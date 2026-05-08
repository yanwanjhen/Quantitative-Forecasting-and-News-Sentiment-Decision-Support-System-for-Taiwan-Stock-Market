"""
資料切分模組
- 固定日期切分
- sequence 不跨 2008 -> 2015 斷點
- regime 只用前一天資訊，不偷看當天
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


def _mark_segments(df: pd.DataFrame, date_col: str = 'Date', gap_days: int = 10) -> pd.DataFrame:
    df = df.sort_values(date_col).reset_index(drop=True).copy()
    gap = df[date_col].diff().dt.days.fillna(0)
    df['_segment_id'] = (gap > gap_days).cumsum().astype(int)
    return df


class TimeSeriesSplitter:
    def __init__(self, config):
        self.config = config

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sort_values('Date').reset_index(drop=True).copy()

        train_df = df[df['Date'] < pd.Timestamp('2023-01-01')].copy()
        val_df = df[(df['Date'] >= pd.Timestamp('2023-01-01')) & (df['Date'] < pd.Timestamp('2025-01-01'))].copy()
        test_df = df[df['Date'] >= pd.Timestamp('2025-01-01')].copy()

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError("固定日期切分後出現空資料集，請檢查資料日期範圍")

        print("📊 固定日期切分:")
        print(f"  ✓ 訓練集: {len(train_df)} 筆 ({train_df['Date'].min().date()} ~ {train_df['Date'].max().date()})")
        print(f"  ✓ 驗證集: {len(val_df)} 筆 ({val_df['Date'].min().date()} ~ {val_df['Date'].max().date()})")
        print(f"  ✓ 測試集: {len(test_df)} 筆 ({test_df['Date'].min().date()} ~ {test_df['Date'].max().date()})")
        return train_df, val_df, test_df


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 regime: Optional[np.ndarray] = None,
                 dates: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.regime = torch.LongTensor(regime) if regime is not None else None
        self.dates = np.array(dates).astype(str) if dates is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {'X': self.X[idx], 'y': self.y[idx]}
        if self.regime is not None:
            item['regime'] = self.regime[idx]
        if self.dates is not None:
            item['date'] = str(self.dates[idx])
        return item


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.splitter = TimeSeriesSplitter(config)
        self.lookback = config.data.lookback_window

    def _normalize_with_train_stats(self, train_df, val_df, test_df, features):
        scaler_params = {}
        for col in features:
            mean_val = train_df[col].mean()
            std_val = train_df[col].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 1.0
            scaler_params[col] = {'mean': float(mean_val), 'std': float(std_val)}

        def _apply(data):
            data = data.copy()
            for col, p in scaler_params.items():
                data[f'{col}_scaled'] = (data[col] - p['mean']) / p['std']
            return data

        return _apply(train_df), _apply(val_df), _apply(test_df), scaler_params

    def _create_sequences(self, data: pd.DataFrame, features: List[str], target: str,
                          regime: Optional[str] = None):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = _mark_segments(data, 'Date', gap_days=10)

        X_list, y_list, date_list, regime_list = [], [], [], []

        for _, seg in data.groupby('_segment_id', sort=True):
            seg = seg.reset_index(drop=True)
            if len(seg) <= self.lookback:
                continue

            feat_vals = seg[features].values
            target_vals = seg[target].values
            regime_vals = seg[regime].values if (regime is not None and regime in seg.columns) else None
            dates = seg['Date'].values

            for i in range(self.lookback, len(seg)):
                X_seq = feat_vals[i - self.lookback:i]
                y_val = target_vals[i]

                if np.isnan(X_seq).any() or np.isnan(y_val):
                    continue

                X_list.append(X_seq)
                y_list.append(y_val)
                date_list.append(dates[i])

                if regime_vals is not None:
                    regime_list.append(int(regime_vals[i - 1]))  # 只用前一天，不偷看當天

        if len(X_list) == 0:
            return (
                np.empty((0, self.lookback, len(features)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=object),
                np.empty((0,), dtype=np.int64)
            )

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
        dates = np.asarray(date_list).astype(str)
        regimes = np.asarray(regime_list, dtype=np.int64) if regime is not None else None

        return X, y, dates, regimes

    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str],
                     target_col: str, regime_col: Optional[str] = None) -> Dict:
        print(f"\n🔧 準備資料集 (lookback={self.lookback})...")

        df = df.sort_values('Date').reset_index(drop=True).copy()

        available_features = [c for c in feature_cols if c in df.columns]
        if len(available_features) == 0:
            raise ValueError("沒有可用特徵欄位")

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"  ⚠ 缺少特徵: {missing}")

        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        train_df, val_df, test_df = self.splitter.split(df)
        train_df, val_df, test_df, scaler_params = self._normalize_with_train_stats(
            train_df, val_df, test_df, available_features
        )

        scaled_features = [f'{c}_scaled' for c in available_features]

        X_train, y_train, d_train, r_train = self._create_sequences(train_df, scaled_features, target_col, regime_col)
        X_val, y_val, d_val, r_val = self._create_sequences(val_df, scaled_features, target_col, regime_col)
        X_test, y_test, d_test, r_test = self._create_sequences(test_df, scaled_features, target_col, regime_col)

        print(f"  ✓ 訓練序列: {X_train.shape}")
        print(f"  ✓ 驗證序列: {X_val.shape}")
        print(f"  ✓ 測試序列: {X_test.shape}")

        return {
            'train': {'X': X_train, 'y': y_train, 'dates': d_train, 'regime': r_train},
            'val': {'X': X_val, 'y': y_val, 'dates': d_val, 'regime': r_val},
            'test': {'X': X_test, 'y': y_test, 'dates': d_test, 'regime': r_test},
            'n_features': len(scaled_features),
            'feature_cols': available_features,
            'scaled_feature_cols': scaled_features,
            'df_train': train_df,
            'df_val': val_df,
            'df_test': test_df,
            'scaler_params': scaler_params,
        }

    def create_dataloaders(self, data: Dict, batch_size: int = 32) -> Dict:
        train_dataset = SequenceDataset(data['train']['X'], data['train']['y'], data['train']['regime'], data['train']['dates'])
        val_dataset = SequenceDataset(data['val']['X'], data['val']['y'], data['val']['regime'], data['val']['dates'])
        test_dataset = SequenceDataset(data['test']['X'], data['test']['y'], data['test']['regime'], data['test']['dates'])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }


class RegimeSplitDataPipeline(DataPipeline):
    def prepare_regime_data(self, df: pd.DataFrame, feature_cols: List[str],
                            target_col: str, regime_col: str) -> Dict:
        return self.prepare_data(df, feature_cols, target_col, regime_col)


def prepare_experiment_data(config, df: pd.DataFrame, feature_cols: List[str],
                            target_col: str, regime_col: str = None,
                            model_type: str = 'single') -> Dict:
    pipeline = DataPipeline(config)
    return pipeline.prepare_data(df, feature_cols, target_col, regime_col)
