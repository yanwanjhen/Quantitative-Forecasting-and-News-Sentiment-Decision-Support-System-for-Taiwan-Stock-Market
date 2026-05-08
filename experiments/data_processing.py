"""
資料處理模組
- 修正 2008 -> 2015 斷點污染
- 報酬率 / target 只在連續區段內計算
"""

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def _ensure_ticker(df: pd.DataFrame, default_ticker: str) -> pd.DataFrame:
    df = df.copy()
    if 'Ticker' not in df.columns:
        df['Ticker'] = default_ticker
    return df


def _mark_segments(df: pd.DataFrame, date_col: str = 'Date', gap_days: int = 10) -> pd.DataFrame:
    df = df.sort_values(date_col).reset_index(drop=True).copy()
    gap = df[date_col].diff().dt.days.fillna(0)
    df['_segment_id'] = (gap > gap_days).cumsum().astype(int)
    return df


class DataLoader:
    """資料載入器"""

    def __init__(self, config):
        self.config = config
        self.stock_data = None
        self.news_data = None
        self.merged_data = None

    def load_stock_data(self) -> pd.DataFrame:
        print("📊 載入股價資料...")
        df = pd.read_csv(self.config.data.stock_file)

        df['Date'] = pd.to_datetime(df['Date'])
        df = _ensure_ticker(df, self.config.data.index_ticker)

        start_date = pd.to_datetime(self.config.data.start_date)
        end_date = pd.to_datetime(self.config.data.end_date)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        print(f"  ✓ 載入 {len(df)} 筆股價資料")
        print(f"  ✓ 日期範圍: {df['Date'].min()} ~ {df['Date'].max()}")
        print(f"  ✓ 股票數量: {df['Ticker'].nunique()}")

        self.stock_data = df
        return df

    def load_news_data(self) -> pd.DataFrame:
        print("📰 載入新聞情緒資料...")

        news_path = Path(self.config.data.news_file)
        if not news_path.exists():
            print("  ⚠ 找不到新聞檔，改用空新聞資料")
            df = pd.DataFrame({
                'Date': pd.to_datetime([]),
                'sentiment_score': [],
                'sentiment_std': [],
                'news_count': [],
            })
            self.news_data = df
            return df

        df = pd.read_csv(news_path)

        date_cols = ['date', 'Date', 'DATE', 'publish_date', 'news_date']
        date_col = next((c for c in date_cols if c in df.columns), None)
        if date_col is None:
            df['Date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        else:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')

        sentiment_cols = ['sentiment_score', 'sentiment', 'score', 'label', 'positive', 'sentiment_label']
        sentiment_col = next((c for c in sentiment_cols if c in df.columns), None)

        if sentiment_col is None:
            df['sentiment_score'] = 0.0
        elif sentiment_col != 'sentiment_score':
            df['sentiment_score'] = df[sentiment_col]

        if df['sentiment_score'].dtype == 'object':
            sentiment_map = {
                'positive': 1, 'neutral': 0, 'negative': -1,
                'Positive': 1, 'Neutral': 0, 'Negative': -1,
                '1': 1, '0': 0, '-1': -1
            }
            df['sentiment_score'] = df['sentiment_score'].map(sentiment_map).fillna(0.0)

        start_date = pd.to_datetime(self.config.data.start_date)
        end_date = pd.to_datetime(self.config.data.end_date)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

        daily_sentiment = (
            df.groupby('Date', as_index=False)
            .agg(
                sentiment_score=('sentiment_score', 'mean'),
                sentiment_std=('sentiment_score', 'std'),
                news_count=('sentiment_score', 'size')
            )
        )
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0.0)

        print(f"  ✓ 載入 {len(daily_sentiment)} 筆每日新聞資料")

        self.news_data = daily_sentiment
        return daily_sentiment

    def merge_data(self, ticker: str = None) -> pd.DataFrame:
        if ticker is None:
            ticker = self.config.data.index_ticker

        print(f"🔗 合併資料 (標的: {ticker})...")

        if self.stock_data is None:
            self.load_stock_data()
        if self.news_data is None:
            self.load_news_data()

        stock_df = self.stock_data[self.stock_data['Ticker'] == ticker].copy()
        if len(stock_df) == 0:
            raise ValueError(f"找不到股票代碼: {ticker}")

        merged = pd.merge(stock_df, self.news_data, on='Date', how='left')
        for col in ['sentiment_score', 'sentiment_std', 'news_count']:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)

        merged = merged.sort_values('Date').reset_index(drop=True)
        self.merged_data = merged

        print(f"  ✓ 合併後資料筆數: {len(merged)}")
        return merged

    def get_all_tickers(self) -> list:
        if self.stock_data is None:
            self.load_stock_data()
        return self.stock_data['Ticker'].unique().tolist()


class DataPreprocessor:
    """資料預處理器"""

    def __init__(self, config):
        self.config = config
        self.scaler_params = {}

    def _prepare_segmented_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = _mark_segments(df, 'Date', gap_days=10)
        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare_segmented_df(df)

        g = df.groupby('_segment_id', group_keys=False)

        prev_close_1 = g['Close'].shift(1)
        prev_close_5 = g['Close'].shift(5)
        prev_close_10 = g['Close'].shift(10)

        df['log_return'] = np.log(df['Close'] / prev_close_1)
        df['log_return_5d'] = np.log(df['Close'] / prev_close_5)
        df['log_return_10d'] = np.log(df['Close'] / prev_close_10)

        for col in ['log_return', 'log_return_5d', 'log_return_10d']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        return df

    def calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare_segmented_df(df)

        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0

        g = df.groupby('_segment_id', group_keys=False)

        df['sentiment_ma5'] = g['sentiment_score'].transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        df['sentiment_ma20'] = g['sentiment_score'].transform(lambda s: s.rolling(window=20, min_periods=1).mean())
        df['sentiment_momentum'] = g['sentiment_score'].transform(lambda s: s - s.shift(5))
        df['sentiment_momentum'] = df['sentiment_momentum'].fillna(0.0)

        return df

    def create_target_horizon(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        target_h{h} = log(Close_{t+h} / Close_t)
        只在同一連續 segment 內計算，避免跨 2008->2015
        """
        df = self._prepare_segmented_df(df)

        future_close = df.groupby('_segment_id')['Close'].shift(-horizon)
        current_close = df['Close']

        col = f'target_h{horizon}'
        df[col] = np.nan

        mask = (future_close > 0) & (current_close > 0)
        df.loc[mask, col] = np.log(future_close[mask] / current_close[mask])
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        return df

    def create_target_h1(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.create_target_horizon(df, 1)

    def create_all_targets(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        df = df.copy()
        for h in horizons:
            df = self.create_target_horizon(df, h)
        return df

    def normalize_features(self, df: pd.DataFrame, feature_cols: list, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        df = df.copy()

        for col in feature_cols:
            if col not in df.columns:
                continue

            if fit:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if pd.isna(std_val) or std_val == 0:
                    std_val = 1.0
                self.scaler_params[col] = {'mean': float(mean_val), 'std': float(std_val)}

            if col in self.scaler_params:
                mean_val = self.scaler_params[col]['mean']
                std_val = self.scaler_params[col]['std']
                df[f'{col}_norm'] = (df[col] - mean_val) / std_val
            else:
                df[f'{col}_norm'] = df[col]

        return df, self.scaler_params
