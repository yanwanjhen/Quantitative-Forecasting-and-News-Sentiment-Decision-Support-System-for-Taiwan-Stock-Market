"""
熊市標籤模組
Regime Labeling Module - Rolling Peak Drawdown
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class RegimeLabeler:
    """市場狀態標籤器（基於 Rolling Peak Drawdown）"""
    
    def __init__(self, config):
        """
        初始化標籤器
        
        Args:
            config: ExperimentConfig 配置物件
        """
        self.config = config
        self.bear_thresholds = config.regime.bear_thresholds
    
    def calculate_rolling_peak(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """
        計算滾動歷史最高價
        
        Peak_t = max(P_1, P_2, ..., P_t)
        
        Args:
            df: 股價資料
            price_col: 價格欄位名稱
            
        Returns:
            pd.DataFrame: 含滾動最高價的資料
        """
        df = df.copy()
        # 改為計算過去 252 天的局部最高點，避免歷史極端高點影響長期判斷
        df['rolling_peak'] = df[price_col].rolling(window=252, min_periods=1).max()
        return df
    
    def calculate_drawdown(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """
        計算每日 Drawdown
        
        Drawdown_t = (P_t - Peak_t) / Peak_t
        
        Args:
            df: 股價資料
            price_col: 價格欄位名稱
            
        Returns:
            pd.DataFrame: 含 Drawdown 的資料
        """
        df = df.copy()
        
        # 確保有 rolling_peak
        if 'rolling_peak' not in df.columns:
            df = self.calculate_rolling_peak(df, price_col)
        
        # 計算 drawdown (負值或零)
        df['drawdown'] = (df[price_col] - df['rolling_peak']) / df['rolling_peak']
        
        return df
    
    def label_regime(self, df: pd.DataFrame, threshold: float = -0.15,
                    price_col: str = 'Close') -> pd.DataFrame:
        """
        根據特定門檻標記熊市/牛市
        
        Args:
            df: 股價資料
            threshold: 熊市門檻 (負值，如 -0.15)
            price_col: 價格欄位名稱
            
        Returns:
            pd.DataFrame: 含 regime 標籤的資料
        """
        df = df.copy()
        
        # 計算 drawdown
        if 'drawdown' not in df.columns:
            df = self.calculate_drawdown(df, price_col)
        
        # 標記 regime
        # 0 = Bull (牛市), 1 = Bear (熊市)
        threshold_name = str(int(abs(threshold) * 100))
        regime_col = f'regime_{threshold_name}'
        
        df[regime_col] = (df['drawdown'] <= threshold).astype(int)
        
        return df
    
    def label_all_regimes(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """
        根據所有門檻標記熊市/牛市
        
        Args:
            df: 股價資料
            price_col: 價格欄位名稱
            
        Returns:
            pd.DataFrame: 含所有 regime 標籤的資料
        """
        df = df.copy()
        
        print("🐻 計算市場狀態標籤...")
        
        # 計算 drawdown
        df = self.calculate_drawdown(df, price_col)
        
        # 為每個門檻創建標籤
        for threshold in self.bear_thresholds:
            df = self.label_regime(df, threshold, price_col)
            threshold_pct = int(abs(threshold) * 100)
            regime_col = f'regime_{threshold_pct}'
            bear_days = df[regime_col].sum()
            bear_pct = bear_days / len(df) * 100
            print(f"  ✓ 門檻 {threshold_pct}%: 熊市天數 {bear_days} ({bear_pct:.1f}%)")
        
        return df
    
    def get_regime_statistics(self, df: pd.DataFrame) -> Dict:
        """
        獲取市場狀態統計
        
        Args:
            df: 含 regime 標籤的資料
            
        Returns:
            Dict: 統計資訊
        """
        stats = {}
        
        for threshold in self.bear_thresholds:
            threshold_pct = int(abs(threshold) * 100)
            regime_col = f'regime_{threshold_pct}'
            
            if regime_col not in df.columns:
                continue
            
            total_days = len(df)
            bear_days = df[regime_col].sum()
            bull_days = total_days - bear_days
            
            # 計算熊市期間的平均報酬
            if 'log_return' in df.columns:
                bear_return = df[df[regime_col] == 1]['log_return'].mean()
                bull_return = df[df[regime_col] == 0]['log_return'].mean()
            else:
                bear_return = 0
                bull_return = 0
            
            stats[threshold_pct] = {
                'total_days': total_days,
                'bear_days': bear_days,
                'bull_days': bull_days,
                'bear_ratio': bear_days / total_days,
                'bull_ratio': bull_days / total_days,
                'bear_avg_return': bear_return,
                'bull_avg_return': bull_return
            }
        
        return stats
    
    def get_regime_periods(self, df: pd.DataFrame, threshold: float = -0.15) -> List[Dict]:
        """
        獲取熊市期間列表
        
        Args:
            df: 含 regime 標籤的資料
            threshold: 熊市門檻
            
        Returns:
            List[Dict]: 熊市期間列表
        """
        threshold_pct = int(abs(threshold) * 100)
        regime_col = f'regime_{threshold_pct}'
        
        if regime_col not in df.columns:
            return []
        
        periods = []
        in_bear = False
        start_date = None
        
        for idx, row in df.iterrows():
            if row[regime_col] == 1 and not in_bear:
                # 進入熊市
                in_bear = True
                start_date = row['Date']
                start_price = row['Close']
            elif row[regime_col] == 0 and in_bear:
                # 離開熊市
                in_bear = False
                periods.append({
                    'start': start_date,
                    'end': row['Date'],
                    'start_price': start_price,
                    'end_price': row['Close'],
                    'duration': (row['Date'] - start_date).days,
                    'max_drawdown': df.loc[
                        (df['Date'] >= start_date) & (df['Date'] <= row['Date']),
                        'drawdown'
                    ].min()
                })
        
        # 如果還在熊市中
        if in_bear:
            periods.append({
                'start': start_date,
                'end': df['Date'].iloc[-1],
                'start_price': start_price,
                'end_price': df['Close'].iloc[-1],
                'duration': (df['Date'].iloc[-1] - start_date).days,
                'max_drawdown': df.loc[df['Date'] >= start_date, 'drawdown'].min()
            })
        
        return periods


class RegimeAwareDataset:
    """市場狀態感知資料集"""
    
    def __init__(self, df: pd.DataFrame, threshold: float = -0.15):
        """
        初始化
        
        Args:
            df: 含 regime 標籤的資料
            threshold: 使用的熊市門檻
        """
        self.df = df.copy()
        self.threshold = threshold
        self.threshold_pct = int(abs(threshold) * 100)
        self.regime_col = f'regime_{self.threshold_pct}'
    
    def split_by_regime(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按市場狀態分割資料
        
        Returns:
            Tuple: (牛市資料, 熊市資料)
        """
        bull_df = self.df[self.df[self.regime_col] == 0].copy()
        bear_df = self.df[self.df[self.regime_col] == 1].copy()
        
        return bull_df, bear_df
    
    def get_regime_labels(self) -> np.ndarray:
        """
        獲取 regime 標籤陣列
        
        Returns:
            np.ndarray: regime 標籤
        """
        return self.df[self.regime_col].values
    
    def get_regime_for_dates(self, dates: np.ndarray) -> np.ndarray:
        """
        獲取特定日期的 regime 標籤
        
        Args:
            dates: 日期陣列
            
        Returns:
            np.ndarray: regime 標籤
        """
        date_to_regime = dict(zip(self.df['Date'], self.df[self.regime_col]))
        return np.array([date_to_regime.get(d, 0) for d in dates])


def add_regime_labels(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    便捷函數：添加所有 regime 標籤
    
    Args:
        df: 股價資料
        config: 配置物件
        
    Returns:
        pd.DataFrame: 含 regime 標籤的資料
    """
    labeler = RegimeLabeler(config)
    return labeler.label_all_regimes(df)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    from config import get_config
    from data_processing import DataLoader
    
    config = get_config()
    
    # 載入資料
    loader = DataLoader(config)
    df = loader.merge_data()
    
    # 計算報酬率
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 添加 regime 標籤
    labeler = RegimeLabeler(config)
    df = labeler.label_all_regimes(df)
    
    # 統計
    stats = labeler.get_regime_statistics(df)
    
    print("\n📊 市場狀態統計:")
    for threshold_pct, stat in stats.items():
        print(f"\n門檻 {threshold_pct}%:")
        print(f"  牛市天數: {stat['bull_days']} ({stat['bull_ratio']*100:.1f}%)")
        print(f"  熊市天數: {stat['bear_days']} ({stat['bear_ratio']*100:.1f}%)")
        print(f"  牛市平均日報酬: {stat['bull_avg_return']*100:.4f}%")
        print(f"  熊市平均日報酬: {stat['bear_avg_return']*100:.4f}%")
    
    # 熊市期間
    print("\n📅 熊市期間 (門檻 15%):")
    periods = labeler.get_regime_periods(df, threshold=-0.15)
    for i, period in enumerate(periods[:5]):
        print(f"  {i+1}. {period['start'].strftime('%Y-%m-%d')} ~ {period['end'].strftime('%Y-%m-%d')}")
        print(f"     持續 {period['duration']} 天, 最大跌幅 {period['max_drawdown']*100:.1f}%")
