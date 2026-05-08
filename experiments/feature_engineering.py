"""
特徵工程模組
- 所有技術指標只在連續區段內計算
- 修正 2008 -> 2015 斷點污染
"""

import pandas as pd
import numpy as np
from typing import List
from importlib import import_module
import warnings

warnings.filterwarnings('ignore')


def _mark_segments(df: pd.DataFrame, date_col: str = 'Date', gap_days: int = 10) -> pd.DataFrame:
    df = df.sort_values(date_col).reset_index(drop=True).copy()
    gap = df[date_col].diff().dt.days.fillna(0)
    df['_segment_id'] = (gap > gap_days).cumsum().astype(int)
    return df


class TechnicalIndicators:
    @staticmethod
    def calculate_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        df = df.copy()
        for period in periods:
            ma = df['Close'].rolling(window=period, min_periods=1).mean()
            df[f'MA_{period}'] = ma
            df[f'MA_{period}_dev'] = (df['Close'] - ma) / ma.replace(0, np.nan)
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        df[f'RSI_{period}'] = rsi.fillna(50.0).clip(0, 100)
        return df

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()

        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()

        up_move = df['High'].diff()
        down_move = -df['Low'].diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
        adx = dx.rolling(window=period, min_periods=1).mean()

        df[f'ADX_{period}'] = adx.fillna(25.0)
        df[f'+DI_{period}'] = plus_di.fillna(0.0)
        df[f'-DI_{period}'] = minus_di.fillna(0.0)
        df[f'ATR_{period}'] = atr.fillna(0.0)
        return df

    @staticmethod
    def calculate_trix(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()
        ema1 = df['Close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, 1e-10)
        df[f'TRIX_{period}'] = trix.fillna(0.0)
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        df = df.copy()
        middle = df['Close'].rolling(window=period, min_periods=1).mean()
        std = df['Close'].rolling(window=period, min_periods=1).std().fillna(0.0)
        upper = middle + std * num_std
        lower = middle - std * num_std

        df['BB_middle'] = middle
        df['BB_upper'] = upper
        df['BB_lower'] = lower
        df['BB_width'] = (upper - lower) / middle.replace(0, np.nan)
        df['BB_pct'] = (df['Close'] - lower) / (upper - lower).replace(0, 1e-10)
        df['BB_pct'] = df['BB_pct'].clip(-1, 2)
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        df = df.copy()
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()

        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd - macd_signal
        return df

    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_volume'] = np.log1p(df['Volume'])
        df['volume_ma5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ma20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma20'].replace(0, 1e-10)

        obv = (np.sign(df['Close'].diff()).fillna(0.0) * df['Volume']).cumsum()
        obv_mean = obv.rolling(window=10, min_periods=1).mean()
        obv_std = obv.rolling(window=10, min_periods=1).std().replace(0, 1.0)

        df['OBV'] = obv
        df['OBV_norm'] = ((obv - obv_mean) / obv_std).fillna(0.0)
        return df

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, periods: List[int] = [5, 10]) -> pd.DataFrame:
        df = df.copy()

        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        for period in periods:
            df[f'volatility_{period}d'] = df['log_return'].rolling(window=period, min_periods=1).std() * np.sqrt(252)
            df[f'parkinson_vol_{period}d'] = np.sqrt(
                (1.0 / (4.0 * np.log(2.0))) *
                (np.log(df['High'] / df['Low']).pow(2).rolling(window=period, min_periods=1).mean())
            ) * np.sqrt(252)

        return df


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.ti = TechnicalIndicators()

    def _apply_by_segment(self, df: pd.DataFrame, func):
        parts = []
        for _, seg in df.groupby('_segment_id', sort=True):
            parts.append(func(seg.copy()))
        out = pd.concat(parts, axis=0).sort_index()
        return out

    def add_regime_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        df['rolling_vol_10'] = df['log_return'].rolling(window=10, min_periods=5).std() * np.sqrt(252)

        rolling_max_60 = df['Close'].rolling(window=60, min_periods=5).max()
        df['drawdown_60'] = (df['Close'] / rolling_max_60) - 1.0

        log_close = np.log(df['Close'].replace(0, np.nan)).ffill()

        def _slope(arr):
            x = np.arange(len(arr), dtype=float)
            if len(arr) < 2:
                return 0.0
            slope, _ = np.polyfit(x, arr, deg=1)
            return float(slope)

        df['trend_slope_10'] = log_close.rolling(window=10, min_periods=5).apply(_slope, raw=True)

        up_ratio = (df['log_return'] > 0).astype(float).rolling(window=10, min_periods=5).mean()
        df['market_breadth_proxy_10'] = (2.0 * up_ratio) - 1.0

        for col in ['rolling_vol_10', 'drawdown_60', 'trend_slope_10', 'market_breadth_proxy_10']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = _mark_segments(df, 'Date', gap_days=10)

        print("📈 計算技術指標（分段避免 2008->2015 亂接）...")

        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_ma(x, periods=[5, 10, 20, 50]))
        print("  ✓ 移動平均線 (MA)")

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_rsi(x, period=14))
        print("  ✓ RSI")

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_adx(x, period=14))
        print("  ✓ ADX / ATR")

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_trix(x, period=14))
        print("  ✓ TRIX")

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_bollinger_bands(x, period=20))
        print("  ✓ Bollinger Bands")

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_macd(x))
        print("  ✓ MACD")

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_volume_indicators(x))
        print("  ✓ Volume features")

        df = self._apply_by_segment(df, lambda x: self.ti.calculate_volatility(x, periods=[5, 10]))
        print("  ✓ Volatility")

        df = self._apply_by_segment(df, self.add_regime_context_features)
        print("  ✓ Regime context")

        return df.sort_values('Date').reset_index(drop=True)

    def get_feature_set(self, feature_combination: str) -> List[str]:
        combo = None
        searched_modules = []

        for module_name in ('config_bull_bias', 'config'):
            try:
                module = import_module(module_name)
                searched_modules.append(module_name)
                combos = getattr(module, 'FEATURE_COMBINATIONS', {})
                if feature_combination in combos:
                    combo = combos[feature_combination]
                    break
            except Exception:
                continue

        if combo is None:
            raise ValueError(f"未知的特徵組合: {feature_combination}；已搜尋模組: {searched_modules}")

        features = []

        if combo.get('use_price', False):
            features.extend([
                'log_return', 'log_return_5d', 'log_return_10d',
                'volatility_5d', 'volatility_10d'
            ])

        if combo.get('use_sentiment', False):
            features.extend([
                'sentiment_score', 'sentiment_ma5', 'sentiment_ma20', 'sentiment_momentum'
            ])

        if combo.get('use_regime_context', False):
            features.extend([
                'rolling_vol_10', 'drawdown_60', 'trend_slope_10', 'market_breadth_proxy_10'
            ])

        if combo.get('tech_set') == 'tech_1':
            features.extend([
                'MA_5_dev', 'MA_10_dev', 'MA_20_dev', 'MA_50_dev',
                'ADX_14', 'RSI_14', 'TRIX_14'
            ])
        elif combo.get('tech_set') == 'tech_2':
            features.extend([
                'BB_width', 'BB_pct',
                'ADX_14', 'RSI_14', 'TRIX_14'
            ])
        elif combo.get('tech_set') == 'tech_3':
            features.extend([
                'MACD_hist', 'MACD_signal', 'parkinson_vol_5d', 'parkinson_vol_10d',
                'ADX_14', 'RSI_14'
            ])

        return features
