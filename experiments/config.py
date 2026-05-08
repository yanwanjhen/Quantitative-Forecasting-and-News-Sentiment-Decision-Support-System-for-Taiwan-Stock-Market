from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "experiments" / "saved_models"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"

for dir_path in [MODELS_DIR, OUTPUT_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    stock_file: str = str(PROJECT_ROOT / "experiments" / "^TWII_historical_data.csv")
    news_file: str = str(DATA_DIR / "news2015-2025_labeled_final.csv")

    start_date: str = "2008-01-01"
    end_date: str = "2025-12-31"

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    lookback_window: int = 60
    index_ticker: str = "^TWII"
    target_scale: float = 1.0


@dataclass
class RegimeConfig:
    bear_thresholds: List[float] = field(default_factory=lambda: [-0.10, -0.12, -0.15, -0.18, -0.20])
    default_threshold: float = -0.15


@dataclass
class FeatureConfig:
    price_features: List[str] = field(default_factory=lambda: [
        'Open', 'High', 'Low', 'Close', 'Volume'
    ])

    return_features: List[str] = field(default_factory=lambda: [
        'log_return', 'log_return_2d', 'log_return_3d', 'log_return_4d', 'log_return_5d'
    ])

    sentiment_features: List[str] = field(default_factory=list)

    tech_indicators_1: List[str] = field(default_factory=lambda: [
        'MA_5_dev', 'MA_10_dev', 'MA_20_dev', 'MA_50_dev',
        'ADX_14', 'RSI_14', 'TRIX_14'
    ])

    tech_indicators_2: List[str] = field(default_factory=lambda: [
        'BB_width', 'BB_pct',
        'ADX_14', 'RSI_14', 'TRIX_14'
    ])


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.10

    batch_size: int = 32
    learning_rate: float = 7e-5
    weight_decay: float = 1e-5
    epochs: int = 160
    patience: int = 28

    model_types: List[str] = field(default_factory=lambda: [
        'regime_embedding',
        'regime_switch'
    ])


@dataclass
class PredictionConfig:
    horizons: List[int] = field(default_factory=lambda: list(range(1, 11)))


@dataclass
class BacktestConfig:
    commission_rate: float = 0.001425
    tax_rate: float = 0.003
    slippage: float = 0.001

    initial_capital: float = 1_000_000
    risk_free_rate: float = 0.02
    min_holding_days: int = 1

    enable_stop_loss: bool = True
    stop_loss: float = 0.03
    stop_cooldown_days: int = 0

    long_only: bool = True
    max_position_size: float = 1.0

    strategy_mode: str = "bull_overlay"
    compare_buy_hold_after_costs: bool = True
    save_raw_buy_hold_too: bool = True

    base_exposure: float = 0.95
    strong_exposure: float = 1.00
    defensive_exposure: float = 0.20
    signal_ema_alpha: float = 0.55
    hold_ratio: float = 0.55
    exit_threshold_ratio: float = 0.28

    allow_overlay_leverage: bool = False
    leveraged_strong_exposure: float = 1.10

    threshold_grid: List[float] = field(default_factory=lambda: [
        0.004, 0.006, 0.008, 0.010
    ])

    holding_days_grid: Dict[int, List[int]] = field(default_factory=lambda: {
        1: [2, 3, 5],
        2: [2, 3, 5],
        3: [3, 5, 8],
        4: [3, 5, 8],
        5: [3, 5, 8],
        6: [5, 8, 10],
        7: [5, 8, 10],
        8: [5, 8, 10],
        9: [5, 8, 10],
        10: [5, 8, 10],
    })

    cooldown_grid: List[int] = field(default_factory=lambda: [0, 1])

    base_exposure_grid: List[float] = field(default_factory=lambda: [0.85, 0.95, 1.00])
    defensive_exposure_grid: List[float] = field(default_factory=lambda: [0.00, 0.20])
    signal_ema_alpha_grid: List[float] = field(default_factory=lambda: [0.35, 0.55])

    min_trades_required: int = 4
    max_trades_allowed: int = 40

    prefer_positive_cagr: bool = True
    max_drawdown_penalty: float = 0.80


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    seed: int = 42
    experiment_name: str = "sharpe_push_overlay"
    use_gpu: bool = True
    verbose: bool = True


FEATURE_COMBINATIONS = {
    "stock_only": {
        "name": "股價",
        "use_price": True,
        "use_sentiment": False,
        "tech_set": None
    },
    "stock_tech1": {
        "name": "股價 + MA/ADX/RSI/TRIX",
        "use_price": True,
        "use_sentiment": False,
        "tech_set": "tech_1"
    },
    "stock_tech2": {
        "name": "股價 + BB/ADX/RSI/TRIX",
        "use_price": True,
        "use_sentiment": False,
        "tech_set": "tech_2"
    }
}

MODEL_VERSIONS = {
    "single": {
        "name": "Single LSTM",
        "use_regime": False,
        "regime_embedding": False
    },
    "regime_embedding": {
        "name": "LSTM + Regime Embedding",
        "use_regime": True,
        "regime_embedding": True
    },
    "regime_switch": {
        "name": "Regime-Switched (Bull/Bear)",
        "use_regime": True,
        "regime_embedding": False
    }
}


def get_config() -> ExperimentConfig:
    return ExperimentConfig()
