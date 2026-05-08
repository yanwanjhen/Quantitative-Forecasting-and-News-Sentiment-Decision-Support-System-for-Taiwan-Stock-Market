"""
回測模組（修正版）
- 不再使用 h=1 / else
- h=1~10 都走 generate_signals_by_horizon()
- 用 validation 選參數，test 真正回測
"""

import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_trade_return: float
    equity_curve: np.ndarray
    daily_returns: np.ndarray
    dates: np.ndarray
    positions: np.ndarray
    signals: np.ndarray
    total_costs: float = 0.0
    total_commission: float = 0.0
    total_tax: float = 0.0
    total_slippage: float = 0.0
    gross_return: float = 0.0


class TradingStrategy:
    def __init__(self, config):
        self.config = config
        self.signal_threshold = getattr(config.backtest, 'signal_threshold', 0.001)
        self.min_holding_days = getattr(config.backtest, 'min_holding_days', 1)

    def generate_signals_by_horizon(self, predictions: np.ndarray, horizon: int,
                                    threshold: float = None,
                                    holding_days: int = None,
                                    cooldown_days: int = 0,
                                    long_only: bool = True) -> np.ndarray:
        preds = np.asarray(predictions, dtype=float)
        n = len(preds)
        signals = np.zeros(n, dtype=float)

        if threshold is None:
            threshold = self.signal_threshold * max(1.0, float(horizon) ** 0.5)

        if holding_days is None:
            holding_days = max(self.min_holding_days, int(horizon))

        next_allowed = 0
        for i in range(n):
            if i < next_allowed:
                continue

            if long_only:
                if preds[i] >= threshold:
                    end_idx = min(i + holding_days, n)
                    signals[i:end_idx] = 1.0
                    next_allowed = end_idx + cooldown_days
            else:
                if preds[i] >= threshold:
                    end_idx = min(i + holding_days, n)
                    signals[i:end_idx] = 1.0
                    next_allowed = end_idx + cooldown_days
                elif preds[i] <= -threshold:
                    end_idx = min(i + holding_days, n)
                    signals[i:end_idx] = -1.0
                    next_allowed = end_idx + cooldown_days

        return signals

    def generate_signals_h1(self, predictions: np.ndarray) -> np.ndarray:
        return self.generate_signals_by_horizon(predictions, horizon=1)

    def generate_signals_h10(self, predictions: np.ndarray, threshold: float = None) -> np.ndarray:
        return self.generate_signals_by_horizon(predictions, horizon=10, threshold=threshold)


class Backtester:
    def __init__(self, config):
        self.config = config
        self.commission_rate = getattr(config.backtest, 'commission_rate', 0.001425)
        self.tax_rate = getattr(config.backtest, 'tax_rate', 0.003)
        self.slippage = getattr(config.backtest, 'slippage', 0.001)

        self.buy_cost = self.commission_rate + self.slippage
        self.sell_cost = self.commission_rate + self.tax_rate + self.slippage

        self.initial_capital = config.backtest.initial_capital
        self.risk_free_rate = getattr(config.backtest, 'risk_free_rate', 0.02)
        self.stop_loss = getattr(config.backtest, 'stop_loss', 0.02)
        self.max_position_size = getattr(config.backtest, 'max_position_size', 1.0)

    def run(self, prices: np.ndarray, signals: np.ndarray, dates: np.ndarray = None) -> BacktestResult:
        n = len(prices)
        if n < 2:
            raise ValueError("價格序列太短，無法回測")

        prices = np.asarray(prices, dtype=float)
        signals = np.asarray(signals, dtype=float).copy()

        daily_returns = np.diff(prices) / prices[:-1]
        desired_signals = signals[:-1]

        strategy_returns = np.zeros(len(daily_returns), dtype=float)
        current_position = 0.0
        trade_returns = []
        current_trade_returns = []

        position_changes = np.diff(np.concatenate([[0], desired_signals]))
        buy_events = (position_changes == 1).astype(float)
        sell_events = (position_changes == -1).astype(float)

        for i in range(len(daily_returns)):
            day_ret = daily_returns[i]
            sig = desired_signals[i]

            if sig == 1:
                if current_position == 0:
                    strategy_returns[i] = day_ret * self.max_position_size - self.buy_cost
                    current_position = 1.0
                    current_trade_returns = [strategy_returns[i]]
                else:
                    strategy_returns[i] = day_ret * self.max_position_size
                    current_trade_returns.append(strategy_returns[i])

                    cum_trade = np.prod(1 + np.array(current_trade_returns)) - 1.0
                    if cum_trade < -self.stop_loss:
                        strategy_returns[i] -= self.sell_cost
                        current_trade_returns[-1] -= self.sell_cost
                        trade_returns.append(np.prod(1 + np.array(current_trade_returns)) - 1.0)
                        current_trade_returns = []
                        current_position = 0.0
            else:
                if current_position == 1.0:
                    strategy_returns[i] = -self.sell_cost
                    current_trade_returns.append(strategy_returns[i])
                    trade_returns.append(np.prod(1 + np.array(current_trade_returns)) - 1.0)
                    current_trade_returns = []
                    current_position = 0.0
                else:
                    strategy_returns[i] = 0.0

        if current_position == 1.0 and len(current_trade_returns) > 0:
            strategy_returns[-1] -= self.sell_cost
            current_trade_returns[-1] -= self.sell_cost
            trade_returns.append(np.prod(1 + np.array(current_trade_returns)) - 1.0)

        equity_curve = self.initial_capital * np.cumprod(1 + strategy_returns)
        equity_curve = np.concatenate([[self.initial_capital], equity_curve])

        total_return = (equity_curve[-1] / self.initial_capital) - 1.0
        gross_returns = desired_signals * daily_returns
        gross_return = np.prod(1 + gross_returns) - 1.0

        n_years = len(strategy_returns) / 252.0
        if n_years > 0 and equity_curve[-1] > 0:
            cagr = (equity_curve[-1] / self.initial_capital) ** (1 / n_years) - 1.0
        else:
            cagr = -1.0

        if len(strategy_returns) > 0 and np.std(strategy_returns) > 1e-8:
            excess_returns = strategy_returns - self.risk_free_rate / 252.0
            sharpe = np.mean(excess_returns) / np.std(strategy_returns) * np.sqrt(252.0)
        else:
            sharpe = 0.0

        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 1e-8:
            excess_return = np.mean(strategy_returns) - self.risk_free_rate / 252.0
            sortino = excess_return / np.std(downside_returns) * np.sqrt(252.0)
        else:
            sortino = 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = np.min(drawdown)

        trades = desired_signals * daily_returns
        trades = trades[trades != 0]
        n_trades = int(np.sum(buy_events))
        win_rate = np.mean(trades > 0) if len(trades) > 0 else 0.0
        avg_trade_return = np.mean(trades) if len(trades) > 0 else 0.0

        total_commission = np.sum(buy_events + sell_events) * self.commission_rate
        total_tax = np.sum(sell_events) * self.tax_rate
        total_slippage_cost = np.sum(buy_events + sell_events) * self.slippage
        total_costs = total_commission + total_tax + total_slippage_cost

        positions = np.concatenate([[0], desired_signals])

        return BacktestResult(
            total_return=total_return,
            annual_return=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=n_trades,
            avg_trade_return=avg_trade_return,
            equity_curve=equity_curve,
            daily_returns=strategy_returns,
            dates=dates,
            positions=positions,
            signals=np.concatenate([[0], desired_signals]),
            total_costs=total_costs,
            total_commission=total_commission,
            total_tax=total_tax,
            total_slippage=total_slippage_cost,
            gross_return=gross_return
        )


class StrategyOptimizer:
    def __init__(self, config):
        self.config = config
        self.strategy = TradingStrategy(config)
        self.backtester = Backtester(config)

    def _score_result(self, result: BacktestResult) -> float:
        score = 2.8 * float(result.sharpe_ratio)
        score += 1.5 * max(result.annual_return, 0.0)
        score -= float(getattr(self.config.backtest, 'max_drawdown_penalty', 0.80)) * abs(result.max_drawdown)

        if result.sharpe_ratio <= 0:
            score -= 2.0
        if result.annual_return <= 0:
            score -= 1.0
        if result.n_trades < getattr(self.config.backtest, 'min_trades_required', 4):
            score -= 4.0
        if result.n_trades > getattr(self.config.backtest, 'max_trades_allowed', 40):
            score -= 2.0

        return float(score)

    def search_best_params(self, predictions, prices, dates, horizon: int) -> Dict:
        threshold_candidates = getattr(self.config.backtest, 'threshold_grid', [0.004, 0.006, 0.008, 0.010])
        holding_candidates = getattr(self.config.backtest, 'holding_days_grid', {}).get(horizon, [2, 3, 5, 8])
        cooldown_candidates = getattr(self.config.backtest, 'cooldown_grid', [0, 1])

        best = None
        all_trials = []

        for threshold in threshold_candidates:
            for holding_days in holding_candidates:
                for cooldown_days in cooldown_candidates:
                    signals = self.strategy.generate_signals_by_horizon(
                        predictions=predictions,
                        horizon=horizon,
                        threshold=threshold,
                        holding_days=holding_days,
                        cooldown_days=cooldown_days,
                        long_only=getattr(self.config.backtest, 'long_only', True),
                    )
                    result = self.backtester.run(prices, signals, dates)

                    trial = {
                        'threshold': float(threshold),
                        'holding_days': int(holding_days),
                        'cooldown_days': int(cooldown_days),
                        'sharpe': float(result.sharpe_ratio),
                        'cagr': float(result.annual_return),
                        'max_drawdown': float(result.max_drawdown),
                        'n_trades': int(result.n_trades),
                        'win_rate': float(result.win_rate),
                        'avg_trade_return': float(result.avg_trade_return),
                    }
                    trial['score'] = self._score_result(result)
                    all_trials.append(trial)

                    if best is None or trial['score'] > best['score']:
                        best = trial.copy()

        best['top_trials'] = sorted(all_trials, key=lambda x: x['score'], reverse=True)[:20]
        return best

    def run_with_best_params(self, predictions, prices, dates, best_params: Dict, horizon: int) -> BacktestResult:
        signals = self.strategy.generate_signals_by_horizon(
            predictions=predictions,
            horizon=horizon,
            threshold=best_params['threshold'],
            holding_days=best_params['holding_days'],
            cooldown_days=best_params['cooldown_days'],
            long_only=getattr(self.config.backtest, 'long_only', True),
        )
        return self.backtester.run(prices, signals, dates)


class BenchmarkComparison:
    def __init__(self, config):
        self.config = config
        self.backtester = Backtester(config)

    def buy_and_hold(self, prices: np.ndarray, dates: np.ndarray = None) -> BacktestResult:
        signals = np.ones(len(prices), dtype=float)
        return self.backtester.run(prices, signals, dates)


def evaluate_model_performance(predictions: np.ndarray, actuals: np.ndarray,
                               prices: np.ndarray, dates: np.ndarray,
                               config, horizon: int = 1) -> Dict:
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    direction_acc = np.mean(np.sign(predictions) == np.sign(actuals))

    backtester = Backtester(config)
    strategy = TradingStrategy(config)
    signals = strategy.generate_signals_by_horizon(predictions, horizon)
    result = backtester.run(prices, signals, dates)

    benchmark = BenchmarkComparison(config)
    bh_result = benchmark.buy_and_hold(prices, dates)

    return {
        'prediction_metrics': {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Direction Accuracy': direction_acc
        },
        'backtest_result': result,
        'benchmark_result': bh_result,
        'outperformance': {
            'vs_benchmark_return': result.total_return - bh_result.total_return,
            'vs_benchmark_sharpe': result.sharpe_ratio - bh_result.sharpe_ratio
        }
    }
