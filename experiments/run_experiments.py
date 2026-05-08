import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

from config import get_config, FEATURE_COMBINATIONS, MODELS_DIR, OUTPUT_DIR
from data_processing import DataLoader, DataPreprocessor
from feature_engineering import FeatureEngineer
from regime_labeling import RegimeLabeler
from data_splitting import DataPipeline
from models import create_model, count_parameters
from training import Trainer, RegimeSwitchTrainer
from backtesting import StrategyOptimizer, BenchmarkComparison


class ExperimentRunner:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._set_seed(self.config.seed)

        self.output_dir = OUTPUT_DIR / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔬 實驗執行器初始化完成")
        print(f"   輸出目錄: {self.output_dir}")

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_and_prepare_data(self, ticker: str = None) -> pd.DataFrame:
        loader = DataLoader(self.config)
        stock_df = loader.load_stock_data()

        # 若有指定 ticker，才篩單一標的；否則保留整份資料
        if ticker is not None:
            df = stock_df[stock_df['Ticker'] == ticker].copy()
            if len(df) == 0:
                raise ValueError(f"找不到股票代碼: {ticker}")
        else:
            df = stock_df.copy()

        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        all_parts = []
        preprocessor = DataPreprocessor(self.config)
        engineer = FeatureEngineer(self.config)
        labeler = RegimeLabeler(self.config)

        if 'Ticker' in df.columns:
            groups = df.groupby('Ticker', sort=True)
        else:
            groups = [(self.config.data.index_ticker, df)]

        for tk, part in groups:
            part = part.sort_values('Date').reset_index(drop=True).copy()
            part = preprocessor.calculate_returns(part)
            part = engineer.add_all_features(part)
            part = labeler.label_all_regimes(part)
            part = preprocessor.create_all_targets(part, self.config.prediction.horizons)
            all_parts.append(part)

        df = pd.concat(all_parts, axis=0).sort_values(['Ticker', 'Date']).reset_index(drop=True)

        print(f"   實際使用股票數量: {df['Ticker'].nunique() if 'Ticker' in df.columns else 1}")
        return df

    @staticmethod
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) <= 1 or len(y) <= 1:
            return 0.0
        c = np.corrcoef(x, y)[0, 1]
        if np.isnan(c):
            return 0.0
        return float(c)

    def _calc_return_metrics(self, daily_returns: np.ndarray) -> dict:
        daily_returns = np.asarray(daily_returns, dtype=float)

        if len(daily_returns) == 0:
            return {
                'total_return': 0.0,
                'cagr': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'equity_curve': np.array([self.config.backtest.initial_capital], dtype=float),
            }

        initial_capital = float(self.config.backtest.initial_capital)
        equity_curve = initial_capital * np.cumprod(1.0 + daily_returns)
        equity_curve = np.concatenate([[initial_capital], equity_curve])

        total_return = float(equity_curve[-1] / initial_capital - 1.0)

        n_years = len(daily_returns) / 252.0
        if n_years > 0 and equity_curve[-1] > 0:
            cagr = float((equity_curve[-1] / initial_capital) ** (1.0 / n_years) - 1.0)
        else:
            cagr = -1.0

        if len(daily_returns) > 1 and np.std(daily_returns) > 1e-8:
            excess_returns = daily_returns - self.config.backtest.risk_free_rate / 252.0
            sharpe = float(np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252.0))
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = float(np.min(drawdown))

        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'equity_curve': equity_curve,
        }

    def _buy_and_hold_metrics(self, prices: np.ndarray, include_costs: bool = False) -> dict:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return {
                'total_return': 0.0,
                'cagr': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
            }

        daily_returns = np.diff(prices) / prices[:-1]

        if include_costs and len(daily_returns) > 0:
            buy_cost = self.config.backtest.commission_rate + self.config.backtest.slippage
            sell_cost = self.config.backtest.commission_rate + self.config.backtest.tax_rate + self.config.backtest.slippage
            daily_returns = daily_returns.copy()
            daily_returns[0] -= buy_cost
            daily_returns[-1] -= sell_cost

        metrics = self._calc_return_metrics(daily_returns)
        return {
            'total_return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['max_drawdown'],
        }

    def _classify_test_market(self, prices: np.ndarray, regime_labels: np.ndarray = None) -> dict:
        prices = np.asarray(prices, dtype=float)
        total_return = float(prices[-1] / prices[0] - 1.0) if len(prices) >= 2 else 0.0

        bull_ratio = None
        bear_ratio = None

        if regime_labels is not None and len(regime_labels) > 0:
            regime_labels = np.asarray(regime_labels)
            bull_ratio = float(np.mean(regime_labels == 0))
            bear_ratio = float(np.mean(regime_labels == 1))

        if bear_ratio is not None and bear_ratio >= 0.65:
            market_regime = 'bear'
        elif bull_ratio is not None and bull_ratio >= 0.65 and total_return > 0:
            market_regime = 'bull'
        else:
            if total_return >= 0.15:
                market_regime = 'bull'
            elif total_return <= -0.15:
                market_regime = 'bear'
            else:
                market_regime = 'sideways'

        return {
            'market_regime': market_regime,
            'market_return': total_return,
            'bull_ratio': bull_ratio,
            'bear_ratio': bear_ratio,
        }

    def run_single_experiment(self, df, feature_combo, model_type, horizon, bear_threshold):
        experiment_name = f"{feature_combo}_{model_type}_h{horizon}_t{int(abs(bear_threshold)*100)}"
        print(f"\n{'=' * 60}")
        print(f"🧪 實驗: {experiment_name}")
        print(f"{'=' * 60}")

        engineer = FeatureEngineer(self.config)
        feature_cols = engineer.get_feature_set(feature_combo)

        target_col = f'target_h{horizon}'
        regime_col = f'regime_{int(abs(bear_threshold) * 100)}'

        pipeline = DataPipeline(self.config)
        data = pipeline.prepare_data(df, feature_cols, target_col, regime_col)

        n_features = data['n_features']
        model = create_model(model_type, n_features, self.config)

        print(f"   模型參數量: {count_parameters(model):,}")

        loaders = pipeline.create_dataloaders(data, self.config.model.batch_size)
        train_regime_labels = data['train'].get('regime', None)

        # 訓練
        if model_type == 'regime_switch':
            trainer = RegimeSwitchTrainer(model, self.config)
            history = trainer.train(data)

            val_predictions, predicted_regime_val = trainer.predict(data['val']['X'])
            predictions, predicted_regime_test = trainer.predict(data['test']['X'])
            actuals = data['test']['y']
        else:
            trainer = Trainer(model, self.config, model_type=model_type, use_weighted_loss=True)
            history = trainer.train(
                loaders['train'],
                loaders['val'],
                train_regime_labels=train_regime_labels
            )

            val_eval = trainer.evaluate(loaders['val'])
            test_eval = trainer.evaluate(loaders['test'])

            val_predictions = val_eval['predictions']
            predictions = test_eval['predictions']
            actuals = test_eval['actuals']

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        direction_acc = np.mean(np.sign(predictions) == np.sign(actuals))
        correlation = self._safe_corr(predictions, actuals)
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        backtest_result = None
        buy_hold_result = None
        best_strategy_params = None
        test_context = {}

        buy_hold_raw = None
        buy_hold_after_cost = None
        selected_benchmark = None
        excess_return_vs_buy_hold = None
        beats_buy_hold = None
        market_info = None

        if 'df_test' in data and 'df_val' in data:
            test_df = data['df_test']
            val_df = data['df_val']

            if 'Close' in test_df.columns and 'Close' in val_df.columns:
                val_dates = pd.to_datetime(data['val']['dates'])
                test_dates = pd.to_datetime(data['test']['dates'])

                val_close_map = val_df.set_index('Date')['Close']
                test_close_map = test_df.set_index('Date')['Close']

                prices_val = val_close_map.reindex(val_dates).to_numpy()
                prices_test = test_close_map.reindex(test_dates).to_numpy()

                print("   [對齊檢查]")
                print(f"   len(val_prices)={len(prices_val)}, len(val_predictions)={len(val_predictions)}")
                print(f"   len(test_prices)={len(prices_test)}, len(test_predictions)={len(predictions)}")

                if (
                    len(prices_test) == len(predictions)
                    and len(prices_val) == len(val_predictions)
                    and not np.isnan(prices_test).any()
                    and not np.isnan(prices_val).any()
                ):
                    optimizer = StrategyOptimizer(self.config)
                    benchmark = BenchmarkComparison(self.config)

                    market_info = self._classify_test_market(prices_test, data['test'].get('regime', None))

                    buy_hold_raw = self._buy_and_hold_metrics(prices_test, include_costs=False)
                    buy_hold_after_cost = self._buy_and_hold_metrics(prices_test, include_costs=True)

                    use_after_cost = bool(getattr(self.config.backtest, 'compare_buy_hold_after_costs', True))
                    selected_benchmark = buy_hold_after_cost if use_after_cost else buy_hold_raw

                    best_strategy_params = optimizer.search_best_params(
                        predictions=val_predictions,
                        prices=prices_val,
                        dates=data['val']['dates'],
                        horizon=horizon,
                    )

                    backtest_result = optimizer.run_with_best_params(
                        predictions=predictions,
                        prices=prices_test,
                        dates=data['test']['dates'],
                        best_params=best_strategy_params,
                        horizon=horizon,
                    )

                    buy_hold_result = benchmark.buy_and_hold(prices_test, data['test']['dates'])

                    excess_return_vs_buy_hold = float(backtest_result.total_return - selected_benchmark['total_return'])
                    beats_buy_hold = bool(backtest_result.total_return > selected_benchmark['total_return'])

                    aligned_slice = slice(self.config.data.lookback_window, None)
                    test_context = {
                        'test_dates': data['test']['dates'],
                        'test_close': prices_test,
                        'test_rsi_14': test_df['RSI_14'].values[aligned_slice] if 'RSI_14' in test_df.columns else None,
                        'test_bb_lower': test_df['BB_lower'].values[aligned_slice] if 'BB_lower' in test_df.columns else None,
                        'test_atr_14': test_df['ATR_14'].values[aligned_slice] if 'ATR_14' in test_df.columns else None,
                        'test_drawdown_60': test_df['drawdown_60'].values[aligned_slice] if 'drawdown_60' in test_df.columns else None
                    }

        model_path = MODELS_DIR / f"{experiment_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'n_features': n_features,
                'model_type': model_type,
                'feature_combo': feature_combo,
                'horizon': horizon,
                'bear_threshold': bear_threshold,
                'task': 'regression_return'
            }
        }, model_path)

        result = {
            'experiment_name': experiment_name,
            'feature_combo': feature_combo,
            'model_type': model_type,
            'horizon': horizon,
            'bear_threshold': bear_threshold,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'direction_accuracy': direction_acc,
            'correlation': correlation,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals,
            'history': history,
            'backtest': backtest_result,
            'model_path': str(model_path),
            'best_strategy_params': best_strategy_params
        }

        if market_info is not None:
            result['test_market_regime'] = market_info['market_regime']
            result['test_market_return'] = market_info['market_return']
            result['test_bull_ratio'] = market_info['bull_ratio']
            result['test_bear_ratio'] = market_info['bear_ratio']

        if buy_hold_raw is not None:
            result['buy_hold_total_return_raw'] = buy_hold_raw['total_return']
            result['buy_hold_cagr_raw'] = buy_hold_raw['cagr']
            result['buy_hold_sharpe_raw'] = buy_hold_raw['sharpe']
            result['buy_hold_max_drawdown_raw'] = buy_hold_raw['max_drawdown']

        if buy_hold_after_cost is not None:
            result['buy_hold_total_return_after_cost'] = buy_hold_after_cost['total_return']
            result['buy_hold_cagr_after_cost'] = buy_hold_after_cost['cagr']
            result['buy_hold_sharpe_after_cost'] = buy_hold_after_cost['sharpe']
            result['buy_hold_max_drawdown_after_cost'] = buy_hold_after_cost['max_drawdown']

        if selected_benchmark is not None:
            result['buy_hold_total_return'] = selected_benchmark['total_return']
            result['buy_hold_cagr'] = selected_benchmark['cagr']
            result['buy_hold_sharpe'] = selected_benchmark['sharpe']
            result['buy_hold_max_drawdown'] = selected_benchmark['max_drawdown']
            result['buy_hold_benchmark_name'] = 'after_cost' if getattr(self.config.backtest, 'compare_buy_hold_after_costs', True) else 'raw'

        if backtest_result:
            result['sharpe_ratio'] = backtest_result.sharpe_ratio
            result['sortino_ratio'] = backtest_result.sortino_ratio
            result['max_drawdown'] = backtest_result.max_drawdown
            result['cagr'] = backtest_result.annual_return
            result['total_return'] = backtest_result.total_return
            result['win_rate'] = backtest_result.win_rate
            result['n_trades'] = backtest_result.n_trades
            result['avg_trade_return'] = backtest_result.avg_trade_return

        if excess_return_vs_buy_hold is not None:
            result['excess_return_vs_buy_hold'] = excess_return_vs_buy_hold
            result['beats_buy_hold'] = beats_buy_hold
        print("\n📊 結果:")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   全樣本方向準確率: {direction_acc*100:.2f}%")

        if backtest_result:
            print("   [Strategy Test]")
            print(f"   Sharpe Ratio: {backtest_result.sharpe_ratio:.3f}")
            print(f"   Max Drawdown: {backtest_result.max_drawdown*100:.2f}%")
            print(f"   CAGR: {backtest_result.annual_return*100:.2f}%")
            print(f"   Total Return: {backtest_result.total_return*100:.2f}%")
            print(f"   交易次數: {backtest_result.n_trades}")
        else:
            print("   [Strategy Test] 未執行")

        if selected_benchmark is not None:
            print("   [Buy & Hold After Cost]")
            print(f"   Return: {selected_benchmark['total_return']*100:.2f}%")
            print(f"   Sharpe: {selected_benchmark['sharpe']:.3f}")
        else:
            print("   [Buy & Hold After Cost] 未產生")

        if excess_return_vs_buy_hold is not None:
            print(f"   Excess Return vs Benchmark: {excess_return_vs_buy_hold*100:.2f}%")
            print(f"   Beats Selected Benchmark: {'YES' if beats_buy_hold else 'NO'}")
        else:
            print("   Excess Return vs Benchmark: 未產生")

        return result



    def run_all_experiments(self):
        df = self.load_and_prepare_data()
        all_results = []

        feature_combos = list(FEATURE_COMBINATIONS.keys())
        model_types = self.config.model.model_types
        horizons = self.config.prediction.horizons
        bear_thresholds = self.config.regime.bear_thresholds

        total_experiments = len(feature_combos) * len(model_types) * len(horizons) * len(bear_thresholds)
        print(f"\n📊 總實驗數: {total_experiments}")

        exp_count = 0
        for feature_combo in feature_combos:
            for model_type in model_types:
                for horizon in horizons:
                    for bear_threshold in bear_thresholds:
                        exp_count += 1
                        print(f"\n[{exp_count}/{total_experiments}]")
                        try:
                            result = self.run_single_experiment(df, feature_combo, model_type, horizon, bear_threshold)
                            all_results.append(result)
                        except Exception as e:
                            print(f"❌ 實驗失敗: {e}")
                            all_results.append({
                                'feature_combo': feature_combo,
                                'model_type': model_type,
                                'horizon': horizon,
                                'bear_threshold': bear_threshold,
                                'error': str(e)
                            })

        self.results = all_results
        self._save_results()
        return all_results

    def _save_results(self):
        serializable_results = []

        for r in self.results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                elif hasattr(v, '__dict__'):
                    sr[k] = str(v)
                else:
                    sr[k] = v
            serializable_results.append(sr)

        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str, ensure_ascii=False)

        summary_data = []
        for r in self.results:
            if 'error' not in r:
                best_params = r.get('best_strategy_params') or {}
                row = {
                    'Experiment Name': r.get('experiment_name'),
                    'Feature Combo': r.get('feature_combo'),
                    'Model Type': r.get('model_type'),
                    'Horizon': r.get('horizon'),
                    'Bear Threshold': r.get('bear_threshold'),
                    'Test Market Regime': r.get('test_market_regime'),
                    'Test Market Return': r.get('test_market_return'),
                    'Test Bull Ratio': r.get('test_bull_ratio'),
                    'Test Bear Ratio': r.get('test_bear_ratio'),
                    'MAE': r.get('mae'),
                    'MSE': r.get('mse'),
                    'RMSE': r.get('rmse'),
                    'Direction Acc': r.get('direction_accuracy'),
                    'Correlation': r.get('correlation'),
                    'R2': r.get('r2'),
                    'Sharpe': r.get('sharpe_ratio'),
                    'Sortino': r.get('sortino_ratio'),
                    'Max DD': r.get('max_drawdown'),
                    'CAGR': r.get('cagr'),
                    'Total Return': r.get('total_return'),
                    'Win Rate': r.get('win_rate'),
                    'N Trades': r.get('n_trades'),
                    'Avg Trade Return': r.get('avg_trade_return'),
                    'Buy & Hold Benchmark Name': r.get('buy_hold_benchmark_name'),
                    'Buy & Hold Return': r.get('buy_hold_total_return'),
                    'Buy & Hold CAGR': r.get('buy_hold_cagr'),
                    'Buy & Hold Sharpe': r.get('buy_hold_sharpe'),
                    'Buy & Hold Max DD': r.get('buy_hold_max_drawdown'),
                    'Buy & Hold Return Raw': r.get('buy_hold_total_return_raw'),
                    'Buy & Hold Return After Cost': r.get('buy_hold_total_return_after_cost'),
                    'Excess Return vs B&H': r.get('excess_return_vs_buy_hold'),
                    'Beats Buy & Hold': r.get('beats_buy_hold'),
                    'Val Sharpe': best_params.get('sharpe'),
                    'Val CAGR': best_params.get('cagr'),
                    'Best Holding Days': best_params.get('holding_days'),
                    'Best Cooldown': best_params.get('cooldown_days'),
                    'Best Threshold': best_params.get('threshold'),
                    'Best Base Exposure': best_params.get('base_exposure'),
                    'Best Defensive Exposure': best_params.get('defensive_exposure'),
                    'Best Signal EMA Alpha': best_params.get('signal_ema_alpha'),
                }
                summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / 'summary.csv', index=False)

        print(f"\n💾 結果已儲存至: {self.output_dir}")


def main():
    config = get_config()
    runner = ExperimentRunner(config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
