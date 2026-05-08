"""
實驗分析模組（修正版）
- 納入 n_trades / win_rate / avg_trade_return
- 不再寫死 h=1 vs h=20
- 區分「單點最高」與「整體平均」
- 增加 MaxDD / 低交易次數 sanity check
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")


class ExperimentAnalyzer:
    def __init__(self, results: List[Dict]):
        self.results = [r for r in results if "error" not in r]
        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            rows.append({
                "feature_combo": r.get("feature_combo"),
                "model_type": r.get("model_type"),
                "horizon": r.get("horizon"),
                "threshold": r.get("bear_threshold"),
                "threshold_pct": int(abs(r.get("bear_threshold", 0)) * 100) if r.get("bear_threshold") is not None else None,
                "mae": r.get("mae"),
                "rmse": r.get("rmse"),
                "direction_acc": r.get("direction_accuracy"),
                "correlation": r.get("correlation"),
                "r2": r.get("r2"),
                "sharpe": r.get("sharpe_ratio"),
                "sortino": r.get("sortino_ratio"),
                "max_dd": r.get("max_drawdown"),
                "cagr": r.get("cagr"),
                "total_return": r.get("total_return"),
                "win_rate": r.get("win_rate"),
                "n_trades": r.get("n_trades"),
                "avg_trade_return": r.get("avg_trade_return"),
                "buy_hold_sharpe": r.get("buy_hold_sharpe"),
                "buy_hold_cagr": r.get("buy_hold_cagr"),
            })
        return pd.DataFrame(rows)

    def _title(self, text: str):
        print("\n" + "=" * 72)
        print(text)
        print("=" * 72)

    def summarize_by_feature(self) -> pd.DataFrame:
        self._title("📊 特徵組合整體表現（平均/中位數）")

        agg_spec = {
            "mae": ["mean", "median"],
            "direction_acc": ["mean", "median"],
            "sharpe": ["mean", "median", "max"],
            "cagr": ["mean", "median", "max"],
            "max_dd": ["mean", "median"],
            "n_trades": ["mean", "median", "max"],
            "win_rate": ["mean", "median"],
            "avg_trade_return": ["mean", "median"],
        }

        usable = {k: v for k, v in agg_spec.items() if k in self.df.columns and self.df[k].notna().any()}
        grouped = self.df.groupby("feature_combo").agg(usable)
        grouped.columns = ["_".join(col).strip() for col in grouped.columns]
        sort_col = "sharpe_mean" if "sharpe_mean" in grouped.columns else "mae_mean"
        ascending = False if sort_col == "sharpe_mean" else True
        grouped = grouped.sort_values(sort_col, ascending=ascending)

        print(grouped.round(6).to_string())
        return grouped

    def summarize_by_horizon(self) -> pd.DataFrame:
        self._title("📊 各 Horizon 整體表現")

        agg_spec = {
            "mae": ["mean", "median"],
            "direction_acc": ["mean", "median"],
            "sharpe": ["mean", "median", "max"],
            "cagr": ["mean", "median", "max"],
            "max_dd": ["mean", "median"],
            "n_trades": ["mean", "median", "max"],
            "win_rate": ["mean", "median"],
        }

        usable = {k: v for k, v in agg_spec.items() if k in self.df.columns and self.df[k].notna().any()}
        grouped = self.df.groupby("horizon").agg(usable)
        grouped.columns = ["_".join(col).strip() for col in grouped.columns]
        print(grouped.round(6).to_string())
        return grouped

    def top_strategy_results(self, min_trades: int = 5, topk: int = 10) -> pd.DataFrame:
        self._title(f"📊 Top {topk} 策略結果（已過濾 N Trades < {min_trades}）")

        df = self.df.copy()

        if "n_trades" in df.columns and df["n_trades"].notna().any():
            df = df[(df["n_trades"].isna()) | (df["n_trades"] >= min_trades)]

        sort_cols = []
        ascending = []

        if "sharpe" in df.columns and df["sharpe"].notna().any():
            sort_cols.append("sharpe")
            ascending.append(False)
        if "cagr" in df.columns and df["cagr"].notna().any():
            sort_cols.append("cagr")
            ascending.append(False)
        if "max_dd" in df.columns and df["max_dd"].notna().any():
            sort_cols.append("max_dd")
            ascending.append(False)
        if "mae" in df.columns and df["mae"].notna().any():
            sort_cols.append("mae")
            ascending.append(True)

        if not sort_cols:
            print("沒有可排序的欄位")
            return pd.DataFrame()

        cols = [
            "feature_combo", "model_type", "horizon", "threshold_pct",
            "mae", "direction_acc", "sharpe", "cagr", "max_dd",
            "n_trades", "win_rate", "avg_trade_return"
        ]
        cols = [c for c in cols if c in df.columns]

        top_df = df.sort_values(sort_cols, ascending=ascending).head(topk)[cols]
        print(top_df.round(6).to_string(index=False))
        return top_df

    def sanity_checks(self) -> Dict:
        self._title("📊 Sanity Check")

        checks = {}

        if "max_dd" in self.df.columns and self.df["max_dd"].notna().any():
            dup = (
                self.df.dropna(subset=["max_dd"])
                .groupby("max_dd")
                .agg(
                    rows=("max_dd", "size"),
                    horizon_nunique=("horizon", "nunique"),
                    feature_nunique=("feature_combo", "nunique"),
                )
                .reset_index()
            )
            suspicious_dup = dup[(dup["rows"] >= 3) & (dup["horizon_nunique"] >= 2)]
            checks["duplicate_maxdd"] = suspicious_dup
            if len(suspicious_dup) > 0:
                print("⚠️ 不同 horizon 出現重複 MaxDD：")
                print(suspicious_dup.round(6).to_string(index=False))
            else:
                print("✓ 沒有明顯的重複 MaxDD 異常")

        if {"sharpe", "n_trades"}.issubset(self.df.columns):
            low_trade_high_sharpe = self.df[
                self.df["sharpe"].notna() &
                self.df["n_trades"].notna() &
                (self.df["sharpe"] > 1.0) &
                (self.df["n_trades"] < 5)
            ]
            checks["low_trade_high_sharpe"] = low_trade_high_sharpe
            if len(low_trade_high_sharpe) > 0:
                print("\n⚠️ 高 Sharpe 但交易次數過少：")
                cols = [c for c in [
                    "feature_combo", "model_type", "horizon", "threshold_pct",
                    "sharpe", "cagr", "max_dd", "n_trades"
                ] if c in low_trade_high_sharpe.columns]
                print(low_trade_high_sharpe[cols].round(6).to_string(index=False))
            else:
                print("\n✓ 沒有發現高 Sharpe 但低交易次數的明顯異常")

        return checks

    def run_full_analysis(self) -> Dict:
        self._title("🔬 完整實驗分析（交易版）")

        if self.df.empty:
            print("沒有可分析的結果")
            return {}

        by_feature = self.summarize_by_feature()
        by_horizon = self.summarize_by_horizon()
        top_results = self.top_strategy_results(min_trades=5, topk=10)
        checks = self.sanity_checks()

        self._title("📋 分析結論")

        if not by_feature.empty:
            best_feature = by_feature.index[0]
            if "sharpe_mean" in by_feature.columns:
                print(f"整體平均 Sharpe 最佳特徵組合：{best_feature}")
            elif "mae_mean" in by_feature.columns:
                print(f"整體平均 MAE 最佳特徵組合：{best_feature}")

        if not top_results.empty:
            top_row = top_results.iloc[0]
            print(
                f"單一最佳策略：{top_row.get('feature_combo')} / "
                f"h={top_row.get('horizon')} / "
                f"threshold={top_row.get('threshold_pct')}%"
            )
            if "sharpe" in top_row:
                print(f"  Sharpe = {top_row.get('sharpe'):.6f}")
            if "n_trades" in top_row and pd.notna(top_row.get("n_trades")):
                print(f"  N Trades = {int(top_row.get('n_trades'))}")

        return {
            "by_feature": by_feature,
            "by_horizon": by_horizon,
            "top_results": top_results,
            "checks": checks,
        }


def analyze_experiments(results: List[Dict]) -> Dict:
    analyzer = ExperimentAnalyzer(results)
    return analyzer.run_full_analysis()


if __name__ == "__main__":
    print("實驗分析模組載入成功")
