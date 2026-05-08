"""
視覺化與報表模組
Visualization and Reporting Module
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# 顏色配置
COLORS = {
    'bull': '#2ecc71',
    'bear': '#e74c3c',
    'neutral': '#3498db',
    'strategy': '#9b59b6',
    'benchmark': '#95a5a6',
    'pred': '#e67e22',
    'actual': '#2c3e50'
}


class Visualizer:
    """視覺化工具"""
    
    def __init__(self, output_dir: Path = None):
        """
        初始化視覺化工具
        
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = output_dir or Path('./figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_price_with_regime(self, df: pd.DataFrame, 
                               threshold: float = -0.15,
                               save_path: str = None):
        """
        繪製價格走勢與熊市區間
        
        Args:
            df: 資料
            threshold: 熊市門檻
            save_path: 儲存路徑
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        threshold_pct = int(abs(threshold) * 100)
        regime_col = f'regime_{threshold_pct}'
        
        # 價格走勢
        ax1 = axes[0]
        ax1.plot(df['Date'], df['Close'], color=COLORS['neutral'], linewidth=1)
        
        # 標記熊市區間
        if regime_col in df.columns:
            bear_periods = df[df[regime_col] == 1]
            for _, period in bear_periods.iterrows():
                ax1.axvspan(period['Date'], period['Date'], 
                           color=COLORS['bear'], alpha=0.3)
        
        ax1.set_ylabel('Price')
        ax1.set_title(f'Price with Bear Market Regions (Threshold: {threshold_pct}%)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[1]
        if 'drawdown' in df.columns:
            ax2.fill_between(df['Date'], df['drawdown'], 0, 
                            color=COLORS['bear'], alpha=0.5)
            ax2.axhline(y=threshold, color='red', linestyle='--', 
                       label=f'Threshold ({threshold_pct}%)')
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Rolling Drawdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Regime
        ax3 = axes[2]
        if regime_col in df.columns:
            ax3.fill_between(df['Date'], df[regime_col], 0,
                            step='pre', color=COLORS['bear'], alpha=0.5)
        ax3.set_ylabel('Bear Market')
        ax3.set_title('Market Regime (0=Bull, 1=Bear)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history: Dict, 
                             title: str = 'Training History',
                             save_path: str = None):
        """
        繪製訓練歷史
        
        Args:
            history: 訓練歷史
            title: 標題
            save_path: 儲存路徑
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 損失
        ax1 = axes[0]
        if 'train_loss' in history:
            ax1.plot(history['train_loss'], label='Train Loss', color=COLORS['strategy'])
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Val Loss', color=COLORS['benchmark'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2 = axes[1]
        if 'train_mae' in history:
            ax2.plot(history['train_mae'], label='Train MAE', color=COLORS['strategy'])
        if 'val_mae' in history:
            ax2.plot(history['val_mae'], label='Val MAE', color=COLORS['benchmark'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Training & Validation MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, predictions: np.ndarray, actuals: np.ndarray,
                        dates: np.ndarray = None,
                        title: str = 'Predictions vs Actuals',
                        save_path: str = None):
        """
        繪製預測值與實際值比較
        
        Args:
            predictions: 預測值
            actuals: 實際值
            dates: 日期
            title: 標題
            save_path: 儲存路徑
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 時間序列比較
        ax1 = axes[0, 0]
        x = dates if dates is not None else np.arange(len(predictions))
        ax1.plot(x, actuals, label='Actual', color=COLORS['actual'], alpha=0.7)
        ax1.plot(x, predictions, label='Predicted', color=COLORS['pred'], alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Return')
        ax1.set_title('Time Series Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 散點圖
        ax2 = axes[0, 1]
        ax2.scatter(actuals, predictions, alpha=0.5, color=COLORS['neutral'])
        
        # 完美預測線
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Prediction vs Actual (Scatter)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 誤差分布
        ax3 = axes[1, 0]
        errors = predictions - actuals
        ax3.hist(errors, bins=50, color=COLORS['neutral'], edgecolor='white', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.axvline(x=np.mean(errors), color='green', linestyle='--', 
                   label=f'Mean: {np.mean(errors):.4f}')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 累積誤差
        ax4 = axes[1, 1]
        cumulative_error = np.cumsum(errors)
        ax4.plot(cumulative_error, color=COLORS['strategy'])
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Cumulative Error')
        ax4.set_title('Cumulative Prediction Error')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_backtest_results(self, result, benchmark_result=None,
                             title: str = 'Backtest Results',
                             save_path: str = None):
        """
        繪製回測結果
        
        Args:
            result: 策略回測結果
            benchmark_result: 基準回測結果
            title: 標題
            save_path: 儲存路徑
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 權益曲線
        ax1 = axes[0, 0]
        dates = result.dates if result.dates is not None else np.arange(len(result.equity_curve))
        ax1.plot(dates, result.equity_curve, label='Strategy', 
                color=COLORS['strategy'], linewidth=2)
        if benchmark_result:
            ax1.plot(dates[:len(benchmark_result.equity_curve)], 
                    benchmark_result.equity_curve, label='Buy & Hold',
                    color=COLORS['benchmark'], linewidth=2, linestyle='--')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[0, 1]
        peak = np.maximum.accumulate(result.equity_curve)
        drawdown = (result.equity_curve - peak) / peak
        ax2.fill_between(dates, drawdown, 0, color=COLORS['bear'], alpha=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown')
        ax2.set_title(f'Drawdown (Max: {result.max_drawdown*100:.2f}%)')
        ax2.grid(True, alpha=0.3)
        
        # 日報酬分布
        ax3 = axes[1, 0]
        ax3.hist(result.daily_returns, bins=50, color=COLORS['neutral'],
                edgecolor='white', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.axvline(x=np.mean(result.daily_returns), color='green', linestyle='--',
                   label=f'Mean: {np.mean(result.daily_returns)*100:.4f}%')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Daily Return Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 持倉狀態
        ax4 = axes[1, 1]
        ax4.fill_between(dates[:len(result.positions)], result.positions, 0,
                        step='pre', color=COLORS['strategy'], alpha=0.5)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Position')
        ax4.set_title('Position Over Time (0=Cash, 1=Long)')
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_experiment_comparison(self, results_df: pd.DataFrame,
                                  metric: str = 'MAE',
                                  group_by: str = 'Model',
                                  title: str = None,
                                  save_path: str = None):
        """
        繪製實驗比較圖
        
        Args:
            results_df: 結果資料框
            metric: 比較指標
            group_by: 分組依據
            title: 標題
            save_path: 儲存路徑
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if group_by in results_df.columns and metric in results_df.columns:
            grouped = results_df.groupby(group_by)[metric].agg(['mean', 'std'])
            
            x = np.arange(len(grouped))
            ax.bar(x, grouped['mean'], yerr=grouped['std'], 
                  capsize=5, color=COLORS['neutral'], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_ylabel(metric)
        
        ax.set_title(title or f'{metric} by {group_by}')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_threshold_analysis(self, results: List[Dict],
                               save_path: str = None):
        """
        繪製門檻分析圖
        
        Args:
            results: 實驗結果列表
            save_path: 儲存路徑
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 準備資料
        data = []
        for r in results:
            if 'error' not in r and 'mae' in r:
                data.append({
                    'Threshold': int(abs(r.get('bear_threshold', 0)) * 100),
                    'Horizon': f"h={r.get('horizon')}",
                    'MAE': r.get('mae')
                })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # 門檻 vs MAE
            ax1 = axes[0]
            for horizon in df['Horizon'].unique():
                subset = df[df['Horizon'] == horizon]
                ax1.plot(subset['Threshold'], subset['MAE'], 
                        marker='o', label=horizon)
            ax1.set_xlabel('Bear Market Threshold (%)')
            ax1.set_ylabel('Test MAE')
            ax1.set_title('Effect of Bear Market Threshold on MAE')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Heatmap
            ax2 = axes[1]
            pivot = df.pivot_table(values='MAE', index='Threshold', 
                                  columns='Horizon', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                       ax=ax2)
            ax2.set_title('MAE Heatmap: Threshold × Horizon')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_architecture(self, save_path: str = None):
        """
        繪製模型架構圖
        
        Args:
            save_path: 儲存路徑
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 繪製模型架構
        components = [
            ('Input\n(seq_len × n_features)', 0.5, 0.9),
            ('Input Projection\n(Linear)', 0.5, 0.8),
            ('Positional\nEncoding', 0.5, 0.7),
            ('Transformer\nEncoder\n(N layers)', 0.5, 0.55),
            ('Flatten /\nAttention Pool', 0.5, 0.4),
            ('Output Layer\n(MLP)', 0.5, 0.25),
            ('Prediction\n(log return)', 0.5, 0.1)
        ]
        
        # Regime 相關組件
        regime_components = [
            ('Regime\nEmbedding', 0.2, 0.7),
            ('Concat', 0.35, 0.65)
        ]
        
        # 繪製主要組件
        for text, x, y in components:
            rect = plt.Rectangle((x-0.12, y-0.04), 0.24, 0.08,
                                fill=True, facecolor=COLORS['neutral'],
                                edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=10)
        
        # 繪製箭頭
        for i in range(len(components)-1):
            ax.annotate('', xy=(components[i+1][1], components[i+1][2]+0.04),
                       xytext=(components[i][1], components[i][2]-0.04),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Regime 組件
        for text, x, y in regime_components:
            rect = plt.Rectangle((x-0.08, y-0.03), 0.16, 0.06,
                                fill=True, facecolor=COLORS['bull'],
                                edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=9)
        
        # 標題
        ax.text(0.5, 0.98, 'Transformer Model Architecture', 
               ha='center', va='top', fontsize=14, fontweight='bold')
        
        ax.text(0.15, 0.98, 'Regime\nComponents', 
               ha='center', va='top', fontsize=10, color=COLORS['bull'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


class ReportGenerator:
    """報表生成器"""
    
    def __init__(self, results: List[Dict], output_dir: Path = None):
        """
        初始化報表生成器
        
        Args:
            results: 實驗結果
            output_dir: 輸出目錄
        """
        self.results = results
        self.output_dir = output_dir or Path('./reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        生成摘要表格
        
        Returns:
            pd.DataFrame: 摘要表格
        """
        data = []
        
        for r in self.results:
            if 'error' in r:
                continue
            
            data.append({
                'Feature Combination': r.get('feature_combo'),
                'Model Type': r.get('model_type'),
                'Horizon': r.get('horizon'),
                'Bear Threshold': f"{int(abs(r.get('bear_threshold', 0))*100)}%",
                'Test MAE': f"{r.get('mae', 0):.6f}",
                'Direction Acc': f"{r.get('direction_accuracy', 0)*100:.2f}%",
                'Sharpe Ratio': f"{r.get('sharpe_ratio', 0):.3f}" if r.get('sharpe_ratio') else 'N/A',
                'Sortino Ratio': f"{r.get('sortino_ratio', 0):.3f}" if r.get('sortino_ratio') else 'N/A',
                'Max Drawdown': f"{r.get('max_drawdown', 0)*100:.2f}%" if r.get('max_drawdown') else 'N/A',
                'CAGR': f"{r.get('cagr', 0)*100:.2f}%" if r.get('cagr') else 'N/A'
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_best_models_report(self) -> str:
        """
        生成最佳模型報告
        
        Returns:
            str: 報告文字
        """
        valid_results = [r for r in self.results if 'error' not in r and 'mae' in r]
        
        if not valid_results:
            return "No valid results to report."
        
        # 按 MAE 排序
        sorted_results = sorted(valid_results, key=lambda x: x.get('mae', float('inf')))
        
        report = []
        report.append("=" * 60)
        report.append("Best Models Report")
        report.append("=" * 60)
        report.append("")
        
        # Top 5
        report.append("Top 5 Models by MAE:")
        report.append("-" * 40)
        
        for i, r in enumerate(sorted_results[:5]):
            report.append(f"\n{i+1}. {r.get('experiment_name', 'Unknown')}")
            report.append(f"   MAE: {r.get('mae', 0):.6f}")
            report.append(f"   Direction Accuracy: {r.get('direction_accuracy', 0)*100:.2f}%")
            if r.get('sharpe_ratio'):
                report.append(f"   Sharpe Ratio: {r.get('sharpe_ratio'):.3f}")
        
        # 按模型類型分組
        report.append("\n" + "=" * 60)
        report.append("Best Model by Type:")
        report.append("-" * 40)
        
        model_types = set(r.get('model_type') for r in valid_results)
        for model_type in model_types:
            type_results = [r for r in valid_results if r.get('model_type') == model_type]
            best = min(type_results, key=lambda x: x.get('mae', float('inf')))
            report.append(f"\n{model_type}:")
            report.append(f"   Best MAE: {best.get('mae', 0):.6f}")
            report.append(f"   Features: {best.get('feature_combo')}")
            report.append(f"   Horizon: h={best.get('horizon')}")
        
        return "\n".join(report)
    
    def generate_threshold_analysis_report(self) -> str:
        """
        生成門檻分析報告
        
        Returns:
            str: 報告文字
        """
        valid_results = [r for r in self.results if 'error' not in r and 'mae' in r]
        
        if not valid_results:
            return "No valid results for threshold analysis."
        
        report = []
        report.append("=" * 60)
        report.append("Threshold Analysis Report")
        report.append("=" * 60)
        
        # 按門檻分組
        thresholds = sorted(set(r.get('bear_threshold') for r in valid_results))
        
        report.append("\nMAE by Bear Market Threshold:")
        report.append("-" * 40)
        
        for threshold in thresholds:
            threshold_results = [r for r in valid_results if r.get('bear_threshold') == threshold]
            avg_mae = np.mean([r.get('mae', 0) for r in threshold_results])
            std_mae = np.std([r.get('mae', 0) for r in threshold_results])
            
            report.append(f"\nThreshold: {int(abs(threshold)*100)}%")
            report.append(f"   Average MAE: {avg_mae:.6f} (±{std_mae:.6f})")
            report.append(f"   Number of experiments: {len(threshold_results)}")
        
        # Horizon 比較
        report.append("\n" + "=" * 60)
        report.append("MAE by Prediction Horizon:")
        report.append("-" * 40)
        
        for horizon in [1, 20]:
            horizon_results = [r for r in valid_results if r.get('horizon') == horizon]
            if horizon_results:
                avg_mae = np.mean([r.get('mae', 0) for r in horizon_results])
                report.append(f"\nHorizon h={horizon}: Average MAE = {avg_mae:.6f}")
        
        return "\n".join(report)
    
    def save_reports(self):
        """儲存所有報告"""
        # 摘要表格
        summary_df = self.generate_summary_table()
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        summary_df.to_html(self.output_dir / 'summary_table.html', index=False)
        
        # 最佳模型報告
        best_report = self.generate_best_models_report()
        with open(self.output_dir / 'best_models_report.txt', 'w') as f:
            f.write(best_report)
        
        # 門檻分析報告
        threshold_report = self.generate_threshold_analysis_report()
        with open(self.output_dir / 'threshold_analysis_report.txt', 'w') as f:
            f.write(threshold_report)
        
        print(f"📝 報告已儲存至: {self.output_dir}")


def generate_all_visualizations(results: List[Dict], output_dir: Path):
    """
    生成所有視覺化圖表
    
    Args:
        results: 實驗結果
        output_dir: 輸出目錄
    """
    visualizer = Visualizer(output_dir)
    
    # 模型架構圖
    visualizer.plot_model_architecture(
        save_path=str(output_dir / 'model_architecture.png')
    )
    
    # 門檻分析
    visualizer.plot_threshold_analysis(
        results,
        save_path=str(output_dir / 'threshold_analysis.png')
    )
    
    # 實驗比較
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        df = pd.DataFrame([{
            'Model': r.get('model_type'),
            'Features': r.get('feature_combo'),
            'MAE': r.get('mae')
        } for r in valid_results if 'mae' in r])
        
        if len(df) > 0:
            visualizer.plot_experiment_comparison(
                df, metric='MAE', group_by='Model',
                title='MAE Comparison by Model Type',
                save_path=str(output_dir / 'model_comparison.png')
            )
    
    print(f"📊 視覺化圖表已生成")


if __name__ == "__main__":
    # 測試視覺化
    visualizer = Visualizer()
    visualizer.plot_model_architecture()
