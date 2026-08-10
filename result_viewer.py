"""
result_viewer.py -- Experiment Results Visualization

A standalone CLI script that reads a RESULTS/ run folder and displays
human-friendly graphs (training curves, GA fitness, equity curves, confusion matrices) 
and tables (metrics, config).

Usage:
    python result_viewer.py RESULTS/baseline_jpm_20260521_094100
"""

import json
import sys
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def print_table(title: str, data: dict) -> None:
    """Print a dictionary as a formatted console table."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 10
    
    pct_keys = (
        "trade_rate", "precision_at_conf", "m2_precision_at_0.5", 
        "win_rate", "avg_win_pct", "avg_loss_pct", "gross_profit_pct", 
        "gross_loss_pct", "total_return_pct", "max_drawdown_pct"
    )
    
    for key, value in data.items():
        if key in ("rmse_bps", "mae_bps"):
            print(f"  {str(key):<{max_key_len + 2}} {value:.2f} bps")
        elif key in pct_keys:
            print(f"  {str(key):<{max_key_len + 2}} {value * 100:.2f}%")
        elif key in ("annualized_sharpe", "profit_factor"):
            print(f"  {str(key):<{max_key_len + 2}} {value:+.2f}")
        elif isinstance(value, float):
            print(f"  {str(key):<{max_key_len + 2}} {value:.6f}")
        else:
            print(f"  {str(key):<{max_key_len + 2}} {value}")
    print(f"{'=' * 50}\n")


def plot_training_curves(run_dir: Path) -> None:
    csv_path = run_dir / "training_log.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#4A90D9", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#E74C3C", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = run_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] Training curves saved to {save_path}")


def plot_ga_fitness(run_dir: Path) -> None:
    csv_path = run_dir / "ga_history.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["generation"], df["min_fitness"], label="Best (Min)", color="#27AE60", linewidth=2)
    ax.plot(df["generation"], df["avg_fitness"], label="Average", color="#F39C12", linewidth=2)
    ax.plot(df["generation"], df["max_fitness"], label="Worst (Max)", color="#E74C3C", linewidth=2, alpha=0.6)
    ax.fill_between(
        df["generation"], df["min_fitness"], df["max_fitness"], alpha=0.1, color="#3498DB"
    )
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title("GA Fitness Evolution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = run_dir / "ga_fitness.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] GA fitness chart saved to {save_path}")


def plot_cumulative_equity(run_dir: Path) -> None:
    csv_path = run_dir / "test_trade_log.csv"
    if not csv_path.exists(): return
        
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    df = df.sort_values("Timestamp")
    strategy_returns = df.groupby("Timestamp")["PnL"].sum()
    baseline_returns = df.groupby("Timestamp")["Actual_Fwd_Return"].mean()
    
    strat_cum = strategy_returns.cumsum()
    base_cum = baseline_returns.cumsum()
    
    running_max = strat_cum.cummax()
    drawdown = running_max - strat_cum
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(strat_cum.values, label="Strategy Cumulative PnL", color="#27AE60", linewidth=2)
    ax1.plot(base_cum.values, label="Buy & Hold (Avg Return)", color="#95A5A6", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Cumulative Return", fontsize=12)
    ax1.set_title("Strategy Equity Curve vs Baseline", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.fill_between(range(len(drawdown)), -drawdown.values, 0, color="#E74C3C", alpha=0.5, label="Drawdown")
    ax2.set_xlabel("Time (Trade Index)", fontsize=12)
    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_path = run_dir / "equity_curve.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] Equity curve saved to {save_path}")


def plot_ticker_pnl(run_dir: Path) -> None:
    csv_path = run_dir / "test_trade_log.csv"
    if not csv_path.exists(): return
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    pnl_by_ticker = df.groupby("Ticker")["PnL"].sum().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(pnl_by_ticker)*0.3)))
    colors = ["#E74C3C" if val < 0 else "#27AE60" for val in pnl_by_ticker.values]
    pnl_by_ticker.plot(kind="barh", ax=ax, color=colors)
    
    ax.set_xlabel("Total PnL", fontsize=12)
    ax.set_ylabel("Ticker", fontsize=12)
    ax.set_title("Total PnL by Ticker", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    fig.tight_layout()
    save_path = run_dir / "pnl_by_ticker.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] Ticker PnL chart saved to {save_path}")


def plot_confidence_distribution(run_dir: Path) -> None:
    csv_path = run_dir / "test_trade_log.csv"
    if not csv_path.exists(): return
    df = pd.read_csv(csv_path)
    if df.empty or "M2_Conf_Score" not in df.columns: return
    
    conf = df["M2_Conf_Score"].dropna()
    if conf.empty: return
    
    threshold = 0.70
    conf_path = run_dir / "config.json"
    if conf_path.exists():
        with open(conf_path, "r") as f:
            c = json.load(f)
            threshold = c.get("confidence_threshold", 0.70)
            
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(conf, bins=50, color="#3498DB", alpha=0.7, edgecolor="black")
    ax.axvline(threshold, color="#E74C3C", linestyle="dashed", linewidth=2, label=f"Threshold ({threshold})")
    
    ax.set_xlabel("M2 Confidence Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Model 2 Confidence Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_path = run_dir / "confidence_dist.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] Confidence distribution saved to {save_path}")


def plot_confusion_matrix(run_dir: Path) -> None:
    path = run_dir / "confusion_matrix.json"
    if not path.exists(): return
    with open(path, "r") as f:
        cm = json.load(f)
        
    labels = ["DOWN", "NEUTRAL", "UP"]
    matrix = np.array([
        [cm.get("actual_down_pred_down", 0), cm.get("actual_down_pred_neutral", 0), cm.get("actual_down_pred_up", 0)],
        [cm.get("actual_neutral_pred_down", 0), cm.get("actual_neutral_pred_neutral", 0), cm.get("actual_neutral_pred_up", 0)],
        [cm.get("actual_up_pred_down", 0), cm.get("actual_up_pred_neutral", 0), cm.get("actual_up_pred_up", 0)]
    ])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)
    
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Actual Class", fontsize=12)
    ax.set_title("Model 1 Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
    
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, str(val), va='center', ha='center', color=color, fontsize=12, fontweight='bold')
            
    fig.tight_layout()
    save_path = run_dir / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] Confusion matrix saved to {save_path}")


def plot_ga_scatter_matrix(run_dir: Path) -> None:
    path = run_dir / "ga_population_history.jsonl"
    if not path.exists(): return
    
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    if not records: return
    
    df = pd.DataFrame(records)
    if "m2_sharpe" not in df.columns: return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if "num_selected_features" in df.columns:
        ax1.scatter(df["num_selected_features"], df["m2_sharpe"], alpha=0.5, color="#8E44AD")
        ax1.set_xlabel("Number of Selected Features")
        ax1.set_ylabel("M2 Sharpe Ratio")
        ax1.set_title("Feature Count vs Sharpe")
        ax1.grid(True, alpha=0.3)
        
    if "threshold_multiplier" in df.columns and "generation" in df.columns:
        sc = ax2.scatter(df["threshold_multiplier"], df["m2_sharpe"], c=df["generation"], cmap="viridis", alpha=0.7)
        ax2.set_xlabel("Threshold Multiplier")
        ax2.set_ylabel("M2 Sharpe Ratio")
        ax2.set_title("Vol Threshold vs Sharpe")
        fig.colorbar(sc, ax=ax2, label="Generation")
        ax2.grid(True, alpha=0.3)
        
    fig.tight_layout()
    save_path = run_dir / "ga_search_space.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] GA scatter matrix saved to {save_path}")


def show_config(run_dir: Path) -> None:
    path = run_dir / "config.json"
    if not path.exists(): return
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print_table("Experiment Configuration", config)


def show_metrics(run_dir: Path) -> None:
    path = run_dir / "metrics.json"
    if not path.exists(): return
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print_table("Test Metrics", metrics)


def show_pnl_stats(run_dir: Path) -> None:
    path = run_dir / "pnl_stats.json"
    if not path.exists(): return
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    print_table("PnL & Trading Statistics", stats)


def show_best_chromosome(run_dir: Path) -> None:
    path = run_dir / "best_chromosome.json"
    if not path.exists(): return
    with open(path, "r", encoding="utf-8") as f:
        chrom = json.load(f)

    features = chrom.pop("selected_features", None)
    print_table("Best Chromosome - Hyperparameters", chrom)

    if features:
        print(f"  Selected Features ({len(features)}):")
        for i, feat in enumerate(features, 1):
            print(f"    {i:>2}. {feat}")
        print()


def view_run(run_dir: Path) -> None:
    print(f"\n{'#' * 60}")
    print(f"  Results Viewer: {run_dir.name}")
    print(f"{'#' * 60}")

    # Tables
    show_config(run_dir)
    show_metrics(run_dir)
    show_pnl_stats(run_dir)
    show_best_chromosome(run_dir)

    # Plots
    plot_training_curves(run_dir)
    plot_ga_fitness(run_dir)
    plot_cumulative_equity(run_dir)
    plot_ticker_pnl(run_dir)
    plot_confidence_distribution(run_dir)
    plot_confusion_matrix(run_dir)
    plot_ga_scatter_matrix(run_dir)

    # Show all generated plots if any were created
    if any(plt.get_fignums()):
        print("\n[INFO] Opening all generated charts...")
        plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python result_viewer.py <path_to_run_folder>")
        print("Example: python result_viewer.py RESULTS/baseline_jpm_20260521_094100")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Error: '{run_dir}' is not a valid directory.")
        sys.exit(1)

    view_run(run_dir)


if __name__ == "__main__":
    main()
