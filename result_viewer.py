"""
result_viewer.py -- Experiment Results Visualization

A standalone CLI script that reads a RESULTS/ run folder and displays
human-friendly graphs (training curves, GA fitness) and tables (metrics, config).

Usage:
    python result_viewer.py RESULTS/baseline_jpm_20260521_094100
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def print_table(title: str, data: dict) -> None:
    """Print a dictionary as a formatted console table.

    Args:
        title: Table heading.
        data: Key-value pairs to display.
    """
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 10
    for key, value in data.items():
        if key in ("rmse_bps", "mae_bps"):
            print(f"  {str(key):<{max_key_len + 2}} {value:.2f} bps")
        elif key in ("trade_rate", "precision_at_conf", "m2_precision_at_0.5"):
            print(f"  {str(key):<{max_key_len + 2}} {value * 100:.1f}%")
        elif key in ("annualized_sharpe",):
            print(f"  {str(key):<{max_key_len + 2}} {value:+.2f}")
        elif isinstance(value, float):
            print(f"  {str(key):<{max_key_len + 2}} {value:.6f}")
        else:
            print(f"  {str(key):<{max_key_len + 2}} {value}")
    print(f"{'=' * 50}\n")


def plot_training_curves(run_dir: Path) -> None:
    """Plot train/val loss curves from training_log.csv.

    Saves the chart as training_curves.png in the run folder.
    """
    csv_path = run_dir / "training_log.csv"
    if not csv_path.exists():
        print("[SKIP] No training_log.csv found.")
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
    plt.show()


def plot_ga_fitness(run_dir: Path) -> None:
    """Plot GA fitness evolution from ga_history.csv.

    Saves the chart as ga_fitness.png in the run folder.
    """
    csv_path = run_dir / "ga_history.csv"
    if not csv_path.exists():
        print("[SKIP] No ga_history.csv found (not a GA run).")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["generation"], df["min_fitness"], label="Best (Min)", color="#27AE60", linewidth=2)
    ax.plot(df["generation"], df["avg_fitness"], label="Average", color="#F39C12", linewidth=2)
    ax.plot(df["generation"], df["max_fitness"], label="Worst (Max)", color="#E74C3C", linewidth=2, alpha=0.6)
    ax.fill_between(
        df["generation"],
        df["min_fitness"],
        df["max_fitness"],
        alpha=0.1, color="#3498DB"
    )
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (RMSE + Penalty)", fontsize=12)
    ax.set_title("GA Fitness Evolution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = run_dir / "ga_fitness.png"
    fig.savefig(save_path, dpi=150)
    print(f"[OK] GA fitness chart saved to {save_path}")
    plt.show()


def show_metrics(run_dir: Path) -> None:
    """Print metrics.json as a formatted table."""
    path = run_dir / "metrics.json"
    if not path.exists():
        print("[SKIP] No metrics.json found.")
        return
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print_table("Test Metrics", metrics)


def show_config(run_dir: Path) -> None:
    """Print config.json as a formatted table."""
    path = run_dir / "config.json"
    if not path.exists():
        print("[SKIP] No config.json found.")
        return
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print_table("Experiment Configuration", config)


def show_best_chromosome(run_dir: Path) -> None:
    """Print best_chromosome.json as a formatted table."""
    path = run_dir / "best_chromosome.json"
    if not path.exists():
        print("[SKIP] No best_chromosome.json found (not a GA run).")
        return
    with open(path, "r", encoding="utf-8") as f:
        chrom = json.load(f)

    # Separate features from hyperparameters for clearer display
    features = chrom.pop("selected_features", None)
    print_table("Best Chromosome - Hyperparameters", chrom)

    if features:
        print(f"  Selected Features ({len(features)}):")
        for i, feat in enumerate(features, 1):
            print(f"    {i:>2}. {feat}")
        print()


def view_run(run_dir: Path) -> None:
    """Display all available results for a single run folder.

    Args:
        run_dir: Path to the run folder inside RESULTS/.
    """
    print(f"\n{'#' * 60}")
    print(f"  Results Viewer: {run_dir.name}")
    print(f"{'#' * 60}")

    show_config(run_dir)
    show_metrics(run_dir)
    show_best_chromosome(run_dir)
    plot_training_curves(run_dir)
    plot_ga_fitness(run_dir)


def main() -> None:
    """CLI entry point."""
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
