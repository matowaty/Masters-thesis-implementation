"""
main.py -- Orchestration Entry Point

The master script that orchestrates the entire pipeline:
Data Ingestion -> Feature Engineering -> Windowing/Scaling ->
Model Training -> GA Optimization -> Final Evaluation.
"""

import argparse
import logging
import sys

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from ga_optimizer import GAOptimizer
from model_builder import BiLSTMModel, BiLSTMAttentionModel, get_device
from result_logger import ResultLogger
from trainer import Trainer

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format for the entire pipeline.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        stream=sys.stdout,
    )


def _prepare_data(data_path: str = "DATA/JPM.csv"):
    """Shared data preparation: load, feature engineer, compute targets, split.

    Returns:
        Tuple of (dp, fe, train_df, val_df, test_df, feature_cols).
    """
    dp = DataProcessor(scaler_type="standard")
    df = dp.load_data(data_path)
    df = dp.handle_missing_intervals(df)

    fe = FeatureEngineer()
    df = fe.add_all_features(df)
    feature_cols = fe.get_feature_names()

    df = dp.compute_targets(df)

    train_df, val_df, test_df = dp.chronological_split(df)
    return dp, fe, train_df, val_df, test_df, feature_cols


def run_baseline_experiment() -> None:
    """Run the Phase 1 baseline experiment with fixed hyperparameters.

    Uses all generated features, window=12, target=t+1, and records
    MSE, MAE, RMSE, R2, and Directional Accuracy.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: BASELINE EXPERIMENT")
    logger.info("=" * 60)

    # --- Configuration ---
    DATA_PATH = "DATA/JPM.csv"
    WINDOW_SIZE = 12
    TARGET_COL = "Target_1_Tick"
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    device = get_device()

    # Result logger
    rl = ResultLogger("baseline", "jpm")
    rl.log_config({
        "data_path": DATA_PATH,
        "window_size": WINDOW_SIZE,
        "target": TARGET_COL,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
    })

    # 1. Data preparation
    dp, fe, train_df, val_df, test_df, feature_cols = _prepare_data(DATA_PATH)

    # 2. Scale features
    dp.fit_scaler(train_df, feature_cols)
    X_train_scaled = dp.transform(train_df, feature_cols)
    X_val_scaled = dp.transform(val_df, feature_cols)
    X_test_scaled = dp.transform(test_df, feature_cols)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # 3. Create sliding windows
    X_train, y_train = dp.create_windows(X_train_scaled, y_train, WINDOW_SIZE)
    X_val, y_val = dp.create_windows(X_val_scaled, y_val, WINDOW_SIZE)
    X_test, y_test = dp.create_windows(X_test_scaled, y_test, WINDOW_SIZE)

    # 4. Build model
    num_features = X_train.shape[2]
    model = BiLSTMModel(
        input_size=num_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        loss_fn="mse"
    )

    # 5. Train
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE
    )

    logger.info("Training pure baseline BiLSTM model...")
    best_val_loss = trainer.train(
        train_loader, val_loader,
        epochs=NUM_EPOCHS, patience=10, verbose=True,
        result_logger=rl,
    )

    # 6. Evaluate
    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION (TEST SET)")
    logger.info("=" * 60)
    metrics = trainer.evaluate(X_test, y_test)

    # 7. Save everything
    chk_path = "checkpoints/baseline_jpm.pt"
    trainer.save_checkpoint(chk_path, dp, feature_cols, WINDOW_SIZE)
    rl.log_metrics(metrics)
    rl.copy_checkpoint(chk_path)

    logger.info("Baseline experiment complete. Results saved to %s", rl.get_run_dir())


def run_ga_optimization(
    mode: str = "features_only",
    ga_epochs: int = 5,
    data_fraction: float = 0.2,
    population_size: int = 10,
    num_generations: int = 5,
) -> None:
    """Run GA optimization and then retrain the best chromosome on full data.

    Args:
        mode: "features_only" or "full".
        ga_epochs: Epochs per individual during GA search.
        data_fraction: Fraction of data for GA fitness evaluations.
        population_size: GA population size.
        num_generations: GA generations.
    """
    label = f"ga_{mode}"
    logger.info("=" * 60)
    logger.info("GA OPTIMIZATION -- mode=%s", mode)
    logger.info("=" * 60)

    # Result logger
    rl = ResultLogger(label, "jpm")
    rl.log_config({
        "mode": mode,
        "ga_epochs": ga_epochs,
        "data_fraction": data_fraction,
        "population_size": population_size,
        "num_generations": num_generations,
        "data_path": "DATA/JPM.csv",
    })

    # 1. Data preparation
    dp, fe, train_df, val_df, test_df, feature_cols = _prepare_data("DATA/JPM.csv")

    # 2. Run GA
    ga = GAOptimizer(
        feature_names=feature_cols,
        train_df=train_df,
        val_df=val_df,
        mode=mode,
        population_size=population_size,
        num_generations=num_generations,
        ga_epochs=ga_epochs,
        data_fraction=data_fraction,
        result_logger=rl,
    )
    result = ga.run()

    # 3. Retrain the best chromosome on the FULL dataset
    best = result["best_chromosome"]
    logger.info("=" * 60)
    logger.info("RETRAINING BEST CHROMOSOME ON FULL DATA")
    logger.info("=" * 60)

    active_features = best.get("selected_features", feature_cols)
    window_size = best.get("window_size", 12)
    hidden_units = best.get("hidden_units", 64)
    dropout = best.get("dropout", 0.2)
    learning_rate = best.get("learning_rate", 1e-3)
    look_forward = best.get("look_forward", 1)
    target_col = f"Target_{look_forward}_Tick"

    device = get_device()

    # Scale on full training data
    dp_full = DataProcessor(scaler_type="standard")
    dp_full.fit_scaler(train_df, active_features)
    X_train_scaled = dp_full.transform(train_df, active_features)
    X_val_scaled = dp_full.transform(val_df, active_features)
    X_test_scaled = dp_full.transform(test_df, active_features)

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    X_train_w, y_train_w = dp_full.create_windows(X_train_scaled, y_train, window_size)
    X_val_w, y_val_w = dp_full.create_windows(X_val_scaled, y_val, window_size)
    X_test_w, y_test_w = dp_full.create_windows(X_test_scaled, y_test, window_size)

    model = BiLSTMModel(
        input_size=len(active_features),
        hidden_size=hidden_units,
        num_layers=2,
        dropout=dropout,
    )

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        loss_fn="mse",
    )

    train_loader, val_loader = trainer.create_dataloaders(
        X_train_w, y_train_w, X_val_w, y_val_w, batch_size=64,
    )

    trainer.train(
        train_loader, val_loader,
        epochs=50, patience=10, verbose=True,
        result_logger=rl,
    )

    # Evaluate
    logger.info("=" * 60)
    logger.info("GA BEST CHROMOSOME -- FINAL TEST EVALUATION")
    logger.info("=" * 60)
    metrics = trainer.evaluate(X_test_w, y_test_w)

    chk_path = f"checkpoints/{label}_best_jpm.pt"
    trainer.save_checkpoint(chk_path, dp_full, active_features, window_size)
    rl.log_metrics(metrics)
    rl.copy_checkpoint(chk_path)

    logger.info("GA experiment complete. Results saved to %s", rl.get_run_dir())


def run_ga_fast_test() -> None:
    """Quick GA test using 20% data and 5 epochs per individual.

    For local CPU testing to verify the pipeline works.
    """
    run_ga_optimization(
        mode="features_only",
        ga_epochs=5,
        data_fraction=0.2,
        population_size=6,
        num_generations=3,
    )


def run_ga_full_gpu() -> None:
    """Full GA run using 100% data and 50 epochs per individual.

    Meant for deployment on an external GPU. No shortcuts.
    """
    run_ga_optimization(
        mode="full",
        ga_epochs=50,
        data_fraction=1.0,
        population_size=20,
        num_generations=30,
    )


def run_time_horizon_experiment() -> None:
    """Run Experiment Phase 2: Time Horizon Sweep.

    Trains 3 separate baseline BiLSTMs -- one for each prediction
    horizon (t+1, t+2, t+3) -- with identical hyperparameters.
    Measures how prediction error degrades as the horizon extends.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: TIME HORIZON SWEEP")
    logger.info("=" * 60)

    DATA_PATH = "DATA/JPM.csv"
    WINDOW_SIZE = 12
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    device = get_device()
    dp, fe, train_df, val_df, test_df, feature_cols = _prepare_data(DATA_PATH)

    dp.fit_scaler(train_df, feature_cols)
    X_train_scaled = dp.transform(train_df, feature_cols)
    X_val_scaled = dp.transform(val_df, feature_cols)
    X_test_scaled = dp.transform(test_df, feature_cols)

    for horizon in [1, 2, 3]:
        target_col = f"Target_{horizon}_Tick"
        logger.info("-" * 40)
        logger.info("Training for horizon: %s", target_col)
        logger.info("-" * 40)

        rl = ResultLogger(f"time_horizon_t{horizon}", "jpm")
        rl.log_config({
            "data_path": DATA_PATH,
            "window_size": WINDOW_SIZE,
            "target": target_col,
            "horizon": horizon,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
        })

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        X_train_w, y_train_w = dp.create_windows(X_train_scaled, y_train, WINDOW_SIZE)
        X_val_w, y_val_w = dp.create_windows(X_val_scaled, y_val, WINDOW_SIZE)
        X_test_w, y_test_w = dp.create_windows(X_test_scaled, y_test, WINDOW_SIZE)

        model = BiLSTMModel(
            input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2,
        )
        trainer = Trainer(model=model, device=device, learning_rate=LEARNING_RATE)

        train_loader, val_loader = trainer.create_dataloaders(
            X_train_w, y_train_w, X_val_w, y_val_w, batch_size=BATCH_SIZE,
        )
        trainer.train(
            train_loader, val_loader,
            epochs=NUM_EPOCHS, patience=10, verbose=True, result_logger=rl,
        )

        metrics = trainer.evaluate(X_test_w, y_test_w)
        chk_path = f"checkpoints/time_horizon_t{horizon}_jpm.pt"
        trainer.save_checkpoint(chk_path, dp, feature_cols, WINDOW_SIZE)
        rl.log_metrics(metrics)
        rl.copy_checkpoint(chk_path)

    logger.info("Time horizon sweep complete.")


def run_attention_comparison() -> None:
    """Run Research Question #4: BiLSTM vs BiLSTM+Attention comparison.

    Trains both models with identical hyperparameters and compares
    test-set metrics to determine if Attention improves results.
    """
    logger.info("=" * 60)
    logger.info("ATTENTION MECHANISM COMPARISON")
    logger.info("=" * 60)

    DATA_PATH = "DATA/JPM.csv"
    WINDOW_SIZE = 12
    TARGET_COL = "Target_1_Tick"
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    device = get_device()
    dp, fe, train_df, val_df, test_df, feature_cols = _prepare_data(DATA_PATH)

    dp.fit_scaler(train_df, feature_cols)
    X_train_scaled = dp.transform(train_df, feature_cols)
    X_val_scaled = dp.transform(val_df, feature_cols)
    X_test_scaled = dp.transform(test_df, feature_cols)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    X_train_w, y_train_w = dp.create_windows(X_train_scaled, y_train, WINDOW_SIZE)
    X_val_w, y_val_w = dp.create_windows(X_val_scaled, y_val, WINDOW_SIZE)
    X_test_w, y_test_w = dp.create_windows(X_test_scaled, y_test, WINDOW_SIZE)

    model_classes = {
        "bilstm": BiLSTMModel,
        "bilstm_attn": BiLSTMAttentionModel,
    }

    for label, ModelClass in model_classes.items():
        logger.info("-" * 40)
        logger.info("Training model: %s", label)
        logger.info("-" * 40)

        rl = ResultLogger(f"attention_{label}", "jpm")
        rl.log_config({
            "data_path": DATA_PATH,
            "model": label,
            "window_size": WINDOW_SIZE,
            "target": TARGET_COL,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
        })

        model = ModelClass(
            input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2,
        )
        trainer = Trainer(model=model, device=device, learning_rate=LEARNING_RATE)

        train_loader, val_loader = trainer.create_dataloaders(
            X_train_w, y_train_w, X_val_w, y_val_w, batch_size=BATCH_SIZE,
        )
        trainer.train(
            train_loader, val_loader,
            epochs=NUM_EPOCHS, patience=10, verbose=True, result_logger=rl,
        )

        metrics = trainer.evaluate(X_test_w, y_test_w)
        chk_path = f"checkpoints/attention_{label}_jpm.pt"
        trainer.save_checkpoint(chk_path, dp, feature_cols, WINDOW_SIZE)
        rl.log_metrics(metrics)
        rl.copy_checkpoint(chk_path)

    logger.info("Attention comparison complete. Use result_viewer.py to compare.")


def run_pipeline() -> None:
    """Execute the full forecasting pipeline end-to-end."""
    run_baseline_experiment()


# ------------------------------------------------------------------
# Multi-Stock Experiments (Universal Model)
# ------------------------------------------------------------------

def _prepare_multi_data(data_dir: str = "DATA"):
    """Multi-stock data preparation: load all, feature engineer, split per-stock.

    Returns:
        Tuple of (dp, train_dfs, val_dfs, test_dfs, feature_cols).
    """
    dp = DataProcessor(scaler_type="standard")
    fe = FeatureEngineer()

    stock_dfs, feature_cols = dp.load_and_engineer_all(data_dir, fe)
    train_dfs, val_dfs, test_dfs = dp.split_all_stocks(stock_dfs)

    return dp, train_dfs, val_dfs, test_dfs, feature_cols


def run_baseline_multi() -> None:
    """Run baseline experiment on ALL stocks combined (universal model).

    Uses per-stock scaling and boundary-safe windowing so that sliding
    windows never cross stock boundaries.  The model architecture and
    hyperparameters are identical to the single-stock baseline.
    """
    logger.info("=" * 60)
    logger.info("MULTI-STOCK BASELINE EXPERIMENT")
    logger.info("=" * 60)

    DATA_DIR = "DATA"
    WINDOW_SIZE = 12
    TARGET_COL = "Target_1_Tick"
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    device = get_device()

    rl = ResultLogger("baseline_multi", "all_stocks")
    rl.log_config({
        "data_dir": DATA_DIR,
        "window_size": WINDOW_SIZE,
        "target": TARGET_COL,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "scaling": "per-stock",
        "mode": "multi-stock",
    })

    dp, train_dfs, val_dfs, test_dfs, feature_cols = _prepare_multi_data(DATA_DIR)

    data = dp.scale_and_window_multi(
        train_dfs, val_dfs, test_dfs,
        feature_cols, TARGET_COL, WINDOW_SIZE,
    )

    # Build model — input shape is still [batch, window, features]
    num_features = data["X_train"].shape[2]
    model = BiLSTMModel(
        input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2,
    )

    trainer = Trainer(model=model, device=device, learning_rate=LEARNING_RATE)

    train_loader, val_loader = trainer.create_dataloaders(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        batch_size=BATCH_SIZE,
    )

    logger.info("Training universal BiLSTM on %d stocks...", len(train_dfs))
    trainer.train(
        train_loader, val_loader,
        epochs=NUM_EPOCHS, patience=10, verbose=True,
        result_logger=rl,
    )

    logger.info("=" * 60)
    logger.info("MULTI-STOCK BASELINE EVALUATION (TEST SET)")
    logger.info("=" * 60)
    metrics = trainer.evaluate(data["X_test"], data["y_test"])

    chk_path = "checkpoints/baseline_multi_all.pt"
    trainer.save_checkpoint(chk_path, dp, feature_cols, WINDOW_SIZE)
    rl.log_metrics(metrics)
    rl.copy_checkpoint(chk_path)

    logger.info("Multi-stock baseline complete. Results saved to %s", rl.get_run_dir())


def run_ga_optimization_multi(
    mode: str = "features_only",
    ga_epochs: int = 5,
    data_fraction: float = 0.2,
    population_size: int = 10,
    num_generations: int = 5,
) -> None:
    """Run GA optimization on ALL stocks combined (universal model).

    Args:
        mode: "features_only" or "full".
        ga_epochs: Epochs per individual during GA search.
        data_fraction: Fraction of per-stock data for GA fitness evaluations.
        population_size: GA population size.
        num_generations: GA generations.
    """
    label = f"ga_{mode}_multi"
    logger.info("=" * 60)
    logger.info("GA MULTI-STOCK OPTIMIZATION -- mode=%s", mode)
    logger.info("=" * 60)

    rl = ResultLogger(label, "all_stocks")
    rl.log_config({
        "mode": mode,
        "ga_epochs": ga_epochs,
        "data_fraction": data_fraction,
        "population_size": population_size,
        "num_generations": num_generations,
        "data_dir": "DATA",
        "scaling": "per-stock",
    })

    dp, train_dfs, val_dfs, test_dfs, feature_cols = _prepare_multi_data("DATA")

    ga = GAOptimizer(
        feature_names=feature_cols,
        train_dfs=train_dfs,
        val_dfs=val_dfs,
        mode=mode,
        population_size=population_size,
        num_generations=num_generations,
        ga_epochs=ga_epochs,
        data_fraction=data_fraction,
        result_logger=rl,
    )
    result = ga.run()

    # Retrain the best chromosome on the FULL multi-stock dataset
    best = result["best_chromosome"]
    logger.info("=" * 60)
    logger.info("RETRAINING BEST CHROMOSOME ON FULL MULTI-STOCK DATA")
    logger.info("=" * 60)

    active_features = best.get("selected_features", feature_cols)
    window_size = best.get("window_size", 12)
    hidden_units = best.get("hidden_units", 64)
    dropout = best.get("dropout", 0.2)
    learning_rate = best.get("learning_rate", 1e-3)
    look_forward = best.get("look_forward", 1)
    target_col = f"Target_{look_forward}_Tick"

    device = get_device()

    dp_full = DataProcessor(scaler_type="standard")
    data = dp_full.scale_and_window_multi(
        train_dfs, val_dfs, test_dfs,
        active_features, target_col, window_size,
    )

    model = BiLSTMModel(
        input_size=len(active_features),
        hidden_size=hidden_units,
        num_layers=2,
        dropout=dropout,
    )

    trainer = Trainer(model=model, device=device, learning_rate=learning_rate)

    train_loader, val_loader = trainer.create_dataloaders(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        batch_size=64,
    )

    trainer.train(
        train_loader, val_loader,
        epochs=50, patience=10, verbose=True,
        result_logger=rl,
    )

    logger.info("=" * 60)
    logger.info("GA MULTI-STOCK BEST CHROMOSOME -- FINAL TEST EVALUATION")
    logger.info("=" * 60)
    metrics = trainer.evaluate(data["X_test"], data["y_test"])

    chk_path = f"checkpoints/{label}_best_all.pt"
    trainer.save_checkpoint(chk_path, dp_full, active_features, window_size)
    rl.log_metrics(metrics)
    rl.copy_checkpoint(chk_path)

    logger.info("GA multi-stock experiment complete. Results saved to %s", rl.get_run_dir())


def run_ga_fast_multi() -> None:
    """Quick multi-stock GA test using 20% data and minimal generations."""
    run_ga_optimization_multi(
        mode="features_only",
        ga_epochs=5,
        data_fraction=0.2,
        population_size=6,
        num_generations=3,
    )


def run_ga_full_multi() -> None:
    """Full multi-stock GA run for external GPU deployment."""
    run_ga_optimization_multi(
        mode="full",
        ga_epochs=50,
        data_fraction=1.0,
        population_size=20,
        num_generations=30,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(description="Financial Forecasting Pipeline")
    parser.add_argument(
        "command",
        choices=[
            # Single-stock experiments
            "baseline", "time_horizon", "attention", "ga_fast", "ga_full",
            "pipeline",
            # Multi-stock (universal model) experiments
            "baseline_multi", "ga_fast_multi", "ga_full_multi",
        ],
        help="Which experiment to run.",
    )
    args = parser.parse_args()

    setup_logging()

    commands = {
        "baseline": run_baseline_experiment,
        "time_horizon": run_time_horizon_experiment,
        "attention": run_attention_comparison,
        "ga_fast": run_ga_fast_test,
        "ga_full": run_ga_full_gpu,
        "pipeline": run_pipeline,
        "baseline_multi": run_baseline_multi,
        "ga_fast_multi": run_ga_fast_multi,
        "ga_full_multi": run_ga_full_multi,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
