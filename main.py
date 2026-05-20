"""
main.py -- Orchestration Entry Point

The master script that orchestrates the entire pipeline:
Data Ingestion -> Feature Engineering -> Windowing/Scaling ->
Model Training -> GA Optimization -> Final Evaluation.
"""

import logging
import sys

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from model_builder import BiLSTMModel, get_device
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

    # 1. Load & clean data
    dp = DataProcessor(scaler_type="standard")
    df = dp.load_data(DATA_PATH)
    df = dp.handle_missing_intervals(df)

    # 2. Feature engineering
    fe = FeatureEngineer()
    df = fe.add_all_features(df)
    feature_cols = fe.get_feature_names()

    # 3. Compute targets
    df = dp.compute_targets(df)

    # 4. Chronological split
    train_df, val_df, test_df = dp.chronological_split(df)

    # 5. Scale features
    dp.fit_scaler(train_df, feature_cols)
    X_train_scaled = dp.transform(train_df, feature_cols)
    X_val_scaled = dp.transform(val_df, feature_cols)
    X_test_scaled = dp.transform(test_df, feature_cols)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # 6. Create sliding windows
    X_train, y_train = dp.create_windows(X_train_scaled, y_train, WINDOW_SIZE)
    X_val, y_val = dp.create_windows(X_val_scaled, y_val, WINDOW_SIZE)
    X_test, y_test = dp.create_windows(X_test_scaled, y_test, WINDOW_SIZE)

    # 7. Build model
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

    # 8. Train
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE
    )
    
    logger.info("Training pure baseline LSTM model...")
    best_val_loss = trainer.train(
        train_loader, val_loader, epochs=NUM_EPOCHS, patience=10, verbose=True
    )
    
    # 9. Evaluate
    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION (TEST SET)")
    logger.info("=" * 60)
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save checkpoint
    chk_path = "checkpoints/baseline_jpm.pt"
    trainer.save_checkpoint(chk_path, dp, feature_cols, WINDOW_SIZE)
    
    logger.info("Experiment finished. Please record these results in 04_EXPERIMENTS.md")


def run_pipeline() -> None:
    """Execute the full forecasting pipeline end-to-end."""
    # For now, just route to baseline
    run_baseline_experiment()


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
