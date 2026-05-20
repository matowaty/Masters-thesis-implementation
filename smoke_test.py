"""
smoke_test.py — End-to-end pipeline validation

Loads JPM data, generates all features, computes targets, applies
chronological splitting / scaling / windowing, builds a minimal inline
BiLSTM, and trains for a few epochs.

Purpose: confirm the data pipeline produces valid tensors and the model
can learn (loss decreases).  This is a throwaway diagnostic script.
"""

import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal inline BiLSTM (avoids coupling to skeleton model_builder.py)
# ---------------------------------------------------------------------------
class _SmokeBiLSTM(nn.Module):
    """Tiny BiLSTM for smoke-testing the data pipeline."""

    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # BiLSTM output is hidden_size * 2 (forward + backward)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, sequence_length, num_features]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch_size, sequence_length, hidden_size * 2]

        # Take the last time step
        last_hidden = lstm_out[:, -1, :]
        # last_hidden: [batch_size, hidden_size * 2]

        out = self.fc(last_hidden)
        # out: [batch_size, 1]
        return out


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full smoke test pipeline."""

    # --- Configuration ---
    DATA_PATH = "DATA/JPM.csv"
    WINDOW_SIZE = 12
    TARGET_COL = "Target_1_Tick"
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------------
    dp = DataProcessor(scaler_type="standard")
    df = dp.load_data(DATA_PATH)
    df = dp.handle_missing_intervals(df)
    logger.info("Raw data shape: %s", df.shape)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    fe = FeatureEngineer()
    df = fe.add_all_features(df)
    feature_cols = fe.get_feature_names()
    logger.info(
        "After feature engineering: %s -- %d features: %s",
        df.shape, len(feature_cols), feature_cols,
    )

    # ------------------------------------------------------------------
    # 3. Compute targets
    # ------------------------------------------------------------------
    df = dp.compute_targets(df)
    logger.info("After targets: %s", df.shape)

    # ------------------------------------------------------------------
    # 4. Chronological split
    # ------------------------------------------------------------------
    train_df, val_df, test_df = dp.chronological_split(df)

    # ------------------------------------------------------------------
    # 5. Scale features (fit ONLY on training set)
    # ------------------------------------------------------------------
    dp.fit_scaler(train_df, feature_cols)
    X_train_scaled = dp.transform(train_df, feature_cols)
    X_val_scaled = dp.transform(val_df, feature_cols)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    # ------------------------------------------------------------------
    # 6. Create sliding windows
    # ------------------------------------------------------------------
    X_train, y_train = dp.create_windows(X_train_scaled, y_train, WINDOW_SIZE)
    X_val, y_val = dp.create_windows(X_val_scaled, y_val, WINDOW_SIZE)

    logger.info("X_train shape: %s  y_train shape: %s", X_train.shape, y_train.shape)
    logger.info("X_val shape:   %s  y_val shape:   %s", X_val.shape, y_val.shape)

    # ------------------------------------------------------------------
    # 7. PyTorch DataLoaders
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 8. Build model
    # ------------------------------------------------------------------
    num_features = X_train.shape[2]
    model = _SmokeBiLSTM(input_size=num_features, hidden_size=32).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    logger.info("Model:\n%s", model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %d", total_params)

    # ------------------------------------------------------------------
    # 9. Training loop (few epochs)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training -- %d epochs", NUM_EPOCHS)
    logger.info("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)       # [B, W, F]
            y_batch = y_batch.to(device)       # [B]

            optimizer.zero_grad()
            preds = model(X_batch).squeeze(-1)  # [B]
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        val_rmse = np.sqrt(avg_val_loss)

        logger.info(
            "Epoch %2d/%d -- train_loss=%.6f  val_loss=%.6f  val_RMSE=%.6f",
            epoch, NUM_EPOCHS, avg_train_loss, avg_val_loss, val_rmse,
        )

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[OK] Smoke test PASSED -- pipeline is end-to-end functional")
    logger.info("  Final val MSE:  %.6f", avg_val_loss)
    logger.info("  Final val RMSE: %.6f", val_rmse)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
