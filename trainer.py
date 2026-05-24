"""
trainer.py — Training Loop, Validation & Evaluation

Handles model training, early stopping, performance metric calculation,
and model persistence (saving/loading checkpoints).
"""

from result_logger import ResultLogger
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_processor import DataProcessor

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: How many epochs to wait after last time validation loss improved.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered after %d epochs without improvement", self.counter)


class DirectionalLoss(nn.Module):
    """Custom loss function that penalizes incorrect direction predictions."""
    def __init__(self, penalty_factor: float = 5.0) -> None:
        super().__init__()
        self.penalty_factor = penalty_factor
        self.base_loss = nn.HuberLoss(reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(y_pred, y_true)
        
        pred_sign = torch.sign(y_pred)
        true_sign = torch.sign(y_true)
        
        # Mask for incorrect direction (ignoring flat true targets)
        wrong_direction = (pred_sign != true_sign) & (true_sign != 0)
        loss[wrong_direction] = loss[wrong_direction] * self.penalty_factor
        
        # Mask for cowardly "lazy" prediction (0.0)
        # Using a small epsilon to catch floats effectively at zero
        lazy_pred = torch.abs(y_pred) < 1e-6
        loss[lazy_pred] = loss[lazy_pred] * self.penalty_factor
        
        return loss.mean()


class Trainer:
    """Handles the PyTorch training loop and evaluation metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_fn: str = "mse",
        target_scaling_factor: float = 1000.0,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: The PyTorch model to train.
            device: Device to run training on (CPU, CUDA, MPS).
            learning_rate: Optimizer learning rate.
            weight_decay: L2 regularization penalty.
            loss_fn: Loss function name -- 'mse' or 'huber' (default 'mse').
            target_scaling_factor: Multiplier used to scale return targets (default 1000.0).
        """
        self.model = model.to(device)
        self.device = device
        self.target_scaling_factor = target_scaling_factor
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        if loss_fn.lower() == "huber":
            self.criterion = nn.HuberLoss()
        elif loss_fn.lower() == "directional":
            self.criterion = DirectionalLoss(penalty_factor=5.0)
        else:
            self.criterion = nn.MSELoss()

    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 64,
    ) -> Tuple[DataLoader, DataLoader]:
        """Convert numpy arrays to PyTorch DataLoaders.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            batch_size: Batch size for both dataloaders.

        Returns:
            Tuple of (train_loader, val_loader).
        """
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    def _train_one_epoch(
        self, dataloader: DataLoader, max_grad_norm: float = 1.0,
        epoch: int = 0, total_epochs: int = 0, show_progress: bool = False,
    ) -> float:
        """Run one pass over the training data."""
        self.model.train()
        total_unscaled_loss = 0.0

        iterator = dataloader
        if show_progress:
            iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch:3d}/{total_epochs}",
                leave=False,
                unit="batch",
            )

        for X_batch, y_batch in iterator:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            preds = self.model(X_batch).squeeze(-1)
            loss = self.criterion(preds, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Compute unscaled loss purely for perfectly comparable logging
            with torch.no_grad():
                unscaled_preds = preds / self.target_scaling_factor
                unscaled_y = y_batch / self.target_scaling_factor
                unscaled_loss = self.criterion(unscaled_preds, unscaled_y)
                total_unscaled_loss += unscaled_loss.item()

            if show_progress:
                iterator.set_postfix(loss=f"{loss.item():.6f}")
            
        # Return the unscaled loss to match val loss exactly on charts
        return total_unscaled_loss / len(dataloader)

    def _validate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                preds = self.model(X_batch).squeeze(-1)
                
                # Unscale predictions and targets to calculate true unscaled val loss
                unscaled_preds = preds / self.target_scaling_factor
                unscaled_y = y_batch / self.target_scaling_factor
                
                loss = self.criterion(unscaled_preds, unscaled_y)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
        result_logger: Optional["ResultLogger"] = None,
    ) -> float:
        """Execute the full training loop with early stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of epochs to train.
            patience: Early stopping patience.
            verbose: If True, logs epoch progress.
            result_logger: Optional ResultLogger to record per-epoch losses.

        Returns:
            The best validation loss achieved.
        """
        early_stopping = EarlyStopping(patience=patience)
        
        if verbose:
            logger.info("Starting training on device %s for up to %d epochs", self.device, epochs)

        epoch_times = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(
                train_loader, epoch=epoch, total_epochs=epochs,
                show_progress=verbose,
            )
            val_loss = self._validate(val_loader)
            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            
            if verbose:
                avg_time = sum(epoch_times) / len(epoch_times)
                remaining = avg_time * (epochs - epoch)
                eta_min, eta_sec = divmod(int(remaining), 60)
                logger.info(
                    "Epoch %3d/%d -- train_loss=%.6f  val_loss=%.6f  "
                    "[%.1fs/epoch, ETA %dm%02ds]",
                    epoch, epochs, train_loss, val_loss,
                    elapsed, eta_min, eta_sec,
                )

            if result_logger is not None:
                result_logger.log_epoch(epoch, train_loss, val_loss)
                
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
                
        return early_stopping.best_loss

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Compute standard regression metrics on the test set.

        Calculates MSE, RMSE, MAE, R2 Score, and Directional Accuracy.

        Args:
            X_test: Test features.
            y_test: Test targets.

        Returns:
            Dictionary of computed metrics.
        """
        self.model.eval()
        
        # For evaluation, we can process all test data at once if it fits in memory,
        # but to be safe we'll use a dataloader.
        test_ds = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch).squeeze(-1)
                all_preds.append(preds.cpu().numpy())
                
        # Unscale predictions and targets for real-world metrics
        y_pred = np.concatenate(all_preds) / self.target_scaling_factor
        y_test_unscaled = y_test / self.target_scaling_factor
        
        # Metrics
        mse = mean_squared_error(y_test_unscaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_unscaled, y_pred)
        r2 = r2_score(y_test_unscaled, y_pred)
        
        # Directional Accuracy
        # Computes percentage of time the model correctly predicts the sign of the return
        correct_direction = np.sign(y_pred) == np.sign(y_test_unscaled)
        da = np.mean(correct_direction) * 100.0
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "rmse_bps": rmse * 10000.0,
            "mae_bps": mae * 10000.0,
            "r2": r2,
            "directional_accuracy": da,
        }
        
        logger.info(
            "Test Evaluation -- RMSE: %.6f (%.1f bps), MAE: %.6f (%.1f bps), R2: %.4f, DA: %.2f%%",
            rmse, rmse * 10000.0, mae, mae * 10000.0, r2, da
        )
        
        return metrics

    def save_checkpoint(
        self, filepath: str, data_processor: DataProcessor, feature_names: list, window_size: int
    ) -> None:
        """Save model, optimizer, scaler state, and configuration to disk.

        Allows for seamless resumption of training or inference on another machine.

        Args:
            filepath: Destination file path (e.g., 'checkpoints/model.pt').
            data_processor: The DataProcessor instance whose scaler state should be saved.
            feature_names: List of feature names used during training.
            window_size: The time window size used during training.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "feature_names": feature_names,
            "window_size": window_size,
            "target_scaling_factor": self.target_scaling_factor,
        }
        
        if data_processor.scaler is not None and data_processor._scaler_fitted:
            checkpoint["scaler"] = data_processor.scaler
        else:
            checkpoint["scaler"] = None

        torch.save(checkpoint, path)
        logger.info("[OK] Checkpoint saved to %s", path)

    def load_checkpoint(self, filepath: str, data_processor: DataProcessor) -> Dict:
        """Load model weights, optimizer state, and scaler from disk.

        Args:
            filepath: Path to the saved checkpoint file.
            data_processor: DataProcessor instance to restore the scaler into.

        Returns:
            Dictionary containing 'feature_names' and 'window_size' used during training.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        # load onto current device
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        scaler = checkpoint.get("scaler")
        if scaler is not None:
            data_processor.scaler = scaler
            data_processor._scaler_fitted = True
            
        if "target_scaling_factor" in checkpoint:
            self.target_scaling_factor = checkpoint["target_scaling_factor"]
            
        logger.info("[OK] Checkpoint loaded from %s", path)
        
        return {
            "feature_names": checkpoint.get("feature_names", []),
            "window_size": checkpoint.get("window_size", 12),
            "target_scaling_factor": self.target_scaling_factor,
        }
