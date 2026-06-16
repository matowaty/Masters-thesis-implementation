"""
trainer_v2.py — Training Loop, Validation & Evaluation (V2 Classification)

Handles model training, early stopping, performance metric calculation,
and model persistence. Adapted for 3-class classification with CrossEntropyLoss
and inverse frequency class weights to handle NEUTRAL class dominance.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_processor_v2 import DataProcessorV2

logger = logging.getLogger(__name__)


class EarlyStoppingV2:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
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


class TrainerV2:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = None  # Will be set in create_dataloaders based on class freq

    def compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """Inverse frequency class weights to combat NEUTRAL dominance."""
        counts = np.bincount(y_train, minlength=3)
        total = len(y_train)
        weights = total / (3.0 * np.maximum(counts, 1))
        logger.info("Computed class weights: %s (counts: %s)", weights, counts)
        return torch.FloatTensor(weights).to(self.device)

    def create_dataloaders(
        self,
        X_train: np.ndarray, y_train: np.ndarray, ret_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, ret_val: np.ndarray,
        batch_size: int = 64,
    ) -> Tuple[DataLoader, DataLoader]:
        
        # Set criterion with class weights computed from training set
        weights = self.compute_class_weights(y_train)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
            torch.FloatTensor(ret_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
            torch.FloatTensor(ret_val)
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    def _train_one_epoch(
        self, dataloader: DataLoader, max_grad_norm: float = 1.0,
        epoch: int = 0, total_epochs: int = 0, show_progress: bool = False,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc=f"Epoch {epoch:3d}/{total_epochs}", leave=False, unit="batch")

        for X_batch, y_batch, _ in iterator:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass: model outputs raw logits for 3 classes
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            if show_progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
            
        return total_loss / len(dataloader)

    def _validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch, _ in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
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
        early_stopping = EarlyStoppingV2(patience=patience)
        if verbose: logger.info("Starting V2 training on %s for up to %d epochs", self.device, epochs)

        epoch_times = []
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, epoch=epoch, total_epochs=epochs, show_progress=verbose)
            val_loss = self._validate(val_loader)
            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            
            if verbose:
                avg_time = sum(epoch_times) / len(epoch_times)
                remaining = avg_time * (epochs - epoch)
                eta_min, eta_sec = divmod(int(remaining), 60)
                logger.info(
                    "Epoch %3d/%d -- train_loss=%.4f  val_loss=%.4f  [%.1fs/epoch, ETA %dm%02ds]",
                    epoch, epochs, train_loss, val_loss, elapsed, eta_min, eta_sec
                )
                
            if result_logger is not None:
                result_logger.log_epoch(epoch, train_loss, val_loss)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
                
        return early_stopping.best_loss

    def save_checkpoint(
        self, filepath: str, data_processor: DataProcessorV2, feature_names: list, window_size: int
    ) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "feature_names": feature_names,
            "window_size": window_size,
        }
        
        if data_processor._scaler_fitted:
            checkpoint["stock_scalers"] = data_processor.stock_scalers

        torch.save(checkpoint, path)
        logger.info("[OK] V2 Checkpoint saved to %s", path)

    def load_checkpoint(self, filepath: str, data_processor: DataProcessorV2) -> Dict:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "stock_scalers" in checkpoint:
            data_processor.stock_scalers = checkpoint["stock_scalers"]
            data_processor._scaler_fitted = True
            
        logger.info("[OK] V2 Checkpoint loaded from %s", path)
        return {
            "feature_names": checkpoint.get("feature_names", []),
            "window_size": checkpoint.get("window_size", 12),
        }
