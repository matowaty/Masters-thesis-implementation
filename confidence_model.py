"""
confidence_model.py — Model 2 Meta-Model (Gradient Boosting)

Trains a Gradient Boosting classifier to predict whether Model 1's
prediction on a given bar is correct, based on the market conditions
and Model 1's confidence scores.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_m2_features(model1: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for Model 2 from Model 1's predictions and inputs."""
    model1.eval()
    m2_features = []
    m2_labels = []

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            logits = model1(X_batch.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_true = y_batch.numpy()
            y_pred = np.argmax(probs, axis=1)

            # Model 2 features: last time step's market features + Model 1 outputs
            last_step = X_batch[:, -1, :].cpu().numpy()  # [batch, num_features]
            max_conf = probs.max(axis=1, keepdims=True)
            
            features = np.concatenate([last_step, probs, max_conf], axis=1)
            was_correct = (y_pred == y_true).astype(int)

            m2_features.append(features)
            m2_labels.append(was_correct)

    return np.vstack(m2_features), np.concatenate(m2_labels)


def train_confidence_model(
    model1: nn.Module, cal_loader: DataLoader, device: torch.device
) -> HistGradientBoostingClassifier:
    """Train the Model 2 confidence estimator on the calibration set."""
    logger.info("Extracting Calibration features for Model 2...")
    m2_cal_features, m2_cal_labels = extract_m2_features(model1, cal_loader, device)

    logger.info("Training Model 2 (HistGradient Boosting Classifier)...")
    model2 = HistGradientBoostingClassifier(
        max_iter=200, 
        max_depth=4, 
        random_state=42,
        learning_rate=0.1
    )
    model2.fit(m2_cal_features, m2_cal_labels)
    
    # Quick sanity check on calibration data
    preds = model2.predict(m2_cal_features)
    cal_precision = precision_score(m2_cal_labels, preds, zero_division=0)
    logger.info("Model 2 Calibration Precision (at 0.5 threshold): %.3f", cal_precision)
    
    return model2


def evaluate_confidence_model(
    model2: HistGradientBoostingClassifier, m2_features: np.ndarray, m2_labels: np.ndarray, threshold: float = 0.70
) -> Tuple[float, float]:
    """Evaluate Model 2 precision at a specific confidence threshold."""
    confidence_scores = model2.predict_proba(m2_features)[:, 1]
    approved = (confidence_scores > threshold).astype(int)

    precision = precision_score(m2_labels, approved, zero_division=0)
    trade_rate = approved.mean()

    logger.info("Model 2 Evaluation -- Precision@%.2f: %.3f | Trade Approval Rate: %.3f", 
                threshold, precision, trade_rate)
    
    return precision, trade_rate
