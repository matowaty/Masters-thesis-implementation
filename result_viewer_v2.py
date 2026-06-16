"""
result_viewer_v2.py — Evaluation Metrics & Visualization for V2

Reports specific metrics for the V2 classification pipeline:
    - Precision@HighConfidence
    - Trade Rate
    - Annualized Sharpe Ratio
    - Model 2 Standalone Precision
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import DataLoader

from confidence_model import extract_m2_features

logger = logging.getLogger(__name__)


class ResultViewerV2:
    def __init__(self, device: torch.device):
        self.device = device

    def evaluate_pipeline(
        self, 
        model1: nn.Module, 
        model2: GradientBoostingClassifier, 
        test_loader: DataLoader, 
        conf_threshold: float
    ) -> dict:
        """Run the full evaluation of Model 1 + Model 2 on the Test set."""
        logger.info("=" * 60)
        logger.info("V2 Pipeline Evaluation on Test Set")
        logger.info("=" * 60)
        
        m2_features, m2_labels = extract_m2_features(model1, test_loader, self.device)
        
        # 1. Model 2 Standalone Precision
        preds = model2.predict(m2_features)
        m2_precision = np.mean(preds[preds == 1] == m2_labels[preds == 1]) if sum(preds) > 0 else 0.0
        logger.info("Model 2 Standalone Precision (at 0.5): %.1f%%", m2_precision * 100)
        
        # 2. Extract Confidence Scores
        conf_scores = model2.predict_proba(m2_features)[:, 1]
        approved_mask = conf_scores > conf_threshold
        trade_rate = approved_mask.mean()
        
        logger.info("Confidence Threshold: %.2f", conf_threshold)
        logger.info("Trade Approval Rate: %.1f%%", trade_rate * 100)
        
        
        metrics = {
            "m2_precision_at_0.5": float(m2_precision),
            "trade_rate": float(trade_rate),
        }
        
        if trade_rate < 0.01:
            logger.warning("Trade rate too low to calculate reliable Sharpe/Precision metrics.")
            metrics["precision_at_conf"] = 0.0
            metrics["annualized_sharpe"] = 0.0
            return metrics
            
        # 3. Precision @ High Confidence
        # Only evaluate Model 1's precision on trades that Model 2 approved
        approved_correct = m2_labels[approved_mask]
        precision_at_conf = approved_correct.mean()
        logger.info("Precision @ High Confidence: %.1f%%", precision_at_conf * 100)
        metrics["precision_at_conf"] = float(precision_at_conf)
        
        # 4. Annualized Sharpe Ratio
        model1.eval()
        all_probs, all_ret = [], []
        with torch.no_grad():
            for X_b, _, ret_b in test_loader:
                logits = model1(X_b.to(self.device))
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_ret.append(ret_b.numpy())
                
        all_probs = np.vstack(all_probs)
        all_ret = np.concatenate(all_ret)
        m1_preds = np.argmax(all_probs, axis=1)
        
        direction = np.where(m1_preds == 2, 1, np.where(m1_preds == 0, -1, 0))
        pnl = direction[approved_mask] * all_ret[approved_mask]
        
        if pnl.std() > 1e-9:
            sharpe = pnl.mean() / pnl.std() * np.sqrt(252 * 13) # Ann. for 30min bars (13/day)
            logger.info("Simulated Annualized Sharpe Ratio: %.2f", sharpe)
            metrics["annualized_sharpe"] = float(sharpe)
        else:
            logger.info("Simulated Annualized Sharpe Ratio: 0.00 (No volatility in returns)")
            metrics["annualized_sharpe"] = 0.0
            
        logger.info("=" * 60)
        return metrics
