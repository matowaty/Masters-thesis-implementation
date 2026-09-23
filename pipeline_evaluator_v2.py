"""
pipeline_evaluator_v2.py — Evaluation Metrics & Visualization for V2

Replaces the old result_viewer_v2.py.
Reports specific metrics for the V2 classification pipeline:
    - Precision@HighConfidence
    - Trade Rate
    - Annualized Sharpe Ratio
    - Model 2 Standalone Precision
    
Also computes and returns detailed logs for deep analysis:
    - Trade Log (DataFrame)
    - Confusion Matrix (dict)
    - PnL Statistics (dict)
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader

from confidence_model import extract_m2_features

logger = logging.getLogger(__name__)


class PipelineEvaluatorV2:
    """Evaluates a trained Model 1 + Model 2 pair on a test set: precision, trade rate,
    annualized Sharpe ratio, confusion matrix, and a per-bar trade log.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def evaluate_pipeline(
        self, 
        model1: nn.Module, 
        model2: HistGradientBoostingClassifier, 
        test_loader: DataLoader, 
        test_times: np.ndarray,
        test_tickers: np.ndarray,
        conf_threshold: float,
        bars_per_day: int = 13
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
        
        # We still need to compute M1 predictions for the trade log even if trade rate is low
        model1.eval()
        all_probs, all_ret, all_targets = [], [], []
        with torch.no_grad():
            for X_b, y_b, ret_b in test_loader:
                logits = model1(X_b.to(self.device))
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_ret.append(ret_b.numpy())
                all_targets.append(y_b.numpy())
                
        all_probs = np.vstack(all_probs)
        all_ret = np.concatenate(all_ret)
        all_targets = np.concatenate(all_targets)
        m1_preds = np.argmax(all_probs, axis=1)
        
        direction = np.where(m1_preds == 2, 1, np.where(m1_preds == 0, -1, 0))
        pnl = direction * all_ret
        
        # 3. Precision @ High Confidence
        if trade_rate > 0.001:
            approved_correct = m2_labels[approved_mask]
            precision_at_conf = approved_correct.mean()
            logger.info("Precision @ High Confidence: %.1f%%", precision_at_conf * 100)
            metrics["precision_at_conf"] = float(precision_at_conf)
            
            approved_pnl = pnl[approved_mask]
            if approved_pnl.std() > 1e-9:
                sharpe = approved_pnl.mean() / approved_pnl.std() * np.sqrt(252 * bars_per_day) # Ann. bars
                logger.info("Simulated Annualized Sharpe Ratio: %.2f", sharpe)
                metrics["annualized_sharpe"] = float(sharpe)
            else:
                logger.info("Simulated Annualized Sharpe Ratio: 0.00 (No volatility in returns)")
                metrics["annualized_sharpe"] = 0.0
        else:
            logger.warning("Trade rate too low to calculate reliable Sharpe/Precision metrics.")
            metrics["precision_at_conf"] = 0.0
            metrics["annualized_sharpe"] = 0.0
            
        logger.info("=" * 60)
        
        # 4. Construct Trade Log
        trade_log_df = pd.DataFrame({
            "Timestamp": test_times,
            "Ticker": test_tickers,
            "Actual_Fwd_Return": all_ret,
            "Target_Class": all_targets,
            "M1_Prob_DOWN": all_probs[:, 0],
            "M1_Prob_NEUTRAL": all_probs[:, 1],
            "M1_Prob_UP": all_probs[:, 2],
            "M1_Pred_Class": m1_preds,
            "M2_Conf_Score": conf_scores,
            "Approved_Trade": approved_mask,
            "PnL": pnl * approved_mask # 0 if not approved
        })
        
        # 5. Compute Confusion Matrix
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(all_targets, m1_preds):
            cm[t, p] += 1
            
        cm_dict = {
            "Actual_DOWN_Pred_DOWN": int(cm[0, 0]),
            "Actual_DOWN_Pred_NEUTRAL": int(cm[0, 1]),
            "Actual_DOWN_Pred_UP": int(cm[0, 2]),
            "Actual_NEUTRAL_Pred_DOWN": int(cm[1, 0]),
            "Actual_NEUTRAL_Pred_NEUTRAL": int(cm[1, 1]),
            "Actual_NEUTRAL_Pred_UP": int(cm[1, 2]),
            "Actual_UP_Pred_DOWN": int(cm[2, 0]),
            "Actual_UP_Pred_NEUTRAL": int(cm[2, 1]),
            "Actual_UP_Pred_UP": int(cm[2, 2]),
        }
        
        # 6. PnL Statistics
        approved_pnl = pnl[approved_mask]
        wins = approved_pnl[approved_pnl > 0]
        losses = approved_pnl[approved_pnl < 0]
        
        pnl_stats = {
            "total_trades_approved": int(approved_mask.sum()),
            "win_rate": float(len(wins) / len(approved_pnl)) if len(approved_pnl) > 0 else 0.0,
            "avg_win_pct": float(wins.mean() * 100) if len(wins) > 0 else 0.0,
            "avg_loss_pct": float(losses.mean() * 100) if len(losses) > 0 else 0.0,
            "gross_profit_pct": float(wins.sum() * 100) if len(wins) > 0 else 0.0,
            "gross_loss_pct": float(losses.sum() * 100) if len(losses) > 0 else 0.0,
            "total_return_pct": float(approved_pnl.sum() * 100)
        }
        
        # Max Drawdown calculation
        cumulative_returns = (1 + approved_pnl).cumprod() if len(approved_pnl) > 0 else np.array([])
        max_drawdown = 0.0
        if len(cumulative_returns) > 0:
            peak = cumulative_returns[0]
            for r in cumulative_returns:
                if r > peak:
                    peak = r
                dd = (peak - r) / peak
                if dd > max_drawdown:
                    max_drawdown = dd
        pnl_stats["max_drawdown_pct"] = float(max_drawdown * 100)
        
        return {
            "metrics": metrics,
            "trade_log": trade_log_df,
            "confusion_matrix": cm_dict,
            "pnl_stats": pnl_stats
        }
