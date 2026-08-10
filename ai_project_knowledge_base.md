# Financial Forecasting Pipeline - Project Knowledge Base

## Core Technology Stack
- **Language**: Python
- **Deep Learning**: PyTorch (`torch.nn`, `DataLoader`, `TensorDataset`, `CrossEntropyLoss`)
- **Machine Learning**: scikit-learn (`HistGradientBoostingClassifier`, `precision_score`)
- **Optimization**: DEAP (Distributed Evolutionary Algorithms in Python)
- **Data Processing**: pandas, numpy
- **Logging/Visualization**: standard `logging`, matplotlib (via `result_viewer.py` for Equity Curves, GA Scatter Matrices, Heatmaps)

## Primary Methodologies
- **Base Forecasting Models**: BiLSTM, BiLSTMAttention
- **Meta-Modeling**: Confidence Estimator (Model 2) acting as a trade approval gate
- **Hyperparameter & Feature Search**: Genetic Algorithms (GA) jointly optimizing feature masks and network topologies
- **Data Normalization**: Per-stock Standardization (`StandardScaler`), Boundary-safe sliding windowing
- **Architecture Strategy**: Universal Model (Multi-stock joint training)

## V1 Implementation Specifics
- **Objective Formulation**: Regression
- **Network Output**: 1-dimensional continuous value (Linear layer, no activation)
- **Loss Function**: Custom Directional Loss
- **Evaluation Metrics**: Validation RMSE, MAE, R2, Directional Accuracy
- **Data Resolution**: 5-minute intervals
- **Prediction Horizon**: 1, 2, 3 ticks (5, 10, 15 minutes)
- **Data Split**: Chronological Train/Validation/Test
- **GA Fitness Function**: Minimizing Validation RMSE (incorporating lambda complexity penalty)

## V2 Implementation Specifics (Current Target Architecture)
- **Objective Formulation**: 3-Class Classification (UP, DOWN, NEUTRAL)
- **Network Output**: 3-dimensional logits
- **Loss Function**: PyTorch `CrossEntropyLoss` with inverse frequency class weighting
- **Primary Evaluation Metrics**: Precision@HighConfidence (>62%), Trade Rate (10-25%), Annualized Sharpe Ratio, Model 2 Precision (>60%)
- **Data Resolution**: Parameterized (Base 15-min or 30-min timeframe; dynamically aggregates 5-min microstructure)
- **Prediction Horizon**: 1 period forward (15 or 30 minutes)
- **Target Labeling**: Dynamic per-stock volatility thresholding (`returns.rolling(window).std() * multiplier`)
- **Data Split**: 3-way Strict Temporal Split (Train: 60%, Calibration: 20%, Test: 20%)
- **Meta-Model Pipeline**: 
  - Train Model 1 (BiLSTM) on Train split
  - Generate Model 1 predictions on Calibration split
  - Train Model 2 (HistGradient Boosting) on Calibration split using Model 1 features + logits + max confidence
- **GA Fitness Function**: Maximizing Simulated Annualized Sharpe Ratio (dynamically calibrated to `bars_per_day` depending on resolution) minus feature complexity penalty
- **GA Chromosome Structure**: `[N feature bits | window_idx | horizon_idx | hidden_idx | dropout_idx | lr_idx | threshold_multiplier_idx | confidence_threshold_idx | class_weight_strategy_idx]`
- **Expanded Feature Space**: 
  - 18 base indicators
  - Intra-bar metrics: RSI divergence, volume surprise, bar range, close position, VWAP distance
  - Cross-stock metrics: Sector-relative returns, Cross-stock z-scores (rolling 20 bars)

## Component Modules
- `data_processor.py` / `data_processor_v2.py`: Data loading, missing interval imputation, multi-resolution alignment, temporal splitting, target generation, sliding windows. Propagates Timestamps & Tickers for evaluation logging.
- `feature_engineer.py` / `feature_engineer_v2.py`: Indicator calculations (TA-based and cross-stock relative features).
- `model_builder.py`: PyTorch Module definitions (`BiLSTMModel`, `BiLSTMAttentionModel`, `ClassificationBiLSTMModel`, `ClassificationBiLSTMAttentionModel`).
- `trainer.py` / `trainer_v2.py`: PyTorch training loops, early stopping, checkpointing.
- `confidence_model.py`: Model 2 HistGradient Boosting logic.
- `pipeline_evaluator_v2.py`: Decoupled evaluation engine generating trade logs, PnL statistics, and confusion matrices (replaced the deprecated `result_viewer_v2.py`).
- `ga_optimizer.py` / `ga_optimizer_v2.py`: DEAP setup, parametrized chromosome evaluation, memory-leak-safe evaluation loops, and Sharpe ratio simulation.
- `result_logger.py`: Advanced metrics persistence (saving `test_trade_log.csv`, `ga_population_history.jsonl`, `confusion_matrix.json`, `pnl_stats.json`).
- `result_viewer.py`: Advanced quantitative plotting dashboard (Cumulative Equity Curves, Drawdowns, Confidence Distribution, Heatmaps, GA Scatter Matrices).

## CLI Entry Points (`main.py`)
- `baseline` / `baseline_multi`: Single and Multi-stock V1 Baseline
- `time_horizon`: Horizon degradation sweep (t+1, t+2, t+3)
- `attention`: V1 Attention mechanism comparison
- `ga_fast` / `ga_full` / `ga_full_multi`: V1 GA optimization routines
- `baseline_v2_multi`: V2 Classification + Meta-Model pipeline
- `attention_v2_multi`: V2 Attention mechanism comparison
- `ga_v2_multi_15min` / `ga_v2_multi_30min`: V2 GA optimization maximizing Sharpe Ratio (parametrized windows)
- `ga_fast_v2_multi_15min` / `ga_fast_v2_multi_30min`: Quick V2 GA routines for testing pipeline integrity

## Utilities
- `convert_xlsx_to_csv.py`: Ingestion layer for Refinitiv exports.
- `smoke_test.py`: Fast CPU E2E validation script.
