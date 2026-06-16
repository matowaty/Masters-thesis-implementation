# System Architecture & Project Specification: ML & GA Financial Forecaster (`01_PROJECT_ARCHITECTURE.md`)

## 1. Project Goal & Research Thesis
**Thesis Title:** "Forecasting the Rate of Return on Selected Financial Instruments Using Deep Learning and a Genetic Algorithm"

The primary objective is to build a highly modular research system to forecast the rate of return on financial instruments using Deep Learning (specifically BiLSTM architectures) optimized by a Genetic Algorithm (GA).

### Core Research Questions
1. How does the time window size (look-back period) affect the prediction error?
2. How does the error change depending on the prediction horizon (1, 2, or 3 ticks/5-minute intervals forward)?
3. Which technical indicators (features) hold the most predictive significance for high-frequency data?
4. Does the Attention mechanism significantly improve results compared to classical LSTM for dense 5-minute data?

## 2. Data Profile & Target Definition
* **Assets:** 20 US Market Companies.
* **Resolution:** 5-minute intervals (OHLCV).
* **Target Variable:** Rate of Return, defined as the percentage change in the Close price. The model will be tested against three distinct prediction horizons:
  * **Target_1_Tick:** `(Close[t+1] - Close[t]) / Close[t]`
  * **Target_2_Tick:** `(Close[t+2] - Close[t]) / Close[t]`
  * **Target_3_Tick:** `(Close[t+3] - Close[t]) / Close[t]`

## 3. Technology Stack Requirements
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch (preferred for modularity, hardware agnosticism, and research flexibility).
* **Data Processing:** Pandas, NumPy, Scikit-learn.
* **Technical Indicators:** `ta` library.
* **Genetic Algorithm:** DEAP or PyGAD.

## 4. Architectural Modularity (Strict Breakdown)
The system must be built in strictly decoupled modules. Changes in one module cannot break others.

### `main.py` (Orchestration)
* The entry point that orchestrates the entire pipeline from data ingestion to GA optimization and final evaluation.

### `data_processor.py` (Ingestion, Preprocessing & Windowing)
* **Tasks:**
    * Load raw data and handle missing intervals (forward fill, then backward fill).
    * Calculate the multi-horizon target variables.
    * **Scaling:** Apply `StandardScaler` or `MinMaxScaler`. **Constraint:** Scalers must be fitted **ONLY** on the training set to prevent data leakage.
    * **Spatio-temporal splitting:** Strictly chronological Train/Validation/Test split (no random shuffling).
    * **Windowing:** Transform 2D tabular data into 3D tensors `[Samples, Time_Window, Features]` where `Time_Window` is dynamic.

### `feature_engineer.py` (Feature Generation)
* Calculates technical indicators using the `ta` library or custom Pandas math:
    * **Volatility & Price Action:** High-Low Spread, Close-Open Spread, Bollinger Bands, ATR.
    * **Trend & Momentum:** SMA, EMA, MACD, RSI.
    * **Volume:** VWAP, OBV.

### `model_builder.py` (Model Architecture)
* Contains the PyTorch Deep Learning classes.
    * **Core Architecture:** Bidirectional LSTM (BiLSTM) with dynamic input sizing (`[batch_size, sequence_length, num_selected_features]`).
    * **Structure:** Input -> BiLSTM Layer(s) -> Dropout Layer(s) -> Fully Connected (Dense) Layer -> Linear Output (for continuous regression).
    * **Loss Function:** MSE (Mean Squared Error) or Huber Loss.
    * **Optimizer:** AdamW.

### `ga_optimizer.py` (Genetic Algorithm)
* Optimizes both hyperparameters and technical feature selection.
* **Chromosome Structure (Vector):**
    * *Window Size:* Integer `[6, 12, 24, 36]`
    * *Look-forward (Target):* Integer `[1, 2, 3]`
    * *Feature Mask:* Binary array (1 = include feature, 0 = drop feature)
    * *LSTM Hidden Units:* Integer `[32, 64, 128, 256]`
    * *Dropout Rate:* Float `[0.1, 0.2, 0.3, 0.4]`
    * *Learning Rate:* Float `[1e-3, 5e-4, 1e-4]`
* **Fitness Function:** Minimize `(RMSE_val + λ * Sum(Feature_Mask))` to maximize predictive accuracy while actively penalizing model complexity/informational noise.

### `trainer.py` (Training & Evaluation)
* Handles the standard training loop, validation tracking, and early stopping.
* **Metrics Evaluated:** MSE, RMSE, MAE, R² Score, and Directional Accuracy.

---

## 5. V2 Classification Pipeline Architecture (Market-Aware)
To combat the low signal-to-noise ratio in 5-minute raw regression, a parallel **V2 Pipeline** was introduced with the following key differences:

* **Target Redefinition:** Predicts 3 classes (UP, NEUTRAL, DOWN) instead of a continuous rate of return. NEUTRAL class dominance is handled via inverse frequency class weighting.
* **Timeframe Aggregation:** 5-minute data is aggregated into 30-minute bars before feature engineering, allowing real market signals to emerge from the microstructure noise.
* **Market-Aware Context:** The system computes the `Index_Return` and `Index_Volatility` across all stocks for every timestamp and injects these into every individual stock's dataset. This provides the model with cross-stock relationships without causing a dimensionality explosion.
* **Model 1 & Model 2 Synergy:**
    * **Model 1 (BiLSTM):** Evaluates market data and predicts the next bar's class using `CrossEntropyLoss`.
    * **Model 2 (Confidence Meta-Model):** A Gradient Boosting Classifier trained strictly on the Calibration Split. It observes Model 1's behavior and predicts if Model 1 will be correct, acting as a trade filter.
* **GA Evolution (Sharpe Fitness):** The GA fitness function simulates trading P&L on the Calibration set using only trades approved by Model 2, maximizing the Annualized Sharpe Ratio.
