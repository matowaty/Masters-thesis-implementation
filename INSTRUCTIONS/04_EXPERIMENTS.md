# Research Plan & Experiment Logbook (`04_EXPERIMENTS.md`)

This file serves as the official roadmap and trial-and-error logbook for the forecasting system. **The IDE agent must strictly record conclusions and metrics here after every completed experiment.**

## 1. Core Research Questions
* How does the time window size (look-back period) affect the prediction error?
* How does the error change depending on the prediction horizon (1, 2, or 3 ticks/5-minute intervals forward)?
* Which technical features have the greatest statistical significance for the model (based on the GA Feature Selection results)?
* Does the Attention mechanism significantly improve results compared to classical LSTM for dense 5-minute data?

## 2. Experiment Phases

### Phase 1: Baseline (Control)
* **Description:** Train a pure baseline LSTM model (from `model_builder.py`) using all generated features and rigidly assigned hyperparameters (e.g., window = 12, predicting $t+1$ step forward).
* **Goal:** Establish the minimum error baseline (MSE, MAE, R² Score) that subsequent phases will strive to beat.
* **RESULTS:** 
  * **Dataset**: JPM (1 ticker)
  * **Hyperparameters**: Window = 12, Prediction = $t+1$, Hidden = 64, Layers = 2
  * **Test Metrics**:
    * **RMSE**: 0.001312
    * **MAE**: 0.000802
    * **R² Score**: -0.1181
    * **Directional Accuracy**: 46.68%
### Phase 2: Time Horizon
* **Description:** Modify the target variable generation in `data_processor.py` to create labels shifted by $t+1$, $t+2$, and $t+3$ ticks into the future.
* **Goal:** Answer research question #2 regarding the degradation or change in prediction error as the horizon extends.
* **RESULTS:** *[To be filled by the Agent after execution]*

### Phase 3: Genetic Feature Optimization (GA Feature Selection)
* **Description:** Run the Genetic Algorithm (`ga_optimizer.py`) utilizing ONLY Part 1 (Binary) of the chromosome structure.
* **Goal:** Find the strongest subset of technical features, answer research question #3, and filter out informational noise.
* **RESULTS:** *[To be filled by the Agent after execution]*

### Phase 4: Full GA Optimization (Tuning)
* **Description:** Run the Genetic Algorithm on the full chromosome (Part 1: Features + Part 2: Hyperparameters & Window Length + Part 3: Learning Rate & Dropout).
* **Goal:** Discover the optimal, final structure of the research model.
* **RESULTS:** *[To be filled by the Agent after execution]*
