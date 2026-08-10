# Financial Forecasting Pipeline — Implementation Change Plan
# Based on current codebase state + session discussion
# All code examples in PyTorch to match existing stack

---

## CURRENT STATE AUDIT

What already exists and is GOOD — do not remove or rewrite:
- BiLSTMModel and BiLSTMAttentionModel in model_builder.py — keep both architectures, they work for classification too
- Boundary-safe windowing in data_processor.py — critical, keep exactly as is
- Per-stock StandardScaler in data_processor.py — correct approach, keep
- 18 features in feature_engineer.py — good base, extend rather than replace
- GA chromosome with 18-bit feature mask + hyperparameter indices in ga_optimizer.py — extend, not replace
- DEAP library setup in ga_optimizer.py — keep, just change fitness function
- Complexity penalty in GA fitness (lambda x num_features) — keep this
- Early stopping in trainer.py — keep
- Checkpointing after every generation in ga_optimizer.py — keep
- Universal Model (multi-stock) approach — keep, it is the right call for 19 stocks
- XLSX to CSV pipeline in convert_xlsx_to_csv.py — keep

What already exists but NEEDS MODIFICATION:
- Directional loss in trainer.py — exists but designed for regression. Needs to be replaced with classification loss
- GA fitness function in ga_optimizer.py — currently uses Validation RMSE. Must change to Sharpe ratio
- GA chromosome in ga_optimizer.py — currently has 18 feature bits + hyperparams. Needs new genes added
- Look-forward horizon [1, 2, 3] ticks — at 5-min resolution this is 5/10/15 minutes. Too short. See timeframe section below
- Output layer in model_builder.py — currently regression (1 output, no activation). Needs to become 3-class softmax

What does NOT yet exist and must be CREATED:
- Timeframe aggregation in data_processor.py — resampling 5-min to 30-min/1H
- 3-class label generation — new function, fits inside data_processor.py or feature_engineer.py
- confidence_model.py — new file, the Model 2 meta-model
- 3-way data split logic — Train / Calibration / Test in data_processor.py
- Cross-stock relative strength features — new features in feature_engineer.py
- Simulation function for Sharpe fitness — needed in ga_optimizer.py

---

## CHANGE 1: TIMEFRAME AGGREGATION (modify data_processor.py)

### Root problem with current 5-minute setup
The current look-forward horizon is [1, 2, 3] ticks in the GA search space. At 5-min resolution:
- t+1 = 5 minutes forward
- t+2 = 10 minutes forward
- t+3 = 15 minutes forward

These are all inside the noise floor. Typical 5-min return is ±0.03–0.15%, often smaller than the bid-ask spread. The directional loss in trainer.py is fighting against the fact that there is no real direction to learn at this scale.

Moving to 30-min bars means t+1 = 30 minutes forward. Returns of ±0.15–0.60% are large enough to be real signal and not microstructure bounce.

### Do not discard 5-min data
Use 5-min bars as a source of microstructure features only. Feed them into the lookback window alongside the 30-min features. This preserves granularity for the LSTM while giving it meaningful labels to learn from.

### Add to data_processor.py
```python
def resample_ohlcv(df_5min, period='30min'):
    df = df_5min.resample(period).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df

def build_multiresolution_frame(df_5min, stock_id):
    df_30 = resample_ohlcv(df_5min, '30min')
    df_1h  = resample_ohlcv(df_5min, '1h')
    # Features from 1H and daily are computed separately and
    # merged into df_30 by timestamp alignment (pd.merge_asof)
    # CRITICAL: only use bars that closed BEFORE the 30-min prediction point
    return df_30, df_1h
```

### Forward return label for 30-min bars
```python
df_30['fwd_return'] = df_30['close'].shift(-1) / df_30['close'] - 1
# Starting threshold: 0.20% = 0.0020
# This replaces the current regression target (raw return value)
```

### Per-stock dynamic threshold (add to data_processor.py)
Different stocks in the 19-stock universe have different volatility. Hard-coding one threshold is wrong.
```python
def compute_dynamic_threshold(returns, window=100, multiplier=0.5):
    # multiplier becomes a new GA gene — see GA section
    return returns.rolling(window).std() * multiplier
```

### Look-ahead bias in multi-resolution alignment
When merging 1H and daily features into the 30-min frame, always use shift(1) on the slower timeframe to ensure only PAST bars are used as features. A 30-min bar at 10:00 must not see any 1H bar that also closed at 10:00.
```python
df_30['1h_rsi'] = df_1h['rsi'].reindex(df_30.index, method='ffill').shift(1)
```

### Impact on window lookback in GA
Current GA search space: [6, 12, 24, 36] steps.
At 30-min bars: 6 steps = 3 hours, 12 steps = 6 hours, 24 steps = 2 trading days, 36 steps = 3 trading days.
These ranges remain sensible — keep the same values in the chromosome, they now mean something more useful.

### Sample count at 30-min resolution
19 stocks x 13 bars/day x 252 days/year x 2 years = ~125,000 samples. Sufficient for deep learning.

---

## CHANGE 2: SWITCH TO 3-CLASS CLASSIFICATION

### What to change in model_builder.py
The final fully connected layer currently outputs 1 value for regression.
Change to 3 outputs with no activation (CrossEntropyLoss in PyTorch takes raw logits).

```python
# In BiLSTMModel and BiLSTMAttentionModel, change the output layer:

# CURRENT (regression):
self.fc = nn.Linear(hidden_size * 2, 1)

# NEW (3-class classification):
self.fc = nn.Linear(hidden_size * 2, 3)
# No softmax here — PyTorch CrossEntropyLoss expects raw logits
# Apply softmax only at inference time when you need probabilities
```

### Label generation (add to data_processor.py or feature_engineer.py)
```python
def label_returns(fwd_returns, threshold):
    # Class 0 = DOWN, Class 1 = NEUTRAL, Class 2 = UP
    labels = np.where(fwd_returns > threshold, 2,
             np.where(fwd_returns < -threshold, 0, 1))
    return labels.astype(np.int64)  # PyTorch CrossEntropyLoss requires int64
```

### Class imbalance — critical
At 30-min resolution, NEUTRAL will still dominate (roughly 50-60% of bars).
Without handling this the model will learn to always predict NEUTRAL and achieve low loss doing it.

```python
# In trainer.py, replace the current loss with weighted CrossEntropy:
from torch.nn import CrossEntropyLoss

def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=3)
    total = len(labels)
    # Inverse frequency weighting
    weights = total / (3 * counts)
    return torch.FloatTensor(weights)

class_weights = compute_class_weights(train_labels)
criterion = CrossEntropyLoss(weight=class_weights.to(device))
```

### What to do with the existing Directional Loss in trainer.py
The existing directional loss penalizes wrong-sign regression predictions. This was the right instinct but the wrong implementation vehicle — it is fighting a regression problem that should be a classification problem. Once you switch to CrossEntropyLoss with class weights, the directional signal is handled implicitly. Keep the old directional loss commented out in case you want to test it on the Stage 2 magnitude regressor later.

---

## CHANGE 3: FEATURES TO ADD IN feature_engineer.py

The current 18 features are a solid base. Do not remove any. Add these on top:

Features missing from current set:
- RSI divergence: rsi - rsi.shift(5) — captures momentum of momentum, not just level
- Volume surprise: volume / volume.rolling(20).mean() — normalized volume, more useful than raw OBV alone
- Bar range: (high - low) / close — volatility proxy per bar
- Close position in bar: (close - low) / (high - low + 1e-9) — where did price settle within the bar
- VWAP distance: (close - vwap) / close — already computing VWAP, this is a free additional feature

Cross-stock features — new and high value with 19 stocks:
- Sector-relative return: stock_return_30min - mean_return_of_sector_stocks_30min
- Cross-stock z-score: (stock_return - universe_mean) / universe_std over rolling 20 bars
- These require computing features AFTER aligning all stocks to the same timestamp index

Note: the existing feature mask in the GA chromosome has 18 bits. After adding new features, update the mask length to match. The DEAP chromosome definition will need the bit count updated — find the individual initialization in ga_optimizer.py and change the length parameter.

---

## CHANGE 4: GA CHROMOSOME EXTENSION (modify ga_optimizer.py)

### Current chromosome structure
[18 feature bits | window_idx | horizon_idx | hidden_idx | dropout_idx | lr_idx]

### New chromosome structure
[N feature bits | window_idx | horizon_idx | hidden_idx | dropout_idx | lr_idx | threshold_multiplier_idx | confidence_threshold_idx | class_weight_strategy_idx]

New genes to add:
- threshold_multiplier_idx: index into [0.3, 0.5, 0.7, 1.0] — multiplier on per-stock dynamic threshold
- confidence_threshold_idx: index into [0.60, 0.65, 0.70, 0.75, 0.80] — Model 2 approval gate
- class_weight_strategy_idx: index into ['balanced', 'sqrt_balanced', 'custom_heavy'] — how aggressively to upweight minorities

Update the feature bit count from 18 to however many features exist after the feature_engineer.py additions above.

### FITNESS FUNCTION — most important change in ga_optimizer.py

Currently uses Validation RMSE. This must change to simulated Sharpe ratio.
RMSE rewards being close to zero on noisy data. Sharpe rewards correct directional calls when you act.

```python
def compute_sharpe_fitness(model, val_loader, confidence_threshold, device):
    model.eval()
    all_probs = []
    all_true_returns = []

    with torch.no_grad():
        for X_batch, y_batch, returns_batch in val_loader:
            # y_batch = class labels (0/1/2)
            # returns_batch = actual forward returns (for P&L simulation)
            logits = model(X_batch.to(device))
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_true_returns.append(returns_batch.numpy())

    all_probs = np.vstack(all_probs)
    all_true_returns = np.concatenate(all_true_returns)

    predicted_class = np.argmax(all_probs, axis=1)
    max_confidence = all_probs.max(axis=1)

    # Only simulate trades where Model 1 confidence is above threshold
    # Model 2 is applied separately — this is just Model 1 inline confidence gate
    trade_mask = max_confidence > confidence_threshold

    if trade_mask.sum() < 10:
        return -999.0  # not enough trades to evaluate

    # Convert class to direction: 0=DOWN=-1, 1=NEUTRAL=0, 2=UP=+1
    direction = np.where(predicted_class == 2, 1,
                np.where(predicted_class == 0, -1, 0))

    pnl = direction[trade_mask] * all_true_returns[trade_mask]
    
    if pnl.std() < 1e-9:
        return -999.0

    sharpe = pnl.mean() / pnl.std() * np.sqrt(252 * 13)  # annualized, 13 bars/day
    
    # Keep the existing complexity penalty on top
    num_features = chromosome_feature_count(individual)
    penalty = lambda_complexity * num_features
    
    return sharpe - penalty

# In DEAP setup, the fitness is now maximized (not minimized)
# Change creator.FitnessMin to creator.FitnessMax if currently set to minimize
```

Note: the val_loader needs to yield the actual forward return value alongside the class label. Add a third element to the dataset in data_processor.py — the raw fwd_return float — so the fitness function can compute real P&L without needing to reverse-engineer it from the label.

---

## CHANGE 5: 3-WAY DATA SPLIT (modify data_processor.py)

Currently the pipeline likely has Train / Validation split. Need to add a third Calibration set for Model 2.

```python
# Suggested split for 2 years of data:
# Train:       first 60% of timeline — used to train Model 1
# Calibration: next 20% of timeline — used to train Model 2
# Test:        final 20% of timeline — used to evaluate both, never touched during training

def temporal_split(df, train_frac=0.60, cal_frac=0.20):
    n = len(df)
    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + cal_frac))
    return (
        df.iloc[:train_end],
        df.iloc[train_end:cal_end],
        df.iloc[cal_end:]
    )
```

Always split by TIME, never by random shuffle. Random shuffle causes look-ahead bias with time series data. The boundary-safe windowing already in data_processor.py must be applied within each split separately to prevent windows crossing split boundaries as well as stock boundaries.

---

## CHANGE 6: MODEL 2 — CONFIDENCE ESTIMATOR (new file: confidence_model.py)

### Concept
Model 2 is trained to answer: given that Model 1 just made this prediction under these market conditions, is it correct? It learns the regime-dependent reliability of Model 1.

Model 1's softmax output is NOT a reliable confidence score — neural networks are overconfident. Model 2 calibrates this using real historical outcomes.

For the architecture, a lightweight model is sufficient here. A gradient boosting classifier (XGBoost or sklearn GradientBoostingClassifier) often outperforms a neural net on this tabular task and is faster to tune. Keep it simple.

### Training procedure
```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score

def train_confidence_model(model1, cal_loader, device):
    model1.eval()
    m2_features = []
    m2_labels = []

    with torch.no_grad():
        for X_batch, y_batch in cal_loader:
            logits = model1(X_batch.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_true = y_batch.numpy()
            y_pred = np.argmax(probs, axis=1)

            # Model 2 features: market features (flattened last step) + Model 1 output
            last_step = X_batch[:, -1, :].cpu().numpy()  # most recent timestep features
            max_conf = probs.max(axis=1, keepdims=True)
            features = np.concatenate([last_step, probs, max_conf], axis=1)

            was_correct = (y_pred == y_true).astype(int)

            m2_features.append(features)
            m2_labels.append(was_correct)

    m2_features = np.vstack(m2_features)
    m2_labels = np.concatenate(m2_labels)

    # Train Model 2
    model2 = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model2.fit(m2_features, m2_labels)

    return model2
```

### PRECISION IS THE ONLY METRIC THAT MATTERS FOR MODEL 2

Precision = TP / (TP + FP)
Translation: of all bars where Model 2 says "Model 1 is correct here, take the trade", how many are actually correct?

We explicitly do NOT optimize for recall. We do NOT care how many correct Model 1 predictions Model 2 misses. Missing trades is fine. Approving bad trades is not.

```python
def evaluate_confidence_model(model2, m2_test_features, m2_test_labels, threshold=0.70):
    confidence_scores = model2.predict_proba(m2_test_features)[:, 1]
    approved = (confidence_scores > threshold).astype(int)

    precision = precision_score(m2_test_labels, approved, zero_division=0)
    trade_rate = approved.mean()

    print(f"Precision at threshold {threshold}: {precision:.3f}")
    print(f"Trade approval rate: {trade_rate:.3f}")
    # Target: precision > 0.60, trade_rate between 0.10 and 0.25
    return precision, trade_rate
```

Sweep the threshold from 0.55 to 0.85 and plot precision vs trade_rate. Pick the threshold where precision crosses 0.65 and trade_rate is still above 0.10. Below 10% trade rate the Sharpe estimate becomes unreliable due to sample size.

### What Model 2 implicitly learns from 30-min data
- Market open bars (9:30–10:00): high volatility, Model 1 unreliable → low confidence output
- Midday bars (12:00–14:00): low volume, mean-reverting, Model 1 unreliable
- Pre-close bars (15:30–16:00): directional momentum, Model 1 more reliable
- High volume bars: stronger signal, higher confidence
- Low-liquidity stocks in the 19-stock universe: Model 1 less reliable, Model 2 learns this per stock

### Data leakage — most critical warning in the entire document
NEVER train Model 2 on any data used to train Model 1.
NEVER let any data from the Test set touch either model during training.
The 3-way split in data_processor.py enforces this structurally.
Any violation here invalidates all results.

---

## CHANGE 7: EVALUATION METRICS (modify result_viewer.py)

Stop reporting Validation RMSE as the headline metric. It is no longer meaningful.

New metrics to compute and plot:

- Precision@HighConfidence: directional accuracy only on trades approved by Model 2. Primary metric. Target > 62%.
- Trade Rate: fraction of 30-min bars where Model 2 approves a signal. Target 10–25%.
- Annualized Sharpe Ratio: of the simulated P&L from approved trades. GA fitness target.
- Model 2 Precision: standalone precision of Model 2's approval decisions. Must be > 60%.
- Calibration plot: plot Model 2's predicted confidence vs actual accuracy in bins. Shows if scores are trustworthy.

Keep reporting training/validation loss curves and GA fitness evolution — these are still useful diagnostics.

```python
def directional_accuracy_on_approved(model1, model2, test_loader, conf_threshold, device):
    # Run both models, filter to approved trades only, compute accuracy
    # This is the number that matters for the thesis results table
    pass
```

---

## FULL ARCHITECTURE AFTER CHANGES

Data flow:
1. Raw 5-min XLSX → convert_xlsx_to_csv.py → cleaned CSV
2. CSV → data_processor.py → resample to 30-min + 1H, compute per-stock dynamic threshold
3. 30-min data → feature_engineer.py → 18 existing features + new features + cross-stock features
4. data_processor.py → 3-way temporal split (60/20/20), boundary-safe windowing per split
5. Train split → trainer.py → Model 1 (BiLSTM or BiLSTMAttention, 3-class, CrossEntropy + class weights)
6. Calibration split → Model 1 predictions → confidence_model.py → Model 2 (GradientBoosting, precision-optimized)
7. GA in ga_optimizer.py → evaluates [Model 1 + confidence threshold gate] using Sharpe fitness on validation
8. Test split → final evaluation of Model 1 + Model 2 combined → result_viewer.py

Model 1: BiLSTMAttentionModel, output layer changed to 3 classes, PyTorch CrossEntropyLoss with class weights
Model 2: GradientBoostingClassifier, trained on calibration set, threshold tuned for precision
GA: DEAP, chromosome extended with new genes, fitness changed from RMSE to Sharpe ratio

---

## IMPLEMENTATION ORDER

Ordered to minimize rework — each step builds on the previous without breaking what came before.

1. data_processor.py: Add resample_ohlcv() and build_multiresolution_frame(). Test that 30-min bars align correctly with no look-ahead.
2. data_processor.py: Add temporal_split() for 3-way split. Apply boundary-safe windowing within each split.
3. data_processor.py: Add label_returns() for 3-class labels. Add raw fwd_return as third output from DataLoader.
4. data_processor.py: Add compute_dynamic_threshold() per stock.
5. feature_engineer.py: Add the 5 missing features (RSI divergence, volume surprise, bar range, close position, VWAP distance). Add cross-stock relative strength. Update feature count.
6. model_builder.py: Change output layer in both BiLSTMModel and BiLSTMAttentionModel from size 1 to size 3. No softmax in forward(). Add inference helper that applies softmax.
7. trainer.py: Replace existing loss with CrossEntropyLoss + class weights. Remove or comment out the regression directional loss. Keep early stopping unchanged.
8. ga_optimizer.py: Extend chromosome with new genes (threshold_multiplier, confidence_threshold, class_weight_strategy). Update feature bit count. Replace RMSE fitness with Sharpe fitness using the simulation function. Change FitnessMin to FitnessMax in DEAP creator if needed.
9. NEW FILE confidence_model.py: Implement train_confidence_model() and evaluate_confidence_model(). Wire into main.py pipeline to run after Model 1 training on calibration split.
10. main.py: Update CLI to add timeframe argument, call new split logic, run confidence model training stage, report new metrics.
11. result_viewer.py: Add precision@threshold plot, trade rate report, calibration curve. Keep existing loss curve and GA evolution plots.
