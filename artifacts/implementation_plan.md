# ML & GA Financial Forecaster — Implementation Plan

## Current State

| Module | Status | Notes |
|---|---|---|
| [convert_xlsx_to_csv.py](file:///c:/Politechnika/Masters/Thesis%20-%202/convert_xlsx_to_csv.py) | ✅ Done | Refinitiv XLSX → clean CSV |
| [data_processor.py](file:///c:/Politechnika/Masters/Thesis%20-%202/data_processor.py) | ✅ Done | Load, fill, targets, split, scale, window — all implemented. Data is sorted ascending (oldest → newest) ✓ |
| [feature_engineer.py](file:///c:/Politechnika/Masters/Thesis%20-%202/feature_engineer.py) | 🟡 Skeleton | Class + method signatures with docstrings, but **all bodies are `pass`** |
| [model_builder.py](file:///c:/Politechnika/Masters/Thesis%20-%202/model_builder.py) | 🟡 Skeleton | `BiLSTMModel` + `BiLSTMAttentionModel` — empty bodies |
| [trainer.py](file:///c:/Politechnika/Masters/Thesis%20-%202/trainer.py) | 🟡 Skeleton | `EarlyStopping`, `Trainer` — empty bodies |
| [ga_optimizer.py](file:///c:/Politechnika/Masters/Thesis%20-%202/ga_optimizer.py) | 🟡 Skeleton | `Chromosome`, `GAOptimizer` — empty bodies |
| [main.py](file:///c:/Politechnika/Masters/Thesis%20-%202/main.py) | 🟡 Skeleton | Pipeline functions — empty bodies |

**Data:** 19 US equities (5-min OHLCV CSVs, ~1.4 MB each) in `DATA/`. The 20th ticker will be added later.

---

## Resolved Decisions

| Question | Decision |
|---|---|
| VWAP session boundary | **Intraday (daily reset)** — reset cumulative sums at each market open |
| Smoke test ticker | **JPM** |
| Leading NaN handling | **Drop** ~20 rows after feature engineering (acceptable) |
| Ticker count | **19 for now**, 20th to be added later |
| Data ordering | Verified: `data_processor.py` already sorts ascending (`df.sort_index(ascending=True)`) ✓ |
| TA library | **`pandas-ta-classic`** (`pip install pandas-ta-classic`, imported as `import pandas_ta as ta`) |

---

## Immediate Deliverables (This Session)

### 1. Feature Engineering — `feature_engineer.py`
### 2. Baseline Smoke Test — `smoke_test.py`

---

## Proposed Changes

### Phase 1A — Feature Engineering (now)

#### [MODIFY] [feature_engineer.py](file:///c:/Politechnika/Masters/Thesis%20-%202/feature_engineer.py)

Implement every method body. The 12 generated feature columns will be:

| Family | Feature | Column Name(s) | `pandas-ta` Command |
|---|---|---|---|
| **Volatility** | High-Low Spread | `HL_Spread` | `df['HL_Spread'] = (df['High'] - df['Low']) / df['Low']` |
| | Close-Open Spread | `CO_Spread` | `df['CO_Spread'] = (df['Close'] - df['Open']) / df['Open']` |
| | Bollinger Bands (20) | `BB_High`, `BB_Mid`, `BB_Low` | `df.ta.bbands(length=20, std=2, append=True)` → rename columns |
| | ATR (14) | `ATR` | `df.ta.atr(length=14, append=True)` → rename to `ATR` |
| **Trend** | SMA | `SMA_10`, `SMA_20` | `df.ta.sma(length=10, append=True)` + `df.ta.sma(length=20, append=True)` |
| | EMA | `EMA_10`, `EMA_20` | `df.ta.ema(length=10, append=True)` + `df.ta.ema(length=20, append=True)` |
| | MACD | `MACD_Line`, `MACD_Histogram` | `df.ta.macd(fast=12, slow=26, signal=9, append=True)` → rename |
| | RSI (14) | `RSI` | `df.ta.rsi(length=14, append=True)` → rename to `RSI` |
| **Volume** | VWAP | `VWAP` | Custom: `groupby(date)` → `cumsum(TP * Vol) / cumsum(Vol)` (daily reset) |
| | OBV | `OBV` | `df.ta.obv(append=True)` → rename to `OBV` |

**Total columns after engineering:** 5 OHLCV base + 17 indicator columns = **22 columns**.
The GA feature mask will cover all 22.

**Key implementation details:**
- `import pandas_ta as ta` — all `df.ta.*` calls come from `pandas-ta-classic`.
- Auto-generated column names (e.g. `BBL_20_2.0`, `ATRr_14`, `RSI_14`) will be **renamed** to our cleaner convention (`BB_Low`, `ATR`, `RSI`) immediately after computation.
- `add_all_features()` calls every method in order, then **drops** rows with NaN (from rolling window warm-up), and stores `self.feature_names`.
- VWAP resets daily — group by `df.index.date`, compute `cumsum(TP * Vol) / cumsum(Vol)` within each day.

---

### Phase 1B — Baseline Smoke Test (now)

#### [NEW] [smoke_test.py](file:///c:/Politechnika/Masters/Thesis%20-%202/smoke_test.py)

A standalone script that:
1. Loads **JPM** from `DATA/JPM.csv`.
2. Runs `FeatureEngineer.add_all_features()`.
3. Computes targets via `DataProcessor.compute_targets()`.
4. Does chronological split + scaling + windowing (using existing `DataProcessor` methods).
5. Builds a **minimal BiLSTM** (inline, not from `model_builder.py` — to avoid coupling to the still-empty skeleton).
6. Trains for 5–10 epochs.
7. Prints MSE / RMSE on the validation set.

**Purpose:** Validate the end-to-end data flow from raw CSV → features → windows → PyTorch tensors → forward pass → loss. We are **not** expecting good predictions — just confirming the pipeline produces valid gradients and the loss decreases.

> [!NOTE]
> This is a throwaway diagnostic script. Once `model_builder.py` and `trainer.py` are implemented, it can be deleted.

---

## Full Roadmap (Future Sessions)

### Phase 2 — Core Model & Training Loop

#### [MODIFY] [model_builder.py](file:///c:/Politechnika/Masters/Thesis%20-%202/model_builder.py)

- Implement `get_device()` — CUDA → MPS → CPU detection.
- Implement `BiLSTMModel.__init__` + `forward`:
  - `nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)`
  - `nn.Dropout(dropout)`
  - `nn.Linear(hidden_size * 2, 1)` — take last time-step hidden state.
  - Inline shape comments at every layer.
- Implement `BiLSTMAttentionModel.__init__` + `forward` + `attention`:
  - Additive attention over all LSTM hidden states.
  - `nn.Linear(hidden_size * 2, 1)` for attention scoring → softmax → weighted sum → dense → output.

#### [MODIFY] [trainer.py](file:///c:/Politechnika/Masters/Thesis%20-%202/trainer.py)

- `EarlyStopping.__init__` + `__call__` — patience counter, best-loss tracking.
- `Trainer.__init__` — set up AdamW + MSE/Huber criterion + move model to device.
- `create_dataloaders` — numpy → `torch.FloatTensor` → `TensorDataset` → `DataLoader`.
- `_train_one_epoch` — standard forward/backward/step loop with gradient clipping.
- `_validate` — `torch.no_grad()` evaluation loop.
- `train` — epoch loop calling `_train_one_epoch` + `_validate` + `EarlyStopping` + logging.
- `evaluate` — compute MSE, RMSE, MAE, R², Directional Accuracy on test set.
- `directional_accuracy` — `np.sign(pred) == np.sign(true)` percentage.

---

### Phase 3 — Baseline Experiment (Experiment Phase 1)

#### [MODIFY] [main.py](file:///c:/Politechnika/Masters/Thesis%20-%202/main.py)

- Implement `setup_logging()` with format: `%(asctime)s %(levelname)-8s %(name)s — %(message)s`.
- Implement `run_baseline_experiment()`:
  - Fixed params: `window=12`, `target=Target_1_Tick`, `hidden=64`, `dropout=0.2`, `lr=1e-3`.
  - All features enabled.
  - Train on one or all 19 tickers, record MSE / MAE / RMSE / R² / DA.
- Log results back into `04_EXPERIMENTS.md` Phase 1 section.

---

### Phase 4 — GA Optimization

#### [MODIFY] [ga_optimizer.py](file:///c:/Politechnika/Masters/Thesis%20-%202/ga_optimizer.py)

- `Chromosome.__init__` — random initialization from allowed value sets.
- `Chromosome.to_dict` / `from_vector` — serialization for DEAP.
- `GAOptimizer.__init__` — store config, call `_setup_deap_toolbox()`.
- `_setup_deap_toolbox` — register `creator.FitnessMin`, `creator.Individual`, population, `cxTwoPoint`, `mutFlipBit` / `mutUniformInt`, `selTournament`.
- `evaluate_individual` — decode chromosome → build model → train (short epochs) → return `(RMSE_val + λ * feature_count,)`.
- `run` — DEAP `eaSimple` or custom loop with hall-of-fame, stats, and logging.
- `log_generation` — per-generation min/avg/max fitness.

#### [MODIFY] [main.py](file:///c:/Politechnika/Masters/Thesis%20-%202/main.py)

- Implement `run_ga_optimization()` — Phase 3 (feature-only GA) + Phase 4 (full GA) experiment flows.
- Implement `run_pipeline()` — the full orchestration tying everything together.

---

## Verification Plan

### Automated Tests (Phase 1A + 1B)

1. **Feature shape check:** After `add_all_features()`, assert the DataFrame has exactly 22 columns (5 OHLCV + 17 indicators) and no NaN values.
2. **Smoke test end-to-end:** Run `smoke_test.py` — confirm:
   - No crashes or shape mismatches.
   - Loss decreases over 5–10 epochs (gradient flow works).
   - Output tensor shape is `[batch_size, 1]`.
3. **Scaling leak check:** Verify scaler is fitted only on train partition (existing `DataProcessor` behavior).

### Manual Verification
- Review printed feature statistics and shapes in the smoke test output.
- Inspect a sample window to visually confirm the data looks reasonable.
