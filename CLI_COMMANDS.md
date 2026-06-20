# CLI Commands Reference

All available commands for the Financial Forecasting Pipeline.

---

## Experiments (`main.py`)

| Command | Description |
|---|---|
| `python main.py baseline` | Train a baseline BiLSTM on JPM with fixed hyperparameters (window=12, hidden=64, dropout=0.2, lr=1e-3). Evaluates on the test set and saves results + checkpoint. **~1 min on CPU.** |
| `python main.py time_horizon` | Train 3 separate BiLSTMs for t+1, t+2, t+3 horizons with identical settings. Measures how prediction error degrades as the look-ahead extends. **~3 min on CPU.** |
| `python main.py attention` | Train BiLSTM and BiLSTM+Attention side-by-side with identical hyperparameters. Determines if the Attention mechanism improves results. **~2 min on CPU.** |
| `python main.py ga_fast` | Quick GA test using 20% of data, 5 epochs per individual, 6 population, 3 generations. For verifying the GA pipeline works locally. **~2 min on CPU.** |
| `python main.py ga_full` | Full GA optimization using 100% data, 50 epochs per individual, 20 population, 30 generations. **Designed for GPU. Will take hours on CPU.** |
| `python main.py pipeline` | Run the full pipeline (currently routes to baseline). |

## Multi-Stock Experiments — Universal Model (`main.py`)

Train a single model on **all 20 stocks** simultaneously. Uses per-stock scaling and boundary-safe windowing (windows never cross stock boundaries).

| Command | Description |
|---|---|
| `python main.py baseline_multi` | Train a universal BiLSTM on all stocks with fixed hyperparameters (same as single-stock baseline). **~20 min on CPU.** |
| `python main.py ga_fast_multi` | Quick multi-stock GA test using 20% of per-stock data, 5 epochs, 6 population, 3 generations. **~30 min on CPU.** |
| `python main.py ga_full_multi` | Full multi-stock GA optimization using 100% data, 50 epochs, 20 population, 30 generations. **Designed for GPU.** |

---

## V2 Classification Experiments (`main.py`)

The V2 Pipeline aligns all stocks chronologically, resamples to 30-min intervals, injects Market-wide features (Index Return, Index Volatility) to provide cross-stock context, and transforms the target into a 3-class problem (UP/DOWN/NEUTRAL) with dynamic volatility thresholding. It trains Model 1 (Classification BiLSTM) and Model 2 (Confidence Meta-Model) sequentially.

| Command | Description |
|---|---|
| `python main.py baseline_v2_multi` | Train the V2 Classification pipeline on all stocks. Computes CrossEntropyLoss, trains Model 2 on the Calibration set, and evaluates Trade Rate and Sharpe Ratio on the Test set. |
| `python main.py attention_v2_multi` | Train both V2 BiLSTM and BiLSTM+Attention on all stocks to compare if the Attention mechanism improves results in the classification setting. |
| `python main.py ga_v2_multi_30min` | Full V2 GA optimization using 30-minute windows. Tunes feature selection and hyperparameters by maximizing the Sharpe Ratio. |
| `python main.py ga_v2_multi_15min` | Full V2 GA optimization using 15-minute windows. Tunes feature selection and hyperparameters by maximizing the Sharpe Ratio. |
| `python main.py ga_fast_v2_multi_30min` | Quick V2 GA test using 30-minute windows. 20% of data, 5 epochs, 6 population, 3 generations. |
| `python main.py ga_fast_v2_multi_15min` | Quick V2 GA test using 15-minute windows. 20% of data, 5 epochs, 6 population, 3 generations. |

---

## Results Viewer (`result_viewer.py`)

| Command | Description |
|---|---|
| `python result_viewer.py RESULTS/<folder_name>` | Visualize a completed experiment run. Shows config table, metrics table, training loss curves, and (for GA runs) fitness evolution chart + best chromosome. Charts are saved as `.png` inside the folder. |

**Example:**
```bash
python result_viewer.py RESULTS/baseline_jpm_20260521_094956
python result_viewer.py RESULTS/ga_features_only_jpm_20260521_095116
```

---

## Data Utilities

| Command | Description |
|---|---|
| `python convert_xlsx_to_csv.py` | Convert all Refinitiv `.xlsx` files in `DATA/` to sorted, clean `.csv` files. One-time utility. |
| `python smoke_test.py` | End-to-end pipeline validation. Loads JPM, generates features, creates windows, trains a tiny inline BiLSTM for 10 epochs. Confirms gradients flow and loss decreases. **~20 sec.** |
