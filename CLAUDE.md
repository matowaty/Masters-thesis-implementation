# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Masters thesis system: **"Forecasting the Rate of Return on Selected Financial Instruments Using Deep Learning and a Genetic Algorithm."** It forecasts returns on 20 US stocks from 5-minute OHLCV data using BiLSTM models whose hyperparameters and feature selection are optimized by a Genetic Algorithm (DEAP).

The authoritative spec lives in `INSTRUCTIONS/` (`01_PROJECT_ARCHITECTURE.md`, `02_DATA_PIPELINE.md`, `03_MODEL_AND_GA.md`, `04_EXPERIMENTS.md`, `05_AGENT_RULES.md`). `CLI_COMMANDS.md` documents every runnable command. Read the relevant instruction file before modifying a module — the project is built incrementally and these files define intended behavior. After running an experiment from `04_EXPERIMENTS.md`, log its results back into that file.

## Environment & dependencies

No `requirements.txt` exists. The code imports: `torch`, `deap` (the GA — *not* DEAP/PyGAD interchangeably, DEAP is used), `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`, and `pandas_ta_classic` for technical indicators. **Note:** the instruction docs mention the `ta` library, but the actual code uses `pandas_ta_classic` — follow the code.

Runs on Windows with PowerShell. Device selection is automatic (CUDA → MPS → CPU); GA "full" runs are designed for GPU and take hours on CPU.

## Commands

All experiments dispatch through `main.py <command>` (see the `commands` dict in `main()`). Common ones:

```bash
python smoke_test.py                  # ~20s end-to-end sanity check: loads JPM, builds features/windows, trains a tiny BiLSTM, confirms loss decreases
python main.py baseline               # ~1min, single-stock BiLSTM on JPM, fixed hyperparams
python main.py ga_fast                # ~2min, quick GA pipeline check (20% data, tiny pop/gens)
python main.py ga_fast_v2_multi_30min # quick V2 classification GA check
python main.py ga_v2_multi_15min      # full V2 GA, 15-min windows, maximizes Sharpe
python result_viewer.py RESULTS/<run_folder>   # render config/metrics/loss/fitness charts (saved as .png in the folder)
python convert_xlsx_to_csv.py         # one-time: convert DATA/*.xlsx → cleaned sorted *.csv
```

Use a `ga_fast*` command to validate GA changes quickly before launching a `ga_full*` / `ga_v2_multi_*` run. There is no test framework — `smoke_test.py` is the fast verification harness.

## Architecture

Two parallel pipelines coexist. Files **without** a suffix are the **V1 regression** pipeline; files with the **`_v2`** suffix are the **V2 classification** pipeline. `main.py` is a monolithic dispatcher holding a `run_*` function per experiment for both.

**V1 (regression)** — predicts a continuous rate of return:
- `data_processor.py` — loading, missing-interval fill (ffill→bfill), multi-horizon targets, **chronological** train/val/test split, training-only scaling, 3D windowing `[Samples, Time_Window, Features]`.
- `feature_engineer.py` — technical indicators (volatility, trend/momentum, volume).
- `model_builder.py` — BiLSTM (+ optional Attention) PyTorch classes, dynamic input sizing.
- `trainer.py` — training loop, early stopping, metrics (MSE/RMSE/MAE/R²/directional accuracy).
- `ga_optimizer.py` — DEAP GA over a chromosome of {window size, look-forward horizon, binary feature mask, hidden units, dropout, learning rate}; fitness minimizes `RMSE_val + λ·sum(feature_mask)` (accuracy vs. complexity).

**V2 (classification, market-aware)** — predicts UP/NEUTRAL/DOWN and simulates trading:
- `data_processor_v2.py` / `feature_engineer_v2.py` — resamples 5-min → 30-min (or 15-min) bars, aligns all stocks chronologically, injects market-wide `Index_Return` / `Index_Volatility` features for cross-stock context, converts target to 3 classes via dynamic volatility thresholding.
- `trainer_v2.py` — Model 1: classification BiLSTM with `CrossEntropyLoss` and inverse-frequency class weighting.
- `confidence_model.py` — Model 2: a `HistGradientBoostingClassifier` (sklearn) trained on a separate **calibration split**, predicting whether Model 1 is correct; acts as a trade filter.
- `pipeline_evaluator_v2.py` — Sharpe Ratio, Trade Rate, Precision@HighConfidence; the V2 GA fitness simulates P&L on the calibration set using only Model-2-approved trades and maximizes the **Annualized Sharpe Ratio**.
- `ga_optimizer_v2.py` — V2 GA driver.

**Shared infrastructure:**
- `result_logger.py` — every run gets a timestamped subfolder under `RESULTS/`; stores config, metrics, GA stats, and checkpoints.
- GA runs checkpoint into `checkpoints/` for resume/persistence; inspect via `view_checkpoint_results.py`.
- `DATA/` holds the 20 stocks as `.csv` (committed) and `.xlsx` (raw source). `RESULTS/`, `checkpoints/`, `artifacts/`, and `*.pt` are gitignored.

## Critical rules (financial ML)

- **No data leakage.** Splits are strictly chronological — never shuffle time-series. Scalers are fit on the **training set only**, then applied to val/test.
- **Per-stock, boundary-safe windowing** in multi-stock modes: windows must never cross stock boundaries, and scaling is per-stock.
- **Tensor shapes:** annotate expected shapes (`[batch_size, sequence_length, num_features]`) in comments around layer transitions — the pipeline is shape-sensitive.
- **Strict modularity:** keep changes within the module that owns the concern (per the `INSTRUCTIONS/` file breakdown); don't bleed training logic into data or GA modules. V1 and V2 files are independent — changing one must not break the other.
- Use the `logging` module (not bare `print`) for GA generations, chosen chromosomes, and metrics.
