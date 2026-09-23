# Financial Forecasting Pipeline — Master's Thesis Code

Code for the master's thesis *"Forecasting the Rate of Return on Selected Financial Instruments Using Deep Learning and a Genetic Algorithm."* Forecasts returns on 19 US stocks from 5-minute OHLCV data using BiLSTM models whose hyperparameters and feature selection are optimized by a genetic algorithm (DEAP).

Two pipelines coexist by design and are compared in the thesis:

- **V1** (files without a suffix) — regression on continuous returns.
- **V2** (files with a `_v2` suffix) — 3-class direction classification (UP/NEUTRAL/DOWN), a Sharpe-ratio-optimized GA, and a confidence meta-model (Model 2, `confidence_model.py`) that filters trades.

See `CLAUDE.md` for the full architecture summary and `INSTRUCTIONS/` for the detailed design spec.

## Setup

```bash
pip install -r requirements.txt
python smoke_test.py   # ~20s end-to-end sanity check
```

Python 3.10+. Device selection is automatic (CUDA -> MPS -> CPU). Full GA runs (`ga_full*`, `ga_v2_multi_*`) are designed for GPU and take hours on CPU — use the `ga_fast*` variants to validate changes quickly.

## Data

`DATA/` (20 US stocks, 5-minute OHLCV, source: LSEG/Refinitiv) is **not included in this repository** — the data license does not permit redistribution. To run the pipeline, place the stocks' cleaned `.csv` files in `DATA/` (see `INSTRUCTIONS/02_DATA_PIPELINE.md` for the expected format; `convert_xlsx_to_csv.py` is the one-time script that produces them from a raw LSEG `.xlsx` export).

## Running experiments

All experiments dispatch through `main.py <command>`. Full command reference: `CLI_COMMANDS.md`. A few examples:

```bash
python main.py baseline              # V1 baseline, single stock, ~1 min on CPU
python main.py ga_fast                # quick V1 GA sanity check, ~2 min on CPU
python main.py ga_v2_multi_15min      # full V2 GA, maximizes Sharpe ratio (designed for GPU)
python result_viewer.py RESULTS/<run_folder>   # charts for a completed run
```

## How the thesis numbers were produced

Every figure, table and in-text number in the thesis is generated from code in `thesis_analysis/` — nothing was entered by hand. Full provenance (which script produced which figure/table, and how to reproduce it) is documented in `thesis_analysis/SOURCES.md`.

```bash
cd thesis_analysis
./run_all.sh
```

Raw experiment outputs (`RESULTS/`, model checkpoints) are not included in this repository; they are provided separately.
