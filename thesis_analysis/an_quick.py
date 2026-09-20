"""AN-11: seeded quick repeats of two V1 experiments on JPM (RQ2 prediction horizon, RQ4 attention).

Runs were produced by quick_runs.py (a thin seeded wrapper around main.py, CPU, torch 2.14) and live in RESULTS/ next to a seed.json.
Three seeds per configuration; identical seeds give identical results for configurations that coincide (sanity check).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, OI, REPO, TEXT_WIDTH_IN, flush_numbers, plt, register_number, savefig, write_table

QUICK = Path(os.environ.get("QUICK_RESULTS", REPO / "RESULTS"))

sys.path.insert(0, str(REPO))
from data_processor import DataProcessor  # noqa: E402
from feature_engineer import FeatureEngineer  # noqa: E402


def coin_flip_da() -> dict[int, float]:
    """Directional accuracy (zero-return bars count as misses) of a random-sign predictor on the JPM V1 test segment, per horizon."""
    dp = DataProcessor()
    df = dp.handle_missing_intervals(dp.load_data(str(DATA_DIR / "JPM.csv")))
    df = dp.compute_targets(FeatureEngineer().add_all_features(df))
    _, _, te = dp.chronological_split(df)
    return {h: 50 * (1 - float((te[f"Target_{h}_Tick"] == 0).mean())) for h in (1, 2, 3)}


def load() -> pd.DataFrame:
    rows = []
    for d in sorted(QUICK.iterdir()):
        sj = d / "seed.json"
        if not sj.exists() or not (d / "metrics.json").exists():
            continue
        seed = json.load(open(sj))["seed"]
        m = json.load(open(d / "metrics.json"))
        name = d.name
        if name.startswith("attention_bilstm_attn"):
            cfg = "BiLSTM + attention (h=1)"
        elif name.startswith("attention_bilstm"):
            cfg = "BiLSTM (h=1)"
        elif name.startswith("time_horizon_t1"):
            cfg = "horizon t+1"
        elif name.startswith("time_horizon_t2"):
            cfg = "horizon t+2"
        elif name.startswith("time_horizon_t3"):
            cfg = "horizon t+3"
        else:
            continue
        rows.append({"cfg": cfg, "seed": seed, "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "da": m["directional_accuracy"], "dir": name})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    if d.empty:
        raise SystemExit("no quick runs found")
    jpm = pd.read_csv(DATA_DIR / "JPM.csv", parse_dates=["Datetime"], index_col="Datetime").sort_index()
    order = ["BiLSTM (h=1)", "BiLSTM + attention (h=1)", "horizon t+1", "horizon t+2", "horizon t+3"]
    coin = coin_flip_da()
    for h in (1, 2, 3):
        register_number(f"QuickCoinH{h}", f"{coin[h]:.1f}\\%")
    hz = {"horizon t+2": 2, "horizon t+3": 3}
    rows = []
    for cfg in order:
        s = d[d["cfg"] == cfg].sort_values("seed")
        rows.append([cfg, str(len(s)), f"{1e3 * s['rmse'].mean():.3f} $\\pm$ {1e3 * s['rmse'].std(ddof=1):.3f}",
                     f"{s['r2'].mean():.5f} $\\pm$ {s['r2'].std(ddof=1):.5f}",
                     f"{s['da'].mean():.1f} $\\pm$ {s['da'].std(ddof=1):.1f}\\%",
                     " / ".join(f"{v:.1f}" for v in s["da"]), f"{coin[hz.get(cfg, 1)]:.1f}\\%"])
    write_table("t13_quick_runs",
                ["Configuration (JPM, 5-min bars)", "Seeds", "RMSE ($\\times 10^{-3}$)", "$R^2$", "DA", "DA per seed (\\%)", "Coin flip"],
                rows, "lrrrrrr",
                "Seeded repeats (seeds 1--3) of the V1 experiments. RMSE, $R^2$ and DA are on the test set (last 10\\% of the JPM series); "
                "mean $\\pm$ standard deviation over seeds. DA counts zero-return bars as misses. ``BiLSTM (h=1)'' and ``horizon $t+1$'' are the same configuration "
                "run by two different experiment functions; identical seeds give identical results.",
                "tab:quick_runs", resize=True, font=r"\footnotesize")

    base = d[d["cfg"] == "BiLSTM (h=1)"].sort_values("seed")
    att = d[d["cfg"] == "BiLSTM + attention (h=1)"].sort_values("seed")
    diff_da = att["da"].to_numpy() - base["da"].to_numpy()
    diff_r2 = att["r2"].to_numpy() - base["r2"].to_numpy()
    register_number("AttDaDiffMean", f"{diff_da.mean():+.1f}")
    register_number("AttDaDiffSd", f"{diff_da.std(ddof=1):.1f}")
    register_number("AttDaDiffMin", f"{diff_da.min():+.1f}")
    register_number("AttDaDiffMax", f"{diff_da.max():+.1f}")
    register_number("AttRTwoDiffMax", f"{np.abs(diff_r2).max():.5f}")
    register_number("BaseDaSeedRange", f"{base['da'].min():.1f}--{base['da'].max():.1f}\\%")
    same = np.allclose(base["rmse"].to_numpy(), d[d["cfg"] == "horizon t+1"].sort_values("seed")["rmse"].to_numpy()) and \
        np.allclose(base["da"].to_numpy(), d[d["cfg"] == "horizon t+1"].sort_values("seed")["da"].to_numpy())
    register_number("QuickRepro", "yes" if same else "no")
    h = d[d["cfg"].str.startswith("horizon")].copy()
    h["h"] = h["cfg"].str[-1].astype(int)
    rm = h.groupby("h")["rmse"].mean()
    ratios = (rm / rm.loc[1]).to_dict()
    for k in (2, 3):
        register_number(f"RmseRatioH{k}", f"{ratios[k]:.2f}")
        register_number(f"SqrtH{k}", f"{np.sqrt(k):.2f}")
    r2h = h.groupby("h")["r2"].mean()
    register_number("HorizonRTwoMin", f"{h['r2'].min():.5f}")
    register_number("HorizonRTwoMax", f"{h['r2'].max():.5f}")
    print("RMSE ratios vs h=1:", {k: round(v, 3) for k, v in ratios.items()}, "sqrt(h):", [round(np.sqrt(k), 3) for k in (2, 3)])
    print("attention DA diffs per seed:", diff_da.round(2), "R2 diffs:", diff_r2.round(6), "| repro:", same)
    print("horizon R2:", r2h.round(5).to_dict())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0))
    hs = np.array([1, 2, 3])
    for seed, mk in zip(sorted(h["seed"].unique()), ["o", "s", "^"]):
        s = h[h["seed"] == seed].sort_values("h")
        ax1.plot(s["h"], 1e3 * s["rmse"], mk, color=OI["blue"], label=f"seed {seed}", ms=5)
    ax1.plot(hs, 1e3 * rm.loc[1] * np.sqrt(hs), "--", color=OI["verm"], label="$\\sqrt{h}\\times$ RMSE at $h=1$")
    ax1.set_xticks(hs)
    ax1.set_xlabel("Prediction horizon $h$ (5-minute bars)")
    ax1.set_ylabel("Test RMSE ($\\times 10^{-3}$)")
    ax1.set_title("(a) Error grows like the volatility itself")
    ax1.legend(fontsize=7)
    for i, (cfg, col) in enumerate([("BiLSTM (h=1)", OI["sky"]), ("BiLSTM + attention (h=1)", OI["orange"])]):
        s = d[d["cfg"] == cfg].sort_values("seed")
        ax2.plot(s["seed"] + (i - 0.5) * 0.12, s["da"], "o", color=col, ms=6, label=cfg)
    ax2.axhline(50, color="black", lw=0.8, ls="--")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Directional accuracy (%)")
    ax2.set_ylim(44, 52)
    ax2.set_title("(b) Attention vs plain BiLSTM")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, "f23_quick_runs")
    flush_numbers()


if __name__ == "__main__":
    main()
