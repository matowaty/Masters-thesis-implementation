"""AN-1b: Version 1 (regression on 5-minute bars) -- T-06 and F-08.

Input: inputs/other_runs.csv (one row per non-GA run, assembled from the metrics.json files of the runs).
V1 'directional accuracy' is mean(sign(pred) == sign(actual)); a bar with an exactly zero return can never count as a hit.
The share of such bars in the JPM test segment is measured here to give the accuracy on non-zero bars (DA / (1 - z)).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

import sys

from common import DATA_DIR, INPUTS, OI, REPO, TEXT_WIDTH_IN, flush_numbers, plt, register_number, savefig, write_table

sys.path.insert(0, str(REPO))
from data_processor import DataProcessor  # noqa: E402
from feature_engineer import FeatureEngineer  # noqa: E402


def v1_test_targets() -> dict[str, np.ndarray]:
    """Next-bar returns of the V1 test segments (last 10% of each series), prepared exactly as the V1 pipeline does
    (forward fill of empty rows, features, chronological split); rows with an empty price were filled forward, so they count as zero returns."""
    out = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        dp = DataProcessor()
        df = dp.handle_missing_intervals(dp.load_data(str(f)))
        df = FeatureEngineer().add_all_features(df)
        df = dp.compute_targets(df)
        _, _, te = dp.chronological_split(df)
        out[f.stem] = te["Target_1_Tick"].to_numpy()
    return out


def main() -> None:
    d = pd.read_csv(INPUTS / "other_runs.csv")
    v1 = d[d["rmse"].notna()].copy()
    v1["bug"] = v1["rmse"] < 1e-4  # two runs report an RMSE of ~1e-6: targets were left unscaled (x1000 missing)
    tt = v1_test_targets()
    tj = tt["JPM"]
    tm = np.concatenate(list(tt.values()))
    z, z_multi = float((tj == 0).mean()), float((tm == 0).mean())
    print(f"JPM V1 test segment: {len(tj)} bars, zero-return share {z:.3f}, positive {np.mean(tj > 0):.3f}, negative {np.mean(tj < 0):.3f}")
    print(f"multi-stock V1 test segment: {len(tm)} bars, zero-return share {z_multi:.3f}, positive {np.mean(tm > 0):.3f}, negative {np.mean(tm < 0):.3f}")
    register_number("VOneZeroShare", f"{100 * z:.1f}\\%")
    register_number("VOneZeroShareMulti", f"{100 * z_multi:.1f}\\%")
    register_number("VOneCoinJpm", f"{50 * (1 - z):.1f}\\%")
    register_number("VOneCoinMulti", f"{50 * (1 - z_multi):.1f}\\%")
    register_number("VOneAlwaysDownJpm", f"{100 * np.mean(tj < 0):.1f}\\%")
    register_number("VOneAlwaysUpJpm", f"{100 * np.mean(tj > 0):.1f}\\%")
    register_number("VOneAlwaysDownMulti", f"{100 * np.mean(tm < 0):.1f}\\%")
    register_number("VOneAlwaysUpMulti", f"{100 * np.mean(tm > 0):.1f}\\%")
    register_number("VOneNTestJpm", f"{len(tj):,}".replace(",", "\\,"))
    ok = v1[~v1["bug"]]
    register_number("VOneRSquaredMin", f"{ok['r2'].min():.2f}")
    register_number("VOneRSquaredMax", f"{ok['r2'].max():.4f}")
    register_number("VOneDAMin", f"{ok['da'].min():.1f}\\%")
    register_number("VOneDAMax", f"{ok['da'].max():.1f}\\%")
    zs = np.where(ok["run"].str.contains("multi"), z_multi, z)
    excl = ok["da"].to_numpy() / (1 - zs)
    register_number("VOneDAExclMin", f"{excl.min():.1f}\\%")
    register_number("VOneDAExclMax", f"{excl.max():.1f}\\%")
    register_number("VOneDAJpmMin", f"{ok[~ok['run'].str.contains('multi')]['da'].min():.1f}\\%")
    register_number("VOneDAJpmMax", f"{ok[~ok['run'].str.contains('multi')]['da'].max():.1f}\\%")
    lo_m, hi_m = ok[ok['run'].str.contains('multi')]['da'].min(), ok[ok['run'].str.contains('multi')]['da'].max()
    register_number("VOneDAMultiRange", f"{lo_m:.1f}\\%" if f"{lo_m:.1f}" == f"{hi_m:.1f}" else f"{lo_m:.1f}--{hi_m:.1f}\\%")
    register_number("VOneRuns", str(len(v1)))
    register_number("VOneRunsBug", str(int(v1["bug"].sum())))
    print(f"V1 runs {len(v1)} (units bug: {int(v1['bug'].sum())}); R2 {ok['r2'].min():.3f}..{ok['r2'].max():.4f}; DA {ok['da'].min():.1f}..{ok['da'].max():.1f}; excl-zero {excl.min():.1f}..{excl.max():.1f}")

    rows = []
    for i, (_, r) in enumerate(v1.iterrows(), 1):
        stamp = r["run"].rsplit("_", 2)
        kind = "baseline" if r["run"].startswith("baseline") else "GA (features only)"
        scope = "multi-stock" if "multi" in r["run"] else "JPM"
        rmse = f"{r['rmse']:.2e}" if r["bug"] else f"{r['rmse']:.6f}"
        rows.append([f"V1-{i}", r["source"], kind, scope, f"{r['lr']:g}" if not pd.isna(r["lr"]) else "--",
                     f"{int(r['window'])}" if not pd.isna(r["window"]) else "--", rmse + ("$^\\dagger$" if r["bug"] else ""),
                     f"{r['r2']:.4f}", f"{r['da']:.1f}\\%", f"{r['da'] / (1 - (z if scope == 'JPM' else z_multi)):.1f}\\%"])
    write_table("t06_v1_results",
                ["Run", "Machine", "Kind", "Scope", "LR", "Window", "RMSE", "$R^2$", "DA", "DA excl. zero bars (est.)"],
                rows, "lllllrrrrr",
                "Version 1 (regression on 5-minute returns, test set = last 10\\% of each series). DA is the share of bars where the predicted sign equals the "
                "realised sign; a zero realised return counts as a miss. The last column divides DA by one minus the share of zero-return bars in the test segment of the "
                f"respective scope ({100 * z:.1f}\\% for JPM, {100 * z_multi:.1f}\\% for all stocks; these shares include rows with an empty price that were filled forward) and is an estimate. $^\\dagger$ Run with an RMSE of about $10^{{-6}}$: the target scaling was not applied, so RMSE and MAE are not comparable "
                "with the other rows.",
                "tab:v1_results", resize=True, font=r"\footnotesize")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0))
    x = np.arange(len(v1))
    colours = [OI["blue"] if "multi" in r else (OI["orange"] if r.startswith("ga_") else OI["sky"]) for r in v1["run"]]
    ax1.bar(x, v1["r2"], color=colours)
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"V1-{i + 1}" for i in x], rotation=90, fontsize=6.5)
    ax1.set_ylabel("$R^2$ on the test set")
    ax1.set_title("(a) Explained variance")
    ax2.bar(x, v1["da"], color=colours)
    ax2.axhline(50 * (1 - z), color="black", lw=0.8, ls="--", label=f"coin flip, JPM ({50 * (1 - z):.1f}%)")
    ax2.axhline(50 * (1 - z_multi), color="black", lw=0.8, ls=":", label=f"coin flip, all stocks ({50 * (1 - z_multi):.1f}%)")
    ax2.legend(fontsize=6, loc="upper right")
    ax2.set_ylim(42, 50)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"V1-{i + 1}" for i in x], rotation=90, fontsize=6.5)
    ax2.set_ylabel("Directional accuracy (%)")
    ax2.set_title("(b) Directional accuracy\n(zero-return bars count as misses)")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=OI["sky"], label="baseline"), Patch(color=OI["orange"], label="GA (features)"), Patch(color=OI["blue"], label="multi-stock")],
               fontsize=6.5, loc="lower right")
    fig.tight_layout()
    savefig(fig, "f08_v1_results")
    flush_numbers()


if __name__ == "__main__":
    main()
