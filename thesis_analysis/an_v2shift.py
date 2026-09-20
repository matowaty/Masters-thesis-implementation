"""AN-12: seeded check of the one-bar window offset of the V2 pipeline (Model 1 only, 30-minute bars, m = 0.3).

Runs were produced by quick_v2_shift.py: 'base' (window = bars i-w..i-1, as in the thesis pipeline) and 'shift'
(window = bars i-w+1..i, i.e. ending at the decision bar) on identical rows, labels, split, scaler and seeds.
Results live in RESULTS_V2SHIFT/<variant>_seed<k>.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import (OI, REPO, TEXT_WIDTH_IN, flush_numbers, plt, register_number, savefig, session_labels, to_et,
                    write_table)

SRC = Path(os.environ.get("V2SHIFT_RESULTS", REPO / "RESULTS_V2SHIFT"))


def load(variant: str) -> list[dict]:
    files = sorted(SRC.glob(f"{variant}_seed*.json"), key=lambda p: int(p.stem.split("seed")[1]))
    return [json.load(open(f)) for f in files]


def ms(v: list[float], f: str = "{:.3f}") -> str:
    a = np.asarray(v, float)
    return f.format(a.mean()) + " $\\pm$ " + f.format(a.std(ddof=1))


SESSIONS = ["pre-market", "regular", "after-hours"]
SESSION_TEX = {"pre-market": "Pre-market", "regular": "Regular", "after-hours": "After-hours"}
B_BOOT = 10000
rng = np.random.default_rng(20260919)


def frame(variant: str, seed: int) -> pd.DataFrame:
    """One row per test bar: decision bar time, position (+1/-1/0), next-bar return in bps, position of the reversal-of-last-bar rule."""
    z = np.load(SRC / f"{variant}_seed{seed}_test.npz", allow_pickle=True)  # own files; the time array is stored as strings
    d = pd.DataFrame({"Ticker": z["ticker"], "Timestamp": pd.to_datetime(z["time"]), "y": z["y"].astype(int),
                      "ret_bps": z["r"] * 1e4, "pred": z["P"].argmax(1)})
    d = d.sort_values(["Ticker", "Timestamp"]).reset_index(drop=True)
    d["pos"] = np.select([d["pred"] == 2, d["pred"] == 0], [1, -1], 0)
    g = d.groupby("Ticker")
    d["turn"] = (d["pos"] - g["pos"].shift(1)).abs().fillna(d["pos"].abs())
    d["gross"] = d["pos"] * d["ret_bps"]
    d["prev_ret"] = g["ret_bps"].shift(1)  # return of the bar that ended at the decision time (bar i-1 -> i)
    d["rev_pos"] = -np.sign(d["prev_ret"]).fillna(0).astype(int)
    d["gap_min"] = (g["Timestamp"].shift(-1) - d["Timestamp"]).dt.total_seconds() / 60
    d["session"] = session_labels(d["Timestamp"])
    d["day"] = to_et(d["Timestamp"]).dt.date
    return d


def sign_acc(pos: np.ndarray, r: np.ndarray) -> float:
    m = (pos != 0) & (r != 0)
    return float((np.sign(r[m]) == pos[m]).mean()) if m.any() else float("nan")


def summarise(d: pd.DataFrame, pos_col: str = "pos") -> dict:
    pos = d[pos_col].to_numpy()
    r = d["ret_bps"].to_numpy()
    call = pos != 0
    turn = (d.assign(_p=d[pos_col]).groupby("Ticker")["_p"].diff().abs().fillna(d[pos_col].abs())).to_numpy()
    n = int(call.sum())
    G = float((pos * r)[call].sum())
    T = float(turn[call].sum())
    return {"n": n, "sign": sign_acc(pos, r), "gross": G / n if n else float("nan"), "turn": T / n if n else float("nan"),
            "be": G / T if T > 0 else float("nan"), "net1": (G - T) / n if n else float("nan"), "net2": (G - 2 * T) / n if n else float("nan")}


def day_boot_gross(d: pd.DataFrame) -> tuple[float, float]:
    g = d[d["pos"] != 0].groupby("day")["gross"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(B_BOOT, len(g)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def rule_by_segment() -> None:
    """Out-of-time check of the model-free reversal rule: the same rule on the training, calibration and test segments of the V2 split."""
    import sys
    sys.path.insert(0, str(REPO))
    from common import DATA_DIR
    from data_processor_v2 import DataProcessorV2
    from feature_engineer_v2 import FeatureEngineerV2
    dp = DataProcessorV2()
    stock_dfs, _ = dp.load_and_engineer_all(str(DATA_DIR), FeatureEngineerV2(), threshold_multiplier=0.3, resample_period="30min")
    train, cal, test = dp.split_all_stocks(stock_dfs)
    seg_res = {}
    for seg_name, seg in (("Train", train), ("Cal", cal), ("Test", test)):
        rows = []
        for tk, df in seg.items():
            prev = stock_dfs[tk]["Close"].pct_change(fill_method=None).reindex(df.index)
            rows.append(pd.DataFrame({"Timestamp": df.index, "prev": prev.to_numpy(), "ret": df["fwd_return"].to_numpy() * 1e4}))
        a = pd.concat(rows, ignore_index=True)
        a["session"] = session_labels(pd.DatetimeIndex(a["Timestamp"]))
        a["pos"] = -np.sign(a["prev"]).fillna(0).astype(int)
        for sess in SESSIONS:
            b = a[a["session"] == sess]
            m = (b["pos"] != 0) & (b["ret"] != 0)
            seg_res[(seg_name, sess)] = (float((np.sign(b.loc[m, "ret"]) == b.loc[m, "pos"]).mean()), int(m.sum()))
    rows_tab = []
    for sess in SESSIONS:
        rows_tab.append([SESSION_TEX[sess]] + [f"{100 * seg_res[(sg, sess)][0]:.1f} ({seg_res[(sg, sess)][1]:,}".replace(",", "\\,") + ")" for sg in ("Train", "Cal", "Test")])
    write_table("t16_rule_segments", ["Session of the decision bar", "Training segment", "Calibration segment", "Test segment"], rows_tab, "lrrr",
                "Sign accuracy (\\%) of the model-free rule that takes the opposite of the sign of the last bar's return, on bars with a non-zero return and a non-zero previous return, "
                "on the three segments of the chronological split of the V2 pipeline (30-minute bars; number of bars in parentheses). The rule uses no parameters, so its accuracy on the "
                "training and calibration segments is out-of-time information with respect to the test period.",
                "tab:rule_segments", font=r"\small")
    for sg in ("Train", "Cal", "Test"):
        for sess, tag in (("pre-market", "Pre"), ("regular", "Reg"), ("after-hours", "After")):
            register_number(f"VShiftRule{tag}{sg}", f"{100 * seg_res[(sg, sess)][0]:.1f}")
    print("rule by segment:", seg_res)


def session_analysis(base: list[dict], shift: list[dict]) -> None:
    if not all((SRC / f"{v}_seed{r['seed']}_test.npz").exists() for v in ("base", "shift") for r in base):
        print("no test-set arrays; session analysis skipped")
        return
    seeds = [r["seed"] for r in base]
    fr = {v: [frame(v, k) for k in seeds] for v in ("base", "shift")}
    rows, res = [], {}
    for sess in SESSIONS + ["all"]:
        cells = {}
        for name, v, col in (("base", "base", "pos"), ("shift", "shift", "pos"), ("rev", "base", "rev_pos")):
            vals = []
            for d in (fr[v] if name != "rev" else fr["base"][:1]):
                dd = d if sess == "all" else d[d["session"] == sess]
                vals.append(summarise(dd, col))
            cells[name] = {k: float(np.nanmean([x[k] for x in vals])) for k in vals[0]}
        res[sess] = cells
        rows.append([SESSION_TEX.get(sess, "All bars"),
                     f"{cells['base']['n']:,.0f}".replace(",", "\\,"), f"{100 * cells['base']['sign']:.1f}", f"{cells['base']['gross']:.2f}",
                     f"{cells['shift']['n']:,.0f}".replace(",", "\\,"), f"{100 * cells['shift']['sign']:.1f}", f"{cells['shift']['gross']:.2f}",
                     f"{cells['rev']['n']:,.0f}".replace(",", "\\,"), f"{100 * cells['rev']['sign']:.1f}", f"{cells['rev']['gross']:.2f}"])
    write_table("t15_v2_shift_sessions",
                ["Session of the\ndecision bar", "Base\ncalls", "Base\nsign (\\%)", "Base\ngross", "Shifted\ncalls", "Shifted\nsign (\\%)", "Shifted\ngross", "Reversal\ncalls", "Reversal\nsign (\\%)", "Reversal\ngross"],
                rows, "lrrrrrrrrr",
                "Directional calls of Model~1 by session of the decision bar (30-minute bars, $m=0.3$, mean over the seeds). Calls: number of UP/DOWN calls; "
                "Sign: sign accuracy on bars with a non-zero return; Gross: mean gross P\\&L per call (bps). Base and Shifted: window ending one bar before and at the decision bar; "
                "Reversal: the model-free rule that takes the opposite of the sign of the return of the last bar (calls on all bars with a non-zero previous return). "
                "The session is that of the start time of the decision bar, so the target of the last regular-session bar falls into the after-hours session.",
                "tab:v2_shift_sessions", resize=True, font=r"\footnotesize")
    # numbers for the text
    for sess, tag in (("pre-market", "Pre"), ("regular", "Reg"), ("after-hours", "After")):
        register_number(f"VShift{tag}SignShift", f"{100 * res[sess]['shift']['sign']:.1f}")
        register_number(f"VShift{tag}SignBase", f"{100 * res[sess]['base']['sign']:.1f}")
        register_number(f"VShift{tag}SignRev", f"{100 * res[sess]['rev']['sign']:.1f}")
        register_number(f"VShift{tag}GrossShift", f"{res[sess]['shift']['gross']:.2f}")
        register_number(f"VShift{tag}GrossBase", f"{res[sess]['base']['gross']:.2f}")
        register_number(f"VShift{tag}GrossRev", f"{res[sess]['rev']['gross']:.2f}")
        register_number(f"VShift{tag}NShift", f"{res[sess]['shift']['n']:,.0f}".replace(",", "\\,"))
        register_number(f"VShift{tag}TurnShift", f"{res[sess]['shift']['turn']:.2f}")
        register_number(f"VShift{tag}BeShift", f"{res[sess]['shift']['be']:.1f}")
        register_number(f"VShift{tag}NetOneShift", f"{res[sess]['shift']['net1']:+.2f}")
        register_number(f"VShift{tag}NetTwoShift", f"{res[sess]['shift']['net2']:+.2f}")
    ext_share = 1 - res["regular"]["shift"]["n"] / res["all"]["shift"]["n"]
    register_number("VShiftExtShare", f"{100 * ext_share:.0f}\\%")
    # economics of the shifted window
    for tag, key in (("Gross", "gross"), ("Turn", "turn"), ("Be", "be"), ("NetOne", "net1"), ("NetTwo", "net2")):
        register_number(f"VShiftAll{tag}Shift", f"{res['all']['shift'][key]:.2f}")
        register_number(f"VShiftAll{tag}Base", f"{res['all']['base'][key]:.2f}")
    ci = [day_boot_gross(d) for d in fr["shift"]]
    register_number("VShiftGrossCiLoMin", f"{min(c[0] for c in ci):.2f}")
    register_number("VShiftGrossCiHiMax", f"{max(c[1] for c in ci):.2f}")
    # robustness of the after-hours effect of the shifted window: day-block interval, days and stocks with a positive mean
    ah_ci, ah_days, ah_tk, ah_ndays, ah_p = [], [], [], [], []
    for d in fr["shift"]:
        ah = d[(d["session"] == "after-hours") & (d["pos"] != 0)]
        ah_ci.append(day_boot_gross(ah))
        dd = ah.groupby("day")["gross"].sum()
        ah_days.append(int((dd > 0).sum()))
        ah_ndays.append(len(dd))
        ah_tk.append(int((ah.groupby("Ticker")["gross"].mean() > 0).sum()))
        nz = ah[ah["ret_bps"] != 0]
        ah_p.append(float(binomtest(int((np.sign(nz["ret_bps"]) == nz["pos"]).sum()), len(nz), 0.5).pvalue))
    register_number("VShiftAfterCiLoMin", f"{min(c[0] for c in ah_ci):.1f}")
    register_number("VShiftAfterCiHiMax", f"{max(c[1] for c in ah_ci):.1f}")
    register_number("VShiftAfterDaysMin", str(min(ah_days)))
    register_number("VShiftAfterDaysN", str(max(ah_ndays)))
    register_number("VShiftAfterTickersMin", str(min(ah_tk)))
    register_number("VShiftAfterBinomPMax", f"{max(ah_p):.0e}".replace("e-0", "e-").replace("e-", "\\times 10^{-") + "}")
    print("after-hours CI", ah_ci, "days positive", ah_days, "of", ah_ndays, "tickers positive", ah_tk, "p", ah_p)
    # agreement of the calls with the reversal rule, and the rule's accuracy on the bars that the model called (after-hours)
    agree = {"base": [], "shift": []}
    rule_on_calls, model_on_calls = [], []
    for v in ("base", "shift"):
        for d in fr[v]:
            c = d[(d["session"] == "after-hours") & (d["pos"] != 0) & (d["rev_pos"] != 0)]
            agree[v].append(float((c["pos"] == c["rev_pos"]).mean()))
            if v == "shift":
                rule_on_calls.append(sign_acc(c["rev_pos"].to_numpy(), c["ret_bps"].to_numpy()))
                model_on_calls.append(sign_acc(c["pos"].to_numpy(), c["ret_bps"].to_numpy()))
    register_number("VShiftAfterAgreeShift", f"{100 * np.mean(agree['shift']):.0f}\\%")
    register_number("VShiftAfterAgreeBase", f"{100 * np.mean(agree['base']):.0f}\\%")
    register_number("VShiftAfterRuleOnCalls", f"{100 * np.mean(rule_on_calls):.1f}")
    register_number("VShiftAfterModelOnCalls", f"{100 * np.mean(model_on_calls):.1f}")
    print("agreement with reversal rule (after-hours calls):", {k: np.round(v, 3) for k, v in agree.items()},
          "rule on the model's bars", np.round(rule_on_calls, 4), "model", np.round(model_on_calls, 4))
    rule_by_segment()
    # targets that span a gap (the next bar is more than 30 minutes later)
    gap_share, gross_gap, gross_nogap, sign_nogap, gap_gross_share = [], [], [], [], []
    for d in fr["shift"]:
        c = d[d["pos"] != 0]
        gp = c["gap_min"] > 30
        gap_share.append(gp.mean())
        gross_gap.append(c.loc[gp, "gross"].mean())
        gross_nogap.append(c.loc[~gp, "gross"].mean())
        gap_gross_share.append(c.loc[gp, "gross"].sum() / c["gross"].sum())
        sign_nogap.append(sign_acc(c.loc[~gp, "pos"].to_numpy(), c.loc[~gp, "ret_bps"].to_numpy()))
    register_number("VShiftGapShare", f"{100 * np.mean(gap_share):.1f}\\%")
    register_number("VShiftGapGross", f"{np.mean(gross_gap):.1f}")
    register_number("VShiftNoGapGross", f"{np.mean(gross_nogap):.2f}")
    register_number("VShiftGapGrossShare", f"{100 * np.mean(gap_gross_share):.0f}\\%")
    register_number("VShiftNoGapSign", f"{100 * np.mean(sign_nogap):.1f}")
    # no-gap, per session (shift)
    ng = {}
    for sess in SESSIONS:
        vals = []
        for d in fr["shift"]:
            dd = d[(d["session"] == sess) & (~(d["gap_min"] > 30))]
            vals.append(summarise(dd))
        ng[sess] = {k: float(np.nanmean([x[k] for x in vals])) for k in vals[0]}
    for sess, tag in (("pre-market", "Pre"), ("regular", "Reg"), ("after-hours", "After")):
        register_number(f"VShiftNoGap{tag}Sign", f"{100 * ng[sess]['sign']:.1f}")
        register_number(f"VShiftNoGap{tag}Gross", f"{ng[sess]['gross']:.2f}")
    print("session results:", json.dumps(res, indent=1))
    print("no-gap per session (shift):", json.dumps(ng, indent=1))
    print("gap share", np.mean(gap_share), "gross gap", np.mean(gross_gap), "gross no gap", np.mean(gross_nogap),
          "share of gross from gap", np.mean(gap_gross_share), "sign no gap", np.mean(sign_nogap))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.8))
    x = np.arange(3)
    for j, (name, lab, col) in enumerate((("base", "base window", OI["sky"]), ("shift", "shifted window", OI["orange"]),
                                          ("rev", "reversal of last bar", OI["green"]))):
        ax1.bar(x + (j - 1) * 0.26, [100 * res[s][name]["sign"] for s in SESSIONS], 0.26, color=col, label=lab)
        ax2.bar(x + (j - 1) * 0.26, [res[s][name]["gross"] for s in SESSIONS], 0.26, color=col, label=lab)
    ax1.axhline(50, color="black", lw=0.8, ls="--")
    ax1.set_ylim(45, 66)
    ax1.set_ylabel("Sign accuracy (%)")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Gross P&L per call (bps)")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(["pre-market", "regular", "after-hours"], fontsize=8)
    ax1.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    savefig(fig, "f25_v2_shift_sessions")


def main() -> None:
    base, shift = load("base"), load("shift")
    if not base or not shift:
        raise SystemExit("no V2 window-shift runs found")
    n = min(len(base), len(shift))
    base, shift = base[:n], shift[:n]
    assert [b["seed"] for b in base] == [s["seed"] for s in shift]
    assert all(b["n_test"] == s["n_test"] for b, s in zip(base, shift))

    metrics = [
        ("sign_acc_nonzero", "Sign accuracy, non-zero returns (\\%)", 100, "{:.2f}"),
        ("sign_acc_regular", "Sign accuracy, regular session (\\%)", 100, "{:.2f}"),
        ("auc_direction", "AUC, direction (UP vs DOWN)", 1, "{:.3f}"),
        ("auc_magnitude", "AUC, magnitude (moving vs neutral)", 1, "{:.3f}"),
        ("kappa", "Cohen's $\\kappa$ (3 classes)", 1, "{:.3f}"),
        ("balanced_accuracy", "Balanced accuracy (3 classes)", 100, "{:.2f}"),
        ("gross_bps_per_call", "Gross P\\&L per call (bps)", 1, "{:.2f}"),
    ]
    rows = []
    for key, label, scale, f in metrics:
        b = np.array([r[key] for r in base], float) * scale
        s = np.array([r[key] for r in shift], float) * scale
        d = s - b
        rows.append([label, ms(b, f), ms(s, f), " / ".join(f"{v:+.{3 if scale == 1 else 2}f}" for v in d)])
        tag = {"sign_acc_nonzero": "Sign", "sign_acc_regular": "SignReg", "auc_direction": "AucDir",
               "auc_magnitude": "AucMag", "kappa": "Kappa", "balanced_accuracy": "BalAcc", "gross_bps_per_call": "Gross"}[key]
        register_number(f"VShift{tag}Base", f.format(b.mean()))
        register_number(f"VShift{tag}Shift", f.format(s.mean()))
        register_number(f"VShift{tag}DiffMin", f"{d.min():+.{3 if scale == 1 else 2}f}")
        register_number(f"VShift{tag}DiffMax", f"{d.max():+.{3 if scale == 1 else 2}f}")
        register_number(f"VShift{tag}DiffMean", f"{d.mean():+.{3 if scale == 1 else 2}f}")
        register_number(f"VShift{tag}Wins", str(int((d > 0).sum())))
    nb = np.array([r["calls_nonzero"] for r in base])
    ns = np.array([r["calls_nonzero"] for r in shift])
    rows.append(["Directional calls on non-zero bars", " / ".join(f"{int(v):,}".replace(",", "\\,") for v in nb),
                 " / ".join(f"{int(v):,}".replace(",", "\\,") for v in ns), ""])
    ps = [r["binom_p"] for r in shift]
    pb = [r["binom_p"] for r in base]
    register_number("VShiftN", str(n))
    register_number("VShiftNTest", f"{base[0]['n_test']:,}".replace(",", "\\,"))
    register_number("VShiftBinomPMinShift", f"{min(ps):.3f}")
    register_number("VShiftBinomPMinBase", f"{min(pb):.3f}")
    register_number("VShiftSignShiftMin", f"{100 * min(r['sign_acc_nonzero'] for r in shift):.2f}")
    register_number("VShiftSignShiftMax", f"{100 * max(r['sign_acc_nonzero'] for r in shift):.2f}")
    register_number("VShiftSignBaseMin", f"{100 * min(r['sign_acc_nonzero'] for r in base):.2f}")
    register_number("VShiftSignBaseMax", f"{100 * max(r['sign_acc_nonzero'] for r in base):.2f}")
    register_number("VShiftEpochs", str(base[0]["epochs_max"]))
    register_number("VShiftPatience", str(base[0]["patience"]))
    write_table("t14_v2_window_shift",
                ["Metric (test set, 30-min bars, $m=0.3$)", "Base window ($t-w..t-1$)", "Shifted window ($t-w+1..t$)", "Difference per seed"],
                rows, "lrrr",
                f"Effect of the one-bar window offset of the V2 pipeline on Model~1 (mean $\\pm$ standard deviation over seeds 1--{n}; "
                f"same rows, labels, split, scaler and seeds; at most {base[0]['epochs_max']} epochs, early stopping with patience {base[0]['patience']}). "
                "The last column gives the shifted minus the base value for each seed. Chance level is 50\\% for the sign accuracy and 0.5 for the AUC of the direction.",
                "tab:v2_window_shift", resize=True, font=r"\footnotesize")

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.8))
    panels = [("sign_acc_nonzero", "Sign accuracy (non-zero)", 100, 50.0), ("auc_direction", "AUC, direction", 1, 0.5),
              ("auc_magnitude", "AUC, magnitude", 1, 0.5)]
    for ax, (key, title, scale, ref) in zip(axes, panels):
        for i, (b, s) in enumerate(zip(base, shift)):
            ax.plot([0, 1], [b[key] * scale, s[key] * scale], "-o", color=OI["blue"], ms=4, lw=1, alpha=0.8,
                    label="seed" if i == 0 else None)
        ax.axhline(ref, color="black", lw=0.8, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["base", "shifted"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Test set value")
    fig.tight_layout()
    savefig(fig, "f24_v2_window_shift")
    session_analysis(base, shift)
    flush_numbers()
    print(json.dumps({r[0]: r[1:] for r in rows}, indent=1))


if __name__ == "__main__":
    main()
