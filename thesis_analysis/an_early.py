"""AN-14: the early runs of the second pipeline (16-18 June 2026) and the fitness scale of the genetic-algorithm logs.

Inputs (read-only): RESULTS/<folder>/ of the early runs (metrics.json, config.json, ga_history.csv,
ga_population_history.jsonl, test_trade_log.csv of E-6) and RESULTS/<run>/ga_history.csv of the eight full runs.
Outputs: out/tables/t19_early_runs.tex, out/figures/f26_early_fitness.pdf|png, numbers (numEarly*).

Runs (see thesis_analysis/inputs/run_manifest_all.csv):
  E-1 baseline_v2 (fixed configuration, laptop)      E-2 BiLSTM, E-3 BiLSTM+attention (same configuration, A100)
  E-4 short GA (A100, 16 June)                        E-5 first full GA run, stopped after 10 generations (A100, 16 June)
  E-6 short GA (laptop, 18 June; the only early run with a test trade log)

Environment: EARLY_RESULTS (default <repo>/RESULTS).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from an_magnitude import auc, day_boot
from common import (A100, LABELS, OI, REPO, RUN_COLORS, TEXT_WIDTH_IN, flush_numbers, load_pop, plt, register_number,
                    savefig, tex_escape, to_et, write_table, RUNS)

EARLY = Path(os.environ.get("EARLY_RESULTS", REPO / "RESULTS"))
FOLD = {
    "E-1": "baseline_v2_multi_all_stocks_20260616_102645",
    "E-2": "attention_bilstm_v2_multi_all_stocks_20260616_163501",
    "E-3": "attention_bilstm_attn_v2_multi_all_stocks_20260616_163730",
    "E-4": "ga_fast_v2_multi_all_stocks_20260616_170234",
    "E-5": "ga_full_v2_multi_all_stocks_20260616_171052",
    "E-6": "ga_fast_v2_multi_all_stocks_20260618_110715",
}
DATE = {"E-1": "16 Jun", "E-2": "16 Jun", "E-3": "16 Jun", "E-4": "16 Jun", "E-5": "16 Jun", "E-6": "18 Jun"}


def jload(eid: str, name: str) -> dict:
    return json.load(open(EARLY / FOLD[eid] / name))


def describe(eid: str) -> str:
    c = jload(eid, "config.json")
    if c.get("mode") == "fast":
        return (f"short GA (population {c['population_size']}, {c['num_generations']} generations, "
                f"{c['ga_epochs']} epochs, {int(round(c['data_fraction'] * 100))}\\% of the data)")
    arch = {"bilstm_v2": "BiLSTM", "bilstm_attn_v2": "BiLSTM with attention"}.get(c.get("model"), "BiLSTM")
    return f"{arch}, fixed configuration"


def neg(s: str) -> str:
    """Typeset a minus sign in text mode."""
    return s.replace("-", "$-$")


def main() -> None:
    # ------------------------------------------------------------------ Table: reported metrics
    fixed_keys = ["window_size", "batch_size", "epochs", "learning_rate", "hidden_size", "num_layers", "dropout", "scaling"]
    cfgs = [{k: jload(e, "config.json")[k] for k in fixed_keys} for e in ("E-1", "E-2", "E-3")]
    assert cfgs[0] == cfgs[1] == cfgs[2], cfgs
    print("fixed configuration of E-1..E-3:", cfgs[0])
    rows = []
    reported = {}
    for eid in ["E-1", "E-2", "E-3", "E-4", "E-6"]:
        m = jload(eid, "metrics.json")
        reported[eid] = m
        rows.append([eid, describe(eid), DATE[eid], f"{100 * m['trade_rate']:.1f}\\%", f"{m['precision_at_conf']:.3f}",
                     neg(f"{m['annualized_sharpe']:.2f}")])
    write_table(
        "t19_early_runs", ["ID", "Configuration", "Date (2026)", "Trade rate", "Precision", "Sharpe"], rows, r"@{}l>{\raggedright\arraybackslash}Xlrrr@{}",
        r"Test-set metrics printed by the pipeline at the end of the early runs of the second pipeline (E-1 to E-4 and E-6, see Appendix~\ref{app:inventory}). "
        r"Trade rate is the share of test bars approved by Model~2, precision is the share of the approved bars on which Model~1 was right, and the Sharpe ratio is annualised. "
        r"E-1 to E-3 have the same hyperparameters (window 12, 64 units, two layers, dropout 0.2, learning rate $10^{-3}$, 50 epochs); E-1 has no entry for the architecture (presumably the default, the BiLSTM of E-2). "
        r"The runs were single runs without fixed random seeds; only for E-6 was the log of the test predictions stored. Run E-5 has no test evaluation (Figure~\ref{fig:early_fitness}).",
        "tab:early_runs", resize=False, font=r"\small", rules=True, full_width=True)
    prec = [reported[e]["precision_at_conf"] for e in reported]
    sh = [reported[e]["annualized_sharpe"] for e in reported]
    tr = [reported[e]["trade_rate"] for e in reported]
    register_number("EarlyPrecMin", f"{min(prec):.2f}")
    register_number("EarlyPrecMax", f"{max(prec):.2f}")
    register_number("EarlyTradeMin", f"{100 * min(tr):.1f}\\%")
    register_number("EarlyTradeMax", f"{100 * max(tr):.1f}\\%")
    register_number("EarlySharpeMax", f"{max(sh):.2f}")
    register_number("EarlySharpeMin", neg(f"{min(sh):.2f}"))
    e1 = reported["E-1"]
    register_number("EarlyBasePrec", f"{e1['precision_at_conf']:.3f}")
    register_number("EarlyBaseTrade", f"{100 * e1['trade_rate']:.1f}\\%")
    register_number("EarlyBaseSharpe", f"{e1['annualized_sharpe']:.2f}")
    register_number("EarlyBaseDiff", f"{abs(reported['E-2']['annualized_sharpe'] - e1['annualized_sharpe']):.1f}")
    a, b = reported["E-2"], reported["E-3"]
    register_number("EarlyArchPrecA", f"{a['precision_at_conf']:.3f}")
    register_number("EarlyArchPrecB", f"{b['precision_at_conf']:.3f}")
    register_number("EarlyArchTradeA", f"{100 * a['trade_rate']:.1f}\\%")
    register_number("EarlyArchTradeB", f"{100 * b['trade_rate']:.1f}\\%")
    register_number("EarlyArchSharpeA", neg(f"{a['annualized_sharpe']:.2f}"))
    register_number("EarlyArchSharpeB", neg(f"{b['annualized_sharpe']:.2f}"))
    register_number("EarlyArchSharpeDiff", f"{abs(b['annualized_sharpe'] - a['annualized_sharpe']):.1f}")

    # ------------------------------------------------------------------ E-6: what the approved trades were
    d = pd.read_csv(EARLY / FOLD["E-6"] / "test_trade_log.csv", parse_dates=["Timestamp"])
    d["Day"] = to_et(d["Timestamp"]).dt.date.astype(str)
    _, day_id = np.unique(d["Day"], return_inverse=True)
    y = d["Target_Class"].to_numpy()
    pred = d["M1_Pred_Class"].to_numpy()
    r = d["Actual_Fwd_Return"].to_numpy()
    n = len(d)
    ud = (pred != 1) & (r != 0)
    hit = (np.sign(r[ud]) == np.where(pred[ud] == 2, 1, -1))
    n_calls, k_hit = int(ud.sum()), int(hit.sum())
    p_bin = stats.binomtest(k_hit, n_calls, 0.5).pvalue
    p_n = d["M1_Prob_NEUTRAL"].to_numpy()
    a_mag = auc(p_n, y == 1)
    mag_ci = day_boot(day_id, p_n, y == 1)
    s_dir = (d["M1_Prob_UP"] - d["M1_Prob_DOWN"]).to_numpy()
    m = y != 1
    a_dir = auc(s_dir[m], y[m] == 2)
    dir_ci = day_boot(day_id[m], s_dir[m], y[m] == 2)
    ap = d["Approved_Trade"].to_numpy().astype(bool)
    prec_check = float((pred[ap] == y[ap]).mean())
    assert abs(prec_check - reported["E-6"]["precision_at_conf"]) < 1e-9, prec_check
    print(f"E-6: n={n} sign acc {k_hit}/{n_calls} = {k_hit / n_calls:.4f} p={p_bin:.3f}; AUC dir {a_dir:.3f} {dir_ci}; AUC mag {a_mag:.3f} {mag_ci}; "
          f"approved {ap.sum()} neutral {(pred[ap] == 1).mean():.3f} nonzero pnl {(d['PnL'][ap] != 0).sum()} prec {prec_check:.4f}")
    register_number("EarlyLogPrec", f"{prec_check:.3f}")
    register_number("EarlyLogRows", f"{n:,}".replace(",", r"\,"))
    register_number("EarlyLogCalls", f"{n_calls:,}".replace(",", r"\,"))
    register_number("EarlyLogSignAcc", f"{100 * k_hit / n_calls:.1f}\\%")
    register_number("EarlyLogSignP", f"{p_bin:.2f}")
    register_number("EarlyLogAucDir", f"{a_dir:.3f}")
    register_number("EarlyLogAucDirLo", f"{dir_ci[0]:.3f}")
    register_number("EarlyLogAucDirHi", f"{dir_ci[1]:.3f}")
    register_number("EarlyLogAucMag", f"{a_mag:.3f}")
    register_number("EarlyLogAucMagLo", f"{mag_ci[0]:.3f}")
    register_number("EarlyLogAucMagHi", f"{mag_ci[1]:.3f}")
    register_number("EarlyLogApproved", str(int(ap.sum())))
    register_number("EarlyLogApprovedNeutral", f"{100 * (pred[ap] == 1).mean():.0f}\\%")
    register_number("EarlyLogNonzero", str(int((d['PnL'][ap] != 0).sum())))

    # ------------------------------------------------------------------ fitness scale of the GA logs
    def hist(eid):
        return pd.read_csv(EARLY / FOLD[eid] / "ga_history.csv")

    h5, h4, h6 = hist("E-5"), hist("E-4"), hist("E-6")
    pop6 = pd.DataFrame([json.loads(l) for l in open(EARLY / FOLD["E-6"] / "ga_population_history.jsonl")])
    pop6["valid"] = pop6["m2_sharpe"] > -900
    # mean fitness of the valid individuals per generation
    mean5 = h5["avg_fitness"].to_numpy()  # E-5: minimum fitness > 0 in every generation, hence all individuals valid
    assert (h5["min_fitness"] > 0).all()
    mean4 = h4["avg_fitness"].to_numpy()
    assert (h4["min_fitness"] > 0).all()
    mean6 = pop6[pop6["valid"]].groupby("generation")["fitness"].mean().to_numpy()
    max5, max4, max6 = h5["max_fitness"].to_numpy(), h4["max_fitness"].to_numpy(), h6["max_fitness"].to_numpy()
    inv6 = int((~pop6["valid"]).sum())

    comp = {}
    for lab in LABELS:
        hh = pd.read_csv(A100 / RUNS[lab][0] / "ga_history.csv")
        pop = load_pop(lab)
        mv = pop[pop["valid"]].groupby("generation")["fitness"].mean()
        g10 = pop[pop["generation"] <= 9]
        comp[lab] = dict(max=hh["max_fitness"].to_numpy(), mean=mv, inv10=float((~g10["valid"]).mean()),
                         best=float(hh["max_fitness"].max()), g0max=float(hh["max_fitness"].iloc[0]), g0mean=float(mv.iloc[0]))

    def rng(vals, f="{:.1f}"):
        return f.format(min(vals)) + "--" + f.format(max(vals))

    register_number("EarlyFirstGens", str(len(h5)))
    register_number("EarlyFirstMaxFit", f"{h5['max_fitness'].max():.1f}")
    register_number("EarlyFirstMaxG0", f"{max5[0]:.1f}")
    register_number("EarlyFirstMeanG0", f"{mean5[0]:.1f}")
    register_number("EarlyFirstMeanLast", f"{mean5[-1]:.1f}")
    register_number("EarlyFirstMinFit", f"{h5['min_fitness'].min():.2f}")
    register_number("EarlyFastMaxFit", f"{max(max4.max(), max6.max()):.1f}")
    register_number("EarlyFastMaxFitA", f"{max4.max():.1f}")
    register_number("EarlyFastMaxFitB", f"{max6.max():.1f}")
    register_number("EarlyFastMeanLastA", f"{mean4[-1]:.1f}")
    register_number("EarlyFastMeanLastB", f"{mean6[-1]:.1f}")
    register_number("EarlyFastInvalidB", str(inv6))
    register_number("EarlyFullBestMin", f"{min(c['best'] for c in comp.values()):.1f}")
    register_number("EarlyFullBestMax", f"{max(c['best'] for c in comp.values()):.1f}")
    register_number("EarlyFullG0MaxMin", f"{min(c['g0max'] for c in comp.values()):.1f}")
    register_number("EarlyFullG0MaxMax", f"{max(c['g0max'] for c in comp.values()):.1f}")
    register_number("EarlyFullG0MeanMin", neg(f"{min(c['g0mean'] for c in comp.values()):.1f}"))
    register_number("EarlyFullG0MeanMax", f"{max(c['g0mean'] for c in comp.values()):.1f}")
    register_number("EarlyFullInvalidMin", f"{100 * min(c['inv10'] for c in comp.values()):.0f}\\%")
    register_number("EarlyFullInvalidMax", f"{100 * max(c['inv10'] for c in comp.values()):.0f}\\%")
    print("E-5 max", max5.round(1), "mean", mean5.round(1))
    print("E-4 max", max4.round(1), "mean", mean4.round(1))
    print("E-6 max", max6.round(1), "mean", mean6.round(1), "invalid", inv6)
    for lab, c in comp.items():
        print(lab, "best", round(c["best"], 2), "g0 max", round(c["g0max"], 2), "g0 mean valid", round(c["g0mean"], 2),
              "invalid share gens 0-9", round(c["inv10"], 2))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0), sharex=True)
    for ax, key, title in [(axes[0], "max", "(a) best fitness in the generation"),
                           (axes[1], "mean", "(b) mean fitness of the valid individuals")]:
        for lab in LABELS:
            v = comp[lab][key]
            v = v.to_numpy() if hasattr(v, "to_numpy") else v
            ax.plot(np.arange(len(v)), v, color="#B0B0B0", lw=0.9, zorder=1,
                    label="eight complete runs" if lab == LABELS[0] else None)
        ser = {"max": [("E-5, first full run (stopped)", max5, OI["verm"]), ("E-4, short GA (16 Jun)", max4, OI["blue"]),
                       ("E-6, short GA (18 Jun)", max6, OI["green"])],
               "mean": [("E-5, first full run (stopped)", mean5, OI["verm"]), ("E-4, short GA (16 Jun)", mean4, OI["blue"]),
                        ("E-6, short GA (18 Jun)", mean6, OI["green"])]}[key]
        for name, v, col in ser:
            ax.plot(np.arange(len(v)), v, color=col, lw=1.6, marker="o", ms=3, zorder=3, label=name)
        ax.set_title(title, loc="left")
        ax.set_xlabel("generation")
        ax.set_xlim(-0.5, 30.5)
    axes[0].set_ylabel("fitness (Sharpe minus penalty)")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    savefig(fig, "f26_early_fitness")
    flush_numbers()


if __name__ == "__main__":
    main()
