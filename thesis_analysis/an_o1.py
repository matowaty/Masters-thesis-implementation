"""AN-14: test O1 -- does the classifier of Model 2 explain the fitness scale of the early GA logs?

Input : RESULTS_O1/{fast,full}_part{0,1}.jsonl written by quick_o1_model2.py (one line per random individual and Model 2 variant;
        20 individuals of a random generation 0, evaluated in the fast mode (20% of the data) and in the full mode).
Output: out/tables/t20_o1_model2.tex   (Section 7.5)
        out/figures/f27_o1_fitness.pdf|png
        numbers: numMtwo*
Environment: O1_RESULTS (default <repo>/RESULTS_O1).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import OI, OUT_TAB, REPO, TEXT_WIDTH_IN, flush_numbers, plt, register_number, savefig, write_table

O1 = Path(os.environ.get("O1_RESULTS", REPO / "RESULTS_O1"))
VARS = ["GB", "HGB_auto", "HGB_off", "HGB_on"]
VLABEL = {"GB": r"\texttt{GradientBoosting}", "HGB_auto": r"\texttt{HistGB}, automatic",
          "HGB_off": r"\texttt{HistGB}, no early stopping", "HGB_on": r"\texttt{HistGB}, early stopping"}
VCAMEL = {"GB": "GB", "HGB_auto": "HgbAuto", "HGB_off": "HgbOff", "HGB_on": "HgbOn"}
MODES = ["fast", "full"]


def neg(s: str) -> str:
    return s.replace("-", "$-$")


def f1(x: float, d: int = 1) -> str:
    return neg(f"{x:.{d}f}")


def main() -> None:
    rows = []
    for p in sorted(O1.glob("*_part*.jsonl")):
        rows += [json.loads(line) for line in open(p)]
    df = pd.DataFrame(rows)
    n_ind = int(df["ind"].nunique())
    assert set(df["mode"]) == set(MODES), set(df["mode"])
    assert (df.groupby(["mode", "variant"]).size() == n_ind).all(), df.groupby(["mode", "variant"]).size()
    print("individuals per cell:", n_ind)
    register_number("MtwoNInd", str(n_ind))

    # the individuals and Model 1 are identical across variants of a mode: check
    for m in MODES:
        sub = df[df["mode"] == m]
        for col in ["n_cal", "n_test", "m1_acc_cal", "m1_acc_test"]:
            assert (sub.groupby("ind")[col].nunique() == 1).all(), (m, col)
    for m in MODES:
        sub = df[(df["mode"] == m) & (df["variant"] == "GB")]
        register_number(f"Mtwo{m.capitalize()}CalMin", f"{int(sub['n_cal'].min()):,}".replace(",", r"\,"))
        register_number(f"Mtwo{m.capitalize()}CalMax", f"{int(sub['n_cal'].max()):,}".replace(",", r"\,"))
        register_number(f"Mtwo{m.capitalize()}TestN", f"{int(sub['n_test'].iloc[0]):,}".replace(",", r"\,"))

    tab_rows = []
    stats = {}
    for m in MODES:
        for v in VARS:
            s = df[(df["mode"] == m) & (df["variant"] == v)].sort_values("ind")
            valid = s["fitness_cal"].notna()
            fc = s.loc[valid, "fitness_cal"]
            ft = s["sharpe_test"].dropna()  # test-set Sharpe ratio (the constant complexity penalty is not applied)
            st = dict(valid=int(valid.sum()), mean=fc.mean(), med=fc.median(), mx=fc.max(),
                      tmed=ft.median(), tmean=ft.mean(), tmax=ft.max(), tn=int(len(ft)),
                      auc_c=s["auc_cal"].median(), auc_t=s["auc_test"].median(), it=s["n_iter"].median(),
                      prec_c=s["prec_app_cal"].mean(), prec_t=s["prec_app_test"].mean())
            stats[(m, v)] = st
            c = f"Mtwo{m.capitalize()}{VCAMEL[v]}"
            register_number(c + "Valid", str(st["valid"]))
            register_number(c + "FitMean", f1(st["mean"]))
            register_number(c + "FitMed", f1(st["med"]))
            register_number(c + "FitMax", f1(st["mx"]))
            register_number(c + "TestMed", f1(st["tmed"]))
            register_number(c + "TestMean", f1(st["tmean"]))
            register_number(c + "TestMax", f1(st["tmax"]))
            register_number(c + "TestValid", str(st["tn"]))
            register_number(c + "AucCal", f"{st['auc_c']:.2f}")
            register_number(c + "AucTest", f"{st['auc_t']:.2f}")
            register_number(c + "Iter", f"{st['it']:.0f}")
            register_number(c + "PrecCal", f"{100 * st['prec_c']:.0f}\\%")
            register_number(c + "PrecTest", f"{100 * st['prec_t']:.0f}\\%")
            print(m, v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in st.items()})
            tab_rows.append([m if v == VARS[0] else "", VLABEL[v], f"{st['it']:.0f}", f"{st['valid']}",
                             f1(st["mean"]), f1(st["mx"]), f"{st['auc_c']:.2f}",
                             f"{st['tn']}", f1(st["tmean"]), f"{st['auc_t']:.2f}"])
        if m != MODES[-1]:
            tab_rows.append("MIDRULE")

    # paired comparisons within an individual (in-sample fitness)
    def paired(m: str, a: str, b: str, key: str = "fitness_cal"):
        x = df[(df["mode"] == m) & (df["variant"] == a)].set_index("ind")[key]
        y = df[(df["mode"] == m) & (df["variant"] == b)].set_index("ind")[key]
        ok = x.notna() & y.notna()
        d = (x[ok] - y[ok])
        p = wilcoxon(d).pvalue if len(d) >= 6 and (d != 0).any() else float("nan")
        return int(ok.sum()), float(d.median()), int((d > 0).sum()), float(p)

    for m in MODES:
        for a, b in [("GB", "HGB_auto"), ("HGB_off", "HGB_on"), ("HGB_auto", "HGB_off"), ("HGB_auto", "HGB_on")]:
            n, med, k, p = paired(m, a, b)
            c = f"Mtwo{m.capitalize()}Pair{VCAMEL[a]}{VCAMEL[b]}"
            register_number(c + "N", str(n))
            register_number(c + "Med", f1(med))
            register_number(c + "Wins", str(k))
            register_number(c + "P", "$<0.001$" if p < 0.001 else f"{p:.3f}")
            print(f"paired {m} {a}-{b}: n={n} median diff {med:.2f} wins {k} p={p:.4g}")

    # identical fits? (HGB_auto == HGB_off in the fast mode since the automatic rule switches early stopping off below 10 000 rows)
    for m in MODES:
        x = df[(df["mode"] == m) & (df["variant"] == "HGB_auto")].set_index("ind")
        y = df[(df["mode"] == m) & (df["variant"] == "HGB_off")].set_index("ind")
        same = np.allclose(x["auc_cal"], y["auc_cal"]) and (x["n_iter"] == y["n_iter"]).all()
        print(f"{m}: HGB_auto identical to HGB_off: {same}")
        register_number(f"Mtwo{m.capitalize()}AutoIsOff", "yes" if same else "no")
        z = df[(df["mode"] == m) & (df["variant"] == "HGB_on")].set_index("ind")
        same_on = np.allclose(x["auc_cal"], z["auc_cal"]) and (x["n_iter"] == z["n_iter"]).all()
        print(f"{m}: HGB_auto identical to HGB_on: {same_on}")
        register_number(f"Mtwo{m.capitalize()}AutoIsOn", "yes" if same_on else "no")
        it = z["n_iter"]
        register_number(f"Mtwo{m.capitalize()}OnIterMin", f"{it.min():.0f}")
        register_number(f"Mtwo{m.capitalize()}OnIterMax", f"{it.max():.0f}")

    tm = [stats[k]["tmean"] for k in stats]
    register_number("MtwoTestMeanMin", f1(min(tm)))
    register_number("MtwoTestMeanMax", f1(max(tm)))

    cap = ("Test O1: fitness of a random generation~0 (\\numMtwoNInd\\ individuals) with four variants of Model~2. The rows of a mode share the individuals "
           "and the trained Model~1; the fast mode uses 20\\% of the data, the full mode all data. ``Trees'' is the median number of boosting iterations kept, "
           "``Valid'' the number of individuals with a valid in-sample fitness (at least ten approved bars, non-zero variance of the P\\&L), "
           "``Mean'' and ``Max'' the mean and maximum of the valid fitness values (Sharpe ratio minus $0.05$ per feature, as in the GA), and AUC the median "
           "area under the ROC curve of Model~2's confidence for the event ``Model~1 was right''. On the test set the annualised Sharpe ratio of the bars approved "
           "with the same threshold is given (no penalty); $n$ is the number of individuals for which it is defined.")
    write_table("t20_o1_model2", ["Mode", "Model 2 (variant)", "Trees", "Valid", "Mean", "Max", "AUC",
                                  "n", "Sharpe", "AUC"], tab_rows, "llrrrrrrrr", cap, "tab:o1_model2",
                note=r"Columns 3--7: calibration set (in-sample fitness, as in the GA); columns 8--10: test set (mean Sharpe ratio).", resize=True,
                font=r"\footnotesize")
    p = OUT_TAB / "t20_o1_model2.tex"
    p.write_text(p.read_text().replace(r"\begin{table}[htbp]", r"\begin{table}[!tb]", 1))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.5), sharey=True)
    rng = np.random.default_rng(3)
    short = {"GB": "GB", "HGB_auto": "HistGB\nauto", "HGB_off": "HistGB\nno stop", "HGB_on": "HistGB\nstop"}
    for ax, m in zip(axes, MODES):
        for k, v in enumerate(VARS):
            s = df[(df["mode"] == m) & (df["variant"] == v)]
            fc, ft = s["fitness_cal"].dropna(), s["sharpe_test"].dropna()
            ax.scatter(k - 0.17 + rng.uniform(-0.08, 0.08, len(fc)), fc, s=14, color=OI["blue"], alpha=0.85, zorder=3,
                       label="calibration set: fitness (in-sample, as in the GA)" if (m == MODES[0] and k == 0) else None)
            ax.scatter(k + 0.17 + rng.uniform(-0.08, 0.08, len(ft)), ft, s=14, color=OI["verm"], marker="D", alpha=0.85, zorder=3,
                       label="test set: Sharpe ratio (out-of-sample)" if (m == MODES[0] and k == 0) else None)
            ax.hlines(fc.mean(), k - 0.32, k - 0.02, color=OI["blue"], lw=1.6, zorder=4)
            ax.hlines(ft.mean(), k + 0.02, k + 0.32, color=OI["verm"], lw=1.6, zorder=4)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(range(len(VARS)))
        ax.set_xticklabels([short[v] for v in VARS], fontsize=7.5)
        nc = df[(df["mode"] == m) & (df["variant"] == "GB")]["n_cal"]
        ax.set_title(("(a) fast mode" if m == "fast" else "(b) full mode") + f"\n{nc.min():,}\u2013{nc.max():,} calibration rows", loc="left")
    axes[0].set_ylabel("fitness / Sharpe ratio")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    savefig(fig, "f27_o1_fitness")
    flush_numbers()


if __name__ == "__main__":
    main()
