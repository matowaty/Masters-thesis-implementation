"""AN-7: what the genetic algorithm actually did (F-09, F-16, F-17, T-11b).

Inputs: ga_population_history.jsonl and ga_history.csv of the eight V2 runs.
Every individual = one full training of Model 1 + Model 2 on the train/calibration split (50 epochs).
Fitness = Sharpe(calibration set) - 0.05 * #features; -999 when Sharpe could not be computed
(no approved trade with non-zero position, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (LABELS, OI, RUN_COLORS, TEXT_WIDTH_IN, fmt_int, flush_numbers, load_pop, plt,
                    register_number, run_dir, savefig, write_table)

rng = np.random.default_rng(7)
GENES = [("window_size", "Window (bars)"), ("hidden_units", "Hidden units"), ("dropout", "Dropout"),
         ("learning_rate", "Learning rate"), ("threshold_multiplier", "Threshold mult. $m$"),
         ("confidence_threshold", "Confidence $\\tau$")]


def main() -> None:
    pops = {lab: load_pop(lab) for lab in LABELS}
    hist = {lab: pd.read_csv(run_dir(lab) / "ga_history.csv") for lab in LABELS}

    # ---------------------------------------------------------------- T-11b summary per run
    rows = []
    inval_share, best_gen, spearman_loss, z_best, z_expected = [], [], [], [], []
    for lab in LABELS:
        p = pops[lab]
        v = p[p["valid"]]
        n, nv = len(p), len(v)
        bi = v["fitness"].idxmax()
        b_gen = int(v.loc[bi, "generation"])
        best_gen.append(b_gen)
        inval_share.append(1 - nv / n)
        rho, pv = stats.spearmanr(v["m1_val_loss"], v["m2_sharpe"])
        spearman_loss.append(rho)
        g0 = v[v["generation"] == 0]["m2_sharpe"]
        gl = v[v["generation"] >= 25]["m2_sharpe"]
        z = (v["m2_sharpe"].max() - v["m2_sharpe"].mean()) / v["m2_sharpe"].std(ddof=1)
        z_best.append(z)
        z_exp = float(stats.norm.ppf((nv - 0.375) / (nv + 0.25)))  # expected maximum of nv iid standard normals (Blom)
        z_expected.append(z_exp)
        rows.append([lab, fmt_int(n), f"{nv} ({100 * nv / n:.0f}\\%)", f"{v['m2_sharpe'].mean():.2f}", f"{v['m2_sharpe'].max():.2f}",
                     f"{v.loc[bi, 'fitness']:.2f}", str(b_gen), f"{z:.1f} ({z_exp:.1f})", f"{rho:+.2f}",
                     f"{g0.mean():.2f}" if len(g0) else "--", f"{gl.mean():.2f}" if len(gl) else "--"])
    write_table("t11b_ga_summary",
                ["Run", "Evaluated", "Valid", "Mean Sharpe (valid)", "Best Sharpe", "Best fitness", "Gen. of best", "$z$ of best (expected under noise)",
                 "$\\rho$(val loss, Sharpe)", "Mean Sharpe gen. 0", "Mean Sharpe gen. $\\geq 25$"],
                rows, "lrrrrrrrrrr",
                "Summary of the genetic-algorithm search per run. ``Valid'' individuals are those for which a Sharpe ratio on the calibration set could be "
                "computed; the others receive a fitness of about $-999$. $z$ is the distance of the best Sharpe ratio from the mean of all valid individuals in units of "
                "their standard deviation; the value in brackets is the expected maximum of the same number of independent standard-normal draws (pure noise). $\\rho$ is Spearman's rank correlation between Model~1 validation loss and the calibration Sharpe ratio among valid individuals.",
                "tab:ga_summary", resize=True, font=r"\footnotesize")
    register_number("InvalidShareMin", f"{100 * min(inval_share):.0f}\\%")
    register_number("InvalidShareMax", f"{100 * max(inval_share):.0f}\\%")
    register_number("BestGenMin", str(min(best_gen)))
    register_number("BestGenMax", str(max(best_gen)))
    register_number("ZExpectedMin", f"{min(z_expected):.1f}")
    register_number("ZExpectedMax", f"{max(z_expected):.1f}")
    register_number("ZBestMin", f"{min(z_best):.1f}")
    register_number("ZBestMax", f"{max(z_best):.1f}")
    excess = np.array(z_best) - np.array(z_expected)
    register_number("ZExcessNearNoiseRuns", str(int((excess < 0.5).sum())))
    register_number("ZExcessOtherMin", f"{excess[excess >= 0.5].min():.1f}")
    register_number("ZExcessOtherMax", f"{excess[excess >= 0.5].max():.1f}")
    print("z excess over noise expectation:", np.round(excess, 2))
    register_number("RhoLossSharpeMin", f"{min(spearman_loss):+.2f}")
    register_number("RhoLossSharpeMax", f"{max(spearman_loss):+.2f}")
    print("invalid share", np.round(inval_share, 2), "best gen", best_gen, "z best", np.round(z_best, 1), "rho", np.round(spearman_loss, 2))

    # ---------------------------------------------------------------- F-09 convergence and cliff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.1))
    for lab in LABELS:
        p = pops[lab]
        v = p[p["valid"]]
        best_so_far = v.groupby("generation")["fitness"].max().cummax()
        ax1.step(best_so_far.index, best_so_far.values, where="post", color=RUN_COLORS[lab], lw=1.1, label=lab)
        share_inv = p.groupby("generation")["valid"].apply(lambda s: 1 - s.mean())
        ax2.plot(share_inv.index, share_inv.rolling(3, min_periods=1).mean().values, color=RUN_COLORS[lab], lw=1.0)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best fitness so far (in-sample)")
    ax1.set_title("(a) Best fitness on the calibration set")
    ax1.legend(ncol=2, fontsize=6.5, loc="lower right")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Share of new individuals with fitness $-999$")
    ax2.set_title("(b) Individuals without a valid Sharpe ratio\n(3-generation moving average)")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    savefig(fig, "f09_ga_convergence")

    # ---------------------------------------------------------------- F-16 gene evolution (pooled over runs)
    allp = pd.concat([p.assign(run=lab) for lab, p in pops.items()])
    allp["gen_bin"] = pd.cut(allp["generation"], bins=[-1, 4, 9, 14, 19, 24, 30], labels=["0-4", "5-9", "10-14", "15-19", "20-24", "25-30"])
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9))
    for ax, gene, title in zip(axes, ["threshold_multiplier", "confidence_threshold"], ["Threshold multiplier $m$", "Confidence threshold $\\tau$"]):
        tab = pd.crosstab(allp["gen_bin"], allp[gene], normalize="index")
        cols = plt.cm.Blues(np.linspace(0.25, 0.95, tab.shape[1]))
        bottom = np.zeros(len(tab))
        for j, c in enumerate(tab.columns):
            ax.bar(range(len(tab)), tab[c].values, bottom=bottom, color=cols[j], edgecolor="white", linewidth=0.4, label=f"{c:g}")
            bottom += tab[c].values
        ax.set_xticks(range(len(tab)))
        ax.set_xticklabels(tab.index)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Share of evaluated individuals")
        ax.set_title(title)
        ax.legend(title=None, ncol=len(tab.columns) if tab.shape[1] <= 4 else 3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    savefig(fig, "f16_gene_evolution")
    share_m03_last = float((allp[allp["generation"] >= 25]["threshold_multiplier"] == 0.3).mean())
    share_m03_first = float((allp[allp["generation"] <= 4]["threshold_multiplier"] == 0.3).mean())
    register_number("MShareLowFirst", f"{100 * share_m03_first:.0f}\\%")
    register_number("MShareLowLast", f"{100 * share_m03_last:.0f}\\%")
    tau_last = allp[allp["generation"] >= 25]["confidence_threshold"].value_counts(normalize=True)
    register_number("TauShareLowLast", f"{100 * tau_last.get(0.6, 0):.0f}\\%")
    print(f"m=0.3 share: first gens {share_m03_first:.2f}, last gens {share_m03_last:.2f}; tau=0.60 last gens {tau_last.get(0.6, 0):.2f}")

    # ---------------------------------------------------------------- F-17 landscape
    v_all = allp[allp["valid"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.0))
    ax = axes[0]
    ax.hist(v_all["m2_sharpe"], bins=50, color="#BBBBBB")
    best_vals = [pops[l].loc[pops[l][pops[l]["valid"]]["fitness"].idxmax(), "m2_sharpe"] for l in LABELS]
    for l, bv in zip(LABELS, best_vals):
        ax.axvline(bv, color=RUN_COLORS[l], lw=1.0)
    ax.set_xlabel("Calibration Sharpe")
    ax.set_ylabel("Individuals")
    ax.set_title("(a) Sharpe distribution\n(lines: run winners)")
    ax = axes[1]
    m_cols = {0.3: OI["blue"], 0.5: OI["green"], 0.7: OI["orange"], 1.0: OI["verm"]}
    for m_, c_ in m_cols.items():
        sub = v_all[v_all["threshold_multiplier"] == m_]
        ax.scatter(sub["m1_val_loss"], sub["m2_sharpe"], s=3, alpha=0.3, color=c_, rasterized=True, label=f"$m$ = {m_:g}")
    rho_all, _ = stats.spearmanr(v_all["m1_val_loss"], v_all["m2_sharpe"])
    ax.set_xlabel("Model 1 validation loss")
    ax.set_ylabel("Calibration Sharpe")
    ax.set_title(f"(b) Validation loss vs Sharpe\n(pooled $\\rho$ = {rho_all:+.2f})")
    leg = ax.legend(fontsize=6.5, markerscale=3, loc="upper left")
    ax = axes[2]
    ax.scatter(v_all["num_selected_features"] + rng.uniform(-0.3, 0.3, len(v_all)), v_all["m2_sharpe"], s=3, alpha=0.25, color=OI["verm"], rasterized=True)
    rho_f, _ = stats.spearmanr(v_all["num_selected_features"], v_all["m2_sharpe"])
    ax.set_xlabel("Number of selected features")
    ax.set_ylabel("Calibration Sharpe")
    ax.set_title(f"(c) Feature count vs Sharpe\n($\\rho$ = {rho_f:+.2f})")
    fig.tight_layout()
    savefig(fig, "f17_ga_landscape")
    register_number("RhoPooledLossSharpe", f"{rho_all:+.2f}")
    register_number("RhoPooledFeaturesSharpe", f"{rho_f:+.2f}")
    rho_within = {}
    for m_ in [0.3, 0.5, 0.7, 1.0]:
        sub = v_all[v_all["threshold_multiplier"] == m_]
        rho_within[m_] = stats.spearmanr(sub["m1_val_loss"], sub["m2_sharpe"])[0]
    register_number("RhoWithinMMin", f"{min(rho_within.values()):+.2f}")
    register_number("RhoWithinMMax", f"{max(rho_within.values()):+.2f}")
    inv_tau = allp.groupby("confidence_threshold")["valid"].apply(lambda x: 1 - x.mean())
    for t_, v_ in inv_tau.items():
        register_number("InvalidTau" + f"{t_:.2f}".replace(".", ""), f"{100 * v_:.0f}\\%")
    print("within-m Spearman", {k: round(v, 3) for k, v in rho_within.items()}, "| invalid share by tau", inv_tau.round(2).to_dict())
    print(f"pooled Spearman: val-loss vs Sharpe {rho_all:+.3f}; #features vs Sharpe {rho_f:+.3f}; valid n={len(v_all)}")

    # ---------------------------------------------------------------- F-17b gene effects on in-sample Sharpe
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_IN, 4.5))
    kw_rows = []
    for ax, (g, title) in zip(axes.ravel(), GENES):
        counts = v_all[g].value_counts()
        levels = sorted(lv for lv in v_all[g].unique() if counts[lv] >= 15)
        means, los, his, ns = [], [], [], []
        # run-demeaned Sharpe: removes the between-run level differences
        y = v_all["m2_sharpe"] - v_all.groupby("run")["m2_sharpe"].transform("mean")
        for lv in levels:
            yy = y[v_all[g] == lv].to_numpy()
            means.append(yy.mean())
            bs = [rng.choice(yy, len(yy)).mean() for _ in range(500)]
            los.append(np.quantile(bs, 0.025))
            his.append(np.quantile(bs, 0.975))
            ns.append(len(yy))
        xs = np.arange(len(levels))
        ax.bar(xs, means, color=OI["blue"], width=0.65)
        ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)], fmt="none", ecolor="black", capsize=2, lw=0.8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lv:g}\n[{ns[i]}]" for i, lv in enumerate(levels)], fontsize=6.5)
        ax.set_title(title, fontsize=8.5)
        groups = [y[v_all[g] == lv].to_numpy() for lv in levels]  # levels with n < 15 are left out
        h, pv = stats.kruskal(*groups)
        ax.text(0.98, 0.02, "KW $p$ " + ("$<$ 0.001" if pv < 0.001 else f"= {pv:.3f}"), transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
        kw_rows.append((g, pv))
    axes[0, 0].set_ylabel("Mean calibration Sharpe\n(run-demeaned)")
    axes[1, 0].set_ylabel("Mean calibration Sharpe\n(run-demeaned)")
    fig.tight_layout()
    savefig(fig, "f17b_gene_effects")
    print("Kruskal-Wallis p by gene:", {g: round(pv, 4) for g, pv in kw_rows})
    for g, pv in kw_rows:
        register_number("KW" + "".join(w.capitalize() for w in g.split("_")), f"{pv:.3f}")
    flush_numbers()


if __name__ == "__main__":
    main()
