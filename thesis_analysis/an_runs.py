"""AN-1, AN-10: run inventory, reported-vs-real metrics, chromosomes, feature stability.

Produces: T-05, T-07, T-08, T-10 (tables), F-10, F-11, F-15 (figures), numbers macros.
"""
from __future__ import annotations

import itertools
import math

from common import *  # noqa: F401,F403

ALL_FEATURES = [
    "Volume", "Past_Return_1_Tick", "Past_Return_2_Tick", "Past_Return_5_Tick", "HL_Spread", "CO_Spread",
    "BB_High", "BB_Mid", "BB_Low", "ATR", "SMA_10", "SMA_20", "EMA_10", "EMA_20", "MACD_Line",
    "MACD_Histogram", "RSI", "VWAP", "OBV", "RSI_Div", "Volume_Surprise", "Bar_Range", "Close_Pos",
    "VWAP_Dist", "Index_Return", "Index_Volatility", "Rel_Return", "Rel_ZScore",
]
N_FEAT = len(ALL_FEATURES)  # 28 (verified from GA population history)


def poisson_binomial_sf(ps: list[float], k: int) -> float:
    """P(X >= k) for X = sum of independent Bernoulli(ps)."""
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum())


def main() -> None:
    reps, chroms, pops = {}, {}, {}
    for lab in LABELS:
        pops[lab] = load_pop(lab)
        chroms[lab] = load_json(lab, "best_chromosome.json")
        reps[lab] = load_json(lab, "metrics.json")

    # ------------------------------------------------------------- T-05 inventory
    rows = []
    for lab in LABELS:
        folder, bar = RUNS[lab]
        pop = pops[lab]
        tr = load_trades(lab)
        date = folder.split("_")[-2]
        rows.append([lab, f"{date[:4]}-{date[4:6]}-{date[6:]}", f"{bar}", str(len(pop)),
                     str(int(pop["generation"].max()) + 1), fmt_int(len(tr)),
                     f"{tr['Timestamp'].min():%Y-%m-%d}", f"{tr['Timestamp'].max():%Y-%m-%d}"])
    write_table("t05_inventory",
                ["Run", "Started", "Bar\n(min)", "Evaluated\nindividuals", "Generations", "Test\nrows",
                 "Test start", "Test end"], rows, "lrrrrrll",
                "Inventory of the eight completed GA runs (population 20, 50 epochs per individual). "
                "R30-0 was run with an earlier version of the code (before the 15-minute option existed) and defaults to 30-minute bars.",
                "tab:inventory", resize=True)
    n_ind = sum(len(p) for p in pops.values())
    register_number("TotalIndividuals", fmt_int(n_ind))
    register_number("IndividualsMin", fmt_int(min(len(p) for p in pops.values())))
    register_number("IndividualsMax", fmt_int(max(len(p) for p in pops.values())))

    # ------------------------------------------------------------- T-07 reported vs real
    rows, stats = [], []
    for lab in LABELS:
        tr = load_trades(lab)
        ap = tr[tr["Approved_Trade"]]
        nz = int((ap["PnL"] != 0).sum())
        neutral_share = float((ap["M1_Pred_Class"] == 1).mean())
        m = reps[lab]
        rows.append([lab, fmt_int(len(tr)), pct(m["trade_rate"]), f"{m['precision_at_conf']:.3f}",
                     f"{m['annualized_sharpe']:.2f}", fmt_int(len(ap)), pct(neutral_share, 1), str(nz)])
        stats.append((len(ap), nz, neutral_share))
    tot_ap = sum(s[0] for s in stats)
    tot_nz = sum(s[1] for s in stats)
    write_table("t07_reported_vs_real",
                ["Run", "Test\nrows", "Trade\nrate", "Precision", "Sharpe", "Approved\nbars", "Approved =\nNEUTRAL",
                 "Non-zero\nP\\&L"], rows, "lrrrrrrr",
                "Metrics reported by the pipeline versus what the approved trades actually were. "
                "``Approved'' are the bars accepted by Model~2; a NEUTRAL prediction carries no position, "
                "so its P\\&L is exactly zero. Reported Sharpe values are therefore not estimates of profitability.",
                "tab:reported_vs_real")
    register_number("ApprovedTotal", fmt_int(tot_ap))
    register_number("ApprovedNonZero", str(tot_nz))

    # ------------------------------------------------------------- T-08 chromosomes
    rows = []
    for lab in LABELS:
        c = chroms[lab]
        rows.append([lab, str(c["window_size"]), str(c["hidden_units"]), f"{c['dropout']:.1f}",
                     f"{c['learning_rate']:g}", f"{c['threshold_multiplier']:.1f}",
                     f"{c['confidence_threshold']:.2f}", str(c["num_selected_features"])])
    write_table("t08_chromosomes",
                ["Run", "Window", "Hidden", "Dropout", "Learning\nrate", "Threshold\nmult. $m$",
                 "Confidence\n$\\tau$", "Features"], rows, "lrrrrrrr",
                "Best chromosome of each GA run (hyper-parameter genes).", "tab:chromosomes")
    # Feature list per run for the appendix
    rows = [[lab, ", ".join(tex_escape(f) for f in chroms[lab]["selected_features"])] for lab in LABELS]
    write_table("t08b_selected_features", ["Run", "Selected features"], rows, "l>{\\raggedright\\arraybackslash}p{11cm}",
                "Features selected by the best chromosome of each GA run.", "tab:selected_features_app",
                font=r"\footnotesize")

    # ------------------------------------------------------------- T-10 feature stability
    sets = {lab: set(chroms[lab]["selected_features"]) for lab in LABELS}
    ks = [len(sets[lab]) for lab in LABELS]
    probs = [k / N_FEAT for k in ks]
    counts = {f: sum(f in sets[lab] for lab in LABELS) for f in ALL_FEATURES}
    order = sorted(ALL_FEATURES, key=lambda f: (-counts[f], f))
    exp_mean = sum(probs)
    exp_sd = math.sqrt(sum(p * (1 - p) for p in probs))
    rows = []
    for f in order:
        pv = poisson_binomial_sf(probs, counts[f]) if counts[f] > 0 else 1.0
        rows.append([tex_escape(f), f"{counts[f]}/8", pct(counts[f] / 8, 0), f"{pv:.3f}"])
    write_table("t10_feature_frequency",
                ["Feature", "Runs", "Rate", "$P(X\\geq x)$ under random selection"], rows, "lrrr",
                "Selection frequency of each of the 28 candidate features in the best chromosomes of the eight GA runs. "
                "The last column is the probability of at least this many selections if each run drew a random subset of the same size "
                "(uncorrected for testing 28 features).", "tab:feature_frequency", font=r"\footnotesize")
    exp_ge7 = sum(poisson_binomial_sf(probs, 7) for _ in ALL_FEATURES)
    exp_ge6 = sum(poisson_binomial_sf(probs, 6) for _ in ALL_FEATURES)
    obs_ge7 = sum(c >= 7 for c in counts.values())
    obs_ge6 = sum(c >= 6 for c in counts.values())
    register_number("FeatExpectedGeSeven", f"{exp_ge7:.1f}")
    register_number("FeatExpectedGeSix", f"{exp_ge6:.1f}")
    register_number("FeatObservedGeSeven", str(obs_ge7))
    register_number("FeatObservedGeSix", str(obs_ge6))
    print(f"features: expected >=7 by chance {exp_ge7:.2f} (observed {obs_ge7}); >=6 {exp_ge6:.2f} (observed {obs_ge6})")

    # pairwise Jaccard vs random baseline
    def jac(a, b):
        return len(a & b) / len(a | b)

    pairs = list(itertools.combinations(LABELS, 2))
    obs_j = np.mean([jac(sets[a], sets[b]) for a, b in pairs])
    rng = np.random.default_rng(0)
    rand_j = []
    for _ in range(4000):
        rs = {lab: set(rng.choice(N_FEAT, size=ks[i], replace=False)) for i, lab in enumerate(LABELS)}
        rand_j.append(np.mean([jac(rs[a], rs[b]) for a, b in pairs]))
    rand_j = np.array(rand_j)
    p_j = float((rand_j >= obs_j).mean())
    register_number("JaccardObs", f"{obs_j:.2f}")
    register_number("JaccardRandom", f"{rand_j.mean():.2f}")
    register_number("JaccardRandomHi", f"{np.quantile(rand_j, 0.975):.2f}")
    register_number("JaccardP", f"{p_j:.3f}")
    print(f"jaccard obs {obs_j:.3f} random {rand_j.mean():.3f} (97.5% {np.quantile(rand_j, .975):.3f}) p={p_j:.4f}")

    # F-15
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ypos = np.arange(N_FEAT)[::-1]
    per_feat_mean = np.mean(probs) * 8
    per_feat_sd = math.sqrt(sum(p * (1 - p) for p in probs))
    lo, hi = per_feat_mean - 2 * per_feat_sd, per_feat_mean + 2 * per_feat_sd
    ax1.axvspan(max(lo, 0), min(hi, 8), color=OI["sky"], alpha=0.18, label="random-selection range ($\\pm 2$ sd)")
    ax1.barh(ypos, [counts[f] for f in order], color=OI["blue"], height=0.7)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([f.replace("_", "\\_") if False else f for f in order], fontsize=7)
    ax1.set_xlim(0, 8)
    ax1.set_xlabel("Number of runs (of 8) in which the feature was selected")
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax1.set_title("(a) Feature selection frequency")
    mat = np.array([[jac(sets[a], sets[b]) for b in LABELS] for a in LABELS])
    im = ax2.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax2.set_xticks(range(len(LABELS)))
    ax2.set_xticklabels(LABELS, rotation=60, ha="right", fontsize=7)
    ax2.set_yticks(range(len(LABELS)))
    ax2.set_yticklabels(LABELS, fontsize=7)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax2.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if mat[i, j] > 0.6 else "black")
    ax2.grid(False)
    ax2.set_title(f"(b) Jaccard similarity of feature sets\n(mean {obs_j:.2f}; random {rand_j.mean():.2f})")
    fig.tight_layout()
    savefig(fig, "f15_feature_stability")

    # ------------------------------------------------------------- F-10 in-sample vs out-of-sample
    ins, oos = [], []
    for lab in LABELS:
        pop = pops[lab]
        best = pop.loc[pop["fitness"].idxmax()]
        ins.append(float(best["m2_sharpe"]))
        oos.append(float(reps[lab]["annualized_sharpe"]))
    register_number("InSampleSharpeMin", f"{min(ins):.1f}")
    register_number("InSampleSharpeMax", f"{max(ins):.1f}")
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    ax.bar(x - 0.19, ins, 0.38, color=OI["blue"], label="GA best individual: in-sample Sharpe (calibration set)")
    ax.bar(x + 0.19, oos, 0.38, color=OI["orange"], label="Reported Sharpe on the test set")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Annualised Sharpe ratio")
    for i, o in enumerate(oos):
        if abs(o) < 1e-12:
            ax.text(i + 0.19, 0.12, "0", ha="center", fontsize=7, color=OI["orange"])
    ax.set_ylim(-1.0, max(ins) * 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Fitness seen by the GA versus out-of-sample outcome")
    fig.tight_layout()
    savefig(fig, "f10_sharpe_gap")

    # ------------------------------------------------------------- F-11 composition of approved trades
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    bottom = np.zeros(len(LABELS))
    comp = []
    for lab in LABELS:
        ap = load_trades(lab).query("Approved_Trade")
        comp.append([float((ap["M1_Pred_Class"] == c).mean()) for c in range(3)])
    comp = np.array(comp)
    for c in range(3):
        ax.bar(x, comp[:, c], bottom=bottom, color=CLASS_COLORS[c], label=f"Model 1 predicts {CLASS_NAMES[c]}", width=0.6)
        bottom += comp[:, c]
    for i, lab in enumerate(LABELS):
        ax.text(i, 1.02, f"{stats[i][1]}", ha="center", fontsize=8)
    ax.text(len(LABELS) - 0.4, 1.10, "non-zero P&L trades", ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Share of approved trades")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.34))
    ax.set_title("What Model 2 approves: almost exclusively NEUTRAL predictions")
    fig.tight_layout()
    savefig(fig, "f11_approved_composition")
    flush_numbers()


if __name__ == "__main__":
    main()
