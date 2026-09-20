"""AN-2, AN-6, AN-8: classification skill of the V2 pipeline (Model 1) and calibration of Model 2.

Produces T-07b (skill metrics), T-07c (Model 2 discrimination), F-12, F-13, F-14, F-22 and number macros.

Convention note: the original A100 analysis reports "sign accuracy" as the hit rate among
directional (UP/DOWN) predictions in which bars with an exactly zero forward return count as wrong.
Roughly 5% of the bars have zero return (stale prices in illiquid hours), which pushes that figure
about 1.2 percentage points below the value computed on non-zero bars. Both are reported.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import (CLASS_COLORS, CLASS_NAMES, LABELS, LABELS_30, OI, RUN_COLORS, TEXT_WIDTH_IN,
                    load_trades, plt, register_number, flush_numbers, savefig, write_table, pct)


def cohen_kappa(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    cm = np.zeros((k, k))
    np.add.at(cm, (a, b), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def macro_f1_and_bal_acc(y: np.ndarray, p: np.ndarray, k: int = 3):
    f1s, recs = [], []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        recs.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)), float(np.mean(recs))


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    logs = {lab: load_trades(lab) for lab in LABELS}
    rows, rows_m2 = [], []
    conf_total = np.zeros((3, 3))
    conf_by_run = {}
    pred_share, act_share = {}, {}
    zero_shares, sign_excl, sign_a100, sign_p = [], [], [], []

    for lab in LABELS:
        d = logs[lab]
        y = d["Target_Class"].to_numpy()
        p = d["M1_Pred_Class"].to_numpy()
        r = d["Actual_Fwd_Return"].to_numpy()
        n = len(d)
        cm = np.zeros((3, 3))
        np.add.at(cm, (y, p), 1)
        conf_by_run[lab] = cm
        conf_total += cm
        act = np.bincount(y, minlength=3) / n
        prd = np.bincount(p, minlength=3) / n
        act_share[lab], pred_share[lab] = act, prd
        acc = float((y == p).mean())
        maj = float(act.max())
        f1, bal = macro_f1_and_bal_acc(y, p)
        kap = cohen_kappa(y, p)
        # sign accuracy among directional predictions
        dirm = p != 1
        nz = r != 0
        hit = ((p == 2) & (r > 0)) | ((p == 0) & (r < 0))
        n_dir = int(dirm.sum())
        n_dir_nz = int((dirm & nz).sum())
        sa_excl = float(hit[dirm & nz].mean())
        sa_a100 = float(hit[dirm].mean())
        k_hits = int(hit[dirm & nz].sum())
        pval = float(stats.binomtest(k_hits, n_dir_nz, 0.5).pvalue)
        up_share_nz = float((r[nz] > 0).mean())
        # majority-sign baseline (always predict the more frequent sign)
        maj_sign = max(up_share_nz, 1 - up_share_nz)
        zero_shares.append(float((~nz).mean()))
        sign_excl.append(sa_excl)
        sign_a100.append(sa_a100)
        sign_p.append(pval)
        rows.append([lab, f"{n:,}".replace(",", r"\,"),
                     "/".join(f"{100 * a:.0f}" for a in act), "/".join(f"{100 * a:.0f}" for a in prd),
                     f"{acc:.3f}", f"{maj:.3f}", f"{bal:.3f}", f"{f1:.3f}", f"{kap:.3f}",
                     f"{100 * sa_excl:.2f}\\%", f"{pval:.2f}", f"{100 * sa_a100:.2f}\\%"])
        # Model 2 discrimination: does the score predict "Model 1 was right"?
        correct = (y == p)
        s = d["M2_Conf_Score"].to_numpy()
        neu = p == 1
        rows_m2.append([lab,
                        f"{auc(s, correct):.3f}",
                        f"{auc(s[neu], correct[neu]):.3f}",
                        f"{auc(s[~neu], correct[~neu]):.3f}",
                        f"{s[neu].mean():.2f}", f"{s[~neu].mean():.2f}",
                        f"{correct[neu].mean():.2f}", f"{correct[~neu].mean():.2f}"])

    write_table("t07b_v2_skill",
                ["Run", "Test rows", "Actual D/N/U (\\%)", "Predicted D/N/U (\\%)", "Acc.", "Majority", "Bal. acc.",
                 "Macro-F1", "$\\kappa$", "Sign acc.", "$p$", "Sign acc. (A100)"],
                rows, "lrrrrrrrrrrr",
                "Model 1 (three-class) on the test set. ``Majority'' is the accuracy of always predicting the most frequent class. "
                "Sign accuracy is the hit rate among UP/DOWN predictions on bars with non-zero forward return; $p$ is a two-sided binomial test "
                "against 0.5 (optimistic, because bars of one day and stocks are not independent). The last column counts zero-return bars as misses, "
                "as the original run summaries do.",
                "tab:v2_skill", resize=True, font=r"\footnotesize")
    write_table("t07c_m2_discrimination",
                ["Run", "AUC all", "AUC within NEUTRAL", "AUC within UP/DOWN", "Mean score NEUTRAL", "Mean score UP/DOWN",
                 "M1 correct | NEUTRAL", "M1 correct | UP/DOWN"],
                rows_m2, "lrrrrrrr",
                "Model 2 as a classifier of ``Model 1 was correct''. The overall AUC is largely driven by the class of the Model 1 prediction (NEUTRAL versus UP/DOWN); "
                "within a predicted class the score discriminates only modestly (AUC 0.52--0.60), and for UP/DOWN calls barely above chance.",
                "tab:m2_discrimination", resize=True, font=r"\footnotesize")

    register_number("SignAccExclMin", f"{100 * min(sign_excl):.1f}\\%")
    register_number("SignAccExclMax", f"{100 * max(sign_excl):.1f}\\%")
    register_number("SignAccAHundredMin", f"{100 * min(sign_a100):.1f}\\%")
    register_number("SignAccAHundredMax", f"{100 * max(sign_a100):.1f}\\%")
    register_number("ZeroReturnShareMin", f"{100 * min(zero_shares):.1f}\\%")
    register_number("ZeroReturnShareMax", f"{100 * max(zero_shares):.1f}\\%")
    n_sig = sum(pv < 0.05 for pv in sign_p)
    register_number("SignSignificantRuns", str(n_sig))
    print(f"sign acc excl zeros {min(sign_excl):.4f}-{max(sign_excl):.4f}; A100 conv {min(sign_a100):.4f}-{max(sign_a100):.4f}; "
          f"zero share {min(zero_shares):.3f}-{max(zero_shares):.3f}; p<0.05 in {n_sig}/8 runs; p={['%.3f' % v for v in sign_p]}")

    # ------------------------------------------------------------ predicted class shares
    m2s = pd.DataFrame(rows_m2, columns=["run", "auc", "auc_n", "auc_d", "sn", "sd", "cn", "cd"])
    print(m2s.to_string(index=False))
    pool = pd.concat([logs[l] for l in LABELS])
    neu_mean = pool.loc[pool.M1_Pred_Class == 1, "M2_Conf_Score"].mean()
    dir_mean = pool.loc[pool.M1_Pred_Class != 1, "M2_Conf_Score"].mean()
    register_number("MTwoMeanNeutral", f"{neu_mean:.2f}")
    register_number("MTwoMeanDirectional", f"{dir_mean:.2f}")
    a_n = [float(x) for x in m2s["auc_n"]]
    a_d = [float(x) for x in m2s["auc_d"]]
    register_number("AucWithinNeutralMax", f"{max(a_n):.2f}")
    register_number("AucWithinDirectionalMax", f"{max(a_d):.2f}")

    # ------------------------------------------------------------ F-14 class shares
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.1))
    w = 0.38
    for off, share, hatch, name in [(-w / 2 - 0.01, act_share, "", "actual"), (w / 2 + 0.01, pred_share, "///", "predicted")]:
        bottom = np.zeros(len(LABELS))
        for c in range(3):
            v = np.array([share[l][c] for l in LABELS])
            ax.bar(x + off, v, w, bottom=bottom, color=CLASS_COLORS[c], hatch=hatch, edgecolor="white", linewidth=0.4,
                   label=f"{CLASS_NAMES[c]}" if name == "actual" else None)
            bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Share of test bars")
    ax.set_title("Target classes (solid, left) versus Model 1 predictions (hatched, right)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    savefig(fig, "f14_class_shares")
    ps = np.array([pred_share[l] for l in LABELS])
    ac = np.array([act_share[l] for l in LABELS])
    register_number("PredDownMin", f"{100 * ps[:, 0].min():.0f}\\%")
    register_number("PredDownMax", f"{100 * ps[:, 0].max():.0f}\\%")
    register_number("PredUpMin", f"{100 * ps[:, 2].min():.0f}\\%")
    register_number("PredUpMax", f"{100 * ps[:, 2].max():.0f}\\%")
    register_number("PredNeutralMin", f"{100 * ps[:, 1].min():.0f}\\%")
    register_number("PredNeutralMax", f"{100 * ps[:, 1].max():.0f}\\%")
    register_number("ActNeutralMin", f"{100 * ac[:, 1].min():.0f}\\%")
    register_number("ActNeutralMax", f"{100 * ac[:, 1].max():.0f}\\%")
    print("pred share D/N/U min", ps.min(0).round(3), "max", ps.max(0).round(3), "actual min", ac.min(0).round(3), "max", ac.max(0).round(3))

    # ------------------------------------------------------------ F-13 confusion matrices
    def draw_cm(ax, cm, title):
        nm = cm / cm.sum(1, keepdims=True)
        ax.imshow(nm, vmin=0, vmax=0.8, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * nm[i, j]:.0f}", ha="center", va="center",
                        color="white" if nm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["D", "N", "U"])
        ax.set_yticklabels(["D", "N", "U"])
        ax.set_title(title, fontsize=8.5)
        ax.grid(False)

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    draw_cm(ax, conf_total, "Pooled over eight runs\n(row-normalised, %)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    savefig(fig, "f13_confusion_pooled")
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH_IN, 3.7))
    for ax, lab in zip(axes.ravel(), LABELS):
        draw_cm(ax, conf_by_run[lab], lab)
    for ax in axes[1]:
        ax.set_xlabel("Predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual")
    fig.tight_layout()
    savefig(fig, "f13b_confusion_grid")

    # ------------------------------------------------------------ F-12 Model 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.2))
    bins = np.linspace(0.0, 1.0, 41)
    for c, name in [(1, "M1 predicts NEUTRAL"), (0, "M1 predicts DOWN"), (2, "M1 predicts UP")]:
        v = pool.loc[pool.M1_Pred_Class == c, "M2_Conf_Score"]
        ax1.hist(v, bins=bins, weights=np.ones(len(v)) / len(v), histtype="stepfilled" if c == 1 else "step",
                 color=CLASS_COLORS[c], alpha=0.45 if c == 1 else 1.0, label=name, linewidth=1.3)
    ax1.axvspan(0.60, 0.65, color=OI["yellow"], alpha=0.35, label="GA-selected $\\tau$ (0.60-0.65)")
    ax1.set_xlim(0.1, 0.9)
    ax1.set_ylim(0, 0.40)
    ax1.set_xlabel("Model 2 confidence score")
    ax1.set_ylabel("Share of bars in the group")
    ax1.set_title("(a) Score by Model 1 prediction")
    ax1.legend(loc="upper right", fontsize=7)
    edges = np.linspace(0.2, 0.8, 13)
    for grp, sel, col, name in [("neu", pool.M1_Pred_Class == 1, OI["black"], "M1 predicts NEUTRAL"),
                                ("dir", pool.M1_Pred_Class != 1, OI["verm"], "M1 predicts UP/DOWN")]:
        sub = pool[sel]
        b = np.digitize(sub["M2_Conf_Score"], edges)
        xs, ys, ns = [], [], []
        for k in range(1, len(edges)):
            m = b == k
            if m.sum() >= 100:
                xs.append(sub.loc[m, "M2_Conf_Score"].mean())
                ys.append((sub.loc[m, "Target_Class"] == sub.loc[m, "M1_Pred_Class"]).mean())
                ns.append(m.sum())
        ax2.plot(xs, ys, "o-", color=col, ms=3.5, lw=1.2, label=name)
    ax2.plot([0.2, 0.8], [0.2, 0.8], "--", color="#888888", lw=0.8, label="perfect calibration")
    ax2.set_xlabel("Model 2 confidence score")
    ax2.set_ylabel("Observed share where Model 1 was correct")
    ax2.set_title("(b) Reliability (bins with $\\geq$100 bars)")
    ax2.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    savefig(fig, "f12_model2_confidence")

    # ------------------------------------------------------------ F-22 agreement between runs
    keyed = {}
    for lab in LABELS_30:
        d = logs[lab][["Timestamp", "Ticker", "M1_Pred_Class", "Target_Class"]].copy()
        keyed[lab] = d.set_index(["Timestamp", "Ticker"])
    kap = pd.DataFrame(np.nan, index=LABELS_30, columns=LABELS_30)
    for a, b in itertools.combinations(LABELS_30, 2):
        j = keyed[a].join(keyed[b], how="inner", lsuffix="_a", rsuffix="_b")
        k = cohen_kappa(j["M1_Pred_Class_a"].to_numpy(), j["M1_Pred_Class_b"].to_numpy())
        kap.loc[a, b] = kap.loc[b, a] = k
    for a in LABELS_30:
        kap.loc[a, a] = 1.0
    off = kap.values[np.triu_indices(len(LABELS_30), 1)]
    kap_true = {lab: cohen_kappa(logs[lab]["Target_Class"].to_numpy(), logs[lab]["M1_Pred_Class"].to_numpy()) for lab in LABELS}
    register_number("KappaBetweenMean", f"{off.mean():.2f}")
    register_number("KappaBetweenMin", f"{off.min():.2f}")
    register_number("KappaBetweenMax", f"{off.max():.2f}")
    register_number("KappaTruthMin", f"{min(kap_true.values()):.3f}")
    register_number("KappaTruthMax", f"{max(kap_true.values()):.3f}")
    print(f"kappa between runs mean {off.mean():.3f} [{off.min():.3f}, {off.max():.3f}]; vs truth {min(kap_true.values()):.3f}..{max(kap_true.values()):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    m = kap.values
    ax1.imshow(m, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(LABELS_30)):
        for j in range(len(LABELS_30)):
            ax1.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if m[i, j] > 0.6 else "black")
    ax1.set_xticks(range(len(LABELS_30)))
    ax1.set_xticklabels(LABELS_30, rotation=60, ha="right")
    ax1.set_yticks(range(len(LABELS_30)))
    ax1.set_yticklabels(LABELS_30)
    ax1.grid(False)
    ax1.set_title("(a) Agreement between runs' predictions\n(Cohen's $\\kappa$, 30-min runs)")
    ax2.bar(range(len(LABELS)), [kap_true[l] for l in LABELS], color=[RUN_COLORS[l] for l in LABELS])
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xticks(range(len(LABELS)))
    ax2.set_xticklabels(LABELS, rotation=60, ha="right")
    ax2.set_ylabel("$\\kappa$ (prediction vs target)")
    ax2.set_title("(b) Agreement with the realised class")
    fig.tight_layout()
    savefig(fig, "f22_prediction_agreement")
    flush_numbers()


if __name__ == "__main__":
    main()
