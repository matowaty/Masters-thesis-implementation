"""AN-11: is Model 1's information about the *magnitude* of the next move larger than its information about the *direction*?

For each run (test set):
  AUC_mag : score P(NEUTRAL) of Model 1 against the event "actual class = NEUTRAL" (small next move), all bars.
  AUC_dir : score P(UP) - P(DOWN) against the event "actual class = UP" among bars whose actual class is UP or DOWN.
Confidence intervals: bootstrap over trading days (B = 1000), which respects the dependence between stocks and bars.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from common import LABELS, TEXT_WIDTH_IN, flush_numbers, fmt_int, load_trades, plt, register_number, savefig, write_table

B = 1000
rng = np.random.default_rng(20260920)


def auc(score, pos):
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def day_boot(day_idx, score, pos, mask=None):
    """Bootstrap AUC over days. day_idx: array of day ids per row."""
    if mask is not None:
        day_idx, score, pos = day_idx[mask], score[mask], pos[mask]
    days = np.unique(day_idx)
    order = np.argsort(day_idx, kind="stable")
    day_sorted = day_idx[order]
    bounds = np.searchsorted(day_sorted, days, side="left")
    ends = np.searchsorted(day_sorted, days, side="right")
    score, pos = score[order], pos[order]
    groups = [(b, e) for b, e in zip(bounds, ends)]
    out = np.empty(B)
    for k in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([np.arange(*groups[i]) for i in pick])
        out[k] = auc(score[sel], pos[sel])
    return np.nanquantile(out, [0.025, 0.975])


def main() -> None:
    rows = []
    mags, dirs, dir_lo, dir_hi = [], [], [], []
    for lab in LABELS:
        d = load_trades(lab)
        days = d["Day"].astype(str)
        _, day_id = np.unique(days, return_inverse=True)
        p_n = d["M1_Prob_NEUTRAL"].to_numpy()
        y = d["Target_Class"].to_numpy()
        a_mag = auc(p_n, y == 1)
        ci_mag = day_boot(day_id, p_n, y == 1)
        m = y != 1
        s_dir = (d["M1_Prob_UP"] - d["M1_Prob_DOWN"]).to_numpy()
        a_dir = auc(s_dir[m], y[m] == 2)
        ci_dir = day_boot(day_id, s_dir, y == 2, mask=m)
        mags.append(a_mag)
        dirs.append(a_dir)
        dir_lo.append(ci_dir[0])
        dir_hi.append(ci_dir[1])
        rows.append([lab, fmt_int(len(d)), fmt_int(int(m.sum())), f"{a_mag:.3f} [{ci_mag[0]:.3f}, {ci_mag[1]:.3f}]",
                     f"{a_dir:.3f} [{ci_dir[0]:.3f}, {ci_dir[1]:.3f}]"])
        print(lab, f"AUC mag {a_mag:.3f} {ci_mag.round(3)}  AUC dir {a_dir:.3f} {ci_dir.round(3)}")
    write_table("t07d_magnitude_direction",
                ["Run", "Test\nbars", "UP/DOWN\nbars", "AUC magnitude\n[95\\% CI]", "AUC direction\n[95\\% CI]"],
                rows, "lrrrr",
                "Information of Model 1 about the size and about the sign of the next move. Magnitude: area under the ROC curve of the predicted probability of NEUTRAL "
                "for the event that the actual class is NEUTRAL (all test bars). Direction: area under the ROC curve of $P(\\mathrm{UP}) - P(\\mathrm{DOWN})$ for the event "
                "that the actual class is UP, computed on the bars whose actual class is UP or DOWN. An AUC of 0.5 means no information. "
                "Intervals are bootstrap intervals over test days.",
                "tab:magnitude_direction", font=r"\small")
    register_number("AucMagMin", f"{min(mags):.2f}")
    register_number("AucMagMax", f"{max(mags):.2f}")
    register_number("AucDirMin", f"{min(dirs):.3f}")
    register_number("AucDirMax", f"{max(dirs):.3f}")
    register_number("AucDirCiExcludesHalf", str(sum(1 for lo, hi in zip(dir_lo, dir_hi) if lo > 0.5 or hi < 0.5)))
    flush_numbers()


if __name__ == "__main__":
    main()
