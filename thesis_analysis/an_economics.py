"""AN-4, AN-5: counterfactual economics of Model 1's directional calls (no Model 2 gate).

Because the approved trades of the pipeline are (almost) all NEUTRAL predictions with zero P&L,
we ask a different question: would the UP/DOWN calls of Model 1 have made money on their own?

Accounting (all in basis points, one bar holding period):
  position  pos_t in {+1 (UP), -1 (DOWN), 0 (NEUTRAL)} per ticker
  gross_t   = pos_t * r_t * 1e4,  r_t = forward return of the next bar
  turnover  = |pos_t - pos_{t-1}| per ticker (one-way units traded; a position kept for two bars is not paid twice)
  net_t(c)  = gross_t - c * turnover_t,   c = one-way cost in bps (assumption; grid 0/1/2/5)
Portfolio: equal weight over all tickers present at a timestamp; daily P&L = sum over the bars of the day;
Sharpe = mean/sd of daily P&L * sqrt(252) (independent of any bars-per-day convention).
Uncertainty: bootstrap over test *days* (10,000 draws; handles cross-sectional and intraday dependence),
plus a timing-free permutation null (the call vectors of a bar are swapped with the same time-of-day slot on another day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (LABELS, OI, RUN_COLORS, TEXT_WIDTH_IN, fmt_int, flush_numbers, load_trades, plt,
                    register_number, savefig, session_labels, to_et, write_table)

B_BOOT = 10_000
B_PERM = 3_000
COSTS = [0, 1, 2, 5]
rng = np.random.default_rng(20260919)


def prep(lab: str) -> pd.DataFrame:
    d = load_trades(lab).sort_values(["Ticker", "Timestamp"]).reset_index(drop=True)
    d["pos"] = np.select([d["M1_Pred_Class"] == 2, d["M1_Pred_Class"] == 0], [1, -1], 0)
    d["ret_bps"] = d["Actual_Fwd_Return"] * 1e4
    d["gross"] = d["pos"] * d["ret_bps"]
    prev = d.groupby("Ticker")["pos"].shift(1)
    d["turn"] = (d["pos"] - prev).abs().fillna(d["pos"].abs())
    d["is_trade"] = (d["pos"] != 0).astype(int)
    d["Session"] = session_labels(d["Timestamp"])
    d["Hour"] = to_et(d["Timestamp"]).dt.hour
    return d


def daily_portfolio(d: pd.DataFrame) -> pd.DataFrame:
    ts = d.groupby("Timestamp")[["gross", "turn"]].mean()
    ts["Day"] = to_et(ts.index).date
    return ts.groupby("Day")[["gross", "turn"]].sum()


def sharpe_daily(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def analyse(lab: str) -> dict:
    d = prep(lab)
    n_tr = int(d["is_trade"].sum())
    G = float(d["gross"].sum())
    T = float(d["turn"].sum())
    out = {"lab": lab, "n_trades": n_tr, "gross_bps": G / n_tr, "turn_per_trade": T / n_tr,
           "breakeven": G / T if T > 0 else float("nan")}
    # per-day sums for the ratio bootstrap
    g_day = d.groupby("Day")[["gross", "turn", "is_trade"]].sum()
    days = g_day.index.to_numpy()
    nd = len(days)
    idx = rng.integers(0, nd, size=(B_BOOT, nd))
    Gs, Ts, Ns = g_day["gross"].to_numpy(), g_day["turn"].to_numpy(), g_day["is_trade"].to_numpy()
    mean_b = Gs[idx].sum(1) / Ns[idx].sum(1)
    be_b = Gs[idx].sum(1) / Ts[idx].sum(1)
    out["gross_ci"] = tuple(np.quantile(mean_b, [0.025, 0.975]))
    out["breakeven_ci"] = tuple(np.quantile(be_b, [0.025, 0.975]))
    out["p_boot_le0"] = float((mean_b <= 0).mean())
    out["n_days"] = nd
    # daily portfolio series
    port = daily_portfolio(d)
    out["port"] = port
    for c in COSTS:
        net = port["gross"].to_numpy() - c * port["turn"].to_numpy()
        out[f"sharpe_{c}"] = sharpe_daily(net)
        out[f"net_trade_{c}"] = (G - c * T) / n_tr
    # bootstrap CI of gross daily Sharpe
    gp = port["gross"].to_numpy()
    ib = rng.integers(0, len(gp), size=(B_BOOT, len(gp)))
    sb = gp[ib].mean(1) / gp[ib].std(1, ddof=1) * np.sqrt(252)
    out["sharpe_gross_ci"] = tuple(np.quantile(sb, [0.025, 0.975]))
    # permutation null (timing-free): the whole cross-section of positions of one bar is reassigned to the same
    # time-of-day slot of another day. This keeps exposure, time-of-day seasonality and the cross-sectional
    # correlation of the calls, and breaks only the day-to-day timing of the predictions.
    P = d.pivot_table(index="Timestamp", columns="Ticker", values="pos", aggfunc="first").fillna(0).to_numpy()
    R = d.pivot_table(index="Timestamp", columns="Ticker", values="ret_bps", aggfunc="first").fillna(0).to_numpy()
    ts_index = d.pivot_table(index="Timestamp", columns="Ticker", values="pos", aggfunc="first").index
    slot = ts_index.strftime("%H:%M").to_numpy()
    slot_groups = [np.where(slot == sl)[0] for sl in np.unique(slot)]
    slot_groups = [g for g in slot_groups if len(g) > 1]
    null = np.empty(B_PERM)
    for b in range(B_PERM):
        Pp = P.copy()
        for g in slot_groups:
            Pp[g] = P[rng.permutation(g)]
        null[b] = float((Pp * R).sum()) / n_tr
    out["exposure"] = float(d["pos"].mean())
    out["perm_p"] = float(((null >= out["gross_bps"]).sum() + 1) / (B_PERM + 1))
    out["null"] = null
    out["frame"] = d
    return out


def main() -> None:
    res = {lab: analyse(lab) for lab in LABELS}

    # ------------------------------------------------------------ T-09 (gross) and T-09c (net of costs)
    rows, rows_net = [], []
    for lab in LABELS:
        r = res[lab]
        rows.append([lab, fmt_int(r["n_trades"]),
                     f"{r['gross_bps']:.2f} [{r['gross_ci'][0]:.2f}, {r['gross_ci'][1]:.2f}]",
                     f"{r['null'].mean():.2f}", f"{r['perm_p']:.3f}", f"{r['breakeven']:.2f}"])
        rows_net.append([lab, f"{r['turn_per_trade']:.2f}", f"{r['net_trade_1']:.2f}", f"{r['net_trade_2']:.2f}", f"{r['net_trade_5']:.2f}",
                         f"{r['sharpe_0']:.1f}", f"{r['sharpe_1']:.1f}", f"{r['sharpe_2']:.1f}"])
    write_table("t09_economics",
                ["Run", "UP/DOWN\ncalls", "Gross, bps/call\n[95\\% CI]", "Null\nmean", "$p_{\\mathrm{perm}}$", "Break-even\n$c$ (bps)"],
                rows, "lrrrrr",
                "Counterfactual gross P\\&L of Model 1's UP/DOWN calls, ignoring Model 2 (one bar holding period, basis points per call). "
                "The CI is a bootstrap over test days. ``Null mean'' is the average gross P\\&L when the same call vectors are randomly reassigned "
                "across days within the same time-of-day slot (whole cross-sections at once); $p_{\\mathrm{perm}}$ is the one-sided "
                "probability that such a reassignment earns at least as much as the actual predictions. The break-even $c$ is the one-way cost per unit of "
                "position change at which the mean net P\\&L is zero.",
                "tab:economics", font=r"\small")
    write_table("t09c_net",
                ["Run", "Turnover\nper call", "Net\n1 bp", "Net\n2 bp", "Net\n5 bp", "Sharpe\ngross", "Sharpe\n1 bp", "Sharpe\n2 bp"],
                rows_net, "lrrrrrrr",
                "Mean net P\\&L per call (bps) after a one-way cost of 1, 2 or 5 bps per unit of position change (an illustrative assumption), "
                "and annualised Sharpe ratios of the daily P\\&L of an equal-weight portfolio over the 19 stocks (daily P\\&L $\\times\\sqrt{252}$) on about "
                f"{res['R15']['n_days']} test days. Turnover per call is the average absolute position change per UP/DOWN call.",
                "tab:economics_net", font=r"\small")

    gross = np.array([res[l]["gross_bps"] for l in LABELS])
    be = np.array([res[l]["breakeven"] for l in LABELS])
    register_number("GrossBpsMin", f"{gross.min():.2f}")
    register_number("GrossBpsMax", f"{gross.max():.2f}")
    register_number("BreakevenMin", f"{be.min():.2f}")
    register_number("BreakevenMax", f"{be.max():.2f}")
    register_number("BreakevenCiHigh", f"{max(res[l]['breakeven_ci'][1] for l in LABELS):.2f}")
    register_number("NTestDays", str(res["R15"]["n_days"]))
    n_pos = int((gross > 0).sum())
    register_number("RunsPositiveGross", str(n_pos))
    perm_sig = sum(res[l]["perm_p"] < 0.05 for l in LABELS)
    register_number("PermSignificantRuns", str(perm_sig))
    ps_sorted = sorted((res[l]["perm_p"], l) for l in LABELS)
    holm_ok = 0
    for k, (pv, _) in enumerate(ps_sorted):
        if pv * (len(LABELS) - k) < 0.05:
            holm_ok += 1
        else:
            break
    register_number("PermHolmSignificantRuns", str(holm_ok))
    print("null means:", [round(float(res[l]["null"].mean()), 3) for l in LABELS], "| null sd:", [round(float(res[l]["null"].std()), 3) for l in LABELS],
          "| exposure (mean pos):", [round(res[l]["exposure"], 3) for l in LABELS], "| Holm-significant:", holm_ok)
    ci_excl0 = sum(res[l]["gross_ci"][0] > 0 for l in LABELS)
    register_number("BootCiExcludesZeroRuns", str(ci_excl0))
    register_number("SharpeGrossMin", f"{min(res[l]['sharpe_0'] for l in LABELS):.1f}")
    register_number("SharpeGrossMax", f"{max(res[l]['sharpe_0'] for l in LABELS):.1f}")
    register_number("SharpeNetOneMax", f"{max(res[l]['sharpe_1'] for l in LABELS):.1f}")
    register_number("NetOneMax", f"{max(res[l]['net_trade_1'] for l in LABELS):.2f}")
    register_number("NetTwoMax", f"{max(res[l]['net_trade_2'] for l in LABELS):.2f}")
    register_number("TurnPerTradeMin", f"{min(res[l]['turn_per_trade'] for l in LABELS):.2f}")
    register_number("TurnPerTradeMax", f"{max(res[l]['turn_per_trade'] for l in LABELS):.2f}")
    print("gross bps/call", gross.round(3), "breakeven", be.round(3))
    print("boot CI excl. 0:", ci_excl0, "perm p<.05:", perm_sig, "perm p:", [round(res[l]['perm_p'], 3) for l in LABELS])
    print("daily sharpe gross:", [round(res[l]['sharpe_0'], 2) for l in LABELS])
    print("daily sharpe net 1bp:", [round(res[l]['sharpe_1'], 2) for l in LABELS])
    print("gross sharpe CI:", [tuple(round(v, 1) for v in res[l]['sharpe_gross_ci']) for l in LABELS])
    print("turn/call", [round(res[l]['turn_per_trade'], 2) for l in LABELS])

    # buy-and-hold drift context: equal-weight average forward return of all bars in the test period
    drift = [res[l]["frame"]["ret_bps"].mean() for l in LABELS]
    register_number("DriftBpsPerBar", f"{np.mean(drift):.2f}")

    # weighting check for R30-5 (the only run whose equal-weight portfolio gains at 2 bps): the portfolio divides by the
    # number of stocks present at each timestamp, so thinly populated timestamps weigh more than 1/19
    d5 = prep("R30-5")
    port5 = daily_portfolio(d5)
    register_number("RThirtyFiveNetTwoPort", f"{(port5['gross'] - 2 * port5['turn']).sum():.0f}")
    register_number("RThirtyFiveNetTwoUniform", f"{(d5['gross'] - 2 * d5['turn']).sum() / 19:.0f}")
    register_number("RThirtyFiveMinStocks", str(int(d5.groupby("Timestamp").size().min())))
    register_number("RThirtyFiveMeanStocks", f"{d5.groupby('Timestamp').size().mean():.1f}")
    print("mean fwd return per bar (bps):", np.round(drift, 2))

    # ------------------------------------------------------------ F-18 equity curves
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.0), sharey=False)
    for ax, c, title in zip(axes, [0, 1, 2], ["(a) gross", "(b) net, 1 bp one-way", "(c) net, 2 bp one-way"]):
        for lab in LABELS:
            p = res[lab]["port"]
            net = p["gross"] - c * p["turn"]
            ax.plot(pd.to_datetime(net.index), net.cumsum().to_numpy(), color=RUN_COLORS[lab], lw=1.0,
                    label=lab if c == 0 else None)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("Cumulative P\\&L (bps of equal-weight portfolio)".replace("\\&", "&"))
    axes[0].legend(ncol=2, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    savefig(fig, "f18_equity_curves")

    # ------------------------------------------------------------ F-19 break-even
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.8))
    x = np.arange(len(LABELS))
    lo = np.array([res[l]["breakeven"] - res[l]["breakeven_ci"][0] for l in LABELS])
    hi = np.array([res[l]["breakeven_ci"][1] - res[l]["breakeven"] for l in LABELS])
    ax.axhspan(1, 5, color=OI["yellow"], alpha=0.35, label="illustrative cost range 1-5 bps (assumption)")
    ax.bar(x, be, color=[RUN_COLORS[l] for l in LABELS], width=0.6)
    ax.errorbar(x, be, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], fmt="none", ecolor="black", capsize=2.5, lw=0.9)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Break-even one-way cost (bps)")
    ax.set_title("Cost that would wipe out the gross edge of the UP/DOWN calls (bars: estimate, whiskers: 95% CI)")
    ax.title.set_fontsize(8.5)
    ax.legend(loc="upper right")
    ax.set_ylim(min(-0.5, (be - lo).min() * 1.1), 5.5)
    fig.tight_layout()
    savefig(fig, "f19_breakeven")

    # ------------------------------------------------------------ F-20 forest + permutation null
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.1), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(LABELS))[::-1]
    for yi, lab in zip(y, LABELS):
        r = res[lab]
        q = np.quantile(r["null"], [0.025, 0.975])
        ax1.plot(q, [yi, yi], color="#BBBBBB", lw=6, solid_capstyle="butt", zorder=1)
        ax1.plot(r["gross_ci"], [yi, yi], color=RUN_COLORS[lab], lw=1.6, zorder=2)
        ax1.plot([r["gross_bps"]], [yi], "o", color=RUN_COLORS[lab], ms=4.5, zorder=3)
    ax1.axvline(0, color="black", lw=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(LABELS)
    ax1.set_xlabel("Gross P\\&L per UP/DOWN call (bps)".replace("\\&", "&"))
    ax1.set_title("(a) Estimate, 95% day-bootstrap CI (colour)\nand permutation-null 95% range (grey)")
    ax1.title.set_fontsize(8.5)
    best = max(LABELS, key=lambda l: res[l]["gross_bps"])
    r = res[best]
    ax2.hist(r["null"], bins=40, color="#BBBBBB")
    ax2.axvline(r["gross_bps"], color=RUN_COLORS[best], lw=1.6, label=f"observed ({best}): {r['gross_bps']:.2f}")
    ax2.set_xlabel("Gross P&L per call under day-swap permutation (bps)")
    ax2.set_ylabel("Permutations")
    ax2.set_title(f"(b) Permutation null, run with the highest\nedge ({best}; one-sided p = {r['perm_p']:.3f})")
    ax2.title.set_fontsize(8.5)
    ax2.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    savefig(fig, "f20_uncertainty")

    # ------------------------------------------------------------ F-21 per ticker and hour (pooled over runs)
    pool = pd.concat([res[l]["frame"] for l in LABELS])
    days = np.array(sorted(pool["Day"].unique()))
    day_idx = {d: i for i, d in enumerate(days)}
    pool["di"] = pool["Day"].map(day_idx)

    def group_edge(keycol: str):
        keys = sorted(pool[keycol].unique())
        kidx = {k: i for i, k in enumerate(keys)}
        Gm = np.zeros((len(days), len(keys)))
        Nm = np.zeros_like(Gm)
        np.add.at(Gm, (pool["di"].to_numpy(), pool[keycol].map(kidx).to_numpy()), pool["gross"].to_numpy())
        np.add.at(Nm, (pool["di"].to_numpy(), pool[keycol].map(kidx).to_numpy()), pool["is_trade"].to_numpy())
        est = Gm.sum(0) / np.maximum(Nm.sum(0), 1)
        ib = rng.integers(0, len(days), size=(2000, len(days)))
        bs = np.stack([Gm[i].sum(0) / np.maximum(Nm[i].sum(0), 1) for i in ib])
        return keys, est, np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0), Nm.sum(0)

    sess_color = {"pre-market": OI["sky"], "regular": OI["blue"], "after-hours": OI["orange"]}

    def hour_color(h: int) -> str:
        if h == 9:
            return "#999999"
        if 4 <= h <= 8:
            return sess_color["pre-market"]
        if 10 <= h <= 15:
            return sess_color["regular"]
        return sess_color["after-hours"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 5.4))
    for ax, key, title in [(ax1, "Ticker", "(a) By stock"),
                           (ax2, "Hour", "(b) By hour of day (US Eastern time; the 09:00 hour mixes pre-market and regular session)")]:
        keys, est, lo_, hi_, n_ = group_edge(key)
        if key == "Ticker":
            order = np.argsort(est)[::-1]
            keys = [keys[i] for i in order]
            est, lo_, hi_, n_ = est[order], lo_[order], hi_[order], n_[order]
        xs = np.arange(len(keys))
        cols = [OI["blue"]] * len(keys) if key == "Ticker" else [hour_color(int(k)) for k in keys]
        ax.bar(xs, est, color=cols, width=0.65)
        ax.errorbar(xs, est, yerr=[est - lo_, hi_ - est], fmt="none", ecolor="black", capsize=1.8, lw=0.7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(keys, rotation=60 if key == "Ticker" else 0, ha="right" if key == "Ticker" else "center", fontsize=7)
        ax.set_ylabel("Gross bps per call (pooled over 8 runs)")
        ax.set_title(title, fontsize=8.5)
        n_pos = int((lo_ > 0).sum())
        n_neg = int((hi_ < 0).sum())
        if key == "Ticker":
            register_number("TickersCiAboveZero", str(n_pos))
            register_number("TickersCiBelowZero", str(n_neg))
            print("tickers with CI above 0:", n_pos, "below 0:", n_neg, "of", len(keys))
        else:
            print("hours with CI above 0:", n_pos, "below 0:", n_neg, "of", len(keys), "| hours", [int(k) for k in keys])
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=sess_color[k], label=k) for k in sess_color] + [Patch(color="#999999", label="mixed")],
               loc="upper left", ncol=4, fontsize=7)
    fig.tight_layout()
    savefig(fig, "f21_edge_by_group")

    # ------------------------------------------------------------ T-09b by trading session (pooled over runs)
    keys, est, lo_, hi_, n_ = group_edge("Session")
    rows = []
    for k, e, l_, h_, n in zip(keys, est, lo_, hi_, n_):
        sub = pool[pool["Session"] == k]
        rows.append([k, fmt_int(len(sub) / len(LABELS)), f"{100 * (sub['ret_bps'] == 0).mean():.1f}\\%",
                     f"{sub['ret_bps'].abs().mean():.1f}", f"{100 * n / pool['is_trade'].sum():.0f}\\%",
                     f"{e:.2f} [{l_:.2f}, {h_:.2f}]"])
        register_number("Edge" + k.replace("-", "").capitalize(), f"{e:.2f}")
        register_number("Edge" + k.replace("-", "").capitalize() + "Lo", f"{l_:.2f}")
        register_number("Edge" + k.replace("-", "").capitalize() + "Hi", f"{h_:.2f}")
        register_number("ZeroShare" + k.replace("-", "").capitalize(), f"{100 * (sub['ret_bps'] == 0).mean():.0f}\\%")
        register_number("CallShare" + k.replace("-", "").capitalize(), f"{100 * n / pool['is_trade'].sum():.0f}\\%")
        print(f"session {k}: rows/run {len(sub) / len(LABELS):.0f}, zero-return {100 * (sub['ret_bps'] == 0).mean():.1f}%, "
              f"mean |r| {sub['ret_bps'].abs().mean():.1f} bps, call share {100 * n / pool['is_trade'].sum():.0f}%, edge {e:.2f} [{l_:.2f}, {h_:.2f}]")
    write_table("t09b_sessions",
                ["Session", "Test bars\nper run", "Zero-return\nbars", "Mean $|r|$\n(bps)", "Share of\ncalls",
                 "Gross bps/call\n[95\\% CI]"],
                rows, "l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r",
                "Gross P\\&L of Model 1's UP/DOWN calls by trading session, pooled over the eight runs (regular session = 09:30-16:00 US Eastern time). "
                "The confidence interval is a bootstrap over test days on the pooled calls.",
                "tab:sessions", font=r"\footnotesize")
    flush_numbers()


if __name__ == "__main__":
    main()
