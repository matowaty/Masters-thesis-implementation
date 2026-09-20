"""AN-3, AN-9: the data, the labels and the baselines.

T-01 data inventory, T-02 split sizes, T-12 reference points (random / majority / persistence / volatility timing),
F-03 split timeline, F-04 bars per hour (sessions, zero returns, missing bars), F-05 stylised facts, F-06 label construction.

Uses the project's own DataProcessorV2 for resampling, labelling and splitting so that the numbers match the experiments exactly.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import (CLASS_COLORS, CLASS_NAMES, DATA_DIR, LABELS_30, OI, REPO, TEXT_WIDTH_IN, et_minutes, fmt_int, flush_numbers,
                    load_trades, plt, register_number, savefig, to_et, write_table)

sys.path.insert(0, str(REPO))
from data_processor_v2 import DataProcessorV2  # noqa: E402
from feature_engineer_v2 import FeatureEngineerV2  # noqa: E402

M_GRID = [0.3, 0.5, 0.7, 1.0]


def session_minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    """Regular US session 09:30-16:00 US Eastern time (DST-aware conversion of the UTC timestamps)."""
    m = et_minutes(idx)
    return (m >= 570) & (m < 960)


def load_raw() -> dict[str, pd.DataFrame]:
    dp = DataProcessorV2()
    out = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        out[f.stem] = dp.load_data(str(f))
    return out


def main() -> None:
    raw = load_raw()
    dp = DataProcessorV2()
    tickers = list(raw)
    print("tickers:", len(tickers))

    # ------------------------------------------------------------ T-01 data inventory
    rows = []
    empty_rows_by_ticker = {}
    pooled_zero, pooled_bars, all_ret5 = 0, 0, []
    for t, df in raw.items():
        r = df["Close"].pct_change(fill_method=None)
        et_day = np.asarray(to_et(df.index).date)
        days = len(set(et_day))
        per_day = df.groupby(et_day).size()
        reg = session_minutes(df.index)
        zero = float((r == 0).mean())
        # share of missing 5-min bars against 16 hours (192 bars) on days that are present
        miss = 1 - len(df) / (days * 192)
        empty = float(df[["Open", "High", "Low", "Close"]].isna().all(axis=1).mean())
        empty_rows_by_ticker[t] = (int(df[["Open", "High", "Low", "Close"]].isna().all(axis=1).sum()), len(df))
        rows.append([t.replace("_", "\\_"), fmt_int(len(df)), str(df.index.min().date()), str(days),
                     f"{per_day.median():.0f}", f"{1e4 * r.std():.1f}", f"{df['Volume'].median():,.0f}".replace(",", r"\,"),
                     f"{100 * zero:.1f}\\%", f"{100 * reg.mean():.0f}\\%", f"{100 * miss:.1f}\\%", f"{100 * empty:.1f}\\%"])
        pooled_zero += int((r == 0).sum())
        pooled_bars += int(r.notna().sum())
    write_table("t01_data_inventory",
                ["Ticker", "5-min rows", "First bar", "Days", "Median rows/day", "Std of 5-min return (bps)",
                 "Median volume", "Zero-return bars", "In regular session", "Absent vs 192/day", "Empty rows"],
                rows, "lrlrrrrrrrr",
                "The 19 stocks used in the experiments. Every series ends on 2026-01-15 and holds exactly 25\\,920 rows of 5-minute data covering pre-market, regular session and after-hours "
                "(up to 192 rows per day); ``regular session'' is 09:30--16:00 US Eastern time. ``Absent'' is the share of the 192 daily slots for which the file has no row at all; "
                "``empty rows'' have a timestamp but no price (no trade in the interval). A bar is counted as zero-return if its close equals the previous close "
                "(computed on rows with a price).",
                "tab:data_inventory", resize=True, font=r"\footnotesize")
    n_bars_total = sum(len(d) for d in raw.values())
    e_all = sum(v[0] for v in empty_rows_by_ticker.values())
    e_oth = sum(v[0] for k, v in empty_rows_by_ticker.items() if k != "BRK.A")
    n_oth = sum(v[1] for k, v in empty_rows_by_ticker.items() if k != "BRK.A")
    register_number("EmptyRowShare", f"{100 * e_all / n_bars_total:.1f}\\%")
    register_number("EmptyRowShareBRK", f"{100 * empty_rows_by_ticker['BRK.A'][0] / empty_rows_by_ticker['BRK.A'][1]:.0f}\\%")
    register_number("EmptyRowShareOthers", f"{100 * e_oth / n_oth:.1f}\\%")
    register_number("RowsPerSeries", fmt_int(25920))
    register_number("NTickers", str(len(tickers)))
    register_number("NBarsTotal", fmt_int(n_bars_total))
    register_number("ZeroBarShare", f"{100 * pooled_zero / pooled_bars:.1f}\\%")
    first = min(d.index.min() for d in raw.values())
    last = max(d.index.max() for d in raw.values())
    register_number("DataFirstDate", str(first.date()))
    register_number("DataLastDate", str(last.date()))
    med_bars = np.median([d.groupby(np.asarray(to_et(d.index).date)).size().median() for d in raw.values()])
    register_number("MedianBarsPerDay", f"{med_bars:.0f}")
    print(f"total bars {n_bars_total}, span {first} -> {last}, zero-return share {pooled_zero / pooled_bars:.3f}, median bars/day {med_bars}")

    # ------------------------------------------------------------ T-02 splits and class shares (project's own processor)
    split_rows, class_rows = [], []
    share_by_m = {}
    fe = FeatureEngineerV2()
    for period, bar_label in [("30min", 30), ("15min", 15)]:
        # exact pipeline preparation: resampling, market context, 28 features (rows with undefined features are dropped), labels
        resampled, _ = dp.load_and_engineer_all(str(DATA_DIR), fe, 0.3, period)
        for m in M_GRID:
            counts = np.zeros((3, 3))  # split x class
            sizes = np.zeros(3)
            rng_dates = [[None, None] for _ in range(3)]
            for t, df in resampled.items():
                lab = dp.compute_targets_and_labels(df.copy(), threshold_multiplier=m)
                parts = dp.temporal_split(lab)
                for si, part in enumerate(parts):
                    sizes[si] += len(part)
                    counts[si] += np.bincount(part["Target_Class"].astype(int), minlength=3)
                    lo, hi = part.index.min(), part.index.max()
                    rng_dates[si][0] = lo if rng_dates[si][0] is None else min(rng_dates[si][0], lo)
                    rng_dates[si][1] = hi if rng_dates[si][1] is None else max(rng_dates[si][1], hi)
            share_by_m[(bar_label, m)] = counts.sum(0) / counts.sum()
            if m == 0.3:
                for si, name in enumerate(["train", "calibration", "test"]):
                    split_rows.append([f"{bar_label} min", name, fmt_int(sizes[si]),
                                       f"{rng_dates[si][0].date()} -- {rng_dates[si][1].date()}",
                                       "/".join(f"{100 * c:.0f}" for c in counts[si] / counts[si].sum())])
            print(f"{bar_label}min m={m}: sizes {sizes.astype(int).tolist()}, class shares all {np.round(share_by_m[(bar_label, m)], 3)}, test {np.round(counts[2] / counts[2].sum(), 3)}")
            if bar_label == 30 and m == 0.3:
                register_number("TrainRows30", fmt_int(sizes[0]))
                register_number("CalRows30", fmt_int(sizes[1]))
                register_number("TestRows30", fmt_int(sizes[2]))
                register_number("TestStart", str(rng_dates[2][0].date()))
                register_number("TestEnd", str(rng_dates[2][1].date()))
                register_number("CalStart", str(rng_dates[1][0].date()))
                register_number("TrainStart", str(rng_dates[0][0].date()))
        if bar_label == 30:
            keep30 = resampled
            # share of rows whose next-bar target spans a gap (overnight / weekend / missing bars): next timestamp more than one bar ahead
            n_gap, n_all = 0, 0
            for tk, dfr in resampled.items():
                gaps = dfr.index.to_series().diff().shift(-1).dropna()
                n_gap += int((gaps > pd.Timedelta(minutes=30)).sum())
                n_all += len(gaps)
            register_number("GapTargetShareThirty", f"{100 * n_gap / n_all:.1f}\\%")
            print("share of 30-min rows whose next bar is not the adjacent slot:", n_gap / n_all)
    write_table("t02_splits",
                ["Bars", "Split", "Rows (all stocks)", "Date range", "Class shares D/N/U (\\%), $m=0.3$"],
                split_rows, "llrll",
                "Chronological 60/20/20 split of each stock, as used by the V2 pipeline, after resampling, computation of the 28 features (rows with undefined features are dropped) "
                "and labelling (rows before removing the initial window of each split). The split is by row count within each stock, so the date ranges shown are envelopes over "
                "all stocks and the split boundaries differ by up to a few days between stocks (BRK.A is sparsely traded and therefore has fewer bars per day). "
                "The class shares are those of the labelling with threshold multiplier $m=0.3$.",
                "tab:splits", resize=True, font=r"\small")

    # ------------------------------------------------------------ F-03 timeline of the split
    df30 = keep30["JPM"]
    n = len(df30)
    tr_end, ca_end = int(n * 0.6), int(n * 0.6) + int(n * 0.2)
    bounds = [(df30.index[0], df30.index[tr_end - 1]), (df30.index[tr_end], df30.index[ca_end - 1]), (df30.index[ca_end], df30.index[-1])]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.1))
    cols = [OI["blue"], OI["orange"], OI["verm"]]
    names = ["train (60%)", "calib. (20%)", "test (20%)"]
    for k, ((lo, hi), c, nm) in enumerate(zip(bounds, cols, names)):
        ax.barh(0, (hi - lo).days, left=lo, color=c, height=0.5)
        ax.text(lo + (hi - lo) / 2, 0, nm, ha="center", va="center", color="white", fontsize=7.5)
        ax.text(lo + (hi - lo) / 2, [0.0, 0.42, -0.42][k] if k else -0.42, f"{lo.date()} to {hi.date()} ({(hi - lo).days} days)",
                ha="center", va="center", fontsize=7, color=c)
    ax.set_yticks([])
    ax.set_ylim(-0.7, 0.7)
    ax.grid(False)
    ax.set_title("Chronological split of the 30-minute data (JPM shown; other stocks are identical up to a few bars)")
    ax.title.set_fontsize(8.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    savefig(fig, "f03_split_timeline")
    register_number("TestDays", str((bounds[2][1] - bounds[2][0]).days))

    # ------------------------------------------------------------ F-04 hours, zero returns, missing
    hour_cnt = np.zeros(24)
    hour_rate = np.zeros(24)
    hour_zero_num = np.zeros(24)
    hour_zero_den = np.zeros(24)
    for t, df in raw.items():
        h = np.asarray(to_et(df.index).hour)
        r0 = (df["Close"].pct_change(fill_method=None) == 0).to_numpy()
        n_days_t = len(set(np.asarray(to_et(df.index).date)))
        for k in range(24):
            m = h == k
            hour_cnt[k] += m.sum()
            hour_rate[k] += m.sum() / n_days_t / len(raw)
            hour_zero_num[k] += r0[m].sum()
            hour_zero_den[k] += m.sum()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.0), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    hours = np.arange(24)
    col_h = [OI["sky"] if 4 <= k <= 8 else OI["blue"] if 10 <= k <= 15 else "#999999" if k == 9 else OI["orange"] if 16 <= k <= 19 else "#DDDDDD" for k in hours]
    ax1.bar(hours, hour_rate, color=col_h, width=0.8)
    ax1.set_xlabel("Hour of day (US Eastern)")
    ax1.set_ylabel("Rows per hour per day (max. 12)")
    ax1.set_ylim(0, 15)
    ax1.set_xlim(3, 21)
    ax2.set_xlim(3, 21)
    ax1.set_title("(a) Rows per hour")
    ax2.bar(hours, 100 * hour_zero_num / np.maximum(hour_zero_den, 1), color=col_h, width=0.8)
    ax2.set_xlabel("Hour of day (US Eastern)")
    ax2.set_ylabel("Zero-return 5-min bars (%)")
    ax2.set_title("(b) Zero-return bars")
    miss = [1 - len(d) / (len(set(np.asarray(to_et(d.index).date))) * 192) for d in raw.values()]
    empt = [float(d[["Open", "High", "Low", "Close"]].isna().all(axis=1).mean()) for d in raw.values()]
    tot_gap = [a + b for a, b in zip(miss, empt)]
    order = np.argsort(tot_gap)[::-1]
    yy = np.arange(len(tickers))
    ax3.barh(yy, [100 * miss[i] for i in order], color=OI["purple"], label="absent")
    ax3.barh(yy, [100 * empt[i] for i in order], left=[100 * miss[i] for i in order], color=OI["orange"], label="empty")
    ax3.set_yticks(yy)
    ax3.set_yticklabels([tickers[i] for i in order], fontsize=6)
    ax3.invert_yaxis()
    ax3.set_xlabel("Share of 192 slots/day (%)")
    ax3.set_title("(c) No price")
    ax3.legend(fontsize=6.5, loc="lower right")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=OI["sky"], label="pre-market"), Patch(color=OI["blue"], label="regular"), Patch(color=OI["orange"], label="after-hours")],
               fontsize=6.5, loc="upper left")
    fig.tight_layout()
    savefig(fig, "f04_data_coverage")
    zr = 100 * hour_zero_num / np.maximum(hour_zero_den, 1)
    register_number("ZeroBarShareRegular", f"{zr[10:16].mean():.1f}\\%")
    register_number("ZeroBarShareExtendedMax", f"{zr[[4, 5, 6, 7, 8, 16, 17, 18, 19]].max():.0f}\\%")
    register_number("MissingBarsMax", f"{100 * max(miss):.1f}\\%")
    register_number("MissingBarsMedian", f"{100 * float(np.median(miss)):.1f}\\%")
    print("zero share by hour:", {int(k): round(float(zr[k]), 1) for k in hours if hour_zero_den[k] > 0}, "| missing max/median", max(miss), np.median(miss))

    # ------------------------------------------------------------ F-05 stylised facts (regular session, 5-min)
    rets, acf_r, acf_abs = [], [], []
    lags = np.arange(1, 61)
    for t, df in raw.items():
        d = df[session_minutes(df.index)].copy()
        d["day"] = d.index.normalize()
        d["r"] = d.groupby("day")["Close"].pct_change(fill_method=None)
        r = d["r"].dropna()
        rets.append(r)
        x = r.to_numpy()
        for arr, coll in [(x - x.mean(), acf_r), (np.abs(x) - np.abs(x).mean(), acf_abs)]:
            den = (arr ** 2).sum()
            coll.append([(arr[:-k] * arr[k:]).sum() / den for k in lags])
    allr = pd.concat(rets)
    z = (allr - allr.mean()) / allr.std()
    kurt = float(stats.kurtosis(allr, fisher=True))
    register_number("KurtosisFiveMin", f"{kurt:.0f}")
    acf_r, acf_abs = np.mean(acf_r, 0), np.mean(acf_abs, 0)
    n_eff = len(allr) / len(tickers)
    band = 1.96 / np.sqrt(n_eff)
    register_number("AcfAbsLagOne", f"{acf_abs[0]:.2f}")
    register_number("AcfRetLagOne", f"{acf_r[0]:.3f}")
    register_number("AcfAbsLagSixty", f"{acf_abs[-1]:.2f}")
    print(f"5-min regular-session returns: n={len(allr)}, excess kurtosis {kurt:.1f}; ACF r lag1 {acf_r[0]:.4f}; ACF |r| lag1 {acf_abs[0]:.3f} lag60 {acf_abs[-1]:.3f}; band {band:.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9))
    bins = np.linspace(-6, 6, 121)
    ax1.hist(np.clip(z, -6, 6), bins=bins, density=True, color="#BBBBBB", label="5-min returns (standardised)")
    xs = np.linspace(-6, 6, 400)
    ax1.plot(xs, stats.norm.pdf(xs), color=OI["verm"], lw=1.2, label="normal")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, 2)
    ax1.set_xlabel("Standardised return")
    ax1.set_ylabel("Density (log scale)")
    ax1.set_title(f"(a) Heavy tails (excess kurtosis {kurt:.0f})")
    ax1.legend(loc="lower center", fontsize=7)
    ax2.plot(lags, acf_abs, "o-", ms=2.5, lw=1.0, color=OI["blue"], label="$|r_t|$")
    ax2.plot(lags, acf_r, "s-", ms=2.5, lw=1.0, color=OI["orange"], label="$r_t$")
    ax2.axhspan(-band, band, color="#CCCCCC", alpha=0.5, label="95% band under independence")
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xlabel("Lag (5-minute bars)")
    ax2.set_ylabel("Autocorrelation (mean over stocks)")
    ax2.set_title("(b) Signs uncorrelated, magnitudes persistent")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "f05_stylised_facts")

    # ------------------------------------------------------------ F-06 label construction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(M_GRID))
    for bl, off, hatch in [(30, -0.2, ""), (15, 0.2, "///")]:
        bottom = np.zeros(len(M_GRID))
        for c in range(3):
            v = np.array([share_by_m[(bl, m)][c] for m in M_GRID])
            ax1.bar(x + off, v, 0.36, bottom=bottom, color=CLASS_COLORS[c], hatch=hatch, edgecolor="white", linewidth=0.4,
                    label=CLASS_NAMES[c] if bl == 30 else None)
            bottom += v
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{m:g}" for m in M_GRID])
    ax1.set_xlabel("Threshold multiplier $m$")
    ax1.set_ylabel("Share of all bars")
    ax1.set_title("(a) Class balance vs $m$\n(solid: 30 min, hatched: 15 min)")
    ax1.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    d = keep30["JPM"].copy()
    d = dp.compute_targets_and_labels(d, threshold_multiplier=0.3)
    seg = d.iloc[-200:-120]
    ax2.plot(np.arange(len(seg)), 1e4 * seg["fwd_return"].to_numpy(), color="#666666", lw=0.7, zorder=1)
    ax2.fill_between(np.arange(len(seg)), -1e4 * seg["volatility_threshold"], 1e4 * seg["volatility_threshold"], color=OI["yellow"], alpha=0.5, label="neutral band $\\pm\\theta$")
    for c in range(3):
        sel = (seg["Target_Class"] == c).to_numpy()
        ax2.scatter(np.arange(len(seg))[sel], 1e4 * seg["fwd_return"].to_numpy()[sel], s=9, color=CLASS_COLORS[c], zorder=3, label=CLASS_NAMES[c])
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("30-minute bar (80 consecutive bars of JPM)")
    ax2.set_ylabel("Next-bar return (bps)")
    ax2.set_title("(b) Labels from a rolling-volatility band ($m=0.3$)")
    ax2.legend(ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    savefig(fig, "f06_labels")
    register_number("ShareNeutralM03", f"{100 * share_by_m[(30, 0.3)][1]:.0f}\\%")
    register_number("ShareNeutralM10", f"{100 * share_by_m[(30, 1.0)][1]:.0f}\\%")

    # ------------------------------------------------------------ T-12 reference points on the actual test rows of R30-1
    lab = "R30-1"
    d = load_trades(lab).sort_values(["Ticker", "Timestamp"]).reset_index(drop=True)
    y = d["Target_Class"].to_numpy()
    pi = np.bincount(y, minlength=3) / len(y)
    ret = d["Actual_Fwd_Return"].to_numpy()
    prev = d.groupby("Ticker")["Actual_Fwd_Return"].shift(1).to_numpy()
    dt = d.groupby("Ticker")["Timestamp"].diff().dt.total_seconds().to_numpy()
    contiguous = (dt <= 1800 + 1) & ~np.isnan(prev)
    ok = contiguous & (ret != 0) & (prev != 0)
    persist_hit = float((np.sign(prev[ok]) == np.sign(ret[ok])).mean())
    always_up = float((ret[ret != 0] > 0).mean())
    # volatility timing: previous bar's |return| as a score for "next bar is NEUTRAL"
    # (note: available at the close of bar t, one bar fresher than the models' inputs)
    vol_score = -np.abs(prev)
    prevs = d.groupby("Ticker")["Actual_Fwd_Return"].transform(lambda s: s.abs().shift(1).rolling(10, min_periods=5).mean()).to_numpy()
    ok2 = contiguous & ~np.isnan(prevs)

    def auc_bin(score, pos):
        n1, n0 = int(pos.sum()), int((~pos).sum())
        r = stats.rankdata(score)
        return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    # ---- directional reference predictors (signs of earlier returns), with information timing spelled out.
    # Row i of the log holds the return of bar i+1 (target). prev1 = return of bar i (row i-1), known at the close of bar i but
    # *not* in the models' window (which ends at bar i-1); prev2 = return of bar i-1, the last return inside the models' window.
    d["ret_bps"] = 1e4 * d["Actual_Fwd_Return"]
    for k in (1, 2):
        d[f"prev{k}"] = d.groupby("Ticker")["Actual_Fwd_Return"].shift(k)
        dt_k = d.groupby("Ticker")["Timestamp"].diff(k).dt.total_seconds()
        d.loc[dt_k > k * 1800 + 1, f"prev{k}"] = np.nan
    d["Day"] = to_et(d["Timestamp"]).dt.date
    d["Session"] = np.asarray(__import__("common").session_labels(d["Timestamp"]))
    days_u = np.array(sorted(d["Day"].unique()))
    day_i = {x: i for i, x in enumerate(days_u)}
    d["di"] = d["Day"].map(day_i)
    rngb = np.random.default_rng(11)

    def directional(pos: np.ndarray, mask: np.ndarray):
        m = mask & (pos != 0) & (d["ret_bps"].to_numpy() != 0)
        g = pos * d["ret_bps"].to_numpy()
        hit = float((g[m] > 0).mean())
        Gd = np.zeros(len(days_u))
        Nd = np.zeros(len(days_u))
        np.add.at(Gd, d["di"].to_numpy()[mask & (pos != 0)], g[mask & (pos != 0)])
        np.add.at(Nd, d["di"].to_numpy()[mask & (pos != 0)], 1)
        ib = rngb.integers(0, len(days_u), size=(5000, len(days_u)))
        bs = Gd[ib].sum(1) / np.maximum(Nd[ib].sum(1), 1)
        return hit, Gd.sum() / max(Nd.sum(), 1), np.quantile(bs, 0.025), np.quantile(bs, 0.975), int(Nd.sum())

    sess = d["Session"].to_numpy()
    allm = np.ones(len(d), bool)
    regm = sess == "regular"
    preds = {
        "Always UP": np.ones(len(d)),
        "Repeat sign of previous bar (bar $t$, not in model window)": np.sign(d["prev1"].fillna(0).to_numpy()),
        "Reverse sign of previous bar (bar $t$, not in model window)": -np.sign(d["prev1"].fillna(0).to_numpy()),
        "Reverse sign of bar $t-1$ (last bar in model window)": -np.sign(d["prev2"].fillna(0).to_numpy()),
    }
    dir_rows = []
    for name, pos in preds.items():
        h, e, lo, hi, n_ = directional(pos, allm)
        h_r, e_r, lo_r, hi_r, n_r = directional(pos, regm)
        dir_rows.append([name, fmt_int(n_), f"{100 * h:.1f}\\%", f"{e:.2f} [{lo:.2f}, {hi:.2f}]", f"{100 * h_r:.1f}\\%", f"{e_r:.2f} [{lo_r:.2f}, {hi_r:.2f}]"])
        key = "".join(w for w in name.replace("$", "").replace("(", " ").replace(")", " ").replace(",", " ").replace("-", " ").title().split() if w.isalnum())[:40]
        print(f"{name}: n={n_} sign acc {100 * h:.2f}% gross {e:.2f} bps [{lo:.2f},{hi:.2f}]; regular session: acc {100 * h_r:.2f}% gross {e_r:.2f} [{lo_r:.2f},{hi_r:.2f}] n={n_r}")
        if name.startswith("Reverse sign of previous"):
            register_number("RevOneHit", f"{100 * h:.1f}\\%")
            register_number("RevOneBps", f"{e:.2f}")
            register_number("RevOneHitReg", f"{100 * h_r:.1f}\\%")
            register_number("RevOneBpsReg", f"{e_r:.2f}")
            register_number("RevOneN", fmt_int(n_))
        if name.startswith("Reverse sign of bar"):
            register_number("RevTwoHit", f"{100 * h:.1f}\\%")
            register_number("RevTwoBps", f"{e:.2f}")
    write_table("t12b_directional_baselines",
                ["Directional reference predictor", "Calls", "Sign acc.", "Gross bps/call [95\\% CI]", "Sign acc. (regular)", "Gross bps/call (regular)"],
                dir_rows, "lrrrrr",
                "Directional reference predictors on the test rows of run R30-1 (gross of costs; every bar with a non-zero reference sign is a call; "
                "sign accuracy on non-zero returns; CI: bootstrap over test days). ``Regular'' restricts to bars starting in the 09:30--16:00 US Eastern session. "
                "The first two reverse/repeat rows use the return of bar $t$, which lies between the end of the models' input window (bar $t-1$) and the start of "
                "the predicted return (bar $t+1$).",
                "tab:directional_baselines", resize=True, font=r"\footnotesize")

    auc_prev = auc_bin(vol_score[contiguous], (y[contiguous] == 1))
    auc_roll = auc_bin(-prevs[ok2], (y[ok2] == 1))
    print(f"class priors {pi.round(3)}; persistence sign hit {persist_hit:.4f} (n={ok.sum()}); always-up {always_up:.4f}; "
          f"AUC(prev |r| -> NEUTRAL) {auc_prev:.3f}; AUC(mean |r| of last 10 bars -> NEUTRAL) {auc_roll:.3f}")
    register_number("PersistenceSignHit", f"{100 * persist_hit:.1f}\\%")
    register_number("AlwaysUpHit", f"{100 * always_up:.1f}\\%")
    register_number("AucPrevAbsNeutral", f"{auc_prev:.2f}")
    register_number("AucRollAbsNeutral", f"{auc_roll:.2f}")
    register_number("MajorityAcc", f"{100 * pi.max():.1f}\\%")
    register_number("PriorsAcc", f"{100 * (pi ** 2).sum():.1f}\\%")
    base_rows = [
        ["Uniform random class", "33.3\\%", "0.333", "0.333"],
        ["Random with class priors", f"{100 * (pi ** 2).sum():.1f}\\%", "0.333", "0.333"],
        ["Always NEUTRAL (majority)", f"{100 * pi.max():.1f}\\%", "0.333", f"{2 * pi[1] / (1 + pi[1]) / 3:.3f}"],
    ]
    write_table("t12_baselines",
                ["Reference predictor (test rows of run R30-1)", "Accuracy", "Balanced acc.", "Macro-F1"],
                base_rows, "lrrr",
                "Reference points for the three-class task on the test set of run R30-1 ($n=" + fmt_int(len(y)) + "$; class shares D/N/U = "
                + "/".join(f"{100 * p:.0f}" for p in pi) + "\\%). For directional reference points: always predicting UP has a sign accuracy of "
                + f"{100 * always_up:.1f}\\%, and predicting the sign of the previous bar's return {100 * persist_hit:.1f}\\% "
                + "(non-zero returns; the latter uses information one bar fresher than the models' inputs).",
                "tab:baselines", font=r"\small")
    flush_numbers()


if __name__ == "__main__":
    main()
