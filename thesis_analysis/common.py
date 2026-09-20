"""Shared helpers for the thesis analysis scripts.

Every number and figure in the thesis is produced by a script in this folder.
Inputs (read-only): A100-results/RESULTS, RESULTS, DATA.
Outputs: out/figures/*.pdf|png, out/tables/*.tex, out/tables/numbers.tex.

Paths can be overridden with environment variables:
    A100_RESULTS, DATA_DIR, THESIS_OUT
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

A100 = Path(os.environ.get("A100_RESULTS", REPO / "A100-results" / "RESULTS"))
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO / "DATA"))
OUT = Path(os.environ.get("THESIS_OUT", HERE / "out"))
OUT_FIG = OUT / "figures"
OUT_TAB = OUT / "tables"
INPUTS = HERE / "inputs"
for _d in (OUT_FIG, OUT_TAB):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Run manifest (chronological order). Label -> (folder, bar length in minutes)
# --------------------------------------------------------------------------
RUNS: dict[str, tuple[str, int]] = {
    "R30-0": ("ga_full_v2_multi_all_stocks_20260618_095350", 30),
    "R15": ("ga_full_v2_multi_15min_all_stocks_20260620_141559", 15),
    "R30-1": ("ga_full_v2_multi_30min_all_stocks_20260621_125311", 30),
    "R30-2": ("ga_full_v2_multi_30min_all_stocks_20260621_215451", 30),
    "R30-3": ("ga_full_v2_multi_30min_all_stocks_20260622_124903", 30),
    "R30-4": ("ga_full_v2_multi_30min_all_stocks_20260622_233819", 30),
    "R30-5": ("ga_full_v2_multi_30min_all_stocks_20260623_111437", 30),
    "R30-6": ("ga_full_v2_multi_30min_all_stocks_20260623_210817", 30),
}
LABELS = list(RUNS)
LABELS_30 = [k for k, v in RUNS.items() if v[1] == 30]
CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]

# Okabe-Ito colour-blind safe palette
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
}
CLASS_COLORS = [OI["verm"], "#999999", OI["blue"]]  # DOWN, NEUTRAL, UP
RUN_COLORS = {
    "R30-0": OI["black"], "R15": OI["orange"], "R30-1": OI["sky"], "R30-2": OI["green"],
    "R30-3": OI["blue"], "R30-4": OI["verm"], "R30-5": OI["purple"], "R30-6": "#7A7A7A",
}

TEXT_WIDTH_IN = 15.0 / 2.54  # 15 cm text block


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "pdf.fonttype": 42,
    })


set_style()


def to_et(ts):
    """UTC timestamps (index or Series of naive datetimes; the data files are in UTC) -> US Eastern wall-clock (DST-aware)."""
    if isinstance(ts, pd.DatetimeIndex):
        return ts.tz_localize("UTC").tz_convert("America/New_York")
    return ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")


def et_minutes(ts):
    e = to_et(ts)
    if isinstance(e, pd.DatetimeIndex):
        return np.asarray(e.hour * 60 + e.minute)
    return (e.dt.hour * 60 + e.dt.minute).to_numpy()


def session_labels(ts) -> np.ndarray:
    """pre-market 04:00-09:30 ET, regular 09:30-16:00 ET, after-hours 16:00-20:00 ET (bar start times)."""
    m = et_minutes(ts)
    return np.select([(m >= 570) & (m < 960), (m >= 240) & (m < 570)], ["regular", "pre-market"], "after-hours")


def run_dir(label: str) -> Path:
    return A100 / RUNS[label][0]


def load_trades(label: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir(label) / "test_trade_log.csv", parse_dates=["Timestamp"])
    et = to_et(df["Timestamp"])
    df["Day"] = et.dt.date  # trading day = US Eastern calendar date (04:00-20:00 ET falls on one date; UTC dates would split it)
    df["Hour"] = et.dt.hour
    return df


def load_pop(label: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(run_dir(label) / "ga_population_history.jsonl")]
    df = pd.DataFrame(rows)
    df["valid"] = df["m2_sharpe"] > -900
    return df


def load_json(label: str, name: str) -> dict:
    return json.load(open(run_dir(label) / name))


def savefig(fig, name: str) -> None:
    fig.savefig(OUT_FIG / f"{name}.pdf")
    fig.savefig(OUT_FIG / f"{name}.png", dpi=170)
    plt.close(fig)
    print("figure:", name)


# --------------------------------------------------------------------------
# LaTeX helpers
# --------------------------------------------------------------------------
def tex_escape(s: object) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", r"\,")


def write_table(name: str, header: list[str], rows: list[list], colfmt: str,
                caption: str, label: str, note: str | None = None,
                resize: bool = False, font: str = r"\small") -> None:
    """Write a booktabs table to out/tables/<name>.tex. Rows are lists of pre-formatted strings."""
    lines = [r"\begin{table}[htbp]", r"\centering", font]
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{colfmt}}}")
    lines.append(r"\toprule")
    hdr = [(r"\makecell{" + h.replace("\n", r"\\{}") + "}") if "\n" in h else h for h in header]
    lines.append(" & ".join(hdr) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        if r == "MIDRULE":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(str(x) for x in r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append("}")
    if note:
        lines.append(rf"\\[2pt]{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    (OUT_TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("table:", name)


_NUMBERS: dict[str, str] = {}


def register_number(name: str, value: str) -> None:
    """Register a LaTeX macro \\num<name> with the given text; flushed by flush_numbers()."""
    import re
    words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    name = re.sub(r"\d", lambda m: words[int(m.group())], name)  # TeX control-sequence names cannot contain digits
    _NUMBERS[name] = value


def flush_numbers(filename: str = "numbers.tex") -> None:
    path = OUT_TAB / filename
    existing: dict[str, str] = {}
    if path.exists():
        import re
        for m in re.finditer(r"\\newcommand\{\\num([A-Za-z]+)\}\{(.*)\}", path.read_text()):
            existing[m.group(1)] = m.group(2)
    existing.update(_NUMBERS)
    body = [r"% Auto-generated by thesis_analysis/*.py -- do not edit by hand"]
    for k in sorted(existing):
        body.append(rf"\newcommand{{\num{k}}}{{{existing[k]}}}")
    path.write_text("\n".join(body) + "\n")
    print("numbers:", len(existing))


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"
