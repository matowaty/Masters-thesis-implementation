"""Shared helpers for the thesis analysis scripts.

Every number and figure in the thesis is produced by a script in this folder.
Inputs (read-only): RESULTS (all runs, including the eight full A100 runs), DATA.
Outputs: out/figures/*.pdf|png, out/tables/*.tex, out/tables/numbers.tex.

Paths can be overridden with environment variables:
    A100_RESULTS, DATA_DIR, THESIS_OUT
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

A100 = Path(os.environ.get("A100_RESULTS", REPO / "RESULTS"))
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


def _register_fonts() -> None:
    """Optionally register the Arial files of the thesis template (THESIS_FONT_DIR=<folder with arial*.ttf>)."""
    d = os.environ.get("THESIS_FONT_DIR")
    if d:
        for f in sorted(Path(d).glob("*.ttf")):
            font_manager.fontManager.addfont(str(f))


def set_style() -> None:
    """Figure style of the faculty template: Arial 9 pt (Liberation Sans, a metric clone, is the fallback)."""
    _register_fonts()
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold", "mathtext.fallback": "cm",
        "font.size": 9, "axes.titlesize": 9.5,
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


def wrap_header(h: str, maxlen: int = 12) -> str:
    """Break a long column header into lines of about maxlen characters (newline-separated; math spans are kept whole)."""
    if "\n" in h or len(h) <= maxlen:
        return h
    toks = re.findall(r"(?:[^\s$]|\$[^$]*\$)+", h)
    lines, cur = [], ""
    for t in toks:
        if cur and len(re.sub(r"[\\$]", "", cur)) + 1 + len(re.sub(r"[\\$]", "", t)) > maxlen:
            lines.append(cur)
            cur = t
        else:
            cur = (cur + " " + t) if cur else t
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def write_table(name: str, header: list[str], rows: list[list], colfmt: str,
                caption: str, label: str, note: str | None = None,
                resize: bool = False, font: str = r"\small", rules: bool = False, full_width: bool = False) -> None:
    """Write a booktabs table to out/tables/<name>.tex. Rows are lists of pre-formatted strings.

    rules=True adds a faded thin rule between the rows (\\rowrule, defined in main.tex) and a little more vertical space,
    for tables with wrapped cells; full_width=True sets the table in tabularx at the text width (colfmt must contain an X column)."""
    lines = [r"\begin{table}[htbp]", r"\centering", font]
    lines.append(rf"\caption{caption_opt(label)}{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    if rules:
        lines.append(r"\renewcommand{\arraystretch}{1.25}")
    env = "tabularx" if full_width else "tabular"
    lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{colfmt}}}" if full_width else rf"\begin{{tabular}}{{{colfmt}}}")
    lines.append(r"\toprule")
    if resize:  # wide tables are scaled down to the text width: wrap long headers so that the scale stays close to 1
        header = [wrap_header(h) for h in header]
    hdr = [(r"\makecell{" + h.replace("\n", r"\\{}") + "}") if "\n" in h else h for h in header]
    lines.append(" & ".join(hdr) + r" \\")
    lines.append(r"\midrule")
    for i, r in enumerate(rows):
        if r == "MIDRULE":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(str(x) for x in r) + r" \\")
            if rules and i + 1 < len(rows) and rows[i + 1] != "MIDRULE":
                lines.append(r"\rowrule")
    lines.append(r"\bottomrule")
    lines.append(rf"\end{{{env}}}")
    if resize:
        lines.append("}")
    if note:
        lines.append(rf"\\[2pt]{{\footnotesize {note}}}")
    lines.append(source_line(label))
    lines.append(r"\end{table}")
    (OUT_TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("table:", name)



# --------------------------------------------------------------------------
# Sources of figures and tables (Sect. "source line" of every float; see SOURCES.md, section 6)
# label -> (short title for the lists of figures / tables, source macro defined in main.tex, script or None)
#   srcdraw = own drawing made in TikZ, srcown = own work (descriptive), srcexp = own calculations from the results of the
#   experiments, srcdata = own calculations from the price data (LSEG export).
# --------------------------------------------------------------------------
FLOAT_META: dict[str, tuple[str, str, str | None]] = {
    # drawings (TikZ)
    "fig:taxonomy": ("Taxonomy of stock forecasting approaches", "srcdraw", None),
    "fig:lstm_cell": ("Internal structure of an LSTM cell", "srcdraw", None),
    "fig:bilstm": ("Bidirectional LSTM unfolded over the input window", "srcdraw", None),
    "fig:ga_flowchart": ("Standard genetic algorithm loop", "srcdraw", None),
    "fig:pipeline": ("Structure of the second pipeline (V2)", "srcdraw", None),
    "fig:chromosome": ("Encoding of an individual of the V2 genetic algorithm", "srcdraw", None),
    # descriptive tables
    "tab:features": ("The 28 candidate features of the V2 pipeline", "srcown", None),
    "tab:ga_space": ("Search space of the V2 genetic algorithm", "srcown", None),
    "tab:modules": ("Modules of the forecasting pipeline", "srcown", None),
    "tab:versions": ("Pinned library versions of the pipeline", "srcown", None),
    "tab:config": ("Configuration of the full GA runs of the second pipeline", "srcown", None),
    "tab:provenance": ("Scripts of the analysis and their outputs", "srcown", None),
    "tab:experiments_overview": ("Overview of all experiments of the thesis", "srcown", "an_inventory.py"),
    "tab:inventory_all": ("Inventory of all runs of the thesis", "srcown", "an_inventory.py"),
    # data
    "fig:data_coverage": ("Coverage of the five-minute data", "srcdata", "an_data.py"),
    "fig:stylised_facts": ("Stylised facts of the five-minute returns", "srcdata", "an_data.py"),
    "fig:split_timeline": ("Chronological 60/20/20 split of the 30-minute data", "srcdata", "an_data.py"),
    "fig:labels": ("Class labels of the V2 pipeline", "srcdata", "an_data.py"),
    "tab:data_inventory": ("The 19 stocks used in the experiments", "srcdata", "an_data.py"),
    "tab:splits": ("Chronological 60/20/20 split of each stock", "srcdata", "an_data.py"),
    "tab:baselines": ("Reference points for the three-class task", "srcdata", "an_data.py"),
    "tab:directional_baselines": ("Directional reference predictors", "srcdata", "an_data.py"),
    # experiments
    "fig:v1_results": ("Test-set results of the 18 runs of the first pipeline", "srcexp", "an_v1.py"),
    "fig:quick_runs": ("Seeded repeats of the first pipeline on JPM", "srcexp", "an_quick.py"),
    "fig:confusion": ("Confusion matrix of Model~1 on the test sets, pooled", "srcexp", "an_v2_skill.py"),
    "fig:class_shares": ("Class shares of the test sets and of the predictions of Model~1", "srcexp", "an_v2_skill.py"),
    "fig:agreement": ("Agreement between the class predictions of the runs", "srcexp", "an_v2_skill.py"),
    "fig:approved": ("Composition of the bars approved by Model~2", "srcexp", "an_runs.py"),
    "fig:m2_confidence": ("Confidence of Model~2 on the test sets", "srcexp", "an_v2_skill.py"),
    "fig:ga_convergence": ("Progress of the eight GA runs", "srcexp", "an_ga.py"),
    "fig:sharpe_gap": ("Sharpe ratio of the best individuals: calibration set and test set", "srcexp", "an_runs.py"),
    "fig:early_fitness": ("Fitness in the GA logs of the early runs", "srcexp", "an_early.py"),
    "fig:o1_fitness": ("Test O1: fitness and test Sharpe ratio for four variants of Model~2", "srcexp", "an_o1.py"),
    "fig:gene_evolution": ("Composition of the population by gene value over the generations", "srcexp", "an_ga.py"),
    "fig:landscape": ("Calibration Sharpe ratios of all valid individuals", "srcexp", "an_ga.py"),
    "fig:gene_effects": ("Mean calibration Sharpe ratio by value of each hyperparameter gene", "srcexp", "an_ga.py"),
    "fig:feature_stability": ("Stability of the feature selection of the best chromosomes", "srcexp", "an_runs.py"),
    "fig:uncertainty": ("Uncertainty of the gross profit per call", "srcexp", "an_economics.py"),
    "fig:breakeven": ("Break-even one-way transaction cost of the directional calls", "srcexp", "an_economics.py"),
    "fig:equity": ("Cumulative profit of the equal-weight portfolio", "srcexp", "an_economics.py"),
    "fig:edge_groups": ("Gross profit per call by stock and by hour of the day", "srcexp", "an_economics.py"),
    "fig:v2_window_shift": ("Effect of the window offset on the test set", "srcexp", "an_v2shift.py"),
    "fig:v2_shift_sessions": ("Effect of the window offset by session of the decision bar", "srcexp", "an_v2shift.py"),
    "fig:confusion_grid": ("Confusion matrices of Model~1 for the eight runs", "srcexp", "an_v2_skill.py"),
    "tab:inventory": ("Inventory of the eight completed GA runs", "srcexp", "an_runs.py"),
    "tab:v1_results": ("Results of the 18 runs of the first pipeline", "srcexp", "an_v1.py"),
    "tab:reported_vs_real": ("Metrics reported by the pipeline and the approved trades", "srcexp", "an_runs.py"),
    "tab:v2_skill": ("Model~1 (three-class) on the test set", "srcexp", "an_v2_skill.py"),
    "tab:m2_discrimination": ("Model~2 as a classifier of ``Model~1 was correct''", "srcexp", "an_v2_skill.py"),
    "tab:magnitude_direction": ("Information of Model~1 about the size and the sign of the next move", "srcexp", "an_magnitude.py"),
    "tab:chromosomes": ("Best chromosome of each GA run", "srcexp", "an_runs.py"),
    "tab:selected_features_app": ("Features selected by the best chromosome of each GA run", "srcexp", "an_runs.py"),
    "tab:economics": ("Counterfactual gross P\\&L of the calls of Model~1", "srcexp", "an_economics.py"),
    "tab:sessions": ("Gross P\\&L of the calls of Model~1 by trading session", "srcexp", "an_economics.py"),
    "tab:economics_net": ("Net P\\&L per call after transaction costs", "srcexp", "an_economics.py"),
    "tab:feature_frequency": ("Selection frequency of the 28 candidate features", "srcexp", "an_runs.py"),
    "tab:ga_summary": ("Summary of the genetic-algorithm search per run", "srcexp", "an_ga.py"),
    "tab:quick_runs": ("Seeded repeats of the V1 experiments", "srcexp", "an_quick.py"),
    "tab:v2_window_shift": ("Effect of the one-bar window offset on Model~1", "srcexp", "an_v2shift.py"),
    "tab:v2_shift_sessions": ("Directional calls of Model~1 by session (window-offset experiment)", "srcexp", "an_v2shift.py"),
    "tab:rule_segments": ("Sign accuracy of the reversal rule by segment", "srcexp", "an_v2shift.py"),
    "tab:early_runs": ("Test-set metrics printed by the early runs", "srcexp", "an_early.py"),
    "tab:o1_model2": ("Test O1: fitness with four variants of Model~2", "srcexp", "an_o1.py"),
}


def caption_opt(label: str) -> str:
    """Optional argument of \\caption (entry of the list of figures / tables) that states the source."""
    short, macro, _ = FLOAT_META[label]
    return f"[{short}. Source: \\{macro}]"


def source_line(label: str) -> str:
    """Line under a figure or table naming its source (macro defined in main.tex) and, if any, the generating script."""
    _, macro, script = FLOAT_META[label]
    extra = "" if script is None else "; script \\texttt{" + script.replace("_", "\\_") + "}"
    return f"\\figsource{{\\{macro}L{extra}}}"


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
