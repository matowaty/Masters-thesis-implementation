"""AN-13: inventory of all experiments (tables only).

Input : inputs/run_manifest_all.csv (one row per result folder: folder, date, machine, group,
        thesis_status, thesis_location, checkpoint). The folders themselves are not read.
Output: out/tables/t17_experiments_overview.tex  (Section 6.5: one row per group of experiments)
        out/tables/t18_inventory_all.tex          (Appendix A.4: every run with its folder name)
        numbers: numInvRuns (all result folders), numInvGroupsRuns...
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict

import pandas as pd

from common import INPUTS, LABELS, OUT_TAB, RUNS, caption_opt, flush_numbers, register_number, source_line, tex_escape

MACHINE = {"local": "laptop", "A100 (Colab)": "A100", "cloud CPU": "cloud CPU"}
SEEDED_ORDER = [  # (folder prefix, description as in Table 7.2)
    ("attention_bilstm_jpm", "BiLSTM (h=1)"),
    ("attention_bilstm_attn_jpm", "BiLSTM + attention (h=1)"),
    ("time_horizon_t1_jpm", "horizon t+1"),
    ("time_horizon_t2_jpm", "horizon t+2"),
    ("time_horizon_t3_jpm", "horizon t+3"),
]


def split_name(folder: str) -> tuple[str, str]:
    m = re.match(r"^(.*)_(\d{8}_\d{6})$", folder)
    assert m, folder
    return m.group(1), m.group(2)


def month_day(d: str) -> str:
    return pd.Timestamp(d).strftime("%-d %b")


def date_range(dates: list[str]) -> str:
    a, b = pd.Timestamp(min(dates)), pd.Timestamp(max(dates))
    if a == b:
        return f"{a.day} {a.strftime('%b')}"
    if a.month == b.month:
        return f"{a.day}--{b.day} {a.strftime('%b')}"
    return f"{a.day} {a.strftime('%b')}--{b.day} {b.strftime('%b')}"


def machines(sub: pd.DataFrame) -> str:
    c = Counter(MACHINE[m] for m in sub["machine"])
    if len(c) == 1:
        return next(iter(c))
    return ", ".join(f"{n} {k}" for k, n in sorted(c.items(), key=lambda x: (-x[1], x[0])))


def main() -> None:
    man = pd.read_csv(INPUTS / "run_manifest_all.csv")
    man["grp"] = man["group"].str[0]
    n_all = len(man)
    register_number("InvRuns", str(n_all))

    # ---------------- overview table (Section 6.5) ----------------
    g = {k: man[man["grp"] == k] for k in "ABCD"}
    assert (len(g["A"]), len(g["B"]), len(g["C"]), len(g["D"])) == (15, 18, 6, 8), [len(v) for v in g.values()]
    rows = [
        ["V1, regression (single runs)", str(len(g["B"])), machines(g["B"]), date_range(list(g["B"]["date"])),
         r"Sect.~\ref{sec:res_v1}; Tab.~\ref{tab:v1_results}"],
        ["V1, seeded repeats", f"{len(g['A'])} ({len(g['A']) // 3}$\\times$3 seeds)", machines(g["A"]),
         date_range(list(g["A"]["date"])), r"Sect.~\ref{sec:res_v1}; Tab.~\ref{tab:quick_runs}"],
        ["V2, early runs", str(len(g["C"])), machines(g["C"]), date_range(list(g["C"]["date"])),
         r"Sect.~\ref{sec:res_selected}, \ref{sec:res_ga}; Tab.~\ref{tab:early_runs}"],
        ["V2, full GA runs", str(len(g["D"])), machines(g["D"]), date_range(list(g["D"]["date"])),
         r"Sect.~\ref{sec:res_v2}--\ref{sec:discussion}; Tab.~\ref{tab:inventory}"],
        ["V2, window-offset check", "6 fits", "cloud CPU", "19 Sep", r"Sect.~\ref{sec:res_shift}"],
        ["V2, Model~2 check", "40 evaluations (20$\\times$2)", "cloud CPU", "20 Sep", r"Sect.~\ref{sec:res_ga}; Tab.~\ref{tab:o1_model2}"],
    ]
    hdr = ["Group of experiments", "Runs", "Machine", "Dates (2026)", "Reported in"]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\caption" + caption_opt("tab:experiments_overview") + r"{Overview of all experiments of the thesis. ``Machine'' is the computer on which the runs were made: the laptop of the author, an NVIDIA A100 accelerator in a hosted notebook (Google Colab) or a CPU in the cloud session used for this thesis. The window-offset check consists of six fits (three seeds, two variants); the Model~2 check consists of 20 random individuals, each evaluated in two modes (Section~\ref{sec:res_ga}). Every run is listed with the name of its folder in Appendix~\ref{app:inventory}.}",
             r"\label{tab:experiments_overview}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lllll}", r"\toprule", " & ".join(hdr) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", source_line("tab:experiments_overview"), r"\end{table}"]
    (OUT_TAB / "t17_experiments_overview.tex").write_text("\n".join(lines) + "\n")
    print("table: t17_experiments_overview")

    # ---------------- full inventory (Appendix A.4) ----------------
    def tt(s: str) -> str:
        return r"\texttt{" + tex_escape(s) + "}"

    body: list[str] = []

    def head(title: str) -> None:
        body.append(r"\multicolumn{5}{l}{\textit{" + title + r"}} \\")

    def row(id_, prefix, stamp, machine, rep) -> None:
        body.append(" & ".join([id_, tt(prefix), tt(stamp), machine, rep]) + r" \\")

    # V1 single runs
    head(r"Version 1, single runs (Table~\ref{tab:v1_results})")
    v1 = g["B"].copy()
    v1["id"] = v1["thesis_status"].str.extract(r"\((V1-\d+)\)")[0]
    v1["n"] = v1["id"].str.extract(r"V1-(\d+)")[0].astype(int)
    for _, r in v1.sort_values("n").iterrows():
        p, s = split_name(r["folder"])
        row(r["id"], p, s, MACHINE[r["machine"]], r"Tab.~\ref{tab:v1_results}")
    body.append(r"\midrule")
    # seeded repeats
    head(r"Version 1, seeded repeats (Table~\ref{tab:quick_runs})")
    for k, (pref, desc) in enumerate(SEEDED_ORDER, 1):
        sub = g["A"][g["A"]["folder"].str.startswith(pref + "_2026")].sort_values("folder")
        assert len(sub) == 3, (pref, len(sub))
        for j, f in enumerate(sub["folder"], 1):  # seeds 1, 2, 3 in the order of the timestamps (checked in seed.json)
            p_, s_ = split_name(f)
            row(f"S-{k}.{j}", p_, s_, "cloud CPU", rf"Tab.~\ref{{tab:quick_runs}}, seed {j}")
    body.append(r"\midrule")
    # V2 early
    head(r"Version 2, early runs (Sections~\ref{sec:res_selected} and~\ref{sec:res_ga})")
    early = g["C"].copy()
    early["t"] = pd.to_datetime(early["folder"].str.extract(r"_(\d{8}_\d{6})$")[0], format="%Y%m%d_%H%M%S")
    # laptop folders carry local time (UTC+2): order by UTC
    early["utc"] = early.apply(lambda r: r["t"] - pd.Timedelta(hours=2) if r["machine"] == "local" else r["t"], axis=1)
    early = early.sort_values("utc").reset_index(drop=True)
    early_ids = {}
    for k, (_, r) in enumerate(early.iterrows(), 1):
        p, s = split_name(r["folder"])
        early_ids[r["folder"]] = f"E-{k}"
        note = (r"Tab.~\ref{tab:early_runs}" if "171052" not in r["folder"]
                else r"Fig.~\ref{fig:early_fitness}")
        row(f"E-{k}", p, s, MACHINE[r["machine"]], note)
    body.append(r"\midrule")
    # V2 full
    head(r"Version 2, full GA runs (Table~\ref{tab:inventory})")
    fold2lab = {v[0]: k for k, v in RUNS.items()}
    for lab in LABELS:
        folder = RUNS[lab][0]
        r = g["D"][g["D"]["folder"] == folder].iloc[0]
        p, s = split_name(folder)
        row(lab, p, s, MACHINE[r["machine"]], r"Sect.~\ref{sec:res_v2}--\ref{sec:discussion}")
    body.append(r"\midrule")
    head(r"Version 2, window-offset check and Model~2 check (not folders of \texttt{RESULTS})")
    body.append(" & ".join(["W-1--6", r"\texttt{RESULTS\_V2SHIFT/\{base,shift\}\_seed\{1,2,3\}}", "19 Sep 2026", "cloud CPU",
                            r"Sect.~\ref{sec:res_shift}"]) + r" \\")
    body.append(" & ".join(["O1", r"\texttt{RESULTS\_O1/\{fast,full\}\_part\{0,1\}.jsonl}", "20 Sep 2026", "cloud CPU",
                            r"Sect.~\ref{sec:res_ga}, Tab.~\ref{tab:o1_model2}"]) + r" \\")

    cap = ("Inventory of all runs of the thesis. The folder of a run is the name in the column ``Folder name'' followed by an underscore and the "
           "timestamp; the timestamps are those in the folder names (UTC for the A100 and the cloud CPU, local time UTC+2 for the laptop). "
           "The IDs V1-$n$ are those of Table~\\ref{tab:v1_results}; S-$k.j$ is configuration $k$ of the seeded repeats with seed $j$ "
           "(Table~\\ref{tab:quick_runs}); E-$k$ are the early runs of the second pipeline (E-5, the first attempt of a full GA run, was stopped after ten generations and only its GA log was kept); "
           "R30-$k$ and R15 are the full GA runs (Table~\\ref{tab:inventory}); W-1--6 and O1 are the two checks made for this thesis (Sections~\\ref{sec:res_shift} and~\\ref{sec:res_ga}). The folders are also listed in \\texttt{thesis\\_analysis/inputs/run\\_manifest\\_all.csv}.")
    out = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
           r"\caption" + caption_opt("tab:inventory_all") + "{" + cap + "}", r"\label{tab:inventory_all}",
           r"\renewcommand{\arraystretch}{1.05}",
           r"\begin{tabular}{@{}l@{\hspace{4pt}}l@{\hspace{4pt}}l@{\hspace{4pt}}l@{\hspace{4pt}}l@{}}",
           r"\toprule", r"ID & Folder name & Timestamp & Machine & Reported in \\", r"\midrule"]
    out += body
    out += [r"\bottomrule", r"\end{tabular}", source_line("tab:inventory_all"), r"\end{table}"]
    (OUT_TAB / "t18_inventory_all.tex").write_text("\n".join(out) + "\n")
    print("table: t18_inventory_all")
    n_rows = sum(1 for b in body if not b.startswith(r"\multicolumn") and b != r"\midrule")
    print("inventory rows:", n_rows)
    flush_numbers()


if __name__ == "__main__":
    main()
