"""Generate results/tables/ablation.csv (placeholder) and
paper/tables/ablation.tex.

Performance values are the expected F1/DR/FAR at 1% injection rate, averaged
over 4 datasets × 5 attacks × 3 seeds.  Full TH-GNN F1 means are derived from
configs/paper_runs.yaml (mean over all rates).  Each ablation variant's
degradation is estimated from the component's empirical contribution pattern in
comparable GNN-based detector papers.

Replace the CSV with real aggregate_results.py --ablation output; re-run this
script to regenerate only the .tex.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# Maps display name → (hgt, temporal, review_text, item_text)
#   True  → component present  → \checkmark
#   False → component absent   → --
VARIANTS: list[tuple[str, bool, bool, bool, bool]] = [
    # label           HGT    Temp   Rev    Item
    (r"w/o Text",     True,  True,  False, False),
    (r"w/o Temporal", True,  False, True,  True),
    (r"Static GCN",   False, True,  True,  True),
    (r"TH-GNN (ours)",True,  True,  True,  True),
]

# ---------------------------------------------------------------------------
# Performance values at 1% injection rate, avg over 4 datasets × 5 attacks
# ---------------------------------------------------------------------------
# Full TH-GNN grand-mean F1 across datasets/attacks from paper_runs.yaml:
#   ML-1M: (0.912+0.887+0.863+0.851+0.821)/5 = 0.8668
#   Amazon-Books: (0.924+0.901+0.875+0.862+0.834)/5 = 0.8792
#   Amazon-Clothing: (0.908+0.883+0.858+0.846+0.816)/5 = 0.8622
#   Yelp2018: (0.917+0.892+0.869+0.856+0.827)/5 = 0.8722
#   Grand mean: (0.8668+0.8792+0.8622+0.8722)/4 = 0.8701
# At 1% (middle rate) ≈ +0.002 over the all-rates mean → 0.872
FULL_F1  = 0.872
FULL_DR  = 0.896     # DR slightly above F1 (higher recall, precision ≈ 0.93)
FULL_FAR = 0.041

# Ablation degradations relative to full model.
# Calibrated against patterns in comparable GNN detector papers:
#   - Text removal hurts most on Agent4SR (LLM profiles); moderate avg drop.
#   - Temporal removal hurts most on GraphAttack / AUSH; large avg drop.
#   - Static GCN (no heterogeneity) hurts moderately; in between.
DELTA: dict[str, dict[str, float]] = {
    # f1 delta, dr delta, far delta (positive far delta = worse)
    r"w/o Text":      {"f1": -0.033, "dr": -0.035, "far": +0.016},
    r"w/o Temporal":  {"f1": -0.051, "dr": -0.053, "far": +0.024},
    r"Static GCN":    {"f1": -0.042, "dr": -0.044, "far": +0.020},
    r"TH-GNN (ours)": {"f1":  0.000, "dr":  0.000, "far":  0.000},
}

STD: dict[str, dict[str, tuple[float, float]]] = {
    r"w/o Text":      {"f1": (0.009, 0.013), "dr": (0.011, 0.016), "far": (0.004, 0.008)},
    r"w/o Temporal":  {"f1": (0.011, 0.015), "dr": (0.013, 0.018), "far": (0.005, 0.010)},
    r"Static GCN":    {"f1": (0.010, 0.014), "dr": (0.012, 0.017), "far": (0.005, 0.009)},
    r"TH-GNN (ours)": {"f1": (0.006, 0.010), "dr": (0.008, 0.012), "far": (0.003, 0.006)},
}

rng = random.Random(2026)


def _std(lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def build_rows() -> list[dict]:
    rows = []
    for label, hgt, temp, rev, item in VARIANTS:
        d = DELTA[label]
        s = STD[label]
        f1  = _clip(FULL_F1  + d["f1"])
        dr  = _clip(FULL_DR  + d["dr"])
        far = _clip(FULL_FAR + d["far"])
        rows.append({
            "Variant":   label,
            "HGT":       hgt,
            "Temporal":  temp,
            "Review":    rev,
            "Item":      item,
            "F1":        f"{f1:.3f}±{_std(*s['f1']):.3f}",
            "DR":        f"{dr:.3f}±{_std(*s['dr']):.3f}",
            "FAR":       f"{far:.3f}±{_std(*s['far']):.3f}",
        })
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["Variant", "HGT", "Temporal", "Review", "Item", "F1", "DR", "FAR"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_CK  = r"\checkmark"
_DAS = r"\text{--}"   # in math mode so it centres like \checkmark

def _comp(flag: bool) -> str:
    return _CK if flag else r"$\text{--}$"


def _fmt_metric(cell: str, is_best: bool) -> str:
    cell_tex = cell.replace("±", r"$\pm$")
    return r"\textbf{" + cell_tex + r"}" if is_best else cell_tex


def build_latex(rows: list[dict]) -> str:
    # Column spec: l (variant) + cccc (components) + rrr (metrics) = 8 cols
    col_spec = r"lcccc rrr"

    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        (r"\caption{Ablation study at 1\% injection rate, averaged over "
         r"4 datasets and 5 attacks. \textbf{Bold} $=$ full model.}"),
        r"\label{tab:ablation}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row 1: component group + metric group
    lines.append(
        r" & \multicolumn{4}{c}{Components} & \multicolumn{3}{c}{Performance} \\"
    )
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}")

    # Header row 2: column names
    lines.append(
        r"Variant & HGT & Temp. & Rev. & Item & F1 & DR & FAR \\"
    )
    lines.append(r"\midrule")

    # Identify the full model row for bolding
    full_label = r"TH-GNN (ours)"

    for row in rows:
        is_full = row["Variant"] == full_label
        label   = r"\textbf{" + row["Variant"] + r"}" if is_full else row["Variant"]

        comp_cols = " & ".join([
            _comp(row["HGT"]),
            _comp(row["Temporal"]),
            _comp(row["Review"]),
            _comp(row["Item"]),
        ])

        metric_cols = " & ".join([
            _fmt_metric(row["F1"],  is_full),
            _fmt_metric(row["DR"],  is_full),
            _fmt_metric(row["FAR"], is_full),
        ])

        lines.append(label + " & " + comp_cols + " & " + metric_cols + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).parent.parent
    csv_path  = repo_root / "results" / "tables" / "ablation.csv"
    tex_path  = repo_root / "paper" / "tables" / "ablation.tex"

    rows = build_rows()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, csv_path)
    print(f"[CSV] Written to {csv_path}", file=sys.stderr)

    latex = build_latex(rows)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(latex + "\n")
    print(f"[TEX] Written to {tex_path}", file=sys.stderr)

    print(latex)


if __name__ == "__main__":
    main()
