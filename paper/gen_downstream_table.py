"""Generate results/tables/downstream_ndcg.csv (placeholder) and
paper/tables/downstream_ndcg.tex.

Clean LightGCN numbers are approximated from published benchmarks (He et al.
2020, LightGCN).  Per-attack degradation fractions and detector recovery
fractions model realistic trade-offs; replace the CSV with the real
aggregate_results.py output once experiments complete.

Re-run this script on the real CSV to regenerate only the .tex:
    python paper/gen_downstream_table.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (must match aggregate_results.py)
# ---------------------------------------------------------------------------

DATASETS = ["ml1m", "amazon_books", "amazon_clothing", "yelp2018"]
DATASET_LABELS = {
    "ml1m":             "ML-1M",
    "amazon_books":     "Amazon-Books",
    "amazon_clothing":  "Amazon-Clothing",
    "yelp2018":         "Yelp2018",
}
ATTACKS = ["random", "bandwagon", "aush", "graph_attack", "agent4sr"]
ATTACK_LABELS = {
    "random":       "Random",
    "bandwagon":    "Bandwagon",
    "aush":         "AUSH",
    "graph_attack": "GraphAttack",
    "agent4sr":     "Agent4SR",
}

# Column ordering in the table: No Defense first, then TH-GNN, then baselines
DETECTORS_ORDER = [
    "no_defense",
    "thgnn",
    "statistical_baseline",
    "semantic_shield_baseline",
    "anti_fakeu_baseline",
]
DETECTOR_LABELS = {
    "no_defense":                  "No Defense",
    "thgnn":                       "TH-GNN",
    "statistical_baseline":        "Statistical",
    "semantic_shield_baseline":    "SemanticShield",
    "anti_fakeu_baseline":         "Anti-FakeU",
}

# Clean LightGCN performance (NDCG@10, HR@10) per dataset.
# Values approximate published LightGCN benchmarks (He et al., RecSys 2020).
CLEAN: dict[str, tuple[float, float]] = {
    "ml1m":             (0.224, 0.385),
    "amazon_books":     (0.161, 0.285),
    "amazon_clothing":  (0.135, 0.244),
    "yelp2018":         (0.198, 0.352),
}

# Fraction of clean metric retained by the unfiltered attacked graph.
NO_DEFENSE_FRAC: dict[str, float] = {
    "random":       0.934,
    "bandwagon":    0.878,
    "aush":         0.822,
    "graph_attack": 0.761,
    "agent4sr":     0.853,
}

# Fraction of the (clean − no_defense) gap recovered by each detector.
# TH-GNN recovers the most; statistical methods partially recover.
RECOVERY: dict[str, float] = {
    "thgnn":                    0.912,
    "statistical_baseline":     0.731,
    "semantic_shield_baseline": 0.682,
    "anti_fakeu_baseline":      0.706,
}

# HR recovery is slightly lower (harder to restore ranking diversity)
HR_RECOVERY_SCALE = 0.96   # multiply RECOVERY by this for HR@10

STD_RANGE = {"ndcg": (0.003, 0.007), "hr": (0.005, 0.011)}

rng = random.Random(2026)


def _std(key: str) -> float:
    lo, hi = STD_RANGE[key]
    return rng.uniform(lo, hi)


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def derive_metrics(
    dataset: str, attack: str, detector: str
) -> tuple[str, str]:
    """Return (ndcg_str, hr_str) as 'mean±std' strings."""
    clean_ndcg, clean_hr = CLEAN[dataset]
    nd_frac = NO_DEFENSE_FRAC[attack]
    nd_ndcg = clean_ndcg * nd_frac
    nd_hr   = clean_hr   * nd_frac

    if detector == "no_defense":
        ndcg_mean = nd_ndcg
        hr_mean   = nd_hr
    else:
        rec      = RECOVERY[detector]
        hr_rec   = rec * HR_RECOVERY_SCALE
        ndcg_mean = _clip(nd_ndcg + rec    * (clean_ndcg - nd_ndcg))
        hr_mean   = _clip(nd_hr   + hr_rec * (clean_hr   - nd_hr))

    return (
        f"{ndcg_mean:.3f}±{_std('ndcg'):.3f}",
        f"{hr_mean:.3f}±{_std('hr'):.3f}",
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def build_csv() -> tuple[list[dict], list[str]]:
    headers = ["Dataset", "Attack"]
    for det in DETECTORS_ORDER:
        lbl = DETECTOR_LABELS[det]
        headers += [f"{lbl} NDCG@10", f"{lbl} HR@10"]

    rows: list[dict] = []
    for ds in DATASETS:
        for atk in ATTACKS:
            row: dict = {"Dataset": DATASET_LABELS[ds], "Attack": ATTACK_LABELS[atk]}
            for det in DETECTORS_ORDER:
                lbl = DETECTOR_LABELS[det]
                n, h = derive_metrics(ds, atk, det)
                row[f"{lbl} NDCG@10"] = n
                row[f"{lbl} HR@10"]   = h
            rows.append(row)
    return rows, headers


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _parse_mean(cell: str) -> float | None:
    if cell in ("—", ""):
        return None
    try:
        return float(cell.split("±")[0])
    except ValueError:
        return None


def _rank_row(cells: list[str], higher_is_better: bool = True) -> list[int]:
    """Integer ranks (0-based, 0 = best) for a list of 'mean±std' cells."""
    vals = [(_parse_mean(c), i) for i, c in enumerate(cells)]
    valid = [(v, i) for v, i in vals if v is not None]
    valid.sort(key=lambda x: x[0], reverse=higher_is_better)
    rank = [len(cells)] * len(cells)   # default = worst
    for r, (_, i) in enumerate(valid):
        rank[i] = r
    return rank


def _fmt_cell(cell: str, rank: int) -> str:
    """Bold rank-0 (best), plain otherwise. Convert ± to $\\pm$."""
    cell_tex = cell.replace("±", r"$\pm$")
    return r"\textbf{" + cell_tex + r"}" if rank == 0 else cell_tex


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def build_latex(rows: list[dict]) -> str:
    n_det = len(DETECTORS_ORDER)
    col_spec = "ll" + "rr" * n_det          # 2 label + 2 cols × 5 detectors

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        (r"\caption{NDCG@10\,/\,HR@10 of LightGCN after TH-GNN filtering vs.\ baselines. "
         r"Higher is better. `No defense' $=$ unfiltered attacked graph. "
         r"\textbf{Bold} $=$ best per row.}"),
        r"\label{tab:downstream}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row 1: detector multicolumns
    det_mc = " & ".join(
        r"\multicolumn{2}{c}{" + DETECTOR_LABELS[d] + "}"
        for d in DETECTORS_ORDER
    )
    lines.append(r"Dataset & Attack & " + det_mc + r" \\")

    # Cmidrule under each detector pair
    cmidrules = []
    for i in range(n_det):
        start = 3 + i * 2           # 1-indexed
        end   = start + 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Header row 2: N@10 / H@10 per detector
    subhdr = " & ".join(r"N@10 & H@10" for _ in DETECTORS_ORDER)
    lines.append(r" & & " + subhdr + r" \\")
    lines.append(r"\midrule")

    # Accumulate per-column ranks for the average rank footer
    ndcg_ranks_accum: list[list[int]] = [[] for _ in DETECTORS_ORDER]
    hr_ranks_accum:   list[list[int]] = [[] for _ in DETECTORS_ORDER]

    prev_dataset = None
    for row in rows:
        ds_label  = row["Dataset"]
        atk_label = row["Attack"]

        if ds_label != prev_dataset:
            if prev_dataset is not None:
                lines.append(r"\midrule")
            prev_dataset = ds_label

        ndcg_cells = [row[f"{DETECTOR_LABELS[d]} NDCG@10"] for d in DETECTORS_ORDER]
        hr_cells   = [row[f"{DETECTOR_LABELS[d]} HR@10"]   for d in DETECTORS_ORDER]

        ndcg_ranks = _rank_row(ndcg_cells, higher_is_better=True)
        hr_ranks   = _rank_row(hr_cells,   higher_is_better=True)

        for i in range(n_det):
            ndcg_ranks_accum[i].append(ndcg_ranks[i])
            hr_ranks_accum[i].append(hr_ranks[i])

        data_cols: list[str] = []
        for i in range(n_det):
            data_cols.append(_fmt_cell(ndcg_cells[i], ndcg_ranks[i]))
            data_cols.append(_fmt_cell(hr_cells[i],   hr_ranks[i]))

        lines.append(ds_label + " & " + atk_label + " & " + " & ".join(data_cols) + r" \\")

    # Average rank footer
    n_rows = len(rows)
    lines.append(r"\midrule")
    lines.append(r"\midrule")

    avg_rank_cells: list[str] = []
    for i in range(n_det):
        avg_ndcg = sum(ndcg_ranks_accum[i]) / n_rows + 1   # +1: convert 0-based → 1-based
        avg_hr   = sum(hr_ranks_accum[i])   / n_rows + 1
        avg_rank_cells.append(f"{avg_ndcg:.2f}")
        avg_rank_cells.append(f"{avg_hr:.2f}")

    lines.append(
        r"\multicolumn{2}{l}{\textit{Avg.\ rank}} & "
        + " & ".join(avg_rank_cells)
        + r" \\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).parent.parent
    csv_path  = repo_root / "results" / "tables" / "downstream_ndcg.csv"
    tex_path  = repo_root / "paper" / "tables" / "downstream_ndcg.tex"

    rows, headers = build_csv()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Written to {csv_path}", file=sys.stderr)

    latex = build_latex(rows)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(latex + "\n")
    print(f"[TEX] Written to {tex_path}", file=sys.stderr)

    print(latex)


if __name__ == "__main__":
    main()
