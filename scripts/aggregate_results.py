"""Aggregate raw experiment results into publication-ready tables.

Reads every JSON file under ``results/raw/`` and produces:

  results/tables/main_detection.csv      — F1/DR/FAR per dataset×attack×detector
  results/tables/downstream_ndcg.csv     — NDCG@10/HR@10 per dataset×attack×detector
  results/tables/budget_curves.csv       — F1 vs injection rate (ML-1M only)
  results/tables/ablation.csv            — F1/DR/FAR for ablation conditions
  results/tables/latex/                  — LaTeX booktabs versions of each table

Also prints a summary of which datasets THGNN wins on and its average
improvement over the best baseline.

Usage
-----
    python scripts/aggregate_results.py [--results-dir results/raw]
                                        [--out-dir results/tables]
                                        [--dataset ml1m]   # optional filter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DETECTORS = ["thgnn", "statistical_baseline", "semantic_shield_baseline", "anti_fakeu_baseline"]
DETECTOR_LABELS = {
    "thgnn":                       "TH-GNN",
    "statistical_baseline":        "Statistical",
    "semantic_shield_baseline":    "SemanticShield",
    "anti_fakeu_baseline":         "Anti-FakeU",
}
ATTACKS = ["random", "bandwagon", "aush", "graph_attack", "agent4sr"]
ATTACK_LABELS = {
    "random":       "Random",
    "bandwagon":    "Bandwagon",
    "aush":         "AUSH",
    "graph_attack": "GraphAttack",
    "agent4sr":     "Agent4SR",
}
DATASETS = ["ml1m", "amazon_books", "amazon_clothing", "yelp2018"]
DATASET_LABELS = {
    "ml1m":             "ML-1M",
    "amazon_books":     "Amazon-Books",
    "amazon_clothing":  "Amazon-Clothing",
    "yelp2018":         "Yelp2018",
}
RATES = [0.005, 0.01, 0.05]

# Ablation labels — "detector" field values produced by run_ablation.py
ABLATION_CONDITIONS = {
    "thgnn":              "Full TH-GNN",
    "thgnn_no_text":      "w/o Text",
    "thgnn_no_temporal":  "w/o Temporal",
    "thgnn_static_gcn":   "Static GCN",
    # legacy keys kept for backward compatibility
    "thgnn_no_fusion":    "w/o Fusion",
    "thgnn_no_hgt":       "w/o HGT",
    "thgnn_no_burstiness":"w/o Burstiness",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path, dataset_filter: Optional[str] = None) -> pd.DataFrame:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("status") != "ok":
            continue

        cfg  = data.get("config", {})
        det  = data.get("detection", {})
        rec  = data.get("recommendation", {})

        if dataset_filter and cfg.get("dataset") != dataset_filter:
            continue

        records.append({
            "dataset":   cfg.get("dataset", ""),
            "attack":    cfg.get("attack",  ""),
            "rate":      float(cfg.get("rate", 0)),
            "detector":  cfg.get("detector", ""),
            "seed":      int(cfg.get("seed", 0)),
            "f1":        float(det.get("f1",   np.nan)),
            "dr":        float(det.get("dr",   np.nan)),
            "far":       float(det.get("far",  np.nan)),
            "ndcg10":    float(rec.get("ndcg@10", np.nan)),
            "hr10":      float(rec.get("hr@10",   np.nan)),
        })

    if not records:
        return pd.DataFrame(columns=[
            "dataset", "attack", "rate", "detector", "seed",
            "f1", "dr", "far", "ndcg10", "hr10",
        ])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _mean_std(vals: np.ndarray) -> str:
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return "—"
    m = valid.mean()
    s = valid.std()
    return f"{m:.3f}±{s:.3f}"


def _bold_best_latex(row_strs: List[str], higher_is_better: bool = True) -> List[str]:
    """Wrap the best cell in ``\\textbf{}``. Handles '—' gracefully."""
    vals = []
    for cell in row_strs:
        if cell == "—":
            vals.append(np.nan)
        else:
            try:
                vals.append(float(cell.split("±")[0]))
            except ValueError:
                vals.append(np.nan)

    if all(np.isnan(v) for v in vals):
        return row_strs

    best_fn = np.nanargmax if higher_is_better else np.nanargmin
    best_idx = best_fn(vals)

    result = []
    for i, cell in enumerate(row_strs):
        if i == best_idx and cell != "—":
            result.append(r"\textbf{" + cell + r"}")
        else:
            result.append(cell)
    return result


def _to_latex(df: pd.DataFrame, caption: str = "", higher_cols: Optional[List[str]] = None) -> str:
    """Convert a DataFrame of string cells to a LaTeX booktabs table.

    Bolds the best value per row in each column group.
    """
    if higher_cols is None:
        higher_cols = []

    # Build header
    col_str = " & ".join(df.columns.tolist())
    rows_latex: List[str] = []
    for _, row in df.iterrows():
        cells = list(row.values)
        # Bold best within each metric group defined by column names
        # We bold per-row within each metric type: F1, DR, FAR, NDCG, HR
        rows_latex.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "}",
        r"\toprule",
        col_str + r" \\",
        r"\midrule",
    ]
    lines += rows_latex
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 1: Main detection results
# ---------------------------------------------------------------------------

def build_main_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: dataset × attack (averaged over injection rates and seeds).
    Columns: detector × metric (F1, DR, FAR).
    """
    groups = df.groupby(["dataset", "attack", "detector"])
    records = []

    for (dataset, attack, detector), grp in groups:
        records.append({
            "dataset":  dataset,
            "attack":   attack,
            "detector": detector,
            "f1":       grp["f1"].values,
            "dr":       grp["dr"].values,
            "far":      grp["far"].values,
        })

    if not records:
        return pd.DataFrame()

    # Pivot: one row per (dataset, attack), one column-group per detector
    col_headers = ["Dataset", "Attack"]
    for det in DETECTORS:
        lbl = DETECTOR_LABELS.get(det, det)
        col_headers += [f"{lbl} F1", f"{lbl} DR", f"{lbl} FAR"]

    table_rows = []
    for dataset in DATASETS:
        for attack in ATTACKS:
            row = [
                DATASET_LABELS.get(dataset, dataset),
                ATTACK_LABELS.get(attack, attack),
            ]
            for det in DETECTORS:
                match = [r for r in records
                         if r["dataset"] == dataset and r["attack"] == attack and r["detector"] == det]
                if match:
                    m = match[0]
                    row += [_mean_std(m["f1"]), _mean_std(m["dr"]), _mean_std(m["far"])]
                else:
                    row += ["—", "—", "—"]
            table_rows.append(row)

    table_rows = [r for r in table_rows if any(c != "—" for c in r[2:])]
    return pd.DataFrame(table_rows, columns=col_headers)


def build_main_detection_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Detection performance (F1 / DR / FAR) across datasets and attack types."
        r"  Mean~$\pm$~std over three seeds and three injection rates."
        r"  \textbf{Bold} = best F1 per row.}",
        r"\label{tab:main_detection}",
    ]

    n_det = len(DETECTORS)
    col_spec = "ll" + ("rrr" * n_det)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Multi-column header row 1
    det_headers = " & ".join(
        r"\multicolumn{3}{c}{" + DETECTOR_LABELS.get(d, d) + "}"
        for d in DETECTORS
    )
    lines.append(r"Dataset & Attack & " + det_headers + r" \\")

    # Header row 2: F1 / DR / FAR per detector
    sub_headers = " & ".join(r"F1 & DR & FAR" for _ in DETECTORS)
    lines.append(r"\cmidrule(lr){3-" + str(2 + 3 * n_det) + "}")
    lines.append(r" &  & " + sub_headers + r" \\")
    lines.append(r"\midrule")

    # Data rows — bold best F1 per row
    f1_col_indices = [2 + 3 * i for i in range(n_det)]  # 0-indexed

    prev_dataset = None
    for _, row in df.iterrows():
        cells = list(row.values)
        # Bold best F1 across detectors
        f1_cells = [cells[i] for i in f1_col_indices]
        f1_bolded = _bold_best_latex(f1_cells, higher_is_better=True)
        for j, idx in enumerate(f1_col_indices):
            cells[idx] = f1_bolded[j]

        if cells[0] != prev_dataset:
            if prev_dataset is not None:
                lines.append(r"\midrule")
            prev_dataset = cells[0]

        lines.append(" & ".join(str(c) for c in cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: Downstream recommendation quality
# ---------------------------------------------------------------------------

def build_downstream_ndcg(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(["dataset", "attack", "detector"])
    records = []
    for (dataset, attack, detector), grp in groups:
        records.append({
            "dataset":  dataset,
            "attack":   attack,
            "detector": detector,
            "ndcg10":   grp["ndcg10"].values,
            "hr10":     grp["hr10"].values,
        })

    col_headers = ["Dataset", "Attack"]
    for det in DETECTORS:
        lbl = DETECTOR_LABELS.get(det, det)
        col_headers += [f"{lbl} NDCG@10", f"{lbl} HR@10"]

    table_rows = []
    for dataset in DATASETS:
        for attack in ATTACKS:
            row = [
                DATASET_LABELS.get(dataset, dataset),
                ATTACK_LABELS.get(attack, attack),
            ]
            for det in DETECTORS:
                match = [r for r in records
                         if r["dataset"] == dataset and r["attack"] == attack and r["detector"] == det]
                if match:
                    m = match[0]
                    row += [_mean_std(m["ndcg10"]), _mean_std(m["hr10"])]
                else:
                    row += ["—", "—"]
            table_rows.append(row)

    table_rows = [r for r in table_rows if any(c != "—" for c in r[2:])]
    return pd.DataFrame(table_rows, columns=col_headers)


def build_downstream_ndcg_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    n_det = len(DETECTORS)
    col_spec = "ll" + ("rr" * n_det)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Downstream recommendation quality (NDCG@10 / HR@10) after detection and filtering."
        r"  \textbf{Bold} = best NDCG@10 per row.}",
        r"\label{tab:downstream_ndcg}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    det_headers = " & ".join(
        r"\multicolumn{2}{c}{" + DETECTOR_LABELS.get(d, d) + "}"
        for d in DETECTORS
    )
    lines.append(r"Dataset & Attack & " + det_headers + r" \\")
    sub_headers = " & ".join(r"N@10 & H@10" for _ in DETECTORS)
    lines.append(r"\cmidrule(lr){3-" + str(2 + 2 * n_det) + "}")
    lines.append(r" &  & " + sub_headers + r" \\")
    lines.append(r"\midrule")

    ndcg_col_indices = [2 + 2 * i for i in range(n_det)]
    prev_dataset = None
    for _, row in df.iterrows():
        cells = list(row.values)
        ndcg_cells = [cells[i] for i in ndcg_col_indices]
        ndcg_bolded = _bold_best_latex(ndcg_cells, higher_is_better=True)
        for j, idx in enumerate(ndcg_col_indices):
            cells[idx] = ndcg_bolded[j]

        if cells[0] != prev_dataset:
            if prev_dataset is not None:
                lines.append(r"\midrule")
            prev_dataset = cells[0]

        lines.append(" & ".join(str(c) for c in cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: Budget curves (ML-1M, all injection rates)
# ---------------------------------------------------------------------------

def build_budget_curves(df: pd.DataFrame) -> pd.DataFrame:
    ml1m = df[df["dataset"] == "ml1m"]
    groups = ml1m.groupby(["attack", "rate", "detector"])
    records = []
    for (attack, rate, detector), grp in groups:
        records.append({
            "attack":   attack,
            "rate":     rate,
            "detector": detector,
            "f1":       grp["f1"].values,
        })

    col_headers = ["Attack", "Rate"]
    for det in DETECTORS:
        col_headers.append(f"{DETECTOR_LABELS.get(det, det)} F1")

    table_rows = []
    for attack in ATTACKS:
        for rate in RATES:
            row = [ATTACK_LABELS.get(attack, attack), f"{rate:.3f}"]
            for det in DETECTORS:
                match = [r for r in records
                         if r["attack"] == attack and abs(r["rate"] - rate) < 1e-9
                         and r["detector"] == det]
                row.append(_mean_std(match[0]["f1"]) if match else "—")
            table_rows.append(row)

    table_rows = [r for r in table_rows if any(c != "—" for c in r[2:])]
    return pd.DataFrame(table_rows, columns=col_headers)


def build_budget_curves_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    n_det = len(DETECTORS)
    col_spec = "lr" + "r" * n_det

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{F1 vs.\ injection rate on ML-1M."
        r"  \textbf{Bold} = best detector per row.}",
        r"\label{tab:budget_curves}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    det_header = " & ".join(DETECTOR_LABELS.get(d, d) for d in DETECTORS)
    lines.append(r"Attack & Rate & " + det_header + r" \\")
    lines.append(r"\midrule")

    det_col_indices = list(range(2, 2 + n_det))
    prev_attack = None
    for _, row in df.iterrows():
        cells = list(row.values)
        det_cells = [cells[i] for i in det_col_indices]
        det_bolded = _bold_best_latex(det_cells, higher_is_better=True)
        for j, idx in enumerate(det_col_indices):
            cells[idx] = det_bolded[j]

        if cells[0] != prev_attack:
            if prev_attack is not None:
                lines.append(r"\midrule")
            prev_attack = cells[0]

        lines.append(" & ".join(str(c) for c in cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4: Ablation study
# ---------------------------------------------------------------------------

def build_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: ablation condition × dataset, cols: F1, DR, FAR averaged over attacks/rates/seeds."""
    ablation_detectors = [d for d in df["detector"].unique() if d in ABLATION_CONDITIONS]
    if not ablation_detectors:
        # No ablation data yet — build a skeleton table with placeholder rows
        col_headers = ["Condition", "Dataset", "F1", "DR", "FAR"]
        return pd.DataFrame(columns=col_headers)

    groups = df[df["detector"].isin(ablation_detectors)].groupby(["detector", "dataset"])
    records = []
    for (detector, dataset), grp in groups:
        records.append({
            "condition": ABLATION_CONDITIONS.get(detector, detector),
            "dataset":   DATASET_LABELS.get(dataset, dataset),
            "f1":        _mean_std(grp["f1"].values),
            "dr":        _mean_std(grp["dr"].values),
            "far":       _mean_std(grp["far"].values),
        })

    col_headers = ["Condition", "Dataset", "F1", "DR", "FAR"]
    table_rows = []
    for cond_key in ABLATION_CONDITIONS:
        cond_label = ABLATION_CONDITIONS[cond_key]
        for dataset in DATASETS:
            ds_label = DATASET_LABELS.get(dataset, dataset)
            match = [r for r in records
                     if r["condition"] == cond_label and r["dataset"] == ds_label]
            if match:
                m = match[0]
                table_rows.append([cond_label, ds_label, m["f1"], m["dr"], m["far"]])

    return pd.DataFrame(table_rows, columns=col_headers)


def build_ablation_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return r"% No ablation data found."
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation study: effect of removing TH-GNN components."
        r"  \textbf{Bold} = best F1 per dataset.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Condition & Dataset & F1 & DR & FAR \\",
        r"\midrule",
    ]

    prev_dataset = None
    for _, row in df.iterrows():
        cells = list(row.values)
        if cells[1] != prev_dataset:
            if prev_dataset is not None:
                lines.append(r"\midrule")
            prev_dataset = cells[1]
        lines.append(" & ".join(str(c) for c in cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 5: Generalisation (train on seen attacks, test on unseen attack)
# ---------------------------------------------------------------------------

# Human-readable labels for train_tag values
_TRAIN_TAG_LABELS = {
    "seen4":    "Random + Bandwagon + AUSH + GraphAttack",
    "agent4sr": "Agent4SR",
}
_TEST_ATTACK_LABELS = {
    "agent4sr":    "Agent4SR (unseen LLM)",
    "graph_attack": "GraphAttack (unseen GNN)",
}


def load_generalisation_results(results_dir: Path) -> pd.DataFrame:
    """Load only generalisation_*.json files from results_dir."""
    records = []
    for path in sorted(results_dir.glob("generalisation_*.json")):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("status") != "ok":
            continue

        cfg = data.get("config", {})
        det = data.get("detection", {})

        records.append({
            "dataset":       cfg.get("dataset",      ""),
            "scenario":      cfg.get("scenario",     ""),
            "train_tag":     cfg.get("train_tag",    ""),
            "train_attacks": cfg.get("train_attacks", []),
            "test_attack":   cfg.get("test_attack",  ""),
            "detector":      cfg.get("detector",     ""),
            "seed":          int(cfg.get("seed", 0)),
            "f1":            float(det.get("f1",  np.nan)),
            "dr":            float(det.get("dr",  np.nan)),
            "far":           float(det.get("far", np.nan)),
        })

    if not records:
        return pd.DataFrame(columns=[
            "dataset", "scenario", "train_tag", "train_attacks",
            "test_attack", "detector", "seed", "f1", "dr", "far",
        ])
    return pd.DataFrame(records)


def build_generalisation(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the generalisation results into a publication-ready DataFrame.

    Rows: (train_attacks, test_attack, dataset) — one row per group.
    Columns: detector × metric (F1, DR, FAR), all as mean ± std strings.
    """
    if df.empty:
        return pd.DataFrame()

    col_headers = ["Train Attacks", "Test Attack", "Dataset"]
    for det in DETECTORS:
        lbl = DETECTOR_LABELS.get(det, det)
        col_headers += [f"{lbl} F1", f"{lbl} DR", f"{lbl} FAR"]

    # Collect raw arrays per (train_tag, test_attack, dataset, detector)
    groups = df.groupby(["train_tag", "test_attack", "dataset", "detector"])
    raw: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for (train_tag, test_attack, dataset, detector), grp in groups:
        raw[(train_tag, test_attack, dataset, detector)] = {
            "f1":  grp["f1"].values,
            "dr":  grp["dr"].values,
            "far": grp["far"].values,
        }

    # Build rows ordered by scenario order, then dataset, then detector
    scenario_order = [
        ("seen4",    "agent4sr"),
        ("agent4sr", "graph_attack"),
    ]
    table_rows = []
    for train_tag, test_attack in scenario_order:
        train_label  = _TRAIN_TAG_LABELS.get(train_tag,    train_tag)
        test_label   = _TEST_ATTACK_LABELS.get(test_attack, test_attack)
        for dataset in DATASETS:
            ds_label = DATASET_LABELS.get(dataset, dataset)
            # Only include datasets that appear in the results
            has_data = any(
                (train_tag, test_attack, dataset, det) in raw
                for det in DETECTORS
            )
            if not has_data:
                continue

            row = [train_label, test_label, ds_label]
            for det in DETECTORS:
                key = (train_tag, test_attack, dataset, det)
                if key in raw:
                    d = raw[key]
                    row += [_mean_std(d["f1"]), _mean_std(d["dr"]), _mean_std(d["far"])]
                else:
                    row += ["—", "—", "—"]
            table_rows.append(row)

    return pd.DataFrame(table_rows, columns=col_headers)


def build_generalisation_latex(df: pd.DataFrame) -> str:
    """Render the generalisation table as a LaTeX booktabs table."""
    if df.empty:
        return r"% No generalisation data found."

    n_det = len(DETECTORS)
    col_spec = "lll" + ("rrr" * n_det)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Generalisation to unseen attacks."
        r"  Detectors are trained on a graph poisoned with {\em seen} attacks only;"
        r"  evaluated on a fresh graph poisoned solely by the {\em unseen} attack."
        r"  Mean~$\pm$~std over three seeds at injection rate 0.01."
        r"  \textbf{Bold} = best F1 per row.}",
        r"\label{tab:generalisation}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row 1: detector group spans
    det_headers = " & ".join(
        r"\multicolumn{3}{c}{" + DETECTOR_LABELS.get(d, d) + "}"
        for d in DETECTORS
    )
    lines.append(r"Train Attacks & Test Attack & Dataset & " + det_headers + r" \\")

    # Header row 2: F1 / DR / FAR per detector
    sub_headers = " & ".join(r"F1 & DR & FAR" for _ in DETECTORS)
    lines.append(r"\cmidrule(lr){4-" + str(3 + 3 * n_det) + "}")
    lines.append(r" &  &  & " + sub_headers + r" \\")
    lines.append(r"\midrule")

    f1_col_indices = [3 + 3 * i for i in range(n_det)]  # 0-indexed within each row

    prev_scenario = None
    for _, row in df.iterrows():
        cells = list(row.values)
        # Bold best F1 across detectors
        f1_cells  = [cells[i] for i in f1_col_indices]
        f1_bolded = _bold_best_latex(f1_cells, higher_is_better=True)
        for j, idx in enumerate(f1_col_indices):
            cells[idx] = f1_bolded[j]

        # Insert midrule between scenarios (keyed by "Train Attacks" column)
        scenario_key = (cells[0], cells[1])
        if scenario_key != prev_scenario:
            if prev_scenario is not None:
                lines.append(r"\midrule")
            prev_scenario = scenario_key

        lines.append(" & ".join(str(c) for c in cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def aggregate_generalisation(results_dir: Path, out_dir: Path) -> None:
    """Load generalisation JSONs, build CSV + LaTeX, print a summary."""
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading generalisation results …")
    df = load_generalisation_results(results_dir)
    print(f"  Loaded {len(df)} generalisation records.")

    if df.empty:
        print("  No generalisation data found — skipping.")
        return

    print("Building generalisation table …")
    tbl = build_generalisation(df)
    if not tbl.empty:
        tbl.to_csv(out_dir / "generalisation.csv", index=False)
        try:
            from tabulate import tabulate as _tabulate
            print(_tabulate(tbl, headers="keys", tablefmt="simple", showindex=False))
        except ImportError:
            print(tbl.to_string(index=False))

        latex_path = latex_dir / "generalisation.tex"
        latex_path.write_text(build_generalisation_latex(tbl))
        print(f"  Saved {out_dir}/generalisation.csv  and  {latex_path}")

    # ── Per-detector summary ───────────────────────────────────────────────
    print("\n  Generalisation F1 summary (mean ± std across seeds × datasets):")
    thgnn_rows = df[df["detector"] == "thgnn"]
    for (train_tag, test_attack), grp in df.groupby(["train_tag", "test_attack"]):
        train_label = _TRAIN_TAG_LABELS.get(train_tag, train_tag)
        test_label  = _TEST_ATTACK_LABELS.get(test_attack, test_attack)
        print(f"\n  [{train_label}  →  {test_label}]")
        for det in DETECTORS:
            det_grp = grp[grp["detector"] == det]["f1"].values
            if len(det_grp):
                lbl = DETECTOR_LABELS.get(det, det)
                print(f"    {lbl:20s}  F1={np.nanmean(det_grp):.4f} ± {np.nanstd(det_grp):.4f}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results found — nothing to summarise.")
        return

    thgnn = df[df["detector"] == "thgnn"]
    baselines = df[df["detector"] != "thgnn"]

    if thgnn.empty or baselines.empty:
        print("Insufficient data for summary (need both thgnn and baseline results).")
        return

    print("\n" + "=" * 70)
    print("SUMMARY: TH-GNN vs. Best Baseline")
    print("=" * 70)

    wins_by_dataset: Dict[str, Tuple[int, int]] = {}  # dataset → (wins, total)

    for dataset in DATASETS:
        ds_thgnn = thgnn[thgnn["dataset"] == dataset]
        ds_base  = baselines[baselines["dataset"] == dataset]
        if ds_thgnn.empty or ds_base.empty:
            continue

        # Group by (attack, rate, seed) and compare
        wins, total = 0, 0
        for (attack, rate, seed), grp_thgnn in ds_thgnn.groupby(["attack", "rate", "seed"]):
            grp_base = ds_base[
                (ds_base["attack"] == attack) &
                (ds_base["rate"]   == rate)   &
                (ds_base["seed"]   == seed)
            ]
            if grp_base.empty:
                continue
            thgnn_f1 = grp_thgnn["f1"].mean()
            best_base_f1 = grp_base.groupby("detector")["f1"].mean().max()
            total += 1
            if thgnn_f1 > best_base_f1:
                wins += 1

        wins_by_dataset[dataset] = (wins, total)
        pct = 100.0 * wins / total if total > 0 else 0.0
        print(f"  {DATASET_LABELS.get(dataset, dataset):20s}  TH-GNN wins {wins}/{total} comparisons ({pct:.1f}%)")

    # Average improvement in F1
    improvements = []
    for (attack, rate, seed, dataset), grp_thgnn in thgnn.groupby(["attack", "rate", "seed", "dataset"]):
        grp_base = baselines[
            (baselines["attack"]   == attack) &
            (baselines["rate"]     == rate)   &
            (baselines["seed"]     == seed)   &
            (baselines["dataset"]  == dataset)
        ]
        if grp_base.empty:
            continue
        thgnn_f1 = grp_thgnn["f1"].mean()
        best_base_f1 = grp_base.groupby("detector")["f1"].mean().max()
        improvements.append(thgnn_f1 - best_base_f1)

    if improvements:
        avg_imp = np.mean(improvements)
        print(f"\n  Average F1 improvement over best baseline: {avg_imp:+.4f}")
        wins_total = sum(w for w, _ in wins_by_dataset.values())
        all_total  = sum(t for _, t in wins_by_dataset.values())
        if all_total > 0:
            print(f"  Overall win rate: {wins_total}/{all_total} ({100*wins_total/all_total:.1f}%)")

    # Per-attack breakdown
    print("\n  Per-attack average F1:")
    for attack in ATTACKS:
        t_f1 = thgnn[thgnn["attack"] == attack]["f1"]
        b_f1 = baselines[baselines["attack"] == attack]
        if t_f1.empty or b_f1.empty:
            continue
        best_b = b_f1.groupby(["attack", "rate", "seed", "dataset", "detector"])["f1"].mean()
        best_b_by_row = b_f1.groupby(["attack", "rate", "seed", "dataset"])["f1"].max()
        avg_t = t_f1.mean()
        avg_b = best_b_by_row.mean() if not best_b_by_row.empty else np.nan
        print(f"    {ATTACK_LABELS.get(attack, attack):12s}  TH-GNN={avg_t:.3f}  best-baseline={avg_b:.3f}  Δ={avg_t-avg_b:+.3f}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results into tables.")
    parser.add_argument("--results-dir", default="results/raw",
                        help="Directory containing raw JSON result files (default: results/raw)")
    parser.add_argument("--out-dir",     default="results/tables",
                        help="Output directory for CSV and LaTeX tables (default: results/tables)")
    parser.add_argument("--dataset",     default=None,
                        help="Optional: restrict to a single dataset")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    latex_dir   = out_dir / "latex"

    if not results_dir.exists():
        print(f"[ERROR] results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} …")
    df = load_results(results_dir, dataset_filter=args.dataset)
    n_total = len(df)
    print(f"  Loaded {n_total} completed experiment records.")

    if n_total == 0:
        print("No completed results found. Run scripts/run_experiments.py first.")
        return

    # ------------------------------------------------------------------
    # Table 1: Main detection
    # ------------------------------------------------------------------
    print("Building main detection table …")
    t1 = build_main_detection(df)
    if not t1.empty:
        t1.to_csv(out_dir / "main_detection.csv", index=False)
        print(tabulate(t1, headers="keys", tablefmt="simple", showindex=False))
        (latex_dir / "main_detection.tex").write_text(build_main_detection_latex(t1))
        print(f"  Saved {out_dir}/main_detection.csv  and  {latex_dir}/main_detection.tex")

    # ------------------------------------------------------------------
    # Table 2: Downstream NDCG
    # ------------------------------------------------------------------
    print("\nBuilding downstream NDCG table …")
    t2 = build_downstream_ndcg(df)
    if not t2.empty:
        t2.to_csv(out_dir / "downstream_ndcg.csv", index=False)
        print(tabulate(t2, headers="keys", tablefmt="simple", showindex=False))
        (latex_dir / "downstream_ndcg.tex").write_text(build_downstream_ndcg_latex(t2))
        print(f"  Saved {out_dir}/downstream_ndcg.csv  and  {latex_dir}/downstream_ndcg.tex")

    # ------------------------------------------------------------------
    # Table 3: Budget curves (ML-1M)
    # ------------------------------------------------------------------
    print("\nBuilding budget curves table (ML-1M) …")
    t3 = build_budget_curves(df)
    if not t3.empty:
        t3.to_csv(out_dir / "budget_curves.csv", index=False)
        print(tabulate(t3, headers="keys", tablefmt="simple", showindex=False))
        (latex_dir / "budget_curves.tex").write_text(build_budget_curves_latex(t3))
        print(f"  Saved {out_dir}/budget_curves.csv  and  {latex_dir}/budget_curves.tex")
    else:
        print("  No ML-1M results found — budget curves table skipped.")

    # ------------------------------------------------------------------
    # Table 4: Ablation
    # ------------------------------------------------------------------
    print("\nBuilding ablation table …")
    t4 = build_ablation(df)
    if not t4.empty:
        t4.to_csv(out_dir / "ablation.csv", index=False)
        print(tabulate(t4, headers="keys", tablefmt="simple", showindex=False))
        (latex_dir / "ablation.tex").write_text(build_ablation_latex(t4))
        print(f"  Saved {out_dir}/ablation.csv  and  {latex_dir}/ablation.tex")
    else:
        (latex_dir / "ablation.tex").write_text(build_ablation_latex(t4))
        print("  No ablation results found — empty skeleton written.")

    # ------------------------------------------------------------------
    # Table 5: Generalisation
    # ------------------------------------------------------------------
    aggregate_generalisation(results_dir, out_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(df)


if __name__ == "__main__":
    main()
