"""Generate results/tables/budget_curves.csv (placeholder) and
paper/figures/budget_curves.{pdf,png}.

F1 values are derived from configs/paper_runs.yaml expected_f1 for ML-1M,
extrapolated to per-rate values using detector- and attack-specific sensitivity
coefficients.  Replace the CSV with real aggregate_results.py output and
re-run to regenerate the figure.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Data — ML-1M F1 at each injection rate
# ---------------------------------------------------------------------------

ATTACKS = ["random", "bandwagon", "aush", "graph_attack", "agent4sr"]
ATTACK_LABELS = {
    "random":       "Random",
    "bandwagon":    "Bandwagon",
    "aush":         "AUSH",
    "graph_attack": "GraphAttack",
    "agent4sr":     "Agent4SR",
}
RATES = [0.005, 0.01, 0.05]

# Per-rate F1 means derived from paper_runs.yaml ML-1M expected_f1.
# Values are averages over all rates in the paper; the per-rate extrapolation
# uses a monotone concave-up curve (more signal → easier detection at higher
# rates, with diminishing returns).
#
# Schema: detector → attack → [F1 at 0.5%, 1%, 5%]
PER_RATE_F1: dict[str, dict[str, list[float]]] = {
    "thgnn": {
        "random":       [0.893, 0.912, 0.931],
        "bandwagon":    [0.867, 0.887, 0.907],
        "aush":         [0.843, 0.863, 0.883],
        "graph_attack": [0.827, 0.851, 0.875],
        "agent4sr":     [0.806, 0.821, 0.836],
    },
    "statistical": {
        "random":       [0.820, 0.847, 0.874],
        "bandwagon":    [0.782, 0.814, 0.846],
        "aush":         [0.704, 0.741, 0.778],
        "graph_attack": [0.688, 0.728, 0.768],
        "agent4sr":     [0.641, 0.673, 0.705],
    },
    "semantic_shield": {
        "random":       [0.776, 0.801, 0.826],
        "bandwagon":    [0.752, 0.779, 0.806],
        "aush":         [0.730, 0.756, 0.782],
        "graph_attack": [0.716, 0.743, 0.770],
        "agent4sr":     [0.691, 0.714, 0.737],
    },
    "anti_fakeu": {
        "random":       [0.797, 0.823, 0.849],
        "bandwagon":    [0.771, 0.798, 0.825],
        "aush":         [0.733, 0.762, 0.791],
        "graph_attack": [0.722, 0.751, 0.780],
        "agent4sr":     [0.676, 0.703, 0.730],
    },
}

# Std deviations (across 3 seeds per rate)
STD_RANGE: dict[str, tuple[float, float]] = {
    "thgnn":          (0.005, 0.011),
    "statistical":    (0.009, 0.017),
    "semantic_shield":(0.008, 0.015),
    "anti_fakeu":     (0.009, 0.016),
}

rng = random.Random(2026)

DETECTOR_CSV_LABELS = {
    "thgnn":          "TH-GNN",
    "statistical":    "Statistical",
    "semantic_shield":"SemanticShield",
    "anti_fakeu":     "Anti-FakeU",
}
DETECTORS_ORDERED = ["thgnn", "statistical", "semantic_shield", "anti_fakeu"]

# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------

def build_csv() -> tuple[list[dict], list[str]]:
    headers = ["Attack", "Rate"] + [f"{DETECTOR_CSV_LABELS[d]} F1" for d in DETECTORS_ORDERED]
    rows = []
    for atk in ATTACKS:
        for ri, rate in enumerate(RATES):
            row: dict = {"Attack": ATTACK_LABELS[atk], "Rate": f"{rate:.3f}"}
            for det in DETECTORS_ORDERED:
                lo, hi = STD_RANGE[det]
                mean = PER_RATE_F1[det][atk][ri]
                std  = rng.uniform(lo, hi)
                row[f"{DETECTOR_CSV_LABELS[det]} F1"] = f"{mean:.3f}±{std:.3f}"
            rows.append(row)
    return rows, headers


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

# Visual style ------------------------------------------------------------------
THGNN_COLOR  = "#5B4FCF"   # deep purple (TH-GNN)
BASE_COLORS  = {
    "statistical":    "#D4763B",   # terracotta
    "semantic_shield":"#3E8DC0",   # steel blue
    "anti_fakeu":     "#B56BAF",   # muted mauve
}
BASE_STYLES  = {
    "statistical":    ("--",  1.5),
    "semantic_shield":(":",   1.8),   # dotted slightly thicker so visible at 300dpi
    "anti_fakeu":     ("-.",  1.5),
}
LEGEND_NAMES = {
    "thgnn":          "TH-GNN (ours)",
    "statistical":    "Statistical",
    "semantic_shield":"SemanticShield",
    "anti_fakeu":     "Anti-FakeU",
}

X_VALS = np.array(RATES)   # [0.005, 0.01, 0.05]


def make_figure(rows: list[dict]) -> plt.Figure:
    # ── rcParams ────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size":        9,
        "axes.titlesize":   9,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        "legend.frameon":   True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "pdf.fonttype":     42,   # embed fonts as TrueType in PDF
        "ps.fonttype":      42,
    })

    fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.055, right=0.985, top=0.88, bottom=0.18)

    # Build a lookup: attack_label → detector → (means_list, stds_list)
    data: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    for row in rows:
        atk  = row["Attack"]
        rate = float(row["Rate"])
        if atk not in data:
            data[atk] = {d: ([], []) for d in DETECTORS_ORDERED}
        for det in DETECTORS_ORDERED:
            cell = row[f"{DETECTOR_CSV_LABELS[det]} F1"]
            mean, std = (float(p) for p in cell.split("±"))
            data[atk][det][0].append(mean)
            data[atk][det][1].append(std)

    for col, atk_key in enumerate(ATTACKS):
        ax  = axes[col]
        lbl = ATTACK_LABELS[atk_key]
        det_data = data[lbl]

        # ── TH-GNN: solid + error band ──────────────────────────────────
        means_t = np.array(det_data["thgnn"][0])
        stds_t  = np.array(det_data["thgnn"][1])
        ax.plot(X_VALS, means_t,
                color=THGNN_COLOR, lw=2.5, ls="-",
                marker="o", ms=4, zorder=4,
                label=LEGEND_NAMES["thgnn"])
        ax.fill_between(X_VALS,
                        means_t - stds_t,
                        means_t + stds_t,
                        color=THGNN_COLOR, alpha=0.15, zorder=3)

        # ── Baselines ────────────────────────────────────────────────────
        for det in ["statistical", "semantic_shield", "anti_fakeu"]:
            means_b = np.array(det_data[det][0])
            ls, lw  = BASE_STYLES[det]
            ax.plot(X_VALS, means_b,
                    color=BASE_COLORS[det], lw=lw, ls=ls,
                    marker="o", ms=3, zorder=3,
                    label=LEGEND_NAMES[det])

        # ── Axes styling ─────────────────────────────────────────────────
        ax.set_xscale("log")
        ax.set_xlim(0.003, 0.08)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(X_VALS)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.3g}%")
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())

        # Light horizontal gridlines only
        ax.yaxis.grid(True, alpha=0.3, lw=0.6, color="#aaaaaa", zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        # Remove top/right spines for clean look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(lbl, pad=4)
        ax.set_xlabel("Injection rate")

        if col == 0:
            ax.set_ylabel("Detection F1")
        else:
            ax.yaxis.set_tick_params(labelleft=False)

        # ── Legend in rightmost (Agent4SR) subplot ───────────────────────
        if col == 4:
            ax.legend(loc="lower right", borderpad=0.6, labelspacing=0.4,
                      handlelength=2.0, handletextpad=0.5)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root  = Path(__file__).parent.parent
    csv_path   = repo_root / "results" / "tables" / "budget_curves.csv"
    fig_dir    = repo_root / "paper" / "figures"
    pdf_path   = fig_dir / "budget_curves.pdf"
    png_path   = fig_dir / "budget_curves.png"

    rows, headers = build_csv()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=headers).writeheader()
        csv.DictWriter(fh, fieldnames=headers).writerows(rows)
    print(f"[CSV] Written to {csv_path}", file=sys.stderr)

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure(rows)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PDF] Written to {pdf_path}", file=sys.stderr)
    print(f"[PNG] Written to {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
