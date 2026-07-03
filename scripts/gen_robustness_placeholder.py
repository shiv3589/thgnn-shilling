"""Placeholder generator for results/figures/robustness_curves.{pdf,png}.

results/tables/robustness.csv does not exist yet (scripts/run_robustness.py has not
been run). This script produces a stub figure matching the expected layout
(3 subplots: Missing Text / Noisy Timestamps / Missing Edges) with flat
placeholder lines so downstream LaTeX compilation is unblocked. Re-run
scripts/run_robustness.py and its aggregate_and_plot() step to replace this
with real data.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITION_TYPES = ["missing_text", "noisy_timestamps", "missing_edges"]
CONDITION_LABELS = {
    "missing_text":     "Missing Text",
    "noisy_timestamps": "Noisy Timestamps",
    "missing_edges":    "Missing Edges",
}
LEVELS = [0.2, 0.5]

STYLES = {
    "thgnn":                    dict(color="#1f77b4", ls="-",  marker="o", lw=2.0, label="TH-GNN",         value=0.85),
    "semantic_shield_baseline": dict(color="#ff7f0e", ls="--", marker="s", lw=1.5, label="SemanticShield", value=0.72),
    "anti_fakeu_baseline":      dict(color="#2ca02c", ls=":",  marker="^", lw=1.5, label="Anti-FakeU",     value=0.68),
}


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.suptitle(
        "TH-GNN robustness under missing modalities\n"
        "(Amazon-Books, Agent4SR, injection rate 0.01) — PLACEHOLDER",
        fontsize=11,
    )

    for ax, cond_type in zip(axes, CONDITION_TYPES):
        ax.set_title(CONDITION_LABELS[cond_type], fontsize=10)
        ax.set_xlabel("Degradation level", fontsize=9)
        ax.set_ylabel("F1", fontsize=9)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels(["20%", "50%"], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        for det, sty in STYLES.items():
            ys = [sty["value"], sty["value"]]
            ax.plot(LEVELS, ys, color=sty["color"], ls=sty["ls"],
                     marker=sty["marker"], lw=sty["lw"], label=sty["label"])

        ax.legend(fontsize=8, loc="lower left")

        ax.text(0.5, 0.5, "PLACEHOLDER — replace after run_robustness.py completes",
                transform=ax.transAxes, fontsize=9, color="red", alpha=0.4,
                ha="center", va="center", rotation=30)

    fig.tight_layout()

    out_dir = Path(__file__).parent.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig_path = out_dir / f"robustness_curves.{ext}"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
