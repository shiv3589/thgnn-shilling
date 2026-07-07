"""Generate paper/figures/budget_curves.pdf from real per-rate results.

Data source: results/tables_fixed/budget_curves.csv (derived from the real
33-configuration ML-1M experiment matrix; verified to match the per-attack
mean/std in results/tables_fixed/main_detection.csv).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RATES = [0.5, 1.0, 5.0]

# (attack, {detector: [F1 at 0.5%, 1%, 5%]})
DATA = {
    "Random":    {"TH-GNN": [0.016, 0.057, 0.976], "Statistical": [0.968, 1.000, 0.823], "Anti-FakeU": [0.938, 0.952, 0.921]},
    "Bandwagon": {"TH-GNN": [0.016, 0.055, 0.973], "Statistical": [0.906, 0.984, 0.743], "Anti-FakeU": [0.984, 0.976, 0.833]},
    "AUSH":      {"TH-GNN": [0.016, 0.058, 0.977], "Statistical": [0.395, 0.695, 0.580], "Anti-FakeU": [0.298, 0.247, 0.394]},
}

STYLE = {
    "TH-GNN":      dict(color="#5B4FCF", lw=2.5, marker="o", linestyle="-"),
    "Statistical": dict(color="#888888", lw=1.3, marker="s", linestyle="--"),
    "Anti-FakeU":  dict(color="#E8735A", lw=1.3, marker="^", linestyle="--"),
}

fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)

for ax, attack in zip(axes, ["Random", "Bandwagon", "AUSH"]):
    for detector, f1s in DATA[attack].items():
        ax.plot(RATES, f1s, label=detector, **STYLE[detector])
    ax.set_title(attack, fontsize=9)
    ax.set_xlabel("Injection rate (%)", fontsize=8)
    ax.set_xticks(RATES)
    ax.set_xticklabels(["0.5", "1", "5"])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, linewidth=0.5)

axes[0].set_ylabel(r"Detection $F_1$", fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08),
           frameon=False, fontsize=8)

fig.tight_layout(rect=(0, 0, 1, 0.92))

out_dir = Path(__file__).parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "budget_curves.pdf"
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
print(f"[PDF] {pdf_path}")
plt.close(fig)
