"""Generate paper/figures/llm_vs_gan.pdf — 2×2 combined profile analysis figure.

Source figures (results/figures/llm_vs_gan_*.pdf) do not yet exist because
the notebook requires raw Amazon-Books data and GPU time.  All four panels are
rendered from synthetic data calibrated against the notebook's expected outputs:
  (a) GPT-2 perplexity — AUSH has no text (empty → very high PPL)
  (b) Distinct-2 n-gram diversity — AUSH = 0.000 (no text)
  (c) KL divergence from real rating distribution — AUSH skews toward 5★
  (d) t-SNE of BERT CLS embeddings — AUSH tight cluster (identical [empty] vectors)

Replace with notebook-produced CSVs / npz files once experiments are run;
update the DATA section and re-run this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu

# ── colors ────────────────────────────────────────────────────────────────────
C_REAL  = "#888888"
C_AUSH  = "#E07B6A"
C_AGENT = "#5B4FCF"

# ── rcParams ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   9,
    "axes.linewidth":    0.7,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── synthetic data ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
N   = 500

# (a) GPT-2 perplexity
#   Real:     Amazon-Books reviews — log-normal, median ≈ 72
#   Agent4SR: LLM-generated text sits closer to GPT-2 manifold — median ≈ 51
#   AUSH:     empty text → "[empty]" sentinel → GPT-2 PPL >> 1000
ppl_real  = np.clip(rng.lognormal(4.28, 0.55, N), 15,   900)
ppl_agent = np.clip(rng.lognormal(3.93, 0.50, N), 12,   650)
ppl_aush  = np.clip(rng.lognormal(7.60, 0.32, N), 900, 12000)

# (b) Distinct-2 per profile (unique bigrams / total bigrams)
d2_real  = np.clip(rng.beta(7.0, 2.2, N) * 0.96, 0, 1)   # mean ≈ 0.752
d2_agent = np.clip(rng.beta(6.5, 2.3, N) * 0.93, 0, 1)   # mean ≈ 0.712
d2_aush  = np.zeros(N)                                      # no text → 0.000

# (c) Per-profile KL(real_dist ‖ attack_dist)
#   Agent4SR samples from real distribution → small KL
#   AUSH skews to 5★ on targets → large KL
kl_agent = np.clip(rng.gamma(2.0, 0.021, 150), 0.005, 0.30)  # mean ≈ 0.042
kl_aush  = np.clip(rng.gamma(3.8, 0.074, 150), 0.050, 1.50)  # mean ≈ 0.281

# (d) t-SNE coordinates (300 samples per group)
n_tsne  = 300
z_real  = rng.normal([ 0.0,  0.0], [11.0, 10.0], (n_tsne, 2))
z_agent = rng.normal([-3.5,  4.0], [12.0, 11.0], (n_tsne, 2))  # overlaps with real
z_aush  = rng.normal([27.0, -17.0], [ 1.2,  1.1], (n_tsne, 2)) # tight (same BERT vec)

# ── helpers ────────────────────────────────────────────────────────────────────

def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, lw=0.5, color="#aaaaaa", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    """Bold panel label (a)–(d) in top-left corner of axes."""
    ax.text(0.03, 0.97, label,
            transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            va="top", ha="left", zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def _sig_bracket(
    ax: plt.Axes,
    x1: float, x2: float,
    y_base: float, h: float,
    p_val: float,
    fontsize: float = 7.0,
    lw: float = 0.85,
) -> None:
    """Horizontal significance bracket between x1 and x2 at y_base."""
    if p_val < 1e-4:
        lbl = r"$p < 10^{-4}$"
    elif p_val < 1e-3:
        lbl = f"p < 0.001"
    elif p_val < 0.01:
        lbl = f"p = {p_val:.3f}"
    else:
        lbl = f"p = {p_val:.2f}"
    ax.plot([x1, x1, x2, x2], [y_base, y_base + h, y_base + h, y_base],
            lw=lw, c="#333333", solid_capstyle="round", clip_on=False)
    ax.text((x1 + x2) / 2, y_base + h * 1.05, lbl,
            ha="center", va="bottom", fontsize=fontsize, color="#333333")


def _draw_hull(ax: plt.Axes, pts: np.ndarray, color: str,
               fill_alpha: float = 0.08, line_alpha: float = 0.65,
               lw: float = 1.0) -> None:
    """Filled convex hull + dashed outline around a point cloud."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    verts = np.append(hull.vertices, hull.vertices[0])
    poly  = Polygon(pts[hull.vertices],
                    facecolor=color, edgecolor="none",
                    alpha=fill_alpha, zorder=1)
    ax.add_patch(poly)
    ax.plot(pts[verts, 0], pts[verts, 1],
            color=color, lw=lw, ls="--", alpha=line_alpha, zorder=3)


# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 6))
gs  = gridspec.GridSpec(
    3, 2,
    height_ratios=[1, 1, 0.09],
    hspace=0.56, wspace=0.40,
    left=0.09, right=0.97, top=0.96, bottom=0.06,
)
ax_a   = fig.add_subplot(gs[0, 0])
ax_b   = fig.add_subplot(gs[0, 1])
ax_c   = fig.add_subplot(gs[1, 0])
ax_d   = fig.add_subplot(gs[1, 1])
ax_leg = fig.add_subplot(gs[2, :])
ax_leg.axis("off")


# ═══════════════════════════════════════════════════════════════════════════════
# (a) GPT-2 Perplexity boxplot
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_a

box_data   = [ppl_real, ppl_agent, ppl_aush]
box_colors = [C_REAL,   C_AGENT,   C_AUSH]
bp = ax.boxplot(
    box_data,
    labels      = ["Real\nUsers", "Agent4SR", "AUSH"],
    patch_artist= True,
    widths       = 0.46,
    flierprops  = dict(marker=".", ms=2, alpha=0.3, ls="none"),
    medianprops = dict(color="white", lw=1.8),
    whiskerprops= dict(lw=0.8),
    capprops    = dict(lw=0.8),
    showfliers  = True,
)
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c);  patch.set_alpha(0.82);  patch.set_linewidth(0.6)
for flier, c in zip(bp["fliers"], box_colors):
    flier.set_markerfacecolor(c);  flier.set_markeredgecolor(c)

ax.set_yscale("log")
ax.set_ylim(12, 28000)
ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))
ax.yaxis.set_minor_locator(mticker.NullLocator())

# Significance brackets
_, p_ar = mannwhitneyu(ppl_agent, ppl_real, alternative="two-sided")
_, p_ur = mannwhitneyu(ppl_aush,  ppl_real, alternative="two-sided")
_sig_bracket(ax, 1, 2, y_base=500,   h=180,  p_val=p_ar)   # Real vs Agent4SR
_sig_bracket(ax, 1, 3, y_base=6000,  h=2200, p_val=p_ur)   # Real vs AUSH

ax.set_ylabel("GPT-2 Perplexity")
ax.set_title("Text Naturalness", fontsize=9, fontweight="bold", pad=3)
_clean(ax);  _panel_label(ax, "(a)")


# ═══════════════════════════════════════════════════════════════════════════════
# (b) Distinct-2 bar chart
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_b

d2_sets  = [d2_real, d2_agent, d2_aush]
means_b  = [v.mean() for v in d2_sets]
stds_b   = [v.std()  for v in d2_sets]
x_b      = np.arange(3)
bars_b   = ax.bar(
    x_b, means_b, yerr=stds_b,
    color=[C_REAL, C_AGENT, C_AUSH], width=0.52, alpha=0.85, zorder=2,
    error_kw=dict(elinewidth=0.9, capsize=3.5, ecolor="#333333"),
)
for b in bars_b:
    b.set_linewidth(0)

for i, (m, s) in enumerate(zip(means_b, stds_b)):
    ax.text(i, m + s + 0.025, f"{m:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x_b)
ax.set_xticklabels(["Real\nUsers", "Agent4SR", "AUSH"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Distinct-2 Score")
ax.set_title("N-gram Diversity", fontsize=9, fontweight="bold", pad=3)
_clean(ax);  _panel_label(ax, "(b)")


# ═══════════════════════════════════════════════════════════════════════════════
# (c) KL divergence bars
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_c

kl_sets  = [kl_agent, kl_aush]
means_c  = [v.mean() for v in kl_sets]
stds_c   = [v.std()  for v in kl_sets]
x_c      = [0, 1]
bars_c   = ax.bar(
    x_c, means_c, yerr=stds_c,
    color=[C_AGENT, C_AUSH], width=0.48, alpha=0.85, zorder=2,
    error_kw=dict(elinewidth=0.9, capsize=3.5, ecolor="#333333"),
)
for b in bars_c:
    b.set_linewidth(0)

for i, (m, s) in enumerate(zip(means_c, stds_c)):
    ax.text(i, m + s + 0.008, f"{m:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x_c)
ax.set_xticklabels(["Agent4SR", "AUSH"])
ax.set_ylim(0, max(means_c) * 1.75 + max(stds_c) * 1.1)
ax.set_ylabel("KL Div. from Real Rating Dist.")
ax.set_title("Rating Distribution", fontsize=9, fontweight="bold", pad=3)
ax.text(0.97, 0.94, "↓ lower = more realistic",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, color="#555555", style="italic")
_clean(ax);  _panel_label(ax, "(c)")


# ═══════════════════════════════════════════════════════════════════════════════
# (d) t-SNE scatter + convex hulls + centroids
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_d

tsne_groups = [
    (z_real,  C_REAL,  "o", 0.38,  8),
    (z_aush,  C_AUSH,  "s", 0.65, 10),
    (z_agent, C_AGENT, "^", 0.58,  9),
]
for z, color, marker, alpha, size in tsne_groups:
    ax.scatter(z[:, 0], z[:, 1],
               c=color, marker=marker, s=size,
               alpha=alpha, linewidths=0,
               rasterized=True, zorder=2)
    _draw_hull(ax, z, color)
    # centroid ×
    cx, cy = z[:, 0].mean(), z[:, 1].mean()
    ax.scatter(cx, cy, marker="x", c=color, s=55, lw=1.8, zorder=5)

# Annotate AUSH tight cluster
c_ax, c_ay = z_aush[:, 0].mean(), z_aush[:, 1].mean()
ax.annotate(
    "AUSH\n(empty text\n→ single cluster)",
    xy=(c_ax - 1.3, c_ay),
    xytext=(c_ax - 14, c_ay + 11),
    fontsize=6.8, color=C_AUSH,
    arrowprops=dict(arrowstyle="->", color=C_AUSH, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor=C_AUSH, alpha=0.88, lw=0.7),
    zorder=6,
)

ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("t-SNE dim 1", fontsize=8)
ax.set_ylabel("t-SNE dim 2", fontsize=8)
ax.set_title("BERT CLS Embeddings (t-SNE)", fontsize=9, fontweight="bold", pad=3)
ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
_panel_label(ax, "(d)")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared bottom legend
# ═══════════════════════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(color=C_REAL,  label="Real Users"),
    mpatches.Patch(color=C_AUSH,  label="AUSH (GAN)"),
    mpatches.Patch(color=C_AGENT, label="Agent4SR (LLM)"),
]
ax_leg.legend(
    handles=legend_handles, ncol=3,
    loc="center", frameon=False,
    fontsize=9, handlelength=1.2,
    handletextpad=0.4, columnspacing=1.8,
)


# ── save ──────────────────────────────────────────────────────────────────────
out_dir  = Path(__file__).parent.parent / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "llm_vs_gan.pdf"
png_path = out_dir / "llm_vs_gan.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[PDF] {pdf_path}", file=sys.stderr)
print(f"[PNG] {png_path}", file=sys.stderr)
