"""Generate paper/figures/architecture.pdf — TH-GNN pipeline diagram.

Layout (left → right):
  [Input G_t]  →  [HGT ×2]  →  [3 streams + GRU → Cross-modal Attn → cat]  →  [MLP → score → Filter → LightGCN]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────
C_PUR   = "#5B4FCF"   # purple  — graph / structural stream
C_TEAL  = "#2A9D8F"   # teal    — text streams
C_ORG   = "#E07840"   # orange  — temporal / burstiness stream
C_DARK  = "#1E1E2E"   # near-black text
C_MED   = "#555577"   # medium  text (secondary)
C_BG_A  = "#F0F0F5"   # input block bg
C_BG_B  = "#EEF0FB"   # encoder block bg
C_BG_C  = "#FDF0EB"   # output block bg
C_LINE  = "#AAAACC"   # block borders

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":      8,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
})

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.set_xlim(0, 7)
ax.set_ylim(0, 3.5)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0.18)

# ── helpers ───────────────────────────────────────────────────────────────────

def rbox(x, y, w, h, fc, ec=C_LINE, lw=0.9, alpha=1.0, zorder=2, pad=0.07):
    """Rounded-corner rectangle; (x,y) is bottom-left, (w,h) is size."""
    p = FancyBboxPatch(
        (x + pad, y + pad), w - 2*pad, h - 2*pad,
        boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)
    return p


def txt(x, y, s, fs=8, color=C_DARK, ha="center", va="center",
        bold=False, zorder=6, wrap=False):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            zorder=zorder, linespacing=1.3)


def arrow(x0, y0, x1, y1, color="#888888", lw=0.9,
          style="->", zorder=4, shrink=2):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            shrinkA=shrink, shrinkB=shrink,
        ),
        zorder=zorder,
    )


def dim_label(x, y, text, color):
    """Small dimension annotation (e.g. '128-dim')."""
    ax.text(x, y, text, fontsize=6.5, color=color,
            ha="center", va="center", style="italic", zorder=7)


def node_icon(x, y, shape, color, size=0.14):
    """Draw user (circle), item (square), or review (rectangle) icon."""
    if shape == "circle":
        p = Circle((x, y), size * 0.95, fc=color, ec="white", lw=0.8, zorder=5)
        ax.add_patch(p)
    elif shape == "square":
        s = size * 1.6
        p = FancyBboxPatch((x - s/2, y - s/2), s, s,
                           boxstyle="round,pad=0.01",
                           fc=color, ec="white", lw=0.8, zorder=5)
        ax.add_patch(p)
    elif shape == "diamond":
        s = size * 1.7
        verts = np.array([[x, y+s], [x+s, y], [x, y-s], [x-s, y]])
        p = Polygon(verts, fc=color, ec="white", lw=0.8, zorder=5)
        ax.add_patch(p)
    elif shape == "rectangle":
        w = size * 2.2
        h = size * 1.1
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.01",
                           fc=color, ec="white", lw=0.8, zorder=5)
        ax.add_patch(p)


# ════════════════════════════════════════════════════════════════════════════
# BLOCK A — Input Graph G_t
# ════════════════════════════════════════════════════════════════════════════
AX, AY, AW, AH = 0.07, 0.18, 1.28, 3.14
rbox(AX, AY, AW, AH, fc=C_BG_A, ec=C_LINE, lw=1.0, zorder=1)
txt(AX + AW/2, AY + AH - 0.22, "Input Graph $G_t$", bold=True, fs=8)

# Node type icons — centred in the box, labels rendered inside each shape
icon_x = AX + AW / 2   # 0.71: centres all shapes (incl. rectangle) within the box
_NODE_SIZE = 0.18
_node_data = [
    (2.72, "circle",    C_PUR,  "User"),
    (2.05, "square",    C_TEAL, "Item"),
    (1.38, "diamond",   C_ORG,  "Review"),
]
for iy, shape, color, label_t in _node_data:
    node_icon(icon_x, iy, shape, color, size=_NODE_SIZE)
    txt(icon_x, iy, label_t, ha="center", fs=5.5, color="white", bold=True)

# Edge arrows between shapes — endpoints computed from exact shape edges
# circle bottom→square top, square bottom→rectangle top
_arrow_pairs = [
    (2.72 - _NODE_SIZE * 0.95,      2.05 + _NODE_SIZE * 1.6 / 2),
    (2.05 - _NODE_SIZE * 1.6 / 2,   1.38 + _NODE_SIZE * 1.7),
]
for y_s, y_t in _arrow_pairs:
    ax.annotate("", xy=(icon_x, y_t), xytext=(icon_x, y_s),
                arrowprops=dict(arrowstyle="-|>", color="#AAAAAA",
                                lw=0.7, mutation_scale=6, shrinkA=3, shrinkB=3),
                zorder=3)

# Timestamped edge note
txt(AX + AW/2, AY + 0.52, "timestamped edges", fs=7, color=C_MED)

# ════════════════════════════════════════════════════════════════════════════
# BLOCK B — TH-GNN Encoder  (the big middle block)
# ════════════════════════════════════════════════════════════════════════════
BX, BY, BW, BH = 1.52, 0.18, 4.10, 3.14
rbox(BX, BY, BW, BH, fc=C_BG_B, ec=C_LINE, lw=1.0, zorder=1)
txt(BX + BW/2, BY + BH - 0.22, "TH-GNN Encoder", bold=True, fs=8)

# ── B1: HGT sub-block ────────────────────────────────────────────────────────
B1X, B1Y, B1W, B1H = 1.63, 0.32, 1.40, 2.80
rbox(B1X, B1Y, B1W, B1H,
     fc="white", ec=C_PUR, lw=1.1, zorder=2, pad=0.05)
txt(B1X + B1W/2, B1Y + B1H - 0.20, "HGT Encoder", bold=True, fs=7.5,
    color=C_PUR)

# HGT layer 2 (top) — shifted +0.08 upward to open gap for sinusoidal label
rbox(B1X + 0.08, B1Y + 1.70, B1W - 0.16, 0.85,
     fc="#EBE9FA", ec=C_PUR, lw=0.8, zorder=3, pad=0.04)
txt(B1X + B1W/2, B1Y + 2.23, "HGT Layer 2", fs=7.5, bold=True, color=C_PUR)
txt(B1X + B1W/2, B1Y + 1.96, "per-type W_Q/K/V", fs=6, color=C_MED)
txt(B1X + B1W/2, B1Y + 1.78, "per-rel W_att/W_msg", fs=6, color=C_MED)

# HGT layer 1 (below)
rbox(B1X + 0.08, B1Y + 0.68, B1W - 0.16, 0.85,
     fc="#EBE9FA", ec=C_PUR, lw=0.8, zorder=3, pad=0.04)
txt(B1X + B1W/2, B1Y + 1.22, "HGT Layer 1", fs=7.5, bold=True, color=C_PUR)
txt(B1X + B1W/2, B1Y + 0.95, "per-type W_Q/K/V", fs=6, color=C_MED)
txt(B1X + B1W/2, B1Y + 0.77, "per-rel W_att/W_msg", fs=6, color=C_MED)

# Temporal encoding label in the gap between Layer 1 top and Layer 2 bottom
txt(B1X + B1W/2, B1Y + 1.615, "+ sinusoidal Δt bias",
    fs=6.2, color=C_ORG)

# Bottom: temporal encoding tag
rbox(B1X + 0.08, B1Y + 0.10, B1W - 0.16, 0.45,
     fc="#FEF0E5", ec=C_ORG, lw=0.8, zorder=3, pad=0.04)
txt(B1X + B1W/2, B1Y + 0.33, "Temporal Encoding", fs=6.8, bold=True, color=C_ORG)
txt(B1X + B1W/2, B1Y + 0.18, "TemporalEncoding(64-dim)", fs=6, color=C_MED)

# Upward arrow through HGT layers
arrow(B1X + B1W/2, B1Y + 0.58, B1X + B1W/2, B1Y + 0.66,
      color=C_PUR, lw=0.8, shrink=1)
arrow(B1X + B1W/2, B1Y + 1.55, B1X + B1W/2, B1Y + 1.67,
      color=C_PUR, lw=0.8, shrink=1)


# ── Horizontal arrow: HGT → streams ─────────────────────────────────────────
arrow(B1X + B1W, BY + BH/2, B1X + B1W + 0.20, BY + BH/2,
      color=C_PUR, lw=1.0, style="-|>", shrink=2)

# ── B2: Three parallel streams ────────────────────────────────────────────────
# Stream start x and fusion box x
SX   = B1X + B1W + 0.24   # x start of stream boxes
FX   = 4.60                # x start of fusion box
FY   = 1.15
FW   = 0.82
FH   = 1.90

# y-centers for the 3 streams + GRU
Y_S1 = 2.70   # stream 1: HGT user node
Y_S2 = 2.05   # stream 2: review text
Y_S3 = 1.35   # stream 3: item text
Y_GR = 0.60   # GRU

SBH  = 0.40   # stream box height
SBW1 = 0.80   # width of stream 1 box (just "HGT user node")
SBW2 = 1.10   # width of stream 2/3 boxes (includes RoBERTa step)

# Stream 1 — HGT user node output (purple)
rbox(SX, Y_S1 - SBH/2, SBW1, SBH,
     fc="#EBE9FA", ec=C_PUR, lw=0.9, zorder=3, pad=0.04)
txt(SX + SBW1/2, Y_S1 + 0.07, "HGT user node", fs=7, color=C_PUR, bold=True)
txt(SX + SBW1/2, Y_S1 - 0.10, "128-dim", fs=6.2, color=C_PUR, bold=False)

# Stream 2 — Review text → RoBERTa (teal)
rbox(SX, Y_S2 - SBH/2, 0.45, SBH,
     fc="#E3F4F1", ec=C_TEAL, lw=0.9, zorder=3, pad=0.04)
txt(SX + 0.225, Y_S2 + 0.07, "review", fs=7, color=C_TEAL, bold=True)
txt(SX + 0.225, Y_S2 - 0.10, "texts", fs=7, color=C_TEAL)

rbox(SX + 0.55, Y_S2 - SBH/2, 0.55, SBH,
     fc="#E3F4F1", ec=C_TEAL, lw=0.9, zorder=3, pad=0.04)
txt(SX + 0.55 + 0.275, Y_S2 + 0.07, "RoBERTa", fs=7, color=C_TEAL, bold=True)
txt(SX + 0.55 + 0.275, Y_S2 - 0.10, "(frozen) 768", fs=5, color=C_TEAL)

# Stream 3 — Item desc. → RoBERTa (teal)
rbox(SX, Y_S3 - SBH/2, 0.45, SBH,
     fc="#E3F4F1", ec=C_TEAL, lw=0.9, zorder=3, pad=0.04)
txt(SX + 0.225, Y_S3 + 0.07, "item", fs=7, color=C_TEAL, bold=True)
txt(SX + 0.225, Y_S3 - 0.10, "desc.", fs=7, color=C_TEAL)

rbox(SX + 0.55, Y_S3 - SBH/2, 0.55, SBH,
     fc="#E3F4F1", ec=C_TEAL, lw=0.9, zorder=3, pad=0.04)
txt(SX + 0.55 + 0.275, Y_S3 + 0.07, "RoBERTa", fs=7, color=C_TEAL, bold=True)
txt(SX + 0.55 + 0.275, Y_S3 - 0.10, "(frozen)  768", fs=6, color=C_TEAL)

# Arrows within stream 2 and 3 (text → RoBERTa)
for ys in [Y_S2, Y_S3]:
    arrow(SX + 0.45, ys, SX + 0.55, ys,
          color=C_TEAL, lw=0.8, style="-|>", shrink=1)

# ── GRU side branch (orange) ─────────────────────────────────────────────────
rbox(SX, Y_GR - 0.20, 0.45, 0.40,
     fc="#FEF0E5", ec=C_ORG, lw=0.9, zorder=3, pad=0.04)
txt(SX + 0.225, Y_GR + 0.05, "timestamps", fs=5.5, color=C_ORG, bold=False)
txt(SX + 0.225, Y_GR - 0.10, "log1p Δt", fs=6, color=C_ORG)

rbox(SX + 0.55, Y_GR - 0.20, 0.65, 0.40,
     fc="#FEF0E5", ec=C_ORG, lw=0.9, zorder=3, pad=0.04)
txt(SX + 0.55 + 0.325, Y_GR + 0.05, "Burstiness GRU", fs=5.2, color=C_ORG, bold=True)
txt(SX + 0.55 + 0.325, Y_GR - 0.10, "32-dim burst", fs=6, color=C_ORG)

arrow(SX + 0.45, Y_GR, SX + 0.55, Y_GR,
      color=C_ORG, lw=0.8, style="-|>", shrink=1)

# ── Fusion box ────────────────────────────────────────────────────────────────
rbox(FX, FY, FW, FH,
     fc="white", ec="#8877DD", lw=1.2, zorder=3, pad=0.05)
txt(FX + FW/2, FY + FH - 0.28, "Cross-modal", fs=7.5, bold=True, color="#5544BB")
txt(FX + FW/2, FY + FH - 0.52, "Attention", fs=7.5, bold=True, color="#5544BB")

# Internals of fusion box
txt(FX + FW/2, FY + FH*0.58,
    "q = subgraph\nk/v = text\nstreams",
    fs=6, color=C_MED)
txt(FX + FW/2, FY + 0.25, "Add-Norm + FF", fs=6, color=C_MED)
txt(FX + FW/2, FY + 0.12, "→ 128-dim", fs=6, color="#5544BB")

# Horizontal arrows: streams → fusion
arrow(SX + SBW1, Y_S1, FX, Y_S1,      color=C_PUR, lw=0.9, style="-|>", shrink=2)
arrow(SX + SBW2, Y_S2, FX, Y_S2,      color=C_TEAL, lw=0.9, style="-|>", shrink=2)
arrow(SX + SBW2, Y_S3, FX, Y_S3,      color=C_TEAL, lw=0.9, style="-|>", shrink=2)

# GRU → fusion (angled upward)
arrow(SX + 0.55 + 0.65, Y_GR, FX + FW/2, FY,
      color=C_ORG, lw=0.9, style="-|>", shrink=2)
dim_label(FX + FW/2 - 0.30, FY - 0.10, "⊕ 32-dim", C_ORG)

# ════════════════════════════════════════════════════════════════════════════
# Main arrow  encoder → output  (left-to-right: fusion right edge → output block)
# ════════════════════════════════════════════════════════════════════════════
arrow(FX + FW, FY + FH/2, 5.90, FY + FH/2,
      color="#555577", lw=1.1, style="-|>", shrink=2)
dim_label((FX + FW + 5.90) / 2, FY + FH/2 + 0.15, "160-dim", C_MED)

# ════════════════════════════════════════════════════════════════════════════
# BLOCK C — Output
# ════════════════════════════════════════════════════════════════════════════
CX, CY, CW, CH = 5.90, 0.18, 1.02, 3.14
rbox(CX, CY, CW, CH, fc=C_BG_C, ec=C_LINE, lw=1.0, zorder=1)
txt(CX + CW/2, CY + CH - 0.22, "Output", bold=True, fs=8)

# MLP head
rbox(CX + 0.12, 2.35, CW - 0.24, 0.65,
     fc="white", ec="#CC8866", lw=0.9, zorder=3, pad=0.04)
txt(CX + CW/2, 2.73, "MLP", fs=7.5, bold=True, color="#AA5533")
txt(CX + CW/2, 2.55, "160 → 64 → 1", fs=6.5, color=C_MED)
txt(CX + CW/2, 2.39, "→ ŷ ∈ [0,1]", fs=6, color=C_MED)

# Score label
rbox(CX + 0.12, 1.80, CW - 0.24, 0.42,
     fc="#FFEEEE", ec="#CC4444", lw=0.9, zorder=3, pad=0.04)
txt(CX + CW/2, 2.01, "shilling score", fs=7, bold=True, color="#CC2222")

# MLP → score arrow
arrow(CX + CW/2, 2.35, CX + CW/2, 2.24,
      color="#AA5533", lw=0.8, style="-|>", shrink=1)

# Filter box
rbox(CX + 0.12, 1.18, CW - 0.24, 0.48,
     fc="white", ec="#886644", lw=0.9, zorder=3, pad=0.04)
txt(CX + CW/2, 1.45, "Filter", fs=7.5, bold=True, color="#664422")
txt(CX + CW/2, 1.28, "remove flagged", fs=6, color=C_MED)

# Score → filter arrow
arrow(CX + CW/2, 1.80, CX + CW/2, 1.68,
      color="#883322", lw=0.8, style="-|>", shrink=1)

# LightGCN box
rbox(CX + 0.12, 0.46, CW - 0.24, 0.58,
     fc="#F5F8FF", ec="#4466AA", lw=0.9, zorder=3, pad=0.04)
txt(CX + CW/2, 0.85, "LightGCN", fs=7.5, bold=True, color="#335599")
txt(CX + CW/2, 0.72, "NDCG@10 / HR@10", fs=5.5, color=C_MED)
txt(CX + CW/2, 0.57, "recommendation", fs=6, color=C_MED)

# Filter → LightGCN arrow
arrow(CX + CW/2, 1.18, CX + CW/2, 1.06,
      color="#664422", lw=0.8, style="-|>", shrink=1)

# ── Main left-to-right arrows (A→B, already done via B→streams above) ────────
# Input → HGT
arrow(AX + AW, BY + BH/2, B1X, BY + BH/2,
      color=C_MED, lw=1.1, style="-|>", shrink=3)

# ════════════════════════════════════════════════════════════════════════════
# Shared legend — placed below the figure via fig.legend()
# ════════════════════════════════════════════════════════════════════════════
import matplotlib.lines as mlines
_legend_handles = [
    mlines.Line2D([], [], color=C_PUR,  lw=2.0, label="Graph structure"),
    mlines.Line2D([], [], color=C_ORG,  lw=2.0, label="Temporal"),
    mlines.Line2D([], [], color=C_TEAL, lw=2.0, label="Review text"),
    mlines.Line2D([], [], color=C_TEAL, lw=2.0, ls="--", label="Item text"),
]
_leg = fig.legend(
    handles=_legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=True,
    framealpha=1.0,
    edgecolor="#CCCCCC",
    fontsize=8,
    handlelength=1.2,
)

# ── save ──────────────────────────────────────────────────────────────────────
out_dir  = Path(__file__).parent.parent / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "architecture.pdf"
png_path = out_dir / "architecture.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

# Report legend bounding box in figure-fraction coordinates
renderer = fig.canvas.get_renderer()
leg_bb = _leg.get_window_extent(renderer=renderer)
fig_bb = fig.get_window_extent(renderer=renderer)
bb_norm = leg_bb.transformed(fig.transFigure.inverted())
print(f"[legend bbox in fig fractions] "
      f"x0={bb_norm.x0:.4f}  y0={bb_norm.y0:.4f}  "
      f"x1={bb_norm.x1:.4f}  y1={bb_norm.y1:.4f}")

plt.close(fig)
print(f"[PDF] {pdf_path}", file=sys.stderr)
print(f"[PNG] {png_path}", file=sys.stderr)
