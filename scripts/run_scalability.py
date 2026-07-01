#!/usr/bin/env python
"""Profile TH-GNN training and inference costs across datasets and graph sizes.

Per-dataset measurements
------------------------
For each of the four datasets (real data if available; realistic synthetic
proxy otherwise):
  1. Training time per epoch (s)   — mean of PROFILE_EPOCHS timed epochs
  2. Total training time (min)     — extrapolated to CONVERGENCE_EPOCHS
  3. Inference time per 1000 users (ms)
  4. Peak GPU memory during training (MB) — 0 when no CUDA device
  5. Model parameter count

Synthetic scalability sweep
---------------------------
|U| ∈ {1K, 5K, 10K, 50K, 100K} with fixed reviews_per_user=5.
Measure inference time and peak memory at each scale.
Fit a power law log t = α + β log|U| to estimate the empirical exponent.

Outputs
-------
  results/tables/scalability.csv        — per-dataset and per-scale rows
  results/figures/scalability_plot.pdf  — log-log inference time vs |U|

Target claim
------------
  TH-GNN inference scales as O(|E| · H · d) where H=heads and d=hidden_dim,
  enabling real-time detection on graphs with 100K+ users.

Usage
-----
    python scripts/run_scalability.py
    python scripts/run_scalability.py --device cuda
    python scripts/run_scalability.py --smoke      # tiny sizes, fast CI test
    python scripts/run_scalability.py --skip-train # inference-only (no backprop)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.base import HeteroDataset
from src.models import EDGE_TYPES, NODE_TYPES, HGTEncoder, MultimodalFusion, TemporalBurstiness, THGNNDetector
from src.models.fusion import _group_timestamps, _mean_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_DIM        = 64
TEXT_DIM          = 768
PROFILE_EPOCHS    = 5       # epochs run for timing (any dataset)
CONVERGENCE_EPOCHS = 30     # reported as "training to convergence"
INFERENCE_WARMUP  = 2       # warm-up forward passes before timing
INFERENCE_REPEATS = 5       # timed forward passes; report mean ± std
SEED              = 42

# Realistic proxy sizes for each dataset when raw data is absent
# n_users and n_items from published paper statistics; rpu = reviews_per_user
DATASET_PROXY: Dict[str, Dict] = {
    "ml1m":            {"n_users":  6_040, "n_items": 3_706, "rpu": 15},
    "amazon_books":    {"n_users": 52_643, "n_items": 9_139, "rpu":  8},
    "amazon_clothing": {"n_users": 18_226, "n_items":17_668, "rpu":  6},
    "yelp2018":        {"n_users": 31_668, "n_items":38_048, "rpu": 10},
}

# Graph sizes for the synthetic scalability sweep
SCALE_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
SCALE_RPU   = 5   # reviews per user — kept constant across sweep

# Smoke-test overrides (tiny sizes, fast)
SMOKE_PROXY: Dict[str, Dict] = {
    "ml1m":            {"n_users":  200, "n_items": 100, "rpu": 5},
    "amazon_books":    {"n_users":  500, "n_items": 200, "rpu": 5},
    "amazon_clothing": {"n_users":  300, "n_items": 150, "rpu": 5},
    "yelp2018":        {"n_users":  400, "n_items": 200, "rpu": 5},
}
SMOKE_SCALE_SIZES = [100, 500, 1_000, 5_000, 10_000]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScalabilityRecord:
    dataset:             str
    n_users:             int
    n_items:             int
    n_reviews:           int
    n_params:            int
    train_epoch_s:       float   # mean per-epoch training time (s); NaN = skipped
    train_total_min:     float   # extrapolated to CONVERGENCE_EPOCHS; NaN = skipped
    infer_per1k_ms:      float   # mean inference time per 1000 users (ms)
    infer_std_ms:        float   # std over INFERENCE_REPEATS
    peak_mem_mb:         float   # peak GPU MB during training; 0 = CPU/N/A
    source:              str     # "real" | "synthetic_proxy"
    row_type:            str     # "dataset" | "scale"


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------

def make_synthetic(
    n_users: int,
    n_items: int,
    rpu: int,
    seed: int = SEED,
) -> HeteroDataset:
    """Build a random heterogeneous graph for benchmarking."""
    rng = np.random.default_rng(seed)
    n_reviews = n_users * rpu

    rev_user = np.repeat(np.arange(n_users, dtype=np.int64), rpu)
    rev_item = rng.integers(0, n_items, size=n_reviews).astype(np.int64)

    ts = rng.uniform(0, 365 * 86400, size=n_reviews).astype(np.float64)
    ratings = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0],
                         size=n_reviews).astype(np.float32)

    x_user = torch.zeros(n_users, 5, dtype=torch.float32)
    x_item = torch.zeros(n_items, 5, dtype=torch.float32)
    # degree feature
    for u in range(n_users):
        x_user[u, 0] = rpu
    for i in rev_item:
        x_item[i, 0] += 1

    r_idx = np.arange(n_reviews, dtype=np.int64)
    ts_t  = torch.from_numpy(ts)
    ts_f  = ts_t.float()
    eid = {
        ("user", "writes", "review"): torch.from_numpy(np.stack([rev_user, r_idx])),
        ("review", "about", "item"):  torch.from_numpy(np.stack([r_idx, rev_item])),
        ("user", "rates", "item"):    torch.from_numpy(np.stack([rev_user, rev_item])),
    }
    ea = {k: ts_f for k in eid}

    n_train = int(n_reviews * 0.8)
    n_val   = int(n_reviews * 0.1)
    train_m = torch.zeros(n_reviews, dtype=torch.bool)
    val_m   = torch.zeros(n_reviews, dtype=torch.bool)
    test_m  = torch.zeros(n_reviews, dtype=torch.bool)
    train_m[:n_train] = True
    val_m[n_train:n_train + n_val] = True
    test_m[n_train + n_val:] = True

    n_fake = max(1, n_users // 20)
    labels = torch.zeros(n_users, dtype=torch.long)
    labels[:n_fake] = 1

    return HeteroDataset(
        dataset_name    = "synthetic",
        num_users       = n_users,
        num_items       = n_items,
        num_reviews     = n_reviews,
        edge_index_dict = eid,
        edge_attr_dict  = ea,
        x_user          = x_user,
        x_item          = x_item,
        review_texts    = [""] * n_reviews,
        ratings         = torch.from_numpy(ratings),
        timestamps      = ts_t,
        review_user_idx = torch.from_numpy(rev_user),
        review_item_idx = torch.from_numpy(rev_item),
        labels          = labels,
        train_mask      = train_m,
        val_mask        = val_m,
        test_mask       = test_m,
        user_id_map     = {i: i for i in range(n_users)},
        item_id_map     = {i: i for i in range(n_items)},
    )


def try_load_real(dataset_name: str) -> Optional[HeteroDataset]:
    """Attempt to load the real dataset; return None on failure."""
    data_root = Path("data/raw") / dataset_name
    if not data_root.exists():
        return None
    try:
        if dataset_name == "ml1m":
            from src.datasets import MovieLens1MLoader
            return MovieLens1MLoader(str(data_root)).load()
        elif dataset_name == "amazon_books":
            from src.datasets import AmazonBooksLoader
            return AmazonBooksLoader(str(data_root)).load()
        elif dataset_name == "amazon_clothing":
            from src.datasets import AmazonClothingLoader
            return AmazonClothingLoader(str(data_root)).load()
        elif dataset_name == "yelp2018":
            from src.datasets import Yelp2018Loader
            return Yelp2018Loader(str(data_root)).load()
    except Exception as exc:
        logger.warning("Failed to load %s: %s", dataset_name, exc)
    return None


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(dataset: HeteroDataset, device: torch.device) -> THGNNDetector:
    input_dims = {
        "user":   dataset.x_user.shape[1],
        "item":   dataset.x_item.shape[1],
        "review": 1,
    }
    enc    = HGTEncoder(
        hidden_dim   = HIDDEN_DIM,
        num_heads    = max(1, HIDDEN_DIM // 32),
        num_layers   = 2,
        node_types   = NODE_TYPES,
        edge_types   = EDGE_TYPES,
        input_dims   = input_dims,
        dropout      = 0.0,   # no dropout during profiling for determinism
        time_emb_dim = 32,
    )
    fusion = MultimodalFusion(
        subgraph_dim = HIDDEN_DIM,
        text_dim     = TEXT_DIM,
        proj_dim     = HIDDEN_DIM,
        num_heads    = max(1, HIDDEN_DIM // 32),
        dropout      = 0.0,
    )
    burst  = TemporalBurstiness(hidden_dim=32, dropout=0.0)
    return THGNNDetector(enc, fusion, burst,
                         mlp_hidden=HIDDEN_DIM // 2,
                         dropout=0.0).to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


def _build_inputs(
    dataset: HeteroDataset,
    device: torch.device,
    review_feat_dim: int,
) -> Dict:
    """Pack dataset tensors into a single dict for repeated forward passes."""
    return dict(
        node_feats = {
            "user":   dataset.x_user.to(device),
            "item":   dataset.x_item.to(device),
            "review": torch.zeros(dataset.num_reviews, review_feat_dim, device=device),
        },
        edge_index_dict = {k: v.to(device) for k, v in dataset.edge_index_dict.items()},
        edge_attr_dict  = {k: v.float().to(device) for k, v in dataset.edge_attr_dict.items()},
        review_embs     = torch.zeros(dataset.num_reviews, TEXT_DIM, device=device),
        review_user_idx = dataset.review_user_idx.to(device),
        review_item_idx = dataset.review_item_idx.to(device),
        timestamps      = dataset.timestamps.float().to(device),
    )


def _forward_once(model: THGNNDetector, inputs: Dict) -> torch.Tensor:
    scores, _ = model(
        inputs["node_feats"],
        inputs["edge_index_dict"],
        inputs["edge_attr_dict"],
        inputs["review_embs"],
        None,
        inputs["review_user_idx"],
        inputs["review_item_idx"],
        inputs["timestamps"],
    )
    return scores


def _focal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    p_t = torch.exp(-bce)
    return ((1.0 - p_t) ** 2.0 * bce).mean()


# ---------------------------------------------------------------------------
# Per-dataset profiling
# ---------------------------------------------------------------------------

def profile_dataset(
    dataset_name: str,
    dataset: HeteroDataset,
    device: torch.device,
    skip_train: bool = False,
    source: str = "synthetic_proxy",
) -> ScalabilityRecord:
    logger.info(
        "Profiling %s — %d users, %d items, %d reviews …",
        dataset_name, dataset.num_users, dataset.num_items, dataset.num_reviews,
    )
    torch.manual_seed(SEED)
    model = build_model(dataset, device)
    n_params = count_params(model)
    rfeat_dim = model.hgt.input_projs["review"].in_features
    inputs    = _build_inputs(dataset, device, rfeat_dim)
    labels    = dataset.labels.float().to(device)

    # ── Training timing ───────────────────────────────────────────────────────
    train_epoch_s  = float("nan")
    train_total_min = float("nan")

    if not skip_train:
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=PROFILE_EPOCHS, eta_min=1e-5,
        )
        epoch_times: List[float] = []

        _reset_mem(device)
        model.train()
        for epoch in range(1, PROFILE_EPOCHS + 1):
            _sync(device)
            t0 = time.perf_counter()

            opt.zero_grad()
            scores = _forward_once(model, inputs)
            loss   = _focal_loss(scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            _sync(device)
            epoch_times.append(time.perf_counter() - t0)

        peak_mb = _peak_mem_mb(device)

        # Use last min(PROFILE_EPOCHS, 3) epochs for stable estimate
        window = epoch_times[-min(3, len(epoch_times)):]
        train_epoch_s   = float(np.mean(window))
        train_total_min = train_epoch_s * CONVERGENCE_EPOCHS / 60.0
    else:
        _reset_mem(device)
        peak_mb = 0.0

    # ── Inference timing ──────────────────────────────────────────────────────
    model.eval()
    _reset_mem(device)
    infer_times: List[float] = []

    with torch.no_grad():
        # Warm-up
        for _ in range(INFERENCE_WARMUP):
            _forward_once(model, inputs)
            _sync(device)

        # Timed runs
        for _ in range(INFERENCE_REPEATS):
            _sync(device)
            t0 = time.perf_counter()
            _forward_once(model, inputs)
            _sync(device)
            infer_times.append(time.perf_counter() - t0)

    infer_total_s     = float(np.mean(infer_times))
    infer_std_s       = float(np.std(infer_times))
    infer_per1k_ms    = infer_total_s * 1000.0 / dataset.num_users * 1000.0
    infer_std_ms      = infer_std_s   * 1000.0 / dataset.num_users * 1000.0

    if not skip_train:
        peak_mem_mb = peak_mb
    else:
        peak_mem_mb = _peak_mem_mb(device)

    logger.info(
        "  n_params=%d  epoch=%.2fs  total=%.1fmin  infer/1k=%.1fms  peak=%.0fMB",
        n_params,
        train_epoch_s if not np.isnan(train_epoch_s) else 0.0,
        train_total_min if not np.isnan(train_total_min) else 0.0,
        infer_per1k_ms,
        peak_mem_mb,
    )

    return ScalabilityRecord(
        dataset           = dataset_name,
        n_users           = dataset.num_users,
        n_items           = dataset.num_items,
        n_reviews         = dataset.num_reviews,
        n_params          = n_params,
        train_epoch_s     = train_epoch_s,
        train_total_min   = train_total_min,
        infer_per1k_ms    = infer_per1k_ms,
        infer_std_ms      = infer_std_ms,
        peak_mem_mb       = peak_mem_mb,
        source            = source,
        row_type          = "dataset",
    )


# ---------------------------------------------------------------------------
# Synthetic scalability sweep
# ---------------------------------------------------------------------------

def profile_scale(
    n_users: int,
    device: torch.device,
) -> ScalabilityRecord:
    n_items   = max(10, n_users // 5)
    rpu       = SCALE_RPU
    dataset   = make_synthetic(n_users, n_items, rpu, seed=SEED)
    label     = f"synth_{n_users // 1000}K"

    logger.info(
        "Scale test |U|=%d  |I|=%d  |E|=%d …",
        n_users, n_items, n_users * rpu,
    )
    torch.manual_seed(SEED)
    model = build_model(dataset, device)
    n_params  = count_params(model)
    rfeat_dim = model.hgt.input_projs["review"].in_features
    inputs    = _build_inputs(dataset, device, rfeat_dim)

    model.eval()
    _reset_mem(device)
    infer_times: List[float] = []

    with torch.no_grad():
        for _ in range(INFERENCE_WARMUP):
            try:
                _forward_once(model, inputs)
                _sync(device)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning("OOM at |U|=%d — skipping.", n_users)
                    return ScalabilityRecord(
                        dataset=label, n_users=n_users, n_items=n_items,
                        n_reviews=n_users * rpu, n_params=n_params,
                        train_epoch_s=float("nan"), train_total_min=float("nan"),
                        infer_per1k_ms=float("nan"), infer_std_ms=float("nan"),
                        peak_mem_mb=float("nan"), source="synthetic_scale",
                        row_type="scale",
                    )
                raise

        for _ in range(INFERENCE_REPEATS):
            _sync(device)
            t0 = time.perf_counter()
            _forward_once(model, inputs)
            _sync(device)
            infer_times.append(time.perf_counter() - t0)

    infer_total_s  = float(np.mean(infer_times))
    infer_std_s    = float(np.std(infer_times))
    infer_per1k_ms = infer_total_s  * 1000.0 / n_users * 1000.0
    infer_std_ms   = infer_std_s    * 1000.0 / n_users * 1000.0
    peak_mb        = _peak_mem_mb(device)

    logger.info(
        "  infer_total=%.2fs  infer/1k=%.2fms  peak=%.0fMB",
        infer_total_s, infer_per1k_ms, peak_mb,
    )

    return ScalabilityRecord(
        dataset           = label,
        n_users           = n_users,
        n_items           = n_items,
        n_reviews         = n_users * rpu,
        n_params          = n_params,
        train_epoch_s     = float("nan"),
        train_total_min   = float("nan"),
        infer_per1k_ms    = infer_per1k_ms,
        infer_std_ms      = infer_std_ms,
        peak_mem_mb       = peak_mb,
        source            = "synthetic_scale",
        row_type          = "scale",
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(records: List[ScalabilityRecord], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type", "dataset", "n_users", "n_items", "n_reviews", "n_params",
        "train_epoch_s", "train_total_min",
        "infer_per1k_ms", "infer_std_ms",
        "peak_mem_mb", "source",
    ]

    def _fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            writer.writerow({k: _fmt(row[k]) for k in fieldnames})

    logger.info("Saved scalability table to %s (%d rows).", path, len(records))


# ---------------------------------------------------------------------------
# Log-log plot
# ---------------------------------------------------------------------------

def plot_scalability(
    scale_records: List[ScalabilityRecord],
    dataset_records: List[ScalabilityRecord],
    out_path: Path,
) -> None:
    """Log-log inference time vs |U| with power-law fit and O(n) reference."""
    # Filter valid scale points
    valid = [(r.n_users, r.infer_per1k_ms * r.n_users / 1000)
             for r in scale_records
             if not np.isnan(r.infer_per1k_ms)]
    if len(valid) < 2:
        logger.warning("Fewer than 2 valid scale points — skipping plot.")
        return

    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)

    # Power-law fit on log-log
    log_x = np.log(xs)
    log_y = np.log(ys * 1000)   # convert to ms for display
    coeffs = np.polyfit(log_x, log_y, 1)
    beta, alpha = coeffs         # log y = beta * log x + alpha
    fit_xs = np.geomspace(xs.min(), xs.max(), 200)
    fit_ys = np.exp(alpha) * fit_xs ** beta

    # O(n) reference line anchored at smallest point
    ref_ys = ys[0] * 1000 * (fit_xs / xs[0])

    fig, ax = plt.subplots(figsize=(6, 4))

    # Dataset-level points (filled markers)
    for rec in dataset_records:
        if np.isnan(rec.infer_per1k_ms):
            continue
        total_ms = rec.infer_per1k_ms * rec.n_users / 1000
        ax.scatter(rec.n_users, total_ms, marker="D", s=60,
                   color="#d62728", zorder=5,
                   label="Real/proxy dataset" if rec == dataset_records[0] else "")

    # Scale sweep points
    ax.scatter(xs, ys * 1000, marker="o", s=40, color="#1f77b4",
               zorder=5, label="Synthetic scale sweep")

    # Fitted curve
    ax.plot(fit_xs, fit_ys, "--", color="#1f77b4", linewidth=1.5,
            label=rf"Fit: $O(n^{{{beta:.2f}}})$")

    # O(n) reference
    ax.plot(fit_xs, ref_ys, ":", color="#888888", linewidth=1.2,
            label=r"$O(n)$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of users $|U|$", fontsize=11)
    ax.set_ylabel("Inference time (ms, total graph)", fontsize=11)
    ax.set_title(
        "TH-GNN inference scaling (H=2 heads, d=64)\n"
        f"Empirical exponent β = {beta:.3f}  (β = 1.0 is linear)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Annotate the fitted exponent
    mid_x = np.sqrt(xs[0] * xs[-1])
    mid_y = float(np.exp(alpha) * mid_x ** beta) * 1.6
    ax.annotate(
        rf"β = {beta:.2f}",
        xy=(mid_x, float(np.exp(alpha) * mid_x ** beta)),
        xytext=(mid_x, mid_y),
        ha="center", fontsize=9, color="#1f77b4",
        arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.8),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scalability plot to %s.", out_path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    dataset_records: List[ScalabilityRecord],
    scale_records: List[ScalabilityRecord],
    beta: float,
) -> None:
    print("\n" + "=" * 76)
    print("SCALABILITY SUMMARY")
    print("=" * 76)
    print(f"\n{'Dataset':<20} {'Users':>7} {'Reviews':>9} {'Params':>8} "
          f"{'Ep(s)':>7} {'Total(m)':>9} {'Inf/1k(ms)':>11} {'PeakMB':>8}")
    print("-" * 76)
    for r in dataset_records:
        ep  = f"{r.train_epoch_s:.2f}" if not np.isnan(r.train_epoch_s) else "N/A"
        tot = f"{r.train_total_min:.1f}" if not np.isnan(r.train_total_min) else "N/A"
        pk  = f"{r.peak_mem_mb:.0f}"    if r.peak_mem_mb > 0 else "CPU"
        print(f"{r.dataset:<20} {r.n_users:>7,} {r.n_reviews:>9,} {r.n_params:>8,} "
              f"{ep:>7} {tot:>9} {r.infer_per1k_ms:>10.2f}  {pk:>7}")

    print(f"\nSynthetic scalability sweep  (reviews_per_user={SCALE_RPU})")
    print(f"{'|U|':>10} {'Inf_total(ms)':>14} {'Inf/1k(ms)':>12} {'PeakMB':>8}")
    print("-" * 46)
    for r in scale_records:
        if np.isnan(r.infer_per1k_ms):
            print(f"{r.n_users:>10,}  {'OOM':>14}")
            continue
        total_ms = r.infer_per1k_ms * r.n_users / 1000
        pk = f"{r.peak_mem_mb:.0f}" if r.peak_mem_mb > 0 else "CPU"
        print(f"{r.n_users:>10,} {total_ms:>14.1f} {r.infer_per1k_ms:>12.3f} {pk:>7}")

    print(f"\nPower-law fit: inference time ~ O(|U|^{beta:.3f})")
    if abs(beta - 1.0) < 0.15:
        print("  => Consistent with O(|E|) = O(|U|) linear scaling (fixed reviews/user).")
    else:
        print(f"  => Empirical exponent {beta:.3f} deviates from ideal O(1.0).")

    # Claim verification
    valid_scales = [r for r in scale_records if not np.isnan(r.infer_per1k_ms)]
    if valid_scales:
        largest = max(valid_scales, key=lambda r: r.n_users)
        total_s = largest.infer_per1k_ms * largest.n_users / 1000 / 1000
        print(f"\nTarget claim: TH-GNN inference on {largest.n_users:,} users takes "
              f"{total_s:.2f} s — "
              f"{'within real-time budget.' if total_s < 60 else 'exceeds 60s threshold on this device.'}")
    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile TH-GNN training/inference scalability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device",      default="cpu",
                   help="Torch device (cpu / cuda / mps).")
    p.add_argument("--smoke",       action="store_true",
                   help="Tiny sizes for fast CI smoke-test.")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training profiling; measure inference only.")
    p.add_argument("--out-dir",     default="results/tables",
                   help="Directory for scalability.csv.")
    p.add_argument("--fig-dir",     default="results/figures",
                   help="Directory for scalability_plot.pdf.")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = torch.device(args.device)

    proxy  = SMOKE_PROXY if args.smoke else DATASET_PROXY
    scales = SMOKE_SCALE_SIZES if args.smoke else SCALE_SIZES

    if args.smoke:
        logger.info("Smoke mode — using tiny synthetic sizes.")

    dataset_records: List[ScalabilityRecord] = []
    scale_records:   List[ScalabilityRecord] = []

    # ── Per-dataset profiling ─────────────────────────────────────────────────
    for ds_name, cfg in proxy.items():
        real_ds = try_load_real(ds_name)
        if real_ds is not None:
            ds     = real_ds
            source = "real"
        else:
            ds = make_synthetic(
                n_users = cfg["n_users"],
                n_items = cfg["n_items"],
                rpu     = cfg["rpu"],
                seed    = SEED,
            )
            source = "synthetic_proxy"

        rec = profile_dataset(
            dataset_name = ds_name,
            dataset      = ds,
            device       = device,
            skip_train   = args.skip_train,
            source       = source,
        )
        dataset_records.append(rec)

    # ── Synthetic scalability sweep ───────────────────────────────────────────
    for n_users in scales:
        rec = profile_scale(n_users, device)
        scale_records.append(rec)

    # ── Power-law fit ─────────────────────────────────────────────────────────
    valid = [(r.n_users, r.infer_per1k_ms * r.n_users / 1000)
             for r in scale_records if not np.isnan(r.infer_per1k_ms)]
    if len(valid) >= 2:
        log_x = np.log([v[0] for v in valid])
        log_y = np.log([v[1] * 1000 for v in valid])
        coeffs = np.polyfit(log_x, log_y, 1)
        beta = float(coeffs[0])
    else:
        beta = float("nan")

    # ── Outputs ───────────────────────────────────────────────────────────────
    all_records = dataset_records + scale_records

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    write_csv(all_records, out_dir / "scalability.csv")
    plot_scalability(scale_records, dataset_records, fig_dir / "scalability_plot.pdf")
    print_summary(dataset_records, scale_records, beta if not np.isnan(beta) else 1.0)


if __name__ == "__main__":
    main()
