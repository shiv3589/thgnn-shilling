#!/usr/bin/env python
"""Full training loop for THGNNDetector — shilling attack detection.

Algorithm summary
-----------------
1. Load heterogeneous dataset and precomputed BERT embeddings.
2. Split users 80 / 10 / 10 at random (seed-controlled); labels come from
   dataset.labels (all-zero for clean data; non-zero after attack injection).
3. Mini-batch training over train users:
   - For each batch of 512 users, extract a 2-hop heterogeneous subgraph.
   - Forward pass through THGNNDetector → raw logit scores.
   - Focal loss (γ=2, α=0.25) for class-imbalance robustness.
   - AdamW optimiser with cosine LR annealing.
4. Validate every epoch on val users; early-stop on val F1 (patience=10).
5. Log train loss, val F1, val DR, val FAR to W&B (optional).
6. Save best checkpoint to checkpoints/{dataset}_{run_id}.pt.

Usage
-----
    python scripts/train.py --dataset ml1m
    python scripts/train.py --dataset yelp2018 --epochs 50 --wandb
    python scripts/train.py --dataset amazon_books --lr 5e-4 --seed 123
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import (
    AmazonBooksLoader,
    AmazonClothingLoader,
    BaseLoader,
    HeteroDataset,
    MovieLens1MLoader,
    Yelp2018Loader,
)
from src.evaluation import (
    best_threshold_f1,
    detection_rate,
    false_alarm_rate,
)
from src.models import (
    EDGE_TYPES,
    NODE_TYPES,
    HGTEncoder,
    MultimodalFusion,
    TemporalBurstiness,
    THGNNDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

LOADERS: Dict[str, type] = {
    "ml1m":             MovieLens1MLoader,
    "amazon_books":     AmazonBooksLoader,
    "amazon_clothing":  AmazonClothingLoader,
    "yelp2018":         Yelp2018Loader,
}

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits: Tensor,
    labels: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Binary focal loss (Lin et al., 2017).

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    where p_t = σ(logit) for positive class, 1 − σ(logit) for negative.
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = labels.float() * alpha + (1.0 - labels.float()) * (1.0 - alpha)
    return (alpha_t * (1.0 - p_t) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# K-hop subgraph sampling
# ---------------------------------------------------------------------------

def _khop_subgraph(
    seed_user_globals: Tensor,
    node_feats: Dict[str, Tensor],
    edge_index_dict: Dict,
    edge_attr_dict: Dict,
    node_timestamps: Optional[Dict[str, Tensor]],
    rev_user_idx: Tensor,
    rev_item_idx: Tensor,
    timestamps: Tensor,
    review_embs: Optional[Tensor],
    item_embs: Optional[Tensor],
    k: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Tuple:
    """Extract k-hop heterogeneous subgraph centred on *seed_user_globals*.

    BFS expands outward from seed users through all edge types for k hops.
    All returned tensors use local (subgraph-level) indices.

    Returns
    -------
    sub_feats        : Dict[str, Tensor] — local node features
    sub_ei           : Dict — local edge_index_dict
    sub_ea           : Dict — local edge_attr_dict
    sub_nt           : Optional[Dict[str, Tensor]] — local node_timestamps
    sub_rev_embs     : Tensor(R_sub, 768)
    sub_item_embs    : Optional[Tensor(I_sub, 768)]
    sub_rev_user_idx : Tensor(R_sub,) — local user index for each sub-review
    sub_rev_item_idx : Tensor(R_sub,) — local item index for each sub-review
    sub_timestamps   : Tensor(R_sub,)
    local_seed_idx   : Tensor(B,) — positions of seed users in sub_feats["user"]
    """
    seed_list = seed_user_globals.cpu().tolist()

    # BFS: track visited nodes and which edge indices to include
    visited: Dict[str, set] = {
        "user":   set(seed_list),
        "item":   set(),
        "review": set(),
    }
    frontier: Dict[str, set] = {"user": set(seed_list), "item": set(), "review": set()}
    included: Dict = {etype: [] for etype in edge_index_dict}

    # Move edge tensors to CPU for BFS (indexing is CPU-side)
    ei_cpu = {et: ei.cpu() for et, ei in edge_index_dict.items()}

    for _ in range(k):
        new_frontier: Dict[str, set] = {"user": set(), "item": set(), "review": set()}
        for etype, ei in ei_cpu.items():
            src_type, _, dst_type = etype
            if not frontier[src_type]:
                continue
            src_tensor = torch.tensor(sorted(frontier[src_type]), dtype=torch.long)
            mask = torch.isin(ei[0], src_tensor)
            edge_idx = torch.where(mask)[0]
            if edge_idx.numel() > 0:
                included[etype].append(edge_idx)
                for n in ei[1][edge_idx].tolist():
                    if n not in visited[dst_type]:
                        new_frontier[dst_type].add(n)
                        visited[dst_type].add(n)
        frontier = new_frontier

    # Deduplicate collected edge indices
    all_edge_idx = {}
    for etype in edge_index_dict:
        parts = included[etype]
        if parts:
            all_edge_idx[etype] = torch.cat(parts).unique()

    # Build global→local maps
    nodes_sorted = {nt: sorted(visited[nt]) for nt in ["user", "item", "review"]}
    g2l = {nt: {g: l for l, g in enumerate(nodes_sorted[nt])} for nt in nodes_sorted}

    # Slice node features
    sub_feats: Dict[str, Tensor] = {}
    sub_nt: Optional[Dict[str, Tensor]] = {} if node_timestamps else None
    for nt in ["user", "item", "review"]:
        globs = nodes_sorted[nt]
        if globs:
            idx = torch.tensor(globs, dtype=torch.long)
            sub_feats[nt] = node_feats[nt][idx].to(device)
            if node_timestamps and nt in node_timestamps:
                sub_nt[nt] = node_timestamps[nt][idx].to(device)  # type: ignore[index]
        else:
            dim = node_feats[nt].shape[1]
            sub_feats[nt] = torch.zeros(0, dim, device=device)

    # Build re-indexed edge dicts
    sub_ei: Dict = {}
    sub_ea: Dict = {}
    for etype, ei in ei_cpu.items():
        src_type, _, dst_type = etype
        if etype not in all_edge_idx or all_edge_idx[etype].numel() == 0:
            continue
        idxs = all_edge_idx[etype]
        src_l = torch.tensor([g2l[src_type][g] for g in ei[0][idxs].tolist()],
                              dtype=torch.long, device=device)
        dst_l = torch.tensor([g2l[dst_type][g] for g in ei[1][idxs].tolist()],
                              dtype=torch.long, device=device)
        sub_ei[etype]  = torch.stack([src_l, dst_l])
        sub_ea[etype]  = edge_attr_dict[etype][idxs].to(device)

    # Build review-level tensors using local indices
    rev_globs = nodes_sorted["review"]
    rev_dim = review_embs.shape[1] if review_embs is not None else 768
    if rev_globs:
        rev_t = torch.tensor(rev_globs, dtype=torch.long)
        sub_rev_user_idx = torch.tensor(
            [g2l["user"][rev_user_idx[r].item()] for r in rev_globs],
            dtype=torch.long, device=device,
        )
        sub_rev_item_idx = torch.tensor(
            [g2l["item"][rev_item_idx[r].item()] for r in rev_globs],
            dtype=torch.long, device=device,
        )
        sub_timestamps = timestamps[rev_t].to(device)
        sub_rev_embs = (review_embs[rev_t].to(device)
                        if review_embs is not None
                        else torch.zeros(len(rev_globs), rev_dim, device=device))
    else:
        sub_rev_user_idx = torch.zeros(0, dtype=torch.long, device=device)
        sub_rev_item_idx = torch.zeros(0, dtype=torch.long, device=device)
        sub_timestamps   = torch.zeros(0, device=device)
        sub_rev_embs     = torch.zeros(0, rev_dim, device=device)

    # Slice item embeddings to subgraph items
    item_globs = nodes_sorted["item"]
    if item_embs is not None and item_globs:
        item_t = torch.tensor(item_globs, dtype=torch.long)
        sub_item_embs: Optional[Tensor] = item_embs[item_t].to(device)
    else:
        sub_item_embs = None

    # Map seed users to their local positions
    local_seed_idx = torch.tensor(
        [g2l["user"][g] for g in seed_list], dtype=torch.long, device=device
    )

    return (
        sub_feats, sub_ei, sub_ea, sub_nt,
        sub_rev_embs, sub_item_embs,
        sub_rev_user_idx, sub_rev_item_idx,
        sub_timestamps, local_seed_idx,
    )


# ---------------------------------------------------------------------------
# Evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(
    model: THGNNDetector,
    user_indices: Tensor,
    dataset: HeteroDataset,
    review_embs: Optional[Tensor],
    item_embs: Optional[Tensor],
    device: torch.device,
    batch_size: int = 512,
    khop: int = 2,
) -> Dict[str, float]:
    """Run detector over *user_indices* in mini-batches, return detection metrics."""
    model.eval()
    all_scores: List[Tensor] = []
    all_labels: List[Tensor] = []

    nf = {nt: dataset.x_user if nt == "user" else
             dataset.x_item if nt == "item" else
             torch.zeros(dataset.num_reviews, 1)
          for nt in NODE_TYPES}
    # Build proper node_feats
    nf = {
        "user":   dataset.x_user,
        "item":   dataset.x_item,
        "review": torch.zeros(dataset.num_reviews, dataset.x_user.shape[1]),
    }
    # Use per-review feature dim matching the review projection
    review_feat_dim = model.hgt.input_projs["review"].in_features
    nf["review"] = torch.zeros(dataset.num_reviews, review_feat_dim)

    for start in range(0, len(user_indices), batch_size):
        batch_globals = user_indices[start : start + batch_size]
        (
            sub_feats, sub_ei, sub_ea, sub_nt,
            sub_rev_embs, sub_item_embs,
            sub_rev_user, sub_rev_item,
            sub_ts, local_seed,
        ) = _khop_subgraph(
            batch_globals, nf,
            dataset.edge_index_dict, dataset.edge_attr_dict,
            None,
            dataset.review_user_idx, dataset.review_item_idx,
            dataset.timestamps.float(),
            review_embs, item_embs,
            k=khop, device=device,
        )

        scores, _ = model(
            sub_feats, sub_ei, sub_ea,
            sub_rev_embs, sub_item_embs,
            sub_rev_user, sub_rev_item,
            sub_ts,
        )
        all_scores.append(torch.sigmoid(scores[local_seed]).cpu())
        all_labels.append(dataset.labels[batch_globals].cpu())

    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy()

    if labels_np.sum() == 0:
        # All genuine users — can't compute meaningful F1/DR; return loss proxy
        logger.debug("No positive labels in evaluation split — F1/DR/FAR are 0")

    f1, thresh = best_threshold_f1(labels_np, scores_np)
    dr  = detection_rate(labels_np, scores_np, threshold=thresh)
    far = false_alarm_rate(labels_np, scores_np, threshold=thresh)
    return {"f1": f1, "dr": dr, "far": far, "threshold": thresh}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(
    dataset: HeteroDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> THGNNDetector:
    input_dims = {
        "user":   dataset.x_user.shape[1],
        "item":   dataset.x_item.shape[1],
        "review": args.review_feat_dim,
    }
    hgt = HGTEncoder(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        input_dims=input_dims,
        dropout=args.dropout,
        time_emb_dim=args.time_emb_dim,
    )
    fusion = MultimodalFusion(
        subgraph_dim=args.hidden_dim,
        text_dim=args.text_dim,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    burst = TemporalBurstiness(hidden_dim=args.burst_dim, dropout=args.dropout)
    return THGNNDetector(hgt, fusion, burst,
                         mlp_hidden=args.mlp_hidden,
                         dropout=args.dropout).to(device)


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------

def _load_embeddings(
    dataset_name: str,
    num_reviews: int,
    num_items: int,
    text_dim: int,
    cache_dir: str = "data/processed",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Load precomputed BERT embeddings from disk if available."""
    base = Path(cache_dir) / dataset_name / "embeddings"
    rev_path  = base / "reviews.npy"
    item_path = base / "items.npy"

    rev_embs = item_embs = None

    if rev_path.exists():
        arr = np.load(rev_path)
        if arr.shape == (num_reviews, text_dim):
            rev_embs = torch.from_numpy(arr).float()
            logger.info("Loaded review embeddings from %s", rev_path)
        else:
            logger.warning("Review embedding shape mismatch: %s vs (%d, %d)",
                           arr.shape, num_reviews, text_dim)
    else:
        logger.warning("No review embeddings at %s — using zeros", rev_path)
        rev_embs = torch.zeros(num_reviews, text_dim)

    if item_path.exists():
        arr = np.load(item_path)
        if arr.shape == (num_items, text_dim):
            item_embs = torch.from_numpy(arr).float()
            logger.info("Loaded item embeddings from %s", item_path)
        else:
            logger.warning("Item embedding shape mismatch: %s vs (%d, %d)",
                           arr.shape, num_items, text_dim)
    else:
        logger.info("No item embeddings at %s — item stream will use zeros", item_path)

    return rev_embs, item_embs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train THGNNDetector for shilling attack detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--dataset", required=True, choices=list(LOADERS),
                   help="Dataset key: ml1m | amazon_books | amazon_clothing | yelp2018")
    p.add_argument("--raw-dir",   default="data/raw",       metavar="DIR")
    p.add_argument("--cache-dir", default="data/processed", metavar="DIR")
    p.add_argument("--ckpt-dir",  default="checkpoints",    metavar="DIR")
    p.add_argument("--min-interactions", type=int, default=5, metavar="K")

    # Training
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--khop",       type=int,   default=2,
                   help="Number of hops for subgraph sampling")
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early-stopping patience on val F1")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default=None, metavar="DEVICE",
                   help="cpu | cuda | mps (autodetect if omitted)")

    # Focal loss
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)

    # Model
    p.add_argument("--hidden-dim",     type=int, default=128)
    p.add_argument("--num-heads",      type=int, default=4)
    p.add_argument("--num-layers",     type=int, default=2)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--time-emb-dim",   type=int, default=64)
    p.add_argument("--text-dim",       type=int, default=768,
                   help="BERT embedding dimension")
    p.add_argument("--proj-dim",       type=int, default=128,
                   help="Cross-modal attention projection dimension")
    p.add_argument("--burst-dim",      type=int, default=32)
    p.add_argument("--mlp-hidden",     type=int, default=64)
    p.add_argument("--review-feat-dim", type=int, default=1,
                   help="Dummy review node feature dim (reviews lack numeric stats)")

    # Logging
    p.add_argument("--wandb",   action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--run-id",  default=None, metavar="ID",
                   help="Run identifier (auto-generated if omitted)")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Validate every N epochs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    run_id = args.run_id or uuid.uuid4().hex[:8]
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.dataset}_{run_id}.pt"

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="thgnn-shilling",
                name=f"{args.dataset}_{run_id}",
                config=vars(args),
            )
            use_wandb = True
        except ImportError:
            logger.warning("wandb not installed — logging disabled")

    # ── Dataset ──────────────────────────────────────────────────────────────
    loader_cls = LOADERS[args.dataset]
    loader = loader_cls(raw_dir=args.raw_dir)
    dataset = loader.load(
        min_user_interactions=args.min_interactions,
        min_item_interactions=args.min_interactions,
    )
    logger.info("%s", dataset.summary())

    # ── BERT embeddings ───────────────────────────────────────────────────────
    review_embs, item_embs = _load_embeddings(
        dataset_name=dataset.dataset_name,
        num_reviews=dataset.num_reviews,
        num_items=dataset.num_items,
        text_dim=args.text_dim,
        cache_dir=args.cache_dir,
        device=torch.device("cpu"),  # kept on CPU; sliced per batch
    )

    # ── User split (80 / 10 / 10 by user index) ───────────────────────────────
    rng = np.random.default_rng(args.seed)
    all_users = np.arange(dataset.num_users)
    rng.shuffle(all_users)
    n_train = int(0.8 * dataset.num_users)
    n_val   = int(0.1 * dataset.num_users)
    train_users = torch.from_numpy(all_users[:n_train])
    val_users   = torch.from_numpy(all_users[n_train : n_train + n_val])
    test_users  = torch.from_numpy(all_users[n_train + n_val :])
    logger.info(
        "User split — train: %d  val: %d  test: %d",
        len(train_users), len(val_users), len(test_users),
    )
    if dataset.labels.sum() == 0:
        logger.warning(
            "All user labels are 0 (clean dataset). "
            "Detection metrics will be uninformative until attacks are injected."
        )

    # ── Model, optimiser, scheduler ──────────────────────────────────────────
    model = _build_model(dataset, args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    # ── Node features (kept on CPU; sliced by khop) ──────────────────────────
    # review nodes use a dummy zero-feature vector (their content is BERT)
    review_feat_dim = model.hgt.input_projs["review"].in_features
    node_feats_full = {
        "user":   dataset.x_user,
        "item":   dataset.x_item,
        "review": torch.zeros(dataset.num_reviews, review_feat_dim),
    }

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_f1     = -1.0
    patience_count  = 0
    wall_t0         = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        # Shuffle train users each epoch
        perm = torch.randperm(len(train_users))
        shuffled = train_users[perm]

        for start in range(0, len(shuffled), args.batch_size):
            batch_globals = shuffled[start : start + args.batch_size]

            (
                sub_feats, sub_ei, sub_ea, sub_nt,
                sub_rev_embs, sub_item_embs,
                sub_rev_user, sub_rev_item,
                sub_ts, local_seed,
            ) = _khop_subgraph(
                batch_globals,
                node_feats_full,
                dataset.edge_index_dict,
                dataset.edge_attr_dict,
                None,
                dataset.review_user_idx,
                dataset.review_item_idx,
                dataset.timestamps.float(),
                review_embs,
                item_embs,
                k=args.khop,
                device=device,
            )

            optimizer.zero_grad()
            scores, _ = model(
                sub_feats, sub_ei, sub_ea,
                sub_rev_embs, sub_item_embs,
                sub_rev_user, sub_rev_item,
                sub_ts,
            )
            batch_scores = scores[local_seed]
            batch_labels = dataset.labels[batch_globals].to(device)

            loss = focal_loss(
                batch_scores, batch_labels,
                gamma=args.focal_gamma, alpha=args.focal_alpha,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──────────────────────────────────────────────────────
        if epoch % args.eval_every == 0:
            val_metrics = _evaluate(
                model, val_users, dataset,
                review_embs, item_embs,
                device=device, batch_size=args.batch_size, khop=args.khop,
            )
            val_f1  = val_metrics["f1"]
            val_dr  = val_metrics["dr"]
            val_far = val_metrics["far"]

            elapsed = time.perf_counter() - wall_t0
            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_F1=%.4f  DR=%.4f  FAR=%.4f  lr=%.2e  t=%ds",
                epoch, args.epochs, avg_loss, val_f1, val_dr, val_far,
                scheduler.get_last_lr()[0], int(elapsed),
            )

            if use_wandb:
                import wandb  # noqa: F811
                wandb.log({
                    "epoch":    epoch,
                    "train/loss": avg_loss,
                    "val/f1":   val_f1,
                    "val/dr":   val_dr,
                    "val/far":  val_far,
                    "lr":       scheduler.get_last_lr()[0],
                })

            # Checkpoint if improved
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_count = 0
                torch.save(
                    {
                        "epoch":      epoch,
                        "model":      model.state_dict(),
                        "optimizer":  optimizer.state_dict(),
                        "scheduler":  scheduler.state_dict(),
                        "val_f1":     val_f1,
                        "val_dr":     val_dr,
                        "val_far":    val_far,
                        "args":       vars(args),
                        "run_id":     run_id,
                    },
                    ckpt_path,
                )
                logger.info("  ✓ Checkpoint saved (val_F1 improved to %.4f)", val_f1)
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    logger.info(
                        "Early stopping at epoch %d (no val_F1 improvement for %d epochs)",
                        epoch, args.patience,
                    )
                    break

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Loading best checkpoint from %s", ckpt_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        logger.warning("No checkpoint found; evaluating with final model weights")

    test_metrics = _evaluate(
        model, test_users, dataset,
        review_embs, item_embs,
        device=device, batch_size=args.batch_size, khop=args.khop,
    )
    logger.info(
        "Test  — F1=%.4f  DR=%.4f  FAR=%.4f",
        test_metrics["f1"], test_metrics["dr"], test_metrics["far"],
    )

    if use_wandb:
        import wandb  # noqa: F811
        wandb.log({
            "test/f1":  test_metrics["f1"],
            "test/dr":  test_metrics["dr"],
            "test/far": test_metrics["far"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
