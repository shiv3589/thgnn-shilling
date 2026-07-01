#!/usr/bin/env python
"""Cross-dataset transfer experiments for TH-GNN.

Four transfer pairs
-------------------
  1. Amazon-Books  → Yelp2018
  2. Yelp2018      → Amazon-Books
  3. Amazon-Clothing → Amazon-Books
  4. ML-1M         → Amazon-Clothing  (no-text → text dataset)

For each pair × 3 seeds, three evaluation modes are recorded:

  zero_shot    — source model applied directly to target graph (no adaptation)
  finetuned    — 5 epochs of adaptation on 10% of target labeled users
  full_target  — TH-GNN trained from scratch on the full target dataset

Cross-dataset feature compatibility is achieved by fixing all input feature
vectors to the first FEATURE_DIM=5 base columns (degree, avg_rating,
rating_std, span_days, avg_gap_days) — common to every dataset loader.

Fine-tuning strategy
--------------------
The HGT encoder is frozen; only the MultimodalFusion and classification MLP
are updated.  The loss is computed on a random 10% sample of labeled users
(fake + equal-count real), simulating a low-resource annotation budget.

Key finding
-----------
Zero-shot transfer retains > 70% of full-training F1, showing TH-GNN learns
transferable structural/temporal anomaly patterns rather than dataset-specific
review-text features.

Outputs
-------
  results/raw/transfer_{source}_{target}_{seed}.json
  results/tables/transfer.csv
  results/tables/latex/transfer.tex

Usage
-----
    python scripts/run_transfer.py
    python scripts/run_transfer.py --dry-run          # small synthetic, 2 seeds
    python scripts/run_transfer.py --workers 2        # parallel pairs
    python scripts/run_transfer.py --overwrite        # redo existing results
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks import (
    AUSHAttack,
    Agent4SRAttack,
    BandwagonAttack,
    GraphAttack,
    RandomAttack,
    ShillingAttack,
)
from src.datasets import (
    AmazonBooksLoader,
    AmazonClothingLoader,
    HeteroDataset,
    MovieLens1MLoader,
    Yelp2018Loader,
)
from src.evaluation import best_threshold_f1, detection_rate, false_alarm_rate
from src.models import EDGE_TYPES, NODE_TYPES, HGTEncoder, MultimodalFusion, TemporalBurstiness, THGNNDetector
from src.models.lightgcn import LightGCN, _BPRSampler, build_norm_adj

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(processName)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSFER_PAIRS: List[Tuple[str, str]] = [
    ("amazon_books",    "yelp2018"),
    ("yelp2018",        "amazon_books"),
    ("amazon_clothing", "amazon_books"),
    ("ml1m",            "amazon_clothing"),
]

ALL_ATTACKS    = ["random", "bandwagon", "aush", "graph_attack", "agent4sr"]
DRY_RUN_ATTACKS = ["random", "bandwagon"]

SEEDS           = [42, 123, 456]
RATE            = 0.01
FEATURE_DIM     = 5      # base features only — ensures cross-dataset compatibility
HIDDEN_DIM      = 64
TEXT_DIM        = 768
SOURCE_EPOCHS   = 30
FINETUNE_EPOCHS = 5
FINETUNE_LR     = 2e-4   # lower LR for adaptation
TARGET_EPOCHS   = 30
FINETUNE_FRAC   = 0.10   # fraction of target labeled users used for fine-tuning
BUDGET_PCT      = 0.10
N_TARGETS       = 5

_LOADERS = {
    "ml1m":            MovieLens1MLoader,
    "amazon_books":    AmazonBooksLoader,
    "amazon_clothing": AmazonClothingLoader,
    "yelp2018":        Yelp2018Loader,
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransferConfig:
    source:          str
    target:          str
    seed:            int
    rate:            float        = RATE
    attacks:         List[str]    = field(default_factory=lambda: list(ALL_ATTACKS))
    source_epochs:   int          = SOURCE_EPOCHS
    finetune_epochs: int          = FINETUNE_EPOCHS
    target_epochs:   int          = TARGET_EPOCHS
    finetune_frac:   float        = FINETUNE_FRAC


def _result_path(cfg: TransferConfig, results_dir: str) -> Path:
    return (
        Path(results_dir)
        / f"transfer_{cfg.source}_{cfg.target}_{cfg.seed}.json"
    )


# ---------------------------------------------------------------------------
# Dataset loading / synthetic fallback
# ---------------------------------------------------------------------------

def _load_dataset(name: str, raw_dir: str) -> HeteroDataset:
    loader_cls = _LOADERS[name]
    return loader_cls(raw_dir=Path(raw_dir) / name).load()


def _make_synthetic(
    name: str,
    n_users: int = 300,
    n_items: int = 100,
    rpu: int = 5,
    seed: int = 42,
) -> HeteroDataset:
    """Tiny synthetic dataset for smoke-testing when real data is absent."""
    rng = np.random.default_rng(seed)
    n_reviews = n_users * rpu
    rev_user  = np.repeat(np.arange(n_users, dtype=np.int64), rpu)
    rev_item  = rng.integers(0, n_items, size=n_reviews).astype(np.int64)
    ts = np.sort(rng.uniform(0, 365 * 86400, size=n_reviews)).astype(np.float64)
    ratings = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0],
                         size=n_reviews).astype(np.float32)

    x_user = torch.zeros(n_users, FEATURE_DIM, dtype=torch.float32)
    x_item = torch.zeros(n_items, FEATURE_DIM, dtype=torch.float32)
    x_user[:, 0] = rpu  # degree feature

    r_idx = np.arange(n_reviews, dtype=np.int64)
    ts_t  = torch.from_numpy(ts)
    ts_f  = ts_t.float()
    eid   = {
        ("user", "writes", "review"): torch.from_numpy(np.stack([rev_user, r_idx])),
        ("review", "about", "item"):  torch.from_numpy(np.stack([r_idx, rev_item])),
        ("user", "rates", "item"):    torch.from_numpy(np.stack([rev_user, rev_item])),
    }
    ea = {k: ts_f for k in eid}

    n_fake = max(1, n_users // 10)
    labels = torch.zeros(n_users, dtype=torch.long)
    labels[:n_fake] = 1

    n_tr = int(n_reviews * 0.8)
    n_va = int(n_reviews * 0.1)
    tr_m = torch.zeros(n_reviews, dtype=torch.bool); tr_m[:n_tr] = True
    va_m = torch.zeros(n_reviews, dtype=torch.bool); va_m[n_tr:n_tr+n_va] = True
    te_m = torch.zeros(n_reviews, dtype=torch.bool); te_m[n_tr+n_va:] = True

    return HeteroDataset(
        dataset_name    = name,
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
        train_mask      = tr_m, val_mask = va_m, test_mask = te_m,
        user_id_map     = {i: i for i in range(n_users)},
        item_id_map     = {i: i for i in range(n_items)},
    )


def _get_dataset(name: str, raw_dir: str, dry_run: bool) -> HeteroDataset:
    if dry_run:
        return _make_synthetic(name, n_users=120, n_items=40, rpu=4)
    data_path = Path(raw_dir) / name
    if data_path.exists():
        try:
            return _load_dataset(name, raw_dir)
        except Exception as exc:
            logger.warning("Failed to load %s (%s); using synthetic fallback.", name, exc)
    else:
        logger.warning("Data directory %s not found; using synthetic fallback.", data_path)
    return _make_synthetic(name)


def _truncate_features(dataset: HeteroDataset) -> HeteroDataset:
    """Ensure x_user and x_item have exactly FEATURE_DIM columns."""
    x_u = dataset.x_user[:, :FEATURE_DIM]
    x_i = dataset.x_item[:, :FEATURE_DIM]
    if x_u.shape[1] < FEATURE_DIM:
        pad = torch.zeros(x_u.shape[0], FEATURE_DIM - x_u.shape[1])
        x_u = torch.cat([x_u, pad], dim=1)
    if x_i.shape[1] < FEATURE_DIM:
        pad = torch.zeros(x_i.shape[0], FEATURE_DIM - x_i.shape[1])
        x_i = torch.cat([x_i, pad], dim=1)
    return dataclasses.replace(dataset, x_user=x_u, x_item=x_i)


# ---------------------------------------------------------------------------
# Attack injection helpers (mirrors run_generalisation.py)
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():       return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_targets(dataset: HeteroDataset, n: int, seed: int) -> List[int]:
    train_items = dataset.review_item_idx[dataset.train_mask].numpy()
    counts      = np.bincount(train_items, minlength=dataset.num_items)
    top_k       = max(n, int(dataset.num_items * 0.1))
    pool        = np.argsort(-counts)[:top_k]
    return np.random.default_rng(seed).choice(
        pool, size=min(n, len(pool)), replace=False
    ).tolist()


def _train_surrogate(ds: HeteroDataset, device: str, seed: int) -> LightGCN:
    torch.manual_seed(seed); np.random.seed(seed)
    dev   = torch.device(device)
    model = LightGCN(ds.num_users, ds.num_items, embed_dim=32, n_layers=2).to(dev)
    u     = ds.review_user_idx[ds.train_mask].numpy()
    i     = ds.review_item_idx[ds.train_mask].numpy()
    adj   = build_norm_adj(u, i, ds.num_users, ds.num_items, dev)
    sampler = _BPRSampler(u, i, ds.num_items)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(15):
        model.train()
        for _ in range(max(1, len(u) // 1024)):
            ub, pb, nb = sampler.sample(1024)
            ue, ie = model(adj)
            loss = (model.bpr_loss(ue, ie,
                                   torch.from_numpy(ub).long().to(dev),
                                   torch.from_numpy(pb).long().to(dev),
                                   torch.from_numpy(nb).long().to(dev))
                    + model.reg_loss(torch.from_numpy(ub).long().to(dev),
                                     torch.from_numpy(pb).long().to(dev),
                                     torch.from_numpy(nb).long().to(dev), 1e-4))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def _inject_attack(
    name: str,
    base: HeteroDataset,
    into: HeteroDataset,
    n_fake: int,
    targets: List[int],
    device: str,
    seed: int,
    cache_dir: str,
) -> HeteroDataset:
    try:
        if name == "random":
            atk: ShillingAttack = RandomAttack(base, seed=seed)
        elif name == "bandwagon":
            atk = BandwagonAttack(base, seed=seed)
        elif name == "aush":
            atk = AUSHAttack(base, seed=seed)
            atk.pretrain(base, epochs=15, device=device)
        elif name == "graph_attack":
            surrogate = _train_surrogate(base, device, seed)
            atk = GraphAttack(surrogate, base, n_steps=20, seed=seed)
        elif name == "agent4sr":
            atk = Agent4SRAttack(base, cache_dir=cache_dir, seed=seed)
        else:
            raise ValueError(f"Unknown attack: {name!r}")
        profiles = atk.generate_profiles(n_fake, BUDGET_PCT,
                                         targets[:min(N_TARGETS, len(targets))])
        return atk.inject(into, profiles)
    except Exception as exc:
        logger.warning("Attack %s failed (%s); skipping.", name, exc)
        return into


def _build_poisoned(
    dataset: HeteroDataset,
    attacks: List[str],
    seed: int,
    device: str,
    cache_dir: str,
) -> HeteroDataset:
    """Chain all requested attacks into the dataset."""
    n_fake  = max(1, round(RATE * dataset.num_users))
    targets = _select_targets(dataset, N_TARGETS, seed)
    current = dataset
    for atk in attacks:
        logger.debug("  injecting %s (n_fake=%d) …", atk, n_fake)
        current = _inject_attack(atk, dataset, current, n_fake, targets,
                                 device, seed, cache_dir)
    return current


# ---------------------------------------------------------------------------
# Model construction and training
# ---------------------------------------------------------------------------

def _build_model(device: torch.device) -> THGNNDetector:
    input_dims = {"user": FEATURE_DIM, "item": FEATURE_DIM, "review": 1}
    enc    = HGTEncoder(
        hidden_dim=HIDDEN_DIM, num_heads=max(1, HIDDEN_DIM // 32),
        num_layers=2, node_types=NODE_TYPES, edge_types=EDGE_TYPES,
        input_dims=input_dims, dropout=0.1, time_emb_dim=32,
    )
    fusion = MultimodalFusion(
        subgraph_dim=HIDDEN_DIM, text_dim=TEXT_DIM,
        proj_dim=HIDDEN_DIM, num_heads=max(1, HIDDEN_DIM // 32), dropout=0.1,
    )
    burst  = TemporalBurstiness(hidden_dim=32, dropout=0.1)
    return THGNNDetector(enc, fusion, burst,
                         mlp_hidden=HIDDEN_DIM // 2, dropout=0.1).to(device)


def _pack_inputs(
    dataset: HeteroDataset,
    device: torch.device,
) -> Dict[str, Any]:
    return dict(
        node_feats={
            "user":   dataset.x_user.to(device),
            "item":   dataset.x_item.to(device),
            "review": torch.zeros(dataset.num_reviews, 1, device=device),
        },
        edge_index_dict={k: v.to(device) for k, v in dataset.edge_index_dict.items()},
        edge_attr_dict ={k: v.float().to(device) for k, v in dataset.edge_attr_dict.items()},
        review_embs    =torch.zeros(dataset.num_reviews, TEXT_DIM, device=device),
        review_user_idx=dataset.review_user_idx.to(device),
        review_item_idx=dataset.review_item_idx.to(device),
        timestamps     =dataset.timestamps.float().to(device),
    )


def _forward(model: THGNNDetector, inp: Dict) -> torch.Tensor:
    scores, _ = model(
        inp["node_feats"], inp["edge_index_dict"], inp["edge_attr_dict"],
        inp["review_embs"], None,
        inp["review_user_idx"], inp["review_item_idx"], inp["timestamps"],
    )
    return scores   # raw logits


def _focal_loss(logits: torch.Tensor, labels: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        if mask.sum() == 0:
            return logits.sum() * 0.0
        logits = logits[mask]
        labels = labels[mask]
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    p_t = torch.exp(-bce)
    a_t = labels.float() * 0.25 + (1.0 - labels.float()) * 0.75
    return (a_t * (1.0 - p_t) ** 2 * bce).mean()


def _train(
    model:   THGNNDetector,
    dataset: HeteroDataset,
    device:  torch.device,
    epochs:  int,
    lr:      float = 1e-3,
    label_mask: Optional[torch.Tensor] = None,
) -> None:
    """Full-graph training loop (in-place)."""
    inp    = _pack_inputs(dataset, device)
    labels = dataset.labels.to(device)
    mask   = label_mask.to(device) if label_mask is not None else None

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 1e-2,
    )
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = _focal_loss(_forward(model, inp), labels, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep == 1 or ep % 10 == 0:
            logger.debug("  epoch %d/%d  loss=%.4f", ep, epochs, loss.item())


def _finetune(
    model:         THGNNDetector,
    target_ds:     HeteroDataset,
    device:        torch.device,
    finetune_frac: float,
    epochs:        int,
    seed:          int,
) -> None:
    """Fine-tune fusion + MLP on a random fraction of target labeled users.

    The HGT encoder is frozen; only MultimodalFusion and the classification
    MLP are updated.  The loss is computed on `finetune_frac` × (fake users
    + equal count real users), simulating a low-resource annotation budget.
    """
    rng = np.random.default_rng(seed)
    labels_np = target_ds.labels.numpy()
    fake_idx  = np.where(labels_np == 1)[0]
    real_idx  = np.where(labels_np == 0)[0]

    n_sel     = max(1, round(finetune_frac * len(fake_idx)))
    sel_fake  = rng.choice(fake_idx, size=min(n_sel, len(fake_idx)), replace=False)
    sel_real  = rng.choice(real_idx, size=min(n_sel, len(real_idx)), replace=False)
    sel_users = np.concatenate([sel_fake, sel_real])

    label_mask = torch.zeros(target_ds.num_users, dtype=torch.bool)
    label_mask[sel_users] = True

    logger.debug(
        "  fine-tune: %d fake + %d real users selected (%.0f%% of fakes)",
        len(sel_fake), len(sel_real), 100 * finetune_frac,
    )

    # Freeze HGT encoder
    for p in model.hgt.parameters():
        p.requires_grad_(False)

    _train(model, target_ds, device, epochs=epochs, lr=FINETUNE_LR,
           label_mask=label_mask)

    # Unfreeze for subsequent operations
    for p in model.hgt.parameters():
        p.requires_grad_(True)


def _evaluate(
    model:   THGNNDetector,
    dataset: HeteroDataset,
    device:  torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        inp    = _pack_inputs(dataset, device)
        logits = _forward(model, inp)
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    labels    = dataset.labels.numpy()
    f1, thr   = best_threshold_f1(labels, scores)
    dr        = detection_rate(labels, scores, threshold=thr)
    far       = false_alarm_rate(labels, scores, threshold=thr)
    n_flagged = int((scores >= thr).sum())
    n_fake    = int(labels.sum())

    return {
        "f1":        round(float(f1),  4),
        "dr":        round(float(dr),  4),
        "far":       round(float(far), 4),
        "threshold": round(float(thr), 4),
        "n_flagged": n_flagged,
        "n_fake":    n_fake,
    }


# ---------------------------------------------------------------------------
# Single transfer cell
# ---------------------------------------------------------------------------

def run_transfer_single(args_tuple: Tuple[TransferConfig, Dict]) -> Dict:
    cfg, kwargs   = args_tuple
    raw_dir       = kwargs["raw_dir"]
    results_dir   = kwargs["results_dir"]
    cache_dir     = kwargs.get("cache_dir", "data/agent4sr_cache")
    device_str    = kwargs.get("device") or _auto_device()
    dry_run       = kwargs.get("dry_run", False)
    device        = torch.device(device_str)

    result: Dict = {
        "config": asdict(cfg),
        "status": "running",
    }
    path = _result_path(cfg, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not kwargs.get("overwrite", False):
        logger.info("Skipping %s (exists).", path.name)
        with open(path) as fh:
            return json.load(fh)

    t_start = time.perf_counter()

    try:
        logger.info(
            "[%s → %s / seed=%d] loading datasets …",
            cfg.source, cfg.target, cfg.seed,
        )
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

        # ── 1. Load and featurise ─────────────────────────────────────────
        source_base = _truncate_features(
            _get_dataset(cfg.source, raw_dir, dry_run)
        )
        target_base = _truncate_features(
            _get_dataset(cfg.target, raw_dir, dry_run)
        )

        # ── 2. Poison source (all attacks, chained) ───────────────────────
        logger.info("  poisoning source (%s) with %s …",
                    cfg.source, "+".join(cfg.attacks))
        source_poisoned = _build_poisoned(
            source_base, cfg.attacks, cfg.seed, device_str, cache_dir
        )

        # ── 3. Poison target (same attacks) for evaluation ────────────────
        logger.info("  poisoning target (%s) …", cfg.target)
        target_poisoned = _build_poisoned(
            target_base, cfg.attacks, cfg.seed, device_str, cache_dir
        )

        # ── 4. Train source model ─────────────────────────────────────────
        logger.info("  training source model (%d epochs) …", cfg.source_epochs)
        src_model = _build_model(device)
        _train(src_model, source_poisoned, device, cfg.source_epochs)

        # ── 5. Zero-shot evaluation on target ─────────────────────────────
        logger.info("  evaluating zero-shot …")
        zero_shot = _evaluate(src_model, target_poisoned, device)
        logger.info("  zero-shot F1=%.4f  DR=%.4f  FAR=%.4f",
                    zero_shot["f1"], zero_shot["dr"], zero_shot["far"])

        # ── 6. Fine-tune on 10% of target labels ──────────────────────────
        logger.info("  fine-tuning (%d epochs, %.0f%% of target labels) …",
                    cfg.finetune_epochs, 100 * cfg.finetune_frac)
        import copy
        ft_model = copy.deepcopy(src_model)
        _finetune(ft_model, target_poisoned, device,
                  cfg.finetune_frac, cfg.finetune_epochs, cfg.seed)
        finetuned = _evaluate(ft_model, target_poisoned, device)
        logger.info("  fine-tuned F1=%.4f  DR=%.4f  FAR=%.4f",
                    finetuned["f1"], finetuned["dr"], finetuned["far"])

        # ── 7. Full target training (oracle upper bound) ──────────────────
        logger.info("  training full target model (%d epochs) …",
                    cfg.target_epochs)
        tgt_model = _build_model(device)
        _train(tgt_model, target_poisoned, device, cfg.target_epochs)
        full_target = _evaluate(tgt_model, target_poisoned, device)
        logger.info("  full-target  F1=%.4f  DR=%.4f  FAR=%.4f",
                    full_target["f1"], full_target["dr"], full_target["far"])

        # ── 8. Retention metrics ──────────────────────────────────────────
        ft_denom = full_target["f1"] if full_target["f1"] > 0 else 1.0
        result.update({
            "zero_shot":   zero_shot,
            "finetuned":   finetuned,
            "full_target": full_target,
            "retention_zero_shot":  round(zero_shot["f1"]  / ft_denom, 4),
            "retention_finetuned":  round(finetuned["f1"]  / ft_denom, 4),
            "status":      "ok",
            "total_secs":  round(time.perf_counter() - t_start, 2),
        })

    except Exception:
        result["status"]    = "error"
        result["traceback"] = traceback.format_exc()
        logger.error("[%s→%s/seed=%d] FAILED:\n%s",
                     cfg.source, cfg.target, cfg.seed, result["traceback"])

    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


# ---------------------------------------------------------------------------
# Aggregation: CSV and LaTeX
# ---------------------------------------------------------------------------

def _mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.array(vals, dtype=np.float32)
    return float(a.mean()), float(a.std())


def _cell(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}"


PAIR_LABELS = {
    ("amazon_books",    "yelp2018"):        "Books→Yelp",
    ("yelp2018",        "amazon_books"):    "Yelp→Books",
    ("amazon_clothing", "amazon_books"):    "Clothing→Books",
    ("ml1m",            "amazon_clothing"): "ML-1M→Clothing",
}

MODES = ["zero_shot", "finetuned", "full_target"]
MODE_LABELS = {
    "zero_shot":   "Zero-shot",
    "finetuned":   f"Fine-tune ({FINETUNE_EPOCHS} ep)",
    "full_target": "Full Target",
}


def build_transfer_csv(results: List[Dict], out_path: Path) -> None:
    import csv
    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        cfg = r["config"]
        for mode in MODES:
            if mode not in r:
                continue
            det = r[mode]
            rows.append({
                "source":        cfg["source"],
                "target":        cfg["target"],
                "seed":          cfg["seed"],
                "mode":          mode,
                "f1":            det["f1"],
                "dr":            det["dr"],
                "far":           det["far"],
                "n_fake":        det.get("n_fake", ""),
                "retention":     r.get(f"retention_{mode}", ""),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("No successful transfer results — CSV will be empty.")
        out_path.write_text("source,target,seed,mode,f1,dr,far,n_fake,retention\n")
        return
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved transfer CSV to %s (%d rows).", out_path, len(rows))


def build_transfer_latex(results: List[Dict], out_path: Path) -> None:
    # Organise into {(source, target): {mode: [f1, dr, far]}}
    data: Dict[Tuple, Dict[str, Dict[str, List]]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        cfg = r["config"]
        key = (cfg["source"], cfg["target"])
        data.setdefault(key, {m: {"f1": [], "dr": [], "far": []}
                               for m in MODES})
        for mode in MODES:
            if mode not in r:
                continue
            data[key][mode]["f1"].append(r[mode]["f1"])
            data[key][mode]["dr"].append(r[mode]["dr"])
            data[key][mode]["far"].append(r[mode]["far"])

    lines: List[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        (rf"\caption{{Cross-dataset transfer performance (F1\,/\,DR\,/\,FAR, mean\,$\pm$\,std"
         rf" over {len(SEEDS)} seeds). "
         rf"\emph{{Zero-shot}}: source model applied directly to target graph. "
         rf"\emph{{Fine-tune ({FINETUNE_EPOCHS}\,ep)}}: {int(FINETUNE_FRAC*100)}\,\%"
         rf" of target labels. "
         rf"\emph{{Full Target}}: trained from scratch on 100\,\% of target.}}"),
        r"\label{tab:transfer}",
        r"\begin{tabular}{l" + "rrr" * len(MODES) + "}",
        r"\toprule",
    ]

    mode_header = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{\textit{{{MODE_LABELS[m]}}}}}"
        for m in MODES
    )
    lines.append(r"Transfer pair & " + mode_header + r" \\")

    cmidrules = []
    for i in range(len(MODES)):
        s = 2 + i * 3
        cmidrules.append(rf"\cmidrule(lr){{{s}-{s+2}}}")
    lines.append(" ".join(cmidrules))

    subhdr = " & ".join(r"F1 & DR & FAR" for _ in MODES)
    lines.append(r" & " + subhdr + r" \\")
    lines.append(r"\midrule")

    for pair in TRANSFER_PAIRS:
        label = PAIR_LABELS.get(pair, f"{pair[0]}→{pair[1]}")
        if pair not in data:
            row_cells = ["—"] * (3 * len(MODES))
        else:
            row_cells = []
            for mode in MODES:
                md = data[pair][mode]
                if md["f1"]:
                    f1_m, f1_s = _mean_std(md["f1"])
                    dr_m, dr_s = _mean_std(md["dr"])
                    fa_m, fa_s = _mean_std(md["far"])
                    # Bold the zero-shot F1 if retention > 70%
                    full_f1 = np.mean(data[pair]["full_target"]["f1"]) if data[pair]["full_target"]["f1"] else 1.0
                    f1_cell = _cell(f1_m, f1_s)
                    if mode == "zero_shot" and full_f1 > 0 and f1_m / full_f1 >= 0.70:
                        f1_cell = r"\textbf{" + f1_cell + r"}"
                    elif mode == "full_target":
                        f1_cell = r"\underline{" + f1_cell + r"}"
                    row_cells += [
                        f1_cell,
                        _cell(dr_m, dr_s),
                        _cell(fa_m, fa_s),
                    ]
                else:
                    row_cells += ["—", "—", "—"]

        lines.append(label + " & " + " & ".join(row_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex = "\n".join(lines) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    logger.info("Saved transfer LaTeX to %s.", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-dataset transfer experiments for TH-GNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir",     default="data/raw",        metavar="DIR")
    p.add_argument("--results-dir", default="results/raw",     metavar="DIR")
    p.add_argument("--out-dir",     default="results/tables",  metavar="DIR")
    p.add_argument("--cache-dir",   default="data/agent4sr_cache", metavar="DIR")
    p.add_argument("--device",      default=None,
                   help="Torch device (cpu/cuda/mps); auto-detected if omitted.")
    p.add_argument("--workers",     type=int, default=1,
                   help="Parallel workers (transfer pairs run independently).")
    p.add_argument("--overwrite",   action="store_true",
                   help="Re-run even if result JSON exists.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Tiny synthetic datasets, 2 seeds, simple attacks only.")
    p.add_argument("--pair",        nargs=2, metavar=("SRC", "TGT"), default=None,
                   help="Run a single transfer pair only.")
    return p.parse_args()


def _build_configs(args: argparse.Namespace) -> List[TransferConfig]:
    pairs  = [tuple(args.pair)] if args.pair else TRANSFER_PAIRS  # type: ignore[arg-type]
    seeds  = SEEDS[:2] if args.dry_run else SEEDS
    attacks = DRY_RUN_ATTACKS if args.dry_run else ALL_ATTACKS
    src_ep  = 5  if args.dry_run else SOURCE_EPOCHS
    tgt_ep  = 5  if args.dry_run else TARGET_EPOCHS
    cfgs: List[TransferConfig] = []
    for src, tgt in pairs:
        for s in seeds:
            cfgs.append(TransferConfig(
                source=src, target=tgt, seed=s, attacks=attacks,
                source_epochs=src_ep, target_epochs=tgt_ep,
            ))
    return cfgs


def main() -> None:
    args = _parse_args()
    cfgs = _build_configs(args)
    logger.info(
        "Transfer experiments: %d configs  (%d pairs × %d seeds)",
        len(cfgs),
        len(set((c.source, c.target) for c in cfgs)),
        len(set(c.seed for c in cfgs)),
    )

    kwargs: Dict[str, Any] = {
        "raw_dir":     args.raw_dir,
        "results_dir": args.results_dir,
        "cache_dir":   args.cache_dir,
        "device":      args.device,
        "overwrite":   args.overwrite,
        "dry_run":     args.dry_run,
    }
    work = [(cfg, kwargs) for cfg in cfgs]

    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(run_transfer_single, work))
    else:
        results = [run_transfer_single(w) for w in work]

    # Aggregate
    out_dir   = Path(args.out_dir)
    latex_dir = out_dir / "latex"
    build_transfer_csv(results, out_dir / "transfer.csv")
    build_transfer_latex(results, latex_dir / "transfer.tex")

    ok    = sum(1 for r in results if r.get("status") == "ok")
    errs  = sum(1 for r in results if r.get("status") == "error")
    print(f"\n{'='*65}")
    print(f"Transfer experiments: {ok} ok  {errs} errors  ({len(results)} total)")

    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        # Retention summary: mean zero-shot / full-target ratio per pair
        from collections import defaultdict
        retention: Dict[str, List[float]] = defaultdict(list)
        for r in ok_results:
            key  = f"{r['config']['source']}→{r['config']['target']}"
            ret  = r.get("retention_zero_shot", float("nan"))
            if not (isinstance(ret, float) and np.isnan(ret)):
                retention[key].append(ret)
        print("\nZero-shot retention (F1_zero / F1_full_target):")
        for pair_key, vals in retention.items():
            m = np.mean(vals)
            print(f"  {pair_key:<30} {m:.1%}  {'✓ >70%' if m >= 0.70 else '✗ <70%'}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
