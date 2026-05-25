#!/usr/bin/env python
"""Ablation experiment runner for THGNNDetector variants.

Evaluates four detector variants across all datasets and attacks at a fixed
injection rate of 1 %:

  Variants          : thgnn (full), thgnn_no_text, thgnn_no_temporal, thgnn_static_gcn
  Datasets          : ml1m, amazon_books, amazon_clothing, yelp2018
  Attacks           : random, bandwagon, aush, graph_attack, agent4sr
  Injection rate    : 0.01 (fixed)
  Seeds             : 42, 123, 456
  Total experiments : 4 × 5 × 4 × 3 = 240

Results are written to ``results/raw/ablation_{dataset}_{attack}_{variant}_{seed}.json``
using the same JSON schema as run_experiments.py so that aggregate_results.py
can process them via its ablation table.

Usage
-----
    # Smoke test (ml1m × random, 2 seeds, full + no_text only)
    python scripts/run_ablation.py --dry-run

    # Full run, 2 parallel workers
    python scripts/run_ablation.py --workers 2

    # Restrict to specific axes
    python scripts/run_ablation.py --dataset ml1m --variant thgnn thgnn_no_text
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Project root on sys.path ──────────────────────────────────────────────────
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
from src.detectors import AblationWrapper, THGNNWrapper
from src.evaluation import best_threshold_f1, detection_rate, false_alarm_rate
from src.models.lightgcn import LightGCN, _BPRSampler, build_norm_adj, train_lightgcn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(processName)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

DATASETS:  List[str] = ["ml1m", "amazon_books", "amazon_clothing", "yelp2018"]
ATTACKS:   List[str] = ["random", "bandwagon", "aush", "graph_attack", "agent4sr"]
SEEDS:     List[int] = [42, 123, 456]
RATE:      float     = 0.01  # fixed injection rate for ablation

ABLATION_VARIANTS: List[str] = [
    "thgnn",             # full model (baseline within ablation)
    "thgnn_no_text",     # no BERT text streams
    "thgnn_no_temporal", # no HGT temporal bias + no burstiness GRU
    "thgnn_static_gcn",  # homogeneous 2-layer GCN backbone
]

N_TARGETS  = 5
BUDGET_PCT = 0.10

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
class AblationConfig:
    dataset: str
    attack:  str
    variant: str   # one of ABLATION_VARIANTS
    seed:    int
    rate:    float = RATE


def _result_path(cfg: AblationConfig, results_dir: str) -> Path:
    return (
        Path(results_dir)
        / f"ablation_{cfg.dataset}_{cfg.attack}_{cfg.variant}_{cfg.seed}.json"
    )


def _build_configs(args: argparse.Namespace) -> List[AblationConfig]:
    if args.dry_run:
        datasets  = ["ml1m"]
        attacks   = ["random"]
        variants  = ["thgnn", "thgnn_no_text"]
        seeds     = SEEDS[:2]
    else:
        datasets  = DATASETS
        attacks   = ATTACKS
        variants  = ABLATION_VARIANTS
        seeds     = SEEDS

    return [
        AblationConfig(d, a, v, s)
        for d in datasets
        for a in attacks
        for v in variants
        for s in seeds
    ]


# ---------------------------------------------------------------------------
# Pipeline helpers (module-level for picklability)
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset(cfg: AblationConfig, raw_dir: str) -> HeteroDataset:
    return _LOADERS[cfg.dataset](raw_dir=raw_dir).load()


def _select_targets(dataset: HeteroDataset, n: int, seed: int) -> List[int]:
    """Sample n target items from the top-10 % most-interacted training items."""
    train_items = dataset.review_item_idx[dataset.train_mask].numpy()
    counts = np.bincount(train_items, minlength=dataset.num_items)
    top_k  = max(n, int(dataset.num_items * 0.1))
    pool   = np.argsort(-counts)[:top_k]
    rng    = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()


def _train_surrogate(
    dataset: HeteroDataset,
    device:  str,
    seed:    int,
    epochs:  int = 15,
) -> LightGCN:
    """Quick surrogate LightGCN for GraphAttack (embed_dim=32, 2 layers)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    model   = LightGCN(dataset.num_users, dataset.num_items, embed_dim=32, n_layers=2).to(dev)
    train_u = dataset.review_user_idx[dataset.train_mask].numpy()
    train_i = dataset.review_item_idx[dataset.train_mask].numpy()
    adj     = build_norm_adj(train_u, train_i, dataset.num_users, dataset.num_items, dev)
    sampler = _BPRSampler(train_u, train_i, dataset.num_items)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs      = 1024
    steps   = max(1, len(train_u) // bs)

    for _ in range(epochs):
        model.train()
        for _ in range(steps):
            u_np, pos_np, neg_np = sampler.sample(bs)
            u_t   = torch.from_numpy(u_np).long().to(dev)
            pos_t = torch.from_numpy(pos_np).long().to(dev)
            neg_t = torch.from_numpy(neg_np).long().to(dev)
            ue, ie = model(adj)
            loss = (
                model.bpr_loss(ue, ie, u_t, pos_t, neg_t)
                + model.reg_loss(u_t, pos_t, neg_t, 1e-4)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


def _build_attack(
    cfg: AblationConfig,
    dataset: HeteroDataset,
    device: str,
    agent_cache_dir: str,
) -> Tuple[ShillingAttack, float]:
    t0 = time.perf_counter()

    if cfg.attack == "random":
        attack: ShillingAttack = RandomAttack(dataset, seed=cfg.seed)

    elif cfg.attack == "bandwagon":
        attack = BandwagonAttack(dataset, seed=cfg.seed)

    elif cfg.attack == "aush":
        attack = AUSHAttack(dataset, seed=cfg.seed)
        attack.pretrain(dataset, epochs=20, device=device)

    elif cfg.attack == "graph_attack":
        surrogate = _train_surrogate(dataset, device, cfg.seed)
        attack = GraphAttack(surrogate, dataset, n_steps=20, seed=cfg.seed)

    elif cfg.attack == "agent4sr":
        cache = Path(agent_cache_dir) / "agent4sr"
        attack = Agent4SRAttack(dataset, cache_dir=str(cache), seed=cfg.seed)

    else:
        raise ValueError(f"Unknown attack: {cfg.attack!r}")

    return attack, time.perf_counter() - t0


def _build_detector(variant: str, seed: int):
    """Return a fresh (unfitted) ablation detector."""
    if variant == "thgnn":
        return THGNNWrapper(seed=seed)
    # Strip the "thgnn_" prefix to get the AblationWrapper variant key
    inner = variant[len("thgnn_"):]   # "no_text", "no_temporal", "static_gcn"
    return AblationWrapper(inner, seed=seed)


def _filter_dataset(
    dataset: HeteroDataset,
    flagged_users: np.ndarray,
) -> HeteroDataset:
    if len(flagged_users) == 0:
        return dataset
    flagged_set = set(int(u) for u in flagged_users)
    keep = torch.tensor(
        [u.item() not in flagged_set for u in dataset.review_user_idx],
        dtype=torch.bool,
    )
    return HeteroDataset(
        dataset_name    = dataset.dataset_name,
        num_users       = dataset.num_users,
        num_items       = dataset.num_items,
        num_reviews     = dataset.num_reviews,
        edge_index_dict = dataset.edge_index_dict,
        edge_attr_dict  = dataset.edge_attr_dict,
        x_user          = dataset.x_user,
        x_item          = dataset.x_item,
        review_texts    = dataset.review_texts,
        ratings         = dataset.ratings,
        timestamps      = dataset.timestamps,
        review_user_idx = dataset.review_user_idx,
        review_item_idx = dataset.review_item_idx,
        labels          = dataset.labels,
        train_mask      = dataset.train_mask & keep,
        val_mask        = dataset.val_mask   & keep,
        test_mask       = dataset.test_mask  & keep,
        user_id_map     = dataset.user_id_map,
        item_id_map     = dataset.item_id_map,
    )


# ---------------------------------------------------------------------------
# Single-experiment pipeline (module-level → picklable)
# ---------------------------------------------------------------------------

def run_ablation_single(args_tuple: Tuple[AblationConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Run one (dataset, attack, variant, seed) cell of the ablation matrix.

    Always writes a result JSON; returns the result dict.
    On error, result["status"] == "error" with a traceback.
    """
    cfg, kwargs = args_tuple
    raw_dir     = kwargs["raw_dir"]
    results_dir = kwargs["results_dir"]
    cache_dir   = kwargs["cache_dir"]
    device      = kwargs.get("device") or _auto_device()

    result: Dict[str, Any] = {
        "config": {**asdict(cfg), "detector": cfg.variant},
        "status": "running",
    }
    path = _result_path(cfg, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    try:
        # ── 1. Load dataset ───────────────────────────────────────────────
        logger.info(
            "[%s/%s/%s/seed=%d] loading dataset …",
            cfg.dataset, cfg.attack, cfg.variant, cfg.seed,
        )
        dataset = _load_dataset(cfg, raw_dir)

        # ── 2. Target items ───────────────────────────────────────────────
        targets = _select_targets(dataset, N_TARGETS, cfg.seed)
        n_fake  = max(1, round(cfg.rate * dataset.num_users))
        logger.info("  targets=%s  n_fake=%d", targets, n_fake)

        # ── 3. Attack ─────────────────────────────────────────────────────
        attack, t_attack = _build_attack(cfg, dataset, device, cache_dir)
        profiles = attack.generate_profiles(n_fake, BUDGET_PCT, targets)
        poisoned = attack.inject(dataset, profiles)
        logger.info("  injected %d fake users (attack_secs=%.1f)", len(profiles), t_attack)

        # ── 4. Fit ablation detector ──────────────────────────────────────
        detector = _build_detector(cfg.variant, cfg.seed)
        t0_det = time.perf_counter()
        detector.fit(poisoned, device=device, seed=cfg.seed)
        t_detect = time.perf_counter() - t0_det

        # ── 5. Score + evaluate detection ────────────────────────────────
        scores = detector.score_users(poisoned)
        labels = poisoned.labels.numpy()

        f1, threshold = best_threshold_f1(labels, scores)
        dr  = detection_rate(labels, scores,   threshold=threshold)
        far = false_alarm_rate(labels, scores, threshold=threshold)
        flagged       = np.where(scores >= threshold)[0]
        n_fake_actual = int(labels.sum())

        logger.info(
            "  detect_secs=%.1f  F1=%.4f  DR=%.4f  FAR=%.4f  flagged=%d/%d",
            t_detect, f1, dr, far, len(flagged), n_fake_actual,
        )
        result["detection"] = {
            "f1":          round(float(f1),        4),
            "dr":          round(float(dr),        4),
            "far":         round(float(far),       4),
            "threshold":   round(float(threshold), 4),
            "n_flagged":   int(len(flagged)),
            "n_fake":      n_fake_actual,
            "detect_secs": round(t_detect, 2),
        }

        # ── 6. Filter + retrain LightGCN ─────────────────────────────────
        filtered = _filter_dataset(poisoned, flagged)

        t0_lgcn = time.perf_counter()
        if filtered.train_mask.sum() == 0:
            logger.warning("  Empty train set after filtering — skipping LightGCN.")
            lgcn_results: Dict = {"ndcg@10": 0.0, "hr@10": 0.0, "best_epoch": 0}
        else:
            lgcn_results = train_lightgcn(
                filtered,
                epochs     = 50,
                embed_dim  = 32,
                n_layers   = 2,
                batch_size = 1024,
                eval_every = 5,
                patience   = 5,
                k          = 10,
                device     = device,
                seed       = cfg.seed,
            )
        t_lgcn = time.perf_counter() - t0_lgcn

        result["recommendation"] = {
            "ndcg@10":    round(float(lgcn_results.get("ndcg@10", 0)), 4),
            "hr@10":      round(float(lgcn_results.get("hr@10",   0)), 4),
            "best_epoch": int(lgcn_results.get("best_epoch", 0)),
            "lgcn_secs":  round(t_lgcn, 2),
        }
        logger.info(
            "  NDCG@10=%.4f  HR@10=%.4f  lgcn_secs=%.1f",
            result["recommendation"]["ndcg@10"],
            result["recommendation"]["hr@10"],
            t_lgcn,
        )

        result["status"]     = "ok"
        result["total_secs"] = round(time.perf_counter() - t_total, 2)

    except Exception:  # noqa: BLE001
        result["status"]    = "error"
        result["traceback"] = traceback.format_exc()
        logger.error(
            "[%s/%s/%s/%d] FAILED:\n%s",
            cfg.dataset, cfg.attack, cfg.variant, cfg.seed,
            result["traceback"],
        )

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Progress summary
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict]) -> None:
    ok     = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]
    skip   = [r for r in results if r.get("status") == "skipped"]
    print(f"\n{'='*60}")
    print(f"  Ablation complete: {len(ok)} ok  {len(errors)} errors  {len(skip)} skipped")

    if ok:
        # Per-variant summary
        by_variant: Dict[str, List[float]] = {}
        for r in ok:
            if "detection" not in r:
                continue
            v = r["config"].get("detector", r["config"].get("variant", "?"))
            by_variant.setdefault(v, []).append(r["detection"]["f1"])
        if by_variant:
            print("\n  Mean F1 per variant:")
            full_f1 = np.mean(by_variant.get("thgnn", [np.nan]))
            for v, f1s in sorted(by_variant.items()):
                mean_f1 = np.mean(f1s)
                delta   = mean_f1 - full_f1 if v != "thgnn" else 0.0
                arrow   = f"  ({delta:+.4f} vs full)" if v != "thgnn" else "  [baseline]"
                print(f"    {v:25s}  F1={mean_f1:.4f}{arrow}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the THGNN ablation matrix (rate=0.01 only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Quick smoke test: ml1m × random, 2 seeds, thgnn + thgnn_no_text.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (ProcessPoolExecutor). 1 = sequential.",
    )
    p.add_argument("--raw-dir",     default="data/raw",         metavar="DIR")
    p.add_argument("--cache-dir",   default="data/processed",   metavar="DIR")
    p.add_argument("--results-dir", default="results/raw",      metavar="DIR")
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="torch device: cpu | cuda | mps (autodetect if omitted).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-run configs whose result file already exists.",
    )
    p.add_argument(
        "--dataset", nargs="+", choices=DATASETS + ["all"], default=None,
    )
    p.add_argument(
        "--attack", nargs="+", choices=ATTACKS + ["all"], default=None,
    )
    p.add_argument(
        "--variant", nargs="+", choices=ABLATION_VARIANTS + ["all"], default=None,
    )
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = args.device or _auto_device()
    logger.info("Device: %s  Workers: %d", device, args.workers)

    configs = _build_configs(args)

    if args.dataset and "all" not in args.dataset:
        configs = [c for c in configs if c.dataset in args.dataset]
    if args.attack  and "all" not in args.attack:
        configs = [c for c in configs if c.attack  in args.attack]
    if args.variant and "all" not in args.variant:
        configs = [c for c in configs if c.variant in args.variant]

    if not args.overwrite:
        pending = [c for c in configs if not _result_path(c, args.results_dir).exists()]
        skipped = len(configs) - len(pending)
        if skipped:
            logger.info("Skipping %d already-complete configs.", skipped)
        configs = pending

    if not configs:
        logger.info("Nothing to run. Use --overwrite to force re-run.")
        return

    logger.info(
        "Running %d ablation experiment(s)%s.",
        len(configs),
        "  [DRY RUN]" if args.dry_run else "",
    )
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    run_kwargs: Dict[str, Any] = {
        "raw_dir":     args.raw_dir,
        "cache_dir":   args.cache_dir,
        "results_dir": args.results_dir,
        "device":      device,
    }
    tasks = [(cfg, run_kwargs) for cfg in configs]

    results: List[Dict] = []

    if args.workers == 1:
        for task in tasks:
            results.append(run_ablation_single(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
            future_to_cfg = {exe.submit(run_ablation_single, t): t[0] for t in tasks}
            for future in concurrent.futures.as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                try:
                    results.append(future.result(timeout=7200))
                except concurrent.futures.TimeoutError:
                    logger.error(
                        "TIMEOUT: %s/%s/%s/%d",
                        cfg.dataset, cfg.attack, cfg.variant, cfg.seed,
                    )
                    results.append({
                        "config": {**asdict(cfg), "detector": cfg.variant},
                        "status": "timeout",
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.error("Worker exception for %s: %s", cfg, exc)
                    results.append({
                        "config": {**asdict(cfg), "detector": cfg.variant},
                        "status": "error", "error": str(exc),
                    })

    _print_summary(results)


if __name__ == "__main__":
    main()
