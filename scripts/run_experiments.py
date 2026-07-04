#!/usr/bin/env python
"""Full experiment matrix runner for THGNN shilling-attack detection.

Experiment grid
---------------
    Datasets            × Attacks       × Injection rates × Detectors          × Seeds
    ---------             ----------      ----------------   --------             ------
    ml1m                  random          0.005              thgnn                42
    video_games           bandwagon       0.010              statistical_baseline 123
    arts_crafts           aush            0.050              semantic_shield_baseline 456
    musical_instruments   graph_attack                       anti_fakeu_baseline
                          agent4sr

Total full run: 4 × 5 × 3 × 4 × 3 = 720 experiments.

Pipeline (per cell)
-------------------
1. Load dataset from disk.
2. Generate N_TARGETS target items (most popular by training degree).
3. Build and (pre-)train attack; generate profiles at the requested rate.
4. Inject profiles → poisoned dataset.
5. Fit detector on poisoned dataset; score all users.
6. Evaluate detection: F1, DR, FAR (best-threshold).
7. Filter flagged users from the graph.
8. Retrain LightGCN on the filtered graph.
9. Evaluate recommendation: NDCG@10, HR@10.
10. Save results/raw/{dataset}_{attack}_{rate}_{detector}_{seed}.json.

Usage
-----
    # Smoke test (2 seeds, ml1m × random only)
    python scripts/run_experiments.py --dry-run

    # Full run, 4 parallel workers
    python scripts/run_experiments.py --workers 4

    # Resume (skips configs whose result file already exists)
    python scripts/run_experiments.py --workers 4

    # Force rerun even if results exist
    python scripts/run_experiments.py --workers 4 --overwrite
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
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
    AmazonArtsCraftsLoader,
    AmazonMusicalInstrumentsLoader,
    AmazonVideoGamesLoader,
    HeteroDataset,
    MovieLens1MLoader,
)
from src.detectors import (
    AntiFakeUBaseline,
    SemanticShieldBaseline,
    StatisticalBaseline,
    THGNNWrapper,
)
from src.evaluation import best_threshold_f1, detection_rate, false_alarm_rate
from src.models.lightgcn import LightGCN, _BPRSampler, build_norm_adj, train_lightgcn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(processName)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment grid constants
# ---------------------------------------------------------------------------

DATASETS:  List[str]   = ["ml1m", "video_games", "arts_crafts", "musical_instruments"]
ATTACKS:   List[str]   = ["random", "bandwagon", "aush", "graph_attack", "agent4sr"]
RATES:     List[float] = [0.005, 0.01, 0.05]
DETECTORS: List[str]   = [
    "thgnn",
    "statistical_baseline",
    "semantic_shield_baseline",
    "anti_fakeu_baseline",
]
SEEDS: List[int] = [42, 123, 456]

N_TARGETS  = 5      # target items per attack run
BUDGET_PCT = 0.10   # fraction of item catalog per fake profile

_LOADERS = {
    "ml1m":                MovieLens1MLoader,
    "video_games":         AmazonVideoGamesLoader,
    "arts_crafts":         AmazonArtsCraftsLoader,
    "musical_instruments": AmazonMusicalInstrumentsLoader,
}


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    dataset:  str
    attack:   str
    rate:     float
    detector: str
    seed:     int


def _result_path(cfg: ExpConfig, results_dir: str) -> Path:
    return (
        Path(results_dir)
        / f"{cfg.dataset}_{cfg.attack}_{cfg.rate}_{cfg.detector}_{cfg.seed}.json"
    )


def _build_configs(args: argparse.Namespace) -> List[ExpConfig]:
    if args.dry_run:
        datasets  = ["ml1m"]
        attacks   = ["random"]
        seeds     = SEEDS[:2]
    else:
        datasets  = DATASETS
        attacks   = ATTACKS
        seeds     = SEEDS

    return [
        ExpConfig(d, a, r, det, s)
        for d   in datasets
        for a   in attacks
        for r   in RATES
        for det in DETECTORS
        for s   in seeds
    ]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset(cfg: ExpConfig, raw_dir: str) -> HeteroDataset:
    loader = _LOADERS[cfg.dataset](raw_dir=raw_dir)
    return loader.load()


def _select_targets(dataset: HeteroDataset, n: int, seed: int) -> List[int]:
    """Return n target items: sampled from the top-10 % most-trained items."""
    train_items = dataset.review_item_idx[dataset.train_mask].numpy()
    counts      = np.bincount(train_items, minlength=dataset.num_items)
    top_k       = max(n, int(dataset.num_items * 0.1))
    pool        = np.argsort(-counts)[:top_k]
    rng         = np.random.default_rng(seed)
    chosen      = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return chosen.tolist()


def _train_surrogate(
    dataset: HeteroDataset,
    device:  str,
    seed:    int,
    epochs:  int = 15,
) -> LightGCN:
    """Fast surrogate LightGCN for GraphAttack (small model, few epochs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    model    = LightGCN(dataset.num_users, dataset.num_items,
                        embed_dim=32, n_layers=2).to(dev)
    train_u  = dataset.review_user_idx[dataset.train_mask].numpy()
    train_i  = dataset.review_item_idx[dataset.train_mask].numpy()
    adj      = build_norm_adj(train_u, train_i,
                              dataset.num_users, dataset.num_items, dev)
    sampler  = _BPRSampler(train_u, train_i, dataset.num_items)
    opt      = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs       = 1024
    steps    = max(1, len(train_u) // bs)

    for _ in range(epochs):
        model.train()
        for _ in range(steps):
            u_np, pos_np, neg_np = sampler.sample(bs)
            u_t   = torch.from_numpy(u_np).long().to(dev)
            pos_t = torch.from_numpy(pos_np).long().to(dev)
            neg_t = torch.from_numpy(neg_np).long().to(dev)
            ue, ie = model(adj)
            loss = (model.bpr_loss(ue, ie, u_t, pos_t, neg_t)
                    + model.reg_loss(u_t, pos_t, neg_t, 1e-4))
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


def _build_attack(
    cfg:             ExpConfig,
    dataset:         HeteroDataset,
    device:          str,
    agent_cache_dir: str,
) -> Tuple[ShillingAttack, float]:
    """Instantiate and (pre-)train the attack.  Returns (attack, elapsed_secs)."""
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


def _build_detector(name: str, seed: int):
    """Return a fresh (unfitted) detector by name."""
    if name == "thgnn":
        return THGNNWrapper(seed=seed)
    if name == "statistical_baseline":
        return StatisticalBaseline(seed=seed)
    if name == "semantic_shield_baseline":
        return SemanticShieldBaseline(seed=seed)
    if name == "anti_fakeu_baseline":
        return AntiFakeUBaseline(seed=seed)
    raise ValueError(f"Unknown detector: {name!r}")


def _filter_dataset(
    dataset:       HeteroDataset,
    flagged_users: np.ndarray,
) -> HeteroDataset:
    """Return a copy of *dataset* with all reviews from *flagged_users* masked out."""
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
# Single-experiment pipeline  (module-level → picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def run_single(args_tuple: Tuple[ExpConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Run one (dataset, attack, rate, detector, seed) cell.

    Always writes a result JSON and returns the result dict.
    On error, the result contains ``status="error"`` and a traceback.
    """
    cfg, kwargs = args_tuple
    raw_dir     = kwargs["raw_dir"]
    results_dir = kwargs["results_dir"]
    cache_dir   = kwargs["cache_dir"]
    device      = kwargs.get("device") or _auto_device()

    result: Dict[str, Any] = {"config": asdict(cfg), "status": "running"}
    path = _result_path(cfg, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    try:
        # ── 1. Load dataset ───────────────────────────────────────────────
        logger.info(
            "[%s/%s/rate=%.3f/%s/seed=%d] loading dataset …",
            cfg.dataset, cfg.attack, cfg.rate, cfg.detector, cfg.seed,
        )
        dataset = _load_dataset(cfg, raw_dir)

        # ── 2. Select target items ────────────────────────────────────────
        targets = _select_targets(dataset, N_TARGETS, cfg.seed)
        n_fake  = max(1, round(cfg.rate * dataset.num_users))
        logger.info(
            "  targets=%s  n_fake=%d  budget=%.0f%%",
            targets, n_fake, BUDGET_PCT * 100,
        )

        # ── 3. Build & run attack ─────────────────────────────────────────
        attack, t_attack = _build_attack(cfg, dataset, device, cache_dir)
        profiles = attack.generate_profiles(n_fake, BUDGET_PCT, targets)
        poisoned = attack.inject(dataset, profiles)
        logger.info(
            "  injected %d fake users  (attack_secs=%.1f)",
            len(profiles), t_attack,
        )

        # ── 4. Fit detector ───────────────────────────────────────────────
        detector = _build_detector(cfg.detector, cfg.seed)
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
            "  detect_secs=%.1f  F1=%.4f  DR=%.4f  FAR=%.4f  "
            "flagged=%d/%d",
            t_detect, f1, dr, far, len(flagged), n_fake_actual,
        )

        result["detection"] = {
            "f1":        round(float(f1),  4),
            "dr":        round(float(dr),  4),
            "far":       round(float(far), 4),
            "threshold": round(float(threshold), 4),
            "n_flagged": int(len(flagged)),
            "n_fake":    n_fake_actual,
            "detect_secs": round(t_detect, 2),
        }

        # ── 6. Filter graph ───────────────────────────────────────────────
        filtered = _filter_dataset(poisoned, flagged)
        n_train_kept = int(filtered.train_mask.sum())
        logger.info(
            "  filtered graph: %d train reviews remain", n_train_kept,
        )

        # ── 7. Retrain LightGCN on filtered graph ─────────────────────────
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
            "hr@10":      round(float(lgcn_results.get("hr@10", 0)),   4),
            "best_epoch": int(lgcn_results.get("best_epoch", 0)),
            "lgcn_secs":  round(t_lgcn, 2),
        }
        logger.info(
            "  NDCG@10=%.4f  HR@10=%.4f  lgcn_secs=%.1f",
            result["recommendation"]["ndcg@10"],
            result["recommendation"]["hr@10"],
            t_lgcn,
        )

        result["status"]      = "ok"
        result["total_secs"]  = round(time.perf_counter() - t_total, 2)

    except Exception:  # noqa: BLE001
        result["status"]    = "error"
        result["traceback"] = traceback.format_exc()
        logger.error(
            "[%s/%s/%.3f/%s/%d] FAILED:\n%s",
            cfg.dataset, cfg.attack, cfg.rate, cfg.detector, cfg.seed,
            result["traceback"],
        )

    # ── Always write to disk ──────────────────────────────────────────────
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
    print(f"  Experiments complete: {len(ok)} ok  {len(errors)} errors  {len(skip)} skipped")
    if ok:
        f1s  = [r["detection"]["f1"]              for r in ok if "detection"     in r]
        ndcg = [r["recommendation"]["ndcg@10"]    for r in ok if "recommendation" in r]
        if f1s:
            print(f"  Detection  F1:    mean={np.mean(f1s):.4f}  "
                  f"min={np.min(f1s):.4f}  max={np.max(f1s):.4f}")
        if ndcg:
            print(f"  NDCG@10:         mean={np.mean(ndcg):.4f}  "
                  f"min={np.min(ndcg):.4f}  max={np.max(ndcg):.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full THGNN experiment matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Run 2 seeds on ml1m×random only to verify the pipeline.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (ProcessPoolExecutor). "
             "1 = sequential (safe default).",
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
        "--dataset",  nargs="+", choices=DATASETS + ["all"], default=None,
        help="Restrict to specific datasets.",
    )
    p.add_argument(
        "--attack",   nargs="+", choices=ATTACKS  + ["all"], default=None,
        help="Restrict to specific attacks.",
    )
    p.add_argument(
        "--detector", nargs="+", choices=DETECTORS + ["all"], default=None,
        help="Restrict to specific detectors.",
    )
    p.add_argument(
        "--seed", nargs="+", type=int, default=None,
        help="Restrict to specific seeds (default: all of %s)." % SEEDS,
    )
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = args.device or _auto_device()
    logger.info("Device: %s  Workers: %d", device, args.workers)

    configs = _build_configs(args)

    # ── Optional subset filters ───────────────────────────────────────────
    if args.dataset and "all" not in args.dataset:
        configs = [c for c in configs if c.dataset in args.dataset]
    if args.attack and "all" not in args.attack:
        configs = [c for c in configs if c.attack  in args.attack]
    if args.detector and "all" not in args.detector:
        configs = [c for c in configs if c.detector in args.detector]
    if args.seed:
        configs = [c for c in configs if c.seed in args.seed]

    # ── Skip already-done configs unless --overwrite ──────────────────────
    if not args.overwrite:
        pending  = [c for c in configs
                    if not _result_path(c, args.results_dir).exists()]
        skipped  = len(configs) - len(pending)
        if skipped:
            logger.info("Skipping %d already-complete configs.", skipped)
        configs = pending

    if not configs:
        logger.info("Nothing to run.  Use --overwrite to re-run completed configs.")
        return

    logger.info(
        "Running %d experiment(s)%s.",
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

    # ── Execute ──────────────────────────────────────────────────────────
    results: List[Dict] = []

    if args.workers == 1:
        # Sequential — avoids spawn overhead and is easier to debug
        for task in tasks:
            results.append(run_single(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
        ) as executor:
            future_to_cfg = {
                executor.submit(run_single, task): task[0] for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                try:
                    results.append(future.result(timeout=7200))  # 2-hour per-job cap
                except concurrent.futures.TimeoutError:
                    logger.error(
                        "TIMEOUT: %s/%s/%.3f/%s/%d",
                        cfg.dataset, cfg.attack, cfg.rate, cfg.detector, cfg.seed,
                    )
                    results.append({"config": asdict(cfg), "status": "timeout"})
                except Exception as exc:  # noqa: BLE001
                    logger.error("Worker exception for %s: %s", cfg, exc)
                    results.append({"config": asdict(cfg), "status": "error",
                                    "error": str(exc)})

    _print_summary(results)


if __name__ == "__main__":
    main()
