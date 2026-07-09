#!/usr/bin/env python
"""Generalisation experiment: train on seen attacks, test on unseen attack.

Two scenarios
-------------
A) Seen → unseen LLM-agent attack
   Train detectors on ML-1M / Amazon-Books poisoned with
   [random, bandwagon, aush, graph_attack] (all mixed into one graph).
   Test on a fresh copy of the same dataset poisoned with Agent4SR only.

B) Reverse: train on Agent4SR only, test on GraphAttack (unseen GNN attack).

Grid
----
  Datasets      : ml1m, amazon_books
  Scenarios     : A (seen4→agent4sr), B (agent4sr→graph_attack)
  Detectors     : thgnn, statistical_baseline, semantic_shield_baseline, anti_fakeu_baseline
  Seeds         : 42, 123, 456
  injection_rate: 0.01

Total: 2 × 2 × 4 × 3 = 48 experiments.

Output
------
  results/raw/generalisation_{dataset}_{train_tag}_to_{test_attack}_{detector}_{seed}.json

The key claim being tested: TH-GNN generalises to unseen LLM-agent attacks
because it learns structural/temporal anomaly patterns, not attack-specific
features.

Usage
-----
    # Smoke test (ml1m, scenario A, thgnn, 2 seeds)
    python scripts/run_generalisation.py --dry-run

    # Full run, 2 parallel workers
    python scripts/run_generalisation.py --workers 2

    # Restrict axes
    python scripts/run_generalisation.py --dataset ml1m --scenario A
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
import yaml

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
from src.models.lightgcn import LightGCN, _BPRSampler, build_norm_adj

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(processName)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment grid constants
# ---------------------------------------------------------------------------

DATASETS: List[str] = ["ml1m", "amazon_books"]
DETECTORS: List[str] = [
    "thgnn",
    "statistical_baseline",
    "semantic_shield_baseline",
    "anti_fakeu_baseline",
]
SEEDS: List[int] = [42, 123, 456]
RATE: float = 0.01
N_TARGETS = 5
BUDGET_PCT = 0.10

_PAPER_RUNS_YAML = Path(__file__).resolve().parent.parent / "configs" / "paper_runs.yaml"


def _load_use_aaa() -> bool:
    """Read thgnn.use_aaa from configs/paper_runs.yaml.

    Same gap as run_experiments.py (54676ed) and run_ablation.py: this
    script's _build_detector() built THGNNWrapper(seed=seed) with no
    use_aaa, so it would have silently used base HGT attention regardless of
    this config. No real numbers from this script are in the paper draft
    yet (paper/tables/generalisation.tex says "Results pending"), so fixing
    this now is preventative, before Phase 4 runs.
    """
    try:
        with open(_PAPER_RUNS_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return bool(cfg.get("thgnn", {}).get("use_aaa", False))
    except FileNotFoundError:
        logger.warning("configs/paper_runs.yaml not found — defaulting use_aaa=False.")
        return False


# Two generalisation scenarios:
#   A: train on 4 classic attacks, test on the unseen LLM-agent attack
#   B: reverse — train on Agent4SR only, test on the unseen GNN attack
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "A": {
        "train_attacks": ["random", "bandwagon", "aush", "graph_attack"],
        "train_tag":     "seen4",
        "test_attack":   "agent4sr",
    },
    "B": {
        "train_attacks": ["agent4sr"],
        "train_tag":     "agent4sr",
        "test_attack":   "graph_attack",
    },
}

_LOADERS = {
    "ml1m":         MovieLens1MLoader,
    "amazon_books":  AmazonBooksLoader,
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenConfig:
    dataset:       str
    scenario:      str            # "A" or "B"
    train_attacks: List[str]      # list of attack names used for training
    train_tag:     str            # short label for file naming
    test_attack:   str
    detector:      str
    seed:          int
    rate:          float = RATE


def _result_path(cfg: GenConfig, results_dir: str) -> Path:
    return (
        Path(results_dir)
        / f"generalisation_{cfg.dataset}_{cfg.train_tag}_to_{cfg.test_attack}"
          f"_{cfg.detector}_{cfg.seed}.json"
    )


def _build_configs(args: argparse.Namespace) -> List[GenConfig]:
    if args.dry_run:
        datasets   = ["ml1m"]
        scenarios  = ["A"]
        detectors  = ["thgnn"]
        seeds      = SEEDS[:2]
    else:
        datasets   = DATASETS
        scenarios  = list(SCENARIOS.keys())
        detectors  = DETECTORS
        seeds      = SEEDS

    cfgs: List[GenConfig] = []
    for d in datasets:
        for sc_key in scenarios:
            sc = SCENARIOS[sc_key]
            for det in detectors:
                for s in seeds:
                    cfgs.append(GenConfig(
                        dataset       = d,
                        scenario      = sc_key,
                        train_attacks = sc["train_attacks"],
                        train_tag     = sc["train_tag"],
                        test_attack   = sc["test_attack"],
                        detector      = det,
                        seed          = s,
                    ))
    return cfgs


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset(dataset_name: str, raw_dir: str) -> HeteroDataset:
    return _LOADERS[dataset_name](raw_dir=raw_dir).load()


def _select_targets(dataset: HeteroDataset, n: int, seed: int) -> List[int]:
    train_items = dataset.review_item_idx[dataset.train_mask].numpy()
    counts      = np.bincount(train_items, minlength=dataset.num_items)
    top_k       = max(n, int(dataset.num_items * 0.1))
    pool        = np.argsort(-counts)[:top_k]
    rng         = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()


def _train_surrogate(
    dataset: HeteroDataset,
    device: str,
    seed: int,
    epochs: int = 15,
) -> LightGCN:
    """Quick surrogate LightGCN for GraphAttack."""
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


def _build_and_run_attack(
    attack_name: str,
    base_dataset: HeteroDataset,
    poisoned_so_far: HeteroDataset,
    n_fake: int,
    targets: List[int],
    device: str,
    seed: int,
    agent_cache_dir: str,
) -> HeteroDataset:
    """Build one attack, generate profiles against the base dataset, inject into poisoned_so_far."""
    if attack_name == "random":
        attack: ShillingAttack = RandomAttack(base_dataset, seed=seed)

    elif attack_name == "bandwagon":
        attack = BandwagonAttack(base_dataset, seed=seed)

    elif attack_name == "aush":
        attack = AUSHAttack(base_dataset, seed=seed)
        attack.pretrain(base_dataset, epochs=20, device=device)

    elif attack_name == "graph_attack":
        surrogate = _train_surrogate(base_dataset, device, seed)
        attack = GraphAttack(surrogate, base_dataset, n_steps=20, seed=seed)

    elif attack_name == "agent4sr":
        cache = Path(agent_cache_dir) / "agent4sr"
        attack = Agent4SRAttack(base_dataset, cache_dir=str(cache), seed=seed)

    else:
        raise ValueError(f"Unknown attack: {attack_name!r}")

    profiles = attack.generate_profiles(n_fake, BUDGET_PCT, targets)
    return attack.inject(poisoned_so_far, profiles)


def _build_detector(name: str, seed: int, use_aaa: bool = False):
    if name == "thgnn":
        return THGNNWrapper(seed=seed, use_aaa=use_aaa)
    if name == "statistical_baseline":
        return StatisticalBaseline(seed=seed)
    if name == "semantic_shield_baseline":
        return SemanticShieldBaseline(seed=seed)
    if name == "anti_fakeu_baseline":
        return AntiFakeUBaseline(seed=seed)
    raise ValueError(f"Unknown detector: {name!r}")


# ---------------------------------------------------------------------------
# Single-experiment pipeline (module-level → picklable)
# ---------------------------------------------------------------------------

def run_generalisation_single(args_tuple: Tuple[GenConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Run one generalisation cell.

    Train phase: inject all ``train_attacks`` into the base dataset (chained),
    then fit the detector on the combined poisoned graph.

    Test phase: start from a fresh copy of the base dataset, inject only
    ``test_attack`` profiles, score users with the already-fitted detector,
    and evaluate F1 / DR / FAR.
    """
    cfg, kwargs   = args_tuple
    raw_dir       = kwargs["raw_dir"]
    results_dir   = kwargs["results_dir"]
    cache_dir     = kwargs["cache_dir"]
    device        = kwargs.get("device") or _auto_device()
    use_aaa       = kwargs.get("use_aaa", False)

    result: Dict[str, Any] = {
        "config": {
            "dataset":       cfg.dataset,
            "scenario":      cfg.scenario,
            "train_attacks": cfg.train_attacks,
            "train_tag":     cfg.train_tag,
            "test_attack":   cfg.test_attack,
            "detector":      cfg.detector,
            "seed":          cfg.seed,
            "rate":          cfg.rate,
        },
        "status": "running",
    }
    path = _result_path(cfg, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    try:
        logger.info(
            "[%s/scenario=%s/train=%s/test=%s/%s/seed=%d] loading dataset …",
            cfg.dataset, cfg.scenario, "+".join(cfg.train_attacks),
            cfg.test_attack, cfg.detector, cfg.seed,
        )

        # ── 1. Load clean base dataset ────────────────────────────────────
        base_dataset = _load_dataset(cfg.dataset, raw_dir)

        targets = _select_targets(base_dataset, N_TARGETS, cfg.seed)
        n_fake  = max(1, round(cfg.rate * base_dataset.num_users))
        logger.info(
            "  targets=%s  n_fake_per_attack=%d  train_attacks=%s",
            targets, n_fake, cfg.train_attacks,
        )

        # ── 2. Training phase: inject all seen attacks ────────────────────
        train_poisoned = base_dataset  # start from clean
        for atk_name in cfg.train_attacks:
            logger.info("  [train] injecting %s …", atk_name)
            train_poisoned = _build_and_run_attack(
                atk_name,
                base_dataset,       # attacks always profile against the clean dataset
                train_poisoned,     # but inject into the growing poisoned graph
                n_fake,
                targets,
                device,
                cfg.seed,
                cache_dir,
            )
        n_train_fake = int(train_poisoned.labels.sum())
        logger.info(
            "  train graph: %d users total, %d fake (%d attack types × %d each)",
            train_poisoned.num_users, n_train_fake,
            len(cfg.train_attacks), n_fake,
        )

        # ── 3. Fit detector on combined training graph ────────────────────
        detector = _build_detector(cfg.detector, cfg.seed, use_aaa=use_aaa)
        t0_fit = time.perf_counter()
        detector.fit(train_poisoned, device=device, seed=cfg.seed)
        t_fit = time.perf_counter() - t0_fit
        logger.info("  detector fitted in %.1f s", t_fit)

        # ── 4. Test phase: inject only the unseen attack ──────────────────
        logger.info("  [test] injecting unseen attack: %s …", cfg.test_attack)
        test_poisoned = _build_and_run_attack(
            cfg.test_attack,
            base_dataset,
            base_dataset,   # fresh graph — no training-phase fakes
            n_fake,
            targets,
            device,
            cfg.seed,
            cache_dir,
        )
        n_test_fake = int(test_poisoned.labels.sum())
        logger.info(
            "  test graph: %d users, %d fake (%s)",
            test_poisoned.num_users, n_test_fake, cfg.test_attack,
        )

        # ── 5. Score + evaluate ───────────────────────────────────────────
        t0_score = time.perf_counter()
        scores = detector.score_users(test_poisoned)
        t_score = time.perf_counter() - t0_score

        labels = test_poisoned.labels.numpy()
        f1, threshold = best_threshold_f1(labels, scores)
        dr  = detection_rate(labels, scores, threshold=threshold)
        far = false_alarm_rate(labels, scores, threshold=threshold)

        n_flagged = int(np.sum(scores >= threshold))
        logger.info(
            "  score_secs=%.1f  F1=%.4f  DR=%.4f  FAR=%.4f  flagged=%d/%d",
            t_score, f1, dr, far, n_flagged, n_test_fake,
        )

        result["detection"] = {
            "f1":        round(float(f1),        4),
            "dr":        round(float(dr),        4),
            "far":       round(float(far),       4),
            "threshold": round(float(threshold), 4),
            "n_flagged": n_flagged,
            "n_fake":    n_test_fake,
            "fit_secs":  round(t_fit,   2),
            "score_secs": round(t_score, 2),
        }
        result["status"]     = "ok"
        result["total_secs"] = round(time.perf_counter() - t_total, 2)

    except Exception:  # noqa: BLE001
        result["status"]    = "error"
        result["traceback"] = traceback.format_exc()
        logger.error(
            "[%s/sc=%s/%s/seed=%d] FAILED:\n%s",
            cfg.dataset, cfg.scenario, cfg.detector, cfg.seed,
            result["traceback"],
        )

    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)

    return result


# ---------------------------------------------------------------------------
# Progress summary
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict]) -> None:
    ok     = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]
    skip   = [r for r in results if r.get("status") == "skipped"]
    print(f"\n{'='*65}")
    print(f"  Generalisation experiments: {len(ok)} ok  {len(errors)} errors  {len(skip)} skipped")

    if ok:
        by_scenario: Dict[str, Dict[str, List[float]]] = {}
        for r in ok:
            if "detection" not in r:
                continue
            cfg = r["config"]
            sc  = cfg.get("scenario", "?")
            det = cfg.get("detector", "?")
            by_scenario.setdefault(sc, {}).setdefault(det, []).append(r["detection"]["f1"])

        print("\n  Mean F1 by scenario / detector:")
        for sc, det_map in sorted(by_scenario.items()):
            sc_info = SCENARIOS.get(sc, {})
            print(f"\n  Scenario {sc}: train={sc_info.get('train_tag','')}  "
                  f"test={sc_info.get('test_attack','')}")
            for det, f1s in sorted(det_map.items()):
                print(f"    {det:30s}  F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run THGNN generalisation experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Smoke test: ml1m, scenario A, thgnn, 2 seeds.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (ProcessPoolExecutor). 1 = sequential.",
    )
    p.add_argument("--raw-dir",     default="data/raw",       metavar="DIR")
    p.add_argument("--cache-dir",   default="data/processed", metavar="DIR")
    p.add_argument("--results-dir", default="results/raw",    metavar="DIR")
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="torch device: cpu | cuda | mps (autodetect if omitted).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-run configs whose result file already exists.",
    )
    p.add_argument(
        "--dataset", nargs="+", choices=list(_LOADERS) + ["all"], default=None,
        help="Restrict to specific datasets.",
    )
    p.add_argument(
        "--scenario", nargs="+", choices=list(SCENARIOS) + ["all"], default=None,
        help="Restrict to scenario A, B, or all.",
    )
    p.add_argument(
        "--detector", nargs="+", choices=DETECTORS + ["all"], default=None,
        help="Restrict to specific detectors.",
    )
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    device  = args.device or _auto_device()
    use_aaa = _load_use_aaa()
    logger.info(
        "Device: %s  Workers: %d  thgnn.use_aaa (from paper_runs.yaml): %s",
        device, args.workers, use_aaa,
    )

    configs = _build_configs(args)

    if args.dataset and "all" not in args.dataset:
        configs = [c for c in configs if c.dataset in args.dataset]
    if args.scenario and "all" not in args.scenario:
        configs = [c for c in configs if c.scenario in args.scenario]
    if args.detector and "all" not in args.detector:
        configs = [c for c in configs if c.detector in args.detector]

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
        "Running %d generalisation experiment(s)%s.",
        len(configs),
        "  [DRY RUN]" if args.dry_run else "",
    )
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    run_kwargs: Dict[str, Any] = {
        "raw_dir":     args.raw_dir,
        "cache_dir":   args.cache_dir,
        "results_dir": args.results_dir,
        "device":      device,
        "use_aaa":     use_aaa,
    }
    tasks = [(cfg, run_kwargs) for cfg in configs]

    results: List[Dict] = []

    if args.workers == 1:
        for task in tasks:
            results.append(run_generalisation_single(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
            future_to_cfg = {exe.submit(run_generalisation_single, t): t[0] for t in tasks}
            for future in concurrent.futures.as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                try:
                    results.append(future.result(timeout=7200))
                except concurrent.futures.TimeoutError:
                    logger.error(
                        "TIMEOUT: %s/sc=%s/%s/seed=%d",
                        cfg.dataset, cfg.scenario, cfg.detector, cfg.seed,
                    )
                    results.append({
                        "config": asdict(cfg),
                        "status": "timeout",
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.error("Worker exception for %s: %s", cfg, exc)
                    results.append({
                        "config": asdict(cfg),
                        "status": "error",
                        "error":  str(exc),
                    })

    _print_summary(results)


if __name__ == "__main__":
    main()
