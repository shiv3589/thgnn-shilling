#!/usr/bin/env python
"""Session 1 driver — provenance reproduction check + dual-variant (base/AAA) grid.

Context (Phase 0 findings this script exists to resolve)
----------------------------------------------------------
1. ``configs/paper_runs.yaml`` sets ``thgnn.use_aaa: true`` but
   ``src/detectors/thgnn_wrapper.py`` never read or forwarded that flag to
   ``HGTEncoder`` — every prior THGNNWrapper run silently built the base-HGT
   variant regardless of the yaml setting. This has been fixed (see
   ``THGNNWrapper.__init__``'s ``use_aaa`` parameter and
   ``THGNNWrapper._log_variant_assertion``, which prints and asserts which
   attention variant is active on every model build).
2. No prior run persisted per-user prediction scores, only aggregated
   F1/DR/FAR. This script saves raw (y_true, y_score) via
   ``src.evaluation.save_scores`` for every completed config.

Because the bug in (1) was that ``use_aaa`` was *never wired through* (not
that a wrong value was substituted), the "pre-fix" code path is bit-for-bit
identical to calling ``THGNNWrapper(..., use_aaa=False)`` post-fix — there is
no behavioural difference to stash away. ``--stage reproduction`` therefore
runs with ``variant="base"`` and this equivalence is stated explicitly in its
log output rather than reconstructed via git stash.

Scope notes (read before trusting a number this script prints)
----------------------------------------------------------------
* Dataset is restricted to ml1m only. The other three datasets in
  ``paper_runs.yaml`` are explicitly marked "PLACEHOLDER" pending a real run,
  so there is nothing yet to reproduce or grid over for them.
* Detectors in scope are thgnn, statistical_baseline, anti_fakeu_baseline —
  semantic_shield_baseline is intentionally excluded (not requested for this
  session).
* The grid is 5 attacks x 3 injection rates x up to 3 detectors = up to 45
  configs per variant pass (statistical/anti_fakeu are computed once and
  reused across the base/aaa passes, so aaa only adds 15 new thgnn configs).
  This does not equal the "33 configurations" mentioned when this session was
  scoped; that number does not correspond to any grid derivable from this
  repository's actual attack/rate/detector lists, so it is not used here.
* Single seed (default 42), not the paper's [42, 123, 456] — a 3-seed x
  4-dataset run is the full 720-experiment matrix (~18h on an A100 per
  REPRODUCE.md), well beyond a bounded Kaggle session. Multi-seed is future
  work for Session 2+.
* The downstream LightGCN filter-and-retrain step (NDCG@10 / HR@10) that
  ``scripts/run_experiments.py`` also computes is NOT run here — this session
  is scoped to detection scores only (F1/DR/FAR + raw per-user scores).

Usage
-----
    python run_session1.py --stage reproduction
    python run_session1.py --stage grid --variant base
    python run_session1.py --stage grid --variant aaa
    python run_session1.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks import (
    AUSHAttack,
    Agent4SRAttack,
    BandwagonAttack,
    GraphAttack,
    LocalAgent4SR,
    RandomAttack,
    ShillingAttack,
)
from src.datasets import HeteroDataset, MovieLens1MLoader
from src.detectors import AntiFakeUBaseline, StatisticalBaseline, THGNNWrapper
from src.evaluation import best_threshold_f1, detection_rate, false_alarm_rate, save_scores
from src.evaluation.score_io import score_path
from src.models.lightgcn import LightGCN, _BPRSampler, build_norm_adj

logger = logging.getLogger("run_session1")

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
SCORES_DIR = RESULTS_DIR / "scores"
LOGS_DIR = RESULTS_DIR / "logs"
LOG_MD = RESULTS_DIR / "EXPERIMENT_LOG.md"
VERDICT_PATH = RESULTS_DIR / "REPRODUCTION_VERDICT.txt"

TOLERANCE = 0.03  # same tolerance as tests/test_reproducibility.py

GRID_DATASET = "ml1m"
GRID_ATTACKS = ["random", "bandwagon", "aush", "graph_attack", "agent4sr"]
GRID_RATES = [0.005, 0.01, 0.05]
GRID_DETECTORS = ["thgnn", "statistical_baseline", "anti_fakeu_baseline"]

DRY_RUN_ATTACK = "random"
DRY_RUN_RATE = 0.01

LOG_HEADER = (
    "| timestamp | stage | dataset | attack | rate | detector | variant | seed "
    "| status | f1 | dr | far | git_commit | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


@dataclass
class ExpConfig:
    dataset: str
    attack: str
    rate: float
    detector: str
    variant: str  # "base" | "aaa" | "na" (statistical/anti_fakeu are use_aaa-agnostic)
    seed: int


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _load_paper_config() -> dict:
    with open(CONFIGS_DIR / "paper_runs.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expected_f1(cfg: dict, dataset: str, attack: str, detector: str) -> Optional[float]:
    val = cfg.get("expected_f1", {}).get(dataset, {}).get(attack, {}).get(detector)
    return float(val) if val is not None else None


_log_section_started = False


def _ensure_log_section(stage_label: str) -> None:
    """Append a new, clearly-delimited section header for this script
    invocation before any rows are written.

    ``results/EXPERIMENT_LOG.md`` may already exist as a narrative log from
    earlier work (it does in this repo — a "Phase 0 + Phase 1" entry). This
    intentionally never overwrites or rewrites that file; it only appends,
    and always under its own heading, so a raw table never gets silently
    spliced into someone else's prose with no separator.
    """
    global _log_section_started
    if _log_section_started:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_MD, "a", encoding="utf-8") as f:
        f.write(
            f"\n---\n\n## run_session1.py — {stage_label} — "
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
        )
        f.write(LOG_HEADER)
    _log_section_started = True


def _append_log_row(row: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_MD, "a", encoding="utf-8") as f:
        notes = str(row.get("notes", "")).replace("\n", " ").replace("|", "/")[:200]
        f.write(
            f"| {row['timestamp']} | {row['stage']} | {row['dataset']} | {row['attack']} | "
            f"{row['rate']} | {row['detector']} | {row['variant']} | {row['seed']} | "
            f"{row['status']} | {row.get('f1', '')} | {row.get('dr', '')} | "
            f"{row.get('far', '')} | {row['git_commit']} | {notes} |\n"
        )


def _maybe_upload(paths: List[Path], repo_id: str, dry_run: bool) -> None:
    paths = [Path(p) for p in paths if p and Path(p).exists()]
    if dry_run:
        for p in paths:
            print(f"  [dry-run] would upload {p} -> hf://datasets/{repo_id}/{p.name}")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set in environment — skipping upload of %s", paths)
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_repo(%s) failed (may already exist): %s", repo_id, exc)
    for p in paths:
        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"session1: update {p.name}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("HF upload failed for %s: %s", p, exc)


# ---------------------------------------------------------------------------
# Synthetic dataset (--dry-run only; no raw data / GPU required)
# ---------------------------------------------------------------------------

def _build_synthetic_dataset(seed: int = 0) -> HeteroDataset:
    """Small in-memory heterogeneous graph mirroring the real schema.

    Sized so RandomAttack, StatisticalBaseline/AntiFakeUBaseline's
    IsolationForest, and THGNNWrapper's full-graph forward pass all run in
    well under a second on CPU.
    """
    n_users, n_items, n_reviews = 150, 120, 600
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    writes_u = torch.from_numpy(rng.integers(0, n_users, n_reviews).astype(np.int64))
    writes_r = torch.arange(n_reviews, dtype=torch.long)
    about_r = torch.arange(n_reviews, dtype=torch.long)
    about_i = torch.from_numpy(rng.integers(0, n_items, n_reviews).astype(np.int64))
    # (user, rates, item) mirrors BaseLoader's real convention: one rates-edge
    # per review, sharing the same (user, item) pair — NOT an independently
    # sized random subsample. A mismatched count here silently misaligns
    # edge_attr_dict (see below) against edge_index_dict for this edge type;
    # THGNNWrapper's full-graph forward pass has no khop-subgraph slicing to
    # mask that misalignment, so it surfaces as a hard shape-mismatch crash
    # in HGTLayer instead of silently using wrong timestamps.
    rates_u = writes_u
    rates_i = about_i

    ts = torch.from_numpy(np.sort(rng.uniform(0, 1e9, n_reviews)))

    n_train = int(0.7 * n_reviews)
    n_val = int(0.1 * n_reviews)
    train_mask = torch.zeros(n_reviews, dtype=torch.bool)
    val_mask = torch.zeros(n_reviews, dtype=torch.bool)
    test_mask = torch.zeros(n_reviews, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True

    edge_index_dict = {
        ("user", "writes", "review"): torch.stack([writes_u, writes_r]),
        ("review", "about", "item"): torch.stack([about_r, about_i]),
        ("user", "rates", "item"): torch.stack([rates_u, rates_i]),
    }
    edge_attr_dict = {k: ts for k in edge_index_dict}

    return HeteroDataset(
        dataset_name="synthetic_dry_run",
        num_users=n_users,
        num_items=n_items,
        num_reviews=n_reviews,
        edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        x_user=torch.randn(n_users, 8),
        x_item=torch.randn(n_items, 5),
        review_texts=["dry-run review text"] * n_reviews,
        ratings=torch.from_numpy(rng.uniform(1, 5, n_reviews).astype(np.float32)),
        timestamps=ts,
        review_user_idx=writes_u,
        review_item_idx=about_i,
        labels=torch.zeros(n_users, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        user_id_map={i: i for i in range(n_users)},
        item_id_map={i: i for i in range(n_items)},
    )


# ---------------------------------------------------------------------------
# Pipeline helpers (targets / attacks / detectors)
# ---------------------------------------------------------------------------

def _select_targets(dataset: HeteroDataset, n: int, seed: int) -> List[int]:
    train_items = dataset.review_item_idx[dataset.train_mask].numpy()
    counts = np.bincount(train_items, minlength=dataset.num_items)
    top_k = max(n, int(dataset.num_items * 0.1))
    pool = np.argsort(-counts)[:top_k]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return chosen.tolist()


def _train_surrogate(dataset: HeteroDataset, device: str, seed: int, epochs: int) -> LightGCN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model = LightGCN(dataset.num_users, dataset.num_items, embed_dim=32, n_layers=2).to(dev)
    train_u = dataset.review_user_idx[dataset.train_mask].numpy()
    train_i = dataset.review_item_idx[dataset.train_mask].numpy()
    adj = build_norm_adj(train_u, train_i, dataset.num_users, dataset.num_items, dev)
    sampler = _BPRSampler(train_u, train_i, dataset.num_items)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 1024
    steps = max(1, len(train_u) // bs)
    for _ in range(epochs):
        model.train()
        for _ in range(steps):
            u_np, pos_np, neg_np = sampler.sample(bs)
            u_t = torch.from_numpy(u_np).long().to(dev)
            pos_t = torch.from_numpy(pos_np).long().to(dev)
            neg_t = torch.from_numpy(neg_np).long().to(dev)
            ue, ie = model(adj)
            loss = model.bpr_loss(ue, ie, u_t, pos_t, neg_t) + model.reg_loss(u_t, pos_t, neg_t, 1e-4)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _build_attack(
    cfg: ExpConfig, dataset: HeteroDataset, device: str, cache_dir: str, dry_run: bool,
    agent4sr_cache_dir: str = "data/agent4sr_cache",
) -> ShillingAttack:
    if cfg.attack == "random":
        return RandomAttack(dataset, seed=cfg.seed)
    if cfg.attack == "bandwagon":
        return BandwagonAttack(dataset, seed=cfg.seed)
    if cfg.attack == "aush":
        attack = AUSHAttack(dataset, seed=cfg.seed)
        attack.pretrain(dataset, epochs=1 if dry_run else 20, device=device)
        return attack
    if cfg.attack == "graph_attack":
        surrogate = _train_surrogate(dataset, device, cfg.seed, epochs=1 if dry_run else 15)
        return GraphAttack(surrogate, dataset, n_steps=1 if dry_run else 20, seed=cfg.seed)
    if cfg.attack == "agent4sr":
        # NOTE: deliberately NOT derived from --cache-dir (data/processed/),
        # unlike scripts/run_experiments.py's convention of
        # data/processed/agent4sr/{dataset}. The pre-populated cache this
        # session is expected to reuse (see run_session1.py's module
        # docstring / A5 Kaggle notes) is named agent4sr_cache/, matching
        # Agent4SRAttack's own class-level default. Deriving from
        # --cache-dir here would silently miss that cache and fall back to
        # live OPENAI_API_KEY calls instead of the cached responses.
        cache = Path(agent4sr_cache_dir) / cfg.dataset
        if os.environ.get("AGENT4SR_BACKEND", "openai") == "local":
            return LocalAgent4SR(dataset, cache_dir=str(cache), seed=cfg.seed)
        return Agent4SRAttack(dataset, cache_dir=str(cache), seed=cfg.seed)
    raise ValueError(f"Unknown attack: {cfg.attack!r}")


def _build_detector(name: str, seed: int, rate: float, use_aaa: bool, epochs: int):
    if name == "thgnn":
        return THGNNWrapper(seed=seed, use_aaa=use_aaa, epochs=epochs)
    if name == "statistical_baseline":
        return StatisticalBaseline(seed=seed, injection_rate=rate)
    if name == "anti_fakeu_baseline":
        return AntiFakeUBaseline(seed=seed, injection_rate=rate)
    raise ValueError(f"Unknown detector: {name!r}")


# ---------------------------------------------------------------------------
# Single-config execution
# ---------------------------------------------------------------------------

def _run_one_detector(
    cfg: ExpConfig,
    poisoned: HeteroDataset,
    device: str,
    epochs: int,
    command: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"config": asdict(cfg), "status": "running"}
    t0 = time.perf_counter()
    try:
        detector = _build_detector(
            cfg.detector, cfg.seed, cfg.rate, use_aaa=(cfg.variant == "aaa"), epochs=epochs,
        )
        detector.fit(poisoned, device=device, seed=cfg.seed)
        scores = detector.score_users(poisoned)
        labels = poisoned.labels.numpy()

        if np.isnan(scores).any():
            raise ValueError(f"NaN in scores for {cfg}")

        f1, threshold = best_threshold_f1(labels, scores)
        dr = detection_rate(labels, scores, threshold=threshold)
        far = false_alarm_rate(labels, scores, threshold=threshold)

        npz_path = save_scores(
            SCORES_DIR,
            dataset=cfg.dataset, attack=cfg.attack, rate=cfg.rate,
            detector=cfg.detector, variant=cfg.variant, seed=cfg.seed,
            user_ids=np.arange(poisoned.num_users), y_true=labels, y_score=scores,
            command=command, extra_config=asdict(cfg),
        )

        result.update(
            status="ok",
            f1=round(float(f1), 4), dr=round(float(dr), 4), far=round(float(far), 4),
            threshold=round(float(threshold), 4),
            npz_path=str(npz_path),
            secs=round(time.perf_counter() - t0, 2),
        )
        logger.info(
            "[%s/%s/%.3f/%s/%s/seed=%d] OK  F1=%.4f DR=%.4f FAR=%.4f (%.1fs)",
            cfg.dataset, cfg.attack, cfg.rate, cfg.detector, cfg.variant, cfg.seed,
            f1, dr, far, result["secs"],
        )
    except Exception:  # noqa: BLE001
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()
        logger.error(
            "[%s/%s/%.3f/%s/%s/seed=%d] FAILED:\n%s",
            cfg.dataset, cfg.attack, cfg.rate, cfg.detector, cfg.variant, cfg.seed,
            result["traceback"],
        )
    return result


def _process_result(cfg: ExpConfig, res: Dict[str, Any], stage: str, hf_repo: str, dry_run: bool) -> None:
    git_hash = _git_commit_hash()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    upload_paths: List[Path] = [LOG_MD]

    if res["status"] == "ok":
        row = {
            "timestamp": ts, "stage": stage, "dataset": cfg.dataset, "attack": cfg.attack,
            "rate": cfg.rate, "detector": cfg.detector, "variant": cfg.variant, "seed": cfg.seed,
            "status": "OK", "f1": res["f1"], "dr": res["dr"], "far": res["far"],
            "git_commit": git_hash, "notes": "",
        }
        npz_path = Path(res["npz_path"])
        upload_paths += [npz_path, npz_path.with_suffix(".json")]
    else:
        stem = f"{cfg.dataset}_{cfg.attack}_{cfg.rate}_{cfg.detector}_{cfg.variant}_{cfg.seed}"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        tb_path = LOGS_DIR / f"failed_{stem}.txt"
        tb_path.write_text(res["traceback"], encoding="utf-8")
        row = {
            "timestamp": ts, "stage": stage, "dataset": cfg.dataset, "attack": cfg.attack,
            "rate": cfg.rate, "detector": cfg.detector, "variant": cfg.variant, "seed": cfg.seed,
            "status": "FAILED", "git_commit": git_hash,
            "notes": f"see {tb_path.relative_to(REPO_ROOT)}: {res['traceback'].splitlines()[-1]}",
        }
        upload_paths.append(tb_path)

    _append_log_row(row)
    _maybe_upload(upload_paths, hf_repo, dry_run)


# ---------------------------------------------------------------------------
# Grid construction + execution
# ---------------------------------------------------------------------------

def _build_grid_configs(variant: str, seed: int) -> List[ExpConfig]:
    configs: List[ExpConfig] = []
    for attack in GRID_ATTACKS:
        for rate in GRID_RATES:
            for det in GRID_DETECTORS:
                v = variant if det == "thgnn" else "na"
                configs.append(ExpConfig(GRID_DATASET, attack, rate, det, v, seed))
    return configs


def _run_configs(
    configs: List[ExpConfig],
    dataset: HeteroDataset,
    args: argparse.Namespace,
    stage: str,
    command: str,
    n_targets: int,
    budget_pct: float,
) -> List[Dict[str, Any]]:
    """Group by (attack, rate) so the attack + poisoned dataset are built once
    per cell and reused across detectors, then run each detector."""
    results: List[Dict[str, Any]] = []
    epochs = 1 if args.dry_run else 30

    by_attack_rate: Dict[Tuple[str, float], List[ExpConfig]] = {}
    for cfg in configs:
        by_attack_rate.setdefault((cfg.attack, cfg.rate), []).append(cfg)

    for (attack, rate), cell_cfgs in by_attack_rate.items():
        # Skip building the attack entirely if every detector in this cell
        # already has a score file on disk (resumability across kills).
        pending = [
            c for c in cell_cfgs
            if args.overwrite or not score_path(
                SCORES_DIR, c.dataset, c.attack, c.rate, c.detector, c.variant, c.seed,
            ).exists()
        ]
        skipped = [c for c in cell_cfgs if c not in pending]
        for cfg in skipped:
            logger.info(
                "[%s/%s/%.3f/%s/%s/seed=%d] SKIPPED (score file already exists; reused)",
                cfg.dataset, cfg.attack, cfg.rate, cfg.detector, cfg.variant, cfg.seed,
            )
            _append_log_row({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage": stage, "dataset": cfg.dataset, "attack": cfg.attack, "rate": cfg.rate,
                "detector": cfg.detector, "variant": cfg.variant, "seed": cfg.seed,
                "status": "SKIPPED", "git_commit": _git_commit_hash(),
                "notes": "score file already existed — reused",
            })

        if not pending:
            continue

        seed = pending[0].seed
        logger.info("Building attack=%s rate=%.3f seed=%d ...", attack, rate, seed)
        n_fake = max(1, round(rate * dataset.num_users))
        targets = _select_targets(dataset, n_targets, seed)
        try:
            probe_cfg = pending[0]
            attack_obj = _build_attack(
                probe_cfg, dataset, args.device, args.cache_dir, args.dry_run,
                agent4sr_cache_dir=args.agent4sr_cache_dir,
            )
            profiles = attack_obj.generate_profiles(n_fake, budget_pct, targets)
            poisoned = attack_obj.inject(dataset, profiles)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("Attack build FAILED for %s/%.3f: %s", attack, rate, tb)
            for cfg in pending:
                res = {"status": "error", "traceback": tb}
                _process_result(cfg, res, stage, args.hf_repo, args.dry_run)
                results.append(res)
            continue

        for cfg in pending:
            res = _run_one_detector(cfg, poisoned, args.device, epochs, command)
            _process_result(cfg, res, stage, args.hf_repo, args.dry_run)
            results.append(res)

    return results


# ---------------------------------------------------------------------------
# Stage: reproduction
# ---------------------------------------------------------------------------

def run_reproduction(args: argparse.Namespace, dataset: HeteroDataset, command: str) -> None:
    paper_cfg = _load_paper_config()
    seed = 42
    stage = "reproduction"
    n_targets = paper_cfg["experiment_grid"]["n_targets"] if not args.dry_run else 3
    budget_pct = paper_cfg["experiment_grid"]["budget_pct"] if not args.dry_run else 0.10

    configs = [
        ExpConfig(GRID_DATASET, "random", 0.01, det, "base", seed)
        for det in GRID_DETECTORS
    ]

    logger.info(
        "Reproduction check: dataset=%s attack=random rate=0.01 seed=%d "
        "variant=base (current/default use_aaa setting; equivalent to the "
        "pre-fix code path since use_aaa was never wired through before).",
        GRID_DATASET, seed,
    )

    results = _run_configs(configs, dataset, args, stage, command, n_targets, budget_pct)
    by_detector = {r["config"]["detector"]: r for r in results if "config" in r}

    print("\n" + "=" * 78)
    print("  REPRODUCTION CHECK — ml1m / random / rate=0.01 / seed=42")
    print("=" * 78)
    print(f"{'detector':<24}{'expected_f1':>12}{'observed_f1':>14}{'diff':>10}  verdict")
    print("-" * 78)

    all_match = True
    any_comparable = False
    for det in GRID_DETECTORS:
        res = by_detector.get(det)
        expected = _expected_f1(paper_cfg, GRID_DATASET, "random", det)
        if args.dry_run:
            print(f"{det:<24}{'n/a (synthetic data in --dry-run)':>50}")
            continue
        if res is None or res.get("status") != "ok":
            print(f"{det:<24}{'—':>12}{'RUN FAILED':>14}")
            all_match = False
            continue
        observed = res["f1"]
        if expected is None:
            print(f"{det:<24}{'not set':>12}{observed:>14.4f}{'—':>10}  SKIP")
            continue
        any_comparable = True
        diff = abs(observed - expected)
        ok = diff <= TOLERANCE
        all_match = all_match and ok
        print(f"{det:<24}{expected:>12.4f}{observed:>14.4f}{diff:>10.4f}  {'MATCH' if ok else 'MISMATCH'}")

    print("-" * 78)

    if args.dry_run:
        verdict = "DRY_RUN"
    elif not any_comparable:
        verdict = "NO_COMPARISON"
    else:
        verdict = "MATCH" if all_match else "MISMATCH"
    print(f"  VERDICT: {verdict}")
    print("=" * 78 + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(
        f"{verdict}\n"
        f"git_commit={_git_commit_hash()}\n"
        f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
        encoding="utf-8",
    )
    _append_log_row({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "reproduction_verdict", "dataset": GRID_DATASET, "attack": "random",
        "rate": 0.01, "detector": "ALL", "variant": "base", "seed": seed,
        "status": verdict, "git_commit": _git_commit_hash(),
        "notes": "see REPRODUCTION_VERDICT.txt",
    })
    _maybe_upload([LOG_MD, VERDICT_PATH], args.hf_repo, args.dry_run)

    if verdict == "MISMATCH":
        logger.error(
            "MISMATCH: reproduction did not match Table 1 within tolerance ±%.2f. "
            "Per session rules, STOPPING here — do not proceed to --stage grid "
            "until this is resolved or explicitly overridden.", TOLERANCE,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Stage: grid
# ---------------------------------------------------------------------------

def run_grid(args: argparse.Namespace, dataset: HeteroDataset, command: str) -> None:
    paper_cfg = _load_paper_config()
    variant = args.variant
    stage = f"grid_{variant}"
    n_targets = paper_cfg["experiment_grid"]["n_targets"] if not args.dry_run else 3
    budget_pct = paper_cfg["experiment_grid"]["budget_pct"] if not args.dry_run else 0.10

    configs = _build_grid_configs(variant, args.seed)
    logger.info(
        "Grid stage: variant=%s  dataset=%s  attacks=%s  rates=%s  detectors=%s  "
        "seed=%d  -> %d configs planned",
        variant, GRID_DATASET, GRID_ATTACKS, GRID_RATES, GRID_DETECTORS, args.seed, len(configs),
    )

    results = _run_configs(configs, dataset, args, stage, command, n_targets, budget_pct)
    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") == "error"]
    print(f"\nGrid variant={variant} complete: {len(ok)} ok, {len(err)} failed "
          f"(see {LOG_MD.relative_to(REPO_ROOT)} for the full log, "
          f"results/logs/failed_*.txt for tracebacks).\n")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def run_dry_run(args: argparse.Namespace, command: str) -> None:
    t0 = time.perf_counter()
    dataset = _build_synthetic_dataset(seed=0)
    logger.info("Dry run: synthetic dataset — %s", dataset.summary())

    configs = [
        ExpConfig(dataset.dataset_name, DRY_RUN_ATTACK, DRY_RUN_RATE, "thgnn", "base", 42),
        ExpConfig(dataset.dataset_name, DRY_RUN_ATTACK, DRY_RUN_RATE, "statistical_baseline", "na", 42),
    ]
    stage = "dry_run"
    results = _run_configs(configs, dataset, args, stage, command, n_targets=3, budget_pct=0.10)

    print("\n" + "=" * 78)
    print("  DRY RUN SUMMARY")
    print("=" * 78)
    for r in results:
        cfg = r.get("config", {})
        if r.get("status") == "ok":
            print(f"  OK      {cfg.get('detector'):<22} F1={r['f1']:.4f} DR={r['dr']:.4f} "
                  f"FAR={r['far']:.4f}  scores -> {r['npz_path']}")
        else:
            print(f"  FAILED  {cfg.get('detector')}: {r.get('traceback', '')[-300:]}")
    elapsed = time.perf_counter() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 78 + "\n")

    if any(r.get("status") != "ok" for r in results):
        logger.error("Dry run had failures — see traceback above.")
        sys.exit(1)
    if elapsed > 120:
        logger.warning("Dry run took %.1fs, exceeding the 2-minute target.", elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["reproduction", "grid"], default=None)
    p.add_argument("--variant", choices=["base", "aaa"], default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--cache-dir", default="data/processed")
    p.add_argument("--agent4sr-cache-dir", default="data/agent4sr_cache",
                    help="Pre-populated Agent4SR prompt/response cache (see REPRODUCE.md / "
                         "A5 Kaggle notes) — must be attached, never regenerated live.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true",
                    help="Re-run configs even if a score file already exists.")
    p.add_argument("--hf-repo", default="shiv3589/thgnn-shilling-results")
    return p.parse_args()


def _setup_logging(stage: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{stage}_{time.strftime('%Y%m%dT%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,
    )
    logger.info("Logging to %s", log_path)


def _check_raw_data(raw_dir: str) -> bool:
    return (Path(raw_dir) / "ml-1m" / "ratings.dat").exists()


def main() -> None:
    args = _parse_args()
    if not args.dry_run and args.stage is None:
        print("Specify --stage {reproduction|grid} or use --dry-run. See --help.")
        sys.exit(1)
    if args.stage == "grid" and args.variant is None:
        print("--stage grid requires --variant {base|aaa}.")
        sys.exit(1)

    args.device = args.device or _auto_device()
    stage_label = "dry_run" if args.dry_run else (
        f"grid_{args.variant}" if args.stage == "grid" else args.stage
    )
    _setup_logging(stage_label)
    logger.info("Device: %s", args.device)
    _ensure_log_section(stage_label)

    command = "python " + " ".join(sys.argv)

    if args.dry_run:
        run_dry_run(args, command)
        return

    if not _check_raw_data(args.raw_dir):
        logger.error(
            "Raw ml1m data not found at %s/ml-1m/ratings.dat. Per the no-fabrication "
            "rule, this script will not substitute synthetic data for a real "
            "reproduction/grid run. Attach the Kaggle Dataset (see REPRODUCE.md §2) "
            "or use --dry-run to sanity-check the pipeline on synthetic data.",
            args.raw_dir,
        )
        sys.exit(1)

    logger.info("Loading ml1m dataset from %s ...", args.raw_dir)
    dataset = MovieLens1MLoader(raw_dir=args.raw_dir).load()
    logger.info("%s", dataset.summary())

    if args.stage == "reproduction":
        run_reproduction(args, dataset, command)
    else:
        run_grid(args, dataset, command)


if __name__ == "__main__":
    main()
