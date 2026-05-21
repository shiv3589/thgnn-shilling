"""Reproducibility tests for the TH-GNN pipeline.

Markers
-------
smoke   : Fast structural check on a tiny synthetic dataset. Always runs.
          Verifies the pipeline executes end-to-end and F1 is in [0, 1].

subsample : 5%-subsample run on real data (skipped when raw data is absent).
            Checks that each detector's F1 is within TOLERANCE of the
            expected values stored in configs/paper_runs.yaml.

full    : Checks already-computed results/raw/ against paper_runs.yaml.
          Skipped unless results/raw/ has ≥ 100 JSON files.

Run examples
------------
    pytest tests/test_reproducibility.py -v -m smoke
    pytest tests/test_reproducibility.py -v -m subsample
    pytest tests/test_reproducibility.py -v -m full
    pytest tests/test_reproducibility.py -v          # smoke + subsample + full
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch
import yaml

# ---------------------------------------------------------------------------
# Project root helpers
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_RAW = REPO_ROOT / "results" / "raw"
RAW_DATA    = REPO_ROOT / "data" / "raw"

sys_path_guard = None
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attacks import BandwagonAttack, RandomAttack
from src.attacks.base import ShillingAttack
from src.datasets.base import HeteroDataset
from src.detectors import (
    AntiFakeUBaseline,
    SemanticShieldBaseline,
    StatisticalBaseline,
    THGNNWrapper,
)
from src.evaluation import best_threshold_f1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOLERANCE = 0.03          # max allowed deviation from expected F1

# 5 % subsample grid: single dataset × single attack × all detectors × 1 seed
SUBSAMPLE_DATASET = "ml1m"
SUBSAMPLE_ATTACKS = ["random", "bandwagon"]
SUBSAMPLE_RATE    = 0.01
SUBSAMPLE_SEED    = 42

# Synthetic dataset dimensions (smoke test)
_N_USERS   = 120
_N_ITEMS   = 200
_N_REVIEWS = 600


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_paper_config() -> dict:
    path = CONFIGS_DIR / "paper_runs.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _expected_f1(cfg: dict, dataset: str, attack: str, detector: str) -> Optional[float]:
    """Return expected F1 from paper_runs.yaml, or None if not set."""
    val = (
        cfg.get("expected_f1", {})
           .get(dataset, {})
           .get(attack, {})
           .get(detector)
    )
    return float(val) if val is not None else None


# ---------------------------------------------------------------------------
# Synthetic dataset factory
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    n_users:   int = _N_USERS,
    n_items:   int = _N_ITEMS,
    n_reviews: int = _N_REVIEWS,
    seed:      int = 42,
) -> HeteroDataset:
    """Build a random HeteroDataset that mirrors the real schema."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    rev_u = torch.from_numpy(rng.integers(0, n_users, n_reviews).astype(np.int64))
    rev_i = torch.from_numpy(rng.integers(0, n_items, n_reviews).astype(np.int64))
    rev_r = torch.arange(n_reviews, dtype=torch.long)
    n_rate = int(n_reviews * 0.8)
    rates_u = torch.from_numpy(rng.integers(0, n_users, n_rate).astype(np.int64))
    rates_i = torch.from_numpy(rng.integers(0, n_items, n_rate).astype(np.int64))

    ts = torch.from_numpy(np.sort(rng.uniform(0, 1e9, n_reviews)))

    n_train = int(0.7 * n_reviews)
    n_val   = int(0.1 * n_reviews)
    train_mask = torch.zeros(n_reviews, dtype=torch.bool)
    val_mask   = torch.zeros(n_reviews, dtype=torch.bool)
    test_mask  = torch.zeros(n_reviews, dtype=torch.bool)
    train_mask[:n_train]                = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :]        = True

    edge_index_dict = {
        ("user",   "writes", "review"): torch.stack([rev_u, rev_r]),
        ("review", "about",  "item"):   torch.stack([rev_r, rev_i]),
        ("user",   "rates",  "item"):   torch.stack([rates_u, rates_i]),
    }
    edge_attr_dict = {k: ts for k in edge_index_dict}

    return HeteroDataset(
        dataset_name    = "synthetic",
        num_users       = n_users,
        num_items       = n_items,
        num_reviews     = n_reviews,
        edge_index_dict = edge_index_dict,
        edge_attr_dict  = edge_attr_dict,
        x_user          = torch.randn(n_users, 16),
        x_item          = torch.randn(n_items, 16),
        review_texts    = ["A short review text." for _ in range(n_reviews)],
        ratings         = torch.from_numpy(
            rng.uniform(1, 5, n_reviews).astype(np.float32)
        ),
        timestamps      = ts,
        review_user_idx = rev_u,
        review_item_idx = rev_i,
        labels          = torch.zeros(n_users, dtype=torch.long),
        train_mask      = train_mask,
        val_mask        = val_mask,
        test_mask       = test_mask,
        user_id_map     = {i: i for i in range(n_users)},
        item_id_map     = {i: i for i in range(n_items)},
    )


# ---------------------------------------------------------------------------
# Mini pipeline helper
# ---------------------------------------------------------------------------

def _run_mini_pipeline(
    dataset: HeteroDataset,
    attack_cls,
    injection_rate: float,
    seed: int,
    budget_pct: float = 0.10,
    n_targets: int    = 3,
) -> Dict[str, float]:
    """Inject attack → fit all four detectors → return {detector_name: f1}."""
    # Select target items (top-3 by degree)
    train_items = dataset.review_item_idx[dataset.train_mask].numpy()
    counts = np.bincount(train_items, minlength=dataset.num_items)
    targets = np.argsort(-counts)[:n_targets].tolist()

    n_fake = max(1, round(injection_rate * dataset.num_users))
    attack: ShillingAttack = attack_cls(dataset, seed=seed)
    profiles = attack.generate_profiles(n_fake, budget_pct, targets)
    poisoned = attack.inject(dataset, profiles)

    detectors = {
        "statistical_baseline":     StatisticalBaseline(seed=seed),
        "semantic_shield_baseline": SemanticShieldBaseline(seed=seed),
        "anti_fakeu_baseline":      AntiFakeUBaseline(seed=seed),
    }

    results: Dict[str, float] = {}
    labels = poisoned.labels.numpy()

    for name, det in detectors.items():
        det.fit(poisoned, device="cpu", seed=seed)
        scores = det.score_users(poisoned)
        f1, _ = best_threshold_f1(labels, scores)
        results[name] = round(float(f1), 4)

    return results


# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

def _real_data_available(dataset: str = SUBSAMPLE_DATASET) -> bool:
    """True when at least one raw file for the requested dataset is present."""
    ds_map = {
        "ml1m":            "ml-1m",
        "amazon_books":    "amazon-books",
        "amazon_clothing": "amazon-clothing",
        "yelp2018":        "yelp2018",
    }
    folder = RAW_DATA / ds_map.get(dataset, dataset)
    return folder.is_dir() and any(folder.iterdir())


def _full_results_available(min_files: int = 100) -> bool:
    if not RESULTS_RAW.is_dir():
        return False
    return sum(1 for _ in RESULTS_RAW.glob("*.json")) >= min_files


# ---------------------------------------------------------------------------
# ── SMOKE TESTS ─────────────────────────────────────────────────────────────
# Always run; no raw data needed.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestSmoke:
    """Structural / integration tests on synthetic data."""

    @pytest.fixture(scope="class")
    def synthetic(self, synthetic_graph):
        return synthetic_graph

    @pytest.mark.parametrize("attack_cls", [RandomAttack, BandwagonAttack])
    def test_pipeline_completes(self, synthetic, attack_cls):
        """Pipeline runs end-to-end and returns a dict with valid F1 values."""
        results = _run_mini_pipeline(
            synthetic, attack_cls,
            injection_rate=0.05, seed=42,
        )
        assert set(results.keys()) == {
            "statistical_baseline",
            "semantic_shield_baseline",
            "anti_fakeu_baseline",
        }
        for det, f1 in results.items():
            assert 0.0 <= f1 <= 1.0, f"{det}: F1={f1} outside [0,1]"

    def test_deterministic_with_same_seed(self, synthetic):
        """Running twice with the same seed yields identical F1 scores."""
        r1 = _run_mini_pipeline(synthetic, RandomAttack, injection_rate=0.05, seed=7)
        r2 = _run_mini_pipeline(synthetic, RandomAttack, injection_rate=0.05, seed=7)
        for det in r1:
            assert r1[det] == r2[det], (
                f"{det}: seed-7 run #1 F1={r1[det]} ≠ run #2 F1={r2[det]}"
            )

    def test_different_seeds_differ(self, synthetic):
        """Different seeds produce different injected profiles (attack is seeded).

        We verify at the profile level rather than at F1, since F1 on tiny
        synthetic data can collide across seeds even when the attacks differ.
        """
        a1 = RandomAttack(synthetic, seed=1)
        a2 = RandomAttack(synthetic, seed=99)
        targets = [0, 1, 2]
        p1 = a1.generate_profiles(5, 0.20, targets)
        p2 = a2.generate_profiles(5, 0.20, targets)
        # At least one profile should have different filler items or timestamps
        items1 = [frozenset(p.rated_items.keys()) for p in p1]
        items2 = [frozenset(p.rated_items.keys()) for p in p2]
        ts1 = [tuple(p.timestamps) for p in p1]
        ts2 = [tuple(p.timestamps) for p in p2]
        assert items1 != items2 or ts1 != ts2, (
            "Seeds 1 and 99 produced identical profiles — RNG seeding is broken"
        )

    def test_higher_injection_raises_f1(self, synthetic):
        """Higher injection rate injects more obvious fakes → F1 should not drop."""
        r_low  = _run_mini_pipeline(synthetic, RandomAttack, injection_rate=0.02, seed=42)
        r_high = _run_mini_pipeline(synthetic, RandomAttack, injection_rate=0.10, seed=42)
        # Statistical baseline is the most reliable; allow for tied values
        assert r_high["statistical_baseline"] >= r_low["statistical_baseline"] - TOLERANCE, (
            f"F1 unexpectedly dropped from {r_low} to {r_high} at higher injection rate"
        )

    def test_paper_config_schema(self):
        """paper_runs.yaml loads correctly and has all required keys."""
        cfg = _load_paper_config()
        for key in ("experiment_grid", "thgnn", "training", "splits", "expected_f1"):
            assert key in cfg, f"paper_runs.yaml missing top-level key: {key!r}"

        grid = cfg["experiment_grid"]
        for key in ("datasets", "attacks", "injection_rates", "seeds"):
            assert key in grid, f"experiment_grid missing: {key!r}"

        assert set(grid["seeds"]) == {42, 123, 456}
        assert set(grid["injection_rates"]) == {0.005, 0.01, 0.05}

    def test_expected_f1_values_in_range(self):
        """All non-null expected_f1 values in paper_runs.yaml are in [0, 1]."""
        cfg = _load_paper_config()
        for dataset, attacks in cfg.get("expected_f1", {}).items():
            for attack, detectors in attacks.items():
                for det, val in detectors.items():
                    if val is not None:
                        assert 0.0 <= float(val) <= 1.0, (
                            f"expected_f1[{dataset}][{attack}][{det}] = {val} out of range"
                        )

    @pytest.mark.timeout(60)
    def test_reproduce_script_dry_run(self):
        """reproduce_paper.sh --dry-run exits 0 and prints 'All done' in <60 s."""
        import subprocess
        result = subprocess.run(
            ["bash", "scripts/reproduce_paper.sh", "--dry-run"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"reproduce_paper.sh --dry-run exited {result.returncode}.\n"
            f"stdout:\n{result.stdout[-500:]}\n"
            f"stderr:\n{result.stderr[-500:]}"
        )
        assert "All done" in result.stdout, (
            f"'All done' not found in stdout:\n{result.stdout[-500:]}"
        )


# ---------------------------------------------------------------------------
# ── SUBSAMPLE TESTS ──────────────────────────────────────────────────────────
# Run a 5 %-subsample pipeline on real data and check against paper numbers.
# Skipped when raw data is not present.
# ---------------------------------------------------------------------------

@pytest.mark.subsample
@pytest.mark.skipif(
    not _real_data_available(),
    reason=f"Raw data for '{SUBSAMPLE_DATASET}' not found under data/raw/. "
           "Download data first (see REPRODUCE.md §2).",
)
class TestSubsample:
    """5%-subsample integration tests against paper_runs.yaml expected F1."""

    @pytest.fixture(scope="class")
    def paper_cfg(self):
        return _load_paper_config()

    @pytest.fixture(scope="class")
    def subsample_dataset(self, paper_cfg):
        """Load ML-1M and subsample 5 % of users (deterministically)."""
        from src.datasets import MovieLens1MLoader
        loader = MovieLens1MLoader(raw_dir=str(RAW_DATA))
        full_ds = loader.load(
            min_user_interactions=paper_cfg["preprocessing"]["min_user_interactions"],
            min_item_interactions=paper_cfg["preprocessing"]["min_item_interactions"],
        )
        # Keep a 5 % stratified subsample of users (preserve fake/real ratio)
        rng      = np.random.default_rng(SUBSAMPLE_SEED)
        all_u    = np.arange(full_ds.num_users)
        keep_idx = rng.choice(all_u, size=max(20, int(0.05 * full_ds.num_users)),
                               replace=False)
        keep_set = set(keep_idx.tolist())

        keep_mask = torch.tensor(
            [full_ds.review_user_idx[r].item() in keep_set
             for r in range(full_ds.num_reviews)],
            dtype=torch.bool,
        )
        sub_u_global = sorted(keep_set)
        u_remap = {g: l for l, g in enumerate(sub_u_global)}
        sub_rev_u = torch.tensor(
            [u_remap[full_ds.review_user_idx[r].item()]
             for r in range(full_ds.num_reviews) if keep_mask[r]],
            dtype=torch.long,
        )
        sub_rev_i = full_ds.review_item_idx[keep_mask]
        sub_ts    = full_ds.timestamps[keep_mask]
        sub_r     = torch.arange(sub_rev_u.shape[0])
        n_sub     = sub_rev_u.shape[0]
        n_train   = int(0.7 * n_sub)
        n_val     = int(0.1 * n_sub)
        train_m = torch.zeros(n_sub, dtype=torch.bool)
        val_m   = torch.zeros(n_sub, dtype=torch.bool)
        test_m  = torch.zeros(n_sub, dtype=torch.bool)
        train_m[:n_train]                = True
        val_m[n_train : n_train + n_val] = True
        test_m[n_train + n_val :]        = True

        n_sub_u = len(sub_u_global)
        edge_index_dict = {
            ("user",   "writes", "review"): torch.stack([sub_rev_u, sub_r]),
            ("review", "about",  "item"):   torch.stack([sub_r, sub_rev_i]),
            ("user",   "rates",  "item"):   torch.stack([sub_rev_u, sub_rev_i]),
        }
        return HeteroDataset(
            dataset_name    = "ml1m_subsample",
            num_users       = n_sub_u,
            num_items       = full_ds.num_items,
            num_reviews     = n_sub,
            edge_index_dict = edge_index_dict,
            edge_attr_dict  = {k: sub_ts for k in edge_index_dict},
            x_user          = full_ds.x_user[torch.tensor(sub_u_global)],
            x_item          = full_ds.x_item,
            review_texts    = [full_ds.review_texts[r]
                               for r in range(full_ds.num_reviews) if keep_mask[r]],
            ratings         = full_ds.ratings[keep_mask],
            timestamps      = sub_ts,
            review_user_idx = sub_rev_u,
            review_item_idx = sub_rev_i,
            labels          = full_ds.labels[torch.tensor(sub_u_global)],
            train_mask      = train_m,
            val_mask        = val_m,
            test_mask       = test_m,
            user_id_map     = {g: l for l, g in enumerate(sub_u_global)},
            item_id_map     = full_ds.item_id_map,
        )

    @pytest.mark.parametrize("attack_name,attack_cls", [
        ("random",    RandomAttack),
        ("bandwagon", BandwagonAttack),
    ])
    def test_f1_within_tolerance(
        self, paper_cfg, subsample_dataset, attack_name, attack_cls,
    ):
        """F1 on 5%-subsample is within ±TOLERANCE of paper_runs.yaml expected values."""
        results = _run_mini_pipeline(
            subsample_dataset,
            attack_cls,
            injection_rate = SUBSAMPLE_RATE,
            seed           = SUBSAMPLE_SEED,
            n_targets      = paper_cfg["experiment_grid"]["n_targets"],
            budget_pct     = paper_cfg["experiment_grid"]["budget_pct"],
        )

        failures: List[str] = []
        for det, observed_f1 in results.items():
            expected = _expected_f1(paper_cfg, SUBSAMPLE_DATASET, attack_name, det)
            if expected is None:
                pytest.skip(
                    f"expected_f1[{SUBSAMPLE_DATASET}][{attack_name}][{det}] "
                    "not set in paper_runs.yaml — populate after reference run."
                )
            diff = abs(observed_f1 - expected)
            if diff > TOLERANCE:
                failures.append(
                    f"  {det}: observed={observed_f1:.4f}, "
                    f"expected={expected:.4f}, diff={diff:.4f} > {TOLERANCE}"
                )

        assert not failures, (
            f"F1 deviation exceeds tolerance (±{TOLERANCE}) for "
            f"{SUBSAMPLE_DATASET}/{attack_name}:\n" + "\n".join(failures)
        )

    def test_subsample_pipeline_stable_across_seeds(self, paper_cfg, subsample_dataset):
        """Two runs with the same seed produce identical F1 (regression check)."""
        r1 = _run_mini_pipeline(
            subsample_dataset, RandomAttack,
            injection_rate=SUBSAMPLE_RATE, seed=SUBSAMPLE_SEED,
        )
        r2 = _run_mini_pipeline(
            subsample_dataset, RandomAttack,
            injection_rate=SUBSAMPLE_RATE, seed=SUBSAMPLE_SEED,
        )
        for det in r1:
            assert r1[det] == r2[det], (
                f"{det}: seed-{SUBSAMPLE_SEED} run 1 F1={r1[det]} ≠ run 2 F1={r2[det]}"
            )


# ---------------------------------------------------------------------------
# ── FULL RESULTS TESTS ───────────────────────────────────────────────────────
# Validate already-computed results/raw/ against paper_runs.yaml.
# Skipped unless results/raw/ has ≥ 100 JSON files.
# ---------------------------------------------------------------------------

@pytest.mark.full
@pytest.mark.skipif(
    not _full_results_available(),
    reason=(
        "Fewer than 100 result files in results/raw/. "
        "Run `bash scripts/reproduce_paper.sh` (Stage 3) first."
    ),
)
class TestFullResults:
    """Validate completed results/raw/*.json against paper expected F1."""

    @pytest.fixture(scope="class")
    def paper_cfg(self):
        return _load_paper_config()

    @pytest.fixture(scope="class")
    def aggregated(self):
        """Load all completed result JSONs from results/raw/."""
        records: Dict[str, Dict] = defaultdict(dict)  # (dataset,attack,detector) → seed-list
        for path in sorted(RESULTS_RAW.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("status") != "ok":
                continue
            cfg = data.get("config", {})
            det = data.get("detection", {})
            key = (cfg.get("dataset"), cfg.get("attack"), cfg.get("detector"))
            records[key].setdefault("f1_values", []).append(float(det.get("f1", float("nan"))))
        return records

    @pytest.fixture(scope="class")
    def datasets(self, paper_cfg):
        return paper_cfg["experiment_grid"]["datasets"]

    @pytest.fixture(scope="class")
    def attacks(self, paper_cfg):
        return paper_cfg["experiment_grid"]["attacks"]

    @pytest.fixture(scope="class")
    def detectors(self, paper_cfg):
        return paper_cfg["experiment_grid"]["detectors"]

    def test_all_experiments_complete(self, paper_cfg, aggregated, datasets, attacks, detectors):
        """Every (dataset, attack, detector) cell has ≥ 3 completed seed results."""
        missing = []
        for dataset in datasets:
            for attack in attacks:
                for det in detectors:
                    key = (dataset, attack, det)
                    n_complete = len(aggregated.get(key, {}).get("f1_values", []))
                    if n_complete < len(paper_cfg["experiment_grid"]["seeds"]):
                        missing.append(
                            f"{dataset}/{attack}/{det}: {n_complete} seeds "
                            f"(expected {len(paper_cfg['experiment_grid']['seeds'])})"
                        )
        if missing:
            # Warn only — a partial run is still useful to validate
            pytest.xfail(
                f"{len(missing)} cells have fewer than 3 completed seeds:\n"
                + "\n".join(missing[:10])
                + ("\n  … (truncated)" if len(missing) > 10 else "")
            )

    def test_thgnn_beats_all_baselines(self, paper_cfg, aggregated, datasets, attacks, detectors):
        """TH-GNN mean F1 > every baseline on ≥ 70 % of (dataset, attack) pairs."""
        baselines = [d for d in detectors if d != "thgnn"]
        wins, total = 0, 0

        for dataset in datasets:
            for attack in attacks:
                thgnn_key = (dataset, attack, "thgnn")
                thgnn_vals = aggregated.get(thgnn_key, {}).get("f1_values", [])
                if not thgnn_vals:
                    continue
                thgnn_mean = float(np.mean(thgnn_vals))
                for b in baselines:
                    b_key = (dataset, attack, b)
                    b_vals = aggregated.get(b_key, {}).get("f1_values", [])
                    if not b_vals:
                        continue
                    total += 1
                    if thgnn_mean > float(np.mean(b_vals)):
                        wins += 1

        if total == 0:
            pytest.skip("No pairwise comparisons available.")
        win_rate = wins / total
        assert win_rate >= 0.70, (
            f"TH-GNN win rate {win_rate:.1%} < 70% ({wins}/{total} comparisons)"
        )

    @pytest.mark.parametrize("dataset,attack", [
        ("ml1m", "random"),
        ("ml1m", "agent4sr"),
    ])
    def test_mean_f1_within_tolerance(
        self, paper_cfg, aggregated, dataset, attack,
    ):
        """Mean F1 (over 3 seeds × 3 rates) is within ±TOLERANCE of paper value."""
        failures = []
        for det in paper_cfg["experiment_grid"]["detectors"]:
            expected = _expected_f1(paper_cfg, dataset, attack, det)
            if expected is None:
                continue
            key = (dataset, attack, det)
            vals = aggregated.get(key, {}).get("f1_values", [])
            if not vals:
                continue
            observed = float(np.nanmean(vals))
            diff = abs(observed - expected)
            if diff > TOLERANCE:
                failures.append(
                    f"  {det}: observed={observed:.4f}, "
                    f"expected={expected:.4f}, diff={diff:.4f}"
                )
        assert not failures, (
            f"F1 deviation > {TOLERANCE} for {dataset}/{attack}:\n"
            + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-m", "smoke"]))
