"""Per-user prediction score persistence.

Phase 0 found that no per-user scores were ever saved from the original
experiment runs — only aggregated F1/DR/FAR metrics landed in
``results/raw/*.json``. Every downstream phase (threshold-free metrics,
calibration, PR curves) needs the raw (y_true, y_score) pairs, so this hook
must run for every detector fit in the experiment grid, not just TH-GNN.

Output layout
-------------
``results/scores/{dataset}_{attack}_{rate}_{detector}_{variant}_{seed}.npz``
    user_ids : int64   (num_users,) — contiguous local user index (matches
               the poisoned HeteroDataset's indexing, so it lines up with
               y_true/y_score without a name lookup)
    y_true   : int64   (num_users,) — ground-truth label, 0=genuine 1=fake
    y_score  : float32 (num_users,) — pre-threshold probability in [0, 1]

Plus a JSON sidecar with the same stem containing the exact config and
command used to produce the file.

Note on filename: the task spec that motivated this module described the
stem as ``{dataset}_{attack}_{rate}_{variant}_{seed}`` (no detector). That
collides across detectors sharing a (dataset, attack, rate, variant, seed)
cell — e.g. thgnn and statistical_baseline would silently overwrite each
other. ``detector`` is included here to avoid that; downstream file globbing
should account for the extra field.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def score_path(
    scores_dir: Path,
    dataset: str,
    attack: str,
    rate: float,
    detector: str,
    variant: str,
    seed: int,
) -> Path:
    stem = f"{dataset}_{attack}_{rate}_{detector}_{variant}_{seed}"
    return Path(scores_dir) / f"{stem}.npz"


def save_scores(
    scores_dir: Path,
    *,
    dataset: str,
    attack: str,
    rate: float,
    detector: str,
    variant: str,
    seed: int,
    user_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    command: str,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save one (dataset, attack, rate, detector, variant, seed) score file.

    Returns the path to the written ``.npz`` file (the JSON sidecar is
    written alongside it with the same stem).
    """
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    npz_path = score_path(scores_dir, dataset, attack, rate, detector, variant, seed)
    np.savez(
        npz_path,
        user_ids=np.asarray(user_ids, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=np.int64),
        y_score=np.asarray(y_score, dtype=np.float32),
    )

    sidecar = {
        "dataset": dataset,
        "attack": attack,
        "rate": rate,
        "detector": detector,
        "variant": variant,
        "seed": seed,
        "n_users": int(len(user_ids)),
        "n_fake": int(np.asarray(y_true).sum()),
        "command": command,
    }
    if extra_config:
        sidecar["config"] = extra_config

    json_path = npz_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return npz_path


def load_scores(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load a scores file back into a dict of arrays (for tests / inspection)."""
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}
