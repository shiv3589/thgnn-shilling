#!/usr/bin/env python
"""Provenance check: every compiled paper table must trace to a committed,
completed (status=="ok") raw result file under results/raw/.

Run before committing changes to paper/tables/*.tex (or wire into a
pre-commit hook — not done automatically here, see REPRODUCE.md). Exits 1
and prints a clear failure if any compiled table lacks a traceable real
result.

Background: paper/main.pdf was previously compiled and committed (git
history: commits d34ac1e, 16a1012) while paper/tables/{main_detection,
ablation,downstream_ndcg}.tex contained 100% fabricated data — see
results/EXPERIMENT_LOG.md's fabrication-audit entry. This check exists so
that can never happen silently again: a raw result file existing on disk
is not enough (run_experiments.py/run_ablation.py always write a result
file even on failure), and a file being status=="ok" is not enough either
if it was never committed (git history is what makes provenance provable
after the fact, which is exactly the capability that was missing).

Usage
-----
    python scripts/check_table_provenance.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_RAW = REPO_ROOT / "results" / "raw"
EXPERIMENTS_TEX = REPO_ROOT / "paper" / "sections" / "experiments.tex"

# Manifest: compiled table (relative to repo root) -> glob pattern(s), under
# results/raw/, for the raw result files it must trace back to. Keep this in
# sync with experiments.tex's \input{} chain — check_input_chain() below
# cross-checks it automatically and warns if they drift apart.
TABLE_MANIFEST: Dict[str, List[str]] = {
    "paper/tables/main_detection.tex": [
        "ml1m_*.json", "amazon_books_*.json", "amazon_clothing_*.json", "yelp2018_*.json",
    ],
    "paper/tables/ablation.tex": [
        "ablation_*.json",
    ],
    "paper/tables/downstream_ndcg.tex": [
        "ml1m_*.json", "amazon_books_*.json", "amazon_clothing_*.json", "yelp2018_*.json",
    ],
}


def _is_git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return result.returncode == 0


def _has_traceable_result(patterns: List[str]) -> Tuple[bool, str]:
    """(ok, detail) — ok=True iff at least one status=="ok" raw result file
    matching any pattern exists on disk AND is committed to git."""
    if not RESULTS_RAW.is_dir():
        return False, "results/raw/ does not exist"

    checked = 0
    for pattern in patterns:
        for path in RESULTS_RAW.glob(pattern):
            checked += 1
            try:
                with open(path, encoding="utf-8") as f:
                    if json.load(f).get("status") != "ok":
                        continue
            except (json.JSONDecodeError, OSError):
                continue
            if _is_git_tracked(path):
                return True, f"traced to {path.relative_to(REPO_ROOT)} (committed, status=ok)"

    if checked == 0:
        return False, f"no files on disk matching {patterns}"
    return False, (
        f"{checked} file(s) on disk matched {patterns}, but none are both "
        f"status=='ok' and committed to git"
    )


def check_input_chain() -> List[str]:
    """Cross-check TABLE_MANIFEST against experiments.tex's actual
    \\input{tables/...} lines, so the manifest can't silently drift from
    what the paper actually compiles."""
    if not EXPERIMENTS_TEX.exists():
        return [f"{EXPERIMENTS_TEX} not found — cannot verify manifest against input chain"]

    text = EXPERIMENTS_TEX.read_text(encoding="utf-8")
    compiled = {
        f"paper/tables/{line.split(r'{tables/')[1].split('}')[0]}.tex"
        for line in text.splitlines()
        if r"\input{tables/" in line
    }
    manifest_keys = set(TABLE_MANIFEST)

    issues: List[str] = []
    missing = compiled - manifest_keys
    stale = manifest_keys - compiled
    if missing:
        issues.append(f"Compiled by experiments.tex but missing from TABLE_MANIFEST "
                       f"(update this script): {sorted(missing)}")
    if stale:
        issues.append(f"In TABLE_MANIFEST but not currently \\input{{}}'d by experiments.tex "
                       f"(remove, or leave as a harmless extra check): {sorted(stale)}")
    return issues


def main() -> int:
    for issue in check_input_chain():
        print(f"[WARN] {issue}")

    failures: List[str] = []
    for table, patterns in TABLE_MANIFEST.items():
        ok, detail = _has_traceable_result(patterns)
        print(f"[{'OK' if ok else 'FAIL'}] {table}: {detail}")
        if not ok:
            failures.append(table)

    print()
    if failures:
        print(f"PROVENANCE CHECK FAILED for {len(failures)} table(s): {failures}")
        print("Do not commit these .tex files as if they reflect real results.")
        print("Either commit the underlying results/raw/*.json first, or keep the")
        print("PROVENANCE UNVERIFIED comment at the top of the table (see")
        print("results/EXPERIMENT_LOG.md).")
        return 1

    print("PROVENANCE CHECK PASSED — every compiled table traces to a committed, completed raw result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
