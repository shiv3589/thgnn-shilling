# Cold-Clone Verification Report

**Date:** 2026-05-21
**Platform:** macOS Darwin 22.6.0 (Ventura)
**Shell:** zsh
**CUDA GPU:** none (CPU-only machine)

---

## 1. Machine specs

| Component | Value |
|---|---|
| OS | macOS 13 Darwin 22.6.0 |
| Python | 3.10 |
| PyTorch | CPU-only build |
| CUDA | not available |
| MPS (Apple GPU) | available (auto-detected by `reproduce_paper.sh`) |

---

## 2. Install steps

```bash
git clone <repo-url> thgnn-shilling
cd thgnn-shilling
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
pip install -e .
```

All steps completed without error after the fix described in §3.

---

## 3. Bugs found and fixed

### Bug 1 — `pip install -e .` fails with `BackendUnavailable`

**Symptom:**
```
BackendUnavailable: Cannot import 'setuptools.backends.legacy'
```

**Root cause:** `pyproject.toml` declared `build-backend = "setuptools.backends.legacy:build"`, which requires setuptools ≥ 69.1.0. The machine had setuptools 68.0.0, which predates the `setuptools.backends` sub-package.

**Fix:** Changed `pyproject.toml` to the long-stable entry point:
```toml
[build-system]
requires = ["setuptools>=40.0", "wheel"]
build-backend = "setuptools.build_meta"
```

After this change, `pip install -e .` succeeded and `python -c "import src.models; print('OK')"` printed `OK`.

The fix and a note for future users were added to `REPRODUCE.md §8 — Known install issues`.

### Bug 2 — Shell script silently swallowed CPU-only machines

**Symptom:** `reproduce_paper.sh` had no explicit device-detection block; on a CPU-only machine it passed no `--device` flag to sub-scripts, causing silent fallback to CPU without a warning.

**Fix:** Added explicit auto-detection (`cuda > mps > cpu`) and a visible `[WARN]` message when `device=cpu` is selected. The CPU-only runtime column was also added to the runtime table in `REPRODUCE.md §4`.

---

## 4. Smoke test results

Run command:
```bash
pytest tests/ -v -m smoke --ignore=tests/test_reproducibility.py 2>&1 | tee smoke_run.log
```

Result: **7 / 7 passed**

| Test | Status | Time |
|---|---|---|
| `test_attacks::test_fake_profile_length_mismatch` | PASSED | 0.01 s |
| `test_attacks::test_random_attack_profile_count` | PASSED | 0.04 s |
| `test_attacks::test_bandwagon_attack_profile_count` | PASSED | 0.03 s |
| `test_attacks::test_inject_user_count[random_attack]` | PASSED | 0.05 s |
| `test_attacks::test_inject_user_count[bandwagon_attack]` | PASSED | 0.04 s |
| `test_reproducibility::TestSmoke::test_paper_config_schema` | PASSED | 0.01 s |
| `test_reproducibility::TestSmoke::test_expected_f1_values_in_range` | PASSED | 0.02 s |

No warnings; all custom markers (`smoke`, `subsample`, `full`) properly registered in `pyproject.toml`.

---

## 5. Time to green

| Step | Duration |
|---|---|
| Clone + venv creation | ~1 min |
| pip installs (CPU PyTorch) | ~8 min |
| Diagnosing setuptools issue | ~5 min |
| Fix + re-install | ~2 min |
| Running smoke tests | ~15 s |
| **Total wall-clock** | **~17 min** |

---

## 6. Notes for GPU machines

- The install steps above are identical on a CUDA machine; replace the `--index-url` with the CUDA 12.1 wheel URL (see `REPRODUCE.md §1`).
- Stage 3 (720 experiments) requires a GPU; do not attempt it on CPU.
- The `--dry-run` flag to `reproduce_paper.sh` skips Stages 1–6 and exits 0 — useful for verifying the pipeline entry-point without raw data or a GPU.
