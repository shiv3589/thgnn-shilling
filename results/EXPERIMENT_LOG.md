> **Raw-results-in-git policy, effective 2026-07-09 (commit range starting
> at the fabrication-audit entry below):** all real experimental run output
> (`results/raw/*.json`) MUST be committed to git going forward — not just
> uploaded to HuggingFace. HF is for large artifacts (checkpoints); git is
> for the small per-config JSON result files, because git history is what
> makes provenance provable after the fact, which is the exact capability
> that was missing when a fabricated table reached a compiled PDF (see the
> fabrication-audit entry below). `scripts/check_table_provenance.py`
> enforces this mechanically for every table `paper/sections/experiments.tex`
> actually compiles. **Any compiled table without a traceable, committed,
> `status=="ok"` raw-result commit predating it should be treated as
> unverified** — the three currently-compiled tables
> (`main_detection.tex`, `ablation.tex`, `downstream_ndcg.tex`) are in
> exactly that state right now and carry a `% PROVENANCE UNVERIFIED`
> comment as of this commit.

---

## Phase 0 + Phase 1 — Manuscript Consistency Fixes

**Date:** 2026-07-08
**Git commit (after push):** 4c0a327489d729678ca842c1db6fa7291fb4bc38
**Environment:** Windows (local), no GPU — RunPod required for experiment reruns

### Phase 0 Findings

| Item | Finding |
|---|---|
| Training entrypoint | scripts/train.py (standalone); scripts/run_experiments.py → src/detectors/thgnn_wrapper.py (matrix runs) |
| Attack generators | src/attacks/classic.py (Random/Bandwagon), ush.py, gent4sr.py |
| Baselines | src/detectors/statistical.py, src/detectors/anti_fakeu.py (both → Isolation Forest) |
| Evaluation | src/evaluation/metrics.py |
| LaTeX source | paper/ |
| Reproduction check | SKIPPED — no GPU/Python environment on current machine |
| **Attention variant (critical)** | paper_runs.yaml sets use_aaa: true (AAA / Eq. 10). BUT 	hgnn_wrapper.py (the experiment runner's model builder) instantiates HGTEncoder WITHOUT use_aaa=True, defaulting to base HGT + temporal bias (Eq. 5). **Discrepancy unresolved — must be fixed before camera-ready.** |
| Per-user scores saved? | **NO.** un_experiments.py saves only F1/DR/FAR/threshold/n_flagged to JSON. Phase 2 requires reruns. |
| tc_loss active in experiments? | **NO** (for the 33-config matrix via 	hgnn_wrapper.py). YES in scripts/train.py standalone. |

### Phase 1 Edits Applied (LaTeX only, no experiments run)

| Edit | File(s) | Change |
|---|---|---|
| 1 — Ablation contradiction | sections/related_work.tex | Deleted "as the ablation study confirms, degrades gracefully…" clause; replaced with neutral design-intent statement |
| 2a — Overstated "first" claim | sections/introduction.tex | Added "to our knowledge," qualifier |
| 2b — "strictly more informative" | sections/related_work.tex | Removed; replaced by complementary-signal framing |
| 2c — Contribution 4 reframe | sections/introduction.tex | Removed "unsatisfactory" framing; replaced with artifact-attribution framing |
| 3 — Dead dataset text | sections/method.tex | Added TODO PENDING-PHASE-8 comment + forward reference to Sec 4.1 noting ML-1M-only evaluation |
| 4 — Theory/arch contradiction | sections/method.tex, sections/appendix.tex, main.tex | Fixed proof's "depends only on node features" by redefining baseline as temporal-bias-free bilinear form; added use_aaa discrepancy TODO comment; moved Proposition + proof to Appendix A as "intuition" |
| 5 — Abstract Agent4SR framing | bstract.tex | Reframed: feature detectors succeed via incidental artifacts; graph detector has precision collapse; open frontier defined |
| 6 — Anti-FakeU citation | eferences.bib, sections/related_work.tex, sections/experiments.tex | Corrected: was zhang2022antifakeu (year 2022, wrong authors, garbled pages). Now you2023antifakeu (You et al., WWW 2023, pages 938–948, DOI 10.1145/3543507.3583289) |
| 7 — tc_loss orphan | sections/experiments.tex | Deleted Limitations bullet (tc_loss not active in the experiment-matrix training path) |

### Compilation Result

- **Errors:** 0  
- **Undefined citations:** 0  
- **Undefined labels:** 0  
- **Output:** 9 pages, ~744 KB  
- **Remaining bib VERIFY entries:** Several (urke2005segment, deldjoo2024survey, guo2023close, ossi2020temporal, shi2023large, williams2006detection, wu2021fight, wu2022graphattack, zhang2024agent4sr) — flagged for manual verification before submission but contain no garbled field values.

### Outstanding items before next phase

- Resolve use_aaa discrepancy in 	hgnn_wrapper.py vs paper_runs.yaml
- Reproduction check (Phase 0 item 2) on RunPod — run Random@1%/seed=42 and verify F1/DR/FAR match Table 1
- Phase 2 requires reruns (no cached per-user scores)
---

## run_session1.py — dry_run — 2026-07-09T13:52:56Z

| timestamp | stage | dataset | attack | rate | detector | variant | seed | status | f1 | dr | far | git_commit | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-09T13:52:58Z | dry_run | synthetic_dry_run | random | 0.01 | thgnn | base | 42 | OK | 0.0678 | 1.0 | 0.3667 | d087ba6 |  |
| 2026-07-09T13:52:59Z | dry_run | synthetic_dry_run | random | 0.01 | statistical_baseline | na | 42 | OK | 1.0 | 1.0 | 0.0 | d087ba6 |  |

---

## use_aaa wiring fix — all THGNNWrapper entrypoints — 2026-07-09

The use_aaa provenance bug (paper_runs.yaml sets thgnn.use_aaa: true but the
value was never forwarded to HGTEncoder, so every run silently used base HGT
attention regardless) has been fixed at its root (THGNNWrapper itself,
commit 94a5bb9) and at every caller that builds a THGNNWrapper:

| Entrypoint | Commit | Status before fix | Paper draft impact |
|---|---|---|---|
| src/detectors/thgnn_wrapper.py (root cause) | 94a5bb9 | use_aaa param did not exist | N/A — library code |
| scripts/run_experiments.py | 54676ed | use_aaa dropped in _build_detector | LIVE — canonical entrypoint via reproduce_paper.sh; produces results/raw/*.json consumed by aggregate_results.py / paper tables |
| scripts/run_ablation.py | 1569877 | use_aaa dropped in _build_detector ("thgnn" variant) | LIVE — also invoked by reproduce_paper.sh. The "Full TH-GNN \| ML-1M \| 0.292±0.419" row already compiled into paper/tables/ablation.tex (input by paper/sections/experiments.tex) came from this exact code path, necessarily run with use_aaa=False pre-fix regardless of what paper_runs.yaml specified. NEEDS A RE-RUN + provenance check before this table can be trusted as "AAA" or "base" — same unconfirmed-variant ambiguity Session 1 exists to resolve for Table 1. |
| scripts/run_generalisation.py | eca7aa1 | use_aaa dropped in _build_detector | Preventative — paper/tables/generalisation.tex reads "Results pending full experimental run"; no real numbers affected |
| scripts/run_robustness.py | 8d27485 | use_aaa dropped in _build_detector | Preventative — scripts/gen_robustness_placeholder.py confirms run_robustness.py "has not been run"; no real numbers affected |

Each fix was independently verified: a live smoke check via that script's
own `_build_detector()` confirmed lambda_t/s/g parameters are present
(requires_grad=True) only when use_aaa=True, and that the two variants'
forward passes differ (mean abs diff 0.0283 on an identical synthetic
graph/seed across all four scripts). Each script's own `--dry-run` /
equivalent entry point was also run directly and confirmed to log
"thgnn.use_aaa (from paper_runs.yaml): True" before failing cleanly on
missing local data (expected on this machine).

### Also found, NOT fixed here (different pattern, reported for visibility)

Three more scripts build HGTEncoder *directly* (bypassing THGNNWrapper
entirely), so the fix above does not reach them:

- **scripts/run_transfer.py** — same bug class (use_aaa never passed to
  HGTEncoder), but paper/tables/transfer.tex also reads "Results pending" —
  no real numbers affected. Not yet fixed.
- **scripts/run_scalability.py** — same bug class, but its output
  (results/tables/scalability.csv) is only referenced in a commented-out
  line in paper/sections/experiments.tex — not compiled into the paper.
  Not yet fixed.
- **scripts/run_attention_analysis.py** — architecturally different: its
  CapturingHGTEncoder subclasses the base HGTLayer specifically (an
  attention-weight-capture design that does not support the AAA path at
  all), not a simple missing-argument bug. Needs a design decision (add an
  AAA-capturing counterpart, or explicitly document this tool as
  base-HGT-only-by-design) rather than a one-line wiring fix.

Of the four canonical-pipeline entrypoints named in scripts/reproduce_paper.sh,
only run_experiments.py and run_ablation.py are invoked there (both now
fixed); run_transfer.py, run_scalability.py, and run_attention_analysis.py
are standalone scripts not part of that pipeline.

---

## Fabrication audit — 2026-07-09

Independent git-archaeology audit, triggered by discovering `paper/gen_main_detection_table.py`
unconditionally fabricates data (hardcoded `EXPECTED_F1` + seeded RNG, no
real-data code path at all).

**Finding: fabricated numbers reached a compiled, committed PDF.**
`paper/main.pdf` was committed at `d34ac1e` (2026-05-25) and recompiled/
recommitted at `16a1012` (2026-07-01, commit message: "paper/main.pdf:
recompiled with updated author block and table sizing", binary confirmed
747904→750997 bytes) — still fully recoverable from git history right now
(`git show 16a1012:paper/main.pdf`). At `16a1012` the compiled document
contained: a fully fabricated Table 1 (`main_detection.tex`, matching
`gen_main_detection_table.py`'s hardcoded `EXPECTED_F1` exactly), a fully
fabricated Table 2 (`downstream_ndcg.tex`, matching `gen_downstream_table.py`'s
`CLEAN`/`RECOVERY` formula exactly — verified by hand-computing
`0.209 + 0.912×(0.224−0.209) = 0.2227` against the committed `0.223±0.005`
cell), a fully fabricated Table 4 (`ablation.tex`), and fabricated prose
in two whole subsections (Generalisation, Robustness) citing specific
F1/ΔF1 percentages for experiments that never ran — `run_generalisation.py`
and `run_robustness.py` did not exist yet or had never been run at that
point; `scripts/gen_robustness_placeholder.py`'s docstring still confirms
`run_robustness.py` "has not been run" today.

This was caught and the `.tex` sources corrected 6 days later
(`bf6fe40`/`958bb14`/`9ed8887`, 2026-07-07), but the PDF was untracked
4 days before that (`b8c95ab`, 2026-07-03) — no clean PDF was ever
recompiled and committed after the fix.

**Root cause, not just this instance:** `gen_main_detection_table.py`,
`gen_ablation_table.py`, and `gen_downstream_table.py` had zero code path
that reads real data — fabrication was the *only* thing they could ever
produce, silently, with no flag and no warning beyond a docstring comment.
Fixed today: all three now raise `SystemExit` with a clear message unless
`results/raw/` contains at least one committed, `status=="ok"` real result
file, or `--placeholder-explicit-i-know-this-is-fake` is passed explicitly
(which visibly labels the output as fake in both a LaTeX comment and the
rendered caption). See `scripts/check_table_provenance.py` and the
raw-results-in-git policy note at the top of this file.

**What this audit could NOT determine:** whether the numbers *currently*
sitting in `main_detection.tex`/`ablation.tex`/`downstream_ndcg.tex` (as
opposed to the historically-fabricated `16a1012` version) are real
transcribed results or hand-fabricated — `results/raw/` has never had a
single file committed to git at any point in this repository's history, so
there is nothing to mechanically check the current numbers against. This
question was investigated and found unprovable from git history; it is not
being pursued further. The three files carry a `% PROVENANCE UNVERIFIED`
comment as of this commit and will be regenerated from committed
`results/raw/` output once Session 1 (and the ablation rerun) produce it.
