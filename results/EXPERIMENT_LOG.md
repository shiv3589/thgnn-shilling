
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
