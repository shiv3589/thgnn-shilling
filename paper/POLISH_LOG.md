# Polish Log — TH-GNN Paper

## Summary of changes

---

## 1. Notation consistency
**No issues found.** `F\textsubscript{1}` is used uniformly across all files.
All mathematical symbols ($D$, $H$, $d_h$, $d_T$, $d_p$, $L$, $L_u$, $\tau$, $\phi$) are
defined once in method.tex and used consistently. The $L$ vs $L_u$ distinction
(layers vs. review count) is unambiguous.

## 2. Cross-reference completeness
**No broken refs.** All `\ref{tab:...}` and `\ref{fig:...}` targets exist.
Section and equation labels (`sec:*`, `eq:*`) that appear unused within the
section files are intentional—they are resolved by the main `.tex` wrapper and
may be cited from sections not yet present (e.g. conclusion). No changes made.

## 3. Tense consistency — FIXED

### experiments.tex (results section — past tense)
| Line | Before | After |
|------|--------|-------|
| 152 | `achieves F1 between` | `achieved F1 between` |
| 154 | `configurations—uniformly` | `configurations, uniformly` (em-dash → comma) |
| 155 | `FAR stays below` | `FAR stayed below` |
| 156 | `baselines incur FAR` | `baselines incurred FAR` |
| 159 | `streams fall back` / `achieves 0.867` | `fell back` / `achieved 0.867` |
| 162 | `gap is starkest` | `gap was starkest` |
| 163 | `defeat detectors that rely` | `defeated detectors that relied` |
| 165 | `achieves a mean F1` | `achieved a mean F1` |
| 168 | `DR...exceeds` | `DR...exceeded` |
| 170–171 | `are detectable` / `is indistinguishable` | `were detectable` / `was indistinguishable` |
| 174 | `measures LightGCN` | `reports LightGCN` |
| 176 | `GraphAttack reduces` | `reduced` |
| 177 | `filtering restores` | `restored` |
| 178 | `detector...reaches` | `reached` |
| 179 | `TH-GNN ranks first` | `ranked first` |
| 186 | `baselines widens / closes` | `widened / closed` |
| 187 | `TH-GNN achieves` | `achieved` |
| 188 | `margin is` | `margin was` |
| 190–192 | `occurs` / `become` / `is too small` | `occurred` / `became` / `was too small` |
| 204–205 | `causes` / `falls` / `doubles` | `caused` / `fell` / `doubled` |
| 208 | `GCN costs` | `cost` |
| 212–214 | `yields` / `is concentrated` / `provides` / `understates` | `yielded` / `was concentrated` / `provided` / `understated` |

### related_work.tex (cited-work verbs — past tense)
| Lines | Before | After |
|-------|--------|-------|
| 14–15 | `treats` / `flags` | `treated` / `flagged` |
| 16–17 | `define` / `capture` | `defined` / `captured` |
| 24–25 | `extend` / `are suspiciously` | `extended` / `were suspiciously` |
| 28–29 | `are fast` / `require` / `fail` | `were fast` / `required` / `failed` |
| 33–34 | `achieve` / `overfit` / `degrade` | `achieved` / `overfitted` / `degraded` |
| 37–38 | `replace` / `discard` | `replaced` / `discarded` |
| 44–47 | `exploit` / `build` / `score` / `outperform` | `exploited` / `built` / `scored` / `outperformed` |
| 109 | `HGT...addresses` | `addressed` |
| 113–114 | `achieves` / `has since been adapted` | `achieved` / `was subsequently adapted` |
| 121–122 | `maintain` | `maintained` |
| 125 | `TGN demonstrates` | `demonstrated` |
| 129 | `approaches show` | `showed` |
| 141–143 | `is the first work` / `is prompted` | `was the first work` / `was prompted` |
| 146–148 | `are grammatically fluent` / `rely on` | `were grammatically fluent` / `relied on` |

## 4. Redundancy — FIXED
**experiments.tex ablation section:** Lines 198–200 contained the conclusion
*before* the data, and line 200 was an exact restatement of line 198–199.
Fixed by:
- Moving the `Table~\ref{tab:ablation}` callout to be the first sentence
- Removing the redundant line 200 entirely
- Moving the conclusion sentence to the end of the subsection

## 5. Table/figure call-outs
**No missing call-outs.** All five floats (`tab:datasets`, `tab:main_detection`,
`tab:downstream`, `tab:ablation`, `fig:budget_curves`, `fig:llm_vs_gan`,
`fig:architecture`) are explicitly mentioned in text before they appear.

## 6. Transition sentences — ADDED
| File | Location | Sentence added |
|------|----------|----------------|
| experiments.tex | End of §4.1 Datasets | "Together, these four datasets provide complementary test conditions spanning rating-only and text-rich settings, sparse and dense interaction graphs, and short and long temporal windows." |
| experiments.tex | End of §4.2 Baselines | "We evaluate all three baselines against the five attack families described next." |
| related_work.tex | End of §2.1 Shilling Detection | "LLM-based generation has compounded this challenge by eliminating the textual fingerprints that historically distinguished synthetic from genuine profiles." |
| related_work.tex | End of §2.2 LLM Detection | "The structural and temporal graph architectures that can provide these complementary signals are surveyed next." |
| method.tex | End of §3.2 Graph Construction | "These node features are the input to the HGT encoder described next." |
| method.tex | End of §3.4 Multimodal Fusion | "We train the full model end-to-end using the focal loss objective described next." |

## 7. Factual corrections (caught during tense pass)
| File | Line | Before | After | Source |
|------|------|--------|-------|--------|
| experiments.tex | 165 | `F1 of 0.824` | `F1 of 0.825` | Consistent with introduction.tex (corrected in prior session); computed mean = 0.8245 |
| experiments.tex | 166 | `+10.8 pp` | `+10.9 pp` | Consistent with introduction.tex correction |
| experiments.tex | 167 | `+14.9 pp` | `+15.0 pp` | 0.825 − 0.675 = 0.150 = 15.0 pp |
| experiments.tex | 168 | `0.830--0.843` | `0.825--0.843` | Actual DR range from main_detection.csv: min=0.825 (Amazon-Clothing) |

## 8. Page budget estimate
Approximate word counts (LaTeX-stripped):
- introduction.tex: ~638 words
- related_work.tex: ~1,200 words
- method.tex: ~1,506 words + ~22 equations
- experiments.tex: ~1,465 words + 3 floats (inline dataset table) + 3 figure refs

Estimated page count in ACM two-column 10pt:
| Component | Est. pages |
|-----------|------------|
| Abstract | 0.2 |
| Introduction | 0.9 |
| Related Work | 1.3 |
| Methodology (text + equations) | 2.5 |
| Experiments text + dataset table | 1.8 |
| main_detection table (full-width) | 0.8 |
| downstream table (full-width) | 0.8 |
| ablation table (half-width) | 0.3 |
| Figures (architecture, budget, llm_vs_gan) | 1.5 |
| **Total** | **~10.1 pages** |

**Status: at budget limit.** The paper is estimated at ~10 pages, within the
8–10 page target but leaving no margin. No cuts required, but any additions
(e.g. limitation paragraph, broader impact) would push it over. The main
detection and downstream tables are the largest contributors; compressing
either to a single-column format would recover ~0.5 pages each.
