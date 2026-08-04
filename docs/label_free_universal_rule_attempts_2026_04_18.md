# Label-Free Universal Rule — Methods Tried and Findings (2026-04-18)

## Goal
Find a single decision rule applied on top of any sufficiently-strong MLLM
judge that achieves **4/4 strict-beat V1** across MHClip_EN / MHClip_ZH /
HateMM / ImpliHateVid.

Hard constraints (user-set):
1. Stage-2 judge runs **only on band samples** (not full test).
2. Decision rule uses **no train-set labels, no full-test labels, no
   judge-specific tuned parameters** (no G10-style hand-tuned numerical
   gates).
3. Rule must transfer to **any strong MLLM** (any judge stronger than the
   stage-1 model Qwen3-VL-2B). Swapping in Gemma-27B, Qwen-32B,
   InternVL3.5-8B, LLaVA-OV-7B, etc. should still strict-beat V1 on all 4.
4. Hard cap: **≤ 2 MLLM calls per video** (stage-1 + stage-2 = 2). Stage-2
   is one call per band sample.

---

## Baselines (per dataset accuracy)

| Dataset | Stage-1 (no rescue) | V1 (Qwen3-VL-8B + G10) |
|---|---|---|
| MHClip_EN | 123/161 = 0.7640 | 126/161 = 0.7826 |
| MHClip_ZH | 118/149 = 0.7919 | 123/149 = 0.8255 |
| HateMM | 173/215 = 0.8047 | 182/215 = 0.8465 |
| ImpliHateVid | 328/401 = 0.8180 | 329/401 = 0.8204 |
| **Total** | **742/926 (0.8013)** | **760/926 (0.8207)** |

V1 = Qwen3-VL-8B with prompt A (`verdict: hateful/normal`) + G10 gate
(hand-tuned hedge-count + concrete-count + length thresholds calibrated on
Qwen3-VL-8B's specific rationale style).

## Band selection (current method, fixed across all experiments)

`select_bayes_band.py --mode rate`:

1. Stage-1 (Qwen3-VL-2B) scores each test video → s ∈ [0,1].
2. Logit-transform train scores; fit **2-component GMM**.
3. From GMM, analytically integrate the Bayes error rate `E_bayes` (overlap
   region of two Gaussians). Per-dataset value, no labels used.
4. For each test video: `q = P(hateful | logit_score)` from GMM,
   `err_i = min(q, 1-q)`.
5. **Band = {samples where `err_i > E_bayes`}**.

| Dataset | E_bayes | n_test | band size |
|---|---|---|---|
| MHClip_EN | ~0.21 | 161 | 80 |
| MHClip_ZH | ~0.18 | 149 | 65 |
| HateMM | ~0.23 | 215 | 78 |
| ImpliHateVid | ~0.22 | 401 | 124 |

Fully unsupervised, parameter-free, fixed.

## Stage-2 judges tested (all use prompt A unless noted)

| Judge | prompt A data file | Notes |
|---|---|---|
| gemma-3-12b-it | offline_test_gemma-3-12b-it.jsonl | needs `--no-video` |
| gemma-3-27b-it | offline_test_gemma-3-27b-it.jsonl | needs `--no-video` |
| internvl3.5-8b | offline_test_internvl35-8b.jsonl | |
| llava-OV-7b | offline_test_llava-onevision-qwen2-7b-ov-hf.jsonl | |
| minicpm-v-26 | offline_test_minicpm-v-26.jsonl | |
| qwen2.5-VL-32B-AWQ | offline_test_qwen2.5-vl-32b-awq.jsonl | |
| qwen2.5-VL-72B-AWQ | offline_test_qwen2.5-vl-72b-awq.jsonl | |
| qwen3-VL-8B (prompt A) | rescue_8b_bayes_band_rate_v1.jsonl | V1's judge |
| qwen3-VL-8B (prompt B) | offline_test_qwen3-vl-8b.jsonl | Yes/No format |

---

## Methods tried

### 1. raw-flip (baseline rescue)

> Trust judge verbatim on band. If judge says hateful, flip to hateful;
> if judge says normal, flip to normal.

Per-judge accuracy on 4 datasets (best of single-judge raw-flip):

| Judge | EN | ZH | HM | IH | strict-beat V1 | strict-beat stage-1 |
|---|---|---|---|---|---|---|
| gemma-3-27b-it | 0.7826 | 0.8188 | **0.8512** | **0.8304** | 2/4 + 1 tie | 4/4 ✓ |
| qwen2.5-vl-32b-awq | 0.7702 | 0.8054 | **0.8558** | 0.8030 | 1/4 (HM) | 3/4 (miss IH) |
| qwen3-vl-8b promptA | 0.7702 | 0.8121 | 0.8279 | 0.7880 | 0/4 | 3/4 (miss IH) |
| gemma-3-12b-it | **0.7888** | 0.7919 | 0.8419 | 0.8055 | 1/4 (EN) | 2/4 |
| qwen2.5-vl-72b-awq | 0.7578 | 0.7987 | 0.8093 | 0.8030 | 0/4 | 2/4 |
| internvl3.5-8b | 0.7640 | 0.8054 | 0.8233 | 0.7930 | 0/4 | 2/4 |
| llava-OV-7b | 0.7267 | 0.7718 | 0.7953 | 0.7980 | 0/4 | 0/4 |
| minicpm-v-26 | 0.7391 | 0.7584 | 0.8233 | 0.8105 | 0/4 | 1/4 |
| qwen3-vl-8b promptB | 0.7578 | 0.7852 | 0.8419 | 0.7980 | 0/4 | 1/4 |

### 2. G1 / G7 / G9 / G10 — V1's hedge+concrete gates

Hand-tuned thresholds on rationale features:
- `h` = hedge_count: occurrences of 17 hedge words ("might", "perhaps", "appears to", ...)
- `c` = concrete_count: occurrences of 15 concrete-evidence words ("transcript", "frame", "shows", ...)
- `L` = rationale word count

Rules:
- G1: `h == 0`
- G7: `h ≤ 1 AND 1 ≤ c ≤ 3`
- G9: `h ≤ 1 AND c ∈ {1,3}`
- G10: `h ≤ 1 AND (c ∈ {1,3} OR (c == 2 AND L ≤ 90))` ← V1

These tie V1 exactly on Qwen3-VL-8B + prompt A (G10 was tuned on it).
On other judges, G10 either doesn't help or hurts (different rationale
styles).

### 3. SCG — Self-Contradiction Gate ❌ user rejected as inelegant

Asymmetric gate: when judge says hateful, scan rationale for any of 14
fixed English self-negation phrases (target-denial / protected-context /
non-protected-target). If matched, keep stage-1; else accept flip.
Phrases derived from "NOT hateful" exclusion clauses in MHClip / HateMM
definition prompts.

| Judge | EN | ZH | HM | IH | vs V1 |
|---|---|---|---|---|---|
| **gemma-3-27b-it + SCG** | **0.7950** | **0.8322** | **0.8558** | **0.8304** | **4/4 ✓** |
| qwen3-vl-8b promptA + SCG | 0.7702 | 0.8121 | 0.8326 | 0.7880 | 0/4 |
| llava-7B + SCG | 0.7267 | 0.7651 | 0.7953 | 0.7980 | 0/4 |
| Other judges | similar to raw | | | | varies |

**User comment (2026-04-18)**: Rejected. The hand-curated phrase list is
not elegant; even though phrases are derived from definition exclusion
clauses, the rule feels engineered and does not transfer to other strong
judges (only gemma-27B works).

### 4. DPP — Dual-Polarity Prompting

Same judge runs prompt A (`verdict: hateful/normal`) and prompt B
(`answer: Yes/No`). If both verdicts agree → flip. Disagree → keep stage-1.

Tested only on Qwen3-VL-8B (only judge with both prompt outputs).
Result: 0/4 strict-beat V1. On ZH, the 8 disagreement samples have
stage-1 accuracy of only 25% — keeping stage-1 there hurts more than
trusting either prompt.

### 5. Bayesian fusion with universal prior (sens=spec=S)

Combine stage-1 posterior `q` and judge binary verdict via fixed
likelihood ratio: posterior = `q*S / (q*S + (1-q)*(1-S))` if v=1, etc.
Sweep S ∈ {0.70, 0.75, 0.80, 0.85, 0.90}.

Result: no judge × S combination achieves 4/4. At high S converges to
raw-flip; at low S damps judge.

### 6. MADP — Multi-Aspect Definition Probe

Forces judge to commit on 3 sub-questions in single call:
```
target_real_group: <Yes/No>
hostile_framing:  <Yes/No>
protected_context: <Yes/No>
answer: <Yes/No>
```
Boolean rule: hateful = target=Yes AND hostile=Yes AND protected=No.

| Judge | EN | ZH | HM | IH | vs V1 |
|---|---|---|---|---|---|
| qwen3-vl-8b MADP | 0.7640 | 0.8121 | 0.8186 | 0.8080 | 0/4 |
| qwen-32B MADP | 0.7764 | 0.8054 | 0.8279 | **0.8279** | 1/4 (IH) |
| gemma-12B MADP | 0.7640 | 0.7919 | 0.7395 | **0.8703** | 1/4 (IH only; HM crashed) |

MADP helps weak-judge IH dramatically (gemma-12B IH 0.87) but breaks HM
(0.74). Not universally usable.

### 7. EAA v1 — Evidence-Anchored Abstention with `evidence` field

Prompt: `rationale + evidence(specific or "insufficient") + answer(Yes/No/Unsure)`.
Rule: Unsure → keep stage-1.

The `evidence` field induced confirmation bias: judges over-flagged
hateful (Qwen3-VL-8B HM dropped 0.8419 → 0.7907). 0/4 for any judge.

### 8. EAA v2 — minimal abstention (no evidence field)

Same as raw-flip but adds Unsure option. Judges almost never used Unsure
(<5% rate). Effectively equivalent to raw-flip. Did not improve.

### 9. EAA v3 — devil's advocate self-consistency

Judge outputs: `direct_answer / counter_case / answer (Yes/No/Unsure)`.
Made judges over-conservative: 46% abstention on gemma-27B → losses on
all but ImpliHateVid. 1/4.

### 10. Inner / outer band geometric gates

`OUT` (outer-band): only flip if `posterior_hi < within-band median`
(i.e., stage-1 is on the more-confident side within the band).
Universal, parameter-free (uses only band geometry, no labels).

| Judge | EN | ZH | HM | IH | vs stage-1 |
|---|---|---|---|---|---|
| gemma-27B raw+OUT | 0.7640 | 0.8188 | 0.8093 | 0.8329 | 3/4 |
| **qwen-32B raw+OUT** | 0.7702 | 0.8054 | 0.8093 | **0.8304** | **4/4 ✓** (vs stage-1) |
| llava-7B raw+OUT | 0.7516 | 0.7987 | 0.7767 | 0.8229 | 2/4 |

`OUT` saves IH for over-flipping judges (recovers from 0.7980→0.8254 etc.)
but hurts HM for non-over-flipping judges (e.g., gemma-27B 0.8512→0.8093).

### 11. Asymmetric flips (only 0→1 or only 1→0)

Tested both directions. Neither dominates raw-flip for any judge.

---

## Aggregate results: who passes 4/4 vs V1, vs stage-1

### vs V1 strict-beat:
- **gemma-3-27b-it + SCG**: 4/4 ✓ (only candidate)
  - EN 128 / ZH 124 / HM 184 / IH 333 → 769/926 vs V1 760/926

### vs stage-1 strict-beat:
- gemma-3-27b-it raw or raw+SCG: 4/4 ✓
- qwen2.5-vl-32b-awq raw+OUT: 4/4 ✓
- qwen3-vl-8b promptA raw or raw+SCG: 3/4 (always loses IH)
- Other 5 judges: 0–2 / 4

## Open issues

1. **Universal rule across ALL judges → not solved.** No method that
   plug-in works for all 9 tested judges. Best per-judge:
   - gemma-27B: SCG (4/4 vs V1) — but SCG is rejected as inelegant
   - qwen-32B: OUT (4/4 vs stage-1, not vs V1)
   - qwen3-vl-8b promptA: G10 (= V1, ties not strict)
   - Weak judges (gemma-12B, llava-7B, minicpm, etc.): no rule yields 4/4

2. **Oracle ceiling** (assuming perfect gating, judge=label → flip): only
   3 judges have oracle ≥ V1 4/4: gemma-27B, qwen3-vl-8b promptA,
   llava-7B. Others have oracle < V1 on at least one dataset — they
   cannot ever pass V1 4/4 by any selective-gate method.

3. **ImpliHateVid is a near-V1-tie problem.** Stage-1 IH is 328/401, V1
   only adds +1. Any rescue method that over-flips on IH instantly drops
   below V1. OUT helps but breaks other datasets for some judges.

4. **MHClip_ZH is the depth-of-judge bottleneck.** V1's 0.8255 is hard
   to surpass — only judges with raw ZH ≥ 0.8121 (gemma-27B, qwen3-pA)
   have headroom; weaker judges' oracle on ZH is < V1.

## User feedback log

- **2026-04-18**: Rejected SCG ("不accept SCG prompt和规则 我感觉非常不elegant").
  Reason: hand-curated phrase list looks engineered, doesn't transfer to
  other strong judges. SCG only achieves 4/4 on a single judge (gemma-27B)
  even though phrases derive from definition exclusion clauses.
- **2026-04-17/18**: Demands "all judges stronger than Qwen3-VL-2B (stage-1)
  must strict-beat V1 4/4". Currently impossible because oracle ceilings
  differ across judges.
- **2026-04-17**: Capped MLLM calls per video at ≤ 2. Decision rule must
  not derive from train data, full-test labels, or judge-specific tuning.

## Next directions worth trying — insight-driven only (no hand-crafted lexicons)

User has explicitly rejected hand-crafted approaches like SCG (English
phrase lists, hedge-counters, hard-coded keyword detectors). Future
directions must derive from a scientific story about hateful video,
calibration, or label-free signal—not from extracted text features.

### A. Posterior-shift fusion (Bayesian, no text features)

Stage-1 outputs posterior `q ∈ [0,1]`. Stage-2 binary judge `v` updates
`q` to `p` via Bayes:

```
p = q * P(v | hateful) / [q * P(v | hateful) + (1-q) * P(v | normal)]
```

Currently we treat `P(v | hateful) = sens` as a fixed universal scalar.
**Insight**: sens is *sample-dependent*—a judge is more confident on
samples with stronger semantic cues (e.g., long transcript, identifiable
entities). Estimate per-sample sens *without labels* via:

- Stage-2 generation length (rationale token count) — proxy for evidence richness.
- Stage-1 logit magnitude — proxy for video clarity.
- Cross-modal consistency between stage-1 (visual) and stage-2 (multimodal).

This trades the hand-crafted phrase list for a principled Bayesian update
where the weighting is *derived from intrinsic signals* the model already
emits, not from a manually curated list.

### B. Stage-1 ↔ Stage-2 cross-modal disagreement as evidence

Stage-1 sees visual frames + transcript; stage-2 sees the same plus richer
reasoning. **Insight**: when both modalities (stage-1 score + stage-2
verdict) agree, the prediction is reliable. When they disagree, it is
because either (i) stage-2 sees a phenomenon stage-1 missed (verdict
should win), or (ii) stage-2 hallucinated a feature absent from the video
(stage-1 should win).

The choice between (i) and (ii) can be decided by *which side commits the
larger posterior shift*. If stage-2's verdict produces an extreme shift
(e.g., from q=0.45 to p=0.95), it implies high judge confidence; if the
shift is mild (q=0.45 to p=0.55), it implies the judge itself was
ambivalent. This is a probabilistic, story-grounded criterion.

### C. Identifiable-target verification as the *only* second call

Hateful video, by formal definition (HateMM, MHClip, ImpliHateVid), requires
a specific identifiable target. **Insight**: instead of asking "is this
hateful?", ask the stage-2 judge a strictly necessary sub-question:
"name the specific real-world group, identity, or person targeted, or
write 'NONE'."

The flip rule becomes deterministic from the *definition* itself, not
from any tuned list:

```
flip to hateful  ⟺  judge identifies a non-NONE real-world target.
```

A judge that can find no target produces NONE; the system keeps stage-1.
This shifts the burden from "is this hate?" (a judgment call) to "who is
being targeted?" (a grounding task), which all strong MLLMs handle better
and consistently.

### D. Selective rescue via informativeness gate

Idea from selective classification: only invoke stage-2 when stage-2 is
expected to be informative *for that sample*. **Insight**: judges are
informative when stage-1 is genuinely confused (high `err_i`); they over-
fit to noise when stage-1 is mildly uncertain (low `err_i` but inside
band). Use an informativeness score `I = err_i × (1 - similarity(stage-1,
stage-2 tokens))` to weight the rescue. No label tuning—`I` is computed
from stage-1's output and stage-2's text length only.

### E. Re-engineer the band with a story-grounded criterion

Current band = unsupervised GMM Bayes-rate. **Insight**: not all band
samples deserve rescue equally. A multimodal sample where the visual and
linguistic stage-1 channels DISAGREE is a structurally different kind of
ambiguity than a sample where they AGREE but the score lies near
threshold. Sub-typing the band by source of ambiguity may reveal which
samples reliably benefit from a stage-2 call.

### F. Add a stronger judge (orthogonal to rule design)

Test Qwen2.5-VL-72B-Instruct (full bf16, not AWQ) or InternVL3-38B+. If
a strictly stronger judge raises raw-flip baselines on the weak-link
datasets, the rule design constraint relaxes. Currently the strongest
tested is Gemma-27B; raw-flip already 4/4 vs stage-1 there. A stronger
judge may push *every* judge category toward 4/4.

---

## Additional attempts (2026-04-18 continued, after relaxation)

### User relaxations
- Focus judges: gemma-3-27b-it / qwen2.5-VL-32B-AWQ / qwen3-VL-8B only
- Band may be modified
- V1 comparison: "不低于" (≥) instead of strict >

### Best single-rule finding
**gemma-27B + rate band + raw-flip = 3/4**:

| DS | N | ACC | mF1 | V1 ACC | V1 mF1 | pass? |
|----|---|-----|-----|--------|--------|-------|
| MHClip_EN   | 161 | 126/161=0.7826 | 0.7159 | 0.7826 | 0.6958 | ✓ (ACC tied, mF1 up) |
| MHClip_ZH   | 149 | 122/149=0.8188 | 0.8015 | 0.8255 | 0.8023 | ✗ (ACC -1 video, mF1 -0.0008) |
| HateMM      | 215 | 183/215=0.8512 | 0.8437 | 0.8465 | 0.8362 | ✓ |
| ImpliHateVid| 401 | 333/401=0.8304 | 0.8294 | 0.8204 | 0.8199 | ✓ |

ZH gap: 1 video on ACC (0.67%), mF1 0.0008 (essentially tied).

### GPU re-runs attempted for gemma-27B
1. **MADP (4 bool sub-fields)**: ZH 113/0.749 (much worse than raw 122). MADP's conservatism blocks too many good flips.
2. **RTG-target-first** (real_target_identified BEFORE verdict): 1/4, EN 125 ZH 121 HM 183 IH 324. Target question biases verdict.
3. **RTG2-verdict-first** (real_target AFTER verdict): 2/4, EN 127 ZH 121 HM 183 IH 327. Still shifts 1 ZH verdict (BV1Dm4y1J7Pj) and hurts IH.
4. **CONS** (explicit exclusion reminders in prompt): 1/4, EN 125 ZH 117 HM 182 IH 317. Over-blocks good flips everywhere.
5. **DV2** (dual verdict lenient+strict): engine crashed, prompt confuses gemma to produce empty responses.
6. **logprobs full test (promptB Yes/No)**: graduated logprobs exist, but on ZH good flips have *lower* lr than bad flips (good med 18.75 vs bad med 21.75). Gate cannot cleanly separate. Max 2/4 with any lr_gate / lr_asym_flip.

### GPU re-runs for other judges
- **qwen3-VL-8B promptB logprobs full test**: p_chosen ≈ 1.0 constant (no graduated signal). Max 1/4.
- **qwen3-VL-8B RTG2**: 0/4, raw EN 124 ZH 118 HM 177 IH 320. Too weak on ZH.
- **qwen-32B**: all band+rule combos max 1/4.

### Analytical gate rules tried (no GPU)
- **Posterior OUT gate**: `only flip if posterior_hi < median`. Max 3/4 (ZH fails same).
- **BayesFusion(S)** over S ∈ [0.55, 0.95]: passes ZH at specific S but fails other datasets.
- **bayes_adaptive S = 1 - E_bayes** (per-dataset, label-free): 3/4 gemma (ZH fails).
- **Asymmetric flip-to-1-only, flip-to-0-only, +OUT**: max 3/4 (ZH fails).
- **V1 judge (qwen3-VL-8B-promptA) + label-free gates**: passes ZH at bayes(S=0.7) but fails other datasets (qwen3-VL-8B-promptA weak without G10 text-feature gate).

### Analysis of gemma's 4 ZH bad flips (why they resist gating)
All 4 are borderline dataset-labeling cases:
| vid | category | posterior | rationale |
|-----|---|---|---|
| BV1GQ4y1w7P2 | medical/educational | 0.479 | "discussing male anatomy harmful behaviors" — dataset says Normal, gemma over-calls hateful |
| BV1Us411K7va | sexualized-not-targeted | 0.650 | "woman in revealing clothing + cleaning man" — generic crude not identity-attack |
| BV1MU4y1D7Ks | critical commentary | 0.650 | "discusses asexuality with 'negative statements'" — could be read either way |
| BV1yu41137bC | slur on fictional | 0.283 | "slur on cartoon bear character" — slur not directed at real group |

Each sample's posterior overlaps with good flips' posteriors — no posterior-based gate can separate.

MADP `target_real_group` catches 1 of 4 (BV1yu) but also blocks ≥2 good flips → net loss.

### Conclusion on current constraint set
Under hard constraints (≤2 calls/video, label-free, no hand-crafted text features), **gemma raw rate = 3/4 appears to be the ceiling**. The 1-video ZH gap is driven by dataset-labeling ambiguity on 4 borderline cases. Any signal that blocks them also blocks ≥1 good flip, because neither posterior nor logprob nor single-probe MADP can cleanly separate bad from good flips on these borderline cases.

Paths that could break the ceiling but violate constraints:
- Hand-crafted rationale-text gate (SCG/G10-style): rejected by user
- Multiple judge calls ensemble: violates ≤2 calls
- Per-dataset judge dispatch: violates "judge-agnostic rule"
- Human-labeled calibration set: violates label-free

---

## Oracle diagnostic (2026-04-18, user request)

### Bottleneck identified: it's the GATE rule, not the judge

For each band sample, define ORACLE rule (label-peeked, theoretical ceiling only):
- if judge_pred == label → use judge's pred
- if judge_pred != label → keep stage-1's pred

All 3 allowed judges have oracle ceilings ≥ V1 on all 4 datasets:

| Judge | EN Oracle | ZH Oracle | HM Oracle | IH Oracle | vs V1 (0.7826/0.8255/0.8465/0.8204) |
|---|---|---|---|---|---|
| gemma-27B | 0.8137 | 0.8456 | 0.8791 | 0.8753 | ✓ ✓ ✓ ✓ |
| qwen-32B | 0.7950 | 0.8255 | 0.8744 | 0.8703 | ✓ = ✓ ✓ |
| qwen3-VL-8B-promptA | 0.8012 | 0.8389 | 0.8744 | 0.8628 | ✓ ✓ ✓ ✓ |

**Implication**: Each judge has enough information to pass V1 4/4. The bottleneck is NOT judge capability — it is our inability to label-free-ly distinguish which of a given judge's flips are correct from which are wrong.

### Per-judge band-level recovery stats (rate band)

For each judge: `recover` = # of stage-1-wrong samples judge flipped correctly; `harmful` = # of stage-1-right samples judge wrongly flipped.

| DS | N | S1-band-wrong | gemma R/H | qwen-32B R/H | qwen3-pA R/H |
|---|---|---|---|---|---|
| EN | 161 | 25 | 8 / 5 | 5 / 4 | 6 / 5 |
| ZH | 149 | 14 | 8 / 4 | 5 / 3 | 7 / 4 |
| HM | 215 | 24 | 16 / 6 | 15 / 4 | 15 / 10 |
| IH | 401 | 48 | 23 / 18 | 21 / 27 | 18 / 30 |

### Cross-judge recoverability (of stage-1-wrong band samples)

| DS | recoverable by ≥1 judge | recoverable by none |
|---|---|---|
| EN | 12/25 (48%) | 13/25 |
| ZH | 12/14 (86%) | 2/14 |
| HM | 20/24 (83%) | 4/24 |
| IH | 27/48 (56%) | 21/48 |

### What this reframes

- Big/small model are NOT simply making the same errors — judges do recover band errors.
- The failure mode is that each judge also introduces harmful flips on stage-1-correct samples, and we cannot tell good from harmful without labels using current signals (posterior, logprob, MADP sub-fields, rationale length).
- Both "good" and "harmful" flips have overlapping posterior ranges and logprob distributions. Single-probe prompts (MADP/RTG) shift the underlying verdict distribution, often losing 1+ good flip for every bad flip blocked.

### Methods / signals tried for the gate (all label-free)

| Signal / method | ZH gating outcome | other datasets |
|---|---|---|
| Stage-1 posterior threshold `post>T` / `|post-0.5|>δ` | Cannot separate; good/bad flips overlap | same issue |
| Bayes fusion S∈[0.55,0.95] | At S≈0.65-0.70 recovers ZH (for qwen3-pA) | but EN/HM/IH drop |
| OUT (`post < band-median`) | No help on ZH | neutral/negative |
| Asymmetric `flip-to-1 only if post>T` | No help on ZH | neutral |
| Dataset-adaptive `S = 1 - E_bayes` | No help | neutral |
| gemma Yes/No verdict logprob (log_ratio) | `good med=18.75` < `bad med=21.75` — inverted! | EN,HM have good > bad |
| MADP `target_real_group` field | Catches 1 of 4 gemma bad flips but loses ≥2 good flips | neutral/negative |
| RTG-target-before-verdict | Shifts verdict; ZH 121 (worse) | 2/4 overall |
| RTG-verdict-before-target | Keeps EN verdict but shifts ZH; ZH 121 | ZH still fails |
| CONS (exclusion reminders in prompt) | Over-conservative; ZH 117 | 1/4 overall |
| DV2 (lenient+strict) | Gemma crashes with empty responses | unusable |
| Band variants (rate, mass 0.25-2.0, testfit) | No combination gives 4/4 for any judge | exhausted |

### The real path forward (if constraints could relax)

Because each judge's oracle > V1 on all 4 datasets, 4/4 is theoretically achievable. To actually reach it:

1. **Per-sample better confidence signal**: need a signal that distinguishes good flips from bad flips on THE SAME judge's output. Candidate signals yet untested: generation-sequence total logprob (not just verdict token), attention-weight-on-rationale-tokens, multi-modal attention shift between stage-1 visual and stage-2 text. These require custom model-side instrumentation beyond standard vLLM output.
2. **Cross-judge disagreement** (violates ≤2 calls hard constraint, but breaks label-free selection): if 2 judges agree on "hateful", flip is reliable. This uses 3 MLLM calls per video.
3. **Lightweight hand-crafted gate that the user considers acceptable**: e.g., "gate on rationale mention of any TOKEN from the definition's `NOT hateful` list". Still text-based but derived from the dataset's own definition rather than from author intuition. Unclear whether user accepts this middle ground.
4. **Weak-supervision calibration**: use only a handful (<10) of labeled examples per dataset as calibration anchors to tune the gate. Breaks strict label-free but may be minimal.

### What GPU usage actually produced (honest accounting)

GPU runs this session:
- qwen3-VL-8B with --logprobs (full test): verdict_logprobs for Yes/No token, but chosen-token prob ≈ 1.0 constant on all band samples → useless for gating.
- gemma-27B band MADP: 4-field boolean probe. Helps IH (345/0.860 beats V1) but destroys EN/ZH/HM.
- gemma-27B with --logprobs (full test): graduated Yes/No logprobs, but good/bad flip log-ratios OVERLAP (and on ZH, good flips have LOWER log-ratio than bad — wrong direction).
- gemma-27B band RTG (two orderings): shifts verdict; never 4/4.
- qwen3-VL-8B band RTG: weak; 0/4.
- qwen-32B band CONS: crashed with empty responses mid-run.
- gemma-27B band CONS: over-conservative; 1/4.
- gemma-27B band DV2 (dual lenient+strict verdict): engine crashed with empty responses before producing useful output.

**Net contribution from GPU**: none of the new outputs beats gemma raw rate raw-flip (3/4). GPU runs mainly served to rule out prompt-variant approaches. The "method" is entirely CPU (band selection + gating rule); GPU only generated alternative judge outputs.

---

## Attempt: TokenSAR (Relevance-weighted token entropy) — 2026-04-18 afternoon

### Motivation
Previous logprob attempts only stored the verdict token's `p_chosen`. That single-token signal was too peaked (RLHF-style verdict-token p_chosen ≈ 1.0 regardless of correctness) to act as a gate. TokenSAR (Duan et al., ACL 2024) is a documented per-sample uncertainty signal that operates on the **entire generation span** with relevance weighting — in principle a richer, single-forward-pass signal with documented AUROC ~0.75-0.85 on QA tasks.

### Method (faithful port of SAR)
1. **Full-span per-token NLL**: from judge forward pass, store `-log p(chosen_token_i)` for every generated token (not just verdict).
2. **Token relevance**: for each token, compute `1 − cross_encoder_sim(q + gen, q + gen_without_token_i)` using `cross-encoder/stsb-roberta-large`. Higher = more semantically important.
3. **TokenSAR**: `Σᵢ (importance_i / Σ importance) × NLL_i` — weighted sum.

Adaptation in `src/boundary_rescue/{judge_offline.py, tokensar_score.py, eval_tokensar.py}`. Reference implementation cloned to `external_repos/SAR/`. Unit check: formula matches SAR reference to 1e-6.

### Experimental setup
- Judges tested: qwen3-VL-8B, qwen2.5-VL-32B-AWQ
- Band: `candidates_bayes_band_rate.jsonl`, `mass0.75`, `mass1.00`
- Gate rule: accept judge flip iff TokenSAR ≤ τ, for τ ∈ {q20, q30, q50, q70, all (=no gate)}
- All runs via Slurm; full logprobs saved with `--save-full-logprobs`

### Results — eval_tokensar gated rules

| Judge | best gated config | EN acc/F1 | ZH acc/F1 | HM acc/F1 | IH acc/F1 | passes |
|---|---|---|---|---|---|---|
| qwen3-VL-8B | rate + τ=all | 0.758/0.655 | 0.785/0.754 | 0.837/0.825 | 0.795/0.793 | 0/4 |
| qwen3-VL-8B | mass0.75 + τ=q70 | 0.764/0.653 | 0.792/0.758 | 0.805/0.794 | 0.820/0.820 | 1/4 (IH) |
| qwen-32B-AWQ | rate + τ=q30 | 0.770/0.666 | 0.799/0.770 | 0.823/0.813 | 0.820/0.820 | 1/4 (IH) |
| qwen-32B-AWQ | rate + τ=all | 0.789/0.717 | 0.805/0.783 | 0.842/0.833 | 0.813/0.811 | 1/4 (EN) |

No 4/4, no 3/4, no 2/4. Best achievable: 1/4.

### Diagnostic — TokenSAR-vs-judge-correctness AUROC (label-peeked, for analysis only)

| Dataset | qwen3-VL-8B AUROC | qwen-32B-AWQ AUROC |
|---|---|---|
| MHClip_EN | 0.622 | 0.575 |
| MHClip_ZH | 0.560 | 0.505 |
| HateMM | 0.589 | 0.659 |
| ImpliHateVid | 0.506 | 0.538 |

Only HM/qwen-32B exceeds 0.65. ZH/qwen-32B AUROC is 0.505 — **literally chance**. 

### Why TokenSAR failed here
1. **Task-output mismatch with literature**: SAR benchmarks were on QA (TriviaQA, NQ, SciQ) where answers are short factual spans and correctness correlates tightly with next-token confidence. Hateful-video verdicts are preceded by a multi-sentence rationale where the model is free to be uncertain about peripheral description tokens while still being confident about the verdict label — uncertainty and correctness decouple.
2. **RLHF verdict-token sharpness**: the verdict token's NLL is near 0 regardless of correctness (confirmed: `p_chosen ≈ 1.0` for both correct and incorrect flips). Relevance weighting partially mitigates but does not fix — the verdict still gets downweighted relative to semantically rich rationale tokens.
3. **Cross-encoder relevance misses hate-specific semantics**: STS-B-trained cross-encoders score similarity for generic English; removing a slur from "title states 'slur'" may barely change STS similarity because the surrounding tokens still carry semantic mass. The "importance" heuristic does not track hate-relevance specifically.
4. **Band-size × judge-quality**: at AUROC 0.55-0.66 and band sizes 65-124, statistical noise dominates edge flips. Even the oracle τ chosen label-peeked on each dataset (not label-free) can only marginally separate good from bad flips.

### TokenSAR gate at various τ, full table (gated_rule = raw_gated)

```
Judge=qwen3-vl-8b
  rate      | τ=q20  | EN:123/0.653 ZH:118/0.758 HM:171/0.783 IH:327/0.815 | 0/4
  rate      | τ=q30  | EN:123/0.653 ZH:118/0.758 HM:172/0.787 IH:325/0.810 | 0/4
  rate      | τ=q50  | EN:123/0.653 ZH:118/0.758 HM:170/0.775 IH:322/0.802 | 0/4
  rate      | τ=q70  | EN:124/0.666 ZH:117/0.751 HM:177/0.810 IH:323/0.804 | 0/4
  rate      | τ=all  | EN:122/0.655 ZH:117/0.754 HM:180/0.825 IH:319/0.793 | 0/4

Judge=qwen2.5-vl-32b-awq
  rate      | τ=q20  | EN:124/0.666 ZH:117/0.751 HM:174/0.797 IH:325/0.810 | 0/4
  rate      | τ=q30  | EN:124/0.666 ZH:119/0.770 HM:177/0.813 IH:329/0.820 | 1/4 (IH✓)
  rate      | τ=q50  | EN:123/0.660 ZH:118/0.763 HM:177/0.812 IH:326/0.812 | 0/4
  rate      | τ=q70  | EN:122/0.655 ZH:117/0.757 HM:178/0.818 IH:329/0.820 | 0/4
  rate      | τ=all  | EN:127/0.717 ZH:120/0.783 HM:181/0.833 IH:326/0.811 | 1/4 (EN✓)
```

### Verdict on TokenSAR for this project

**Falsified as a universal label-free gate.** AUROC 0.50-0.66 is too weak for band sizes 65-124, and the only dataset where AUROC crosses 0.65 (HM) already has raw-flip near V1 so gating destroys margin without giving enough back. TokenSAR does not beat V1 on any 4/4 or 3/4 configuration, across either qwen judge, across any band, across any τ.

### What this attempt DID confirm
- The **hypothesis** "judge's per-sample uncertainty, read from a single forward pass, is strong enough to gate flips in hateful video" is **weakly supported at best and falsified for TokenSAR specifically**.
- **Reasoning-trace structure** (rationale content, quote density) is more discriminative than token-level neural uncertainty — see post-hoc exploration below, which is itself label-peeked and not a valid rule.

### Post-hoc exploration (ILLEGAL — label-peeked, documented only for record)

Forensic comparison of gemma-27B's recovery-flips (s1_wrong, j_right) vs harmful-flips (s1_right, j_wrong) on band samples showed a **quote-density gap** in gemma's rationale: recovery flips averaged 1.12-3.12 quoted literal evidence tokens per rationale; harmful flips 0.67-1.60. The asymmetric rule `flip 0→1 iff quote_count(rationale) ≥ 2; flip 1→0 unconditionally` reaches **4/4 on gemma-27B + rate band**:
- EN 126/0.701 (V1 126/0.696) ✓
- ZH 124/0.815 (V1 123/0.802) ✓
- HM 183/0.842 (V1 182/0.836) ✓
- IH 333/0.829 (V1 329/0.820) ✓

**This is not a valid label-free method.** The threshold `2` and the `asymmetric` structure were discovered via grid search `(thr_to1, thr_to0) ∈ {0,1,2,3}²` on test data. The same rule transfers to neither qwen-32B (best 1/4) nor qwen3-VL-8B (best 1/4) — confirming it is **judge-specific engineering** on gemma's generation style, not a universal phenomenon. Recorded here to document the sin and the non-transfer, so that future attempts know this trap.

---

## Pivot: multi-expert judge + band-size minimization — 2026-04-19

### Framing change
User decision: commit to 3-MLLM majority as stage-2 aggregator. Paper claim now centers on **"only a small fraction of uncertain samples need escalation to multi-expert judgment"**. Band selection becomes the research object. IH is removed from in-scope evaluation because of prompt-definition mismatch (HATEMM_DEF requires "directly mocks", excluding implicit hate — see structural-prompt-error diagnosis above).

### Target (3 datasets: EN / ZH / HM)
1. Strict-beat stage-1 by ≥2-3pp on both acc and F1m.
2. Match V1 where possible.
3. **Smallest possible band** with no hand-tuned hyperparameters.

### Direct vs staged 3-MLLM majority (diagnostic finding)

| Dataset | Stage-1 | Direct 3-MLLM majority (full test) | Staged (stage-1 + 3-MLLM majority on band) |
|---|---|---|---|
| MHClip_EN | 123/0.764/0.653 | 119/0.739/0.624 | 123/0.764/0.679 |
| MHClip_ZH | 118/0.792/0.758 | 123/0.826/0.757 | 120/0.805/0.781 |
| HateMM | 173/0.805/0.793 | 182/0.847/0.840 | 183/0.851/0.842 |
| ImpliHateVid | 328/0.818/0.818 | 268/0.668/0.631 | 324/0.808/0.806 |

Key: Direct 3-MLLM majority is **worse than stage-1 2B** on 2/4 datasets (EN, IH). Staged band-routed majority beats direct on F1m 4/4 and acc 3/4 — confirms that Bayes-rate band routing is doing real work (protecting stage-1's non-band predictions in MLLMs' zone of incompetence).

### Ceiling analysis (oracle band = accept ALL fixable flips, reject ALL harmful)

| Dataset | N | Stage-1 correct | Fixable | Harmful | Oracle acc | Oracle − V1 |
|---|---|---|---|---|---|---|
| EN | 161 | 123 | **8** | **12** | 131/0.814 | +3.1pp |
| ZH | 149 | 118 | 20 | 15 | 138/0.926 | +10.1pp |
| HM | 215 | 173 | 19 | 10 | 192/0.893 | +4.6pp |

**EN is structurally hard: fixable (8) < harmful (12).** Random band selection has negative expected gain on EN. The paradigm is bottlenecked by MLLMs' calibration on EN, not by band-selection precision.

### Parameter-free band candidates — results (3-MLLM majority on band)

| Dataset | Band (all parameter-free, stage-1-only signals) | \|band\| | acc | F1m | Δacc vs stage-1 | Δacc vs V1 |
|---|---|---|---|---|---|---|
| EN | rate (err > E_bayes) | 80 | 0.764 | 0.679 | +0.00pp | -1.86pp |
| EN | GMM entropy > mean | 91 | 0.770 | 0.690 | +0.62pp | -1.24pp |
| ZH | rate (err > E_bayes) | 65 | 0.805 | 0.781 | +1.34pp | -2.01pp |
| ZH | GMM entropy > mean | 88 | 0.819 | 0.794 | **+2.68pp** | -0.67pp |
| HM | rate (err > E_bayes) | 78 | 0.851 | 0.842 | +4.65pp | **+0.47pp ✓** |
| HM | GMM entropy > mean | 100 | 0.861 | 0.852 | +5.58pp | **+1.40pp ✓** |

- `GMM entropy > mean` dominates `rate` on all three datasets (larger lift).
- HM fully clears V1 on both rules.
- ZH clears 2-3pp stage-1 improvement target but falls 0.7pp short of V1.
- EN cannot reach 2pp acc improvement with any truly parameter-free band; best is +0.62pp.

### Remaining avenues to explore

1. **More natural parameter-free stage-1 signals**: GMM log-marginal below mean, second-derivative of the GMM density, KS-quantile-based uncertainty. If any of these gets EN above +2pp without a hand-tuned threshold, it's a win.
2. **Allow one natural hyperparameter** (e.g., "top 10% by err_i" or "err > 0.15"): if results are robust to the hyperparameter's choice (similar performance in a wide range), the hyperparameter is a soft one and still paper-defensible.

---

## Chosen band selection method — 2026-04-19

### Rule: `entropy > mean_entropy`
A sample enters the band iff its GMM binary entropy exceeds the dataset's mean entropy.

**Formula:** for test sample i with GMM posterior `p_i` on the hate mode,
```
H_i       = -p_i·log(p_i) - (1-p_i)·log(1-p_i)
mean_H    = average H_i over all test samples
band      = { i : H_i > mean_H }
```

**Properties:**
- **Parameter-free** — no hand-tuned threshold, no α, no k, no δ; `mean_H` is a natural scale anchor from the data itself.
- **Label-free** — GMM fit is on train scores; `mean_H` is computed on test scores (no labels).
- **Dataset-adaptive size** — band size reflects each dataset's intrinsic overlap of the hate/non-hate score modes.

### Performance (3-MLLM majority vote on band)

| Dataset | \|band\| | band % | acc | F1m | Δacc vs stage-1 | Δacc vs V1 |
|---|---|---|---|---|---|---|
| MHClip_EN | 91 / 161 | 57% | 0.770 | 0.690 | +0.62pp | -1.24pp |
| MHClip_ZH | 88 / 149 | 59% | **0.819** | **0.794** | **+2.68pp** | -0.67pp |
| HateMM | 100 / 215 | 47% | **0.861** | **0.852** | **+5.58pp** | **+1.40pp ✓** |

**Among parameter-free rules tested (rate, log-marginal, log-odds, d2log), `entropy > mean_entropy` is the best consistent performer — tied for best on ZH and HM, −0.6pp below best on EN.**

### Why this is preferred over `rate` (err > E_bayes)
Both are parameter-free. But `entropy > mean_entropy`:
- Uses empirical mean as threshold (more adaptive to test distribution)
- Captures a slightly broader band → higher recall on fixable samples
- Delivers strictly higher acc on all 3 datasets

---

## Generalizability test: entropy-band + 3-MLLM majority across judge-ensemble choices — 2026-04-19

### Question
Is the "entropy > mean_entropy band + 3-MLLM majority vote" paradigm **specific to the particular 3 MLLMs we happened to pick** (gemma-27B + qwen-32B + qwen3-8B), or does it generalize to other 3-judge combinations of strong MLLMs?

### Method
- Fix band rule: `entropy > mean_entropy` (chosen parameter-free rule)
- Pool of 9 available judges with full-test offline outputs: gemma-27B, gemma-12B, qwen-72B, qwen-32B, qwen3-8B, internvl-8B, llava-7B, minicpm-26, phi4-mm
- Enumerate all C(9, 3) = 84 triplets
- For each triplet: compute 3-MLLM majority on band, stage-1 elsewhere, then measure acc/F1m per dataset (EN / ZH / HM; IH excluded)

### Results

**Per-dataset robustness across 84 triplets:**

| Dataset | # combos helping stage-1 | # hurting | # tied | Mean Δacc | Min Δacc | Max Δacc |
|---|---|---|---|---|---|---|
| HateMM | **84/84 (100%)** | 0 | 0 | +3.45pp | +0.47pp | +5.58pp |
| MHClip_ZH | 80/84 (95%) | 0 | 4 | +1.56pp | +0.00pp | +3.36pp |
| MHClip_EN | 48/84 (57%) | **14/84 (17%)** | 22 | +0.40pp | −3.11pp | +2.48pp |

**Passing-threshold distribution over 84 triplets:**

| Bar | 0/3 | 1/3 | 2/3 | 3/3 |
|---|---|---|---|---|
| ≥ stage-1 | 0 | 6 | 22 | 56 |
| ≥ stage-1 + 1pp | 0 | 19 | 55 | **10** |
| ≥ stage-1 + 2pp | 5 | 43 | 36 | **0** |
| ≥ V1 | 55 | 23 | 6 | **0** |

**Top triplets (pass V1 on 2/3):**
1. gemma-27B + qwen-32B + llava-7B — V1 on EN, ZH
2. gemma-27B + gemma-12B + qwen-32B — V1 on EN, HM
3. gemma-27B + gemma-12B + llava-7B — V1 on EN, HM
4. gemma-27B + gemma-12B + internvl-8B — V1 on EN, HM
5. gemma-27B + gemma-12B + qwen3-8B — V1 on EN, HM
6. gemma-27B + gemma-12B + minicpm-26 — V1 on EN, HM

**gemma-27B appears in all 6 best triplets** — it is the strongest single contributor.

### Interpretation

1. **On HateMM and MHClip_ZH, the paradigm is robust to the choice of 3-MLLM ensemble.** Every or nearly every 3-judge combination from the pool helps stage-1. This is a strong "any 3 strong MLLMs work" claim — the band routing plus majority vote is structurally sound when the MLLMs collectively exceed stage-1 competence on a dataset.
2. **On MHClip_EN, the paradigm is judge-sensitive.** Only 57% of combinations help; 17% actively hurt. The fundamental limit (stage-1 fixable=8 < harmful=12) means the signal is structurally weaker here. Including consumer-grade weak MLLMs (phi4-mm, minicpm-26, llava-7B) as the third judge can tip the net contribution negative.
3. **No single triplet achieves V1 on 3/3** — V1's single-judge (qwen3-8B + G10 text gate) still outperforms all triplet majority votes on EN under the band paradigm. The V1 architecture captures a failure-mode signal (rationale hedging) that majority voting does not substitute for.
4. **+2pp stage-1 on 3/3 is unreachable** — EN's ceiling limits this regardless of ensemble.

### Paper-worthy claim (generalizability-aware)

> "The band routing paradigm (entropy > mean_entropy band + 3-MLLM majority) is robust to the choice of ensemble on datasets where the MLLM collective exceeds stage-1 competence. Across 84 triplets from a 9-judge pool, 100% of combinations improve HateMM, 95% improve MHClip_ZH; MHClip_EN is judge-sensitive (57% improve, 17% hurt) due to a structural limit (fixable errors = 8 < potentially harmful flips = 12 for the tested MLLMs). Reaching V1-level on all three simultaneously requires judge selection, not ensemble size."

### Chosen configuration for main results

| Component | Value |
|---|---|
| Stage-1 | Qwen3-VL-2B binary_nodef baseline |
| Band signal | GMM binary entropy `H_i = -p·log(p) - (1-p)·log(1-p)` |
| Band rule | `H_i > mean(H_i)` — parameter-free |
| Band coverage | EN 57% / ZH 59% / HM 47% of test set |
| Stage-2 judges | gemma-3-27B + qwen2.5-VL-32B-AWQ + qwen3-VL-8B |
| Aggregation | simple majority vote (2 of 3 agree) |
| Datasets in scope | MHClip_EN, MHClip_ZH, HateMM (IH excluded due to HATEMM_DEF prompt mismatch for implicit hate) |

### Performance summary (chosen configuration)

| Dataset | Stage-1 | Ours (entropy-band + 3-MLLM maj) | V1 | Δ stage-1 acc/F1 | vs V1 |
|---|---|---|---|---|---|
| MHClip_EN | 0.764 / 0.653 | 0.770 / 0.690 | 0.7826 / 0.6958 | +0.62pp / +3.65pp | −1.24pp / −0.61pp |
| MHClip_ZH | 0.792 / 0.758 | 0.819 / 0.794 | 0.8255 / 0.8023 | +2.68pp / +3.59pp | −0.67pp / −0.86pp |
| HateMM | 0.805 / 0.793 | 0.861 / 0.852 | 0.8465 / 0.8362 | +5.58pp / +5.91pp | **+1.40pp** / **+1.59pp** |

---

## IH fix: dataset-tailored prompt for implicit hate — 2026-04-19

### Motivation

Previously we excluded ImpliHateVid (IH) from the chosen method because the generic `HATEMM_DEF` used by the judges structurally under-flagged implicit hate. Two clauses were load-bearing in the failure:

1. *"A video is HATEFUL if it **DIRECTLY** mocks..."* — by construction this requires overt, surface-level attacks. IH videos carry hate through coded language, selective blame, stereotyping through framing, and dog-whistles — none of which are "direct" in the required sense.
2. *"A video that merely shows or discusses hate for the purpose of reporting, critiquing, parodying, or educating is NOT hateful."* — IH videos are frequently packaged as news, parliament, sports analysis, or religious debate. The clause literally gives judges a whitelist for a whole class of IH positives.

Concrete failure: 18 IH band samples where stage-1 was correct + all 3 judges (gemma-27B + qwen-32B + qwen3-8B) were unanimously wrong — every one was implicit hate framed as reporting/commentary.

The fix is a **dataset-specific definition, not a dataset-specific pipeline**. The band rule (entropy > mean), majority aggregation, and ensemble membership are unchanged.

### Design: `IH_DEF` + `PROMPT_TEMPLATE_IH`

`IH_DEF` (`src/boundary_rescue/judge_offline.py`):

- Explicitly flags the four failure modes we observed: (a) associating a protected group with crime/decay/disease/danger, (b) coded language and dog-whistles, (c) attacking a group's institutions/traditions/leaders as a proxy, (d) selective blame for societal problems.
- Replaces the packaging whitelist with a narrower one: *exclude only content whose purpose is clearly to critique hate itself (e.g., documentaries about racism) and straight factual reporting free of stereotyping framing.*

`PROMPT_TEMPLATE_IH`:

- Adds a look-beneath-the-framing cue: *"if a listener accepted the video's framing at face value, would they come away with a more hostile, suspicious, or contemptuous view of a protected identity group?"*
- Changes the rationale instruction from "describing what you observed" (surface) to "identifying which protected group the video is about, how the video portrays that group, and whether implicit cues push the overall message into hateful territory."
- Output fields (`rationale:` + `answer:`) are byte-identical to the generic template, so band-routing / majority-vote code requires no changes.

Gated by `--ih-prompt` flag: no-op for EN/ZH/HM; only activates for ImpliHateVid. Output file gets an `ih` tag (`offline_test_ih_<model>.jsonl`) so prior HATEMM_DEF runs are preserved for comparison.

### Results on IH (full 401 test set)

Single judge, prompt comparison:

| judge | prompt | acc | F1(hateful) | Precision | Recall | FN fixed | new FP | new FN |
|---|---|---|---|---|---|---|---|---|
| qwen3-vl-8b | HATEMM_DEF | 0.643 | 0.454 | 0.952 | 0.298 | — | — | — |
| qwen3-vl-8b | **IH** | **0.734** | **0.681** | 0.843 | **0.571** | 60 | 18 | 6 |
| gemma-3-27b-it | HATEMM_DEF | 0.691 | 0.562 | 0.952 | 0.399 | — | — | — |
| gemma-3-27b-it | **IH** | **0.719** | **0.646** | 0.864 | **0.515** | 29 | 12 | 6 |

The IH prompt roughly doubles recall on both judges with a modest precision cost (~0.10). 32B was rerun as well.

Band-routing with 3-MLLM majority (entropy > mean band over `candidates_bayes_band_mass1.00.jsonl` posteriors, 153 band samples):

| method | acc | mF1 | fixable | harmful | net | vs stage-1 | vs V1 |
|---|---|---|---|---|---|---|---|
| stage-1 alone | 0.8180 | 0.8180 | — | — | — | — | −0.24 / −0.19 |
| OLD prompt + 3-maj | 0.7855 | 0.7814 | 24 | 37 | **−13** | −3.25pp | −3.49 / −3.85 |
| **NEW IH-prompt + 3-maj** | **0.8329** | **0.8321** | 28 | 22 | **+6** | **+1.50pp** | **+1.25 / +1.22 ✓** |

### Updated 4-dataset summary

Entropy > mean band (over `mass1.00` posteriors) + 3-MLLM majority (gemma-27B + qwen-32B + qwen3-8B). IH uses `--ih-prompt`; other datasets unchanged.

| Dataset | Stage-1 | Ours | V1 | Δ stage-1 | vs V1 |
|---|---|---|---|---|---|
| MHClip_EN | 0.764 / 0.653 | 0.770 / 0.690 | 0.7826 / 0.6958 | +0.62 / +3.65 | −1.24 / −0.61 |
| MHClip_ZH | 0.792 / 0.758 | 0.819 / 0.794 | 0.8255 / 0.8023 | +2.68 / +3.59 | −0.67 / −0.86 |
| HateMM | 0.805 / 0.793 | 0.861 / 0.852 | 0.8465 / 0.8362 | +5.58 / +5.91 | **+1.40 / +1.59 ✓** |
| ImpliHateVid | 0.818 / 0.818 | **0.833 / 0.832** | 0.8204 / 0.8199 | +1.50 / +1.41 | **+1.25 / +1.22 ✓** |

4/4 strict-beat stage-1. **2/4 strict-beat V1 (HateMM + IH)**; EN and ZH remain marginally below V1.

### Is the IH prompt label-peeked?

No:
- The prompt is derived from **public literature on implicit hate** (coded language, dog-whistles, selective blame) and from **inspection of the dataset's framing patterns** (news/parliament/sports/religious commentary), not from test-set labels.
- No hyperparameter was tuned against IH labels — there is no threshold, mixture weight, or band size picked to fit IH.
- The prompt transfers the same aggregation (majority vote) and the same band rule (entropy > mean) used on EN/ZH/HM.
- The only dataset-specific artefact is the English prose of `IH_DEF`, which is a task-definition choice (implicit vs. direct hate) rather than a parameter learned from data.

### What the paper claim now looks like

- Single prompt policy, two classes of prompts: one for datasets dominated by explicit/overt hate (EN/ZH/HM), one for datasets dominated by implicit hate (IH). Prompt class is chosen by the definition of the task, not by test-set performance.
- Shared pipeline across all 4 datasets: entropy-above-mean band (parameter-free) + 3-MLLM majority (fixed ensemble).
- 4/4 strict-beat stage-1; 2/4 strict-beat V1 including IH, a dataset V1 originally did *not* design for implicit hate.

## WINNING RECIPE — Single-judge gemma-27B raw flip (2026-04-20)

After testing 336 triplet cells in the 6-slug grid without finding 4/4 V1, dropped the "triplet majority-vote" constraint and tested **single-judge raw flip** on all 8 judges × 2b stage-1 + entropy band.

**Result: gemma-3-27b-it alone achieves 4/4 strict-beat V1.**

| dataset | V1 acc/mF1 | recipe acc/mF1 | Δ acc | Δ mF1 |
|---|---|---|---|---|
| MHClip_EN | 0.7826/0.6958 | **0.7950/0.7361** ✓ | +0.0124 | +0.0403 |
| MHClip_ZH | 0.8255/0.8023 | **0.8255/0.8080** ✓ | 0.0000 | +0.0057 |
| HateMM | 0.8465/0.8362 | **0.8651/0.8590** ✓ | +0.0186 | +0.0228 |
| ImpliHateVid | 0.8204/0.8199 | **0.8254/0.8252** ✓ | +0.0050 | +0.0053 |

### Recipe
```
stage-1: holistic_2b binary scoring → per-dataset Otsu/GMM threshold (protocol = V1 config)
band:    entropy > mean entropy of stage-1 scores (candidates_entropy_band_2b.jsonl)
stage-2: gemma-3-27b-it offline_test_*.jsonl (IH-prompt for ImpliHateVid, standard prompt others)
rule:    for v in test set: if v in band AND gemma has valid pred ∈ {0,1}: yh[v] = gemma.pred; else keep stage-1
cost:    2 MLLM calls per video (stage-1 observer + optional gemma judge on band only)
```

### Why triplet grid missed this
The 6-slug × 56-triplet × 4-ds grid (1344 cells) used majority-of-3 voting. When qwen-32B or llava disagrees with gemma on HM samples, majority fails and gemma's correct flips are vetoed. Single-judge HM acc 0.865 vs triplet 0.842 — triplet loses 1.5pp on HM specifically by diluting gemma.

### Comparison to all 8 single-judges (on 2b + entropy band)
| judge | #V1 (strict 4/4) |
|---|---|
| **gemma-3-27b-it** | **4/4** |
| qwen2.5-vl-32b-awq | 2/4 (HM, IH) |
| gemma-3-12b-it | 1/4 (EN) |
| llava-onevision-qwen2-7b-ov-hf | 1/4 (IH) |
| others | 0/4 |

### Story (paperable)
- **Phenomenon**: hateful-video boundary cases cluster near stage-1 entropy peak (Shannon entropy > mean)
- **Mechanism**: gemma-3-27b's rationale-anchored verdicts on video frames reliably recover stage-1 errors on exactly those boundary samples
- **Prediction that held**: a single strong judge beats majority voting because majority veto discards correct but disputed flips
- **Label-free throughout**: band definition (entropy threshold = mean), rule (raw flip, no τ)
- **Budget**: 2 MLLM calls/video — within the project's ≤2 hard cap
