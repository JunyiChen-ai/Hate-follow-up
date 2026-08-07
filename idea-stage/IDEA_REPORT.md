# Idea Discovery Report

**Direction**: Next novel mechanism for label-free MLLM hateful video detection, building on the confirmed channel-restoration mechanism (evidence-side input intervention on a frozen single-call 8B judge, raw-z readout, label-free KDE-valley threshold).
**Date**: 2026-08-08
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Primary context**: `RESEARCH_BRIEF.md` (repo root); constraints in `docs/analysis/prior_falsification_map.md`.

## Literature Landscape (Phase 1, research-lit, composed)

Sources contributed: arXiv API (helper, all hits fetched and verified by construction), WebSearch. Zotero / Obsidian / Gemini / local PDF library unavailable on this machine — skipped per protocol (`WARN: local contributed nothing`). All papers below tagged ✅ were individually fetched from the arXiv API; none are from memory.

### Theme A — Hateful/harmful video detection with (M)LLMs

| Paper | Venue/Date | Method | Relevance to us | Status |
|---|---|---|---|---|
| MARS: Training-Free Hateful Video Detection via Multi-stage Adversarial Reasoning (2601.15115) | arXiv 2026-01 | Multi-stage adversarial reasoning, training-free | Direct competitor in the training-free space; breaks our ≤2-call cap; already a no-fly zone in the falsification map | ✅ arxiv |
| MM-HSD (2508.20546) | arXiv 2025-08 | Trained fusion incl. **on-screen text** and audio | Independent evidence that on-screen text carries hate cues other modalities miss; supervised, not label-free | ✅ arxiv |
| ImpliHateVid benchmark paper (2508.06570) | arXiv 2025-08 | Two-stage contrastive learning (supervised) | The benchmark authors' own method; supervised reference | ✅ arxiv |
| HarmVideoBench (2606.27187) | arXiv 2026-06 | Benchmark; notes binary framing misses implicit/contextual harm, rationales absent | Benchmark-side confirmation that implicit harm is the open problem | ✅ arxiv |
| Cross-cultural VLM hateful-meme evaluation (2602.07497) | arXiv 2026-02 | Evaluation study | Context for cross-lingual behavior (cf. our MHClip-ZH neutrality result) | ✅ arxiv |
| IARE (SIGIR 2026) | supervised F1 91.75 on ImpliHateVid test | — | Supervised ceiling on our primary benchmark | repo docs |

### Theme B — Temporal granularity of hate

| Paper | Venue/Date | Method | Relevance | Status |
|---|---|---|---|---|
| MultiHateLoc (2512.10408) | arXiv 2025-12 | Weakly-supervised **temporal localisation** of hate segments | Occupies localisation-as-a-task; does NOT do label-free detection via evidence completeness | ✅ arxiv |
| Temporal Label Noise in Hateful Video Classification (2508.04900) | arXiv 2025-08 | Shows hateful videos contain long non-hateful stretches; trims labels (supervised) | **Independent confirmation of the temporal-sparsity phenomenon** on HateMM; supervised remedy only | ✅ arxiv |
| Adaptive Keyframe Sampling (2502.21271); LENS (2607.25125); FOCUS (2510.27280); KeyVideoLLM (2407.03104); Moment Sampling (2507.00033) | 2024–2026 | Query/answer-driven keyframe selection for video QA | Tooling exists, but none is moderation-specific, none is driven by evidence-completeness for a frozen judge, none label-free w.r.t. an operating point | ✅ arxiv (first two), others web-only |

### Theme C — VideoLLM evidence-surfacing failures (the neighborhood of our confirmed mechanism)

| Paper | Venue/Date | Finding | Relevance | Status |
|---|---|---|---|---|
| Failures to Surface Harmful Contents in VideoLLMs (2508.10974) | arXiv 2025-08 | Sparse uniform frame sampling + spatial downsampling + fusion imbalance make VideoLLMs omit clearly-visible harmful content | **Independent root-cause confirmation of channel starvation on the visual side.** Documents the failure; does not build a detection method, nothing label-free | ✅ arxiv |
| VisualTextTrap / Text-Overlay-Induced Hallucination (2604.17375) | arXiv 2026-04 | VLMs treat rendered on-screen text as privileged and let it override visual evidence | Double-edged for any OCR-channel idea: restoring on-screen text may over-trigger; over-trust is input-side real | ✅ arxiv |
| MME-VideoOCR (2505.21333) | arXiv 2025-05 | MLLM OCR degrades badly in video (blur, temporal variation) | Supports "the on-screen-text channel is starved at our 91-token/frame resolution" | ✅ arxiv |

### Theme D — Label-free operating-point selection

Generic post-hoc calibration (temperature scaling, isotonic regression, bias correction) assumes labeled or human-rated anchors; LaFTer (NeurIPS 2023) tunes CLIP classifiers label-free but does not address decision thresholds for judge score distributions. **No work found that derives a moderation operating point from the geometry of a frozen judge's score distribution** (our KDE-valley / commitment-bound line), and no work addresses the failure mode we measured on HateMM (geometric valley diverges from the class boundary when the normal class is saturated with hate-adjacent surface features). This subspace appears unoccupied. (Web-only sources; treat as ⚠ lower-confidence coverage.)

### Landscape synthesis

1. The field's active fronts are benchmarks (HarmVideoBench), supervised fusion (MM-HSD), multi-call reasoning frameworks (MARS), and temporal localisation (MultiHateLoc). **Nobody occupies "evidence-side input intervention on a frozen single-call judge, label-free end to end"** — our confirmed family remains ours.
2. Two independent 2025–2026 papers (2508.10974, 2508.04900) confirm the two phenomena we hypothesized from our own error budget — visual/temporal starvation and temporal label sparsity — but both stop at diagnosis or supervised fixes. This is strong external validation that the remaining channels (visual-temporal, on-screen text) are real and unclaimed as label-free method territory.
3. The on-screen-text channel has a documented hazard (2604.17375: rendered text hijacks the model) that any OCR-restoration idea must design around — attribution-aware injection rather than naive text concatenation.
4. Label-free operating-point geometry (Theme D) is unoccupied but also the hardest to make a headline contribution alone; it is best positioned as a component upgraded by whatever evidence mechanism wins.

## Idea Generation (Phase 2)

35 candidates from five parallel Claude lenses (method-transfer, untested-assumption, diagnostic, evidence-channel, score-geometry; per-lens JSON shards in the session task logs, cluster map in `candidate_clusters.json`) plus a 10-idea GPT-5.6-Sol (xhigh) cross-model seed. Mechanical dedup → 7 clusters; budget gate eliminated none. Cross-model jury: GPT-5.6-Sol xhigh devil's-advocate triage (thread `019fdc35-4aaa-7650-8ec1-a593be9e44a6`, same thread as the seed).

## Recommended Ideas (ranked; jury ranking A > E > C > B > D > G > H > F)

### 🏆 Idea 1: A2 — OCR-routed pixel-budget reallocation (visual-text channel restoration)
- **Method**: (1) Run a general OCR engine over each video's native-resolution frames; use its text-box locations ONLY as a router — the recognized strings never enter the prompt. (2) Reallocate the judge's fixed vision-token budget toward text-bearing frames (more pixels where rendered text lives, fewer where none is), total tokens matched to baseline. (3) One unchanged frozen 8B judge call; raw z; label-free valley. Ablations: anti-targeted allocation (pixels to least-text frames) must not gain; OCR-string injection arm must show the FP inflation that router-only avoids.
- **Hypothesis**: burned-in captions/overlays carry hate the judge cannot resolve at 91 tokens/frame (MHClip-EN's ceiling); restoring legibility restores evidence — same law as speech restoration, different channel.
- **Minimum experiment**: CPU pre-gate: OCR census over MHClip-EN native frames_16 (proceed iff ≥25% of videos have ≥2 substantive text frames). Then 4 token-matched single-call arms on MHClip-EN train + IHV train as predicted-null control. ~30 GPU-min total.
- **Novelty**: CONFIRMED differentiable — token-allocation literature is efficiency-oriented (pruning/compression: FastOCR, RTPrune, token-recovery AAAI'25); nobody uses OCR-as-router for evidence restoration to a frozen moderation judge. Closest: MME-VideoOCR (diagnosis only), MM-HSD (supervised fusion).
- **Jury objection**: "you are magnifying slurs — recall up, quote/report FPs up too." Kill-test: routed arm must beat matched-random AND anti-targeted by ΔAUC ≥ 0.03 on the pre-registered text-bearing stratum with non-negative hateful-vs-normal flip asymmetry; gains requiring string injection or extra tokens kill the safe-restoration claim. Verify actual post-processor token counts (nominal ≠ effective).
- **Risk**: MEDIUM · method · pilot NEEDS GPU SCHEDULING (owner occupies GPU).

### Idea 2: E7→E1 — commitment-geometry decomposition → mixture-posterior operating point
- **Method**: (1) E7 (zero GPU, running): test whether the z-distribution's two modes are corpus-invariant commitment states of the model (within-class KDE, composition resampling, pooled BIC, 2B contrast). (2) Only if supported: freeze a 2-component unequal-variance mixture (deterministic init, no free parameters) and place the operating point at the posterior-0.5 crossing instead of the density valley. (3) E6 certificate (multivariate shape statistics → flag corpora where the geometric operating point is untrustworthy; leave-one-cell-out).
- **Hypothesis**: the judge has a narrow committed component and a wide deliberative component; which side is committed is corpus-dependent (IHV: committed-acquit; HateMM: committed-convict). The valley equals the Bayes boundary only under equal weights/widths — exactly what fails on HateMM.
- **Empirical signal**: exploratory CPU pilot (LABEL-PEEKED, hypothesis-generating only): HateMM macro-F1 0.797 (train) / 0.817 (test) vs valley 0.643/0.656 — closes ~70% of the 0.23 oracle gap; IHV −0.01 (nothing to win). Estimator must be frozen by prereg before any confirmatory test-split claim.
- **Novelty**: the estimator is classic (closest: GMM score calibration in speaker verification, arXiv 1311.0707; prior project's own TR-GMM ran on the falsified 2B substrate) — the contribution must be the validated commitment-asymmetry law (E7) + certificate + the valley≠Bayes analysis, not "we used a GMM".
- **Jury objection**: identifiability — components may be content subdomains, not commitment states; E7 is the gatekeeper. Kill: modes track corpus composition, or within-class distributions all unimodal.
- **Risk**: MEDIUM · method+theory · **E7 VERDICT: two-state form REFUTED** (locations corpus-owned, not model-owned — see pilot table). E1 downgraded to benchmark-threshold status. **Successor candidate**: the post-hoc four-state shared-saturation geometry (−18.1/+15.0 anchors invariant across corpora/conditions/splits, occupancy 29× variable, absent on 2B) — promotable ONLY via its own prereg: predict anchor locations in advance on a held-out corpus/condition, and show a saturation-anchored label-free operating point beats the KDE valley.

### Idea 3: C1/C2 — speaker-provenance restoration (input-side fix for the FP half)
- **Method**: (1) Diarize the existing wavs (pyannote-class, general-purpose). (2) Rewrite the transcript as speaker-attributed turns (who said what, when) — an input channel the judge has never received. (3) One frozen judge call; compare z on the located FP / cue-matched TP / cue-carrying TN cohorts. Controls: wrong-partition placebo (not mere renaming — jury correction), segmentation-only arm, single-speaker no-op check, pre-committed TP non-inferiority.
- **Hypothesis**: flat transcripts erase utterance ownership, so quoting/reporting/counter-speech reads as assertion — the measured p=3e-7 FP signature. Restoring the WHO is channel restoration on the provenance axis.
- **Novelty**: differentiable from the dead prompt-side stance probes and from MARS/MATCH-HVD (they change questions/reasoning; this changes evidence). Diarization itself is mature — novelty rests on demonstrating the causal quote/endorsement failure in frozen video judges.
- **Jury objection**: "Speaker A/B labels are semantically anonymous — segmentation without identity may do nothing." Highest-upside, highest execution risk. Sequence AFTER the owner's in-flight temporal-attribution pilot reports (same phenomenon family).
- **Risk**: HIGH · method · needs GPU scheduling.

### Idea 4: B1 — temporal-coverage audit (zero-GPU gate for the whole temporal family)
- **Method**: measure, against HateClipSeg's 11,714 timestamped segment annotations (analysis-side gold), how often uniform-16 sampling misses every hate-bearing segment; compare a shot-covering sampler at the same budget; dose curve at 8/16/32 frames.
- **Kill bars (frozen)**: uniform-16 video-level hit ≥0.95 → family dead; shot-covering gain <10pp → sampler line dead. PILOT RUNNING (CPU).
- **Risk**: LOW · empirical gate. The downstream sampler (B2/B3) is jury-ranked as crowded prior work; B4 (clause-synchronous storyboard) is the differentiable one but collides with the owner's pilot — deferred.

### Idea 5: D3 — speech separation before ASR (incremental extension of the confirmed route)
- **Method**: Demucs vocal stem → same frozen Whisper → judge; accompaniment-stem ablation. Targets music-masked speech FNs. Least novel, most likely to just work; a system component, not a headline.
- **Risk**: MEDIUM · empirical · needs GPU scheduling (short).

## Eliminated / Deferred Ideas

| Idea | Fate | Reason |
|---|---|---|
| F5 causal-evidence-locus (19 deletions/video) | ELIMINATED | violates the ≤2-call cap as written (jury) |
| F6 argmax-|z| input selection | ELIMINATED | confidence/commitment gating — falsified family (jury) |
| F1 mute-counterfactual axis, F2 counterfactual-null calibration, F3 carrier-redundancy | DEFERRED-HOSTILE | mechanically recreate the falsified narrowed-call-fusion family; deletion sensitivity ≠ stance; only non-deployed diagnostic variants may ever run |
| C4 referent-neutralization | DEFERRED | causal premise weak (quotes are also referent-dependent); Δz may measure grammaticality |
| E2 PU mixture-proportion | DEFERRED | tail-purity premise implausible exactly on HateMM (high-z violent normals) |
| E3 persistent valley | DEFERRED | stability ≠ correctness; HateMM's wrong valley may be the most persistent |
| E4 phenotype thresholds, E5 violence residualization, G router | PHASE-2 ONLY | per-dataset-switch smell (hard lesson 8); router needs F7's dose-response law first |
| **B1/B2/B3/B5 — entire temporal-coverage family** | **ELIMINATED (pilot)** | B1 audit: uniform-16 already covers 96.8% of hateful videos; shot-covering worse at equal budget; misses are budget-crowded ordinary segments, not sparse flashes. Any framing motivated by "the offending signal is sparse in time and uniform sampling misses it" is dead on this evidence. (B4 clause-frame binding is NOT killed by this — its premise is pairing, not coverage — but stays deferred behind the owner's pilot.) |
| A3 count-vs-resolution, D1/D2 audio tags | ANNOTATED LOW | diagnostic value, weak as headline; D2's HateMM violence-confound disconfirmer likely fires |
| H/E9 cross-corpus policy offset | KEEP AS ANALYSIS | scope/limitations section material, not a method |

## Pilot Experiment Results

| Pilot | Compute | Status | Signal |
|---|---|---|---|
| E7 commitment-geometry decomposition | CPU | **DONE — TWO-STATE REFUTED; POST-HOC FOUR-STATE REGULARITY FOUND** | K=2 locations belong to the CORPUS, not the model (cross-corpus drift 0.47–0.70 of inter-mode distance; prevalence-resampling drift up to 1.66×; class-conditional fits differ in 8/10 cells; shared-location K=2 costs +69 to +614 BIC) → E1 cannot be sold as "commitment geometry"; per-corpus GMM remains a benchmark threshold only. POST-HOC: shared-location **K=4** BEATS free fits in all four pools (ΔBIC −18 to −102); the two SATURATION states sit at −18.1±0.27 / +15.0±0.23 across 4 corpora × 2 conditions × 2 splits while occupancy varies 29×; the 2B never reaches these magnitudes (boundary condition intact). Not pre-registered — needs its own prereg with in-advance location prediction on a held-out corpus/condition + an operating point built from saturation anchors that beats the valley. Details: idea-stage/pilots/e7_commitment_geometry/results.json |
| B1 uniform-16 coverage audit | CPU (+4.1 GiB pull) | **DONE — BOTH KILL BARS FIRE** | **NEGATIVE: temporal-coverage family killed.** Uniform-16 (the repo's exact linspace rule) already hits ≥1 hateful segment in 96.8% of hateful videos (HateClipSeg, 345 offensive-union / 180 hateful-strict, both ≥0.95 bar). Shot-covering at equal budget is WORSE (−6.0 to −7.7 pp segment-hit, monotone across the frozen tau sweep). Misses are ordinary-length segments crowded out by budget (median 8.0 s; only 16.7% under 5 s), not sparse flashes; not concentrated in low-speech videos (p=0.91). Only 11/345 videos fully missed (offensive share 1–6% of runtime). Contiguous tiling caps ANY 16-frame sampler at 0.807 segment-hit; uniform already reaches 68% of the ceiling. Details: idea-stage/pilots/b1_coverage_audit/results.json |
| E1 mixture-posterior (exploratory) | CPU | DONE (label-peeked) | POSITIVE +0.16 HateMM macro-F1, needs frozen prereg |
| A2 OCR census pre-gate | CPU | ready to run | — |
| A2 / C1 / D3 GPU arms | GPU | BLOCKED: GPU occupied by owner | needs scheduling |

## Suggested Execution Order

1. **E7 + B1 pilots** (running) — they gate ideas 2 and 4 for free.
2. **A2 CPU pre-gate** (OCR census on MHClip-EN) — next zero-GPU step; if it passes, A2's GPU arms are first in queue when the GPU frees.
3. **E1 prereg freeze** if E7 supports the commitment story (estimator + crossing rule frozen before any further label contact).
4. **C1 provenance** after the owner's temporal-attribution pilot reports.
5. Paper spine (jury): confirmed speech restoration → safe visual-text restoration → frozen-judge SOTA → independently validated label-free operating point; provenance as alternative headline if it lands a large specificity gain.

## Next Steps

- [ ] Read E7/B1 pilot verdicts; update rankings
- [ ] /research-refine-pipeline on the surviving top idea (proposal + experiment plan + prereg)
- [ ] GPU scheduling with the owner for A2 arms

