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

### Follow-up: symmetric saturation-anchor replication

The post-hoc E7 four-state regularity received a separate held-out mechanism
pilot on the previously excluded stance-reader score conditions. **Verdict:
FAIL.** The positive tail replicated near `+15.0` and extreme states were less
reader-sensitive than the interior, but the negative tail moved to about
`-15.1` in two HateMM reader conditions, outside the frozen `-18.1 +/- 2.5`
bar. The symmetric two-anchor mechanism is retired; no threshold-performance
experiment is licensed. See `docs/duplex/SATURATION_ANCHOR_PILOT_NOTE.md`.

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

# Round 2 — Post-attribution idea discovery (2026-08-09/10)

**Date**: 2026-08-09/10. **Primary context**: `RESEARCH_BRIEF.md` rev 2 (commit
802996e). **Pipeline**: targeted literature survey → three specialist lens
agents with full repo access → cluster merge with a ten-idea GPT-5.6-Sol seed →
cross-model jury (GPT-5.6-Sol xhigh via Codex, thread
`019fe631-f21e-7c43-a64a-0b70255b36b2`) → three preregistered pilots.

Round 1 asked which evidence channel to restore. Round 2 asked a different
question, and this section records the answer.

## 1. What this round asked

The five-corpus error attribution closed the previous round's premise. No
model-side blind spot survived its controls, the decision boundary was shown to
be annotation-owned, and the owner vetoed further data-preprocessing-shaped
contributions. The brief therefore replaced "which channel is starved" with a
constraint question: **which of the project's own standing constraints should be
relaxed to buy mechanism-level novelty, and at what cost to the label-free
claim?** The brief listed five relaxations. R1 is a second call with a role that
is not in the falsified family. R2 is label-free training or adaptation at the
representation level rather than on the answer posterior. R3 is a readout wider
than one scalar. R4 is task specification as a first-class input rather than as a
prompt. R5 is the judge backbone, admissible only if a scale or architecture
phenomenon is itself the claim.

Every idea generated this round was required to name which relaxation it spends
and to say what it would cost if the relaxation is refused.

## 2. Landscape delta against round 1

The survey was run against R1 through R5 rather than against the hateful-video
topic, and it changed three things. Latent handoff between calls is now claimed
prior art: LatentMAS (2511.20639, ICML 2026 Spotlight, search-result-only)
passes last-layer hidden states and key-value caches between agents without
text, so "we pass hidden states between two calls" is no longer novel on its
own and only the *role* of the second call can be. The project's own scalar
readout is now independently derived in the literature: "When Does a Language
Model Commit?" (2605.06723, abstract-verified) defines the finite-answer
projection as the Yes-minus-No log-odds difference, shows it stabilises 17 to 31
tokens before the answer is parseable, and recovers it from compact hidden
states, which strengthens the base mechanism's theoretical footing while
removing any claim to the readout quantity itself. Specification-conditioned
weight generation appeared during the round: Compliance2LoRA (2607.27594,
abstract-verified) treats safety policies as inputs to a LoRA generator, which
closes the "hypernetwork emits an adapter from policy text" door and leaves only
"specification conditions the readout geometry of a frozen judge" open. Three
further papers frame the neighbourhood without occupying it: CLIPTTA
(2507.14312, search-result-only) argues that entropy minimisation is misaligned
with contrastive vision-language pretraining and replaces it, which is the
project's own saturation finding one architecture generation earlier; NExT-Guard
(2603.02219, abstract-verified) uses off-the-shelf sparse autoencoders as a
training-free text safety readout; FBHM (2605.31349, abstract-verified) steers a
vision-language model for hateful memes with roughly 500 supervised samples. GMP
(2603.01724, abstract-verified) is the empirical form of this project's
ill-posedness proof, showing that language-model moderation degrades when
guidelines are unstable or context-dependent. One negative result from the
survey is worth recording: a search synthesis attributed bimodal decision
projections to "Geometry of Decision Making in Language Models" (2511.20315),
the abstract does not contain that claim, and no published paper was located
that reports the bimodal logit-contrast phenomenon this project measures.

**Competitor alert.** LELA, "Towards Training-free Multimodal Hate Localisation
with Large Language Models" (2602.09637, 10 February 2026, abstract-verified),
claims to be the first training-free large-language-model framework for hate
video localisation, decomposes video into five modalities, and evaluates on
HateMM and MultiHateClip, which are two of this project's five corpora. It
appears to be a multi-stage prompting cascade and therefore almost certainly
exceeds the two-call cap, and it targets localisation rather than a video-level
operating point. It nonetheless removes the round-1 claim that nobody occupies
training-free hateful video. SafeLens (2605.17610, abstract-verified) takes the
adjacent slot on the guardrail side with a trained fast-and-slow video
architecture. The single-call frozen-judge label-free position is still
unoccupied, but novelty must now be argued at the mechanism layer and never at
the "training-free" layer.

## 3. The five clusters and the jury verdict

Three lens agents (R4 specification-conditioned readout; the R1-by-R3 empty cell
of a second call that manufactures a contrast; R2 combined with R5) were run
with full repository access, and one of them ran zero-GPU measurements. Their
output was merged with a ten-idea cross-model seed into five clusters. The jury
ranked them 2 > 1 > 3 > 5 > 4.

**Cluster 2, specification-conditioned readout geometry (jury rank 1).** Judge
each video twice under two rule lists that already exist verbatim in the
repository, discard both scalars, and read the layer-27 displacement between the
two states as a predictor of which videos change class when the policy changes.
The jury called this the strongest of the five and the only one that addresses
the ill-posedness proof directly, because HateClipSeg supplies the same corpus
under two label collapses and therefore a rare decisive falsification. Its
predicted laboratory failure was named in advance and is exactly what happened:
almost all displacement energy sits in a constant rule-list carrier, a
length-matched off-construct policy produces an equally strong direction, and
the per-video interaction residual is too weak to predict flips.

**Cluster 1, answer-side contrast readout (jury rank 2).** Hold the prompt
byte-identical, append the first token of "Yes" in one forward and of "No" in
the other off a shared key-value cache, and read the leading direction of the
per-video difference after the corpus-mean answer direction is removed. The jury
judged this conditionally defensible and named its collision precisely: the
commitment paper already theorises the Yes-versus-No separation and reads it
from hidden states, so the claim cannot be that the two states differ but only
that the per-video interaction residual carries construct information the scalar
discards. The predicted failure was that mean subtraction leaves a residual
whose first component still correlates with the scalar.

**Cluster 3, scale emergence with geometry distillation (jury rank 3).** The 2B
judge holds the construct as linearly as the 8B and ranks nearly as well, but its
residual stream does not rotate toward the answer direction, so no label-free
operating point exists at that scale. The jury rated the finding strong and the
method half weak, demoted the ridge distillation from claimed novelty to a cost
demonstration pending the owner's ruling on cross-model supervision, and warned
that two model sizes do not establish emergence.

**Cluster 5, illocution and endorsement axis (jury rank 4, parked).** A second
call under a non-evaluative role restates the speaker's claim, the generated
text is discarded, and the difference between the judge-role state and the
author-role state is read as an endorsement axis aimed at the measured MHClip-ZH
failure where a keyword in the title is treated as an assertion. The jury's
objection is that the role change is not an identifiable endorsement
intervention, because it alters perspective, task, and generation behaviour at
once, and that asking the model to restate a mentioned claim may manufacture the
very endorsement it aims to measure. Parked, not killed.

**Cluster 4, cross-modal redundancy direction (jury rank 5, killed).**
Canonical correlation between a video-only forward and a transcript-only
forward, after the scalar is projected out, was proposed as a label-free
criterion that selects the construct because modality-specific nuisance does not
transfer across views. The jury killed it as a submission direction on two
grounds. Canonical analysis finds shared identity and topic rather than hate,
and, decisively for this project, the contribution depends on creating narrowed
modality views, which is input restructuring as measurement apparatus and falls
under the owner's veto whether or not it happens offline.

## 4. Pilot results

Three preregistered pilots ran. All three are dead. Each preregistration was
committed before any judge call, and each verdict note reports every clause
including the ones that passed.

| Idea | Prereg commit | Verdict commit | One-line death cause |
|---|---|---|---|
| Cluster 2, specification displacement | `34fbdc3` | `b2b0f05` | Displacement is genuine but too small on MHClip-EN (0.710 against a 0.72 floor) and on HateClipSeg is a single-arm readout in disguise (placebo 0.665 against 0.724, second arm contributes 0.003). |
| Cluster 1, answer-side contrast | `0271a68` | `030d310` | The Yes-minus-No leading direction correlates with the scalar at Spearman 0.983, firing the preregistered renaming abort at 0.90 on all four evaluation corpora. |
| Interleaved timeline (owner's parallel kill test) | `c78955e` | `4f962d1` | The misaligned control captured the whole gain (+0.038 against +0.033 on the binding stratum, ratio 1.16), so the judge does not bind text to adjacent frames. |

**Specification displacement, detail.** The carrier-aligned readout passed two of
five frozen clauses and the residual axis passed none. On MHClip-EN the pairing
is load-bearing (the shuffled placebo falls to 0.492, chance) but the separation
is 0.710 against a 0.72 floor. On HateClipSeg the separation clears its 0.70
floor at 0.724, and then the shuffled placebo reaches 0.665, because the score
is reproduced to within 0.003 by the strict arm's projection alone at 0.721. The
off-construct control separated the two readouts and killed the residual axis
outright: a length-matched spam-and-copyright policy reached 0.720 on MHClip-EN,
above the real construct swap's own 0.663 on the same stratum. Both directions
are close to orthogonal to the direction that carries the distinction, at cosine
0.034 for the residual component and 0.037 for the carrier against the
supervised probe refitted on the same videos.

**Answer-side contrast, detail.** The mechanism half of the story survived and
the readout half did not. The degeneracy abort did not fire, at median cosine
0.902 between each video's difference and the corpus mean, and the pairing
placebo margin of 0.092 cleared its 0.05 bar, so the manufactured contrast is
genuinely evidence-conditioned. The renaming abort fired at 0.983, 0.979, 0.949
and 0.946 on the four evaluation corpora. Manufacturing the contrast made the
problem worse rather than better: the prompt-side component tracked the scalar
at 0.992 with 46 percent of the variance, and the answer-side component tracks
it at 0.983 with 64 percent. Layer 18 is the only depth that escapes the
renaming abort, at 0.860, and it escapes by being degenerate instead, at median
cosine 0.996.

**Stage A of that pilot corrected two premises and is the more durable result.**
Inside the frozen saturation band the scalar is not flat: its in-band range is
+13.000 to +19.500 with a standard deviation of 1.50, and it reaches AUC 0.6472
on the in-band arena. The band-conditioned principal-component control, run as a
free rescue of the earlier corpus-wide failure, reached only 0.6348 against a
0.70 floor. Most importantly, a leave-one-out supervised probe on the same 183
in-band videos reaches 0.6253, which is *below* the scalar it was meant to beat,
while the same probe reaches 0.837 on the MHClip-EN construct stratum. The
saturation band is therefore not a hidden arena where the information waits to
be found; it is close to label-unseparable at layer 27 even with labels. Any
future work that chooses that arena should suspect the arena as much as the
method, particularly given that HateClipSeg's negative class is already known to
contain protected-group hostility at 36 percent.

## 5. Synthesis: unsupervised readout access is now falsified from both sides

The single most important thing this round produced is a closed statement, not a
method.

The supervised probe at layer 27 separates no-protected-target offence from
protected-target hate at leave-one-out AUC 0.837, where the scalar reads 0.321
in the same direction. The information exists in the representation. Every
attempt to reach it without labels has now failed, and the failures cover the
space rather than sampling it:

1. **Corpus-wide principal components** rename the scalar. The first component
   correlates with it at 0.992 and 0.995 at layers 27 and 36 and reaches only
   0.746 against a 0.799 floor, with cross-corpus replication at 0.617.
2. **Band-conditioned principal components**, which remove the scalar's variance
   by construction, reach 0.6348 against a 0.70 floor, and the arena itself has
   a supervised ceiling of 0.6253.
3. **Extreme-pseudo-label heads** buy 0.0398 over the scalar on HateMM while
   losing ranking (0.8893 against 0.9232) and require two-sided saturation
   occupancy that three of five corpora lack.
4. **Specification-displacement directions**, which draw the direction from
   outside the corpus, are near-orthogonal to the working direction at cosine
   0.034 and 0.037, and their off-construct placebo outperforms the real
   construct swap.
5. **Forced-answer contrasts**, which manufacture the contrast on the answer
   side with the prompt held byte-identical, concentrate 64 percent of the
   variance onto the scalar.

Stated precisely: **every unsupervised summary of a single judge call's hidden
states either renames the scalar readout or points somewhere orthogonal to the
construct, while supervised access at 0.837 proves the information is present.**
The two failure modes are not independent accidents. Both places where the answer
lives, namely the state that produces the logits and the difference between the
states that follow the two answers, have their dominant variance aligned with the
logit contrast, because the logit contrast is a linear functional of very nearly
that difference read through the unembedding. Unsupervised variance answers the
question it is asked, and the question it is being asked is the readout.

What follows is a fork rather than a next experiment. Either the single call
must be relaxed, or the source of the direction must come from outside the
model's own unlabeled variance. Section 7 states the second option and the
ruling it needs.

## 6. Surviving assets

**The scale-emergence finding (Cluster 3).** The following are **round-2
measurements, hypothesis-generating, not preregistered**. They were run on CPU
over ImpliHateVid `train_clean`, 1,283 videos, comparing the 8B judge against
the 2B on artifacts already on disk.

| Quantity | 2B | 8B | Ratio |
|---|---:|---:|---:|
| Linear probe AUC on hidden states (layer 21 on 2B, layer 27 on 8B, ridge, five-fold) | 0.9655 | 0.9647 | 1.00× |
| Ranking AUC, hateful against normal | 0.898 | 0.934 | 1.04× |
| Standard deviation of the raw logit contrast | 0.82 | 11.89 | 14.5× |
| Relative KDE trough depth, stored bf16 | 0.0477 | 0.4043 | 8.5× |
| Relative KDE trough depth, de-quantized fp32 | 0.0315 | 0.4011 | 12.7× |
| Standard deviation of the cosine between the state and the Yes-minus-No unembedding direction | 0.0032 | 0.0295 | 9.1× |
| Norm of the Yes-minus-No unembedding direction | 1.545 | 1.654 | 1.07× |
| State norm after the final normalisation | 142.6 | 226.5 | 1.59× |

Three readings follow. The bf16 quantization confound that the readout postmortem
suspected is refuted: with exact fp32 logits the 2B's trough gets *shallower*,
not deeper, and relative trough depth is affine-invariant, so no temperature or
rescaling fix can exist either. The de-quantization is exact, with a maximum
deviation from the stored value of 0.2525, which is one grid step, and it
reproduces the committed 8B trough depth of 0.4043 from the E7 pilot. The
dissociation is clean: the 2B holds the construct as linearly as the 8B and
ranks nearly as well, and what it lacks is angular commitment, with the 9.1-fold
gap in cosine spread accounting for essentially the whole 14.5-fold
dynamic-range gap while the unembedding geometry is effectively identical.

What remains before this can be claimed as emergence rather than as a two-point
artifact: a 4B checkpoint as a third scale point, a non-instruct control at the
same scale to test whether instruction tuning rather than scale creates the
rotation, and cross-corpus replication on the 2B states already on disk. The
jury's warning stands, that depth, tokenizer differences, and normalisation are
live alternative explanations and that layer comparisons must use normalised
depth. The estimated cost is about one GPU-hour. The practical corollary, if it
holds, is sharp for the moderation community: a small guardrail model can rank
almost as well as a large one and still be impossible to threshold without
labels, so small-model guardrail deployment inherits a hidden labelling cost.

**A second round-2 measurement, same caveat.** The corpus-mean displacement
caused by the project's own confirmed transcript-restoration intervention has
cosine +0.090 with the readout direction, so the restoration direction is
genuinely not the readout direction, and the intervention's asymmetry on that
paired set is +3.487 in the score. Read as a readout the corpus-mean version
fails: on the coarse ImpliHateVid task the restoration axis reaches 0.725
against the scalar's 0.951, and after residualising the scalar it falls to
0.458. The coarse task is not where the bottleneck lives, so this is a partial
negative rather than a closed door, but it was not promoted to a pilot.

**The ill-posedness proof and the attribution corpus.** These are the round's
most reusable non-method assets and they are already committed. Same corpus,
same scores, two label collapses, and the optimal label-free rule flips. Five
corpora have blind-audited strata on disk. Any future method must be argued
against them, and any claim that a corpus is "weak" now has to survive them.

**Reusable controls produced by the dead pilots.** The off-construct
spam-and-copyright policy is a working negative control for any policy-swap
experiment, and it separated two readouts where the pairing placebo alone would
not have. The layer-0 identity check, which must read exactly 0.500 because the
final prompt token is identical across arms, and the exact reproduction of the
frozen scores are a pair of cheap integrity tests worth keeping in later runs.
The answer-contrast pre-flight caught a transcript-configuration mismatch at
Spearman 0.827 and forced a full re-extraction of 3,197 videos before any clause
was computed, which is the gate working as designed.

**Timestamped ASR, a side-product with independent value.** The interleaved
kill test re-transcribed 45.0 audio-hours under the frozen Whisper
configuration to recover chunk timestamps. In 1,105 of 1,105 videos the
re-transcription reproduced the stored text byte for byte, so the timestamps
apply to the exact characters the judge reads, and 1,067 of the 1,105 took the
timestamped route rather than any fallback. Any future design needing a
temporal index over the judged transcript now has one, at no further cost.

## 7. The fork that needs an owner ruling

Section 5 leaves exactly one door in the R3 direction that has not been tried:
**obtain the readout direction from supervision that is not human and not
hate-specific**, for example a general-purpose language model labelling the
project's own unlabeled corpus once, offline, to define a direction that is then
applied by a frozen linear readout at inference. The owner has to rule on
whether this remains label-free. Both sides are stated here and neither is
adopted.

**The case that it stays label-free.** The charter bans human hate labels, not
supervision as such, and it explicitly permits general-purpose pretrained models
because they carry world knowledge rather than task-specific hate supervision.
Nothing about the target corpus is annotated by a person at any point. The
project already depends on such models for the judge itself and for the speech
transcription, so the line, if it exists, has already been crossed in the base
method. The information is provably present at 0.837 and every route that avoids
outside supervision has now been closed, so refusing this door is equivalent to
accepting that the scalar readout is the ceiling.

**The case that it does not.** A general-purpose model's notion of hate is
itself distilled from human annotation, so the supervision is human hate labels
laundered through a model rather than absent. The choice of teacher is a free
parameter with no principled answer, which is the same objection that killed
cross-dataset pooling under Anti-pattern 3, and a reviewer will ask what changes
under a different teacher. Choosing a teacher *because* it is good at hate is
the anti-pattern in its pure form. The honest framing would become "no human
labels at our end", which is weaker than the current claim and invites the
reader to compare against ordinary distillation baselines rather than against
label-free ones.

**What the ruling decides.** A yes reopens R3 with a concrete first experiment
and turns Cluster 3's ridge map from a demoted cost demonstration into an
admissible method component. A no closes R3 and leaves R1 (a second call with a
role not yet tried) and R5 (the scale phenomenon as the claim in its own right)
as the surviving relaxations, with the scale-emergence finding as the nearest
publishable object. The three preregistered deaths in this round do not depend
on the ruling either way.
