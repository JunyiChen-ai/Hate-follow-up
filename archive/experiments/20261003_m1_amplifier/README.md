Archived2026-10-03: complete main has no qualifying gain and HMM PR drops; no mechanism claim.

# M1 Amplifier: visual attention intervention with a language-only reference

Candidate11, declared2026-10-03 before implementation or performance inspection.
Independent of Integrator and archived visual contrast. Development-selected.
This declaration does not imply novelty approval or a measured gain.

## Hypothesis and source

Existing analysis found that a joint question often follows speech even when the
visual branch is positive (`experiments/20260928_infer/README.md` section13).
Hard attention restrictions and deleting media have failed in this iteration.
Test a different intervention: increase existing image attention without removing
any context, and discount answer-token preferences that survive without images.
Hypothesis: this improves visual-window ordering while preserving the native
speech/global readings. It need not succeed, especially for cross-modal meaning.

Source: Liu et al., PAI, ECCV2024, https://arxiv.org/pdf/2407.21771,
Eq3/4 and implementation details. Author code:
https://github.com/LALBJ/PAI/blob/master/attention.py and
https://github.com/LALBJ/PAI/blob/master/CFG.py .
The source adds alpha times the absolute image attention logit before softmax,
then contrasts multimodal and text-only token logits. This transfers those two
mechanisms, not a claim to invent attention amplification or CFG, nor an exact
reproduction of every author decoding rule. Author CFG.py also masks candidates
below0.1times the amplified maximum probability (APC). R1 explicitly omits APC
to preserve finite graded Yes/No evidence; this can change the source behavior
and must not be hidden as an equivalent implementation.
Paper Eq4 uses probability notation; author CFG.py actually contrasts log-softmax
values before APC. Our token-logit arithmetic matches that code's contrast term
up to a token-independent normalizer, not a literal mixture of probabilities.
Independent rule4 review must search target hateful-video use before coding.
Review PASS: `docs/reviews/20261003_m1_amplifier_proposal.md`.

## Exact R1 declaration

Qwen3-VL-8B BF16; original20frames, ASR, policy, global/window prompt text,8s
windows,4fps and unchanged r6. Same inputs/constants for both main corpora.
Native prefix, native global numeric margin and its selected answer are retained.
Every speech window uses its original native read. For each visual question only:

1. Reuse the native prefix plus global question/answer cache. In language layers
   with zero-based indices2 through35, change the LAST query token's attention
   logits to all prefix image-token keys: a <- a + .5*abs(a), before softmax.
   All heads, same existing image attention direction, no head selection or
   temporal restriction. Earlier query rows and the reused native prefix/global-QA
   cache stay unchanged; the last query token's later hidden states and KV change.
   Only the local question's final token is intervened on, consistent
   with the source's token readout. Do not edit the vision encoder or mRoPE.
2. Build one language-only prefix via Judge.prefix_messages with_frames=False,
   preserving all ASR, policy and system text. This also omits image timestamps
   and uses the existing no-frame introductory wording; it is a language-only
   reference, not an isolated pixel intervention. Ask the original global
   question, force the ORIGINAL native answer, then ask the same visual questions.
   No attention amplification in this branch. Its own global answer is diagnostic.
3. Combine per-token answer logits as 1.1*amplified - .1*text_only, then apply the
   original Yes/No token-set logsumexp margin. Contrast BEFORE class aggregation;
   do not subtract already aggregated margins. Vocabulary normalization constants
   cancel in the final margin, permitting exact answer-token subset computation.

Alpha=.5 comes from the paper's linear-projector LLaVA setting; gamma=1.1 is
the source setting across models. Skip the first2layers as in the author command,
extending the end to Qwen's36layers. These are declared transfer assumptions,
not Qwen-optimal constants; no scan. No noise, new model, frame extraction,
learned parameter, temperature, threshold, routing or decoder modification.
The null branch is the SAME frozen model with different input, not an ensemble.

Native full-cache visual states can already contain earlier images, and speech
queries still have full context. This does not establish isolated modalities or
semantic cross-modal fusion. Boost all20frames, not only the queried interval;
any localization claim must be demonstrated by resulting window order.

## Controls, gate and falsification

Complete333 paired arms: native; matched attention implementation with alpha0
and no token contrast; PAI main. The matched arm controls eager/SDPA numerical
differences; apply promotion against native/current r6 AND that control.
Record amplified-only and text-only answer-token logits for later ablations.
The intervention uses identical precision/attention arithmetic in alpha0 and.5.
Validate actual image positions, query row/layer scope, GQA, mask, KV crop,
native restoration and actual model-forward counts, not only algebraic formulas.

Main gate: within+.01 both corpora, no pooled loss>.005 or within loss>.01,
canonical evaluator, all3main metrics. Any final main gain>=.01 allows revision
under rule9; otherwise archive. Do not select an ablation as a new main silently.
If main qualifies, evaluate cached removal controls: attention only (gamma1),
contrast only (unamplified matched token logits), and neither. Each component
claimed as novelty must cost>=.01 on a common main metric in both corpora.

Report raw combined and branch within (available frames), final within and pooled
metrics, single-window contributions, native-verdict strata, bootstrap uncertainty,
and positive/negative examples. A gain only from video shifts is insufficient.
If attention is necessary, an additional matched wrong-image-support intervention
is required before claiming temporal grounding (its exact support will be declared
before that control runs). It must change actual media/support: permuting the
members of the set of ALL image keys does not change the enhanced set and is
not a valid intervention. No use of test GT in any scoring/fitting path.

## Cost and run plan

Native deployment3+B outer forwards, B=V+S. Main deployment6+B+V: native
prefix/global/answer, amplified V queries and native S queries, language-only
prefix/global/forced-answer and V reference queries. No repeated video prefill per
window. Paired3-arm collection6+B+3V (native/eager0/amplified visual queries plus
text-only queries; native speech shared), smoke restoration adds3+B.
Reuse all current derived inputs and model weights. Dense last-row attention can
increase branch time; full333 estimate20–35 GPUmin paired on5090, to be replaced
by five-video smoke actual cost. Sequential prefix caches; release visual cache
before language-only prefill and save token logits on CPU. No new preprocessing.

First2manifest videos/corpus plus HMMhate_video_114 for plumbing only, no GT.
Then one complete run per corpus, canonical raw/r6 metrics, shared branch report.
Host selected by live preflight; run.log records host. No GPU job before independent
proposal and code review. Current method and paper remain unchanged.

## Implementation checks

Implemented2026-10-03. `amplifier.py` wraps the installed SDPA attention interface
only during visual suffix reads. Other rows use the original SDPA output; the
last row uses explicit GQA attention with common arithmetic for alpha0/.5.
This avoids copying model-specific Q/K norms, RoPE or cache-update code. The
reused prefix/global-QA cache is cropped after every question, and the native
speech/global output is copied exactly. Language-only reference uses a fresh
cache after the visual cache is released. No cross-experiment imports.

`selfcheck.py` passes actual small Qwen text-model FP32/BF16, independent
GQA row algebra, last-row-only change, cache preservation, native restoration,
unchanged weights and token-normalizer cancellation. Output
`runs/20261003_m1_amplifier/selfcheck/checks.json`; not a corpus performance test.
Read-only inspection of lab2 transformers5.15.1 confirms the same attention
interface accepts the wrapper, while the local witness uses4.57.6. GPU execution
is still needed to verify that deployment path and exact historical baseline.

Independent rule6 review PASS: `docs/reviews/20261003_m1_amplifier_code.md`.
Actual multimodal36-layer Qwen FP32/BF16 checks all34intervened layers, GQA,
mRoPE/DeepStack, query/cache isolation, native restoration, the complete reader,
native speech/global/answer, image-free reference and actual forwards. Synthetic
prepare/report and canonical r6 commands pass; report now rejects nonfinite
decoded scores or changed global values before loading GT. Artifacts:
`runs/20261003_m1_amplifier/independent_review/check_amplifier.{py,json}`.
No real GT or corpus performance was inspected. GPU smoke target sc474399
(lab2) if fresh preflight confirms availability, conda HateVLM as in launch.

Launch `bash experiments/20261003_m1_amplifier/launch/run_lab.sh smoke`, then
`main` after plumbing PASS. Canonical evaluation
`bash experiments/20261003_m1_amplifier/launch/run_analysis.sh`; shared branch
diagnostic uses `scripts/analysis/m1_branch_diagnostics.py --arms base eager pai`.

## GPU smoke PASS, proceeding to full333

Host sc474399, smoke artifacts returned locally:
`runs/20261003_m1_amplifier/r1_smoke/plumbing_summary.json`. Five native globals
and292branches exactly match base_gridA; restoration, original speech/global/
answer, identical local queries, no-image reference, actual counts,34-layer visits,
token-first contrast reconstruction and finite4fps pass. Peak17.8242GiB;
matched eager visual drift at most.28363. Mean visual attention per input changes
from.0213–.0322 to.3569–.3815; this is activation by construction, not accuracy.
Smoke43.5s includes native restoration, no GT/performance read, constants unchanged.
First2/corpus extrapolation: native182.0/237.7s, PAI251.5/368.6s,
paired365.4/613.1s for HMM/HCS (standalone10.33min, paired16.31min).
These small-input estimates exclude the stress video; full actual time will replace
them. Full333 paired run follows on the same live-idle host after fresh preflight.

## Full333 result: archive, no qualifying main gain

Host sc474399; all results returned locally before reporting. All333 native
whole-video reads and13939 native branches exactly match base_gridA. Canonical
metrics `runs/20261003_m1_amplifier/r1_main_decoded/<arm>/metrics.json`:

| Corpus | arm | ROC | PR | within | raw within |
|---|---|---:|---:|---:|---:|
| HateMM | base | .897119 | .694235 | .750782 | .680011 |
| HateMM | eager | .897129 | .694243 | .750915 | .679839 |
| HateMM | pai | .893032 | .685114 | .756861 | .682000 |
| HateClipSeg | base | .716825 | .671072 | .637349 | .610130 |
| HateClipSeg | eager | .716872 | .671087 | .637452 | .609303 |
| HateClipSeg | pai | .712161 | .669913 | .628277 | .607175 |

Within+.006079 (84,95%CI[-.010663,.028046]) / -.009072
(99,[-.023444,.004604]); HMM PR-.009120. No final main metric gains>=.01;
matched eager changes remain small. Raw visual ordering-.015198/+.004561,
max-combined+.001988/-.002955; native speech exactly unchanged, eligible82/97.
Sources `r1_main_analysis/{alignment,summary,branch_diagnostics,cost}.json`.
Post-scoring analysis read complete raw/decoded predictions and test GT arrays;
no media/ASR semantic inspection, no constants changed. More image attention by
construction is not evidence that it attended to useful evidence.

Measured standalone native302.45/244.80s, eager340.57/280.11s, pai451.78/375.65s
(HMM/HCS); pai total13.79min,1.512x native; paired wall22.23min,
peak17.8238GiB. PAI new-video calls6+B+V vs native3+B; no preprocessing added.
Archive per rule9, no component controls or coefficient/layer search after the
failed main gate. This is tenth cumulative archive; source implementation remains
for reproducibility and all negative results are retained. Development-selected.
