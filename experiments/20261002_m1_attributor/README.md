# M1 Attributor: temporal attribution of a joint decision

Date 2026-10-02. Status: proposal PASS; implementation/numeric checks underway.
Independent review: `docs/reviews/20261002_m1_attributor_proposal.md`.
Planned host uoa-lab2 / sc474399, subject to live availability.
Third candidate of `experiments/20261002_m1_iteration/README.md`.

## Hypothesis and antecedents

Grounder and Selector physically restricted query access but did not improve
localization. Their official raw/final metrics and post-scoring analyses were
read after completion: `runs/20261002_m1_grounder/r1_full_analysis/` and
`runs/20261002_m1_selector/r1_main_analysis/`. Their test labels were used only
in analysis. No additional GT is used to design or fit this reader.

Prior repository failure is explicitly acknowledged:
`archive/detection-2026-08/docs/duplex/PREREG_temporal_attribution_pilot.md`
and `TEMPORAL_ATTRIBUTION_PILOT_NOTE.md` (same directory) tested absolute
embedding gradient-times-input and late attention under the old input/protocol.
The results tracked token density. This candidate instead uses signed shared
cached-value gates and a path integral under current inputs; it may fail for
the same reason. These old segment-level numbers are not comparable to current
4-fps metrics. The new method must beat a token-density control to claim that
semantic contribution, rather than input quantity, explains localization gains.

Hypothesis: the model's joint whole-video Yes/No decision can supply temporal
evidence through signed internal attribution, without repeatedly asking the
model to label each window or removing context during prefix encoding.
This is an application of integrated gradients to cached media-value paths,
not an invention of integrated gradients or proof of raw-input causality.
Source: Sundararajan et al., https://arxiv.org/abs/1703.01365 .
Related transformer attribution: Chefer et al., CVPR 2021,
https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf .
Independent review must search actual hateful-video attribution antecedents,
including https://doi.org/10.1145/3774905.3796488 (publisher access failed for
the authoring agent; title: An Interpretable Agentic Framework for Multimodal
Hate Video Analysis with Explicit Evidence Attribution).

## Fixed computation

Same frozen Qwen3-VL-8B, existing 20 frames, repaired transcript, policy,
`VIDEO_QUESTION`, 8-second windows and canonical r6 downstream algorithm.
Encode the ordinary full multimodal prefix once; all prefix K/V are detached.
Compute the unchanged global Yes-minus-No log-sum-exp margin `z_video`.
No global answer is appended for attribution; it directly explains this margin.

At every language attention layer, multiply each cached media token's value
vector by a scalar gate g_i. The same gate for token i is shared across layers
and KV heads. Keys, non-media prefix values, and the global question's own
values are unchanged. Media spans are exactly `src/window_token_regions.py`
(timestamped frame blocks and transcript, including transcript timestamps).
Policy/scaffolding values remain fully available throughout. At g=1 this is
the native decision. At g=0 media values are zero: this is a zero-value
reference, NOT removal of frames/transcript, and contextualized keys and
scaffolding can still carry their information.

For F(g) equal to that global margin, compute signed attribution
`A_i = integral_0^1 partial F(alpha * 1) / partial g_i d alpha`.
Autograd differentiates only gates; weights never update. Query states may
depend on gates and are differentiated normally. Prefix encoding is not
differentiated. This attributes direct cached-media-value paths under a
specified baseline; it is not a full decomposition of semantic evidence.

Gauss-Legendre grids are 16, 32, 64, 128 and 256 nodes. Accept the first grid
whose completeness residual `abs(sum A - (F(1)-F(0)))` is at most
`max(.25, .05*abs(F(1)-F(0)))` AND whose full signed token vector differs
from the previous grid by relative L1 <= .05 (`L1(delta)/max(L1(A),1e-12)`).
Thus the earliest possible acceptance is 32, requiring 16+32 gradient reads.
These numerical settings are identical for both corpora and use no GT.
Retain every residual and node count. At the cap, record failure; do not
silently rescale A to force completeness or exclude a failed video. Numerical
failure blocks performance conclusions until repaired. Smoke evaluates all grids
and reports per-window values/rank stability against the 256-node reference.

Numerical-only revision before any performance read: the first smoke
`runs/20261002_m1_attributor/r1_smoke/` tested 16/32/64. Native endpoints and
cache invariance were exact, peak memory 19.96 GiB. Three videos had token
L1 < .017 and window Spearman >= .9995 versus 64; `non_hate_video_4` needed
64 for completeness but had 16/64 rank .90 and 32/64 rank .70. Therefore
completeness alone was insufficient and the vector-convergence guard plus
128/256 refinements were added. Original smoke outputs are retained unchanged.
No localization metric or GT was read in making this numerical revision.

Map signed contributions into visual/speech windows using exact token spans.
When one token belongs to multiple windows, split its contribution equally
among those memberships. Tokens with no temporal support remain an explicitly
reported unattributed remainder. For modality m/window w, set its readout to
`N_windows * sum_i membership(w,i) * A_i` over that modality. Multiplication
by N converts conserved contribution into average-window evidence density;
it is common to all windows, preserves within-video order, and is not tuned.
Visual windows without sampled frames score zero. Omit a speech readout when
the original window transcript is empty, as in base; any resulting remainder
is recorded. Raw window curve remains the max of observed modality readouts.
Feed these readouts and unchanged z_video to the same r6 algorithm, refitting
its existing unsupervised distributions without labels. No other score blend.

## Sequence, controls, gates

1. Independent proposal review under rule 4, then implementation and independent
   code review under rule 6. Seed 0, same precision/environment as current reader.
2. First two manifest videos per corpus: numeric and cost smoke only. Verify
   native/g=1 equality, finite gate gradients, cache immutability, temporal
   contribution conservation and endpoint/path stability. No GT or AUC pilot.
3. Complete base/Attributor paired run on both corpora. Reproduce every baseline
   readout against current cache. Use only the canonical evaluator, all three
   metrics, raw and r6; paired within bootstrap (seed 0, 2000 resamples).
4. If no final metric gains .01 anywhere, archive. If qualifying gain appears,
   run the declared controls: endpoint gradient (no path integration), absolute
   contribution (drops sign), half-rotated media-token attribution within each
   modality (wrong time), and base shifted by the attribution-minus-base per-video
   mean max-readout (same base raw ranking). Also run a token-density arm that
   sets each media-token contribution to one, with exactly the same temporal
   membership and N-window scaling. These are separate diagnostic arms.
5. For supported improvement require within +.01 both corpora and no main metric
   deterioration beyond .005 pooled/.01 within. Every claimed novel component
   must meet rule 14g on both corpora; quadrature accuracy can remain an
   implementation detail if endpoint gradients are equivalent.
6. If positive, intervene on cached values: zero the highest-scoring window's
   media versus its half-video-offset window, retaining original keys/context.
   Compare actual global-margin drops, signed attribution, and localization
   cases (correct/wrong global decision, short/diffuse positives, missing-frame
   windows). Report failures and do not confuse model faithfulness with GT
   correctness or explain improvements solely by global mean-score shifts.

All results are development-selected; original method/paper remain unchanged.
No performance-based threshold, label fitting, extra dataset or ensemble.

## Cost

Reuse frames/ASR/model and one prefix encoding per video; no new preprocessing.
Deployment replaces the original 3+B forwards (B observed modality branches)
with 1 prefix + 2 endpoint forwards (one also backward, stored as endpoint
control) + K global-query forwards/backwards, where
K is at least 48 (16+32), then 112, 240 or 496 on further refinement. No per-window
model calls. Parameter gradients and prefix backprop are disabled, but query
backprop has real GPU/memory cost. Initial estimate before numerical smoke was 15–35 GPU minutes
for 333 videos on 5090; paired baseline adds approximately 10 minutes. Actual
smoke time/memory and per-corpus standalone estimates must be reported before
full launch; if convergence makes cost excessive, numerical feasibility is
resolved before launching, without looking at performance. Controls reuse
stored attribution except the deletion test, which adds two forwards/video.

All code/launchers local then scoped git sync; complete corpus on one machine;
return results before STATUS outcome updates. Log host, paths, model and date,
never file hashes or run-provenance commit IDs.

## Refined numerical smoke, before performance (2026-10-02)

Host sc474399; `runs/20261002_m1_attributor/r1_smoke_refined/`, four manifest
videos, all outputs local. Baseline global scores and all 136 window margins
match current reader exactly; g=1/native and prefix-cache comparisons are exact.
Frozen-model gradient plumbing also passed independent small-Qwen CPU tests:
FP32 gate finite-difference error 7.36e-6; 32-node completeness 3.42e-7;
BF16 gradients finite, non-media gradients zero, model parameter grads absent.
Independent record: `docs/reviews/20261002_m1_attributor_code.md`.

Three videos accept at 32 (48 total quadrature calls); `non_hate_video_4`
accepts at 256 (496 calls): residual .3018 against tolerance .4180 and
successive-vector relative L1 .0270. For the first three, accepted versus
256-node window Spearman is 1 / 1 / .999656 and token relative L1 < .0092.
For the difficult video, 64/256 and 128/256 rank correlations are .90;
it passes the declared value-convergence test but small rank swaps remain.
No claim of exact BF16 calculus or raw-input causal attribution is made.

Measured deployed-path (prefix included) cost is 5.80 times native on these
four videos, with 19.96 GiB maximum allocated memory. Actual smoke runs all
496 nodes even on early-accept videos and took 104.7 seconds total; use
`numeric.deployed_seconds`, not smoke wall time, to estimate deployment.
Full-pair estimate revised to roughly 45–75 GPU minutes, highly sensitive to
how many videos need refinement. All calls/retries are included in the cost.
No GT or localization metric was read. Full scoring stops if any video fails
the fixed cap instead of silently excluding it or evaluating unreliable scores.

## Numerical stop and repair investigation (before performance, 2026-10-03)

The attempted `r1_main` run stopped at manifest video 26, `HateMM/non_hate_video_82`,
after saving diagnostics and 25 complete pairs. At 256 the completeness residual
.2853 met tolerance .3091, but relative token L1 was .1888; no reliable acceptance.
All outputs returned locally; no metrics or GT were read. This is not a method
performance result and this subset will not be evaluated.

Investigate adaptive integration on this numerical failure only (not a GT-chosen
case): SciPy quad_vec, Gauss-Kronrod 21, L1 vector norm, epsabs .05, epsrel .025,
limit 32 intervals, one worker. Record sampled alpha, margin, gradient sum/L1,
local interval errors, estimated total error and endpoint completeness. This is
an integration feasibility diagnostic, not an alternate selected performance arm.
No normalization of contributions to force completeness. The full method will
be redeclared numerically only after these checks, then independently confirmed.


Adaptive BF16 diagnostic also failed its accuracy target: 2457 gradient reads,
115.62 seconds, estimated L1 integration error 2.872; scalar completeness residual
.1094 passes but does not rescue vector uncertainty. Local output
`runs/20261002_m1_attributor/numeric_adaptive/checks.jsonl`. No performance read.

Next diagnostic promotes only language query computations to FP32, retaining
exactly the original frozen BF16 parameter values (cast, no updates) and native
BF16 prefix/cache. Temporarily offload unused vision/embedding/LM-head weights,
then restore BF16 for the next native prefix/baseline. Compare gated g=1 to an
ungated FP32 query reference and separately record its drift from native BF16.
This explains a higher-precision cached-query counterpart, not an exactly equal
BF16 margin. Test the same failure video with declared 16/32/64/128/256 grids;
no GPU performance experiment resumes until numerical reliability is established.
Extra ungated reference read and precision-transfer time are diagnostic overhead
and must be reported separately if this becomes the deployed numeric pathway.

If FP32 query arithmetic becomes necessary and the candidate improves metrics,
add an FP32-window control: original window questions, native BF16 encoded
prefix/global-answer cache and z_video, but the same FP32 language query math.
This distinguishes attribution gains from arithmetic precision alone. No claim
of attribution-driven improvement may be based solely on outperforming a lower
precision window reader. This control is declared before any outcome inspection.


FP32 failure-case diagnostic succeeded on lab2:
`runs/20261002_m1_attributor/numeric_fp32/checks.jsonl`.
Acceptance at 64 (112 accumulated quadrature pairs); residual .08038, adjacent
vector relative L1 .04018; 256-node residual .00948. All 16/32/64/128 versus
256 window Spearman values are 1. Native FP32/g=1 difference is zero; the
FP32-minus-BF16 endpoint drift is −.01274. Restored BF16 global margin is exact,
prefix cache immutable, peak allocated memory 30.07 GiB. Full all-grid smoke
49.7 seconds; accepted integration 10.91 seconds before transfers/reference.
No performance inspected. Next memory check chooses `HateMM/hate_video_114`
solely because it has the largest existing prefix (5829 tokens) among all333
in `runs/20261002_m1_grounder/r1_full/base/predictions.jsonl`; no GT used.
Command: `launch/run_numeric_fp32.sh hate_video_114 numeric_fp32_max`.


Largest-prefix FP32 smoke exceeded the 32GB card during gradient attention
(`runs/20261002_m1_attributor/numeric_fp32_max/launch.out`). It also had a
GPU cache snapshot used only for validation. Remove that validation allocation
by snapshotting on CPU, and use decoder activation checkpointing for FP32
queries. Prefix cache is read-only: a functional DynamicCache update returns
prefix+query K/V without storing query values. This makes decoder recomputation
idempotent; query positions still derive from the unchanged prefix length.
The same gate remains active through backward/recomputation. Verify checkpointed
margin/gradient equality and native-prefix invariance independently. Count the
recomputed decoder layers and report their time; this is a memory optimization,
not a new scoring mechanism or a reason to claim fewer computations.
Repeat longest-prefix numeric smoke as `numeric_fp32_max_checkpoint`.


### Current numerical execution contract (before full FP32 results)

The mathematical shared-value-gate attribution, token mapping, scoring and
quadrature acceptance rule are unchanged. The BF16 integration implementation
is superseded by FP32 language queries with decoder activation recomputation;
the model's learned parameter values and native prefix encoder remain unchanged.
All 333 videos will restart in `r1_main_fp32`; the partial BF16 run is not reused.
The original BF16 z_video remains the downstream global score, requiring an
additional native global-query forward in FP32 attribution deployment.
Forward count is therefore 1 prefix + 1 BF16 global + 2 FP32 endpoints + K
FP32 quadrature queries; backward count 1+K, each with decoder recomputation.
Standalone seconds include native global read and precision transfer/recomputation,
excluding the separately counted ungated FP32 diagnostic reference. Smoke also
counts an extra native restoration check. Actual paired wall time is reported.

After maximum-prefix memory/equivalence passes, run `launch/run_lab2.sh main_fp32`
and return all outputs. `launch/run_analysis.sh` evaluates only the new complete
`r1_main_fp32`, writing `r1_main_fp32_decoded` and `r1_main_fp32_analysis`.
A numerical failure still stops the complete experiment; no subset result.


Maximum-prefix checkpointed FP32 smoke passed:
`runs/20261002_m1_attributor/numeric_fp32_max_checkpoint/checks.jsonl`.
Peak 28.48 GiB; acceptance at 32; residual .007024, adjacent token L1 .001198;
accepted versus 256 token L1 .001110, window Spearman .997411 and max window
difference .04044. At 256 residual .0000251. Native FP32/g=1 and restored BF16
checks exact; immutable prefix verified. Actual 497 backwards recomputed
17,892 language layers (=497*36). Accepted deployment is 52 outer forwards
including prefix/native global, 49 backwards with 1,764 layer recomputations.
Measured standalone prefix + attribution is 15.51 seconds for this longest
input, versus 7.01 seconds native. Full all-grid smoke took 147.7 seconds;
this is not deployment latency. Independent small-Qwen tests show ordinary,
read-only cache and checkpointed gate gradients/margins match exactly in
BF16/FP32 at alpha 0/.37/1. Full-pair estimate remains roughly 45–90 minutes,
subject to the observed refinement distribution. Proceed with complete FP32 pair.


2026-10-03 continuous-run memory repair, before any performance evaluation:
`r1_main_fp32` stopped before video25 attribution, while promoting decoder
weights: allocated26.35GiB plus4.06GiB reserved/unallocated;192MiB allocation
failed. First24 numerical checks passed. All outputs returned locally; no GT
or partial metrics read. Fix the storage transition by moving native decoder
blocks/norm to CPU before allocating FP32 CUDA weights, and enable expandable
allocator segments. Values, precision, gradients, grids and scoring unchanged.
Transfers remain included in measured deployment time. Start all333 anew in
`r1_main_fp32_mem`; analysis now targets only that complete run. Preserve both
partial runs as diagnostics; do not merge them into the final experiment.

Prepared (not performance-evaluated) CPU falsification arms in `controls.py`:
endpoint gradient, absolute contribution, within-modality half-token rotation,
media density, and a common per-video shift preserving native raw max order.
Generation requires a qualifying complete primary result. A four-video saved
smoke check verified attribution-to-window reproduction, unchanged global and
modality availability, finite output, and shift centering/rank invariance;
`runs/20261002_m1_attributor/controls_selfcheck/checks.json`. No labels read.
Independent targeted control review and canonical evaluation are still pending.
