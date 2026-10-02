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
