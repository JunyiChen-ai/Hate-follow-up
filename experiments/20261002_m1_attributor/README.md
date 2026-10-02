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

Gauss-Legendre quadrature starts at 16 nodes. If the completeness residual
`abs(sum A - (F(1)-F(0)))` exceeds `max(.25, .05*abs(F(1)-F(0)))`, recompute
at 32, then 64 nodes. These are numerical integration settings, identical
for both corpora, selected solely from model outputs. Retain every residual
and node count. At the cap, record failure; do not silently rescale A to force
completeness. A systematic numerical failure blocks performance conclusions
until repaired. Smoke compares 16/32/64 attribution stability without GT.

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
K is 16 normally, total 48 if retried at 32, or 112 if retried at 64. No per-window
model calls. Parameter gradients and prefix backprop are disabled, but query
backprop has real GPU/memory cost. Estimated normal reader 15–35 GPU minutes
for 333 videos on 5090; paired baseline adds approximately 10 minutes. Actual
smoke time/memory and per-corpus standalone estimates must be reported before
full launch; if convergence makes cost excessive, numerical feasibility is
resolved before launching, without looking at performance. Controls reuse
stored attribution except the deletion test, which adds two forwards/video.

All code/launchers local then scoped git sync; complete corpus on one machine;
return results before STATUS outcome updates. Log host, paths, model and date,
never file hashes or run-provenance commit IDs.
