Archived2026-10-03: no final main metric improved by .01; raw visual ordering worsened on both corpora.

# M1 Preserver: preserve visual question representations during context integration

Candidate17, declared2026-10-03 before implementation or performance. Frozen
Qwen3-VL-8B, existing frames/ASR/policy/questions, 8s windows and4fps; unchanged
r6 and canonical evaluator. Independent of Highlighter and Stabilizer, never
combined with them. All eventual comparisons are development-selected.

## Hypothesis and source

Window questions need context, but their internal representations may be dominated
by the transcript and the preceding global answer. Removing context altogether
has already failed. Here each visual question is first read with frames and policy
alone; its layer-wise attention outputs then enter the same question's normal
context-conditioned read. This tests preservation of image-conditioned evidence
while retaining contextual reasoning, rather than asserting that all context is
harmful or that the original MLLM cannot reason jointly.

Source: Zhao et al., *When RAG Hurts: Diagnosing and Mitigating Attention
Distraction in Retrieval-Augmented LVLMs*, ICML2026,
https://proceedings.mlr.press/v306/zhao26g.html and
https://github.com/ubc-tea/MAD-RAG . Read full method5.1–5.2, experiment6.1 and
official Qwen25vl/run_mad_rag.py. Source snapshots are read-only under
third_party/mad_rag_source_read/. No source datasets/oracle chunks were downloaded.

MAD-RAG places identical questions before and after retrieved text and mixes their
attention outputs at every layer, alpha=.5. Its Eq3 selects image-only attention,
whereas Eq5 and the official Qwen wrapper mix complete attention outputs, which
also contain preceding instructions/question tokens. They are not generally
equivalent. We explicitly transfer the code's complete-output mechanism, not the
image-only equation or a claim of pure visual semantics.

Our adaptation uses two reusable caches instead of inserting every window question
into the global prefix: one frames/policy reference, one unchanged full context.
The full-context cache does not contain reference-question tokens. This is not
numerically equivalent to the source's dual-question input. We also mix the full
identical query suffix, including chat headers, rather than only the bare question
span. These changes preserve the original global and speech computations and
avoid re-encoding images separately for every window. They must be tested here;
source task results do not establish our mechanism or low cost.

## Exact R1 computation, fixed before scoring

1. Construct the ordinary native prefix with all cached frames, full transcript
   and policy. Read native global question/answer and native speech branches as
   usual. Record exact baseline visual branches for paired evaluation.
2. Construct one separate reference prefix with the SAME cached frames/timestamps
   and policy, using existing prefix_messages(with_context=False). It has no ASR,
   global question or global answer. Reset and save its own native mRoPE deltas.
   Use actual frame counts:20 is the budget, one existing HCS cache has18frames.
3. For each visual window, use EXACTLY the native full-context visual query suffix
   token IDs on the reference prefix. Capture self-attention outputs after o_proj
   for all query rows at all36language layers. This reference read has no attention
   intervention. Crop its KV back after each window; do not append answers.
4. Read those same token IDs on the full native context. At every language layer,
   replace its query attention output O by .5*O+.5*O_reference, computed inFP32
   then cast to the originalBF16. Residual/MLP computations remain native and
   receive this output. All36layers/allqueryrows are fixed, no selection or scan.
   The reference is read-only and is not recursively mixed. Restore the native
   full-context mRoPE state before this call and crop its KV afterwards.
5. Read final Yes/No margin with the unchanged12-token FP32 native head. Combine
   visual and original speech with the existing maximum; keep native global
   numeric score and answer. Feed the complete reader arm to unchanged r6.

No new encoder/model, output voting, learned controller, labels, per-corpus
constants or confidence routing. One shared frozen MLLM supplies both internal
views; no combination of independently parameterized models. No other candidate
intervention or earlier best prediction is included. Reference-only Yes/No values
may be saved as a predeclared diagnostic, never used to select a branch.

## Cost and integrity checks

Let V be visual windows, B=V+available speech. Native deployment3+B forwards;
Preserver4+B+V (one extra prefix and one extra reference query per window).
Paired collection4+B+2V; smoke adds3+B+V for fresh native restoration and all-window alpha0 checks. Two prefix
KV caches coexist; capture only one window's36attention-output arrays at a time.
All reference/prefix/mixing/head costs count toward deployment, including both
image encodes. Initial estimate15–25GPUmin deployed,20–35min paired on5090 for
333videos; replace with five-video smoke before full run. No precomputed new-video
reference is treated as free. Reuse only original frame/ASR/weight assets.

Independent code review must test actual36layer Qwen,32/8heads,FP32/BF16;
post-o_proj tensor oracle; all query rows vs untouched cached prefix; alpha0 exact
native; identity-reference injection equality within declared cast arithmetic;
per-prefix rotary restoration; exact query token alignment; no cross-window cache
contamination; missing speech/18frames; active mechanism in final margins;
forward counts,4fps alignment,no GT path and canonical evaluation. GPU smoke is
first2manifest videos per corpus plus HateMM hate_video_114, without GT screening.
Full native333 must reproduce base_gridA before performance conclusions.

## Falsifiable decision and controls

Same main gate: final within+.01 on BOTH main corpora against paired native and
current r6; no pooled loss>.005 or within loss>.01. No final metric gain>=.01
means archive; a qualifying single-corpus gain permits at most3declared revisions.

If main qualifies, run fixed controls:
- Remove mixing (alpha0), keeping the same execution. This must reproduce native.
- Reference without images: policy-only reference, same question token IDs, same
  all-layer .5 mixing. Tests whether gains require image-conditioned reference.
  It also changes prefix length and native mRoPE positions; differences support
  the reference-input package, not image semantics isolated from position effects.
- Temporally mismatched reference: retain chronological timestamp slots but move
  reference images by a circular shift of ceil(N/2), N=actual frame count. Same
  native target context/questions and .5 mixing. This perturbs both image order
  and timestamp association, so it cannot isolate only timestamp binding. Report
  windows for which frame support actually changes, and single-window separately.
- Reference-only diagnostic: retain the native global and speech, use reference
  visual scores. Tests the need for context integration rather than full removal.

Removing a claimed mechanism must cost>=.01 in a common main metric on BOTH
corpora. Main vs native alone cannot establish visual preservation. Report raw
visual/speech/max orderings, final three metrics, changes confined to videos with
unchanged raw ordering, single-window effects, paired video bootstrap2000seed0,
representative gain/loss content cases and complete cost. No causal attribution
solely from attention magnitude, an activation difference or source-paper claims.

Prior evidence read: iteration14negative candidates, verdict analysis, original
prefix/query code and MAD-RAG primary sources. Highlighter scoring is complete
but final performance was not read when declaring this candidate. No new GT,
input extraction or label-based fitting for this proposal. Implementation waits
for independent rule4 review; full scoring waits for independent rule6 review.


Independent proposal review PASS: `docs/reviews/20261003_m1_preserver_proposal.md`.
Implementation started after PASS, with the policy-only position/length caveat
retained above. Reference-only values are saved as diagnostics; main evaluation
only compares base/preserve. Smoke explicitly adds V alpha0 queries in addition
to the declared fresh native pass; no full-run constants or scoring change.


Implementation2026-10-03 complete. Selfcheck PASS for actual36-layer Qwen with
32/8heads,128attention dimensions, native mRoPE[24,20,20], FP32/BF16 and18/20images:
`runs/20261003_m1_preserver/selfcheck/numerics.json`. Both prefix caches restore
exactly, alpha0 and native replay exact, identity reference exact, nonzero mixing
reaches final logits; reference tensors remain read-only. This is a random-weight
CPU witness, not an8B performance result. Independent code review requested.
Intended target sc448960 via `launch/lab.sbatch smoke`, then main after validation;
Slurm default/pilot partition is not used. Reuse its reviewed HateVLM environment.


Independent code review PASS2026-10-03:
`docs/reviews/20261003_m1_preserver_code.md`. Actual36-layer Qwen retains32/8heads,
head_dim128 and native mRoPE on torch2.14/transformers5.16.1 CPU; both dtypes,
18/20images, all-suffix post-o_proj oracle, shared native12-token FP32 head,
alpha0/fresh restore, separate real positions/caches, complete reader call counts
and333 synthetic canonical report pass. Target5.15.1 position implementation was
also read independently; implementer target5.15.1 selfcheck passed and returned
as `runs/20261003_m1_preserver/selfcheck_lab_server/numerics.json`.
GPU5video smoke remains required before any full performance result.

GPU smoke completed on sc448960 through Slurm job54,2026-10-03. All5videos,
292native branches and5globals reproduce the historical native input/readout
exactly; all-window alpha0 and fresh native replay pass. No GT was used.
Source: `runs/20261003_m1_preserver/r1_smoke/plumbing_summary.json`, checks.jsonl
and run.log. Total smoke45.4s (including extra checks), peak17.824GiB.
The first2videos/corpus extrapolate deployed Preserver332.1s/417.7s
(HateMM/HateClipSeg), paired385.4s/536.1s, about15.36min paired total.
These are small-sample scheduling estimates, replacing the initial estimate;
full-run measured costs will be reported. No constants or input rules changed
after smoke. Full333 runs on the same sc448960 Slurm local partition.


## Full R1 result and disposition (2026-10-03)

Host sc448960, Slurm job55,333videos completed and returned locally before CPU
analysis. Canonical base reproduces current r6; all333native globals and13939
branches reproduce historical base_gridA exactly. Source:
`runs/20261003_m1_preserver/r1_main_decoded/preserve/metrics.json`.

| Corpus | pooled ROC | pooled PR | within | within delta | eligible videos |
|---|---:|---:|---:|---:|---:|
| HateMM | .895609545 | .689650711 | .752657821 | +.001875778 | 84 |
| HateClipSeg | .715188724 | .672281752 | .629401070 | -.007948044 | 99 |

No final main metric improved by.01; archive17 under rule9, no mixture-weight
revision or component controls. All main differences fall within the declared
noise bounds, including HMM PR-.004584, but this supplies no positive mechanism.
Paired final within95%CI HMM[-.016519,.019880],HCS[-.030946,.016416],2000draws seed0.
Raw visual within falls .613908 to .547162 and .546450 to .512565. Raw max within
falls .680011 to .655015 and .610130 to .574752; paired95%CI for those raw-max
changes [-.045666,-.010172]/[-.054857,-.017753]. Speech and global are unchanged.
Thus preserving this reference attention output did not preserve useful visual
localization in this adaptation; the small final HMM shift is not new evidence.
No theorem about general visual preservation follows from the source or activation
mix. This negative result does not by itself identify which layer/context component
causes the loss. Analysis: `runs/20261003_m1_preserver/r1_main_analysis/` summary,
branch_diagnostics, per_video, cost and table files.

Measured deployed cost513.90s HMM/420.58s HCS, total15.57min; native329.00/269.21s,
9.97min,1.562x. Preserver4+B+V vs native3+B calls; both prefix encodes and all
reference visual queries included. Full paired wall1181.4s; peak17.834/17.700GiB.
No source frame extraction cost was added; all native caches were reused.

Development-selected files read: full paired reader/decoded predictions and
canonical metrics, original `data/gt_4fps/{HateMM,HateClipSeg}.npz`, summary and
branch diagnostics above. Labels only enter completed-score evaluation/analysis,
not scoring, fitting or constants. Decision: archive rather than tune alpha/layers.
Explorer was independently proposed/reviewed and its first primitives implemented
before these results were read. Current r6/paper/Overleaf remain unchanged.
