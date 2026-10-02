# M1 Recycler: content-dependent recycling of attention from sink tokens

Candidate12, declared2026-10-03 while Integrator R2 and Amplifier R1 run;
their full metrics have not been inspected. No implementation/GPU run yet.
Independent candidate, no combination with either running method.
Development-selected, frozen Qwen3-VL-8B, unchanged r6 and canonical4fps evaluator.

## Hypothesis and source

VAR, Kang et al., ICLR2025, *See What You Are Told: Visual Attention Sink in
Large Multimodal Models*, https://arxiv.org/html/2503.03321v1, sections4–6.
The source identifies sink tokens through unusually large normalized activation
in backbone-specific channels. It selects heads by visual mass and visual
non-sink mass fraction, then reallocates part of sink attention to visual
non-sink tokens. Equations3/4 define the selection/redistribution. We transfer
this attention-computation method, not claim to invent VAR or attention sinks.
Author implementation: https://github.com/seilk/VisAttnSink. Independent review
PASS: `docs/reviews/20261003_m1_recycler_proposal.md`. This implementation follows
the paper equations, with explicit Qwen3 adaptation below. Official code has no
Qwen3 indices or automatic channel-discovery rule; Qwen2 indices cannot be reused.
Paper/code differences are frozen before running: our transferred fraction .6
follows the paper (author code retains .6 and transfers .4); we use phi>=20 and
exact RMS (author code uses >20 and RMS epsilon1e-6); layers0..34 (code starts sink
scan at2); text sinks can trigger redistribution without visual sinks (code skips
that case); current query tokens are also scanned (code scans only prefill).
These choices constitute a paper-defined transfer, not an exact code reproduction.

Hypothesis: local visual decisions can improve by redirecting attention that is
dominated by activation-defined sink tokens, while preserving context in other
heads. Whether those tokens are semantically redundant in Qwen3/video is an
empirical question. High activation alone does not prove irrelevance.
Different from archived Selector: no temporal hard masking or local/other-window
head criterion. Different from PAI: no uniform image-logit increase, no text-only
reference or logit contrast. Different from Integrator: causal prefix unchanged.
Independent rule4 review must verify target-task novelty before implementation.

## Exact R1 and declared Qwen3 adaptation

Use original20frames, full ASR, policy, prompts,8s windows and4fps. Native prefix,
global question/answer, numeric global and speech branch remain unchanged.
Only the VISUAL WINDOW QUERY computation changes, across layers0..34 (all
except the last), all its input query rows, with causal edges preserved.

The source's published sink dimensions are specific to its older backbones;
they are not silently reused for Qwen3. Declare this label-free adaptation:
for each language layer, choose the two largest absolute channels in the
FIRST PREFIX TOKEN's residual hidden state entering that decoder block.
Resolve exact ties by smaller channel index. This first token is a constant
chat-template token, encoded causally before any media/text; its hidden state
does not depend on later video content. No labels or corpus fit are involved.
The number2 follows the source's LLaMA-2 example, not a Qwen-optimal claim.
Capture these indices during the native prefill, no extra forward pass.

For every cached and current-query token, compute phi=max absolute activation
on these two channels divided by its RMS across ALL hidden channels. Use residual
block-input states before learned RMSNorm, as in paper x notation. Mark sink
iff phi>=20. RMSzero means phi0. Save only sink masks and selected channel ids,
not all hidden states. Capture original prefix/global-QA masks during their
ordinary calls; query masks use that query's current states at the current layer.
Thus later-layer query masks reflect earlier interventions. No future query keys
can receive attention; the sink predicate of each key uses its own current state.

At each modified layer and each query row/head, compute ordinary masked
softmax attention (FP32 softmax, original BF16 QK/V arithmetic). Select the head
for that row iff total attention on image keys>=.2 AND the fraction of that
image mass on NON-SINK image keys>=.5. If no non-sink image or no sink mass,
leave it unchanged. Among selected rows/heads:
- multiply attention to every visible sink key (text or image) by.4;
- redistribute the removed.6 times total sink mass to non-sink image keys,
  proportionally to their original attention;
- preserve attention sum1; all other entries unchanged.
Use the resulting attention to compute V mixing and ordinary final Yes/No
token-set margin. No clipping of probabilities, extra temperature, score mixing,
new encoder, generated hypothesis or routing by corpus/GT.

Constants are source tau20,p.6,visualmass.2, hallucination-task rho.5; adapted
sinkchannel count2; all eligible layers except last. No scan. Source VAR changes
all text computations; this experiment deliberately limits it to local visual
questions to preserve native global/speech. Earlier query rows CAN change and
alter later query representations. Cached prefix/global-QA states stay original.
This is not a direct temporal support restriction and does not establish ASR fusion.

## Cost, tests and decisions

Deployment keeps native3+B outer calls (B=V+S), but each visual query requires
dense attention and per-row/head redistribution; there is real extra work.
Store layerwise boolean sink masks for prefix and global QA; retain no hidden-state
cache beyond native KV. Up to roughly P*Q*H attention values per layer/query,
not P² during native prefix. Reuse existing frames/ASR/weights.
Complete paired native / matched-dense-no-redistribution / recycle requires
3+B+2V calls, smoke native restoration another3+B. Provisional full333 paired
20–40GPUmin on5090, to replace with five-video smoke time/memory. No new
video preprocessing. If smoke does not fit32GiB, lower-memory equivalent row
chunking may be used, but not lower resolution/fewer frames or changed precision.

Independent code checks: source algebra and probability conservation, causal/GQA
masks, actual first-token channel selection without labels, native cached masks
and query-prefix alignment, capture/branch lifecycle, only visual queries, exact
native/speech/global/answer restoration, all-layer activation, actual calls,
timestamps4fps, complete333 canonical evaluation. Small multimodal Qwen FP32/BF16
and five real-input smoke videos (first2/corpus plus HMMhate_video_114); no GT in
smoke. Full run follows only after independent proposal/code and smoke PASS.

Main gate versus native/current AND matched dense: within+.01 both, no pooled
loss>.005 or within loss>.01. Rule9 revision only if some main gain>=.01;
otherwise archive. No component or alternate arm chosen silently after results.
If qualified, remove head selection (redistribute all heads), and compare to
matched-count random sink masks (fixed seed0), preserving visible key counts per
row/type. The controls need their exact replay/mapping declared before running;
neither can be claimed to isolate semantics merely by holding token counts fixed.
Any novel claimed component must cost>=.01 common main metric both corpora.
Report activation/sink/head rates, raw visual/combined ordering, final three
metrics, native-verdict strata, single-window contributions, paired uncertainty,
cost and actual positive/negative media examples. More attention by construction
or gains caused only by video offsets are not sufficient mechanism evidence.

## Run state

Proposal review PASS; implementation and independent code review next. Host not yet assigned.
No real labels read for this proposal, no new performance used to select constants.

## Implementation and code checks

Implemented `recycler.py` (block-input capture + temporary SDPA dispatcher),
`measure.py` (complete paired reader), canonical `analyze.py`, `smoke_report.py`
and `launch/`. CPU selfcheck FP32/BF16 passes independent row/head algebra,
4096-dimensional sink predicate, causal zeros, probability conservation, real
small-Qwen cache and native restoration. The small model uses explicit synthetic
sink masks solely to witness plumbing; it is not a semantic/real-media result.
Independent code review PASS: `docs/reviews/20261003_m1_recycler_code.md`, actual
36-layer multimodal Qwen FP32/BF16, controlled nonempty sinks, first-token channels,
prefix/globalQA/current-query mask capture, all35eligible layers/all suffix rows,
cache/inputs/native/global/answer/speech restoration, real call counts and canonical
report plumbing. A report key copied as pai_raw was fixed to recycle_raw before
GPU; synthetic reporting passed after correction. Artifacts under
`runs/20261003_m1_recycler/{selfcheck,independent_review}/`.

Timing caveat: paired native prefix includes sink-mask capture overhead; its time
is an upper bound on untouched native processing. Do not use the candidate/base
ratio to claim an exact native-relative overhead; report absolute times and this
measurement limitation. Source capture overhead is part of candidate deployment.

GPU smoke on idle sc448960 after git sync and full preflight:
`bash experiments/20261003_m1_recycler/launch/run_lab.sh smoke`, then local
`python experiments/20261003_m1_recycler/smoke_report.py` after result return.
Only if plumbing/cost pass: `.../run_lab.sh main`; local analysis
`bash experiments/20261003_m1_recycler/launch/run_analysis.sh`.
Host HateVLM torch2.11+cu128/transformers5.15.1; existing inputs/weights reused.
No labels/performance read for these checks. Real sink activation is to be measured,
not assumed from controlled small-model tests.

## Five-video GPU smoke PASS

sc448960, results returned locally:
`runs/20261003_m1_recycler/r1_smoke/plumbing_summary.json`. Five native globals
and292native branches exactly match historical baseline; original global/answer,
speech, inputs, query lifecycle/call counts and finite4fps pass. Largest prefix5829,
peak17.8247GiB, wall49.5s including restoration. Actual selected head-row fraction
averages.0103–.0267 across the five videos; transferred attention mass averages
.00167–.00458. Sink predicate is active on real Qwen3 inputs. Dense-eager visual
max drift reaches.5048; recycle changes reach2.268. These are activation/numerical
checks, not semantic or performance support. Conditional gating means not every
eligible layer transfers mass; all eligible layers are nevertheless executed.

First2/corpus extrapolation, excluding stress: capture-including baseline210.35/
287.29s, recycle236.11/347.78s, paired378.81/663.94s. Total standalone9.73min,
paired17.38min; estimates only, full measured times will replace them. No new
labels/performance read, no constants changed. Full333 begins after sync and
fresh preflight on the same sc448960 host.
