# M1 Contraster: depth-contrastive local evidence reading

Declared2026-10-03, seventh candidate. Proposal PASS; implementation and CPU
checks completed, code review PASS, no performance. Eraser and Factorizer are running; either supported result takes
priority. Development host sc474397. All eventual results development-selected.

## Hypothesis and source boundary

Earlier tests changed input availability or attributed the global answer to
media. This candidate retains the entire native computation and reads how a
window's answer changes across language-model depth. A final answer can retain
strong associations already present at a premature layer; the depth contrast
may emphasize information incorporated by later computation. The testable
hypothesis is better within-video ordering from this new readout, with matched
temporal pairing mattering. It is NOT established that early layers represent
topic and late layers represent hateful acts, or that later is always better.

Source: [DoLa, ICLR2024](https://arxiv.org/pdf/2309.03883), layer-contrastive
decoding with a dynamically selected premature distribution. We test a binary
evidence-reader adaptation, not new contrastive-decoding mathematics. The
[author implementation](https://github.com/voidism/DoLa/blob/main/dola.py)
also scores supplied continuations. Its KL argument order differs from the
standard Jensen-Shannon definition; this candidate explicitly uses standard
JS below, and does not claim bitwise reproduction of that code. Adjacent
[LayerCD](https://arxiv.org/html/2509.25177v1) contrasts vision-encoder feature
depths, whereas this candidate reads language-decoder residual states from one
unchanged multimodal forward. Independent rule4 review must actually search
hateful-video detection/localization, and decide whether this internal readout
is an eligible task transfer rather than pure calibration under repository rules.

Local sources read: current STATUS; GLR, TAD, NGA and headroom READMEs; current
`src/mllm_judge.py`. Final-state supervised-probe failures do not test a depth
trajectory. No pending Eraser/Factorizer scores or GT have been read. The already
archived attribution/attention failures are motivation to change the readout,
not evidence that this particular readout succeeds.

## Exact method and constants

Use the original Qwen3-VL-8B, native BF16,20frames, repaired ASR, policy,
VIDEO_QUESTION, selected native answer and8-second isolated visual/speech
queries. Original z_video and answer remain unchanged. Native raw maximum,
missing-speech handling,4fps and r6 inference algorithm remain fixed.

For each available window branch, capture only its final query token's
post-block residual state after language blocks2,4,...,L-2 (L=36 here).
Apply the model's native final RMSNorm to each captured state. The actual final
normalized state is the mature state. Project them using the same LM head in
FP32 and obtain full-vocabulary distributions p_l and p_L, temperature1.
Select l*=argmax_l JS(p_L,p_l), with ties taking the shallowest candidate.
JS(p,q)=0.5 KL(p||(p+q)/2)+0.5 KL(q||(p+q)/2), natural logs. No labels or corpus
statistics participate in selection. Save layer choice and all JS values.
Also save all candidate/mature Yes/No-token logits and full-vocabulary log
normalizers. These small arrays reconstruct the fixed-middle-layer control
and expose the native probabilities of dominant token variants. Full-vocabulary
tensors are not saved. The source selects layer buckets on a validation set;
our all-interior-even-layer pool is fixed, without such selection, so its source
performance and tuning claims do not transfer.

For each token t in the native Yes/No token sets, d_t=log p_L(t)-log p_l*(t).
Window-branch evidence is logsumexp(d_Yes)-logsumexp(d_No). The shared full-
vocabulary normalizers cancel between the two label sets; compute the final
margin directly from FP32 logit differences to avoid unnecessary cancellation
error. Do not replace it with z_L-z_l, which is generally different when a label
has multiple token variants. Coefficient1, no blending or temperature scan.
No plausibility truncation: this is a finite binary evidence score, not open-
ended token generation. This omission is a stated adaptation from source DoLa.
It may amplify extremely unlikely label spellings; report which variants
dominate before/after contrast and their mature/premature probabilities, and
do not interpret such amplification alone as new factual evidence.
The report counts contrast-dominant variants with mature full-vocabulary
probability below.001, a descriptive diagnostic never used by the reader.
No layer skipping, attention edits, extra models, training or label calibration.

## Experiment and controls

1. Independent proposal review; if PASS, implement then independent code review.
   Verify block indexing, final RMSNorm exactly once, actual final-margin native
   parity, hook removal, cache restoration, stable JS and logit-difference
   equivalence. Save scalar diagnostics, not every full-vocabulary tensor.
2. First two manifest videos per corpus plus the largest5829-token prefix
   (HateMM/hate_video_114, selected from prefix-length metadata) for GPU plumbing
   and measured cost, without GT/performance screening.
3. Full333 videos in one paired native/contrastive measurement. Canonical raw
   and r6 evaluator, all3 metrics, paired-video bootstrap2000 seed0. Require
   within+.01 on both corpora and no main loss beyond.005 pooled/.01 within
   against paired native and current r6. Any final main gain>=.01 permits up to
   three declared revisions; no qualifying gain means archive under rule9.
4. Only after qualifying gain, measure controls: native mature-only(remove
   contrast); fixed middle layer L/2(remove dynamic selection); permute the
   selected premature label logits across windows within each modality and
   video using default_rng(0), preserving native mature logits(no matched
   depth trajectory); common-shift diagnostic preserving native raw-max order.
   Record actual altered pairs; one-window branches are not an effective
   permutation. If dynamic selection fails14g, it cannot be claimed as novelty
   and must be removed or downgraded with a tested simpler method.
5. Raw-vs-decoded ordering, selected-layer distribution, correct/wrong native
   verdict, sparse positives and gain/loss cases. If claiming reduced topic
   bias, explicitly test GT-by-group-mention strata with the existing frozen
   group-word diagnostic; neither selected layers nor JS alone proves that
   semantic interpretation. Gains only after r6 or equally under a common
   shift do not establish improved raw localization. All tests are development
   evidence, not a label-free guarantee of correctness.

## Cost and outputs

Reuse current media/ASR/weights and native prefix cache. Outer model forwards
remain3+B per new video, no extra transformer pass or backwards. Each branch
adds17 intermediate full-vocabulary projections plus one mature full-vocabulary
projection for JS (native needs only its Yes/No rows). Qwen's FP32 LM-head copy
requires approximately2.3GiB; the17 captured final-token states are small. This
extra projection work and bandwidth must be timed, not called free. Initial
estimate10-25GPUmin for both corpora on a free5090, subject to smoke before full
launch. If memory requires row-chunked projection, preserve exact definition
and report implementation; do not choose layer subsets after performance.
Paired baseline shares the same native forward; save per-branch readout time
and per-corpus native versus contrastive deployment totals. Output root
`runs/20261003_m1_contraster/`. No paper or current-method changes before evidence.

Independent proposal PASS: `docs/reviews/20261003_m1_contraster_proposal.md`.
CPU FP32/BF16 actual six-layer Qwen3-VL text-model checks passed: captured
post-block states equal returned intermediate states, native final hidden/KV
unchanged, one final norm and projection exact, standard JS matches double
precision reference, log-probability/logit-difference equivalence within2.4e-7,
weights and hooks restored. Extreme-logit stability and shallowest tie checked.
Source `runs/20261003_m1_contraster/selfcheck/invariance.json`, no GT.
Baseline time in the shared-forward experiment includes intermediate-state
capture overhead and is therefore an upper bound, not a pure native benchmark.
Smoke additionally repeats ordinary unhooked queries to verify every margin.
Actual full measurement3+B forwards; smoke3+2B. FP32 head setup is recorded
separately; full-vocabulary projections use one batch, with its mature-margin
numeric drift from the native small-row projection recorded explicitly.
Independent code PASS: `docs/reviews/20261003_m1_contraster_code.md`. A report-only
field-name bug was fixed before GPU runs and confirmed through a complete
synthetic report. Independent36-layer FP32/BF16 and reader orchestration tests
are in `runs/20261003_m1_contraster/independent_review/check_contraster.json`.
Steady-state per-video timings exclude model loading and the one-time FP32-head
setup; report these distinctions rather than presenting them as cold-start cost.

## GPU plumbing and launch, 2026-10-03

Five-video smoke completed on sc448960. Returned source:
`runs/20261003_m1_contraster/r1_smoke/plumbing_summary.json`. All292 native
window reads and five global margins exactly match base_gridA; ordinary repeated
query parity,4fps alignment, finite scores and saved-logit reconstruction pass.
Peak20.142GiB; log-probability/difference equivalence error7.63e-6; mature batched
projection margin drift from native small-row projection at most.0002022.
Before full performance, add the following numeric control if a qualifying gain
appears: reconstruct mature-only margins from saved batched logits. This keeps
the exact projection plumbing of Contraster and tests whether tiny GEMM drift
can explain a result. No new model calls are needed and no control performance
has been inspected. The paired native arm remains exact and is the primary gate.

First-two-per-corpus timing extrapolates3.61/5.09 minutes for Contraster; added
projection time on those samples is.084/.298sec respectively. The sample is
small and not representative; budget10-20GPUmin for both corpora. Head FP32
setup.0099sec, excluded from steady-state totals. Main command, after live GPU
check on sc448960: `bash experiments/20261003_m1_contraster/launch/run_lab.sh main`.
Output `runs/20261003_m1_contraster/r1_main/`; native+contrast share3+B forwards.

Full paired333-video run started on sc448960 at03:01NZ. Eraser completed negatively
and was archived after the Contraster declaration and smoke; no method change.
