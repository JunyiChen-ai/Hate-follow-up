# M1 Eraser: re-encoding after temporal media removal

Date2026-10-03. Status: proposal PASS, implementation under code review. Fifth candidate of the M1 iteration;
Attributor is still running and has no inspected performance. Marginalizer was
STOPped at proposal, not implemented. Planned host lab-server/sc448960 subject
to live availability. Independent proposal review required before implementation.

## Mechanism and prior negative evidence

The current reader asks whether each window contains a violation. Instead,
measure how the original whole-video decision changes when the observed media
in that window are physically absent from the model input. Re-encode the entire
remaining prefix for each intervention: cached contextualized keys/values from
the original full video are not reused after removal. This prevents erased
content remaining in other tokens' cached states. The question always remains
exactly VIDEO_QUESTION; no instruction to imagine ignoring an interval.

F is the fixed global Yes-minus-No log-sum-exp margin, exactly the current
VIDEO_QUESTION scoring function. For window i and modality m,
z_i^m = F(full video)-F(video minus media i,m).
Visual removal drops all sampled timestamped frames in that window, retaining
speech. Speech removal drops the window's words from the transcript, retaining
frames. Both therefore test one modality's necessity given the other modality
and all remaining temporal context. A cross-modal conjunction can affect both
interventions. Redundant hateful evidence may have little individual influence;
this is a limitation, not something the score automatically solves. Difference
from a fixed full-video score is only a definition of attribution; subtraction
by itself cannot improve within-video order. The scientific test concerns the
new re-encoded removal reads, not cancellation of a constant offset.

Current visual/speech raw max and r6 inference algorithm remain fixed. Original
native global z_v stays downstream. No labels, added encoder, learned component,
score blend, ensemble, dataset routing or new smoothing. This is input occlusion
attribution transferred to label-free hateful-video localization; do not claim
new causal mathematics or counterfactual ground truth.

Prior documents read for design (development analysis, no new GT read):
- archive/experiments/20260912_cva/README.md: hypothetical verbal exclusion
  failed. Its prefix still contained the purportedly ignored interval.
- archive/experiments/20261002_m1_grounder/README.md and m1_selector counterpart:
  restricting window queries' cache attention failed. They did not recompute
  the media-conditioned prefix or measure an original global-question response
  to physically removed input.
- experiments/20261002_m1_attributor/README.md: signed value-path IG currently
  being tested; it leaves contextualized keys/scaffolding and uses zero values.
  Its performance is unknown. Eraser instead removes actual media and recomputes
  every dependent state, with a substantial cost that is counted explicitly.
- Prior frame coverage and GLR failures warn about content quantity and word
  prediction confounds. Do not re-test GLR as a new idea.

## Exact input transformation and fixed constants

Same Qwen3-VL-8B-Instruct native BF16,20-frame cache, repaired Whisper loader,
policy, global question,8-second windows,4fps outputs,seed0. No prompt edits.
Visual membership: timestamps in [start,end), with last window including end.
Zero sampled frames => z_visual=0 exactly; no identical re-encoding call.
Speech membership: each original ASR segment is split into whitespace words;
indices are rounded proportional-time boundaries exactly as window_text.
Delete those indexed words, retain original order of remaining words and the
original segment's start/end timestamps. Preserve untouched segment text exactly;
only affected text is rejoined with spaces. Empty segments are omitted by the
existing transcript renderer. The remaining words still lie in their original
coarse ASR interval; no new word alignment is inferred. Log deleted indices,
word counts and original times. Omit a speech score when the original window
has no speech, as current. No speech-token clipping or length normalization.

Current baseline is rerun with its ordinary chosen-verdict windows first, before
any altered prefix. Original global margin and every native baseline window
must match current cached reader. Every altered prefix resets the full model
prefill and positional state; no cache crosses interventions. Restore and
re-read the full video in smoke to verify history/position independence.

## Primary and falsification plan

1. One independent rule4 proposal review, with actual target-task literature
   search; then implementation and one rule6 code review.
2. First two manifest videos per corpus: smoke for exact baseline parity,
   per-window erasure membership/conservation, native identity when no media
   removed, fresh-prefill repeatability, shapes/missing branches and real cost.
   No GT or localization metrics in smoke.
3. Complete333 paired base/Eraser videos on one host, canonical evaluator,
   raw and unchanged r6, all3metrics, within paired bootstrap2000 seed0.
4. No final metric +.01 anywhere => archive. Any qualifying improvement allows
   declared follow-up (max3 revisions). Promotion requires within+.01 both and
   no main drop beyond .005 pooled/.01 within versus paired/current. Every
   novelty claim must pass14g on both corpora.
5. If positive, evaluate controls: hypothetical-exclusion global reads with
   original prefix (CVA intervention versus actual removal); removed-frame/word
   density alone; half-video-offset erasure scores within modality; constant
   per-video common shift that matches Eraser/base mean-max difference while
   preserving native raw max ranking. Compare actual original modality queries
   under the same re-encoded removed input for selected deterministic cases,
   and report both improvements and harm from lost context.
6. Post-score analysis: per-video/time ordering changes, positive/negative
   windows' erasure effects, deleted quantity strata, correct/wrong global
   judgement, sparse/diffuse positives, missing-frame cases, strongest paired
   gains/losses with original media/transcripts. A model response to erasure is
   faithful to that intervention by construction but is not proof of correctness.
   No labels enter inference, fitting, constant choice or routing.

## Prior art to verify, and cost

Primary source family: input occlusion sensitivity (Zeiler/Fergus,
https://arxiv.org/abs/1311.2901) and efficient LLM context attribution
(AttriBoT, https://arxiv.org/abs/2411.15102). Independent reviewer must search
whether actual temporal/modality media erasure has already been used in hateful
video detection/localization, including post-hoc explanation, not just whether
our exact formula is absent. Source-already-used means STOP under rule4.

Reuse frames/ASR/weights only. Original prefix K/V cannot be reused after an
intervention; do not describe this as cheap cached/offline processing. Let B be
original observed modality branches and E the number of nonempty removals.
Base standalone3+B forwards; Eraser standalone2+2E forwards (one native prefix
and its global query, then E fresh prefixes and global queries). Paired run
adds native local branches/answer extension. No backwards. Nonempty visual
removals<=20 and speech removals<=N; full re-encoding typically dominates.
Initial full-pair estimate60-120 GPUminutes on5090; replace by smoke-based
estimate before launch. Corpus timings/call counts and missing-media skips are
recorded. Controls' additional model cost separate. All outputs in project,
no hashes; local code then scoped git sync; return results before STATUS update.
All outcomes development-selected. Current method/paper/Overleaf unchanged.


After proposal declaration, numerical-only exploration on first60 completed
Attributor videos (no GT or localization metrics): zeroing cached media values
leaves F0/F1 rank correlation .994887, same decision sign90%, median absolute
margin change1.248709. Local source
`runs/20261002_m1_attributor/r1_main_fp32_mem/reference_diagnostic_60.json`.
This is consistent with the already-declared limitation that non-media states
retain input information. It does not establish that attribution localization
fails, or distinguish retained semantic evidence from a generic prior. No
scoring design, constants or subset changed from this diagnostic.


## Implementation checks before GPU performance

Proposal PASS: `docs/reviews/20261003_m1_eraser_proposal.md`, with explicit
full-text access limits. Code uses `Judge` for every fresh raw-media prefix;
no original KV cache is retained for an intervention. Complete CPU membership
check:333 videos,7359 windows,11676 nonempty modality removals. Removed words
match `window_text` exactly; indices do not overlap and cover the whole-duration
word subset, and all in-duration frames are assigned once.712 ASR words are
outside the manifest video duration and remain in the unchanged global context,
as in the baseline; they are not silently reassigned to a window. This is an
input/protocol limitation, no GT read or input fix. Source:
`runs/20261003_m1_eraser/selfcheck/transform.json`.

Implementation/launch: `measure.py`, `erasure.py`,
`launch/run_lab_server.sh smoke|main`; after full results return locally,
`launch/run_analysis.sh`. Primary outputs `runs/20261003_m1_eraser/r1_main/`,
canonical raw/r6 outputs `r1_main/{base,erase}/metrics.json` and
`r1_main_decoded/{base,erase}/metrics.json`. No model changes or metrics yet.
