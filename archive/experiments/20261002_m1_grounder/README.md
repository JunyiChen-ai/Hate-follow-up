Archived 2026-10-02: no main metric improved by .01; no evidence for the proposed late-access mechanism.

# Grounder: early context, late evidence access

Declared 2026-10-02, before implementation/outcome inspection. Development host
uoa-lab1/sc474397; intended compute host uoa-lab2, subject to live availability.
Parent autonomous task: `experiments/20261002_m1_iteration/README.md`.

## Hypothesis and difference from earlier attempts

Current M1 asks each window question while allowing every language-model layer
to attend to the entire video and its own global verdict. A timestamp in the
question does not enforce which evidence the final decision uses. Earlier error
analysis found many false alarms far from the annotated segment and strong topic
responses. Candidate: retain contextual interpretation in early layers, then
restrict direct evidence access in late layers to the target window.

This is the unrun suggestion in `docs/reviews/20260928_codex_mechanism_review_round2.md`,
now tested directly under the user's M1 iteration instruction. Earlier tests of
input-context removal/score mixing do not execute this layer-dependent computation.
It differs from CVA's hypothetical exclusion question and from subtracting completed
logits. No prompt, backbone, frames, transcript, output head or Decoder changes.

Important limit: cached media representations and query residuals remain
contextualized. We restrict **direct attention access**, not all information flow;
no claim of complete causal isolation, decontamination, or semantic grounding by
construction. Whether this helps is an empirical question.

## Fixed mechanism

- Qwen3-VL-8B-Instruct, frozen; current shared Judge and repaired ASR loader.
- Same 20 frames, timestamped transcript, global Yes/No turn, 8-second windows,
  isolated visual/speech questions, and missing-speech handling as current M1.
- Prefix encoding and video verdict unchanged. In each window branch, the first
  three quarters of text layers see the whole prefix. In the final ceil(L/4)
  layers, allow policy/system/non-evidence scaffolding, that window's prefix
  frames/transcript tokens, and the branch's own causal query tokens; block
  direct access to other media and the global question/answer turn.
- Map frame tokens by their existing timestamps and transcript words by the
  same proportional overlap rule as `window_text`. Do not add a nearest frame
  to empty windows. Empty visual evidence is recorded, not filled with new data.
- Both local modalities remain available to both branches at the mask level;
  keep current questions to isolate this computation change.
- One fixed layer split, seed 0, same constants and procedure on both corpora.
  No layer sweep. Changing depth later requires a declared revision, all results.

## Controls declared before running

All arms use the same explicit causal attention-mask plumbing and frozen prompts:

1. `base`: all prefix evidence accessible at every layer (plumbing baseline).
2. `late`: candidate above.
3. `verdict_only`: in the final quarter block only the global Q/A turn, retaining
   all media. Tests whether any gain merely comes from reduced verdict access.
4. `all_local`: the local restriction at every query layer. Prefix encoding still
   contextualized; tests the need for early full-prefix access in the branch.
5. `shifted`: same late restriction and token count as `late`, but rotate the
   kept media positions halfway through each modality's prefix token positions.
   Preserve scaffolding and the question (including its copied local speech).
   This is a budget-matched wrong-prefix-evidence control, not total replacement
   of the window's information. Log overlap with the true mask.
6. `early`: same local mask and same ceil(L/4) restricted layers as `late`, but
   apply it in the first quarter of query layers. Added before running on the
   independent reviewer's recommendation: distinguishes layer placement from
   simply restricting fewer layers. A stage-placement claim needs a >=.01
   improvement over this control on both corpora too.

First verify token mapping, cache crop/copy parity, causal mask, hooked vs ordinary
base, same global read in every arm, and per-video/window coverage. Two videos per
corpus may be used only for plumbing/speed checks; no outcome-based subset screening.
Then run complete HateMM and HateClipSeg for all six arms; one corpus is never sliced
across machines. Apply unchanged r6_bma through its CLI and the canonical evaluator.

## What would support the explanation

- Relative to current matched base: both corpora no decline beyond .005 pooled /
  .01 within, and within improvement >=.01 on both (target M1 localization).
- `late` must outperform `shifted` on both; raw local discrimination should also
  improve, not only a corpus-fit shift. Quantify within-video rank changes.
- To claim staged context/evidence access, `late` versus `all_local` and
  `verdict_only` must each show a >=.01 main-metric gain on both datasets (rule14g).
  Otherwise narrow the claim/delete the unsupported part; do not invent a story.
- Paired video bootstrap, correct/wrong original verdict groups, sparse-positive
  videos and windows with/without sampled frames; report regressions as well as
  selected gain cases. These are exploratory diagnostics, not semantic labels.
- Global-query output must remain identical; with fixed Decoder any changes must
  originate in M1. Separate raw-reader changes from downstream distribution effects.
- CPU-only distribution control, declared before metrics: take base raw reads and
  add one common constant per video equal to mean(max late branches) minus
  mean(max base branches), then apply the same r6 fitting. This preserves raw
  within-video max/modality ordering. If it explains the gain, do not attribute
  improvement to better temporal evidence ranking. It uses both experimental
  caches and is a diagnostic reconstruction, not a deployment method.
- Window-rank diagnostic declared while GPU reading is in progress, before any
  performance inspection: per-video percentile-rank change for positive windows
  (GT fraction >=.5), pure negative windows whose midpoint is >8s from a positive
  GT frame, and nearer pure negatives. Bootstrap per-video group means, not
  individual windows. The hypothesis predicts decreasing far-negative rank and
  retained/increased positive rank; ambiguous partial windows are omitted from
  these diagnostics. These thresholds never enter the method or primary metrics.

## Cost and execution

Existing frames, ASR and weights reusable. Intervention requires fresh reads; old
logit caches cannot simulate it. New-video model calls equal base (one prefix,
one global query, its answer extension, one branch per observed modality/window).
Hooking a mask adds no model forward. Estimate ~1–3 seconds/video on a 5090 from
previous same-reader runs, 6–17 GPU minutes per complete 333-video arm, six arms
~0.6–1.7 GPU hours before prefix-sharing savings. Experimental arms share each
video's unchanged prefix/global computation; deployment cost still charges it to
each method. Measure actual overhead in the plumbing run; report if >25%.
GPU witness and existing environment checked before launch. Repository no-hash,
Git-only sync, detached jobs and output-return rules override skill defaults.

Proposal review: PASS, `docs/reviews/20261002_m1_grounder_proposal.md` (rule4,
independent instance, actual literature search). Attention steering and cross-layer
evidence scheduling have prior work; potential novelty is application/mechanism
validation in this task, not inventing attention steering. No layer cognitive
specialization is assumed or claimed. Reviewer-requested `early` control added.
Code review: PASS, `docs/reviews/20261002_m1_grounder_code.md` (rule6). Calls metadata
was corrected to actual 3+n_branches forwards per deployed arm; no scoring bug found.
GPU smoke on sc474399: four videos, all six arms; token mapping valid, all crop/copy
and explicit-base/ordinary-SDPA differences exactly zero. Late branch time / base
was .991 on these four videos; this is a plumbing estimate, not full-corpus timing.
Smoke returned to `runs/20261002_m1_grounder/r1_smoke/`. Independent CPU real-layer
checks verify depth selection, causality, branch order/cache invariance and all
333 videos' proportional ASR-word slicing. No performance metrics inspected yet.

Environment: existing lab2 `HateVLM`, Python 3.12, torch 2.11.0+cu128,
transformers 5.15.1 (same library versions as current Reader); 5090 seeded GPU
matrix witness finite and correct shape. No environment rebuilt. Local analysis
uses HateVideo. Selected remote checkout was fast-forwarded and matches local
code; unrelated legacy `results/` inputs and other home projects are untouched.

Run `bash experiments/20261002_m1_grounder/launch/run_lab2.sh full` detached with
logs under `runs/20261002_m1_grounder/r1_full/`; rsync that run back after completion,
then `launch/run_analysis.sh` locally for raw evaluator outputs, unchanged r6
inference, shift-only control and paired/bootstrap report. GT enters analysis only.
For this run, `launch/run_all_analysis.sh` performs the same stages, with the
seven independent CPU decoding arms in parallel on lab1 (16 CPUs, 48GB available
before launch; one BLAS thread per arm). It waits for every successful exit before
reporting. `cost_alignment.json` records full-corpus parity against the existing
current Reader and observed standalone-time estimates, including each arm's share
of prefix computation. This changes execution scheduling, not any numerical fit.

Input-only coverage check while the GPU run is pending (no GT/performance read):
`runs/20261002_m1_grounder/input_coverage.json` uses the current shared input
loader. HateMM has 3,768 windows (1,030 without a sampled frame; 118 without a
frame or speech); HCS has 3,591 (1,233 and 169 respectively). Thus frame-coverage
strata are material. The hard-mask arms retain contextualized prefix/query states
even for these windows; they must not be described as having removed all external
information. No missing-input fallback or constants were changed after this check.

## Full result and decision (2026-10-02)

All six arms completed on sc474399, 333 paired videos, 2,596 seconds elapsed
after model loading. Returned locally before analysis. Every global and window
logit in the base arm exactly reproduces `runs/20260926_glr/base_gridA` (maximum
absolute difference zero). All downstream arms completed on sc474397, using the
same r6 algorithm with separately refitted label-free parameters. Development-selected.

Official sources: `runs/20261002_m1_grounder/r1_full_decoded/<arm>/metrics.json`;
raw sources: `runs/20261002_m1_grounder/r1_full/<arm>/metrics.json`.
Order below is pooled ROC / pooled PR / within ROC; within N=84 / 99.

| Arm | HateMM | HateClipSeg |
|---|---|---|
| base | .897119 / .694235 / .750782 | .716825 / .671072 / .637349 |
| late | .897570 / .695834 / .750171 | .716291 / .670781 / .637249 |
| verdict_only | .897551 / .695239 / .749620 | .716391 / .671259 / .637407 |
| all_local | .895879 / .693183 / .705164 | .701506 / .655243 / .588674 |
| shifted | .897616 / .696013 / .749408 | .716177 / .670634 / .637110 |
| early | .895925 / .691426 / .749139 | .717713 / .673074 / .630512 |
| shift_only (diagnostic) | .897598 / .695979 / .754803 | .716379 / .670834 / .638305 |

No candidate/control improves a final main metric by .01. Archive under rule 9;
do not promote the design or write it as a contribution. Current method/paper unchanged.

Mechanism evidence (`r1_full_analysis/summary.json`, `read_diagnostics.json`, and
`window_rank_diagnostics.json`):

- Late versus base within differences: -.000611 [95% paired-video CI -.004019,
  +.002558] / -.000100 [-.003042,+.002662]. Raw within: +.004628 / -.000044;
  both intervals include zero. No consistent improvement before or after decoding.
- Late versus shifted final within: only +.000763 / +.000139. The average overlap
  of their local-media support is .0080 / .0055, yet mean absolute branch-logit
  differences are only .0593 / .0518. Correct support in the last quarter does
  not have the predicted substantial effect. This does not identify whether
  earlier query states or contextualized prefix K/V is responsible.
- Restricting all query layers loses .0456 / .0487 within relative to base;
  HCS pooled metrics also drop. Removing broad context everywhere is not a fix.
- The far-negative rank does not consistently decrease; on HateMM it slightly
  increases (+.00347 percentile-rank units). Frame strata do not rescue the claim.
- Shift-only reconstruction is slightly better than late on both corpora, still
  within noise versus base. There is no supported localization gain to explain.

Cost (`r1_full/cost_alignment.json`): estimated standalone synchronized wall time
base 305.2 / 246.7 seconds, late 304.8 / 246.4, excluding model loading and original
frame/ASR preparation. Mean forwards per video 36.52 / 60.05, exactly the baseline
call count; no added calls. Experimental prefix sharing is not counted as a
deployment saving.

Test-read log: inspected all raw/decoded official metrics, paired bootstrap,
window ranks, per-video gain/loss rankings, and descriptive branch changes after
all GPU reads completed; analysis reads `data/gt_4fps/{HateMM,HateClipSeg}.npz`.
No semantic case class is inferred from these numeric rankings. These findings
motivate the next separate candidate, Selector: preserve context heads while
selectively modifying heads whose current query retrieves local evidence.
This is a new mechanism, not a sweep over Grounder's layer split.
