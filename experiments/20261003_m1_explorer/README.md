# M1 Explorer: uncertainty- and support-aware visual evidence acquisition

Candidate18, declared2026-10-03 before Preserver performance is read.
Self-designed acquisition mechanism informed by the native frame-support diagnostic
and EcoFrame, not a reproduction of that paper. Development-selected; frozen
Qwen3-VL-8B, native global/speech, original policy/questions,8s windows,4fps and
unchanged r6. No implementation before independent rule4 proposal review.

## Hypothesis, evidence and source limits

Native20-frame cache timestamp annotations place no frame in1030/3768 HateMM
windows and1233/3591 HCS windows. Source `runs/20261002_m1_iteration/frame_support/summary.json`
uses manifest/frame times only, no GT. These are nominal seek timestamps, not
verified decoded PTS: old extraction could retry .5s earlier without renaming.
This diagnoses missing nominal support, not missed hateful events or exact pixel
timing, and is not a causal explanation of errors. Interventions
on existing representations have not produced a qualifying dual-corpus result.
Hypothesis: bounded acquisition of actual, temporally relevant new frames can
correct insufficient window reads while retaining the native global context.
The global verdict remains fallible; additional local evidence can change a
window score but this mechanism does not guarantee correction of a wrong verdict.

Primary inspiration: EcoFrame, arXiv2608.03918v1,
https://arxiv.org/html/2608.03918v1, sections2/4/5.1 and appendixA.3/B/C.
The author repository https://github.com/AK-DREAM/EcoFrame has only a README as
checked2026-10-03. The source combines full-vocabulary answer entropy, attention
prior times temporal distance for candidate acquisition, and separate CLIP-based
reselection. Here the target is a fixed localization window, so we directly acquire
new frames inside that window. We do not claim CLIP reselection or an exact source
implementation. Binary Yes/No entropy is a declared adaptation, not source entropy.
The explicit no-frame rule follows our structural input diagnosis, not a GT threshold.
Source appendixB uses deep layers19–21, whereas this proposal uses all36layers;
this is another explicit adaptation. Proposal review identified MATCH (TCSVT2026,
author source https://jianlang.org/papers/MATCH.pdf) as a close target-task neighbor
using CLIP-based positive/negative cue retrieval from a uniform frame pool.
Do not claim the first evidence/frame retrieval for hateful videos; the tested
question is whether this specific bounded feedback acquisition improves localization.

## Exact R1, fixed for both corpora before scoring

1. Build native20-frame full-video prefix (actual18 for the existing short cache),
   original timestamped transcript/policy; compute native global margin and append
   its own native hard answer. Read native speech windows unchanged. This global
   context remains the starting cache for every visual query.
2. Read the original visual window question on that cache. Save its native margin
   z and a frame prior: from the final query token, compute pre-RoPE attention over
   image keys only, after native q_norm/k_norm, at all36language layers/all32query
   heads (repeat the8KV heads normally). Softmax within image keys, average over
   heads/layers, then mean over each frame's visual tokens and renormalize over
   frames. No interventions in this pass. Cached image keys must be captured
   before RoPE during prefix encoding; query vectors captured at the same site.
3. Let p=sigmoid(z) and h=-p log(p)-(1-p)log(1-p), computed stably. At round0,
   if the window contains at least one original frame by its unchanged cache
   timestamp annotation and h<.1, return z.
   Otherwise acquire2new frames. After the first acquisition, stop if h<.3;
   otherwise acquire2more. Stop after4new frames, regardless of entropy. No labels,
   model fitting, per-corpus rules or alternate output selection. Return the LAST
   evaluated margin, not an average, maximum or confidence-selected past margin.
4. Candidate target times are4fps bin centers t=(i+.5)/4 within the current
   [window_start,window_end). Probe decoded-frame presentation timestamps and
   source indices without GT; normalize PTS by the container presentation origin.
   Each target maps to the first decoded frame at or after it. Reject mappings
   outside the window or actual video coverage; do not substitute a neighboring
   window's frame. Deduplicate source indices among candidates and newly acquired
   frames. For the old JPEGs, reconstruct requested seek times from extraction
   index/duration (rounded3decimals) and conservatively exclude source-index
   mappings for both that time and its possible .5s-earlier retry. Original PTS
   were not saved, so this is a conservative legacy exclusion, not a proven exact
   identity audit. Native JPEGs and native timestamp tokens must stay unchanged.
   If fewer than2 candidates remain, use all; if none, return the last margin.
   Store actual new-frame PTS/index and use actual PTS in added timestamp tokens.
5. Sort original frames by their native annotated time and new frames by actual
   PTS, and propagate their
   attention priors to nearest-frame temporal cells (midpoint boundaries). At
   candidate time t use A(t)^.5 * min_s |t-s|/video_duration, where s ranges over
   all original plus newly acquired frames for this query. Greedily pick the max,
   update distances and pick the next; earliest actual source-frame time breaks
   ties. The prior A is fixed within an acquisition batch and refreshed from the
   next query read. No raw pixel alteration or synthetic frames.
   Here candidate t is the mapped actual PTS, never the requested grid center.
   Equidistant nearest-frame ties use the earlier time; exactly coincident
   observed times use the mean of their frame priors. These rules are deterministic.
6. For each acquisition read, start from the exact same native verdict-conditioned
   cache, append all acquired local frames with their actual timestamps and the
   EXACT original visual question. The appended images are new observations in
   the current user turn; no previous window answer is appended. Use native Qwen
   multimodal positions for the appended block, restore its correct rotary state,
   crop back to the shared cache after reading. Original images/ASR/global answer
   remain available. The whole acquired set is re-encoded at round2; charge that
   cost. Capture attention for original and appended frame tokens for the refreshed
   prior. Keep the original12-token FP32 Yes/No head. Changes enter final visual
   scores only; original global, speech, max combination and r6 stay fixed.

Constants: support=at least1original cache-annotated frame; entropy thresholds.1/.3nats;
2frames per acquisition, max4new; candidate4fps; attention exponent.5; all36layers;
chronological earliest-time ties. Source .1/.2 entropy schedule inspires thresholds,
but these are explicit new binary-entropy constants, with no scan. No other
candidate's intervention, extra model, CLIP score or prediction ensemble is used.

## Cost and required implementation checks

Let V be visual windows, B=V+available speech, R_w in{0,1,2} be acquisition reads.
Native deployment3+B; Explorer3+B+sum_w R_w. Original global and native visual
query can be shared in paired collection, so collecting the baseline requires no
extra MLLM forward. Each expanded read additionally runs the vision encoder on
2or4images; the4-image second round reprocesses the first2. Maximum6V additional
image encodes,2Vextra language forwards, no full-prefix re-encoding. Pre-RoPE
capture costs are charged to Explorer; native timings exclude capture or state
a matched instrumented baseline explicitly. Reuse original image/ASR/weights,
but decode actual new video frames and charge decoding. Online scoring reads raw
video/cache only; save acquired-image witnesses and metadata in the run output.
Any reusable derived cache is promoted separately into `data/m1_explorer_frames/`
with PROVENANCE; scoring never writes into data/. Runs/output/logs stay under runs/.
Initial estimate30–70GPUmin for333videos on5090, depending on acquisition rate
and video decoding; replace with fixed5video smoke before the full run. Peak
extra4small images/window should fit32G; verify actual longest-prefix smoke.
Full corpus remains on one machine, lab Slurm local partition only. Both raw
video directories verified on uoa-lab2 (HateMM/video, HateClipSeg/videos).

Independent code review must check native333baseline parity, no label path,
full pre-RoPE/GQA/norm coordinate oracle, source-index deduplication/time alignment,
actual image processor sizes, bounded deterministic acquisition, support and
entropy decision paths including confident wrong-signed examples, appended image
multimodal positions with existing KV, separate windows and round cache restoration,
exact original suffix wording, attention over original+new frames, missing speech,
18frames, all added image/call costs and canonical4fps evaluation. Actual36layer
32/8head Qwen FP32/BF16 tests plus real8B5video noGT smoke before full performance.
Smoke first2manifest videos per corpus plus HMM hate_video_114; not GT-selected.
No change to shared Judge unless a genuinely reusable interface is needed and
independently reviewed; no experiment-to-experiment imports.

## Decision and falsification

Main gate: final within+.01 on BOTH corpora vs native/current, no pooled loss>.005
or within loss>.01. No final main metric>=.01 gain =>archive; a qualifying
single-corpus gain allows at most3predeclared revisions under rule9.
If main qualifies, execute controls on both full corpora:
- Zero acquisitions: exact native reader restoration.
- Uniform candidate acquisition at the SAME per-video/window round counts as the
  main trace. This is a diagnostic with matched calls/frame counts, not a deployable method.
  It tests the attention-guided choice independently of new-pixel budget; the
  whole adaptive selection cannot be claimed merely because adding images helps.
  Precisely: with final main count c, choose targets a+(j+.5)*(b-a)/c for
  j=0..c-1. In this order select the nearest remaining eligible candidate by
  actual PTS, earlier PTS breaks ties. Replay the main per-round added counts
  using this list, no entropy re-gating. Only the final round margin is reported.
  This matches calls/frame budgets, not measured elapsed time. For all methods,
  use the same native image-processor settings. Verify that new frames of each
  video share the same processed grid/token count before interpreting this
  as a token-matched comparison; otherwise report the mismatch and do not make
  a compute-controlled mechanism claim until the discrepancy is resolved.
- Distance-only acquisition with A(t)=1, otherwise identical candidates, greedy
  distance updates and main replayed round counts. This isolates the attention
  prior within the declared selector; uniform comparison alone does not.
- Always acquire4 uniformly spaced new local frames, same cached-context interface:
  compares the full acquisition mechanism to a simple fixed-input baseline.
  Use the same quantile rule with c=min(4,number_of_eligible_candidates), and
  read after first2 (or fewer), then after the remaining images if any.
- Timestamp-mismatched local frames, matched final count: within each video,
  group windows by their actual final acquired-frame count c>0. For each group
  with k>=2, sort by window index and rotate donor image lists by ceil(k/2),
  retaining the receiver's original added timestamp slots and main round counts.
  Within each donor list retain its image order. For a singleton group or c=0,
  leave that window unchanged and mark it unmatched. Replay main round counts
  without entropy re-gating. This preserves acquired image multiplicities and
  per-window counts; it perturbs content/time association, not only timestamps.
  Report matched/unmatched window and video counts and effects separately while
  retaining the complete-corpus canonical comparison. No valid mismatch for a
  single-window video; do not claim evidence from that subset.

Claim a component only if its removal costs>=.01 in a common main metric on BOTH
corpora. Report all three final metrics, raw visual/speech/max order, native-global
correct/incorrect analysis (GT only after scoring), with/without original frame
support, acquisition counts, paired video bootstrap2000seed0, gain/loss content
cases, and per-corpus full cost. No causal claim from entropy or attention alone.
If matched uniform/fixed4 explains the gains, the acquisition mechanism is not
established; additional inputs alone are not a novel M1 under rule4.
The entropy gate is initially a bounded-compute implementation choice, not a
separate novelty claim: replayed counts do not independently validate its decisions.

Prior evidence read: frame-support diagnostic; native scoring code; earlier16
archived results including Stabilizer; primary EcoFrame and previous unselected
VideoTree/VTimeCoT methods. Preserver smoke only, no Preserver GT/performance yet.
No new GT was read for this proposal; any later error analysis is logged explicitly.


Independent proposal review PASS2026-10-03:
`docs/reviews/20261003_m1_explorer_proposal.md`. Implementation started after
that review, before Preserver performance was read. `explorer.py` implements
read-only all-layer Q/K capture, bounded deterministic frame selection, actual
PyAV PTS indexing and run-local image witnesses. `measure.py` reuses the native
context and explicitly recomputes complete multimodal positions before slicing
the appended block; it asserts cached prefix positions are unchanged. Native
replay after each expanded window is checked in smoke. CPU numeric and independent
code review are required before any real GPU scoring. Timing uses a matched
instrumented native baseline and explicitly says so; it cannot imply total
attention extraction overhead is free. Source indexing/decode is included in
Explorer timing even when the same raw input is reused across windows.

CPU selfcheck PASS: actual36-layer Qwen,32/8heads,128head dimensions,18/20native
images and2/4appended images, FP32/BF16. Captured all36pre-RoPE layers; cached
prefix positions and all KV entries restore exactly. Cached-vs-fresh last-state
max absolute difference <1e-6FP32 and <=.027344BF16 under predeclared .00002/.06
allclose tolerances. `runs/20261003_m1_explorer/selfcheck/numerics.json`. Actual
raw HMM1 input-only probe also decoded2845PTS entries, mapped28legal candidates
in the first8s and retrieved the selected source frames139/4 by exact PTS;
`runs/20261003_m1_explorer/input_selfcheck/`. No GT/scoring used for this probe.
Independent code review requested; real5video GPU smoke not launched yet.

Target uoa-lab2 HateVLM/transformers5.15.1 CPU selfcheck also PASS, returned as
`runs/20261003_m1_explorer/selfcheck_lab2/numerics.json`. The raw-input audit for
HCS bit_AxrVklzh9Cyf (the18-frame native cache) finds24931decoded frames with
last actual PTS207.75s versus manifest226.439002s; source report
`runs/20261003_m1_explorer/input_selfcheck_18/report.json`. Later windows have
no legal new visual candidates and therefore return the native read with an
explicit candidate_exhausted trace. No frame is borrowed from another interval,
no original input/GT/duration is modified. This is a noGT media-coverage fact.

The target input availability audit found all333 raw files and decoded the first
frame of every file (215 HateMM,118 HateClipSeg); no missing/failed entries.
Codecs:330h264,2libdav1d,1vp9. This is a first-frame availability check, not a
complete-media integrity claim. Source
`runs/20261003_m1_explorer/input_availability/report.json`, host sc474399,
returned locally before documentation; no GT was read.

Independent rule6 code review PASS2026-10-03:
`docs/reviews/20261003_m1_explorer_code.md`. An independent pre-RoPE numeric
oracle, cached/fresh appended-image comparisons, exact native replay, actual
source PTS decoding, full synthetic333 evaluator pipeline, and entropy/support/
candidate-exhaustion branches passed. Ready for the fixed5video real8B smoke;
this is implementation evidence, not performance or mechanism evidence.

Real8B smoke PASS on sc474399, Slurm56,2026-10-03. All5global margins and292
native branch margins exactly match historical base_gridA, every post-acquisition
native replay exact; actual source PTS/window/source-index checks pass. Returned
artifacts: `runs/20261003_m1_explorer/r1_smoke/plumbing_summary.json` and full
traces/images. Collection39.5s including native replay diagnostics; maximum
17.9454GiB. Acquisition reads per video4/5/9/33/59. First2manifest-video mean
extrapolation (excluding the added long-prefix stress video) estimates Explorer
369.1s HMM +733.3s HCS =18.37min, versus instrumented native188.2s+252.2s.
This tiny-input estimate is uncertain and replaces the earlier30–70min guess;
plan20–40min, report actual full-run timing. No GT or smoke accuracy was read.
Constants and production code unchanged. Proceed to full333 paired collection
using `sbatch experiments/20261003_m1_explorer/launch/lab.sbatch main` on lab2.

Post-scoring diagnostics added without changing the running reader:
`trace_diagnostics.py` reconstructs candidate sets from saved source PTS and checks
actual selection, then counts differences from distance-only at the SAME observed
history. This is a conditional trace diagnostic, not a performance control with
independently recomputed histories. Five-video smoke has35/110acquisition rounds
with a different selected set from distance-only; all new images within each video
share a processed grid. Source `r1_smoke/trace_diagnostics.json` under the same run
root; no GT. Do not infer selection quality from selection differences.
`case_analysis.py` is a post-scoring-only script reading canonical test GT after
full evaluation, with nominal support/acquisition masks, visual/raw/final branches,
native-global correctness and largest final gains/losses. These are descriptive
subgroups, not causal controls; no selected cases/constants feed back into R1.

Conditional control implementation is in `controls.py`, with
`analyze_controls.py` and `launch/control_lab.sbatch`. It does not modify the
running R1 reader or select new constants. The original four declared controls
rebuild native context independently and replay main counts without entropy
re-gating; the fixed4 arm instead always acquires up to4legal frames. Mismatch
uses original receiver timestamp slots and rotated donor content. All controls
retain exact native speech/global, read original visual once, and return only
the last expanded margin. Source decoding/image encodes are charged. Independent
review requested for this new path; no control GPU run has been launched. R1 full performance has not been read;
controls will run only if the main gate warrants the declared mechanism tests. Smoke uses r1_smoke traces;
formal controls use complete r1_main traces. `prepare` reports actual token
matching separately from image/call counts; no token-matched claim if it fails.
No new proposal constants or mechanisms are introduced.

Conditional controls independent code review PASS2026-10-03:
`docs/reviews/20261003_m1_explorer_controls_code.md`. No control reader correction
was needed. Two post-scoring report gaps were fixed before any control GPU run:
matched/unmatched window and video-group counts/effects are now separate, and
all decoded rate/duration/length/finite/global checks occur before GT access.
The reviewer verified these with synthetic333 cases, including18frames and an
unmatched video, plus actual36-layer FP32/BF16 tests for all four control readers.
This establishes implementation validity only. Full R1 is still collecting;
no real main/control accuracy has yet been read for this candidate.

## R1 complete: one-corpus improvement, not promoted (2026-10-03)

Host sc474399, Slurm57,333videos, collection1559.9s. All output returned locally
before this result/status update. Native333global/13939branch values and current
r6 metrics exact. Canonical source
`runs/20261003_m1_explorer/r1_main_decoded/explore/metrics.json`:

| corpus | pooled ROC | pooled PR | within (eligible videos) |
|---|---:|---:|---:|
| HateMM | .8976002308700018 | .6895375598692042 | .7548295590258155 (84) |
| HateClipSeg | .731874629967137 | .6839855741455672 | .658439650597689 (99) |

Relative to current/native, within +.0040475 / +.0210905; HMM PR -.0046970,
HCS ROC +.0150498 / PR +.0129135. HCS paired final CI [.002057,.043759]; HMM
[-.024510,.032065]. Main dual-corpus gate fails, but qualifying HCS gains permit
up to3 revisions under rule9. No control GPU scoring yet, no mechanism claim,
no promotion or Overleaf change. All evidence is development-selected.

Raw visual within .613908→.595101 HMM and .546450→.582562 HCS; raw max
.680011→.667500 and .610130→.634069. Speech unchanged. HMM's sole eligible
single-window video contributes +.007143 to its final mean, while the other83
videos contribute -.003095: the HMM final gain is not new within-window evidence.
HCS raw visual/max and final paired CIs exclude zero. Sources
`r1_main_analysis/{summary,branch_diagnostics,case_analysis}.json` under this run root;
all main numbers above are from the canonical metrics file, not these diagnostics.

Actual deployment-equivalent time including all selection/decoding/reads:
HMM758.684s, HCS794.850s (25.89min), vs instrumented native315.847/257.687s
(9.56min), ratio2.709. Peak17.945/17.672GiB. Mean forward counts47.088/82.322
vs36.521/60.051; additional acquisition reads2272/2628. New frames4538/5251,
image encodes including repeats5338/6163. Source `r1_main_analysis/cost.json`
and `r1_main/trace_diagnostics.json`. All new images within each video have the
same grid; selected sets differ from distance-only in1005/2272 HMM and1235/2628
HCS rounds, conditional on the actual history. This is not a mechanism control.

Post-scoring GT read log: `data/gt_4fps/{HateMM,HateClipSeg}.npz`, full R1/base
raw and decoded predictions, canonical metrics, traces, original ASR via
`src.video_inputs.load_asr`, and acquired PNG witnesses. The support-masked
HMM subgroup (34 videos with both classes in no-nominal-support frames) loses
.08515 raw-max within; CI[-.16884,-.01588]. Same mask HCS90videos is +.00025,
while its nominal-support subgroup improves +.01637 raw max. These masks change
the evaluated subset; they are descriptive, not alternative headline metrics.
Read top final losses HMM hate_video_45/151/304 and gains HCS bit_7EOOUGa9y9h4 /
yt_0Y_8MoLKn0I. Inspected actual PNGs for HMM45 index3908, HMM151 index9019,
HMM304 index267 and HCS0Y index7222. HMM151's GT-negative tail still visibly
contains a supremacist recruitment slogan/symbol, so its high new visual score
must not be called semantically wrong merely because this GT labels that tail
negative. HMM45 tail is an animated violent scene, not by itself evidence of
protected-group hate; HMM304 has persistent organizational branding. HCS0Y's
negative tail is a publication end card. We did not inspect every video or listen
to audio. These cases limit a simplistic 'new frames remove false positives' claim.

## R2 declaration: entropy-only acquisition (first revision, before R2 scoring)

R1's special no-nominal-frame override forces acquisition even when the native
window answer is confident. Missing a nominal frame is not a direct measurement
of missing semantic context: original full-video visual context and speech remain
available, and new frames may repeat persistent branding or static hateful imagery
that does not follow the temporal annotation. R1's support-group degradation and
some confident-negative overrides (HMM304 windows1/25, initial h .05470/.08631)
motivate testing this exception directly, without treating all bad cases as the
same cause. HMM45/151 also contain uncertain failures; this revision is not claimed
to resolve them or annotation disagreements.

R2 removes ONLY the no-frame override: for every window, stop immediately if
initial binary entropy h<.1, regardless of nominal frame support. Otherwise use
exactly the original attention/distance selector,2+2 acquisition, first-round
h<.3 stop, final-margin output, native context/global/speech and unchanged r6.
No threshold scan, new constants, new model or per-corpus branch. The same rule
applies to both complete corpora. The mechanism under test is bounded feedback
acquisition; the removed support exception is not retained as a contribution.

R2 can first be replayed EXACTLY from R1 caches: each window has an independent
restored native cache, and R2 only skips entire acquisitions of initial-confident
windows. For retained windows every image/prior/read/stop is identical to R1.
An explicit noGT replay script will record sources, complete333 coverage, the
removed-window counts and deployment forward counts; it will not fabricate
measured R2 time. This is full-corpus development evidence, not a new GPU run.
If it meets the performance gate, run the final R2 reader on all333 to verify
score parity and actual cost before claiming success or starting mechanism tests.
Expected fresh R2 cost <= R1's25.89min because it only skips reads/decoding; actual
runtime needs measurement. The replay itself uses CPU, no new MLLM calls.
The original uniform/distance/fixed4/mismatch controls remain the planned tests,
using the chosen final method's round traces; changing the gate alone is not a
new independent novelty claim. This consumes revision1 of the allowed3.

R2 replay implementation `replay_r2.py` and `launch/run_replay_r2.sh` write only
`r2_cache*` outputs; source R1 records are read-only. Replay records remove measured
R1 timing fields and explicitly record no new model calls. The deployment call
count is recomputed from retained acquisition rounds. Independent code review
requested before replay execution. Actual reader support is a single default-
compatible `support_override` flag, selected by `--version r2`; R1 remains the
default. Fresh R2 outputs are isolated in `r2_smoke`/`r2_main`, and the noGT prepare
step requires all fresh window values, curves and call counts to equal replay.
Slurm invocation, if cached performance qualifies:
`sbatch experiments/20261003_m1_explorer/launch/lab.sbatch smoke r2`, then `main r2`.
CPU fresh-run analysis uses `launch/run_analysis.sh r2`. No real R2 output exists
yet; all method constants were declared above before implementing this path.

## R2 cache result and R3 declaration (2026-10-03)

R2 replay and minimal reader code independently PASS:
`docs/reviews/20261003_m1_explorer_r2_replay_code.md`. Complete333 CPU replay and
canonical evaluation finished, with zero new model calls and no invented timing.
Source `runs/20261003_m1_explorer/r2_cache_decoded/explore/metrics.json`:
HMM ROC/PR/within .8979400346061241/.6901514840919151/.7523310911070054;
HCS .7303704475729428/.6830040006451092/.6539606639444846.
Within gains +.0015490 / +.0166116; HMM PR -.0040831, HCS ROC/PR
+.0135456/+.0119319. Paired within CIs [-.026309,.027596] / [-.001340,.036979].
Still not promoted; no fresh R2 GPU confirmation or controls, since dual-corpus
gate fails. Removed715/784forced windows and769/876acquisition reads. HMM raw
visual remains -.015528 below native, so removing the exception is insufficient.

Additional post-scoring decomposition reads the same canonical GT, raw R1/R2
predictions and R1 traces, and calls canonical within on four positive/negative
frame subsets according to acquisition status. The weighted contributions sum
exactly to full visual within delta. Source
`runs/20261003_m1_explorer/r1_main_analysis/read_format_diagnostic.json`.
HMM R1 loss -.018807 comprises -.005873 from comparisons across the two read
formats and -.012934 within acquired frames. R2 loss -.015528 comprises -.008690
across formats and -.006838 within acquired frames. HCS has positive contributions
in both categories. Acquisition status is selected/confounded: this is a rank
error decomposition, NOT proof of a causal input-format effect. It explicitly
rules out blaming all R1 degradation on mixing read formats.

R3 is the second revision, declared here before R3 scoring: ALWAYS acquire the
first two eligible local frames for every window, regardless of initial entropy
or nominal frame support. The original native visual read still supplies its
pre-RoPE frame prior. After that first read, preserve the original .3-nat stop;
otherwise acquire up to two more using the refreshed prior. Everything else
(candidate4fps/actualPTS, A^.5 times distance, earliest ties, cached full context,
original prompts/global/speech, final margin and r6) stays identical. Physical
candidate exhaustion still returns the last available read and is reported.
The initial .1 threshold is not used by R3, not scanned or retuned. Same rule on
both corpora. This is a bounded feedback reader with a common minimum local
input, not a new unrelated method or per-corpus routing.

Hypothesis: eliminating the mixture of native-only and expanded local input may
recover some cross-format ranking loss; it cannot by itself guarantee fixing
within-acquired errors, static hateful visuals or annotation disagreement.
The experiment may therefore fail. R1/R2 already-expanded windows must reproduce
exactly, while previously skipped windows need new reads. Full R3 will still be
run as one complete333 paired collection on lab2, not spliced into a claimed fresh
result. Initial estimate35–45GPUmin (native3+B plus at least one acquisition per
window and an optional second; max6V image encodes); replace with the same five
video smoke before full run. Reuse the model/original inputs, charge all new reads
and decoding. No threshold scan, added model, label fitting, score averaging or
changed downstream inference. This consumes revision2 of3; one revision remains.

If R3 passes the complete dual-corpus gate, the same predeclared control family is
still required using R3's own actual counts/traces. Uniform and distance-only
would test its selector; mismatched images would test time/content association;
fixed4 uniform would test whether a simple constant input budget explains its
gain. No entropy contribution will be claimed from matched-count replay alone.

R3 implementation adds only the explicit `always_acquire_first` flag and version
paths; configuration records `[null,.3]` entropy thresholds so the unused initial
.1 is not misrepresented as active. Existing R1/R2 defaults stay intact. R3 noGT
prepare additionally checks all R1-already-expanded windows: selected source
entries, each-round margins and final window values must reproduce exactly.
Fresh previously skipped windows are the only new reads. Independent narrow code
review requested before the original five-video GPU smoke and full333 run.

R3 narrow review PASS: `docs/reviews/20261003_m1_explorer_r3_code.md`.
The prepare audit now also reconstructs eligible first-round candidates from saved
source PTS and legacy exclusions: a legal candidate requires acquisition, while
zero legal candidates require an explicit exhaustion record. Independent tests
cover32 control-flow fixtures, complete333 synthetic prepare and the fixed5 smoke
set, including negative checks. No real R3 scores were read before this review.

R3 actual smoke completed on sc474399 (lab2), Slurm61,2026-10-03. All five native
globals and292 native branches exactly match the frozen reader; all98 windows
already expanded by R1 have identical selected entries, round margins and final
records. Post-acquisition native replays are exact. Source:
`runs/20261003_m1_explorer/r3_smoke/plumbing_summary.json`. Full smoke48.9s,
peak17.9454GiB. Extrapolation from the original two manifest videos per corpus
(excluding the added stress video): HMM491.90s, HCS1018.17s, total25.17min;
this replaces the preliminary35–45min estimate but remains a small-sample cost
estimate. No GT or performance metric was read for this plumbing run. Proceed to
the declared complete333 R3 collection on the same Slurm node/environment.

## Autonomous continuation, 2026-10-04

User renewed the instruction to continue until performance and mechanistic goals
are met. R1 is positive development evidence, especially on HCS; its HMM PR loss
of .004697 is inside the .005 pooled noise floor. It does not yet satisfy the
predeclared two-corpus within gain or establish a selection mechanism.
R3 full333 was submitted on sc474399 via Slurm job79, using the reviewed code
and prior five-video smoke. No constants, reader inputs or r6 changed on resume.

The predeclared controls now accept an explicit version to read R3 traces and
write isolated r3_controls/r3_controls_decoded outputs. R1 remains the default
with its original paths. This is path/config plumbing only: all four control
algorithms, budgets, prompts and canonical evaluator/r6 calls remain identical.
A narrow independent review is required before any real R3 control run. Controls
will run after the full-corpus performance gate; no new source recipe or tuning
is selected while the R3 main collection proceeds.

Version plumbing independent review PASS:
`docs/reviews/20261004_m1_explorer_version_code.md`. Complete synthetic333 cases
cover all three versions/four arms and case diagnostics; evaluator/r6 commands
were captured rather than run, and no real GT/performance was read for review.

Conditional controls can run independently as complete333 experiments on lab2,
lab3 or lab-server, using control_lab.sbatch/control_lab3.sbatch/control_server.sbatch
respectively. Each script fixes the target local partition, one GPU,4CPU/32G and
repository-only outputs. Existing HateVLM runtimes are reused; fresh target smoke
and native parity are required before any full control, not import success alone.
No dataset is split between machines. No control is submitted before the main gate.

## R3 complete and R4 declaration (2026-10-04)

R3 full333 completed on sc474399, Slurm79,2117.2s, and was returned to the
local repository before evaluation/documentation. NoGT prepare verifies all333
native globals/13939native branches exactly and4044R1-expanded windows exactly.
Canonical source `runs/20261003_m1_explorer/r3_main_decoded/explore/metrics.json`:
HMM ROC/PR/within .8973644548066193/.6889007126470338/.7577197411625319(84);
HCS .7350612119211078/.6856850467787075/.6632335093864848(99).
Relative to r6, within+.0069377/+.0258844; HMM PR-.0053339, HCS ROC/PR
+.0182363/+.0146130. Within paired CIs[-.022153,.035225]/[.002129,.052897].
The two-corpus gate still fails. Native r6 exactly reproduced; no controls or
mechanism claim yet. R3 is further positive HCS development evidence.

Deployment-equivalent reading time, including source indexing/decoding/selection:
HMM1068.975s/HCS1041.455s(total35.174min), vs native315.964/257.855s, ratio3.678.
Mean outer forwards56.279/95.288 vs36.521/60.051; new frames8489/8311,
image encodes including repeats9469/9455; peak17.945/17.672GiB.
Sources `r3_main_analysis/cost.json`, `r3_main/trace_diagnostics.json` under the
same run root. Original20-frame/ASR inputs are reused here; their extraction
remains a common new-video prerequisite, not a free deployment input.

Post-scoring read log: canonical GT arrays, complete R3 raw/decoded/base
predictions, metrics, alignment and traces; `r3_main_analysis/case_analysis.json`.
HMM raw visual within .613908→.587017 and raw max .680011→.656391;
HCS visual .546450→.588579 and raw max .610130→.629204. HMM's sole eligible
one-window video contributes+.0071429 to final within; the other83contribute
-.0002052. Uniform minimum local input did not establish new HMM localization.
HMM losses151/304/45 again have high expanded visual scores near GT-negative
tails; e.g.45last window visual-1.804→11.551 with no speech read. The prior
image inspection for151 found real supremacist content there, so this tail must
not be described as semantically non-hateful merely from GT. R3 analysis does
not itself inspect all new image semantics or prove a self-conditioning bias.

R4 is revision3of3, declared before implementation/scoring. Retain R3's native
global, native speech, native initial visual/prior, first2local-frame acquisition,
.3-nat continuation, actualPTS candidates, attention/distance selector, final
margin and r6. Change ONLY expanded-read conditioning: ask the exact original
visual question with all acquired local frames on the original observational
prefix (policy,20frames,full ASR) BEFORE the model's global question/answer
dialogue. The native global question/answer remains for native reads and M4,
but is absent from expanded local reads. No alternate Yes/No, new prompt wording,
score averaging, per-corpus rule or model. No-label context choice is fixed in
both corpora. Candidate exhaustion returns the original native read.

Hypothesis: an acquired visual observation may be judged with less reinforcement
from the model's own earlier global answer. This may improve ranking in cases
where generic imagery inherits the earlier verdict; it cannot resolve annotation
disagreements or guarantee correcting the global judgement. The matched older
Reader analysis `experiments/20261002_verdict_analysis/README.md` motivates testing
this conditioning with actual new local pixels, not treating its older numbers
as the latest-reader ablation or proving a causal Yes/No-token effect. The changed
treatment is the ENTIRE self-verdict dialogue, as in that analysis.

Implementation uses an independently restored pre-verdict KV prefix for expanded
reads. All cached-prefix mRoPE positions must equal a fresh observational-prefix
render. Original native cache/global/speech and post-acquisition native replays
must remain exact. A second prefix copy costs memory/copy time and is charged;
there is no extra language or vision forward from making the copy. Calls remain
3+B+sum(R_w); budget at most4new images/2expanded reads per window and6image
encodes including the second-round repeats. Estimate35–40GPUmin for full333,
replace with the same fixed5smoke. No constant scan, ensemble or new dataset.

The same four declared complete-corpus controls use R4's own conditioning and
traces. Matched uniform/distance/fixed4/mismatch still distinguish budget,
selection and content/time association. If R4 passes, an additional treatment
control restoring the self-verdict at R4's acquired sets/counts can test this
context hypothesis without attributing all differences from R3 to one token;
declare and review its exact replay before use. The context choice is not a
standalone novelty claim. Independent proposal-delta review before implementation,
then a narrow code review and fresh fixed5smoke/full333. If it still fails the
performance/required mechanism gates, record the best R1/R3 evidence and archive
this family; continue another complete method under the autonomous goal.

R4 proposal-delta review PASS:
`docs/reviews/20261004_m1_explorer_r4_proposal.md`. The original, stricter Explorer
gate remains final within >=+.01 on BOTH corpora, relative to native/current,
and pooled losses <=.005. R3-to-R4 differences are the strategy's total effect:
first selected pairs stay fixed, but changed first-round read/prior/entropy may
change the second round. They do not isolate a context effect at fixed pixels.

R4 implementation, awaiting independent code review: lazily deep-copy the native
KV cache at its restored full length, then crop only the copy to the observational
length P. Recompute positions from the original observational token IDs/grids;
check they equal the first P positions of the native prefix. Capture/restore the
observational rotary delta independently. Copy/preparation time is charged to
acquisition and a separate per-video field reports it. Native cache is never
cropped to P and expanded suffixes never remain between windows/rounds.
In the fixed5 smoke, for each cumulative new-image count first encountered in a
video, encode a fresh full observational-plus-local rendering with all original
and new images. Its token IDs/grids must equal the cached-prefix-plus-suffix
encoding; all expanded reads verify unchanged prefix positions. These extra
processor-only smoke diagnostics are charged to smoke acquisition time (making
its deployment extrapolation conservative), add no model forwards, and do not
run in the full collection. Independent CPU model tests must also check actual
cached-vs-fresh hidden outputs and KV isolation; a textual render check alone is
not evidence of equivalent model computation. No R4 performance is read before
these checks and the target smoke.

Independent R4 code review PASS (2026-10-04):
`docs/reviews/20261004_m1_explorer_r4_code.md`, evidence under
`runs/20261003_m1_explorer/r4_review/`. Actual 36-layer Qwen CPU tests cover
FP32/BF16,18/20 original images,1/2/3/4 new images, independent clone/crop,
cached/fresh hidden outputs and positions, native exact replay, main/four controls.
Synthetic333/fixed5 prepare,32 decision cases,13 rejection cases and version/CLI
paths pass; canonical evaluator/r6 commands were captured with unchanged flags.
This CPU environment is torch2.7.1/transformers4.57.6, not target5.15 or actual8B
weights. Target fixed5 smoke remains required before full333. No GT/R4 performance
was used in this review; no evaluator, r6 or reader constants changed.

R4 fixed5 target smoke completed on sc474399, Slurm86,2026-10-04:48.1s,
peak18.3231GiB. Returned locally before noGT prepare. All5 native globals and292
native branches exact; all158 initial acquisition sets and initial priors equal
R3. All post-acquisition native replays exact. Actual 8B/torch2.11+cu128/
transformers5.15.1 full-token/image-grid seam and cached-prefix position checks
passed, including real cumulative1/2/4 images. No extra full-8B fresh-hidden
forward was performed; CPU cached/fresh numerical evidence remains as reported
above. Prefix copy/preparation measured .01891s summed over5videos, charged.
Source `runs/20261003_m1_explorer/r4_smoke/plumbing_summary.json`, checks and
details. No GT/metrics were read for smoke. The first2manifest videos/corpus
extrapolate HMM525.73s/HCS958.72s,total24.74min; this is a small-sample estimate
with processor diagnostics included, and R3's analogous estimate underestimated
its full collection by about10min. Retain35–40min as a practical budget pending
actual R4 time. Proceed to unchanged complete333 collection on the same target.

## Conditional fixed-trajectory context control, declared before R4 results

Declaration2026-10-04, while Slurm87 collects full R4 and before any R4 metric/GT
analysis. If the main gate passes, run `verdict_replay` on complete333 and the
same fixed5 smoke; paths `r4_controls[_smoke]/verdict_replay` and
`r4_controls_decoded/verdict_replay`. It is a control, not revision4 or a new
method. Obtain the unchanged native prefix/global answer/visual/speech. For each
window use exactly the R4 trace's round counts, added source entries, cumulative
chronological sets, actual timestamp strings and original visual question.
Replay every declared round on the ORIGINAL full native cache with its global
question/answer dialogue, rather than R4's observation-only cache. No re-gating,
re-selection, changed answer, alternate prompt, R3 margins, score averaging or
label input. Return the final replay margin; zero-round windows return the
unchanged native read. Original global and speech remain exact. Reject use on
R1/R2/R3. All tokens/grids other than the removed/restored dialogue must match
the R4 acquired observations at fixed sets. Native replay checks still required.

Budget matches the realized main `3+B+sum(R_w)` and cumulative image encodes;
it needs no second observational cache. Reuse original20-frame/ASR inputs and
R4 trace entries; pixels are re-encoded for every replay. Charge actual source
indexing/decoding/reading time, estimated35–40GPUmin total. Shared canonical
evaluation and the same independent full-corpus r6 fit apply. Compare all three
final metrics and paired raw visual/max/decoded within; final comparisons include
changed r6 fitting and must not be described as solely a hidden-model effect.
This estimates the effect of the ENTIRE dialogue treatment conditional on R4's
chosen trajectories, including changed positions/conversation structure, not a
Yes/No-token causal effect or an effect averaged over arbitrary policies. It does
not establish acquisition novelty. The four already declared acquisition controls
still use R4 observation-only conditioning and its own traces. Independent narrow
review of this fixed replay before implementation/use, then target smoke, is
required. No control GPU job is submitted before R4's full performance gate.

Fixed-trajectory proposal and narrow code review PASS:
`docs/reviews/20261004_m1_explorer_verdict_replay_{proposal,code}.md`, evidence
`runs/20261003_m1_explorer/verdict_replay_review/`. Actual CPU replay and old-four-
arm compatibility,333synthetic analysis,negative-version/trace checks and CPU
launcher pass. An independent derived CPU Judge supplies cache_position for
old4.57 native incremental calls; this is a test-runtime adapter, not a production
change. Target5.15.1 actual source1108–1141 uses cache.get_seq_length() directly
when incremental native calls omit attention_mask, so no production adaptation
is needed. CPU evidence does not replace target8B control smoke. Main R4 code
and its running collection are unchanged. The conditional GPU gate remains.
