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
