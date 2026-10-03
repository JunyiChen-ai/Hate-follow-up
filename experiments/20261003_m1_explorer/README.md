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
but decode actual new video frames, charge decoding and store new cache under
`data/m1_explorer_frames/` with PROVENANCE. Runs/output/logs stay under runs/.
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
  main trace. This is a diagnostic with matched compute, not a deployable method.
  It tests the attention-guided choice independently of new-pixel budget; the
  whole adaptive selection cannot be claimed merely because adding images helps.
- Always acquire4 uniformly spaced new local frames, same cached-context interface:
  compares the full acquisition mechanism to a simple fixed-input baseline.
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

Prior evidence read: frame-support diagnostic; native scoring code; earlier16
archived results including Stabilizer; primary EcoFrame and previous unselected
VideoTree/VTimeCoT methods. Preserver smoke only, no Preserver GT/performance yet.
No new GT was read for this proposal; any later error analysis is logged explicitly.
