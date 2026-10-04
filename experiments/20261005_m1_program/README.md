# M1 candidate23 deferred proposal: executable temporal evidence program

Declared 2026-10-05, before implementation/GPU/GT analysis. Original unfiltered
candidate: executable_temporal_evidence_program in
experiments/20261004_m1_ideation/CANDIDATES.json (jury rank5). Deferred source/proposal
preparation while Tree21 runs; Lattice22 remains the next reviewed backup.
Execution update2026-10-05: independent fixed5 plumbing may overlap Lattice22 R3
on the idle lab3 GPU; no complete Program main before the Lattice result branch.
All Program acquisition/scoring stays together onsc474398, with identical code,
constants and fixed5 as previously declared. This scheduling change does not use
Lattice test predictions or modify this candidate's scientific specification.
Formal method is still r6_bma; all future results development-selected.

Execution host selectedsc474398/uoa-lab3 on2026-10-05. Full333 current source
inventory/20cached frames CPU preflight PASS, noGT/GPU; readable evidence
`runs/20261005_m1_program/lab3_source_preflight.json`. All laboratory code synced
and clean/current partition idle/1.4Tfree; scoped machine check and unchanged
foreign-home note in `runs/20261005_m1_program/machines_before_smoke{,_note}.txt`.
Slurm130 submitted the originalfixed5 via committed `launch/lab3.sbatch` (only
partition differs from reviewedlab2 wrapper); acquisition and paired scoring
stay together onthishost. Actual8B/schema/clone/native parity pending; no main
or performance evaluation/GT use at this point.

Actualfixed5 Slurm130 completedsc474398 at08:00:22. Both complete run and derived
program inputs returned immediately to local paths. Acquisition/schema binding,
native and cloned reads completed, but the declared noGT `analyze.py --stage
prepare --smoke` FAILed at `mechanism not exercised in this corpus smoke`:
both corpora have0actual fresh module calls. Authority error log
`runs/20261005_m1_program/r1_full_smoke_analysis/run.log`; actual acquisition
records `data/temporal_evidence_program/{HateMM,HateClipSeg}/<video>.json`, host,
prompt/tokens/parse/execute records included. Complete Program main is NOT
launched; no performance metric/GT/idea verdict. The guard is unchanged.

Observed generated interface violations include segment argument `"s0"` instead
of integer0 and timestamp94.94 in a character-offset slot; programs with21
frame/emit operations exceed declared12; missing requested windows, invalid
top-level arrays and one truncated chunk. Strict execution records UNKNOWN and
does not coerce invented IDs/coordinates or fabricate perception. An independent
narrow diagnosis is requested to distinguish real generator/interpreter/cache
bugs from correctly rejected noncompliant model outputs before any repair. No
scientific constants/prompt/parse contract have been changed after this smoke.

Independent narrow diagnosis froze
`docs/reviews/20261005_m1_program_gpu_interface_diagnosis.md`: no observed
generation/cache/interpreter binding bug. Actual22chunks/158windows current
source/prompt/image/input-output tokens and execution replay PASS; compliant
source fixture triggers both scope/action, actual targetHF5.15.1 increment
position branch verified onCPU. Raw output interface noncompliance is the
observed cause of0modules. Guard remainsFAIL; no coercion/repair/cap relaxation
or main. Explicit new source/decoder design is needed; candidate24 proceeds
independently while this candidate has no performance/idea verdict.

## Mechanism, source and scope

A program selects source objects and composes factual perception through explicit
temporal/scope dependencies; the interpreter executes these operations on actual
media and transcript spans. It does not merely generate a completed explanation
or check whether generated JSON is syntactically valid. Fresh factual module
outputs are not moderation predictions. Exactly one final moderation measurement
per available native branch enters fixed r6.

Primary source: [ViperGPT](https://viper.cs.columbia.edu/),
[paper](https://arxiv.org/pdf/2303.08128) sections3.1-3.3,4.4 and AppendixB.
Actual downloaded paper/text: runs/20261004_m1_ideation/backup_source_reads/program/.
Original composes pretrained perception modules with generated unrestricted Python;
VideoSegment exposes trimming/ordered frames and actual execution. Official
main_batch.run_program and video_segment.{trim,frame_iterator,select_answer} read.
Here the restricted interpreter, exact ASR-span objects, quotation scope and
same-Qwen module reuse are explicit self-designed adaptations; not exact original
reproduction or a claim of original numerical effectiveness on this task.

Primary target comparisons: MARS2.2 generates factual description and competing
hate/nonhate reasoning; MATCHIII-C retrieves localized multimodal evidence by
CLIP similarity and uses an LMM verifier, III-D trains a rationale-enhanced
predictor; RAMF3.2 uses adversarial reasoning then learned fusion; LEAF3.2 uses
label-corrected explanation grounding and distillation; IARE4.2-4.3 uses annotated
context/SFT and DPO. Generic CoT, source references, captioning, retrieval and
verification are already present. Narrow proposed delta: actual generated
temporal/source/scope programs whose dependencies control fresh perception and
the local evidence record. No opposing hate hypotheses or decision averaging.
MAESTRO's accessible official appendix also executes generated tool requests
(including DeepFace/YOLO) and iteratively updates the content representation.
Tool orchestration/fresh perception in hateful video is therefore already known;
neither is claimed as a first. Its full paper access was limited during review;
the independent report records the accessed sections and remaining scope limit.

A search snippet initially appeared to place visual program distillation under
video hate. Actual authors' 2025 slides4/9/12 and paper2502.07138 separate VPD
meme methods from video embedding/order fusion. This is a scope correction, not
a novelty verdict. Source scopes/queries: same run directory/source_scope.json.
Independent rule4 review must search complete target-domain program use.

## Common R1 specification

Both full HateMM215 and HateClipSeg118; same frozen Qwen3-VL-8B, same native20
overview/full ASR,8s windows and canonical4fps/fixed r6. Recollect paired native.
Global/native stance remains exact; only the two local branches are remeasured
with executed records. No labels, GT, prior predictions/metrics enter any source,
planner, module, interpreter, scoring or threshold. No score blending.

Source table uses native frame indices with the nominal times parsed by
src.video_inputs.frame_paths from cache filenames, and native ASR
segment indices, actual text and start/end. Text spans are half-open character
offsets into one segment. Their interval uses the declared proportional character
allocation within the native segment; this is not newly measured word timing. Entire native window body
remains available to the final reader. Inputs are read-only and existing caches
are reused; no new encoder, detector, OCR model or video pixels in R1.
Frame coordinates retain the native seek/fallback provenance; they have not been
verified as exact decoded presentation timestamps. Temporal operations below use
these declared nominal coordinates, not a claim of precise media timing.

Planner is the same Qwen on a factual-only source prefix: overview/frame IDs,
indexed full ASR and a neutral system instruction. Excludes moderation rules,
global verdict and any previous local answer. Process consecutive nonoverlapping
chunks of8 windows, greedy max2048 new tokens/chunk, seed0, no retry or alternate
program voting. Save raw output and parse failures. Parse exactly one JSON array;
invalid/truncated output yields explicit UNKNOWN plan records for those windows,
not fabricated successful execution. This behavior is charged and audited.

Frozen literal planner system:
```text
You write source-grounded evidence programs, not moderation decisions. Use only the supplied indexed video sources and the API below. Select observable sources, then compose temporal and discourse evidence. Do not classify hate, supply confidence scores, invent sources, or output a rationale. Return exactly one JSON array and no other text. Include each requested window exactly once. Use an empty ops list if no program is supported.
```
Frozen literal API/schema block (user text preceding the source table):
```text
All IDs and offsets are zero-based integers. Source times are nominal cache coordinates; transcript character times are proportional estimates, not measured word times. Character offsets count Unicode characters in the supplied original segment text. Windows and character spans are half-open. Use at most 12 operations and 2 perception calls per window. Variable names must be unique; arguments may reference only earlier variables in that window. Context interprets a local utterance and does not establish a local occurrence. Output schema: [{"window":integer,"ops":[operation,...]},...].
API:
["span",name,segment,start_char,end_char] -> exact original transcript substring and its proportional time interval.
["frame",name,frame_index] -> the original cached frame and nominal time.
["local",name,input_name] -> fully contained characters of a span in the requested window, or an in-window frame; otherwise UNKNOWN.
["context",name,span_name] -> interpretation-only transcript context, at most 2 spans, first 96 characters each.
["scope",name,local_span_name,[context_names]] -> fresh factual reading of the local utterance with those contexts: speaker role, mode, exact target span reference, and support.
["action",name,local_frame_name] -> fresh factual reading of the selected local frame: visible actor, action, target, and support.
["join",name,span_name,frame_name] -> paired local witnesses only if span.start <= frame.time < span.end and both are in the requested window; otherwise UNKNOWN.
["emit",[names]] -> existing values and their dependencies; use at most once, as the last operation.
Do not use other operations or add decision fields.
```
The source table is canonical JSON with keys frames (id,time,time_source),
segments (id,start,end,text,time_source), requested_windows (id,start,end).
Images precede this table, each labeled with its frame id and nominal time.
Time-source values are respectively native_filename_nominal and
native_asr_segment_proportional_chars. The user turn ends with literal
"Return programs for requested_windows using only the supplied sources and API."
No few-shot program, previous chunk output, or hidden decision instruction.
Missing requested windows yield UNKNOWN. Duplicate window IDs yield UNKNOWN
for that window (all duplicate plans rejected); out-of-chunk/noninteger window
objects are rejected and recorded. An empty ops list yields an explicit empty
record. Failure to parse exactly one array yields UNKNOWN for the entire chunk.

Each returned object is {"window":integer,"ops":[operation,...]}.
At most12 operations/window, at most2 perception operations/window.
Only these typed operations, in topological order with earlier variable references:

- ["span",name,segment,start_char,end_char]: materialize exact original text and
  proportional interval. Invalid offsets/type/reference -> UNKNOWN.
- ["frame",name,frame_index]: materialize original image and nominal timestamp;
  missing/out-of-range -> UNKNOWN.
- ["local",name,input_name]: temporal intersection with that object's destination
  window. A span clips to its actual overlapping characters; a frame outside the
  window yields UNKNOWN. No timestamp rewriting or evidence of absence.
- ["context",name,span_name]: designate an exact interpretation-only span from
  anywhere in the same video. At most2 context spans, first96 characters each.
- ["scope",name,local_span_name,[context_names]]: fresh same-Qwen factual module.
  Receives exact materialized local text and the chosen true context spans; no
  draft answer, moderation rules/verdict, previous module answer or free-text
  planner rationale. Returns speaker role, quotation/rejection/reporting mode,
  a target span reference or UNKNOWN, and supporting source references.
- ["action",name,local_frame_name]: fresh same-Qwen factual module on the actual
  selected in-window frame. Returns short actor/action/target description with its
  frame support. Outside-window/missing source -> UNKNOWN without model call.
- ["join",name,span_name,frame_name]: the frame is a time point, not a
  zero-length interval. Return paired witnesses only if
  span.start <= frame.time < span.end and both sources are local to the
  destination window on the declared native coordinates.
  Otherwise UNKNOWN; it never treats distant context as local occurrence.
- ["emit",[names]]: return only existing typed values and source dependencies.
  At most one emit, last. No arbitrary code, imports, loops, user arithmetic,
  file/network primitives, confidence scores or hate-class fields.

Before execution, cap each plan's operations to12 and perception operations to2
by retaining the first topologically valid entries; save all rejected entries.
All local operations refer to the destination window. No cross-video references.
No automatic fabricated span/frame fallback. The interpreter actually fetches
the referenced original text/images, evaluates temporal intersections and executes
the requested scope/action functions; each returned value carries its source
dependencies. Model-supplied support references must exist among that call's
actual inputs, and a returned target span must parse exactly. Unsupported fields
become UNKNOWN. Valid provenance does not guarantee factual semantic correctness.

Exact character clipping: for original segment [s,e), length L Unicode characters,
offset interval [a,b), retain [a',b') where
a'=max(a,ceil((window.start-s)*L/(e-s)-1e-9),0) and
b'=min(b,floor((window.end-s)*L/(e-s)+1e-9),L).
The 1e-9 tolerance is in character-index units and only stabilizes arithmetic at
integer boundaries. Return UNKNOWN when b'<=a'. Retained interval is
[s+a'*(e-s)/L,s+b'*(e-s)/L), with original offsets saved; there is no word-boundary
rounding or source-text rewriting. context truncation to96 characters updates
the saved end offset and proportional interval. All offsets/IDs reject booleans.

scope/action each use greedy max96 new tokens, same system instruction:
"You describe only the supplied source content. Do not classify hate or infer
unsupported facts. Return only the requested JSON; use UNKNOWN when uncertain."
Scope question:
"Identify the speaker role (speaker/quoted/reported/UNKNOWN), mode
(direct/quoted/rejected/reported/UNKNOWN), the exact target span reference if
present, and supporting source references for this utterance. Interpretation
context is not an additional local utterance. Return keys speaker,mode,target,support."
Action question:
"Describe the visible actor, action and target in this frame. Return keys
actor,action,target,support. Keep each description at most twelve words and
support only by the supplied frame identifier; use UNKNOWN where not visible."

Fresh module calls use only their materialized source packet and neutral system,
not the planner answer or native global/stance cache. Inputs/outputs/real calls/time,
physical images/tokens, emitted dependencies, invalid/UNKNOWN fields are saved.
There is no claim that speaker identity can be established from ASR alone:
speaker roles refer to discourse attribution, not biometric identity.
Module support entries are the input variable names in that call. A scope target
is {"segment":integer,"start_char":integer,"end_char":integer}, contained in one
of its actual input spans; unsupported targets become UNKNOWN. Scope sources
remain typed local/interpretation-only. The action source contains its exact
frame index. Generation uses a fresh ordinary Qwen cache, FP32 lm-head greedy
argmax, no sampling, stops at native EOS or the declared token cap. Truncated
planner output yields UNKNOWN even if its last token completes a JSON array.
Final JSON is UTF-8 canonical (sorted keys, compact separators); no generation
or fact measurement is hidden in a reused native cache.

Final paired reader builds exact native global/own hard-stance cache once. Each
native local branch independently appends "Executed local evidence (uncertain
measurements, not a moderation decision):\n" plus canonical JSON of its emitted
record, then the exact native yesno_question. Visual uses action/join frame records
only; speech uses span/scope/context records only. Full native interpretation
prefix remains visible; no hard isolation claim. Missing speech remains absent
according to native availability. Crop/restore original cache/delta after every
branch. One FP32 Yes/No margin/branch, raw max(new visual,new speech), unchanged
r6 and canonical evaluator. No tree score, program numeric score or module verdict
enters the method. Uniform behavior on both datasets.

## Cost and checks

Additional work per new video with W windows: ceil(W/8) planner generations,
0-2 factual generations/window according to requested valid sources, and normal
local rereading. No new media decoding/features; native overview/ASR reused.
All planner/module generation work is included in standalone new-video time,
including invalid outputs. Native/global and reference-local collection are
accounted separately. Preliminary full333 estimate120-360 GPUmin, unmeasured;
the original candidate45-100min estimate omitted realistic generation work.
Measure actual five-video cost and declare any revised runtime estimate before
full launch; never call this free cached inference. No time-based truncation or
automatic candidate rejection. Slurm only; one whole corpus/host.

Meaningful CPU checks before independent rule6 review: exact offset/time clips,
typed dependency execution, a synthetic quotation target bound across contexts,
UNKNOWN vs absence, wrong-window/unsupported-source rejection, and fresh-branch
cache isolation. Fixed five noGT GPU smoke: first2/corpus plus HateMM114; exact
native global/windows/curves, actual nonempty execution in each corpus, independent
cache replay, and full input/source/call/coverage accounting. Format failures are
visible; no labels or numerical performance selection in smoke.

Main gate: same main metric>=.01 BOTH, no pooled loss>.005/within loss>.01,
all three final metrics with within eligible84/99. None anywhere>=.01 -> archive;
a qualifying gain permits at most3 predeclared result-guided revisions. Full
333 prediction/inputs returned local before STATUS. No evaluator changes.

If main passes, full333 controls (not implemented):
unexecuted program in final reader; matched planner/module budget with ordinary
nearest-context perception rather than generated dependency execution; replace
typed source/scope composition with flat identical perceptual outputs; rotate
window/argument dependencies floor(W/2) retaining donors' true source coordinates;
remove scope or temporal join if separately claimed. Perception outputs and token
budgets must be accounted rather than treating unequal added calls as proof.
Complete-program claim requires a dual-corpus common-metric .01 advantage over
matched perception/flat execution, plus wrong-binding falsification; each
separately claimed operator must pass rule14g or be removed/demoted. Audit actual
source corrections, raw visual/speech/max ordering and paired video bootstrap;
valid syntax/provenance alone is not mechanism evidence. Independent final
novelty/mechanism review required before completion. No scientific result yet.

## Implementation and deferred execution

CPU implementation: program.py is the typed interpreter and literal material;
inputs.py binds native source inventory and deterministically replays saved
executions; extract.py performs fresh planner/module generations with full
input/token/grid/source records; measure.py appends the executed modality record
to one native independent branch. Shared native cache is src/stance_cache.py.
analyze.py only orchestrates canonical evaluator/fixed r6 and imports the
canonical within-video helper. No inter-experiment import or evaluator rewrite.

23 independently enumerated character-cell cases and a synthetic quotation
packet pass, together with half-open point joins, duplicate/missing/truncated
plans, unsupported targets/support, UNKNOWN, exact input replay and literal
planner checks. Evidence runs/20261005_m1_program/cpu_checks/selfcheck.log.
Actual five native inventories/JPEG parsing/CPU processor rendering also pass:
runs/20261005_m1_program/cpu_source_preflight/summary.json. These are input and
interpreter checks, not measured semantic accuracy or native GPU parity.
Single independent rule4 PASS: docs/reviews/20261005_m1_program_proposal.md,
same-family provisional, including MAESTRO overlap/access limitations.
Independent rule6 code review PASS: docs/reviews/20261005_m1_program_code.md.
Independent1500 character-cell oracle, real source packet/perception cap, fresh
generation FP32/EOS/call checks and normal/exception native-cache restoration pass.
Evidence runs/20261005_m1_program/code_review/independent_checks.log.
Actual Qwen GPU native parity/semantic correctness remain untested; no GPU/GT/result yet.

Conditional commands, only after Tree/Lattice result-based branches permit:
```bash
sbatch experiments/20261005_m1_program/launch/lab2.sbatch smoke
# Immediately return run outputs AND data/temporal_evidence_program from lab2.
python experiments/20261005_m1_program/analyze.py --stage prepare --smoke
# Only after exact native/source/call checks pass and runtime estimate is recorded:
sbatch experiments/20261005_m1_program/launch/lab2.sbatch main
# Immediately return outputs/inputs, then local detached CPU analysis:
setsid nohup bash experiments/20261005_m1_program/launch/run_analysis.sh > runs/20261005_m1_program/analysis_launcher.log 2>&1 </dev/null &
```
These commands are not yet dispatched. GPU target/provenance: sc474399/uoa-lab2
local-sc474399 Slurm, whole333 acquisition then paired reader; all actual source
metadata/run logs record generating hostname. First line of each run.log is host.
Canonical input cache: data/temporal_evidence_program, with PROVENANCE.md generated
by extract.py; inputs never include labels. Outputs r1_extract_{smoke,main},
r1_full_{smoke,main}, r1_full_main_decoded and corresponding analysis directories.
