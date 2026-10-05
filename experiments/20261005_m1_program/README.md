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

## R1 source-handle interface B, declared before implementation/GPU

The unchanged free-JSON interface A is retained in its original files and cache.
Its actual22chunks/158windows/zero module calls remain a failed noGT result, not
an idea-performance verdict. Diagnosis found no implementation bug: this is an
explicit new source/planning/decoder design, not repaired/coerced old output.
No A/B performance comparison or GT has been seen. Interface B tests the SAME
complete restricted source-execution method with a legally executable planner.
It does not itself supply novelty or count as an extra result-guided revision.

Native20 frames, full ASR,8s windows, source proportional character times and
nominal frame times remain unchanged. Prebind one candidate for each real ASR
segment's fully contained character clip in a window (the exact original `local`
operator), all actual in-window frame IDs, and all other segment spans as
interpretation-only context. Handles S<zero-padded8digit segment index> and
F<zero-padded8digit frame index> only alias actual source coordinates, not
invented entities. No lexical target/policy filtering or GT. The planner sees
original20 images, full source inventory and the actual requested window source
choices. One fresh planner per8windows as A, but constrained greedy decoding
selects one local span when any exists, one local frame when any exists, and zero
to2 DISTINCT remote context segments when speech exists. NONE is forced only
when no actual local source exists; STOP is always allowed for context. Join is
selected true/false only when the actual two local witnesses overlap; otherwise
false. Source/time aliases are compiled to ordinary A span/local/context/scope/
action/join/emit instructions, never repaired from an invalid raw program. A
12-operation/2-perception/2-context maximum still applies: two contexts plus
scope+action+join+emit use12operations. Local spans and module outputs, plus a
selected actual join, are emitted; context alone is never a local occurrence.

Literal output skeleton per window is a JSON object with keys window,speech,
frame,contexts,join, in that order. Array/window keys/punctuation are forced,
while handles, STOP and legal booleans use FP32 next-token argmax ONLY among
legal continuations of the literal choice trie. All forced tokens are actually
forwarded, counted, and saved; there is no hidden retry, beam, voting or free
output repair. Choice strings include their closing punctuation so no shorter
choice is accepted by bypassing a model decision. At most2048 total planner
tokens, same seed0. Incomplete/capped plans retain UNKNOWN, never fabricated
perception. Identical constants and exact frozen grammar for both corpora.

Factual modules still use fresh source-only Qwen branches, max96total tokens.
Scope JSON skeleton keys speaker,mode,target,support: speaker in
speaker/quoted/reported/UNKNOWN; mode direct/quoted/rejected/reported/UNKNOWN;
target UNKNOWN or one of the actual local span's contiguous1–4 whitespace words
(with original Unicode character boundaries). This is a bounded target proposal
set, not certified semantic ownership. Support contains the actual local source
ID only; it records the measured source, not evidence that its interpretation is
correct. Scope/target values are constrained greedy choices. Action keeps free
actor/action/target descriptions (<=12words per field) and actual frame support.
JSON syntax is forced; a field greedily generates only tokens decoding to valid
JSON-string content without quote/backslash/control/replacement characters,
plus the closing quote. At24content tokens or12words the closing quote is forced.
Forced punctuation is measured; field truncation is recorded separately from
whole96-token cap. Uncertain semantics may be UNKNOWN; grammatical validity is
not factual accuracy. No moderation words/decision/score fields in these modules.

The original A interpreter executes compiled B instructions unchanged. At most
2 actual perception generations/window; no alternative program aggregation.
Final independent native visual/speech questions receive the factual executed
records exactly as A; native global/own hard stance and fixed-r6 remain frozen.
All source choices, compiled ops, allowed option sequences, forced/generated
actual IDs, cap/field-stop events and actual forward counts are saved for replay.
Separate version `R1 source-handle interface B; sources2026-10-05`, cache
`data/temporal_evidence_program_handles/`, outputs `r1_handles_extract_*` and
`r1_handles_full_*`; original A outputs are not reused/relabelled. Full new-video
processing includes all planner and actual module generations. Preliminary
120–360GPUmin remains unmeasured; constrained syntax adds forwards, a one-time
vocabulary scan and source inventories but no extra model/encoder/calls.

Independent narrow code/interface review and CPU source/token/interpreter checks
must precede actual SAME fixed5 onsc474398. The unchanged prepare guard requires
actual perception calls and meaningful emitted evidence in EACH corpus, native
allraw exact and real cloned margins. No main until actual noGT PASS. Full333
and the originally declared complete main/mechanism gates remain unchanged.

B implementation note before GPU: planner material contains each complete remote
segment text ONCE in the indexed source inventory; each window's remote options
repeat only handles/coordinates, not the full transcript. Local clips contain
actual clip text. Scope target options cover only the selected real local span.
The original smoke guard is retained and additionally requires at least one
structurally non-UNKNOWN factual module in each corpus. This does not certify
facts. CPU author fixture158plans/141scope/76action/44join/6185target-span checks
PASS in `runs/20261005_m1_program/handle_cpu_checks/run.log`; fixture outputs are
not measured GPU perceptions or performance. Narrow independent review pending.

Interface B independent narrow code/interface review PASS (same-family provisional):
`docs/reviews/20261005_m1_program_handle_code.md`, evidence
`runs/20261005_m1_program/handle_code_review/`. One actual observation bug was
fixed: new factual-validity summary counters were not replay-bound to executed
calls. Both counters and source/program/cost/forward counts now match actual
metadata; independent corruption checks confirm rejection. Independent real
Qwen tokenizer/FP32 legal argmax/forced token forwards/EOS/cap/Unicode/source
oracles, tiny two-layer BF16 KV and current firstvideo source/reader bindings
PASS. These are CPU fixtures, no GT or actual 8B perceptions. Same fixed5 on
sc474398 through `launch/lab3_handles.sbatch smoke` is next after synchronization;
original A run/cache remain unchanged, no performance result. Main still requires
actual noGT prepare including the unchanged exercise/native/clone guard and
new structurally-valid factual-call check for EACH corpus.

B actualfixed5 Slurm133 dispatched2026-10-05 onsc474398 after code PASS and
`machines_before_handle_smoke{,_note}.txt`: all laboratory code synchronized/clean,
selected5090 idle/1.4Tfree, exact foreign-home names unchanged. Both acquisition
and paired native/new scoring stay together onlab3. No actual B result yet.

B Slurm133 fixed5 failed onsc474398 at08:49:52 after2m55s: actual4/5 acquisition
completed (24/5/38/53module calls respectively), fifth hate_video_114 long planner
PREFILL hit CUDA OOM, needing902MiB with769MiB free and3.25GiB allocator reserved
but unused. Both partial run and input cache immediately returned to sc474397;
source `runs/20261005_m1_program/slurm_133.out`. No paired B scoring/noGT prepare
or performance verdict. Four successful source records retained, not relabelled
as full validation. Failure runtime is additional incurred experiment cost.
Only execution allocator setting changes: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
in B launch. No scientific prompt, token, source, cap, model, operation, scoring
or input-version change. Same5 retry after narrow launch confirmation and code
sync; existing exact-bound successful source records may be reused. Unchanged
allraw/clone/exercise/factual guards still required before main.

Allocator-only narrow confirmation PASS:
`docs/reviews/20261005_m1_program_handle_allocator_fix.md`. It reduces allocation
fragmentation, does not guarantee long-prefill capacity; same source/token/model/
ops/guard contract and strict replay of four existing source records retained.

Identical B fixed5 allocator-only retry dispatched onsc474398/Slurm134 at08:56:51,
current machine/code check `machines_before_allocator_retry{,_note}.txt` clean/
synced, lab3 idle/1.4Tfree, foreign names unchanged. Actual CPU current-source
size audit reports37,798 image-expanded tokens for the first hate_video_114
planner chunk (`handle_oom_diagnosis/source_sizes.json`), not a GT measurement.
Original source/prompt/token contract remains frozen; retry result pending.

B Slurm134 identical5 completed09:00:18; both inputs/runs returned immediately.
Local noGT prepare PASS: authority
`runs/20261005_m1_program/r1_handles_full_smoke_analysis/plumbing_summary.json`.
Native allraw/global exact; actual valid perception calls126HMM/91HCS (UNKNOWN0),
all10cloned branches exact. Source program/call/cost counts replay-bound; no GT,
performance or semantic-truth claim. Long video peak27.864268GiB, largest saved
prompt38,410tokens; input acquisition163.804216s. Failed133 adds175s experiment
runtime, not concealed inside method standalone accounting.

The complete333 source-only size audit finished onsc474397, no labels/scores or
method parameter choice. `handle_full_size_preflight/source_sizes.json` records
all333/chunks plain lengths and CPU image-expanded lengths for the three
largest plain candidates; their maximum is53,137tokens at
HateMM/non_hate_video_134, chunk offset40 (adjacent53,037/53,027). Larger than
fixed5's long case. Before main, release the cached FP32 full-vocabulary copy
BEFORE EACH fresh prefill and reconstruct the identical frozen weight.float()
AFTER it; clone only the final hidden row rather than retaining the whole
prefill hidden-output storage. Inputs/positions/attention/weights/FP32 logits/
legal choices/tokens/caps/calls/guard are unchanged; this is uniform execution
memory management, not a new scientific input version or corpus rule. Acquisition
cost records include all re-creation work; earlier exact-source cached inputs
remain valid mathematical measurements, not forced new outputs.

Narrow independent confirmation and actual source-only largest-input capacity
check precede full333. `launch/lab3_capacity.sbatch` invokes `handle_extract.py
--capacity-check` for the size-selected whole non_hate_video_134, not a score/GT
pilot. It saves actual program inputs and calls in the same exact-bound cache,
separate `r1_handles_extract_capacity` run. No native/final performance is computed
by this capacity check; fixed5 native/clone/exercise guards stay required. Full
scientific gates and initial+3 result-guided revision budget remain unchanged.

Independent memory/capacity-entry narrow confirmation PASS (same-family
provisional): `docs/reviews/20261005_m1_program_handle_prefill_memory_fix.md`.
Two consecutive real two-layer BF16 Qwen CPU generations keep every legal-choice
FP32 logit, token, event and forward count exact; current source-only CPU replay
confirms N134 offset40 expanded53,137. This is not actual8B GPU capacity evidence.

Actual fixed5 source/factual inspection, without GT or performance:
`runs/20261005_m1_program/handle_smoke_input_audit/summary.json`. All96HMM and62HCS
programs selected zero remote contexts; planned joins were zero in both corpora.
HMM90/90 scope speaker/mode were UNKNOWN/UNKNOWN; HCS49/51 were UNKNOWN/UNKNOWN
and2/51 reported/reported. Scope target choices were non-UNKNOWN59/90 and30/51;
action generations were36/40. These are actual selected fields, not factual
accuracy. Structural validity of217 factual calls does not establish ownership,
context retrieval, joining or semantic correctness. Preserve these inactive
components explicitly in complete-run interpretation and mechanism decisions;
do not force activation or weaken the existing guard after this observation.

Source-only size-selected whole N134 capacity check dispatched09:15:25 on
sc474398/Slurm136 after independent narrow PASS and four clean synchronized
laboratory code checks. Selected partition idle/1.4Tfree; foreign home names
unchanged. Evidence `machines_before_capacity{,_note}.txt`. No GT/scoring in
this check; full333 waits for actual success.

Slurm136 completed09:17:59 onsc474398, whole N13457windows/8planner calls/62
module calls/4554 actual forwards,151.571920s/29.028573GiB peak. The saved largest
actual image-expanded prompt is53,137tokens. Both run/cache returned immediately;
local source/token/compiler/generation replay PASS, authority
`runs/20261005_m1_program/handle_capacity_validation/summary.json`. No GT or final
score was read. The earlier actualfixed5/native/clone/factual guard remains PASS.
Identical complete333 B acquisition+paired native/new measurement is now ready
forlab3 after synchronization and fresh machine check; source-bound prior B
cache records retain original actual costs, no scientific change or score tuning.

Complete333 SAME R1 B dispatched09:20 onsc474398/Slurm137, after synchronized
clean four laboratory code checks and idle lab3/1.4Tfree. Current record
`machines_before_handles_main{,_note}.txt`; foreign home names unchanged. Both
whole source acquisition and paired native/new scoring stay onlab3. No labels
in reader/acquisition, no performance conclusion until all333 return and the
local canonical evaluator runs. Longest capacity/fixed5 caches reused only after
current source/grammar/token execution replay; their actual incurred costs retained.


Complete R1 B results (development-selected, 2026-10-05)

Host sc474398/Slurm137 completed source333 at12:54:58 and paired333 at13:18:34;
whole runs and source333 returned to sc474397 before canonical evaluation.
Authority `runs/20261005_m1_program/r1_handles_full_main_decoded/optimized/metrics.json`:
HateMM ROC/PR/within .8972189729166409/.6912739651813877/.7697881662815763 (84),
HateClipSeg .7193770662555653/.6724209479708321/.6395515623972258 (99).
Relative to current r6: HMM +.000100/−.002961/+.019006; HCS +.002552/+.001349/+.002202.
No loss beyond declared noise, but no common dual-corpus +.01: performance FAIL.
Preserve HMM within positive progress under rule9; zero result-guided revisions used.
Native all raw reads and all six canonical metrics exact. Full source interpreter
replay PASS, `complete_source_audit/summary.json`;7359 valid programs,11807 actual
module calls,context/join0. Scope speaker/mode resolved only1HMM/3HCS; source
existence and constrained fields are not semantic accuracy or mechanism evidence.
Paired final within bootstrap CI includes0 in both; rawmax within decreases
−.013625/−.013037, HMM rawvisual −.030462. No full mechanism controls yet.
Cost authority `r1_handles_full_main_analysis/alignment.json`: new processing
7474.809361sHMM +6554.825201sHCS =233.827243min, including219.254124min source
acquisition, native550.047034s;25.50624x native. Source cached reads retain their
actual acquisition time; no new local pixel samples beyond native frames20.

Actual post-score error analysis and R2 declaration (before R2 execution)

Read `runs/20261005_m1_program/r1_handles_full_main_analysis/per_video.json`,
`r1_handles_full_main/{base,optimized}/predictions.jsonl`, all333
`data/temporal_evidence_program_handles/<dataset>/*.json`, and
`data/gt_4fps/{HateMM,HateClipSeg}.npz`. Exact case/window details and source
counts in `runs/20261005_m1_program/r1_error_analysis/program_diagnostics.json`.
HMM .012979 of the .019006 final within gain comes from videos whose rawmax
within score did not change; this scalar diagnostic alone does not prove all
ranks identical. Worst HMM H215/H295/H206 and HCS yt_SUwb0mNbqLk/bit_ckVu3UOtuy7O/
bit_r7Am2yJu0HpG lose final .2195/.1620/.1568 and .3732/.2921/.1887 respectively.
Actual window GT fractions and original/new V/S are retained for these cases.
Many emitted scopes have UNKNOWN ownership/mode despite supplied target spans;
actions can be all literal UNKNOWN or longer UNKNOWN-prefixed generation.
Full333 has3768/3591 windows;2651/2351 visual windows contain an action with at
least one field other than literal UNKNOWN, and only1/3 scopes resolve speaker
or mode. This motivates removing unsupported record injection, not numeric
calibration or labels in the new-video computation.

R2 is source-availability fallback, revision1/3. Reuse exact immutable B source
cache and execution, charge its complete original costs. For each branch:
visual adds the original executed record iff it contains an action with at
least one actor/action/target field not literal UNKNOWN or a valid actual join;
speech adds it iff it contains a scope with speaker or mode not UNKNOWN, or a
real context. Otherwise invoke a FRESH unchanged native question on the current
native prefix+own stance. No historical score splicing, averaging, score gates,
label access or corpus routing. This predicate does not treat UNKNOWN-prefixed
nonliteral descriptions as UNKNOWN and does not certify factual truth.
No additional numerical constants, prompts, model, source choices or M2–4 changes.
All333 are freshly paired in the final R2 run. Smoke requires original allraw,
clone/execution/factual guards and exact native margin on fallback queries;
retaining zero resolved speech scopes is honest, not a reason to force activity.
Expected fresh local reading cost approx25min plus complete219.25min acquisition
cost for new333 (cached acquisition remains charged); source processing call
counts unchanged. Actual cost and all six results determine rule9 disposition.

R2 author full-source/current-token check PASS (333videos,7359windows,
13939 original R1 branch queries/IDs exact):
`runs/20261005_m1_program/available_cpu_checks/summary.json`.
Visual application2651/2351, fallback1117/1240; speech application1/3,
fallback3438/3138. Source-only counts, not scores or semantic correctness.
Independent narrow confirmation PASS:
`docs/reviews/20261005_m1_program_available_facts_code.md`; real36layerBF16 CPU
fresh fallback/current prefix+stance, clone/crop/rope/calls,14predicate cases,
5corruption rejections and fixed5 actualsource/token binding. 8B GPU R2 pending.

R2 identicalfixed5 submitted2026-10-05 13:44:07 onsc474398/Slurm149 after allfourlabs
synchronized clean, lab3idle/1.4Tfree, exact foreign-home STRAY names unchanged.
Evidence `machines_before_available_smoke{,_note}.txt`;148/146 occupy sharedQOS2,
149 mayqueue without bypass. Original B source/costs reused only after strict
current source/grammar/token replay. No R2 performance claim.

Additional actual rank check uses scipy rankdata on original/new window max, not only equal AUC: HMM7/84 exact-order videos contribute .012617615131276334 of final within gain; HCS0/99. Exact case lists are in the same error artifact. H329 is one constant raw8s window, while fixed r6 has two cell-level final values whose order reverses; its +.6 video-within gain is not new M1 temporal ordering. Existing evaluator and r6 remain fixed.

Actual R2 fixed5 Slurm149 completed13:57:01, runs returned immediately; no new
source cache was generated (same full333 immutable B). Local noGT prepare PASS,
`runs/20261005_m1_program/r2_handles_full_smoke_analysis/plumbing_summary.json`:
native allraw/G exact,10actual clones exact,70newV/2newS changed. All fallback
queries were freshly measured and exactly native by production assertions and
source/token validation;70V/88V fallback,2S/132S fallback match source preflight.
Actual standalone205.905423sHMM+102.261208sHCS includes all original source
acquisition, not historical score reuse. Complete R2 paired333 ready, no GT/main
performance yet. New video retains the complete acquisition cost, so this reduces
unsupported record injection, not the source generation workload.

Complete333 R2 dispatched2026-10-05 onsc474398/Slurm150 after current allfourlabs
clean synced check, lab3idle1.4Tfree and146 sole activeQOSjob. Evidence
`machines_before_available_main{,_note}.txt`. SourcecacheB/sourcegeneration costs
unchanged, every native/new output is a fresh current-model call; no new GT in reader.

## R2 full333 result2026-10-05

Runhostsc474398/Slurm150 DONE14:20:56, entirepairedruns immediatelyreturned via `runs/20261005_m1_program/return_available_main.log`. LocalnoGT source/token/nativeallraw/G PASS andcanonicalnative all6exact. Authority `runs/20261005_m1_program/r2_handles_full_main_decoded/optimized/metrics.json`: HMM ROC/PR/within .8974793605420147/.6962635795408186/.7661858786046777 (84), HCS .71681863230139/.6730202046094492/.6409840593522289 (99). Within+.015403834638710756/+.003634945548130153, allotherlosseswithintolerance, commonmetric.01 gateFAIL. PreservepositiveHMMprogress; Rule9 permitsremaining2revisions afteractualerroranalysis. Finalpaired95% intervals contain0 both; rawmaxwithin-.006523590277980433/-.009781588743189946, no mechanismclaim. Newvideo actualstandalone229.4507940746519min/native25.065285048268407x, including originalB acquisition; actual sourcecostnotfree. All development-selected; report `runs/20261005_m1_program/r2_handles_full_main_analysis/summary.json`.

ActualR2 GT/source erroranalysis `runs/20261005_m1_program/r2_error_analysis/source_diagnostics.json` read fullbase/newrawandfinalpervideo, fullBsourcecache andboth4fpsGT. Examples HMMH215 loss-.228724,HCSyt_SUwb0mNbqLk-.350271 andbit_ckVu3UOtuy7O-.224956 retained withactualwindowpositivefractions andsourcefields. Sourceaction descriptivefields often begin UNKNOWN but are longerstrings: actor741/633, action754/611, target1121/1112 (HMM/HCS). H215windows10/12 contain `UNKNOWN action:UNKNOWN ...` ratherthanresolvedfacts; R2 intentionallyonly recognizedliteralUNKNOWN andthereforeappliedthesepackets. Actor-only109/64 windows also donotestablish anaction. These areactualfactualavailabilityfailures, not certifiedincorrectpredictions; wewilldeclareR3 unavailablemarker/action-requirement semantics before measurement, not choose a numeric scorethreshold.

## R3 predeclared2026-10-05: resolved action availability

Revision2/3, afteractualR2 fullGT/sourceerroranalysis above. OriginalimmutableB source/planner/interpreter/modules/cost unchanged. Samebothcorpora: invisualactionrecord only, normalize actor/action/target strings whose left-stripped text starts regex `^UNKNOWN\b`, case-insensitive, to literalUNKNOWN. Preserve originalunmodifiedsourcecache andsupport/provenance; this is uncertaintyavailability, not factualtruth. Applyvisualprogram packet onlyif actionfielddoesnotbeginthatmarker orvalidactualjoin. Actor-only/target-only noaction cannotestablish anaction. SpeechusesidenticalR2 availability/context rules (four actualresolvedscopewindows); no newspeech constant. No numericmargin gate, hatekeyword ordatasetroute. Noavailablepacket -> freshunchangednativebranch, never copyhistoricmargins. Bothbranchesfresh onnativecache/ownstance, max/fixedr6unchanged. Sourceacquisition stillchargedfull; no newmodel/frame. LiteralUNKNOWN-prefixedcontentexception nowdeclaredbeforeGPU; no thresholdscan. Complete333CPUoriginalR1/R2query/tokencompatibility+R3normalization/immutabilitychecks andindependentnarrowcodeconfirmation required, samefixed5 beforewhole333. R3scientificexecution/performanceunknown. RemainingonefuturerevisionafterR3 ifqualifyinggainpersists.

R3authorCPU `runs/20261005_m1_program/resolved_cpu_checks/summary.json` PASS: all333 sourceimmutable,27878actualR1/R2query/token checks exact, normalized2616/2356 fields; visual applies1788/1675 andfallback1980/1916 HMM/HCS, speech1/3 unchanged. Fixture initiallyomittedexplicitselected_rows(False), fixedfixtureonly, initialfailurelogretained. Independentnarrow `docs/reviews/20261005_m1_program_resolved_action_code.md` PASS: actual36layerBF16 CPU main/smoke,11regexboundaries,12corruptions, full333sourceoracle; noGT/GPUinreview. Actual8Bfixed5pending.

R3 identicalfixed5 submitted2026-10-05 onsc474398/Slurm154, queuedbehindTTF153 andInterval151 underQOS2. The actualallfourlabs clean4aa4687 check immediatelypreceded153, no interveningcodechange; exactforeignSTRAYnamesunchanged. Evidence `runs/20261005_m1_program/machines_before_resolved_smoke{,_note}.txt`; originalBsourcecacheimmutable, no newsourcegeneration. No R3 performance or mechanism resultyet.

ActualR3 fixed5 Slurm154 DONE15:02:27, wholeruns immediatelyreturned via `return_resolved_smoke.log`; no newsourcecache. LocalstrictnoGT source/nativeallraw/G/10clone/freshfallback PASS, authority `r3_handles_full_smoke_analysis/plumbing_summary.json`. Samewhole333R3ready; no R3mainGT/metricsyet.
