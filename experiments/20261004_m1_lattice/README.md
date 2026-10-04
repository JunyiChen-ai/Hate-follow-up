# M1 candidate22 backup: frozen temporal speech lattice reader

Declared 2026-10-04 while candidate21 was running; its allowed revisions are now
exhausted and the family archived. Current candidate22: fixed5 GPU plumbing
verification started on sc474399. No performance result or GT analysis yet. Formal
method remains r6_bma; development-selected. Not an acoustic-alignment revision:
newly decoded competing lexical content, lattice reachability and longest-path
positions replace a single-transcript alignment-support intervention.

## Source and hypothesis

ASR substitution/deletion can change an act, its target or negation. Treating
the sole transcript as certain prevents the semantic reader from using acoustic
alternatives. A single frozen reader can encode an entire word-confusion DAG
with mutually exclusive alternatives and their acoustic mass before it makes
one speech decision. No independently scored hypothesis is averaged.

Primary structural source: [Huang and Chen, ASRU2019](https://www.csie.ntu.edu.tw/~yvchen/doc/ASRU19_LatticeSLU.pdf),
sections3.1–3.3/equations10–13 and training setup. It combines predecessor
attention, optional log conditional path probabilities and longest-path positions
after subword splitting. The experiments fine-tune GPT with SLU labels. This
proposal uses those operators on frozen Qwen and does not claim source numerical
performance or an exact reproduction. Official code actually inspected:
https://github.com/MiuLab/Lattice-Transformer-SLU, `src/lattice_utils.py` and
`src/modeling_openai.py`.

[Everson et al., 2024](https://arxiv.org/html/2401.02921v1), sections2.1–2.4 and
limitations, motivates uncertainty-preserving input but uses separator strings
and removes null alternatives. Its results do not establish frozen structural
encoding. Here acoustic null mass remains in the DAG, despite having no text KV
node. Source scopes and failed/successful accesses are recorded in
`runs/20261004_m1_ideation/backup_source_reads/word_lattice_source_scope.json`.
Target neighbors include MultiHateLoc's single-transcript temporal alignment and
existing video-language reasoning methods in the independent ideation jury. An
independent rule4 search/review must check whether this complete source method
already occurs in hateful video work; input changes or separator prompts alone
are not the novelty claim.

## Exact proposed common R1, before implementation or scoring

1. Both full test corpora use original native20-frame overview/full ASR, native
   Qwen global question/own hard stance, native visual margins and fixed8s windows.
   Recollect an exact paired native reference. Scoring/ASR/DAG construction never
   opens labels, GT, previous predictions or metrics. Same flow and constants.
2. Decode original video's audio as mono16kHz float32 with its actual presentation
   origin relative to the same video's presentation origin. Save the audio origin,
   observed sample intervals and actual resampling mapping; never assume the first
   audio sample is video time0. Each native
   window with nonempty native speech body supplies exactly its real8s audio
   samples (final window ends at its manifest duration), without neighbors or
   invented padding before the observed samples. Native empty speech remains
   absent. If the actual audio ends before a window, record empty actual audio;
   the new speech body for that window is absent, not an invented transcription.
   Standard Whisper feature padding to30s is recorded and is not source audio.
   Overlapping resampled blocks retain the first observed sample at each actual
   presentation coordinate, in decode order; later overlapping samples are discarded
   with exact intervals recorded. No averaging, shifting or time compression.
3. Same frozen `openai/whisper-large-v3`, FP16, automatic language detection,
   transcribe task, deterministic beam search `num_beams=5`, `num_return_sequences=5`,
   do_sample=False, length_penalty=1, early_stopping=False, maximum448 total decoder
   tokens, no timestamps or temperature fallback. Suppressed token defaults come
   from the model's parsed generation configuration and are saved. Detect language
   once per video from its first up-to30s actual audio, reuse the same detected
   language for all its windows; no forced language by corpus. Every recognition,
   detection, preprocessing and decoding call is charged. Save exact beam tokens,
   EOS/truncation, text, sequence scores and actual consumed samples. Weight each
   beam by FP32 softmax of its length-normalized sequence score (temperature1),
   merge identical decoded text by adding mass. These are normalized beam weights,
   not calibrated posterior probabilities or a complete ASR lattice.
4. Split each decoded text into nonempty whitespace-delimited words, preserving
   exact case/punctuation. Sort hypotheses by descending score, beam order ties;
   highest scoring nonempty hypothesis is the alignment anchor. Levenshtein
   word alignment to the anchor has match0, substitution/insertion/deletion1;
   compare exact casefolded words; ties prefer diagonal, then deletion, then
   insertion. For each anchor word make a substitution/deletion slot. For each
   boundary make an insertion-phrase slot, retaining each hypothesis's complete
   inserted phrase as one alternative; remove a boundary only if every alternative
   is null. In each slot merge exactly identical strings, sum their beam mass,
   sort mass descending then earliest contributing beam/text. Null alternative
   is an epsilon edge with its mass, not a spoken word or a generated `[silence]`
   text. This is a word-confusion approximation that allows recombination across
   slots; it is not the original beam DAG or a true word-time alignment. All slots
   are grounded to their actually cropped8s window, not invented word timestamps.
5. Each nonempty alternative becomes a branch of Qwen subword nodes, tokenized
   independently as `one leading space + exact alternative text`, with no added
   tokens/vocabulary. Its within-branch edges have probability1; entering the
   alternative has its slot mass. Different options of one slot are unreachable
   from each other. Slot completion joins the branches; epsilon skips that slot.
   The DAG is serialized in slot order, alternative order, subword order. Every
   DAG position is the longest-path distance from its source after subword splitting;
   a slot advances by the maximum nonempty alternative token count. Consequently
   alternatives share logical starts without pretending they are sequential speech.
6. Append one independent speech branch to the unchanged native stance cache.
   Render its chat header/scaffold using native Qwen templates. Scaffold literal:
   `Consider only window {i+1} of {n}, from {a:.1f}s to {b:.1f}s of this video.\n`
   `The following speech lattice contains alternative transcriptions of the same audio. Alternatives in one slot are mutually exclusive; a null edge means no word. The complete video transcript is interpretation context.\nSpeech lattice:\n`
   Then append the declared DAG token sequence, then independently tokenized suffix:
   `\n\nIs THIS window one of the segments where speech that violates the above rules occurs?\n\nAnswer "Yes" or "No".`
   Exact scaffold and suffix IDs/text, token-to-slot/alternative map, mass and logical
   positions are recorded. This manual graph tokenization is declared; it is not
   required to equal ordinary BPE tokenization of its flattened text.
7. Use the native cache as an immutable prefix and a single Qwen forward for all
   suffix/DAG/query tokens. An additive attention bias applies to all36 layers/heads:
   scaffold tokens see the native prefix plus their preceding scaffold tokens;
   a graph token sees that prefix/scaffold, its own branch's earlier tokens/self,
   and all earlier slots. Same-slot other alternatives and all future nodes are
   masked with negative infinity. Earlier-slot keys receive log of their alternative
   mass, own-branch predecessors/self receive0. The final question/chat suffix sees
   all graph keys with log of alternative mass and ordinary prior scaffold/query
   keys with0. This implements conditional predecessor probability for the declared
   independent-slot DAG. No probability floor or selected head/layer. Original
   shared cache keys remain accessible with0; no hard context-only isolation claim.
   Positions of graph tokens use the declared longest-path distance starting after
   the scaffold's last native multimodal position; final question positions start
   after the longest DAG path. The original prefix positions are unchanged. Crop
   and restore cache/rotary state after each branch; no previous local answers.
8. Read native FP32 Yes/No margin once at the final question. This replaces only
   native speech for windows whose native body was nonempty and actual audio exists;
   native visual/global remain unchanged. New raw max(native visual,new speech),
   missing speech -> visual. Use unchanged r6 subprocess and the single canonical
   evaluator at4fps. No hypothesis-specific moderation predictions, voting,
   backbone fitting, label calibration or dataset routing. Fixed seed0.

Empty recognized text yields a DAG with no word nodes and the same speech query;
it is explicitly distinguished from missing actual audio. Language-detection and
generation APIs must be verified against the actual target HF version before
implementation. Runtime/interface errors are repaired without changing constants
or evaluating the idea. No silent model/decoding substitution.

## Cost, implementation validation and mechanism gates

Common native preprocessing is reported separately. Native media/ASR/20JPEG/model
weights may be reused, but every new8s beam recognition is charged for a new video.
There is up to1 language-detection encoder pass/video,1 padded audio encoder and
actual beam decoder steps/nonempty speech window, then1 new Qwen speech forward
per available window. Feature/beam token memory, all graph masks/position computation
and source decoding are charged. A beam is5 hypotheses from one model, not5
independent models or5 moderation calls. Initial estimate60–180GPUmin/333 on5090;
unmeasured. No parallel window slicing. Fixed same5 noGT smoke verifies real APIs,
native parity, observed intervention, costs and memory; it never picks constants.

Required meaningful checks: independently enumerated small-DAG conditional
probabilities/epsilon branches; longest-path positions and same-slot masking;
observed crop sample alignment; exact native global/visual/speech reproduction;
new raw speech actually entering fixed r6; None/multimodal rotary restoration;
one-best graph with probability1 agrees with an otherwise identical sequential
token forward; no local answer-state leakage; parsed input versions/complete333;
all runtime costs; canonical evaluator only. Independent rule6 review required.

Main gate: a common main metric gains>=.01 on BOTH datasets versus current/native
r6, no pooled loss>.005/within loss>.01. Full215/118, report all three metrics and
84/99 eligible within videos. No ANY main gain>=.01 -> archive; one qualifying
gain -> at most3 predeclared, error-analysis-guided revisions. Development-selected.

If main passes, run full333 controls:

- `onebest`: same new beam acquisition, use only highest-scoring text with unit
  mass and ordinary serial positions in the same scaffold/query. Tests uncertain
  graph reading beyond obtaining a new transcript/crop.
- `flat`: identical DAG token multiset/physical length, ordinary causal attention
  and serial positions. Tests complete structural encoding against extra words.
- `binary`: same graph/shared positions, replace all finite log-mass biases with0.
  A diagnostic of acoustic weighting; not separately claimed without rule14g.
- `wrong_mass`: deterministically rotate option masses by floor(K/2) within each
  multi-option slot; retain exact tokens and graph/positions. Null mass participates;
  singleton is a recorded no-op. Falsifies acoustic-mass interpretation.
- `wrong_audio_window`: rotate each video's acquired window DAGs by floor(W/2),
  retaining donors' true source interval rather than falsifying timestamps. Final
  speech questions refer to destination windows. Singleton no-op recorded. Tests
  whether newly recognized local content's source correspondence matters.

The complete method must beat both new-onebest and same-token flat on a common
main metric>=.01 in BOTH datasets to claim the structural uncertainty mechanism.
Each separately claimed operator must pass rule14g or be deleted/demoted. A
postscore ASR/disagreement/source-scope audit, raw speech/max ordering and paired
video bootstrap assess actual lexical changes; no reference words/hate labels
enter decoding/scoring. Final independent novelty/mechanism review required.
No Overleaf or external messages. No scientific result yet.

## Preparation after independent proposal review

Independent rule4 PASS: `docs/reviews/20261004_m1_lattice_proposal.md` (2026-10-05).
Only CPU implementation and source/interface checks are prepared while Tree runs;
GPU launch and evaluation remain conditional on Tree's result-based branch.

Actual target HF5.15.1 source inspection found its Whisper public long-form wrapper
repeats each input `num_return_sequences` times and sets the inner return count1.
Using that wrapper for deterministic beam5 would yield repeated one-best outputs,
not the declared five-best beam. The implementation must use the same model's
standard short-form `GenerationMixin.generate` with one encoder result, explicit
Whisper start/language/transcribe/no-timestamps prefix and parsed default suppression
processors. Retain beam5/return5/total max_length448 exactly; no altered scientific
algorithm or numerical selection. Explicitly suppress model timestamp IDs as part
of the declared no-timestamps decoding. Save actual prefix/config/processors and
verify five physical beam returns before any scoring. Every language-detection
decoder/encoder and recognition decoder call is charged. This is an API adaptation,
not evidence that five alternatives are useful or distinct in every window.

FP32 zero-mass alternatives have no positive-probability path and contribute no
graph token; preserve their zero mass in input records without renormalizing the
remaining nonempty alternatives. No mass floor. The onebest control's tokenization
is `one leading space + exact highest-scoring text` as a single tokenizer call,
ordinary serial positions; the unit-mass degenerate correctness check instead
compares the identical manually compiled IDs/mask against an ordinary sequential
forward. These comparisons answer different questions.

Implementation files: `audio.py`, `lattice.py`, `extract.py`, `reader.py`,
`measure.py`, `analyze.py`, `selfcheck.py`, and committed `launch/` scripts.
Native stance/cache machinery is shared in `src/stance_cache.py`, without changing
the original Judge or evaluator. No other experiment is imported. Current Tree
code/inputs/parameters have not changed. Independent rule6 PASS (2026-10-05,
same-family provisional): `docs/reviews/20261005_m1_lattice_code.md`.

Meaningful CPU checks PASS, actual stdout:
`runs/20261004_m1_lattice/cpu_checks/selfcheck.log`. An independently enumerated
option-path oracle verifies predecessor probabilities including epsilon mass;
a separate longest-distance graph oracle verifies logical positions; all original
beam strings reconstruct through the slots. A real delayed-audio MKV verifies
PTS/resampling/cropping, missing/partial intervals and uncompressed gaps. Zero-mass
omission and single-path mask/position degeneration also pass. Python compilation
and all launch shell syntax pass. No real beam or Qwen measurement parity claimed.

Independent review found a resume-input consistency gap before any GPU run.
Saved traces now reconstruct the complete graph from the current confusion slots
and actual Qwen tokenizer, requiring exact alternatives/masses/token IDs/node
geometry and recomputed logical positions. Head/tail tokenization and the current
window scaffold are also checked. Native `prefix_tokens` retains its original
pre-global-question length; separately recorded `stance_cache_tokens` and
`stance_cache_logical_start` identify the immutable cache after global/stance turns.
The reviewer caught and corrected their initial length conflation during this fix.
Five deliberate stale/corrupt graph cases are rejected by the CPU regression check;
no score, beam generation rule or attention operator changed.

The independent reviewer also exercised the actual structural reader on a tiny
CPU Qwen: both layers received the declared 4D bias; cloned-cache replay and
cache/delta restoration passed. Target HF5.15.1 source inspection confirms an
existing 4D mask is passed through and generic beam returns the requested five
sequences. Evidence: `runs/20261004_m1_lattice/code_review/`. These checks do not
replace actual Whisper/Qwen8B Slurm smoke or establish performance/mechanism.

Before any GPU launch, the fixed smoke additionally performs one cloned-cache
structural replay and two probability-one structural/sequential forwards per video
with available speech. Exact replay required; graph/sequential final margin
tolerance .01 is an operational BF16-kernel check, not a hateful-score threshold.
Input/matrix oracle checks separately establish exact geometry. Its three diagnostic
Qwen calls/time are reported separately from deployment work and included in
actual GPU workload/peak memory. Main has no diagnostic extra calls.

The paired measurement collects common native prefix/global/visual once, reference
native speech separately, then new speech independently on the same immutable
prefix. Standalone native time = prefix +visual +reference speech; standalone new
time = all new ASR preparation +prefix +visual +new speech. Reference speech and
diagnostics are additional collection work, not silently attributed to deployment.
Calls in each method's record still equal3+W+its available speech windows. Actual
paired forwards are independently counted and reconciled with common/reused work.

### Real audio CPU preflight, 2026-10-05

Before GPU acquisition, decoding the fixed five actual smoke media passed four
videos but failed on HateMM `hate_video_114`: AAC48k source PTS7168/7169/7170
represent overlapping decoded blocks, triggering the initial overlap assertion.
This was an input implementation failure, before any beam or moderation score.
Failure evidence: `runs/20261004_m1_lattice/cpu_audio_preflight/failure.json`.

The repaired, declared common policy places each resampled block at its actual
presentation coordinate and keeps the first observed sample there in decode order.
Later overlaps are discarded with exact block intervals saved; audio is never
averaged, shifted or concatenated across a gap. Cache version/config/validation
record this policy. Independent hand overlapping/backward/clipped block cases and
the prior delayed-audio/gap oracle pass. All five real media now decode/crop
successfully; recorded placed intervals exactly reconstruct the observed mask,
and every window's waveform equals its source timeline slice. Actual evidence:
`runs/20261004_m1_lattice/cpu_audio_preflight_fixed/{run.log,summary.json}`.
This CPU parsing check does not substitute for actual ASR beam/Qwen8B smoke or
full performance/mechanism validation. Independent narrow fix confirmation PASS
(same-family provisional): `docs/reviews/20261005_m1_lattice_audio_overlap_fix.md`.
Its separate per-sample first-wins oracle matches 100 deterministic overlapping
blocks and verifies the saved actual-media interval bookkeeping. This policy
does not claim to recover a unique waveform from malformed overlapping timestamps.

### First actual fixed5 GPU attempt: interface repair only

Host sc474399, Slurm121 (2026-10-05). All5 true-audio beam inputs were generated;
Qwen first structural call failed before a completed score/video with CUDA SDPA
`invalid dtype for bias - should match query's dtype`. Outputs and inputs returned
to uoa-lab1 before documentation. Evidence: local `slurm_121.out`,
`r1_extract_smoke/summary.json` and `r1_extract_smoke/actual_beam_audit.json` under
`runs/20261004_m1_lattice/`. No GT or metric; not a method-negative observation.

Narrow implementation fix: retain FP32 construction of log-mass/zero/negative-
infinity mask, convert it to the model's BF16 query dtype only at the SDPA API
boundary. This necessarily rounds finite log-masses at BF16 precision; zero and
negative infinity remain exact, graph edges/positions/tokens unchanged. No prompt,
beam, mass normalization or scientific constant change. Independent narrow
confirmation required before repeating exactly the same5 and checks.

Bias dtype fix independent narrow PASS, same-family provisional:
`docs/reviews/20261005_m1_lattice_bias_dtype_fix.md`. Actual36-layer tiny BF16
Qwen CPU/SDPA checks reach every layer with BF16 query/bias, clone exact, and
normal/injected-exception cache/delta restoration. Finite logmass fixture max
rounding .00274181 (not a global bound);0/negative infinity exact. This CPU evidence
does not replace target8B GPU parity. Repeat exactly the same fixed5.

No-GT descriptive `input_audit.py --smoke` reads actual source beam inputs only,
records alternative/epsilon/truncation counts and compares casefolded Unicode word
sequences, never selects a method input or score. Authority
`runs/20261004_m1_lattice/r1_extract_smoke/input_audit.json`:134 actual windows,
134 distinct exact-text alternatives,114 with lexical differences ignoring
punctuation/case;29 of670 beams truncated, all retained by declared448-token cap.
These are input observations, not ASR accuracy or hateful localization evidence.

### Fixed5 actual GPU verification complete

sc474399/Slurm122 DONE5; all run outputs and source inputs returned locally before
no-GT prepare PASS. Authority
`runs/20261004_m1_lattice/r1_full_smoke_analysis/plumbing_summary.json`. Native
all G/windows/curves exactly reproduce current native; G/V unchanged in new arm,
134 speech margins changed. All5 cloned-cache margins exact; probability-one DAG
and otherwise identical sequential token forwards also EXACT in all5 (difference0,
within original operational tolerance .01). Actual graph token/current input
bindings pass; no GT or metric used.

Standalone sample HMM56.12289s (84 recognized windows), HCS16.33920s (50), including
actual original beam preparation46.90260/11.88669s. Native8.40762/4.07519s.
Peaks18.09434/17.28584GiB; paired Qwen forwards282/174 including separately timed
diagnostics9/6 (.31564/.19191s). ASR encoder87/52, decoder4800/1097 including
language detection. Graph tokens5768/1425, slots2093/841. Rough full-corpus cost
33.88266/16.06688min =49.94954min; an extrapolation, not actual complete cost.
Proceed identical complete333/main, without changed constants or selected windows.

### Control execution details, declared before control implementation/result

Complete333 main submitted sc474399/Slurm123. Controls remain conditional on the
complete main performance gate, never partial results. There are six independent
new speech arms: full, onebest, flat, binary, wrong_mass, wrong_audio_window;
native G/own stance/V and reference S are recollected once and held common.
Each arm makes one new S forward per actually available window on its own restored
cache. Joint workload3+W+native_S+6*available_S; fixed5 additionally clone the first
available S of EACH arm, separately timed6 calls/video. Input acquisition is the
same real beam cache, charged to every arm's standalone new-video cost but performed
only once in the physical experiment. No new ASR/model/label/threshold/metric input.
Full and native must exactly reproduce the corresponding completed main BEFORE
any control evaluator reads GT. Main performance PASS is a launcher hard guard.

onebest tokenizes `one leading space + exact onebest text` in one call between the
exact same chat scaffold and question, with ordinary causal mask/serial positions.
flat retains the exact graph IDs/head/tail, causal mask and serial positions;
binary retains IDs/graph positions and all mask exclusions, finite logmass->0.
wrong_mass rotates option masses by floor(K/2) and retains original serialized
token IDs/branch identity/positions even when mass order changes. Null alternatives
participate. Slots containing any zero mass are recorded as unchanged because
moving positive mass to a previously omitted text node would change the matched
token inventory; this coverage limitation is reported, not hidden. Singleton
slots are unchanged. No per-corpus filtering or score-based arm selection.

wrong_audio_window rotates only actually acquired available windows by floor(W/2).
Its scaffold states the DONOR's true window/time; immediately before the original
final speech question it adds `Judge destination window {i} of {n}, from {a:.1f}s
to {b:.1f}s of this video. The transcription above comes from source window {j}
of {n}, from {c:.1f}s to {d:.1f}s.` These are truthful source/destination coordinates,
not mislabeled evidence. No available window/singleton stays unchanged. This arm
necessarily changes the source/destination instruction and is a scoped diagnostic,
not a perfectly text-matched isolation claim. Native full transcript remains visible.

Controls save current-source confusion/compiled IDs/positions/mask specification,
actual changed mass bindings/source windows, per-arm raw/native curves and physical
call/time/memory records. Resume and noGT prepare rebuild from current beam inputs
and current native chat/template/tokenizer, without hashes. Same canonical evaluator
and fixed r6 apply to all seven output streams. Require common same-metric+.01
over BOTH onebest AND flat on BOTH datasets for structural novelty, raw-ordering
and paired intervals, actual incorrect-input coverage and independent final review.
binary/mass weighting is diagnostic unless its own rule14g removal gate passes.
Preliminary six-arm reader work20–60GPUmin/333 on5090 in addition to main collection,
shared beam acquisition reused for this experiment; unmeasured. No control GPU yet.

Controls prepared, independent rule6 PASS (same-family provisional):
`docs/reviews/20261005_m1_lattice_controls_code.md`. Actual evidence
`runs/20261004_m1_lattice/controls_code_review/`: actual5 CPU processor/ASR/beam
source/token binding and12 corruption rejections; memory-stub reader6arm/call/copy
checks and noavailable boundary; real36-layer tiny BF16 Qwen6arm clones/exception
KV+delta recovery; independent epsilon rotation/zero/causal/binary/donor cases;
canonical7streams14 subprocess commands and launcher four gate cases. These are
CPU checks, not actual8B control predictions or mechanism evidence. Reviewer found
cost binding gap; validator now ties prep/ASR counts to current metadata and
reconstructs every native/arm standalone time from actual components. Three cost
corruption cases rejected. Actual author CPU fixed5 compile summary in
`controls_cpu_compile/summary.json` binds native expanded image IDs and token
inventories. Native/full GPU parity still required after complete main PASS.
No control GPU or control score/GT analysis has run.
