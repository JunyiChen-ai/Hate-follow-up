Archived2026-10-05: R1/R2 positive HateMM within retained, but neither passed the full dual-corpus gate; R3 has no qualifying +.01 and losses outside noise, rule9 archive. No R4/control GPU.

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

## R1 complete result and R2 declaration (2026-10-05)

R1 ran on sc474399, Slurm123, completed333 at06:39:12. All output and actual beam
inputs were immediately returned to sc474397 before detached CPU validation and
evaluation. Native global/window/curves and all six fixed-r6 metrics exactly
reproduce the current method. G/V are unchanged;6580 speech windows changed.
Authority: `runs/20261004_m1_lattice/r1_full_main_decoded/optimized/metrics.json`.
Paired report/intervals: `r1_full_main_analysis/{summary,per_video}.json` under
the same run root. All results are development-selected.

| dataset | pooled ROC | pooled PR | within | eligible | delta ROC/PR/within |
|---|---:|---:|---:|---:|---|
| HateMM | .8965709075 | .6774663450 | .7990019800 |84|−.000547754 / −.016768258 / +.048219936|
| HateClipSeg | .7144145989 | .6671199008 | .6400126663 |99|−.002410265 / −.003952184 / +.002663553|

HateMM final within paired95% interval[.017739774,.084486120] is positive; retain
this positive result. HCS interval[−.017121808,.024132034]. Raw max within gains
.011059259/+.005913626; raw shared-S gains+.039093118(82)/+.019182930(97), both
intervals cross0. V changes0. Performance FAIL: no common metric+.01 both and
HMM pooled PR loss exceeds.005. Rule9 permits revision because HMM within+.0482.
No R1 control GPU; prepared controls are not mechanism evidence.

Actual per-new-video attributed work totals70.509133min (HMM34.176989/HCS36.332144),
native9.186967min, ratio7.674909. Includes59.950735min of actual independent audio
recognition, even though later revisions can physically reuse that cache. Peaks
17.823754/17.547223GiB. Acquisition encoder/decoder calls3653/173984 and
3259/194654; paired reader calls11291/10227, diagnostic0. Full actual input audit
`r1_extract_main/input_audit.json`:6580 recognized windows,5893 with different
lexical candidates,1835/32900 truncated beams retained, zero-mass alternatives0.
These source properties do not establish semantic correctness or mechanism.

Postscore analysis read exactly native/optimized raw predictions, both decoded
predictions, canonical summary/per-video and `data/gt_4fps/{HateMM,HateClipSeg}.npz`;
the complete readable file list is `r1_error_analysis/summary.json`. Actual window
beam/native sources and automatically selected diagnostic examples are in
`r1_error_analysis/{per_video,examples}.json`, generated by `error_analysis.py`.
These are postscore diagnostic inputs only; no scoring/constant/threshold path
imports the analysis. The new transcripts recover content missing in native
repetition/music crops, while some label-negative innocuous speech (including
an ordinary short Yes/No exchange) gains a large speech margin. HMM positive/
negative frame-weighted raw-S shifts are−2.305967/+.648504; HCS−1.027156/+.523828.
These descriptive shifts and source examples motivate checking sentence-level
correlation, not a label-based filter or a claimed cause already established.

R2 hypothesis: R1's independent slot recombination and separately encoded words
discard dependencies within a complete recognized sentence. Keep the actual
complete hypothesis as one branch; preserve its punctuation/subword history and
exclude other full hypotheses. This is revision1/3 of the same uncertainty-reader
family, not a new family or an ASR/encoder/prompt-only swap. No GPU has run yet.

R2 constants/input are exactly the existing5 beams/448-token cap/FP32 score
softmaxT1/seed0/8s/16k and same native availability, policy, G, own stance, V,
4fps and fixed r6. Keep ALL actual texts, including truncated texts; merge exact
identical decoded text by summed existing beam weights. Sort mass descending,
then earliest beam/text. Encode each entire nonempty positive-mass path once as
`one leading space + exact decoded text`; empty text is an epsilon path with
its mass, no fabricated token or positive-mass renormalization. Each branch's
subwords use consecutive logical positions beginning after the scaffold; the
query starts after the maximum complete path length. Physical storage serializes
paths in the stated order, no aligned-slot recombination.

R2 literal scaffold: `Consider only window {i} of {n}, from {a:.1f}s to {b:.1f}s
of this video.\nThe following speech lattice contains mutually exclusive complete
transcriptions of the same audio. Words within one alternative belong together;
alternatives must not be combined. A null path means no words. The complete video
transcript is interpretation context.\nSpeech lattice:\n`. Final question unchanged.
All36-layer BF16 bias: scaffold sees native prefix/earlier scaffold; path token
sees native prefix/scaffold and its own earlier tokens/self only, all other path
tokens excluded. Query sees all path keys with log of their merged path mass and
native prefix/scaffold/earlier query with0. This retains the utterance's joint
path identity inside its representation. Original native full ASR remains visible;
no hard global isolation or calibrated acoustic posterior claim. Exactly one
final Qwen margin/newS, no independent hypothesis classifiers or score averaging.

`path_graph.py`, `path_reader.py`, `path_measure.py`, `path_analyze.py` preserve the
R1 implementation/results. R2 uses complete atomic per-video records, current
native CPU processor-expanded prefix binding plus exact source/graph/token/range/
position/cost replay before reuse. Shared same-experiment native record/compiler
helpers are used; canonical evaluator is never copied. New physical ASR calls0
in this revision, but new-video cost still includes the full unchanged acquisition.
Additional reader planning estimate10–25GPUmin/333, total70–90min includingASR,
unmeasured; complete paths can use more tokens than the factorized graph. Actual
cost/peak/calls replace estimates after the fixed5 and full333 runs.

Before GPU, independently check new path mask/source bindings and unchanged native
flow; actualfixed5 must then pass native exact, first-newS clone exact and the
probability-one complete path versus identical ordinary serial token margin
within the unchanged .01 BF16 operational tolerance. No pilot GT selection. Full
333 uses the identical R2. No performance gate changes. If main PASS, controls
must be adapted BEFORE their scores to intact-path onebest/flat/binary/mass/source
operators and pass independent checks; R1's guarded control launcher stays R1.
No R1 controls are launched to rescue a failed main gate. Structural novelty still
requires common-metric+.01 over both onebest and matched flat on BOTH corpora,
actual wrong-source coverage, raw ordering, intervals and final independent review.

R2 author hand path/mass/matrix/degeneration CPU checks PASS:
`runs/20261004_m1_lattice/path_cpu_checks/selfcheck.log`. Independent narrow code
review PASS: `docs/reviews/20261005_m1_lattice_path_code.md`; actual evidence in
`path_code_review/` under the run root includes real36-layer BF16 attention/clone/
unit/exception KV tests, actual5 CPU source/native expanded-ID binding with15
corruption cases, emptyavailability, canonical function identity and subprocess
flags. The sole reported count bug is fixed: epsilon counts empty text; nonempty
zero-mass paths have a separate counter. This does not change scores, and actual
R1 zero-mass count0. Actual8B R2 fixed5/full333 GPU checks remain pending.

Actual R2 fixed5 Slurm124 completed onsc474399 at07:04:43; output and unchanged
input cache returned locally. NoGT `path_analyze.py --stage prepare --smoke` PASS:
native all G/window/curve exact, G/V unchanged,134new S changed,5clones exact.
All5 probability-one graph/otherwise identical serial margins are EXACT (0
difference), not merely within the unchanged .01 tolerance. Authority:
`runs/20261004_m1_lattice/r2_full_smoke_analysis/plumbing_summary.json`; actual
unit values in `r2_full_smoke/records/`. Peaks18.134567/17.310364GiB;
actual paired forwards282/174 including9/6diagnostics. Physical graph tokens
18069/5115 for84/50recognized windows,420/250complete paths,epsilon/zero counts0.
These checks support identical full333 dispatch, not a performance or mechanism
claim. The short-sample extrapolation remains descriptive; known full source
acquisition59.95min is still charged and supersedes any smaller sample projection.

Full R2 submittedsc474399/Slurm125 after the actualfixed5 PASS. Same code and
configuration, full215/118; no R2 GT or score-based selection before completion.

R2 controls are declared/prepared before their scores, not executed:
`path_control_{reader,measure,analyze}.py` and
`launch/{lab2_path_controls.sbatch,run_path_control_analysis.sh}`. Exact same
six arms as R1, now applied to complete hypothesis paths:
full calls the productionR2 reader; onebest ordinary serial reads the actual
highest-scoring complete text with the same scaffold/question; flat uses ALL
identical compiled full-path tokens but serial causal bias/positions; binary
keeps exclusions/positions with every finite bias0. wrong_mass rotates merged
path masses byfloor(K/2), retaining tokens/order/positions and epsilon mass.
Groups containing a zero-mass path remain unchanged to preserve physical token
inventory; singleton/no-op cases explicit. Report both changed float masses and
changed BF16 logmasses, not just intended perturbations. wrong_audio_window uses
the original available-window rotation and truthful donor coordinates plus
literal destination instruction, with the same unmatched-text scope limit asR1.

Native prefix/ownstance/V/referenceS are acquired once, six newS reads are
independent; firstavailable cloned replay for each arm adds6diagnostics/video in
smoke only. Actual physical calls, every arm's separate reading/diagnostic cost,
current source/compiled-ID/geometry/native processor binding and atomic records
are saved. Unchanged actual acoustic acquisition is physically shared but charged
to each standalone method for new-video inference. All7streams call the canonical
evaluator/fixedr6. full must exactly reproduce completedR2 main in noGT prepare.
The launcher hard-guards completedR2 performance PASS and333coverage BEFORE any
GPU reader, never passes that summary into scoring. Additional6arm planning
estimate20–60GPUmin/333, unmeasured. Independent narrow code review PASS:
`docs/reviews/20261005_m1_lattice_path_controls_code.md`, with actual CPU evidence
in `runs/20261004_m1_lattice/path_controls_code_review/`. Real36-layer BF16
six-arm forward/clone/exception tests, actualfixed5 CPU input binding and17
corruption cases, independentR2/native/full stub parity,14canonical command
checks and4launcher guard cases passed. Author actualfixed5 CPU compilation
also confirms identical full/flat/binary/wrong-mass token IDs and effective
BF16 mass changes; evidence `path_controls_cpu_compile/` under the run root.
These are CPU preparation checks; no actual control GPU/GT analysis. Claims
and rule14g gates remain exactly as declared.

R2 full333 acquisition completedsc474399/Slurm125 at07:26:23 on2026-10-05.
Both output and unchanged source input returned locally immediately, without
checksum/delete options; transfer logs `r2_main_{run,input}_transfer.log` under
the run root. Detached CPU noGT prepare and canonical evaluation are running;
no completed R2 performance or mechanism claim yet.

## R2 completed result and R3 predeclaration, 2026-10-05

R2 complete333 native raw G/windows/curves and six final metrics EXACT. Authority
`runs/20261004_m1_lattice/r2_full_main_decoded/optimized/metrics.json`:
HateMM ROC/PR/within .8962667642452506/.6774431550730213/.8024099744861634
(84eligible); HateClipSeg .7129393756053307/.6678464475414709/.6392448429477551
(99eligible). HMM within+.05162793052019643 with95% pairedCI
[.01988807419459929,.0882880597997688] is preserved positive progress. HMM
PR−.01679144827318968; HCS within+.0018957291436563217. No common dual-corpus
gain and a loss outside noise: performanceFAIL. Six R2 control arms remain
prepared only, no controlGPU and no mechanism claim. Raw shared speech within
changes+.051619701603807265/+.023100667764692343, raw max+.027426282117242944/
.004811282404509483; native visual exactly unchanged. All development-selected.
Cost authority `r2_full_main_analysis/alignment.json` under the run root:
new-video35.076396/37.373454min, total72.449850min/native7.883060 times, including
the actual unchanged ASR59.950735min. 6580new S; reader11291/10227 actual paired
forwards, diagnostics0; graph752874/842297tokens and17182/15696complete paths.
No epsilon/zero-mass paths; peaks17.823454/17.807392GiB. Physical reuse of ASR
does not remove its standalone new-video cost.

Postscore R2 error analysis read the actual native/optimized raw and fixed-r6
predictions, summary/per_video, `data/gt_4fps/{HateMM,HateClipSeg}.npz`, and the
actual ASR cache. Exact read paths are in
`runs/20261004_m1_lattice/r2_error_analysis/summary.json`; diagnostics and actual
source texts in `per_video.json`/`examples.json`. Native ASR “Music”/repetition
misses local language recovered by beams, consistent with positive within cases.
Counterexamples include non_hate_video_188 recovered profanity (newS+11.091 vs
native−14.608, GTnegative), hate_video_114 first-window profanity (newS+13.800
vsnative−17.627, GTnegative), HCS hearing questions/quoted threats and lyric
profanity with positive newS in GTnegative windows. These are descriptive cases,
not an automatic profanity filter or proof that any one cause explains the loss.
Frame-weighted raw speech shifts positive/negative are−2.090669/+.304951 onHMM
and−1.019886/+.426233 onHCS. No label enters scoring, fitting, constant selection
or thresholds; the next design is explicitly development-selected.

An additional descriptive key-budget calculation read the actual R2 graph records
and completed per-window error analysis; saved in
`r2_error_analysis/path_key_budget.json`. Mean weighted word-key mass is43.883710/
53.666988 per available window (not measured realized attention), while merged
path mass sums to1. Almost5 distinct paths/window contain218.922361/268.162050
tokens. Thus a hypothesis contributes its mass once per token to the query's
prior key inventory; R2 preserves sentence dependencies but still exposes every
candidate word directly. Long repetitive hypotheses can supply hundreds of keys.
This is a mathematical/readout observation, not a proven cause of the errors.

R3 hypothesis: encode each complete uncertain utterance, then read one end-state
per path instead of all of its word keys. This is revision2/3 of the same family,
not a new candidate. No R3 GPU/GT result yet. Sources,5beams, scores/FP32softmaxT1,
seed0,448ASR cap,8s/16k availability, G/own hard stance/V/policy,4fps/fixedr6 and
the literal R2 scaffold/question are unchanged on both corpora. Append exactly
one literal newline `\n` token to each nonempty positive-mass complete path;
assert the frozen Qwen tokenizer encodes this literal as one token. It is a
neutral boundary, not an independently generated label or class vote. Exact
duplicate texts merge as before; epsilon/zero mass remain no-KV and unnormalized.
Each boundary has its path's consecutive logical position and sees native prefix,
scaffold and its own complete utterance at all36BF16 layers. Candidate histories
remain mutually exclusive. Every query/tail row sees native prefix/scaffold,
earlier query/self and ONLY the path boundary keys with log merged path mass;
all candidate lexical keys are−inf for that query. Query logical start follows
the maximum path length including boundary. Native full ASR/overview remain
visible; no global hard isolation or calibrated acoustic-posterior claim.
One final Qwen newS; no hypothesis classification, output average, rerouting,
score calibration, postprocessing or label-dependent vocabulary filter.

R3 `terminal_{graph,reader,measure,analyze}.py` and corresponding launch files
preserve R1/R2 files/results. Atomic current-source/CPU native-prefix/geometry/
token/margin/cost replay remains required. Hand masks and independent real36-layer
CPU tests must establish endpoint-only query visibility and exception restoration
before GPU. Actualfixed5 native allraw exact and cloned R3 replay exact remain
mandatory. A one-path end-state reader does intentionally differ from an ordinary
all-word causal query; it instead must agree within the unchanged .01 tolerance
with a separately constructed one-path end-state matrix on IDENTICAL token IDs.
This reference is independently checked, uses the same boundary-only operator,
and never substitutes for native parity. No performance or numeric tolerance
is relaxed. Empty graph reduces to the original scaffold/query causal reader.

Physical new ASR calls0 with cache reuse. New videos still pay full59.95min source
acquisition for333; additional Qwen reader planning10–25min, total70–90min,
unmeasured until smoke/main. At most5additional boundary tokens/window; input
encoder/decoder acquisition count unchanged. One newS per available window;
fixed5 clone/unit/reference adds3diagnostics per eligible video, excluded from
standalone deployment cost but reported physically. Full333 has no diagnostics.

If and only if complete R3 main passes, declare/review/run matched controls before
their scores: ordinary-onebest with boundary and same scaffold, flat with all
identical R3 tokens/serial positions, binary endpoint accessibility with finite
bias0, wrong-mass with fixed ID inventory, truthful wrong-audio window and an
all-word query operator on otherwise identical boundary tokens. Structural novelty
must beat both onebest and matchedflat in the same main metric≥.01 on BOTH corpora;
the terminal bottleneck can only be a novelty component if its removal also drops
a common main metric≥.01 on BOTH corpora. The historical R2 is descriptive only:
its token inventory differs, so it is not the matched all-word ablation. Require
raw ordering, paired intervals, actual intervention coverage and final independent
mechanism assessment. No R2 control run or historical-number hybrid rescues a
failed R3 gate. Otherwise rule9/max3revisions governs the next branch.

R3 author hand graph/mass/end-state accessibility/position/empty/underflow oracles
and actual frozen Qwen one-token newline check PASS:
`runs/20261004_m1_lattice/terminal_cpu_checks/selfcheck.log`. Independent narrow
code review PASS, same-family provisional:
`docs/reviews/20261005_m1_lattice_terminal_code.md`; actual evidence in
`terminal_code_review/` under the run root. Real36-layer BF16 attention/clone/
nonempty-and-empty unit/exception restoration tests, actualfixed5 current source/
CPU-expanded native prefix with18corruptions and canonical identity/4subprocess
checks passed. Two diagnostic-only bugs were fixed and independently confirmed:
FP32 duplicate mass can sum to1.000000014901 rather than exact Python1, and empty
unit paths require no terminal key. Production path mass/readout unchanged;
.01 GPU reference tolerance unchanged. Actual8B fixed5/full333 remains pending.

Actual R3 fixed5 Slurm128 completedsc474399 at07:43:53; both outputs and unchanged
inputs returned locally immediately. NoGT prepare PASS, authority
`runs/20261004_m1_lattice/r3_full_smoke_analysis/plumbing_summary.json`: native
all G/window/curve exact, G/V unchanged,134newS,5clones exact. All5 independently
constructed unit end-state reference margins are EXACT (difference0), not merely
within.01. Values−13.705020905/−11.671485901/−8.203338623/−11.155235291/
−2.231147766 in the declared fixed5 order. Actual282/174paired forwards include
9/6diagnostics; peaks18.125379/17.311294GiB;18489/5365graph tokens,420/250terminal
paths. Smoke proves plumbing only; noGT/metric selection. Full333 dispatch uses
identical R3 code and constants after code sync/current machine check.

Full R3 submittedsc474399/Slurm129 after current machine/code check and actual
fixed5 PASS; full215/118, same configuration. No main score-based change during
this acquisition. R3 controls are prepared before any control score:
`terminal_control_{reader,measure,analyze}.py` and
`launch/{lab2_terminal_controls.sbatch,run_terminal_control_analysis.sh}`.
Seven arms: full actualproductionR3, ordinaryserial onebest including its newline,
flat ALL identical R3 tokens with causal/serial geometry, binary original
endpoint-only geometry with finitebias0, wrong_mass fixedpath token/order rotation,
truthful wrong_audio_window, and allword. allword preserves EVERY R3 input token,
head/tail/logical position/path-history exclusion; only the tail-query graph
access changes from weighted terminal keys to weighted ALL lexical/terminal keys.
It uses the original R2 graph accessibility on the R3 graph, not historical R2
scores or its different token inventory. This isolates the end-state bottleneck.

Current-source/native CPU expanded-prefix/compiled-token/geometry/operator flags,
atomic records/resume and native/full exact reference are bound before canonical
evaluation. All8streams use the single canonical evaluator and fixedr6. Smoke
adds7clones per video with available audio; full none. Actual arm calls/timing,
shared physical joint cost, repeated standalone source cost and BF16 quality/source
perturbation coverage are separate. Existing zero/singleton/noop/truthful donor
limits apply. Launcher hard-guards complete R3 main performancePASS,84/99 and333
before GPU/model reads. Planning additional7arm acquisition25–70GPUmin/333,
unmeasured, actual source cache reuse means0physical ASR calls but each standalone
arm includes the actual source acquisition59.95min. Full must beat BOTH onebest
and flat by the same main metric≥.01 on BOTH corpora; end-state novelty additionally
requires matchedallword removal≥.01 on BOTH corpora. Report the intersection of
these metrics too; not just intended masks. Raw ordering/intervals/actualcoverage
and final independent mechanism review are still required. Independent narrow
control code review PASS, same-family provisional:
`docs/reviews/20261005_m1_lattice_terminal_controls_code.md`; actual evidence in
`runs/20261004_m1_lattice/terminal_controls_code_review/`. Real36-layer BF16
seven-arm504layer checks/clone/exception restoration, same-ID/same-position
allword-only-query access, actualfixed5 CPU source/native binding19corruptions,
independentR3 stub reference prepare exact/corruption rejection,16canonical
subprocess flags and4dispatch guard cases all PASS. Author actualfixed5 CPU
compiled IDs also match actualR3 main-reader smoke IDs exactly; full/flat/binary/
wrong_mass/allword inputs identical and effective BF16 mass changes confirmed.
Evidence `terminal_controls_cpu_compile/` under the run root. These are CPU
preparation checks only; no actual controlGPU/predictions/GT yet.

## R3 final and branch closure, 2026-10-05

R3 complete333 native raw G/windows/curves and six final metrics EXACT. Authority
`runs/20261004_m1_lattice/r3_full_main_decoded/optimized/metrics.json`:
HateMM ROC/PR/within .8924204974096293/.673876055773841/.7501122925823352
(84eligible); HateClipSeg .7056003314314789/.6449858131158581/.6014832974536528
(99eligible). Delta HMM−.004698164/−.020358548/−.000669751; HCS−.011224532/
−.026086271/−.035865816. PerformanceFAIL, anyqualifyinggainFalse. Raw max within
−.053077855/−.042481294 with both95% CIs wholly negative; raw shared speech
−.021648806/−.042145964. No realized improvement supports the end-state readout.
G/V unchanged,6580newS actually entered scores. Development-selected.

Actual new-video cost 35.126324/37.420590min, total72.546913min/native7.883237 times,
including unchanged actualASR59.950735min. Graph770056/857993tokens including
17182/15696terminal keys; reader11291/10227paired forwards, diagnostics0,
peaks17.823454/17.807124GiB. Detailed authority
`runs/20261004_m1_lattice/r3_full_main_analysis/alignment.json`.

R3 noqualifyinggain triggers rule9 archive. Initial+two revisions ran; noR4
or full controls, and no mechanism/promotion claim. R1 HMM within+.048219936
and R2+.051627931/positive paired intervals remain preserved with all six
metrics/costs above; never blend their numbers across versions/corpora. All
inputs/outputs local. Three control sets are only prepared/independentCPU-code
PASS, not control results. Next independentcandidate24 may start because
candidate23's actualfixed5 cannot exercise its typed-program mechanism and
requires an explicitly new interface before any main. Formalr6 unchanged.

**2026-10-09 disk cleanup (user-approved, category A):** the derived cache data/temporal_speech_lattice (with its PROVENANCE.md) was deleted on sc474397. Run outputs and metrics under runs/ are kept.
