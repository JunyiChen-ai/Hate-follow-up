# M1 candidate22 backup: frozen temporal speech lattice reader

Declared 2026-10-04 while candidate21's complete run is in progress. Prepared
independent backup only: CPU implementation prepared, no GPU output or GT analysis. Run
only after the current candidate's result-based branch permits switching. Formal
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
