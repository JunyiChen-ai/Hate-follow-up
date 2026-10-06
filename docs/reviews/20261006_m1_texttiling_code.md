# Candidate38 Rule6 code review — PASS after two specific fixes

2026-10-06, independent agent instance, same-family provisional. This is the
single Rule6 review of `experiments/20261006_m1_texttiling/`, including the
promoted shared `src/audio_inputs.py` and `src/native_input_binding.py` and
inspection of their existing shared reader/generation/evaluation interfaces.
The Rule4 rank2 PASS is not reopened. No author code, ReKV implementation,
other experiment, evaluator, GT, or model weights were modified by this review.

**PASS: no known unresolved bug that changes the declared experiment's
observations or conclusions, after B1 and B2 below were fixed and checked.**
This is implementation evidence only, not source-quality, actual8B, GPU,
performance, mechanism, or promotion evidence. It does not stop the research
iteration; the complete measured experiments and mechanism interventions
remain necessary.

## Bugs found and narrowly rechecked

1. **B1 — capped Whisper teacher input exceeded capacity.** The original
   `Recognizer.block` appended teacher EOS unconditionally. Greedy generation
   reaching 448 tokens already contains four prefix tokens plus 444 observed
   content tokens, so the teacher input became 449 rows. A real HF decoder
   with max_target_positions448 rejects row448. The fix preserves all actual
   generated content: teacher EOS is included only when capacity permits;
   otherwise the final content query closes the boundary and the no-EOS
   attention slice still supplies L+1 DTW rows. The distinction is recorded
   as `teacher_eos`, validated, and explicitly declared in the specification.
   Independent production `Recognizer.block` execution with the real HF
   generic greedy generator reached 448 tokens, retained all444 content
   tokens, and returned445 DTW rows. This was a narrow random one-layer CPU
   Whisper, real tokenizer, fixture suppression allowing one ordinary token,
   and CUDA-to-CPU tensor placement translation. Its generated text is not
   a transcript-quality observation. No generated output was mocked.
2. **B2 — final incomplete pseudo sentence counted as complete.** The original
   gap range used the ceiling number of pseudo sentences; a239-word sequence
   incorrectly offered a gap requiring twelve complete20-word sentences.
   It now uses the floor number of complete sentences for eligible gaps,
   while retaining the final partial sentence and all original source words
   in the partition. Independent239/240/259/260-word checks produce gaps
   `[]/[6]/[6]/[6,7]`, respectively, with full source-word coverage.

Evidence: `runs/20261006_m1_texttiling/independent_code_review/fix_check.py`,
`fix_summary.json`, and `fix.log`. The original449-row real decoder rejection
is also recorded in `summary.json`. Its239-word check ran after the author's
fix and is explicitly final-state evidence, not an archived old-code run.
Only these two bugs were rechecked after their fixes; no second general
review or new proposal gate was introduced.

## Scientific path inspected

- Source audio uses actual resampled PTS, explicit zero-filled internal gaps,
  first-observed overlap ownership, and actual crop endpoints. Nominal30s
  windows do not compress missing time. Shared source decoding is replayed
  against saved sample and observation arrays before cache reuse/analysis.
- The same Whisper generates greedy content and teacher-forced alignment;
  production uses the generation-config official alignment heads. The eager
  probability matrices are not softmaxed twice. FP32 normalization, median
  filtering, CPU DTW tie rules, raw jumps, actual tokenizer word groups and
  clipped versus raw endpoints follow the declared adaptation. No new label
  or prediction enters this source path.
- The complete lexical chain is present: normalization, complete blocks,
  frequency cosine, one smoothing pass, peak depth, valley-plateau selection,
  mean-depth threshold, spacing, original-word-safe boundaries and partition.
  LOCAL ownership is half-open midpoint ownership; CONTEXT consists of other
  words in touched lexical segments, not a substituted fixed neighborhood.
- Grammar choices constrain literal original IDs, contiguous same-role spans
  and lengths. UNKNOWN/capped/oversized packets select a fresh fallback S;
  UNKNOWN is not interpreted as a no-hate score. Accepted records explicitly
  separate LOCAL acts and explaining CONTEXT and enter a fresh S branch.
  The original question remains verbatim at the end of the new question.
- Native20/full original ASR, own G and hard stance, independent native V,
  max(V,S), canonical4fps and fixedr6 commands remain in the execution path.
  Production cache branches restore their native lengths and rope state.
  Analysis requires exact paired native allraw results before annotation
  evaluation and checks whole coverage and compiler execution per corpus.
- Source/current native input/runtime records are replayed before completed
  reader records are reused. Native pixels are compared as actual arrays;
  parser prompts, original IDs, scope and generated token traces are replayed.
  Source and reader host agreement plus whole333 coverage prevent mixing a
  pilot cache from a different host into a claimed complete run.
- Source decode/generation/alignment/proof writes and reader native/parser/new
  speech/binding times are recorded. Setup, validation, record I/O, failures
  and resume work remain visible in pipeline attempts; standalone per-video
  times are not total experimental elapsed time. Paired native S and clone
  checks are separately identifiable overhead. Both extraction and reader
  injected setup failures retain a failed attempt with positive elapsed time.
- Only the analysis evaluation stage opens the canonical evaluator/GT path.
  The command capture checks the existing `src.eval.evaluate_four_datasets`
  and fixedr6 `nscore/bma/length/min_windows2/grid6/m2` arguments. No evaluator
  implementation is duplicated or modified by this candidate.

## Independent CPU execution evidence and limits

`runs/20261006_m1_texttiling/independent_code_review/check.py` exercised the
actual HF36-layer Qwen model class with narrow random weights, a synthetic
tokenizer,20 synthetic images, and explicit synthetic timed speech. Actual
production `read_video`, `parse`, structured Stream forwards and bundle
validation ran. The span-choice schedule is an explicitly controlled fixture,
not a pretrained model selection. Ten native/fresh/clone branch invocations
preserved every original KV tensor. Fresh S changed; the no-LOCAL/native-S
fallback stayed exact. Wrong-role spans, changed source text and changed
native pixels were rejected. The original question remained verbatim.
Results are in `summary.json`, with the generated fixture bundle in
`bundle.json` and captured evaluator commands in `commands.json`.

The same script independently checked negative leading audio placement,
first-observed overlap and internal zero gaps using synthetic sample arrays.
This does not claim that all333 actual media timelines have been decoded.
`cost_check.py` and `cost_summary.json` provide the injected failure evidence.
All review-created files are below the review's own run directory, apart from
this single review document. No GPU, Slurm submission, pretrained weights,
GT read, content hash, or Git identifier was used by these checks.

Author evidence inspected separately includes lexical/operator, real-tokenizer
compiler, alignment primitives, and32+32-layer random Whisper teacher checks.
The333 input preflight establishes raw/native/processor availability only.
Neither it nor any random-weight test establishes fresh ASR correctness,
actual8B parser utility, runtime GPU capacity or performance. Source semantic
errors and the mechanism's effect remain empirical questions for the
already-declared complete runs and deletion/wrong-binding interventions.

## Narrow launcher activation correction — independently confirmed PASS

2026-10-06, same reviewer instance and same Rule6 review. The parent reported
owned job223 was canceled while PENDING before source/GPU execution; that
scheduler history is parent evidence in
`runs/20261006_m1_texttiling/launcher_environment_fix.txt`, not an independent
GPU observation. The actual lab1 deployment defect was its activation of a
named conda HateVLM environment instead of this host's repository-local venv.

Independently confirmed the corrected lab1 line
`source .cache/envs/HateVLM/bin/activate` after its repository `cd`. Under
`set -euo pipefail`, actual activation selects the repository's
`.cache/envs/HateVLM/bin/python` and imports torch2.11.0+cu128,
transformers5.15.1 and av18.1.0 on CPU. `bash -n` passes. Executing the actual
launcher control flow with only `nvidia-smi` and Python payload commands
stubbed confirms smoke sends `--smoke` to extraction then measurement, main
sends neither flag, and invalid scope exits2 before either payload. Lab1
partition, GPU count and memory directives remain intact. No GPU query,
model/source payload, submission, Git operation or GT read was performed.

Evidence: `runs/20261006_m1_texttiling/independent_code_review/launcher_fix_check.py`,
`launcher_fix_summary.json`, and `launcher_fix_run.log`. This confirms only
the concrete environment/launcher correction, not a new general review,
scientific variant, actual GPU readiness or performance result.
