# Candidate38: lexical discourse partition and source-scoped speech

Selected from the unchanged 20261006 nine-candidate pool, rank2. Its one
independent Rule4 PASS is `docs/reviews/20261006_m1_ideation_jury.md`;
same-family provisional. This is not a new review or a reset of Acoustic20,
Lattice22, QuoteGraph24 or Provenance25. No main predictions/GT or new ASR
were read to choose these constants. All eventual comparisons are
development-selected. The complete source/reader/strict replay/canonical CLI
and Slurm prototype is implemented; actual8B/Whisper measurements and its one
Rule6 independent review are still required. No GPU experiment is authorized
by a CPU PASS alone.

The hypothesis is that a literal act should be attributed to its actual8s
window, while a lexically connected utterance elsewhere can explain it.
Unrelated-topic attacks should not become direct evidence for this window.
This replaces S only: native20(actual18–20), full original ASR, original G,
its own hard Yes/No stance, independent native V, max(V,S), fixed r6 and
canonical4fps are unchanged. Both fixed datasets use the same pipeline.

## Sources and declared adaptation

Actual primary reading on2026-10-06: [Hearst ACL1994 algorithm section](https://people.ischool.berkeley.edu/~hearst/papers/tiling-acl94/acl94.html),
tokenization/block cosine/one smoothing/depth/spacing; [official Whisper timing
implementation](https://raw.githubusercontent.com/openai/whisper/main/whisper/timing.py),
alignment heads, teacher-forced cross attention, normalization/median/DTW/word
boundaries and later heuristics. Source scope saved under this run's
source_reading/. The implemented Unicode/no-stop-list/mean-depth/no paragraph
snapping is a declared adaptation, not a reproduction of the paper's hidden
image threshold. The source parser and local speech measurement are our
composition. DTW estimates time; it does not verify an act's semantics.

## Frozen R1 specification before GPU

`spec.json` is the machine-readable authority. New ASR uses the same frozen
Whisper-large-v3, greedy beam1, max448 total decoder tokens, transcribe/no
timestamp output and one language identification on first actual audio.
Actual audio is placed at resampled16000Hz PTS relative to the video's origin;
overlap keeps the first observed sample, gaps are zero at their original times,
not compressed. Non-overlapping nominal30s blocks cover actual video time.
Empty audio blocks do not create words. Each block generates a separate single
transcript and teacher-forces those actual tokens for official-head DTW at20ms.
Save raw DTW and used clipped times separately; no subsequent duration/median
word heuristics, no proportional segment word times. A capped ASR block remains
explicitly incomplete and its observed token prefix remains attributable; no
retry or continuation invents missing words. Distinct words on different block
sides retain distinct IDs even if their literal text repeats; no transcript
overlap is introduced. Word assembly follows actual HF original-token groups:
whitespace/punctuation for spaced languages, Unicode for zh/ja/th/lo/my/yue.
No punctuation merge changes their times. HF eager attention already contains
softmax probabilities, so normalize it without a second softmax. Use the
Whisper generation-config official10 alignment heads, FP32 per-head weights
normalized across all teacher input rows, width7 reflection median, then mean
heads and rows from no-timestamps through the last content token (exclude final
teacher EOS). Zero standard deviation becomes zero normalized weights. Crop
to ceil(actual samples/320) encoder frames at20ms, bounded by encoder length.
DTW costs areFP32, official CPU strict diagonal/strict up/otherwise left ties;
save raw jumps before group lookup. Cumulative word token boundaries index
those jumps; clip only to actual crop endpoints and retain raw endpoints too.
Teacher EOS closes alignment when it fits the448 decoder capacity. A capped
prefix4+content444 preserves every actual token and teacher-forces those448
rows without an extraEOS; its terminal content query supplies the final DTW
boundary. This explicit capacity adaptation normalizes the actual448 rows
and does not silently drop the last word. Teacher EOS is never a generated
word or evidence. These details are fixed before source extraction.

Lexical sequence: Unicode letter/number runs with internal apostrophes,
casefold, no stemming or stop list. A source word with multiple lexical runs
contributes each run but preserves the same original word ID and midpoint.
Pseudo sentences contain20 lexical runs, including a final short sentence.
Only gaps with6 complete pseudo sentences on each side have a cosine;
shorter documents are one segment. Cosine uses raw frequency vectors, zero
norm returns0. Smooth once with the available adjacent gap scores in width3
(no invented padding). At each gap follow nondecreasing scores left/right
until the first decrease to find peaks. Depth is both peak differences.
Eligible valley plateaus choose their earliest gap; flat zero-depth plateaus
do not split. Retain depth strictly above the mean of all valid gap depths.
Visit depth descending, tie early; accepted boundaries must be at least3
pseudo sentences apart. Do not cut an original source word even if a multi-run
word crosses a pseudo boundary: advance to the next original word ID, dedup
resulting identical boundaries. Segments partition original words once.

Each8s half-open window owns source words whose used time midpoint falls
inside; exact8s is owned by the next window, a midpoint at/after manifest
duration is explicitly out of scope. Every lexical segment touched by a LOCAL
word contributes its other original words as CONTEXT. Keep IDs/times/order;
no fixed neighborhood is substituted. More than2048 source words or6144
encoded parser input tokens rejects the whole parser packet with a recorded
reason; nothing is silently truncated. Grammar-selected span endpoints refer
to actual word IDs, up to2 spans/role, each at most32 original words, same
role and contiguous supplied original IDs. UNKNOWN/invalid/incomplete parser
uses a fresh direct-aligned-LOCAL S; UNKNOWN never means no hate. No LOCAL
words uses a fresh original S if the original window has speech, otherwise
keeps native absent-S. Source model selections can be wrong even when IDs are
valid. Execution guard requires at least one compiled LOCAL act in each
dataset, exact native allraw/G/V/stance, real input/times, and untouched cloned
native cache before a full run; no GT or score is used in this guard.

The final new S is a fresh independent branch of its own native stance cache.
The original speech question/window coordinates remain verbatim after the
declared source instruction and a literal compiled record. Native prefix/full
ASR is not edited. This record lists accepted LOCAL acts and explaining
CONTEXT with actual times, never relocates the latter to the current window.
Fresh fallback also remains independent of V. One window still emits one S.

## Cost and falsifiable mechanism

For B=ceil(T/30), Ns aligned-speech windows: B greedy ASR generations and B
teacher-forced alignments, one language detection, up to Ns grammar parses and
Ns new S measurements; V/G stay native. Original ASR/native20 acquisition and
all new audio decode/transfer/Whisper/parse/branch work are charged for a new
video. Paired native S, repeats and controls are experimental overhead and
reported separately. Initial unmeasured5090 estimate for120s is60–240s added
work, plus native costs; long-topic parser prefill may be more expensive.
No claim that frozen/cached extraction removes this deployment cost.

Claim one complete part: lexical partition scoped LOCAL-act/CONTEXT compiler.
Deletion uses the exact same new words/times and fresh direct LOCAL S, so
better ASR alone cannot prove the claim. Additional matched-nearby-context,
uncompiled-role, circular wrong CONTEXT and swapped LOCAL/CONTEXT interventions
are mandatory before mechanism success. No new corpus, ensembling or hate
score postprocessing. Main gates compare all6 final metrics to
`runs/20260926_twolevel/r6_bma/metrics.json`: one same primary metric >=+.01
on BOTH, pooled losses<=.005, within losses<=.01. Claimed part deletion must
lose one common primary >=.01 on BOTH, with real raw V/S/max ordering and
wrong binding controls. Whole333 and all sources must run on one host and be
returned to ROOT before any STATUS claim. No actual performance exists yet.

## Software evidence before GPU

`runs/20261006_m1_texttiling/operator_cpu_checks/summary.json` validates the
frequency cosine against independent dense vectors, flat/short/empty streams,
an800-word two-topic boundary, Unicode and original-word coverage, exact8s
midpoint ownership. `alignment_cpu_checks/summary.json` matches15 official
DTW path examples, exact official-head probability normalization/median
matrix, zero-std/short-tail and actual Whisper tokenizer literal coverage.
Initial test-only short-token endpoint assumption was corrected; failed log
preserved, no scientific timing operator changed.

`compiler_cpu_checks/summary.json` validates actual Qwen tokenizer prefix-free
span options, shared Stream replay, wrong role rejection and frozen question
suffix; initial fixture incorrectly expected context beyond its own topic
boundary, corrected to one-topic fixture, failure kept. No compiler algorithm
or execution guard changed. `whisper_cpu_checks/summary.json` exercises actual
HF32 encoder+32 decoder layers/20 heads with official10 heads and random
narrow weights on1s synthetic features; not pretrained recognition or ASR
truth. All are CPU implementation evidence, not actual8B or performance.

One fresh independent Rule6 instance is reviewing the final prototype and may
write its own meaningful CPU evidence under `independent_code_review/`.
Actual333 raw/native/real processor preflight is running; future source word
capacity and actual source semantics are not thereby validated. No new source
cache has been generated, no GT read, no main result and no budget reset.

Independent Rule6 confirmed only B1 capped teacher-capacity and B2 incomplete lexical block fixes. No content tokens were dropped and no execution guard changed. Actual narrow random-weight HF production Recognizer.block greedy reached448 tokens/retained444 content/445 DTW rows; actual36-layer random Qwen production reader/parser/validator exercised fresh S, ten complete native KV restorations and wrong role/source text/native pixel rejection. Evidence runs/20261006_m1_texttiling/independent_code_review/{summary,fix_summary,cost_summary}.json. This is software evidence only, not actual pretrained8B/ASR semantics/performance. Actual333 raw/native/real-processor preflight PASS runs/full_input_preflight/summary.json. The one Rule6 review is PASS at docs/reviews/20261006_m1_texttiling_code.md (same-family provisional). No GPU started; actual fixed5 is next.
