# Duplex reading kill-test: postmortem

Sources: `docs/duplex/PREREG_duplex_killtest.md` (frozen 2026-08-05, plus the same-day
batch-size amendment), the two report JSONs in `docs/duplex/reports/`, and the ten scored
files under `results/duplex_probe/ImpliHateVid/`. Every figure below was recomputed from
those files.

## Verdict

**KILL.** The trigger is the P5a clause of the pre-registered kill rule, and it fires on
both models. P5a required the effort placebo to separate IM from EX at least 0.05 AUC below
the pragmatic reader. The observed gap is +0.0051 on the 2B (0.5047 versus 0.4996) and
+0.0027 on the 8B (0.6069 versus 0.6042). Whatever the divergence statistic D measures, the
thoroughness-matched placebo reproduces it almost exactly, which is the pre-registered
signature of an effort effect rather than a pragmatic one. The hypothesis was that implicit
hate lives in the gap between a surface reading and a full reading of the same video. What
died is the claim that a prompt can open that gap: telling the model to read harder moves D
as much as telling it to read for conveyed meaning. D measures effort, not pragmatics. Per
the pre-registration, the method is not to be built.

## Pre-registered results

The retained sample is identical on both models and across all five readers: IM 264, EX 259,
NH 632, total 1155 of the 1283 `train_clean` videos. Both lit variants pass P4a and P4b on
both models, so the selection rule takes the more minimal variant, `lit_v1`, on both. All
statistics below use it.

| # | Statistic | 2B | 8B | Threshold | 2B | 8B |
|---|---|---|---|---|---|---|
| P1 | AUC(IM vs EX) via D | 0.5047 | 0.6069 | ≥ 0.60 | fail | pass |
| P1 | AUC(IM vs NH) via D | 0.5660 | 0.7738 | ≥ 0.60 | fail | pass |
| P2 | IM share, (lit-low, prag-high) | 0.117 | 0.042 | top cell for IM | fail | fail |
| P3 | AUC(D) in mid s_prag tercile | 0.5344 | 0.6510 | ≥ 0.55 | weak | pass |
| P4a | AUC(EX vs NH) via s_lit | 0.8886 | 0.9569 | ≥ 0.70 | pass | pass |
| P4b | median IM z_lit − z_prag | −6.0e-08 | −2.9801 | < 0 | pass | pass |
| P5a | AUC(IM vs EX) via D_effort | 0.4996 | 0.6042 | ≥ 0.05 below P1 | **fail** | **fail** |
| P5b | AUC(IM vs EX) via D_para | 0.4585 | 0.5313 | ≤ 0.55 | pass | pass |

Median D by group, prag minus lit_v1: the 2B gives 6.0e-08 (IM), 6.0e-08 (EX), 0.0 (NH); the
8B gives 2.980 (IM), 1.460 (EX), 0.0 (NH). The 2B P4b pass is nominal only, since −6.0e-08 is
a float-rounding artifact rather than measurable suppression. The discarded lit variant is
close on both: `lit_v2` gives P4a 0.9020 and IM suppression −0.2500 on the 2B, P4a 0.9559 and
−2.6051 on the 8B. Quadrant occupancy at pool medians:

| Model | Group | low_low | low_high | high_low | high_high |
|---|---|---|---|---|---|
| 2B | IM | 0.223 | 0.117 | 0.049 | 0.610 |
| 2B | EX | 0.093 | 0.093 | 0.031 | 0.784 |
| 2B | NH | 0.759 | 0.047 | 0.043 | 0.150 |
| 8B | IM | 0.159 | 0.042 | 0.004 | 0.795 |
| 8B | EX | 0.046 | 0.023 | 0.004 | 0.927 |
| 8B | NH | 0.813 | 0.040 | 0.013 | 0.134 |

## The IM fingerprint: failure A

The pre-registration named two diagnostic fingerprints on the IM subset. The data shows
failure A, the literal reader leaking. IM videos concentrate in the high-lit, high-prag cell
at 0.610 on the 2B and 0.795 on the 8B, while the cell the hypothesis predicted, low-lit and
high-prag, holds 0.117 and 0.042. The literal reader gives IM videos a median s_lit of 0.269
on the 8B against 1.1e-07 for NH, so it is reading implicit hate it was instructed to ignore.

Comprehension is automatic. Instructing a model to judge only face-value meaning does not
suspend the pragmatic inference that already happened during encoding; it only shifts how the
model reports. On the 8B that shift is real (median IM suppression −2.98 logits) yet nowhere
near enough to push IM videos below the literal-reader median. Instruction-level control of
reading depth is not a usable instrument.

This also closes off the pivot failure B would have motivated. Failure B, lit-low and prag-low
together, would have meant the model lacks the coded knowledge, and would have argued for a
distributional-discovery paradigm that recovers it from unlabeled data. That is not what
happened. Both readers know, and neither can be made to forget, so a discovery paradigm
answers a question this experiment did not raise.

## The censoring artifact behind the 8B P1 pass

This section is post-hoc. It is not among the pre-registered arms, and no threshold in it was
fixed in advance.

The 8B nominally passes P1 (0.607 and 0.774), but the statistic is dominated by censoring. Of
the 1155 retained videos, 622 (53.9%) have D exactly zero, and all 622 are zero because both
readers were clipped to the same bound by the `EPS = 1e-4` guard in
`analyze_duplex_probe.py`, 509 at the low bound and 113 at the high bound. Censoring is
severely unbalanced across the groups P1 compares: NH 480/632 (75.9%), EX 90/259 (34.7%), IM
52/264 (19.7%). Because AUC scores ties at 0.5, a group that is three quarters ties is ranked
mostly by the tie rule rather than by anything the readers measured. Restricting to the 533
uncensored videos collapses P1 to 0.559 for IM versus EX and 0.518 for IM versus NH, both
below the pass line. The 8B pass is an artifact of the readout.

The two scales fail the same readout in opposite ways. On the 2B nothing clips at all (0% of
s_lit at either bound), but the scores are quantized: 99.9% of s_lit logits fall on a 0.25
grid, s_lit takes 21 distinct values, and D collapses to 11 levels on that grid. Exact zeros
account for 11.3% of D, rising to 31.9% once values within 1e-6 of zero count as grid-zero.
On the 8B the middle is continuous but the tails are censored, with 52.8% of s_lit at the low
clip bound and 9.8% at the high bound.

The lesson is about the instrument. A renormalized P(Yes) read from the next token and
clipped at 1e-4 before the logit destroys ranking information exactly where implicit content
lives, near the decision boundary on one model and in the saturated tails on the other. Any
successor probe must fix the readout before it tests a mechanism.

## Differential missingness

Exactly 128 videos were skipped, and the skipped set is byte-identical across all ten reader
files and both models, so the retained sample is comparable throughout. The skips are context
overflows, and the driver is frame resolution. Every skipped video has 1080p or larger frames
(126 at 1920×1080, one at 2560×1440, one at 1080×1920, all at least 2,073,600 pixels), while
the largest retained frames reach 1,163,520 pixels (two videos at 808×1440, then a tier at
1280×720). The separation is clean, with no overlap.

The cause is that `mm_processor_kwargs={"max_pixels": 100352}` in `score_duplex_probe.py`
never reaches the resizer. Under vLLM 0.11.0 with transformers 4.57.1,
`Qwen3VLProcessingInfo._get_vision_info` reads `image_processor.size["shortest_edge"]` and
`size["longest_edge"]`, so a bare `max_pixels` kwarg is silently dropped for Qwen3-VL. Frames
were processed at native resolution instead. A 1920×1080 frame resizes to 1920×1088 and costs
about 2040 vision tokens, so 16 of them cost roughly 32,600 against `max_model_len=32768`,
matching the observed cutoff.

The missingness is label-correlated. EX lost 66/325 (20.3%) and IM lost 60/324 (18.5%), while
NH lost 2/634 (0.3%). High resolution therefore tracks the hate labels in this dataset and is
a live shortcut candidate: a model keying on production quality alone would score above
chance. Future evaluations on ImpliHateVid must stratify by resolution, and must not let a
resolution-driven skip rule silently reshape the evaluated set.

## Execution record

Both stages ran off-cluster on a single RTX 5090 (32 GB) in `frames_16` `--no-video` mode,
python 3.12.3, vllm 0.11.0, torch 2.8.0+cu128, transformers 4.57.1. The 2B stage completed
clean in about 33 minutes for all five readers. The 8B stage did not. The batch-8
configuration hit CUDA OOM, and the pre-registration was amended the same day to
`--batch-size 2`, recorded as an execution-throughput parameter only, since greedy per-video
constrained scoring makes per-sample scores independent of batch composition. The 8B then hit
a GPU device fault (unspecified launch failure, requiring a machine reboot) and later a
memory-fragmentation OOM during the `effort` reader, restart 1 of 2. Both crashes killed the
engine mid-run and left null-score rows behind, and both times the polluted tail was stripped
per the crash protocol against the 128-ID reference skip set. File timestamps still show the
outage: 8B `lit_v1` landed at 10:19 and `lit_v2` not until 19:29, with the rest following at
roughly 14-minute intervals.

Two infrastructure issues are worth fixing before any successor probe. First, `get_media_path`
in `src/our_method/data_utils.py` checks only `frames/<vid>` and never `frames_16`, so the run
depended on a `frames -> frames_16` symlink in the data root. Second, `load_done_ids` in
`score_duplex_probe.py` adds a `video_id` to the done set whenever the field is present,
without checking whether `score` is null. A resume after an engine death therefore treats
null-polluted rows as complete and silently skips those videos forever unless the tail is
stripped first. That is a data-integrity bug, not an inconvenience.

## What survives

The model knows the content at both scales. The literal reader alone separates EX from NH at
0.889 (2B) and 0.957 (8B), and on the 8B its own median s_lit is 0.995 for EX, 0.269 for IM,
and 0.000 for NH. Coded material is visible to the model without hate-specific supervision,
which is the premise label-free framing rests on.

A single-call pragmatic signal is real. In the middle tercile of s_prag, where the score is
most ambiguous, s_prag alone separates hateful from normal at 0.665 (2B) and 0.771 (8B). D
adds nothing on top: the same tercile gives D 0.534 and 0.651, below s_prag on both models.
One well-specified reading is worth more than the difference between two readings.

Instruction-based suppression is dead as a mechanism. A successor that needs the model not to
see something must enforce that structurally, through a bottleneck that removes the evidence
from the input, rather than by asking the model to ignore it. The description-bottleneck
literal reader named in the pre-registration is such a structure, and it would require a fresh
pre-registration.

Wording sensitivity is a real noise source. P5b stays under the noise-floor threshold, but a
pure paraphrase of the pragmatic block still shifts the median score by one 0.25 logit quantum
on the 2B, in every group. Any effect smaller than one quantum is not distinguishable from
rewording.
