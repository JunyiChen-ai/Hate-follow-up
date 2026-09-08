# Label-free frame-level hateful-video localization: consolidated results

## Scope and why the two protocols stay apart

This document collects every frame-level hateful-video localization result
produced without any hate annotation, across two repositories: `Hate-follow-up`
(this repository) and `Retrieval-hate`. Both study the same four corpora —
HateMM, MultiHateClip EN, MultiHateClip ZH and HateClipSeg — but they measure
different things on different grids, so their numbers are reported in two
separate sections and are never pooled into one ranking.

Three differences make a merged table meaningless. The frame grid differs: this
repository scores on a 1 fps grid, `Retrieval-hate` on a 4 fps grid, with several
methods evaluated at a coarser native rate and upsampled. The metric set differs:
this repository reports pooled ROC-AUC, pooled PR-AUC, a within-hate macro (the
mean per-video ROC-AUC over hateful videos whose gold array contains both
classes) and a video-level AUC, whereas `Retrieval-hate` reports pooled frame
ROC-AUC and PR-AUC only. The cohorts differ: `Retrieval-hate` evaluates on its
own test split with its own gold arrays, and this repository uses a separately
frozen split with SHA-pinned gold arrays. A cell in Section A and the cell
directly above it in Section B are therefore two measurements of related
quantities, not two measurements of one quantity.

One convention applies throughout. "Label-free" here means no hate annotation of
the target corpus is consumed at any stage. Methods that consume temporal
supervision from an external corpus at pre-training time are separated into their
own group and marked, because they are label-free with respect to hate but not
with respect to temporal grounding. Weakly supervised methods, which train on
video-level hate labels, fall outside the scope of both sections and are listed
by name at the end without numbers.

---

## Section A: 1 fps protocol (this repository)

Protocol: `docs/duplex/FRAME_EVAL_PROTOCOL.md`. All rows share a 1 fps frame
grid, SHA-pinned gold arrays, and one evaluator (`eval_baseline_scores.py` over
`scripts/duplex/frame_eval_common.py`). Pooled ROC-AUC and PR-AUC run over every
frame of every scored video. The within-hate macro is the mean per-video ROC-AUC
restricted to hateful videos whose gold array carries both classes, and `n`
counts those videos. Video AUC max-pools each video's frame scores and ranks them
against the video label.

Chance is 0.5 for the ROC columns. Chance for the PR column is the frame positive
rate: 0.2419 on HateMM, 0.2505 on MHC EN, 0.2327 on MHC ZH and 0.5255 on
HateClipSeg. The HateClipSeg base rate is the reason a PR value near 0.55 on that
corpus is near chance rather than well above it.

| Corpus | Method | Supervision | Pooled ROC-AUC | Pooled PR-AUC | Within-hate macro (n) | Video AUC |
|---|---|---|---:|---:|---:|---:|
| HateMM (214 videos, 29,266 frames) | Ours (masked locator, 1 forward) | zero labels | 0.7451 | 0.5601 | 0.5706 | 0.9010 |
| HateMM | Vad-R1, zero-shot arm | external VAD pre-training | 0.5696 | 0.2722 | 0.5000 (85) | 0.5288 † |
| HateMM | Vad-R1, term-adaptation arm | external VAD pre-training | 0.5750 | 0.2740 | 0.5000 (85) | 0.5451 † |
| HateMM | EventVAD reimplementation | training-free | 0.5174 | 0.2519 | 0.4988 (85) | 0.4519 |
| MHC EN (158 videos, 5,600 frames) | Ours | zero labels | 0.6198 | 0.4141 | 0.6154 (44) | 0.7015 |
| MHC EN | Vad-R1, zero-shot arm | external VAD pre-training | 0.5427 | 0.2699 | 0.5000 (44) | 0.5247 † |
| MHC EN | Vad-R1, term-adaptation arm | external VAD pre-training | 0.6053 | 0.3046 | 0.5005 (44) | 0.5778 † |
| MHC EN | EventVAD reimplementation | training-free | 0.5041 | 0.2568 | 0.4784 (44) | 0.5179 |
| MHC ZH (153 videos, 4,817 frames) | Ours | zero labels | 0.6004 | 0.3813 | 0.6076 (7) ‡ | 0.6153 |
| MHC ZH | Vad-R1, zero-shot arm | external VAD pre-training | 0.5987 | 0.2838 | 0.5000 (7) ‡ | 0.6427 † |
| MHC ZH | Vad-R1, term-adaptation arm | external VAD pre-training | 0.7262 | 0.3721 | 0.5000 (7) ‡ | 0.7272 † |
| MHC ZH | EventVAD reimplementation | training-free | 0.5202 | 0.2440 | 0.4923 (7) ‡ | 0.5623 |
| HateClipSeg (79 videos, 18,839 frames) | Ours | zero labels | 0.5414 | 0.6149 | 0.5324 (67) | 0.7536 |
| HateClipSeg | Vad-R1, zero-shot arm | external VAD pre-training | 0.6382 | 0.6115 | 0.5001 (67) | 0.6826 |
| HateClipSeg | Vad-R1, term-adaptation arm | external VAD pre-training | 0.5655 | 0.5610 | 0.5001 (67) | 0.6196 |
| HateClipSeg | EventVAD reimplementation | training-free | not run | not run | not run | not run |

† Vad-R1 emits one binary verdict per video on the three MultiHateClip and HateMM
corpora, so the last column is the balanced accuracy `(TPR + TNR) / 2` rather than
a ranking statistic. Its frame scores are binary in every corpus, which leaves the
ROC curve a single interior operating point and makes both pooled AUCs coarse by
construction.

‡ The MHC ZH within-hate macro rests on 7 videos. That column carries no weight on
this corpus and no claim should lean on it.

Source files. Every Section A cell is written by the shared evaluator into
`results/reproduction/{baselines/<method>,ours}/<corpus>/frame_eval.json`, and the
tables are transcribed from `docs/duplex/BASELINE_RESULTS.md`: the consolidated
HateMM, MHC EN and MHC ZH tables at lines 879-918, the EventVAD table at lines
971-975, the Vad-R1 arms at lines 661-665 and 774-781, and the HateClipSeg table
at lines 1131-1152. The HateMM row for our method is additionally documented in
`docs/duplex/FRAME_LEVEL_EVAL_NOTE.md`; note that this note records 212 videos and
28,751 frames for that run, against the 214 videos and 29,266 frames given in the
consolidated header, a discrepancy carried over unresolved and flagged below.

Cost, from `docs/duplex/BASELINE_RESULTS.md` lines 943-953. Our locator uses one
packed forward per video at 0.11 s with a shared prefix. Vad-R1 uses one call per
video over 16 frames at 3.7 s. EventVAD calls the MLLM once per detected event,
which came to 6.3 to 20.6 calls per video with a worst case of 185, totalling
6,436 calls and 13.4 GPU-hours over the three corpora it ran on; 40 % of its
events return no parseable score under the paper's own prompt.

EventVAD was not run on HateClipSeg by owner decision, recorded at lines 1154-1157
of the same file: it read the floor on all three prior corpora at that cost, and a
fourth floor row was judged not worth another four-plus GPU-hours.

LAVAD has an adapter and evaluation plumbing in this repository (commit `14d5fa1`,
`scripts/reproduction_baselines/lavad/`), but no LAVAD inference has been run
here, so it holds no row in Section A. Its numbers appear in Section B only.

---

## Section B: 4 fps protocol (`Retrieval-hate`, read-only)

Protocol: common test split, 4 fps evaluation grid, frame ROC-AUC and frame
PR-AUC. Every method's scores are upsampled to that grid from its own native
rate, which is recorded per method below. Main pre-registered variants only; no
post-hoc selection among variants. T3AL is the mean over three seeds.

Test-split frame positive rates, which are the PR chance levels: 0.2422 on
HateMM, 0.2734 on MHC-EN, 0.2649 on MHC-ZH and 0.4730 on HateClipSeg.

Cells read `ROC / PR`. The final column is the unweighted mean over the four
corpora.

### Group 1: strictly label-free (zero-shot, training-free, or test-time adaptation on unlabelled video)

| Method | Source | Native rate | HateMM | MHC-EN | MHC-ZH | HateClipSeg | Mean ROC / PR |
|---|---|---|---:|---:|---:|---:|---:|
| ZS-CLIP | baseline row defined in LAVAD, CVPR 2024 | 4 fps | .537/.278 | .501/.268 | .608/.341 | .499/.461 | .536/.337 |
| ZS-ImageBind (image) | LAVAD's ImageBind baseline, CVPR 2024 | 4 fps | .592/.314 | .594/.329 | .598/.358 | .593/.554 | .594/.389 |
| ZS-ImageBind (video) | as above | 0.5 fps | .591/.310 | .564/.306 | .573/.355 | .581/.543 | .577/.378 |
| ZS-ImageBind (audio) | extension beyond LAVAD's visual baseline | 0.5 fps | .565/.291 | .616/.368 | .653/.396 | .565/.512 | .600/.392 |
| Qwen2.5-VL-7B native grounding | Qwen2.5-VL, own harness | interval output | .519/.252 | .522/.281 | .511/.270 | .503/.475 | .514/.319 |
| LAVAD | Zanella et al., CVPR 2024 | 1 fps | .559/.291 | .556/.311 | .492/.263 | .577/.546 | .546/.353 |
| URF-HVAA | NeurIPS 2025 | 0.1 fps | .574/.318 | .549/.297 | .545/.287 | .586/.553 | .564/.364 |
| AV²A | CVPR 2025 | 1 fps / 0.1 fps | .539/.252 | .531/.322 | .560/.321 | .486/.468 | .529/.341 |
| T3AL | Liberatori et al., CVPR 2024 | per-video test-time adaptation | .627/.305 | .568/.314 | .720/.440 | .636/.567 | **.638/.406** |

### Group 2: trained on unlabelled or normal-only target data

| Method | Source | Native rate | HateMM | MHC-EN | MHC-ZH | HateClipSeg | Mean ROC / PR |
|---|---|---|---:|---:|---:|---:|---:|
| MULDE (clipL336, one-class) | Micorek et al., CVPR 2024 | 4 fps | .600/.309 | .487/.259 | .513/.250 | .533/.500 | .533/.329 |
| CLAP (unlabelled pool) | Emad et al., CVPR 2024 | 2 fps | .586/.352 | .494/.279 | .328/.189 | .471/.453 | .470/.318 |

MULDE consumes the target dataset's normal-only training partition, which is a
label in the weak sense that normality is asserted; CLAP consumes an unlabelled
target pool with no labels of any kind. Neither consumes a hate annotation.

### Group 3: auxiliary temporal pre-training (no hate labels, external temporal supervision)

| Method | Source | Native rate | HateMM | MHC-EN | MHC-ZH | HateClipSeg | Mean ROC / PR |
|---|---|---|---:|---:|---:|---:|---:|
| LaGoVAD | ICLR 2026 | 0.5 fps | .558/.305 | .524/.262 | .597/.312 | .500/.467 | .545/.336 |
| UniTime | NeurIPS 2025 | interval output | .478/.235 | .495/.272 | .488/.260 | .453/.455 | .479/.305 |
| SeViLA Localizer | Yu et al., NeurIPS 2023 | 1 fps | .627/.332 | **.629/.330** | .670/.386 | .576/.561 | .626/.402 |

These three carry checkpoints trained with temporal supervision on external
corpora. They use no hate annotation, so they belong in this document, but they
are not comparable to Group 1 on the question of how much a system can localize
without any temporal supervision at all.

### Control rows

| Control | HateMM | MHC-EN | MHC-ZH | HateClipSeg | Mean ROC / PR |
|---|---:|---:|---:|---:|---:|
| GOLD_BROADCAST (video gold label broadcast to every frame) | .886/.583 | .943/.766 | .984/.919 | .626/.544 | .860/.703 |
| RANDOM_UNIFORM (uniform random scores) | .500/.242 | .500/.274 | .499/.265 | .501/.472 | .500/.313 |

GOLD_BROADCAST is not a method. It is the score a system would earn by knowing
each video's label perfectly and pointing nowhere inside the video, and it is the
ceiling that any purely video-level signal can reach on a frame metric. Its
HateClipSeg value (.626) is far below its value on the other three corpora
because 87 % of that corpus's videos are positive, which leaves little
video-level signal to broadcast.

Source files. Group tables are transcribed from
`/home/jehc223/Retrieval-hate/idea-stage/repro_campaign/BASELINE_PERFORMANCE_ARCHIVE_2026-08-22.md`
(headline table, lines 24-39), cross-checked against the machine-readable
`/home/jehc223/Retrieval-hate/idea-stage/repro_campaign/summary_test.csv`. T3AL's
row comes from `/home/jehc223/Retrieval-hate/idea-stage/repro_t3al/eval/test_agg.json`,
`main` variant, because the generic table builder does not ingest T3AL's
three-seed schema and the master CSV therefore records it as not run. Control
rows come from `/home/jehc223/Retrieval-hate/idea-stage/repro_campaign/gt_controls.json`,
`split_test_4fps` block, with RANDOM_UNIFORM read from `random_ROC_AUC_mean` and
`random_AP_mean`. Per-method provenance and native rates come from
`/home/jehc223/Retrieval-hate/idea-stage/REPRO_CAMPAIGN_RESULTS.md`, its
`sections/` directory and `idea-stage/repro_campaign/MODEL_ASSETS_STATUS.md`.

LAVAD in Section B was evaluated on the test split only. Its full-corpus run was
not performed, and the reason is recorded at §K.2 of
`REPRO_CAMPAIGN_RESULTS.md`. Combined with its absence from Section A, LAVAD has
exactly one measurement in this document: the 4 fps test-split row above.

---

## Section C: what the two tables say

1. T3AL holds the best four-corpus mean among strictly label-free methods
   (.638 ROC, .406 PR) and SeViLA Localizer is second overall (.626/.402), though
   SeViLA reaches that number with external temporal supervision and therefore
   answers a weaker question than T3AL does.

2. UniTime falls below random ROC on all four corpora and CLAP falls below on
   three, so transplanting a temporally pre-trained localizer or an unlabelled
   anomaly head into this domain can produce a score that is worse than guessing,
   not merely uninformative. CLAP's MHC-ZH cell (.328) is the single worst
   measurement in either section.

3. Vad-R1 never emits a sub-interval. Across all four corpora, 604 videos per
   arm and both arms, every positive prediction spans the whole clip, which pins its
   within-hate macro at the tie value (0.5000 to 0.5005). Its frame row is a
   video-level verdict broadcast across the timeline, and substituting hate
   vocabulary for anomaly vocabulary moves the verdict threshold without
   recovering a single interval.

4. Among purely feed-forward zero-shot scorers, ImageBind-audio is the strongest
   cell on both MultiHateClip corpora (.616 EN, .653 ZH), which is consistent with
   hate in this material being carried largely by speech. T3AL beats it on MHC-ZH
   (.720), but T3AL adapts on each unlabelled test video rather than scoring it in
   one pass.

5. Every method sits far below GOLD_BROADCAST. The best Section B mean (.638) is
   .222 ROC below the broadcast control (.860), and on MHC-ZH the gap is .264
   (.720 against .984). Since GOLD_BROADCAST points nowhere inside a video, that
   gap measures how much of the frame metric these methods fail to capture even
   at the video level, before localization is considered at all.

6. Under the 1 fps protocol our masked locator leads the pooled column on HateMM
   (.7451, against .5696 for the next row that uses no hate label), leads the within-hate macro
   on both MultiHateClip corpora (.6154 EN, .6076 ZH) and leads the video column
   on HateMM (.9010) and HateClipSeg (.7536). On HateClipSeg, the one corpus whose
   annotation supports a well-powered within-video measurement over 67 videos, it
   reads .5324, inside the same narrow band every other method occupies. The
   localization claim is unsupported there for every system measured.

---

## Out of scope: weakly supervised methods

The following are evaluated in `docs/duplex/BASELINE_RESULTS.md` under the same
1 fps protocol but train on video-level hate labels, so they lie outside the
label-free scope of this document and their numbers are deliberately omitted
here:

- VadCLIP (AAAI 2024)
- DSANet (AAAI 2026)
- MACIL-SD, in its audio-visual, audio-only and visual-only configurations
- MultiHateLoc reimplementation, in its fused, DMS, visual, audio, text and union branches
- Audio-only MIL over VGGish features

Refer to `docs/duplex/BASELINE_RESULTS.md` for those rows. They belong in a
supervision-axis comparison, not in a label-free one.
