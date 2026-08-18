# Reproduction baselines: measured results

VadCLIP (AAAI 2024) and DSANet (AAAI 2026), ported to this study's 1 fps grid by
`scripts/reproduction_baselines/` (port commit `177fd5c`, patch list in
`scripts/reproduction_baselines/PATCHES.md`), trained and scored on all three
corpora. Six train / score / evaluate cycles ran strictly one after another on a
single RTX 5090 through `scripts/reproduction_baselines/run_all.sh`, started
2026-08-19 04:16 NZST and finished 04:18, about two minutes of wall time in
total: the CLIP features are precomputed, so a training epoch is one or two
seconds.

Every number below comes from `results/reproduction/baselines/<method>/<corpus>/frame_eval.json`,
which is written by `eval_baseline_scores.py` over `scripts/duplex/frame_eval_common.py`,
the same evaluator any method in this study is scored by. The video-level column
is the one thing computed outside that file: it max-pools each video's frame
scores into a single number and ranks those against the corpus video label.

## Results

Pooled ROC-AUC and PR-AUC run over every frame of every scored video. The
within-hate macro is the mean per-video ROC-AUC restricted to hateful videos
whose gold array contains both classes, so it measures where inside a hateful
video the score peaks. `n` is how many videos that mean covers. Video AUC
max-pools the frame scores per video and ranks them against the video label.

| method | corpus | branch | pooled ROC-AUC | pooled PR-AUC | within-hate macro (n) | video AUC |
| --- | --- | --- | --- | --- | --- | --- |
| VadCLIP | hatemm | score_mlp | 0.6855 | 0.4457 | 0.4848 (85) | 0.7242 |
| VadCLIP | hatemm | score_align | 0.5685 | 0.3359 | 0.5037 (85) | 0.6473 |
| VadCLIP | mhclip_en | score_mlp | 0.6281 | 0.3611 | 0.3331 (44) | 0.6405 |
| VadCLIP | mhclip_en | score_align | 0.4791 | 0.2347 | 0.4621 (44) | 0.5306 |
| VadCLIP | mhclip_zh | score_mlp | 0.5676 | 0.2705 | 0.3562 (7) | 0.5537 |
| VadCLIP | mhclip_zh | score_align | 0.3880 | 0.1806 | 0.3225 (7) | 0.3981 |
| DSANet | hatemm | score_mlp | 0.7063 | 0.4824 | 0.5453 (85) | 0.7470 |
| DSANet | hatemm | score_refined | 0.7063 | 0.4824 | 0.5453 (85) | 0.7470 |
| DSANet | hatemm | score_align | 0.6828 | 0.4540 | 0.5689 (85) | 0.7259 |
| DSANet | mhclip_en | score_mlp | 0.6684 | 0.4354 | 0.3844 (44) | 0.6768 |
| DSANet | mhclip_en | score_refined | 0.6684 | 0.4354 | 0.3844 (44) | 0.6768 |
| DSANet | mhclip_en | score_align | 0.5602 | 0.3596 | 0.7230 (44) | 0.5860 |
| DSANet | mhclip_zh | score_mlp | 0.5749 | 0.2921 | 0.3557 (7) | 0.5588 |
| DSANet | mhclip_zh | score_refined | 0.5749 | 0.2921 | 0.3557 (7) | 0.5588 |
| DSANet | mhclip_zh | score_align | 0.5904 | 0.3082 | 0.5279 (7) | 0.5792 |

Chance is 0.5 for every ROC column. For PR-AUC chance is the frame positive
rate, which is 0.2419 on hatemm, 0.2505 on mhclip_en and 0.2327 on mhclip_zh.
The scored cohort is the gold cohort: 214 videos and 29266 frames on hatemm,
158 and 5600 on mhclip_en, 153 and 4817 on mhclip_zh.

`score_mlp` is the MIL branch, the reading both papers headline. `score_align`
is the text-alignment branch. DSANet's `score_refined` equals `score_mlp` to
eight decimal places, which is the expected consequence of the binary collapse
already documented in the port README: with one non-normal class the
hierarchical refinement redistributes the MLP score over a single column and
returns it unchanged. The two branches are kept separate in the table only so
that the equality is on record.

Reading the three columns together: on HateMM both baselines separate hateful
from non-hateful videos reasonably well (video AUC 0.72 to 0.75) and carry some
of that into the frame grid (pooled ROC 0.69 to 0.71), but neither localises
inside a hateful video, since the within-hate macro sits at or below chance for
every branch except DSANet's alignment branch. On MultiHateClip the pooled
number degrades and the within-hate macro falls well below chance for the MIL
branch of both methods, which means the MIL score systematically peaks on the
non-hateful seconds of hateful videos. VadCLIP's alignment branch on mhclip_zh
is the worst cell in the study at 0.3880 pooled, below chance by a clear margin.

## Run settings

Every hyperparameter is the published XD-Violence default except the three
listed in PATCHES.md patch O1 (`classes-num` 2, and the per-corpus
`visual-length` / `attn-window`). `run_all.sh` passes nothing, so the option
modules' defaults are what ran: seed 234, lr 1e-5, batch size 96, 10 epochs,
`visual-length` 256 with `attn-window` 64 on hatemm and 64 with 16 on
MultiHateClip. Model selection is on a seeded, label-stratified 10 % validation
carve-out of the train split, by video-level average precision; the test split
is never opened during training (patch V3).

| corpus | train / val videos | hateful in train | optimizer steps at batch 96 |
| --- | --- | --- | --- |
| hatemm | 766 / 85 | 307 | 8 per epoch, 80 total |
| mhclip_en | 567 / 63 | 174 | 6 per epoch, 60 total |
| mhclip_zh | 591 / 66 | 187 | 7 per epoch, 70 total |

## Loss curves and the contingency

The porting agent pre-declared one contingency before any run: if a corpus came
back near chance **with a flat training loss**, that cell was to be rerun at
`--batch-size 16 --max-epoch 50`, on the argument that 80 steps at lr 1e-5 might
be too few for the model to move at all.

**The contingency did not fire on any cell, and no rerun was performed.** The
loss moved in all six. Below is the MIL classification loss (`loss1` for
VadCLIP, `l1` for DSANet) at the first and last epoch, with the epoch model
selection kept.

| cell | first epoch | last epoch | selected epoch | val video AP at selection |
| --- | --- | --- | --- | --- |
| VadCLIP / hatemm | 0.8666 | 0.5060 | 9 | 0.7932 |
| VadCLIP / mhclip_en | 0.9140 | 0.5633 | 8 | 0.4394 |
| VadCLIP / mhclip_zh | 0.7124 | 0.5351 | 2 | 0.6446 |
| DSANet / hatemm | 0.8538 | 0.4967 | 7 | 0.7988 |
| DSANet / mhclip_en | 0.9058 | 0.5351 | 10 | 0.4725 |
| DSANet / mhclip_zh | 0.7171 | 0.5368 | 3 | 0.6615 |

Beyond the drop in loss, validation AP itself plateaus well inside the 10 epoch
budget in five of the six cells: VadCLIP on hatemm reaches 0.7891 by epoch 5 and
ends at 0.7932, and VadCLIP on mhclip_zh peaks at epoch 2 and never recovers
that value. A run that has already stopped improving on validation before the
budget ends is not a run starved of optimizer steps, so raising the step count
would have been tuning rather than the pre-declared fix. The only cell still
improving at the last epoch is DSANet on mhclip_en, whose validation AP rises
monotonically from 0.3471 to 0.4725 and selects epoch 10; the batch-16 setting
would plausibly move that cell, but its loss is not flat either, so the
pre-declared trigger is not met there and the published setting is what the
table reports.

## Sanity checks

Run for every branch of every cell, all passing.

Score-to-gold length: zero mismatches. `eval_baseline_scores.py` raises on a
length mismatch, and independently the length of each per-video score array was
compared against the gold array. Zero gold videos missing from the score file
and zero scored videos absent from the gold in all six cells.

Non-constant scores: no video in any cell has a constant score array. Globally,
the MIL branch spans roughly 0.003 to 0.99 on hatemm with a standard deviation
of 0.29; the alignment branch is the narrower of the two, spanning 0.333 to
0.468 on VadCLIP / hatemm with a standard deviation of 0.021. The alignment
branch is narrow but not degenerate, and it is a rank metric that reads it.

Finiteness: the evaluator raises on non-finite scores and did not.

## Two things to know before quoting these numbers

**The mhclip_zh within-hate macro rests on 7 videos.** Of the 43 hateful videos
in the mhclip_zh gold cohort, 36 are annotated hateful for their entire
duration, leaving no within-video ranking to score. The macro column for that
corpus therefore averages 7 videos, and its standard deviation is around 0.4.
It should not be read as a stable localisation measurement. The comparable
counts are 85 of 85 on hatemm and 44 of 46 on mhclip_en, so those macros are
sound.

**Nothing in this table is test-selected.** Both upstream training scripts pick
their checkpoint by test AP; this port does not, which makes these numbers lower
than the corresponding upstream protocol would produce and comparable with a
method that also never opens the test split. `--val-frac 0 --select last`
restores the upstream behaviour if a strictly-as-published number is ever
wanted.

# MultiHateLoc reimplementation (our protocol, stated assumptions)

MultiHateLoc (WWW 2026, arXiv 2512.10408) is the closest published competitor
to this study. **These numbers do not come from the authors' code.** The
repository the paper announces, `github.com/mmilabuk/multihateloc`, holds a
LICENSE file and nothing else, so the rows below are a from-scratch
reimplementation from the paper text, run under this study's frozen protocol.
Every architectural detail the paper leaves unstated was filled with the
simplest reading that makes the described object run, and each such choice is
enumerated in `scripts/reproduction_baselines/multihateloc/DESIGN.md` and
marked `INFERRED` at the line of code that makes it.

Three things must be said before any number is quoted.

**The paper does not state its frame rate.** T is only "the number of frames".
Its evaluation grid and its span-to-frame rule are likewise unstated. We freeze
1 fps, this study's gold grid, and say so; on any other grid T changes, the MIL
pool size `ceil(T/3)` changes, and a frame-level AUC means something else.

**The published 0.645 frame mAP / 0.799 AUC on HateMM is not a target these
rows can hit or miss.** Grid, gold rasterization, splits, cohort, model
selection and metric all differ, and four of those six differ because the
paper does not specify them. The rows belong beside the other retrained
baselines under this study's protocol, not beside the paper's own table.

**The reimplementation is honest about the parts it invented.** The largest
inference is where the Dynamic Modality Selection weights enter the network:
the paper uses them only in the final frame selection, where they would never
receive a gradient, so here they also scale each modality's contribution to
the fused branch. A different reading gives a different model.

## Results

Same evaluator as every row above (`eval_baseline_scores.py` over
`scripts/duplex/frame_eval_common.py`), same gold cohort, same video-AUC
convention (max-pool the frame scores, rank against the video label).

| method | corpus | branch | pooled ROC-AUC | pooled PR-AUC | within-hate macro (n) | video AUC |
| --- | --- | --- | --- | --- | --- | --- |
| MultiHateLoc-reimpl | hatemm | score_fused | 0.7504 | 0.4856 | 0.6008 (85) | 0.8622 |
| MultiHateLoc-reimpl | hatemm | score_dms | 0.7595 | 0.5165 | 0.6029 (85) | 0.8625 |
| MultiHateLoc-reimpl | hatemm | score_visual | 0.6434 | 0.4053 | 0.5495 (85) | 0.7126 |
| MultiHateLoc-reimpl | hatemm | score_audio | 0.7777 | 0.5115 | 0.6106 (85) | 0.8156 |
| MultiHateLoc-reimpl | hatemm | score_text | 0.6777 | 0.4137 | 0.5398 (85) | 0.8006 |
| MultiHateLoc-reimpl | hatemm | score_union | 0.5249 | 0.2517 | 0.5283 (85) | 0.5000 |
| MultiHateLoc-reimpl | mhclip_en | score_fused | 0.6740 | 0.3700 | 0.4611 (44) | 0.6498 |
| MultiHateLoc-reimpl | mhclip_en | score_dms | 0.6832 | 0.3890 | 0.4553 (44) | 0.6543 |
| MultiHateLoc-reimpl | mhclip_en | score_visual | 0.6378 | 0.4110 | 0.4902 (44) | 0.6557 |
| MultiHateLoc-reimpl | mhclip_en | score_audio | 0.6711 | 0.3528 | 0.5206 (44) | 0.7124 |
| MultiHateLoc-reimpl | mhclip_en | score_text | 0.6219 | 0.3050 | 0.4902 (44) | 0.5312 |
| MultiHateLoc-reimpl | mhclip_en | score_union | 0.4916 | 0.2474 | 0.4661 (44) | 0.5000 |
| MultiHateLoc-reimpl | mhclip_zh | score_fused | 0.6749 | 0.4032 | 0.4126 (7) | 0.7382 |
| MultiHateLoc-reimpl | mhclip_zh | score_dms | 0.7022 | 0.4085 | 0.4299 (7) | 0.7233 |
| MultiHateLoc-reimpl | mhclip_zh | score_visual | 0.7011 | 0.4004 | 0.3697 (7) | 0.7156 |
| MultiHateLoc-reimpl | mhclip_zh | score_audio | 0.6487 | 0.3314 | 0.5254 (7) | 0.6622 |
| MultiHateLoc-reimpl | mhclip_zh | score_text | 0.6659 | 0.4134 | 0.4404 (7) | 0.6444 |
| MultiHateLoc-reimpl | mhclip_zh | score_union | 0.4978 | 0.2319 | 0.3503 (7) | 0.5000 |

`score_fused` is the primary branch and the one to quote: the fused stream's
frame probability, which is what the paper headlines. `score_visual`,
`score_audio` and `score_text` are the three modality branches, each supervised
by the same MIL loss. `score_union` is the paper's literal output, the
importance-gated union of the four top-K frame sets. `score_dms` is **our**
continuous reading of the same importance weights, a convex combination of the
three modality probabilities; it is not in the paper and is reported because
the union rule throws away the ranking our evaluator reads.

## What the table says

**The reimplementation beats VadCLIP and DSANet on all three corpora** — the
two rows in the table above it as of this run; the MACIL-SD port is a separate
track and its rows are not yet here. Against the best of those two per corpus,
`score_fused` moves pooled
ROC-AUC from 0.7063 to 0.7504 on hatemm, 0.6684 to 0.6740 on mhclip_en and
0.5904 to 0.6749 on mhclip_zh, and video AUC from 0.7470 to 0.8622 on hatemm.
It is also the first row whose within-hate macro is above chance everywhere on
hatemm: 0.6008 for the fused branch against 0.5453 for DSANet's MIL branch and
0.4848 for VadCLIP's. That is the column that measures localization inside a
hateful video rather than separation between videos, and it is the column
every earlier baseline failed.

The comparison is not clean, and the reason is the input, not the method: this
row sees audio and text, and VadCLIP and DSANet see only CLIP frames. The
modality columns make the point directly. On hatemm the **audio branch alone**
scores 0.7777 pooled ROC-AUC, above the fused branch, and the visual branch
alone scores 0.6434, below every CLIP-based row in the table. On mhclip_en the
audio branch again has the best video AUC of any branch, 0.7124. A three-modal
model beating two visual-only models is mostly a statement about VGGish and
Whisper, not about MultiHateLoc's fusion.

**The union rule is degenerate under a ranking metric, and the video AUC of
exactly 0.5000 in all three corpora is not a coincidence.** The union always
contains at least the fused branch's top third of frames, so every video has at
least one frame set to 1, so max-pooling gives every video the identical score
and the ranking is one giant tie. Its pooled frame ROC-AUC, 0.49 to 0.52, is
the same fact at frame level: a 0/1 array carries one operating point and no
ranking. This is a property of the paper's stated output, not of our
implementation of it, and it is the reason `score_dms` exists.

**Fusion buys almost nothing over the best single modality.** `score_dms`, the
weighted modality mix, edges out `score_fused` on pooled ROC-AUC in all three
corpora (0.7595 / 0.6832 / 0.7022 against 0.7504 / 0.6740 / 0.6749), and the
audio branch alone beats both on hatemm. The learned importance weights are
close to uniform and never collapse onto one modality: averaged over the test
videos they are 0.414 / 0.188 / 0.398 (visual / audio / text) on hatemm,
0.309 / 0.263 / 0.428 on mhclip_en and 0.526 / 0.157 / 0.318 on mhclip_zh. On
hatemm the block puts its lowest weight on audio, which is the modality whose
branch scores best — so on this data the Dynamic Modality Selection block is
not selecting well.

## Run settings

Published settings, used verbatim: Adam, lr 1e-4, batch size 32, 100 epochs,
K = 3 (the top third of frames), smoothness lambda 0.1, contrastive lambda 0.2.
Inferred settings: hidden 256, embedding 128, dropout 0.1, contrastive
temperature 0.07, 0.67 M parameters. Protocol: seed 234, a seeded
label-stratified 10 % validation carve out of the train split, selection on
validation video-level AP, test split never opened during training (the same
rule as patch V3 for the other ports).

Features. Visual is ImageNet ViT-B/16 (`google/vit-base-patch16-224`, CLS
token, 768-d), which is the encoder the paper cites, not CLIP. Audio is VGGish
128-d. Text is `bert-base-uncased` (`bert-base-chinese` for mhclip_zh) over the
frozen whisper-large-v3 fragments, repeat-padded across each fragment's
interval onto the frame grid, zero where no fragment covers the second.
Extraction of all 2672 text matrices took 13 s on the 5090, zero failures; mean
frame coverage is 0.810 on hatemm, 0.854 on mhclip_en, 0.860 on mhclip_zh, and
59 / 40 / 32 videos have no ASR fragment at all and are therefore all-zero in
the text modality. The paper's linear interpolation of VGGish up to T is the
identity on our grid — VGGish is already one row per second — so it is not
performed.

| corpus | train / val videos | hateful in train | steps per epoch | selected epoch | val video AP | wall time |
| --- | --- | --- | --- | --- | --- | --- |
| hatemm | 766 / 85 | 307 | 24 | 12 | 0.8465 | 150 s |
| mhclip_en | 567 / 63 | 174 | 18 | 7 | 0.4700 | 26 s |
| mhclip_zh | 591 / 66 | 187 | 19 | 83 | 0.6786 | 26 s |

All three ran one after another on one RTX 5090 through
`scripts/reproduction_baselines/multihateloc/run_all.sh`, 2026-08-19 05:08 to
05:12 NZST.

## Loss evidence

Every loss term moved, in every corpus. The MIL term is the sum of the four
branch BCEs, so its epoch-1 value near 2.75 is four branches at chance.

| corpus | MIL e1 -> e100 | smoothness e1 -> e100 | contrastive e1 -> e100 |
| --- | --- | --- | --- |
| hatemm | 2.7440 -> 0.4410 | 0.0002 -> 0.0381 | 3.4896 -> 1.8817 |
| mhclip_en | 2.7176 -> 0.4830 | 0.0003 -> 0.0317 | 3.5137 -> 1.8280 |
| mhclip_zh | 2.7232 -> 0.5663 | 0.0003 -> 0.0367 | 3.4917 -> 2.0266 |

Per-branch MIL at epoch 100 (visual / audio / text / fused): 0.045 / 0.266 /
0.126 / 0.004 on hatemm, 0.016 / 0.340 / 0.126 / 0.001 on mhclip_en, 0.016 /
0.345 / 0.205 / 0.001 on mhclip_zh. The fused branch fits the training video
labels almost perfectly in all three; the audio branch is the one that never
does, which is the reverse of the test-set ordering on hatemm and is the
cleanest single sign that the fused branch is overfitting.

Smoothness *rises* from near zero, which is the expected direction and not a
failure: at initialisation every frame probability is near 0.5 and the score is
already flat, so the term starts at its floor and grows as the model learns to
vary its score across a video. Weighted at lambda 0.1 it contributes under
0.004 to the total loss throughout, so it is regularising rather than driving.

Selection matters and is not cosmetic. On hatemm validation video AP peaks at
0.8465 on epoch 12 and falls to 0.7545 by epoch 100 while the training MIL loss
keeps dropping — the model overfits well inside the published budget. The
budget is kept at 100 epochs as published; selection decides which of those
epochs is scored. mhclip_zh is the opposite case, selecting epoch 83.

## Sanity checks

`smoke_cpu.py` in the port directory runs 22 checks, all passing: all three
feature matrices present at the stated dimensionality for every video of every
split (1066 + 792 + 814); the three modality lengths agree video by video, zero
mismatches; feature rows equal gold frames for all 214 + 158 + 153 gold videos;
a padded batch reproduces one-at-a-time scoring to 6e-8, so padding leaks
nowhere; the MIL pool size is exactly `ceil(T/3)` and its mean matches a manual
per-video sort; every loss term is finite, every parameter including the DMS
block receives a gradient, smoothness is zero on a constant score, the
contrastive term is lower when the modalities agree; the union set contains the
fused top-K and no padded frame.

On the score files: zero length mismatches against the gold arrays, zero gold
videos missing, all scores finite. The four extra scored videos per
MultiHateClip corpus and one on hatemm are test-split videos with no gold
array, the same cohort gap the rows above have.

One real degeneracy, and it is the paper's design rather than a bug. **The text
branch returns a constant score for a video whose transcript is a single
fragment**: repeat-padding one sentence vector across the whole video makes
every frame identical, so a frame-wise classifier must return one value. Within
the within-hate macro cohort this affects 4 of 85 videos on hatemm, 5 of 44 on
mhclip_en and 2 of 7 on mhclip_zh, and those videos drop out of the text
branch's macro. No fused-branch array is constant anywhere in the macro cohort.
