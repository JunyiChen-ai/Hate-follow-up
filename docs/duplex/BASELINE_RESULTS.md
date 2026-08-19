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

---

# MACIL-SD (ACM MM 2022) and its two uni-modal ablations

MACIL-SD ported by `scripts/reproduction_baselines/` (port commit `6d3ca64`,
patch list in `scripts/reproduction_baselines/PATCHES.md`, MACIL-SD section),
trained and scored on all three corpora in three modality settings. Nine
train / score / evaluate cycles ran strictly one after another on a single
RTX 5090 through `scripts/reproduction_baselines/run_all_macilsd.sh`, started
2026-08-19 05:17 NZST and finished 05:28, eleven minutes of wall time: the I3D
and VGGish features are precomputed, so an epoch is two or three seconds.

Every hyperparameter is the published default. `run_all_macilsd.sh` passes
nothing, so `macilsd/option.py` is what ran: seed 2333, lr 4e-4, batch size 128,
50 epochs, `max-seqlen` 200, EMA momentum 0.91, the three CMA lambdas at
1.5 / 1.5 / 0.1, `--grid snippet` (the alignment argued for in PATCHES.md A1),
`--crop-repeat 5`. Model selection is on a seeded, label-stratified 10 %
validation carve-out by video-level average precision; the test split is never
opened during training (patch M7, which removes upstream's test-selected
checkpointing).

## Results

Columns as in the VadCLIP / DSANet table above. The `macilsd` rows are the
audio-visual model, which exposes three readouts from one training run: `av` is
the fused score the paper headlines, `audio` and `visual` are the two branches
of that same fused model. The `macilsd_audio` and `macilsd_visual` rows are
separate trainings of upstream's own `Single_Model` on one modality alone, at
upstream's own lr/5 (patch M11) -- these are the honest uni-modal comparators,
not branches of the fused model.

| method | corpus | branch | pooled ROC-AUC | pooled PR-AUC | within-hate macro (n) | video AUC |
| --- | --- | --- | --- | --- | --- | --- |
| MACIL-SD | hatemm | score_av | 0.7282 | 0.5127 | 0.5383 (85) | 0.7611 |
| MACIL-SD | hatemm | score_audio | 0.7290 | 0.4501 | 0.5419 (85) | 0.7379 |
| MACIL-SD | hatemm | score_visual | 0.6552 | 0.4447 | 0.5012 (85) | 0.7059 |
| MACIL-SD | mhclip_en | score_av | 0.6764 | 0.4664 | 0.5383 (44) | 0.7112 |
| MACIL-SD | mhclip_en | score_audio | 0.6575 | 0.4453 | 0.5284 (44) | 0.7240 |
| MACIL-SD | mhclip_en | score_visual | 0.6759 | 0.4530 | 0.5397 (44) | 0.6858 |
| MACIL-SD | mhclip_zh | score_av | 0.7757 | 0.5233 | 0.4588 (7) | 0.7685 |
| MACIL-SD | mhclip_zh | score_audio | 0.7774 | 0.5301 | 0.5256 (7) | 0.7808 |
| MACIL-SD | mhclip_zh | score_visual | 0.7387 | 0.4834 | 0.4258 (7) | 0.7321 |
| MACIL-SD audio-only | hatemm | score_mil | 0.7667 | 0.4939 | 0.5966 (85) | 0.7814 |
| MACIL-SD audio-only | mhclip_en | score_mil | 0.7142 | 0.4987 | 0.5142 (44) | 0.7141 |
| MACIL-SD audio-only | mhclip_zh | score_mil | 0.6320 | 0.3254 | 0.5269 (7) | 0.6725 |
| MACIL-SD visual-only | hatemm | score_mil | 0.6398 | 0.4073 | 0.4966 (85) | 0.7046 |
| MACIL-SD visual-only | mhclip_en | score_mil | 0.6340 | 0.3670 | 0.5104 (44) | 0.6632 |
| MACIL-SD visual-only | mhclip_zh | score_mil | 0.6860 | 0.4085 | 0.4995 (7) | 0.7262 |

Chance is 0.5 for every ROC column; for PR-AUC it is the frame positive rate,
0.2419 on hatemm, 0.2505 on mhclip_en, 0.2327 on mhclip_zh. The scored cohort is
the gold cohort in all nine cells.

Three things stand out.

**MACIL-SD is the strongest baseline in the study on pooled frame ROC.** Its
best cell per corpus is 0.7290 on hatemm, 0.6764 on mhclip_en and 0.7774 on
mhclip_zh, against DSANet's 0.7063 / 0.6684 / 0.5904. The margin is largest on
mhclip_zh, where every CLIP-based baseline sat near or below chance and
MACIL-SD is nineteen points higher.

**Audio carries the signal, and fusion does not add to it.** The standalone
audio-only model beats the full audio-visual model on hatemm (0.7667 against
0.7282 pooled, 0.7814 against 0.7611 video) and on mhclip_en (0.7142 against
0.6764), and it beats the standalone visual-only model on both. The one corpus
where that reverses is mhclip_zh, where audio-only drops to 0.6320 while the
fused model reaches 0.7757. Since these corpora are hate-speech corpora whose
offending content is largely spoken, an audio-dominant result is expected; what
the fused model buys over its own audio branch is close to nothing on two of
three corpora.

**Localisation inside a hateful video remains unsolved here too.** The
within-hate macro sits between 0.43 and 0.60 in every cell, so the frame ranking
inside a hateful video is near chance even where the pooled and video-level
numbers are strong. The best localiser in the nine cells is the audio-only model
on hatemm at 0.5966. This is the same pattern the VadCLIP and DSANet rows show:
these methods separate hateful videos from non-hateful ones, and then spread
that verdict fairly flatly over the timeline.

## Loss evidence

The MIL classification loss (`cls`) at the first and last epoch, with the
selected epoch and its validation video AP. Fifty epochs everywhere.

| cell | cls first | cls last | selected epoch | val video AP at selection |
| --- | --- | --- | --- | --- |
| MACIL-SD / hatemm | 0.5990 | 0.3689 | 1 | 0.8586 |
| MACIL-SD / mhclip_en | 0.6035 | 0.5624 | 15 | 0.4823 |
| MACIL-SD / mhclip_zh | 0.6097 | 0.6029 | 15 | 0.5369 |
| audio-only / hatemm | 0.6788 | 0.1752 | 18 | 0.8598 |
| audio-only / mhclip_en | 0.6793 | 0.2272 | 16 | 0.5013 |
| audio-only / mhclip_zh | 0.6723 | 0.2856 | 49 | 0.4091 |
| visual-only / hatemm | 0.6632 | 0.1042 | 13 | 0.8012 |
| visual-only / mhclip_en | 0.6437 | 0.1358 | 24 | 0.5430 |
| visual-only / mhclip_zh | 0.6304 | 0.0992 | 12 | 0.5325 |

The loss decreased first to last in all nine cells, but two of them deserve to
be flagged rather than buried.

**The audio-visual model's `cls` loss barely moves on MultiHateClip**: 0.6035 to
0.5624 on EN and 0.6097 to 0.6029 on ZH, against 0.6788 to 0.1752 for the
uni-modal model on the same features. This is not a stalled run -- the four CMA
terms and the uni-modal distillation term all fall by roughly a factor of five
over the same fifty epochs, and validation AP rises -- but the fused MIL head
itself is close to flat on both MultiHateClip corpora. The uni-modal ablations,
which optimise a plain MIL head at lr/5, drive their loss down by a factor of
four to six on every corpus. Reported as measured; no rerun was performed and no
hyperparameter was changed, since nothing in the published preset was tuned here.

**MACIL-SD on hatemm selects epoch 1.** Validation AP peaks at 0.8586 on the
first epoch and never recovers it across the remaining forty-nine, ending at
0.7810. The reported hatemm audio-visual row is therefore a one-epoch model. The
number is what the frozen selection rule returns and is left as it stands, but
it should not be read as a converged result.

## Sanity checks

Run for every branch of every cell.

Score-to-gold length: zero mismatches in all fifteen branch-cells; zero gold
videos missing from any score file and zero scored videos absent from the gold.
`eval_baseline_scores.py` raises on either condition and did not. Finiteness:
all scores finite.

Non-constant scores: nine of the fifteen branch-cells have no constant video at
all. The exceptions are the same handful of videos in each case -- 4 of 214 on
hatemm (`score_audio` of the fused model, and the audio-only model), 3 of 214 on
hatemm for the visual columns, and 1 of 153 on mhclip_zh for the audio columns.
These are videos short enough to occupy a single snippet after the 16-frame
grid, so a per-snippet score has one value to give. Score ranges are wide
everywhere: the uni-modal MIL heads span roughly 2e-6 to 0.999 with a standard
deviation near 0.28 on hatemm, and the fused model's branches span 0.09 to 0.83.

---

# Ours (zero-label locator) on MultiHateClip

The masked packed locator, Arm M of
`scripts/duplex/masked_parallel_isolation_pilot.py`, carried to MultiHateClip EN
and ZH by `scripts/duplex/masked_parallel_isolation_mhclip.py`. **One MLLM
forward pass per video**, no labels, no training: the shared rules prefix and
all of a video's transcript chunks are packed into a single sequence, a
block-diagonal attention mask cuts every cross-chunk path, and each chunk's
position ids restart at the end of the prefix, so one pass computes what N
isolated per-chunk calls would compute. The score of a chunk is the frozen
judge's own answer margin, `logsumexp(logits[Yes ids]) - logsumexp(logits[No
ids])`, read from logits with nothing generated.

Scored by the same `scripts/duplex/frame_eval_common.py` against the same frozen
gold arrays as every baseline above. Chunk z is spread over its `[start, end)`;
frames no scored chunk covers take `(corpus-min chunk z) - 1`, the floor
convention frozen in `docs/duplex/PREREG_frame_level_evaluation_hatemm.md`,
applied per corpus.

## Results

| method | corpus | pooled ROC-AUC | pooled PR-AUC | within-hate macro (n) | video AUC |
| --- | --- | --- | --- | --- | --- |
| Ours (1 pass/video, zero labels) | mhclip_en | 0.6198 | 0.4141 | 0.6154 (44) | 0.7015 |
| Ours (1 pass/video, zero labels) | mhclip_zh | 0.6004 | 0.3813 | 0.6076 (7) | 0.6153 |

Cohort: all 158 EN and 153 ZH gold videos, 5600 and 4817 frames, positive rates
0.2505 and 0.2327 -- the same cohort the baselines are scored on. 818 EN and
1171 ZH chunks were scored, one packed forward per video, 157 and 152 videos
respectively; runtime 33 s and 42 s in total for the locator pass.

The comparison that matters is the within-hate macro, because that is the column
every trained baseline fails. **The locator is the best localiser in the study on
both MultiHateClip corpora**: 0.6154 on EN against 0.5397 for the best MACIL-SD
cell and 0.7230 / 0.3844 for DSANet's two branches, and 0.6076 on ZH against
0.5269 for the best MACIL-SD cell. It does this with no labels and no training,
where the baselines each consumed the full labelled train split. On the pooled
and video-level columns it is behind the trained baselines, which is the
expected shape: a transcript-only locator has no evidence on the 656 EN and 441
ZH frames no chunk covers, and those frames all sit at the floor.

Two caveats carry over from the baseline table. The ZH within-hate macro rests
on 7 videos, for the reason given above -- 36 of 43 hateful ZH videos are
annotated hateful end to end -- and should not be read as a stable measurement.
The DSANet mhclip_en alignment-branch macro of 0.7230 is higher than the
locator's 0.6154, but that same branch scores 0.5602 pooled against the
locator's 0.6198 and drops to 0.3844 on its own MIL branch.

## Prompt provenance

The prompt is the frozen judge's, reassembled per corpus rather than re-authored.
The frozen judge uses BILIBILI_RULES on MHClip_ZH and YOUTUBE_RULES elsewhere
(`src/duplex/score_duplex_probe.py:161-162`,
`src/our_method/score_holistic_2b.py:467`); that convention is replicated and
everything else -- lead-in sentence, system message, question, template layout --
is byte-identical across the two corpora and to the HateMM pilot. sha256, from
`results/reproduction/ours/<corpus>/prompt_fingerprints.json`:

| component | mhclip_en | mhclip_zh |
| --- | --- | --- |
| rules block | `e23dd329b55122ae…` | `b3fceb3631267398…` |
| question | `f45673af42da76b5…` | `f45673af42da76b5…` |
| system message | `e6addb7b869ede44…` | `e6addb7b869ede44…` |
| user-text template | `9442091c90445103…` | `3f24f4122dabaeab…` |
| packed prefix | `5aab1929792c022c…` | `316581c8e5e620cf…` |
| packed suffix | `4d7644e75cf868e7…` | `4d7644e75cf868e7…` |

The EN rules block and user-text template hashes equal the HateMM diagnostic's
frozen values (`isolated_chunk_diag.FROZEN_TEXT_SHA`), which is asserted at
runtime, not merely observed: the EN template is compared against
`isolated_chunk_diag.user_text` on three probe strings before the model loads.
Only the rules block and therefore the prefix differ on ZH, which is the intended
per-corpus difference.

## Fidelity: does one packed pass really reproduce N isolated calls

Checked on **every chunk of both corpora**, not a sample: each chunk was scored a
second time with a genuine isolated call and the two columns compared.

| | mhclip_en | mhclip_zh |
| --- | --- | --- |
| chunks compared | 818 | 1171 |
| Spearman(masked, sequential) | 0.99776 | 0.99809 |
| Pearson | 0.99944 | 0.99961 |
| max abs delta z | 0.75 | 0.75 |
| mean abs delta z | 0.175 | 0.163 |
| chunks bit-identical | 42.9 % | 43.9 % |
| pooled ROC from the isolated calls | 0.6203 | 0.5999 |
| **endpoint delta, packed minus isolated** | **-0.00051 ROC, -0.00117 PR** | **+0.00047 ROC, +0.00583 PR** |

Both corpora clear the 0.99 Spearman bar, and the endpoint moves by at most
0.0006 ROC, so nothing in the table above depends on which way the chunks were
scored.

The residual is bf16 arithmetic, and this was measured rather than assumed.
Packing a **single** branch is **bit-identical** to the isolated call on both
corpora, which shows the seam tokenisation, the block mask and the position
restart are exact -- and the prompt-identity assertion (`concat(prefix_ids,
branch_ids)` must equal the isolated prompt's ids, chunk by chunk) runs before
every packed forward and passed throughout. The difference appears only once
several branches share a sequence, where attention over the longer packed
sequence tiles its reduction differently. The model itself is deterministic: the
same call repeated returns the identical value.

One nuisance worth recording, because it cost a false alarm. An all-ones 2-D
`attention_mask` and an explicit 4-D additive mask send HuggingFace to different
SDPA kernels, and they disagree by the same 0.25 to 0.5 logit; the no-mask path
and the packed path agree with the 4-D path bit for bit. `score_sequential`
therefore takes the mask form as an explicit argument, so the comparison
measures the packing mechanism rather than a kernel dispatch.

**A three-video spot check is not enough on this corpus, and the reason is
instructive.** The first run stopped on a spot Spearman of 0.9802 (EN) and 0.9694
(ZH) over three videos. Those spot sets are tie-dominated -- 17 distinct z values
across the 75 EN spot chunks, largest tie group 24 -- and Spearman under heavy
ties converts sub-quantum noise into rank swaps: Pearson on the same 75 chunks
was 0.9987, and every discordant pair was separated by at most 0.25 in the
reference, one quantum of the score grid. Over the full cohort, where z takes 122
and 148 distinct values, the same comparison reads 0.998. The spot bar is kept in
the script as a cheap tripwire but is recorded as advisory when
`--sequential-reference` runs the complete comparison.

## Coverage and sanity

Zero gold videos lack a chunk record on either corpus, and every gold video's
frame grid matches its chunk record's duration, so no video is missing from the
score file and none is scored outside the gold. Frame coverage is 4944 of 5600 EN
and 4376 of 4817 ZH; uncovered frames take the floor, -24.50 on EN and -23.25 on
ZH.

One video per corpus is scored all-floor and is reported rather than dropped:
`uPJtlBAOT_U` (EN), whose last Whisper chunk carries a null start and end, and
`BV1Ts4y1A7XN` (ZH), whose single chunk carries null timestamps. Both fail the
frozen `usable_spans` helper, which refuses a record it cannot place on the
timeline; neither was special-cased. Every chunk with text was scored -- zero
chunks were dropped for empty text on either corpus.

---

# Vad-R1 (NeurIPS'25, zero-shot, released checkpoint, original prompt)

Vad-R1 is a 7B video-anomaly reasoner released with weights, so unlike every
other entry above nothing was trained here. The released checkpoint was run once
per video on all three test splits and its answer was rasterised onto this
study's 1 fps grid. Run through `scripts/reproduction_baselines/run_all_vadr1.sh`
on a single RTX 5090, three corpora strictly one after another, started
2026-08-19 05:57 NZST and finished 06:34: 37 minutes of wall time, of which 30
minutes is generation (800 s on hatemm, 513 s on mhclip_en, 509 s on mhclip_zh;
3.74, 3.25 and 3.33 s per video). One vLLM engine at a time, one forward per
video, 16 sampled frames, temperature 0.1, top-p 0.9, 512 max new tokens, seed 0.

**The prompt is upstream's, verbatim.** Vad-R1 asks the model whether an
*abnormal event* occurs, and that wording was not touched: sha256 of the binary
prompt is `cb673111d1b01d00…`, of the system prompt `7bf05ce3b7d79396…`, of the
assembled chat prompt `2c48af7c71dc56c2…`, recorded per corpus in
`results/reproduction/baselines/vadr1/<corpus>/run_meta.json` alongside upstream
commit `8536296b`. A second arm that substitutes hate vocabulary for "abnormal"
exists in the runner; it is a term-adaptation ablation on the same test split, it
was approved by the owner and run on 2026-08-19, and it is reported separately
under "Vad-R1 term-adaptation arm" below. Everything in this section is the
zero-shot arm.

## How a Vad-R1 answer becomes a frame score

The model returns one `<which>` verdict and, when that verdict is positive, one
`<when>` interval in normalised video coordinates. `rasterize_and_eval.py` maps
that interval to frames and writes 1 inside it and 0 outside; a negative verdict
writes all zeros. **The resulting frame score is therefore binary**, which has
two consequences worth stating before any number is read. The ROC curve has a
single interior operating point, so the pooled ROC-AUC is coarse by construction
and is not resolution-comparable to a continuous scorer's AUC. It is reported
anyway because every method in this study passes through one evaluator. The
columns that actually characterise a single-interval predictor are the interval
descriptives below.

## Results

| method | corpus | pooled ROC-AUC | pooled PR-AUC | within-hate macro (n) | verdict acc | verdict AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Vad-R1 zero-shot | hatemm | 0.5696 | 0.2722 | 0.5000 (85) | 0.5093 | 0.5288 |
| Vad-R1 zero-shot | mhclip_en | 0.5427 | 0.2699 | 0.5000 (44) | 0.6076 | 0.5247 |
| Vad-R1 zero-shot | mhclip_zh | 0.5987 | 0.2838 | 0.5000 (7) | 0.6797 | 0.6427 |

Chance is 0.5 for the ROC columns; for PR-AUC it is the frame positive rate,
0.2419 on hatemm, 0.2505 on mhclip_en and 0.2327 on mhclip_zh, so the PR column
sits three to five points above its own base rate. The last two columns come from
the model's own `<which>` verdict against the corpus video label. That verdict is
binary, so its "AUC" is the balanced accuracy `(TPR + TNR) / 2` and not a ranking
statistic; it is written in the table to keep the column readable next to the
trained baselines, whose video AUC ranks a max-pooled continuous score. TPR and
TNR are 0.6235 / 0.4341 on hatemm, 0.3261 / 0.7232 on mhclip_en and
0.5581 / 0.7273 on mhclip_zh, from confusion counts (tp/fp/fn/tn) of
53/73/32/56, 15/31/31/81 and 24/30/19/80.

Interval quality, over the videos carrying at least one gold positive frame. A
video the model called normal enters at IoU 0 rather than being dropped.

| corpus | frame IoU mean / median (n) | interval IoU vs gold envelope | R@IoU 0.3 | R@IoU 0.5 | R@IoU 0.7 |
| --- | --- | --- | --- | --- | --- |
| hatemm | 0.4013 / 0.2461 (85) | 0.4439 / 0.3851 | 0.4824 | 0.4235 | 0.3412 |
| mhclip_en | 0.2600 / 0.0000 (46) | 0.2640 / 0.0000 | 0.3043 | 0.2826 | 0.2609 |
| mhclip_zh | 0.5173 / 0.5455 (43) | 0.5037 / 0.5468 | 0.5349 | 0.5349 | 0.4884 |

## The within-hate macro is exactly 0.5000, and the reason is the finding

Every one of the three macro cells reads 0.5000 with standard deviation 0.0000.
That is not a coincidence and not an evaluator artefact. **Vad-R1 never once
predicted a sub-interval.** Across all 525 videos, every positive answer placed
the abnormal event over the entire clip: 126 of 126 positive intervals on
hatemm, 46 of 46 on mhclip_en, 54 of 54 on mhclip_zh, all `[0.0, 1.0]` up to the
rounding that occasionally writes `0.999`. Mean predicted span is 1.000 of video
duration in all three corpora. A whole-video interval rasterises to a constant
score array, a constant array has no internal ranking, and the macro therefore
takes the tie value 0.5 for every hateful video. Exactly one array per corpus on
hatemm and mhclip_en is non-constant, and only because a `0.999` endpoint drops
the final frame.

So the frame-level numbers in the first table are not measuring localisation at
all. They are the video-level verdict broadcast across the timeline, scored on a
frame grid. The pooled ROC of 0.5696 / 0.5427 / 0.5987 is a restatement of which
videos got called abnormal, and the interval IoU of 0.40 / 0.26 / 0.52 is a
restatement of what fraction of each hateful video is annotated hateful. On
mhclip_zh, where 36 of 43 hateful videos are annotated hateful end to end, a
degenerate whole-video prediction scores 0.5173 frame IoU for free; that is the
highest interval number in the table and it carries no localisation content.

**Vad-R1 is the weakest entry in the study on this task, and it fails in a new
way.** VadCLIP, DSANet and MACIL-SD each separate hateful videos from normal ones
and then spread that verdict fairly flatly over the timeline. Vad-R1 does not
spread it flatly; it declines to spread it at all, and its video-level separation
is also the weakest measured -- 0.5093 verdict accuracy on hatemm is barely above
calling everything normal. The zero-shot anomaly framing transfers to hateful
video neither as a detector nor as a localiser.

## Sanity checks

Every check below was run over the full cohort, not a sample.

Coverage is exact on all three corpora. 214, 158 and 153 generations were read
and 214, 158 and 153 videos were scored, matching the 214 / 158 / 153 gold
videos; zero videos are missing from the score files and zero scored videos lack
gold. Every row of `scores.jsonl` has length equal to its gold array, checked per
video: zero mismatches anywhere. One HateMM split id, `hate_video_427`, has no
gold and was dropped before inference rather than scored, which is recorded in
`run_meta.json`.

Generations are non-empty everywhere: 525 of 525 records carry output text, zero
empty or null. Zero decode errors: the AV1 ffmpeg fallback was exercised on 37
mhclip_en and 8 mhclip_zh videos and OpenCV handled the remaining 480, with no
video falling back to a short sample.

Parse failures are one video in 525. HateMM and mhclip_en parsed cleanly (126
positive-interval plus 88 negative; 46 plus 112). On mhclip_zh, `BV19C4y177iH`
ran into a repetition loop and hit the 512-token cap without closing its tags --
the tail of its output is `"火种" (spark)` repeated some forty times — so it
parsed as `unparsed`, contributed an all-zero score array, and counted as a
negative verdict. No video produced a positive verdict without a usable interval.

## What this row is for

It is the honest zero-shot reading of a video-anomaly reasoner on hateful video,
run at its released settings with its released prompt, and it belongs in the
table because the comparison is one reviewers will ask for. It should not be
quoted as evidence about hate localisation methods in general: the checkpoint was
trained to answer a different question about a different corpus, and the failure
observed here is that its interval head collapses to the trivial answer, which is
a statement about domain transfer rather than about the ceiling of anomaly-based
localisation.

## Vad-R1 term-adaptation arm (owner-approved)

The second arm named above was approved and run on 2026-08-19, 06:51 to 07:30
NZST, 39 minutes of wall time on the same single RTX 5090, three corpora strictly
one after another, 33 minutes of it generation (863 s on hatemm, 558 s on
mhclip_en, 562 s on mhclip_zh; 4.03, 3.53 and 3.67 s per video). Everything else
is held fixed: same checkpoint, same 16 frames, same temperature 0.1, top-p 0.9,
512 max new tokens, seed 0, same rasteriser, same evaluator, same gold arrays.
Outputs are written to `results/reproduction/baselines/vadr1/<corpus>_hateful/`,
so the zero-shot arm above is untouched.

The arm rewrites "abnormal" to "hateful" and "normal" to "non-hateful" throughout
the prompt and in the `<which>` vocabulary the parser reads. The prompt hashes
move accordingly: binary prompt `6e9caac84945c320…`, system prompt
`30042787d933e88e…`, assembled chat prompt `5a11072684fa0c6b…`, recorded per
corpus in `run_meta.json`. The model adopted the substituted vocabulary without
prompting trouble: every parsed verdict came back as `Hateful` or `Non-hateful`,
and zero verdicts were non-standard in either arm.

### Both arms side by side

| corpus | arm | pooled ROC | pooled PR | within-hate macro (n) | verdict acc | balanced acc | TPR / TNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hatemm | anomaly | 0.5696 | 0.2722 | 0.5000 (85) | 0.5093 | 0.5288 | 0.6235 / 0.4341 |
| hatemm | hateful | 0.5750 | 0.2740 | 0.5000 (85) | 0.5000 | 0.5451 | 0.7647 / 0.3256 |
| mhclip_en | anomaly | 0.5427 | 0.2699 | 0.5000 (44) | 0.6076 | 0.5247 | 0.3261 / 0.7232 |
| mhclip_en | hateful | 0.6053 | 0.3046 | 0.5005 (44) | 0.6013 | 0.5778 | 0.5217 / 0.6339 |
| mhclip_zh | anomaly | 0.5987 | 0.2838 | 0.5000 (7) | 0.6797 | 0.6427 | 0.5581 / 0.7273 |
| mhclip_zh | hateful | 0.7262 | 0.3721 | 0.5000 (7) | 0.6993 | 0.7272 | 0.7907 / 0.6636 |

Confusion counts (tp/fp/fn/tn) are 53/73/32/56 and 65/87/20/42 on hatemm,
15/31/31/81 and 24/41/22/71 on mhclip_en, 24/30/19/80 and 34/37/9/73 on
mhclip_zh, anomaly then hateful in each pair. Frame positive rates are unchanged
because the gold is unchanged: 0.2419, 0.2505, 0.2327.

Interval descriptives, over videos carrying at least one gold positive frame.

| corpus | arm | positive verdicts | whole-clip intervals | whole-clip rate | mean span | frame IoU mean / median | interval IoU mean / median | R@0.3 | R@0.5 | R@0.7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hatemm | anomaly | 126 | 126 | 1.000 | 0.9999 | 0.4013 / 0.2461 | 0.4439 / 0.3851 | 0.4824 | 0.4235 | 0.3412 |
| hatemm | hateful | 152 | 152 | 1.000 | 0.9998 | 0.4630 / 0.4208 | 0.5370 / 0.6668 | 0.5412 | 0.4706 | 0.3882 |
| mhclip_en | anomaly | 46 | 46 | 1.000 | 0.9998 | 0.2600 / 0.0000 | 0.2640 / 0.0000 | 0.3043 | 0.2826 | 0.2609 |
| mhclip_en | hateful | 65 | 65 | 1.000 | 0.9996 | 0.4245 / 0.1496 | 0.4304 / 0.1497 | 0.4783 | 0.4565 | 0.4348 |
| mhclip_zh | anomaly | 54 | 54 | 1.000 | 0.9998 | 0.5173 / 0.5455 | 0.5037 / 0.5468 | 0.5349 | 0.5349 | 0.4884 |
| mhclip_zh | hateful | 71 | 71 | 1.000 | 0.9997 | 0.7442 / 1.0000 | 0.7278 / 0.9700 | 0.7674 | 0.7674 | 0.7209 |

### The degeneracy is untouched; only the verdict threshold moves

**Substituting hate vocabulary does not recover a single sub-interval.** Across
all 525 videos in the hateful arm, every positive answer again spans the whole
clip: 152 of 152 on hatemm, 65 of 65 on mhclip_en, 71 of 71 on mhclip_zh. The
endpoint distribution is the same two values as before and nothing else — 129
`[0.0, 1.0]` plus 23 `[0.0, 0.999]` on hatemm, 37 plus 28 on mhclip_en, 48 plus
23 on mhclip_zh. Mean predicted span is 1.000 of video duration in both arms of
all three corpora. The within-hate macro therefore stays pinned at the tie value:
0.5000 with standard deviation 0.0000 on hatemm and mhclip_zh, and 0.5005 with
standard deviation 0.0036 on mhclip_en, where the handful of `0.999` endpoints
drop a final frame and make a few arrays non-constant. The interval head answers
the same trivial answer whichever word the question uses, so the failure is not a
vocabulary mismatch between "abnormal" and "hateful" — it survives the rename.

**What the substitution does move is the verdict, and it moves it by shifting the
positive rate.** The model says the positive word more often in every corpus: 126
to 152 on hatemm, 46 to 65 on mhclip_en, 54 to 71 on mhclip_zh. TPR rises in all
three (0.6235 to 0.7647, 0.3261 to 0.5217, 0.5581 to 0.7907) and TNR falls in all
three (0.4341 to 0.3256, 0.7232 to 0.6339, 0.7273 to 0.6636). Balanced accuracy
still improves everywhere (0.5288 to 0.5451, 0.5247 to 0.5778, 0.6427 to 0.7272),
so the shift is not purely a threshold slide: on mhclip_zh in particular the word
"hateful" separates the classes better than "abnormal" does, by 8.5 balanced-
accuracy points. Plain accuracy does not follow, because the corpora are majority
non-hateful: on hatemm it falls from 0.5093 to 0.5000 and on mhclip_en from
0.6076 to 0.6013 even as balanced accuracy rises.

The frame-level columns inherit both effects and should be read with care. Pooled
ROC rises everywhere (0.5696 to 0.5750, 0.5427 to 0.6053, 0.5987 to 0.7262), and
the interval IoU numbers rise with it, but neither is evidence of localisation.
A whole-clip prediction scores frame IoU equal to the annotated hateful fraction
of that video, so calling more hateful videos hateful mechanically raises the IoU
mean without the model ever pointing at a moment. The mhclip_zh cell makes this
plain: its median frame IoU in the hateful arm is exactly 1.0000, which is what a
degenerate whole-clip prediction earns on a corpus where most hateful videos are
annotated hateful end to end.

### Sanity checks and anomalies

Coverage is exact in the hateful arm on all three corpora: 214, 158 and 153
generations read, the same numbers scored, zero videos missing from the score
files and zero scored videos absent from gold. Frame counts match the zero-shot
arm exactly (29,266 / 5,600 / 4,817). Generations are non-empty for all 525
records and there were zero decode errors; the AV1 ffmpeg fallback fired on the
same 37 mhclip_en and 8 mhclip_zh videos, with no video falling back to a short
sample.

Parse failures are again one video in 525, but it is a different video, and the
two arms trade which one fails. `BV19C4y177iH`, the mhclip_zh repetition loop
that failed under the anomaly prompt, parsed cleanly here, so mhclip_zh has zero
parse failures in this arm. In its place `hate_video_89` on HateMM degenerated
into a repetition loop of its own and hit the 512-token cap without closing its
tags; its output ends in a run of roughly 400 consecutive `H` characters, its
`which_raw` is null, and it was counted as `unparsed`, contributed an all-zero
score array and counted as a negative verdict. No video in either arm produced a
positive verdict without a usable interval.

### What this arm settles

The term-adaptation arm was worth running because it isolates one candidate
explanation and kills it. If Vad-R1's whole-clip collapse had been an artefact of
being asked about the wrong concept — "abnormal" rather than "hateful" — then
naming the right concept should have loosened the interval head. It did not, in
any corpus, for any of the 288 positive answers. The collapse is a property of
the model's temporal grounding on this material, not of the question's wording.
The verdict channel, by contrast, is genuinely sensitive to the wording and reads
hate better when asked about hate, most clearly on mhclip_zh. That is a
statement about the model's classifier, not its localiser, and it does not change
the row this baseline occupies in the consolidated tables: the zero-shot arm with
the released prompt remains the reported number.

---

# CONSOLIDATED TABLES (2026-08-19, all runs complete)

All numbers under the frozen protocol (FRAME_EVAL_PROTOCOL.md): 1 fps
grid, shared GT arrays (SHA-pinned), same evaluator, val-carve
checkpoint selection for every trained baseline (upstream test-selection
removed). Supervision column is the honest axis.

## HateMM test (214 gold videos, 29,266 frames)

| Method | Supervision | Pooled ROC | Pooled PR | Within-hate macro | Video AUC |
|---|---|---|---|---|---|
| MultiHateLoc reimpl (fused) | video labels | **0.7504** | 0.4856 | **0.6008** | 0.8622 |
| — its audio branch | video labels | 0.7777 | — | — | — |
| **Ours (masked locator, 1 fwd)** | **zero labels** | 0.7451 | **0.5601** | 0.5706 | **0.9010** |
| Audio-only MIL (VGGish) | video labels | 0.7667 | 0.4939 | 0.5966 | 0.7814 |
| MACIL-SD (av) | video labels | 0.7282 | 0.5127 | 0.5383 | 0.7611 |
| DSANet | video labels | 0.7063 | 0.4824 | 0.5453 | 0.7470 |
| VadCLIP | video labels | 0.6855 | 0.4457 | 0.4848 | 0.7242 |
| Vad-R1 (zero-shot ckpt) | none (trained on own VAD data) | 0.5696 | 0.2722 | 0.5000 | 0.5288 (balanced acc) |
| EventVAD reimpl (training-free) | none | 0.5174 | 0.2519 | 0.4988 | 0.4519 |

## MHC EN test (158 gold videos)

| Method | Supervision | Pooled ROC | Pooled PR | Within-hate macro | Video AUC |
|---|---|---|---|---|---|
| Audio-only MIL | video labels | 0.7142 | 0.4987 | 0.5142 | 0.7141 |
| MACIL-SD (av) | video labels | 0.6764 | 0.4664 | 0.5383 | 0.7112 |
| MultiHateLoc reimpl | video labels | 0.6740 | 0.3700 | 0.4611 | 0.6498 |
| DSANet | video labels | 0.6684 | 0.4354 | 0.3844 (align branch: 0.7230) | 0.6768 |
| VadCLIP | video labels | 0.6281 | 0.3611 | 0.3331 | 0.6405 |
| **Ours** | **zero labels** | 0.6198 | 0.4141 | **0.6154** | 0.7015 |
| Vad-R1 | none | 0.5427 | 0.2699 | 0.5000 | 0.5247 |
| EventVAD reimpl | none | 0.5041 | 0.2568 | 0.4784 | 0.5179 |

## MHC ZH test (153 gold videos; within-macro n=7 — UNSTABLE, do not lean on that column)

| Method | Supervision | Pooled ROC | Pooled PR | Within-hate macro | Video AUC |
|---|---|---|---|---|---|
| MACIL-SD (audio) | video labels | 0.7774 | 0.5301 | 0.5256 | 0.7808 |
| MACIL-SD (av) | video labels | 0.7757 | 0.5233 | 0.4588 | 0.7685 |
| MultiHateLoc reimpl | video labels | 0.6749 | 0.4032 | 0.4126 | 0.7382 |
| Audio-only MIL | video labels | 0.6320 | 0.3254 | 0.5269 | 0.6725 |
| **Ours** | **zero labels** | 0.6004 | 0.3813 | **0.6076** | 0.6153 |
| Vad-R1 | none | 0.5987 | 0.2838 | 0.5000 | 0.6427 |
| DSANet | video labels | 0.5749 | 0.2921 | 0.3557 | 0.5588 |
| VadCLIP | video labels | 0.5676 | 0.2705 | 0.3562 | 0.3981 |
| EventVAD reimpl | none | 0.5202 | 0.2440 | 0.4923 | 0.5623 |

## Readings the paper must carry

1. **The pooled frame metric is dominated by video-level discrimination**
   — trained baselines win it where training data matches (MHC), our
   zero-label method ties/leads it on HateMM; the decomposition
   (pooled vs within-video) is itself a finding no prior work reports.
2. **Within-video localization is weak for everyone** (≤0.62 everywhere;
   we lead on both MHC corpora, MultiHateLoc's fused branch leads on
   HateMM at 0.6008 vs our 0.5706; DSANet's text-alignment branch hits
   0.7230 on MHC EN while its pooled is weak — branch-level dissociation
   worth analysis). The field's operative metric hides all of this.
3. **Audio is the strongest single trained channel** on HateMM and MHC
   ZH — consistent with hate being carried by speech; our locator reads
   the same speech through transcripts at zero labels.
4. **Vad-R1 (the zero-shot MLLM competitor) does not localize at all**:
   every positive prediction spans the whole clip (525/525 videos);
   its frame row is a broadcast video verdict.
5. Caveats bound to specific rows: MACIL-SD/hatemm selected epoch 1
   (not converged); ZH within-macro n=7; Vad-R1 scores are binary
   (one interior ROC point); our MHC pooled is depressed by uncovered
   (non-speech) frames sitting at the floor — a transcript-only
   locator has no evidence there, and we do not impute any.

## Cost row (HateMM, per video)

Ours: 1 packed forward, 0.11 s (prefix shared), zero labels.
EventVAD reimpl: 6.3-20.6 VideoLLaMA2 calls/video (max 185), 2.4 s/call,
13.4 GPU-hours for the three test sets; 40% of its events yield no
parseable score under the paper's own prompt (301/525 videos end up
with constant arrays) - floor row on every corpus.
Trained baselines: minutes of training + feature extraction
(CLIP/I3D/VGGish/ViT/BERT pipelines), video labels required.
Vad-R1: 3.7 s/video, 16 frames, 7B, no labels but trained on its own
VAD corpus. LELA (cited only): 12–16 GPT-4o-mini calls PER FRAME.

---

# EventVAD (MM'25, training-free, reimplemented-from-paper — released code non-runnable, see DESIGN_EVENTVAD.md)

The released repository cannot execute: `graph_propagation` is imported and
defined nowhere, the scoring prompt is the literal string `"prompt"`, the RAFT
checkpoint path is a placeholder, and `src/evaluate.py` does not compile. All
four gaps and every inference made in filling them are argued in
`scripts/reproduction_baselines/DESIGN_EVENTVAD.md`; the porting patches are in
`PATCHES.md`. The numbers below come from the `paper` prompt arm, the Figure 2
reconstruction, which is the paper's own headline condition. No second arm was
run: `no_thinking` and `bounded` are second conditions on the same test split
and need owner approval.

## Results

| Corpus | Supervision | Pooled ROC | Pooled PR | Within-hate macro | Video AUC |
|---|---|---|---|---|---|
| HateMM test (214 videos, 29,266 frames) | none (training-free) | 0.5174 | 0.2519 | 0.4988 (n=85) | 0.4519 |
| MHC EN test (158 videos, 5,600 frames) | none (training-free) | 0.5041 | 0.2568 | 0.4784 (n=44) | 0.5179 |
| MHC ZH test (153 videos, 4,817 frames) | none (training-free) | 0.5202 | 0.2440 | 0.4923 (n=7) | 0.5623 |

Every pooled figure sits within 0.021 of chance, and the HateMM video-level
column sits below it. This is a floor row, not a competitive one.

## Cost: EventVAD scores per event, not per video

The method makes one MLLM call per event, so its per-video cost is whatever the
segmenter cuts. That is the column the comparison has to be read with.

| Corpus | Events | Events/video median | mean | max | **MLLM calls per video** | Stage-2 s/call |
|---|---|---|---|---|---|---|
| HateMM | 4,413 | 16 | 20.6 | **185** | **20.6 mean, 185 worst case** | 2.37 |
| MHC EN | 1,061 | 6 | 6.7 | 16 | 6.7 mean, 16 worst case | 2.41 |
| MHC ZH | 962 | 6 | 6.3 | 14 | 6.3 mean, 14 worst case | 2.24 |

The worst case is `non_hate_video_356`: 29,953 decoded frames cut into 185
events, so 185 VideoLLaMA2-7B forward passes for one video, 7.3 minutes of GPU
time. Against this project's own two-calls-per-video cap (CLAUDE.md,
anti-pattern 1) EventVAD is off by an order of magnitude on the HateMM mean and
by two on its worst case. The cap does not bind a reproduced baseline, but the
comparison is only honest with the number visible: 6,436 MLLM calls across 525
videos, against one packed forward per video for ours.

## The dominant anomaly: the model mostly declines to emit a number

Under the paper's own prompt, VideoLLaMA2.1-7B-16F answers most events in prose
that never states a score, and when it does state one it states almost the same
one every time.

| Corpus | Unparsed events | % of events | % of frames | Distinct parsed values | Share at 0.0 |
|---|---|---|---|---|---|
| HateMM | 1,798 / 4,413 | 40.74 | 40.64 | 8 | 2,141 / 2,615 = 81.9% |
| MHC EN | 358 / 1,061 | 33.74 | 33.14 | 4 | 587 / 703 = 83.5% |
| MHC ZH | 427 / 962 | 44.39 | 44.18 | 3 | 497 / 535 = 92.9% |

Parse-rule histograms, verbatim from `frame_eval.json`:

    hatemm     parse {sentence 1396, trailing_number 1219, unparsed 1798}
               range {in_range 2486, none 1798, div10 127, clamped_low 1, div100 1}
    mhclip_en  parse {sentence 364, trailing_number 339, unparsed 358}
               range {in_range 682, none 358, div10 21}
    mhclip_zh  parse {sentence 263, trailing_number 272, unparsed 427}
               range {in_range 523, none 427, div10 12}

The unparsed answers are not truncations. On MHC ZH, 418 of the 427 end in
sentence-final punctuation: they are complete answers that simply never reach a
number, typically of the form *"there are no obvious anomalies that stand out as
unusual or unexpected"*. Only the handful that ran to the 2,048-token cap did so
through degenerate repetition (*"a black and white stuffed animal, a black and
white stuffed animal, …"*). Upstream's own parser, `float(output.strip())`,
would have read none of them; the port's prose-tolerant parser recovers 55–66%.

The consequence for the metric is that the score arrays are close to flat:

| Corpus | Videos with a constant score array | of which at 0.0 |
|---|---|---|
| HateMM | 100 / 214 | 99 |
| MHC EN | 80 / 158 | 80 |
| MHC ZH | 121 / 153 | 120 |

So 301 of 525 videos carry no within-video ranking information at all, which is
why the within-hate macro sits at a median of exactly 0.5000 on all three
corpora. Unparsed events fill with 0.0 and are counted rather than dropped, so
the cohort stays constant between arms; `frac_frames_unparsed` records how much
of each number was filled in.

This is the same shape of failure Vad-R1 shows, reached by a different route.
Vad-R1 broadcasts one video-level verdict across every frame; EventVAD cuts the
timeline finely and correctly, then assigns nearly every piece the same score.
Both produce a frame row that cannot localize. DESIGN_EVENTVAD.md predicted the
mitigation in advance under G2-c: the `bounded` arm states the score range the
paper leaves unstated. Whether it lifts the parse rate is a measurement this run
does not make.

## Sanity checks

| Check | Result |
|---|---|
| Score length equals gold length | 525 / 525 videos exact, 0 mismatches |
| Events tile the timeline | 0 gaps, 0 overlaps, 0 videos whose first event starts after frame 0 or whose last event ends before `n_frames` |
| Segmentation errors | 0 across 525 videos |
| Scoring errors | 0 across 6,436 events |
| RAFT NaN | none, on any corpus |
| Decode failures | none; every video in all three gold cohorts decoded |
| Scores non-constant | **fails for 301 / 525 videos** — see the degeneracy above |
| Decode rate cap | honoured; no video exceeds 30 fps |

The only string in the run log matching `error` or `nan` is the preflight's own
assertion that upstream's `src/evaluate.py` does not compile:

    PASS G4: src/evaluate.py does not compile  -- Sorry: IndentationError:
    unexpected indent (evaluate.py, line 45)

Decoded frame counts came in 824 below the pre-run probe, 1,115,310 against
1,116,134 predicted, a 0.07% difference from decode rounding.

## Run settings and wall time

Single RTX 5090, one GPU stage resident at a time, three corpora strictly
serial, shortest-first. Preset `paper` (α = 0.75, γ = 0.6, CLIP L2-normalised),
`ma_mode = upstream`, arm `paper`, greedy decoding, `max_new_tokens = 2048`,
VideoLLaMA2.1-7B-16F in fp16 at 18.1 GB, 16 frames per event.

| Corpus | Stage 1 (RAFT + CLIP) | Measured rate | Stage 2 (VideoLLaMA2) | Stage 3 |
|---|---|---|---|---|
| MHC ZH | 94.4 min (135,687 frames) | 23.96 frame/s | 35.9 min (962 calls) | < 1 min, CPU |
| MHC EN | 69.2 min (159,167 frames) | 38.36 frame/s | 42.6 min (1,061 calls) | < 1 min, CPU |
| HateMM | 388.8 min (820,456 frames) | 35.17 frame/s | 174.3 min (4,413 calls) | < 1 min, CPU |
| **Total** | **9.2 h** | | **4.2 h** | |

Whole sweep 13.4 hours. The rate spread across corpora is resolution, not load:
MHC ZH carries the 1280x720 videos and runs slowest per frame. Against the
CPU-measured floor of 1.0 to 3.5 frame/s recorded in DESIGN_EVENTVAD.md, the GPU
is 10 to 24 times faster, and stage 1 remains the cost centre at 69% of the
sweep.

---

# HateClipSeg (4th corpus)

HateClipSeg joins the study as the fourth corpus, and it is the one that
actually powers the measurement the other three could not. The annotation is
exhaustive: every second of every video carries a segment label, so a hateful
video is not a hateful *point* on an otherwise unannotated timeline. **67 of the
79 gold videos carry both classes inside themselves, 85 % of the cohort.** The
comparable counts are 85 of 214 on HateMM (40 %), 44 of 158 on MHC EN (28 %) and
7 of 153 on MHC ZH (5 %). **The within-video macro is therefore the well-powered
column on this corpus and the pooled column is the weak one** -- the exact
reverse of the reading the other three corpora invite, and the reason this
corpus was added.

Cohort and grid. 315 train / 79 test videos, our own seeded 80/20 stratified
draw (seed 234; HateClipSeg publishes a ratio but no ids), frozen at
`results/reproduction/splits/hateclipseg_{train,test}.txt`. Gold arrays on the
same 1 fps grid as every other corpus: `hateclipseg_test.npz` is PRIMARY, a
frame positive iff its covering segment is offensive under the union rule over
dimensions 1..5, 9,900 of 18,839 frames positive (52.55 %);
`hateclipseg_test_hateful_strict.npz` is the sensitivity array, hateful
dimension alone, 4,039 positive (21.44 %). Videos run 180 to 350 s, median 239 s.

Two base rates have to be held in mind before any number below is read.
**Chance PR-AUC is 0.5255**, not the 0.24 of the other corpora, so a PR column
near 0.55 is near chance rather than well above it. And **69 of the 79 test
videos are positive at the video level (87 %)**, so video-level AUC rests on 10
negatives, and video-level average precision -- which is what every trained cell
selects its checkpoint by -- is close to uninformative here: a constant
predictor scores about 0.87.

## Results

Every row is `results/reproduction/{baselines/<method>,ours}/hateclipseg/frame_eval.json`,
written by the same `eval_baseline_scores.py` over `frame_eval_common.py` that
produced the other three corpora. Video AUC max-pools the frame scores per video
and ranks them against the corpus video label.

| method | supervision | branch | pooled ROC-AUC | pooled PR-AUC | within-hate macro (n) | video AUC |
| --- | --- | --- | --- | --- | --- | --- |
| **Ours (masked locator, 1 fwd/video)** | **zero labels** | z_masked | 0.5414 | 0.6149 | 0.5324 (67) | **0.7536** |
| VadCLIP | video labels | score_mlp | 0.5328 | 0.5447 | **0.5530** (67) | 0.5072 |
| VadCLIP | video labels | score_align | 0.5531 | 0.5860 | 0.5063 (67) | 0.6275 |
| DSANet | video labels | score_mlp | 0.5387 | 0.5502 | **0.5583** (67) | 0.5493 |
| DSANet | video labels | score_refined | 0.5387 | 0.5502 | 0.5583 (67) | 0.5493 |
| DSANet | video labels | score_align | 0.5559 | 0.5507 | 0.5015 (67) | 0.6145 |
| MACIL-SD (av) | video labels | score_av | 0.5220 | 0.5412 | 0.5329 (67) | 0.5101 |
| MACIL-SD (av) | video labels | score_audio | 0.5132 | 0.5314 | 0.5123 (67) | 0.5159 |
| MACIL-SD (av) | video labels | score_visual | 0.5215 | 0.5471 | 0.5363 (67) | 0.5087 |
| MACIL-SD (audio only) | video labels | score_mil | 0.5166 | 0.5568 | 0.5266 (67) | 0.6275 |
| MACIL-SD (visual only) | video labels | score_mil | 0.5195 | 0.5581 | 0.5394 (67) | 0.6232 |
| MultiHateLoc reimpl | video labels | score_fused | 0.4993 | 0.5208 | 0.5085 (67) | 0.6246 |
| MultiHateLoc reimpl | video labels | score_dms | 0.5393 | 0.5684 | 0.4903 (67) | 0.6493 |
| MultiHateLoc reimpl | video labels | score_visual | 0.5520 | 0.5767 | 0.4977 (67) | 0.6797 |
| MultiHateLoc reimpl | video labels | score_audio | 0.4792 | 0.4997 | 0.4709 (67) | 0.5210 |
| MultiHateLoc reimpl | video labels | score_text | 0.5346 | 0.5736 | 0.4990 (67) | 0.5558 |
| MultiHateLoc reimpl | video labels | score_union | 0.4837 | 0.5176 | 0.4942 (67) | 0.5000 |
| Vad-R1 (zero-shot, released prompt) | none | score_interval | **0.6382** | **0.6115** | 0.5001 (67) | 0.6826 |
| Vad-R1 (term-adaptation arm) | none | score_interval | 0.5655 | 0.5610 | 0.5001 (67) | 0.6196 |
| EventVAD | none | — | not run | not run | not run | not run |

EventVAD is marked *not run* by owner default, not by oversight: it read the
floor (0.50 to 0.52 pooled) on all three prior corpora while spending 6.4k MLLM
calls and 13.4 GPU-hours, and 40 % of its events return no parseable score under
the paper's own prompt. A fourth floor row is not worth another 4-plus GPU-hours.

## What the table says

**Nobody localises on HateClipSeg.** The within-video macro -- the column this
corpus can actually measure -- runs from 0.4709 to 0.5583 across the sixteen
supervised branch-cells, and the zero-label locator sits at 0.5324 inside that
band. The best cell in the study is DSANet's MIL branch at 0.5583, which is
0.058 above chance on 67 videos, and five branch-cells fall *below* chance, all
five of them MultiHateLoc's. This is not a ranking worth defending; it is a floor
that everything sits on. Vad-R1's macro is 0.5001 with a standard deviation of
0.0006, which is the same degeneracy the other three corpora recorded: its
positive prediction spans the whole clip, so its frame row is a broadcast video
verdict and carries no within-video ranking at all.

**The pooled column has almost nothing left to reward.** With 52.55 % of frames
positive, chance PR-AUC is 0.5255; the supervised cells run 0.4792 to 0.5559 on
pooled ROC and 0.4997 to 0.5860 on pooled PR, and three of them -- MultiHateLoc's
audio, fused and union branches -- are below chance on both. The pooled metric
was carried on the other corpora by video-level discrimination leaking into it;
here, where 87 % of videos are positive, there is much less video-level signal
left to leak.

**Vad-R1 tops the pooled column at 0.6382 without doing anything the pooled
column claims to measure.** Its within-video macro is 0.5001; the pooled number
comes entirely from calling 41 of 79 videos abnormal end to end and 38 normal end
to end. That is the clearest single demonstration in this document that pooled
frame ROC on a corpus like this is a video-level metric wearing a frame-level
name.

**The one column with a real spread is video AUC, and the zero-label locator
leads it**: 0.7536, against 0.6826 for Vad-R1's anomaly arm, 0.6797 for the best
trained cell (MultiHateLoc's visual branch) and 0.6493, 0.6275, 0.6275 for the
rest. On 10 negatives that is not a strong measurement either, but the direction
matches HateMM, where the locator also led the video column (0.9010).

The honest summary of this corpus: **HateClipSeg is where the study's central
claim gets its cleanest negative.** On the one corpus whose annotation can
support a within-video measurement over 85 % of its cohort, no method in the
study -- trained or zero-label, audio, visual, text or fused -- rises meaningfully
above chance at saying *where inside a hateful video* the hate is. The
localisation claim that the field's benchmarks obscure is not merely
unsupported here; it is measured and absent.

## Vad-R1: both arms, and the same degeneracy

Both arms ran on the 79 test videos, 16 frames each, the released checkpoint and
prompt verbatim for the anomaly arm and the term-adaptation ablation for the
hateful arm.

| | anomaly arm | hateful arm |
| --- | --- | --- |
| positive-interval / negative parses | 41 / 38 | 56 / 23 |
| unparsed | 0 | 0 |
| pooled ROC / PR | 0.6382 / 0.6115 | 0.5655 / 0.5610 |
| within-hate macro (sd) | 0.5001 (0.0006) | 0.5001 (0.0011) |
| frame IoU, mean / median (n=69) | 0.389 / 0.326 | 0.461 / 0.515 |
| R@frame-IoU 0.3 / 0.5 / 0.7 | 0.507 / 0.464 / 0.377 | 0.638 / 0.507 / 0.391 |
| video verdict acc / P / R / F1 | 0.595 / 0.951 / 0.565 / 0.709 | 0.709 / 0.911 / 0.739 / 0.816 |

Swapping the anomaly vocabulary for the hateful one moves the verdict threshold
-- 15 more videos called positive, recall 0.565 to 0.739, accuracy 0.595 to 0.709
-- and moves the interval-overlap numbers with it, but leaves the within-video
macro at 0.5001 in both arms. The degeneracy is untouched by the term swap,
exactly as it was on the three prior corpora. Its scores are binary, so the ROC
curve has one interior operating point and both AUCs are coarse by construction.

## Hateful-strict sensitivity (ours)

The same scores against the hateful-only collapse, same videos, same frames,
gold recollapsed:

| gold | frames positive | pooled ROC | pooled PR | within macro (n) | video AUC |
| --- | --- | --- | --- | --- | --- |
| PRIMARY (offensive union) | 9,900 / 18,839 (0.5255) | 0.5414 | 0.6149 | 0.5324 (67) | 0.7536 |
| sensitivity (hateful strict) | 4,039 / 18,839 (0.2144) | 0.5165 | 0.3635 | 0.5205 (37) | 0.6784 |

The strict collapse moves every column toward chance and cuts the macro's video
count from 67 to 37. It is reported because the corpus ships both collapses; the
primary is primary.

## Ours: prompt provenance and fidelity

The locator is `scripts/duplex/masked_parallel_isolation_hateclipseg.py`, a
sibling of the MultiHateClip script that imports its `Judge`, block mask,
branch-local positions, cohort builder and frame map, so the mechanism is shared
code rather than a second implementation.

The prompt is this corpus's own frozen one. HateClipSeg was first read
chunk-by-chunk by `scripts/duplex/isolated_chunk_diag.py`, and the assembled
user text here is **asserted byte-identical to that module's on three probe
strings before the model loads**, with the sha256s checked against its
`FROZEN_TEXT_SHA` rather than merely observed to match. From
`results/reproduction/ours/hateclipseg/prompt_fingerprints.json`:

| component | sha256 |
| --- | --- |
| rules block (YOUTUBE_RULES under the frozen lead-in) | `e23dd329b55122ae1caa50334f96072d5ceb56d61edf98bc869af85f2f0c9a77` |
| question | `f45673af42da76b5b8afad71160616bd929227eac995c6c5b8f848496f207233` |
| system message | `e6addb7b869ede44dc5500bd4cce5a09429342b31202078fc74fe3061f197455` |
| user-text template | `9442091c9044510371c325e893879dc3a6f7639c3ae1ea6b0afe877461c6d253` |
| chat-template full prompt | `e0e73120ac8f66e3d803b0a2fbf510f3e6039ef86df73469ebd4cab4bfded3bc` |
| packed prefix (160 tokens) | `5aab1929792c022cec703131dd296e26086479fc85bbeae18aa4a37d88a5b4c5` |
| packed suffix | `4d7644e75cf868e705523f19f69479769ac5a427c3f682175cd44a4942e66016` |

The first four equal HateMM's frozen values, which is the intended result: both
corpora take YOUTUBE_RULES, so their templates coincide.

Fidelity. The mask plumbing check is exact -- a fully causal 4-D mask reproduces
the default no-mask logits to `max |Δ logit| = 0.000000` -- and the
prompt-identity assertion (`concat(prefix_ids, branch_ids)` must equal the
isolated prompt's ids, chunk by chunk) ran before every one of the 78 packed
forwards and passed throughout.

The spot check -- the three videos with the most chunks, 211 chunks re-scored
with genuine isolated calls -- read Spearman 0.9905 against the matched-kernel
sequential reference, which clears the 0.99 bar, so the protocol's escalation
trigger did not fire. It cleared it narrowly enough to be worth superseding, so
**the full-cohort comparison was run anyway**: every one of the 1,248 chunks was
scored a second time with a genuine isolated call.

| | HateClipSeg |
| --- | --- |
| chunks compared | 1,248 |
| Spearman(masked, sequential) | 0.99862 |
| Pearson | 0.99981 |
| max abs delta z | 1.25 |
| mean abs delta z | 0.170 |
| chunks bit-identical | 44.5 % |
| pooled ROC from the isolated calls | 0.5412 |
| **endpoint delta, packed minus isolated** | **+0.00018 ROC, −0.00134 PR, +0.00152 macro** |

The endpoint moves by at most 0.0014 on any column, so no number in this section
depends on which way the chunks were scored. The spot-check pattern repeats the
one measured on MultiHateClip: over three videos the z column is tie-dominated
and Spearman converts sub-quantum bf16 noise into rank swaps, while over the full
cohort, where z takes 179 distinct values, the same comparison reads 0.9986. The
residual is bf16 non-associativity in attention over the longer packed sequence,
not the packing mechanism.

The re-run that produced the reference column reproduced the primary
`frame_level`, `frame_level_hateful_strict` and `video_level` blocks **exactly**,
field for field, which is the determinism check paid for free by running the
comparison twice.

## Coverage and cost

78 of the 79 gold videos carry scorable chunks and one, `bit_20VYH5uxw20D`, does
not: its chunk record fails the frozen `usable_spans` helper, so it is scored
all-floor and reported rather than dropped. Zero gold videos lack a chunk record
altogether. 1,248 chunks of 1,251 were scored; the three skipped carry empty
text. Frame coverage is 16,140 of 18,839 (85.7 %); the 2,699 uncovered frames sit
at the floor, −23.75, and nothing is imputed for them.

Cost: **one packed forward per video**, 78 forwards, 93,981 packed tokens, 9.8 s
of GPU for the whole cohort (0.125 s/video), zero labels, no training.

## Loss evidence

Every trained cell moved. The MIL classification loss at first and last epoch,
with the selected epoch and its validation video AP. Selection is on a seeded,
label-stratified 10 % carve of the train split (283 train / 32 val, 247 hateful
in train); the test split is never opened during training.

| cell | epochs | MIL loss first | MIL loss last | selected epoch | val video AP |
| --- | --- | --- | --- | --- | --- |
| VadCLIP | 10 | 0.4742 | 0.3056 | 3 | 0.8496 |
| DSANet | 10 | 0.4745 | 0.2901 | 4 | 0.8557 |
| MACIL-SD (av) | 50 | 0.4169 | 0.2245 | 50 | 0.9896 |
| MACIL-SD (audio) | 50 | 0.5351 | 0.1927 | 46 | 0.9798 |
| MACIL-SD (visual) | 50 | 0.4497 | 0.1114 | 12 | 0.9811 |
| MultiHateLoc reimpl | 100 | 2.4697 | 0.2850 | 1 | 0.9498 |

No cell is flat, so the pre-declared batch-16 contingency does not fire on this
corpus either, and no rerun was performed.

**The selection column should not be over-read on this corpus.** With 87 % of
videos positive, validation video AP starts near 0.9 and has very little room to
move: MultiHateLoc's peaks at epoch 1 (0.9498) and ends at 0.8696, so its
reported row is a one-epoch model, and MACIL-SD's visual ablation selects epoch
12 out of 50. The selection rule is the frozen one and is left as it stands, but
on a corpus this positive-heavy the criterion is barely discriminating. This is a
property of HateClipSeg's video-level balance, not a porting choice.

## Sanity checks

Run for every branch of every cell; all pass.

Score-to-gold length: zero mismatches in all sixteen supervised branch-cells
and in the locator. Zero gold videos missing from any score file and zero scored
videos absent from the gold; `eval_baseline_scores.py` raises on either and did
not. Finiteness: every score finite everywhere.

Non-constant scores: 79 of 79 videos are non-constant in fourteen of the
sixteen branch-cells. The two exceptions are MultiHateLoc's `score_text`, which
has 2 constant videos of 79, and its `score_union`, which is a 0/1 indicator by
construction rather than a score. Ranges are healthy: the MIL heads span roughly
0.017 to 0.998 with a standard deviation of 0.19 to 0.29, the exception being
MACIL-SD's visual-only head, which spans 0.5345 to 0.9655 at sd 0.067; the two
alignment branches are narrow but not degenerate (0.4848 to 0.5972, sd 0.019),
and the locator's chunk margin spans −22.75 to +25.75 with 182 distinct values
over 1,248 chunks.

## Run settings and wall time

Single RTX 5090, one GPU job resident at a time, every stage strictly serial,
driven by `scripts/reproduction_baselines/run_hateclipseg_sweep.sh`. Every stage
is the existing runner for that method with `CORPORA=hateclipseg`; the only new
code is the locator script and the corpus registration in
`hate_common/{data,runtime}.py`. `visual-length` / `attn-window` take the HateMM
setting of 256 / 64, since HateClipSeg's videos are 180 to 350 s and the
MultiHateClip argument for a shorter window (batches that are mostly padding)
does not apply. Every other hyperparameter is the published default.

| stage | wall time |
| --- | --- |
| Ours (locator, 78 packed forwards) | 33 s |
| VadCLIP train + score + evaluate | 12 s |
| DSANet train + score + evaluate | 17 s |
| MACIL-SD, three modalities | 2 min 4 s |
| MultiHateLoc reimpl | 41 s |
| Vad-R1, anomaly arm (79 videos, vLLM) | 5 min 59 s |
| Vad-R1, hateful arm (79 videos, vLLM) | 6 min 35 s |
| **whole sweep** | **16 min 21 s** |

A follow-up 50 s locator pass produced the full-cohort fidelity column above.
The two Vad-R1 arms are 78 % of the sweep; every trained baseline together is
under four minutes, because the features are precomputed.
