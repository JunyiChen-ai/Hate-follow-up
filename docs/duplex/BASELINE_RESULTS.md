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
