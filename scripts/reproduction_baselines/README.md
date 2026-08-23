# Reproduction baselines

> **Protocol amendment (2026-08-23).** New `official-val` runs preserve the
> released validation sets for HateMM, MHC-EN and MHC-ZH. Earlier rows merged
> train+validation and made a seeded 10% carve; those measurements remain
> archived as `legacy-resplit-val` and must not be mixed into an official-val
> ranking. HateClipSeg releases no split IDs, so its frozen 394-video cohort
> keeps the original 79-video test manifest (SHA256 `0d648643...`) and divides
> the former train cohort into 252 train / 63 validation videos, stratified by
> the video-level offensive-union label with seed 234.

The validation-only search entry point is `tune_official_val.py`. It supports
VadCLIP, DSANet, MACIL-SD (AV/audio/visual), MultiHateLoc, CMHKF and both the
centralized and three-client Fed-WSVAD protocols; trials never load test.
VERA is staged in `vera_adapter.py`: `select` chooses guiding questions with
official video-level validation labels, `infer` performs resumable official
InternVL2-8B sliding-window inference, and `postprocess` applies VERA's visual
neighbour propagation and Gaussian smoothing. Test inference is a separate,
explicit command after the validation choice has been archived.

Video-anomaly-detection baselines, ported to the hateful-video corpora of the
reproduction study. Each predicts a score per temporal unit; the study reads
those scores as frame-level hate localisation on the 1 fps grid.

| method | venue | supervision | modality | upstream | commit |
| --- | --- | --- | --- | --- | --- |
| VadCLIP | AAAI 2024 | weak, video-level | CLIP ViT-B/16, 1 fps | https://github.com/nwpu-zxr/VadCLIP | `c41067f` |
| DSANet | AAAI 2026 | weak, video-level | CLIP ViT-B/16, 1 fps | https://github.com/lessiYin/DSANet | `eb335b2` |
| MACIL-SD | ACM MM 2022 | weak, video-level | I3D 5-crop + VGGish | https://github.com/JustinYuu/MACIL_SD | `c20943f` |
| EventVAD | ACM MM 2025 | **training free** | CLIP + RAFT, VideoLLaMA2 7B | https://github.com/YihuaJerry/EventVAD | `25cacd8` |
| LAVAD | CVPR 2024 | **training free** | 5x BLIP-2 + Llama-2-13B + ImageBind | https://github.com/lucazanella/lavad | `1ad46c6` |

The three weakly-supervised ports train on video-level labels only. EventVAD
trains nothing and runs on the test cohorts alone.

Everything between here and the "MACIL-SD" heading describes the two CLIP
baselines. **MACIL-SD and EventVAD each have their own section at the end of
this file**, because neither shares code with the CLIP pair and each consumes
different inputs. EventVAD additionally has `DESIGN_EVENTVAD.md`, because its
released code cannot be run and two of its components had to be reconstructed
from the paper.

LAVAD is separately specified in `DESIGN_LAVAD.md`. Its cohort adapter and
shared-evaluator handoff are complete, but no final scores exist yet: the
released seven-stage pipeline requires five BLIP-2 passes, ImageBind retrieval
and two Llama-2-13B generations per temporal point, and its official command
expects two GPUs. Do not interpret its presence in this table as a completed
full-corpus run.

DSANet's README says it consumes VadCLIP's released features, and the code
bears that out: the two repositories carry byte-identical copies of
`src/clip/`, `src/utils/layers.py` and `src/utils/tools.py`, and DSANet's
`XDDataset` is VadCLIP's with two extra return values. DSANet's model adds a
text adapter, a self-guided normality branch and two extra alignment losses on
top of VadCLIP's dual-branch architecture.

## What upstream expects, and what we feed it

Upstream reads a two-column CSV, `path,label`, whose `path` column points at
one `.npy` per video crop and whose `label` column holds a class string
(`Normal`, `Abuse`, ... for UCF-Crime; `A`, `B1`, `B2`, `B4`, `B5`, `B6`, `G`
for XD-Violence, hyphen-joined for multi-label rows). Each `.npy` is
`(T, 512)` float, **one row per 16-frame snippet**, and `test()` lifts snippet
scores onto the frame grid with `np.repeat(scores, 16)`. The XD training CSV
has five rows per video (`__0` .. `__4`, the five spatial crops); the test CSV
keeps only `__0`.

This study's features are `results/reproduction/features/clip_b16_1fps/<corpus>/<video_id>.npy`,
shape `(T, 512)` float32, one row per second on the 1 fps grid that
`docs/duplex/FRAME_EVAL_PROTOCOL.md` fixes and that the gold arrays in
`results/reproduction/gt/` are rasterised onto. They come from
`openai/clip-vit-base-patch16` `image_embeds`, i.e. the same post-projection
512-d space as OpenAI CLIP's `encode_image`, unnormalised (row norms near 11),
so they sit in the space the frozen text encoder's embeddings live in. Single
crop, no five-crop augmentation.

**The mapping: one feature row = one snippet = one second = one gold frame.**
Two consequences follow, and they are the whole of the temporal adaptation.
The `np.repeat(..., 16)` upsampling is dropped, because the scores already sit
on the gold grid. And every hyperparameter counted in snippets is re-read in
seconds. `smoke_cpu.py` asserts that feature row counts equal gold frame counts
for all 214 + 158 + 153 gold videos in the three corpora; they do, exactly.
**No feature is re-extracted or resampled.**

The CSV is replaced by `hate_common/data.py`, which reads the frozen split
manifests under `results/reproduction/splits/` and the upstream label files
directly.

### Labels and the binary collapse

| corpus | source | mapping |
| --- | --- | --- |
| hatemm | `HateMM_annotation.csv`, `label` column | `Hate` -> 1, `Non Hate` -> 0 |
| mhclip_en / mhclip_zh | `{en,zh}_{train,valid,test}.tsv`, `Majority_Voting` | `Hateful` + `Offensive` -> 1, `Normal` -> 0 |

The MultiHateClip collapse is the one CLAUDE.md fixes. Both corpora therefore
have exactly one anomalous class, and the class-name text branch reduces to two
prompts: **`"normal content"`** and **`"hateful content"`**, in that order.
`dsanet/descriptions.py` carries the full argument; the short version is that
slot 0 must be the normal class because `CLAS2`, `CLASM_BKG`, the text
orthogonality term and the inference formula all read column 0 as normal, and
the two strings are kept minimally different so the contrast the alignment loss
sees is the hateful/normal axis rather than a difference in phrasing.

**Consequence to know before reading DSANet's numbers.** DSANet's headline
test-time score comes from `refine_scores_hierarchical`, which splits
`sigmoid(logits1)` across the non-normal columns of the alignment softmax in
proportion to those columns. With two classes there is one non-normal column,
the proportion is 1, and the refinement returns `sigmoid(logits1)` unchanged.
Under the binary collapse `score_refined` *is* `score_mlp`, verified to 3e-8 in
`smoke_cpu.py` and exactly equal on the real dry run. That is a property of the
collapse, not a porting bug. `dsanet/infer.py` therefore also writes
`score_align`, an alignment-only reading of `logits2` by VadCLIP's own formula,
so the text branch remains inspectable.

## Hyperparameters

Both ports take the **published XD-Violence preset**, not the UCF-Crime one.
XD-Violence is scored as a binary anomaly task with a handful of anomaly
subclasses, which is the shape of this collapse; UCF-Crime is a 13-way
classification benchmark and its preset is tuned for that. Everything below is
verbatim `xd_option.py` except the three marked rows.

| | VadCLIP | DSANet |
| --- | --- | --- |
| embed-dim / visual-width | 512 / 512 | 512 / 512 |
| visual-head / visual-layers | 1 / 1 | 1 / 1 |
| prompt-prefix / prompt-postfix | 10 / 10 | 10 / 10 |
| **classes-num** | **2** (published 7) | **2** (published 7) |
| **visual-length** | **256 hatemm, 64 mhclip** (published 256) | same |
| **attn-window** | **64 hatemm, 16 mhclip** (published 64) | same |
| max-epoch / batch-size | 10 / 96 | 10 / 96 |
| lr | 1e-5 | 1e-5 |
| schedule | MultiStepLR [3, 6, 10], rate 0.1 | CosineAnnealingLR + warm-cosine on the refiner (warmup 100) |
| loss3 weight | 1e-4 | 1.0 |
| loss2 weight | 1.0 | 5.0 |
| DNP | -- | on, 16 prototypes, decoder depth 8, normal ratio 0.8 |
| text adapter | -- | `text_adapt_until` 1, `t_w` 0.6 |
| temp | -- | 1.0 |
| seed | 234 | 234 |

**Why `visual-length` moves.** Upstream's 256 snippets is 171 s on XD and 137 s
on UCF. Measured on this study's features:

| corpus | videos | median | p90 | max | > 256 s |
| --- | --- | --- | --- | --- | --- |
| hatemm | 1066 | 108 s | 254 s | 5809 s | 9.9 % |
| mhclip_en | 792 | 36 s | 60 s | 61 s | 0 % |
| mhclip_zh | 814 | 31 s | 52 s | 60 s | 0 % |

256 rows on HateMM leaves 90 % of the corpus untruncated and happens to equal
the published number, so it is kept. On MultiHateClip, whose longest video is
61 s, 256 rows would make four rows in five zero padding -- and upstream passes
`padding_mask=None` into the temporal transformer during training, so those
padded rows are attended to. 64 rows covers every MultiHateClip video whole and
keeps the published 4:1 window ratio, hence `attn-window` 16.

**Why the MIL top-k does not move.** `k = int(length / 16 + 1)` looks like a
snippet-length constant but is a *fraction* of the sequence, the top ~6 %. It
transfers to the 1 fps grid untouched: on a 96 s video it reads the 7 most
anomalous seconds exactly as on a 96-snippet XD video it read the 7 most
anomalous snippets.

**Model selection is the one substantive protocol change.** Both upstream
training scripts evaluate the *test* set after every epoch and keep the
best-test-AP checkpoint. A test-selected baseline is not comparable with a
method that is not test-selected, so this port never opens the test split
during training. It carves a seeded, label-stratified 10 % validation subset out
of the train split and selects on video-level average precision there
(HateMM 766/85, mhclip_en 567/63, mhclip_zh 591/66). `--val-frac 0 --select last`
restores upstream's behaviour. See PATCHES.md patch V3.

**One caveat worth flagging before the first run.** The published batch size of
96 was set for XD-Violence's ~9000 training crops. HateMM has 766 training
videos after the validation carve, so 10 epochs at batch 96 is about 80
optimiser steps at lr 1e-5. That may simply be too few steps for the model to
move. The defaults are left at the published values because that is what
"published default hyperparameters" means, but if the first HateMM run produces
a near-chance AUC with a flat loss curve, the first thing to vary is
`--batch-size 16 --max-epoch 50`, not the architecture.

## Layout

```
scripts/reproduction_baselines/
  clone_upstream.sh          pinned clones into third_party/ + the CLIP checkpoint
  run_all.sh                 six sequential GPU runs, one GPU at a time
  smoke_cpu.py               CPU-only shape/finiteness/plumbing checks
  eval_baseline_scores.py    shared evaluator over frame_eval_common
  train_{vadclip,dsanet}_hatemm.py   launchers (--corpus selects the corpus)
  test_{vadclip,dsanet}_hatemm.py    launchers
  hate_common/               clip/, tools.py, layers.py (vendored); data.py, runtime.py (new)
  vadclip/                   model.py (vendored), option.py, train.py, infer.py
  dsanet/                    model.py + 4 utils (vendored), option.py, train.py, infer.py
```

`third_party/` holds pristine clones and is gitignored. Every difference
between a vendored file and its upstream original is listed in **PATCHES.md**
and carries a `PORT PATCH` comment at the point of change.

## Running

Setup, once:

```bash
bash scripts/reproduction_baselines/clone_upstream.sh
CUDA_VISIBLE_DEVICES="" /home/jehc223/venvs/SafetyContradiction/bin/python \
    scripts/reproduction_baselines/smoke_cpu.py
```

Everything, sequentially, on one GPU:

```bash
cd /home/jehc223/Hate-follow-up
setsid nohup bash scripts/reproduction_baselines/run_all.sh \
    > results/reproduction/baselines/run_all.log 2>&1 &
```

One method on one corpus:

```bash
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
$PY scripts/reproduction_baselines/train_vadclip_hatemm.py --corpus hatemm --device cuda
$PY scripts/reproduction_baselines/test_vadclip_hatemm.py  --corpus hatemm --device cuda
$PY scripts/reproduction_baselines/eval_baseline_scores.py --corpus hatemm \
    --scores results/reproduction/baselines/vadclip/hatemm/scores.jsonl \
    --json-out results/reproduction/baselines/vadclip/hatemm/frame_eval.json
```

## Outputs

Under `results/reproduction/baselines/<method>/<corpus>/`:

| file | contents |
| --- | --- |
| `model.pth` | selected checkpoint (`state_dict`) |
| `train_meta.json` | resolved args, train/val id lists, per-epoch loss and val-AP history, selected epoch, class prompts |
| `scores.jsonl` | one object per video: `video_id`, `n_frames`, and each `score_*` branch as a length-`n_frames` list on the 1 fps grid |
| `frame_eval.json` | the evaluator's full result dict, per branch |

## Evaluation

`eval_baseline_scores.py` computes nothing of its own. It hands
`{video_id: (scores, labels)}` to `scripts/duplex/frame_eval_common.evaluate`,
so a baseline number and a method number come out of the same implementation of
the frame grid, the rank ROC-AUC and the step-wise average precision. It reports
pooled ROC-AUC and PR-AUC over every frame of every scored video, the
within-hate macro ROC-AUC (the per-video mean restricted to videos the corpus
labels hateful, since a normal video has an all-negative gold array and no
within-video ranking to score), and the frame counts and positive rate so a
pooled number can be read against its base rate. It also reports any gold video
absent from the score file rather than silently skipping it.

The scored cohort is the gold cohort: the test-split ids whose media was present
when `results/reproduction/gt/` was built (214 of 215 HateMM, 158 of 162
mhclip_en, 153 of 157 mhclip_zh). The gold JSON files record why each of the
others was dropped.

`evaluate_scores(scores, gt, hate_ids)` is importable, so any other method in
the study can be scored the same way without going through a file.

## Environment

`/home/jehc223/venvs/SafetyContradiction` -- python 3.12.3, torch 2.8.0+cu128,
torchvision 0.23.0, numpy 2.2.6, scipy 1.18.0, einops, tqdm, regex.

`ftfy` was installed for this port (CLIP's `simple_tokenizer` imports it; pure
python, pulls in `wcwidth`). Nothing else was added. `pandas` and
`scikit-learn`, which upstream needs for its CSV dataset and its inline metrics,
are **not** required: `hate_common/data.py` reads the manifests with the stdlib
`csv` module, and scoring goes through `frame_eval_common`, which uses scipy.
`visdom` and `apex` appear nowhere in either repository.

The frozen CLIP ViT-B/16 checkpoint (`~/.cache/clip/ViT-B-16.pt`, sha256
`5806e77c...`) is fetched by `clone_upstream.sh`.

---

# MACIL-SD

Modality-Aware Contrastive Instance Learning with Self-Distillation, ACM MM
2022, https://github.com/JustinYuu/MACIL_SD @ `c20943f`. The study's
audio-visual baseline, and the source of its pure-audio row.

MACIL-SD shares no code with VadCLIP or DSANet: it descends from XDVioDet and
RTFM, not from CLIP. It reads I3D RGB and VGGish instead of CLIP, has no text
branch, and is already a binary MIL scorer, so the hateful/normal collapse
touches only the label map. The port reuses `hate_common.data` and
`hate_common.runtime` read-only and is scored by the same
`eval_baseline_scores.py`, so its numbers sit on the same grid as the other two.

The audio-visual model plus its self-distillation partner comes to **0.678M
parameters**, matching the "Ours (full) 0.678M" line in the paper's results
table.

## Setup

```bash
git clone https://github.com/JustinYuu/MACIL_SD.git third_party/MACIL_SD
git -C third_party/MACIL_SD checkout c20943f
```

No checkpoint download: MACIL-SD trains from frozen features and nothing else,
so `clone_upstream.sh` is not extended.

## The temporal alignment, which is the whole of the adaptation

MACIL-SD's `Att_MMIL` concatenates the audio and visual sequences on a new
axis, so it needs **one audio row per visual row, describing the same
instant**. On XD-Violence that is free; the released arrays come pre-paired.
Here they do not.

| | rows | unit | coverage |
| --- | --- | --- | --- |
| I3D | `(n_snippets, 5, 1024)` | 16 frames at 24 fps = 0.666667 s | drops tail frames that do not fill a snippet |
| VGGish | `(T, 128)` | 1 s, row `i` = `[i, i+1)` | the whole waveform |

`T` is the gold length: the arrays in `results/reproduction/gt/` have length
exactly `T` for all 214 + 158 + 153 gold videos. The grids also cover different
spans -- audio outlives visual in 1042 / 790 / 808 of the 1066 / 792 / 814
videos, by at most 5.33 s, 1.67 s and 2.00 s.

**The resolution: train on the I3D snippet grid, resample VGGish onto it, map
the scores back to the second grid at inference.** `--grid snippet`, the
default. `macilsd/align.py` carries the argument and PATCHES.md section A1
carries the precise definitions; in brief:

- **Why the snippet grid.** This study's I3D was extracted at 24 fps with
  16-frame snippets, the same decode rate and snippet length XD-Violence used,
  so a row is 0.666667 s in both places. Every hyperparameter MACIL-SD counts
  in rows keeps the physical meaning it was tuned with and **none has to be
  re-read** -- unlike the VadCLIP and DSANet ports, whose 1 fps features forced
  `--visual-length` and `--attn-window` to move per corpus. Pooling I3D down to
  1 s would instead discard a third of the visual resolution the extraction run
  paid for.
- **Audio up.** Snippet `j`'s audio row is the overlap-length-weighted mean of
  the VGGish rows intersecting `[start_j, end_j)` from `<id>.times.json`, i.e.
  the time-average of the piecewise-constant VGGish signal over the snippet. A
  snippet straddling a second boundary gets both seconds in proportion.
- **Scores back down.** Gold second `i` takes the score of the snippet
  containing its midpoint `i + 0.5`, clamped to the last snippet. The clamp is
  the hold-last rule for the dropped-tail seconds. This replaces upstream's
  `np.repeat(scores, 16)`, which targeted a 24 fps frame grid; here the target
  is 1 fps, the ratio is not an integer, and a lookup is the honest form.
- **The mirror image.** `--grid second` pools I3D onto the 1 fps grid, leaves
  VGGish untouched, and makes the back-map the identity. Read the audio-only
  row against it if you suspect the audio resampling of doing work.

## Five crops

Checked against upstream, because the two ends differ. **Training** uses five
separate samples per video, one per crop, each paired with that video's single
VGGish array (`audio_list[index // 5]`), shuffled so the five crops land in
different batches. **Testing** takes the **crop mean**, not crop `__0`:
`infer.py` loads the five-crop list at `batch_size=5, shuffle=False` so one
batch is one video, and `avce_test` averages the sigmoids over that axis.
`__0` appears only in `list/make_gt.py`, where it iterates videos once while
rasterising ground truth. Both conventions are replicated.

## The audio-only ablation

`--modality audio` trains **upstream's own `Single_Model`**, at **upstream's own
lr/5**, on VGGish alone. That module is not something this port introduces: in
`main.py` it is the uni-modal partner the audio-visual model is distilled from
every epoch. The only change is its input width, 128 instead of 1024.

This is the honest comparator. "MACIL-SD's audio branch alone" is not well
defined -- `a_out` is the output of cross-attention *against the video*, so
removing the video removes the branch -- and a fresh MIL head on VGGish would
confound modality with architecture.

`--modality visual` is the matched visual-only row, the same network on I3D. It
costs nothing extra and without it the audio-only number has only the
audio-visual number to be read against, which confounds "audio is weaker" with
"one modality is weaker".

`--crop-repeat`, default 5 for every modality, keeps the audio-only run's
optimiser-step count equal to the audio-visual run's: the audio branch of the
audio-visual model sees each VGGish array five times per epoch, once per crop.
`--crop-repeat 1` gives the one-item-per-video reading.

## Hyperparameters

The published preset, verbatim. Nothing had to be adapted -- see PATCHES.md
patch O2, and the note above on why `--max-seqlen 200` carries over untouched.

| | value | |
| --- | --- | --- |
| lr | 4e-4 | uni-modal partner at lr/5 |
| batch-size | 128 | |
| max-seqlen | 200 rows | = 133.3 s, as on XD-Violence |
| max-epoch | 50 | |
| hid-dim / ffn-dim | 128 / 128 | |
| nhead / dropout | 4 / 0.1 | |
| num-classes | 1 | already binary; the collapse touches only the label map |
| m (EMA base) | 0.91 | `cosine_scheduler(m, 1, epoch, 50)` |
| lamda_a2b / a2n / cof | 1.5 / 1.5 / 0.1 | ramp `min(lamda, cof * epoch)` |
| optimiser / schedule | Adam, CosineAnnealingLR T_max 60 | 60 against 50 epochs, as published |
| seed | 2333 | |

Two published values look like transcription errors and are not, so they are
flags rather than literals: `--sched-tmax 60` against 50 epochs means the
cosine never reaches its trough, and `--ema-epochs 50` is a literal in the
distillation schedule that does not follow `--max-epoch`.

**Model selection is the one substantive protocol change**, the same one the
other two ports make. `main.py` evaluates the *test* split after every epoch
and keeps the best-test-AP checkpoint. This port never opens the test split
during training; it selects on video-level AP over a seeded, stratified 10 %
carve from train. `--val-frac 0 --select last` restores the last-epoch
behaviour. PATCHES.md patch M7.

**One upstream quirk, reproduced and flagged.** `AVCE_Model.forward` returns
`(..., v_out, a_out)` and both call sites unpack it as
`(audio_rep, visual_rep)`, so each representation reaches the contrastive loss
under the other modality's name while the logits selecting the top-k positions
do not. Kept as published, since the reported 83.40 AP was obtained with it.
`--fix-rep-swap` corrects the pairing; the swap is not cosmetic (the four
InfoNCE terms sum to 9.4406 as published against 3.2192 corrected on a fixed
synthetic batch). PATCHES.md patch M13.

## Running

CPU smoke first -- it touches no GPU:

```bash
CUDA_VISIBLE_DEVICES="" /home/jehc223/venvs/SafetyContradiction/bin/python \
    scripts/reproduction_baselines/smoke_cpu_macilsd.py
```

All nine runs (three modalities x three corpora), sequentially, one GPU:

```bash
cd /home/jehc223/Hate-follow-up
setsid nohup bash scripts/reproduction_baselines/run_all_macilsd.sh \
    > results/reproduction/baselines/run_all_macilsd.log 2>&1 &
```

`run_all_macilsd.sh` is separate from `run_all.sh` on purpose and does not
touch it. Restrict the sweep with `MODALITIES` and `CORPORA`.

One modality on one corpus:

```bash
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
$PY scripts/reproduction_baselines/train_macilsd_hatemm.py \
    --corpus hatemm --modality av --device cuda
$PY scripts/reproduction_baselines/test_macilsd_hatemm.py \
    --corpus hatemm --modality av --device cuda
$PY scripts/reproduction_baselines/eval_baseline_scores.py --corpus hatemm \
    --scores results/reproduction/baselines/macilsd/hatemm/scores.jsonl \
    --json-out results/reproduction/baselines/macilsd/hatemm/frame_eval.json
```

`--modality` selects both the architecture and the output directory, so it must
match between train and test.

## Outputs

Under `results/reproduction/baselines/<method>/<corpus>/`, where `<method>` is
`macilsd`, `macilsd_audio` or `macilsd_visual`. Same four files the other ports
write. The score branches differ:

| modality | branches in `scores.jsonl` |
| --- | --- |
| av | `score_av` (upstream's `pred`, the crop-mean sigmoid of `av_logits`), `score_audio` and `score_visual` (the two per-frame branch probabilities, for inspectability) |
| audio / visual | `score_mil` (upstream's `pred3`) |

`train_meta.json` additionally records `grid`, `row_seconds` and
`n_train_items`, so the alignment a checkpoint was trained under is recoverable
from its own metadata.

---

# EventVAD

Training-Free Event-Aware Video Anomaly Detection, ACM MM 2025,
https://github.com/YihuaJerry/EventVAD @ `25cacd8`, paper arXiv:2504.13092.
The study's training-free event-segmentation baseline. It trains nothing, so
it runs on the **test cohorts only** and there is no train split to carve a
validation set out of; the model-selection question the other three ports
answer does not arise.

Two stages. Stage 1 turns each video into events: CLIP ViT-B/16 and RAFT
optical flow per frame, a dynamic spatiotemporal graph over frames with a
temporal decay, training-free graph attention propagation, then statistical
boundary detection on the propagated features. Stage 2 hands each event's 16
frames to VideoLLaMA2.1-7B-16F once and reads an anomaly score out of the
answer.

## Read DESIGN_EVENTVAD.md first

EventVAD is the one baseline in this study whose released code **cannot be
run**. `src/event_seg/uniseg_processor.py` imports `graph_propagation` from
`graph_operations`, and `graph_operations.py` is a byte-identical duplicate of
`video_processing.py` that defines only `process_video`. The function exists
nowhere in the release, so the pipeline raises `ImportError` before decoding a
frame. Two further gaps follow from the same state: the scoring prompt is the
literal string `"prompt"`, and the RAFT checkpoint path is `/path/raft-things.pth`.
`src/evaluate.py` does not compile.

`DESIGN_EVENTVAD.md` reconstructs the missing propagation module from the
paper's Eq. (5)-(8) and the missing prompt from its Figure 2 and section 3.4,
quotes the sources, and lists every choice that is an inference together with
the flag that reverses it. `PATCHES.md` covers the porting changes.
`smoke_cpu_eventvad.py` re-checks every mechanical claim in both, including
that the three gaps are still present in the pinned clone.

Because the release was never executed, its `config.py` is not a tested preset:
the paper's published values are the defaults, and `--preset upstream` selects
the config literals.

| | paper | `config.py` | default here |
| --- | --- | --- | --- |
| α, semantic-motion fusion | 0.75 | 0.8 | **0.75** |
| γ, time decay | 0.6 | 0.05 | **0.6** |
| CLIP L2-normalised in the node feature | yes | no | **yes** |
| moving average | centred | trailing | **trailing** |
| Savitzky-Golay width / MAD multiplier | 60 @ 30 fps / 3 | 2.0 s / 3.0 | 2.0 s / 3.0 |
| GAT iterations / projection dim | 1 / 64 | 1 / 64 | 1 / 64 |

The moving average is the one row where the code wins over the paper, and the
reason is measured rather than argued: a centred window of width w cannot
detect a change whose smoothed width is also w, because it averages over the
peak it is being compared against. On the synthetic in the selftest the centred
ratio reaches 1.194 against a threshold of 1.281 and detects nothing, while the
trailing window reaches 1.602 against 1.577 and fires. `--ma-mode centered`
keeps the claim checkable.

## Setup

```bash
bash scripts/reproduction_baselines/clone_upstream.sh
```

now also clones EventVAD, `princeton-vl/RAFT` and `DAMO-NLP-SG/VideoLLaMA2` at
the commit EventVAD's own requirements pin, fetches `raft-things.pth` from
RAFT's Dropbox release (sha256 `fcfa4125...a7e1`) into
`/home/jehc223/data/checkpoints/raft/`, and downloads VideoLLaMA2.1-7B-16F
(~16 GB) into `/home/jehc223/data/checkpoints/videollama2/`.

**Environment.** The existing `/home/jehc223/venvs/SafetyContradiction` runs
both stages; no second environment. One package was added, `timm`, installed
`--no-deps` so torch and torchvision are untouched -- VideoLLaMA2's
`stc_connector_v35` projector needs `RegStage`. Upstream's other pins are not
needed: `flash-attn` is replaced by `sdpa` (patch E1), `deepspeed` and
`bitsandbytes` are training-only or quantisation-only, and LAVIS resolves to
the OpenAI CLIP ViT-B/16 this repository already vendors and caches (patch E2,
asserted bit-identical in the smoke test).

## Running

CPU smoke first -- it touches no GPU and takes about a minute:

```bash
CUDA_VISIBLE_DEVICES="" /home/jehc223/venvs/SafetyContradiction/bin/python \
    scripts/reproduction_baselines/smoke_cpu_eventvad.py
```

All three corpora, sequentially, one GPU:

```bash
cd /home/jehc223/Hate-follow-up
setsid nohup bash scripts/reproduction_baselines/run_all_eventvad.sh \
    > results/reproduction/baselines/run_all_eventvad.log 2>&1 &
```

Corpora are ordered shortest-first (`mhclip_zh mhclip_en hatemm`, 4817 s /
5600 s / 29266 s of video) so a configuration problem surfaces in the first
hour rather than the tenth. Restrict with `CORPORA` and `STAGES`; both GPU
stages are resumable and skip video ids already recorded without an error.

One corpus, one stage at a time:

```bash
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
EV=scripts/reproduction_baselines/eventvad
$PY $EV/segment_events.py --corpus hatemm --device cuda        # GPU: CLIP + RAFT
$PY $EV/score_events.py   --corpus hatemm --arm paper           # GPU: VideoLLaMA2 7B
CUDA_VISIBLE_DEVICES="" $PY $EV/rasterize_and_eval.py --corpus hatemm --arm paper
```

The two GPU stages are separate processes on purpose. Stage 1 holds CLIP and
RAFT, about 0.3 GB, and is bound by RAFT's per-frame-pair forward. Stage 2
holds VideoLLaMA2 in fp16, about 16 GB, and is bound by generation. Fusing them
would pin 16 GB through the whole of stage 1 for nothing.

`--arm` defaults to `paper`, the Figure 2 prompt, which is the result to quote.
`no_thinking` is the paper's own Table 5 ablation -- the `#Instruction` line
removed -- and `bounded` states the score range the paper leaves unstated. Both
are second conditions on the same test split and need owner approval.

## Outputs

Under `results/reproduction/baselines/eventvad/<corpus>/`:

| file | contents |
| --- | --- |
| `events.jsonl` | per video: probe, `n_frames`, `decode_fps`, `n_edges`, the `[start, end)` event partition, the raw boundaries, the detector's median/MAD/threshold, per-stage timings |
| `segment_meta.json` | resolved config, cohort size, split ids without gold |
| `event_scores.jsonl` | per video: one record per event with the model's full text, the parsed number, the normalised score, the parse rule and the range rule |
| `score_meta.json` | the exact prompt, model path, attention implementation, decode settings |
| `scores.jsonl` | per video: `video_id`, `n_frames`, `score_event` on the 1 fps grid |
| `frame_eval.json` | the evaluator's result dict, plus `n_events`, `events_per_video_{mean,median,max}`, `n_events_unparsed`, `frac_frames_unparsed` and the parse/range histograms |

Read the number against `frac_frames_unparsed` and `range_rules`. An event
whose answer carried no number scores 0.0 and is counted rather than dropped,
because dropping would change the cohort between arms.

## One thing to know before comparing

EventVAD makes **one MLLM call per event**, not per video, so its inference
cost scales with how finely the segmenter cuts. `events_per_video_median` and
`events_per_video_max` are reported for exactly that reason. As a reproduced
baseline that is fine; it is worth stating that the same profile would not
satisfy this project's own two-calls-per-video cap if EventVAD were being
proposed as a method rather than measured as a comparison.
