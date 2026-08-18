# Port patches

## Vendoring policy

Both upstream repositories are cloned **pristine** into `third_party/`
(gitignored) by `clone_upstream.sh`, and modified copies of the files this
study needs are vendored under `scripts/reproduction_baselines/`. Nothing under
`third_party/` is edited. Every difference between a vendored file and its
upstream original is listed below and carries a `PORT PATCH (patch <id>)`
comment at the point of change.

| upstream | commit | date |
| --- | --- | --- |
| https://github.com/nwpu-zxr/VadCLIP | `c41067f07d252efcda18008bea367886070c33b0` | 2024-03-10 |
| https://github.com/lessiYin/DSANet | `eb335b23fd6f01810bcd176c948c10348764a504` | 2026-03-26 |

`diff -rq third_party/VadCLIP/src/clip third_party/DSANet/src/clip` is empty
and so are the same comparisons for `utils/layers.py` and `utils/tools.py`:
the two repositories carry byte-identical copies. Those three are therefore
vendored once, under `hate_common/`, rather than twice.

## Vendored file map

| vendored path | upstream origin | state |
| --- | --- | --- |
| `hate_common/clip/` | `VadCLIP/src/clip/` | verbatim |
| `hate_common/tools.py` | `VadCLIP/src/utils/tools.py` | verbatim |
| `hate_common/layers.py` | `VadCLIP/src/utils/layers.py` | patches L1, L2 |
| `vadclip/model.py` | `VadCLIP/src/model.py` | patches V1, V2 |
| `dsanet/model.py` | `DSANet/src/model.py` | patches V1, V2, D2, D4 |
| `dsanet/adapter_modules.py` | `DSANet/src/utils/adapter_modules.py` | patch D1 |
| `dsanet/dnp_vision_transformer.py` | `DSANet/src/utils/dnp_vision_transformer.py` | verbatim |
| `dsanet/StableAdamW.py` | `DSANet/src/utils/StableAdamW.py` | verbatim |
| `dsanet/descriptions.py` | `DSANet/src/utils/descriptions.py` | patch D3 |
| `vadclip/train.py` | `VadCLIP/src/xd_train.py` | patches V3, V4, V5, T1 |
| `vadclip/infer.py` | `VadCLIP/src/xd_test.py` | patches V6, V7 |
| `vadclip/option.py` | `VadCLIP/src/xd_option.py` | patch O1 |
| `dsanet/train.py` | `DSANet/src/xd_train.py` | patches V3, V4, V5, T1 |
| `dsanet/infer.py` | `DSANet/src/xd_test.py` | patches V6, V7 |
| `dsanet/option.py` | `DSANet/src/xd_option.py` | patch O1 |
| `hate_common/data.py` | -- | new |
| `hate_common/runtime.py` | -- | new (CLAS2 / CLASM verbatim from either `xd_train.py`) |
| `eval_baseline_scores.py` | -- | new |
| `smoke_cpu.py` | -- | new |

Files under `third_party/*/src/utils/` that this port does not use at all:
`xd_detectionMAP.py`, `ucf_detectionMAP.py`, `lr_warmup.py`, `crop.py`,
`ucf_train.py`, `ucf_test.py`, `ucf_option.py`, `dataset.py`. The detection-mAP
modules score temporal-action-localisation segments against XD/UCF's own
`gt_segment.npy`; this study scores per-frame arrays through
`scripts/duplex/frame_eval_common.py`, so they have no role.

## Compatibility patches (torch 2.8, no visdom, no apex)

Neither repository imports visdom or apex, so nothing had to be removed on that
account. Three things did break under torch 2.8 / numpy 2.2.

**L1 -- `DistanceAdj` hard-codes CUDA.** `hate_common/layers.py`. Upstream calls
`.to('cuda')` twice inside `forward`, so the module cannot run on CPU at all,
and it recomputes the `scipy.spatial.distance.pdist` matrix on every forward
pass. The port takes the device from the module's own parameter and caches the
matrix per `(device, max_seqlen)`. The returned tensor is numerically identical.

**L2 -- `GraphAttentionLayer` uses removed APIs.** `hate_common/layers.py`.
`nn.init.xavier_uniform` (no trailing underscore) and `torch.cuda.FloatTensor`
are gone in torch 2.x, so importing the module raised. The class is dead code
in both repositories -- nothing constructs it -- so the initialiser is rewritten
to the modern equivalents purely so the import succeeds.

**D1 -- `import ipdb`.** `dsanet/adapter_modules.py`. Upstream imports a
debugger that is not installed and not used in the file. Import removed.

## Task patches (XD/UCF anomaly classes to binary hate)

**D2 -- description-table dispatch.** `dsanet/model.py`,
`DSANet.get_text_features`. Upstream selects between `DESCRIPTIONS_ORI` (14 UCF
classes) and `DESCRIPTIONS_ORI_XD` (7 XD classes) by `len(text) == 14`, and
ignores the prompt list its caller passes. This port has one table, so the
dispatch is deleted. The `text` argument stays in the signature -- callers still
pass it -- and stays unused, as upstream.

**D3 -- binary class prompts.** `dsanet/descriptions.py`. New
`DESCRIPTIONS_HATE` with two entries, in this order:

```python
DESCRIPTIONS_HATE = {
    "normal":  ["normal content"],
    "hateful": ["hateful content"],
}
```

The full argument for the wording and the ordering is in that file's docstring.
In short: HateMM is binary at the video level and MultiHateClip's three-way
`Majority_Voting` collapses to `Hateful + Offensive` versus `Normal` per
CLAUDE.md, so there is exactly one anomalous class; slot 0 must be the normal
class because `CLAS2`, `CLASM_BKG`, the orthogonality term and the inference
formula all read column 0 as normal; and the two strings stay minimally
different so the axis the alignment loss sees is hateful-versus-normal rather
than a difference in phrasing.

**T1 -- text-orthogonality normaliser.** `hate_common/runtime.py`. Upstream
divides the accumulated cosine by the literal `6`, the XD anomaly-class count.
The port divides by `num_class - 1`, which is 6 on XD and 1 here. The formula is
otherwise untouched.

**O1 -- option modules.** `vadclip/option.py`, `dsanet/option.py`. Rewritten
from the corresponding `xd_option.py` (the XD preset, not the UCF one: XD is
scored as a binary anomaly task, which is the shape of this collapse). Every
published XD value is preserved; the changed ones are `classes-num` 7 -> 2 and
the per-corpus `visual-length` / `attn-window`, both documented in the module
docstrings and in `hate_common.runtime.default_visual_length`. The CSV-path,
gt-path and model-path arguments are dropped, since this port reads the study's
own manifests.

## Correctness patches

**V3 -- test-set model selection removed.** `*/train.py`. Both `xd_train.py`
files call `test()` after every epoch and keep the checkpoint with the best
**test** AP. A baseline selected on the test set is not comparable with a method
that is not. This port never opens the test split during training: it carves a
seeded, label-stratified validation subset out of the train split
(`hate_common.data.split_train_val`, `--val-frac`, default 0.1) and selects on
video-level average precision there. `--val-frac 0 --select last` restores
upstream's behaviour of simply taking the final epoch.

**V4 -- per-epoch checkpoint reload removed.** `*/train.py`. Upstream reloads
the best-so-far checkpoint from disk at the end of *every* epoch, so an epoch
that fails to improve is discarded along with its optimiser trajectory, turning
the run into a restart search rather than continuous training. The best state is
held in memory here and restored once, after the last epoch.

**V5 -- logging.** `*/train.py`. Upstream logs on `step % 4800 == 0` where
`step` is a local reset to 0 at the top of each iteration and then set to
`i * batch_size`, which fires erratically. Replaced with one summary line per
epoch carrying every loss term, the validation AP and the wall time.

**D4 -- stale text-feature cache.** `dsanet/model.py`. `get_text_features`
memoises the class embeddings the first time it runs with `self.training ==
False` and never invalidates them. The text adapter is trainable, so upstream's
per-epoch evaluation scores every later epoch with the *first* epoch's text
features. `DSANet.train()` is overridden to drop the cache on every train/eval
switch; within a single evaluation pass the cache still does its job.

**V6 -- snippet-to-frame upsampling removed.** `*/infer.py`. Upstream applies
`np.repeat(scores, 16, 0)` because one XD/UCF feature row covers a 16-frame
snippet. This study's features are one row per second, sampled on the same 1 fps
grid the gold spans are rasterised onto, so a length-T score vector already sits
on the gold grid. `smoke_cpu.py` asserts feature rows equal gold frames for all
214 + 158 + 153 gold videos. The features are **not** re-extracted or resampled.

**V7 -- scoring separated from inference.** `*/infer.py`. Upstream's `test()`
computes `sklearn.metrics.roc_auc_score` / `average_precision_score` and the
detection mAP inline. This port writes per-video score arrays to
`scores.jsonl` and does no scoring; `eval_baseline_scores.py` reads that file
and calls `scripts/duplex/frame_eval_common.py`, so baselines and methods are
scored by one implementation. It also removes the `scikit-learn` dependency.

**V2 -- `clip_download_root`.** `*/model.py`. Constructor keyword added so
`clip.load` can be pointed at a shared cache. Default behaviour (`~/.cache/clip`)
is unchanged.

**V1 -- import paths.** `*/model.py`. Upstream runs with `src/` as the working
directory and imports `clip` and `utils.*` as top-level modules. Repointed at
this port's package layout.

## Deliberately not patched

`process_feat`'s `uniform_extract` averages a long video down to
`visual_length` rows rather than truncating. On HateMM's 10 % of videos longer
than 256 s this compresses time, so a score row no longer maps to one second.
That only affects **training** items; inference uses `process_split`, which
chops rather than averages and therefore preserves the one-row-per-second
mapping the gold needs. Upstream behaviour is kept.

The MIL top-k, `k = int(length / 16 + 1)`, is left verbatim. Despite the 16 it
is a *fraction* of the sequence (the top ~6 %), not a count tied to the 16-frame
snippet, so it transfers to the 1 fps grid without adaptation: on a 96 s video
it reads the 7 most anomalous seconds exactly as on a 96-snippet XD video it
read the 7 most anomalous snippets.

Upstream passes `padding_mask=None` into the temporal transformer during
training, so zero-padded rows are attended to. Kept, but it is the reason
`--visual-length` is set per corpus rather than left at 256 everywhere: on
MultiHateClip, whose longest video is 61 s, a 256-row block would be four
fifths padding.

---

# MACIL-SD

Added by the audio-visual arm of the study. Same vendoring policy: the clone in
`third_party/MACIL_SD` is pristine, the working copy lives in
`scripts/reproduction_baselines/macilsd/`, and every difference is listed here
and carries a `PORT PATCH (patch <id>)` comment at the point of change.

| upstream | commit | date |
| --- | --- | --- |
| https://github.com/JustinYuu/MACIL_SD | `c20943fd51ea7b0ed23e719f65fdfc82a35be530` | 2022-07-13 |

MACIL-SD shares no code with VadCLIP or DSANet -- it descends from XDVioDet and
RTFM, not from CLIP -- so nothing is vendored into `hate_common/`. It does reuse
`hate_common.data` (labels, splits, the stratified validation carve, the gold
arrays) and `hate_common.runtime` (common CLI flags, seeding, device
resolution, output paths) read-only, and it is scored by the same
`eval_baseline_scores.py`.

`clone_upstream.sh` is not extended, because MACIL-SD needs no checkpoint
download; the clone is
`git clone https://github.com/JustinYuu/MACIL_SD.git third_party/MACIL_SD &&
git -C third_party/MACIL_SD checkout c20943f`.

## Vendored file map

| vendored path | upstream origin | state |
| --- | --- | --- |
| `macilsd/InfoNCE.py` | `MACIL_SD/InfoNCE.py` | verbatim, byte-identical |
| `macilsd/Transformer.py` | `MACIL_SD/Transformer.py` | patch M1 |
| `macilsd/CMA_MIL.py` | `MACIL_SD/CMA_MIL.py` | patches M3, M6 |
| `macilsd/avce_network.py` | `MACIL_SD/avce_network.py` | patches M3, M4, M5, M6 |
| `macilsd/utils.py` | `MACIL_SD/utils.py` | patch M2 |
| `macilsd/option.py` | `MACIL_SD/option.py` | patch O2 |
| `macilsd/train.py` | `MACIL_SD/main.py` + `train.py` | patches M4, M7, M8, M9, M10, M11, M13 |
| `macilsd/infer.py` | `MACIL_SD/test.py` + `infer.py` | patches M4, M12, M14, M15 |
| `macilsd/dataset.py` | `MACIL_SD/avce_dataset.py` | rewritten, see A1--A3 |
| `macilsd/align.py` | -- | new, the alignment design |
| `smoke_cpu_macilsd.py` | -- | new |
| `run_all_macilsd.sh` | -- | new |

Upstream files this port does not use: `tSNE.py` (t-SNE plotting behind an
unreachable `i == 10000` branch), `list/make_list.py` and `list/make_gt.py`
(they build the XD-Violence path lists and rasterise XD's own ground truth),
`list/gt.npy` and `ckpt/macil_sd.pkl` (XD-Violence artefacts).

## A1 -- the temporal alignment

This is the design decision the port turns on, and `macilsd/align.py` carries
the full argument. The short version.

`Att_MMIL.forward` concatenates the audio and visual sequences on a new axis,
so MACIL-SD requires **one audio row per visual row, describing the same
instant**. On XD-Violence that is free: the released RGB and VGGish arrays come
pre-paired and upstream never checks. Here the two feature sets sit on
different grids.

| | rows | unit | coverage |
| --- | --- | --- | --- |
| I3D `i3d_rgb_5crop` | `(n_snippets, 5, 1024)` | 16 frames at 24 fps = 0.666667 s | drops the tail frames that do not fill a whole snippet |
| VGGish `vggish_1s` | `(T, 128)` | 1 s, row `i` = `[i, i+1)` | the whole waveform |

The second grid is the gold grid: the arrays in
`results/reproduction/gt/<corpus>_test.npz` have length exactly `T` for all
214 + 158 + 153 gold videos, asserted in `smoke_cpu_macilsd.py`. The two grids
also cover different spans. Because of the dropped tail, audio outlives visual
in 1042 / 790 / 808 of the 1066 / 792 / 814 videos, by at most **5.33 s**
(hatemm `non_hate_video_149`), 1.67 s and 2.00 s; visual outlives audio in
8 / 0 / 3 videos, by at most 2.67 s.

**Resolved as: train on the I3D snippet grid, resample VGGish onto it, map the
scores back to the second grid at inference.** `--grid snippet`, the default.

1. *The snippet grid is upstream's grid, physically.* This study's I3D
   features were extracted at 24 fps with 16-frame snippets, the same decode
   rate and snippet length XD-Violence used, so one row is 0.666667 s in both
   places. Every hyperparameter MACIL-SD counts in rows keeps the physical
   meaning it was tuned with, and **none has to be re-read** -- unlike the
   VadCLIP and DSANet ports, where the 1 fps CLIP features forced
   `--visual-length` and `--attn-window` to move per corpus. `--max-seqlen 200`
   is 133.3 s here and 133.3 s on XD.
2. *Resampling degrades the coarser signal, not the finer one.* Pooling I3D
   down to 1 s would discard a third of the visual temporal resolution the
   extraction run paid for. Lifting VGGish up to 0.667 s invents nothing and
   loses nothing: a snippet window falls inside one or two second-long VGGish
   windows.
3. *The back-map is a lookup, not an interpolation.* Gold second `i` takes the
   score of the snippet containing its midpoint `i + 0.5`; seconds past the end
   of the visual coverage -- exactly the dropped-tail seconds above -- hold the
   last snippet's score. Both rules are asserted in the smoke test.

Precise definitions, all in `macilsd/align.py`:

*Audio onto the snippet grid.* `resample_intervals` treats VGGish as a
piecewise-constant signal, row `i` constant over `[i, i+1)`, and sets snippet
`j`'s audio row to the **overlap-length-weighted mean** over `[start_j, end_j)`
read from `<id>.times.json`. That is the time-average of the signal over the
snippet, so a snippet straddling a second boundary gets both seconds in
proportion rather than the nearer one whole. A snippet with no overlap at all
holds the nearest audio row; this fires only on the 8 + 0 + 3 videos where
visual outlives audio.

*Scores back onto the gold grid.* `snippet_index_for_seconds` returns
`clip(searchsorted(starts, i + 0.5) - 1, 0, n_snippets - 1)`. The clamp is the
hold-last-snippet rule. This replaces upstream's `np.repeat(pred, 16)`, which
lifted a snippet score onto 16 frames of a 24 fps grid; the target here is 1 fps,
the ratio `1 / 0.666667` is not an integer, and a lookup is the honest form.

*The mirror image.* `--grid second` pools I3D onto the 1 fps grid with the same
overlap-weighted average, leaves VGGish untouched, and makes the back-map the
identity. It is not the default. It is what the audio-only row should be read
against if anyone suspects the audio resampling of doing work, since on that
grid the VGGish rows are consumed exactly as written and the score grid *is*
the gold grid. Verified end to end on CPU.

## A2 -- the five-crop convention

Checked against upstream rather than assumed, because the two ends differ.

**Training: five separate samples per video.** `avce_dataset.Dataset` indexes
the RGB list directly and the audio list as `audio_list[index // 5]`, and
`make_list.py` writes the five `__0` .. `__4` files of a video consecutively.
So the training list is 5N rows, each a single crop paired with that video's one
VGGish array, and `main.py` shuffles over all of them, scattering a video's five
crops across different batches.

**Testing: the crop mean, not crop `__0`.** `infer.py` builds the same
five-row-per-video list with `batch_size=5, shuffle=False`, so one batch is
exactly one video's five crops, and `avce_test` does
`torch.mean(torch.sigmoid(av_logits), 0)` over that batch axis. `__0` appears
only in `list/make_gt.py`, where it is used to iterate videos once while
rasterising the ground truth -- not as a test-time crop choice.

Both are replicated. The only difference is mechanical: this study's crops live
in one `(n_snippets, 5, 1024)` array per video rather than five files, so
`index // 5` becomes `index // 5` for the video and `index % 5` for a crop-axis
slice, and the test batch of five becomes a five-row stack built in the dataset.
The crop order recorded in `times.json` is top_left, top_right, bottom_left,
bottom_right, centre; upstream never names its crops, so the correspondence is
positional either way.

## A3 -- dataset rewrite

`macilsd/dataset.py` replaces `avce_dataset.py`. Upstream reads two parallel
text files of `.npy` paths and takes the video label from whether the substring
`_label_A` occurs in the path -- both XD-Violence conventions. This port reads
the study's frozen split manifests and label files through `hate_common.data`,
with the same binary collapse the other two ports use (HateMM `Hate` -> 1;
MultiHateClip `Hateful` + `Offensive` -> 1 per CLAUDE.md).

`upstream.process_feat(..., is_random=False)` is used unchanged for training
items -- uniform subsample when longer than `--max-seqlen`, zero-pad when
shorter -- and test items are fed raw and unchunked, as upstream feeds them. The
longest video in any gold cohort is 1499 snippets (hatemm), so the quadratic
attention over a full test sequence stays small and no chunking guard is needed.

The audio array is resampled once per video at dataset construction and shared
across that video's five crops; the visual side is memory-mapped and sliced per
crop, so a training item reads a fifth of the bytes.

## Compatibility patches (torch 2.8, no visdom, no apex, cpu-runnable)

Neither visdom nor apex appears in the repository. Four things had to change.

**M1 -- `attention` hard-codes CUDA.** `macilsd/Transformer.py`. The local mask
is allocated with `torch.ones(scores.size()).cuda()`. Device taken from the
scores tensor instead. The branch is dead in every published configuration --
`masksize` is never moved off its default 1 -- so this is purely so the file
runs device-agnostically.

**M2 -- `Prepare_logger` dropped.** `macilsd/utils.py`. It opens a file handler
under a relative `log/` path, which fails unless the process happens to be
cwd'd into the clone. Replaced by stdout logging (patch M8). Every other
function in the module is byte-identical.

**M3 -- `.cuda()` in the loss and the MIL head.** `macilsd/CMA_MIL.py`,
`macilsd/avce_network.py`. `CMAL` allocates six accumulators and both `clas`
methods allocate one, all with `.cuda()`, so nothing in the model can forward on
cpu. Device taken from the incoming tensors. Values unchanged.

**M6 -- import paths.** `macilsd/CMA_MIL.py`, `macilsd/avce_network.py`.
Upstream runs with the repository root as the working directory and imports
`InfoNCE` and `Transformer` as top-level modules. Repointed at this port's
package layout.

## Correctness patches

**M4 -- batch-of-one squeeze.** `macilsd/avce_network.py`, `macilsd/train.py`,
`macilsd/infer.py`. Upstream writes a bare `squeeze()` on the per-frame logits
in both `clas` methods, on three tensors in `avce_train`, and on `av_logits` in
`avce_test`. For a batch of two or more this only removes the trailing
`num_classes == 1` axis and is correct. For a batch of exactly one it also
removes the batch axis, `logits[i]` becomes a scalar, and `torch.topk` raises.
Upstream trains with `drop_last` left at its default `False`, so any corpus
whose item count is `1 mod batch_size` would hit it. Changed to `squeeze(-1)` on
the logits and `reshape(-1)` on the bag score, which is identical for every
batch size of two or more. In `infer.py` the same fix is load bearing for the
audio-only path, which forwards a single crop by design.

**M7 -- test-set model selection removed.** `macilsd/train.py`. `main.py` calls
`avce_test` on the **test** loader after every epoch and keeps the checkpoint
with the best test AP, so the published number is test-selected. A
test-selected baseline is not comparable with a method that is not. This port
never opens the test split during training: it carves a seeded,
label-stratified 10 % validation subset out of the train split
(`hate_common.data.split_train_val`, `--val-frac`, default 0.1) and selects on
video-level average precision of the MIL bag score there. `--val-frac 0
--select last` takes the final epoch instead. Identical rule to patch V3.

**M10 -- pre-training-loop test call removed.** `macilsd/train.py`. `main.py`
runs `test()` once before epoch 0 to log the random-initialisation AP. It reads
the test split, so it is dropped.

**M12 -- t-SNE import and its dead branch removed.** `macilsd/infer.py`.
`test.py` imports `batch_tsne` and guards it with `if i == 10000:`, where `i` is
a batch index over a few hundred videos. Unreachable, and the import pulls in
matplotlib and scikit-learn.

**M14 -- snippet-to-frame upsampling replaced.** `macilsd/infer.py`. See A1.

**M15 -- scoring separated from inference.** `macilsd/infer.py`. `avce_test`
computes `precision_recall_curve` inline and returns an AP. This port writes
per-video score arrays to `scores.jsonl` and does no scoring;
`eval_baseline_scores.py` reads that file and calls
`scripts/duplex/frame_eval_common.py`, so baselines and methods go through one
evaluator. Also removes the scikit-learn dependency. Identical rule to patch V7.

**M8 -- logging.** `macilsd/train.py`. Upstream logs twice per epoch through
the dropped file logger. Replaced with one summary line per epoch carrying
every loss term, both lambda ramps, the EMA rate, the validation AP and the
wall time.

**M9 -- GPU environment assignment removed.** `macilsd/train.py`.
`torch.multiprocessing.set_start_method('spawn')` and
`os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus` are gone; the device comes
from `--device` through `hate_common.runtime.resolve_device`.

## M11 -- the uni-modal ablations, including the audio-only row

`--modality audio` and `--modality visual`. Both train **upstream's own
`Single_Model`**, at **upstream's own lr/5**, on one modality alone.

This is what makes the audio-only row an honest comparator rather than a new
architecture. `Single_Model` is not something this port introduces: in
`main.py` it is the uni-modal partner that the audio-visual model is distilled
from every epoch, and it is trained there with `Adam(lr / 5)` and the same
`CosineAnnealingLR(T_max=60)`. `--modality audio` builds exactly that module
with its input width set to VGGish's 128 instead of I3D's 1024, feeds it the
audio, and changes nothing else. Nothing is added and nothing is tuned.

The alternative readings were rejected. "MACIL-SD's audio branch alone" is not
well defined: `a_out` is the output of cross-attention *against the video*, so
deleting the video deletes the branch. A fresh MIL head on VGGish would be a
new model, and any difference from the audio-visual row would then confound
modality with architecture.

`--modality visual` is the matched visual-only row. It costs nothing extra --
upstream trains this network anyway -- and without it the audio-only number has
only the audio-visual number to be read against, which confounds "audio is
weaker" with "one modality is weaker".

**The five-crop count for audio-only.** `--crop-repeat`, default 5 for every
modality. The audio branch of the audio-visual model sees each video's VGGish
array five times per epoch, once per crop, so an audio-only comparator visiting
it once per epoch would differ from the branch it is meant to be compared
against by a factor of five in optimiser steps rather than by modality. Setting
it to 5 matches the step count exactly; `--crop-repeat 1` gives the
one-item-per-video reading. Neither is obviously the only right answer, so the
flag is explicit and the value lands in `train_meta.json`.

At inference the audio-only model forwards one crop rather than five: all five
carry the same VGGish array, so their sigmoids are identical and the crop mean
is that value.

## M13 -- an upstream quirk, reproduced and flagged

`AVCE_Model.forward` returns `(..., v_out, a_out)`, and both call sites unpack
that pair as `(audio_rep, visual_rep)`:

```python
mmil_logits, audio_logits, visual_logits, _, audio_rep, visual_rep = model_av(f_a, f_v, seq_len)
```

so `audio_rep` is the **visual** representation and `visual_rep` is the
**audio** one. The logits that select the top-k positions inside `CMAL` are not
swapped, so the contrastive loss indexes one modality's representation with the
other modality's chosen frames.

This is left exactly as published, because the reported 83.40 AP was obtained
with it, and reproducing a paper means reproducing what it ran.
`--fix-rep-swap` pairs each representation with its own logits. The swap is not
cosmetic: on a synthetic batch with both self-guided banks populated the four
InfoNCE terms sum to 9.4406 as published and 3.2192 corrected, checked in
`smoke_cpu_macilsd.py`.

## O2 -- option module

`macilsd/option.py`, rewritten from `MACIL_SD/option.py`. Every published value
is kept verbatim: `lr 4e-4`, `batch-size 128`, `max_seqlen 200`, `max-epoch 50`,
`m 0.91`, `lamda_a2b 1.5`, `lamda_a2n 1.5`, `lamda_cof 0.1`, `hid_dim 128`,
`ffn_dim 128`, `nhead 4`, `dropout 0.1`, `num_classes 1`, `a_feature_size 128`,
`v_feature_size 1024`, seed 2333, the lr/5 partner rate, `T_max 60`, and the
literal 50 in the EMA schedule.

**Nothing in this preset had to be adapted**, which is worth stating because it
is unlike the other two ports. `--visual-length` and `--attn-window` had to move
per corpus for VadCLIP and DSANet because their 1 fps CLIP features changed what
a row means; MACIL-SD's rows are 0.666667 s here exactly as on XD-Violence, so
`--max-seqlen 200` carries over untouched. `num_classes` did not have to move
either: MACIL-SD is already a binary MIL scorer, so the hateful/normal collapse
touches only the label map.

Two published values look like transcription errors and are not, so they are
exposed as flags rather than buried:

- `--sched-tmax 60` against `--max-epoch 50`. `CosineAnnealingLR(T_max=60)` over
  50 epochs never reaches its trough; the run ends at `0.033 * lr`.
- `--ema-epochs 50` against `--max-epoch 50`. The mixing rate comes from
  `cosine_scheduler(m, 1, epoch, 50)` with the 50 written as a literal,
  independent of `--max-epoch`. Changing the epoch budget without this flag
  would silently reshape the distillation schedule.

Dropped: the six XD-Violence path arguments (`--rgb-list`, `--audio-list`, the
two test lists, `--gt`, `--model_dir`) and `--gpus`, replaced by this study's
manifests and `--device`; `--num_stages 3`, which nothing constructs;
`--pretrained-ckpt`, unused; `--dataset-name`, unused. `--workers` becomes
`--num-workers` from `hate_common.runtime`, so all three ports take the same
common flags. `--modality` upstream defaults to `'MIX2'` and is read once in
`avce_dataset.Dataset.__init__` and then never tested by any branch; the name is
reused here for the live choice between the audio-visual model and the two
uni-modal ablations.

## Deliberately not patched

`process_feat`'s `uniform_extract` subsamples a long video down to
`max_seqlen` rows rather than truncating, which compresses time on the 9.9 % of
HateMM videos longer than 133 s. That affects **training** items only;
inference feeds the raw sequence, so the one-row-per-snippet mapping the gold
needs is preserved. Upstream behaviour kept, same call as patch V-not-patched
in the VadCLIP port.

The MIL top-k, `int(seq_len // 16 + 1)`, is left verbatim. Despite the 16 it is
a *fraction* of the sequence, the top ~6 %, not a count tied to the 16-frame
snippet, and here it reads the same fraction of the same physical window as it
did on XD.

`CMAL` selects its abnormal and normal banks by the model's own bag score
(`mmil_logits[i] > 0.5`), not by the video label, and returns four literal
zeros when either bank is empty. Both are upstream behaviour and both are kept;
the four-zero short circuit is asserted in the smoke test so a zero CMA column
in a training log is not mistaken for a bug.

`avce_train`'s `model_av.requires_grad = True` / `model_uni.requires_grad =
False` assignments set a plain attribute on the Module rather than on its
parameters and therefore do nothing. Kept verbatim: the `zero_grad` pair before
each backward is what actually separates the two graphs, and the two losses
share no parameters in any case.

Upstream applies no padding mask in the temporal transformer. Kept. It matters
less here than in the other two ports, because `avce_train` truncates each batch
to `max(seq_len)` before the forward, so padding is bounded by the longest real
sequence in the batch rather than by `--max-seqlen`.
