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

---

# EventVAD

Training-Free Event-Aware Video Anomaly Detection, ACM MM 2025,
https://github.com/YihuaJerry/EventVAD @ `25cacd8`, paper arXiv:2504.13092.
The study's training-free event-segmentation baseline, and the second of its
two MLLM baselines beside Vad-R1.

**Read PATCHES.md's sibling, DESIGN_EVENTVAD.md, first.** EventVAD is the one
port in this study whose upstream **cannot be run at all**: the release imports
a `graph_propagation` it never defines, so `main.py` raises `ImportError`
before decoding a frame. The design note records the four gaps in the release,
reconstructs the missing module and the missing prompt from the paper with the
quotes they rest on, and lists every choice that is an inference. This section
records only the porting patches -- the changes made to run the method on this
study's corpora, not the changes made to make it exist.

Nothing under `third_party/` is edited. Unlike the VadCLIP and DSANet ports,
no upstream file is vendored: with the graph propagation missing and the
prompt a placeholder, there is no file worth copying. `eventvad/` is written
against the paper, and each module's docstring names the upstream file it
replaces.

## File map

| path | replaces | state |
| --- | --- | --- |
| `eventvad/config.py` | `src/event_seg/config.py` | rewritten, paper values |
| `eventvad/video_io.py` | `src/event_seg/utils.py` | rewritten, patch E4/E7 |
| `eventvad/features.py` | `src/event_seg/feature_extractor.py` | rewritten, patches E2, E3, E4 |
| `eventvad/graph.py` | `src/event_seg/uniseg_processor.py` + the missing `graph_propagation` | reconstructed, DESIGN G1 |
| `eventvad/boundary.py` | `src/event_seg/boundary_detection.py` | ported, DESIGN G7 |
| `eventvad/prompt.py` | `src/score/event_score.py:23` | reconstructed, DESIGN G2 |
| `eventvad/segment_events.py` | `src/event_seg/main.py` + `video_processing.py` | rewritten, patch E5 |
| `eventvad/score_events.py` | `src/score/event_score.py` | rewritten, patches E1, E6 |
| `eventvad/rasterize_and_eval.py` | `src/evaluate.py` | rewritten, patch E8 |
| `run_all_eventvad.sh`, `smoke_cpu_eventvad.py` | -- | new |

Upstream files with no role here: none -- the release is ten Python files and
every one of them is either replaced above or is the duplicate
`graph_operations.py`.

## Dependency decisions

EventVAD ships two `requirements.txt` files pinning two conda environments,
between them 300-odd packages including torch 2.1/2.2 on CUDA 11.8, a jupyter
stack, spacy, open3d and streamlit. This port adds **one** package to the
existing `/home/jehc223/venvs/SafetyContradiction` and creates no new
environment.

| pinned upstream | needed? | decision |
| --- | --- | --- |
| `torch==2.1.0+cu121` / `2.2.0+cu118` | -- | the venv's torch 2.8.0+cu128 is used; nothing in either stage touches a removed API |
| `flash-attn==2.5.8` | **no** | patch E1: `sdpa` instead |
| `deepspeed==0.13.1` | **no** | training only; `load_pretrained_model` never imports it for inference |
| `bitsandbytes==0.43.0` | **no** | reached only through `load_8bit` / `load_4bit`, both off. `from transformers import BitsAndBytesConfig` resolves without the package |
| `decord==0.6.0` | present | imported by `videollama2.mm_utils` at module scope, so it must import; it is never called, because patch E6 hands `process_video` a decoded array |
| `salesforce-lavis` | **no** | patch E2: resolves to OpenAI CLIP ViT-B/16, which this repo already vendors and caches |
| `timm` | **yes** | `videollama2/model/projector.py` needs `RegStage` for the `stc_connector_v35` projector. Installed with `--no-deps` so torch and torchvision are untouched; timm 1.0.28 still aliases the deprecated `timm.models.layers` path the projector imports |
| `transformers==4.33.2` / `4.40.0` | -- | 4.57.1 is used. `videollama2` imports and builds under it; the checkpoint's stale `transformers_version: 4.40.0` needs no processor fix-up, unlike Vad-R1's |
| `opencv`, `imageio`, `einops`, `sentencepiece`, `accelerate`, `scipy`, `networkx` | present | already in the venv |

`networkx` is present but **not used**: see patch E9.

## E1 -- the SigLIP tower demands flash-attn unconditionally

`videollama2/model/encoder.py` line 96, inside `SiglipVisionTower.__init__`:

```python
config._attn_implementation = 'flash_attention_2'
```

`load_pretrained_model`'s `use_flash_attn` flag defaults to False and governs
only the language model; the vision tower ignores it. flash-attn 2.5.8 does
not build against torch 2.8 / CUDA 12.8 on Blackwell in any reasonable time,
and it is not needed here: SigLIP's encoder is a plain bidirectional
transformer over the patch tokens of 16 frames, with no causal mask and no
sequence long enough for the memory argument to bite, so `sdpa` computes the
same attention. `score_events.patch_siglip_attention` replaces the tower's
`__init__` at import time with a copy of upstream's body, lines 86-100,
differing in that one string. `--attn` selects, so `flash_attention_2` is
still reachable on a machine that has it.

## E2 -- LAVIS is dropped for the CLIP already in this repository

`feature_extractor.py` loads CLIP through
`lavis.models.load_model_and_preprocess(name="clip", model_type="ViT-B-16")`.
LAVIS's `configs/models/clip_vit_base16.yaml` resolves that to
`pretrained: openai` with `vis_processor.eval: clip_image_eval, image_size:
224`, i.e. the OpenAI CLIP ViT-B/16 checkpoint this study already caches at
`~/.cache/clip/ViT-B-16.pt` (sha256 `5806e77c...`, fetched by
`clone_upstream.sh` for VadCLIP and DSANet), preprocessed by

```
Resize(224, BICUBIC) -> CenterCrop(224) -> convert("RGB") -> ToTensor()
    -> Normalize((0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711))
```

which is step for step the transform `hate_common/clip/clip.py:_transform(224)`
builds -- LAVIS's mean and std come from `BlipImageBaseProcessor`'s defaults
and are the same six constants. `smoke_cpu_eventvad.py` rebuilds LAVIS's
transform from those definitions and asserts it produces a **bit-identical**
tensor to the vendored one on a random image, so the substitution is checked
rather than argued. Installing LAVIS would pin an old transformers against the
same venv VideoLLaMA2 runs in.

`Config.fp16_enabled` is False and upstream calls `.float()` on the loaded
model; the port does the same, so CLIP runs in fp32 as published.

## E3 -- the RAFT checkpoint path

Upstream's `'/path/raft-things.pth'` resolved to
`/home/jehc223/data/checkpoints/raft/raft-things.pth`, sha256
`fcfa4125...a7e1`, from `princeton-vl/RAFT`'s own `download_models.sh`.
`clone_upstream.sh` fetches and checksums it. Upstream's import,
`from RAFT.core.raft import RAFT`, does not work as written either -- RAFT's
`core/raft.py` does `from update import ...`, so `core/` has to be on
`sys.path`, which is what `features.py` does.

## E4 -- extraction streams instead of materialising the video

`utils.video_to_frames` returns `np.array(frames)` over **every** decoded
frame. For the longest HateMM test video, 1000 s at 30 fps and 1280x720, that
is 83 GB. The features derived from those frames are (n, 640) float32, 77 MB;
the frames are the only thing that does not fit. CLIP is a per-frame function
and RAFT a per-adjacent-pair function, so `FeatureExtractor.extract` consumes
a frame iterator and interleaves them in one pass, holding one `chunk_size`
buffer. The two feature arrays are the arrays upstream's whole-array code
would have produced.

## E7 -- one ffmpeg decode path, and the frame rate

Upstream decodes with `cv2.VideoCapture`. The OpenCV build here has no AV1
decoder and 2 of 12 sampled MultiHateClip English test videos and 2 of 12
Chinese ones are AV1, with HEVC also present in the Chinese split. Elsewhere in
this study that is handled by an ffmpeg fallback beside an OpenCV main path;
here there is no reason to keep two, because the stage wants a whole decoded
stream rather than indexed frames and the system ffmpeg reads all three codecs.
Pixels are matched to upstream's: `rgb24` is the channel order
`cv2.cvtColor(..., BGR2RGB)` produces, `scale=...:flags=bilinear` is
`cv2.resize`'s default `INTER_LINEAR` rather than ffmpeg's bicubic default, and
`EventVADConfig.output_size` is upstream's cap-then-round-to-even rule
character for character.

**Rate (P2).** Upstream reads every frame; the paper fixes "FPS = 30". These
corpora run at 24, 25, 29.97, 30, 59.94 and 60. `max_fps` **caps** rather than
resamples: a 60 fps file decodes at 30, a 25 fps file stays at 25. Upsampling
25 to 30 would insert duplicated frames, and a duplicate frame has optical flow
exactly zero and cosine dissimilarity exactly zero -- a stretch of perfect
event continuity the video does not contain, fed straight into the boundary
detector.

**Time decay (P3).** `ema_window` and `min_segment_gap` are already written in
seconds upstream and multiplied by fps where they are used, so they carry over
untouched. γ does not: it multiplies a **frame-index** difference, so the
paper's γ = 0.6 at 30 fps is a decay of 18 per second, and reproducing that
decay at rate f needs `γ · 30 / f`. `gamma_mode = per_second` does that;
`per_frame` uses γ literally. This is the same adaptation the VadCLIP and
DSANet ports make when they re-read snippet-counted hyperparameters in seconds.

## E5 -- events are carried as boundaries, not as re-encoded video

`video_processing.process_video` writes every segment to its own
`segment_XXXX.mp4` with `cv2.VideoWriter`, trying `mp4v`, `avc1`, `xvid` in
turn, and hands the paths to the scorer. Three reasons the port writes a JSON
boundary list instead. The re-encode is a lossy generation between the frames
the segmenter measured and the frames the scorer sees. `cv2.VideoWriter`
cannot write the AV1 inputs back out, and its `mp4v` fallback would silently
change the pixels for a large minority of MultiHateClip. And 525 test videos
cut into events would write tens of thousands of files to serve 16 frames
each, which the scorer can seek in the source. The boundaries are the entire
information content of upstream's segment directory.

`events_from_boundaries` reproduces upstream's arithmetic -- consecutive
merged boundaries, a final segment `min_segment_gap * fps` long, a prepended
`(0, first)` and an appended `(last, end)` -- with two departures, both
recorded in the returned diagnostics. Ends are half-open so the events
partition `range(n_frames)` exactly, which is what the rasteriser needs; and
upstream's "drop a segment shorter than two frames" rule can in principle
leave a hole, so any hole is closed by extending the previous event and
counted in `n_gaps_closed`. Merging already forces boundaries
`min_segment_gap * fps` apart, so it should never fire; it is a guarantee, not
a correction.

## E6 -- 16 frames per event, from the source

Upstream calls `processor['video'](segment_path)`, which decodes the segment
file with decord and samples 16 frames from it. With no segment files, the
port applies VideoLLaMA2's own index rule --
`mm_utils.frame_sample(duration, mode='uniform', num_frames=16)`, reproduced
verbatim in `score_events.frame_sample_uniform` and asserted equal to the
upstream function in the selftest -- to the event's frame range, pulls those
absolute indices out of the source, and hands `process_video` the resulting
`np.ndarray`, which its own dispatch accepts. The model therefore sees the
frames upstream's sampler would have picked out of that segment, without the
re-encode. `collect_event_frames` makes **one** decode pass per video and
emits each event's 16 frames as the stream reaches them, so a video with many
events costs one decode, not one per event.

## E8 -- the evaluator

`src/evaluate.py` does not compile (`for line in f:s`, line 44) and targets a
UCF-Crime `tag.txt`. `rasterize_and_eval.py` maps events onto the 1 fps gold
grid and hands the result to `eval_baseline_scores.evaluate_scores`, so
EventVAD's number and every other baseline's come out of one implementation of
`frame_eval_common.evaluate`.

**The mapping.** An event covers decoded frames `[s, e)`, i.e. seconds
`[s/fps, e/fps)`; gold second `i` takes the score of the event containing its
midpoint `i + 0.5`, clamped to the last event. That is the convention the
MACIL-SD port already uses to cross a non-integer grid ratio, and it is a
lookup rather than upstream's `scores[s:e] = score` because `fps` is 29.97 as
often as 30 here.

**Unparsed events (P5).** An event whose text carried no number scores 0.0 and
is counted. Dropping the video instead would change the cohort between prompt
arms and make two arms' pooled numbers incomparable. `frame_eval.json` reports
`n_events_unparsed`, `frac_events_unparsed` and `frac_frames_unparsed`, so a
number can be read against how much of it was filled in.

## E9 -- scipy.sparse instead of networkx

Upstream builds a `networkx.Graph`. Its edge count is not the kNN fan-out it
looks like: the top-k is taken **per block pair**, so a node collects `init_k`
edges against each of the `ceil(n / 200)` column blocks. On a 30 000-frame
video that is `5 x 150 = 750` edges per node, 22.5 M edges, tens of GB in
networkx. The same edges as a CSR matrix are about 270 MB, and the propagation
becomes one sparse-dense product instead of a Python loop over adjacency
lists. The edge set and the weights are unchanged. Upstream's `combined_sim`
is symmetric in (i, j) -- each of its three terms is -- so networkx's
last-write-wins on a repeated unordered pair and this port's de-duplication by
maximum agree by construction, and the smoke test asserts the resulting matrix
is symmetric.

## E10 -- RAFT does not run unpadded, and has a resolution floor

`feature_extractor.extract_flow_features` calls
`self.raft_model(prev_frame, curr_frame, iters=...)` directly, with no
`InputPadder`. RAFT's `forward` sizes its coordinate grid as `H // 8, W // 8`
while its encoder produces `ceil(H / 8), ceil(W / 8)`, so on any side that is
not a multiple of 8 the two disagree and `bilinear_sampler` raises:

    RuntimeError: grid_sampler(): expected grid and input to have same batch
    size, but got input with sizes [6420, 1, 60, 107] and grid with sizes
    [6360, 9, 9, 2]

That is 854x480, which is 108 of the 214 HateMM test videos. RAFT's own
`demo.py` wraps every call in `InputPadder`, which replicate-pads to the next
multiple of 8; `features._mean_flow` does the same and crops the flow back with
`padder.unpad` before averaging, so the mean covers exactly the original frame.
This is not a choice between readings -- the stage cannot run without it.

RAFT also has a **floor**. Its correlation pyramid halves the `H/8` feature map
four times, so a side under about 128 px collapses the coarsest level to width
1, where `bilinear_sampler` normalises by `W - 1` and every output is NaN.
Measured across all 525 gold videos, the smallest decoded side is **144 px**
(`non_hate_video_197`, 176x144), so the corpora clear the floor -- but not by
much, and `smoke_cpu_eventvad.py` asserts it on every video it probes rather
than leaving it to chance.

## Deliberately not patched

`dynamic_k = max(3, init_k - (i // (n // 10)))` shrinks the fan-out from 5 to 3
as the row-block sweep advances, a positional asymmetry the paper describes
nowhere. Kept, because it is what the released code does. The only change is a
guard: `n // 10` is zero for a video under ten frames and upstream divides by
it.

`extract_features` applies `clip_weight` to the CLIP half and then
`build_dynamic_graph` applies it again inside `combined_sim`, so upstream
weights the semantic branch twice in the similarity while the flow branch
enters the distance already scaled by `1 - alpha`. The port follows Eq. (4),
which weights each branch once, and `--preset upstream` does not restore the
double weighting -- that one is a straightforward inconsistency with the
paper's own equation rather than a defensible alternative reading.

The flow branch is two numbers per frame. `f_flow = P^T E[o]` averages the
whole RAFT field to a single 2-vector before lifting it to 128 dimensions
through a matrix with orthonormal rows, which is an isometry -- the smoke test
asserts it. So the 128-dimensional motion branch carries exactly two degrees
of freedom. That is upstream's design and the paper's Eq. (2), kept as is, and
it is worth knowing before reading Table 5's `+1.42` for RAFT alone.
# LAVAD

Added 2026-08-21. Upstream is
https://github.com/lucazanella/lavad at
`1ad46c666d1b3cfb262f3dd84769acf873285056`. The clone remains pristine under
`third_party/lavad`; no upstream source is vendored or edited. The complete
adapter and deviation ledger is `DESIGN_LAVAD.md`. In brief: prepare the
frozen test cohort as one JPEG per gold second, run upstream with
`frame_interval=1`, strictly pack raw/refined JSON scores onto the common grid,
and replace upstream sklearn evaluation with `eval_baseline_scores.py`. The
published anomaly prompt is retained as the primary arm. No full inference or
hate-specific prompt arm has been run.
