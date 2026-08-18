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
