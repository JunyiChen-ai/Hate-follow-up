# MultiHateLoc reimplementation: what the paper fixes and what we inferred

MultiHateLoc (WWW 2026, arXiv 2512.10408) is the closest published competitor
to this study: weakly-supervised frame-level hate localization, trained on
video-level labels, evaluated on HateMM and MultiHateClip. It reports HateMM
frame mAP 0.645 and AUC 0.799.

**There is no reference implementation.** The repository the paper announces,
`github.com/mmilabuk/multihateloc`, contains a LICENSE file and nothing else
(single commit, 2026-01-27; verified 2026-08-18). Everything below was
reimplemented from the paper text. Emails to the authors are a separate track;
nothing here waits on them.

This document is the honesty ledger for that reimplementation. Section 1 lists
what the paper states and we implement as stated. Section 2 lists every place
the paper is silent and we had to choose, with the choice and the reason.
Section 3 lists the protocol differences between our evaluation and theirs,
which are the reason our numbers are not comparable to the numbers in their
tables. Nothing in Section 2 or 3 is hidden inside the code: each choice
carries an `INFERRED` comment at the line that makes it.

## 1. Implemented as the paper states

| element | paper | here |
| --- | --- | --- |
| output | per-frame hate probabilities, sigmoid, one per frame of T | same |
| visual features | ViT-B/16, 768-d per frame, cited to Dosovitskiy (ImageNet, not CLIP) | `results/reproduction/features/vit_b16_imagenet_1fps`, `google/vit-base-patch16-224` CLS token, 768-d |
| audio features | VGGish, 128-d per 1-second clip | `results/reproduction/features/vggish_1s`, 128-d per second |
| text features | Whisper transcript, sentence fragments with timestamps, BERT 768-d per fragment, repeat-padded over the fragment's interval | `results/reproduction/features/bert_sentence_1fps`, built by `extract_bert_sentence_features.py` |
| streams | one branch per modality plus a fused branch, each emitting frame probabilities | `model.MultiHateLoc`: three `ModalityBranch` plus a fused head |
| MIL | top-K where K is a proportion; their Table 4 best is K = 3, the top 33 % of frames | `topk_counts` returns `ceil(T / 3)`, floor of one frame |
| MIL loss | binary cross-entropy of the top-K mean against the video label | `MultiHateLoc.mil_loss` |
| smoothness | temporal smoothness regulariser, lambda = 0.1 | `smoothness_loss`, mean squared first difference |
| contrastive | cross-modal contrastive loss, lambda = 0.2 | `contrastive_loss`, InfoNCE |
| modality weighting | a Dynamic Modality Selection block producing importance weights | `DynamicModalitySelection` |
| final frames | union of the fused branch's top-K frames with the modality-specific top-K frames, weighted by those importance weights | `union_frames`, reported as `score_union` |
| optimizer | Adam, learning rate 1e-4 | `torch.optim.Adam`, default `--lr 1e-4` |
| batch size | 32 | default `--batch-size 32` |
| epochs | 100 | default `--max-epoch 100` |

## 2. Inferred choices

Ordered from the ones most likely to move a number to the ones least likely.

### 2.1 Frame rate: frozen at 1 fps

The paper never states its frame rate. It says only that T is "the number of
frames". Its evaluation grid and its span-to-frame rasterization rule are
likewise unstated, so the published 0.645 mAP / 0.799 AUC cannot be
reconstructed from the paper alone
(`docs/duplex/LOCALIZATION_PROTOCOL_SURVEY.md` records the same gap for LELA).

We freeze **1 fps**, this study's gold grid
(`docs/duplex/FRAME_EVAL_PROTOCOL.md`): frame i covers second i, and row i of
every feature matrix is frame i of the frozen gold array by construction. The
choice is not neutral — a denser grid would change T, change `ceil(T/3)`, and
change what a frame-level AUC means — but it is the only grid on which this
reimplementation can be scored against VadCLIP, DSANet and this study's own
methods with one evaluator and no per-video crop.

### 2.2 Audio interpolation is the identity here

The paper linearly interpolates VGGish's 1-second vectors up to T. Our T is
one row per second, so VGGish's native grid already **is** the frame grid and
the interpolation has nothing to do. No resampling is performed. This is a
consequence of choice 2.1, not a departure from the method: on any grid
denser than 1 fps the interpolation would be live.

### 2.3 BERT checkpoint

The paper says "BERT", 768-d, without naming a checkpoint. We use
`bert-base-uncased` for HateMM, MultiHateClip-EN and HateClipSeg, and
`bert-base-chinese` for MultiHateClip-ZH. Both are 768-d, which is what the
stated dimensionality requires. HateMM carries a minority of non-English speech; it still goes
through the uncased English model, because HateMM is an English corpus and
switching checkpoints per utterance would add a component the paper does not
describe.

### 2.4 Sentence vector and the fragment-to-frame rule

- Sentence vector: last-hidden-state CLS token, pooler head not built. The
  paper says only "768-d per sentence".
- A fragment covering `[start, end)` is written to every frame whose second
  overlaps that interval. Where two fragments overlap one second, the fragment
  with the larger overlap wins; ties go to the earlier fragment.
- Frames no fragment covers get a **zero vector**. The paper does not say what
  happens during silence, and zero is the only value that adds no information.
  Measured coverage: mean 0.810 of frames on HateMM, 0.854 on
  MultiHateClip-EN, 0.860 on MultiHateClip-ZH; 59 / 40 / 32 videos have no ASR
  fragment at all and are therefore all-zero in the text modality.
- Whisper size: the paper does not state one. We do not rerun ASR; we consume
  the transcripts already frozen for this study (whisper-large-v3), so the
  text branch sees the identical transcription every other component sees.

### 2.5 Branch architecture

The paper names the branches and gives no widths, depths or activations. Each
branch is `LayerNorm -> Linear(d, 256) -> ReLU -> Dropout(0.1) -> Linear(256,
128) -> ReLU`, with a `Linear(128, 1)` frame head. The fused branch takes the
concatenated 384-d frame embedding through the same shape. Total 0.67 M
parameters, inside the 0.3 M to 20 M range every weakly-supervised localizer
in `docs/duplex/BASELINE_REPRODUCTION_LIST.md` occupies.

Two sub-choices inside this:

- **Input LayerNorm.** The paper states no input normalisation. The three
  feature families arrive on incompatible scales — on HateMM the mean row norm
  is 25.4 for ViT-B/16 and 2.9 for VGGish — so an unnormalised concatenation
  would give vision a ninefold scale advantage in the fused branch for reasons
  that have nothing to do with what the modalities carry. LayerNorm removes
  the scale difference and adds no capacity.
- **No temporal mixing inside a branch.** The branch is applied frame by
  frame. The paper describes per-frame prediction and names no temporal
  encoder; its only temporal coupling is the smoothness regulariser. Adding
  self-attention or a temporal convolution would be adding a component the
  paper does not have.

### 2.6 Dynamic Modality Selection: exact form, and where it enters training

The paper names the block and uses its output in the final frame selection,
but gives no equation. Ours: pool each modality's frame embeddings over the
valid frames, score each pooled vector with **one shared** two-layer head
(`Linear(128, 64) -> Tanh -> Linear(64, 1)`), softmax the three scores. Shared
rather than per-modality parameters, so the block compares modalities on one
scale instead of learning three independent biases.

Weights used only at inference would receive no gradient and would sit at
their initialisation forever. So the same weights also scale each modality's
contribution to the fused branch — the reading under which "dynamic modality
selection" actually selects something. They are multiplied by the modality
count, so a uniform weighting reproduces plain concatenation exactly.

**This is the single largest inference in the reimplementation.** A different
reading of where the weights act would give a different model.

### 2.7 Which branches the losses cover

The paper states one MIL objective and does not say whether the modality
branches are supervised. They must be: the paper reads a top-K set off each
modality branch, and without a video-level loss of its own a branch's frame
probabilities are unconstrained. All four branches get the same BCE and the
four terms are summed. Smoothness is likewise applied to all four and averaged
over them, so the published lambda = 0.1 keeps its meaning with four streams
contributing.

### 2.8 Contrastive pairing

The paper names a cross-modal contrastive loss and does not state what is
contrasted against what. Ours is the standard reading: for each unordered pair
of modalities, the two video-level pooled embeddings of the *same* video are
the positive pair, the other videos in the batch supply the negatives,
symmetric in both directions, temperature 0.07, averaged over the three pairs.
Label-agnostic, as a cross-modal alignment term should be. Temperature is our
choice; the paper gives none.

### 2.9 The union rule, and why we also report a continuous reading

The paper's final output is a frame **set**: the union of the fused branch's
top-K frames with the modality-specific top-K frames weighted by the
importance weights. A positive scalar cannot change a top-K set within one
modality, so the weights can only decide *which* modality sets join the union.
We include modality m's set when its weight is at least uniform, 1/3 — the
reading with no free threshold in it.

That set is a binary array, so its ranking metrics collapse to a single
operating point and it will always look weak next to a continuous score under
ROC-AUC. We report it as `score_union` because it is what the paper describes,
and we additionally report `score_dms`, the importance-weighted convex
combination of the three modality probabilities, which keeps the ranking the
union discards. `score_dms` is **our** continuous reading, not the paper's
output, and is labelled as such wherever it appears.

The primary branch for every headline comparison is `score_fused`, the fused
branch's frame probability.

### 2.10 Variable-length batching

The paper is silent. Each batch is padded to its longest video and carries a
boolean mask; every pooling, top-K, smoothness and loss term reads the mask,
so a padded row never contributes. Nothing is truncated and nothing is
uniformly averaged down, so a score row maps to one second at both train and
test time. (This differs from the VadCLIP / DSANet ports, which inherit
upstream's `process_feat` uniform-averaging on the training side; see
PATCHES.md, "Deliberately not patched".)

### 2.11 Numerical details the paper does not fix

Dropout 0.1; no weight decay; no learning-rate schedule; no gradient clipping;
PyTorch default initialisation; seed 234 (the seed the other ports in this
directory use). Probabilities are clamped to `[1e-7, 1 - 1e-7]` before the BCE.

## 3. Protocol differences from the paper's own evaluation

These are why the numbers in `docs/duplex/BASELINE_RESULTS.md` must not be read
against the 0.645 / 0.799 in the paper.

1. **Frame grid.** Ours is 1 fps and frozen; theirs is unstated. Frame-level
   AUC and mAP are grid-dependent.
2. **Span-to-frame gold.** Ours is the frozen rasterization in
   `docs/duplex/FRAME_EVAL_PROTOCOL.md`; theirs is unstated.
3. **Splits.** Ours are the frozen splits in `results/reproduction/splits/`;
   theirs are not published.
4. **Model selection.** We never open the test split during training: a
   seeded, label-stratified 10 % of the train split is held out and the epoch
   is selected on its video-level average precision (the same rule as patch V3
   for the other ports). The paper does not state how it selects. The
   published 100-epoch budget is kept in full; selection only decides which of
   those 100 epochs is scored.
5. **Cohort.** We score the gold cohort — 214 HateMM, 158 MultiHateClip-EN,
   153 MultiHateClip-ZH videos — which is what makes the row comparable with
   the other baselines in that table, and is not necessarily their cohort.
6. **Label collapse.** MultiHateClip `Majority_Voting` collapses as
   `Hateful + Offensive` versus `Normal`, per CLAUDE.md. The paper's own
   description of MultiHateClip is inaccurate (`LOCALIZATION_LANDSCAPE.md`
   records it claiming an even class split), so its collapse rule cannot be
   read off the text.
7. **Metric.** We report pooled frame ROC-AUC / PR-AUC, a within-hate macro
   AUC and a max-pooled video AUC through
   `scripts/duplex/frame_eval_common.py`, the one evaluator every method in
   this study is scored by. The paper reports frame mAP and AUC.

## 4. Files

| path | role |
| --- | --- |
| `extract_bert_sentence_features.py` | Whisper fragments to BERT, repeat-padded onto the 1 fps grid (GPU, ~13 s for all 2672 videos) |
| `data.py` | three-modality dataset, pad-and-mask collate |
| `model.py` | the model, the four losses, the union and DMS readouts |
| `train.py` | training, validation selection, `scores.jsonl` |
| `video_auc.py` | max-pooled video-level AUC column |
| `smoke_cpu.py` | CPU checks: feature shapes, grid alignment, masking, top-K, loss finiteness |
| `run_all.sh` | three corpora, one GPU job at a time |
