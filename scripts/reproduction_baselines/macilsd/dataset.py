"""Corpus plumbing for MACIL-SD, replacing MACIL_SD @ c20943f avce_dataset.py.

Upstream reads two parallel text files of `.npy` paths, one line per RGB crop
and one line per video for VGGish, pairs them with `audio_list[index // 5]`,
and reads the video label from whether the substring `_label_A` occurs in the
path. All three of those are XD-Violence conventions; this port reads the
study's frozen split manifests and label files through `hate_common.data`
instead, and gets the temporal pairing from `macilsd/align.py`.

The five-crop convention, replicated exactly
--------------------------------------------
Upstream trains on **five separate samples per video**, one per spatial crop,
each paired with that video's single VGGish array -- that is what
`audio_list[index // 5]` does, given `make_list.py` writes the five `__0` ..
`__4` files consecutively. `main.py` shuffles over all 5N of them, so the five
crops of one video land in different batches.

At test time upstream builds the same five-row-per-video list, loads it with
`batch_size=5, shuffle=False` so that one batch is exactly one video's five
crops, and takes `torch.mean(torch.sigmoid(av_logits), 0)` -- the **crop mean**,
not crop `__0`. (`__0` appears only in `list/make_gt.py`, to iterate videos once
when rasterising the ground truth.)

This port keeps both. Training items are `(video, crop)` pairs over the same
5N index space, and inference scores all five crops of a video in one forward
and averages the sigmoids. The only difference is mechanical: this study's
crops live in one `(n_snippets, 5, 1024)` array per video rather than five
files, so `index // 5` becomes a crop axis slice.

The audio-only ablation keeps the 5N index space too. See `Modality` below.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.utils.data as data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hate_common import data as hdata          # noqa: E402
from . import align                            # noqa: E402
from .utils import process_feat                # noqa: E402

MODALITIES = ("av", "audio", "visual")


def feature_dim(modality):
    return align.A_DIM if modality == "audio" else align.V_DIM


def usable_ids(corpus, video_ids):
    """Split ids that have both an I3D and a VGGish feature file."""
    return [v for v in video_ids if align.has_features(corpus, v)]


class MacilTrainDataset(data.Dataset):
    """One item per (video, crop). Mirrors upstream's 5N-row training list.

    Returns `(f_v, f_a, label)` where both feature tensors are
    `(max_seqlen, D)` after upstream's `process_feat(..., is_random=False)`:
    uniform subsample when the video is longer, zero-pad when shorter. The
    training loop recovers the true length from the non-zero rows exactly as
    upstream does.

    For `modality="audio"` the visual tensor is still returned -- the training
    loop needs it to derive `seq_len` the upstream way -- but the model never
    sees it. For `modality="visual"` the audio tensor is returned and unused.

    `crop_repeat` controls how many of the five crop slots are visited. It is 5
    for every modality by default, including audio-only. The audio branch of
    the audio-visual model sees each video's VGGish array five times per epoch,
    once per crop, so an audio-only comparator that saw it once per epoch would
    differ from the branch it is meant to be compared against by a factor of
    five in optimiser steps rather than by modality. `--crop-repeat 1` gives the
    one-item-per-video reading instead; the two are recorded separately because
    neither is obviously the only right answer.
    """

    def __init__(self, corpus, video_ids, labels, max_seqlen,
                 grid="snippet", modality="av", crop_repeat=align.N_CROPS):
        if modality not in MODALITIES:
            raise ValueError("modality must be one of %s" % (MODALITIES,))
        if not 1 <= crop_repeat <= align.N_CROPS:
            raise ValueError("crop_repeat must be in [1, %d]" % align.N_CROPS)
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.labels = labels
        self.max_seqlen = int(max_seqlen)
        self.grid = grid
        self.modality = modality
        self.crop_repeat = int(crop_repeat)

        missing = [v for v in self.video_ids if v not in labels]
        if missing:
            raise KeyError("%d ids have no label, e.g. %s"
                           % (len(missing), missing[:5]))
        absent = [v for v in self.video_ids
                  if not align.has_features(corpus, v)]
        if absent:
            raise FileNotFoundError("%d ids lack an I3D or VGGish feature "
                                    "file, e.g. %s" % (len(absent), absent[:5]))

        # Precomputed once per video: the audio is identical across the five
        # crops, so resampling it per item would repeat the same work five
        # times. The visual side stays lazy and per-crop, which is the branch
        # that would actually cost memory.
        self._audio = {}
        self._meta = {}
        for vid in self.video_ids:
            audio, n_seconds, snip = align.aligned_audio(corpus, vid, grid)
            self._audio[vid] = audio
            self._meta[vid] = (n_seconds, snip)

    def __len__(self):
        return len(self.video_ids) * self.crop_repeat

    def __getitem__(self, index):
        vid = self.video_ids[index // self.crop_repeat]
        crop = index % self.crop_repeat
        n_seconds, snip = self._meta[vid]
        f_a = self._audio[vid]
        if self.modality == "audio":
            # Nothing reads the visual features, but seq_len is derived from
            # them upstream, so a same-length placeholder keeps that code path
            # identical. Using the audio itself would make an all-zero VGGish
            # row look like padding; a ones column cannot.
            f_v = np.ones((f_a.shape[0], 1), dtype=np.float32)
        else:
            f_v = align.aligned_visual_crop(self.corpus, vid, crop, self.grid,
                                            n_seconds, snip)
        f_v = process_feat(f_v, self.max_seqlen, is_random=False)
        f_a = process_feat(f_a, self.max_seqlen, is_random=False)
        label = float(self.labels[vid])
        return (torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32)),
                label)


class MacilTestDataset(data.Dataset):
    """One item per video, five crops stacked. Mirrors upstream's batch of 5.

    Returns `(f_v, f_a, snippet_index, n_seconds, video_id)` where

        f_v             (n_crops, n_rows, 1024)
        f_a             (n_crops, n_rows, 128), the same array repeated
        snippet_index   (n_seconds,) int64, the map from gold second to model
                        row that macilsd/infer.py applies to the scores
        n_seconds       the gold array length for this video

    Upstream feeds the raw, untruncated sequence at test time -- no
    `process_feat`, no chunking -- and this port does the same. The longest
    video in any gold cohort is 1499 snippets (hatemm), so the quadratic
    attention over the full sequence stays small.

    `n_crops` is 5 for the audio-visual and visual-only models. For audio-only
    it is 1: all five crops carry the same VGGish array, so their sigmoids are
    identical and their mean is that value.
    """

    def __init__(self, corpus, video_ids, max_seqlen, grid="snippet",
                 modality="av"):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.grid = grid
        self.modality = modality
        self.n_crops = 1 if modality == "audio" else align.N_CROPS
        self.max_seqlen = int(max_seqlen)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        f_a, n_seconds, snip = align.aligned_audio(self.corpus, vid, self.grid)
        if self.modality == "audio":
            f_v = np.ones((f_a.shape[0], 1), dtype=np.float32)[None]
        else:
            crops = [align.aligned_visual_crop(self.corpus, vid, c, self.grid,
                                               n_seconds, snip)
                     for c in range(self.n_crops)]
            f_v = np.stack(crops, axis=0)
        f_a = np.repeat(f_a[None], self.n_crops, axis=0)
        if self.grid == "second":
            index_map = np.arange(n_seconds, dtype=np.int64)
        else:
            index_map = align.snippet_index_for_seconds(snip, n_seconds)
        return (torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32)),
                torch.from_numpy(index_map), int(n_seconds), vid)


def describe(corpus, split, grid="snippet"):
    """Small summary for the training log."""
    ids = usable_ids(corpus, hdata.load_split(corpus, split))
    labels = hdata.load_labels(corpus)
    rows = []
    for v in ids:
        n = len(align.snippet_bounds(corpus, v))
        rows.append(n)
    rows = np.asarray(rows)
    pos = sum(labels[v] for v in ids)
    return {
        "corpus": corpus, "split": split, "grid": grid,
        "n_videos": len(ids), "n_hateful": int(pos),
        "n_normal": len(ids) - int(pos),
        "snippets_median": int(np.median(rows)),
        "snippets_p90": int(np.percentile(rows, 90)),
        "snippets_max": int(rows.max()),
    }
