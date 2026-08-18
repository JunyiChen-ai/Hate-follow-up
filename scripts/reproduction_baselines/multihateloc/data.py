"""Three-modality corpus plumbing for the MultiHateLoc reimplementation.

Labels, splits, the gold arrays and the seeded validation carve all come from
`hate_common.data`, so this port sees exactly the cohort, the label collapse
and the train/val protocol the VadCLIP and DSANet ports see. What is new here
is only that a video is three matrices rather than one, and that the batch is
padded rather than cropped.

Temporal unit. All three matrices are already one row per second on the frozen
1 fps gold grid, and their lengths are equal video by video (checked in
`smoke_cpu.py`). MultiHateLoc linearly interpolates VGGish's 1-second audio
vectors up to its frame count; here the audio grid already *is* the frame
grid, so the interpolation is the identity and is not performed. That is a
consequence of our frozen 1 fps choice, not a deviation from the paper.

Batching. The paper is silent on how variable-length videos share a batch.
This port pads each batch to its longest video and carries a boolean mask;
every pooling, top-K, smoothness and loss term reads the mask, so a padded row
never contributes. Nothing is truncated and nothing is uniformly averaged
down, so a score row still maps to one second at both train and test time.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.utils.data as tdata

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..")))

from hate_common import data as hdata  # noqa: E402

REPO_ROOT = hdata.REPO_ROOT
FEATURE_ROOT = os.path.join(REPO_ROOT, "results", "reproduction", "features")

# modality -> (feature directory, dimensionality). The visual encoder is the
# ImageNet ViT-B/16 the paper cites (Dosovitskiy), not CLIP.
MODALITIES = ("visual", "audio", "text")
FEATURE_DIRS = {
    "visual": "vit_b16_imagenet_1fps",
    "audio": "vggish_1s",
    "text": "bert_sentence_1fps",
}
FEATURE_DIMS = {"visual": 768, "audio": 128, "text": 768}


def feature_path(modality, corpus, video_id):
    return os.path.join(FEATURE_ROOT, FEATURE_DIRS[modality], corpus,
                        "%s.npy" % video_id)


class MultiModalDataset(tdata.Dataset):
    """One item per video: three (T, d) matrices, the video label and T."""

    def __init__(self, corpus, video_ids, labels=None):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.labels = labels if labels is not None else hdata.load_labels(corpus)
        missing = [v for v in self.video_ids if v not in self.labels]
        if missing:
            raise KeyError("%d ids have no label, e.g. %s"
                           % (len(missing), missing[:5]))
        for mod in MODALITIES:
            absent = [v for v in self.video_ids
                      if not os.path.exists(feature_path(mod, corpus, v))]
            if absent:
                raise FileNotFoundError("%d ids have no %s feature, e.g. %s"
                                        % (len(absent), mod, absent[:5]))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        feats = {}
        length = None
        for mod in MODALITIES:
            arr = np.load(feature_path(mod, self.corpus, vid)).astype(np.float32)
            if arr.ndim != 2 or arr.shape[1] != FEATURE_DIMS[mod]:
                raise ValueError("%s/%s: %s feature has shape %s, expected "
                                 "(T, %d)" % (self.corpus, vid, mod,
                                              arr.shape, FEATURE_DIMS[mod]))
            if length is None:
                length = arr.shape[0]
            elif arr.shape[0] != length:
                raise ValueError("%s/%s: %s has %d rows but another modality "
                                 "has %d" % (self.corpus, vid, mod,
                                             arr.shape[0], length))
            feats[mod] = torch.from_numpy(arr)
        return feats, int(self.labels[vid]), int(length), vid


def collate(batch):
    """Pad to the batch's longest video and return the validity mask."""
    lengths = [item[2] for item in batch]
    tmax = max(lengths)
    feats = {}
    for mod in MODALITIES:
        out = torch.zeros(len(batch), tmax, FEATURE_DIMS[mod])
        for i, item in enumerate(batch):
            out[i, :item[2]] = item[0][mod]
        feats[mod] = out
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    lens = torch.tensor(lengths, dtype=torch.long)
    mask = torch.arange(tmax)[None, :] < lens[:, None]
    vids = [item[3] for item in batch]
    return feats, labels, lens, mask, vids


def describe(corpus, split):
    ids = hdata.load_split(corpus, split)
    labels = hdata.load_labels(corpus)
    lengths = [np.load(feature_path("visual", corpus, v),
                       mmap_mode="r").shape[0] for v in ids]
    pos = sum(labels[v] for v in ids)
    return {"corpus": corpus, "split": split, "n_videos": len(ids),
            "n_hateful": int(pos), "n_normal": len(ids) - int(pos),
            "seconds_min": int(min(lengths)),
            "seconds_median": int(np.median(lengths)),
            "seconds_max": int(max(lengths))}
