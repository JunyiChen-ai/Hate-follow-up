"""Corpus plumbing shared by the VadCLIP and DSANet ports.

Both upstream repos read a two-column CSV (`path,label`) whose rows are
pre-cropped feature files, and both take the temporal unit of those files for
granted. This module replaces that CSV with the reproduction study's own
manifests and states the temporal mapping explicitly.

The temporal unit
-----------------
Upstream features are per-snippet: XD-Violence and UCF-Crime CLIP features are
one 512-d row per 16-frame snippet, so a row is roughly 0.53 s (UCF, 30 fps) or
0.67 s (XD, 24 fps) of video, and upstream evaluation repeats every snippet
score 16 times to reach the frame grid.

This study's features are one 512-d row per second, sampled on the same 1 fps
grid the gold spans are rasterised onto (docs/duplex/FRAME_EVAL_PROTOCOL.md).
So:

    one feature row  ==  one snippet (the model's temporal unit)
                     ==  one second
                     ==  one gold frame

Two things follow. First, no upsampling is needed at inference: upstream's
`np.repeat(scores, 16, 0)` is dropped, and a length-T score vector lines up
element-for-element with the length-T gold array. Second, every hyperparameter
counted in snippets now means seconds, so `--visual-length` is re-read as a
window in seconds and set per corpus (see the option modules).

Nothing here re-extracts or resamples features. The .npy files are consumed as
they were written.
"""

from __future__ import annotations

import ast
import csv
import json
import os

import numpy as np
import torch
import torch.utils.data as data

from . import tools

# ------------------------------------------------------------------- paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FEATURE_ROOT = os.path.join(REPO_ROOT, "results", "reproduction",
                            "features", "clip_b16_1fps")
SPLIT_ROOT = os.path.join(REPO_ROOT, "results", "reproduction", "splits")
GT_ROOT = os.path.join(REPO_ROOT, "results", "reproduction", "gt")

HATEMM_ANNOTATION_CANDIDATES = (
    "/home/jehc223/data/HateMM/upstream_spans/HateMM_annotation.csv",
    "/home/jehc223/Retrieval-hate/data/gt/HateMM/HateMM_annotation.csv",
)
MHCLIP_ROOT_CANDIDATES = (
    "/home/jehc223/data/Multihateclip/upstream_spans",
    "/home/jehc223/Retrieval-hate/data/gt/mhc_votes",
)
HCS_SEGMENT_CSV_CANDIDATES = (
    os.path.join(REPO_ROOT, "idea-stage", "pilots", "b1_coverage_audit",
                 "data", "segment_level_annotation.csv"),
    "/home/jehc223/data/HateClipSeg/Dataset/segment_level_annotation.csv",
)

CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")

# Class order is load bearing: slot 0 is the normal class everywhere. See
# dsanet/descriptions.py for the full argument.
CLASS_NAMES = ("normal", "hateful")
PROMPT_TEXT = ["normal content", "hateful content"]
NUM_CLASSES = 2


# ------------------------------------------------------------------ labels
def load_labels(corpus):
    """video_id -> 0 (normal) / 1 (hateful), for every id the corpus knows.

    HateMM: the `label` column of the upstream annotation CSV, Hate -> 1.
    MultiHateClip: upstream Majority_Voting over the train/valid/test TSVs,
    with the binary collapse CLAUDE.md fixes, Hateful + Offensive -> 1.
    HateClipSeg: a video is 1 iff at least one of its segments is offensive
    under the union rule over dimensions 1..5 (hateful, insulting, sexual,
    violence, harm). The segment-level CSV is read directly rather than the
    shipped video-level file, because that is the source the frame gold
    (scripts/duplex/build_gt_arrays_hateclipseg.py) and the split
    stratification (scripts/duplex/reproduction_splits.py) both derive from,
    so the three cannot disagree.
    """
    if corpus == "hatemm":
        out = {}
        path = next((p for p in HATEMM_ANNOTATION_CANDIDATES
                     if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError("no HateMM annotation CSV found")
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                vid = row["video_file_name"].rsplit(".", 1)[0]
                label = row["label"].strip()
                if label not in ("Hate", "Non Hate"):
                    raise ValueError("unexpected HateMM label %r for %s"
                                     % (label, vid))
                out[vid] = 1 if label == "Hate" else 0
        return out

    if corpus in ("mhclip_en", "mhclip_zh"):
        lang = corpus.split("_")[1]
        language = {"en": "English", "zh": "Chinese"}[lang]
        out = {}
        for part in ("train", "valid", "test"):
            candidates = []
            for root in MHCLIP_ROOT_CANDIDATES:
                candidates.extend((
                    os.path.join(root, "%s_%s.tsv" % (lang, part)),
                    os.path.join(root, "mhc_%s_%s.tsv" % (language, part)),
                ))
            path = next((p for p in candidates if os.path.isfile(p)), None)
            if path is None:
                raise FileNotFoundError("no MHC %s/%s TSV found" % (lang, part))
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    vote = row["Majority_Voting"].strip()
                    if vote not in ("Hateful", "Offensive", "Normal"):
                        raise ValueError("unexpected Majority_Voting %r in %s"
                                         % (vote, path))
                    out[row["Video_ID"]] = 0 if vote == "Normal" else 1
        return out

    if corpus == "hateclipseg":
        csv.field_size_limit(1 << 30)
        out = {}
        path = next((p for p in HCS_SEGMENT_CSV_CANDIDATES
                     if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError("no HateClipSeg segment annotation CSV found")
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                labels = ast.literal_eval(row["Segment-Level Label"])
                spans = ast.literal_eval(row["Segment Timestamp"])
                if len(labels) != len(spans):
                    # The same guard reproduction_splits.py applies: a row
                    # whose two lists disagree carries no usable segment table
                    # and never entered the split.
                    continue
                out[row["Video Id"].strip()] = int(any(
                    any(int(x) == 1 for x in lab[1:6]) for lab in labels))
        return out

    raise ValueError("unknown corpus %r (expected one of %s)"
                     % (corpus, ", ".join(CORPORA)))


def load_split(corpus, split):
    """The frozen id list for one split, order preserved."""
    path = os.path.join(SPLIT_ROOT, "%s_%s.txt" % (corpus, split))
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def feature_path(corpus, video_id):
    return os.path.join(FEATURE_ROOT, corpus, "%s.npy" % video_id)


def gt_arrays(corpus, split="test"):
    """video_id -> uint8 frame labels, from the frozen gold npz."""
    path = os.path.join(GT_ROOT, "%s_%s.npz" % (corpus, split))
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def split_train_val(video_ids, labels, val_frac, seed):
    """Deterministic stratified carve of a validation subset out of train.

    Upstream selects its checkpoint on the *test* set (both xd_train.py and
    ucf_train.py call test() every epoch and keep the best-AP state). That
    makes the reported number a test-selected number, which is not usable as a
    baseline here. This port instead holds out a fixed fraction of the training
    split, stratified on the video label, and never reads the test split during
    training. Pass --val-frac 0 to reproduce the upstream last-epoch model with
    no selection at all.

    Determinism does not depend on the input order: ids are sorted first.
    """
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0, 1), got %r" % (val_frac,))
    if val_frac == 0.0:
        return list(video_ids), []

    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for cls in (0, 1):
        members = sorted(v for v in video_ids if labels[v] == cls)
        order = rng.permutation(len(members))
        n_val = int(round(len(members) * val_frac))
        n_val = min(max(n_val, 1 if members else 0), max(len(members) - 1, 0))
        chosen = {members[i] for i in order[:n_val]}
        val_ids.extend(sorted(chosen))
        train_ids.extend(m for m in members if m not in chosen)
    return sorted(train_ids), sorted(val_ids)


def load_train_val(corpus, labels=None, val_frac=0.1, seed=234,
                   legacy_resplit=False):
    """Return the frozen official train/validation manifests.

    ``legacy_resplit=True`` reproduces the pre-2026-08-23 protocol which
    carved validation from a merged training manifest. New experiments must
    use the default official manifests. HateClipSeg's validation manifest is
    the frozen local split because that dataset releases no split IDs.
    """
    train_ids = load_split(corpus, "train")
    if legacy_resplit:
        labels = labels if labels is not None else load_labels(corpus)
        return split_train_val(train_ids, labels, val_frac, seed)
    val_ids = load_split(corpus, "val")
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError("train/validation overlap for %s: %s"
                         % (corpus, overlap[:5]))
    return train_ids, val_ids


def label_vectors(class_indices, device=None):
    """One-hot targets in CLASS_NAMES order, slot 0 = normal.

    Replaces utils.tools.get_batch_label, whose two-class branch keys off the
    literal string 'Normal' from the XD/UCF CSVs.
    """
    idx = torch.as_tensor(list(class_indices), dtype=torch.long)
    out = torch.zeros(len(idx), NUM_CLASSES)
    out[torch.arange(len(idx)), idx] = 1.0
    return out.to(device) if device is not None else out


# ----------------------------------------------------------------- dataset
class HateVideoDataset(data.Dataset):
    """One item per video, feature rows on the 1 fps grid.

    Mirrors utils/dataset.py's XDDataset: training items go through
    `tools.process_feat` (uniform-average down to `visual_length` rows when the
    video is longer, zero-pad up when shorter) and test items through
    `tools.process_split` (chop into consecutive `visual_length` blocks, pad the
    tail). Both helpers are used unmodified.

    Returns (feature, class_index, length[, video_id]). `class_index` is an int
    rather than upstream's label string, since the string was only ever a key
    into the label map.
    """

    def __init__(self, corpus, video_ids, visual_length, test_mode,
                 labels=None):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.visual_length = int(visual_length)
        self.test_mode = bool(test_mode)
        self.labels = labels if labels is not None else load_labels(corpus)
        missing = [v for v in self.video_ids if v not in self.labels]
        if missing:
            raise KeyError("%d ids have no label, e.g. %s"
                           % (len(missing), missing[:5]))
        absent = [v for v in self.video_ids
                  if not os.path.exists(feature_path(corpus, v))]
        if absent:
            raise FileNotFoundError("%d ids have no feature file, e.g. %s"
                                    % (len(absent), absent[:5]))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        feat = np.load(feature_path(self.corpus, vid)).astype(np.float32)
        if self.test_mode:
            feat, length = tools.process_split(feat, self.visual_length)
        else:
            feat, length = tools.process_feat(feat, self.visual_length)
        feat = torch.from_numpy(np.ascontiguousarray(feat))
        label = int(self.labels[vid])
        if self.test_mode:
            return feat, label, length, vid
        return feat, label, length


def describe_corpus(corpus, split):
    """Small summary used by the training logs."""
    ids = load_split(corpus, split)
    labels = load_labels(corpus)
    lengths = [np.load(feature_path(corpus, v), mmap_mode="r").shape[0]
               for v in ids]
    pos = sum(labels[v] for v in ids)
    return {
        "corpus": corpus,
        "split": split,
        "n_videos": len(ids),
        "n_hateful": int(pos),
        "n_normal": len(ids) - int(pos),
        "seconds_min": int(min(lengths)),
        "seconds_median": int(np.median(lengths)),
        "seconds_p90": int(np.percentile(lengths, 90)),
        "seconds_max": int(max(lengths)),
    }


def load_scores_jsonl(path):
    """{video_id: {branch: np.ndarray}} from a scores.jsonl written by infer."""
    out = {}
    with open(path) as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec["video_id"]
            if vid in out:
                raise ValueError(
                    f"{path}:{line_number}: duplicate video_id {vid!r}")
            branches = {k: np.asarray(v, dtype=float)
                        for k, v in rec.items() if k.startswith("score_")}
            if not branches:
                raise ValueError(
                    f"{path}:{line_number}: no score_* branch for {vid!r}")
            out[vid] = branches
    return out
