"""
Data utilities for MHClip datasets.

Provides media path resolution, annotation loading, and clean split generation.
"""

import argparse
import csv
import json
import os
import glob as globmod

# Dataset location. Default is this cluster's shared data directory; on
# other machines set HVD_DATA_ROOT to a directory laid out the same way
# (e.g. $HVD_DATA_ROOT/ImpliHateVid/{annotation(new).json,splits/,frames_16/}).
_DATA_ROOT = os.environ.get("HVD_DATA_ROOT", "/data/jehc223")

DATASET_ROOTS = {
    "MHClip_EN": os.path.join(_DATA_ROOT, "Multihateclip", "English"),
    "MHClip_ZH": os.path.join(_DATA_ROOT, "Multihateclip", "Chinese"),
    "HateMM": os.path.join(_DATA_ROOT, "HateMM"),
    "ImpliHateVid": os.path.join(_DATA_ROOT, "ImpliHateVid"),
}

SPLIT_ALIASES = {
    "train": "train",
    "test": "test",
    "valid": "validation",
    "val": "validation",
    "validation": "validation",
}

RAW_VALIDATION_SPLITS = {
    "MHClip_EN": "valid.csv",
    "MHClip_ZH": "valid.csv",
    "HateMM": "valid.csv",
    "ImpliHateVid": "val.csv",
}

MP4_SUBDIRS = {
    "MHClip_EN": "video_mp4",
    "MHClip_ZH": "video",
    "HateMM": "video",
    "ImpliHateVid": "video",
}

# Videos to unconditionally skip across every scoring pipeline. These 8
# MHClip_ZH mp4 files pass the `os.path.isfile` + >1000-byte check but
# fail vLLM's Qwen3-VL video decoder ("Expected reading N frames, but
# only loaded 0 frames from video.") — observed across our holistic-score
# pipeline, naive 2B text, and MARS faithful on 2026-04-14. Every pipeline
# that feeds mp4 → vLLM dies on them. Manually maintained list; add new
# entries here when more broken videos surface.
SKIP_VIDEOS = {
    "MHClip_EN": set(),
    "MHClip_ZH": {
        "BV1Gw411E7i2",
        "BV1Qx411V7tT",
        "BV16N4y1q7WU",
        "BV1nJ4m1p7BG",
        "BV1KK411P7uJ",
        "BV1zD4y1Y7ec",
        "BV1du411g7tk",
        "BV1bA41137we",
    },
    "HateMM": set(),
    "ImpliHateVid": set(),
}

_EXTRA_SKIP_FILE = os.environ.get("EXTRA_SKIP_VIDEOS_FILE")
if _EXTRA_SKIP_FILE and os.path.isfile(_EXTRA_SKIP_FILE):
    with open(_EXTRA_SKIP_FILE) as _sf:
        extra = {line.strip() for line in _sf if line.strip()}
    for _skip_set in SKIP_VIDEOS.values():
        _skip_set.update(extra)

_EXTRA_SKIP_VIDEOS = os.environ.get("EXTRA_SKIP_VIDEOS")
if _EXTRA_SKIP_VIDEOS:
    extra = {v.strip() for v in _EXTRA_SKIP_VIDEOS.split(",") if v.strip()}
    for _skip_set in SKIP_VIDEOS.values():
        _skip_set.update(extra)


def get_media_path(vid, dataset):
    """Return (path, media_type) or None.

    Videos in `SKIP_VIDEOS[dataset]` unconditionally return None so every
    scoring script skips them at the top of the loop. Otherwise: check
    mp4 first (must exist and be >1000 bytes), then fall back to
    `frames/<vid>/*.jpg`.
    """
    if vid in SKIP_VIDEOS.get(dataset, set()):
        return None

    root = DATASET_ROOTS[dataset]
    mp4_dir = MP4_SUBDIRS[dataset]

    # Check mp4
    mp4_path = os.path.join(root, mp4_dir, f"{vid}.mp4")
    if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 1000:
        return (mp4_path, "video")

    # Check frames
    frames_dir = os.path.join(root, "frames", vid)
    if os.path.isdir(frames_dir):
        jpgs = globmod.glob(os.path.join(frames_dir, "*.jpg"))
        if len(jpgs) >= 1:
            return (frames_dir, "frames")

    return None


def load_annotations(dataset):
    """Load annotation(new).json and return dict[vid -> {title, transcript, label}]."""
    root = DATASET_ROOTS[dataset]
    ann_path = os.path.join(root, "annotation(new).json")
    with open(ann_path) as f:
        data = json.load(f)

    result = {}
    for entry in data:
        vid = entry["Video_ID"]
        result[vid] = {
            "title": entry.get("Title", ""),
            "transcript": entry.get("Transcript", ""),
            "label": entry.get("Label", ""),
        }
    return result


def canonical_split(split):
    try:
        return SPLIT_ALIASES[split]
    except KeyError as exc:
        raise ValueError(
            f"unknown split={split!r}; expected train/test/validation"
        ) from exc


def raw_split_path(dataset, split):
    split = canonical_split(split)
    if split == "validation":
        fname = RAW_VALIDATION_SPLITS[dataset]
    else:
        fname = f"{split}.csv"
    return os.path.join(DATASET_ROOTS[dataset], "splits", fname)


def clean_split_path(dataset, split):
    split = canonical_split(split)
    return os.path.join(DATASET_ROOTS[dataset], "splits", f"{split}_clean.csv")


def load_clean_split_ids(dataset, split):
    split = canonical_split(split)
    path = clean_split_path(dataset, split)
    if not os.path.isfile(path):
        generate_clean_splits(dataset, splits=[split])
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def generate_clean_splits(dataset, splits=None):
    """Generate clean split csvs excluding missing-media or not-in-annotation videos.

    Validation is normalized to `validation_clean.csv` even when the raw
    dataset file is named `valid.csv` or `val.csv`.
    """
    root = DATASET_ROOTS[dataset]
    splits_dir = os.path.join(root, "splits")
    annotations = load_annotations(dataset)
    splits = [canonical_split(s) for s in (splits or ["train", "test", "validation"])]

    for split_name in splits:
        src_path = raw_split_path(dataset, split_name)
        if not os.path.isfile(src_path):
            print(f"  [WARN] {src_path} not found, skipping")
            continue

        with open(src_path) as f:
            all_ids = [line.strip() for line in f if line.strip()]

        clean_ids = []
        excluded_no_annotation = 0
        excluded_no_media = 0

        for vid in all_ids:
            if vid not in annotations:
                excluded_no_annotation += 1
                continue
            if get_media_path(vid, dataset) is None:
                excluded_no_media += 1
                continue
            clean_ids.append(vid)

        out_path = clean_split_path(dataset, split_name)
        with open(out_path, "w") as f:
            for vid in clean_ids:
                f.write(vid + "\n")

        # Count label distribution
        label_counts = {}
        for vid in clean_ids:
            lbl = annotations[vid]["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        print(f"  {dataset} {split_name}: {len(all_ids)} total -> {len(clean_ids)} clean "
              f"(excluded: {excluded_no_annotation} no-annotation, {excluded_no_media} no-media)")
        for lbl in sorted(label_counts.keys()):
            print(f"    {lbl}: {label_counts[lbl]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-splits", action="store_true",
                        help="Generate clean splits for both datasets")
    args = parser.parse_args()

    if args.generate_splits:
        for ds in ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]:
            print(f"\n=== {ds} ===")
            generate_clean_splits(ds)
