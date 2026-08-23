#!/usr/bin/env python3
"""Freeze the train/validation/test manifests used by baseline reproductions.

The manifests are the single source of truth for which videos each baseline may
train on and which it is evaluated on.  They are derived from the upstream split
files intersected with the media actually present on disk, so that any download
attrition is recorded once, here, instead of being silently absorbed by each
baseline's dataloader.

Split rules (frozen by owner decision, see the Phase 0 plan):

* HateMM preserves the released ``train_clean`` / ``valid`` / ``test_clean``
  split, intersected with available media.
* MHClip preserves the released ``train`` / ``valid`` / ``test`` split,
  intersected with available media.  ``k9OtaMbK0Ac`` is removed from train
  because upstream also lists it in test.
* HateClipSeg has **no published split**.  The paper reports an 80/20 division
  but releases no video ids, so no upstream manifest exists to intersect with.
  The split written here is therefore *ours*, not a reproduction of theirs:
  a seeded, video-level-stratified 64/16/20 train/validation/test draw over the
  annotated videos whose media is present locally. See ``hateclipseg_split`` below for the exact
  rule; every number computed on it must be reported as being on our split.

Outputs ``results/reproduction/splits/*.txt``, one video id per line, sorted,
plus the SHA256 of each file.  Run with ``--check`` to recompute the manifests
and fail if they differ from what is on disk.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/home/jehc223/data")
OUT_DIR = REPO / "results" / "reproduction" / "splits"

# Upstream English video present in both the train and the test TSV.  It is
# dropped from train so that no baseline can see a test video during training.
MHC_EN_TRAIN_TEST_OVERLAP = "k9OtaMbK0Ac"

HATEMM_SPLITS = DATA / "HateMM" / "splits"
HATEMM_VIDEO_DIRS = [
    DATA / "HateMM" / "video",
    REPO / "results" / "testruns" / "hatemm" / "media",
]

MHC_SPAN_CANDIDATES = (
    DATA / "Multihateclip" / "upstream_spans",
    Path("/home/jehc223/Retrieval-hate/data/gt/mhc_votes"),
)
MHC_VIDEO_DIRS = {
    "en": [
        DATA / "Multihateclip" / "English" / "video_mp4",
        REPO / "results" / "testruns" / "mhclip_en" / "media",
    ],
    "zh": [
        DATA / "Multihateclip" / "Chinese" / "video",
        REPO / "results" / "testruns" / "mhclip_zh" / "media",
    ],
}

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v"}

# ---- HateClipSeg -----------------------------------------------------------
# Segment-level gold: one row per video, a python-literal list of 6-dim
# multi-hot labels [normal, hateful, insulting, sexual, violence, harm] and a
# parallel list of ['start', 'end'] string seconds.
HCS_GOLD_CSV_CANDIDATES = (
    REPO / "idea-stage" / "pilots" / "b1_coverage_audit" / "data"
    / "segment_level_annotation.csv",
    DATA / "HateClipSeg" / "Dataset" / "segment_level_annotation.csv",
)
HCS_VIDEO_DIRS = [
    DATA / "HateClipSeg" / "videos",
    DATA / "HateClipSeg" / "video",
    Path("/home/jehc223/Retrieval-hate/data/video/HateClipSeg"),
]
HCS_SEED = 234
HCS_TEST_FRACTION = 0.2
HCS_VALID_FRACTION_OF_REMAINDER = 0.2
# This video arrived after the 394-video cohort and 79-video test manifest were
# frozen on 2026-08-19. Excluding it preserves the published test SHA256
# 0d648643... while allowing a validation split to be carved from the original
# frozen training cohort.
HCS_POST_FREEZE_MEDIA = {"yt_DnrYK1FXKgk"}
# Present by filename but truncated before the first decodable frame. It fell
# in train in the frozen draw; remove it after drawing so val/test IDs stay
# byte-for-byte stable.
HCS_UNUSABLE_TRAIN_MEDIA = {"yt_NzvfkIYS5Yg"}
# Fixed stratum order, so the draw does not depend on dict iteration order.
HCS_STRATA = ("negative", "positive")


def available_ids(dirs: list[Path]) -> dict[str, Path]:
    """Map video id -> path, preferring the first directory that supplies it."""
    found: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS and p.stat().st_size > 0:
                found.setdefault(p.stem, p)
    return found


def read_id_list(path: Path) -> list[str]:
    """Read a HateMM split file: one bare video id per line."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(line)
    return ids


def read_mhc_tsv(path: Path) -> list[str]:
    """Read the Video_ID column of an upstream MultiHateClip span TSV."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [r["Video_ID"].strip() for r in reader if r.get("Video_ID", "").strip()]


def mhc_tsv(lang: str, split: str) -> Path:
    """Resolve either the original lowercase TSVs or the archived copies."""
    language = {"en": "English", "zh": "Chinese"}[lang]
    names = (f"{lang}_{split}.tsv", f"mhc_{language}_{split}.tsv")
    for root in MHC_SPAN_CANDIDATES:
        for name in names:
            path = root / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"no MHC {lang}/{split} TSV in {MHC_SPAN_CANDIDATES}")


def dedup(ids: list[str]) -> list[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- HateClipSeg
def hcs_is_offensive_union(label: list) -> bool:
    """Segment label is offensive under the union rule.

    Identical to ``is_offensive_union`` in
    ``scripts/duplex/sentinel_localization_pilot.py``: any of the five
    non-normal dimensions (hateful, insulting, sexual, violence, harm) set.
    """
    return any(int(x) == 1 for x in label[1:6])


def read_hcs_gold() -> dict[str, list[list[int]]]:
    """video id -> segment labels, from the segment-level annotation CSV."""
    csv.field_size_limit(1 << 30)
    out: dict[str, list[list[int]]] = {}
    path = next((p for p in HCS_GOLD_CSV_CANDIDATES if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no HateClipSeg annotation CSV in {HCS_GOLD_CSV_CANDIDATES}")
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            labels = ast.literal_eval(row["Segment-Level Label"])
            spans = ast.literal_eval(row["Segment Timestamp"])
            if len(labels) != len(spans):
                continue
            out[row["Video Id"].strip()] = labels
    return out


def hateclipseg_split() -> tuple[list[str], list[str], list[str], dict]:
    """Our seeded 64/16/20 HateClipSeg split, stratified by video label.

    HateClipSeg publishes no split ids, so there is nothing upstream to
    intersect with and nothing to reproduce.  This draw is ours and is frozen
    here:

    1. Eligible videos are those with a segment-level annotation row *and*
       local media.  Annotated videos without media are attrition and are
       listed in the report, never silently absorbed.
    2. A video is *positive* if at least one of its segments is offensive
       under the union rule, and *negative* otherwise.  The video-level
       annotation file is not consulted: the frame gold is built from the
       segment labels, so the stratification variable is derived from the
       same source, and the two can never disagree.
    3. Within each stratum, ids are sorted and then shuffled by
       ``random.Random(234)`` -- one generator, strata visited in the fixed
       order (negative, positive) -- and the first ``round(0.2 * n)`` go to
       test.  Sorting before shuffling makes the draw independent of
       filesystem order.
    """
    gold = read_hcs_gold()
    avail = available_ids(HCS_VIDEO_DIRS)
    annotated = sorted(gold)
    eligible = [v for v in annotated
                if v in avail and v not in HCS_POST_FREEZE_MEDIA]
    missing_media = [v for v in annotated if v not in avail]
    media_unannotated = sorted(set(avail) - set(gold))

    strata: dict[str, list[str]] = {name: [] for name in HCS_STRATA}
    for vid in eligible:
        positive = any(hcs_is_offensive_union(lab) for lab in gold[vid])
        strata["positive" if positive else "negative"].append(vid)

    rng = random.Random(HCS_SEED)
    train: list[str] = []
    valid: list[str] = []
    test: list[str] = []
    per_stratum = {}
    for name in HCS_STRATA:
        ids = sorted(strata[name])
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_test = int(round(HCS_TEST_FRACTION * len(shuffled)))
        test.extend(shuffled[:n_test])
        remainder = shuffled[n_test:]
        n_valid = int(round(HCS_VALID_FRACTION_OF_REMAINDER * len(remainder)))
        valid.extend(remainder[:n_valid])
        train.extend(remainder[n_valid:])
        per_stratum[name] = {
            "eligible": len(ids),
            "test": n_test,
            "validation": n_valid,
            "train": len(remainder) - n_valid,
        }

    bad_outside_train = HCS_UNUSABLE_TRAIN_MEDIA & (set(valid) | set(test))
    if bad_outside_train:
        raise RuntimeError("frozen unusable media entered val/test: %s" %
                           sorted(bad_outside_train))
    train = [v for v in train if v not in HCS_UNUSABLE_TRAIN_MEDIA]
    for name in HCS_STRATA:
        per_stratum[name]["train"] = sum(v in train for v in strata[name])

    info = {
        "manifest": "hateclipseg",
        "provenance": (
            "ours, not upstream: HateClipSeg publishes an 80/20 ratio but no "
            "video ids"
        ),
        "seed": HCS_SEED,
        "test_fraction": HCS_TEST_FRACTION,
        "validation_fraction_of_remainder": HCS_VALID_FRACTION_OF_REMAINDER,
        "stratification": (
            "video positive iff at least one segment is offensive under the "
            "union rule over dims 1..5"
        ),
        "annotated": len(annotated),
        "available": len(eligible),
        "missing_media": len(missing_media),
        "missing_media_ids": missing_media,
        "media_without_annotation": media_unannotated,
        "post_freeze_media_excluded": sorted(HCS_POST_FREEZE_MEDIA & set(avail)),
        "unusable_train_media_excluded": sorted(
            HCS_UNUSABLE_TRAIN_MEDIA & set(avail)),
        "per_stratum": per_stratum,
    }
    return sorted(train), sorted(valid), sorted(test), info


def build() -> tuple[dict[str, list[str]], list[dict]]:
    manifests: dict[str, list[str]] = {}
    report: list[dict] = []

    # ---- HateMM -----------------------------------------------------------
    hm_avail = available_ids(HATEMM_VIDEO_DIRS)
    hm_train_up = dedup(read_id_list(HATEMM_SPLITS / "train_clean.csv"))
    hm_valid_path = HATEMM_SPLITS / "validation_clean.csv"
    if not hm_valid_path.is_file():
        hm_valid_path = HATEMM_SPLITS / "valid.csv"
    hm_valid_up = dedup(read_id_list(hm_valid_path))
    hm_test_up = dedup(read_id_list(HATEMM_SPLITS / "test_clean.csv"))
    hm_test_set = set(hm_test_up)
    hm_train_up = [i for i in hm_train_up if i not in hm_test_set]
    hm_valid_up = [i for i in hm_valid_up if i not in hm_test_set]

    for name, upstream in (("hatemm_train", hm_train_up),
                           ("hatemm_val", hm_valid_up),
                           ("hatemm_test", hm_test_up)):
        kept = sorted(i for i in upstream if i in hm_avail)
        missing = sorted(i for i in upstream if i not in hm_avail)
        manifests[name] = kept
        report.append(
            {
                "manifest": name,
                "upstream": len(upstream),
                "available": len(kept),
                "missing": len(missing),
                "missing_ids": missing,
            }
        )

    # ---- MultiHateClip ----------------------------------------------------
    for lang in ("en", "zh"):
        avail = available_ids(MHC_VIDEO_DIRS[lang])
        train_up = dedup(read_mhc_tsv(mhc_tsv(lang, "train")))
        valid_up = dedup(read_mhc_tsv(mhc_tsv(lang, "valid")))
        test_up = dedup(read_mhc_tsv(mhc_tsv(lang, "test")))
        test_set = set(test_up)
        overlap = sorted(set(train_up) & test_set)
        train_up = [i for i in train_up if i not in test_set]
        valid_up = [i for i in valid_up if i not in test_set]

        for name, upstream in (
            (f"mhclip_{lang}_train", train_up),
            (f"mhclip_{lang}_val", valid_up),
            (f"mhclip_{lang}_test", test_up),
        ):
            kept = sorted(i for i in upstream if i in avail)
            missing = sorted(i for i in upstream if i not in avail)
            manifests[name] = kept
            entry = {
                "manifest": name,
                "upstream": len(upstream),
                "available": len(kept),
                "missing": len(missing),
                "missing_ids": missing,
            }
            if name.endswith("_train"):
                entry["train_test_overlap_removed"] = overlap
            report.append(entry)

        if lang == "en" and MHC_EN_TRAIN_TEST_OVERLAP not in overlap:
            print(
                f"WARNING: expected {MHC_EN_TRAIN_TEST_OVERLAP} in the EN "
                f"train/test overlap, found {overlap}",
                file=sys.stderr,
            )

    # ---- HateClipSeg ------------------------------------------------------
    hcs_train, hcs_valid, hcs_test, hcs_info = hateclipseg_split()
    manifests["hateclipseg_train"] = hcs_train
    manifests["hateclipseg_val"] = hcs_valid
    manifests["hateclipseg_test"] = hcs_test
    for name, ids in (("hateclipseg_train", hcs_train),
                      ("hateclipseg_val", hcs_valid),
                      ("hateclipseg_test", hcs_test)):
        report.append(
            {
                "manifest": name,
                "upstream": hcs_info["annotated"],
                "available": len(ids),
                "missing": 0,
                "missing_ids": [],
                "note": "seeded local split, no upstream ids exist",
            }
        )
    report.append(hcs_info)

    return manifests, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="recompute and diff against the manifests on disk instead of writing",
    )
    args = ap.parse_args()

    manifests, report = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failed = False
    digests = {}
    for name, ids in sorted(manifests.items()):
        path = OUT_DIR / f"{name}.txt"
        body = "".join(f"{i}\n" for i in ids)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != body:
                print(f"MISMATCH {path}", file=sys.stderr)
                failed = True
        else:
            path.write_text(body, encoding="utf-8")
        if path.exists():
            digests[name] = sha256_file(path)

    for row in report:
        if "upstream" not in row:  # the HateClipSeg provenance block
            print(
                f"{row['manifest']:20s} seed={row['seed']} "
                f"annotated={row['annotated']:4d} "
                f"available={row['available']:4d} "
                f"missing_media={row['missing_media']:4d}  "
                + ", ".join(
                    f"{k}: {v['eligible']} -> {v['train']}/{v['validation']}/{v['test']}"
                    for k, v in row["per_stratum"].items()
                )
            )
            continue
        overlap = row.get("train_test_overlap_removed")
        extra = f"  overlap_removed={overlap}" if overlap else ""
        print(
            f"{row['manifest']:20s} upstream={row['upstream']:4d} "
            f"available={row['available']:4d} missing={row['missing']:4d}{extra}"
        )
        if row["missing_ids"]:
            shown = row["missing_ids"][:20]
            tail = " ..." if len(row["missing_ids"]) > 20 else ""
            print(f"    missing: {', '.join(shown)}{tail}")

    print()
    for name, digest in sorted(digests.items()):
        print(f"SHA256  {name}.txt  {digest}")

    (OUT_DIR / "manifest_report.json").write_text(
        json.dumps({"counts": report, "sha256": digests}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
