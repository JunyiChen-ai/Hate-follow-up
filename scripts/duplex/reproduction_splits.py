#!/usr/bin/env python3
"""Freeze the train/test video-ID manifests used by every baseline reproduction.

The manifests are the single source of truth for which videos each baseline may
train on and which it is evaluated on.  They are derived from the upstream split
files intersected with the media actually present on disk, so that any download
attrition is recorded once, here, instead of being silently absorbed by each
baseline's dataloader.

Split rules (frozen by owner decision, see the Phase 0 plan):

* HateMM  train = ``train_clean`` + ``validation_clean`` (851 upstream ids)
  intersected with available media; test = ``test_clean`` (215 ids).
* MHClip  train = upstream ``train`` + ``valid`` intersected with available
  media, minus ``k9OtaMbK0Ac`` (an English video that upstream lists in both
  train and test); test = upstream ``test`` intersected with available media.

Outputs ``results/reproduction/splits/*.txt``, one video id per line, sorted,
plus the SHA256 of each file.  Run with ``--check`` to recompute the manifests
and fail if they differ from what is on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

MHC_SPANS = DATA / "Multihateclip" / "upstream_spans"
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


def build() -> tuple[dict[str, list[str]], list[dict]]:
    manifests: dict[str, list[str]] = {}
    report: list[dict] = []

    # ---- HateMM -----------------------------------------------------------
    hm_avail = available_ids(HATEMM_VIDEO_DIRS)
    hm_train_up = dedup(
        read_id_list(HATEMM_SPLITS / "train_clean.csv")
        + read_id_list(HATEMM_SPLITS / "validation_clean.csv")
    )
    hm_test_up = dedup(read_id_list(HATEMM_SPLITS / "test_clean.csv"))
    hm_test_set = set(hm_test_up)
    hm_train_up = [i for i in hm_train_up if i not in hm_test_set]

    for name, upstream in (("hatemm_train", hm_train_up), ("hatemm_test", hm_test_up)):
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
        train_up = dedup(
            read_mhc_tsv(MHC_SPANS / f"{lang}_train.tsv")
            + read_mhc_tsv(MHC_SPANS / f"{lang}_valid.tsv")
        )
        test_up = dedup(read_mhc_tsv(MHC_SPANS / f"{lang}_test.tsv"))
        test_set = set(test_up)
        overlap = sorted(set(train_up) & test_set)
        train_up = [i for i in train_up if i not in test_set]

        for name, upstream in (
            (f"mhclip_{lang}_train", train_up),
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
