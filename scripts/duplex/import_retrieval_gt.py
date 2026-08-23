#!/usr/bin/env python3
"""Restore the frozen 1-fps test GT from Retrieval-hate's canonical archive.

The large ``results/reproduction`` tree is intentionally gitignored and may be
absent after a fresh clone. Retrieval-hate stores the same rasterized labels in
its structured ``frame_gt_4fps/*.npz`` archives, including the exact ``y1``
arrays. This importer selects only the frozen test manifest and applies the
study's existing rule that a positive video with no usable span is excluded.
"""

from __future__ import annotations

import hashlib
import ast
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_gt_arrays import save_npz_deterministic, wav_header_duration  # noqa: E402
from frame_eval_common import build_gt_array  # noqa: E402
from hatemm_span_gold import parse_snippet  # noqa: E402
from mhclip_span_gold import parse_spans  # noqa: E402

SOURCE = Path("/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps")
OUT = REPO / "results" / "reproduction" / "gt"
SPLITS = REPO / "results" / "reproduction" / "splits"
CORPORA = {
    "hatemm": ("HateMM", "HateMM"),
    "mhclip_en": ("MHC", "MHC"),
    "mhclip_zh": ("MHC_zh", "MHC_zh"),
    "hateclipseg": ("HateClipSeg", "HateClipSeg"),
}
WAV_ROOT = Path("/home/jehc223/Retrieval-hate/data/AV2A_wav")
HATEMM_CSV = Path("/home/jehc223/Retrieval-hate/data/gt/HateMM/HateMM_annotation.csv")
MHC_TSV_ROOT = Path("/home/jehc223/Retrieval-hate/data/gt/mhc_votes")


def source_rows(corpus: str):
    """video_id -> (positive video label, raw temporal spans)."""
    if corpus == "hatemm":
        out = {}
        with HATEMM_CSV.open(newline="") as fh:
            for row in csv.DictReader(fh):
                vid = row["video_file_name"].rsplit(".", 1)[0]
                spans, _problems = parse_snippet(row["hate_snippet"])
                out[vid] = (row["label"].strip() == "Hate", spans)
        return out
    language = {"mhclip_en": "English", "mhclip_zh": "Chinese"}[corpus]
    path = MHC_TSV_ROOT / f"mhc_{language}_test.tsv"
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            label = row["Majority_Voting"].strip()
            out[row["Video_ID"].strip()] = (
                label in ("Hateful", "Offensive"),
                parse_spans(row["Duration"]),
            )
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for corpus, (source_name, wav_name) in CORPORA.items():
        ids = [x.strip() for x in
               (SPLITS / f"{corpus}_test.txt").read_text().splitlines()
               if x.strip()]
        rows = None if corpus == "hateclipseg" else source_rows(corpus)
        with np.load(SOURCE / f"{source_name}.npz", allow_pickle=True) as z:
            pos = {str(v): i for i, v in enumerate(z["video_ids"])}
            arrays = {}
            excluded = []
            for vid in ids:
                if vid not in pos:
                    raise KeyError(f"{corpus}: {vid} absent from canonical GT")
                i = pos[vid]
                if rows is None:
                    positive = bool(int(z["y_video"][i]))
                    raw = np.asarray(z["spans"][i], dtype=float).reshape(-1, 2)
                    spans = [(float(s), float(e)) for s, e in raw if e > s]
                else:
                    positive, raw = rows[vid]
                    spans = [(float(s), float(e)) for s, e in raw if e > s]
                if positive and not spans:
                    excluded.append(vid)
                    continue
                if not positive:
                    spans = []
                wav = WAV_ROOT / wav_name / f"{vid}.wav"
                if not wav.is_file():
                    raise FileNotFoundError(wav)
                arrays[vid] = build_gt_array(
                    spans, wav_header_duration(str(wav)), fps=1.0)

        path = OUT / f"{corpus}_test.npz"
        digest = save_npz_deterministic(str(path), arrays)
        sidecar = {
            "corpus": corpus,
            "split": "test",
            "fps": 1.0,
            "source": str(SOURCE / f"{source_name}.npz"),
            "source_sha256": sha256(SOURCE / f"{source_name}.npz"),
            "split_sha256": sha256(SPLITS / f"{corpus}_test.txt"),
            "npz_sha256": digest,
            "n_videos": len(arrays),
            "n_frames": sum(len(v) for v in arrays.values()),
            "n_positive_frames": sum(int(v.sum()) for v in arrays.values()),
            "excluded_positive_without_span": excluded,
        }
        (OUT / f"{corpus}_test.json").write_text(
            json.dumps(sidecar, indent=2) + "\n")
        print(corpus, sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
