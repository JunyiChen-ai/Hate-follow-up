#!/usr/bin/env python3
"""Train-only sparse K=16 VERA producer for the audited Relation-V6 input.

The candidate locations are a fixed, label-free uniform subset of the complete
one-second grid.  This file is separate from the frozen dense VERA adapter and
uses the exact-A/B-approved batch-2 inference implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import vera_adapter as legacy
import vera_fast_infer as fast
from hate_common import data as hdata

K = 16
INDEX_RULE = (
    "unique(round(linspace(0,L-1,min(K,L)))); "
    "L=ceil(min(media_duration,vggish_1s_n_frames)); K=16; "
    "videos without a decodable visual stream are excluded label-free"
)
VGG_INDEX = (legacy.REPO / "results/reproduction/features/vggish_1s/"
             "hateclipseg/index.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sparse_indices(length: int, k: int = K) -> np.ndarray:
    if length <= 0:
        raise ValueError("timeline length must be positive")
    count = min(k, length)
    # np.rint implements the requested round rule and is deterministic here.
    return np.unique(np.rint(np.linspace(0, length - 1, count)).astype(int))


def frozen_train_cohort():
    train = hdata.load_split("hateclipseg", "train")
    val = set(hdata.load_split("hateclipseg", "val"))
    test = set(hdata.load_split("hateclipseg", "test"))
    if len(train) != len(set(train)):
        raise RuntimeError("duplicate HateClipSeg train IDs")
    if set(train) & (val | test):
        raise RuntimeError("HateClipSeg frozen train intersects val/test")
    return train


def aligned_length(n_frames: int, duration: float) -> int:
    """Number of integer 1-fps starts inside both frozen and media support."""
    if n_frames <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid timeline/media support")
    return max(1, int(np.ceil(min(float(n_frames), float(duration)))))


def expected_starts(video_id, timeline, duration):
    length = aligned_length(int(timeline[video_id]["n_frames"]), duration)
    return sparse_indices(length).astype(float)


def valid_sparse_raw(path: Path, video_id: str, starts: np.ndarray,
                     n_frames: int) -> bool:
    try:
        row = json.loads(path.read_text())
        duration = float(row["duration"])
        segments = row["segments"]
        return (
            row.get("video_id") == video_id
            and np.isfinite(duration) and duration > 0
            and len(segments) == len(starts)
            and all(
                float(segment["start"]) == float(start)
                and 0 <= float(segment["start"]) < float(segment["end"])
                <= min(duration, float(n_frames))
                and segment.get("score") in (0, 1)
                and isinstance(segment.get("response"), str)
                for start, segment in zip(starts, segments)
            )
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def prompt_digest(prompts) -> str:
    canonical = json.dumps(prompts, ensure_ascii=False,
                           separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(raw_root: Path, manifest_path: Path, cohort, excluded,
                   selection, timeline, media_audit_path):
    raw_root = raw_root.resolve()
    expected = set(cohort)
    actual = {path.stem for path in raw_root.glob("*.json")}
    if actual != expected:
        raise RuntimeError(f"raw JSON set mismatch: missing={sorted(expected-actual)[:3]}, "
                           f"extra={sorted(actual-expected)[:3]}")
    files, aggregate = {}, []
    for video_id in sorted(cohort):
        path = raw_root / f"{video_id}.json"
        n_frames = int(timeline[video_id]["n_frames"])
        row = json.loads(path.read_text())
        starts = expected_starts(video_id, timeline, float(row["duration"]))
        if not valid_sparse_raw(path, video_id, starts, n_frames):
            raise RuntimeError(f"invalid sparse raw: {video_id}")
        digest = sha256(path)
        files[video_id] = {"sha256": digest}
        aggregate.append(f"{video_id}\t{digest}\n")
    payload = {
        "corpus": "hateclipseg",
        "split": "train",
        "k": K,
        "selection_cohort_ids": list(cohort),
        "excluded_train_ids": excluded,
        "media_audit": str(Path(media_audit_path).resolve()),
        "media_audit_sha256": sha256(Path(media_audit_path).resolve()),
        "root": str(raw_root),
        "files": files,
        "prompt": selection["prompts"],
        "prompt_sha256": prompt_digest(selection["prompts"]),
        "backend": selection["attention_backend"],
        "backbone": selection["backbone"],
        "index_rule": INDEX_RULE,
        "root_set_sha256": hashlib.sha256("".join(aggregate).encode()).hexdigest(),
        "label_access": "none",
        "cohort_policy": (
            "all frozen HateClipSeg train videos with a decodable visual stream; "
            "non-decodable train media excluded without label access; val/test zero intersection"
        ),
    }
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    return payload


def run(args):
    if args.corpus != "hateclipseg" or args.split != "train" or args.k != K:
        raise RuntimeError("this audited producer is fixed to hateclipseg/train/K16")
    raw_root = Path(args.raw_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    if manifest_path.parent == raw_root:
        raise RuntimeError("manifest must be outside raw root")
    raw_root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.prompt_json).read_text())
    if (selection.get("corpus") != "hateclipseg" or
            selection.get("selection_split") != "official-val" or
            not legacy.valid_selection(args.prompt_json, "hateclipseg")):
        raise RuntimeError("invalid frozen HateClipSeg prompt selection")
    timeline = json.loads(VGG_INDEX.read_text())
    frozen_cohort = frozen_train_cohort()
    if set(frozen_cohort) - set(timeline):
        raise RuntimeError("train cohort absent from frozen VGGish timeline")
    audit_path = Path(args.media_audit).resolve()
    audit = json.loads(audit_path.read_text())
    if (audit.get("corpus") != "hateclipseg" or audit.get("split") != "train"
            or audit.get("label_access") != "none"
            or audit.get("frozen_train_ids") != frozen_cohort
            or set(audit.get("records", {})) != set(frozen_cohort)):
        raise RuntimeError("invalid frozen label-free media audit")
    cohort, excluded = [], {}
    for video_id in frozen_cohort:
        record = audit["records"][video_id]
        path = Path(legacy.video_path("hateclipseg", video_id)).resolve()
        if (record.get("path") != str(path) or record.get("media_sha256") != sha256(path)):
            raise RuntimeError(f"media audit hash/path mismatch: {video_id}")
        if record.get("status") == "decodable_visual_stream":
            cohort.append(video_id)
        elif record.get("status") == "no_decodable_visual_stream":
            excluded[video_id] = {"reason": "no_decodable_visual_stream"}
        else:
            raise RuntimeError(f"unknown media audit status: {video_id}")
    if set(cohort) | set(excluded) != set(frozen_cohort) or set(cohort) & set(excluded):
        raise RuntimeError("decodable/excluded train partition invalid")
    if not cohort:
        raise RuntimeError("no decodable HateClipSeg train videos")
    print(f"audited visual cohort: {len(cohort)} included, {len(excluded)} excluded",
          flush=True)

    model, tokenizer, _ = legacy.load_model(selection["attention_backend"])
    for position, video_id in enumerate(cohort, 1):
        path = raw_root / f"{video_id}.json"
        n_frames = int(timeline[video_id]["n_frames"])
        reader = fast.ReusableVideoReader(legacy.video_path("hateclipseg", video_id))
        starts = expected_starts(video_id, timeline, reader.duration)
        if valid_sparse_raw(path, video_id, starts, n_frames):
            print(f"[{position}/{len(cohort)}] {video_id}: already complete", flush=True)
            continue
        aligned_end = min(reader.duration, float(n_frames))
        records = []
        for offset in range(0, len(starts), 2):
            batch_starts = starts[offset:offset + 2]
            images = [reader.frames(float(start), args.window, 8)
                      for start in batch_starts]
            predictions = fast.predict_batch(model, tokenizer, images,
                                             selection["prompts"], 2)
            for start, (score, response) in zip(batch_starts, predictions):
                records.append({"start": float(start),
                                "end": min(aligned_end, float(start) + args.window),
                                "score": score, "response": response})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"video_id": video_id,
                                         "duration": reader.duration,
                                         "segments": records}, ensure_ascii=False))
        os.replace(temporary, path)
        print(f"[{position}/{len(cohort)}] {video_id}: {len(records)} sparse windows",
              flush=True)
    build_manifest(raw_root, manifest_path, cohort, excluded, selection, timeline,
                   audit_path)
    print(f"wrote audited manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="hateclipseg")
    parser.add_argument("--split", default="train")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument(
        "--media-audit",
        default=("results/reproduction/official_val/final/vera/hateclipseg/"
                 "seed_234/train_sparse_k16/media_audit.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
