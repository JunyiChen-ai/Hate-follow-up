"""Build the frame-level ground-truth arrays for HateClipSeg (our test split).

Protocol: docs/duplex/FRAME_EVAL_PROTOCOL.md, HateClipSeg section. This is a
sibling of ``build_gt_arrays.py`` rather than a fourth corpus inside it: the
three corpora there are scored against *hate spans* drawn on an otherwise
unannotated timeline, whereas HateClipSeg annotates the timeline exhaustively,
segment by segment, so the conversion rules differ enough to deserve their own
file. Nothing in ``build_gt_arrays.py`` changes, and none of its three arrays
move a byte.

Two arrays are produced over the same cohort and the same grid:

    results/reproduction/gt/hateclipseg_test.npz
        PRIMARY. A frame is positive iff the segment covering it is offensive
        under the union rule -- any of the five non-normal dimensions
        (hateful, insulting, sexual, violence, harm) set. This is the rule
        ``sentinel_localization_pilot.is_offensive_union`` already uses, so
        the frame gold and the earlier pilot cohort agree by construction.

    results/reproduction/gt/hateclipseg_test_hateful_strict.npz
        SENSITIVITY. Positive iff dimension 1 (hateful) alone is set. Same
        videos, same frame counts, so the two are directly comparable; a
        method's numbers may be reported on both but the primary is primary.

Both are written with the deterministic zip layout of ``build_gt_arrays.py``
(fixed member order, fixed timestamps, no compression), so re-running on the
same inputs reproduces the same SHA256 byte for byte.

CPU only, no model calls. Sources:

    segments   idea-stage/pilots/b1_coverage_audit/data/segment_level_annotation.csv
    cohort     results/reproduction/splits/hateclipseg_test.txt
    durations  results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl
               (wav_duration), with the wav header as fallback
    container  results/hateclipseg/audio_meta.jsonl (clock cross-check only)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from datetime import datetime, timezone

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

import numpy as np  # noqa: E402

from build_gt_arrays import (  # noqa: E402
    read_jsonl,
    save_npz_deterministic,
    wav_header_duration,
)
from frame_eval_common import frame_times  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS, "reproduction", "gt")
SPLIT_DIR = os.path.join(RESULTS, "reproduction", "splits")

PROTOCOL_DOC = "docs/duplex/FRAME_EVAL_PROTOCOL.md"
FPS = 1.0

GOLD_CSV = os.path.join(
    PROJECT_ROOT, "idea-stage", "pilots", "b1_coverage_audit", "data",
    "segment_level_annotation.csv")
CHUNKS_JSONL = os.path.join(
    RESULTS, "interleaved_timeline", "hateclipseg", "timestamped_chunks.jsonl")
AUDIO_META = os.path.join(RESULTS, "hateclipseg", "audio_meta.jsonl")
SPLIT_FILE = os.path.join(SPLIT_DIR, "hateclipseg_test.txt")

WAV_DIRS = [
    "{data_root}/HateClipSeg/wav",
    os.path.join(RESULTS, "hateclipseg", "wav"),
]

# A video whose last usable segment ends more than this many seconds past the
# media is excluded: see CLOCK_RULE below. One second is exactly one frame on
# the 1 fps grid, so an overshoot under it cannot move any frame label. The
# threshold is fixed by the grid, not chosen to hit a cohort size.
CLOCK_TOLERANCE_S = 1.0

CLOCK_RULE = (
    "exclude a video whose last usable segment ends more than 1.0 s (one "
    "frame) past the media duration, measured against the container where a "
    "container duration is known and against the wav otherwise"
)

# Label dimension order in the segment annotation.
DIMS = ("normal", "hateful", "insulting", "sexual", "violence", "harm")


# ------------------------------------------------------------------ parsing
def is_offensive_union(label):
    """Union rule: any of the five non-normal dimensions set."""
    return any(int(x) == 1 for x in label[1:6])


def is_hateful_strict(label):
    """Sensitivity rule: the hateful dimension alone."""
    return int(label[1]) == 1


def load_segments():
    """video id -> list of (start, end, label) in file order."""
    csv.field_size_limit(1 << 30)
    out = {}
    with open(GOLD_CSV, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels = ast.literal_eval(row["Segment-Level Label"])
            spans = ast.literal_eval(row["Segment Timestamp"])
            if len(labels) != len(spans):
                raise SystemExit(
                    "label/timestamp length mismatch for %s" % row["Video Id"])
            out[row["Video Id"].strip()] = [
                (float(a), float(b), list(lab))
                for (a, b), lab in zip(spans, labels)
            ]
    return out


def load_durations(ids, data_root):
    """id -> (wav_duration, source); plus the ids with no audio at all."""
    from_chunks = {}
    for rec in read_jsonl(CHUNKS_JSONL):
        dur = rec.get("wav_duration")
        if dur:
            from_chunks[rec["video_id"]] = float(dur)
    wav_dirs = [d.format(data_root=data_root) for d in WAV_DIRS]
    out, source, no_audio = {}, {}, []
    for vid in ids:
        if vid in from_chunks:
            out[vid], source[vid] = from_chunks[vid], "timestamped_chunks"
            continue
        found = None
        for wav_dir in wav_dirs:
            cand = os.path.join(wav_dir, vid + ".wav")
            if os.path.isfile(cand):
                found = cand
                break
        if found is None:
            no_audio.append(vid)
            continue
        out[vid], source[vid] = wav_header_duration(found), "wav_header"
    return out, source, no_audio


def load_container_durations():
    """id -> container duration, for the annotation-clock cross-check only."""
    out = {}
    for path in (AUDIO_META, CHUNKS_JSONL):
        if not os.path.exists(path):
            continue
        for rec in read_jsonl(path):
            dur = rec.get("container_duration")
            if dur and rec["video_id"] not in out:
                out[rec["video_id"]] = float(dur)
    return out


def read_split_ids(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


# ------------------------------------------------------------------ tiling
def tiling_report(segments_by_video):
    """Measure how well the segments actually tile, corpus-wide.

    Reported rather than assumed: the frame rule below only needs *some*
    covering segment, but whether one frame can be covered twice, and whether
    a frame can be covered not at all, are facts about the annotation and are
    measured here before any rule is frozen.
    """
    rep = {
        "videos": len(segments_by_video),
        "segments": 0,
        "videos_not_starting_at_zero": 0,
        "adjacent_gaps": 0,
        "adjacent_gap_seconds": 0.0,
        "adjacent_overlaps": 0,
        "adjacent_overlap_seconds": 0.0,
        "degenerate_segments": 0,
        "videos_with_degenerate_segment": 0,
        "degenerate_segments_not_last": 0,
    }
    for vid, segs in sorted(segments_by_video.items()):
        rep["segments"] += len(segs)
        if segs and abs(segs[0][0]) > 1e-9:
            rep["videos_not_starting_at_zero"] += 1
        has_degenerate = False
        for i, (s, e, _lab) in enumerate(segs):
            if e <= s:
                rep["degenerate_segments"] += 1
                has_degenerate = True
                if i != len(segs) - 1:
                    rep["degenerate_segments_not_last"] += 1
        if has_degenerate:
            rep["videos_with_degenerate_segment"] += 1
        for i in range(len(segs) - 1):
            delta = segs[i + 1][0] - segs[i][1]
            if delta > 1e-6:
                rep["adjacent_gaps"] += 1
                rep["adjacent_gap_seconds"] += delta
            elif delta < -1e-6:
                rep["adjacent_overlaps"] += 1
                rep["adjacent_overlap_seconds"] += -delta
    rep["adjacent_gap_seconds"] = round(rep["adjacent_gap_seconds"], 6)
    rep["adjacent_overlap_seconds"] = round(rep["adjacent_overlap_seconds"], 6)
    return rep


def clock_audit(segments_by_video, durations, containers):
    """Annotation-clock overshoot over every video with audio, not just test.

    The clock rule is frozen for the whole corpus, so its effect is measured
    over the whole corpus. Reporting only the test-split firings would hide
    that most of the affected videos happen to sit in train, where they
    matter to any baseline that builds train-side frame targets.
    """
    rep = {"videos_with_audio": len(durations), "over_tolerance": [],
           "audio_shorter_than_container": []}
    for vid, wav in sorted(durations.items()):
        segs = segments_by_video.get(vid) or []
        usable = [(s, e) for s, e, _lab in segs if e > s]
        if not usable:
            continue
        container = containers.get(vid)
        reference = container if container else wav
        last_end = max(e for _s, e in usable)
        if last_end - reference > CLOCK_TOLERANCE_S:
            rep["over_tolerance"].append({
                "video_id": vid,
                "last_segment_end": round(last_end, 3),
                "wav_duration": round(wav, 3),
                "container_duration": (round(container, 3) if container
                                       else None),
                "overshoot_seconds": round(last_end - reference, 3),
            })
        elif container and container - wav > CLOCK_TOLERANCE_S:
            rep["audio_shorter_than_container"].append({
                "video_id": vid,
                "container_duration": round(container, 3),
                "wav_duration": round(wav, 3),
                "seconds_of_annotation_past_the_wav": round(
                    min(last_end, container) - wav, 3),
            })
    rep["n_over_tolerance"] = len(rep["over_tolerance"])
    rep["n_audio_shorter_than_container"] = len(
        rep["audio_shorter_than_container"])
    return rep


def build_frame_labels(segments, duration, positive_fn):
    """Frame labels plus a per-frame coverage count, on the 1 fps grid.

    A frame is positive iff some segment covering it -- half-open,
    ``start <= t < end`` -- is positive under ``positive_fn``. Overlapping
    segments union, so a positive segment wins over a negative one; the
    coverage count is returned so a caller can report how often that rule
    could fire at all. Degenerate segments (``end <= start``) cover nothing
    and are dropped upstream, where the drop is counted.
    """
    t = frame_times(duration, FPS)
    labels = np.zeros(len(t), dtype=np.uint8)
    covered = np.zeros(len(t), dtype=np.int32)
    for start, end, label in segments:
        if not (end > start):
            continue
        hit = (t >= start) & (t < end)
        covered[hit] += 1
        if positive_fn(label):
            labels[hit] = 1
    return labels, covered


# ------------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--data-root", default="/home/jehc223/data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "build_gt_arrays_hateclipseg.log")
    handle = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        handle.write(msg + "\n")

    segments_by_video = load_segments()
    log("HateClipSeg frame gold, protocol %s" % PROTOCOL_DOC)
    log("fps=%g, half-open [start, end), duration = wav_duration" % FPS)
    log("")
    log("--- segment tiling, all %d annotated videos ---"
        % len(segments_by_video))
    tiling = tiling_report(segments_by_video)
    for key in ("videos", "segments", "videos_not_starting_at_zero",
                "adjacent_gaps", "adjacent_gap_seconds",
                "adjacent_overlaps", "adjacent_overlap_seconds",
                "degenerate_segments", "videos_with_degenerate_segment",
                "degenerate_segments_not_last"):
        log("  %-34s %s" % (key, tiling[key]))

    split_ids = read_split_ids(SPLIT_FILE)
    durations, dur_source, no_audio = load_durations(split_ids, args.data_root)
    containers = load_container_durations()

    all_ids = sorted(segments_by_video)
    all_durations, _src, _na = load_durations(all_ids, args.data_root)
    clock = clock_audit(segments_by_video, all_durations, containers)
    log("")
    log("--- annotation clock, all %d annotated videos with audio ---"
        % clock["videos_with_audio"])
    log("  over tolerance (%.1f s)             %d  %s"
        % (CLOCK_TOLERANCE_S, clock["n_over_tolerance"],
           [e["video_id"] for e in clock["over_tolerance"]]))
    log("  audio shorter than container       %d  %s"
        % (clock["n_audio_shorter_than_container"],
           [e["video_id"] for e in clock["audio_shorter_than_container"]]))

    excluded_clock = []
    excluded_no_segments = []
    per_video = {}
    arrays = {"union": {}, "hateful_strict": {}}
    counts = {
        "test_split_size": len(split_ids),
        "videos_without_audio": len(no_audio),
        "videos_included": 0,
        "videos_excluded_annotation_clock": 0,
        "videos_excluded_no_usable_segment": 0,
        "frames_total": 0,
        "frames_uncovered": 0,
        "frames_multiply_covered": 0,
        "degenerate_segments_dropped": 0,
        "videos_with_degenerate_segment": 0,
        "seconds_total": 0.0,
        "segments_reaching_past_audio_end": 0,
        "segments_wholly_past_audio_end": 0,
    }
    for key in arrays:
        counts["frames_positive_%s" % key] = 0
        counts["videos_all_positive_%s" % key] = 0
        counts["videos_all_negative_%s" % key] = 0
        counts["videos_both_classes_%s" % key] = 0

    for vid in sorted(durations):
        segs = segments_by_video.get(vid)
        if not segs:
            excluded_no_segments.append({"video_id": vid,
                                         "reason": "no annotation row"})
            counts["videos_excluded_no_usable_segment"] += 1
            continue
        usable = [(s, e, lab) for s, e, lab in segs if e > s]
        n_degenerate = len(segs) - len(usable)
        if not usable:
            excluded_no_segments.append(
                {"video_id": vid,
                 "reason": "every segment degenerate (end <= start)"})
            counts["videos_excluded_no_usable_segment"] += 1
            continue

        wav = durations[vid]
        container = containers.get(vid)
        reference = container if container else wav
        last_end = max(e for _s, e, _lab in usable)
        overshoot = last_end - reference
        if overshoot > CLOCK_TOLERANCE_S:
            excluded_clock.append({
                "video_id": vid,
                "last_segment_end": round(last_end, 3),
                "wav_duration": round(wav, 3),
                "container_duration": (round(container, 3) if container
                                       else None),
                "overshoot_seconds": round(overshoot, 3),
                "reason": ("annotation clock runs past the media by more "
                           "than one frame"),
            })
            counts["videos_excluded_annotation_clock"] += 1
            continue

        if n_degenerate:
            counts["degenerate_segments_dropped"] += n_degenerate
            counts["videos_with_degenerate_segment"] += 1
        counts["segments_reaching_past_audio_end"] += sum(
            1 for _s, e, _l in usable if e > wav)
        counts["segments_wholly_past_audio_end"] += sum(
            1 for s, _e, _l in usable if s >= wav)

        entry = {
            "duration": round(wav, 3),
            "duration_source": dur_source[vid],
            "container_duration": (round(container, 3) if container else None),
            "n_segments": len(segs),
            "n_segments_usable": len(usable),
            "n_degenerate_segments": n_degenerate,
            "annotation_overshoot_seconds": round(overshoot, 3),
        }
        n_frames = None
        for key, fn in (("union", is_offensive_union),
                        ("hateful_strict", is_hateful_strict)):
            labels, covered = build_frame_labels(usable, wav, fn)
            arrays[key][vid] = labels
            n_pos = int(labels.sum())
            entry["n_frames"] = int(len(labels))
            entry["n_positive_frames_%s" % key] = n_pos
            entry["n_positive_segments_%s" % key] = sum(
                1 for _s, _e, lab in usable if fn(lab))
            counts["frames_positive_%s" % key] += n_pos
            if n_pos == 0:
                counts["videos_all_negative_%s" % key] += 1
            elif n_pos == len(labels):
                counts["videos_all_positive_%s" % key] += 1
            else:
                counts["videos_both_classes_%s" % key] += 1
            if key == "union":
                n_frames = int(len(labels))
                entry["n_frames_uncovered"] = int((covered == 0).sum())
                entry["n_frames_multiply_covered"] = int((covered > 1).sum())
                counts["frames_uncovered"] += entry["n_frames_uncovered"]
                counts["frames_multiply_covered"] += \
                    entry["n_frames_multiply_covered"]

        per_video[vid] = entry
        counts["videos_included"] += 1
        counts["frames_total"] += n_frames
        counts["seconds_total"] += wav

    counts["seconds_total"] = round(counts["seconds_total"], 3)

    written = {}
    for key, name in (("union", "hateclipseg_test"),
                      ("hateful_strict", "hateclipseg_test_hateful_strict")):
        npz_path = os.path.join(args.out_dir, name + ".npz")
        written[key] = {
            "name": name,
            "path": npz_path,
            "sha256": save_npz_deterministic(npz_path, arrays[key]),
        }

    sidecar = {
        "protocol": PROTOCOL_DOC,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": "hateclipseg",
        "split": "test (ours, seed 234, 80/20; HateClipSeg publishes no ids)",
        "fps": FPS,
        "frame_grid": "t = 0, 1, 2, ... while t < wav_duration",
        "span_convention": "half-open [start, end)",
        "positive_rule_primary": (
            "the covering segment is offensive under the union rule over "
            "dims 1..5 (hateful, insulting, sexual, violence, harm)"),
        "positive_rule_sensitivity": "the covering segment sets dim 1 (hateful)",
        "label_dimensions": list(DIMS),
        "overlap_rule": (
            "union: a frame covered by both a positive and a negative segment "
            "is positive. Measured need for it corpus-wide: 0 adjacent "
            "overlaps in 11714 segments"),
        "uncovered_frame_rule": (
            "a frame no segment covers is negative. The annotation tiles from "
            "0 with no gaps, so this fires only in the sub-second tail after "
            "the last segment ends"),
        "degenerate_segment_rule": (
            "segments with end <= start are dropped and counted. Every one in "
            "this corpus is the final segment of its video and its end equals "
            "the media duration, so dropping it removes no covered interval"),
        "annotation_clock_rule": CLOCK_RULE,
        "clock_tolerance_seconds": CLOCK_TOLERANCE_S,
        "duration_source": ("wav_duration from the timestamped-chunk manifest "
                            "where it has an entry, otherwise the wav header"),
        "cohort_definition": os.path.relpath(SPLIT_FILE, PROJECT_ROOT),
        "sources": {
            "segments": os.path.relpath(GOLD_CSV, PROJECT_ROOT),
            "durations": os.path.relpath(CHUNKS_JSONL, PROJECT_ROOT),
            "container_durations": os.path.relpath(AUDIO_META, PROJECT_ROOT),
            "cohort": os.path.relpath(SPLIT_FILE, PROJECT_ROOT),
        },
        "arrays": {
            "primary": {
                "npz": os.path.relpath(written["union"]["path"], PROJECT_ROOT),
                "npz_sha256": written["union"]["sha256"],
                "npz_bytes": os.path.getsize(written["union"]["path"]),
            },
            "sensitivity_hateful_strict": {
                "npz": os.path.relpath(written["hateful_strict"]["path"],
                                       PROJECT_ROOT),
                "npz_sha256": written["hateful_strict"]["sha256"],
                "npz_bytes": os.path.getsize(
                    written["hateful_strict"]["path"]),
            },
        },
        "segment_tiling_all_annotated_videos": tiling,
        "annotation_clock_audit_all_videos_with_audio": clock,
        "counts": counts,
        "excluded_annotation_clock": excluded_clock,
        "excluded_no_usable_segment": excluded_no_segments,
        "test_split_videos_without_audio": sorted(no_audio),
        "cohort_note": (
            "The cohort is our test split minus the videos excluded by the "
            "annotation-clock rule. Media for 41 of the 435 annotated videos "
            "is not held locally and never entered the split; that attrition "
            "is recorded in results/reproduction/splits/manifest_report.json. "
            "Any number computed against one SHA256 below is not comparable "
            "with a number computed against another."),
        "per_video": per_video,
    }
    side_path = os.path.join(args.out_dir, "hateclipseg_test.json")
    with open(side_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    log("")
    log("--- cohort ---")
    log("test split                 : %d" % counts["test_split_size"])
    log("  without audio            : %d %s"
        % (counts["videos_without_audio"], sorted(no_audio)))
    log("  excluded, annotation clock: %d  %s"
        % (counts["videos_excluded_annotation_clock"],
           [e["video_id"] for e in excluded_clock]))
    log("  excluded, no usable segment: %d  %s"
        % (counts["videos_excluded_no_usable_segment"],
           [e["video_id"] for e in excluded_no_segments]))
    log("  included                 : %d" % counts["videos_included"])
    log("degenerate segments dropped: %d in %d videos"
        % (counts["degenerate_segments_dropped"],
           counts["videos_with_degenerate_segment"]))
    log("segments past audio end    : %d reaching past, %d wholly past"
        % (counts["segments_reaching_past_audio_end"],
           counts["segments_wholly_past_audio_end"]))
    log("frames                     : %d total, %d uncovered, %d covered twice"
        % (counts["frames_total"], counts["frames_uncovered"],
           counts["frames_multiply_covered"]))
    log("audio seconds              : %.1f" % counts["seconds_total"])
    log("")
    log("%-16s %8s %8s %10s %10s %10s %10s"
        % ("array", "videos", "frames", "pos", "pos_rate", "both_cls",
           "all_neg"))
    for key, label in (("union", "primary"),
                       ("hateful_strict", "sensitivity")):
        log("%-16s %8d %8d %10d %10.4f %10d %10d"
            % (label, counts["videos_included"], counts["frames_total"],
               counts["frames_positive_%s" % key],
               counts["frames_positive_%s" % key]
               / float(max(counts["frames_total"], 1)),
               counts["videos_both_classes_%s" % key],
               counts["videos_all_negative_%s" % key]))
    log("")
    for key, label in (("union", "primary"),
                       ("hateful_strict", "sensitivity")):
        log("%-12s %s  %s" % (label, written[key]["sha256"],
                              os.path.relpath(written[key]["path"],
                                              PROJECT_ROOT)))
    log("sidecar      %s" % os.path.relpath(side_path, PROJECT_ROOT))
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
