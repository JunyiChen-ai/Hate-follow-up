"""Build the released frame-level ground-truth arrays for the reproduction study.

Protocol: docs/duplex/FRAME_EVAL_PROTOCOL.md (frozen 2026-08-18). This
script is the only producer of the arrays every method in the study is
scored against, so that no method can quietly bring its own gold.

Outputs, one per corpus:

    results/reproduction/gt/hatemm_test.npz
    results/reproduction/gt/mhclip_en_test.npz
    results/reproduction/gt/mhclip_zh_test.npz

Each npz holds one uint8 array per included video, keyed by video_id, on
the 1 fps grid t = 0, 1, 2, ... while t < duration, with 1 marking a
frame inside a hate span under half-open containment. Each npz has a
JSON sidecar (<name>.json) carrying the cohort counts, the exclusion
lists with reasons, the per-video frame and positive counts, and the
SHA256 of the npz itself.

The npz is written with a deterministic zip layout (fixed member order,
fixed timestamps, no compression), so re-running this script on the same
inputs reproduces the same SHA256 byte for byte.

CPU only, no model calls. Sources:

    HateMM   spans     results/hatemm_localization/span_gold.json
             durations results/hatemm_localization/timestamped_chunks.jsonl
    MHC EN   spans     results/mhclip_localization/span_gold_en.json
             durations results/interleaved_timeline/mhclip_en/timestamped_chunks.jsonl
    MHC ZH   spans     results/mhclip_localization/span_gold_zh.json
             durations results/interleaved_timeline/mhclip_zh/timestamped_chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
from numpy.lib import format as npformat

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

from frame_eval_common import build_gt_array, frame_times  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS, "reproduction", "gt")
SPLIT_DIR = os.path.join(RESULTS, "reproduction", "splits")

PROTOCOL_DOC = "docs/duplex/FRAME_EVAL_PROTOCOL.md"
FPS = 1.0

MHC_POSITIVE_LABELS = ("Hateful", "Offensive")

# Wav directories searched, in order, when a test video is absent from the
# timestamped-chunk manifest. The Phase 0 media pull added test videos the
# frozen testruns never covered, so the chunk manifests no longer define the
# cohort; the split manifests do, and the duration comes off the wav header.
WAV_DIRS = {
    "hatemm": [os.path.join(RESULTS, "testruns", "hatemm", "wav"),
               "{data_root}/HateMM/wav"],
    "en": [os.path.join(RESULTS, "testruns", "mhclip_en", "wav"),
           "{data_root}/Multihateclip/English/wav"],
    "zh": [os.path.join(RESULTS, "testruns", "mhclip_zh", "wav"),
           "{data_root}/Multihateclip/Chinese/wav"],
}


# ------------------------------------------------------------------- io
def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def durations_from_chunks(path):
    """video_id -> wav_duration, from a timestamped-chunk manifest."""
    out = {}
    for rec in read_jsonl(path):
        vid = rec["video_id"]
        dur = rec.get("wav_duration")
        if dur is None or float(dur) <= 0:
            raise SystemExit("no positive wav_duration for %s in %s"
                             % (vid, path))
        if vid in out:
            raise SystemExit("duplicate video_id %s in %s" % (vid, path))
        out[vid] = float(dur)
    return out


def wav_header_duration(path):
    """Duration in seconds straight off the wav header, no decode."""
    import wave
    with wave.open(path, "rb") as handle:
        frames, rate = handle.getnframes(), handle.getframerate()
    if rate <= 0:
        raise SystemExit("non-positive sample rate in %s" % path)
    return frames / float(rate)


def read_split_ids(name):
    """Frozen split manifest -> ordered list of video ids."""
    path = os.path.join(SPLIT_DIR, name)
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_durations(ids, chunk_path, code, data_root):
    """video_id -> (duration, source) for every id that has audio on disk.

    The timestamped-chunk manifest wins where it has an entry, so every
    video the frozen runs already covered keeps byte-identical numbers.
    Videos the manifest never saw -- the test media that only arrived with
    the Phase 0 pull -- fall back to their wav header.
    """
    from_chunks = durations_from_chunks(chunk_path)
    wav_dirs = [d.format(data_root=data_root) for d in WAV_DIRS[code]]
    out, source, no_audio = {}, {}, []
    for vid in ids:
        if vid in from_chunks:
            out[vid] = from_chunks[vid]
            source[vid] = "timestamped_chunks"
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
        out[vid] = wav_header_duration(found)
        source[vid] = "wav_header"
    return out, source, no_audio


def save_npz_deterministic(path, arrays):
    """np.savez-compatible archive with a byte-reproducible layout."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            npformat.write_array(buf, np.asarray(arrays[name]),
                                 allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy",
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --------------------------------------------------------------- corpora
def clean_spans(raw_spans):
    """(kept, n_degenerate) — degenerate spans (end <= start) removed."""
    kept, degenerate = [], 0
    for span in raw_spans or []:
        start, end = float(span[0]), float(span[1])
        if end <= start:
            degenerate += 1
            continue
        kept.append((start, end))
    return kept, degenerate


def span_range_notes(spans, duration):
    """How many spans reach past, or start past, the end of the audio."""
    past_end = sum(1 for s, e in spans if e > duration)
    wholly_past = sum(1 for s, e in spans if s >= duration)
    return past_end, wholly_past


def collect_hatemm(data_root):
    """HateMM test_clean: video-level label from the id prefix."""
    span_path = os.path.join(RESULTS, "hatemm_localization", "span_gold.json")
    chunk_path = os.path.join(RESULTS, "hatemm_localization",
                              "timestamped_chunks.jsonl")
    gold = json.load(open(span_path, encoding="utf-8"))
    spans_by_video = gold["spans"]
    split_ids = read_split_ids("hatemm_test.txt")
    durations, dur_source, no_audio = resolve_durations(
        split_ids, chunk_path, "hatemm", data_root)

    records = []
    for vid in sorted(durations):
        is_hate = vid.startswith("hate_video_")
        kept, degenerate = clean_spans(spans_by_video.get(vid))
        records.append({
            "video_id": vid,
            "label": "hate" if is_hate else "non_hate",
            "positive_video": is_hate,
            "spans": kept,
            "n_degenerate_spans": degenerate,
            "duration": durations[vid],
            "duration_source": dur_source[vid],
        })
    return {
        "corpus": "hatemm",
        "split": "test_clean",
        "sources": {
            "spans": os.path.relpath(span_path, PROJECT_ROOT),
            "durations": os.path.relpath(chunk_path, PROJECT_ROOT),
            "cohort": os.path.relpath(os.path.join(SPLIT_DIR,
                                                   "hatemm_test.txt"),
                                      PROJECT_ROOT),
        },
        "records": records,
        "split_size": len(split_ids),
        "upstream_split_size": int(gold.get("split_videos")
                                   or len(split_ids)),
        "absent_from_local_mirror": [],
        "missing_media": sorted(no_audio),
        "gold_from_upstream_tsv_only": [],
        "label_field": "id prefix (hate_video_* / non_hate_video_*)",
    }


def upstream_test_ids(data_root, code):
    """Video ids in the upstream test TSV, or None if it is not readable.

    span_gold_{en,zh}.json only covers videos the local annotation mirror
    kept, so it under-reports the true split size. Reading the upstream
    TSV separates the two losses: videos absent from the mirror, and
    videos in the mirror whose media has not been fetched yet.
    """
    path = os.path.join(data_root, "Multihateclip", "upstream_spans",
                        "%s_test.tsv" % code)
    if not os.path.exists(path):
        return None
    import csv
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["Video_ID"].strip()
                for row in csv.DictReader(handle, delimiter="\t")}


def upstream_test_rows(data_root, code):
    """video_id -> {majority_label, spans} from the upstream test TSV.

    span_gold_{en,zh}.json only keeps videos that are *also* in the local
    annotation mirror, but the gold itself -- the majority vote and the
    Duration spans -- comes from the upstream TSV alone; the mirror only
    supplies a cross-check label. So a test video absent from the mirror
    still has complete gold upstream, and is read from here rather than
    dropped for want of a mirror row.
    """
    path = os.path.join(data_root, "Multihateclip", "upstream_spans",
                        "%s_test.tsv" % code)
    if not os.path.exists(path):
        return {}
    sys.path.insert(0, _THIS)
    from mhclip_span_gold import parse_spans  # noqa: E402
    import csv
    out = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            out[row["Video_ID"].strip()] = {
                "majority_label": row["Majority_Voting"].strip(),
                "spans": parse_spans(row["Duration"]),
            }
    return out


def collect_mhclip(code, data_root):
    """MultiHateClip EN or ZH: majority-vote label from the upstream TSVs."""
    span_path = os.path.join(RESULTS, "mhclip_localization",
                             "span_gold_%s.json" % code)
    chunk_path = os.path.join(RESULTS, "interleaved_timeline",
                              "mhclip_%s" % code, "timestamped_chunks.jsonl")
    gold = json.load(open(span_path, encoding="utf-8"))
    by_video = {v["video_id"]: v for v in gold["videos"]}
    upstream_rows = upstream_test_rows(data_root, code)

    # The cohort is the frozen split manifest: upstream test videos whose
    # media is present locally.
    split_ids = read_split_ids("mhclip_%s_test.txt" % code)
    durations, dur_source, no_audio = resolve_durations(
        split_ids, chunk_path, code, data_root)

    upstream_ids = upstream_test_ids(data_root, code)
    mirror_test_ids = {v["video_id"] for v in gold["videos"]
                       if "test" in [v["split"]]
                       + list(v.get("extra_splits") or [])}
    absent_from_mirror = (sorted(upstream_ids - mirror_test_ids)
                          if upstream_ids is not None else None)

    records = []
    not_in_gold = []
    not_in_test = []
    from_upstream_only = []
    for vid in sorted(durations):
        entry = by_video.get(vid)
        if entry is not None:
            splits = [entry["split"]] + list(entry.get("extra_splits") or [])
            if "test" not in splits:
                not_in_test.append({"video_id": vid, "splits": splits})
                continue
            raw_spans, label = entry["spans"], entry["majority_label"]
        else:
            row = upstream_rows.get(vid)
            if row is None:
                not_in_gold.append(vid)
                continue
            from_upstream_only.append(vid)
            splits = ["test"]
            raw_spans, label = row["spans"], row["majority_label"]
        kept, degenerate = clean_spans(raw_spans)
        records.append({
            "video_id": vid,
            "label": label,
            "positive_video": label in MHC_POSITIVE_LABELS,
            "spans": kept,
            "n_degenerate_spans": degenerate,
            "duration": durations[vid],
            "duration_source": dur_source[vid],
            "listed_in_splits": splits,
        })
    return {
        "corpus": "mhclip_%s" % code,
        "split": "test",
        "sources": {
            "spans": os.path.relpath(span_path, PROJECT_ROOT),
            "spans_fallback": "Multihateclip/upstream_spans/%s_test.tsv"
                              % code,
            "durations": os.path.relpath(chunk_path, PROJECT_ROOT),
            "cohort": os.path.relpath(
                os.path.join(SPLIT_DIR, "mhclip_%s_test.txt" % code),
                PROJECT_ROOT),
        },
        "records": records,
        "not_in_span_gold": not_in_gold,
        "not_in_test_split": not_in_test,
        "split_size": len(mirror_test_ids),
        "missing_media": sorted(
            ((set(upstream_ids) if upstream_ids is not None else set())
             | set(no_audio)) - set(durations)),
        "upstream_split_size": (len(upstream_ids)
                                if upstream_ids is not None else None),
        "absent_from_local_mirror": absent_from_mirror,
        "gold_from_upstream_tsv_only": from_upstream_only,
        "label_field": "upstream Majority_Voting (Hateful/Offensive/Normal)",
    }


# ----------------------------------------------------------------- build
def build(corpus, out_dir, log):
    name = corpus["corpus"] + "_test"
    arrays = {}
    per_video = {}
    excluded_no_span = []
    normal_with_span = []
    absent_mirror = corpus.get("absent_from_local_mirror")
    counts = {
        "upstream_test_split_size": corpus.get("upstream_split_size"),
        "test_videos_absent_from_local_annotation_mirror":
            None if absent_mirror is None else len(absent_mirror),
        "test_split_size_in_span_gold": corpus.get("split_size"),
        "videos_missing_local_media":
            len(corpus.get("missing_media") or []),
        "videos_available": len(corpus["records"]),
        "videos_with_gold_from_upstream_tsv_only":
            len(corpus.get("gold_from_upstream_tsv_only") or []),
        "videos_with_duration_from_wav_header": sum(
            1 for r in corpus["records"]
            if r.get("duration_source") == "wav_header"),
        "videos_included": 0,
        "videos_excluded_positive_without_span": 0,
        "videos_positive_with_span": 0,
        "videos_all_negative": 0,
        "frames_total": 0,
        "frames_positive": 0,
        "frames_negative": 0,
        "seconds_total": 0.0,
        "degenerate_spans_dropped": 0,
        "videos_with_degenerate_span": 0,
        "spans_reaching_past_audio_end": 0,
        "spans_wholly_past_audio_end": 0,
    }
    by_label = {}

    for rec in corpus["records"]:
        vid = rec["video_id"]
        spans = rec["spans"]
        if rec["n_degenerate_spans"]:
            counts["degenerate_spans_dropped"] += rec["n_degenerate_spans"]
            counts["videos_with_degenerate_span"] += 1

        # Rule (b): a video the annotators called Hateful/Offensive but
        # left without any usable span carries no localization gold. It
        # is excluded, not silently turned into an all-negative video.
        if rec["positive_video"] and not spans:
            excluded_no_span.append({
                "video_id": vid,
                "label": rec["label"],
                "reason": "positive video with no usable span",
                "n_degenerate_spans": rec["n_degenerate_spans"],
            })
            counts["videos_excluded_positive_without_span"] += 1
            continue

        # Rule (a): a Normal-majority video that nonetheless carries
        # leftover spans is all-negative for the whole video — the
        # video-level majority vote governs.
        use_spans = spans
        if (not rec["positive_video"]) and spans:
            normal_with_span.append({
                "video_id": vid,
                "label": rec["label"],
                "n_spans_ignored": len(spans),
            })
            use_spans = []

        past_end, wholly_past = span_range_notes(use_spans, rec["duration"])
        counts["spans_reaching_past_audio_end"] += past_end
        counts["spans_wholly_past_audio_end"] += wholly_past

        labels = build_gt_array(use_spans, rec["duration"], fps=FPS)
        arrays[vid] = labels
        n_pos = int(labels.sum())
        per_video[vid] = {
            "label": rec["label"],
            "duration": round(rec["duration"], 3),
            "duration_source": rec.get("duration_source"),
            "n_frames": int(len(labels)),
            "n_positive_frames": n_pos,
            "n_spans": len(use_spans),
        }
        counts["videos_included"] += 1
        counts["frames_total"] += int(len(labels))
        counts["frames_positive"] += n_pos
        counts["seconds_total"] += rec["duration"]
        if n_pos:
            counts["videos_positive_with_span"] += 1
        else:
            counts["videos_all_negative"] += 1
        by_label[rec["label"]] = by_label.get(rec["label"], 0) + 1

    counts["frames_negative"] = counts["frames_total"] - counts["frames_positive"]
    counts["seconds_total"] = round(counts["seconds_total"], 3)
    counts["videos_included_by_label"] = dict(sorted(by_label.items()))

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, name + ".npz")
    sha = save_npz_deterministic(npz_path, arrays)

    sidecar = {
        "protocol": PROTOCOL_DOC,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": corpus["corpus"],
        "split": corpus["split"],
        "fps": FPS,
        "frame_grid": "t = 0, 1, 2, ... while t < wav_duration",
        "span_convention": "half-open [start, end)",
        "duration_source": ("wav_duration from the timestamped-chunk "
                            "manifest where it has an entry, otherwise the "
                            "wav header"),
        "cohort_definition": ("the frozen split manifest under "
                             "results/reproduction/splits/"),
        "videos_with_gold_from_upstream_tsv_only":
            corpus.get("gold_from_upstream_tsv_only") or [],
        "label_field": corpus["label_field"],
        "sources": corpus["sources"],
        "npz": os.path.relpath(npz_path, PROJECT_ROOT),
        "npz_sha256": sha,
        "npz_bytes": os.path.getsize(npz_path),
        "counts": counts,
        "excluded_positive_without_span": excluded_no_span,
        "test_split_videos_without_local_media":
            corpus.get("missing_media") or [],
        "test_split_videos_absent_from_local_annotation_mirror":
            absent_mirror,
        "normal_majority_with_leftover_spans_forced_all_negative":
            normal_with_span,
        "videos_absent_from_span_gold": corpus.get("not_in_span_gold", []),
        "videos_available_but_not_in_test_split":
            corpus.get("not_in_test_split", []),
        "cohort_note": (
            "The cohort is the test-split videos whose media is present "
            "locally at build time. Media for the remaining test videos is "
            "still being fetched; when it lands, this script must be re-run "
            "and the SHA256 below will change. Any number computed against "
            "an older SHA256 is not comparable with numbers computed "
            "against a newer one."
        ),
        "per_video": per_video,
    }
    side_path = os.path.join(out_dir, name + ".json")
    with open(side_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    log("")
    log("=== %s (%s) ===" % (corpus["corpus"], corpus["split"]))
    log("test split                : upstream %s, in span gold %s, "
        "%s absent from the annotation mirror, %d without local media"
        % (counts["upstream_test_split_size"],
           counts["test_split_size_in_span_gold"],
           counts["test_videos_absent_from_local_annotation_mirror"],
           counts["videos_missing_local_media"]))
    log("available videos          : %d  (%d with the duration off the wav "
        "header, %d with gold from the upstream TSV only)"
        % (counts["videos_available"],
           counts["videos_with_duration_from_wav_header"],
           counts["videos_with_gold_from_upstream_tsv_only"]))
    log("included                  : %d  (%s)"
        % (counts["videos_included"],
           ", ".join("%s %d" % (k, v)
                     for k, v in counts["videos_included_by_label"].items())))
    log("  with positive frames    : %d" % counts["videos_positive_with_span"])
    log("  all-negative            : %d" % counts["videos_all_negative"])
    log("excluded, positive w/o span: %d  %s"
        % (counts["videos_excluded_positive_without_span"],
           [e["video_id"] for e in excluded_no_span]))
    log("normal-majority spans dropped (rule a): %d  %s"
        % (len(normal_with_span), [e["video_id"] for e in normal_with_span]))
    log("degenerate spans dropped  : %d in %d videos"
        % (counts["degenerate_spans_dropped"],
           counts["videos_with_degenerate_span"]))
    log("spans past audio end      : %d reaching past, %d wholly past"
        % (counts["spans_reaching_past_audio_end"],
           counts["spans_wholly_past_audio_end"]))
    log("frames                    : %d total, %d positive (%.3f), %d negative"
        % (counts["frames_total"], counts["frames_positive"],
           counts["frames_positive"] / float(counts["frames_total"]),
           counts["frames_negative"]))
    log("audio seconds             : %.1f" % counts["seconds_total"])
    log("npz                       : %s" % os.path.relpath(npz_path,
                                                           PROJECT_ROOT))
    log("SHA256                    : %s" % sha)
    log("sidecar                   : %s" % os.path.relpath(side_path,
                                                           PROJECT_ROOT))
    return sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--data-root", default="/home/jehc223/data",
                    help="read-only, only for the upstream MHC test TSVs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "build_gt_arrays.log")
    handle = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        handle.write(msg + "\n")

    log("frame-level ground-truth arrays, protocol %s" % PROTOCOL_DOC)
    log("fps=%g, half-open [start, end), duration = wav_duration" % FPS)

    sidecars = [
        build(collect_hatemm(args.data_root), args.out_dir, log),
        build(collect_mhclip("en", args.data_root), args.out_dir, log),
        build(collect_mhclip("zh", args.data_root), args.out_dir, log),
    ]

    log("")
    log("%-16s %8s %8s %10s %10s  %s"
        % ("corpus", "videos", "frames", "pos", "pos_rate", "sha256"))
    for s in sidecars:
        c = s["counts"]
        log("%-16s %8d %8d %10d %10.4f  %s"
            % (s["corpus"], c["videos_included"], c["frames_total"],
               c["frames_positive"],
               c["frames_positive"] / float(c["frames_total"]),
               s["npz_sha256"]))
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
