"""Interleaved-timeline kill test, stage A (GPU): recover Whisper chunk timestamps.

The frozen cross-benchmark ASR stage already called the Whisper pipeline with
`return_timestamps=True`; it stored only the joined text and the chunk count and
threw the timestamps away. This module re-runs the identical configuration --
`openai/whisper-large-v3` through the transformers ASR pipeline, fp16 on cuda,
automatic language detection, 30-second chunked long-form decoding, batch size
8, reading the same wav files already on disk -- and keeps the chunk boundaries.

Nothing about the transcription is re-decided. `MODEL_ID` is imported from
`scripts/duplex/channel_restoration_asr.py`, unmodified, and the pipeline call
is byte-identical to `scripts/duplex/crossbench_asr.py`. The stored text is
compared against the frozen `fresh_transcripts.jsonl` and the comparison result
is recorded per video: a mismatch disqualifies that video from the timestamped
segmentation route and sends it to the documented proportional fallback.

Pre-registration: docs/duplex/PREREG_interleaved_timeline_killtest.md.

Output: results/interleaved_timeline/<slug>/timestamped_chunks.jsonl.

Usage:
  python scripts/duplex/interleaved_timeline_asr.py --corpus mhclip_zh
"""

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from channel_restoration_asr import MODEL_ID  # noqa: E402

# Corpus slug -> (dataset name, frozen working directory holding wav/ and the
# frozen ASR + gate outputs).
CORPORA = {
    "mhclip_zh": ("MHClip_ZH", "results/testruns/mhclip_zh"),
    "mhclip_en": ("MHClip_EN", "results/testruns/mhclip_en"),
    "implihatevid": ("ImpliHateVid", "results/testruns/implihatevid"),
    "hateclipseg": ("HateClipSeg", "results/hateclipseg"),
    # HateMM test_clean, for the HateMM isolated-chunk localization diagnostic.
    # Same frozen working directory (wav/, audio_meta.jsonl, fresh_transcripts.jsonl)
    # as every other HateMM run; the key doubles as the output sub-directory, so
    # `--out-root results` writes results/hatemm_localization/timestamped_chunks.jsonl.
    "hatemm_localization": ("HateMM", "results/testruns/hatemm"),
}

# ---------------------------------------------------------------------------
# Reproduction-study corpora (baseline reproduction, Phase 2).
#
# The entries above all read a *frozen working directory* that already holds
# `audio_meta.jsonl` (durations) and `fresh_transcripts.jsonl` (the video list
# plus the text to reproduce). The reproduction study's train splits have
# neither: they are plain wav directories plus a frozen split manifest. These
# entries therefore name a wav directory and an explicit id source, and the
# duration is read straight off the wav header. Nothing about the Whisper
# configuration changes -- same MODEL_ID, same fp16-on-cuda pipeline, same
# chunk_length_s=30 / batch_size=8, same automatic language detection (the ZH
# `forced_zh` arm in results/testruns/mhclip_zh was a separate diagnostic side
# arm and was never the convention behind the timestamped-chunk manifests the
# study consumes; those were decoded with language auto-detection).
DATA_ROOT = os.environ.get("HVD_DATA_ROOT", "/home/jehc223/data")
SPLITS = os.path.join(ROOT, "results", "reproduction", "splits")

SPLIT_CORPORA = {
    # Unified manifests used by the official-validation reproduction.  These
    # deliberately cover train + validation + test in one resumable output so
    # the run does not depend on legacy, unversioned testrun artifacts being
    # present on a particular machine.
    "hatemm_all": {
        "dataset": "HateMM",
        "wav_dir": "/home/jehc223/Retrieval-hate/data/AV2A_wav/HateMM",
        "ids_files": [os.path.join(SPLITS, "hatemm_%s.txt" % split)
                      for split in ("train", "val", "test")],
    },
    "mhclip_en_all": {
        "dataset": "MHClip_EN",
        "wav_dir": "/home/jehc223/Retrieval-hate/data/AV2A_wav/MHC",
        "ids_files": [os.path.join(SPLITS, "mhclip_en_%s.txt" % split)
                      for split in ("train", "val", "test")],
    },
    "mhclip_zh_all": {
        "dataset": "MHClip_ZH",
        "wav_dir": "/home/jehc223/Retrieval-hate/data/AV2A_wav/MHC_zh",
        "ids_files": [os.path.join(SPLITS, "mhclip_zh_%s.txt" % split)
                      for split in ("train", "val", "test")],
    },
    "hateclipseg_all": {
        "dataset": "HateClipSeg",
        "wav_dir": "/home/jehc223/Retrieval-hate/data/AV2A_wav/HateClipSeg",
        "ids_files": [os.path.join(SPLITS, "hateclipseg_%s.txt" % split)
                      for split in ("train", "val", "test")],
    },
    "hatemm_train": {
        "dataset": "HateMM",
        "wav_dir": os.path.join(DATA_ROOT, "HateMM", "wav"),
        "ids_file": os.path.join(SPLITS, "hatemm_train.txt"),
    },
    "mhclip_en_train": {
        "dataset": "MHClip_EN",
        "wav_dir": os.path.join(DATA_ROOT, "Multihateclip", "English", "wav"),
        "ids_file": os.path.join(SPLITS, "mhclip_en_train.txt"),
    },
    "mhclip_zh_train": {
        "dataset": "MHClip_ZH",
        "wav_dir": os.path.join(DATA_ROOT, "Multihateclip", "Chinese", "wav"),
        "ids_file": os.path.join(SPLITS, "mhclip_zh_train.txt"),
    },
    # Test-split videos whose media only arrived with the Phase 0 pull, so
    # they are absent from the legacy results/testruns/* manifests.
    "mhclip_en_test_new": {
        "dataset": "MHClip_EN",
        "wav_dir": os.path.join(DATA_ROOT, "Multihateclip", "English", "wav"),
        "ids": ["hXv7bR9i5Q4"],
    },
    "mhclip_zh_test_new": {
        "dataset": "MHClip_ZH",
        "wav_dir": os.path.join(DATA_ROOT, "Multihateclip", "Chinese", "wav"),
        "ids": ["BV16N4y1q7WU", "BV1Gw411E7i2", "BV1KK411P7uJ",
                "BV1Qx411V7tT", "BV1bA41137we", "BV1du411g7tk",
                "BV1nJ4m1p7BG", "BV1zD4y1Y7ec"],
    },
    # HateClipSeg videos, train or test, absent from the frozen manifest at
    # results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl. The
    # id list is written by scripts/duplex/hateclipseg_asr_coverage.py, which
    # also records the coverage it measured; run that first. As of the Phase 2
    # inventory the list is empty -- the frozen manifest already covers all
    # 394 locally held videos -- so this entry exists for the case where more
    # media lands later.
    "hateclipseg_missing": {
        "dataset": "HateClipSeg",
        "wav_dir": os.path.join(DATA_ROOT, "HateClipSeg", "wav"),
        "ids_file": os.path.join(ROOT, "results", "reproduction", "asr",
                                 "hateclipseg_missing", "ids.txt"),
    },
}


def wav_duration_seconds(path):
    """Duration in seconds, read from the wav header (no decode)."""
    import wave
    with wave.open(path, "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if rate <= 0:
        raise ValueError("non-positive sample rate in %s" % path)
    return round(frames / float(rate), 6)


def load_jsonl(path, key="video_id"):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r[key]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    choices=sorted(CORPORA) + sorted(SPLIT_CORPORA))
    ap.add_argument("--out-root", default=os.path.join(
        ROOT, "results", "interleaved_timeline"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most this many remaining videos (smoke test)")
    ap.add_argument("--video-batch", type=int, default=8,
                    help="Number of video files submitted to the pipeline together")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "timestamped_chunks.jsonl")
    missing_wav = []

    if args.corpus in CORPORA:
        dataset, work_rel = CORPORA[args.corpus]
        work = os.path.join(ROOT, work_rel)
        meta = load_jsonl(os.path.join(work, "audio_meta.jsonl"))
        frozen = load_jsonl(os.path.join(work, "fresh_transcripts.jsonl"))
        wav_dir = os.path.join(work, "wav")
        wanted = list(frozen)
    else:
        spec = SPLIT_CORPORA[args.corpus]
        dataset = spec["dataset"]
        wav_dir = spec["wav_dir"]
        frozen = {}
        if "ids_files" in spec:
            wanted = []
            for ids_file in spec["ids_files"]:
                with open(ids_file, encoding="utf-8") as handle:
                    wanted.extend(ln.strip() for ln in handle if ln.strip())
            # A split bug must not silently cause a duplicated ASR record.
            if len(wanted) != len(set(wanted)):
                raise ValueError("duplicate ids across %s" % spec["ids_files"])
        elif "ids_file" in spec:
            with open(spec["ids_file"], encoding="utf-8") as handle:
                wanted = [ln.strip() for ln in handle if ln.strip()]
        else:
            wanted = list(spec["ids"])
        meta = {}
        for vid in wanted:
            wav = os.path.join(wav_dir, vid + ".wav")
            if not os.path.isfile(wav):
                missing_wav.append(vid)
                continue
            try:
                meta[vid] = {"wav_duration": wav_duration_seconds(wav),
                             "container_duration": None}
            except Exception as exc:  # unreadable header -> report, skip
                missing_wav.append(vid)
                print(f"  UNREADABLE WAV {vid}: {type(exc).__name__}: {exc}",
                      flush=True)
        if missing_wav:
            print(f"  MISSING/UNREADABLE WAV ({len(missing_wav)}): "
                  f"{missing_wav}", flush=True)

    done = set(load_jsonl(out_path))
    # A video container may legitimately have no audio stream.  Keep such a
    # video in the experiment and record an explicit empty transcript instead
    # of silently dropping it (the downstream text grid then stays all-zero).
    # This is also resumable: never append the sentinel twice.
    missing_records = [v for v in missing_wav if v not in done]
    if missing_records:
        with open(out_path, "a", encoding="utf-8") as handle:
            for vid in missing_records:
                handle.write(json.dumps({
                    "video_id": vid,
                    "text": "",
                    "chunks": [],
                    "n_chunks": 0,
                    "matches_frozen_text": None,
                    "frozen_n_chunks": None,
                    "wav_duration": None,
                    "container_duration": None,
                    "asr_seconds": 0.0,
                    "error": "missing_or_unreadable_audio",
                }, ensure_ascii=False) + "\n")
        done.update(missing_records)
        print("  recorded %d explicit empty-audio transcripts" %
              len(missing_records), flush=True)
    todo = [v for v in wanted
            if v not in done and os.path.isfile(os.path.join(wav_dir, v + ".wav"))]
    todo.sort(key=lambda v: (meta.get(v, {}).get("wav_duration") or 0.0))
    total_sec = sum(meta.get(v, {}).get("wav_duration") or 0.0 for v in todo)
    print(f"stage A [{args.corpus}/{dataset}]: {len(wanted)} videos wanted, "
          f"{len(done)} already timestamped, {len(todo)} to run "
          f"({total_sec / 3600:.2f} audio-hours)", flush=True)
    if args.limit is not None:
        todo = todo[:args.limit]
        total_sec = sum(meta.get(v, {}).get("wav_duration") or 0.0 for v in todo)
        print(f"  --limit: {len(todo)} videos", flush=True)
    if not todo:
        return

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; this stage is CUDA-only")

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to("cuda")
    model.eval()
    devices = {p.device.type for p in model.parameters()}
    dtypes = {str(p.dtype) for p in model.parameters()}
    if devices != {"cuda"}:
        raise SystemExit(f"ABORT: Whisper parameters are on {devices}, not cuda")
    if "torch.float16" not in dtypes:
        raise SystemExit(f"ABORT: Whisper parameter dtypes are {dtypes}, not fp16")

    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor,
                    torch_dtype=torch.float16, device="cuda",
                    chunk_length_s=30, batch_size=8)
    gen_kwargs = {"task": "transcribe"}

    t0 = time.time()
    audio_done = 0.0
    n_match = 0
    n_error = 0
    fh = open(out_path, "a")
    processed = 0
    for batch_start in range(0, len(todo), args.video_batch):
        vids = todo[batch_start:batch_start + args.video_batch]
        wavs = [os.path.join(wav_dir, vid + ".wav") for vid in vids]
        tb = time.time()
        try:
            outs = list(pipe(wavs, batch_size=args.video_batch,
                             return_timestamps=True, return_language=True,
                             generate_kwargs=gen_kwargs))
            if len(outs) != len(vids):
                raise RuntimeError("pipeline returned %d outputs for %d inputs" %
                                   (len(outs), len(vids)))
        except Exception as batch_exc:
            # Preserve per-video fault isolation: a malformed recording must
            # not turn every other member of its batch into an empty record.
            print("  batch fallback (%s: %s)" %
                  (type(batch_exc).__name__, str(batch_exc)[:200]), flush=True)
            outs = []
            for vid, wav in zip(vids, wavs):
                try:
                    outs.append(pipe(wav, return_timestamps=True,
                                     return_language=True,
                                     generate_kwargs=gen_kwargs))
                except Exception as exc:
                    outs.append({"text": "", "chunks": [],
                                 "_asr_error": "%s: %s" %
                                 (type(exc).__name__, str(exc)[:350])})

        per_video_seconds = (time.time() - tb) / max(len(vids), 1)
        for vid, out in zip(vids, outs):
            text = (out.get("text") or "").strip()
            chunks = [{"start": (c.get("timestamp") or (None, None))[0],
                       "end": (c.get("timestamp") or (None, None))[1],
                       "text": c.get("text") or ""}
                      for c in out.get("chunks", [])]
            err = out.get("_asr_error")
            if err:
                n_error += 1
                print(f"  ASR ERROR {vid}: {err}", flush=True)

            if vid in frozen:
                frozen_text = frozen[vid].get("fresh_text") or ""
                matches = (text == frozen_text)
                frozen_n_chunks = frozen[vid].get("n_chunks")
            else:
                # Split-manifest corpora have no frozen transcript to reproduce.
                matches, frozen_n_chunks = None, None
            n_match += int(bool(matches))
            rec = {
                "video_id": vid,
                "text": text,
                "chunks": chunks,
                "n_chunks": len(chunks),
                "matches_frozen_text": matches,
                "frozen_n_chunks": frozen_n_chunks,
                "wav_duration": meta.get(vid, {}).get("wav_duration"),
                "container_duration": meta.get(vid, {}).get("container_duration"),
                "asr_seconds": round(per_video_seconds, 2),
                "error": err,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

            processed += 1
            audio_done += rec["wav_duration"] or 0.0
        if processed % 20 < len(vids) or processed == len(vids):
            el = time.time() - t0
            rt = audio_done / max(el, 1e-9)
            print(f"  [{processed}/{len(todo)}] dur={rec['wav_duration']}s "
                  f"chunks={len(chunks)} match={matches} "
                  f"{rec['asr_seconds']}s | {rt:.1f}x realtime, "
                  f"eta {(total_sec - audio_done) / max(rt, 1e-9) / 60:.1f} min",
                  flush=True)
    fh.close()
    print(f"stage A [{args.corpus}] done: {len(todo)} clips, "
          f"{n_match}/{len(todo)} reproduced the frozen text, "
          f"{n_error} errored, {len(missing_wav)} without a readable wav, "
          f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
