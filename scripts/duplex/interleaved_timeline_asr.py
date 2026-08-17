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
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--out-root", default=os.path.join(
        ROOT, "results", "interleaved_timeline"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most this many remaining videos (smoke test)")
    args = ap.parse_args()

    dataset, work_rel = CORPORA[args.corpus]
    work = os.path.join(ROOT, work_rel)
    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "timestamped_chunks.jsonl")

    meta = load_jsonl(os.path.join(work, "audio_meta.jsonl"))
    frozen = load_jsonl(os.path.join(work, "fresh_transcripts.jsonl"))
    wav_dir = os.path.join(work, "wav")

    done = set(load_jsonl(out_path))
    todo = [v for v in frozen
            if v not in done and os.path.isfile(os.path.join(wav_dir, v + ".wav"))]
    todo.sort(key=lambda v: (meta.get(v, {}).get("wav_duration") or 0.0))
    total_sec = sum(meta.get(v, {}).get("wav_duration") or 0.0 for v in todo)
    print(f"stage A [{args.corpus}/{dataset}]: {len(frozen)} frozen ASR rows, "
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
        MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
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
                    dtype=torch.float16, device="cuda",
                    chunk_length_s=30, batch_size=8)
    gen_kwargs = {"task": "transcribe"}

    t0 = time.time()
    audio_done = 0.0
    n_match = 0
    fh = open(out_path, "a")
    for i, vid in enumerate(todo, 1):
        wav = os.path.join(wav_dir, vid + ".wav")
        tv = time.time()
        try:
            out = pipe(wav, return_timestamps=True, return_language=True,
                       generate_kwargs=gen_kwargs)
            text = (out.get("text") or "").strip()
            chunks = [{"start": (c.get("timestamp") or (None, None))[0],
                       "end": (c.get("timestamp") or (None, None))[1],
                       "text": c.get("text") or ""}
                      for c in out.get("chunks", [])]
            err = None
        except Exception as exc:
            text, chunks = "", []
            err = f"{type(exc).__name__}: {exc}"[:400]

        frozen_text = frozen[vid].get("fresh_text") or ""
        matches = (text == frozen_text)
        n_match += int(matches)
        rec = {
            "video_id": vid,
            "text": text,
            "chunks": chunks,
            "n_chunks": len(chunks),
            "matches_frozen_text": matches,
            "frozen_n_chunks": frozen[vid].get("n_chunks"),
            "wav_duration": meta.get(vid, {}).get("wav_duration"),
            "container_duration": meta.get(vid, {}).get("container_duration"),
            "asr_seconds": round(time.time() - tv, 2),
            "error": err,
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

        audio_done += rec["wav_duration"] or 0.0
        if i % 20 == 0 or i == 1:
            el = time.time() - t0
            rt = audio_done / max(el, 1e-9)
            print(f"  [{i}/{len(todo)}] dur={rec['wav_duration']}s "
                  f"chunks={len(chunks)} match={matches} "
                  f"{rec['asr_seconds']}s | {rt:.1f}x realtime, "
                  f"eta {(total_sec - audio_done) / max(rt, 1e-9) / 60:.1f} min",
                  flush=True)
    fh.close()
    print(f"stage A [{args.corpus}] done: {len(todo)} clips, "
          f"{n_match}/{len(todo)} reproduced the frozen text, "
          f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
