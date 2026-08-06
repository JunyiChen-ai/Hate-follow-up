"""Cross-benchmark stage B (GPU): Whisper large-v3 re-transcription.

The ImpliHateVid channel-restoration stage B, generalized over the dataset.
The ASR configuration is inherited verbatim from
`scripts/duplex/channel_restoration_asr.py`, which is unmodified and supplies
`normalize`, `edit_norm`, `gzip_ratio` and the model id: `openai/whisper-large-v3`
through the `transformers` ASR pipeline, fp16 on cuda, automatic language
detection, 30-second chunked long-form decoding, batch size 8, and the
30-minute cap already applied when the wav was written.

Automatic language detection is deliberate and matters here: MHClip_ZH is
Mandarin and MHClip_EN is English, so no `language` is passed and Whisper
selects per chunk.

CUDA is mandatory. Placement and dtype are asserted after load and the run
aborts rather than falling back to CPU.

Usage:
  python scripts/duplex/crossbench_asr.py --dataset MHClip_ZH \
      --out-dir results/crossbench/mhclip_zh
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from channel_restoration_asr import MODEL_ID, edit_norm, gzip_ratio  # noqa: E402
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    wav_dir = os.path.join(args.out_dir, "wav")
    meta_path = os.path.join(args.out_dir, "audio_meta.jsonl")
    asr_path = os.path.join(args.out_dir, "fresh_transcripts.jsonl")

    ids = load_clean_split_ids(args.dataset, args.split)
    ann = load_annotations(args.dataset)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta[r["video_id"]] = r

    done = set()
    if os.path.exists(asr_path):
        with open(asr_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:
                    continue

    todo = [v for v in ids
            if v not in done and meta.get(v, {}).get("wav_ok")
            and os.path.isfile(os.path.join(wav_dir, v + ".wav"))]
    todo.sort(key=lambda v: meta[v].get("wav_duration") or 0.0)
    no_audio = [v for v in ids if not meta.get(v, {}).get("wav_ok")]
    total_sec = sum(meta[v].get("wav_duration") or 0.0 for v in todo)
    print(f"stage B [{args.dataset}]: {len(ids)} ids, {len(done)} done, "
          f"{len(todo)} to transcribe ({total_sec / 3600:.2f} audio-hours), "
          f"{len(no_audio)} without usable audio", flush=True)
    if not todo:
        return

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; this arm is CUDA-only")

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
    print(f"Whisper on {devices} dtypes={dtypes} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor,
                    dtype=torch.float16, device="cuda",
                    chunk_length_s=30, batch_size=8)

    t0 = time.time()
    audio_done = 0.0
    fh = open(asr_path, "a")
    for i, vid in enumerate(todo, 1):
        wav = os.path.join(wav_dir, vid + ".wav")
        tv = time.time()
        try:
            out = pipe(wav, return_timestamps=True, return_language=True,
                       generate_kwargs={"task": "transcribe"})
            text = (out.get("text") or "").strip()
            langs = Counter(c.get("language") for c in out.get("chunks", [])
                            if c.get("language"))
            err = None
            n_chunks = len(out.get("chunks", []))
        except Exception as exc:
            text, langs, n_chunks = "", Counter(), 0
            err = f"{type(exc).__name__}: {exc}"[:400]

        old = ann[vid]["transcript"] or ""
        rec = {
            "video_id": vid,
            "fresh_text": text,
            "fresh_chars": len(text),
            "old_chars": len(old),
            "old_visible_chars": len(old[:300]),
            "languages": dict(langs),
            "top_language": (langs.most_common(1)[0][0] if langs else None),
            "n_chunks": n_chunks,
            "wav_duration": meta[vid].get("wav_duration"),
            "hit_duration_cap": bool(meta[vid].get("hit_duration_cap")),
            "vad_speech_frac": meta[vid].get("vad_speech_frac"),
            "gzip_ratio_raw": gzip_ratio(text),
            "edit_norm_vs_dataset": round(edit_norm(text, old), 6),
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
            print(f"  [{i}/{len(todo)}] {vid} dur={rec['wav_duration']}s "
                  f"chars={rec['fresh_chars']} lang={rec['top_language']} "
                  f"{rec['asr_seconds']}s | {rt:.1f}x realtime, "
                  f"eta {(total_sec - audio_done) / max(rt, 1e-9) / 60:.1f} min",
                  flush=True)
    fh.close()
    print(f"stage B [{args.dataset}] done: {len(todo)} clips, "
          f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
