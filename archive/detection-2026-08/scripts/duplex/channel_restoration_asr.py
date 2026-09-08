"""Channel restoration, stage B (GPU): Whisper large-v3 re-transcription.

The C2 arm's ASR configuration is frozen in the pre-registration: `large-v3`
through the `transformers` ASR pipeline, automatic language detection, 30-second
chunked long-form decoding, 30-minute audio cap per clip. Nothing here is tuned
after the bulk run starts.

CUDA is mandatory. The model placement is asserted after load and the run aborts
rather than falling back to CPU. Clips are processed in ascending wav duration
so that throughput is observable early. Records are appended and fsynced, so the
stage resumes after a crash.

Pre-registration: docs/duplex/PREREG_channel_restoration.md.
"""

import gzip
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter

OUT_DIR = os.environ.get(
    "CR_OUT_DIR", "/home/jehc223/Hate-follow-up/results/channel_restoration")
WAV_DIR = os.path.join(OUT_DIR, "wav")
META_PATH = os.path.join(OUT_DIR, "audio_meta.jsonl")
ASR_PATH = os.path.join(OUT_DIR, "fresh_transcripts.jsonl")

MODEL_ID = "openai/whisper-large-v3"


def normalize(s):
    s = unicodedata.normalize("NFKC", (s or "")).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_norm(a, b):
    """Normalized Levenshtein over normalized text; 0 identical, 1 disjoint."""
    from rapidfuzz.distance import Levenshtein
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def gzip_ratio(text):
    raw = (text or "").encode("utf-8")
    if not raw:
        return None
    return len(raw) / len(gzip.compress(raw, 9))


def main():
    ids_path = sys.argv[1]
    ids = json.load(open(ids_path))["dismissed_all"]

    meta = {}
    with open(META_PATH) as f:
        for line in f:
            r = json.loads(line)
            meta[r["video_id"]] = r

    sys.path.insert(0, "/home/jehc223/Hate-follow-up/src/our_method")
    from data_utils import load_annotations
    ann = load_annotations("ImpliHateVid")

    done = set()
    if os.path.exists(ASR_PATH):
        with open(ASR_PATH) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:
                    continue

    todo = [v for v in ids
            if v not in done and meta.get(v, {}).get("wav_ok")
            and os.path.isfile(os.path.join(WAV_DIR, v + ".wav"))]
    todo.sort(key=lambda v: meta[v].get("wav_duration") or 0.0)
    no_audio = [v for v in ids if not meta.get(v, {}).get("wav_ok")]
    total_sec = sum(meta[v].get("wav_duration") or 0.0 for v in todo)
    print(f"stage B: {len(ids)} ids, {len(done)} done, {len(todo)} to transcribe "
          f"({total_sec/3600:.2f} audio-hours), {len(no_audio)} without usable audio",
          flush=True)
    if not todo:
        return

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; the C2 arm is CUDA-only")

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
    fh = open(ASR_PATH, "a")
    for i, vid in enumerate(todo, 1):
        wav = os.path.join(WAV_DIR, vid + ".wav")
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
            print(f"  [{i}/{len(todo)}] {vid} dur={rec['wav_duration']}s "
                  f"chars={rec['fresh_chars']} lang={rec['top_language']} "
                  f"{rec['asr_seconds']}s | {audio_done/max(el,1e-9):.1f}x realtime, "
                  f"eta {(total_sec-audio_done)/max(audio_done/max(el,1e-9),1e-9)/60:.1f} min",
                  flush=True)
    fh.close()
    print(f"stage B done: {len(todo)} clips, {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
