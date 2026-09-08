"""Timestamped Whisper ASR for the frozen temporal-attribution cohort."""

import json
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = os.path.join(ROOT, "results", "temporal_attribution")
COHORTS = os.path.join(RUN, "cohorts.json")
OUT = os.path.join(RUN, "timestamped_asr.jsonl")
WAV = os.path.join(ROOT, "results", "c2_fullcorpus", "wav")
MP4 = "/home/jehc223/data/ImpliHateVid_video_dismissed"
MODEL_ID = "openai/whisper-large-v3"


def done_ids():
    out = set()
    if not os.path.exists(OUT):
        return out
    for line in open(OUT):
        try:
            r = json.loads(line)
            if r.get("video_id") and r.get("error") is None:
                out.add(r["video_id"])
        except Exception:
            pass
    return out


def main():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    cohorts = json.load(open(COHORTS))["cohorts"]
    ids = sorted({v for values in cohorts.values() for v in values})
    done = done_ids()
    todo = [v for v in ids if v not in done]
    print(f"timestamp ASR: {len(ids)} cohort videos, {len(done)} done, {len(todo)} todo", flush=True)
    if not todo:
        return
    # Some full-corpus wavs were pruned after the source run. Restore the exact
    # frozen 16 kHz mono / 1800 s-capped extraction from the retained mp4s;
    # cohort membership is never changed because of cache state.
    sys_path = os.path.dirname(__file__)
    import sys
    sys.path.insert(0, sys_path)
    from channel_restoration_audio import extract_wav
    for vid in todo:
        wav = os.path.join(WAV, vid + ".wav")
        if not os.path.isfile(wav):
            mp4 = os.path.join(MP4, vid + ".mp4")
            ok, err = extract_wav(mp4, wav)
            if not ok:
                raise SystemExit(f"cannot restore wav for {vid}: {err}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    model.eval()
    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
                    dtype=torch.float16, device="cuda", chunk_length_s=30, batch_size=8)
    t0 = time.time()
    for i, vid in enumerate(todo, 1):
        path = os.path.join(WAV, vid + ".wav")
        tv = time.time()
        try:
            raw = pipe(path, return_timestamps=True, return_language=True,
                       generate_kwargs={"task": "transcribe"})
            chunks = []
            for c in raw.get("chunks", []):
                ts = c.get("timestamp") or (None, None)
                chunks.append({"start": ts[0], "end": ts[1],
                               "text": (c.get("text") or "").strip()})
            rec = {"video_id": vid, "text": (raw.get("text") or "").strip(),
                   "chunks": chunks, "error": None,
                   "seconds": round(time.time() - tv, 3)}
        except Exception as exc:
            rec = {"video_id": vid, "text": "", "chunks": [],
                   "error": f"{type(exc).__name__}: {exc}"[:500],
                   "seconds": round(time.time() - tv, 3)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if i == 1 or i % 10 == 0:
            print(f"[{i}/{len(todo)}] {vid} chunks={len(rec['chunks'])} "
                  f"chars={len(rec['text'])} {rec['seconds']}s", flush=True)
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
