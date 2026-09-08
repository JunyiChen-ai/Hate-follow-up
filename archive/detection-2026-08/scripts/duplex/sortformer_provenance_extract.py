#!/usr/bin/env python3
"""End-to-end speaker-turn restoration with NVIDIA Streaming Sortformer."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from nemo.collections.asr.models import SortformerEncLabelModel


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/speaker_provenance"
MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def main():
    cohort = json.load(open(ROOT / "results/temporal_attribution/cohorts.json"))
    ids = sorted({v for values in cohort["cohorts"].values() for v in values})
    asr = {r["video_id"]: r for r in map(json.loads, open(ROOT / "results/temporal_attribution/timestamped_asr.jsonl")) if r.get("video_id") in set(ids)}
    visual = {r["video_id"]: r for r in map(json.loads, open(RUN / "provenance.jsonl"))}
    out = RUN / "sortformer_provenance.jsonl"
    done = {r["video_id"] for r in map(json.loads, open(out))} if out.exists() else set()
    model = SortformerEncLabelModel.from_pretrained(MODEL_ID).eval().cuda()

    for i, vid in enumerate(ids, 1):
        if vid in done:
            continue
        wav = ROOT / "results/c2_fullcorpus/wav" / f"{vid}.wav"
        pred = model.diarize(audio=[str(wav)], batch_size=1)[0]
        segs = []
        for line in pred:
            a, b, spk = line.split()
            segs.append({"start": float(a), "end": float(b), "speaker": spk})
        duration = defaultdict(float)
        for s in segs: duration[s["speaker"]] += s["end"] - s["start"]
        primary = max(duration, key=duration.get) if duration else None
        old_chunks = visual.get(vid, {}).get("chunks", [])
        chunks = []
        unknown = 0
        for ci, c0 in enumerate(asr.get(vid, {}).get("chunks", [])):
            a, b = c0.get("start"), c0.get("end")
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b <= a:
                unknown += 1
                chunks.append({"text": (c0.get("text") or "").strip(), "start": a, "end": b,
                               "speaker": None, "tag": "SPEAKER_UNKNOWN"})
                continue
            scores = {s: sum(overlap(float(a), float(b), x["start"], x["end"])
                             for x in segs if x["speaker"] == s) for s in duration}
            speaker = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else None
            frac = old_chunks[ci].get("face_visible_fraction") if ci < len(old_chunks) else None
            visual_role = "ONSCREEN" if isinstance(frac, (int, float)) and frac >= .4 else "VOICEOVER" if isinstance(frac, (int, float)) else "UNKNOWN"
            if speaker is None:
                tag = "SPEAKER_UNKNOWN"; unknown += 1
            else:
                owner = "PRIMARY" if speaker == primary else "SECONDARY"
                tag = f"{owner}_{visual_role}"
            chunks.append({"text": (c0.get("text") or "").strip(), "start": a, "end": b,
                           "speaker": speaker, "face_visible_fraction": frac, "tag": tag})
        rec = {"video_id": vid, "model": MODEL_ID, "segments": segs,
               "speaker_durations": dict(duration), "n_speakers": len(duration),
               "primary_speaker": primary, "unknown_chunks": unknown, "chunks": chunks,
               "provenance_transcript": "\n".join(f"[{c['tag']}]: {c['text']}" for c in chunks if c["text"])}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
        if i == 1 or i % 10 == 0:
            print(f"[{i}/90] {vid} speakers={len(duration)} segments={len(segs)} unknown={unknown}", flush=True)


if __name__ == "__main__": main()
