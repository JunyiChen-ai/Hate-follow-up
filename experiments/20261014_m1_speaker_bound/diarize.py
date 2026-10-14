#!/usr/bin/env python
"""Candidate 44 input (rule 5): speaker diarization of the HateMM / HateClipSeg test audio.

Writes data/speaker_turns/<dataset>/turns.jsonl (one record per video: speaker turns in seconds) and
data/speaker_turns/PROVENANCE.md. No labels are read. GPU use only inside a Slurm job.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for f in (Path.home() / ".cache/huggingface/token", Path(os.environ.get("HF_HOME", "/nonexistent")) / "token"):
        if f.is_file():
            return f.read_text().strip()
    return None


def load_pipeline(name, device):
    import torch
    from pyannote.audio import Pipeline

    tok = hf_token()
    try:
        pipe = Pipeline.from_pretrained(name, token=tok)
    except TypeError:
        pipe = Pipeline.from_pretrained(name, use_auth_token=tok)
    if pipe is None:
        raise RuntimeError(f"Pipeline.from_pretrained({name}) returned None: gated model not accepted, or offline")
    pipe.to(torch.device(device))
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--pipeline", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--audio", default=str(ROOT / "data/audio16k"))
    ap.add_argument("--out", default=str(ROOT / "data/speaker_turns"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N files per dataset (smoke; writes turns_smoke.jsonl)")
    a = ap.parse_args()
    import torch
    import torchaudio

    host = socket.gethostname()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f"host {host} commit {commit} pipeline {a.pipeline} device {a.device} torch {torch.__version__}", flush=True)
    pipe = load_pipeline(a.pipeline, a.device)
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    totals = {}
    for ds in a.datasets:
        wavs = sorted((Path(a.audio) / ds).glob("*.wav"))
        if a.limit:
            wavs = wavs[: a.limit]
        d = out_root / ds
        d.mkdir(parents=True, exist_ok=True)
        name = "turns_smoke.jsonl" if a.limit else "turns.jsonl"
        n_ok = n_fail = 0
        audio_s = 0.0
        t_ds = time.time()
        with open(d / name, "w") as fout:
            for i, w in enumerate(wavs):
                t0 = time.time()
                try:
                    wav, sr = torchaudio.load(str(w))
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    res = pipe({"waveform": wav, "sample_rate": sr})
                    ann = getattr(res, "speaker_diarization", res)
                    turns = [{"spk": str(spk), "s": round(float(seg.start), 3), "e": round(float(seg.end), 3)}
                             for seg, _, spk in ann.itertracks(yield_label=True)]
                    rec = {"video_id": w.stem, "dataset": ds, "status": "ok", "n_speakers": len(ann.labels()),
                           "n_turns": len(turns), "audio_seconds": round(wav.shape[1] / sr, 3),
                           "speech_seconds": round(sum(t["e"] - t["s"] for t in turns), 3),
                           "seconds": round(time.time() - t0, 2), "turns": turns}
                    n_ok += 1
                    audio_s += rec["audio_seconds"]
                except Exception as ex:  # noqa: BLE001 - record and continue
                    rec = {"video_id": w.stem, "dataset": ds, "status": "failed", "error": repr(ex)[:400], "turns": []}
                    n_fail += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                if (i + 1) % 20 == 0 or i + 1 == len(wavs):
                    print(f"{ds} {i + 1}/{len(wavs)} ok {n_ok} failed {n_fail} {time.time() - t_ds:.0f}s", flush=True)
        totals[ds] = {"files": len(wavs), "ok": n_ok, "failed": n_fail, "audio_seconds": round(audio_s),
                      "wall_seconds": round(time.time() - t_ds)}
    print("TOTALS " + json.dumps(totals), flush=True)
    if not a.limit:
        (out_root / "PROVENANCE.md").write_text(
            f"""# data/speaker_turns

Speaker turns of the HateMM and HateClipSeg test videos from `{a.pipeline}` (pyannote.audio, default
hyper-parameters, number of speakers not given), run on the 16 kHz mono audio in `data/audio16k/` (one call per file,
whole file). `<dataset>/turns.jsonl`: one record per video with `status` (`ok` / `failed`), `n_speakers`,
`turns: [{{spk, s, e}}]` in seconds. Generated on {host} ({datetime.date.today().isoformat()}, commit {commit}) by
`experiments/20261014_m1_speaker_bound/diarize.py` (candidate 44 pre-check, rule 5 input). Totals: {json.dumps(totals)}.
No labels read.
"""
        )


if __name__ == "__main__":
    main()
