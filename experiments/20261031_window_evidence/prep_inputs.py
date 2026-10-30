#!/usr/bin/env python3
"""CPU inputs of the window-evidence reader for one corpus: 16 kHz mono wav (`data/audio16k/<ds>/`, for the forced
alignment) and the 0.5 fps frame cache (`data/temporal_retrieved_kv/<ds>/<vid>/`). Idempotent: videos whose wav and
metadata.json exist are skipped. No labels read. Run detached:
  setsid nohup python -u experiments/20261031_window_evidence/prep_inputs.py --dataset DeHate --jobs 12 \
      > runs/20261031_window_evidence/prep_DeHate.log 2>&1 &
"""
import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_manifest  # noqa: E402
from src.window_evidence import FRAME_CACHE, acquire_frames  # noqa: E402
from src.actual_video_frames import resolve_video  # noqa: E402

MANIFESTS = {"DeHate": ROOT / "data/manifests/DeHate_test.jsonl"}
GENERATOR = "experiments/20261031_window_evidence/prep_inputs.py; 2026-10-31"


def one(row):
    out = dict(dataset=row["dataset"], video_id=row["video_id"])
    wav = ROOT / "data/audio16k" / row["dataset"] / (row["video_id"] + ".wav")
    try:
        if not wav.exists():
            wav.parent.mkdir(parents=True, exist_ok=True)
            tmp = wav.with_suffix(".tmp.wav")
            p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(resolve_video(row)), "-vn", "-ac", "1", "-ar", "16000",
                                "-acodec", "pcm_s16le", str(tmp)], capture_output=True, text=True)
            if p.returncode == 0 and tmp.exists():
                tmp.replace(wav)
            else:
                tmp.unlink(missing_ok=True)
                out["audio_error"] = p.stderr[-300:]
        if wav.exists():
            out["audio_seconds"] = (wav.stat().st_size - 44) / 32000
        if not (FRAME_CACHE / row["dataset"] / row["video_id"] / "metadata.json").exists():
            meta = acquire_frames(row, GENERATOR)
            out["frames"] = len(meta["source"]["selected_indices"])
            out["decode_seconds"] = meta["source"]["decode_seconds"]
        else:
            out["frames"] = "cached"
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(MANIFESTS))
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    rows = load_manifest(MANIFESTS[a.dataset], [a.dataset])
    t0 = time.perf_counter()
    log = []
    with Pool(a.jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(one, rows), 1):
            log.append(r)
            if k % 50 == 0 or "error" in r:
                print(k, len(rows), r, f"{time.perf_counter() - t0:.0f}s", flush=True)
    out = ROOT / "runs/20261031_window_evidence" / f"prep_{a.dataset}.jsonl"
    out.write_text("".join(json.dumps(x) + "\n" for x in log))
    errors = [x for x in log if "error" in x]
    no_audio = [x for x in log if "audio_error" in x]
    print(f"videos {len(log)} frame_errors {len(errors)} audio_errors {len(no_audio)} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    print("PREP_DONE" if not errors else "PREP_FAILED", flush=True)


if __name__ == "__main__":
    main()
