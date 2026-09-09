#!/usr/bin/env python3
"""Extract K uniformly spaced frames per test video for SPVL.

t_k = (k + 0.5) * duration / K, k = 0..K-1. Native resolution (the Qwen processor applies
the pixel cap at load time). Output: data/frames_k<K>/<dataset>/<video_id>/f{k:02d}_t{t:.2f}.jpg
"""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def extract(row, k_frames, out_root):
    vid = row["video_id"]; ds = row["dataset"]; dur = float(row["duration"])
    src = Path(row["video_path"])
    if not src.exists():
        cands = [c for sub in ("video", "videos") for c in sorted((Path.home() / "data" / ds / sub).glob(f"{vid}.*"))
                 if c.exists()]  # ~/data/HateClipSeg/video/ holds broken symlinks; videos/ holds the files
        if not cands:
            print(f"MISSING {ds}/{vid}", flush=True); return ds, vid, 0
        src = cands[0]
    out = out_root / ds / vid
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for k in range(k_frames):
        t = (k + 0.5) * dur / k_frames
        dst = out / f"f{k:02d}_t{t:.2f}.jpg"
        if dst.exists():
            done += 1; continue
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src),
               "-frames:v", "1", "-q:v", "2", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and dst.exists():
            done += 1
        else:
            # near the end of the file -ss may overshoot; retry 0.5 s earlier
            cmd[5] = f"{max(0.0, t - 0.5):.3f}"
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and dst.exists():
                done += 1
            else:
                print(f"FAIL {ds}/{vid} k={k} t={t:.2f}: {r.stderr.strip()[:200]}", flush=True)
    return ds, vid, done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out_root = ROOT / f"data/frames_k{a.frames}"
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    rows = [r for r in rows if r["dataset"] in a.datasets]
    print(f"videos={len(rows)} frames={a.frames} out={out_root}", flush=True)
    ok = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for ds, vid, done in ex.map(lambda r: extract(r, a.frames, out_root), rows):
            if done == a.frames: ok += 1
            else: print(f"INCOMPLETE {ds}/{vid}: {done}/{a.frames}", flush=True)
    print(f"DONE complete={ok}/{len(rows)}", flush=True)

if __name__ == "__main__":
    main()
