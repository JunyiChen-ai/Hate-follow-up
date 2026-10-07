#!/usr/bin/env python3
"""ZS-CLIP on DeHate with the exact setup of the HateMM / HateClipSeg rows (runs/20261008_baselines/zs_clip/).

Port of Retrieval-hate `scripts/repro_campaign/extract_dense.py` (visual channel) and `zs_clip.py`
(stages `prompts` + `visual`, prompt set `main`); uoa-lab2 `~/Retrieval-hate` is read only.

  features  ffmpeg `fps=4,scale=w=336:h=336:force_original_aspect_ratio=increase:flags=bicubic,crop=336:336`,
            rgb24, /255, CLIP mean/std, `CLIPVisionModel` (openai/clip-vit-large-patch14-336) fp16,
            `pooler_output` (1024-d), batch 48. Frame i = content at t = i/4 s.
  score     cos(img, "a hateful video frame") - cos(img, "a normal video frame"), with the joint-space image
            embedding = visual_projection(pooler_output), using the campaign's own `prompt_emb.npz` (text
            embeddings and projection matrix, copied to data/retrieval_hate_repro/repro_zs_clip/).
  grid      native 4 fps, used as is; frames past the last decoded frame hold its value (as for HMM/HCS).

Stages
  verify    re-extract 5 HateMM test videos and compare with the campaign's cached features and scores
  extract   DeHate test videos (all 1341 of the manifest) -> data/dense4fps_clipL336/DeHate/<vid>.npy
  score     scores + exact-cohort finalisation -> runs/20261008_baselines/zs_clip/DeHate/
No labels are read.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402

CODE_PATH = "experiments/20261008_baselines/zs_clip/zs_clip.py"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"
FPS = 4.0
SIZE = 336
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
PROMPT_EMB = REPO / "data/retrieval_hate_repro/repro_zs_clip/prompt_emb.npz"
FEAT_DIR = REPO / "data/dense4fps_clipL336/DeHate"
OUT = ec.OUT_ROOT / "zs_clip" / "DeHate"
RH = Path.home() / "Retrieval-hate"          # read only (uoa-lab2), verify stage only
HMM_VIDEO = Path.home() / "data/HateMM/video"


def frame_stream(path: Path, chunk: int):
    """Retrieval-hate extract_dense.frame_stream, unchanged."""
    vf = (f"fps={FPS:g},scale=w={SIZE}:h={SIZE}:force_original_aspect_ratio=increase:"
          f"flags=bicubic,crop={SIZE}:{SIZE}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    nbytes = SIZE * SIZE * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes * chunk)
    try:
        while True:
            buf = proc.stdout.read(nbytes * chunk)
            if not buf:
                break
            n = len(buf) // nbytes
            if n == 0:
                break
            yield np.frombuffer(buf[: n * nbytes], dtype=np.uint8).reshape(n, SIZE, SIZE, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "ignore")
        proc.wait()
        if proc.returncode not in (0, None) and err.strip():
            print(f"[ffmpeg-v] rc={proc.returncode} {path.name}: {err.strip()[:200]}", flush=True)


def decode_all(path: Path, batch: int):
    return list(frame_stream(path, batch))


def encode(model, chunks, mean, std, dev):
    import torch
    feats = []
    with torch.no_grad():
        for arr in chunks:
            x = torch.from_numpy(np.ascontiguousarray(arr)).to(dev)
            x = x.permute(0, 3, 1, 2).half().div_(255.0)
            x = (x - mean) / std
            feats.append(model(pixel_values=x).pooler_output.float().cpu().numpy())
    if not feats:
        return None
    return np.concatenate(feats, 0).astype(np.float32)


def load_model():
    import torch
    from transformers import CLIPVisionModel
    dev = "cuda"
    vm = CLIPVisionModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float16).to(dev).eval()
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return vm, mean, std, dev


def atomic_save(d: Path, vid: str, arr: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{vid}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, d / f"{vid}.npy")


def extract(jobs, out_dir: Path, batch: int, log) -> dict:
    """jobs: list of (vid, path). Decoding of the next videos runs on a worker thread (does not change numbers)."""
    import torch
    vm, mean, std, dev = load_model()
    todo = [(v, p) for v, p in jobs if not (out_dir / f"{v}.npy").exists()]
    log(f"extract: {len(todo)} of {len(jobs)} videos to do -> {out_dir}")
    q: queue.Queue = queue.Queue(maxsize=3)

    def work():
        for v, p in todo:
            try:
                q.put((v, p, decode_all(p, batch), None))
            except Exception as e:  # recorded, the video falls to F2 later
                q.put((v, p, None, f"{type(e).__name__}: {e}"))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    stats = {"n": 0, "empty": [], "errors": {}, "oom_retry": []}
    t0 = time.time()
    while True:
        got = q.get()
        if got is None:
            break
        v, p, chunks, err = got
        if err is not None or not chunks:
            if err is None:
                stats["empty"].append(v)
            else:
                stats["errors"][v] = err
            log(f"[EMPTY] {v} {err or 'no frames decoded'}")
            continue
        F = None
        bs = batch
        while F is None:
            try:
                if bs != batch:  # re-chunk after an OOM
                    flat = np.concatenate(chunks, 0)
                    chunks = [flat[i:i + bs] for i in range(0, len(flat), bs)]
                F = encode(vm, chunks, mean, std, dev)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 4:
                    raise
                bs //= 2
                stats["oom_retry"].append([v, bs])
        atomic_save(out_dir, v, F)
        stats["n"] += 1
        if stats["n"] % 25 == 0:
            el = time.time() - t0
            log(f"PROGRESS {stats['n']}/{len(todo)} elapsed={el:.0f}s rate={stats['n'] / el:.2f} vid/s")
    stats["wall_seconds"] = round(time.time() - t0, 1)
    return stats


def score_feats(F: np.ndarray, pr) -> np.ndarray:
    """Retrieval-hate zs_clip.stage_visual for the `main` prompt set."""
    names = [str(x) for x in pr["names"]]
    emb, W = pr["emb"], pr["visual_projection"]
    d = (emb[names.index("main|pos")] - emb[names.index("main|neg")]).astype(np.float32)
    img = F.astype(np.float32) @ W.T
    img /= np.linalg.norm(img, axis=-1, keepdims=True) + 1e-12
    return (img @ d).astype(np.float32)


def stage_verify(args, log) -> int:
    from scipy.stats import spearmanr
    ids = ec.cohort("HateMM")[:5]
    vdir = OUT / "verify_hatemm_features"
    stats = extract([(v, HMM_VIDEO / f"{v}.mp4") for v in ids], vdir, args.batch, log)
    pr = np.load(PROMPT_EMB, allow_pickle=True)
    camp = np.load(RH / "archive/idea-stage/repro_zs_clip/scores_visual_HateMM.npz", allow_pickle=True)
    set_names = [str(x) for x in camp["set_names"]]
    res, ok = {}, True
    for v in ids:
        mine = np.load(vdir / f"{v}.npy")
        ref = np.load(RH / f"data/CLIP_Embedding/HateMM/dense4fps_clipL336/{v}.npy")
        n = min(len(mine), len(ref))
        cos = float(np.mean(np.sum(mine[:n] * ref[:n], 1) /
                            (np.linalg.norm(mine[:n], axis=1) * np.linalg.norm(ref[:n], axis=1) + 1e-12)))
        s_mine = score_feats(mine, pr)
        s_ref = np.asarray(camp[v][set_names.index("main")], dtype=np.float32)
        m = min(len(s_mine), len(s_ref))
        res[v] = {"T_mine": len(mine), "T_campaign": len(ref), "mean_cos_features": cos,
                  "max_abs_feature_diff": float(np.abs(mine[:n] - ref[:n]).max()),
                  "max_abs_score_diff": float(np.abs(s_mine[:m] - s_ref[:m]).max()),
                  "score_spearman": float(spearmanr(s_mine[:m], s_ref[:m]).correlation)}
        ok &= len(mine) == len(ref) and cos > 0.999
        log(f"VERIFY {v} {json.dumps(res[v])}")
    (OUT / "verify.json").write_text(json.dumps({"ok": ok, "videos": res, "extract": stats}, indent=2) + "\n")
    log(f"VERIFY {'PASSED' if ok else 'FAILED'} (same frame count and mean feature cosine > 0.999 on 5 HateMM videos)")
    return 0 if ok else 4


def stage_extract(args, log) -> int:
    man = ec.manifest_rows("DeHate")
    jobs = [(v, Path(r["video_path"])) for v, r in sorted(man.items())]
    missing = [v for v, p in jobs if not p.exists()]
    if missing:
        log(f"FAILED {len(missing)} DeHate videos missing on this machine, e.g. {missing[:3]}")
        return 5
    stats = extract(jobs, FEAT_DIR, args.batch, log)
    (OUT / "extract_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"extract done: {stats['n']} new, empty {len(stats['empty'])}, errors {len(stats['errors'])}, "
        f"{stats['wall_seconds']} s")
    return 0


def stage_score(args, log) -> int:
    pr = np.load(PROMPT_EMB, allow_pickle=True)
    dur = ec.durations("DeHate")
    curves, extra, fail = {}, {}, {}
    n_tail = 0
    for v in ec.cohort("DeHate"):
        p = FEAT_DIR / f"{v}.npy"
        if not p.exists():
            fail[v] = "no feature file (decode failure)"
            continue
        s = score_feats(np.load(p), pr)
        T = ec.curve_length(dur[v])
        tail = ec.tail_frames(len(s), FPS, T)
        n_tail += tail
        curves[v] = ec.broadcast_to_4fps(s, FPS, T)
        extra[v] = {"native_samples": int(len(s)), "tail_hold_frames": int(tail)}
    rep = ec.finalize("zs_clip_main", "DeHate", OUT, curves, native_rate=FPS, code_path=CODE_PATH, log=log,
                      extra=extra, failures=fail, notes={"tail_hold_frames_total": n_tail},
                      config={"variant": "main ('a normal video frame' / 'a hateful video frame')",
                              "backbone": CLIP_MODEL + " CLIPVisionModel fp16 pooler_output + visual_projection",
                              "features": str(FEAT_DIR.relative_to(REPO)),
                              "prompt_embeddings": str(PROMPT_EMB.relative_to(REPO)),
                              "native_rate": "4 fps", "score": "cos(img, hateful) - cos(img, normal)",
                              "grid": "native 4 fps; frames past the last decoded frame hold its value"})
    return 0 if rep.get("exact_test_set") else 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify", "extract", "score"])
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = ec.RunLog(OUT / "run.log", append=True)
    (OUT / "run.pid").write_text(f"{os.getpid()}\n")
    log(f"stage {a.stage}; code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
    rc = {"verify": stage_verify, "extract": stage_extract, "score": stage_score}[a.stage](a, log)
    log(f"stage {a.stage} {'DONE' if rc == 0 else f'FAILED rc={rc}'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
