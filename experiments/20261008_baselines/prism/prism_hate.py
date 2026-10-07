#!/usr/bin/env python3
"""PRISM (ICML 2026, github.com/ytC2026/ICML2026-PRISM) on HateMM / HateClipSeg / DeHate.

Release files followed: feature_extraction/extract_qwen_xd.py (video side) and Code/Qwen_xd.py (text side and
scoring, method "M6 PRISM (Full)"). Backbone Qwen/Qwen3-VL-Embedding-2B with the release's Qwen3VLEmbedder.

Video side (extract_qwen_xd.py): decord reads the file; every 16th native frame is sampled (SAMPLE_INTERVAL = 16);
windows of 7 consecutive samples with stride 1 (WINDOW_SIZE = 7, WINDOW_STRIDE = 1) are embedded with the
instruction "Represent the video content for anomaly detection." (Qwen3VLEmbedder(max_pixels=600*1000); the
embedder's own frame sampler expands each 7-frame list to 64 frames, as in the release).
  Engineering changes, same arithmetic: windows of one video are embedded in batches of 8 (all windows of a video
  have the same size, so there is no padding); the model is loaded in bfloat16 (the checkpoint's dtype; the release
  passes no dtype); each sampled frame is resized once per video instead of once per occurrence in every window
  (`frame_cache`; `check_fast_path` asserts identical processor inputs), and CPU preprocessing of the next batch
  overlaps the GPU forward. A file decord cannot open is re-encoded to H.264 once (lf_common.transcode_h264) and listed.

Text side (Qwen_xd.py, unchanged arithmetic): normal pool embedded with the prefix "Video footage of ", each class
pool with "Real-world video footage of anomalous event: " (the release's XD templates; the UCF template "A
surveillance video of" is not used); normal centre = mean normal embedding; covariance of all text embeddings,
whitening P = (S + 0.01 I)^(-1/2); class axis_c = normalize(P (mean_c - centre)); window score
s = sum_c l_c softmax(l / 0.2)_c with l = (normalize(v) - centre) . axis; Gaussian smoothing over windows with
sigma = 16. Pools: experiments/20261008_baselines/prism/hate_descriptions.json (make_descriptions.py; 9 rules + the
positive query as 10 classes, "normal content" + 20 normal scenes).

Mapping to frames (coordinator rule 5, 2026-10-08): the release scores window i at its start (gt index i <-> native
frames [16i, 16i+16)) and leaves the last 6 sample positions unscored. Here window i covers native frames
[16 i, 16 (i + 7)), and a native frame takes the mean smoothed score of the windows that cover it. Frames after the
last covered frame (fewer than 16 native frames) hold the last value. 4 fps frame j takes the native frame at
(j + 0.5) / 4 s. Videos with fewer than 7 samples (fewer than 97 native frames) have no window: plan F2 (median).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, BASE)
import lf_common as L                                            # noqa: E402

METHOD = "prism"
OUT_ROOT = os.path.join(REPO, "runs", "20261008_baselines", "prism")
CODE_PATH = "experiments/20261008_baselines/prism/prism_hate.py"
DESC_PATH = os.path.join(HERE, "hate_descriptions.json")
SAMPLE_INTERVAL, WINDOW_SIZE, WINDOW_STRIDE = 16, 7, 1
INSTRUCTION = "Represent the video content for anomaly detection."
NORMAL_PREFIX = "Video footage of "
ANOM_PREFIX = "Real-world video footage of anomalous event: "
TEMP_B, SIGMA, REG_LAMBDA, TEXT_BATCH = 0.2, 16, 0.01, 16


def load_embedder(model_dir, dtype="bfloat16"):
    import torch
    sys.path.insert(0, model_dir)
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    return Qwen3VLEmbedder(model_name_or_path=model_dir, max_pixels=600 * 1000, dtype=getattr(torch, dtype))


def read_samples(path):
    """extract_qwen_xd.get_sliding_windows, returning the sampled frames once (windows share frames)."""
    from decord import VideoReader, cpu
    from PIL import Image
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    idx = list(range(0, total, SAMPLE_INTERVAL))
    frames = [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()] if idx else []
    return frames, total, float(vr.get_avg_fps())


def frame_cache(emb, frames):
    """Per-frame part of qwen_vl_utils.fetch_video (list-of-frames branch), computed once per sampled frame.

    In the release path every window is a list of 7 frames that the embedder expands to 64 (each frame repeated
    about 9 times) and fetch_video resizes all 64 frames of every window: fetch_image per frame (PIL), then one
    bicubic antialiased torch resize to the size set by total_pixels / 64. Both steps act on each frame alone and all
    windows of a video share frames and target size, so each sampled frame is resized once here and windows index
    the cache. `check_fast_path` asserts that the processor inputs are identical to the release path."""
    import torch
    from qwen_vl_utils import vision_process as vp
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    image_factor = 16 * vp.SPATIAL_MERGE_SIZE                       # image_patch_size=16 in _preprocess_inputs
    info = {"total_pixels": emb.total_pixels}                       # format_model_input's kwargs for a frame list
    imgs = [vp.fetch_image({"image": f, **info}, image_factor) for f in frames]
    stacked = torch.stack([torch.from_numpy(np.array(im).transpose(2, 0, 1)) for im in imgs])
    nframes = vp.ceil_by_factor(min(emb.num_frames, emb.max_frames), vp.FRAME_FACTOR)
    min_pixels = vp.VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    max_pixels = max(min(vp.VIDEO_MAX_TOKEN_NUM * image_factor * image_factor,
                         info["total_pixels"] / nframes * vp.FRAME_FACTOR), int(min_pixels * 1.05))
    rh, rw = vp.smart_resize(stacked.shape[2], stacked.shape[3], factor=image_factor, min_pixels=min_pixels,
                             max_pixels=max_pixels)
    return transforms.functional.resize(stacked, [rh, rw], interpolation=InterpolationMode.BICUBIC,
                                        antialias=True).float()


def window_inputs(emb, frames, cache, starts):
    """Qwen3VLEmbedder._preprocess_inputs for a batch of windows, reading the per-frame cache."""
    rel = np.linspace(0, WINDOW_SIZE - 1, emb.num_frames, dtype=int)[:emb.max_frames]
    convs = [emb.format_model_input(video=frames[i:i + WINDOW_SIZE], instruction=INSTRUCTION) for i in starts]
    text = emb.processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
    n = len(rel)
    videos = [cache[i + rel] for i in starts]
    meta = [dict(fps=2.0, frames_indices=list(range(n)), total_num_frames=(n / 2.0) * 2.0) for _ in starts]
    return emb.processor(text=text, images=None, videos=videos, video_metadata=meta, truncation=True,
                         max_length=emb.max_length, padding=True, do_resize=False, return_tensors="pt",
                         do_sample_frames=False)


def embed_inputs(emb, inputs):
    """Qwen3VLEmbedder.process after preprocessing: forward, last-token pooling, L2 normalisation."""
    import torch.nn.functional as F
    inputs = {k: v.to(emb.model.device) for k, v in inputs.items()}
    out = emb.forward(inputs)
    return F.normalize(emb._pooling_last(out["last_hidden_state"], out["attention_mask"]), p=2, dim=-1)


def check_fast_path(emb, frames, starts):
    """True if the cached path gives the release path's processor inputs exactly."""
    import torch
    convs = [emb.format_model_input(video=frames[i:i + WINDOW_SIZE], instruction=INSTRUCTION) for i in starts]
    ref = emb._preprocess_inputs(convs)
    new = window_inputs(emb, frames, frame_cache(emb, frames), starts)
    return set(ref) == set(new) and all(torch.equal(ref[k], new[k]) for k in ref)


def cmd_extract(args):
    import torch
    ds = args.dataset
    dest = os.path.join(OUT_ROOT, ds)
    fdir = os.path.join(dest, "features")
    os.makedirs(fdir, exist_ok=True)
    L.start_run_log(dest)
    man = L.manifest(ds)
    ids = L.cohort(ds)
    if args.limit:
        ids = ids[:args.limit]
    meta_path = os.path.join(dest, "extract_meta.jsonl")
    done = set()
    if os.path.isfile(meta_path):
        for line in open(meta_path):
            r = json.loads(line)
            if not r.get("error"):
                done.add(r["video_id"])
    todo = [v for v in ids if v not in done]
    print("%s extract: %d cohort, %d done, %d to do" % (ds, len(ids), len(done), len(todo)), flush=True)
    if not todo:
        return 0
    emb = load_embedder(args.model, args.dtype)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)
    with open(os.path.join(dest, "extract_config.json"), "w") as fh:
        json.dump({"dataset": ds, "model": args.model, "dtype": args.dtype, "sample_interval": SAMPLE_INTERVAL,
                   "window_size": WINDOW_SIZE, "window_stride": WINDOW_STRIDE, "instruction": INSTRUCTION,
                   "embedder_max_pixels": 600 * 1000, "batch": args.batch,
                   "preprocessing": "per-frame cache (frame_cache) + prefetch of the next batch", "code_version": L.git_version(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    t_start, n_win = time.time(), 0
    with open(meta_path, "a") as out:
        for k, vid in enumerate(todo, 1):
            src = man[vid]["video_path"]
            rec = {"video_id": vid, "video_path": src}
            t0 = time.time()
            tmp = None
            try:
                path = src
                if not L.decord_ok(src):
                    tmp = os.path.join(dest, "_transcode", vid + ".mp4")
                    path = L.transcode_h264(src, tmp)
                    rec["transcoded"] = True
                frames, total, fps = read_samples(path)
                rec.update(n_native_frames=total, native_fps=fps, n_samples=len(frames))
                starts = list(range(0, len(frames) - WINDOW_SIZE + 1, WINDOW_STRIDE))
                rec["n_windows"] = len(starts)
                if starts:
                    embs = []
                    cache = frame_cache(emb, frames)
                    batches = [starts[b:b + args.batch] for b in range(0, len(starts), args.batch)]
                    nxt = pool.submit(window_inputs, emb, frames, cache, batches[0])
                    for j in range(len(batches)):
                        inputs = nxt.result()
                        if j + 1 < len(batches):                   # CPU preprocessing of the next batch overlaps
                            nxt = pool.submit(window_inputs, emb, frames, cache, batches[j + 1])
                        with torch.no_grad():
                            embs.append(embed_inputs(emb, inputs).float().cpu())
                    arr = torch.cat(embs).numpy().astype(np.float32)
                    np.save(os.path.join(fdir, vid + ".npy"), arr)
                    n_win += len(arr)
            except Exception as exc:                                       # noqa: BLE001
                rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                torch.cuda.empty_cache()
            finally:
                if tmp and os.path.isfile(tmp):
                    os.remove(tmp)
            rec["wall_s"] = round(time.time() - t0, 2)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            el = time.time() - t_start
            print("[%4d/%4d] %-22s %4s win %6.1fs | %.2f win/s | eta %.1f min%s" % (
                k, len(todo), vid, rec.get("n_windows"), rec["wall_s"], n_win / max(el, 1e-9),
                (len(todo) - k) * el / k / 60, ("  ERROR " + rec["error"][:200]) if rec.get("error") else ""),
                flush=True)
    print("EXTRACT_DONE %s %.1f min" % (ds, (time.time() - t_start) / 60), flush=True)
    return 0


def build_axes(emb, desc):
    """Qwen_xd.py section 2 (text side), M6 only."""
    import torch
    import torch.nn.functional as F
    device = emb.model.device

    def get_embeddings(texts, prefix):
        texts = [prefix + t for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), TEXT_BATCH):
                out.append(emb.process([{"text": t} for t in texts[i:i + TEXT_BATCH]]).to(device).float())
        return torch.cat(out)

    norm_texts = []
    for _, v in desc["prompt_config"]["scenario_normals"].items():
        norm_texts.extend(v)
    norm_embs = F.normalize(get_embeddings(norm_texts, NORMAL_PREFIX), p=2, dim=1)
    centre = norm_embs.mean(0)
    code_map = desc["prompt_config"]["class_overrides"]
    anom = desc["content"]["anomalies"]
    embs, means = [], []
    for code in sorted(code_map):
        feats = F.normalize(get_embeddings(anom[code_map[code]], ANOM_PREFIX), p=2, dim=1)
        embs.append(feats)
        means.append(feats.mean(0))
    ref = torch.cat([norm_embs] + embs, 0)
    diff = ref - ref.mean(0)
    sigma = diff.T @ diff / (len(ref) - 1)
    sigma = sigma + REG_LAMBDA * torch.eye(sigma.shape[0], device=device)
    lam, vec = torch.linalg.eigh(sigma)
    precision = vec @ torch.diag(1.0 / torch.sqrt(torch.clamp(lam, min=1e-7))) @ vec.T
    axes = torch.stack([F.normalize(precision @ (m - centre), p=2, dim=0) for m in means]).t()
    return centre.cpu().numpy(), axes.cpu().numpy(), {"n_normal": len(norm_texts), "n_classes": len(means),
                                                      "n_text": int(len(ref)), "classes": sorted(code_map)}


def window_scores(feats, centre, axes):
    """Qwen_xd.py M6 + gaussian_filter1d(SIGMA) over the window sequence."""
    from scipy.ndimage import gaussian_filter1d
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    lg = (f - centre) @ axes
    w = np.exp((lg - lg.max(1, keepdims=True)) / TEMP_B)
    w = w / w.sum(1, keepdims=True)
    return gaussian_filter1d((lg * w).sum(1), SIGMA)


def windows_to_native(scores, n_native):
    """Window i covers native frames [16 i, 16 (i + 7)); a frame takes the mean score of covering windows;
    frames past the last covered frame hold the last value."""
    acc = np.zeros(n_native)
    cnt = np.zeros(n_native)
    for i, s in enumerate(scores):
        a, b = SAMPLE_INTERVAL * i * WINDOW_STRIDE, min(n_native, SAMPLE_INTERVAL * (i * WINDOW_STRIDE + WINDOW_SIZE))
        acc[a:b] += s
        cnt[a:b] += 1
    out = np.full(n_native, np.nan)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    last = np.flatnonzero(m)[-1]
    out[last + 1:] = out[last]
    return out, int(n_native - last - 1)


def cmd_score(args):
    ds = args.dataset
    dest = os.path.join(OUT_ROOT, ds)
    L.start_run_log(dest)
    man = L.manifest(ds)
    with open(DESC_PATH) as fh:
        desc = json.load(fh)
    emb = load_embedder(args.model, args.dtype)
    centre, axes, tinfo = build_axes(emb, desc)
    del emb
    meta = {}
    for line in open(os.path.join(dest, "extract_meta.jsonl")):
        r = json.loads(line)
        if not r.get("error") or r["video_id"] not in meta:
            meta[r["video_id"]] = r
    rows, stats = {}, {"no_window": [], "transcoded": [], "tail_hold_native_frames": 0, "windows": 0}
    for vid in L.cohort(ds):
        dur = man[vid]["duration"]
        n = L.n_frames(dur)
        r = meta.get(vid)
        if r is None or r.get("error"):
            rows[vid] = L.row(METHOD, ds, vid, dur, [], "windows", CODE_PATH, error=(r or {}).get("error", "no row"))
            continue
        if r.get("transcoded"):
            stats["transcoded"].append(vid)
        if not r["n_windows"]:
            stats["no_window"].append(vid)
            rows[vid] = L.row(METHOD, ds, vid, dur, [], "windows", CODE_PATH,
                              error="fewer than %d sampled frames (%d native frames)" % (WINDOW_SIZE,
                                                                                        r["n_native_frames"]))
            continue
        feats = np.load(os.path.join(dest, "features", vid + ".npy"))
        s = window_scores(feats, centre, axes)
        native, tail = windows_to_native(s, r["n_native_frames"])
        stats["tail_hold_native_frames"] += tail
        stats["windows"] += len(s)
        fps = r["native_fps"]
        idx = np.clip(np.floor((np.arange(n) + 0.5) / L.RATE * fps).astype(int), 0, len(native) - 1)
        rows[vid] = L.row(METHOD, ds, vid, dur, native[idx], "windows", CODE_PATH,
                          extra={"n_windows": len(s), "native_fps": fps, "n_native_frames": r["n_native_frames"],
                                 "transcoded": bool(r.get("transcoded"))}, calls=len(s))
    med, failed = L.apply_f2(rows, ds)
    summary = {"n_videos": len(rows), "n_f2": len(failed), "f2": failed, "f2_median": med, "text": tinfo,
               "no_window": stats["no_window"], "transcoded": stats["transcoded"],
               "tail_hold_native_frames": stats["tail_hold_native_frames"], "n_windows": stats["windows"]}
    with open(os.path.join(dest, "score_stats.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(dest, "run.log"), "a") as fh:
        fh.write("score: %s\n" % json.dumps({k: v for k, v in summary.items() if k != "f2"}))
        for vid, why in failed:
            fh.write("F2 %s: %s\n" % (vid, why))
    print(json.dumps({k: v for k, v in summary.items() if k != "f2"}, indent=1))
    with open(os.path.join(dest, "config.json"), "w") as fh:
        json.dump({"method": METHOD, "dataset": ds, "code": CODE_PATH, "code_version": L.git_version(),
                   "descriptions": "experiments/20261008_baselines/prism/hate_descriptions.json",
                   "prefixes": [NORMAL_PREFIX, ANOM_PREFIX], "temp_b": TEMP_B, "sigma_windows": SIGMA,
                   "reg_lambda": REG_LAMBDA, "variant": "M6 PRISM (Full)", "extract_config": "extract_config.json",
                   "to_native": "window i covers native frames [16i, 16(i+7)); mean over covering windows; tail hold",
                   "to_4fps": "native frame at (j+0.5)/4 s", "fallback_F2": "median frame score (videos with no "
                   "window or a failed decode)", "date": time.strftime("%Y-%m-%d")}, fh, indent=2)
    if len(failed) > 0.01 * len(rows):
        print("STOP: %d F2 videos > 1%% of %d" % (len(failed), len(rows)))
        np.save(os.path.join(dest, "_axes_debug.npy"), axes)
        return 1
    with open(os.path.join(dest, "scored_rows.jsonl"), "w") as fh:
        for r in rows.values():
            fh.write(json.dumps(r) + "\n")
    return 0


def cmd_check(args):
    """CPU check of the cached preprocessing against the release path on the first videos of a corpus."""
    sys.path.insert(0, args.model)
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    import scripts.qwen3_vl_embedding as q

    class TextOnly(Qwen3VLEmbedder):                       # the release's attributes, without loading weights
        def __init__(self, path):
            self.max_length, self.min_pixels, self.max_pixels = q.MAX_LENGTH, q.MIN_PIXELS, 600 * 1000
            self.total_pixels, self.fps, self.num_frames, self.max_frames = q.MAX_TOTAL_PIXELS, q.FPS, \
                q.MAX_FRAMES, q.MAX_FRAMES
            self.default_instruction = "Represent the user's input."
            self.processor = Qwen3VLProcessor.from_pretrained(path, padding_side="right")
    emb = TextOnly(args.model)
    man = L.manifest(args.dataset)
    ok = True
    for vid in L.cohort(args.dataset)[:args.limit or 2]:
        frames, _, _ = read_samples(man[vid]["video_path"])
        n = len(frames) - WINDOW_SIZE + 1
        for starts in ([0, 1, 2, 3], list(range(max(0, n - 3), n))):
            same = check_fast_path(emb, frames, starts)
            ok &= same
            print(vid, starts, "identical" if same else "DIFFERENT", flush=True)
    print("CHECK_OK" if ok else "CHECK_FAILED")
    return 0 if ok else 1


def cmd_eval(args):
    ds = args.dataset
    dest = os.path.join(OUT_ROOT, ds)
    rows = [json.loads(l) for l in open(os.path.join(dest, "scored_rows.jsonl"))]
    L.finalize(ds, rows, dest, METHOD)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("extract", "score", "eval", "check"))
    ap.add_argument("--dataset", choices=L.DATASETS, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="Qwen3-VL-Embedding-2B snapshot directory")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    return {"extract": cmd_extract, "score": cmd_score, "eval": cmd_eval, "check": cmd_check}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
