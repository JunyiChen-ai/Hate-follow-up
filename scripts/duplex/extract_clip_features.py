"""Reproduction study, Phase 2 task 2: CLIP ViT-B/16 visual features at 1 fps.

VadCLIP (AAAI'24) and DSANet (AAAI'26) both consume per-frame CLIP image
embeddings; extracting them once here is what lets the two baselines be
compared on identical input rather than on two separately-tuned feature
pipelines.

Frame grid. The features live on *the same grid as the frozen frame-level
ground truth* (docs/duplex/FRAME_EVAL_PROTOCOL.md,
scripts/duplex/frame_eval_common.frame_times): timestamps t = 0, 1, 2, ...
while t < duration, with duration taken from the wav file, exactly as
build_gt_arrays.py takes it. That is deliberate. Any other grid -- a plain
floor(duration), or a grid keyed off the video container instead of the audio
-- would force an ad-hoc crop between a baseline's per-frame scores and the
gold array, and the crop would differ per video. Here row i of the feature
matrix is frame i of the gold array by construction.

Timestamps are converted to frame indices as round(t * avg_fps), clamped to
the last decodable frame. A clamp means the audio outlives the video stream;
the count is recorded per video so the mismatch is visible rather than silent.

Decoding uses decord, and falls back to the system ffmpeg (`-vf fps=1`)
whenever decord fails -- both when it refuses the file up front and when its
threaded decoder dies part-way through the forward pass, which is why the
retry re-runs the encoder from scratch rather than resuming. That fallback is
not cosmetic: a large minority of the MultiHateClip mp4 files carry AV1 video
and decord's bundled ffmpeg has no AV1 decoder, so without it 202 EN and 48
ZH videos would silently drop out of every baseline trained on these
features; five HateClipSeg files fail the second way, mid-stream on damaged
h264 packets. Which backend produced a video, and the decord error that sent
it to the fallback, are recorded per video in index.json.

Preprocessing is CLIP's own released transform, run through
`CLIPImageProcessor` rather than reimplemented: shortest-edge bicubic resize to
224, centre crop, rescale, normalise. The encoder is `openai/clip-vit-base-
patch16` in fp16.

Output: <out-root>/<corpus>/<video_id>.npy, float32, shape (T, 512), the raw
(unnormalized) CLIP image-embedding space. Plus <corpus>/index.json with the
per-video grid bookkeeping and the failure list.

  python scripts/duplex/extract_clip_features.py --corpus hatemm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

from frame_eval_common import frame_times  # noqa: E402

DATA_ROOT = os.environ.get("HVD_DATA_ROOT", "/home/jehc223/data")
SPLITS = os.path.join(PROJECT_ROOT, "results", "reproduction", "splits")
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "features",
                        "clip_b16_1fps")

MODEL_ID = "openai/clip-vit-base-patch16"
FPS = 1.0

# corpus -> video directory, wav directories searched in order for the
# duration, and the split manifests that define the id set.
CORPORA = {
    "hatemm": {
        "video_dir": os.path.join(DATA_ROOT, "HateMM", "video"),
        "wav_dirs": ["/home/jehc223/Retrieval-hate/data/AV2A_wav/HateMM",
                     os.path.join(DATA_ROOT, "HateMM", "wav"),
                     os.path.join(PROJECT_ROOT, "results", "testruns",
                                  "hatemm", "wav")],
        "splits": ["hatemm_train.txt", "hatemm_val.txt", "hatemm_test.txt"],
        "chunk_manifests": [
            "results/reproduction/asr/hatemm_all/timestamped_chunks.jsonl",
        ],
    },
    "mhclip_en": {
        "video_dir": os.path.join(DATA_ROOT, "Multihateclip", "English",
                                  "video_mp4"),
        "wav_dirs": ["/home/jehc223/Retrieval-hate/data/AV2A_wav/MHC",
                     os.path.join(DATA_ROOT, "Multihateclip", "English",
                                  "wav"),
                     os.path.join(PROJECT_ROOT, "results", "testruns",
                                  "mhclip_en", "wav")],
        "splits": ["mhclip_en_train.txt", "mhclip_en_val.txt",
                   "mhclip_en_test.txt"],
        "chunk_manifests": [
            "results/reproduction/asr/mhclip_en_all/timestamped_chunks.jsonl",
        ],
    },
    "hateclipseg": {
        "video_dir": os.path.join(DATA_ROOT, "HateClipSeg", "videos"),
        # HateClipSeg's audio was extracted into the repo rather than
        # alongside the media, so there is a single wav directory here.
        "wav_dirs": ["/home/jehc223/Retrieval-hate/data/AV2A_wav/HateClipSeg",
                     os.path.join(PROJECT_ROOT, "results", "hateclipseg",
                                  "wav")],
        "splits": ["hateclipseg_train.txt", "hateclipseg_val.txt",
                   "hateclipseg_test.txt"],
        "chunk_manifests": [
            "results/reproduction/asr/hateclipseg_all/"
            "timestamped_chunks.jsonl",
        ],
    },
    "mhclip_zh": {
        "video_dir": os.path.join(DATA_ROOT, "Multihateclip", "Chinese",
                                  "video"),
        "wav_dirs": ["/home/jehc223/Retrieval-hate/data/AV2A_wav/MHC_zh",
                     os.path.join(DATA_ROOT, "Multihateclip", "Chinese",
                                  "wav"),
                     os.path.join(PROJECT_ROOT, "results", "testruns",
                                  "mhclip_zh", "wav")],
        "splits": ["mhclip_zh_train.txt", "mhclip_zh_val.txt",
                   "mhclip_zh_test.txt"],
        "chunk_manifests": [
            "results/reproduction/asr/mhclip_zh_all/timestamped_chunks.jsonl",
        ],
    },
}


def read_ids(spec):
    ids = []
    for name in spec["splits"]:
        with open(os.path.join(SPLITS, name), encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.append(line.strip())
    seen, out = set(), []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


def find_video_path(video_dir, video_id):
    """Resolve mixed container extensions without changing video identity."""
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv"):
        path = os.path.join(video_dir, video_id + ext)
        if os.path.isfile(path):
            return path
    return os.path.join(video_dir, video_id + ".mp4")


def wav_duration_seconds(path):
    import wave
    with wave.open(path, "rb") as handle:
        frames, rate = handle.getnframes(), handle.getframerate()
    if rate <= 0:
        raise ValueError("non-positive sample rate in %s" % path)
    return frames / float(rate)


def video_duration_seconds(path):
    """Container-duration fallback for released videos with no audio track."""
    import subprocess
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def load_chunk_durations(spec):
    """video_id -> wav_duration, from this corpus's chunk manifests.

    This has to be the first duration source, and it has to win over the wav
    header, because build_gt_arrays.py resolves duration the same way and the
    gold arrays are what the features are scored against. The frozen
    audio_meta.jsonl behind the older manifests rounded wav_duration to two
    decimals, so a video whose true length is a hair over a whole second --
    45.000312 stored as 45.0 -- gets a 45-frame gold grid where the wav header
    would give 46. Reading the header here instead would hand five videos a
    feature matrix one row longer than their gold vector.
    """
    out = {}
    for rel in spec["chunk_manifests"]:
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                dur = rec.get("wav_duration")
                if rec["video_id"] in out or dur is None or float(dur) <= 0:
                    continue
                out[rec["video_id"]] = (float(dur), rel)
    return out


def find_duration(vid, spec, chunk_durations, video_path=None):
    """(duration, source): the chunk manifest first, the wav header second."""
    hit = chunk_durations.get(vid)
    if hit is not None:
        return hit
    for wav_dir in spec["wav_dirs"]:
        path = os.path.join(wav_dir, vid + ".wav")
        if os.path.isfile(path):
            duration = wav_duration_seconds(path)
            if duration > 0:
                return duration, os.path.relpath(
                    path, PROJECT_ROOT if path.startswith(PROJECT_ROOT) else "/")
    if video_path and os.path.isfile(video_path):
        duration = video_duration_seconds(video_path)
        if duration > 0:
            return duration, "video-container (no released wav)"
    return None, None


def ffmpeg_extract_1fps(path, work_dir):
    """Decode a video to one PNG per second with the system ffmpeg.

    decord ships its own ffmpeg build, and that build has no AV1 decoder: a
    large minority of the MultiHateClip mp4 files carry AV1 video and fail in
    `decord.VideoReader` with "cannot find video stream". Those files are not
    corrupt -- the system ffmpeg reads them -- so this is the fallback decode
    path, used only after decord has refused a file. The `fps=1` filter emits
    one frame per output second starting at t = 0, which is the same grid the
    decord path samples.

    Returns the sorted list of written PNG paths. Raises with ffmpeg's own
    stderr attached when ffmpeg cannot read the file either.
    """
    import subprocess
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
           "-vf", "fps=1", "-vsync", "0",
           "-f", "image2", os.path.join(work_dir, "%08d.png")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    frames = sorted(f for f in os.listdir(work_dir) if f.endswith(".png"))
    if res.returncode != 0 and not frames:
        raise RuntimeError("ffmpeg rc=%d: %s"
                           % (res.returncode, (res.stderr or "").strip()[:300]))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames: %s"
                           % (res.stderr or "").strip()[:300])
    return [os.path.join(work_dir, f) for f in frames]


def decord_frame_source(path, grid, batch_size):
    """(meta, batches, cleanup) reading the 1 fps grid through decord.

    Raises whatever decord raises. The caller is expected to retry through
    ffmpeg_frame_source, because decord fails on this study's media in two
    different places: `VideoReader` refuses AV1 outright, and its threaded
    decoder dies part-way through a damaged h264 stream, long after the
    reader was constructed.
    """
    import decord
    from PIL import Image

    vr = decord.VideoReader(path, num_threads=4)
    avg_fps = float(vr.get_avg_fps())
    n_video = len(vr)
    if not (avg_fps > 0) or n_video <= 0:
        raise ValueError("unusable video stream: fps=%r frames=%r"
                         % (avg_fps, n_video))
    raw_idx = np.rint(grid * avg_fps).astype(np.int64)
    idx = np.clip(raw_idx, 0, n_video - 1)
    meta = {
        "decode_backend": "decord",
        "video_frames": int(n_video),
        "video_avg_fps": round(avg_fps, 6),
        "video_duration": round(n_video / avg_fps, 6),
        "frames_clamped_past_video_end": int((raw_idx > n_video - 1).sum()),
    }

    def batches():
        for s in range(0, len(idx), batch_size):
            arr = vr.get_batch(list(idx[s:s + batch_size])).asnumpy()
            yield [Image.fromarray(a) for a in arr]

    return meta, batches, lambda: None


def ffmpeg_frame_source(path, n_target, batch_size, tmp_dir, decord_exc):
    """(meta, batches, cleanup) reading the 1 fps grid through system ffmpeg.

    The fallback path. `decord_exc` is the failure that sent the video here
    and is recorded in the per-video metadata, so a file that needed the
    fallback -- and the reason -- stays visible in index.json.
    """
    from PIL import Image

    work = tempfile.mkdtemp(prefix="frames1fps_", dir=tmp_dir)
    try:
        png = ffmpeg_extract_1fps(path, work)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    n_video = len(png)
    if n_video >= n_target:
        png = png[:n_target]
        n_clamped = 0
    else:
        # Audio outlives the video stream: hold the last frame, the same
        # thing the decord path's index clamp does.
        n_clamped = n_target - n_video
        png = png + [png[-1]] * n_clamped
    meta = {
        "decode_backend": "ffmpeg",
        "decord_error": ("%s: %s" % (type(decord_exc).__name__,
                                     decord_exc))[:300],
        "video_frames_at_1fps": int(n_video),
        "video_avg_fps": None,
        "video_duration": float(n_video),
        "frames_clamped_past_video_end": int(n_clamped),
    }

    def batches():
        for s in range(0, len(png), batch_size):
            yield [Image.open(p).convert("RGB") for p in png[s:s + batch_size]]

    return meta, batches, lambda: shutil.rmtree(work, ignore_errors=True)


def encode_with_fallback(encode, path, grid, n_target, batch_size, tmp_dir):
    """(feats, meta): run `encode` over the video's frames, decord then ffmpeg.

    `encode` takes the batch generator and returns the feature matrix. It is
    called again from scratch if the decord attempt fails at any point --
    including part-way through the forward pass, which is where decord's
    threaded decoder actually dies on a damaged stream. Re-running the
    encoder is what makes the fallback cover mid-decode failures as well as
    the AV1 files decord refuses to open at all.
    """
    try:
        meta, batches, cleanup = decord_frame_source(path, grid, batch_size)
    except Exception as exc:
        decord_exc = exc
    else:
        try:
            return encode(batches), meta
        except Exception as exc:
            decord_exc = exc
        finally:
            cleanup()

    meta, batches, cleanup = ffmpeg_frame_source(
        path, n_target, batch_size, tmp_dir, decord_exc)
    try:
        return encode(batches), meta
    finally:
        cleanup()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--batch", type=int, default=64,
                    help="frames per CLIP forward pass")
    ap.add_argument("--tmp-dir", default=None,
                    help="scratch directory for the ffmpeg fallback frames")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)

    ids = read_ids(spec)
    todo = [v for v in ids
            if not os.path.isfile(os.path.join(out_dir, v + ".npy"))]
    print("clip [%s]: %d videos in the manifests, %d already extracted, "
          "%d to run" % (args.corpus, len(ids), len(ids) - len(todo),
                         len(todo)), flush=True)
    if args.limit is not None:
        todo = todo[:args.limit]
        print("  --limit: %d videos" % len(todo), flush=True)
    if not todo:
        return 0

    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; this stage is CUDA-only")

    proc = CLIPImageProcessor.from_pretrained(MODEL_ID)
    model = CLIPVisionModelWithProjection.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to("cuda")
    model.eval()
    devices = {p.device.type for p in model.parameters()}
    if devices != {"cuda"}:
        raise SystemExit("ABORT: CLIP parameters are on %s, not cuda" % devices)

    index_path = os.path.join(out_dir, "index.json")
    index = {}
    if os.path.isfile(index_path):
        index = json.load(open(index_path, encoding="utf-8"))

    chunk_durations = load_chunk_durations(spec)
    failures = []
    t0 = time.time()
    n_frames_total = 0
    for i, vid in enumerate(todo, 1):
        path = find_video_path(spec["video_dir"], vid)
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            duration, dur_src = find_duration(vid, spec, chunk_durations, path)
            if duration is None or duration <= 0:
                raise ValueError("no positive wav duration for %s" % vid)
            grid = frame_times(duration, FPS)
            n_target = len(grid)

            def encode(batches, n_target=n_target):
                feats = np.empty((n_target, model.config.projection_dim),
                                 dtype=np.float32)
                written = 0
                with torch.no_grad():
                    for imgs in batches():
                        # CLIP's own released transform, unmodified:
                        # shortest-edge bicubic resize to 224, centre crop,
                        # rescale, normalise.
                        x = proc(images=imgs,
                                 return_tensors="pt")["pixel_values"]
                        x = x.to("cuda", dtype=torch.float16)
                        emb = model(pixel_values=x).image_embeds
                        feats[written:written + len(imgs)] = \
                            emb.float().cpu().numpy()
                        written += len(imgs)
                if written != n_target:
                    raise ValueError("decoded %d frames for a %d-frame grid"
                                     % (written, n_target))
                return feats

            feats, meta = encode_with_fallback(
                encode, path, grid, n_target, args.batch, args.tmp_dir)

            # np.save appends .npy unless the name already ends in it, so the
            # temporary name has to carry the suffix itself.
            tmp = os.path.join(out_dir, vid + ".tmp.npy")
            np.save(tmp, feats)
            os.replace(tmp, os.path.join(out_dir, vid + ".npy"))
            index[vid] = dict(
                {"n_frames": int(n_target),
                 "wav_duration": round(duration, 6),
                 "duration_source": dur_src,
                 "dim": int(feats.shape[1])}, **meta)
            n_frames_total += n_target
        except Exception as exc:
            msg = "%s: %s" % (type(exc).__name__, exc)
            failures.append({"video_id": vid, "error": msg[:400]})
            print("  FAILED %s -- %s" % (vid, msg[:200]), flush=True)
            continue

        if i % 25 == 0 or i == 1:
            el = time.time() - t0
            print("  [%d/%d] %s T=%d clamped=%d | %.1f vid/min, eta %.1f min"
                  % (i, len(todo), vid, index[vid]["n_frames"],
                     index[vid]["frames_clamped_past_video_end"],
                     i / max(el, 1e-9) * 60,
                     (len(todo) - i) / max(i / max(el, 1e-9), 1e-9) / 60),
                  flush=True)
        if i % 100 == 0:
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=1, sort_keys=True)

    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1, sort_keys=True)
    # The .npy files are the record of what succeeded, so failures.json
    # carries this attempt's failures only -- a video that a later run
    # recovers must not stay listed as a failure.
    fail_path = os.path.join(out_dir, "failures.json")
    if failures:
        with open(fail_path, "w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=1)
    elif os.path.isfile(fail_path):
        os.remove(fail_path)

    clamped = sum(1 for v in index.values()
                  if v["frames_clamped_past_video_end"])
    via_ffmpeg = sum(1 for v in index.values()
                     if v.get("decode_backend") == "ffmpeg")
    print("clip [%s] done: %d/%d extracted this run, %d in the manifests "
          "with features, %d frames this run, %d videos decoded through the "
          "ffmpeg fallback, %d videos with frames clamped past the video "
          "end, %d failures, %.1fs"
          % (args.corpus, len(todo) - len(failures), len(todo), len(index),
             n_frames_total, via_ffmpeg, clamped, len(failures),
             time.time() - t0), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
