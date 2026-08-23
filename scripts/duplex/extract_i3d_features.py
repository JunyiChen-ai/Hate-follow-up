"""Reproduction study, Phase 2 task 3: I3D RGB snippet features, 5-crop.

MACIL-SD (MM'22) is trained and released against the XD-Violence feature
release, and it reads the visual stream in that release's exact layout: one
1024-d I3D RGB vector per non-overlapping 16-frame snippet of a 24 fps decode,
five spatial crops per snippet, i.e. a (n_snippets, 5, 1024) tensor per video.
Handing it anything else -- a different snippet length, a single centre crop,
a 1 fps grid -- would not be MACIL-SD with new data; it would be a different
model wearing MACIL-SD's name. So the XD-Violence convention is reproduced
here rather than adapted.

Weights. `rgb_imagenet.pt` from piergiaj/pytorch-i3d
(https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt,
sha256 2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538): the
RGB I3D of Carreira and Zisserman trained on Kinetics-400 from
ImageNet-inflated initialisation, which is the release the violence-detection
literature standardised on. The architecture is vendored verbatim in
scripts/duplex/i3d_model.py so the state dict loads with strict=True; a
failure to load strictly aborts rather than falling back to partial weights.
The feature is the `extract_features` endpoint -- global average pool after
Mixed_5c, before dropout and the classifier -- squeezed to 1024-d.

Decoding. Every video goes through the system ffmpeg, not decord: decord's
bundled ffmpeg cannot read the AV1 files that make up a large minority of
MultiHateClip, and at 24 fps the whole corpus has to stream rather than sit in
memory. ffmpeg resamples to 24 fps and scales the shortest edge to 256 in one
filter chain, then writes PPM frames to a pipe. PPM is used instead of raw
video because each frame carries its own width and height in its header: the
scaled size depends on the source aspect ratio and on any rotation metadata
ffmpeg applies for us, and reading it off the stream removes an entire class
of silent frame-desynchronisation bug that guessing the size from ffprobe
would invite.

Preprocessing. 5 crops of 224x224 per snippet in the fixed order
(top-left, top-right, bottom-left, bottom-right, centre), then the I3D
convention x -> x / 127.5 - 1. Cropping and normalisation run on the GPU; the
CPU thread only demuxes, so decode and forward overlap.

Snippet grid. Snippet i covers decoded frames [16i, 16i + 16), i.e. seconds
[16i / 24, (16i + 16) / 24). A trailing run of fewer than 16 frames is
dropped, as in XD-Violence, and the number of dropped frames is recorded per
video. The per-video times.json states the mapping explicitly so that a
baseline's snippet scores can be resampled onto the frozen 1 fps gold grid
without anyone having to re-derive it.

Output: <out-root>/<corpus>/<video_id>.npy, float32 (n_snippets, 5, 1024),
plus <video_id>.times.json, plus <corpus>/index.json and failures.json.

  python scripts/duplex/extract_i3d_features.py --corpus hatemm
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

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

from extract_clip_features import (  # noqa: E402
    CORPORA, find_duration, find_video_path, load_chunk_durations, read_ids)

OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "features",
                        "i3d_rgb_5crop")
DEFAULT_WEIGHTS = "/home/jehc223/data/checkpoints/i3d/rgb_imagenet.pt"
WEIGHTS_SHA256 = "2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538"
WEIGHTS_URL = ("https://github.com/piergiaj/pytorch-i3d/raw/master/"
               "models/rgb_imagenet.pt")

DECODE_FPS = 24
SNIPPET = 16
CROP = 224
SHORT_SIDE = 256
CROP_ORDER = ["top_left", "top_right", "bottom_left", "bottom_right", "centre"]

# fps=24 resamples; the scale expression puts the *shorter* edge at 256 and
# lets the longer edge follow the aspect ratio, rounded to an even size.
VF = ("fps=%d,scale='if(gt(iw,ih),-2,%d)':'if(gt(iw,ih),%d,-2)'"
      % (DECODE_FPS, SHORT_SIDE, SHORT_SIDE))


def _read_exact(stream, n):
    """Read exactly n bytes, or None if the stream ends first.

    A short read means ffmpeg was cut off mid-frame; returning the partial
    buffer padded out would hand the model a half-black frame that looks like
    real content, so the truncated frame is dropped instead.
    """
    chunks, got = [], 0
    while got < n:
        buf = stream.read(n - got)
        if not buf:
            return None
        chunks.append(buf)
        got += len(buf)
    return b"".join(chunks)


def _ppm_token(stream):
    """Next whitespace-delimited PPM header token, skipping '#' comments."""
    tok = b""
    while True:
        ch = stream.read(1)
        if not ch:
            return None if not tok else tok
        if ch == b"#":
            while ch and ch not in b"\r\n":
                ch = stream.read(1)
            continue
        if ch.isspace():
            if tok:
                return tok
            continue
        tok += ch


def ppm_frames(stream):
    """Yield HxWx3 uint8 arrays from a stream of concatenated binary PPMs."""
    while True:
        magic = _ppm_token(stream)
        if magic is None:
            return
        if magic != b"P6":
            raise ValueError("expected a P6 PPM header, got %r" % magic[:16])
        width = int(_ppm_token(stream))
        height = int(_ppm_token(stream))
        maxval = int(_ppm_token(stream))
        if maxval != 255:
            raise ValueError("expected 8-bit PPM, got maxval=%d" % maxval)
        payload = _read_exact(stream, width * height * 3)
        if payload is None:
            return
        yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)


def snippet_producer(path, out_q, batch_snippets, err_box):
    """Decode `path` at 24 fps and push uint8 snippet batches onto out_q.

    Runs in its own thread so ffmpeg's decode overlaps the GPU forward pass.
    Pushes (batch, n_frames_seen) tuples and finally None; any exception is
    recorded in err_box and also terminates the queue with None.
    """
    proc = None
    try:
        proc = subprocess.Popen(
            # -pix_fmt rgb24 is load-bearing, not decoration: for a 10-bit
            # source ffmpeg's ppm muxer picks 16-bit rgb48 and every pixel
            # doubles in width. Four MultiHateClip videos are 10-bit.
            ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
             "-vf", VF, "-pix_fmt", "rgb24",
             "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1024 * 1024)
        buf, batch, n_seen = [], [], 0
        for frame in ppm_frames(proc.stdout):
            n_seen += 1
            buf.append(frame)
            if len(buf) == SNIPPET:
                batch.append(np.stack(buf))
                buf = []
                if len(batch) == batch_snippets:
                    out_q.put((np.stack(batch), n_seen))
                    batch = []
        if batch:
            out_q.put((np.stack(batch), n_seen))
        # A trailing run shorter than a full snippet is dropped, as in the
        # XD-Violence release; the count is reported so it is not silent.
        err_box["n_frames"] = n_seen
        err_box["n_tail_frames_dropped"] = len(buf)
        proc.stdout.close()
        rc = proc.wait()
        stderr = proc.stderr.read().decode("utf-8", "replace").strip()
        err_box["ffmpeg_returncode"] = rc
        if rc != 0 and n_seen == 0:
            err_box["error"] = RuntimeError(
                "ffmpeg rc=%d: %s" % (rc, stderr[:300]))
        elif stderr:
            err_box["ffmpeg_stderr"] = stderr[:300]
    except Exception as exc:  # noqa: BLE001 -- surfaced by the consumer
        err_box["error"] = exc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
    finally:
        out_q.put(None)


def five_crop(x):
    """(B, 3, T, H, W) -> (5B, 3, T, 224, 224) in the documented crop order.

    The 5B ordering is crop-major within each snippet: rows 5b .. 5b+4 are the
    five crops of snippet b, which is the layout the (n_snippets, 5, 1024)
    output expects.
    """
    h, w = x.shape[-2], x.shape[-1]
    if h < CROP or w < CROP:
        raise ValueError("frame %dx%d is smaller than the %d crop"
                         % (h, w, CROP))
    top, left = 0, 0
    bottom, right = h - CROP, w - CROP
    cy, cx = bottom // 2, right // 2
    corners = [(top, left), (top, right), (bottom, left), (bottom, right),
               (cy, cx)]
    crops = [x[..., y:y + CROP, xx:xx + CROP] for y, xx in corners]
    # stack on a new axis 1 so the flatten below is crop-major per snippet
    import torch
    return torch.stack(crops, dim=1).flatten(0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--batch", type=int, default=8,
                    help="snippets per forward pass (x5 crops of 16 frames)")
    ap.add_argument("--queue", type=int, default=6,
                    help="decoded batches held in flight ahead of the GPU")
    ap.add_argument("--tmp-dir", default=None,
                    help="accepted for runner symmetry; unused (ffmpeg pipes)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated video ids, for the smoke test")
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)

    ids = read_ids(spec)
    if args.only:
        want = set(args.only.split(","))
        ids = [v for v in ids if v in want]
        todo = list(ids)
    else:
        todo = [v for v in ids
                if not os.path.isfile(os.path.join(out_dir, v + ".npy"))]
    print("i3d [%s]: %d videos in the manifests, %d already extracted, "
          "%d to run" % (args.corpus, len(ids), len(ids) - len(todo),
                         len(todo)), flush=True)
    if args.limit is not None:
        todo = todo[:args.limit]
        print("  --limit: %d videos" % len(todo), flush=True)
    if not todo:
        return 0

    import hashlib

    import torch
    from i3d_model import InceptionI3d

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; this stage is CUDA-only")

    digest = hashlib.sha256(open(args.weights, "rb").read()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise SystemExit("ABORT: %s has sha256 %s, expected %s (%s)"
                         % (args.weights, digest, WEIGHTS_SHA256, WEIGHTS_URL))
    model = InceptionI3d(400, in_channels=3)
    # strict: the vendored definition must match the released state dict
    # exactly. A partial load would produce features that look fine and mean
    # nothing.
    model.load_state_dict(torch.load(args.weights, map_location="cpu"),
                          strict=True)
    # fp32 parameters plus autocast, not a wholesale .half(): I3D's batch-norm
    # layers run with eps = 1e-3 on frozen running statistics, and casting
    # those to fp16 shifts the normalisation enough to move the features.
    # Autocast keeps the convolutions in fp16 and the normalisation in fp32.
    model.eval().cuda()
    devices = {p.device.type for p in model.parameters()}
    if devices != {"cuda"}:
        raise SystemExit("ABORT: I3D parameters are on %s, not cuda" % devices)

    index_path = os.path.join(out_dir, "index.json")
    index = {}
    if os.path.isfile(index_path):
        index = json.load(open(index_path, encoding="utf-8"))

    chunk_durations = load_chunk_durations(spec)
    failures = []
    t0 = time.time()
    n_snip_total, n_sec_total = 0, 0.0
    for i, vid in enumerate(todo, 1):
        path = find_video_path(spec["video_dir"], vid)
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            wav_duration, dur_src = find_duration(
                vid, spec, chunk_durations, path)

            out_q = queue.Queue(maxsize=args.queue)
            box = {}
            thread = threading.Thread(
                target=snippet_producer,
                args=(path, out_q, args.batch, box), daemon=True)
            thread.start()

            feats, frame_shape = [], None
            with torch.no_grad():
                while True:
                    item = out_q.get()
                    if item is None:
                        break
                    batch, _ = item
                    if frame_shape is None:
                        frame_shape = (int(batch.shape[2]),
                                       int(batch.shape[3]))
                    # (B, T, H, W, 3) uint8 -> (B, 3, T, H, W) uint8 on GPU
                    x = torch.from_numpy(batch).cuda(non_blocking=True)
                    x = x.permute(0, 4, 1, 2, 3)
                    x = five_crop(x)
                    # I3D's own input convention: [0, 255] -> [-1, 1]
                    x = x.float().div_(127.5).sub_(1.0)
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = model.extract_features(x)
                    out = out.reshape(out.shape[0], 1024)
                    feats.append(out.float().cpu().numpy()
                                 .reshape(-1, 5, 1024))
            thread.join()
            if box.get("error") is not None:
                raise box["error"]
            if not feats:
                raise ValueError(
                    "no complete %d-frame snippet decoded (%d frames seen, "
                    "ffmpeg rc=%s)" % (SNIPPET, box.get("n_frames", 0),
                                       box.get("ffmpeg_returncode")))
            arr = np.concatenate(feats, axis=0).astype(np.float32)
            n_snip = int(arr.shape[0])

            tmp = os.path.join(out_dir, vid + ".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, os.path.join(out_dir, vid + ".npy"))
            times = [[round(s * SNIPPET / DECODE_FPS, 6),
                      round((s + 1) * SNIPPET / DECODE_FPS, 6)]
                     for s in range(n_snip)]
            with open(os.path.join(out_dir, vid + ".times.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"video_id": vid,
                           "decode_fps": DECODE_FPS,
                           "snippet_frames": SNIPPET,
                           "n_snippets": n_snip,
                           "crop_order": CROP_ORDER,
                           "times": times}, handle)

            covered = n_snip * SNIPPET / float(DECODE_FPS)
            index[vid] = {
                "n_snippets": n_snip,
                "dim": int(arr.shape[2]),
                "n_crops": int(arr.shape[1]),
                "decode_fps": DECODE_FPS,
                "snippet_frames": SNIPPET,
                "frames_decoded": int(box.get("n_frames", 0)),
                "tail_frames_dropped": int(box.get("n_tail_frames_dropped", 0)),
                "scaled_frame_hw": list(frame_shape) if frame_shape else None,
                "video_seconds_covered": round(covered, 6),
                "wav_duration": (round(wav_duration, 6)
                                 if wav_duration else None),
                "duration_source": dur_src,
                "ffmpeg_stderr": box.get("ffmpeg_stderr"),
            }
            n_snip_total += n_snip
            n_sec_total += covered
        except Exception as exc:
            msg = "%s: %s" % (type(exc).__name__, exc)
            failures.append({"video_id": vid, "error": msg[:400]})
            print("  FAILED %s -- %s" % (vid, msg[:200]), flush=True)
            continue

        if i % 10 == 0 or i == 1:
            el = max(time.time() - t0, 1e-9)
            print("  [%d/%d] %s snips=%d %dx%d | %.2f vid/min, %.1fx realtime,"
                  " eta %.1f min"
                  % (i, len(todo), vid, index[vid]["n_snippets"],
                     frame_shape[0], frame_shape[1], i / el * 60,
                     n_sec_total / el, (len(todo) - i) / (i / el) / 60),
                  flush=True)
        if i % 50 == 0:
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=1, sort_keys=True)

    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1, sort_keys=True)
    fail_path = os.path.join(out_dir, "failures.json")
    if failures:
        with open(fail_path, "w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=1)
    elif os.path.isfile(fail_path):
        os.remove(fail_path)

    el = max(time.time() - t0, 1e-9)
    print("i3d [%s] done: %d/%d extracted this run, %d in the manifests with "
          "features, %d snippets (%.1f h of video) this run, %d failures, "
          "%.1f min wall, %.1fx realtime"
          % (args.corpus, len(todo) - len(failures), len(todo), len(index),
             n_snip_total, n_sec_total / 3600.0, len(failures), el / 60,
             n_sec_total / el), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
