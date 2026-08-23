"""Reproduction study, Phase 2 task 5: ImageNet ViT-B/16 visual features at 1 fps.

The MultiHateLoc reimplementation (WWW'26, no code released) specifies an
ImageNet-pretrained ViT-B/16 as its visual branch, not CLIP. Feeding it the
CLIP features already extracted by extract_clip_features.py would quietly
change the baseline into something the paper does not describe, so the
paper's own visual encoder gets its own pass here.

Everything except the encoder is identical to extract_clip_features.py, and
deliberately so: the same 1 fps grid tied to the audio duration, the same
duration resolver (chunk manifest first, wav header second), the same decord
decode with a system-ffmpeg fallback -- shared code, imported from that file,
not a second copy -- the same index.json / failures.json bookkeeping. Row i of
the feature matrix is frame i of the frozen gold array by construction, which
is what lets MultiHateLoc's per-frame scores be compared against VadCLIP's
and DSANet's without a per-video crop.

Encoder: `google/vit-base-patch16-224` in fp16 -- the original ViT-B/16,
pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. Preprocessing is
that checkpoint's own released transform, run through `ViTImageProcessor`
rather than reimplemented: bilinear resize to 224x224, rescale, normalise
with mean = std = 0.5. Note this is *not* the CLIP transform (no
shortest-edge resize, no centre crop, different statistics); reusing the CLIP
processor here would put the images off the distribution the ImageNet weights
were trained on.

The per-frame vector is the final-layer CLS token after the encoder's output
layernorm, 768-d. The randomly-initialised pooler head is not built.

Output: <out-root>/<corpus>/<video_id>.npy, float32, shape (T, 768). Plus
<corpus>/index.json with the per-video grid bookkeeping, and failures.json.

  python scripts/duplex/extract_vit_features.py --corpus hatemm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

from extract_clip_features import (  # noqa: E402
    CORPORA, encode_with_fallback, find_duration, find_video_path, load_chunk_durations,
    read_ids)
from frame_eval_common import frame_times  # noqa: E402

OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "features",
                        "vit_b16_imagenet_1fps")

MODEL_ID = "google/vit-base-patch16-224"
FPS = 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--batch", type=int, default=64,
                    help="frames per ViT forward pass")
    ap.add_argument("--tmp-dir", default=None,
                    help="scratch directory for the ffmpeg fallback frames")
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
    print("vit [%s]: %d videos in the manifests, %d already extracted, "
          "%d to run" % (args.corpus, len(ids), len(ids) - len(todo),
                         len(todo)), flush=True)
    if args.limit is not None:
        todo = todo[:args.limit]
        print("  --limit: %d videos" % len(todo), flush=True)
    if not todo:
        return 0

    import torch
    from transformers import ViTImageProcessor, ViTModel

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; this stage is CUDA-only")

    proc = ViTImageProcessor.from_pretrained(MODEL_ID)
    # add_pooling_layer=False: the pooler head has no pretrained weights in
    # this checkpoint and we take the CLS token directly, so building it would
    # only put randomly-initialised parameters in the graph.
    model = ViTModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                     add_pooling_layer=False).to("cuda")
    model.eval()
    devices = {p.device.type for p in model.parameters()}
    if devices != {"cuda"}:
        raise SystemExit("ABORT: ViT parameters are on %s, not cuda" % devices)
    dim = int(model.config.hidden_size)

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
            duration, dur_src = find_duration(
                vid, spec, chunk_durations, path)
            if duration is None or duration <= 0:
                raise ValueError("no positive wav duration for %s" % vid)
            grid = frame_times(duration, FPS)
            n_target = len(grid)

            def encode(batches, n_target=n_target):
                feats = np.empty((n_target, dim), dtype=np.float32)
                written = 0
                with torch.no_grad():
                    for imgs in batches():
                        x = proc(images=imgs,
                                 return_tensors="pt")["pixel_values"]
                        x = x.to("cuda", dtype=torch.float16)
                        # last_hidden_state is post-layernorm in HF's ViT;
                        # token 0 is the CLS token.
                        out = model(pixel_values=x).last_hidden_state[:, 0]
                        feats[written:written + len(imgs)] = \
                            out.float().cpu().numpy()
                        written += len(imgs)
                if written != n_target:
                    raise ValueError("decoded %d frames for a %d-frame grid"
                                     % (written, n_target))
                return feats

            feats, meta = encode_with_fallback(
                encode, path, grid, n_target, args.batch, args.tmp_dir)

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
    print("vit [%s] done: %d/%d extracted this run, %d in the manifests "
          "with features, %d frames this run, %d videos decoded through the "
          "ffmpeg fallback, %d videos with frames clamped past the video "
          "end, %d failures, %.1fs"
          % (args.corpus, len(todo) - len(failures), len(todo), len(index),
             n_frames_total, via_ffmpeg, clamped, len(failures),
             time.time() - t0), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
