"""Reproduction study, Phase 2 task 3: VGGish audio features on the 1 fps grid.

MACIL-SD (MM'22) and the MultiHateLoc re-implementation (WWW'26) both take a
128-d VGGish embedding per second of audio, so the embeddings are extracted
once here and shared, exactly as the CLIP features are shared by VadCLIP and
DSANet.

Which VGGish. `torchvggish` (pip, version 0.2), which is the PyTorch port of
Google's released AudioSet VGGish: the released TensorFlow checkpoint's
weights, loaded through the package's own torch.hub URL, and the released
log-mel front end (16 kHz, 64 mel bands 125-7500 Hz, 25 ms window / 10 ms hop,
96-frame patches). No substitute embedding is used anywhere in this script.
The AudioSet post-processing head -- the PCA rotation plus 8-bit quantisation
that the *published feature files* carry -- is deliberately switched off
(`postprocess=False`), so what is stored is the raw 128-d activation of the
embedding layer. Quantised features would throw away precision that the
downstream MIL heads are trained on in float.

Frame grid. One embedding per frame of the frame-level evaluation grid
(docs/duplex/FRAME_EVAL_PROTOCOL.md): row i covers [i, i + 0.96) seconds, and
T equals the number of gold frames for that video, because the duration comes
from the same wav file build_gt_arrays.py reads. The waveform is zero-padded
at the tail so the last second of audio still yields a full patch instead of
being dropped, which is what the stock 0.96 s hop would do.

Output: <out-root>/<corpus>/<video_id>.npy, float32, shape (T, 128), plus
<corpus>/index.json and, on any failure, <corpus>/failures.json.

  python scripts/duplex/extract_vggish_features.py --corpus hatemm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

from frame_eval_common import frame_times  # noqa: E402
from extract_clip_features import (CORPORA, find_duration, find_video_path,  # noqa: E402
                                   load_chunk_durations, read_ids)

OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "features",
                        "vggish_1s")
FPS = 1.0
HOP_SECONDS = 1.0


def read_wav(path):
    """(mono float64 in [-1, 1], sample_rate) from a PCM wav file."""
    import soundfile as sf
    data, rate = sf.read(path, dtype="float64", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, rate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--batch", type=int, default=256,
                    help="log-mel patches per VGGish forward pass")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)

    ids = read_ids(spec)
    index_path = os.path.join(out_dir, "index.json")
    index = json.load(open(index_path, encoding="utf-8")) \
        if os.path.isfile(index_path) else {}
    # A frozen split can deliberately exclude an unusable sample after an
    # earlier extraction run.  Do not leave that sample discoverable through
    # stale bookkeeping even when every active feature file already exists.
    wanted = set(ids)
    index = {vid: meta for vid, meta in index.items() if vid in wanted}
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1, sort_keys=True)
    todo = [v for v in ids
            if not os.path.isfile(os.path.join(out_dir, v + ".npy"))]
    print("vggish [%s]: %d videos in the manifests, %d already extracted, "
          "%d to run" % (args.corpus, len(ids), len(ids) - len(todo),
                         len(todo)), flush=True)
    if args.limit is not None:
        todo = todo[:args.limit]
        print("  --limit: %d videos" % len(todo), flush=True)
    if not todo:
        return 0

    import torch
    import torchvggish
    from torchvggish import vggish_input, vggish_params

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; this stage is CUDA-only")

    # One patch per second instead of the stock 0.96 s hop, so row i of the
    # output is second i of the evaluation grid.
    vggish_params.EXAMPLE_HOP_SECONDS = HOP_SECONDS
    model = torchvggish.vggish(postprocess=False)
    model.eval().to("cuda")
    print("vggish weights: %s (postprocess=False, hop=%.2f s, window=%.2f s)"
          % (torchvggish.torchvggish.VGGISH_WEIGHTS, HOP_SECONDS,
             vggish_params.EXAMPLE_WINDOW_SECONDS), flush=True)

    chunk_durations = load_chunk_durations(spec)
    failures = []
    t0 = time.time()
    n_rows = 0
    for i, vid in enumerate(todo, 1):
        try:
            wav_path = None
            for wav_dir in spec["wav_dirs"]:
                cand = os.path.join(wav_dir, vid + ".wav")
                if os.path.isfile(cand):
                    wav_path = cand
                    break
            # Same duration resolver as the CLIP features and the gold
            # arrays: the chunk manifest first, the wav header second.
            video_path = find_video_path(spec["video_dir"], vid)
            duration, dur_src = find_duration(
                vid, spec, chunk_durations, video_path)
            if duration is None or duration <= 0:
                raise ValueError("no positive duration for %s" % vid)
            n_target = len(frame_times(duration, FPS))

            if wav_path is None:
                # Released videos without an audio stream are valid examples;
                # represent the absent modality as deterministic silence.
                rate = 16000
                data = np.zeros(int(np.ceil(duration * rate)), dtype=np.float64)
            else:
                data, rate = read_wav(wav_path)
            # Pad the tail so the final grid second still forms a full patch.
            need = int(np.ceil(((n_target - 1) * HOP_SECONDS
                                + vggish_params.EXAMPLE_WINDOW_SECONDS
                                + 0.05) * rate))
            patches = None
            for extra in (0, int(0.5 * rate), int(2.0 * rate)):
                padded = data
                if len(padded) < need + extra:
                    padded = np.concatenate(
                        [padded, np.zeros(need + extra - len(padded))])
                patches = vggish_input.waveform_to_examples(padded, rate)
                if len(patches) >= n_target:
                    break
            if patches is None or len(patches) < n_target:
                raise ValueError("VGGish produced %d patches for %d grid "
                                 "seconds" % (0 if patches is None
                                              else len(patches), n_target))
            patches = patches[:n_target].detach()

            feats = np.empty((n_target, vggish_params.EMBEDDING_SIZE),
                             dtype=np.float32)
            with torch.no_grad():
                for s in range(0, n_target, args.batch):
                    x = patches[s:s + args.batch].to("cuda")
                    emb = model(x)
                    feats[s:s + len(x)] = emb.float().cpu().numpy()

            # np.save appends .npy unless the name already ends in it.
            tmp = os.path.join(out_dir, vid + ".tmp.npy")
            np.save(tmp, feats)
            os.replace(tmp, os.path.join(out_dir, vid + ".npy"))
            index[vid] = {
                "n_frames": int(n_target),
                "wav_duration": round(duration, 6),
                "duration_source": dur_src,
                "sample_rate": int(rate),
                "wav": (os.path.relpath(wav_path, PROJECT_ROOT)
                        if wav_path and wav_path.startswith(PROJECT_ROOT)
                        else wav_path),
                "dim": int(feats.shape[1]),
            }
            n_rows += n_target
        except Exception as exc:
            msg = "%s: %s" % (type(exc).__name__, exc)
            failures.append({"video_id": vid, "error": msg[:400]})
            print("  FAILED %s -- %s" % (vid, msg[:200]), flush=True)
            continue

        if i % 100 == 0 or i == 1:
            el = time.time() - t0
            print("  [%d/%d] %s T=%d | %.1f vid/min, eta %.1f min"
                  % (i, len(todo), vid, index[vid]["n_frames"],
                     i / max(el, 1e-9) * 60,
                     (len(todo) - i) / max(i / max(el, 1e-9), 1e-9) / 60),
                  flush=True)
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=1, sort_keys=True)

    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1, sort_keys=True)
    if failures:
        fail_path = os.path.join(out_dir, "failures.json")
        old = json.load(open(fail_path, encoding="utf-8")) \
            if os.path.isfile(fail_path) else []
        with open(fail_path, "w", encoding="utf-8") as handle:
            json.dump(old + failures, handle, indent=1)

    print("vggish [%s] done: %d/%d extracted, %d rows, %d failures, %.1fs"
          % (args.corpus, len(todo) - len(failures), len(todo), n_rows,
             len(failures), time.time() - t0), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
