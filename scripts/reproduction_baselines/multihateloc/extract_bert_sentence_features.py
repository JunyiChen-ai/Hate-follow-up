"""MultiHateLoc reimplementation: BERT sentence features on the 1 fps grid.

MultiHateLoc (WWW'26, arXiv 2512.10408) describes its text branch as: Whisper
transcribes the audio into sentence fragments carrying timestamps, each
fragment is embedded by BERT into a 768-d vector, and that vector is
"repeat-padded" across the frames the fragment's interval covers. This script
produces exactly that array, one file per video, on the same 1 fps grid the
frozen gold arrays live on (docs/duplex/FRAME_EVAL_PROTOCOL.md), so row i of
the text matrix is frame i of the gold array and of the ViT / VGGish matrices
already extracted by scripts/duplex/.

The paper does not release code (github.com/mmilabuk/multihateloc is a
LICENSE-only repository) and leaves several details unstated. Every choice
this script makes in their place is listed in
scripts/reproduction_baselines/multihateloc/DESIGN.md; the four that live here
are repeated below so the file is readable on its own.

  1. ASR. The paper says "Whisper" without a size. This study does not rerun
     Whisper: it consumes the transcripts already frozen for the corpus
     (whisper-large-v3, the same chunk manifests scripts/duplex's feature
     extractors take their durations from). Reusing them keeps the text branch
     on the identical transcription every other component of this study sees.

  2. BERT variant. The paper says "BERT", 768-d, without a checkpoint.
     bert-base-uncased for hatemm, mhclip_en and hateclipseg (all three are
     English corpora), bert-base-chinese for mhclip_zh. Both are 768-d base
     models, which is what the stated
     dimensionality requires. HateMM contains a minority of non-English
     speech; it still goes through the uncased English model, since the corpus
     is an English corpus and switching checkpoints per utterance would be an
     addition the paper does not describe.

  3. Sentence vector. Last-hidden-state CLS token, no pooler head. The paper
     states only "768-d per sentence". CLS is the conventional reading of a
     sentence-level BERT vector and needs no extra untrained parameters; the
     pooler head is not built, since its tanh projection is not part of what
     the paper describes.

  4. Fragment-to-frame rule. Frame i covers [i, i+1) seconds. A fragment
     covering [start, end) is written to every frame whose second overlaps it.
     Where two fragments overlap the same second, the one with the larger
     overlap wins, ties going to the earlier fragment. Frames no fragment
     covers get a zero vector -- the paper does not say what happens to
     silence, and zero is the only value that adds no information.

Output: <out-root>/<corpus>/<video_id>.npy, float32, shape (T, 768), plus
<corpus>/index.json with per-video coverage bookkeeping.

  python extract_bert_sentence_features.py --corpus hatemm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "duplex"))

from extract_clip_features import CORPORA, read_ids  # noqa: E402

OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "features",
                        "bert_sentence_1fps")
# The frozen grid. T is read from the ViT matrix rather than recomputed from
# the wav duration: the ViT extractor already resolved the duration for every
# video in the manifests, and taking T from its output makes a length mismatch
# between the two branches impossible by construction.
GRID_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "features",
                         "vit_b16_imagenet_1fps")

BERT_ID = {
    "hatemm": "bert-base-uncased",
    "mhclip_en": "bert-base-uncased",
    "mhclip_zh": "bert-base-chinese",
    "hateclipseg": "bert-base-uncased",
}
MAX_TOKENS = 64  # ASR fragments are short; 64 word-pieces covers them.


def load_chunks(spec):
    """video_id -> list of {start, end, text}, from the corpus chunk manifests.

    The manifests are disjoint by construction (one per split), but the merge
    is checked rather than assumed: a repeated id aborts.
    """
    out = {}
    for rel in spec["chunk_manifests"]:
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                vid = rec["video_id"]
                if vid in out:
                    raise ValueError("video %s appears in two chunk manifests"
                                     % vid)
                chunks = []
                for ch in rec.get("chunks") or []:
                    text = (ch.get("text") or "").strip()
                    start, end = ch.get("start"), ch.get("end")
                    if not text or start is None or end is None:
                        continue
                    start, end = float(start), float(end)
                    if not (math.isfinite(start) and math.isfinite(end)):
                        continue
                    if end <= start:
                        # Whisper occasionally emits a zero-width or inverted
                        # fragment; give it the single second it starts in so
                        # the text is not dropped.
                        end = start + 1.0
                    chunks.append({"start": start, "end": end, "text": text})
                out[vid] = chunks
    return out


def assign_frames(chunks, n_frames):
    """frame index -> chunk index, by largest overlap; -1 where uncovered."""
    owner = np.full(n_frames, -1, dtype=np.int64)
    best = np.zeros(n_frames, dtype=np.float64)
    for ci, ch in enumerate(chunks):
        lo = max(0, int(math.floor(ch["start"])))
        hi = min(n_frames, int(math.ceil(ch["end"])))
        for f in range(lo, hi):
            ov = min(ch["end"], f + 1.0) - max(ch["start"], float(f))
            if ov <= 0:
                continue
            if ov > best[f]:          # strict: ties keep the earlier fragment
                best[f] = ov
                owner[f] = ci
    return owner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--grid-root", default=GRID_ROOT)
    ap.add_argument("--batch", type=int, default=64,
                    help="sentence fragments per BERT forward pass")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true",
                    help="smoke-test escape hatch; the real run is CUDA")
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)
    grid_dir = os.path.join(args.grid_root, args.corpus)

    ids = read_ids(spec)
    todo = [v for v in ids
            if not os.path.isfile(os.path.join(out_dir, v + ".npy"))]
    print("bert [%s]: %d videos in the manifests, %d already extracted, "
          "%d to run" % (args.corpus, len(ids), len(ids) - len(todo),
                         len(todo)), flush=True)
    if args.limit is not None:
        todo = todo[:args.limit]
        print("  --limit: %d videos" % len(todo), flush=True)
    if not todo:
        return 0

    chunks_by_video = load_chunks(spec)
    missing_asr = [v for v in todo if v not in chunks_by_video]
    if missing_asr:
        raise SystemExit("ABORT: %d videos have no ASR record, e.g. %s"
                         % (len(missing_asr), missing_asr[:5]))

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA is not available; pass --cpu to smoke test")
    model_id = BERT_ID[args.corpus]
    tok = AutoTokenizer.from_pretrained(model_id)
    # add_pooling_layer=False: the CLS vector is taken from the last hidden
    # state directly, so the pooler's tanh projection is never used.
    model = AutoModel.from_pretrained(model_id, add_pooling_layer=False)
    model = model.to(device).eval()
    dim = int(model.config.hidden_size)
    if dim != 768:
        raise SystemExit("ABORT: %s is %d-d, the paper states 768" % (model_id, dim))

    index_path = os.path.join(out_dir, "index.json")
    index = {}
    if os.path.isfile(index_path):
        index = json.load(open(index_path, encoding="utf-8"))

    failures = []
    t0 = time.time()
    for i, vid in enumerate(todo, 1):
        try:
            grid_path = os.path.join(grid_dir, vid + ".npy")
            if not os.path.isfile(grid_path):
                raise FileNotFoundError(grid_path)
            n_frames = int(np.load(grid_path, mmap_mode="r").shape[0])
            chunks = chunks_by_video[vid]
            feats = np.zeros((n_frames, dim), dtype=np.float32)

            if chunks:
                owner = assign_frames(chunks, n_frames)
                used = sorted({int(c) for c in owner if c >= 0})
                if used:
                    texts = [chunks[c]["text"] for c in used]
                    vecs = np.empty((len(used), dim), dtype=np.float32)
                    with torch.no_grad():
                        for s in range(0, len(texts), args.batch):
                            enc = tok(texts[s:s + args.batch],
                                      padding=True, truncation=True,
                                      max_length=MAX_TOKENS,
                                      return_tensors="pt").to(device)
                            out = model(**enc).last_hidden_state[:, 0]
                            vecs[s:s + len(enc["input_ids"])] = \
                                out.float().cpu().numpy()
                    slot = {c: k for k, c in enumerate(used)}
                    for f in range(n_frames):
                        c = int(owner[f])
                        if c >= 0:
                            feats[f] = vecs[slot[c]]
                covered = int((owner >= 0).sum())
            else:
                covered = 0

            tmp = os.path.join(out_dir, vid + ".tmp.npy")
            np.save(tmp, feats)
            os.replace(tmp, os.path.join(out_dir, vid + ".npy"))
            index[vid] = {
                "n_frames": n_frames,
                "dim": dim,
                "bert": model_id,
                "n_chunks": len(chunks),
                "n_frames_covered": covered,
                "coverage": round(covered / n_frames, 6) if n_frames else 0.0,
            }
        except Exception as exc:
            msg = "%s: %s" % (type(exc).__name__, exc)
            failures.append({"video_id": vid, "error": msg[:400]})
            print("  FAILED %s -- %s" % (vid, msg[:200]), flush=True)
            continue

        if i % 200 == 0 or i == 1:
            el = time.time() - t0
            print("  [%d/%d] %s T=%d coverage=%.2f | %.0f vid/min"
                  % (i, len(todo), vid, index[vid]["n_frames"],
                     index[vid]["coverage"], i / max(el, 1e-9) * 60),
                  flush=True)

    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1, sort_keys=True)
    fail_path = os.path.join(out_dir, "failures.json")
    if failures:
        with open(fail_path, "w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=1)
    elif os.path.isfile(fail_path):
        os.remove(fail_path)

    cov = [v["coverage"] for v in index.values()]
    silent = sum(1 for v in index.values() if v["n_chunks"] == 0)
    print("bert [%s] done: %d/%d extracted this run, %d in the manifests with "
          "features, mean frame coverage %.3f, %d videos with no ASR "
          "fragments at all, %d failures, %.1fs"
          % (args.corpus, len(todo) - len(failures), len(todo), len(index),
             float(np.mean(cov)) if cov else 0.0, silent, len(failures),
             time.time() - t0), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
