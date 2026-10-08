#!/usr/bin/env python3
"""CLARA step 2 OCR (`get_ocr.py`), run in the PaddleOCR venv (.cache/envs/paddleocr: paddlepaddle-gpu 3.2.0,
paddleocr 3.4.0 -> PP-OCRv5_server_det / PP-OCRv5_server_rec, the pipeline defaults).

    python clara_ocr.py --dataset DS [--device cpu|gpu:0]

Same OCR object as the authors (`PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
use_textline_orientation=False)`), same confidence filter (rec_scores >= 0.5 kept). It runs on the frames the
model consumes: `frame_40` of train / val clips and `frame_w` of test-window clips (the authors OCR every budget
folder; only the budget-40 words enter the embeddings). Output per clip: `clip_XXX/ocr_text/ocr_clip.json` with
`clip_words_keep` / `clip_scores_keep` (the fields `extract_video_emb.py` reads).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("CLARA_DATA", REPO / "data"))  # env override: smoke tests only
RAW = DATA / "clara_raw"
CONF_TH = 0.5


def frame_key(p: Path) -> int:
    m = re.findall(r"\d+", p.stem)
    return int(m[0]) if m else 0


def clip_dirs(ds: str):
    for vdir in sorted(p for p in (RAW / ds).iterdir() if p.is_dir()):
        for fdir in sorted(vdir.glob("clip_*/frames/frame_40")) + sorted(vdir.glob("win_*/clip_*/frames/frame_w")):
            yield fdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--check", action="store_true", help="exit 0 if every clip frame folder has its OCR file")
    ap.add_argument("--reverse", action="store_true", help="walk the clip folders from the end (a GPU pass can "
                    "finish what a CPU pass started from the front; both skip folders already written)")
    args = ap.parse_args()
    if args.check:
        missing = [d for d in clip_dirs(args.dataset)
                   if not (d.parent.parent / "ocr_text" / "ocr_clip.json").is_file()]
        print(f"OCR check {args.dataset}: {len(missing)} clip folders without OCR")
        sys.exit(1 if missing else 0)
    log_dir = Path(os.environ.get("DETWIN_RUNS", REPO / "runs" / "20261008_baselines")) / "clara" / args.dataset / "ocr"
    log_dir.mkdir(parents=True, exist_ok=True)
    logf = log_dir / f"run_shard{args.shard}.log"
    fresh = not logf.is_file()
    fh = logf.open("a")
    if fresh:
        fh.write(f"host {socket.gethostname()}\n")

    def log(msg):
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        fh.write(line + "\n")
        fh.flush()
        print(line, flush=True)

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                    device=args.device)
    dirs = [d for i, d in enumerate(clip_dirs(args.dataset)) if i % args.nshards == args.shard]
    if args.reverse:
        dirs = dirs[::-1]
    log(f"OCR {args.dataset} shard {args.shard}/{args.nshards}: {len(dirs)} clip frame dirs, device {args.device}")
    t0, n_img, n_done = time.time(), 0, 0
    for k, fdir in enumerate(dirs):
        out = fdir.parent.parent / "ocr_text" / "ocr_clip.json"
        if out.is_file() and out.stat().st_size > 0:
            continue
        if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
            log(f"stop after {args.max_minutes} min (budget)")
            break
        frames = sorted(fdir.glob("frame_*.jpg"), key=frame_key)
        words_keep, scores_keep, words_all, scores_all, per_frame = [], [], [], [], []
        if frames:
            results = ocr.predict(input=[str(p) for p in frames])
            if len(results) != len(frames):
                raise RuntimeError(f"predict returned {len(results)} results for {len(frames)} images in {fdir}")
            for p, res in zip(frames, results):
                d = res.json
                d = d.get("res", d)
                t = list(d.get("rec_texts", []) or [])
                s = [float(x) for x in (d.get("rec_scores", []) or [])]
                n = min(len(t), len(s))
                t, s = t[:n], s[:n]
                kt = [str(a) for a, b in zip(t, s) if b >= CONF_TH]
                ks = [b for b in s if b >= CONF_TH]
                words_all += t
                scores_all += s
                words_keep += kt
                scores_keep += ks
                per_frame.append({"frame_file": p.name, "words_keep": kt})
            n_img += len(frames)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"budget_dir": fdir.name, "conf_th": CONF_TH, "num_frames": len(frames),
                                   "device": args.device,
                                   "clip_words_keep": words_keep, "clip_scores_keep": scores_keep,
                                   "clip_words_all": words_all, "clip_scores_all": scores_all,
                                   "frames": per_frame}, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 500 == 0:
            el = time.time() - t0
            log(f"  {k + 1}/{len(dirs)} dirs, {n_img} images, {n_img / max(el, 1e-6):.2f} img/s")
    log(f"DONE ocr shard {args.shard}: {n_done} dirs written, {n_img} images, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
