#!/usr/bin/env python
"""Stage 1 of the text-region pixel-budget probe: locate on-screen text boxes.

Runs the A2 census detector, unchanged, over the 16 native-resolution grid
frames of every MHClip-EN test_clean video. The detector module is imported
from `idea-stage/pilots/a2_ocr_census/ocr_census.py` rather than copied, so the
engine, the default CRAFT thresholds and the box normalisation are provably the
same ones the census used.

Detection only. Recognised strings are never requested and never stored: this
stage emits box geometry and nothing else.

Output: results/text_region_budget/boxes.jsonl, one record per video.
"""

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

CENSUS = os.path.join(ROOT, "idea-stage", "pilots", "a2_ocr_census", "ocr_census.py")


def load_census():
    spec = importlib.util.spec_from_file_location("a2_ocr_census", CENSUS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHClip_EN")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "text_region_budget", "boxes.jsonl"))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    census = load_census()
    from data_utils import DATASET_ROOTS, load_clean_split_ids

    ids = load_clean_split_ids(args.dataset, args.split)
    seen, uniq = set(), []
    for v in ids:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    frames_root = os.path.join(DATASET_ROOTS[args.dataset], "frames_16")

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [v for v in uniq if v not in done]
    print(f"[detect] {len(uniq)} videos, {len(done)} done, {len(todo)} to go, "
          f"device={args.device}", flush=True)
    if not todo:
        return

    import cv2

    import easyocr
    reader = easyocr.Reader(["en"], gpu=(args.device == "cuda"), verbose=False)

    t0 = time.time()
    with open(args.out, "a") as fout:
        for i, vid in enumerate(todo, 1):
            paths = sorted(glob.glob(os.path.join(frames_root, vid, "*.jpg")))
            rec = {"video_id": vid, "n_frames": len(paths), "frames": []}
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    rec["frames"].append({"f": os.path.basename(p), "err": "unreadable"})
                    continue
                H, W = img.shape[0], img.shape[1]
                horiz, free = reader.detect(img)
                boxes = census.boxes_from_detect(horiz[0] if horiz else [],
                                                 free[0] if free else [])
                rec["frames"].append({
                    "f": os.path.basename(p),
                    "H": H, "W": W,
                    "n_boxes": len(boxes),
                    "union_frac": round(census.union_area_frac(boxes, H, W), 6),
                    "boxes": [[round(c, 1) for c in b] for b in boxes],
                })
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[detect] {i}/{len(todo)} el={el/60:.1f}m "
                      f"eta={(el/i)*(len(todo)-i)/60:.1f}m", flush=True)
    print(f"[detect] complete in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
