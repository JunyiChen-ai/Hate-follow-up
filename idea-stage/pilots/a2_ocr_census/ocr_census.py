#!/usr/bin/env python
"""
A2 pre-gate: on-screen-text census for MHClip-EN vs ImpliHateVid (CPU-only).

Detection-only OCR (EasyOCR/CRAFT detector). Recognised strings are never
requested and never stored -- only geometry (boxes) is used.

Per frame we record:
  - number of detected text boxes
  - union text-box area as a fraction of frame area
  - median box height in native pixels
  - the same box heights after applying the judge's smart_resize
    (Qwen-VL, factor=28, max_pixels=100352); a box whose scaled height
    is < 12 px is presumptively illegible to the judge.

Usage:
  python ocr_census.py --stage detect   # writes per-frame shards (jsonl)
  python ocr_census.py --stage agg      # writes results.json
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("HVD_DATA_ROOT", "/home/jehc223/data")

MH_FRAMES = os.path.join(DATA_ROOT, "Multihateclip/English/frames_16")
MH_SPLIT = os.path.join(DATA_ROOT, "Multihateclip/English/splits/train_clean.csv")
MH_ANN = os.path.join(DATA_ROOT, "Multihateclip/English/annotation(new).json")

IHV_FRAMES = os.path.join(DATA_ROOT, "ImpliHateVid/frames_16")
IHV_SPLIT = os.path.join(DATA_ROOT, "ImpliHateVid/splits/train_clean.csv")
IHV_ANN = os.path.join(DATA_ROOT, "ImpliHateVid/annotation(new).json")

IHV_SAMPLE_N = 300
IHV_SAMPLE_SEED = 0

# judge-side resize spec
FACTOR = 28
MIN_PIXELS = 56 * 56
MAX_PIXELS = 100352          # == 128 * 28 * 28
LEGIBLE_MIN_H = 12.0         # scaled box height below this -> presumed illegible

SUBSTANTIVE_AREA_FRAC = 0.01  # >=1% of frame area
SUBSTANTIVE_MIN_FRAMES = 2    # >=2 such frames -> video counts
PROCEED_BAR = 0.25            # >=25% of MHClip-EN train_clean videos


# --------------------------------------------------------------------------- #
# judge resize
# --------------------------------------------------------------------------- #
def smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Qwen-VL smart_resize. Returns (h_bar, w_bar)."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def judge_scale(h, w):
    hb, wb = smart_resize(h, w)
    return hb / float(h), (hb, wb)


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #
def read_ids(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_labels(path):
    recs = json.load(open(path))
    return {r["Video_ID"]: r.get("Label", "") for r in recs}


def build_tasks():
    mh_ids = read_ids(MH_SPLIT)
    mh_lab = load_labels(MH_ANN)
    mh = [("MHClip_EN", v, os.path.join(MH_FRAMES, v), mh_lab.get(v, "NA"))
          for v in mh_ids if os.path.isdir(os.path.join(MH_FRAMES, v))]

    ihv_ids = read_ids(IHV_SPLIT)
    ihv_ids = [v for v in ihv_ids if os.path.isdir(os.path.join(IHV_FRAMES, v))]
    rng = random.Random(IHV_SAMPLE_SEED)
    ihv_sel = sorted(rng.sample(ihv_ids, min(IHV_SAMPLE_N, len(ihv_ids))))
    ihv_lab = load_labels(IHV_ANN)
    ihv = [("ImpliHateVid", v, os.path.join(IHV_FRAMES, v), ihv_lab.get(v, "NA"))
           for v in ihv_sel]

    return mh + ihv, len(mh_ids), len(ihv_ids)


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
_READER = None


def get_reader(threads, use_gpu=False):
    global _READER
    if _READER is None:
        import torch
        torch.set_num_threads(threads)
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=bool(use_gpu), verbose=False)
    return _READER


def boxes_from_detect(horiz, free):
    """Normalise EasyOCR detect() output to (x0, y0, x1, y1) axis-aligned boxes."""
    out = []
    for b in horiz or []:
        x0, x1, y0, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    for poly in free or []:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def union_area_frac(boxes, H, W, grid=256):
    """Union area of boxes as fraction of frame, via a coarse raster mask."""
    if not boxes:
        return 0.0
    import numpy as np
    gh = grid
    gw = max(1, int(round(grid * W / float(H))))
    mask = np.zeros((gh, gw), dtype=bool)
    sy, sx = gh / float(H), gw / float(W)
    for (x0, y0, x1, y1) in boxes:
        r0 = max(0, int(math.floor(y0 * sy)))
        r1 = min(gh, int(math.ceil(y1 * sy)))
        c0 = max(0, int(math.floor(x0 * sx)))
        c1 = min(gw, int(math.ceil(x1 * sx)))
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = True
    return float(mask.mean())


def process_video(args):
    corpus, vid, fdir, label, threads, use_gpu = args
    reader = get_reader(threads, use_gpu)
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    frames = sorted(glob.glob(os.path.join(fdir, "frame_*.jpg")))
    rec = {"corpus": corpus, "video_id": vid, "label": label,
           "n_frames": len(frames), "frames": [], "error": None}
    # decode JPEGs on worker threads so GPU detection is not IO-bound
    with ThreadPoolExecutor(max_workers=4) as ex:
        imgs = list(ex.map(cv2.imread, frames))
    for fp, img in zip(frames, imgs):
        try:
            if img is None:
                rec["frames"].append({"f": os.path.basename(fp), "err": "unreadable"})
                continue
            H, W = img.shape[0], img.shape[1]
            horiz, free = reader.detect(img)
            boxes = boxes_from_detect(horiz[0] if horiz else [],
                                      free[0] if free else [])
            scale, (hb, wb) = judge_scale(H, W)
            heights = [max(0.0, b[3] - b[1]) for b in boxes]
            areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in boxes]
            frame_area = float(H * W)
            sum_area = sum(areas)
            illeg_area = sum(a for a, h in zip(areas, heights)
                             if h * scale < LEGIBLE_MIN_H)
            rec["frames"].append({
                "f": os.path.basename(fp),
                "H": H, "W": W,
                "jh": hb, "jw": wb, "scale": round(scale, 6),
                "n_boxes": len(boxes),
                "union_frac": round(union_area_frac(boxes, H, W), 6),
                "sum_area_frac": round(sum_area / frame_area, 6),
                "illeg_area_frac_of_text": round(illeg_area / sum_area, 6) if sum_area > 0 else None,
                "med_h_native": round(float(np.median(heights)), 2) if heights else None,
                "med_h_judge": round(float(np.median(heights)) * scale, 3) if heights else None,
                "heights_native": [round(h, 1) for h in heights],
                "areas_frac": [round(a / frame_area, 7) for a in areas],
            })
        except Exception as e:  # noqa: BLE001
            rec["frames"].append({"f": os.path.basename(fp), "err": repr(e)[:200]})
    return rec


# --------------------------------------------------------------------------- #
def run_detect(nproc, threads, use_gpu=False):
    tasks, n_mh_total, n_ihv_total = build_tasks()
    print(f"[detect] {len(tasks)} videos "
          f"(MH split={n_mh_total}, IHV pool={n_ihv_total}), "
          f"nproc={nproc} threads/proc={threads} gpu={use_gpu}", flush=True)
    out_path = os.path.join(OUT_DIR, "per_frame.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["video_id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [(c, v, d, l, threads, use_gpu) for (c, v, d, l) in tasks if v not in done]
    print(f"[detect] resuming: {len(done)} done, {len(todo)} to go", flush=True)

    t0 = time.time()
    if nproc <= 1:
        with open(out_path, "a") as fout:
            for i, t in enumerate(todo, 1):
                rec = process_video(t)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                if i % 20 == 0:
                    el = time.time() - t0
                    print(f"[detect] {i}/{len(todo)} el={el/60:.1f}m "
                          f"eta={(el/i)*(len(todo)-i)/60:.1f}m", flush=True)
        print(f"[detect] complete in {(time.time()-t0)/60:.1f} min", flush=True)
        return
    with open(out_path, "a") as fout, Pool(nproc) as pool:
        for i, rec in enumerate(pool.imap_unordered(process_video, todo, chunksize=1), 1):
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 20 == 0:
                el = time.time() - t0
                print(f"[detect] {i}/{len(todo)} el={el/60:.1f}m "
                      f"eta={(el/i)*(len(todo)-i)/60:.1f}m", flush=True)
    print(f"[detect] complete in {(time.time()-t0)/60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
def pct(x, n):
    return round(100.0 * x / n, 2) if n else None


def summarise(recs, corpus):
    import numpy as np
    rs = [r for r in recs if r["corpus"] == corpus]
    n_vid = len(rs)

    vid_rows = []
    all_heights_native, all_heights_judge, all_weights = [], [], []
    frame_union, frame_nboxes = [], []
    res_counter = {}
    n_frames_total = 0
    n_frames_err = 0

    for r in rs:
        n_sub = 0
        n_any = 0
        for fr in r["frames"]:
            if "err" in fr:
                n_frames_err += 1
                continue
            n_frames_total += 1
            key = f"{fr['W']}x{fr['H']}"
            res_counter[key] = res_counter.get(key, 0) + 1
            frame_union.append(fr["union_frac"])
            frame_nboxes.append(fr["n_boxes"])
            if fr["n_boxes"] > 0:
                n_any += 1
            if fr["union_frac"] >= SUBSTANTIVE_AREA_FRAC:
                n_sub += 1
            sc = fr["scale"]
            for h, a in zip(fr["heights_native"], fr["areas_frac"]):
                all_heights_native.append(h)
                all_heights_judge.append(h * sc)
                all_weights.append(a)
        vid_rows.append({
            "video_id": r["video_id"],
            "label": r["label"],
            "n_sub_frames": n_sub,
            "n_frames_any_text": n_any,
            "mean_union_frac": round(float(np.mean([f["union_frac"] for f in r["frames"]
                                                    if "err" not in f])), 5)
            if any("err" not in f for f in r["frames"]) else None,
        })

    hn = np.array(all_heights_native) if all_heights_native else np.array([])
    hj = np.array(all_heights_judge) if all_heights_judge else np.array([])
    w = np.array(all_weights) if all_weights else np.array([])

    qual = [v for v in vid_rows if v["n_sub_frames"] >= SUBSTANTIVE_MIN_FRAMES]

    def half(v):
        return "hateful" if v["label"] in ("Hateful", "Offensive") else "normal"

    strata = {}
    for hlf in ("hateful", "normal"):
        sub = [v for v in vid_rows if half(v) == hlf]
        if not sub:
            continue
        strata[hlf] = {
            "n_videos": len(sub),
            "n_qualifying": sum(1 for v in sub if v["n_sub_frames"] >= SUBSTANTIVE_MIN_FRAMES),
            "pct_qualifying": pct(sum(1 for v in sub if v["n_sub_frames"] >= SUBSTANTIVE_MIN_FRAMES), len(sub)),
            "mean_sub_frames_per_video": round(float(np.mean([v["n_sub_frames"] for v in sub])), 3),
            "median_sub_frames_per_video": float(np.median([v["n_sub_frames"] for v in sub])),
            "mean_frames_any_text": round(float(np.mean([v["n_frames_any_text"] for v in sub])), 3),
            "mean_union_frac": round(float(np.mean([v["mean_union_frac"] for v in sub
                                                    if v["mean_union_frac"] is not None])), 5),
        }

    # text-mass-weighted illegibility at judge scale
    if w.size:
        illeg_mask = hj < LEGIBLE_MIN_H
        illeg_mass = float(w[illeg_mask].sum() / w.sum())
        illeg_boxes = float(illeg_mask.mean())
    else:
        illeg_mass, illeg_boxes = None, None

    # restricted to substantive frames only
    sub_h, sub_w = [], []
    for r in rs:
        for fr in r["frames"]:
            if "err" in fr or fr["union_frac"] < SUBSTANTIVE_AREA_FRAC:
                continue
            for h, a in zip(fr["heights_native"], fr["areas_frac"]):
                sub_h.append(h * fr["scale"])
                sub_w.append(a)
    if sub_w:
        sh = np.array(sub_h)
        sw = np.array(sub_w)
        illeg_mass_sub = float(sw[sh < LEGIBLE_MIN_H].sum() / sw.sum())
        illeg_boxes_sub = float((sh < LEGIBLE_MIN_H).mean())
    else:
        illeg_mass_sub, illeg_boxes_sub = None, None

    def q(a, p):
        return round(float(np.percentile(a, p)), 3) if a.size else None

    # legibility histogram over judge-scaled box heights (by count and by text mass)
    bands = [(0, 6), (6, 9), (9, 12), (12, 16), (16, 24), (24, 1e9)]
    hist = {}
    if hj.size:
        for lo, hi in bands:
            msk = (hj >= lo) & (hj < hi)
            name = f"{lo}-{hi}px" if hi < 1e9 else f">={lo}px"
            hist[name] = {"frac_boxes": round(float(msk.mean()), 4),
                          "frac_text_mass": round(float(w[msk].sum() / w.sum()), 4)}

    # per-frame view: is the *typical* text in this frame illegible?
    sub_frame_med_illeg, any_frame_med_illeg = [], []
    for r in rs:
        for fr in r["frames"]:
            if "err" in fr or fr["med_h_judge"] is None:
                continue
            any_frame_med_illeg.append(fr["med_h_judge"] < LEGIBLE_MIN_H)
            if fr["union_frac"] >= SUBSTANTIVE_AREA_FRAC:
                sub_frame_med_illeg.append(fr["med_h_judge"] < LEGIBLE_MIN_H)

    # per-label-half illegibility (text-mass weighted)
    half_illeg = {}
    for hlf in ("hateful", "normal"):
        hh, ww = [], []
        for r in rs:
            lab = "hateful" if r["label"] in ("Hateful", "Offensive") else "normal"
            if lab != hlf:
                continue
            for fr in r["frames"]:
                if "err" in fr:
                    continue
                for h, a in zip(fr["heights_native"], fr["areas_frac"]):
                    hh.append(h * fr["scale"])
                    ww.append(a)
        if ww:
            hh_a, ww_a = np.array(hh), np.array(ww)
            half_illeg[hlf] = {
                "frac_text_mass_illegible": round(float(ww_a[hh_a < LEGIBLE_MIN_H].sum() / ww_a.sum()), 4),
                "frac_boxes_illegible": round(float((hh_a < LEGIBLE_MIN_H).mean()), 4),
                "median_box_height_at_judge_px": round(float(np.median(hh_a)), 3),
            }
    for hlf, d in half_illeg.items():
        if hlf in strata:
            strata[hlf].update(d)

    return {
        "n_videos": n_vid,
        "n_frames_scored": n_frames_total,
        "n_frames_error": n_frames_err,
        "native_resolutions_top": sorted(res_counter.items(), key=lambda kv: -kv[1])[:10],
        "frac_frames_any_text": round(float(np.mean([b > 0 for b in frame_nboxes])), 4) if frame_nboxes else None,
        "frac_frames_substantive": round(float(np.mean([u >= SUBSTANTIVE_AREA_FRAC for u in frame_union])), 4) if frame_union else None,
        "mean_boxes_per_frame": round(float(np.mean(frame_nboxes)), 3) if frame_nboxes else None,
        "mean_union_frac_per_frame": round(float(np.mean(frame_union)), 5) if frame_union else None,
        "median_union_frac_per_frame": round(float(np.median(frame_union)), 5) if frame_union else None,
        "videos_with_ge2_substantive_frames": len(qual),
        "pct_videos_with_ge2_substantive_frames": pct(len(qual), n_vid),
        "box_height_native_px": {"p10": q(hn, 10), "p25": q(hn, 25), "median": q(hn, 50),
                                 "p75": q(hn, 75), "p90": q(hn, 90), "n": int(hn.size)},
        "box_height_at_judge_px": {"p10": q(hj, 10), "p25": q(hj, 25), "median": q(hj, 50),
                                   "p75": q(hj, 75), "p90": q(hj, 90), "n": int(hj.size)},
        "illegible_at_judge": {
            "threshold_px": LEGIBLE_MIN_H,
            "frac_of_text_mass_all_frames": round(illeg_mass, 4) if illeg_mass is not None else None,
            "frac_of_boxes_all_frames": round(illeg_boxes, 4) if illeg_boxes is not None else None,
            "frac_of_text_mass_substantive_frames": round(illeg_mass_sub, 4) if illeg_mass_sub is not None else None,
            "frac_of_boxes_substantive_frames": round(illeg_boxes_sub, 4) if illeg_boxes_sub is not None else None,
            "frac_frames_whose_median_text_illegible": round(float(np.mean(any_frame_med_illeg)), 4) if any_frame_med_illeg else None,
            "frac_substantive_frames_whose_median_text_illegible": round(float(np.mean(sub_frame_med_illeg)), 4) if sub_frame_med_illeg else None,
        },
        "judge_scaled_height_bands": hist,
        "by_label_half": strata,
    }


def run_agg():
    import numpy as np  # noqa: F401
    path = os.path.join(OUT_DIR, "per_frame.jsonl")
    raw = [json.loads(ln) for ln in open(path) if ln.strip()]
    # a resumed run can re-emit a video; keep the last record per video id
    by_id = {}
    for r in raw:
        by_id[r["video_id"]] = r
    recs = list(by_id.values())
    print(f"[agg] {len(raw)} lines -> {len(recs)} unique videos", flush=True)
    mh = summarise(recs, "MHClip_EN")
    ihv = summarise(recs, "ImpliHateVid")

    verdict = "PROCEED" if (mh["pct_videos_with_ge2_substantive_frames"] or 0) >= PROCEED_BAR * 100 else "STOP"

    out = {
        "pilot": "A2 CPU pre-gate: on-screen-text census",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": {
            "name": "EasyOCR 1.7.2 CRAFT detector (detection only, no recognition)",
            "device": "mixed: first ~163 videos scored on CPU, remainder on CUDA "
                      "(RTX 5090). Same CRAFT weights and same default thresholds "
                      "in both cases; the detector is deterministic, so device does "
                      "not change the geometry that is measured.",
            "thresholds": "engine defaults (text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1.0)",
            "note": "recognised strings were never requested or stored",
        },
        "definitions": {
            "substantive_frame": f"union OCR-box area >= {SUBSTANTIVE_AREA_FRAC:.0%} of frame area",
            "qualifying_video": f">= {SUBSTANTIVE_MIN_FRAMES} substantive frames out of 16",
            "judge_resize": f"Qwen-VL smart_resize(factor={FACTOR}, max_pixels={MAX_PIXELS})",
            "illegible_box": f"box height after judge resize < {LEGIBLE_MIN_H} px",
            "text_mass": "sum of OCR box areas (frame-area-normalised)",
        },
        "proceed_bar": {
            "rule": f">= {PROCEED_BAR:.0%} of MHClip-EN train_clean videos have >= {SUBSTANTIVE_MIN_FRAMES} substantive frames",
            "observed_pct": mh["pct_videos_with_ge2_substantive_frames"],
            "verdict": verdict,
        },
        "MHClip_EN_train_clean": mh,
        "ImpliHateVid_train_clean_seed0_n300": ihv,
        "contrast_EN_vs_IHV": {
            "pct_qualifying_videos": [mh["pct_videos_with_ge2_substantive_frames"],
                                      ihv["pct_videos_with_ge2_substantive_frames"]],
            "frac_frames_substantive": [mh["frac_frames_substantive"], ihv["frac_frames_substantive"]],
            "mean_boxes_per_frame": [mh["mean_boxes_per_frame"], ihv["mean_boxes_per_frame"]],
            "mean_union_frac_per_frame": [mh["mean_union_frac_per_frame"], ihv["mean_union_frac_per_frame"]],
        },
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["proceed_bar"], indent=2))
    print(json.dumps(out["contrast_EN_vs_IHV"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["detect", "agg"], required=True)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.stage == "detect":
        run_detect(a.nproc, a.threads, use_gpu=(a.device == "cuda"))
    else:
        run_agg()
