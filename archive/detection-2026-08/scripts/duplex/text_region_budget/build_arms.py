#!/usr/bin/env python
"""Stage 2 of the text-region pixel-budget probe: render the three visual arms.

Reads the box geometry from stage 1 and writes, for each of TEXT, RAND and
ANTI, a manifest of 16 image slots per video plus the crop images those
manifests point at. Full-frame slots point at the original frames_16 jpg, byte
for byte the file the BASELINE run read, so the only difference between arms is
the content of the crop slots.

Frozen rendering scheme (see docs/duplex/PREREG_text_region_budget_probe.md):

  * text area of a frame = union OCR-box area as a fraction of frame area, the
    A2 census measure; a frame is a candidate donor when that fraction is at
    least 0.01, the census's own substantive-frame threshold.
  * donors = the k frames with the largest text area, k = min(4, number of
    candidate donors); the k frames with the smallest text area among the
    non-donors are dropped.
  * crop = union bounding box of that donor's detected boxes, each side
    expanded by 8 % of the corresponding bounding-box side, clamped to the
    frame.
  * every crop is resampled to a merge-grid whose token count is within two
    tokens of a full frame's token count for that video, choosing the grid
    whose aspect ratio is closest to the crop's.
  * slot order: temporal, with each crop placed immediately after its donor.
  * RAND and ANTI reuse the donor frames and the crop width and height in
    native pixels and move the rectangle: RAND to a uniformly random legal
    position (seed 20260808), ANTI to the legal position that maximises the
    minimum distance to every detected box in that frame.
  * a video with no candidate donor keeps BASELINE rendering in all three arms
    and is analysed as the no-text stratum.

Output under results/text_region_budget/:
  manifest_<arm>.json   video_id -> {"images": [...16 paths...], ...}
  accounting.json       per-video and corpus-level image-token accounting
  crops/<arm>/<video_id>_<slot>.png
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

# Judge-side vision geometry, read off src/duplex/extract_duplex_readout.py.
FACTOR = 32          # patch_size 16 * merge_size 2
MIN_PIXELS = 65536   # == 64 merge cells
MAX_PIXELS = 100352  # == 98 merge cells
MIN_TOKENS = MIN_PIXELS // (FACTOR * FACTOR)
MAX_TOKENS = MAX_PIXELS // (FACTOR * FACTOR)

SUBSTANTIVE_AREA_FRAC = 0.01   # A2 census threshold
N_DONORS = 4
PAD_FRAC = 0.08
TOKEN_TOL = 2                  # per-crop slack, in merge cells
RAND_SEED = 20260808
ANTI_GRID = 65                 # candidate positions per axis
BUDGET_TOL = 0.02              # +/- 2 % of the baseline per-video image tokens

ARMS = ("text", "rand", "anti")


def smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Qwen3-VL smart_resize as the judge's image processor applies it."""
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


def frame_tokens(h, w):
    hb, wb = smart_resize(h, w)
    return (hb // FACTOR) * (wb // FACTOR)


def choose_grid(target_tokens, crop_h, crop_w):
    """Merge grid (gh, gw) for a crop: token count within TOKEN_TOL of a full
    frame's, aspect ratio as close to the crop's as the factorisation allows."""
    want = math.log(crop_h / float(crop_w))
    lo = max(MIN_TOKENS, target_tokens - TOKEN_TOL)
    hi = min(MAX_TOKENS, target_tokens + TOKEN_TOL)
    best = None
    for prod in range(lo, hi + 1):
        for gh in range(1, prod + 1):
            if prod % gh:
                continue
            gw = prod // gh
            key = (abs(math.log(gh / float(gw)) - want), abs(prod - target_tokens), -gh)
            if best is None or key < best[0]:
                best = (key, (gh, gw, prod))
    return best[1]


def pad_box(x0, y0, x1, y1, W, H):
    bw, bh = x1 - x0, y1 - y0
    px, py = PAD_FRAC * bw, PAD_FRAC * bh
    nx0 = int(max(0, math.floor(x0 - px)))
    ny0 = int(max(0, math.floor(y0 - py)))
    nx1 = int(min(W, math.ceil(x1 + px)))
    ny1 = int(min(H, math.ceil(y1 + py)))
    return nx0, ny0, max(nx0 + 1, nx1), max(ny0 + 1, ny1)


def rect_gap(a, b):
    """Euclidean gap between two axis-aligned rectangles; 0 when they touch."""
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)


def overlap_area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def anti_position(cw, ch, W, H, boxes):
    """Legal top-left maximising the minimum distance to every detected box."""
    xs = np.unique(np.linspace(0, max(0, W - cw), ANTI_GRID).astype(int))
    ys = np.unique(np.linspace(0, max(0, H - ch), ANTI_GRID).astype(int))
    best, best_key = None, None
    for y in ys:
        for x in xs:
            rect = (x, y, x + cw, y + ch)
            d = min(rect_gap(rect, b) for b in boxes)
            key = (-d, int(y), int(x))
            if best_key is None or key < best_key:
                best_key, best = key, (int(x), int(y), d)
    return best


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHClip_EN")
    ap.add_argument("--split", default="test")
    ap.add_argument("--boxes", default=os.path.join(
        ROOT, "results", "text_region_budget", "boxes.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(
        ROOT, "results", "text_region_budget"))
    args = ap.parse_args()

    from data_utils import DATASET_ROOTS, load_clean_split_ids
    from PIL import Image

    ids = load_clean_split_ids(args.dataset, args.split)
    seen, uniq = set(), []
    for v in ids:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    frames_root = os.path.join(DATASET_ROOTS[args.dataset], "frames_16", "")

    boxes_by_vid = {}
    with open(args.boxes) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                boxes_by_vid[r["video_id"]] = r
    missing = [v for v in uniq if v not in boxes_by_vid]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} videos have no box record")

    for arm in ARMS:
        os.makedirs(os.path.join(args.out_dir, "crops", arm), exist_ok=True)

    rng = np.random.default_rng(RAND_SEED)
    manifests = {a: {} for a in ARMS}
    accounting = {}
    n_notext = 0

    for vid in uniq:
        rec = boxes_by_vid[vid]
        paths = sorted(glob.glob(os.path.join(frames_root, vid, "*.jpg")))
        frames = rec["frames"]
        if len(paths) != len(frames) or len(paths) != 16:
            raise SystemExit(f"ABORT: {vid}: {len(paths)} frames, {len(frames)} records")

        tok = [frame_tokens(fr["H"], fr["W"]) for fr in frames]
        if len(set(tok)) != 1:
            raise SystemExit(f"ABORT: {vid}: non-uniform per-frame tokens {set(tok)}")
        T = tok[0]
        base_total = 16 * T

        cand = [i for i, fr in enumerate(frames)
                if fr.get("n_boxes", 0) > 0 and fr["union_frac"] >= SUBSTANTIVE_AREA_FRAC]
        k = min(N_DONORS, len(cand))

        if k == 0:
            n_notext += 1
            for arm in ARMS:
                manifests[arm][vid] = {"images": paths, "no_text": True,
                                       "predicted_image_tokens": base_total}
            accounting[vid] = {"no_text": True, "frame_tokens": T,
                               "baseline_total": base_total,
                               "arm_total": base_total, "rel_dev": 0.0,
                               "n_donors": 0, "n_candidate_frames": 0}
            continue

        order = sorted(cand, key=lambda i: (-frames[i]["union_frac"], i))
        donors = sorted(order[:k])
        rest = [i for i in range(16) if i not in set(donors)]
        drop = set(sorted(rest, key=lambda i: (frames[i]["union_frac"], i))[:k])

        slots = {a: [] for a in ARMS}
        crop_tokens = []
        donor_info = []

        for i in range(16):
            if i in drop:
                continue
            for a in ARMS:
                slots[a].append(paths[i])
            if i not in donors:
                continue

            fr = frames[i]
            W, H = fr["W"], fr["H"]
            bs = [tuple(b) for b in fr["boxes"]]
            x0 = min(b[0] for b in bs)
            y0 = min(b[1] for b in bs)
            x1 = max(b[2] for b in bs)
            y1 = max(b[3] for b in bs)
            cx0, cy0, cx1, cy1 = pad_box(x0, y0, x1, y1, W, H)
            cw, ch = cx1 - cx0, cy1 - cy0
            gh, gw, prod = choose_grid(T, ch, cw)
            crop_tokens.append(prod)

            rx = int(rng.integers(0, max(1, W - cw + 1)))
            ry = int(rng.integers(0, max(1, H - ch + 1)))
            ax, ay, adist = anti_position(cw, ch, W, H, bs)

            placements = {
                "text": (cx0, cy0),
                "rand": (rx, ry),
                "anti": (ax, ay),
            }
            box_area_sum = sum(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                               for b in bs)

            def covered(place):
                px, py = place
                r = (px, py, px + cw, py + ch)
                return sum(overlap_area(r, b) for b in bs) / max(1e-9, box_area_sum)

            src = Image.open(paths[i]).convert("RGB")
            for a in ARMS:
                px, py = placements[a]
                out = os.path.join(args.out_dir, "crops", a, f"{vid}_s{i:02d}.png")
                src.crop((px, py, px + cw, py + ch)).resize(
                    (gw * FACTOR, gh * FACTOR), Image.BICUBIC).save(out)
                slots[a].append(out)
            src.close()

            donor_info.append({
                "slot": i,
                "frame_wh": [W, H],
                "union_frac": fr["union_frac"],
                "n_boxes": fr["n_boxes"],
                "crop_wh": [cw, ch],
                "crop_frac_of_frame": round(cw * ch / float(W * H), 5),
                "grid": [gh, gw],
                "crop_tokens": prod,
                "magnification_vs_baseline": round(
                    math.sqrt((cw * ch) / float(W * H)) ** -1
                    * math.sqrt(prod / float(T)), 4),
                "text_overlap_frac": {a: round(covered(placements[a]), 4)
                                      for a in ARMS},
                "anti_min_distance_px": round(adist, 1),
                "anti_overlaps_text": bool(adist <= 0.0),
            })

        arm_total = (16 - k) * T + sum(crop_tokens)
        rel = (arm_total - base_total) / float(base_total)
        if abs(rel) > BUDGET_TOL:
            raise SystemExit(f"ABORT: {vid}: token budget deviation {rel:+.4f}")

        for a in ARMS:
            if len(slots[a]) != 16:
                raise SystemExit(f"ABORT: {vid}/{a}: {len(slots[a])} slots")
            manifests[a][vid] = {"images": slots[a], "no_text": False,
                                 "predicted_image_tokens": arm_total}
        accounting[vid] = {
            "no_text": False, "frame_tokens": T,
            "baseline_total": base_total, "arm_total": arm_total,
            "rel_dev": round(rel, 6),
            "n_donors": k, "n_candidate_frames": len(cand),
            "donors": donor_info,
        }

    for a in ARMS:
        with open(os.path.join(args.out_dir, f"manifest_{a}.json"), "w") as f:
            json.dump(manifests[a], f)

    devs = [accounting[v]["rel_dev"] for v in uniq]
    tot_base = sum(accounting[v]["baseline_total"] for v in uniq)
    tot_arm = sum(accounting[v]["arm_total"] for v in uniq)
    n_anti_overlap = sum(1 for v in uniq for d in accounting[v].get("donors", [])
                         if d["anti_overlaps_text"])
    n_donor_total = sum(len(accounting[v].get("donors", [])) for v in uniq)
    summary = {
        "n_videos": len(uniq),
        "n_no_text_stratum": n_notext,
        "n_with_text": len(uniq) - n_notext,
        "n_donor_slots": n_donor_total,
        "per_video_rel_dev": {
            "min": min(devs), "max": max(devs),
            "mean_abs": round(float(np.mean(np.abs(devs))), 6),
            "n_outside_2pct": sum(1 for d in devs if abs(d) > BUDGET_TOL),
        },
        "corpus_image_tokens": {
            "baseline": tot_base, "arms": tot_arm,
            "rel_dev": round((tot_arm - tot_base) / float(tot_base), 6),
        },
        "n_anti_slots_forced_to_overlap_text": n_anti_overlap,
        "median_crop_frac_of_frame": round(float(np.median(
            [d["crop_frac_of_frame"] for v in uniq
             for d in accounting[v].get("donors", [])])), 5) if n_donor_total else None,
        "median_linear_magnification_of_text": round(float(np.median(
            [d["magnification_vs_baseline"] for v in uniq
             for d in accounting[v].get("donors", [])])), 4) if n_donor_total else None,
        "rand_mean_text_overlap_frac": round(float(np.mean(
            [d["text_overlap_frac"]["rand"] for v in uniq
             for d in accounting[v].get("donors", [])])), 4) if n_donor_total else None,
        "anti_mean_text_overlap_frac": round(float(np.mean(
            [d["text_overlap_frac"]["anti"] for v in uniq
             for d in accounting[v].get("donors", [])])), 4) if n_donor_total else None,
    }
    with open(os.path.join(args.out_dir, "accounting.json"), "w") as f:
        json.dump({"summary": summary, "per_video": accounting}, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)

    # ------------------------------------------------------- pre-flight decode
    n_img, bad = 0, []
    for a in ARMS:
        for vid, m in manifests[a].items():
            for p in m["images"]:
                try:
                    with Image.open(p) as im:
                        im.load()
                    n_img += 1
                except Exception as e:  # noqa: BLE001
                    bad.append((a, vid, p, repr(e)[:120]))
    print(f"[preflight] decoded {n_img} images across {len(ARMS)} arms, "
          f"{len(bad)} failures", flush=True)
    if bad:
        for b in bad[:20]:
            print("  FAIL", b, flush=True)
        raise SystemExit("ABORT: pre-flight decode failures")


if __name__ == "__main__":
    build()
