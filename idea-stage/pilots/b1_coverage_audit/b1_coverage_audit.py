"""B1: does uniform-16 frame sampling miss the hate-bearing segments of HateClipSeg videos?

Zero-GPU. All parameters are frozen in PREREG.md; this file is the executable copy of them.

Stages (each caches to data/cache/*.json so the expensive ones run once):
  probe   ffprobe every media file -> n_frames, fps, duration
  shots   ffmpeg 4fps/64x36/gray -> absdiff change points -> shot boundaries
  vad     ffmpeg 16kHz mono -> webrtcvad voiced fraction (transcript-length proxy)
  analyze annotations x samplers -> results.json

Usage: python b1_coverage_audit.py [probe|shots|vad|analyze|all]
"""

import ast
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
VDIR = os.path.join(DATA, "videos")
CACHE = os.path.join(DATA, "cache")
os.makedirs(CACHE, exist_ok=True)

IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
OFFENSIVE_DIMS = [1, 2, 3, 4, 5]

# ---- frozen sampler / detector parameters (see PREREG.md) ----
KS = [8, 16, 32]
PRIMARY_K = 16
SHOT_FPS = 4.0
SHOT_W, SHOT_H = 64, 36
TAUS = [0.05, 0.10, 0.15]
PRIMARY_TAU = 0.10
MIN_SHOT_SEC = 1.0
N_RANDOM_DRAWS = 200
RANDOM_SEED = 20260807
VAD_AGGRESSIVENESS = 2
WORKERS = 12


# --------------------------------------------------------------------------- annotations
def load_annotations():
    vids, segs = {}, {}
    with open(os.path.join(DATA, "video_level_annotation.csv")) as f:
        for row in csv.DictReader(f):
            vids[row["Video Id"].strip()] = {
                "labels": ast.literal_eval(row["Video-Level Label"]),
                "victims": ast.literal_eval(row["Target Victim"]),
            }
    with open(os.path.join(DATA, "segment_level_annotation.csv")) as f:
        for row in csv.DictReader(f):
            vid = row["Video Id"].strip()
            lab = ast.literal_eval(row["Segment-Level Label"])
            ts = [[float(a), float(b)] for a, b in ast.literal_eval(row["Segment Timestamp"])]
            assert len(lab) == len(ts), vid
            segs[vid] = {"labels": lab, "ts": ts, "ann_dur": ts[-1][1] if ts else 0.0}
    return vids, segs


def media_path(vid):
    for ext in (".mp4", ".mkv", ".webm"):
        p = os.path.join(VDIR, vid + ext)
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------- probe
def _probe_one(vid):
    p = media_path(vid)
    if p is None:
        return vid, None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=nb_frames,avg_frame_rate,duration:format=duration",
           "-of", "json", p]
    try:
        out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout)
    except Exception as e:
        return vid, {"error": str(e)}
    st = (out.get("streams") or [{}])[0]
    fr = st.get("avg_frame_rate", "0/0")
    try:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    dur = None
    for cand in (st.get("duration"), (out.get("format") or {}).get("duration")):
        try:
            dur = float(cand)
            break
        except (TypeError, ValueError):
            continue
    nb = st.get("nb_frames")
    try:
        n = int(nb)
    except (TypeError, ValueError):
        n = int(round(dur * fps)) if (dur and fps) else 0
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", p],
        capture_output=True, text=True).stdout.strip() != ""
    return vid, {"path": os.path.basename(p), "fps": fps, "n_frames": n,
                 "duration": dur, "has_audio": has_audio}


def stage_probe(all_vids):
    res = {}
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_probe_one, v) for v in all_vids]
        for i, fu in enumerate(as_completed(futs)):
            v, r = fu.result()
            if r is not None:
                res[v] = r
            if (i + 1) % 100 == 0:
                print(f"  probe {i+1}/{len(futs)}", flush=True)
    json.dump(res, open(os.path.join(CACHE, "probe.json"), "w"))
    print(f"probe: {len(res)} media probed")
    return res


# --------------------------------------------------------------------------- shots
def _shots_one(vid):
    """Return {tau: [boundary_times]} using the frozen absdiff rule."""
    p = media_path(vid)
    if p is None:
        return vid, None
    cmd = ["ffmpeg", "-v", "error", "-i", p, "-an",
           "-vf", f"fps={SHOT_FPS},scale={SHOT_W}:{SHOT_H}",
           "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    try:
        pr = subprocess.run(cmd, capture_output=True, timeout=1800)
    except Exception as e:
        return vid, {"error": str(e)}
    buf = np.frombuffer(pr.stdout, dtype=np.uint8)
    fsz = SHOT_W * SHOT_H
    nf = len(buf) // fsz
    if nf < 2:
        return vid, {"n_sampled": int(nf), "boundaries": {str(t): [] for t in TAUS}}
    fr = buf[: nf * fsz].reshape(nf, fsz).astype(np.int16)
    d = np.abs(np.diff(fr, axis=0)).mean(axis=1) / 255.0  # d[i] compares frame i and i+1
    out = {"n_sampled": int(nf), "boundaries": {}}
    for tau in TAUS:
        cand = np.nonzero(d > tau)[0]
        acc, last = [], -1e9
        for i in cand:
            t = (i + 1) / SHOT_FPS  # boundary sits at the later frame's time
            if t - last >= MIN_SHOT_SEC:
                acc.append(round(float(t), 3))
                last = t
        out["boundaries"][str(tau)] = acc
    return vid, out


def stage_shots(vids_with_media):
    path = os.path.join(CACHE, "shots.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [v for v in vids_with_media if v not in res]
    print(f"shots: {len(todo)} to do ({len(res)} cached)")
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_shots_one, v) for v in todo]
        for i, fu in enumerate(as_completed(futs)):
            v, r = fu.result()
            if r is not None:
                res[v] = r
            if (i + 1) % 25 == 0:
                print(f"  shots {i+1}/{len(futs)}", flush=True)
                json.dump(res, open(path, "w"))
    json.dump(res, open(path, "w"))
    return res


# --------------------------------------------------------------------------- vad
def _vad_one(vid):
    import webrtcvad
    p = media_path(vid)
    if p is None:
        return vid, None
    cmd = ["ffmpeg", "-v", "error", "-i", p, "-vn", "-ac", "1", "-ar", "16000",
           "-f", "s16le", "-"]
    try:
        pr = subprocess.run(cmd, capture_output=True, timeout=1800)
    except Exception as e:
        return vid, {"error": str(e)}
    pcm = pr.stdout
    if len(pcm) < 32000:
        return vid, {"voiced_frac": None, "audio_sec": len(pcm) / 32000.0}
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    step = int(16000 * 0.03) * 2  # 30 ms, 16-bit
    n = len(pcm) // step
    voiced = 0
    for i in range(n):
        if vad.is_speech(pcm[i * step:(i + 1) * step], 16000):
            voiced += 1
    return vid, {"voiced_frac": voiced / n if n else None,
                 "audio_sec": len(pcm) / 32000.0, "n_vad_frames": n}


def stage_vad(vids_with_media):
    path = os.path.join(CACHE, "vad.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [v for v in vids_with_media if v not in res]
    print(f"vad: {len(todo)} to do ({len(res)} cached)")
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_vad_one, v) for v in todo]
        for i, fu in enumerate(as_completed(futs)):
            v, r = fu.result()
            if r is not None:
                res[v] = r
            if (i + 1) % 50 == 0:
                print(f"  vad {i+1}/{len(futs)}", flush=True)
                json.dump(res, open(path, "w"))
    json.dump(res, open(path, "w"))
    return res


# --------------------------------------------------------------------------- samplers
def uniform_times(n_frames, fps, K):
    """The repo's frames_16 rule, in seconds."""
    idx = np.linspace(0, n_frames - 1, K, dtype=int)
    return (idx / fps).tolist()


def shots_from_boundaries(bnds, duration):
    edges = [0.0] + [b for b in bnds if 0.0 < b < duration] + [duration]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def shot_cover_times(shots, K):
    """Frozen budget allocation: >=K shots -> K longest, midpoints;
    <K shots -> 1 each, remainder by largest-remainder on duration."""
    if not shots:
        return []
    order = sorted(range(len(shots)), key=lambda i: -(shots[i][1] - shots[i][0]))
    if len(shots) >= K:
        pick = order[:K]
        return sorted((shots[i][0] + shots[i][1]) / 2.0 for i in pick)
    durs = np.array([e - s for s, e in shots], dtype=float)
    extra = K - len(shots)
    share = durs / durs.sum() * extra
    base = np.floor(share).astype(int)
    rem = extra - int(base.sum())
    for i in np.argsort(-(share - base))[:rem]:
        base[i] += 1
    times = []
    for i, (s, e) in enumerate(shots):
        m = 1 + int(base[i])
        for j in range(m):
            times.append(s + (e - s) * (j + 0.5) / m)
    return sorted(times)


def hit_mask(times, ts):
    """Which annotated segments contain >=1 sampled time."""
    hit = [False] * len(ts)
    last = len(ts) - 1
    for t in times:
        for i, (s, e) in enumerate(ts):
            if s <= t < e or (i == last and s <= t <= e):
                hit[i] = True
                break
    return hit


# --------------------------------------------------------------------------- analysis
def frac(a, b):
    return (a / b) if b else None


def analyze(vids, segs, probe, shots, vad):
    rng = np.random.default_rng(RANDOM_SEED)
    have = [v for v in segs if v in probe and probe[v].get("n_frames") and probe[v].get("fps")]
    have.sort()

    # --- schema census over the full annotation set ---
    schema = {"n_videos_annotated": len(segs),
              "n_segments_annotated": sum(len(s["labels"]) for s in segs.values()),
              "n_videos_with_media": len(have),
              "n_segments_with_media": sum(len(segs[v]["labels"]) for v in have),
              "per_dim": {}}
    for i, name in enumerate(IDX):
        schema["per_dim"][name] = {
            "segments_all": sum(sum(x[i] for x in s["labels"]) for s in segs.values()),
            "videos_all": sum(1 for s in segs.values() if any(x[i] for x in s["labels"])),
            "segments_media": sum(sum(x[i] for x in segs[v]["labels"]) for v in have),
            "videos_media": sum(1 for v in have if any(x[i] for x in segs[v]["labels"])),
        }
    schema["offensive_union"] = {
        "segments_all": sum(sum(1 for x in s["labels"] if any(x[d] for d in OFFENSIVE_DIMS))
                            for s in segs.values()),
        "videos_all": sum(1 for s in segs.values()
                          if any(any(x[d] for d in OFFENSIVE_DIMS) for x in s["labels"])),
        "segments_media": sum(sum(1 for x in segs[v]["labels"] if any(x[d] for d in OFFENSIVE_DIMS))
                              for v in have),
        "videos_media": sum(1 for v in have
                            if any(any(x[d] for d in OFFENSIVE_DIMS) for x in segs[v]["labels"])),
    }
    # annotation-vs-container duration agreement
    dd = [(segs[v]["ann_dur"], probe[v]["duration"]) for v in have if probe[v].get("duration")]
    rel = [abs(a - b) / max(a, 1e-6) for a, b in dd]
    schema["ann_vs_container_duration"] = {
        "n": len(rel), "median_rel_err": float(np.median(rel)),
        "frac_within_5pct": float(np.mean([r <= 0.05 for r in rel])),
    }

    # --- per-video sampler evaluation ---
    per_video = {}
    for v in have:
        pr, sg = probe[v], segs[v]
        ts, labs = sg["ts"], sg["labels"]
        dur = pr["duration"] or sg["ann_dur"]
        rec = {"n_seg": len(ts), "ann_dur": sg["ann_dur"], "dur": dur,
               "voiced_frac": (vad.get(v) or {}).get("voiced_frac"),
               "samplers": {}}
        for K in KS:
            t = uniform_times(pr["n_frames"], pr["fps"], K)
            rec["samplers"][f"uniform_{K}"] = hit_mask(t, ts)
        # random-16: mean over draws, plus per-draw video-level hit handled later
        rec["random_16_seg_hit_prob"] = None
        draws = rng.random((N_RANDOM_DRAWS, PRIMARY_K)) * dur
        cnt = np.zeros(len(ts))
        vid_hit_union = np.zeros(N_RANDOM_DRAWS, dtype=bool)
        vid_hit_strict = np.zeros(N_RANDOM_DRAWS, dtype=bool)
        u_idx = [i for i, x in enumerate(labs) if any(x[d] for d in OFFENSIVE_DIMS)]
        s_idx = [i for i, x in enumerate(labs) if x[1]]
        for k in range(N_RANDOM_DRAWS):
            m = hit_mask(draws[k].tolist(), ts)
            cnt += np.array(m, dtype=float)
            vid_hit_union[k] = any(m[i] for i in u_idx)
            vid_hit_strict[k] = any(m[i] for i in s_idx)
        rec["random_16_seg_hit_prob"] = (cnt / N_RANDOM_DRAWS).tolist()
        rec["random_16_vid_hit_union"] = float(vid_hit_union.mean())
        rec["random_16_vid_hit_strict"] = float(vid_hit_strict.mean())
        # shot-covering
        sh = shots.get(v) or {}
        for tau in TAUS:
            b = (sh.get("boundaries") or {}).get(str(tau))
            if b is None:
                rec["samplers"][f"shot_{tau}"] = None
                continue
            sl = shots_from_boundaries(b, dur)
            t = shot_cover_times(sl, PRIMARY_K)
            rec["samplers"][f"shot_{tau}"] = hit_mask(t, ts)
            rec[f"n_shots_{tau}"] = len(sl)
        per_video[v] = rec

    # --- aggregate ---
    def agg(target_idx_fn, name):
        """target_idx_fn(labels) -> list of segment indices that count as hate-bearing."""
        out = {"definition": name, "samplers": {}}
        vids_t = [v for v in have if target_idx_fn(segs[v]["labels"])]
        out["n_videos"] = len(vids_t)
        out["n_segments"] = sum(len(target_idx_fn(segs[v]["labels"])) for v in vids_t)
        keys = [f"uniform_{K}" for K in KS] + [f"shot_{t}" for t in TAUS]
        for key in keys:
            nv = nvh = ns = nsh = 0
            for v in vids_t:
                m = per_video[v]["samplers"].get(key)
                if m is None:
                    continue
                idxs = target_idx_fn(segs[v]["labels"])
                nv += 1
                nvh += int(any(m[i] for i in idxs))
                ns += len(idxs)
                nsh += sum(int(m[i]) for i in idxs)
            out["samplers"][key] = {"n_videos": nv, "video_hit_rate": frac(nvh, nv),
                                    "n_segments": ns, "segment_hit_rate": frac(nsh, ns)}
        # random-16
        vh = [per_video[v]["random_16_vid_hit_union" if name == "offensive_union"
                           else "random_16_vid_hit_strict"] for v in vids_t]
        sp, sn = 0.0, 0
        for v in vids_t:
            probs = per_video[v]["random_16_seg_hit_prob"]
            for i in target_idx_fn(segs[v]["labels"]):
                sp += probs[i]
                sn += 1
        out["samplers"]["random_16"] = {"n_videos": len(vh),
                                        "video_hit_rate": float(np.mean(vh)) if vh else None,
                                        "n_segments": sn, "segment_hit_rate": frac(sp, sn)}
        return out

    def f_union(labs):
        return [i for i, x in enumerate(labs) if any(x[d] for d in OFFENSIVE_DIMS)]

    def f_strict(labs):
        return [i for i, x in enumerate(labs) if x[1]]

    results = {"schema": schema, "definitions": {}, "per_dim": {}}
    results["definitions"]["offensive_union"] = agg(f_union, "offensive_union")
    results["definitions"]["hateful_strict"] = agg(f_strict, "hateful_strict")
    for d in OFFENSIVE_DIMS:
        results["per_dim"][IDX[d]] = agg(
            (lambda dd: (lambda labs: [i for i, x in enumerate(labs) if x[dd]]))(d), IDX[d])

    # --- stratification (offensive_union, uniform-16 primary) ---
    def strat(key, defn_fn, name):
        vids_t = [v for v in have if defn_fn(segs[v]["labels"])]
        by_segdur = defaultdict(lambda: [0, 0])
        by_viddur = defaultdict(lambda: [0, 0, 0, 0])  # segs hit, segs, vids hit, vids
        by_nseg = defaultdict(lambda: [0, 0, 0, 0])
        seg_bins = [(-1e9, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, 60), (60, 1e9)]
        # the corpus is duration-homogeneous (all clips 180-350 s), so fixed decade bins
        # collapse to one cell; use empirical quintiles instead.
        dvals = sorted(per_video[v]["dur"] or 0 for v in vids_t)
        qe = [dvals[int(len(dvals) * f)] for f in (0.2, 0.4, 0.6, 0.8)]
        vid_bins = [(0, qe[0]), (qe[0], qe[1]), (qe[1], qe[2]), (qe[2], qe[3]), (qe[3], 1e9)]
        nseg_bins = [(0, 10), (10, 20), (20, 40), (40, 1e9)]

        def blab(bins, x):
            for lo, hi in bins:
                if lo <= x < hi:
                    lo_s = "neg" if lo < -1e8 else f"{lo:g}"
                    return f"{lo_s}-{'inf' if hi > 1e8 else f'{hi:g}'}"
            return "na"
        for v in vids_t:
            m = per_video[v]["samplers"][key]
            if m is None:
                continue
            idxs = defn_fn(segs[v]["labels"])
            ts = segs[v]["ts"]
            vb = blab(vid_bins, per_video[v]["dur"] or 0)
            nb = blab(nseg_bins, per_video[v]["n_seg"])
            vhit = int(any(m[i] for i in idxs))
            by_viddur[vb][2] += vhit
            by_viddur[vb][3] += 1
            by_nseg[nb][2] += vhit
            by_nseg[nb][3] += 1
            for i in idxs:
                sb = blab(seg_bins, ts[i][1] - ts[i][0])
                by_segdur[sb][0] += int(m[i])
                by_segdur[sb][1] += 1
                by_viddur[vb][0] += int(m[i])
                by_viddur[vb][1] += 1
                by_nseg[nb][0] += int(m[i])
                by_nseg[nb][1] += 1
        return {
            "by_segment_duration_s": {k: {"segments": b, "hit": a, "segment_hit_rate": frac(a, b)}
                                      for k, (a, b) in sorted(by_segdur.items())},
            "by_video_duration_s": {k: {"segments": b, "seg_hit_rate": frac(a, b),
                                        "videos": d, "video_hit_rate": frac(c, d)}
                                    for k, (a, b, c, d) in sorted(by_viddur.items())},
            "by_segment_count": {k: {"segments": b, "seg_hit_rate": frac(a, b),
                                     "videos": d, "video_hit_rate": frac(c, d)}
                                 for k, (a, b, c, d) in sorted(by_nseg.items())},
        }

    results["stratification_uniform16_offensive_union"] = strat(
        f"uniform_{PRIMARY_K}", f_union, "offensive_union")

    # --- budget ceiling: best segment_hit_rate any K-frame sampler can reach ---
    # Segments tile the video contiguously, so K frames touch at most K distinct segments.
    ceil = {}
    for K in KS:
        for nm, fn in (("offensive_union", f_union), ("hateful_strict", f_strict)):
            vids_t = [v for v in have if fn(segs[v]["labels"])]
            num = 0
            den = 0
            for v in vids_t:
                idxs = fn(segs[v]["labels"])
                num += min(K, len(idxs))   # an oracle spends every frame on a distinct target
                den += len(idxs)
            ceil.setdefault(nm, {})[f"K{K}"] = {
                "oracle_segment_hit_rate": frac(num, den),
                "note": "K frames land in at most K of the contiguous annotation segments",
            }
    results["budget_ceiling"] = ceil
    results["segments_per_video"] = {
        "median": float(np.median([per_video[v]["n_seg"] for v in have])),
        "mean": float(np.mean([per_video[v]["n_seg"] for v in have])),
    }

    # --- forensics on the videos uniform-16 misses entirely ---
    missed_vids = []
    for v in have:
        idxs = f_union(segs[v]["labels"])
        if not idxs:
            continue
        m = per_video[v]["samplers"][f"uniform_{PRIMARY_K}"]
        if not any(m[i] for i in idxs):
            od = sum(segs[v]["ts"][i][1] - segs[v]["ts"][i][0] for i in idxs)
            missed_vids.append({
                "n_offensive_segments": len(idxs), "n_segments": len(m),
                "offensive_seconds": round(od, 2), "duration": round(per_video[v]["dur"], 2),
                "offensive_time_share": round(od / max(per_video[v]["dur"], 1e-6), 4),
                "voiced_frac": per_video[v]["voiced_frac"],
            })
    results["videos_fully_missed_by_uniform16"] = {
        "n": len(missed_vids),
        "median_n_offensive_segments": float(np.median([x["n_offensive_segments"] for x in missed_vids])) if missed_vids else None,
        "median_offensive_time_share": float(np.median([x["offensive_time_share"] for x in missed_vids])) if missed_vids else None,
        "cases": missed_vids,
    }

    # --- missed-segment forensics (uniform-16, offensive_union) ---
    missed, hit_d = [], []
    for v in have:
        idxs = f_union(segs[v]["labels"])
        if not idxs:
            continue
        m = per_video[v]["samplers"][f"uniform_{PRIMARY_K}"]
        for i in idxs:
            d = segs[v]["ts"][i][1] - segs[v]["ts"][i][0]
            (hit_d if m[i] else missed).append(d)
    q = [5, 25, 50, 75, 95]
    results["missed_segment_durations"] = {
        "n_missed": len(missed), "n_hit": len(hit_d),
        "missed_percentiles_s": {f"p{p}": float(np.percentile(missed, p)) for p in q} if missed else {},
        "hit_percentiles_s": {f"p{p}": float(np.percentile(hit_d, p)) for p in q} if hit_d else {},
        "missed_mean_s": float(np.mean(missed)) if missed else None,
        "hit_mean_s": float(np.mean(hit_d)) if hit_d else None,
        "frac_missed_under_5s": float(np.mean([d < 5 for d in missed])) if missed else None,
        "frac_missed_under_10s": float(np.mean([d < 10 for d in missed])) if missed else None,
        "missed_duration_share_of_offensive_time": None,
    }
    tot_off = sum(missed) + sum(hit_d)
    if tot_off:
        results["missed_segment_durations"]["missed_duration_share_of_offensive_time"] = \
            float(sum(missed) / tot_off)

    # --- transcript-length stratum ---
    vids_t = [v for v in have if f_union(segs[v]["labels"])]
    vf = [(per_video[v]["voiced_frac"], v) for v in vids_t
          if per_video[v]["voiced_frac"] is not None]
    tr = {"n_with_vad": len(vf), "n_videos": len(vids_t)}
    if vf:
        vals = np.array([x[0] for x in vf])
        qs = np.quantile(vals, [0.25, 0.5, 0.75])
        tr["voiced_frac_quartile_edges"] = [float(x) for x in qs]
        buckets = defaultdict(lambda: [0, 0, 0, 0])
        for val, v in vf:
            b = "Q1_lowest" if val <= qs[0] else "Q2" if val <= qs[1] else "Q3" if val <= qs[2] else "Q4_highest"
            m = per_video[v]["samplers"][f"uniform_{PRIMARY_K}"]
            idxs = f_union(segs[v]["labels"])
            buckets[b][0] += sum(int(m[i]) for i in idxs)
            buckets[b][1] += len(idxs)
            buckets[b][2] += int(all(m[i] for i in idxs))  # fully covered
            buckets[b][3] += 1
        tr["by_voiced_quartile"] = {
            k: {"videos": d, "segments": b, "segment_hit_rate": frac(a, b),
                "frac_videos_fully_covered": frac(c, d)}
            for k, (a, b, c, d) in sorted(buckets.items())}
        # do videos with >=1 missed offensive segment skew low-voiced?
        miss_v = [val for val, v in vf
                  if not all(per_video[v]["samplers"][f"uniform_{PRIMARY_K}"][i]
                             for i in f_union(segs[v]["labels"]))]
        full_v = [val for val, v in vf
                  if all(per_video[v]["samplers"][f"uniform_{PRIMARY_K}"][i]
                         for i in f_union(segs[v]["labels"]))]
        tr["voiced_frac_mean_videos_with_missed_segment"] = float(np.mean(miss_v)) if miss_v else None
        tr["voiced_frac_mean_videos_fully_covered"] = float(np.mean(full_v)) if full_v else None
        tr["n_videos_with_missed_segment"] = len(miss_v)
        tr["n_videos_fully_covered"] = len(full_v)
    # video-level miss (not a single offensive segment hit) vs voiced fraction
    lost = [per_video[v]["voiced_frac"] for v in vids_t
            if per_video[v]["voiced_frac"] is not None
            and not any(per_video[v]["samplers"][f"uniform_{PRIMARY_K}"][i]
                        for i in f_union(segs[v]["labels"]))]
    kept = [per_video[v]["voiced_frac"] for v in vids_t
            if per_video[v]["voiced_frac"] is not None
            and any(per_video[v]["samplers"][f"uniform_{PRIMARY_K}"][i]
                    for i in f_union(segs[v]["labels"]))]
    tr["video_level_miss_vs_voiced"] = {
        "n_missed_videos": len(lost), "n_hit_videos": len(kept),
        "voiced_frac_mean_missed": float(np.mean(lost)) if lost else None,
        "voiced_frac_mean_hit": float(np.mean(kept)) if kept else None,
        "voiced_frac_median_missed": float(np.median(lost)) if lost else None,
        "voiced_frac_median_hit": float(np.median(kept)) if kept else None,
    }
    if lost and kept:
        obs = np.mean(kept) - np.mean(lost)
        pool = np.array(lost + kept)
        rs = np.random.default_rng(RANDOM_SEED)
        null = []
        for _ in range(10000):
            rs.shuffle(pool)
            null.append(pool[len(lost):].mean() - pool[:len(lost)].mean())
        tr["video_level_miss_vs_voiced"]["permutation_p_two_sided"] = \
            float(np.mean(np.abs(np.array(null)) >= abs(obs)))
    results["transcript_proxy"] = tr

    # annotation anomalies worth flagging
    neg = [(v, i) for v in have for i, (s, e) in enumerate(segs[v]["ts"]) if e <= s]
    results["annotation_anomalies"] = {
        "n_segments_with_nonpositive_duration": len(neg),
        "note": "start >= end in the shipped timestamps; excluded from no analysis, "
                "they simply can never be hit",
    }

    # --- shot detector diagnostics ---
    sd = {}
    for tau in TAUS:
        ns = [per_video[v].get(f"n_shots_{tau}") for v in vids_t
              if per_video[v].get(f"n_shots_{tau}") is not None]
        sd[str(tau)] = {"n_videos": len(ns),
                        "median_shots": float(np.median(ns)) if ns else None,
                        "mean_shots": float(np.mean(ns)) if ns else None,
                        "frac_videos_ge16_shots": float(np.mean([x >= 16 for x in ns])) if ns else None,
                        "frac_videos_1_shot": float(np.mean([x == 1 for x in ns])) if ns else None,
                        "p10_p90": [float(np.percentile(ns, 10)), float(np.percentile(ns, 90))] if ns else None}
    results["shot_detector_diagnostics"] = sd

    # --- kill bar verdicts ---
    u16u = results["definitions"]["offensive_union"]["samplers"]["uniform_16"]
    u16s = results["definitions"]["hateful_strict"]["samplers"]["uniform_16"]
    sh = results["definitions"]["offensive_union"]["samplers"][f"shot_{PRIMARY_TAU}"]
    shs = results["definitions"]["hateful_strict"]["samplers"][f"shot_{PRIMARY_TAU}"]
    results["kill_bars"] = {
        "bar1_uniform16_video_hit_ge_0.95": {
            "offensive_union": {"value": u16u["video_hit_rate"],
                                "KILLED": u16u["video_hit_rate"] >= 0.95},
            "hateful_strict": {"value": u16s["video_hit_rate"],
                               "KILLED": u16s["video_hit_rate"] >= 0.95},
        },
        "bar2_shot_minus_uniform_segment_hit_lt_10pp": {
            "offensive_union": {
                "uniform_16": u16u["segment_hit_rate"], "shot_16": sh["segment_hit_rate"],
                "delta_pp": (sh["segment_hit_rate"] - u16u["segment_hit_rate"]) * 100,
                "KILLED": (sh["segment_hit_rate"] - u16u["segment_hit_rate"]) * 100 < 10},
            "hateful_strict": {
                "uniform_16": u16s["segment_hit_rate"], "shot_16": shs["segment_hit_rate"],
                "delta_pp": (shs["segment_hit_rate"] - u16s["segment_hit_rate"]) * 100,
                "KILLED": (shs["segment_hit_rate"] - u16s["segment_hit_rate"]) * 100 < 10},
        },
    }

    results["frozen_params"] = {
        "uniform_rule": "np.linspace(0, n_frames-1, K, dtype=int) / fps",
        "K": KS, "shot_fps": SHOT_FPS, "shot_size": [SHOT_W, SHOT_H],
        "taus": TAUS, "primary_tau": PRIMARY_TAU, "min_shot_sec": MIN_SHOT_SEC,
        "n_random_draws": N_RANDOM_DRAWS, "seed": RANDOM_SEED,
        "vad_aggressiveness": VAD_AGGRESSIVENESS,
        "lexicons_json": "quarantined, not downloaded, not read",
    }
    return results


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    vids, segs = load_annotations()
    all_vids = sorted(segs)
    if stage in ("probe", "all"):
        stage_probe(all_vids)
    probe = json.load(open(os.path.join(CACHE, "probe.json")))
    with_media = [v for v in all_vids if v in probe]
    if stage in ("shots", "all"):
        stage_shots(with_media)
    if stage in ("vad", "all"):
        stage_vad([v for v in with_media if probe[v].get("has_audio")])
    if stage in ("analyze", "all"):
        shots = json.load(open(os.path.join(CACHE, "shots.json")))
        vad = json.load(open(os.path.join(CACHE, "vad.json")))
        res = analyze(vids, segs, probe, shots, vad)
        out = os.path.join(HERE, "results.json")
        json.dump(res, open(out, "w"), indent=1)
        print(json.dumps(res["kill_bars"], indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()
