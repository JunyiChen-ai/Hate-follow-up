#!/usr/bin/env python3
"""Exploratory paired analysis. GT is used only by diagnostics/evaluation."""
import argparse
import csv
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.evaluate import binary_intervals, within_video_macro
from src.video_inputs import load_asr, window_text

OUT = ROOT / "runs/20261002_verdict_analysis"
DATASETS = ("HateMM", "HateClipSeg", "DeHate")
READS = {
    "main": ("runs/20260910_spvl/mllm/q3vl-8b/full", "runs/20260910_spvl/mllm/q3vl-8b/nostance"),
    "DeHate": ("runs/20260927_dehate_external/reads_gridA", "runs/20260928_infer/dehate/reads_nostance"),
}
FINALS = {
    "main": ("runs/20261002_verdict_analysis/main_full", "runs/20261002_verdict_analysis/main_nostance"),
    "DeHate": ("runs/20260926_twolevel/final_dehate/final_m2", "runs/20260928_infer/dehate/abl_nostance"),
}


def read(path):
    latest = {}
    for line in (ROOT / path / "predictions.jsonl").open():
        r = json.loads(line)
        latest[(r["dataset"], r["video_id"])] = r
    assert not any(r.get("error") for r in latest.values()), path
    return latest


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csvdump(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def auc(y, s):
    return within_video_macro({"v": y}, {"v": s})["within_video_macro_ROC_AUC"]


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[ok], np.asarray(b)[ok]
    return float(spearmanr(a, b).statistic) if len(a) > 1 and np.ptp(a) and np.ptp(b) else None


def curve(wins, field, n):
    """Diagnostic zero-order hold on each half-open window; missing branch is NaN."""
    t = np.arange(n) / 4
    result = np.full(n, np.nan)
    for w in wins:
        if field in w:
            result[(t >= w["start"]) & (t < w["end"])] = w[field]
    return result


def bootstrap(xs):
    a = np.asarray([x for x in xs if x is not None], float)
    if not len(a):
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(20261002)
    bs = a[rng.integers(len(a), size=(10000, len(a)))].mean(axis=1)
    return {"n": len(a), "mean": float(a.mean()), "ci95": np.quantile(bs, [.025, .975]).tolist(),
            "improved": int((a > 1e-10).sum()), "worse": int((a < -1e-10).sum()),
            "tied": int((np.abs(a) <= 1e-10).sum())}


def summarize(rows):
    out = {"n_all": len(rows), "final_delta": bootstrap([r.get("final_delta") for r in rows]),
           "raw_max_delta": bootstrap([r["z_auc_delta"] for r in rows])}
    for key in ("final_full", "final_no", "z_auc_full", "z_auc_no", "z_speech_auc_delta", "z_visual_auc_delta",
                "z_shift", "z_shift_positive", "z_shift_negative", "z_positive_minus_negative_shift",
                "z_visual_shift", "z_speech_shift", "z_signflip"):
        out[key + "_mean"] = mean([r.get(key) for r in rows])
    for key in ("z_spearman", "z_visual_spearman", "z_speech_spearman", "z_offset_energy_share"):
        out[key + "_median"] = median([r.get(key) for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-only", action="store_true")
    args = ap.parse_args()
    out = OUT / ("raw_analysis" if args.raw_only else "analysis")
    out.mkdir(parents=True, exist_ok=True)
    print("host", socket.gethostname(), flush=True)
    (out / "run.pid").write_text(str(os.getpid()))
    reads = {k: tuple(read(p) for p in v) for k, v in READS.items()}
    finals = {} if args.raw_only else {k: tuple(read(p) for p in v) for k, v in FINALS.items()}
    aligned = {"pairs": {}}
    for family, (full, no) in reads.items():
        assert full.keys() == no.keys()
        for k in full:
            a, b = full[k]["extra"], no[k]["extra"]
            assert a["z_video"] == b["z_video"]
            assert [(w["start"], w["end"]) for w in a["windows"]] == [(w["start"], w["end"]) for w in b["windows"]]
            assert [[m for m in ("z_visual", "z_speech") if m in w] for w in a["windows"]] == [[m for m in ("z_visual", "z_speech") if m in w] for w in b["windows"]]
        aligned["pairs"][family] = {"n_paired": len(full), "identical_global_z_and_grids_and_branch_presence": True}
    allrows, winrows, cases, summary = [], [], {}, {}
    for ds in DATASETS:
        fam = "DeHate" if ds == "DeHate" else "main"
        full, no = reads[fam]
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(g["y4"][i], np.int8) for i, v in enumerate(g["video_ids"]) if str(g["split"][i]) == "test"}
        asr = load_asr(ds, fill_untimed=ds == "DeHate")
        ids = sorted(v for v in gt if (ds, v) in full)
        aligned[ds] = {"n_gt": len(gt), "n_overlap": len(ids), "missing_ids": sorted(set(gt) - set(ids))}
        rows = []
        for vid in ids:
            key = (ds, vid)
            a, b = full[key], no[key]
            wa, wb = a["extra"]["windows"], b["extra"]["windows"]
            n = min(len(gt[vid]), len(a["score_curve"]), len(b["score_curve"]))
            if not args.raw_only:
                fa, fb = (f[key] for f in finals[fam])
                n = min(n, len(fa["score_curve"]), len(fb["score_curve"]))
            y = gt[vid][:n]
            pos = bool(y.any()); yes = a["extra"]["z_video"] > 0
            r = {"dataset": ds, "video_id": vid, "duration": a["duration"], "n_frames_eval": n,
                 "n_windows": len(wa), "positive_fraction": float(y.mean()), "global_z": a["extra"]["z_video"],
                 "verdict": "Yes" if yes else "No", "global_class": ("TP" if pos else "FP") if yes else ("FN" if pos else "TN"),
                 "mixed": bool(y.min() != y.max()), "n_gt_spans": len(binary_intervals(y))}
            if not args.raw_only:
                r["final_full"] = auc(y, fa["score_curve"][:n])
                r["final_no"] = auc(y, fb["score_curve"][:n])
                r["final_delta"] = r["final_full"] - r["final_no"] if r["final_full"] is not None else None
            for m in ("z", "z_visual", "z_speech"):
                ca, cb = curve(wa, m, n), curve(wb, m, n)
                aa = np.array([w.get(m, np.nan) for w in wa])
                bb = np.array([w.get(m, np.nan) for w in wb])
                good = np.isfinite(aa) & np.isfinite(bb)
                d = aa[good] - bb[good]
                da = ca - cb
                r[m + "_auc_full"], r[m + "_auc_no"] = auc(y, ca), auc(y, cb)
                r[m + "_auc_delta"] = r[m + "_auc_full"] - r[m + "_auc_no"] if r[m + "_auc_full"] is not None and r[m + "_auc_no"] is not None else None
                r[m + "_shift"] = mean(d)
                r[m + "_shift_positive"] = mean(da[y == 1])
                r[m + "_shift_negative"] = mean(da[y == 0])
                r[m + "_positive_minus_negative_shift"] = (r[m + "_shift_positive"] - r[m + "_shift_negative"]
                    if r[m + "_shift_positive"] is not None and r[m + "_shift_negative"] is not None else None)
                r[m + "_spearman"] = corr(aa, bb)
                r[m + "_signflip"] = float(np.mean((aa[good] > 0) != (bb[good] > 0))) if good.any() else None
                r[m + "_offset_energy_share"] = float(d.mean() ** 2 / np.mean(d ** 2)) if len(d) and np.mean(d ** 2) > 0 else None
            rows.append(r)
            for i, (x, z) in enumerate(zip(wa, wb)):
                mask = (np.arange(n) / 4 >= x["start"]) & (np.arange(n) / 4 < x["end"])
                wr = {"dataset": ds, "video_id": vid, "window": i, "start": x["start"], "end": x["end"],
                      "gt_fraction": float(y[mask].mean()) if mask.any() else None,
                      "text": window_text(asr.get(vid, []), x["start"], x["end"])}
                for m in ("z", "z_visual", "z_speech"):
                    wr[m + "_full"], wr[m + "_no"] = x.get(m), z.get(m)
                winrows.append(wr)
        allrows.extend(rows)
        mixed = [r for r in rows if r["mixed"]]
        summary[ds] = {"all": summarize(rows), "mixed": summarize(mixed), "by_global": {}, "by_coverage": {}}
        for group in ("TP", "FP", "FN", "TN"):
            summary[ds]["by_global"][group] = summarize([r for r in rows if r["global_class"] == group])
        for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1)):
            summary[ds]["by_coverage"][f"[{lo},{hi})"] = summarize([r for r in mixed if lo <= r["positive_fraction"] < hi])
        for name, choose in (("all", mixed), ("global_No", [r for r in mixed if r["verdict"] == "No"])):
            key = "z_auc_delta" if args.raw_only else "final_delta"
            ordered = sorted(choose, key=lambda r: (r[key], r["video_id"]))
            for sign, selected in (("loss", ordered[:3]), ("gain", ordered[-3:][::-1])):
                for rank, r in enumerate(selected):
                    vid = r["video_id"]
                    cases[f"{ds}_{name}_{sign}_{rank + 1}"] = {"stats": r, "gt_intervals": binary_intervals(gt[vid]),
                        "transcript": asr.get(vid, []),
                        "windows": [w for w in winrows if w["dataset"] == ds and w["video_id"] == vid]}
        print(ds, json.dumps(summary[ds]["all"]), flush=True)
    csvdump(out / "per_video.csv", allrows)
    csvdump(out / "per_window.csv", winrows)
    dump(out / "summary.json", summary)
    dump(out / "alignment.json", aligned)
    dump(out / "candidate_cases.json", cases)
    dump(out / "config.json", {"reads": READS, "finals": None if args.raw_only else FINALS,
         "gt_dir": "data/gt_4fps", "split": "test", "bootstrap": "10000 paired video resamples; seed 20261002",
         "scope": "exploratory, development-selected; no multiplicity correction", "host": socket.gethostname(),
         "code": "experiments/20261002_verdict_analysis/analyze.py; 2026-10-02"})
    print("ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
