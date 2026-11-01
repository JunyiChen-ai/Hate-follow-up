#!/usr/bin/env python3
"""Reading cost of the adopted reader from the stored read logs (README §3, task 7). Reads per-video records only
(no GT). Writes runs/20261101_paper_redecode/cost/cost.json and prints a table.

- DeHate: runs/20261031_window_evidence/DeHate/records/DeHate/*.json, field `seconds` (one pass of
  experiments/20261031_window_evidence/read.py that computes all four read sets r6_replay / frames / words / adopted:
  per window the native visual, local visual, proportional speech and aligned speech branches). An upper bound for the
  adopted reader alone.
- HateMM / HateClipSeg: the adopted reads were composed from two runs, so no single-pass time exists:
  * runs/20261007_m1_streamingtom/placement_controls_main/records/<ds>/*.json, `checks.times` (per component,
    sc474399): prefix (prefix + verdict turn), native_visual, native_speech (r6's two window branches),
    adjacent_local (window branches with in-window frames next to the question). Estimate of the adopted reader =
    prefix + adjacent_local + native_speech (the aligned speech branch is one text branch per window, like the
    proportional one); r6 = prefix + native_visual + native_speech, from the same records.
  * runs/20261014_m1_audio_bound/vl_aligned/records/<ds>/*.json, `seconds` (sc474397): prefix + aligned speech branches.
- Prefix tokens: extra.prefix_tokens of the adopted predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/20261101_paper_redecode/cost"


def stats(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "sum_min": float(x.sum() / 60)}


def main():
    res = {}
    # DeHate, one pass with all four read sets
    recs = [json.load(open(p)) for p in sorted((ROOT / "runs/20261031_window_evidence/DeHate/records/DeHate").glob("*.json"))]
    res["DeHate"] = {"read_seconds_all_four_sets": stats([r["seconds"] for r in recs]),
                     "hosts": sorted({r["host"] for r in recs}),
                     "windows": stats([r["windows"] for r in recs]),
                     "peak_GiB_max": float(max(r["peak_GiB"] for r in recs))}
    pred = ROOT / "runs/20261031_window_evidence/DeHate/adopted/predictions.jsonl"
    res["DeHate"]["prefix_tokens"] = stats([json.loads(l)["extra"]["prefix_tokens"] for l in open(pred)])
    adopted_main = ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl"
    pt = {}
    for l in open(adopted_main):
        r = json.loads(l); pt.setdefault(r["dataset"], []).append(r["extra"]["prefix_tokens"])
    for ds in ("HateMM", "HateClipSeg"):
        pl = [json.load(open(p)) for p in sorted((ROOT / f"runs/20261007_m1_streamingtom/placement_controls_main/records/{ds}").glob("*.json"))]
        t = [r["checks"]["times"] for r in pl]
        va = [json.load(open(p)) for p in sorted((ROOT / f"runs/20261014_m1_audio_bound/vl_aligned/records/{ds}").glob("*.json"))]
        res[ds] = {
            "placement_hosts": sorted({r["checks"]["host"] for r in pl}),
            "prefix_seconds": stats([x["prefix"] for x in t]),
            "r6_native_visual_seconds": stats([x["native_visual"] for x in t]),
            "r6_native_speech_seconds": stats([x["native_speech"] for x in t]),
            "local_visual_seconds": stats([x["adjacent_local"] for x in t]),
            "adopted_estimate_seconds": stats([x["prefix"] + x["adjacent_local"] + x["native_speech"] for x in t]),
            "r6_same_records_seconds": stats([x["prefix"] + x["native_visual"] + x["native_speech"] for x in t]),
            "adjacent_local_standalone_seconds": stats([r["predictions"]["adjacent_local"]["extra"]["standalone_seconds"] for r in pl]),
            "vl_aligned_hosts": sorted({r["host"] for r in va}),
            "vl_aligned_prefix_plus_aligned_speech_seconds": stats([r["seconds"] for r in va]),
            "windows": stats([r["windows"] for r in va]),
            "prefix_tokens": stats(pt[ds]),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost.json").write_text(json.dumps(res, indent=2))
    for ds, d in res.items():
        print(f"== {ds}")
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"  {k:48s} n {v['n']:5d}  mean {v['mean']:8.3f}  median {v['median']:8.3f}  p10 {v['p10']:8.3f}  "
                      f"p90 {v['p90']:8.3f}  total {v['sum_min']:7.1f} min")
            else:
                print(f"  {k:48s} {v}")


if __name__ == "__main__":
    main()
