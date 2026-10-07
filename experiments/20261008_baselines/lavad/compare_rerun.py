#!/usr/bin/env python3
"""Compare the LAVAD rerun of HateClipSeg videos with the Retrieval-hate campaign's stored outputs (uoa-lab2).

run_plan.md L1b: old outputs may be mixed with rerun rows only if the ported code reproduces the old scores on
5 overlapping videos. For each video in `--ids-file` this compares, stage by stage, the rerun
(`data/blip2_captions_1fps/`, `runs/20261008_baselines/lavad/work/`, `.../curves/`) with the campaign's files
(`~/Retrieval-hate/data/captions/blip2_1fps/`, `~/Retrieval-hate/data/lavad/`,
`~/Retrieval-hate/archive/idea-stage/repro_lavad/curves/`, all read only), and writes
`runs/20261008_baselines/lavad/HateClipSeg_rerun_check.json`. `reproduced` is true when every overlap video
(all ids except the one being completed) has the same NaN pattern in `base` and max |diff| <= 1e-6.
No labels are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RH = Path.home() / "Retrieval-hate"
RUN_DIR = REPO / "runs/20261008_baselines/lavad"
DS = "HateClipSeg"
TOL = 1e-6


def same_json(a: Path, b: Path):
    if not a.exists() or not b.exists():
        return None
    ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
    if ja == jb:
        return {"identical": True}
    keys = set(ja) | set(jb)
    diff = sorted(k for k in keys if ja.get(k) != jb.get(k))
    return {"identical": False, "n_keys": len(keys), "n_diff": len(diff), "first_diff": diff[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--target", default="bit_ZaY0S1anrdep")
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    res, ok = {}, True
    for v in ids:
        r = {"captions": same_json(REPO / f"data/blip2_captions_1fps/{DS}/{v}.json",
                                   RH / f"data/captions/blip2_1fps/{DS}/{v}.json")}
        for stage in ("clean", "summary", "score", "refined"):
            r[stage] = same_json(RUN_DIR / f"work/{stage}/{DS}/{v}.json", RH / f"data/lavad/{stage}/{DS}/{v}.json")
        pa, pb = RUN_DIR / f"curves/{DS}/{v}.npz", RH / f"archive/idea-stage/repro_lavad/curves/{DS}/{v}.npz"
        if pa.exists() and pb.exists():
            ca, cb = np.load(pa)["base"], np.load(pb)["base"]
            same_len = len(ca) == len(cb)
            n = min(len(ca), len(cb))
            nan_same = bool(np.array_equal(np.isnan(ca[:n]), np.isnan(cb[:n])))
            fin = np.isfinite(ca[:n]) & np.isfinite(cb[:n])
            md = float(np.abs(ca[:n][fin] - cb[:n][fin]).max()) if fin.any() else 0.0
            r["base_curve"] = {"len_rerun": int(len(ca)), "len_campaign": int(len(cb)), "same_nan_pattern": nan_same,
                               "max_abs_diff": md, "nan_rerun": int(np.isnan(ca).sum()),
                               "nan_campaign": int(np.isnan(cb).sum())}
            good = same_len and nan_same and md <= TOL
        else:
            r["base_curve"] = None
            good = False
        r["reproduced"] = good
        if v != a.target:
            ok &= good
        res[v] = r
        print(v, json.dumps(r), flush=True)
    n_overlap = sum(v != a.target for v in ids)
    out = {"target": a.target, "overlap_videos": n_overlap, "tolerance": TOL,
           "reproduced": bool(ok and n_overlap >= 5), "videos": res}
    (RUN_DIR / f"{DS}_rerun_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print("REPRODUCED" if out["reproduced"] else "NOT REPRODUCED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
