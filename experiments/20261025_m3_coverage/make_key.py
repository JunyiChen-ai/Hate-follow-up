#!/usr/bin/env python3
"""Per-video key file for the r6_bma decoder (CPU; no label file read).

experiments/20260926_twolevel/twolevel_r2.py takes extra video-level reads through --dvd-reads / --dvd-conds: with
--key calib --dvd-conds T the video term becomes logit P_T(x), where x is the field z_target of the record and P_T is
key_calibration (the label-free two-Gaussian calibration r6 applies to K) fitted on x per corpus; the time level (EM,
M2 posteriors, centred rank) does not read these records. So writing K' into z_target and passing --dvd-conds T gives
the r6_bma decoder with the key K replaced by K' and M2 untouched; --dvd-conds viol,T gives the noisy-AND of the r6 key
and P_T(K'). This script writes that file.

  --form r6           K' = z_video + mean window z (the r6 key itself; plumbing check: must reproduce r6_bma)
  --form cv           K' = c_v
  --form zv_plus_cv   K' = z_video + c_v              (window mean replaced by c_v)
  --form r6_plus_cv   K' = z_video + mean window z + c_v
z_video and the window z come from the reads the decoder runs on (--run, default the r6 reads base_gridA). c_v comes
from --values (jsonl: dataset, video_id, value; the format of common.write_values) or from a read_extra.py output:
  --reads R --read-mode stance|direct --reduce expect --option-values 0 .05 .17 .37 .62 .87   (sum_k p_k v_k)
  --reads R --read-mode stance|direct --reduce logodds --pos D E F                            (log P(pos) - log P(rest))

  python experiments/20261025_m3_coverage/make_key.py --form r6 --out runs/20261025_m3_coverage/keys/r6_key.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def window_reads(run):
    out = {}
    for line in open(Path(run) / "predictions.jsonl"):
        r = json.loads(line)
        if r.get("error"):
            continue
        z = [float(w["z"]) for w in r["extra"]["windows"]]
        out[(r["dataset"], r["video_id"])] = (float(r["extra"]["z_video"]), float(np.mean(z)) if z else 0.0)
    return out


def cv_from_reads(path, mode, reduce, option_values, pos):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            continue
        lp = r["reads"][mode]["logp"]
        labs = r["options"]
        if reduce == "expect":
            if len(option_values) != len(labs):
                raise SystemExit(f"--option-values has {len(option_values)} values for {len(labs)} options")
            x = sum(math.exp(lp[l]) * v for l, v in zip(labs, option_values))
        else:
            if not pos or any(p not in labs for p in pos) or set(pos) == set(labs):
                raise SystemExit(f"--pos {pos} must be a proper subset of the options {labs}")
            a = np.logaddexp.reduce([lp[l] for l in labs if l in pos])
            b = np.logaddexp.reduce([lp[l] for l in labs if l not in pos])
            x = float(a - b)
        out[(r["dataset"], r["video_id"])] = float(x)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--form", choices=["r6", "cv", "zv_plus_cv", "r6_plus_cv"], required=True)
    ap.add_argument("--run", default=str(ROOT / "runs/20260926_glr/base_gridA"), help="reads the decoder runs on")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--values", help="jsonl with dataset, video_id, value (c_v)")
    ap.add_argument("--reads", help="read_extra.py reads.jsonl (c_v reduced from the option distribution)")
    ap.add_argument("--read-mode", choices=["stance", "direct"], default="stance")
    ap.add_argument("--reduce", choices=["expect", "logodds"], default="expect")
    ap.add_argument("--option-values", nargs="*", type=float, default=[])
    ap.add_argument("--pos", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = {k: v for k, v in window_reads(a.run).items() if k[0] in a.datasets}
    cv = {}
    if a.form != "r6":
        if bool(a.values) == bool(a.reads):
            raise SystemExit("give exactly one of --values / --reads for this form")
        if a.values:
            for line in open(a.values):
                if line.strip():
                    r = json.loads(line)
                    cv[(r["dataset"], r["video_id"])] = float(r["value"])
        else:
            cv = cv_from_reads(a.reads, a.read_mode, a.reduce, a.option_values, a.pos)
        missing = [k for k in base if k not in cv]
        if missing:
            raise SystemExit(f"c_v missing for {len(missing)} of {len(base)} videos, e.g. {missing[:3]}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for (ds, vid), (zv, wm) in sorted(base.items()):
            c = cv.get((ds, vid))
            k = {"r6": zv + wm, "cv": c, "zv_plus_cv": None if c is None else zv + c,
                 "r6_plus_cv": None if c is None else zv + wm + c}[a.form]
            fh.write(json.dumps({"dataset": ds, "video_id": vid, "error": None, "z_target": float(k), "form": a.form,
                                 "z_video": zv, "window_mean": wm, "c_v": c}) + "\n")
    print(f"wrote {len(base)} keys ({a.form}) to {out}")


if __name__ == "__main__":
    main()
