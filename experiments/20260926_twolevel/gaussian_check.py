"""Post hoc check of the Gaussian evidence model (README §22).

Per corpus and modality: transform the window reads to normal scores of their corpus rank (as in twolevel_r2.py
--transform nscore), label each window by the test GT (hateful when at least half of its 4-fps frames are hateful),
and report mean, standard deviation, skewness and excess kurtosis of each state. Test labels are read only here,
for diagnosis; they enter no fitting.

    python experiments/20260926_twolevel/gaussian_check.py
"""
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"   # the reads of r6_bma (its config "run")
GT = ROOT / "data/gt_4fps"
OUT = ROOT / "runs/20260926_twolevel/gaussian_check"


def stats(x):
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()),
            "skew": float(skew(x)), "excess_kurtosis": float(kurtosis(x))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "run.log", "w")
    def say(s):
        print(s); log.write(s + "\n")
    say(f"host {socket.gethostname()}  date {datetime.now():%Y-%m-%d %H:%M}  reads {RUN}  gt {GT}")
    recs = [json.loads(l) for l in open(RUN)]
    out = {}
    for ds in ("HateMM", "HateClipSeg"):
        g = np.load(GT / f"{ds}.npz", allow_pickle=True)
        gt = dict(zip(g["video_ids"], g["y4"]))
        for m in ("z_visual", "z_speech"):
            ys, lab = [], []
            for r in recs:
                if r["dataset"] != ds or r.get("error") or r["video_id"] not in gt:
                    continue
                y4 = np.asarray(gt[r["video_id"]], float)
                for w in r["extra"]["windows"]:
                    if m not in w:
                        continue
                    seg = y4[int(round(w["start"] * 4)):int(round(w["end"] * 4))]
                    if len(seg):
                        ys.append(w[m]); lab.append(seg.mean() >= .5)
            ys, lab = np.array(ys), np.array(lab)
            u = norm.ppf((rankdata(ys) - 0.5) / len(ys))
            out[f"{ds}/{m}"] = {"raw": stats(ys), "u_nonhate": stats(u[~lab]), "u_hate": stats(u[lab]),
                                "hate_share": float(lab.mean())}
            say(f"{ds} {m}: n {len(ys)}, hate share {lab.mean():.2f}; raw skew {skew(ys):+.2f} "
                f"excess kurtosis {kurtosis(ys):+.2f}")
            for k in ("u_nonhate", "u_hate"):
                s = out[f"{ds}/{m}"][k]
                say(f"  {k:9s} mean {s['mean']:+.2f} sd {s['sd']:.2f} skew {s['skew']:+.2f} "
                    f"excess kurtosis {s['excess_kurtosis']:+.2f}")
    json.dump(out, open(OUT / "summary.json", "w"), indent=1)
    say(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
