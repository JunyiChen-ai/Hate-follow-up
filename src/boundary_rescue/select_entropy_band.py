"""Entropy-above-mean band per (stage-1 MLLM, dataset).

Fits a 2-component GMM in logit space on the stage-1 MLLM's test
scores (re-using `select_bayes_band.fit_gmm`), derives per-sample
`posterior_hi`, computes binary entropy `H_i = -p·log(p) -
(1-p)·log(1-p)`, and writes the set of test video_ids whose entropy
exceeds the mean entropy across test.

Output: results/boundary_rescue/<ds>/candidates_entropy_band_<slug>.jsonl

Each line:
  {video_id, score, logit, posterior_hi, entropy, in_band,
   threshold, pred_baseline, side, fit_source}

Usage:
  python src/boundary_rescue/select_entropy_band.py --model-tag 2b
  python src/boundary_rescue/select_entropy_band.py --model-tag qwen2.5-vl-7b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "our_method"))
sys.path.insert(0, str(_HERE.parent / "naive_baseline"))

from quick_eval_all import load_scores_file  # noqa: E402
from data_utils import SKIP_VIDEOS  # noqa: E402
from select_bayes_band import to_logit, fit_gmm  # noqa: E402

PROJECT_ROOT = Path("/data/jehc223/EMNLP3")
OUT_ROOT = PROJECT_ROOT / "results" / "boundary_rescue"
DS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]


def _ent(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def baseline_path(slug, ds, criterion):
    if slug == "2b" and criterion == "protocol":
        return OUT_ROOT / ds / "baseline_preds_v2.jsonl"
    suf = f"_{slug}_{criterion}"
    return OUT_ROOT / ds / f"baseline_preds_v2{suf}.jsonl"


def v2_baseline_path(slug, criterion):
    if slug == "2b" and criterion == "protocol":
        return OUT_ROOT / "v2_baseline.json"
    suf = f"_{slug}_{criterion}"
    return OUT_ROOT / f"v2_baseline{suf}.json"


def run_one(slug, ds, criterion):
    bp = baseline_path(slug, ds, criterion)
    if not bp.exists():
        print(f"[skip] {slug}/{ds}/{criterion}: {bp} missing")
        return None
    base_rows = [json.loads(l) for l in open(bp) if l.strip()]
    vb = json.load(open(v2_baseline_path(slug, criterion)))
    info = vb[ds]
    fit_scores = np.array(list(load_scores_file(info["fit_path"]).values()), dtype=float)
    gmm, hi = fit_gmm(fit_scores)
    test_scores = np.array([r["score"] for r in base_rows], dtype=float)
    z = to_logit(test_scores).reshape(-1, 1)
    post = gmm.predict_proba(z)[:, hi]
    ent = np.array([_ent(p) for p in post])
    mean_e = float(ent.mean())
    skip = SKIP_VIDEOS.get(ds, set())

    records = []
    in_band_count = 0
    for i, r in enumerate(base_rows):
        vid = r["video_id"]
        if vid in skip:
            continue
        e = float(ent[i])
        in_band = bool(e > mean_e)
        if in_band:
            in_band_count += 1
        records.append({
            "video_id": vid,
            "score": float(r["score"]),
            "logit": float(z[i, 0]),
            "posterior_hi": float(post[i]),
            "entropy": e,
            "in_band": in_band,
            "threshold": float(info["threshold"]),
            "pred_baseline": int(r["pred_baseline"]),
            "side": "above" if r["pred_baseline"] == 1 else "below",
            "fit_source": info["fit_source"],
            "mean_entropy_dataset": mean_e,
        })
    out = OUT_ROOT / ds / f"candidates_entropy_band_{slug}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "slug": slug, "ds": ds, "criterion": criterion,
        "n_test": len(records), "n_band": in_band_count,
        "mean_entropy": mean_e, "out_path": str(out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="2b",
                    help="stage-1 slug; reads baseline_preds_v2*_<slug>_*.jsonl")
    ap.add_argument("--criterion", default="protocol",
                    choices=["protocol", "otsu", "gmm", "li_lee"],
                    help="Match the criterion used in Phase C "
                         "(default protocol keeps V1 semantics).")
    args = ap.parse_args()

    print(f"{'slug':<20} {'ds':<14} {'crit':<10} {'n_band':>7} {'/':>2} {'n_test':>6}  mean_H  out")
    print("-" * 100)
    for ds in DS:
        r = run_one(args.model_tag, ds, args.criterion)
        if r is None:
            continue
        print(f"{r['slug']:<20} {r['ds']:<14} {r['criterion']:<10} "
              f"{r['n_band']:>7} / {r['n_test']:>6}  {r['mean_entropy']:.4f}  "
              f"{r['out_path']}")


if __name__ == "__main__":
    main()
