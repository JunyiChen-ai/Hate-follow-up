"""Back half of the label-free pipeline: entropy-routed sequential verification.

Input  : 2B Stage-1 score files (results/boundary_rescue/<DS>/baseline_preds_v2.jsonl)
         plus the per-dataset entropy-band file and the three offline judge files
         (gemma-3-27b-it, qwen2.5-vl-32b-awq, qwen2.5-vl-72b-awq).

Output : ACC, Macro F1, Macro Precision, Macro Recall per dataset and averaged
         over the four datasets.

Pipeline (label-free, single rho per dataset):
  1. Hbar_D = mean H(p) over the unlabeled Stage-1 GMM posteriors of dataset D.
  2. rho_D  = upper solution of H(rho_D) = Hbar_D.  (no labels used)
  3. For each video v:
       p   = posterior_hi from Stage-1.
       if H(p) <= Hbar_D    -> keep Stage-1 label.
       else                  -> sequentially call g27 > q32 > q72; after each
                                verdict update logit(p) += sign(verdict)*log(rho_D/(1-rho_D)),
                                stop as soon as H(p') <= Hbar_D.
       final = 1 if p >= 0.5 else 0.

Usage: python src/our_method/eval_sequential_back_half.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3")
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

from data_utils import SKIP_VIDEOS  # noqa: E402
from grid_eval_all import (  # noqa: E402
    DS,
    N_TEST,
    baseline_pred_path,
    entropy_band_path,
    judge_path,
    ld_jsonl,
    load_labels,
)

ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")
SLUG = "2b"


def H(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rho_from_hbar(hbar: float) -> float:
    """Upper root of H(rho) = hbar on (0.5, 1)."""
    lo, hi = 0.5, 1 - 1e-12
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if H(mid) > hbar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def macro_prf(y: list[int], yh: list[int]) -> tuple[float, float, float]:
    classes = sorted(set(y))
    fs, ps, rs = [], [], []
    for c in classes:
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return sum(fs) / len(fs), sum(ps) / len(ps), sum(rs) / len(rs)


def eval_dataset(ds: str) -> dict:
    labels = load_labels(ds)
    base = {r["video_id"]: int(r["pred_baseline"])
            for r in ld_jsonl(baseline_pred_path(SLUG, ds, "protocol"))}
    band_rows = ld_jsonl(entropy_band_path(SLUG, ds))
    band = {r["video_id"]: r for r in band_rows}
    hbar = sum(float(r["entropy"]) for r in band_rows) / len(band_rows)
    rho = rho_from_hbar(hbar)
    lam = math.log(rho / (1 - rho))
    judges = [{r["video_id"]: r for r in ld_jsonl(judge_path(j, ds))}
              for j in ORDER]

    skip = SKIP_VIDEOS.get(ds, set())
    vids = [v for v in base if v not in skip and labels.get(v) in (0, 1)]

    y, yh = [], []
    for v in vids:
        y.append(labels[v])
        s1 = base[v]
        b = band.get(v, {})
        if not bool(b.get("in_band")):
            yh.append(s1)
            continue
        ell = logit(float(b.get("posterior_hi", 0.5)))
        for table in judges:
            r = table.get(v, {}).get("pred")
            if r in (0, 1):
                ell += (2 * int(r) - 1) * lam
                if H(sigmoid(ell)) <= hbar:
                    break
        yh.append(1 if sigmoid(ell) >= 0.5 else 0)

    acc = sum(1 for a, b in zip(y, yh) if a == b) / N_TEST[ds]
    mf1, mp, mr = macro_prf(y, yh)
    return {"ds": ds, "hbar": hbar, "rho_D": rho,
            "acc": acc, "mf1": mf1, "mp": mp, "mr": mr}


def main() -> None:
    rows = [eval_dataset(ds) for ds in DS]
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in ("acc", "mf1", "mp", "mr")}

    w = 14
    print(f"{'Dataset':<{w}} {'rho_D':>7} {'ACC':>7} {'MacroF1':>8} "
          f"{'MacroP':>8} {'MacroR':>8}")
    print("-" * (w + 7 + 7 + 8 + 8 + 8 + 5))
    for r in rows:
        print(f"{r['ds']:<{w}} {r['rho_D']:>7.3f} "
              f"{100*r['acc']:>7.2f} {r['mf1']:>8.4f} "
              f"{r['mp']:>8.4f} {r['mr']:>8.4f}")
    print("-" * (w + 7 + 7 + 8 + 8 + 8 + 5))
    print(f"{'Average':<{w}} {'-':>7} "
          f"{100*avg['acc']:>7.2f} {avg['mf1']:>8.4f} "
          f"{avg['mp']:>8.4f} {avg['mr']:>8.4f}")


if __name__ == "__main__":
    main()
