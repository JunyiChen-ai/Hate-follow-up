#!/bin/bash
#SBATCH --job-name=hm_trainfit_2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hatemm_trainfit_2b_%j.out

set -euo pipefail

cd /data/jehc223/EMNLP2
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction

MODEL="Qwen/Qwen3-VL-2B-Instruct"
BS=32

python src/our_method/score_holistic_2b.py \
  --dataset HateMM \
  --split train \
  --mode binary \
  --model "$MODEL" \
  --model-slug 2b \
  --batch-size "$BS"

python - <<'PY'
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

ROOT = Path("/data/jehc223/EMNLP2")
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

from grid_eval_all import SKIP_VIDEOS, judge_path, ld_jsonl, load_labels  # noqa: E402
from quick_eval_all import load_scores_file  # noqa: E402
from thresholds import gmm_threshold, li_lee_threshold, otsu_threshold  # noqa: E402

DS = "HateMM"
ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")
CRITS = {
    "gmm": gmm_threshold,
    "otsu": otsu_threshold,
    "li_lee": li_lee_threshold,
}
TRAIN_PATH = ROOT / "results" / "holistic_2b" / DS / "train_binary.jsonl"
TEST_PATH = ROOT / "results" / "holistic_2b" / DS / "test_binary.jsonl"
OUT = ROOT / "results" / "boundary_rescue" / DS / "hatemm_trainfit_full_pipeline.csv"
OUT_JSON = ROOT / "results" / "boundary_rescue" / DS / "hatemm_trainfit_full_pipeline.json"


def ent(p):
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def logit(p):
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def rho_from_hbar(hbar):
    lo, hi = 0.5, 1 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        if ent(mid) > hbar:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def to_logit_arr(scores):
    scores = np.clip(np.array(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(scores / (1 - scores)).reshape(-1, 1)


def ordered_score_rows(path):
    scores = load_scores_file(path)
    rows, seen = [], set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            if vid in scores and vid not in seen:
                seen.add(vid)
                rows.append((vid, float(scores[vid])))
    return rows


def macro(y, yh):
    acc = sum(a == b for a, b in zip(y, yh)) / len(y)
    fs, ps, rs = [], [], []
    for c in (0, 1):
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return acc, sum(fs) / 2, sum(ps) / 2, sum(rs) / 2


def eval_trainfit(crit):
    train_path = TRAIN_PATH
    test_path = TEST_PATH
    if not os.path.isfile(train_path):
        raise FileNotFoundError(train_path)
    fit_scores = np.array(list(load_scores_file(train_path).values()), dtype=float)
    test_rows = ordered_score_rows(test_path)
    test_scores = np.array([s for _, s in test_rows], dtype=float)

    threshold = CRITS[crit](fit_scores)
    base = {vid: int(score >= threshold) for vid, score in test_rows}

    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200).fit(to_logit_arr(fit_scores))
    hi = int(np.argmax(gmm.means_.flatten()))
    post = gmm.predict_proba(to_logit_arr(test_scores))[:, hi]
    entropies = np.array([ent(p) for p in post])
    hbar = float(entropies.mean())
    rho = rho_from_hbar(hbar)
    lam = math.log(rho / (1 - rho))
    band = {
        vid: {"posterior_hi": float(post[i]), "in_band": bool(entropies[i] > hbar)}
        for i, (vid, _) in enumerate(test_rows)
    }
    judges = [{r["video_id"]: r for r in ld_jsonl(judge_path(j, DS))} for j in ORDER]
    labels = load_labels(DS)
    skip = SKIP_VIDEOS.get(DS, set())
    valid = [v for v in base if v not in skip and labels.get(v) in (0, 1)]

    y, yh = [], []
    calls, band_n = 0, 0
    for v in valid:
        y.append(labels[v])
        b = band.get(v, {"in_band": False})
        if not b["in_band"]:
            yh.append(base[v])
            continue
        band_n += 1
        ell = logit(b["posterior_hi"])
        for table in judges:
            pred = table.get(v, {}).get("pred")
            if pred in (0, 1):
                calls += 1
                ell += (2 * int(pred) - 1) * lam
                if ent(sigmoid(ell)) <= hbar:
                    break
        yh.append(1 if sigmoid(ell) >= 0.5 else 0)

    acc, mf1, mp, mr = macro(y, yh)
    return {
        "dataset": DS,
        "criterion": crit,
        "source": "train set",
        "threshold": float(threshold),
        "n_fit": int(len(fit_scores)),
        "n_eval": int(len(valid)),
        "band_n": int(band_n),
        "calls_per_video": float(calls / len(valid)),
        "acc": float(acc),
        "macro_f1": float(mf1),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
    }


rows = [eval_trainfit(c) for c in ("gmm", "otsu", "li_lee")]
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
with open(OUT_JSON, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Wrote {OUT}")
for r in rows:
    print(
        f"{r['criterion']:<6} {r['source']:<9} "
        f"ACC={r['acc']*100:.1f} M-F1={r['macro_f1']*100:.1f} "
        f"M-P={r['macro_precision']*100:.1f} M-R={r['macro_recall']*100:.1f} "
        f"threshold={r['threshold']:.6f} band={r['band_n']} calls={r['calls_per_video']:.3f}"
    )
PY
