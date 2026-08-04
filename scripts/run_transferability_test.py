#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
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


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
CRITERIA = {
    "gmm": gmm_threshold,
    "otsu": otsu_threshold,
    "li_lee": li_lee_threshold,
}
ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")
OUT = ROOT / "results" / "boundary_rescue" / "transferability_full_pipeline_2b.csv"
BAND_OUT = ROOT / "results" / "boundary_rescue" / "transferability_band_membership_2b.jsonl"


def train_path(dataset: str) -> Path:
    return ROOT / "results" / "holistic_2b" / dataset / "train_binary.jsonl"


def test_path(dataset: str) -> Path:
    if dataset == "MHClip_ZH":
        prerepro = ROOT / "results" / "holistic_2b" / dataset / "test_binary.jsonl.prerepro_20260413"
        if prerepro.exists():
            return prerepro
    return ROOT / "results" / "holistic_2b" / dataset / "test_binary.jsonl"


def entropy(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def rho_from_hbar(hbar: float) -> float:
    lo, hi = 0.5, 1 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        if entropy(mid) > hbar:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def to_logit_arr(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.array(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(scores / (1 - scores)).reshape(-1, 1)


def ordered_score_rows(path: Path) -> list[tuple[str, float]]:
    scores = load_scores_file(path)
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
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


def full_metrics(y: list[int], yh: list[int]) -> dict[str, float]:
    acc = sum(a == b for a, b in zip(y, yh)) / len(y)
    fs, ps, rs = [], [], []
    for c in (0, 1):
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return {"acc": acc, "mf1": sum(fs) / 2, "mp": sum(ps) / 2, "mr": sum(rs) / 2}


def eval_transfer(source: str, target: str, criterion: str) -> dict[str, object]:
    source_scores = np.array(list(load_scores_file(train_path(source)).values()), dtype=float)
    target_rows = ordered_score_rows(test_path(target))
    target_scores = np.array([score for _, score in target_rows], dtype=float)

    threshold = CRITERIA[criterion](source_scores)
    base = {vid: int(score >= threshold) for vid, score in target_rows}

    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200).fit(to_logit_arr(source_scores))
    high_component = int(np.argmax(gmm.means_.flatten()))
    post = gmm.predict_proba(to_logit_arr(target_scores))[:, high_component]
    entropies = np.array([entropy(p) for p in post])
    hbar = float(entropies.mean())
    rho = rho_from_hbar(hbar)
    lam = math.log(rho / (1 - rho))
    band = {
        vid: {"posterior_hi": float(post[i]), "in_band": bool(entropies[i] > hbar)}
        for i, (vid, _) in enumerate(target_rows)
    }

    judges = [{r["video_id"]: r for r in ld_jsonl(judge_path(j, target))} for j in ORDER]
    labels = load_labels(target)
    skip = SKIP_VIDEOS.get(target, set())
    valid = [v for v in base if v not in skip and labels.get(v) in (0, 1)]

    y, yh_stage1, yh_full = [], [], []
    calls = 0
    band_n = 0
    band_video_ids: list[str] = []
    flipped = 0
    rescued = 0
    harmed = 0
    for v in valid:
        y.append(labels[v])
        s1 = base[v]
        yh_stage1.append(s1)
        b = band.get(v, {"in_band": False})
        if not b["in_band"]:
            yh_full.append(s1)
            continue
        band_n += 1
        band_video_ids.append(v)
        ell = logit(b["posterior_hi"])
        for table in judges:
            pred = table.get(v, {}).get("pred")
            if pred in (0, 1):
                calls += 1
                ell += (2 * int(pred) - 1) * lam
                if entropy(sigmoid(ell)) <= hbar:
                    break
        final = 1 if sigmoid(ell) >= 0.5 else 0
        if final != s1:
            flipped += 1
            if final == labels[v] and s1 != labels[v]:
                rescued += 1
            if final != labels[v] and s1 == labels[v]:
                harmed += 1
        yh_full.append(final)

    s1 = full_metrics(y, yh_stage1)
    full = full_metrics(y, yh_full)
    return {
        "criterion": criterion,
        "source": source,
        "target": target,
        "is_diagonal": int(source == target),
        "threshold": float(threshold),
        "n_fit": int(len(source_scores)),
        "n_target": int(len(valid)),
        "band_n": int(band_n),
        "band_rate": float(band_n / len(valid)),
        "calls_per_video": float(calls / len(valid)),
        "flipped": int(flipped),
        "rescued": int(rescued),
        "harmed": int(harmed),
        "stage1_acc": s1["acc"],
        "stage1_mf1": s1["mf1"],
        "stage1_mp": s1["mp"],
        "stage1_mr": s1["mr"],
        "full_acc": full["acc"],
        "full_mf1": full["mf1"],
        "full_mp": full["mp"],
        "full_mr": full["mr"],
        "band_video_ids": band_video_ids,
    }


def main() -> None:
    rows = []
    for criterion in CRITERIA:
        for source in DATASETS:
            for target in DATASETS:
                rows.append(eval_transfer(source, target, criterion))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        metric_rows = [{k: v for k, v in r.items() if k != "band_video_ids"} for r in rows]
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    with open(BAND_OUT, "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "criterion": r["criterion"],
                        "source": r["source"],
                        "target": r["target"],
                        "threshold": r["threshold"],
                        "band_n": r["band_n"],
                        "band_video_ids": r["band_video_ids"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {OUT}")
    print(f"Wrote {BAND_OUT}")
    for criterion in CRITERIA:
        off = [r for r in rows if r["criterion"] == criterion and not r["is_diagonal"]]
        diag = [r for r in rows if r["criterion"] == criterion and r["is_diagonal"]]
        print(
            f"{criterion:<6} offdiag full ACC={np.mean([r['full_acc'] for r in off])*100:.1f} "
            f"M-F1={np.mean([r['full_mf1'] for r in off])*100:.1f}; "
            f"diag full ACC={np.mean([r['full_acc'] for r in diag])*100:.1f} "
            f"M-F1={np.mean([r['full_mf1'] for r in diag])*100:.1f}"
        )


if __name__ == "__main__":
    main()
