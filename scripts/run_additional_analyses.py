from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/data/jehc223/EMNLP2")
OUT_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "paper" / "analysis"

sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))

import build_experiments as exp  # noqa: E402


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
DATASET_LABELS = {
    "HateMM": "HateMM",
    "MHClip_EN": "MHClip-EN",
    "MHClip_ZH": "MHClip-ZH",
    "ImpliHateVid": "ImpliHateVid",
}
MAIN_ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")


def pct(x: float) -> float:
    return 100.0 * x


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def logit_scalar(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def entropy_scalar(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def macro_pos_neg(y: list[int], pred: list[int]) -> dict[str, float]:
    out = {}
    for c, name in ((1, "hateful"), (0, "normal")):
        tp = sum(1 for a, b in zip(y, pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, pred) if a == c and b != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
    return out


def calibration_metrics(y: list[int], prob: list[float], bins: int = 10) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y_arr) ** 2))
    nll = float(-np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y_arr[mask]))
        ece += float(np.mean(mask)) * abs(conf - acc)
    return {"brier": brier, "nll": nll, "ece": ece}


def fit_gmm_1d(x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    best = None
    best_ll = -1e300
    for _ in range(4):
        qs = np.quantile(x, [0.25, 0.75])
        means = qs + rng.normal(0, max(np.std(x), 1e-3) * 0.05, size=2)
        vars_ = np.array([np.var(x), np.var(x)], dtype=float) + 1e-4
        weights = np.array([0.5, 0.5], dtype=float)
        for _ in range(100):
            logp = []
            for k in range(2):
                lp = (
                    math.log(max(weights[k], 1e-12))
                    - 0.5 * math.log(2 * math.pi * vars_[k])
                    - 0.5 * ((x - means[k]) ** 2) / vars_[k]
                )
                logp.append(lp)
            logp = np.vstack(logp).T
            m = np.max(logp, axis=1, keepdims=True)
            resp = np.exp(logp - m)
            resp = resp / np.sum(resp, axis=1, keepdims=True)
            nk = np.sum(resp, axis=0) + 1e-8
            weights = nk / len(x)
            means = np.sum(resp * x[:, None], axis=0) / nk
            vars_ = np.sum(resp * (x[:, None] - means) ** 2, axis=0) / nk
            vars_ = np.maximum(vars_, 1e-4)
        logp = []
        for k in range(2):
            lp = (
                math.log(max(weights[k], 1e-12))
                - 0.5 * math.log(2 * math.pi * vars_[k])
                - 0.5 * ((x - means[k]) ** 2) / vars_[k]
            )
            logp.append(lp)
        logp = np.vstack(logp).T
        ll = float(np.sum(np.max(logp, axis=1) + np.log(np.sum(np.exp(logp - np.max(logp, axis=1, keepdims=True)), axis=1))))
        if ll > best_ll:
            best_ll = ll
            best = (weights.copy(), means.copy(), vars_.copy())
    assert best is not None
    weights, means, vars_ = best
    order = np.argsort(means)
    return weights[order], means[order], vars_[order]


def posterior_hi(z: np.ndarray, weights: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    logp = []
    for k in range(2):
        lp = (
            math.log(max(float(weights[k]), 1e-12))
            - 0.5 * math.log(2 * math.pi * float(vars_[k]))
            - 0.5 * ((z - float(means[k])) ** 2) / float(vars_[k])
        )
        logp.append(lp)
    logp = np.vstack(logp).T
    m = np.max(logp, axis=1, keepdims=True)
    resp = np.exp(logp - m)
    resp = resp / np.sum(resp, axis=1, keepdims=True)
    hi = int(np.argmax(means))
    return resp[:, hi]


def final_with_custom_band(
    cache: exp.EvalCache,
    ds: str,
    posterior_by_vid: dict[str, float],
    in_band_by_vid: dict[str, bool],
    hbar: float,
) -> tuple[float, float, float]:
    labels = cache.load_labels_for(ds)
    base = cache.load_base("2b", ds)
    judges = [cache.load_judge(j, ds) for j in MAIN_ORDER]
    rho = exp.rho_from_hbar(hbar)
    lam = math.log(rho / (1 - rho))
    y, pred = [], []
    calls = 0
    for vid in cache.valid_vids("2b", ds):
        y.append(labels[vid])
        p = posterior_by_vid.get(vid, 0.5)
        if not in_band_by_vid.get(vid, False):
            pred.append(base[vid])
            continue
        ell = logit_scalar(p)
        for table in judges:
            calls += 1
            r = table.get(vid, {}).get("pred")
            if r in (0, 1):
                ell += (2 * int(r) - 1) * lam
                if entropy_scalar(1 / (1 + math.exp(-ell))) <= hbar:
                    break
        pred.append(1 if ell >= 0 else 0)
    acc = sum(1 for a, b in zip(y, pred) if a == b) / len(y)
    mf1, _, _ = exp.macro_prf(y, pred)
    return acc, mf1, calls / len(y)


def run_gmm_stability(cache: exp.EvalCache) -> tuple[list[dict], list[dict]]:
    rows = []
    summary = []
    rng = random.Random(17)
    for ds in DATASETS:
        labels = cache.load_labels_for(ds)
        base = cache.load_base("2b", ds)
        valid = cache.valid_vids("2b", ds)
        score_by_vid = {}
        for vid, row in cache.load_band("2b", ds).items():
            score_by_vid[vid] = float(row["score"])
        z_all = np.asarray([logit_scalar(score_by_vid[v]) for v in valid], dtype=float)
        full_band = cache.load_band("2b", ds)
        full_in_band = {v for v in valid if full_band.get(v, {}).get("in_band")}
        full_errors = {v for v in valid if base[v] != labels[v]}
        full_row = exp.eval_sequential(cache, MAIN_ORDER, slug="2b", rho_mode="entropy", route="band", stop=True)
        full_acc_by_ds = {r["ds"]: r["acc"] for r in full_row["per_dataset"]}
        full_acc = full_acc_by_ds[ds]

        for frac in (0.5, 0.7, 0.9):
            n = max(8, int(len(valid) * frac))
            for rep in range(40):
                sample_idx = rng.sample(range(len(valid)), n)
                weights, means, vars_ = fit_gmm_1d(z_all[sample_idx], seed=1000 + rep)
                q = posterior_hi(z_all, weights, means, vars_)
                entropies = np.asarray([entropy_scalar(float(p)) for p in q])
                hbar = float(np.mean(entropies))
                in_band = entropies >= hbar
                posterior_map = {v: float(p) for v, p in zip(valid, q)}
                in_band_map = {v: bool(b) for v, b in zip(valid, in_band)}
                band_set = {v for v, b in in_band_map.items() if b}
                errors_in = len(band_set & full_errors)
                acc, mf1, calls = final_with_custom_band(cache, ds, posterior_map, in_band_map, hbar)
                rows.append(
                    {
                        "dataset": ds,
                        "fraction": frac,
                        "rep": rep,
                        "band_rate": len(band_set) / len(valid),
                        "jaccard_vs_full_band": len(band_set & full_in_band) / len(band_set | full_in_band) if band_set | full_in_band else 1.0,
                        "error_capture": errors_in / len(full_errors) if full_errors else 0.0,
                        "hbar": hbar,
                        "rho": exp.rho_from_hbar(hbar),
                        "final_acc": acc,
                        "final_mf1": mf1,
                        "calls": calls,
                        "delta_acc_vs_full": acc - full_acc,
                    }
                )
        for frac in (0.5, 0.7, 0.9):
            sub = [r for r in rows if r["dataset"] == ds and r["fraction"] == frac]
            for metric in ("band_rate", "jaccard_vs_full_band", "error_capture", "final_acc", "delta_acc_vs_full", "calls"):
                vals = np.asarray([r[metric] for r in sub], dtype=float)
                summary.append(
                    {
                        "dataset": ds,
                        "fraction": frac,
                        "metric": metric,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "p05": float(np.quantile(vals, 0.05)),
                        "p95": float(np.quantile(vals, 0.95)),
                    }
                )
    return rows, summary


def run_full_predictions(cache: exp.EvalCache) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for ds in DATASETS:
        labels = cache.load_labels_for(ds)
        base = cache.load_base("2b", ds)
        band = cache.load_band("2b", ds)
        hbar = cache.hbar[("2b", ds)]
        rho = cache.rhod[("2b", ds)]
        lam = math.log(rho / (1 - rho))
        judges = [cache.load_judge(j, ds) for j in MAIN_ORDER]
        out[ds] = {}
        for vid in cache.valid_vids("2b", ds):
            b = band.get(vid, {})
            p0 = float(b.get("posterior_hi", 0.5))
            ell = logit_scalar(p0)
            votes = []
            used = 0
            if b.get("in_band"):
                for table in judges:
                    used += 1
                    r = table.get(vid, {}).get("pred")
                    votes.append(r if r in (0, 1) else None)
                    if r in (0, 1):
                        ell += (2 * int(r) - 1) * lam
                        if entropy_scalar(1 / (1 + math.exp(-ell))) <= hbar:
                            break
            p_final = 1 / (1 + math.exp(-ell))
            pred_final = 1 if p_final >= 0.5 else 0
            if not b.get("in_band"):
                pred_final = base[vid]
            out[ds][vid] = {
                "label": labels[vid],
                "s1_pred": base[vid],
                "s1_prob": p0,
                "final_prob": p_final,
                "final_pred": pred_final,
                "in_band": bool(b.get("in_band")),
                "used": used,
                "votes": votes,
            }
    return out


def run_calibration(full: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    all_y, all_s1, all_final = [], [], []
    for ds in DATASETS:
        y = [r["label"] for r in full[ds].values()]
        s1 = [r["s1_prob"] for r in full[ds].values()]
        final = [r["final_prob"] for r in full[ds].values()]
        all_y += y
        all_s1 += s1
        all_final += final
        m1 = calibration_metrics(y, s1)
        m2 = calibration_metrics(y, final)
        rows.append({"dataset": ds, "stage": "Stage 1", **m1})
        rows.append({"dataset": ds, "stage": "Full", **m2})
        rows.append(
            {
                "dataset": ds,
                "stage": "Delta Full-Stage1",
                "brier": m2["brier"] - m1["brier"],
                "nll": m2["nll"] - m1["nll"],
                "ece": m2["ece"] - m1["ece"],
            }
        )
    m1 = calibration_metrics(all_y, all_s1)
    m2 = calibration_metrics(all_y, all_final)
    rows.append({"dataset": "Average", "stage": "Stage 1", **m1})
    rows.append({"dataset": "Average", "stage": "Full", **m2})
    rows.append(
        {
            "dataset": "Average",
            "stage": "Delta Full-Stage1",
            "brier": m2["brier"] - m1["brier"],
            "nll": m2["nll"] - m1["nll"],
            "ece": m2["ece"] - m1["ece"],
        }
    )
    return rows


def run_conflict_trajectory(full: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    bucket: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for ds in DATASETS:
        for r in full[ds].values():
            if not r["in_band"]:
                continue
            s1 = r["s1_pred"]
            votes = r["votes"]
            used_votes = [v for v in votes if v in (0, 1)]
            if not used_votes:
                path = "no valid verifier"
            elif len(used_votes) == 1:
                path = "V1 agrees" if used_votes[0] == s1 else "V1 conflicts"
            elif len(used_votes) == 2:
                if used_votes[0] == s1:
                    path = "V1 agrees, V2 used"
                elif used_votes[1] == s1:
                    path = "V1 conflicts, V2 returns to S1"
                else:
                    path = "V1 conflicts, V2 confirms V1"
            else:
                seq = "".join("H" if v == 1 else "N" for v in [s1] + used_votes[:3])
                path = f"3-call path {seq}"
            key = (ds, path)
            bucket[key]["n"] += 1
            bucket[key]["s1_correct"] += int(r["s1_pred"] == r["label"])
            bucket[key]["final_correct"] += int(r["final_pred"] == r["label"])
            bucket[key]["flipped"] += int(r["final_pred"] != r["s1_pred"])
            bucket[key]["rescued"] += int(r["s1_pred"] != r["label"] and r["final_pred"] == r["label"])
            bucket[key]["harmed"] += int(r["s1_pred"] == r["label"] and r["final_pred"] != r["label"])
    for (ds, path), c in sorted(bucket.items()):
        n = c["n"]
        rows.append(
            {
                "dataset": ds,
                "trajectory": path,
                "n": n,
                "share_in_band": n / sum(v["n"] for (d, _), v in bucket.items() if d == ds),
                "stage1_acc": c["s1_correct"] / n if n else 0.0,
                "final_acc": c["final_correct"] / n if n else 0.0,
                "flip_rate": c["flipped"] / n if n else 0.0,
                "rescue_rate_all": c["rescued"] / n if n else 0.0,
                "harm_rate_all": c["harmed"] / n if n else 0.0,
                "rescued_n": c["rescued"],
                "harmed_n": c["harmed"],
            }
        )
    return rows


def run_classwise(full: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    for ds in DATASETS:
        y = [r["label"] for r in full[ds].values()]
        s1 = [r["s1_pred"] for r in full[ds].values()]
        final = [r["final_pred"] for r in full[ds].values()]
        m1 = macro_pos_neg(y, s1)
        m2 = macro_pos_neg(y, final)
        row = {"dataset": ds}
        for key in sorted(m1):
            row[f"s1_{key}"] = m1[key]
            row[f"full_{key}"] = m2[key]
            row[f"delta_{key}"] = m2[key] - m1[key]
        rows.append(row)
    return rows


def clean_text(text: str, max_len: int = 340) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def run_case_studies(cache: exp.EvalCache, full: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    for ds in DATASETS:
        judge_maps = [cache.load_judge(j, ds) for j in MAIN_ORDER]
        categories = {
            "rescued": [vid for vid, r in full[ds].items() if r["in_band"] and r["s1_pred"] != r["label"] and r["final_pred"] == r["label"]],
            "confirmed": [vid for vid, r in full[ds].items() if r["in_band"] and r["used"] == 1 and r["s1_pred"] == r["label"] and r["final_pred"] == r["label"]],
            "hard_failed": [vid for vid, r in full[ds].items() if r["in_band"] and r["used"] == 3 and r["final_pred"] != r["label"]],
        }
        for category, vids in categories.items():
            for vid in vids[:3]:
                r = full[ds][vid]
                rationales = []
                for name, jm in zip(MAIN_ORDER, judge_maps):
                    if vid in jm:
                        rationales.append(f"{name}: {clean_text(jm[vid].get('rationale', ''))}")
                rows.append(
                    {
                        "dataset": ds,
                        "category": category,
                        "video_id": vid,
                        "label": r["label"],
                        "stage1_pred": r["s1_pred"],
                        "final_pred": r["final_pred"],
                        "calls": r["used"],
                        "votes": " ".join("H" if v == 1 else "N" if v == 0 else "?" for v in r["votes"]),
                        "rationale_snippet": " || ".join(rationales[:2]),
                    }
                )
    return rows


def build_summary(
    gmm_summary: list[dict],
    calibration: list[dict],
    conflict: list[dict],
    classwise: list[dict],
    cases: list[dict],
) -> str:
    lines = ["# Additional Experiment Probe", ""]
    lines.append("## Quick Read")
    lines.append("")

    gmm_acc = [r for r in gmm_summary if r["metric"] == "delta_acc_vs_full" and r["fraction"] == 0.7]
    worst_gmm = min(gmm_acc, key=lambda r: r["p05"]) if gmm_acc else None
    if worst_gmm:
        lines.append(
            f"- GMM stability: at 70% unlabeled subsampling, worst 5th-percentile ACC drift is "
            f"{pct(worst_gmm['p05']):+.2f} pp on {worst_gmm['dataset']}; this is usable if the stds are small."
        )

    avg_delta = [r for r in calibration if r["dataset"] == "Average" and r["stage"] == "Delta Full-Stage1"]
    if avg_delta:
        r = avg_delta[0]
        lines.append(
            f"- Posterior calibration: aggregate Brier delta {r['brier']:+.4f}, ECE delta {r['ece']:+.4f}. "
            "Negative is good."
        )

    top_conflict = sorted(conflict, key=lambda r: r["n"], reverse=True)[:5]
    if top_conflict:
        lines.append("- Conflict trajectory: largest buckets are " + "; ".join(f"{r['dataset']} {r['trajectory']} n={r['n']}" for r in top_conflict) + ".")

    hate_recalls = [(r["dataset"], r["delta_hateful_recall"]) for r in classwise]
    if hate_recalls:
        lines.append(
            "- Class-wise behavior: hateful recall deltas are "
            + ", ".join(f"{d} {pct(v):+.1f}pp" for d, v in hate_recalls)
            + "."
        )

    lines.append(f"- Qualitative cases: collected {len(cases)} candidate rows across rescued / confirmed / hard-failed categories.")
    lines.append("")
    lines.append("## Files")
    for name in (
        "gmm_stability_raw.csv",
        "gmm_stability_summary.csv",
        "posterior_calibration.csv",
        "conflict_trajectory.csv",
        "classwise_behavior.csv",
        "qualitative_cases.csv",
    ):
        lines.append(f"- `paper/analysis/{name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cache = exp.EvalCache()

    gmm_raw, gmm_summary = run_gmm_stability(cache)
    full = run_full_predictions(cache)
    calibration = run_calibration(full)
    conflict = run_conflict_trajectory(full)
    classwise = run_classwise(full)
    cases = run_case_studies(cache, full)

    write_csv(ANALYSIS_DIR / "gmm_stability_raw.csv", gmm_raw)
    write_csv(ANALYSIS_DIR / "gmm_stability_summary.csv", gmm_summary)
    write_csv(ANALYSIS_DIR / "posterior_calibration.csv", calibration)
    write_csv(ANALYSIS_DIR / "conflict_trajectory.csv", conflict)
    write_csv(ANALYSIS_DIR / "classwise_behavior.csv", classwise)
    write_csv(ANALYSIS_DIR / "qualitative_cases.csv", cases)
    (ANALYSIS_DIR / "additional_experiment_probe.md").write_text(
        build_summary(gmm_summary, calibration, conflict, classwise, cases),
        encoding="utf-8",
    )

    print(ANALYSIS_DIR / "additional_experiment_probe.md")


if __name__ == "__main__":
    main()
