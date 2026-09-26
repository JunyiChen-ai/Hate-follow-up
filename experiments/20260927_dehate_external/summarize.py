#!/usr/bin/env python3
"""DeHate table and paired bootstrap (README §2). Evaluation only: reads the gold.

Point estimates are copied from the evaluator's metrics.json files; the weakly supervised rows are seed means. The
bootstrap resamples the 1151 scored videos (4000 resamples, seed 0) and recomputes each metric with sklearn on
frame weights equal to the video's multiplicity, which equals duplicating the video. Pooled ROC and PR use every
frame; within averages the per-video ROC over videos with both classes. Per-video curves are cut to the gold length
as in the evaluator. A method with several seeds is scored as its seed mean in every resample. The unweighted
estimate is checked against metrics.json before any interval is reported.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20260927_dehate_external"
OURS = {"spvl_r2": "SPVL-r2", "current": "SPVL-r2 + duration prior (current)", "r3_m2": "r3_m2 (candidate)",
        "r3_full": "r3_full (candidate, interval output)"}
ZERO = {"zs_imagebind": "ZS-ImageBind (zero-label)"}
WEAK = {"fed_wsvad_3client": "Fed-WSVAD, 3 clients (video labels)", "multihateloc": "MultiHateLoc (video labels)",
        "dsanet": "DSANet (video labels)", "macilsd": "MACIL-SD (video labels)"}
SEEDS = (234, 2025, 3407)
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
SHORT = {"frame_ROC_AUC": "ROC", "frame_PR_AUC": "PR", "within_video_macro_ROC_AUC": "within"}
N_BOOT, SEED = 4000, 0

G = {}


def load_curves(pred_path, method):
    out = {}
    for line in open(pred_path):
        r = json.loads(line)
        if r.get("dataset") == "DeHate" and r.get("method") == method and r.get("score_curve") and not r.get("error"):
            out[r["video_id"]] = np.asarray(r["score_curve"], float)
    return out


def pack(curves, Y, vids):
    """Frame arrays in the evaluator's alignment, plus each frame's video index and per-video within terms."""
    ys, ss, idx, wv, wauc = [], [], [], [], []
    for k, v in enumerate(vids):
        y = Y[v]; s = curves[v]; n = min(len(y), len(s))
        y, s = y[:n], s[:n]
        ok = np.isfinite(s); y, s = y[ok], s[ok]
        ys.append(y); ss.append(s); idx.append(np.full(len(y), k))
        if len(y) and y.min() != y.max():
            wv.append(k); wauc.append(roc_auc_score(y, s))
    return {"y": np.concatenate(ys), "s": np.concatenate(ss), "i": np.concatenate(idx),
            "wv": np.asarray(wv), "wauc": np.asarray(wauc)}


def metrics(p, counts):
    w = counts[p["i"]].astype(float)
    m = w > 0
    wd = counts[p["wv"]].astype(float)
    return np.array([roc_auc_score(p["y"][m], p["s"][m], sample_weight=w[m]),
                     average_precision_score(p["y"][m], p["s"][m], sample_weight=w[m]),
                     float((wd * p["wauc"]).sum() / wd.sum())])


def method_metrics(name, counts):
    return np.mean([metrics(p, counts) for p in G["packs"][name]], axis=0)


def boot_chunk(seeds_counts):
    return [{name: method_metrics(name, c) for name in G["names"]} for c in seeds_counts]


def main():
    g = np.load(ROOT / "data/gt_4fps/DeHate.npz", allow_pickle=True)
    Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
    vids = sorted(Y)
    packs, point = {}, {}
    for tag in OURS:
        mp = json.load(open(R / tag / "metrics.json"))
        point[tag] = [p for p in mp["per_dataset"] if p["dataset"] == "DeHate"][0]
        method = point[tag]["method"]
        packs[tag] = [pack(load_curves(R / tag / "predictions.jsonl", method), Y, vids)]
    for tag in ZERO:
        if (R / tag / "metrics.json").exists():
            point[tag] = json.load(open(R / tag / "metrics.json"))["per_dataset"][0]
            packs[tag] = [pack(load_curves(R / tag / "predictions.jsonl", "zs_imagebind"), Y, vids)]
    wm = {p["method"]: p for p in json.load(open(R / "weaksup/metrics.json"))["per_dataset"]}
    for tag in WEAK:
        rows = [wm[f"{tag}__seed{s}"] for s in SEEDS]
        point[tag] = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
        point[tag]["sd"] = {k: float(np.std([r[k] for r in rows])) for k in METRICS}
        packs[tag] = [pack(load_curves(R / "weaksup/predictions.jsonl", f"{tag}__seed{s}"), Y, vids) for s in SEEDS]

    G["packs"] = packs
    ones = np.ones(len(vids), int)
    for tag in packs:  # the unweighted estimate must equal the evaluator's number
        est = method_metrics(tag, ones)
        ref = np.array([point[tag][k] for k in METRICS])
        assert np.allclose(est, ref, atol=1e-9), (tag, est, ref)

    baselines = [t for t in list(ZERO) + list(WEAK) if t in point]
    best = {k: max(baselines, key=lambda t: point[t][k]) for k in METRICS}
    pairs = [("r3_m2", "current")] + [(o, "best") for o in ("current", "r3_m2", "spvl_r2")]
    G["names"] = sorted(set(["r3_m2", "current", "spvl_r2"] + list(best.values())))

    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, len(vids), len(vids)), minlength=len(vids)) for _ in range(N_BOOT)]
    chunks = [draws[i::14] for i in range(14)]
    with Pool(14) as pool:
        res = [x for part in pool.map(boot_chunk, chunks) for x in part]

    lines = ["DeHate test, 4 fps, 1151 scored videos (234 hateful), frame base rate "
             f"{point['current'].get('base_rate', float('nan')):.3f}. ROC / PR / within; weakly supervised rows are "
             "seed means (sd).", ""]
    for tag, label in list(OURS.items()) + list(ZERO.items()) + list(WEAK.items()):
        if tag not in point:
            continue
        p = point[tag]
        sd = p.get("sd")
        cells = [f"{p[k]:.4f}" + (f" ({sd[k]:.3f})" if sd else "") for k in METRICS]
        extra = ""
        if tag == "r3_full":
            extra = f"   interval F1@.3/.5/.7 {p['interval_F1@0.3']:.3f} / {p['interval_F1@0.5']:.3f} / {p['interval_F1@0.7']:.3f}"
        lines.append(f"{label:42s} " + " / ".join(cells) + extra)
    lines.append("")
    lines.append("Strongest baseline per metric: " + ", ".join(f"{SHORT[k]} {best[k]}" for k in METRICS))
    summary = {"point": point, "best_baseline": best, "pairs": {}}
    for a, b in pairs:
        for j, k in enumerate(METRICS):
            bb = best[k] if b == "best" else b
            d = np.array([r[a][j] - r[bb][j] for r in res])
            diff = point[a][k] - point[bb][k]
            lo, hi = np.quantile(d, [.025, .975])
            summary["pairs"][f"{a}-{bb}:{SHORT[k]}"] = [diff, float(lo), float(hi)]
            lines.append(f"  {a} - {bb} {SHORT[k]:6s} {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    out = R / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
