"""Per-(MLLM, dataset) unsupervised threshold search.

For each stage-1 MLLM slug and dataset, fit all three unsupervised
criteria {otsu, gmm, li_lee} on the appropriate fit source (TR for
EN/ZH/IH, TF for HateMM), apply to test, and report acc / macro-F1.
The 'oracle' pick per cell is the criterion with highest macro-F1
(acc is used as tiebreak).

Also emits `baseline_preds_v2_<slug>_<crit>.jsonl` and
`v2_baseline_<slug>_<crit>.json` for each (slug, crit) pair — these
are the artefacts Phase D / Phase E consume.

Writes:
  results/boundary_rescue/threshold_search_summary.json
  results/boundary_rescue/threshold_search_summary.md

No GPU. Expected runtime: ~1 min for 5 slugs × 3 criteria × 4 ds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "our_method"))
sys.path.insert(0, str(_HERE.parent / "naive_baseline"))

from quick_eval_all import load_scores_file  # noqa: E402
from data_utils import load_annotations, SKIP_VIDEOS  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402
from thresholds import otsu_threshold, gmm_threshold, li_lee_threshold  # noqa: E402

PROJECT_ROOT = Path("/data/jehc223/EMNLP3")
OUT_ROOT = PROJECT_ROOT / "results" / "boundary_rescue"
DS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
CRITERIA = {"otsu": otsu_threshold, "gmm": gmm_threshold, "li_lee": li_lee_threshold}
# 2B uses a legacy ZH test path; everything else uses plain test_binary.jsonl
FIT_SOURCE = {"MHClip_EN": "train", "MHClip_ZH": "train",
              "HateMM": "test", "ImpliHateVid": "train"}


def score_files(slug):
    def tr(ds):
        return PROJECT_ROOT / "results" / f"holistic_{slug}" / ds / "train_binary.jsonl"
    def te(ds):
        if slug == "2b" and ds == "MHClip_ZH":
            return PROJECT_ROOT / "results" / "holistic_2b" / "MHClip_ZH" / "test_binary.jsonl.prerepro_20260413"
        return PROJECT_ROOT / "results" / f"holistic_{slug}" / ds / "test_binary.jsonl"
    return {ds: (tr(ds), te(ds)) for ds in DS}


def metrics(y, yh):
    assert len(y) == len(yh)
    n = len(y)
    if n == 0:
        return 0.0, 0.0, 0, 0, 0, 0
    tp = sum(1 for a, b in zip(y, yh) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, yh) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yh) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, yh) if a == 0 and b == 0)
    acc = (tp + tn) / n
    p1 = tp / (tp + fp) if tp + fp else 0.0
    r1 = tp / (tp + fn) if tp + fn else 0.0
    f1p = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0
    r0 = tn / (tn + fp) if tn + fp else 0.0
    f1n = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    return acc, (f1p + f1n) / 2, tp, fp, fn, tn


def evaluate_cell(slug, ds, crit_name):
    tr_path, te_path = score_files(slug)[ds]
    fs = FIT_SOURCE[ds]
    fit_path = tr_path if fs == "train" else te_path
    if not fit_path.exists() or not te_path.exists():
        return None
    fit_scores = np.array(list(load_scores_file(str(fit_path)).values()), dtype=float)
    test_score_dict = load_scores_file(str(te_path))
    thr = CRITERIA[crit_name](fit_scores)

    ann = load_annotations(ds)
    skip = SKIP_VIDEOS.get(ds, set())
    y, yh, vids = [], [], []
    for vid, s in test_score_dict.items():
        if vid in skip or vid not in ann:
            continue
        lab = collapse_label(ds, ann[vid]["label"])
        if lab not in (0, 1):
            continue
        pred = int(s >= thr)
        y.append(lab); yh.append(pred); vids.append(vid)
    acc, mf1, tp, fp, fn, tn = metrics(y, yh)
    return {
        "slug": slug, "ds": ds, "criterion": crit_name,
        "fit_source": fs, "threshold": float(thr),
        "n_fit": int(len(fit_scores)), "n_eval": int(len(y)),
        "acc": float(acc), "mf1": float(mf1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fit_path": str(fit_path), "test_path": str(te_path),
    }


def run_baseline_preds(slug, crit):
    """Call baseline_preds_v2.py to emit the artefacts needed downstream.
    Idempotent: baseline_preds_v2.py re-reads scores and overwrites."""
    cmd = [sys.executable,
           str(_HERE / "baseline_preds_v2.py"),
           "--model-tag", slug, "--criterion", crit]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stdout, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="+",
                    default=["2b", "qwen2.5-vl-7b", "gemma-3-12b-it",
                             "minicpm-v-26", "pixtral-12b-2409", "internvl3-14b"],
                    help="Stage-1 slugs to search over")
    ap.add_argument("--emit-preds", action="store_true",
                    help="Also run baseline_preds_v2.py for each (slug, crit) "
                         "pair so Phase D/E artefacts are on disk.")
    args = ap.parse_args()

    rows = []
    for slug in args.slugs:
        for ds in DS:
            for crit in CRITERIA:
                r = evaluate_cell(slug, ds, crit)
                if r is None:
                    continue
                rows.append(r)

    # Per-(slug, ds) oracle pick by (mf1 desc, acc desc)
    oracles = {}
    for r in rows:
        key = (r["slug"], r["ds"])
        cur = oracles.get(key)
        if cur is None or (r["mf1"], r["acc"]) > (cur["mf1"], cur["acc"]):
            oracles[key] = r

    summary = {
        "rows": rows,
        "oracles": [{"slug": k[0], "ds": k[1], **v} for k, v in oracles.items()],
    }
    out_json = OUT_ROOT / "threshold_search_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown table (one block per slug)
    lines = ["# Threshold search — per (MLLM, dataset, criterion)", ""]
    slugs_present = sorted(set(r["slug"] for r in rows))
    for slug in slugs_present:
        lines += [f"## slug = `{slug}`", "",
                  "| dataset | criterion | fit_src | threshold | acc | mF1 | TP | FP | FN | TN |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for ds in DS:
            for crit in CRITERIA:
                r = next((x for x in rows if x["slug"] == slug and x["ds"] == ds and x["criterion"] == crit), None)
                if r is None:
                    continue
                oracle_mark = " ⭐" if oracles.get((slug, ds)) and oracles[(slug, ds)]["criterion"] == crit else ""
                lines.append(f"| {ds} | {crit}{oracle_mark} | {r['fit_source']} | "
                             f"{r['threshold']:.4f} | {r['acc']:.4f} | {r['mf1']:.4f} | "
                             f"{r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |")
        lines.append("")
        # oracle picks
        lines += ["### Oracle pick (highest mF1)", "",
                  "| dataset | oracle crit | acc | mF1 |",
                  "|---|---|---|---|"]
        for ds in DS:
            o = oracles.get((slug, ds))
            if o:
                lines.append(f"| {ds} | **{o['criterion']}** | {o['acc']:.4f} | {o['mf1']:.4f} |")
        lines.append("")
    (OUT_ROOT / "threshold_search_summary.md").write_text("\n".join(lines))

    print(f"Summary → {out_json}")
    print(f"Markdown → {OUT_ROOT / 'threshold_search_summary.md'}")

    if args.emit_preds:
        print("\nEmitting baseline_preds_v2_*.jsonl + v2_baseline_*.json per (slug, crit)...")
        for slug in args.slugs:
            for crit in CRITERIA:
                # only emit if this slug has enough data
                any_row = any(r["slug"] == slug for r in rows)
                if not any_row:
                    continue
                ok, out, err = run_baseline_preds(slug, crit)
                print(f"  {slug:20s} {crit:8s} {'OK' if ok else 'FAIL'}")
                if not ok:
                    print(err[-400:])


if __name__ == "__main__":
    main()
