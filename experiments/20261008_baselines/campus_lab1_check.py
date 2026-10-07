#!/usr/bin/env python3
"""uoa-lab1 re-check of predictions made on uoa-campus2 (2026-10-08; campus_*.sbatch): exact cohort, finite score on every GT frame, then the
canonical evaluator (unchanged) into metrics_lab1.json next to the campus metrics.json, and a field-by-field
comparison, written to lab1_check.json. Only GT ids and lengths are read here; labels reach only the evaluator
subprocess.

    python experiments/20261008_baselines/campus_lab1_check.py runs/20261008_baselines/<method>/<DS>:<DS> ...
"""
import json, os, socket, subprocess, sys, datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/jehc223/Hate-follow-up")
sys.path.insert(0, str(REPO / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402

KEYS = ["frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC", "n_videos_predicted", "n_videos_overlap",
        "n_frames", "n_videos_defined"]


def check(d: Path, ds: str) -> dict:
    pred = d / "predictions.jsonl"
    rows = {}
    for line in open(pred):
        r = json.loads(line)
        if r["dataset"] == ds:
            rows[r["video_id"]] = r
    ids = ec.cohort(ds)
    t = ec.gt_lengths(ds)
    bad = [v for v in ids if v not in rows or rows[v].get("error") or len(rows[v]["score_curve"]) < t[v]
           or not np.isfinite(np.asarray(rows[v]["score_curve"][:t[v]], float)).all()]
    extra = sorted(set(rows) - set(ids))
    out = d / "metrics_lab1.json"
    cmd = [sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(pred),
           "--gt-dir", str(REPO / "data/gt_4fps"), "--out", str(out), "--datasets", ds]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    a = [r for r in json.load(open(d / "metrics.json"))["per_dataset"] if r["dataset"] == ds][0]
    b = [r for r in json.load(open(out))["per_dataset"] if r["dataset"] == ds][0]
    rep = {"code": "experiments/20261008_baselines/campus_lab1_check.py (" + ec.code_version() + ")",
           "host": socket.gethostname(), "date": datetime.datetime.now().isoformat(timespec="seconds"),
           "dataset": ds, "predictions": str(pred.relative_to(REPO)), "cohort_size": len(ids),
           "rows": len(rows), "missing_or_nonfinite": bad, "extra_ids": extra,
           "exact_test_set": not bad and not extra,
           "evaluator": " ".join(cmd[1:]),
           "campus_metrics": {k: a[k] for k in KEYS}, "lab1_metrics": {k: b[k] for k in KEYS},
           "identical": all(a[k] == b[k] for k in KEYS)}
    (d / "lab1_check.json").write_text(json.dumps(rep, indent=2) + "\n")
    return rep


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        d, ds = spec.rsplit(":", 1)
        r = check(REPO / d, ds)
        m = r["lab1_metrics"]
        print(f"{d} {ds}: exact={r['exact_test_set']} rows={r['rows']} identical={r['identical']} "
              f"ROC {m['frame_ROC_AUC']:.4f} PR {m['frame_PR_AUC']:.4f} within {m['within_video_macro_ROC_AUC']:.4f} "
              f"n_videos {m['n_videos_overlap']} n_frames {m['n_frames']} n_within {m['n_videos_defined']}")
