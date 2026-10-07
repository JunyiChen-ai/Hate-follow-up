#!/usr/bin/env python3
"""Collect the evaluator's metrics.json of the SAGE / CLARA window runs into one summary (mean and sd over seeds).

    python experiments/20261008_baselines/detwin/collect.py

Reads runs/20261008_baselines/{sage,clara,clara_norationale}/<DS>/seed<k>/{metrics.json,coverage.json} and writes
runs/20261008_baselines/sage_clara_summary.json. Numbers are copied from the evaluator outputs; nothing is
recomputed.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")


def main():
    out = {"source": "runs/20261008_baselines/<method>/<DS>/seed<k>/metrics.json", "rows": []}
    for mdir in ("sage", "clara", "clara_norationale"):
        for ds in C.DATASETS:
            per_seed = {}
            for s in C.SEEDS:
                d = C.RUNS / mdir / ds / f"seed{s}"
                mp, cp = d / "metrics.json", d / "coverage.json"
                if not mp.is_file():
                    continue
                cov = json.loads(cp.read_text()) if cp.is_file() else {}
                row = [r for r in json.loads(mp.read_text())["per_dataset"] if r["dataset"] == ds][0]
                per_seed[s] = {**{k: row[k] for k in METRICS}, "n_videos": row["n_videos_overlap"],
                               "exact_test_set": cov.get("exact_test_set"), "f2_videos": len(cov.get("f2_videos", {})),
                               "f1_tail_windows": sum(len(x) for x in cov.get("f1_tail_windows", {}).values()),
                               "metrics_json": str(mp.relative_to(C.REPO))}
            if not per_seed:
                continue
            agg = {}
            for k in METRICS:
                vals = [r[k] for r in per_seed.values() if r[k] is not None]
                agg[k] = {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}
            out["rows"].append({"method": mdir, "dataset": ds, "seeds": per_seed, "mean_sd": agg,
                                "n_seeds": len(per_seed)})
    path = C.RUNS / "sage_clara_summary.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for r in out["rows"]:
        a = r["mean_sd"]
        print(f"{r['method']:18s} {r['dataset']:12s} seeds={r['n_seeds']} "
              + " ".join(f"{k.split('_')[0] if k != 'within_video_macro_ROC_AUC' else 'within'}="
                         f"{a[k]['mean']:.4f}±{a[k]['sd']:.4f}" for k in METRICS))
    print(f"written {path}")


if __name__ == "__main__":
    main()
