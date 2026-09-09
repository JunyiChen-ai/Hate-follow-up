#!/usr/bin/env python3
"""Evaluate a run on a named video subset with the shared evaluator.

Subsets (rule 10: the GT and the Whisper coverage are read only to pick videos):
  silent_hate : hateful videos with > 50 % of their positive frames outside any Whisper speech segment
  within      : videos whose grid has both classes
Usage: subset_eval.py --run-dir runs/20260910_spvl/full --compose ispvl_rrank --subset silent_hate
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/20260910_spvl"))
import spvl  # noqa: E402


def silent_hate_ids(datasets):
    out = set()
    for ds in datasets:
        asr = spvl.load_asr(ds); g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        for vid, y in zip(g["video_ids"], g["y4"]):
            y = np.asarray(y); L = len(y); cov = np.zeros(L, bool)
            for s, e, t in asr.get(str(vid), []):
                if t.strip():
                    cov[int(np.floor(s * 4)):min(L, int(np.ceil(e * 4)))] = True
            p = y > 0
            if p.sum() and (p & ~cov).sum() / p.sum() > 0.5:
                out.add((ds, str(vid)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--compose", default="ispvl_rrank")
    ap.add_argument("--subset", choices=["silent_hate", "within"], default="silent_hate")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    keep = silent_hate_ids(a.datasets) if a.subset == "silent_hate" else spvl.within_defined_ids(a.datasets)
    run = Path(a.run_dir); src = run / f"predictions_{a.compose}.jsonl"
    dst = run / f"predictions_{a.compose}__{a.subset}.jsonl"; n = 0
    with open(dst, "w") as fh:
        for l in open(src):
            r = json.loads(l)
            if (r["dataset"], r["video_id"]) in keep:
                fh.write(l); n += 1
    m = run / f"metrics_{a.compose}__{a.subset}.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(dst),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(m), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True)
    d = json.load(open(m))
    print(f"{run.name} [{a.subset}, {n} videos] " + "  ".join(
        f"{p['dataset']} within {p['within_video_macro_ROC_AUC']:.4f} (n={p['n_videos_defined']})" for p in d["per_dataset"]))


if __name__ == "__main__":
    main()
