#!/usr/bin/env python3
"""E0 (rule 10 test read): is the structured hypothesis reliable?

(a) does it vary with content: TARGET / FORM distributions, relation of "none" to the verdict z;
(b) can the cited EVIDENCE spans localize: paint 1 inside cited spans, 0 elsewhere -> within-video macro ROC
    via the shared evaluator (metrics_cited.json), compared with the SPVL-r2 and per-window-alone numbers;
(c) coverage: fraction of GT-positive frames inside cited spans; precision: fraction of cited frames that are GT-positive.
Reads GT only for (c) and only to report; nothing here feeds a score.
"""
import argparse, json, math, os, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    run = Path(a.run_dir)
    rows = [json.loads(l) for l in open(run / "predictions.jsonl") if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    gt = {ds: np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True) for ds in a.datasets}
    gtmap = {(ds, str(v)): np.asarray(y) for ds, g in gt.items() for v, y in zip(g["video_ids"], g["y4"])}
    out_pred = run / "predictions_cited.jsonl"
    stats = {ds: {"n": 0, "none": 0, "none_z_neg": 0, "z_neg": 0, "with_evidence": 0, "n_ev": [], "targets": Counter(),
                  "forms": Counter(), "cov_num": 0, "cov_den": 0, "prec_num": 0, "prec_den": 0} for ds in a.datasets}
    with open(out_pred, "w") as fh:
        for r in rows:
            ds, vid = r["dataset"], r["video_id"]
            h = r["extra"].get("hypothesis") or {}
            st = stats[ds]
            st["n"] += 1
            z = r["extra"]["z_video"]
            if z < 0:
                st["z_neg"] += 1
            if h.get("is_none"):
                st["none"] += 1
                if z < 0:
                    st["none_z_neg"] += 1
            ev = h.get("evidence") or []
            if ev:
                st["with_evidence"] += 1
                st["n_ev"].append(len(ev))
            st["targets"][(h.get("target") or "").lower()[:40]] += 1
            for f in h.get("form") or []:
                st["forms"][f] += 1
            L = int(math.ceil(float(r["duration"]) * FPS))
            curve = np.zeros(L)
            for e in ev:
                i0, i1 = max(0, int(math.floor(e["start"] * FPS))), min(L, int(math.ceil(e["end"] * FPS)))
                if i1 > i0:
                    curve[i0:i1] = 1.0
            y = gtmap.get((ds, vid))
            if y is not None:
                n = min(len(y), L)
                st["cov_num"] += int(((y[:n] > 0) & (curve[:n] > 0)).sum()); st["cov_den"] += int((y[:n] > 0).sum())
                st["prec_num"] += int(((y[:n] > 0) & (curve[:n] > 0)).sum()); st["prec_den"] += int((curve[:n] > 0).sum())
            fh.write(json.dumps({**r, "method": r["method"] + "__cited", "score_curve": curve.tolist()}) + "\n")
    metrics = run / "metrics_cited.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(out_pred),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(metrics), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    m = {p["dataset"]: p for p in json.load(open(metrics))["per_dataset"]}
    for ds, st in stats.items():
        p = m.get(ds, {})
        print(f"== {ds}: videos {st['n']}  z<0 {st['z_neg']}  hypothesis none {st['none']} (of which z<0 {st['none_z_neg']})  "
              f"with evidence {st['with_evidence']}  items/video {np.mean(st['n_ev']) if st['n_ev'] else 0:.1f}")
        print(f"   cited-span within {p.get('within_video_macro_ROC_AUC', float('nan')):.4f} (n={p.get('n_videos_defined')})  "
              f"pooled ROC {p.get('frame_ROC_AUC', float('nan')):.4f}  coverage of GT+ frames {st['cov_num'] / max(1, st['cov_den']):.3f}  "
              f"precision of cited frames {st['prec_num'] / max(1, st['prec_den']):.3f}")
        print("   targets:", st["targets"].most_common(12))
        print("   forms:", sorted(st["forms"].items()))


if __name__ == "__main__":
    main()
