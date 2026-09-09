#!/usr/bin/env python3
"""Risk checks R1/R2 for an SPVL run (no labels used).

- cross-video: Spearman(z_video, legacy whole-video z)                        (parity of the verdict)
- within-video: Spearman(window z, legacy per-chunk log-odds) per video, median (how much the
  window scores are the old chunk scores renamed); legacy chunks are matched to the window that
  contains their midpoint
- collapse check: median within-video std of window z, share of videos with < 2 distinct window z
Writes diagnose.json into the run dir.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
LEGACY_CHUNKS = ROOT / "data/omsl_v6_inputs/text_unified_qwen3vl8b_chunk_scores_b1_fullcoverage.jsonl"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); a = ap.parse_args()
    run = Path(a.run_dir)
    rows = [json.loads(l) for l in open(run / "predictions.jsonl") if l.strip()]
    rows = [r for r in rows if not r.get("error") and r.get("extra")]
    legacy_z = {}
    for ds in ("HateMM", "HateClipSeg"):
        for l in open(ROOT / f"data/omsl_v6_inputs/holistic_consistent/{ds}/scores.jsonl"):
            d = json.loads(l); legacy_z[(ds, d["video_id"])] = float(d["z"])
    chunks = {}
    for l in open(LEGACY_CHUNKS):
        d = json.loads(l)
        if d["dataset"] in ("HateMM", "HateClipSeg"):
            chunks.setdefault((d["dataset"], d["video_id"]), []).append((float(d["start"]), float(d["end"]), float(d["log_odds"])))
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds]
        zv = [(r["extra"]["z_video"], legacy_z.get((ds, r["video_id"]))) for r in rs
              if r["extra"].get("z_video") is not None and (ds, r["video_id"]) in legacy_z]
        rho_v = float(spearmanr([a for a, _ in zv], [b for _, b in zv]).correlation) if len(zv) > 2 else None
        per_video, stds, collapsed = [], [], 0
        for r in rs:
            w = r["extra"]["windows"]
            z = np.array([x["z"] for x in w], float)
            if len(z) == 0:
                continue
            stds.append(float(z.std()))
            if len(np.unique(np.round(z, 3))) < 2:
                collapsed += 1
            lc = chunks.get((ds, r["video_id"]), [])
            pairs = []
            for s, e, lo in lc:
                mid = 0.5 * (s + e)
                j = [i for i, x in enumerate(w) if x["start"] <= mid < x["end"] or (i == len(w) - 1 and mid >= x["start"])]
                if j:
                    pairs.append((z[j[0]], lo))
            if len(pairs) >= 4 and np.ptp([p[0] for p in pairs]) > 0 and np.ptp([p[1] for p in pairs]) > 0:
                per_video.append(float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation))
        out[ds] = {"n_videos": len(rs), "spearman_zvideo_vs_legacy_z": rho_v,
                   "median_within_spearman_window_vs_legacy_chunk": float(np.median(per_video)) if per_video else None,
                   "n_videos_with_chunk_pairs": len(per_video),
                   "median_within_std_window_z": float(np.median(stds)) if stds else None,
                   "videos_with_collapsed_windows": collapsed}
    (run / "diagnose.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
