#!/usr/bin/env python3
"""Pre-check of README §2 (reads the gold; analysis only, never part of a method).

Are stretches of "topic" (the window mentions the target group) longer than stretches of hate, so that a three-level
time model (off topic / topic without attack / attack) has something to identify? Inputs: the window-level topic reads
`t` and act reads `a` of experiments/20260912_tad (runs/20260912_tad/e0/predictions_topic.jsonl; 183 videos, same 8 s
grid, older ASR loader) and the 4 fps gold. Output: runs/20260928_infer/precheck_topic/table.txt and summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 4
OUT = ROOT / "runs/20260928_infer/precheck_topic"


def runs_of(b):
    """Lengths of maximal runs of True in a boolean array, with (start, end) indices."""
    out, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reads = {}
    for line in open(ROOT / "runs/20260912_tad/e0/predictions_topic.jsonl"):
        r = json.loads(line)
        if not r.get("error"):
            reads.setdefault(r["dataset"], {})[r["video_id"]] = r
    lines, summary = [], {}
    for ds in ("HateMM", "HateClipSeg"):
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
        ts = np.array([w["t"] for r in reads[ds].values() for w in r["extra"]["windows"] if "t" in w])
        thr = float(np.median(ts))                      # label-free threshold: corpus median of the topic read
        hate_runs, topic_runs, ratio, grp = [], [], [], {"none": [], "topic_only": [], "hate": []}
        n_vid = 0
        for v, r in sorted(reads[ds].items()):
            if v not in Y:
                continue
            ws = sorted(r["extra"]["windows"], key=lambda w: w["start"])
            if any("t" not in w for w in ws):
                continue
            y = Y[v]
            share = np.array([y[int(round(w["start"] * FPS)):max(int(round(w["start"] * FPS)) + 1, min(int(round(w["end"] * FPS)), len(y)))].mean()
                              for w in ws])
            hate = share >= .5
            topic = np.array([w["t"] for w in ws]) > thr
            act = np.array([w["a"] for w in ws])
            n_vid += 1
            hr, tr = runs_of(hate), runs_of(topic)
            hate_runs += [b - a for a, b in hr]; topic_runs += [b - a for a, b in tr]
            for a, b in hr:
                cover = [(c, d) for c, d in tr if c <= a < d]        # the topic run holding the hate run's first window
                ratio.append((cover[0][1] - cover[0][0]) / (b - a) if cover else 0.0)
            for i in range(len(ws)):
                grp["hate" if hate[i] else ("topic_only" if topic[i] else "none")].append(act[i])
        hate_runs, topic_runs, ratio = map(np.array, (hate_runs, topic_runs, ratio))
        med = {k: float(np.median(v)) for k, v in (("hate_run", hate_runs), ("topic_run", topic_runs), ("ratio", ratio))}
        s = {"videos": n_vid, "topic_threshold": thr, "n_hate_runs": int(len(hate_runs)), "n_topic_runs": int(len(topic_runs)),
             "median_hate_run_windows": med["hate_run"], "median_topic_run_windows": med["topic_run"],
             "median_ratio_topic_run_over_hate_run": med["ratio"], "share_hate_runs_not_in_topic_run": float((ratio == 0).mean()),
             "mean_act_read": {k: float(np.mean(v)) for k, v in grp.items()}, "n_windows": {k: len(v) for k, v in grp.items()},
             "hate_run_pct": np.percentile(hate_runs, [25, 50, 75]).tolist(), "topic_run_pct": np.percentile(topic_runs, [25, 50, 75]).tolist(),
             "ratio_pct": np.percentile(ratio, [25, 50, 75]).tolist()}
        summary[ds] = s
        lines += [f"== {ds}: {n_vid} videos, topic threshold (corpus median of t) {thr:.2f}",
                  f"  hate runs: n {len(hate_runs)}, windows q25/50/75 {np.percentile(hate_runs, [25, 50, 75]).round(1).tolist()}",
                  f"  topic runs: n {len(topic_runs)}, windows q25/50/75 {np.percentile(topic_runs, [25, 50, 75]).round(1).tolist()}",
                  f"  topic run holding the hate run / hate run length: q25/50/75 {np.percentile(ratio, [25, 50, 75]).round(2).tolist()}; "
                  f"hate runs starting outside any topic run {100 * (ratio == 0).mean():.1f} %",
                  "  mean act read: " + "  ".join(f"{k} {np.mean(v):+.2f} (n {len(v)})" for k, v in grp.items())]
        ok = med["ratio"] >= 1.5 and med["topic_run"] > med["hate_run"]
        lines.append(f"  pre-check {'PASS' if ok else 'FAIL'} (needs median ratio >= 1.5 and median topic run > median hate run)")
        s["pass"] = bool(ok)
    (OUT / "table.txt").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
