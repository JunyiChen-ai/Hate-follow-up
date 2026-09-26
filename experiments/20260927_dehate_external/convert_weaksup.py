#!/usr/bin/env python3
"""Weakly supervised references on the 4 fps grid (README §2), evaluated by the shared evaluator.

Source: the Retrieval-hate DeHate run on uoa-lab2, copied to `runs/20260927_dehate_external/weaksup_src/<method>/seed_<s>/`
(scores.jsonl, frame_eval.json, frozen_config.json). Each method was tuned on DeHate validation with train-split
video labels and trained at three seeds. The branch is the one that run reports as its headline number
(its README §5.1): Fed-WSVAD `score_align`, MultiHateLoc `score_fused`, DSANet `score_mlp`, MACIL-SD `score_av`.

Mapping: second s of the 1 fps curve covers the 4 fps frames of [s, s + 1), so each value is repeated 4 times. The
curve is then cut or padded with its last value to ceil(duration * 4) frames, with the duration from
data/manifests/DeHate_test.jsonl, the grid the other methods here use.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20260927_dehate_external"
BRANCH = {"fed_wsvad_3client": "score_align", "multihateloc": "score_fused", "dsanet": "score_mlp",
          "macilsd": "score_av"}
SEEDS = (234, 2025, 3407)


def main():
    dur = {json.loads(l)["video_id"]: float(json.loads(l)["duration"])
           for l in open(ROOT / "data/manifests/DeHate_test.jsonl") if l.strip()}
    out = R / "weaksup"
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "predictions.jsonl"
    with open(pred, "w") as fh:
        for m, branch in BRANCH.items():
            for s in SEEDS:
                n_len_diff = []
                for line in open(R / "weaksup_src" / m / f"seed_{s}" / "scores.jsonl"):
                    r = json.loads(line)
                    a = np.repeat(np.asarray(r[branch], float), 4)
                    n = int(math.ceil(dur[r["video_id"]] * 4))
                    n_len_diff.append(n - len(a))
                    a = a[:n] if len(a) >= n else np.r_[a, np.full(n - len(a), a[-1])]
                    fh.write(json.dumps({"schema_version": 1, "method": f"{m}__seed{s}", "dataset": "DeHate",
                                         "video_id": r["video_id"], "duration": dur[r["video_id"]], "native_rate": 1.0,
                                         "score_curve": [float(x) for x in a], "intervals": [], "error": None,
                                         "extra": {"branch": branch, "source": f"weaksup_src/{m}/seed_{s}"}}) + "\n")
                d = np.asarray(n_len_diff)
                print(f"{m} seed {s}: {len(d)} videos; frames padded (4 fps grid minus repeated curve): "
                      f"min {d.min()} median {np.median(d):.0f} max {d.max()}")
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(out / "metrics.json"), "--datasets", "DeHate"],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    for p in json.load(open(out / "metrics.json"))["per_dataset"]:
        print(f"{p['method']:28s} {p['frame_ROC_AUC']:.4f} / {p['frame_PR_AUC']:.4f} / "
              f"{p['within_video_macro_ROC_AUC']:.4f}  n {p['n_videos_overlap']}")


if __name__ == "__main__":
    main()
