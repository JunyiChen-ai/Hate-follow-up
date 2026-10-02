#!/usr/bin/env python3
"""GT-free diagnostic reconstruction: preserve all within-video raw rankings."""
import copy
import json
import os
import socket
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/20261002_verdict_analysis/shift_reads"


def read(arm):
    p = ROOT / f"runs/20260910_spvl/mllm/q3vl-8b/{arm}/predictions.jsonl"
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, p.open()) if not r.get("error")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("host", socket.gethostname(), flush=True)
    (OUT / "run.pid").write_text(str(os.getpid()))
    full, no = read("full"), read("nostance")
    assert full.keys() == no.keys()
    offsets = []
    with (OUT / "predictions.jsonl").open("w") as f:
        for k in sorted(no):
            r = copy.deepcopy(no[k])
            ws, fs = r["extra"]["windows"], full[k]["extra"]["windows"]
            assert [(w["start"], w["end"]) for w in ws] == [(w["start"], w["end"]) for w in fs]
            assert r["extra"]["z_video"] == full[k]["extra"]["z_video"]
            shift = float(np.mean([x["z"] - y["z"] for x, y in zip(fs, ws)]))
            before = {m: np.array([w[m] for w in ws if m in w]) for m in ("z", "z_visual", "z_speech")}
            for w in ws:
                for m in ("z", "z_visual", "z_speech"):
                    if m in w:
                        w[m] += shift
            for m, old in before.items():
                new = np.array([w[m] for w in ws if m in w])
                assert np.array_equal(np.argsort(old, kind="stable"), np.argsort(new, kind="stable"))
                assert np.array_equal(np.diff(old) == 0, np.diff(new) == 0)
            r["method"] = "diagnostic_common_shift_only"
            r["code_path"] = "experiments/20261002_verdict_analysis/shift_control.py"
            r["extra"]["diagnostic_shift"] = shift
            t = np.arange(len(r["score_curve"])) / 4
            sc = np.zeros(len(t))
            for w in ws:
                sc[(t >= w["start"]) & (t < w["end"])] = w["z"]
            r["score_curve"] = sc.tolist()
            f.write(json.dumps(r) + "\n")
            offsets.append({"dataset": k[0], "video_id": k[1], "shift": shift})
    (OUT / "offsets.json").write_text(json.dumps(offsets, indent=2) + "\n")
    (OUT / "config.json").write_text(json.dumps({"host": socket.gethostname(), "date": "2026-10-02",
        "construction": "nostance + mean(full.max - nostance.max), same scalar for all branches/windows of a video",
        "purpose": "diagnostic reconstruction; both observed conditions needed; not a deployable method",
        "inputs": ["runs/20260910_spvl/mllm/q3vl-8b/full", "runs/20260910_spvl/mllm/q3vl-8b/nostance"],
        "GT_used": False, "all_raw_within_video_rankings_preserved": True}, indent=2) + "\n")
    print("SHIFT_READS_DONE", len(offsets), flush=True)


if __name__ == "__main__":
    main()
