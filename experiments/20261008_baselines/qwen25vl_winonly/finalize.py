#!/usr/bin/env python3
"""Qwen2.5-VL-7B per-window scoring without context (`winonly` arm of experiments/20260910_spvl/spvl.py,
run_plan.md L8) -> exact-cohort predictions and metrics.

Sources (spvl.py and compose.py outputs, used unchanged):
  HateMM, HateClipSeg  runs/20260910_spvl/mllm/q25vl-7b/winonly/            (2026-09-10, uoa-lab3)
  DeHate               runs/20261008_baselines/qwen25vl_winonly/DeHate_spvl/ (this experiment, uoa-lab2)

Two read-outs of the same forward passes, each evaluated separately:
  raw          `predictions.jsonl`: every 4 fps frame holds the Yes/No log-odds of the 8 s window containing its
               centre (the model's own per-window score, no post-processing).
  ispvl_rrank  `predictions_ispvl_rrank.jsonl` (compose.py --intercept spvl --residual rank): whole-video
               log-odds + zero-mean centred rank of the window scores inside the video. This is the "per-window
               alone" row of runs/20260910_spvl/mllm_table.md. In this arm the whole-video question sees no video
               content (no frames, no transcript), so the intercept is one constant for all videos and the
               read-out keeps only the within-video order.
Outputs: runs/20261008_baselines/qwen25vl_winonly/<read-out>/<DS>/. No labels are read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402

CODE_PATH = "experiments/20261008_baselines/qwen25vl_winonly/finalize.py"
BASE = ec.OUT_ROOT / "qwen25vl_winonly"
SOURCE = {"HateMM": REPO / "runs/20260910_spvl/mllm/q25vl-7b/winonly",
          "HateClipSeg": REPO / "runs/20260910_spvl/mllm/q25vl-7b/winonly",
          "DeHate": BASE / "DeHate_spvl"}
FILES = {"raw": "predictions.jsonl", "ispvl_rrank": "predictions_ispvl_rrank.jsonl"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg", "DeHate"])
    a = ap.parse_args()
    rc = 0
    for ds in a.datasets:
        src_dir = SOURCE[ds]
        cfg_src = json.loads((src_dir / "config.json").read_text())
        for readout, fname in FILES.items():
            out = BASE / readout / ds
            log = ec.RunLog(out / "run.log")
            (out / "run.pid").write_text(f"{os.getpid()}\n")
            log(f"code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
            log(f"source {src_dir / fname} (spvl.py run on {cfg_src.get('host')}, {cfg_src.get('date')})")
            rows = {}
            with open(src_dir / fname) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r["dataset"] == ds:
                        rows[r["video_id"]] = r          # last row per video, as the evaluator keeps it
            curves, fail, extra = {}, {}, {}
            for v in ec.cohort(ds):
                r = rows.get(v)
                if r is None or r.get("error") or not r.get("score_curve"):
                    fail[v] = (r or {}).get("error") or "no row"
                    continue
                curves[v] = np.asarray(r["score_curve"], dtype=np.float64)
                ex = r.get("extra") or {}
                extra[v] = {"z_video": ex.get("z_video"), "n_windows": ex.get("n_windows")}
            zv = sorted({round(float(e["z_video"]), 4) for e in extra.values() if e["z_video"] is not None})
            rep = ec.finalize(f"qwen25vl7b_winonly_{readout}", ds, out, curves, native_rate=ec.FPS,
                              code_path=CODE_PATH, log=log, extra=extra, failures=fail,
                              notes={"source": str(src_dir / fname), "distinct_z_video": zv[:5],
                                     "n_distinct_z_video": len(zv)},
                              config={"readout": readout, "source": str((src_dir / fname).relative_to(REPO)),
                                      "spvl_config": {k: cfg_src.get(k) for k in (
                                          "model", "frames", "no_transcript_context", "windows", "window_seconds",
                                          "branches", "window_question", "stance", "isolation", "method_name",
                                          "host", "date", "transformers", "torch")},
                                      "arm": "winonly: --isolation cache --frames 0 --no-transcript-context "
                                             "--branches joint --window-question rules --windows fixed "
                                             "--window-seconds 8"})
            rc |= 0 if rep.get("exact_test_set") else 6
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
