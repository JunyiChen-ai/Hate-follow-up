#!/usr/bin/env python3
"""LAVAD curves -> exact-cohort predictions and metrics (HateClipSeg completion, DeHate).

Input: 1 fps curves `curves/<DS>/<vid>.npz`, key `base` (stage-06 refined score, the reported row of the
HateMM / HateClipSeg re-evaluation), written by `lavad_chain.py curves`.

  HateClipSeg  117 cohort videos keep the campaign's stored curves (`Retrieval-hate/archive/idea-stage/
               repro_lavad/curves/HateClipSeg/`, read only, the same files `convert_rh_baselines.py` read);
               `bit_ZaY0S1anrdep` takes the rerun curve from `runs/20261008_baselines/lavad/curves/HateClipSeg/`.
               Allowed only if `rerun_check.json` shows the rerun reproduces the campaign's curves on the 5
               overlap videos (run_plan.md L1b); otherwise pass `--all-rerun` after a full HateClipSeg rerun.
  DeHate       all cohort videos from the rerun curves.

Fallbacks (run_plan.md §1.3): F3 = a 1 fps sample with no score (every one of its 10 neighbour summaries was a
Llama-2 refusal) takes the value of the nearest scored sample of the same video (ties: the earlier one); F2 (in
`exact_cohort.finalize`) if a video has no scored sample at all. Then 1 fps -> 4 fps by piecewise-constant
broadcast (frame i <- sample floor(i/4)), frames past the last sample hold its value. No labels are read.
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

CODE_PATH = "experiments/20261008_baselines/lavad/finalize.py"
RUN_DIR = ec.OUT_ROOT / "lavad"
RERUN_CURVES = RUN_DIR / "curves"
CAMPAIGN_CURVES = Path("/home/jehc223/Retrieval-hate/archive/idea-stage/repro_lavad/curves")  # read only
RERUN_VIDEO_HCS = "bit_ZaY0S1anrdep"
RATE = 1.0


def fill_nearest(c: np.ndarray) -> tuple[np.ndarray, int]:
    c = np.asarray(c, dtype=np.float64).copy()
    bad = ~np.isfinite(c)
    if not bad.any() or bad.all():
        return c, 0
    good = np.flatnonzero(~bad)
    for i in np.flatnonzero(bad):
        j = np.searchsorted(good, i)
        cands = [good[k] for k in (j - 1, j) if 0 <= k < len(good)]
        c[i] = c[min(cands, key=lambda g: (abs(g - i), g))]
    return c, int(bad.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["HateClipSeg", "DeHate"])
    ap.add_argument("--all-rerun", action="store_true", help="HateClipSeg: take every video from the rerun")
    a = ap.parse_args()
    ds = a.dataset
    out = RUN_DIR / ds
    out.mkdir(parents=True, exist_ok=True)
    log = ec.RunLog(out / "run.log")
    (out / "run.pid").write_text(f"{os.getpid()}\n")
    log(f"code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
    source = {}
    if ds == "HateClipSeg" and not a.all_rerun:
        chk = json.loads((RUN_DIR / "HateClipSeg_rerun_check.json").read_text())
        if not chk.get("reproduced"):
            log("FAILED rerun_check.json: the rerun does not reproduce the campaign curves on the overlap videos; "
                "rerun the whole corpus and pass --all-rerun")
            return 7
        for v in ec.cohort(ds):
            source[v] = (RERUN_CURVES if v == RERUN_VIDEO_HCS else CAMPAIGN_CURVES) / ds / f"{v}.npz"
    else:
        for v in ec.cohort(ds):
            source[v] = RERUN_CURVES / ds / f"{v}.npz"
    dur = ec.durations(ds)
    curves, extra, fail = {}, {}, {}
    n_f3, n_tail = 0, 0
    for v, p in source.items():
        if not p.exists():
            fail[v] = "no curve (stage failed or no frames)"
            continue
        z = np.load(p)
        c = np.asarray(z["base"], dtype=np.float64)
        if not np.isfinite(c).any():
            fail[v] = "no scored 1 fps sample"
            continue
        c, nf = fill_nearest(c)
        T = ec.curve_length(dur[v])
        tail = ec.tail_frames(len(c), RATE, T)
        n_f3 += nf
        n_tail += tail
        curves[v] = ec.broadcast_to_4fps(c, RATE, T)
        extra[v] = {"source": str(p), "native_samples": int(len(c)), "tail_hold_frames": int(tail)}
        if nf:
            extra[v]["fallback"] = {"code": "F3", "n_native_samples_filled": nf,
                                    "rule": "nearest scored 1 fps sample of the same video"}
            log(f"F3 {ds}/{v}: {nf} unscored 1 fps samples filled from the nearest scored sample")
    rep = ec.finalize("lavad_base", ds, out, curves, native_rate=RATE, code_path=CODE_PATH, log=log, extra=extra,
                      failures=fail, notes={"f3_native_samples": n_f3, "tail_hold_frames_total": n_tail,
                                            "sources": ("campaign curves for 117 videos + rerun for "
                                                        f"{RERUN_VIDEO_HCS}" if ds == "HateClipSeg"
                                                        and not a.all_rerun else "rerun curves")},
                      config={"variant": "base (LAVAD stage-06 refined score, verbatim law-enforcement prompt)",
                              "pipeline": "BLIP-2 opt-6.7b-coco captions at 1 fps, ImageBind caption cleaning, "
                                          "Llama-2-13b-chat NF4 greedy summary + score, ImageBind refinement "
                                          "(10 neighbours, softmax(similarity) weights)",
                              "native_rate": "1 fps", "grid": "frame i <- sample floor(i/4); tail holds last",
                              "fallback_F3": "unscored 1 fps sample <- nearest scored sample of the same video"})
    log("DONE" if rep.get("exact_test_set") else "FAILED not evaluated")
    return 0 if rep.get("exact_test_set") else 6


if __name__ == "__main__":
    raise SystemExit(main())
