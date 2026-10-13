#!/usr/bin/env python3
"""Candidate 43 follow-up control (README §11, declared before its run): the forced-aligned word slice under the
Qwen3-VL-8B reader of r6. Only the speech reads are new: per window the r6 speech wording with the aligned slice
(no read when the slice is empty, as r6). The visual reads come from the stored runs, so the arms are
  vl_text_aligned = max(adjacent_local's visual_local read, aligned speech read)   -> compare with adjacent_local
  vl_r6_aligned   = max(r6's native visual read, aligned speech read)              -> compare with r6_bma
Replay check: on the first two videos of each corpus the proportional speech read is recomputed and compared with
r6's stored z_speech (same reader, same text). GPU inside Slurm only; no label file is read.
Output: runs/20261014_m1_audio_bound/vl_aligned/{<arm>/predictions.jsonl, records/<ds>/<vid>.json, run.log}.
"""
import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_audio import WINDOW_S, aligned_slice, aligned_words, prediction, selected_rows  # noqa: E402
from src.mllm_judge import MODEL as VL_MODEL, Judge, yesno_question  # noqa: E402
from src.stance_cache import build, margin  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, window_text  # noqa: E402

EXP = ROOT / "runs/20261014_m1_audio_bound"
R6 = ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"
ADJ = ROOT / "runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local/predictions.jsonl"
NATIVE_FRAMES = 20
ARMS = ("vl_text_aligned", "vl_r6_aligned")
IMPLEMENTATION = "candidate 43 follow-up: aligned word slice under the Qwen3-VL-8B reader; 2026-10-14"


def tick():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def stored(path):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(path)) if not r.get("error")}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="vl_aligned")
    a = ap.parse_args()
    out = EXP / a.name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    import transformers
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=VL_MODEL, implementation=IMPLEMENTATION, arms=list(ARMS), window_seconds=WINDOW_S,
               native_frames=NATIVE_FRAMES, GT_read=False, smoke=a.smoke, datasets=a.datasets, torch=torch.__version__, transformers=transformers.__version__,
               visual_sources={"vl_text_aligned": str(ADJ.relative_to(ROOT)), "vl_r6_aligned": str(R6.relative_to(ROOT))},
               command="python -u " + " ".join(sys.argv), source="experiments/20261014_m1_audio_bound/measure_vl_aligned.py; 2026-10-14")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    rows = selected_rows(a.datasets, a.smoke)
    r6, adj = stored(R6), stored(ADJ)
    asr = {ds: load_asr(ds) for ds in a.datasets}
    words = {ds: aligned_words(ds) for ds in a.datasets}
    torch.manual_seed(0)
    j = Judge(VL_MODEL)
    results = {arm: [] for arm in ARMS}
    replay_videos = {ds: [x["video_id"] for x in rows if x["dataset"] == ds][:2] for ds in a.datasets}
    for ordinal, row in enumerate(rows, 1):
        ds, vid = row["dataset"], row["video_id"]
        begin = tick()
        segments = asr[ds].get(vid, [])
        rec = words[ds].get(vid)
        native = frame_paths(ds, vid, NATIVE_FRAMES)
        cache, ctx = build(j, native, segments)
        wins = fixed_windows(float(row["duration"]), WINDOW_S)
        n = len(wins)
        ref_r6, ref_adj = r6[(ds, vid)]["extra"]["windows"], adj[(ds, vid)]["extra"]["windows"]
        assert len(ref_r6) == len(ref_adj) == n, (vid, len(ref_r6), len(ref_adj), n)
        speech, traces, replay = [], [], []
        for i, (t1, t2) in enumerate(wins):
            body = window_text(segments, t1, t2)
            abody = aligned_slice(rec, t1, t2, body)
            assert (ref_r6[i].get("z_speech") is not None) == bool(body.strip()), (vid, i)
            s = margin(j, cache, ctx, yesno_question(i, n, t1, t2, abody, "speech")) if abody is not None and abody.strip() else None
            speech.append(s)
            t = dict(i=i, bounds=[t1, t2], text_prop=body, text_aligned=abody, z_aligned=s, z_prop_stored=ref_r6[i].get("z_speech"))
            if vid in replay_videos[ds] and body.strip():
                zp = margin(j, cache, ctx, yesno_question(i, n, t1, t2, body, "speech"))
                t["z_prop_replay"] = zp
                replay.append(abs(zp - ref_r6[i]["z_speech"]))
            traces.append(t)
        secs = tick() - begin
        ctx2 = dict(ctx, prefix_tokens=ctx.get("prefix_tokens"))
        visual = {"vl_text_aligned": [w["z_visual"] for w in ref_adj], "vl_r6_aligned": [w["z_visual"] for w in ref_r6]}
        preds = {arm: prediction(row, ctx2, visual[arm], speech, arm, secs) for arm in ARMS}
        for arm in ARMS:
            preds[arm]["code_path"] = "experiments/20261014_m1_audio_bound/measure_vl_aligned.py"
            results[arm].append(preds[arm])
        bundle = dict(implementation=IMPLEMENTATION, model=VL_MODEL, host=socket.gethostname(), GT_read=False, stance=ctx["stance"], stance_r6=r6[(ds, vid)]["extra"]["stance"],
                      global_margin=ctx["global_margin"], aligned_status=None if rec is None else rec["status"], windows=n, seconds=secs,
                      replay_max_abs=max(replay) if replay else None, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, traces=traces, predictions=preds)
        path = out / "records" / ds / f"{vid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, allow_nan=False) + "\n")
        logging.info("%d/%d %s/%s windows=%d aligned_reads=%d %.0fs stance=%s/%s replay_max_abs=%s", ordinal, len(rows), ds, vid, n, sum(s is not None for s in speech), secs,
                     ctx["stance"], bundle["stance_r6"], bundle["replay_max_abs"])
        del cache
    for arm, rr in results.items():
        (out / arm).mkdir(exist_ok=True)
        (out / arm / "config.json").write_text(json.dumps({**cfg, "arm": arm}, indent=2) + "\n")
        (out / arm / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
    logging.info("DONE coverage=%d", len(rows))


if __name__ == "__main__":
    main()
