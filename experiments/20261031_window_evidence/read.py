#!/usr/bin/env python3
"""Adopted reader (user 2026-10-31): r6's shared prefix and verdict turn, with each window branch carrying its own
evidence (src/window_evidence.py): the window's 0.5 fps frames right before the visual question, the forced-aligned
word slice in the speech question. Qwen3-VL-8B, frozen. GPU inside Slurm only; no label file is read.

Per video every read below comes from one shared cache, so four read sets are written at the cost of one run:
  r6_replay = native visual + proportional speech   (must equal the stored r6 reads)
  frames    = in-window frames visual + proportional speech
  words     = native visual + aligned speech
  adopted   = in-window frames visual + aligned speech (the adopted configuration)
--smoke: the first two HateMM and HateClipSeg videos; also checks the frames / aligned reads against the stored
HateMM / HateClipSeg runs they were promoted from (placement_controls adjacent_local, vl_aligned).
Output: runs/20261031_window_evidence/<name>/{records/<ds>/<vid>.json, <arm>/predictions.jsonl, config.json, run.log}.
Resumable: videos with a record are not read again.
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
from src.mllm_judge import MODEL, Judge, yesno_question  # noqa: E402
from src.stance_cache import build, margin  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402
from src.window_evidence import WINDOW_S, aligned_slice, aligned_words, frames_margin, in_window, local_frames, window_prediction  # noqa: E402

EXP = ROOT / "runs/20261031_window_evidence"
NATIVE_FRAMES = 20
ARMS = {"r6_replay": ("native", "prop"), "frames": ("local", "prop"), "words": ("native", "aligned"), "adopted": ("local", "aligned")}
MANIFESTS = {"DeHate": ROOT / "data/manifests/DeHate_test.jsonl"}
ALL_TEST = ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"
R6 = {"DeHate": ROOT / "runs/20260927_dehate_external/reads_gridA/predictions.jsonl",
      "HateMM": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl", "HateClipSeg": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"}
SMOKE_REFS = {"frames_visual": ROOT / "runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local/predictions.jsonl",
              "aligned_speech": ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl"}
CODE = "experiments/20261031_window_evidence/read.py"
TOL = 1e-3


def stored(path, datasets):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(path)) if not r.get("error") and r["dataset"] in datasets}


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def read(j, row, segments, rec):
    begin = tick()
    torch.cuda.reset_peak_memory_stats()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    assert 0 < len(native) <= NATIVE_FRAMES
    frames = local_frames(row)
    cache, ctx = build(j, native, segments)
    wins = fixed_windows(float(row["duration"]), WINDOW_S)
    n = len(wins)
    reads = {k: [] for k in ("native", "local", "prop", "aligned")}
    traces = []
    for i, (t1, t2) in enumerate(wins):
        body = window_text(segments, t1, t2)
        abody = aligned_slice(rec, t1, t2, body)
        vq = yesno_question(i, n, t1, t2, body, "visual")
        v = margin(j, cache, ctx, vq)
        s = margin(j, cache, ctx, yesno_question(i, n, t1, t2, body, "speech")) if body.strip() else None
        a = margin(j, cache, ctx, yesno_question(i, n, t1, t2, abody, "speech")) if abody is not None and abody.strip() else None
        inside = in_window(frames, t1, t2)
        lv = frames_margin(j, cache, ctx, inside, vq) if inside else v
        for k, x in zip(("native", "local", "prop", "aligned"), (v, lv, s, a)):
            reads[k].append(x)
        traces.append(dict(i=i, bounds=[t1, t2], local_times=[t for t, _ in inside], text_prop=body, text_aligned=abody,
                           z_visual_native=v, z_visual_local=lv, z_speech_prop=s, z_speech_aligned=a))
    secs = tick() - begin
    preds = {}
    for arm, (vk, sk) in ARMS.items():
        p = window_prediction(row, ctx, reads[vk], reads[sk], f"window_evidence_{arm}", secs, CODE)
        preds[arm] = p
    del cache
    return dict(model=MODEL, host=socket.gethostname(), GT_read=False, global_margin=ctx["global_margin"], stance=ctx["stance"],
                stance_cache_tokens=ctx["stance_cache_tokens"], aligned_status=None if rec is None else rec["status"], windows=n,
                windows_with_frames=sum(bool(t["local_times"]) for t in traces), seconds=secs, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30,
                traces=traces, predictions=preds)


def check(bundle, row, r6, refs):
    """Max |difference| to the stored reads; None where no reference exists."""
    key = (row["dataset"], row["video_id"])
    out = {}
    ref = r6.get(key)
    if ref is not None:
        ws = ref["extra"]["windows"]
        assert len(ws) == bundle["windows"], (key, len(ws), bundle["windows"])
        out["r6_video"] = abs(ref["extra"]["z_video"] - bundle["global_margin"])
        out["r6_visual"] = max(abs(w["z_visual"] - t["z_visual_native"]) for w, t in zip(ws, bundle["traces"]))
        sp = [abs(w["z_speech"] - t["z_speech_prop"]) for w, t in zip(ws, bundle["traces"]) if "z_speech" in w and t["z_speech_prop"] is not None]
        out["r6_speech"] = max(sp, default=0.0)
        out["r6_speech_presence"] = sum(("z_speech" in w) != (t["z_speech_prop"] is not None) for w, t in zip(ws, bundle["traces"]))
    for name, field, tkey in (("frames_visual", "z_visual", "z_visual_local"), ("aligned_speech", "z_speech", "z_speech_aligned")):
        ref = refs.get(name, {}).get(key)
        if ref is None:
            continue
        ws = ref["extra"]["windows"]
        out[name] = max((abs(w[field] - t[tkey]) for w, t in zip(ws, bundle["traces"]) if field in w and t[tkey] is not None), default=0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["DeHate"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    datasets = ["HateMM", "HateClipSeg"] if a.smoke else a.datasets
    name = a.name or ("smoke" if a.smoke else "_".join(datasets))
    out = EXP / name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    rows = [r for ds in datasets for r in load_manifest(MANIFESTS.get(ds, ALL_TEST), [ds])]
    if a.smoke:
        rows = [r for ds in datasets for r in [x for x in rows if x["dataset"] == ds][:2]]
    import transformers
    import subprocess
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"], capture_output=True, text=True).stdout.strip()
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, arms={k: list(v) for k, v in ARMS.items()}, window_seconds=WINDOW_S,
               native_frames=NATIVE_FRAMES, GT_read=False, smoke=a.smoke, datasets=datasets, videos=len(rows), torch=torch.__version__,
               transformers=transformers.__version__, code=f"{CODE} + src/window_evidence.py; commit {commit}", command="python -u " + " ".join(sys.argv))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    r6 = {}
    for ds in datasets:
        r6.update(stored(R6[ds], [ds]))
    refs = {k: stored(p, datasets) for k, p in SMOKE_REFS.items()} if a.smoke else {}
    asr = {ds: load_asr(ds) for ds in datasets}
    words = {ds: aligned_words(ds) for ds in datasets}
    torch.manual_seed(0)
    j = Judge(MODEL)
    worst, errors, t0 = {}, 0, time.perf_counter()
    for k, row in enumerate(rows, 1):
        ds, vid = row["dataset"], row["video_id"]
        path = out / "records" / ds / f"{vid}.json"
        if path.exists():
            continue
        try:
            bundle = read(j, row, asr[ds].get(vid, []), words[ds].get(vid))
        except Exception as ex:  # noqa: BLE001
            errors += 1
            logging.exception("ERROR %s/%s %s", ds, vid, ex)
            continue
        bundle["check"] = check(bundle, row, r6, refs)
        for c, v in bundle["check"].items():
            worst[c] = max(worst.get(c, 0.0), v)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(bundle, allow_nan=False) + "\n")
        tmp.replace(path)
        logging.info("%d/%d %s/%s windows=%d with_frames=%d %.1fs peak=%.1fGiB check=%s elapsed=%.0fs", k, len(rows), ds, vid, bundle["windows"],
                     bundle["windows_with_frames"], bundle["seconds"], bundle["peak_GiB"], {c: round(v, 5) for c, v in bundle["check"].items()},
                     time.perf_counter() - t0)
    logging.info("worst differences to stored reads (this session): %s", worst)
    bad = {c: v for c, v in worst.items() if c != "r6_speech_presence" and v > TOL} | ({"r6_speech_presence": worst["r6_speech_presence"]} if worst.get("r6_speech_presence") else {})
    for arm in ARMS:
        recs = []
        for row in rows:
            path = out / "records" / row["dataset"] / f"{row['video_id']}.json"
            if path.exists():
                recs.append(json.loads(path.read_text())["predictions"][arm])
        (out / arm).mkdir(exist_ok=True)
        (out / arm / "config.json").write_text(json.dumps({**cfg, "arm": arm}, indent=2) + "\n")
        (out / arm / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    missing = len(rows) - len(recs)
    logging.info("records %d / %d, errors this session %d", len(recs), len(rows), errors)
    if missing or bad:
        logging.info("FAILED missing=%d differences_above_tolerance=%s", missing, bad)
        sys.exit(1)
    logging.info("DONE coverage=%d", len(rows))


if __name__ == "__main__":
    main()
