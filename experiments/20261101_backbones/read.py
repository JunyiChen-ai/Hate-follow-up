#!/usr/bin/env python3
"""Backbone study with the adopted reader (README §2-§4). GPU inside Slurm only; no label file is read.

One backbone x one corpus per call:  read.py --tag <backbone> --dataset <HateMM|HateClipSeg> [--arms main ...]
Smoke (porting checks, README §3):    read.py --tag <backbone> --smoke   (first two HateMM and HateClipSeg videos)

Arms (arm-major order):
  main  the adopted reader on the backbone (src/window_evidence_family.read_video: prefix encoded once, verdict, stance
        turn, per window an isolated visual branch with the window's 0.5 fps frames and an isolated speech branch with
        the aligned words) and the per-window read alone (read_alone: per window one plain call with only that window's
        frames and aligned words and the joint question). Writes full/ and alone/ predictions.
  no_transcript, no_prefix_frames, no_stance, asr_windows, joint
        the reader-ablation arms A1, A2, A3, A4, A6 (experiments/20261101_reader_ablation/README.md §2;
        src/window_evidence_family.read_arm).
The first --check-plain videos of the main arm also get the plain-call check (check b); --smoke with q3vl-8b also
compares every main read with the stored adopted reads (check a). Arms whose prefix and verdict equal the main
reader's (no_stance, asr_windows, joint) must give the same z_video as this run's main record of the video.
Output runs/20261101_backbones/<tag>/<dataset>/ (smoke: runs/20261101_backbones/smoke/<tag>/):
records/<arm>/<ds>/<vid>.json, <full|alone|arm>/predictions.jsonl, config.json, run.log, run.pid. Resumable:
(arm, video) pairs with a record are not read again.
"""
import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import window_evidence_family as F  # noqa: E402
from src.mllm_judge import Judge  # noqa: E402
from src.reader_arms import record as arm_record  # noqa: E402
from src.video_inputs import frame_paths, load_asr, load_manifest  # noqa: E402
from src.window_evidence import aligned_words, local_frames, window_prediction  # noqa: E402

EXP = ROOT / "runs/20261101_backbones"
CODE = "experiments/20261101_backbones/read.py"
MODELS = {"q3vl-2b": "Qwen/Qwen3-VL-2B-Instruct", "q3vl-4b": "Qwen/Qwen3-VL-4B-Instruct", "q3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
          "q3vl-32b": "Qwen/Qwen3-VL-32B-Instruct", "q25vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct", "internvl35-8b": "OpenGVLab/InternVL3_5-8B-HF",
          "llava-ov-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "gemma3-12b": "google/gemma-3-12b-it"}
ALL_TEST = ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"
ADOPTED = ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl"
NATIVE_FRAMES = 20
GATE_NATS = 3.0   # 2026-09-10 family study (experiments/20260910_spvl/README.md §11): cached branch vs plain call, max |dz| < 3
ADOPTED_TOL = 1e-3  # window_evidence smoke tolerance; the adopted path is expected to reproduce with 0.0
ARMS = ("main",) + F.ARMS
SAME_VERDICT = {"no_stance", "asr_windows", "joint"}


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def adopted_check(rec, ref):
    ws = ref["extra"]["windows"]
    assert len(ws) == len(rec["traces"])
    out = dict(video=abs(ref["extra"]["z_video"] - rec["global_margin"]),
               visual=max(abs(w["z_visual"] - t["z_visual"]) for w, t in zip(ws, rec["traces"])),
               speech=max((abs(w["z_speech"] - t["z_speech"]) for w, t in zip(ws, rec["traces"]) if "z_speech" in w and t["z_speech"] is not None), default=0.0),
               speech_presence=sum(("z_speech" in w) != (t["z_speech"] is not None) for w, t in zip(ws, rec["traces"])))
    return out


@torch.no_grad()
def read(j, tag, row, segments, words, check_plain):
    begin = tick()
    torch.cuda.reset_peak_memory_stats()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    assert 0 < len(native) <= NATIVE_FRAMES
    frames = local_frames(row)
    ctx, traces, cache = F.read_video(j, row, native, frames, segments, words)
    del cache
    secs_full = tick() - begin
    plain = F.plain_check(j, row, frames, segments, ctx, traces) if check_plain else None
    t0 = tick()
    alone = F.read_alone(j, row, frames, segments, words)
    secs_alone = tick() - t0
    for t, z in zip(traces, alone):
        t["z_alone"] = z
    full = window_prediction(row, ctx, [t["z_visual"] for t in traces], [t["z_speech"] for t in traces], f"backbones_full_{tag}", secs_full, CODE)
    return dict(model=j.model_id, tag=tag, family=j.family, host=socket.gethostname(), GT_read=False, global_margin=ctx["global_margin"],
                stance=ctx["stance"], prefix_tokens=ctx["prefix_tokens"], stance_cache_tokens=ctx["stance_cache_tokens"],
                image_tokens_per_frame=(ctx.get("image_counts") or [None])[0], aligned_status=None if words is None else words["status"],
                windows=len(traces), windows_with_frames=sum(bool(t["local_times"]) for t in traces), seconds_full=secs_full,
                seconds_alone=secs_alone, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, plain_check=plain, traces=traces,
                predictions=dict(full=full, alone=F.alone_prediction(row, alone, secs_alone, CODE)))


@torch.no_grad()
def read_arm(j, tag, arm, row, segments, words):
    begin = tick()
    torch.cuda.reset_peak_memory_stats()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    assert 0 < len(native) <= NATIVE_FRAMES
    frames = local_frames(row)
    ctx, wins, reads = F.read_arm(j, arm, row, native, frames, segments, words)
    secs = tick() - begin
    pred = arm_record(row, ctx, wins, reads, f"backbones_{arm}_{tag}", secs, CODE, segment_windows=arm == "asr_windows")
    return dict(model=j.model_id, tag=tag, arm=arm, family=j.family, host=socket.gethostname(), GT_read=False, global_margin=ctx["global_margin"],
                stance=ctx["stance"], prefix_tokens=ctx["prefix_tokens"], stance_cache_tokens=ctx["stance_cache_tokens"], windows=len(wins),
                seconds_full=secs, seconds_alone=0.0, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, windows_with_frames=None,
                plain_check=None, reads=reads, prediction=pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", choices=["HateMM", "HateClipSeg"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-plain", type=int, default=1, help="videos (from the start) that get the plain-call check")
    ap.add_argument("--arms", nargs="+", default=["main"], choices=ARMS)
    a = ap.parse_args()
    assert a.smoke or a.dataset, "--dataset or --smoke"
    datasets = ["HateMM", "HateClipSeg"] if a.smoke else [a.dataset]
    out = EXP / "smoke" / a.tag if a.smoke else EXP / a.tag / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    rows = [r for ds in datasets for r in load_manifest(ALL_TEST, [ds])]
    if a.smoke:
        rows = [r for ds in datasets for r in [x for x in rows if x["dataset"] == ds][:2]]
    check_plain = len(rows) if a.smoke else a.check_plain
    import transformers
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src/window_evidence_family.py", CODE], capture_output=True, text=True).stdout.strip()
    arms = list(ARMS) if a.smoke else a.arms
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODELS[a.tag], tag=a.tag, datasets=datasets, videos=len(rows), smoke=a.smoke, arms=arms,
               window_seconds=8.0, native_frames=NATIVE_FRAMES, local_fps=0.5, GT_read=False, check_plain_videos=check_plain, gate_nats=GATE_NATS,
               torch=torch.__version__, transformers=transformers.__version__, hf_home=os.environ.get("HF_HOME"),
               code=f"{CODE} + src/window_evidence_family.py (+ src/window_evidence.py, src/stance_cache.py, src/mllm_judge.py); commit {commit}"
               + (f"; uncommitted: {dirty}" if dirty else ""), command="python -u " + " ".join(sys.argv))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    asr = {ds: load_asr(ds) for ds in datasets}
    words = {ds: aligned_words(ds) for ds in datasets}
    ref = {}
    if a.smoke and a.tag == "q3vl-8b":
        ref = {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(ADOPTED))}
    torch.manual_seed(0)
    j = Judge(MODELS[a.tag])
    logging.info("family %s same_turn %s list_content %s mrope %s", j.family, j.same_turn, j.list_content, F.is_mrope(j))
    worst, errors, t0, bad = {}, 0, time.perf_counter(), []
    jobs = [(arm, row) for arm in arms for row in rows]
    for k, (arm, row) in enumerate(jobs, 1):
        ds, vid = row["dataset"], row["video_id"]
        path = out / "records" / arm / ds / f"{vid}.json"
        if path.exists():
            continue
        try:
            if arm == "main":
                rec = read(j, a.tag, row, asr[ds].get(vid, []), words[ds].get(vid), rows.index(row) < check_plain)
            else:
                rec = read_arm(j, a.tag, arm, row, asr[ds].get(vid, []), words[ds].get(vid))
        except Exception as ex:  # noqa: BLE001
            errors += 1
            logging.exception("ERROR %s %s/%s %s", arm, ds, vid, ex)
            torch.cuda.empty_cache()
            continue
        if arm == "main" and (ds, vid) in ref:
            rec["adopted_check"] = adopted_check(rec, ref[(ds, vid)])
        checks = {}
        if rec["plain_check"]:
            p = rec["plain_check"]
            checks.update(plain_max_dz=max(p["verdict_dz"], p["visual_max_dz"], p["speech_max_dz"]), plain_spearman=p["spearman"])
            if checks["plain_max_dz"] >= GATE_NATS or not p["token_seam_identical"]:
                bad.append((arm, ds, vid, "plain", p))
        if rec.get("adopted_check"):
            c = rec["adopted_check"]
            checks.update(adopted_max=max(c["video"], c["visual"], c["speech"]), adopted_presence=c["speech_presence"])
            if checks["adopted_max"] > ADOPTED_TOL or c["speech_presence"]:
                bad.append((arm, ds, vid, "adopted", c))
        main_path = out / "records" / "main" / ds / f"{vid}.json"
        if arm in SAME_VERDICT and main_path.exists():
            checks["z_video_vs_main"] = abs(json.loads(main_path.read_text())["global_margin"] - rec["global_margin"])
            if checks["z_video_vs_main"] > ADOPTED_TOL:
                bad.append((arm, ds, vid, "z_video_vs_main", checks["z_video_vs_main"]))
        for c, v in checks.items():
            if v is not None:
                key = f"{arm}:{c}"
                worst[key] = max(worst.get(key, v), v) if c != "plain_spearman" else min(worst.get(key, v), v)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(rec, allow_nan=False) + "\n")
        tmp.replace(path)
        logging.info("%d/%d %s %s/%s windows=%d with_frames=%s full=%.1fs alone=%.1fs peak=%.1fGiB prefix=%d checks=%s plain=%s elapsed=%.0fs",
                     k, len(jobs), arm, ds, vid, rec["windows"], rec["windows_with_frames"], rec["seconds_full"], rec["seconds_alone"], rec["peak_GiB"],
                     rec["prefix_tokens"], {c: (round(v, 5) if v is not None else None) for c, v in checks.items()}, rec["plain_check"], time.perf_counter() - t0)
        if bad:
            break
    logging.info("worst checks (this session): %s", worst)
    recs, missing = {}, 0
    for arm in arms:
        for row in rows:
            path = out / "records" / arm / row["dataset"] / f"{row['video_id']}.json"
            if not path.exists():
                missing += 1
                continue
            r = json.loads(path.read_text())
            for name, pred in (r["predictions"].items() if arm == "main" else [(arm, r["prediction"])]):
                recs.setdefault(name, []).append(pred)
    for name, rr in recs.items():
        (out / name).mkdir(exist_ok=True)
        (out / name / "config.json").write_text(json.dumps({**cfg, "arm": name}, indent=2) + "\n")
        (out / name / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
    logging.info("records %s, missing %d, errors this session %d", {n: len(r) for n, r in recs.items()}, missing, errors)
    if missing or bad:
        logging.info("FAILED missing=%d check_failures=%s", missing, bad)
        sys.exit(1)
    logging.info("DONE videos=%d arms=%s", len(rows), arms)


if __name__ == "__main__":
    main()
