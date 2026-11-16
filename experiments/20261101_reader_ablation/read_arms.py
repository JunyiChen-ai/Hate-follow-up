#!/usr/bin/env python3
"""Reader ablation arms of the adopted window-evidence reader (README §2). Qwen3-VL-8B, frozen; GPU inside Slurm only;
no label file is read.

Arms (src/reader_arms.py + the adopted src/stance_cache, src/window_evidence):
  base              the adopted reader (must equal the stored adopted reads; --smoke checks it)
  no_transcript     A1 prefix without the whole transcript
  no_prefix_frames  A2 prefix without the 20 frames (window visual branches keep their 0.5 fps frames)
  no_stance         A3 verdict read, not written back; windows follow the prefix directly
  asr_windows       A4 Whisper segments as windows (gaps unscored)
  no_isolation      A5 branches see each other in order; the model's Yes/No written back after every branch
  joint             A6 one branch per window: window frames + aligned words, joint question; key z_joint
  no_window_evidence  the older window inputs (README §8): visual question alone (no in-window frames), speech question
                    with the proportional transcript slice; must equal the stored older-reader reads
No-verdict-turn family (README §8, 2026-11-16): `ns_<arm>` = the arm with the verdict read but not written back
(stance=False, as no_stance), for no_transcript, no_prefix_frames, asr_windows, no_isolation, joint,
no_window_evidence. The family's base is the stored `no_stance` arm.
Combined arms (README §9, 2026-11-17): `<part>__<part>` changes two reader parts at once (verdict turn kept):
joint__no_isolation (one joint branch per window in the growing conversation), joint__no_window_evidence (one joint
branch with the older window inputs), no_isolation__no_window_evidence (growing conversation with the older inputs),
joint__no_isolation__no_window_evidence (all three: one joint branch per window, older inputs, growing conversation).
Arm-major order (every video of one arm, then the next arm). Output runs/20261101_reader_ablation/<name>/
{records/<arm>/<ds>/<vid>.json, <arm>/predictions.jsonl, <arm>/config.json, config.json, run.log, run.pid}.
Resumable: (arm, video) pairs with a record are not read again.
--smoke: first two videos of HateMM, HateClipSeg, DeHate, every arm of the 2026-11-01 set (or the arms given with
--arms), plus every listed no_isolation / ns_no_isolation arm on the longest HateMM video.
Timing run (main-session request 2026-11-01): --arms base --name timing_<corpus>, one corpus per job; each record keeps
seconds (whole video, after model load), prefix_seconds (prefix forward + verdict + stance turn), window_seconds,
prefix_tokens and peak_GiB; the base check against the stored adopted reads runs on every video.
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
from src.mllm_judge import MODEL, Judge, yesno_question  # noqa: E402
from src.reader_arms import SequentialReader, asr_windows, build_context, record  # noqa: E402
from src.stance_cache import margin  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402
from src.window_evidence import WINDOW_S, aligned_slice, aligned_words, frames_margin, in_window, local_frames  # noqa: E402

EXP = ROOT / "runs/20261101_reader_ablation"
NATIVE_FRAMES = 20
ARMS_V1 = ["base", "no_transcript", "no_prefix_frames", "no_stance", "asr_windows", "no_isolation", "joint"]  # 2026-11-01 set
NS_KINDS = ["no_transcript", "no_prefix_frames", "asr_windows", "no_isolation", "joint", "no_window_evidence"]
COMBINED = ["joint__no_isolation", "joint__no_window_evidence", "no_isolation__no_window_evidence",   # README §9
            "joint__no_isolation__no_window_evidence"]
ARMS = ARMS_V1 + ["no_window_evidence"] + ["ns_" + k for k in NS_KINDS] + COMBINED
# prefix and verdict question as the base (the verdict is read before anything is written back or cropped)
SAME_VERDICT = {"base", "no_stance", "asr_windows", "no_isolation", "joint", "no_window_evidence",
                "ns_asr_windows", "ns_no_isolation", "ns_joint", "ns_no_window_evidence", *COMBINED}
MANIFESTS = {"HateMM": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", "HateClipSeg": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "DeHate": ROOT / "data/manifests/DeHate_test.jsonl"}
STORED = {"HateMM": ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl",
          "HateClipSeg": ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl",
          "DeHate": ROOT / "runs/20261031_window_evidence/DeHate/adopted/predictions.jsonl"}
# the older reader's reads (r6: visual question alone, proportional speech slice), reference of no_window_evidence
OLD = {"HateMM": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl", "HateClipSeg": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl",
       "DeHate": ROOT / "runs/20261031_window_evidence/DeHate/r6_replay/predictions.jsonl"}
CODE = "experiments/20261101_reader_ablation/read_arms.py"
TOL = 1e-3


def stored(path, ds):
    return {r["video_id"]: r for r in map(json.loads, open(path)) if not r.get("error") and r["dataset"] == ds}


def arm_reads(arm, ds):
    """Stored reads of a 2026-11-01 arm (runs/20261101_reader_ablation/<corpus>/<arm>), reference of the ns family."""
    return stored(EXP / ds / arm / "predictions.jsonl", ds)


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def visual_read(j, cache, ctx, frames, q):
    """Adopted visual branch: window frames next to the question; a window without a frame asks the question alone."""
    return frames_margin(j, cache, ctx, frames, q) if frames else margin(j, cache, ctx, q)


@torch.no_grad()
def read(j, arm, row, segments, rec):
    begin = tick()
    torch.cuda.reset_peak_memory_stats()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    assert 0 < len(native) <= NATIVE_FRAMES
    frames = local_frames(row)
    ns = arm.startswith("ns_")          # no-verdict-turn family: the arm `kind` without the stance turn
    kind = arm[3:] if ns else arm
    parts = set(kind.split("__"))       # a combined arm changes two parts (README §9); a single arm has one part
    old_inputs = "no_window_evidence" in parts
    cache, ctx = build_context(j, native, segments, with_context="no_transcript" not in parts,
                               with_frames="no_prefix_frames" not in parts, stance=not ns and "no_stance" not in parts)
    prefix_secs = tick() - begin  # 20 frames + transcript prefix forward, verdict read and stance turn (timing only)
    if "asr_windows" in parts:
        segs = asr_windows(segments)
        wins = [(s, e) for s, e, _ in segs]
        bodies = [aligned_slice(rec, s, e, t) for s, e, t in segs]
    else:
        wins = fixed_windows(float(row["duration"]), WINDOW_S)
        if old_inputs:   # the older reader: proportional transcript slice, no forced alignment
            bodies = [window_text(segments, t1, t2) for t1, t2 in wins]
        else:
            bodies = [aligned_slice(rec, t1, t2, window_text(segments, t1, t2)) for t1, t2 in wins]
    n = len(wins)
    seq = SequentialReader(j, cache, ctx) if "no_isolation" in parts else None
    reads, traces = [], []
    for i, ((t1, t2), body) in enumerate(zip(wins, bodies)):
        inside = [] if old_inputs else in_window(frames, t1, t2)   # older reader: question alone
        has_speech = body is not None and body.strip() != ""
        r = {}
        if "joint" in parts:
            q = yesno_question(i, n, t1, t2, body, "joint")
            r["z_joint"] = seq.read(inside, q) if seq is not None else visual_read(j, cache, ctx, inside, q)
        else:
            vq = yesno_question(i, n, t1, t2, body, "visual")
            sq = yesno_question(i, n, t1, t2, body, "speech") if has_speech else None
            if seq is not None:
                r["z_visual"] = seq.read(inside, vq)
                if sq is not None:
                    r["z_speech"] = seq.read([], sq)
            else:
                r["z_visual"] = visual_read(j, cache, ctx, inside, vq)
                if sq is not None:
                    r["z_speech"] = margin(j, cache, ctx, sq)
        reads.append(r)
        traces.append(dict(i=i, bounds=[t1, t2], local_times=[t for t, _ in inside], words=len(body.split()) if has_speech else 0, **r))
    secs = tick() - begin
    pred = record(row, ctx, wins, reads, f"reader_ablation_{arm}", secs, CODE, segment_windows="asr_windows" in parts)
    out = dict(model=MODEL, host=socket.gethostname(), arm=arm, GT_read=False, global_margin=ctx["global_margin"], stance=ctx["stance"],
               prefix_tokens=ctx["prefix_tokens"], stance_cache_tokens=ctx["stance_cache_tokens"],
               final_cache_tokens=cache.get_seq_length(), aligned_status=None if rec is None else rec["status"], windows=n,
               windows_with_frames=sum(bool(t["local_times"]) for t in traces), seconds=secs, prefix_seconds=prefix_secs,
               window_seconds=secs - prefix_secs, branches=sum(len(r) for r in reads),
               peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, traces=traces, prediction=pred)
    del cache, seq
    return out


def window_diff(ws, mine, out, tag=""):
    """Every window read and the speech presence against a stored read set of the same windows."""
    assert len(ws) == len(mine), (len(ws), len(mine))
    out[tag + "z_visual"] = max(abs(a["z_visual"] - b["z_visual"]) for a, b in zip(ws, mine))
    out[tag + "z_speech"] = max((abs(a["z_speech"] - b["z_speech"]) for a, b in zip(ws, mine) if "z_speech" in a and "z_speech" in b), default=0.0)
    out[tag + "speech_presence"] = float(sum(("z_speech" in a) != ("z_speech" in b) for a, b in zip(ws, mine)))


def check(bundle, ref, other=None):
    """Max |difference| to the stored adopted reads of the same video (`ref`), for the parts the arm shares with the
    base; `other` = the stored read set the arm must reproduce in part or in full (README §8.3):
    no_window_evidence -> the older reader's reads (every read); no_stance -> the stored no_stance reads (every read);
    ns_no_transcript / ns_no_prefix_frames -> the stored arm with the verdict turn (z_video: same prefix);
    ns_no_isolation -> the stored no_stance reads (first branch: nothing before it but the prefix);
    joint__no_isolation -> the stored joint reads (first branch); no_isolation__no_window_evidence -> the older
    reader's reads (first branch)."""
    arm, out = bundle["arm"], {}
    mine = bundle["prediction"]["extra"]["windows"]
    if other is not None:
        ow = other["extra"]["windows"]
        if arm in ("no_window_evidence", "no_stance"):
            out["stored_z_video"] = abs(other["extra"]["z_video"] - bundle["global_margin"])
            window_diff(ow, mine, out, "stored_")
        elif arm in ("ns_no_transcript", "ns_no_prefix_frames"):
            out["arm_z_video"] = abs(other["extra"]["z_video"] - bundle["global_margin"])
        elif arm == "ns_no_isolation":
            out["first_visual_no_stance"] = abs(ow[0]["z_visual"] - mine[0]["z_visual"])
        elif arm == "joint__no_isolation":
            out["first_joint"] = abs(ow[0]["z_joint"] - mine[0]["z_joint"])
        elif arm == "no_isolation__no_window_evidence":
            out["first_visual_old"] = abs(ow[0]["z_visual"] - mine[0]["z_visual"])
        elif arm == "joint__no_isolation__no_window_evidence":
            out["first_joint_old"] = abs(ow[0]["z_joint"] - mine[0]["z_joint"])
    if ref is None:
        return out
    ws = ref["extra"]["windows"]
    if arm in SAME_VERDICT:
        out["z_video"] = abs(ref["extra"]["z_video"] - bundle["global_margin"])
    if arm == "base":
        window_diff(ws, mine, out)
        out["score_curve"] = max(abs(a - b) for a, b in zip(ref["score_curve"], bundle["prediction"]["score_curve"]))
        assert len(ref["score_curve"]) == len(bundle["prediction"]["score_curve"])
    if arm == "no_isolation":
        out["first_visual"] = abs(ws[0]["z_visual"] - mine[0]["z_visual"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM"])
    ap.add_argument("--arms", nargs="+", default=None, choices=ARMS,
                    help="default: the six 2026-11-01 arms (smoke: those plus base)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    datasets = ["HateMM", "HateClipSeg", "DeHate"] if a.smoke else a.datasets
    arms = a.arms or (ARMS_V1 if a.smoke else [x for x in ARMS_V1 if x != "base"])
    seq_arms = [x for x in arms if "no_isolation" in x]
    name = a.name or ("smoke" if a.smoke else "_".join(datasets))
    out = EXP / name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    rows = {ds: load_manifest(MANIFESTS[ds], [ds]) for ds in datasets}
    jobs = []  # (arm, row)
    if a.smoke:
        longest = max(rows["HateMM"], key=lambda r: float(r["duration"]))
        rows = {ds: rs[:2] for ds, rs in rows.items()}
        jobs = [(arm, r) for arm in arms for ds in datasets for r in rows[ds]] + [(arm, longest) for arm in seq_arms]
    else:
        jobs = [(arm, r) for arm in arms for ds in datasets for r in rows[ds]]
    import transformers
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"], capture_output=True, text=True).stdout.strip()
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, arms=arms, window_seconds=WINDOW_S, native_frames=NATIVE_FRAMES,
               GT_read=False, smoke=a.smoke, datasets=datasets, videos={ds: len(r) for ds, r in rows.items()}, torch=torch.__version__,
               transformers=transformers.__version__, code=f"{CODE} + src/reader_arms.py; commit {commit}", command="python -u " + " ".join(sys.argv))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    refs = {ds: stored(STORED[ds], ds) for ds in datasets}
    others = {}     # (arm, ds) -> stored read set the arm is checked against (README §8.3)
    for arm in arms:
        for ds in datasets:
            if arm in ("no_window_evidence", "no_isolation__no_window_evidence"):
                others[arm, ds] = stored(OLD[ds], ds)
            elif arm == "joint__no_isolation":
                others[arm, ds] = arm_reads("joint", ds)
            elif arm == "joint__no_isolation__no_window_evidence":   # first branch = joint__no_window_evidence, window 1
                ref_path = EXP / f"mg_{ds}" / "joint__no_window_evidence" / "predictions.jsonl"
                others[arm, ds] = stored(ref_path, ds) if ref_path.exists() else {}
            elif arm in ("no_stance", "ns_no_isolation"):
                others[arm, ds] = arm_reads("no_stance", ds)
            elif arm in ("ns_no_transcript", "ns_no_prefix_frames"):
                others[arm, ds] = arm_reads(arm[3:], ds)
    asr = {ds: load_asr(ds) for ds in datasets}
    words = {ds: aligned_words(ds) for ds in datasets}
    torch.manual_seed(0)
    j = Judge(MODEL)
    worst, errors, t0 = {}, 0, time.perf_counter()
    for k, (arm, row) in enumerate(jobs, 1):
        ds, vid = row["dataset"], row["video_id"]
        path = out / "records" / arm / ds / f"{vid}.json"
        if path.exists():
            continue
        try:
            bundle = read(j, arm, row, asr[ds].get(vid, []), words[ds].get(vid))
        except Exception as ex:  # noqa: BLE001
            errors += 1
            logging.exception("ERROR %s %s/%s %s", arm, ds, vid, ex)
            torch.cuda.empty_cache()
            continue
        bundle["check"] = check(bundle, refs[ds].get(vid), others.get((arm, ds), {}).get(vid))
        for c, v in bundle["check"].items():
            worst[f"{arm}:{c}"] = max(worst.get(f"{arm}:{c}", 0.0), v)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(bundle, allow_nan=False) + "\n")
        tmp.replace(path)
        logging.info("%d/%d %s %s/%s windows=%d with_frames=%d cache=%d %.1fs peak=%.1fGiB check=%s elapsed=%.0fs", k, len(jobs), arm, ds, vid,
                     bundle["windows"], bundle["windows_with_frames"], bundle["final_cache_tokens"], bundle["seconds"], bundle["peak_GiB"],
                     {c: round(v, 5) for c, v in bundle["check"].items()}, time.perf_counter() - t0)
    logging.info("worst differences to stored adopted reads (this session): %s", worst)
    bad = {c: v for c, v in worst.items() if v > TOL}
    missing = 0
    for arm in arms:
        recs = []
        for ds in datasets:
            for row in rows[ds]:
                path = out / "records" / arm / ds / f"{row['video_id']}.json"
                if path.exists():
                    recs.append(json.loads(path.read_text())["prediction"])
                else:
                    missing += 1
        (out / arm).mkdir(exist_ok=True)
        (out / arm / "config.json").write_text(json.dumps({**cfg, "arm": arm}, indent=2) + "\n")
        (out / arm / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        logging.info("arm %s: %d records", arm, len(recs))
    if a.smoke:
        missing += sum(not (out / "records" / arm / "HateMM" / f"{longest['video_id']}.json").exists() for arm in seq_arms)
    logging.info("errors this session %d, missing %d", errors, missing)
    if missing or bad:
        logging.info("FAILED missing=%d differences_above_tolerance=%s", missing, bad)
        sys.exit(1)
    logging.info("DONE jobs=%d", len(jobs))


if __name__ == "__main__":
    main()
