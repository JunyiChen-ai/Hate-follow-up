#!/usr/bin/env python3
"""Free-form per-video detection plan in the adopted reader (experiments/20261101_video_plan, declared 2026-11-01).

Per video (Qwen3-VL-8B, frozen, GPU inside Slurm only; no label file is read anywhere):
  1. the adopted shared context: r6 prefix (20 frames + transcript + rules) and the whole-video verdict turn
     (src/stance_cache.build);
  2. PLAN_QUESTION asked once; the model writes a free-form plan (greedy, at most MAX_PLAN_TOKENS tokens);
     the plan is written back onto the shared context as the model's own turn (src/stance_cache.append_turn);
  3. every 8 s window read exactly as the adopted reader (src/window_evidence: in-window 0.5 fps frames next to the
     visual question, forced-aligned word slice in the speech question; isolated branches, no memory of earlier
     windows), now after the plan turn.
Arms: own_plan (stage own: generate + read), other_plan (stage other: the plan of another video of the same corpus,
fixed derangement seed 0 over the corpus video ids, written as this video's own turn). base = the stored adopted reads;
--smoke re-reads the first two videos of each corpus with the plan switched off and checks them against those reads.
Output: runs/20261101_video_plan/<name>/{records,records_other}/<ds>/<vid>.json, <arm>/predictions.jsonl,
plans_<ds>.jsonl, derangement_<ds>.json, config.json, run.log. Resumable: videos with a record are not read again.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mllm_judge import MODEL, Judge, yesno_question  # noqa: E402
from src.stance_cache import append_turn, build, generate_turn, margin  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402
from src.window_evidence import WINDOW_S, aligned_slice, aligned_words, frames_margin, in_window, local_frames, window_prediction  # noqa: E402

EXP = ROOT / "runs/20261101_video_plan"
NATIVE_FRAMES = 20
MAX_PLAN_TOKENS = 256
SEED = 0
ALL_TEST = ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"
ADOPTED = ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl"
CODE = "experiments/20261101_video_plan/read_plan.py"
TOL = 1e-3
PLAN_QUESTION = (
    "This video will now be checked moment by moment. Before that, write a short plan in your own words for "
    "checking this video: what in this video could violate the rules above, or why nothing in it does, and what to "
    "look and listen for in each moment. Keep it to one short paragraph of plain prose, about 100 words."
)


def stored(path, datasets):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(path)) if not r.get("error") and r["dataset"] in datasets}


def derange(ids, seed=SEED):
    """Fixed random derangement of the sorted ids: rejection sampling of permutations with no fixed point."""
    ids = sorted(ids)
    assert len(ids) >= 2
    rng = np.random.RandomState(seed)
    while True:
        perm = rng.permutation(len(ids))
        if all(perm[i] != i for i in range(len(ids))):
            return {ids[i]: ids[int(perm[i])] for i in range(len(ids))}


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def read_windows(j, cache, ctx, row, segments, rec, frames):
    """The adopted window reads on the given shared context: (visual, speech, traces)."""
    wins = fixed_windows(float(row["duration"]), WINDOW_S)
    n = len(wins)
    visual, speech, traces = [], [], []
    for i, (t1, t2) in enumerate(wins):
        body = window_text(segments, t1, t2)
        abody = aligned_slice(rec, t1, t2, body)
        vq = yesno_question(i, n, t1, t2, body, "visual")
        inside = in_window(frames, t1, t2)
        v = frames_margin(j, cache, ctx, inside, vq) if inside else margin(j, cache, ctx, vq)
        s = margin(j, cache, ctx, yesno_question(i, n, t1, t2, abody, "speech")) if abody is not None and abody.strip() else None
        visual.append(v)
        speech.append(s)
        traces.append(dict(i=i, bounds=[t1, t2], local_times=[t for t, _ in inside], text_aligned=abody, z_visual=v, z_speech=s))
    return visual, speech, traces


def compare(ref, z_video, traces):
    """Max |difference| to a stored prediction record (z_video, per-window visual and speech), and presence mismatches."""
    ws = ref["extra"]["windows"]
    assert len(ws) == len(traces), (len(ws), len(traces))
    out = dict(z_video=abs(ref["extra"]["z_video"] - z_video),
               visual=max(abs(w["z_visual"] - t["z_visual"]) for w, t in zip(ws, traces)),
               speech=max((abs(w["z_speech"] - t["z_speech"]) for w, t in zip(ws, traces) if "z_speech" in w and t["z_speech"] is not None), default=0.0),
               speech_presence=sum(("z_speech" in w) != (t["z_speech"] is not None) for w, t in zip(ws, traces)))
    return out


@torch.no_grad()
def stage_own(j, row, segments, rec, ref):
    """Verdict, plan generation, own-plan window reads; with `ref`: also the plan-off reads checked against it."""
    torch.cuda.reset_peak_memory_stats()
    begin = tick()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    assert 0 < len(native) <= NATIVE_FRAMES
    frames = local_frames(row)
    cache, ctx = build(j, native, segments)
    t_prefix = tick() - begin
    out = dict(model=MODEL, host=socket.gethostname(), GT_read=False, global_margin=ctx["global_margin"], stance=ctx["stance"],
               stance_cache_tokens=ctx["stance_cache_tokens"], aligned_status=None if rec is None else rec["status"], seconds_prefix=t_prefix)
    if ref is not None:  # plan switched off: must reproduce the stored adopted reads
        t0 = tick()
        bv, bs, btr = read_windows(j, cache, ctx, row, segments, rec, frames)
        out["base_check"] = compare(ref, ctx["global_margin"], btr)
        out["base_traces"] = btr
        out["base_prediction"] = window_prediction(row, ctx, bv, bs, "video_plan_base_replay", tick() - t0, CODE)
    t0 = tick()
    plan, n_tok, hit_max = generate_turn(j, cache, ctx, PLAN_QUESTION, MAX_PLAN_TOKENS)
    out["plan"] = dict(text=plan, tokens=n_tok, hit_max_tokens=hit_max, seconds=tick() - t0, question=PLAN_QUESTION, max_tokens=MAX_PLAN_TOKENS)
    t0 = tick()
    ctx_plan = append_turn(j, cache, ctx, PLAN_QUESTION, plan)
    v, s, tr = read_windows(j, cache, ctx_plan, row, segments, rec, frames)
    secs = tick() - t0
    out.update(turn_tokens=ctx_plan["turn_tokens"], plan_cache_tokens=ctx_plan["stance_cache_tokens"], windows=len(tr),
               windows_with_frames=sum(bool(t["local_times"]) for t in tr), seconds_reads=secs, seconds_total=tick() - begin,
               peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, traces=tr,
               prediction=window_prediction(row, ctx_plan, v, s, "video_plan_own_plan", secs, CODE))
    del cache
    return out


@torch.no_grad()
def stage_other(j, row, segments, rec, own, source_vid, source_plan):
    """Another video's plan written as this video's own turn, then the adopted window reads."""
    torch.cuda.reset_peak_memory_stats()
    begin = tick()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    frames = local_frames(row)
    cache, ctx = build(j, native, segments)
    assert abs(ctx["global_margin"] - own["global_margin"]) <= TOL and ctx["stance"] == own["stance"], "verdict differs from stage own"
    ctx_plan = append_turn(j, cache, ctx, PLAN_QUESTION, source_plan)
    v, s, tr = read_windows(j, cache, ctx_plan, row, segments, rec, frames)
    secs = tick() - begin
    out = dict(model=MODEL, host=socket.gethostname(), GT_read=False, global_margin=ctx["global_margin"], stance=ctx["stance"],
               plan_source=source_vid, plan=dict(text=source_plan), turn_tokens=ctx_plan["turn_tokens"], plan_cache_tokens=ctx_plan["stance_cache_tokens"],
               windows=len(tr), seconds_total=secs, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, traces=tr,
               prediction=window_prediction(row, ctx_plan, v, s, "video_plan_other_plan", secs, CODE))
    del cache
    return out


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM"])
    ap.add_argument("--smoke", action="store_true", help="first two videos of HateMM and HateClipSeg; checks the plan-off reads")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    datasets = ["HateMM", "HateClipSeg"] if a.smoke else a.datasets
    name = a.name or ("smoke" if a.smoke else "_".join(datasets))
    out = EXP / name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    rows = load_manifest(ALL_TEST, datasets)
    if a.smoke:
        rows = [r for ds in datasets for r in [x for x in rows if x["dataset"] == ds][:2]]
    import transformers
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"], capture_output=True, text=True).stdout.strip()
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, window_seconds=WINDOW_S, native_frames=NATIVE_FRAMES,
               plan_question=PLAN_QUESTION, max_plan_tokens=MAX_PLAN_TOKENS, decoding="greedy", seed=SEED, GT_read=False, smoke=a.smoke,
               datasets=datasets, videos=len(rows), torch=torch.__version__, transformers=transformers.__version__,
               base_reads=str(ADOPTED.relative_to(ROOT)), code=f"{CODE} + src/window_evidence.py + src/stance_cache.py; commit {commit}",
               command="python -u " + " ".join(sys.argv))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    refs = stored(ADOPTED, datasets) if a.smoke else {}
    asr = {ds: load_asr(ds) for ds in datasets}
    words = {ds: aligned_words(ds) for ds in datasets}
    torch.manual_seed(SEED)
    j = Judge(MODEL)
    worst, errors, t0 = {}, 0, time.perf_counter()
    # ---- stage own: verdict, plan, own-plan reads
    for k, row in enumerate(rows, 1):
        ds, vid = row["dataset"], row["video_id"]
        path = out / "records" / ds / f"{vid}.json"
        if path.exists():
            continue
        try:
            rec = stage_own(j, row, asr[ds].get(vid, []), words[ds].get(vid), refs.get((ds, vid)))
        except Exception as ex:  # noqa: BLE001
            errors += 1
            logging.exception("ERROR own %s/%s %s", ds, vid, ex)
            continue
        for c, v in rec.get("base_check", {}).items():
            worst[c] = max(worst.get(c, 0.0), v)
        write_json(path, rec)
        logging.info("own %d/%d %s/%s stance=%s plan_tokens=%d hit_max=%s windows=%d gen=%.1fs reads=%.1fs total=%.1fs peak=%.1fGiB check=%s elapsed=%.0fs",
                     k, len(rows), ds, vid, rec["stance"], rec["plan"]["tokens"], rec["plan"]["hit_max_tokens"], rec["windows"], rec["plan"]["seconds"],
                     rec["seconds_reads"], rec["seconds_total"], rec["peak_GiB"], {c: round(v, 5) for c, v in rec.get("base_check", {}).items()},
                     time.perf_counter() - t0)
        if a.smoke:
            logging.info("PLAN %s/%s: %s", ds, vid, rec["plan"]["text"])
    # ---- plans and derangement per corpus (ids only; no labels)
    plans = {}
    for ds in datasets:
        for row in [r for r in rows if r["dataset"] == ds]:
            path = out / "records" / ds / f"{row['video_id']}.json"
            if path.exists():
                plans[(ds, row["video_id"])] = json.loads(path.read_text())
        (out / f"plans_{ds}.jsonl").write_text("".join(json.dumps(dict(dataset=ds, video_id=v, plan=r["plan"]["text"], tokens=r["plan"]["tokens"],
                                                                      hit_max_tokens=r["plan"]["hit_max_tokens"], z_video=r["global_margin"], stance=r["stance"])) + "\n"
                                                      for (d, v), r in sorted(plans.items()) if d == ds))
    missing_own = len(rows) - len(plans)
    if missing_own:
        logging.info("FAILED stage own incomplete: missing=%d errors=%d", missing_own, errors)
        sys.exit(1)
    mapping = {}
    for ds in datasets:
        m = derange([v for d, v in plans if d == ds])
        assert all(src != v for v, src in m.items())
        mapping[ds] = m
        (out / f"derangement_{ds}.json").write_text(json.dumps(dict(seed=SEED, dataset=ds, source_of=m), indent=1) + "\n")
    # ---- stage other: another video's plan
    for k, row in enumerate(rows, 1):
        ds, vid = row["dataset"], row["video_id"]
        path = out / "records_other" / ds / f"{vid}.json"
        if path.exists():
            continue
        src = mapping[ds][vid]
        try:
            rec = stage_other(j, row, asr[ds].get(vid, []), words[ds].get(vid), plans[(ds, vid)], src, plans[(ds, src)]["plan"]["text"])
        except Exception as ex:  # noqa: BLE001
            errors += 1
            logging.exception("ERROR other %s/%s %s", ds, vid, ex)
            continue
        write_json(path, rec)
        logging.info("other %d/%d %s/%s plan_from=%s windows=%d total=%.1fs peak=%.1fGiB elapsed=%.0fs", k, len(rows), ds, vid, src, rec["windows"],
                     rec["seconds_total"], rec["peak_GiB"], time.perf_counter() - t0)
    # ---- prediction files
    counts = {}
    for arm, folder, key in (("own_plan", "records", "prediction"), ("other_plan", "records_other", "prediction"), ("base_replay", "records", "base_prediction")):
        recs = []
        for row in rows:
            path = out / folder / row["dataset"] / f"{row['video_id']}.json"
            if path.exists():
                r = json.loads(path.read_text())
                if key in r:
                    recs.append(r[key])
        if not recs:
            continue
        (out / arm).mkdir(exist_ok=True)
        (out / arm / "config.json").write_text(json.dumps({**cfg, "arm": arm}, indent=2) + "\n")
        (out / arm / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        counts[arm] = len(recs)
    logging.info("prediction records %s of %d videos; errors this session %d", counts, len(rows), errors)
    if a.smoke:
        logging.info("worst differences of the plan-off reads to the stored adopted reads: %s", worst)
    bad = {c: v for c, v in worst.items() if c != "speech_presence" and v > TOL} | ({"speech_presence": worst["speech_presence"]} if worst.get("speech_presence") else {})
    if counts.get("own_plan") != len(rows) or counts.get("other_plan") != len(rows) or bad:
        logging.info("FAILED counts=%s videos=%d differences_above_tolerance=%s", counts, len(rows), bad)
        sys.exit(1)
    logging.info("DONE coverage=%d", len(rows))


if __name__ == "__main__":
    main()
