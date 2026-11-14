#!/usr/bin/env python3
"""EventVAD (ACM MM 2025, arXiv 2504.13092) with Qwen3-VL-8B as the event scorer. README §2.

EventVAD is training-free: CLIP + RAFT features, a dynamic graph and boundary detection cut the video into events,
then a video LLM scores each event with a chain-of-thought prompt. The segmentation stage is reused unchanged from
the run in the main table (runs/20261008_baselines/eventvad/<DS>/events.jsonl, released config constants); only the
scorer changes, from VideoLLaMA2.1-7B-16F to Qwen3-VL-8B:
- frames: 16 frames uniformly over the event (VideoLLaMA2's 16-frame sampler), taken from a 4 fps decode, at the
  image size of the naive Qwen3-VL-8B row (src/mllm_judge.py MAX_PIXELS);
- speech: "Speech during this segment: <Whisper words of the event span>" before the question, as the
  transcript variant in the main table (eventvad_av), at most 512 tokens;
- prompt: EventVAD's Figure 2 template (scripts/reproduction_baselines/eventvad/prompt.py PAPER_QUESTION /
  PAPER_INSTRUCTION) with a plain anomaly definition, " An anomaly is hateful content." (the main-table run put the
  nine rules of our Reader there; README §2);
- decoding: greedy, at most 2048 new tokens (as the main-table run).
Read-outs: `prob` = the expected score under the model's fp32 distribution at "\\nTherefore, the final score is "
appended to its own answer (the paper's Figure 2 closing sentence, used by the main-table run as its extraction rule),
first digit and the decimal after "0." (vllm_common.expected_score); `text` = the number the answer itself states
(the release's parser, prompt.parse_score / normalise_score; 0 when none, as the main-table run's rule C).
4 fps: a frame takes the score of the event containing its centre (frames past the last event take the last one).

Stages: `infer` (GPU, vLLM env) -> runs/20261113_lf_baselines/eventvad/<DS>/events.jsonl; `readout` (GPU, HateVLM
env) -> readout.jsonl; `raster` (CPU, HateVideo env) -> curves + canonical evaluator. No label is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_common as Q         # noqa: E402
from vllm_common import L       # noqa: E402

sys.path.insert(0, os.path.join(Q.REPO, "scripts", "reproduction_baselines", "eventvad"))
import prompt as pmod           # noqa: E402

EVENTS = os.path.join(os.environ.get("LF_BASELINES_RUNS") or os.path.join(Q.REPO, "runs", "20261008_baselines"), "eventvad")
FPS = 4.0
NUM_FRAMES = 16
SPEECH_CAP = 512
NO_SPEECH = "(no speech)"
SUFFIX = "\nTherefore, the final score is "
QUESTION = "#Question: %s An anomaly is hateful content.\n#Instruction: %s" % (
    pmod.PAPER_QUESTION, pmod.PAPER_INSTRUCTION)


def build_prompt(speech):
    return "Speech during this segment: " + speech + "\n" + QUESTION


def out_dir(ds):
    return os.path.join(Q.RUNS, "eventvad", ds)


def done_ids(path):
    if not os.path.isfile(path):
        return set()
    with open(path) as fh:
        return {json.loads(x)["video_id"] for x in fh if x.strip()}


def load_records(path):
    recs = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["video_id"]] = r
    return recs


def load_events(ds):
    """video_id -> list of (start_s, end_s) from the main-table EventVAD segmentation."""
    out = {}
    with open(os.path.join(EVENTS, ds, "events.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            fps = float(r["decode_fps"])
            out[r["video_id"]] = [(s / fps, e / fps) for s, e in r["events"]]
    return out


def event_frames(frames, t1, t2):
    return Q.uniform(Q.frames_in(frames, t1, t2), NUM_FRAMES)


def capped(text, n_tokens):
    """Whole-word prefix with at most SPEECH_CAP tokens (eventvad_4fps.py speech_text)."""
    if not text:
        return NO_SPEECH
    if n_tokens(text) <= SPEECH_CAP:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens(" ".join(words[:mid])) <= SPEECH_CAP:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo]) or NO_SPEECH


def cmd_infer(args):
    ds = args.dataset
    dest = out_dir(ds)
    L.start_run_log(dest)
    log = Q.log_to(os.path.join(dest, "infer.log"))
    path = os.path.join(dest, "events.jsonl")
    man = L.manifest(ds)
    vids = L.cohort(ds)
    if args.limit:
        vids = vids[:args.limit]
    ev = load_events(ds)
    done = done_ids(path)
    todo = [v for v in vids if v not in done]
    log("eventvad %s: %d cohort videos, %d to do; code %s" % (ds, len(vids), len(todo), L.git_version()))
    if not todo:
        log("DONE infer (nothing to do)")
        return
    segs = L.transcript_segments(ds, {v: man[v]["duration"] for v in vids})
    eng = Q.Engine(max_model_len=16384, max_images=NUM_FRAMES, log=log)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        try:
            return v, Q.decode_frames(man[v]["video_path"], FPS, "ev_%s_%s" % (ds, v)), None
        except Exception as e:  # noqa: BLE001
            return v, [], repr(e)

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    fut = pool.map(decode, chunks[0])
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        decoded = list(fut)
        if ci + 1 < len(chunks):
            fut = pool.map(decode, chunks[ci + 1])
        units, failed = [], {}
        for v, frames, err in decoded:
            if err or not frames or v not in ev:
                failed[v] = err or ("no frames decoded" if v in ev else "no EventVAD events")
                continue
            for k, (t1, t2) in enumerate(ev[v]):
                files = event_frames(frames, t1, t2)
                speech = capped(L.span_text(segs.get(v, []), t1, t2), eng.n_tokens)
                units.append((v, k, t1, t2, len(files), speech,
                              eng.request(Q.open_images(files, reader_size=True), build_prompt(speech),
                                          "%s/%s/e%04d" % (ds, v, k))))
        outs = eng.generate([u[6] for u in units], args.max_tokens) if units else []
        per = {}
        for (v, k, t1, t2, nf, speech, _), o in zip(units, outs):
            val, status, raw = pmod.parse_score(o.text)
            norm, rule = pmod.normalise_score(val)
            per.setdefault(v, []).append({"k": k, "start": t1, "end": t2, "n_frames": nf, "speech": speech,
                                          "answer": o.text, "finish": o.finish_reason, "parsed": norm,
                                          "parse_status": status, "range_rule": rule})
        with open(path, "a") as fh:
            for v in chunk:
                fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"],
                                     "error": failed.get(v), "events": per.get(v, [])}) + "\n")
        for v, _, _ in decoded:
            Q.drop_frames("ev_%s_%s" % (ds, v))
        log("chunk %d/%d: %d videos, %d events, %d failed videos, %.0f s" % (
            ci + 1, len(chunks), len(chunk), len(units), len(failed), time.time() - t0))
    log("DONE infer %s" % ds)


def cmd_readout(args):
    ds = args.dataset
    dest = out_dir(ds)
    log = Q.log_to(os.path.join(dest, "readout.log"))
    recs = load_records(os.path.join(dest, "events.jsonl"))
    path = os.path.join(dest, "readout.jsonl")
    done = done_ids(path)
    todo = [v for v in recs if v not in done]
    log("eventvad readout %s: %d videos, %d to do; code %s" % (ds, len(recs), len(todo), L.git_version()))
    if not todo:
        log("DONE readout (nothing to do)")
        return
    man = L.manifest(ds)
    sc = Q.Scorer()
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        if recs[v].get("error"):
            return v, []
        return v, Q.decode_frames(man[v]["video_path"], FPS, "ev_ro_%s_%s" % (ds, v))

    ahead = args.workers * 2
    futs = [pool.submit(decode, v) for v in todo[:ahead]]
    t0, n = time.time(), 0
    for i, v in enumerate(todo):
        if i + ahead < len(todo):
            futs.append(pool.submit(decode, todo[i + ahead]))
        _, frames = futs[i].result()
        scores = []
        for e in recs[v]["events"]:
            files = event_frames(frames, e["start"], e["end"])
            if len(files) != e["n_frames"]:
                raise RuntimeError("%s event %d: %d frames now, %d at generation" % (v, e["k"], len(files), e["n_frames"]))
            lp1, lp2 = sc.score_digits(Q.open_images(files, reader_size=True), build_prompt(e["speech"]),
                                       e["answer"].rstrip() + SUFFIX)
            scores.append(Q.expected_score(lp1, lp2))
            n += 1
        Q.drop_frames("ev_ro_%s_%s" % (ds, v))
        with open(path, "a") as fh:
            fh.write(json.dumps({"video_id": v, "score": scores}) + "\n")
        if (i + 1) % 50 == 0:
            log("%d / %d videos, %d events, %.0f s" % (i + 1, len(todo), n, time.time() - t0))
    log("DONE readout %s: %d events in %.0f s" % (ds, n, time.time() - t0))


def cmd_raster(args):
    ds = args.dataset
    dest = out_dir(ds)
    recs = load_records(os.path.join(dest, "events.jsonl"))
    prob = {r["video_id"]: r["score"] for r in load_records(os.path.join(dest, "readout.jsonl")).values()}
    stats = {"videos": len(recs), "events": sum(len(r["events"]) for r in recs.values()),
             "finish_length": sum(e["finish"] == "length" for r in recs.values() for e in r["events"]),
             "text_unparsed": sum(e["parsed"] is None for r in recs.values() for e in r["events"])}
    for readout in ("prob", "text"):
        rows = {}
        for v, r in recs.items():
            n = L.n_frames(r["duration"])
            es = r["events"]
            if r.get("error") or not es:
                rows[v] = L.row("eventvad_q3vl_" + readout, ds, v, r["duration"], [], None, __file__,
                                error=r.get("error") or "no events")
                continue
            vals = prob[v] if readout == "prob" else [e["parsed"] if e["parsed"] is not None else 0.0 for e in es]
            curve = L.units_to_4fps([e["start"] for e in es], [e["end"] for e in es], vals, n)
            rows[v] = L.row("eventvad_q3vl_" + readout, ds, v, r["duration"], curve, None, __file__)
        med, failed = L.apply_f2(rows, ds)
        stats["f2_videos_" + readout] = failed
        L.finalize(ds, list(rows.values()), os.path.join(dest, readout), "eventvad_q3vl_" + readout)
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("infer", "readout", "raster"))
    ap.add_argument("--dataset", required=True, choices=L.DATASETS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16, help="videos per vLLM batch")
    ap.add_argument("--workers", type=int, default=4, help="ffmpeg decoders")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    {"infer": cmd_infer, "readout": cmd_readout, "raster": cmd_raster}[args.stage](args)


if __name__ == "__main__":
    main()
