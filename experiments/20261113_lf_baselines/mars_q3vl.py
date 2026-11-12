#!/usr/bin/env python3
"""MARS (ICASSP 2026, arXiv 2601.15115, github.com/Multimodal-Intelligence-Lab-MIL/MARS, MIT) on 8 s windows with
Qwen3-VL-8B. README arm B1.

MARS is a training-free hateful video detector: four calls of one VLM on the same frames and transcript,
(1) objective description, (2) evidence under the assumption that the content is hateful, (3) evidence under the
assumption that it is not, (4) a final decision that reads the three JSON answers. Prompts are verbatim from the
release (`code/MARS/Qwen.py` `create_turn_prompt`); each turn is a fresh single user message with the frames first,
then the prompt and "\\n\\nTranscript: <text>" ("No transcript available." when empty).

Applied per window of the fixed 8 s grid of the Reader (the unit of the naive per-window rows):
- frames: the window's 1 fps frames (MARS samples up to 16 frames from a 1 fps folder and uses all of them when there
  are fewer; an 8 s window has 8), at the image size of the naive Qwen3-VL-8B row (src/mllm_judge.py MAX_PIXELS);
- transcript: the window's Whisper large-v3 words, sliced proportionally (as the naive rows; MARS uses Whisper);
- decoding: greedy, at most 2048 new tokens per turn (the release samples at temperature 0.7 with 4096);
- read-outs of turn 4: `prob` = log P("1") - log P("0") at the position where the final decision writes its label,
  from one HF forward with an fp32 output layer on the turn-4 prompt plus the generated text up to that position
  (README §2: the same arithmetic as the naive rows, so that scores are not rounded); `label` = the 0/1 label itself
  (the paper's output). The verbalised confidence and the two hypothesis strengths are recorded.

Stages: `infer` (GPU, vLLM env) writes runs/20261113_lf_baselines/mars/<DS>/windows.jsonl (one line per video,
resumable); `readout` (GPU, HateVLM env) writes readout.jsonl; `raster` (CPU, HateVideo env) builds the 4 fps curves
of both read-outs and calls the canonical evaluator.
No label is read.
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
import vllm_common as Q          # noqa: E402
from vllm_common import L       # noqa: E402

WINDOW = 8.0
FPS = 1.0
MAX_FRAMES = 16
NO_TRANSCRIPT = "No transcript available."

DEFS = """HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.
OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech.
IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech."""

TURN1 = """You are a content analysis expert. Provide a detailed, objective description of this video content.

Task: Describe what you observe in the video frames and transcript:
1. objective_visual_description: Describe what you see - objects, people, actions, settings, text, symbols, colors, expressions, gestures, etc.

Rules:
- Be comprehensive and detailed
- NO analysis, interpretations, or value judgments
- Simply report what you observe
- Return ONLY valid JSON with one key: ["objective_visual_description"]"""

TURN2 = """You are a hateful content detection specialist. Analyze the video and transcript content under the ASSUMPTION that it contains hateful or offensive content.

""" + DEFS + """

Task: Identify evidence supporting hateful(offensive) interpretation:
1. evidence: Describe specific evidence supporting the hateful nature of the content
2. reasoning: Explain why it is hateful content.
3. strength: Your confidence in this hate interpretation [0-1]

Focus on:
- Language targeting religion, ethnicity, race, gender, nationality, etc.
- Group-based dehumanizing language or symbols
- Identity-based discriminatory attacks (not personal insults)
- Incitement against identity groups

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""

TURN3 = """You are a content analyst. Analyze this content under the ASSUMPTION that it does NOT constitute hate speech and offensive content.

""" + DEFS + """

Task: Identify evidence supporting non-hate(non-offensive) interpretation:
1. evidence: Describe specific evidence supporting the non-hateful nature of the content
2. reasoning: Explain why this content does not hateful
3. strength: Your confidence in this non-hate interpretation [0-1]

Consider:
- Is this a personal dispute rather than group targeting?
- Are insults directed at individuals rather than identity groups?
- Is there artistic, satirical, or educational context?
- Does the content lack group-based discriminatory language?

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""


def turn4(t1, t2, t3):
    return """You are a senior content moderation specialist making the final determination.

OBJECTIVE DESCRIPTION: {d}
HATE ARGUMENT: {h}
NON-HATE ARGUMENT: {n}

Task: Conduct critical analysis and make final determination:
1. evidence_comparison: Compare the quality and strength of both arguments, identifying which evidence is more compelling
2. contextual_analysis: Analyze how the content's context (platform, audience, intent) affects interpretation
3. harm_assessment: Evaluate potential real-world impact and harm to targeted groups
4. final_decision:
   - label: 0 (non-hate) or 1 (hate/offensive)
   - confidence: Overall confidence in decision [0-1]
   - key_factors: The decisive elements that determined your judgment
   - reasoning: 2-3 sentences explaining your decision

Rules:
- Weigh evidence objectively, not just confidence scores
- Consider both explicit and subtle indicators
- Prioritize potential for real-world harm
- Base decision on strongest evidence, not balanced arguments

Return ONLY valid JSON with key "final_decision" containing the above structure""".format(
        d=json.dumps(t1), h=json.dumps(t2), n=json.dumps(t3))


def with_transcript(prompt, text):
    return prompt + "\n\nTranscript: " + (text if text else NO_TRANSCRIPT)


def out_dir(ds):
    return os.path.join(Q.RUNS, "mars", ds)


def done_ids(path):
    if not os.path.isfile(path):
        return set()
    with open(path) as fh:
        return {json.loads(x)["video_id"] for x in fh if x.strip()}


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def cmd_infer(args):
    ds = args.dataset
    dest = out_dir(ds)
    L.start_run_log(dest)
    log = Q.log_to(os.path.join(dest, "infer.log"))
    path = os.path.join(dest, "windows.jsonl")
    man = L.manifest(ds)
    vids = L.cohort(ds)
    if args.limit:
        vids = vids[:args.limit]
    done = done_ids(path)
    todo = [v for v in vids if v not in done]
    log("mars %s: %d cohort videos, %d to do; code %s" % (ds, len(vids), len(todo), L.git_version()))
    if not todo:
        log("DONE infer (nothing to do)")
        return
    segs = L.transcript_segments(ds, {v: man[v]["duration"] for v in vids})
    eng = Q.Engine(max_model_len=16384, max_images=MAX_FRAMES, log=log)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        try:
            return v, Q.decode_frames(man[v]["video_path"], FPS, "mars_%s_%s" % (ds, v)), None
        except Exception as e:  # noqa: BLE001
            return v, [], repr(e)

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    fut = pool.map(decode, chunks[0])
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        decoded = list(fut)
        if ci + 1 < len(chunks):
            fut = pool.map(decode, chunks[ci + 1])
        units = []                       # (vid, window index, t1, t2, images, text)
        failed = {}
        for v, frames, err in decoded:
            if err or not frames:
                failed[v] = err or "no frames decoded"
                continue
            dur = man[v]["duration"]
            for i, (t1, t2) in enumerate(L_fixed(dur)):
                files = Q.uniform(Q.frames_in(frames, t1, t2), MAX_FRAMES)
                text = L.span_text(segs.get(v, []), t1, t2)
                units.append((v, i, t1, t2, Q.open_images(files, reader_size=True), text))
        reqs = []
        for v, i, t1, t2, imgs, text in units:
            for k, p in enumerate((TURN1, TURN2, TURN3)):
                reqs.append(eng.request(imgs, with_transcript(p, text), "%s/%s/w%03d/t%d" % (ds, v, i, k + 1)))
        outs = eng.generate(reqs, args.max_tokens) if reqs else []
        parsed = []
        for j, u in enumerate(units):
            o = outs[3 * j:3 * j + 3]
            parsed.append(([x.text for x in o], [Q.clean_json(x.text) for x in o],
                           [x.finish_reason for x in o]))
        reqs4 = [eng.request(u[4], with_transcript(turn4(*p[1]), u[5]), "%s/%s/w%03d/t4" % (ds, u[0], u[1]))
                 for u, p in zip(units, parsed)]
        outs4 = eng.generate(reqs4, args.max_tokens) if reqs4 else []
        per = {}
        for u, p, o in zip(units, parsed, outs4):
            v, i, t1, t2, imgs, text = u
            j4 = Q.clean_json(o.text)
            fd = get(j4, "final_decision") if isinstance(get(j4, "final_decision"), dict) else j4
            per.setdefault(v, []).append({
                "i": i, "start": t1, "end": t2, "n_frames": len(imgs), "text": text,
                "t4_prompt": with_transcript(turn4(*p[1]), text), "t4_prefix": Q.label_prefix(o.text),
                "label": get(fd, "label"), "confidence": get(fd, "confidence"),
                "hate_strength": get(p[1][1], "strength"), "nonhate_strength": get(p[1][2], "strength"),
                "parsed": [x is not None for x in p[1]] + [j4 is not None],
                "finish": p[2] + [o.finish_reason], "answers": p[0] + [o.text]})
        with open(path, "a") as fh:
            for v in chunk:
                rec = {"dataset": ds, "video_id": v, "duration": man[v]["duration"], "error": failed.get(v),
                       "windows": sorted(per.get(v, []), key=lambda w: w["i"])}
                fh.write(json.dumps(rec) + "\n")
        for v, _, _ in decoded:
            Q.drop_frames("mars_%s_%s" % (ds, v))
        nz = sum(w["t4_prefix"] is not None for ws in per.values() for w in ws)
        log("chunk %d/%d: %d videos, %d windows, %d with a label read, %d failed videos, %.0f s" % (
            ci + 1, len(chunks), len(chunk), len(units), nz, len(failed), time.time() - t0))
    log("DONE infer %s" % ds)


def L_fixed(duration):
    from src.video_inputs import fixed_windows
    return fixed_windows(duration, WINDOW)


def load_records(path):
    recs = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["video_id"]] = r
    return recs


def cmd_readout(args):
    ds = args.dataset
    dest = out_dir(ds)
    log = Q.log_to(os.path.join(dest, "readout.log"))
    recs = load_records(os.path.join(dest, "windows.jsonl"))
    path = os.path.join(dest, "readout.jsonl")
    done = done_ids(path)
    todo = [v for v in recs if v not in done]
    log("mars readout %s: %d videos, %d to do; code %s" % (ds, len(recs), len(todo), L.git_version()))
    if not todo:
        log("DONE readout (nothing to do)")
        return
    man = L.manifest(ds)
    sc = Q.Scorer()
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        if recs[v].get("error"):
            return v, []
        return v, Q.decode_frames(man[v]["video_path"], FPS, "mars_ro_%s_%s" % (ds, v))

    t0, n = time.time(), 0
    futs = [pool.submit(decode, v) for v in todo[:args.workers * 2]]
    for k, v in enumerate(todo):
        if k + args.workers * 2 < len(todo):
            futs.append(pool.submit(decode, todo[k + args.workers * 2]))
        _, frames = futs[k].result()
        zs = []
        for w in recs[v]["windows"]:
            if w["t4_prefix"] is None:
                zs.append(None)
                continue
            files = Q.uniform(Q.frames_in(frames, w["start"], w["end"]), MAX_FRAMES)
            if len(files) != w["n_frames"]:
                raise RuntimeError("%s window %d: %d frames now, %d at generation" % (v, w["i"], len(files), w["n_frames"]))
            lp = sc.next_logprobs(Q.open_images(files, reader_size=True), w["t4_prompt"], w["t4_prefix"], ["0", "1"])
            zs.append(lp["1"] - lp["0"])
            n += 1
        Q.drop_frames("mars_ro_%s_%s" % (ds, v))
        with open(path, "a") as fh:
            fh.write(json.dumps({"video_id": v, "z": zs}) + "\n")
        if (k + 1) % 50 == 0:
            log("%d / %d videos, %d windows read, %.0f s" % (k + 1, len(todo), n, time.time() - t0))
    log("DONE readout %s: %d windows read in %.0f s" % (ds, n, time.time() - t0))


def cmd_raster(args):
    ds = args.dataset
    dest = out_dir(ds)
    recs = load_records(os.path.join(dest, "windows.jsonl"))
    zread = {r["video_id"]: r["z"] for r in load_records(os.path.join(dest, "readout.jsonl")).values()}
    stats = {"videos": len(recs), "windows": 0, "no_label_position": 0, "label_missing": 0, "video_median_fill": 0,
             "finish_length": 0, "turn_parse_fail": [0, 0, 0, 0]}
    for readout in ("prob", "label"):
        rows = {}
        for v, r in recs.items():
            n = L.n_frames(r["duration"])
            ws = r["windows"]
            if r.get("error") or not ws:
                rows[v] = L.row("mars_q3vl_" + readout, ds, v, r["duration"], [], 1.0 / WINDOW, __file__,
                                error=r.get("error") or "no windows")
                continue
            if readout == "prob":
                vals = zread[v]
            else:
                vals = [float(w["label"]) if w["label"] in (0, 1, "0", "1") else None for w in ws]
            ok = [x for x in vals if x is not None]
            if not ok:
                rows[v] = L.row("mars_q3vl_" + readout, ds, v, r["duration"], [], 1.0 / WINDOW, __file__,
                                error="no window read")
                continue
            fill = float(np.median(ok))
            if readout == "prob":
                stats["windows"] += len(ws)
                stats["video_median_fill"] += sum(x is None for x in vals)
                for w in ws:
                    stats["no_label_position"] += w["t4_prefix"] is None
                    stats["label_missing"] += w["label"] not in (0, 1, "0", "1")
                    stats["finish_length"] += sum(f == "length" for f in w["finish"])
                    for k in range(4):
                        stats["turn_parse_fail"][k] += not w["parsed"][k]
            vals = [fill if x is None else x for x in vals]
            curve = L.units_to_4fps([w["start"] for w in ws], [w["end"] for w in ws], vals, n)
            rows[v] = L.row("mars_q3vl_" + readout, ds, v, r["duration"], curve, 1.0 / WINDOW, __file__)
        med, failed = L.apply_f2(rows, ds)
        if readout == "prob":
            stats["f2_videos"] = failed
        L.finalize(ds, list(rows.values()), os.path.join(dest, readout), "mars_q3vl_" + readout)
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
