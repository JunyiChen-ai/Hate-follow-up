#!/usr/bin/env python3
"""Cog-VADU (TMLR 2026, openreview QcuSMNG7J8, third_party/Cog-VADU at b440861) on Qwen3-VL-8B (README §3.4).

Stage 1 (the chain, `Raw_CoADTP_STAGE1.py`): one generation per 8 s segment with the release's chain-of-thought prompt;
from the second segment on the prompt carries the previous segment's DESCRIPTION, ANOMALY_TYPE and smoothed score;
the release's parser and temporal smoothing (0.3 x previous + 0.7 x this). Stages 3-6 (ImageBind video / text
embeddings, per-video index of distinct descriptions, cross-modal re-ranking, softmax-weighted neighbour scores) as
`get_visual_embeddings.py`, `create_summary_index_FAST.py`, `video_text_TEST3_fast.py`, `eval2_k_exp.py`.
Deviations: README §3.4 (8 s segments, 8 frames at 1 fps, Whisper words, anomaly = "An anomaly is hateful content.",
fp32 expected-score read-out, retrieval limited to existing entries, numpy instead of faiss).

Stages: `infer` (vLLM env) -> units.jsonl; `readout` (HateVLM env) -> readout.jsonl; `embed` (HateVideo env,
ImageBind) -> imagebind/<video>.npz; `raster` (CPU, HateVideo env) -> <config>/predictions.jsonl + evaluator.
No label is read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_common as Q         # noqa: E402
from vllm_common import L       # noqa: E402

UNIT = 8.0                      # segment length (s); one segment = one release step (16 frames)
FPS = 1.0
MAX_FRAMES = 8
SMOOTH = 0.3                    # temporal_consistency_weight
DECAY = 2.0                     # 2 x frame_interval, in steps = segments
NO_SPEECH = "(no speech)"
SYSTEM = "You are a helpful assistant specialized in video analysis and anomaly detection."
ANOMALY_DB = {
    "NORMAL": "Regular, expected activities with no safety concerns like normal waliking ,playing ,shopping,talking, cooking, etc",
    "HATEFUL": "An anomaly is hateful content.",
}
IB_ROOT = os.path.join(Q.REPO, "third_party", "Cog-VADU", "ImageBind")
IB_WEIGHTS = os.path.join(Q.REPO, "data", "assets", "imagebind", "imagebind_huge.pth")


def P1(anomaly_descriptions):
    return f"""You are an expert video surveillance analyst identifying unusual events and potential safety issues and anomalies and to rate the video clip in the range [0.000 to 1.000] as an anomlay score based on your analysis. Let's think step by step to analyze this video clip.
      
    ANOMALY CATEGORIES:
    {anomaly_descriptions}"""


def P2(prev_context):
    return f"""

                PREVIOUS CLIP CONTEXT:
                - Previous scene: {prev_context['prev_description']}
                - Previous type: {prev_context.get('prev_type', 'NORMAL')}
                - Previous score: {prev_context.get('prev_score', 0.0):.3f}
                
        Consider temporal consistency. If the current clip shows similar activities, maintain similar ANOMALY_TYPE and score, but rephrase the DESCRIPTION to reflect only what you observe in this clip."""


P3 = "\n\n            Let's analyze this video step-by-step. Provide your response in this EXACT format:\n\n            DESCRIPTION: [Write 2 clear sentences describing ONLY what you observe: people, objects, actions, environment. Be factual and specific. Focus on what is happening, not what is NOT happening.]\n\n            REASONING: [Think through this step-by-step:\n            1) What specific actions are taking place?\n            2) Are these actions normal for this environment or unusual/threatening?\n            3) Do the actions match any anomaly category? Which one and why?\n            4) What is the severity level - minor, moderate, or severe?]\n\n            EXPLANATION: [Based on your reasoning above, provide a clear 1-2 sentence explanation of why this scene is normal or anomalous. Be direct and specific.]\n\n            ANOMALY_SCORE: [Single number between 0.000 and 1.000 low score means normal and high score means abnormal or unusual or potential safety issues]\n            ANOMALY_TYPE: [See what best suits for this video clip. Choose ONE if available otherwise provide yourself: {', '.join(anomaly_types)}]\n           \n\n            SCORING GUIDELINES:\n            • 0.000 = Completely normal, everyday activities with no concerns whatsoever\n            • 1.000 = Severe criminal activity or emergency requiring immediate intervention\n\n            Your score should reflect the SEVERITY and THREAT LEVEL of what you observe on a continuous scale between these two extremes. Consider:\n            - How dangerous or criminal is the activity?\n            - What is the potential for harm to people or property?\n            - How urgent is the need for intervention?\n\n            Assign scores based on your step-by-step reasoning. The score should naturally emerge from your analysis of the actions, context, and severity. There are no fixed boundaries - use the full continuous range from 0.000 to 1.000 based on your judgment.\n\n            CRITICAL INSTRUCTIONS:\n            - Think step-by-step through your reasoning\n            - Be specific about what you observe\n            - Base your score on the severity of the activity\n            - Use the EXACT format shown above\n            - Provide all fields (DESCRIPTION, REASONING, EXPLANATION, ANOMALY_SCORE, ANOMALY_TYPE)\n            Let's begin the analysis:"


def build_cot_prompt(prev_context=None):
    """Raw_CoADTP_STAGE1.py ChainOfThoughtAnomalyProcessor.build_cot_prompt, same strings (P3 is a plain string in
    the release, so its `{', '.join(anomaly_types)}` stays literal)."""
    anomaly_descriptions = '\n'.join([f"  • {k}: {v}" for k, v in ANOMALY_DB.items()])
    prompt = P1(anomaly_descriptions)
    if prev_context and prev_context.get('prev_description'):
        prompt += P2(prev_context)
    prompt += P3
    return prompt


def user_text(words, prev_context):
    return "Transcript: %s\n\n%s" % (words or NO_SPEECH, build_cot_prompt(prev_context))


class Chain:
    """Raw_CoADTP_STAGE1.py VideoAnomalyDetector._parse_videollama_output and the prev_analysis bookkeeping of
    generate_raw_anomaly_scores, for one video. Returns per segment the parsed analysis and how the score was formed
    (`kind`), so the read-out can follow the same rule."""
    REFUSAL = ["can't assist", "cannot assist", "i'm sorry", "i cannot", "i apologize"]

    def __init__(self):
        self.last = None       # _last_analysis
        self.prev = None       # prev_analysis

    def context(self):
        if self.prev is None:
            return None
        return {'prev_description': self.prev["description"], 'prev_type': self.prev["anomaly_type"],
                'prev_score': self.prev["anomaly_score"]}

    def step(self, output):
        default_score = self.prev["anomaly_score"] if self.prev else 0.0
        if any(p in output.lower() for p in self.REFUSAL):
            if self.last:
                a = dict(self.last, reasoning="Model refused. Previous context maintained.", kind="refusal_prev")
            else:
                a = {"description": "Model declined to analyze", "reasoning": "Safety filter triggered",
                     "explanation": "No previous context available", "anomaly_score": 0.0, "anomaly_type": "NORMAL",
                     "kind": "refusal_none"}
            a["parsed"] = None
            self.prev = a
            return a
        lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
        description, reasoning, explanation = "No description provided", "No reasoning provided", "No explanation provided"
        anomaly_score, anomaly_type, parsed = default_score, "NORMAL", None
        current = None
        for line in lines:
            low = line.lower()
            if low.startswith('description:'):
                current = 'description'; description = line.split(':', 1)[1].strip() if ':' in line else line
            elif low.startswith('reasoning:'):
                current = 'reasoning'; reasoning = line.split(':', 1)[1].strip() if ':' in line else line
            elif low.startswith('explanation:'):
                current = 'explanation'; explanation = line.split(':', 1)[1].strip() if ':' in line else line
            elif low.startswith('anomaly_score:'):
                m = re.search(r'(\d*\.?\d+)', line.split(':', 1)[1].strip())
                if m:
                    anomaly_score = round(max(0.0, min(1.0, float(m.group(1)))), 3); parsed = anomaly_score
                current = None
            elif low.startswith('anomaly_type:'):
                anomaly_type = line.split(':', 1)[1].strip().upper()
                if anomaly_type not in ANOMALY_DB:
                    anomaly_type = "NORMAL"
                current = None
            elif current == 'description':
                description += " " + line
            elif current == 'reasoning':
                reasoning += " " + line
            elif current == 'explanation':
                explanation += " " + line
        kind = "first"
        if self.last is not None:
            anomaly_score = round(SMOOTH * self.last["anomaly_score"] + (1 - SMOOTH) * anomaly_score, 3)
            kind = "smooth"
        a = {"description": description.strip(), "reasoning": reasoning.strip(), "explanation": explanation.strip(),
             "anomaly_score": anomaly_score, "anomaly_type": anomaly_type, "kind": kind, "parsed": parsed}
        self.last = a
        self.prev = a
        return a


def score_prefix(answer):
    """The answer up to the ANOMALY_SCORE number the release parser reads (the last line starting with
    `anomaly_score:` that has a number), else the first `ANOMALY_SCORE` followed by a number anywhere; None if
    neither."""
    pos, off = None, 0
    for raw in answer.split('\n'):
        st = raw.strip()
        if st.lower().startswith('anomaly_score:'):
            m = re.search(r'\d', st.split(':', 1)[1])
            if m:
                lead = len(raw) - len(raw.lstrip())
                colon = st.index(':') + 1
                pos = off + lead + colon + m.start()
        off += len(raw) + 1
    if pos is None:
        m = re.search(r'anomaly_score[^\d\n]{0,12}(?=\d)', answer, re.I)
        if m:
            pos = m.end()
    return answer[:pos] if pos is not None else None


def chain_values(units, values):
    """The release's score rule applied to per-segment values (the fp32 read-out): first segment as is, then
    0.3 x the last non-refusal value + 0.7 x this; a refusal reuses the last value (0 if none). No rounding."""
    out, last = [], None
    for u, x in zip(units, values):
        k = u["kind"]
        if k == "refusal_prev":
            s = last
        elif k == "refusal_none":
            s = 0.0
        elif k == "first" or last is None:
            s = x; last = s
        else:
            s = SMOOTH * last + (1 - SMOOTH) * x; last = s
        out.append(s)
    return out


def out_dir(ds):
    return os.path.join(Q.RUNS, "cogvadu", ds, "u8")


def load_records(path):
    recs = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                recs[r["video_id"]] = r
    return recs


def unit_frames(frames, t1, t2):
    return Q.uniform(Q.frames_in(frames, t1, t2), MAX_FRAMES)


# ----------------------------------------------------------------------------- stage 1: the chain (vLLM env)
def cmd_infer(args):
    from src.video_inputs import fixed_windows
    ds = args.dataset
    dest = out_dir(ds)
    L.start_run_log(dest)
    log = Q.log_to(os.path.join(dest, "infer.log"))
    path = os.path.join(dest, "units.jsonl")
    man = L.manifest(ds)
    vids = L.cohort(ds)
    if args.limit:
        vids = vids[:args.limit]
    done = set(load_records(path))
    todo = [v for v in vids if v not in done]
    log("cogvadu %s: %d cohort videos, %d to do; code %s" % (ds, len(vids), len(todo), L.git_version()))
    if not todo:
        log("DONE infer (nothing to do)")
        return
    segs = L.transcript_segments(ds, {v: man[v]["duration"] for v in vids})
    eng = Q.Engine(max_model_len=8192, max_images=MAX_FRAMES, log=log)
    with open(os.path.join(dest, "prompt.txt"), "w") as fh:
        ctx = {"prev_description": "<previous description>", "prev_type": "NORMAL", "prev_score": 0.123}
        for c in (None, ctx):
            fh.write(eng.request([None] * 2, user_text("<words>", c), "x", system=SYSTEM)["prompt"] + "\n\n=====\n\n")
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        try:
            return v, Q.decode_frames(man[v]["video_path"], FPS, "cog_%s_%s" % (ds, v)), None
        except Exception as e:  # noqa: BLE001
            return v, [], repr(e)

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    fut = pool.map(decode, chunks[0])
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        decoded = list(fut)
        if ci + 1 < len(chunks):
            fut = pool.map(decode, chunks[ci + 1])
        plan, failed = {}, {}
        for v, frames, err in decoded:
            if err or not frames:
                failed[v] = err or "no frames decoded"
                continue
            plan[v] = [(k, t1, t2, unit_frames(frames, t1, t2), L.span_text(segs.get(v, []), t1, t2))
                       for k, (t1, t2) in enumerate(fixed_windows(man[v]["duration"], UNIT))]
        chains = {v: Chain() for v in plan}
        per = {v: [] for v in plan}
        steps = max((len(p) for p in plan.values()), default=0)
        n_gen = 0
        for t in range(steps):              # the chain: segment t of every video waits for its segment t - 1
            act = [v for v in plan if t < len(plan[v])]
            reqs, meta = [], []
            for v in act:
                k, t1, t2, files, words = plan[v][t]
                text = user_text(words, chains[v].context())
                reqs.append(eng.request(Q.open_images(files, reader_size=True), text, "%s/%s/u%04d" % (ds, v, k),
                                        system=SYSTEM))
                meta.append((v, k, t1, t2, len(files), words, text))
            outs = eng.generate(reqs, args.max_tokens)
            n_gen += len(outs)
            for (v, k, t1, t2, nf, words, text), o in zip(meta, outs):
                a = chains[v].step(o.text)
                per[v].append({"i": k, "start": t1, "end": t2, "n_frames": nf, "words": words, "text": text,
                               "answer": o.text, "finish": o.finish_reason, "description": a["description"],
                               "anomaly_type": a["anomaly_type"], "parsed": a["parsed"], "score": a["anomaly_score"],
                               "kind": a["kind"]})
        with open(path, "a") as fh:
            for v in chunk:
                fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"],
                                     "error": failed.get(v), "units": per.get(v, [])}) + "\n")
        for v, _, _ in decoded:
            Q.drop_frames("cog_%s_%s" % (ds, v))
        log("chunk %d/%d: %d videos, %d segments, %d chain steps, %d failed videos, %.0f s" % (
            ci + 1, len(chunks), len(chunk), n_gen, steps, len(failed), time.time() - t0))
    log("DONE infer %s" % ds)


# ----------------------------------------------------------------------------- read-out (HateVLM env)
def cmd_readout(args):
    ds = args.dataset
    dest = out_dir(ds)
    log = Q.log_to(os.path.join(dest, "readout.log"))
    recs = load_records(os.path.join(dest, "units.jsonl"))
    path = os.path.join(dest, "readout.jsonl")
    done = set(load_records(path))
    todo = [v for v in recs if v not in done]
    log("cogvadu readout %s: %d videos, %d to do; code %s" % (ds, len(recs), len(todo), L.git_version()))
    if not todo:
        log("DONE readout (nothing to do)")
        return
    man = L.manifest(ds)
    sc = Q.Scorer()
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        if recs[v].get("error"):
            return []
        return Q.decode_frames(man[v]["video_path"], FPS, "cog_ro_%s_%s" % (ds, v))

    ahead = args.workers * 2
    futs = [pool.submit(decode, v) for v in todo[:ahead]]
    t0, n, no_num = time.time(), 0, 0
    for i, v in enumerate(todo):
        if i + ahead < len(todo):
            futs.append(pool.submit(decode, todo[i + ahead]))
        frames = futs[i].result()
        p = []
        for u in recs[v]["units"]:
            files = unit_frames(frames, u["start"], u["end"])
            if len(files) != u["n_frames"]:
                raise RuntimeError("%s segment %d: %d frames now, %d at generation" % (v, u["i"], len(files), u["n_frames"]))
            if u["kind"].startswith("refusal"):
                p.append(None); continue
            pre = score_prefix(u["answer"])
            if pre is None:
                pre = u["answer"].rstrip() + "\nANOMALY_SCORE: "; no_num += 1
            lp1, lp2 = sc.score_digits(Q.open_images(files, reader_size=True), u["text"], pre, system=SYSTEM)
            p.append(Q.expected_score(lp1, lp2)); n += 1
        Q.drop_frames("cog_ro_%s_%s" % (ds, v))
        with open(path, "a") as fh:
            fh.write(json.dumps({"video_id": v, "p": p}) + "\n")
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            log("%d / %d videos, %d segments, %.0f s" % (i + 1, len(todo), n, time.time() - t0))
    log("DONE readout %s: %d segments, %d answers without an ANOMALY_SCORE number, %.0f s" % (
        ds, n, no_num, time.time() - t0))


# ----------------------------------------------------------------------------- ImageBind (HateVideo env)
def imagebind():
    import torch
    sys.path.insert(0, IB_ROOT)
    from imagebind import data
    from imagebind.models.imagebind_model import imagebind_huge
    data.BPE_PATH = os.path.join(IB_ROOT, "bpe", "bpe_simple_vocab_16e6.txt.gz")
    m = imagebind_huge(pretrained=False)
    m.load_state_dict(torch.load(IB_WEIGHTS, map_location="cpu"))
    return m.eval().to("cuda"), data


def cmd_embed(args):
    import torch
    ds = args.dataset
    dest = out_dir(ds)
    log = Q.log_to(os.path.join(dest, "embed.log"))
    recs = load_records(os.path.join(dest, "units.jsonl"))
    edir = os.path.join(dest, "imagebind")
    os.makedirs(edir, exist_ok=True)
    todo = [v for v in recs if recs[v]["units"] and not recs[v].get("error")
            and not os.path.isfile(os.path.join(edir, v + ".npz"))]
    log("cogvadu embed %s: %d videos to do; code %s" % (ds, len(todo), L.git_version()))
    if not todo:
        log("DONE embed (nothing to do)")
        return
    man = L.manifest(ds)
    model, data = imagebind()
    from imagebind.models.imagebind_model import ModalityType
    t0, n = time.time(), 0
    for i, v in enumerate(todo):
        frames = Q.decode_frames(man[v]["video_path"], FPS, "cog_ib_%s_%s" % (ds, v))
        us = recs[v]["units"]
        clips = [unit_frames(frames, u["start"], u["end"]) for u in us]
        vid = []
        with torch.no_grad():
            for b in range(0, len(clips), 16):
                x = data.load_and_transform_video_data(clips[b:b + 16], "cuda")
                vid.append(model({ModalityType.VISION: x})[ModalityType.VISION].float().cpu().numpy())
            texts = [u["description"] for u in us]
            txt = []
            for b in range(0, len(texts), 64):
                x = data.load_and_transform_text(texts[b:b + 64], "cuda")
                txt.append(model({ModalityType.TEXT: x})[ModalityType.TEXT].float().cpu().numpy())
        Q.drop_frames("cog_ib_%s_%s" % (ds, v))
        np.savez(os.path.join(edir, v + ".npz"), video=np.concatenate(vid).astype(np.float32),
                 text=np.concatenate(txt).astype(np.float32))
        n += len(us)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            log("%d / %d videos, %d segments, %.0f s" % (i + 1, len(todo), n, time.time() - t0))
    log("DONE embed %s: %d segments, %.0f s" % (ds, n, time.time() - t0))


# ----------------------------------------------------------------------------- stages 4-6 (CPU)
def refine_video(desc, vemb, temb, scores, cfg):
    """Stages 4-6 for one video. desc: per-segment descriptions; vemb / temb: per-segment ImageBind video / text
    embeddings; scores: per-segment raw scores (chain values). Returns per-segment refined scores."""
    first = {}
    for j, d in enumerate(desc):                       # 4. one index entry per distinct description
        first.setdefault(d, j)
    keys = sorted(first, key=lambda d: first[d])
    rep = [first[d] for d in keys]                     # representative segment (the earliest)
    members = [[j for j, d in enumerate(desc) if d == k] for k in keys]
    T = temb[rep] / np.maximum(np.linalg.norm(temb[rep], axis=1, keepdims=True), 1e-12)
    V = vemb / np.maximum(np.linalg.norm(vemb, axis=1, keepdims=True), 1e-12)
    m, e = len(desc), len(keys)
    k_init = min(cfg["initial_k"], e)
    k_keep = min(cfg["num_neighbors"], k_init)
    out = []
    for i in range(m):
        D = T @ V[i]                                   # inner product of normalised vectors (faiss IndexFlatIP)
        cand = np.argsort(-D, kind="stable")[:k_init]
        dist = np.abs(np.arange(m) - i)
        w = np.where(dist <= cfg["temporal_window"], 1.0 + 0.5 * np.exp(-dist / DECAY), 1.0)
        vis = (V @ V[i]) * w
        comb = [(cfg["text_weight"] * (1.0 / (1.0 + D[c])) + cfg["visual_weight"] * max(vis[j] for j in members[c]), c)
                for c in cand]
        order = sorted(range(len(comb)), key=lambda q: comb[q][0], reverse=True)[:k_keep]   # stable, as list.sort
        kept = [comb[q][1] for q in order]
        sim = np.array([D[c] for c in kept]); val = np.array([scores[rep[c]] for c in kept])
        if len(kept) > cfg["eval_neighbors"]:          # 6. top num_neighbors by similarity, softmax weights
            top = np.argsort(sim, kind="stable")[-cfg["eval_neighbors"]:]
            sim, val = sim[top], val[top]
        wts = np.exp(sim) / np.sum(np.exp(sim))
        out.append(float(np.sum(val * wts)))
    return out


DEFAULT = {"initial_k": 30, "num_neighbors": 10, "eval_neighbors": 10, "text_weight": 0.5, "visual_weight": 0.5,
           "temporal_window": 5}


def load_all(ds):
    dest = out_dir(ds)
    recs = load_records(os.path.join(dest, "units.jsonl"))
    ro = {r["video_id"]: r["p"] for r in load_records(os.path.join(dest, "readout.jsonl")).values()}
    emb = {}
    for v, r in recs.items():
        if r["units"] and not r.get("error"):
            z = np.load(os.path.join(dest, "imagebind", v + ".npz"))
            emb[v] = (z["video"].astype(np.float64), z["text"].astype(np.float64))
    return recs, ro, emb


def curves(ds, readout, cfg, loaded=None):
    recs, ro, emb = loaded or load_all(ds)
    out = {}
    for v, r in recs.items():
        us = r["units"]
        if r.get("error") or not us:
            out[v] = None; continue
        if readout == "prob":
            raw = chain_values(us, ro[v])
        else:
            raw = [u["score"] for u in us]
        s = refine_video([u["description"] for u in us], emb[v][0], emb[v][1], raw, cfg)
        out[v] = L.units_to_4fps([u["start"] for u in us], None, s, L.n_frames(r["duration"]))
    return out


def cmd_raster(args):
    ds = args.dataset
    dest = out_dir(ds)
    loaded = load_all(ds)
    recs = loaded[0]
    us_all = [u for r in recs.values() for u in r["units"]]
    stats = {"videos": len(recs), "segments": len(us_all),
             "finish_length": sum(u["finish"] == "length" for u in us_all),
             "kinds": {k: sum(u["kind"] == k for u in us_all) for k in ("first", "smooth", "refusal_prev", "refusal_none")},
             "no_parsed_score": sum(u["parsed"] is None and not u["kind"].startswith("refusal") for u in us_all),
             "no_score_prefix": sum(score_prefix(u["answer"]) is None for u in us_all),
             "configs": {}}
    for readout in ("prob", "text"):
        name = "%s_default" % readout
        method = "cogvadu_q3vl_u8_" + name
        cur = curves(ds, readout, DEFAULT, loaded)
        rows = {}
        for v, r in recs.items():
            if cur[v] is None:
                rows[v] = L.row(method, ds, v, r["duration"], [], None, __file__, error=r.get("error") or "no segments")
            else:
                rows[v] = L.row(method, ds, v, r["duration"], cur[v], 1.0 / UNIT, __file__)
        med, failed = L.apply_f2(rows, ds)
        stats["configs"][name] = {"f2_videos": failed}
        L.finalize(ds, list(rows.values()), os.path.join(dest, name), method)
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("infer", "readout", "embed", "raster"))
    ap.add_argument("--dataset", required=True, choices=L.DATASETS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64, help="videos whose chains run side by side")
    ap.add_argument("--workers", type=int, default=4, help="ffmpeg decoders")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    {"infer": cmd_infer, "readout": cmd_readout, "embed": cmd_embed, "raster": cmd_raster}[args.stage](args)


if __name__ == "__main__":
    main()
