#!/usr/bin/env python3
"""CoReVAD (ICPR 2026, arXiv 2605.23116, github.com/Muk-00/CoReVAD) on Qwen3-VL-8B with frames + Whisper words.
README §3.

CoReVAD is training-free and uses one frozen VLM: for every segment the VLM looks at 8 frames and writes two lines,
a binary decision ("Anomalous scenes." / "Normal scenes.") and one sentence describing the scene. The decisions are
then refined with frozen CLIP ViT-B/16 features, in this order (src/ucf/LRC.py, src/ucf/eval_ucf.py):
  1. Local Response Cleaning: segment i takes the response of the neighbour j in [i - 1, i + 1] whose text embedding
     is closest to the segment's mean frame embedding;
  2. softmax refinement: the score of segment i is the average of all segments' decisions, weighted by
     softmax(cos(frames_i, description_k) / tau) over k, tau = 0.05;
  3. Gaussian smoothing of the segment scores (kernel size 9, sigma 5, in segments);
  4. position weighting: the frame scores are multiplied by a Gaussian centred at the middle of the video with a
     standard deviation of half its length.

Kept: the prompt structure and output format (P_VLM_format.txt), 8 frames per segment, the four refinement steps and
their formulas and default constants. Changed (README §3 lists every deviation):
- backbone Qwen3-VL-8B instead of InternVL2-8B, frames at the image size of the naive row;
- the segment's Whisper words are added to the prompt input ("Transcript: <words>"), as in every row of §1-§2;
- prompt wording: "anomaly" is replaced by "hateful content" and the decision line is "Hateful scenes." /
  "Normal scenes." (PROMPT below; fixed before any run);
- segment length `--unit` seconds (the release uses 30 frames, about 1 s), for cost; default 8 s with 1 fps frames,
  the unit and frames of the naive row;
- the assistant turn is started with the opening code fence, so the decision is the first thing written;
- read-outs of the decision: `prob` = P(Hateful) from the fp32 output layer at the position where the decision is
  written (restricted to the first tokens of the two decision lines; README §2 shared rule), `label` = the decoded
  0 / 1 decision (the paper's);
- CLIP frame embedding of a segment = mean over its 8 sampled frames (the release averages all frames of the
  segment), standard CLIP preprocessing.

Stages: `infer` (GPU, vLLM env) -> runs/20261113_lf_baselines/corevad/<DS>/u<unit>/units.jsonl; `readout` (GPU,
HateVLM env) -> readout.jsonl (decision log-odds) and clip.npz (segment frame embeddings, text embeddings of the
responses); `raster` (CPU, HateVideo env) -> one directory per configuration with the canonical evaluator's
metrics.json. No label is read.
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

MAX_FRAMES = 8
NO_SPEECH = "(no speech)"
CLIP = "openai/clip-vit-base-patch16"
FENCE = "```\n"
POS, NEG = "Hateful scenes.", "Normal scenes."
# P_VLM_format.txt of the release with the anomaly wording replaced by hateful content and the transcript added to
# the input; "[$Data]" is replaced by the frames ("Frame<k>: <image>") and the transcript line.
PROMPT_HEAD = """You are the model.

**Model Description:**
You are designed for hateful video detection. The input is a sequence of video frames and the transcript of the speech for identifying whether there is hateful content in the video.

**Prompt Questions:**
Answer the following questions based on what you see from the video frames and read in the transcript:
- First, decide whether the scene shows hateful content or just normal/non-hateful content.
- Then, write ONE short description of what the scene shows.
Based on the analysis above, please conclude your answer.

**Input:**
["""
PROMPT_TAIL = """]

**Output Rules (STRICT):**
- Based on the analysis above, reply with ONLY a fenced code block (no language tag) that has EXACTLY two lines.
  1) write **"Hateful scenes."** if you detected any hateful content in the frames or the transcript; otherwise write **"Normal scenes."**
  2) write ONE short sentence describing what is happening.
- Do NOT add any other text before or after the code block."""


def messages(n_images, text):
    content = [{"type": "text", "text": PROMPT_HEAD}]
    for k in range(n_images):
        content += [{"type": "text", "text": "Frame%d: " % (k + 1)}, {"type": "image"}, {"type": "text", "text": "\n"}]
    content.append({"type": "text", "text": "Transcript: %s%s" % (text or NO_SPEECH, PROMPT_TAIL)})
    return [{"role": "user", "content": content}]


def out_dir(ds, unit):
    return os.path.join(Q.RUNS, "corevad", ds, "u%g" % unit)


def load_records(path):
    recs = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["video_id"]] = r
    return recs


def unit_frames(frames, t1, t2):
    return Q.uniform(Q.frames_in(frames, t1, t2), MAX_FRAMES)


def parse(answer):
    """(decision line, description line) of the generated text that follows the opening fence."""
    lines = [x.strip() for x in (answer or "").replace("```", "\n").splitlines() if x.strip()]
    first = lines[0] if lines else ""
    return first, (lines[1] if len(lines) > 1 else first)


def decoded_label(first):
    """The release's first_sentence_to_label: 1 for the positive decision line, else 0."""
    return 1.0 if first.lower().startswith("hateful scenes") else 0.0


def cmd_infer(args):
    from src.video_inputs import fixed_windows
    ds = args.dataset
    dest = out_dir(ds, args.unit)
    L.start_run_log(dest)
    log = Q.log_to(os.path.join(dest, "infer.log"))
    path = os.path.join(dest, "units.jsonl")
    man = L.manifest(ds)
    vids = L.cohort(ds)
    if args.limit:
        vids = vids[:args.limit]
    done = set(load_records(path)) if os.path.isfile(path) else set()
    todo = [v for v in vids if v not in done]
    fps = max(1.0, MAX_FRAMES / args.unit)
    log("corevad %s unit %g s (%g fps decode): %d cohort videos, %d to do; code %s" % (
        ds, args.unit, fps, len(vids), len(todo), L.git_version()))
    if not todo:
        log("DONE infer (nothing to do)")
        return
    segs = L.transcript_segments(ds, {v: man[v]["duration"] for v in vids})
    eng = Q.Engine(max_model_len=8192, max_images=MAX_FRAMES, log=log)
    with open(os.path.join(dest, "prompt.txt"), "w") as fh:
        fh.write(eng.processor.apply_chat_template(messages(2, "<words>"), tokenize=False,
                                                   add_generation_prompt=True) + FENCE)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        try:
            return v, Q.decode_frames(man[v]["video_path"], fps, "crv_%s_%s" % (ds, v)), None
        except Exception as e:  # noqa: BLE001
            return v, [], repr(e)

    def request(files, text, uid):
        images = Q.open_images(files, reader_size=True)
        prompt = eng.processor.apply_chat_template(messages(len(images), text), tokenize=False,
                                                   add_generation_prompt=True) + FENCE
        return {"prompt": prompt, "multi_modal_data": {"image": images},
                "multi_modal_uuids": {"image": ["%s/img%03d" % (uid, i) for i in range(len(images))]}}

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    fut = pool.map(decode, chunks[0])
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        decoded = list(fut)
        if ci + 1 < len(chunks):
            fut = pool.map(decode, chunks[ci + 1])
        units, failed = [], {}
        for v, frames, err in decoded:
            if err or not frames:
                failed[v] = err or "no frames decoded"
                continue
            for k, (t1, t2) in enumerate(fixed_windows(man[v]["duration"], args.unit)):
                files = unit_frames(frames, t1, t2)
                text = L.span_text(segs.get(v, []), t1, t2)
                units.append((v, k, t1, t2, len(files), text, request(files, text, "%s/%s/u%04d" % (ds, v, k))))
        outs = eng.generate([u[6] for u in units], args.max_tokens) if units else []
        per = {}
        for (v, k, t1, t2, nf, text, _), o in zip(units, outs):
            per.setdefault(v, []).append({"i": k, "start": t1, "end": t2, "n_frames": nf, "text": text,
                                          "answer": o.text, "finish": o.finish_reason})
        with open(path, "a") as fh:
            for v in chunk:
                fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"],
                                     "error": failed.get(v), "units": per.get(v, [])}) + "\n")
        for v, _, _ in decoded:
            Q.drop_frames("crv_%s_%s" % (ds, v))
        log("chunk %d/%d: %d videos, %d segments, %d failed videos, %.0f s" % (
            ci + 1, len(chunks), len(chunk), len(units), len(failed), time.time() - t0))
    log("DONE infer %s unit %g" % (ds, args.unit))


def cmd_readout(args):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from src.mllm_judge import Judge
    ds = args.dataset
    dest = out_dir(ds, args.unit)
    log = Q.log_to(os.path.join(dest, "readout.log"))
    recs = load_records(os.path.join(dest, "units.jsonl"))
    path = os.path.join(dest, "readout.jsonl")
    done = set(load_records(path)) if os.path.isfile(path) else set()
    todo = [v for v in recs if v not in done]
    log("corevad readout %s unit %g: %d videos, %d to do; code %s" % (ds, args.unit, len(recs), len(todo),
                                                                     L.git_version()))
    if not todo:
        log("DONE readout (nothing to do)")
        return
    man = L.manifest(ds)
    fps = max(1.0, MAX_FRAMES / args.unit)
    j = Judge(Q.MODEL)
    first = [j.tok.encode(w, add_special_tokens=False)[0] for w in (POS, NEG)]
    if first[0] == first[1]:
        raise SystemExit("the two decision lines start with the same token")
    cand = torch.tensor(first, device=j.device)
    log("decision tokens: %r vs %r" % (j.tok.decode(first[:1]), j.tok.decode(first[1:])))
    clip = CLIPModel.from_pretrained(CLIP).to(j.device).eval()
    cproc = CLIPProcessor.from_pretrained(CLIP)

    def embed_text(texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = cproc(text=[t or "" for t in texts[i:i + 64]], return_tensors="pt", padding=True, truncation=True,
                        max_length=77).to(j.device)
            f = clip.get_text_features(**enc).float()
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        f = torch.cat(out).numpy()
        f[[not t for t in texts]] = 0.0          # the release's text_encode: a zero vector for an empty response
        return f

    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        if recs[v].get("error"):
            return []
        return Q.decode_frames(man[v]["video_path"], fps, "crv_ro_%s_%s" % (ds, v))

    ahead = args.workers * 2
    futs = [pool.submit(decode, v) for v in todo[:ahead]]
    t0, n = time.time(), 0
    for i, v in enumerate(todo):
        if i + ahead < len(todo):
            futs.append(pool.submit(decode, todo[i + ahead]))
        frames = futs[i].result()
        z, img = [], []
        with torch.inference_mode():
            for u in recs[v]["units"]:
                files = unit_frames(frames, u["start"], u["end"])
                if len(files) != u["n_frames"]:
                    raise RuntimeError("%s segment %d: %d frames now, %d at generation" % (
                        v, u["i"], len(files), u["n_frames"]))
                images = Q.open_images(files, reader_size=True)
                full = j.render(messages(len(images), u["text"]), add_generation_prompt=True) + FENCE
                out = j.model.model(**j.model_inputs(j.encode(full, images)), use_cache=False)
                lg = j._logits_fp32(out.last_hidden_state[0, -1:], cand)[0]
                z.append(float(lg[0] - lg[1]))
                del out
                enc = cproc(images=images, return_tensors="pt").to(j.device)
                f = torch.nn.functional.normalize(clip.get_image_features(**enc).float(), dim=-1)
                img.append(f.mean(0).cpu().numpy())      # mean of the normalised frame features, not renormalised
                n += 1
            us = recs[v]["units"]
            pairs = [parse(u["answer"]) for u in us]
            full_t = embed_text([" ".join(p).strip() if p[0] else "" for p in pairs]) if us else np.zeros((0, 512))
            body_t = embed_text([p[1] for p in pairs]) if us else np.zeros((0, 512))
        Q.drop_frames("crv_ro_%s_%s" % (ds, v))
        if us:
            os.makedirs(os.path.join(dest, "clip"), exist_ok=True)
            np.savez(os.path.join(dest, "clip", v + ".npz"), image=np.asarray(img, dtype=np.float32),
                     text_full=full_t.astype(np.float32), text_body=body_t.astype(np.float32))
        with open(path, "a") as fh:
            fh.write(json.dumps({"video_id": v, "z": z}) + "\n")
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            log("%d / %d videos, %d segments, %.0f s" % (i + 1, len(todo), n, time.time() - t0))
    log("DONE readout %s unit %g: %d segments in %.0f s" % (ds, args.unit, n, time.time() - t0))


# ----------------------------------------------------------------------------- refinement (CPU)
def gaussian_smooth_1d(data, size, sigma):
    """The release's gaussian_smooth_1d (eval_ucf.py): kernel on linspace(-size // 2, size // 2, size), zero padding
    at the ends (np.convolve mode "same")."""
    kernel = np.exp(-np.linspace(-size // 2, size // 2, size) ** 2 / (2 * sigma ** 2))
    full = np.convolve(data, kernel / kernel.sum(), mode="full")
    lo = (size - 1) // 2                 # mode="same" for len(data) >= size; the same slice for shorter videos
    return full[lo:lo + len(data)]


def refine(y, image, text_full, text_body, cfg):
    """Segment scores of one video after steps 1-3; y = the per-segment decision (probability or 0 / 1)."""
    y = np.asarray(y, dtype=np.float64)
    m = len(y)
    body = text_body
    if cfg["lrc"]:                      # 1. Local Response Cleaning, window 1
        sims = image @ text_full.T       # (segment, response)
        pick = np.empty(m, dtype=int)
        for i in range(m):
            lo, hi = max(0, i - 1), min(m, i + 2)
            pick[i] = lo + int(np.argmax(sims[i, lo:hi]))
        y, body = y[pick], text_body[pick]
    if cfg["tau"] is not None:          # 2. softmax refinement over all segments of the video
        sims = image @ body.T
        w = np.exp((sims - sims.max(axis=1, keepdims=True)) / cfg["tau"])
        y = (w / (w.sum(axis=1, keepdims=True) + 1e-12)) @ y
    if cfg["size"] > 1:                 # 3. Gaussian smoothing
        y = gaussian_smooth_1d(y, cfg["size"], cfg["sigma"])
    return y


def position_weight(curve):
    """4. The release's gaussian_smoothing: multiply by a Gaussian centred at the middle frame, sigma = half the
    number of frames."""
    n = len(curve)
    sigma = int(n * 0.5)
    if sigma < 1:
        return curve
    return curve * np.exp(-0.5 * ((np.arange(n) - int(n / 2)) / sigma) ** 2)


def config_name(c):
    return "%s_lrc%d_tau%s_g%d-%g_pos%d" % (c["readout"], c["lrc"], "off" if c["tau"] is None else "%g" % c["tau"],
                                            c["size"], c["sigma"], c["pos"])


def parse_config(s):
    """e.g. prob_lrc1_tau0.05_g9-5_pos1"""
    m = re.fullmatch(r"(prob|label)_lrc([01])_tau(off|[0-9.]+)_g(\d+)-([0-9.]+)_pos([01])", s)
    if not m:
        raise SystemExit("bad configuration name %r" % s)
    return {"readout": m.group(1), "lrc": int(m.group(2)), "tau": None if m.group(3) == "off" else float(m.group(3)),
            "size": int(m.group(4)), "sigma": float(m.group(5)), "pos": int(m.group(6))}


DEFAULT_CONFIGS = ["prob_lrc1_tau0.05_g9-5_pos1", "label_lrc1_tau0.05_g9-5_pos1"]


def cmd_raster(args):
    ds = args.dataset
    dest = out_dir(ds, args.unit)
    recs = load_records(os.path.join(dest, "units.jsonl"))
    zs = {r["video_id"]: r["z"] for r in load_records(os.path.join(dest, "readout.jsonl")).values()}
    stats = {"videos": len(recs), "segments": sum(len(r["units"]) for r in recs.values()),
             "finish_length": sum(u["finish"] == "length" for r in recs.values() for u in r["units"]),
             "no_decision_line": sum(parse(u["answer"])[0].lower().rstrip(".") not in ("hateful scenes", "normal scenes")
                                     for r in recs.values() for u in r["units"]),
             "decoded_positive": sum(decoded_label(parse(u["answer"])[0]) for r in recs.values() for u in r["units"]),
             "configs": {}}
    feats = {}
    for v, r in recs.items():
        if r["units"] and not r.get("error"):
            feats[v] = np.load(os.path.join(dest, "clip", v + ".npz"))
    for name in args.config:
        cfg = parse_config(name)
        method = "corevad_q3vl_u%g_%s" % (args.unit, name)
        rows = {}
        for v, r in recs.items():
            n = L.n_frames(r["duration"])
            us = r["units"]
            if r.get("error") or not us:
                rows[v] = L.row(method, ds, v, r["duration"], [], None, __file__, error=r.get("error") or "no segments")
                continue
            if cfg["readout"] == "prob":
                y = 1.0 / (1.0 + np.exp(-np.asarray(zs[v], dtype=np.float64)))
            else:
                y = np.asarray([decoded_label(parse(u["answer"])[0]) for u in us])
            f = feats[v]
            s = refine(y, f["image"].astype(np.float64), f["text_full"].astype(np.float64),
                       f["text_body"].astype(np.float64), cfg)
            curve = L.units_to_4fps([u["start"] for u in us], None, s, n)
            if cfg["pos"]:
                curve = position_weight(curve)
            rows[v] = L.row(method, ds, v, r["duration"], curve, 1.0 / args.unit, __file__)
        med, failed = L.apply_f2(rows, ds)
        stats["configs"][name] = {"f2_videos": failed}
        L.finalize(ds, list(rows.values()), os.path.join(dest, name), method)
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("infer", "readout", "raster"))
    ap.add_argument("--dataset", required=True, choices=L.DATASETS)
    ap.add_argument("--unit", type=float, default=8.0, help="segment length in seconds")
    ap.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS, help="raster: configuration names")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16, help="videos per vLLM batch")
    ap.add_argument("--workers", type=int, default=4, help="ffmpeg decoders")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()
    {"infer": cmd_infer, "readout": cmd_readout, "raster": cmd_raster}[args.stage](args)


if __name__ == "__main__":
    main()
