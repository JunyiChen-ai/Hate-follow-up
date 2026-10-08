#!/usr/bin/env python3
"""CLARA step 3: video-level VLM rationale (`get_rationale_qwen.py`), run in the vLLM venv (.cache/envs/vllm_q3vl:
vllm 0.11.0, torch 2.8.0+cu128, transformers 4.57.6).

    python clara_rationale.py --dataset DS

One rationale per video (train, val and test-cohort videos), exactly the authors' two calls: step A (objective
description) and step B (decision), each with the 20 rationale frames, VIDEO_TITLE, VIDEO_DESCRIPTION and the
full transcription; prompts and tag parsers are imported from the authors' file. A test video's rationale is
shared by all its windows (coordinator decision 2026-10-08: the rationale is a video-level read; regenerating it
per window would double the VLM cost of the published method), so every window carries this video-level term.

Generation: Qwen/Qwen3-VL-8B-Instruct, bf16, temperature 0.2, top_p 0.9, max 2048 new tokens; top_k 20 and
repetition penalty 1.0 come from the model's generation_config, which HF `generate` (the authors' call) applies.
Served with vLLM instead of HF generate for throughput. Prefix caching and the multimodal processor cache are off
and every image gets an explicit id, so vLLM computes no content hash of the frames (hash ban, CLAUDE.md); a guard
makes any call to vLLM's MultiModalHasher fail once the engine is up (start-up memory profiling keys synthetic dummy
images only). Per-request sampling seed = 1000 * step + index of the video in the
sorted id list (no hash-derived seed).
Title / description: DeHate `DeHate_labels.csv` `title` / `desc`; HateMM and HateClipSeg have none ("N/A", the
authors' value for a missing field). Transcription: Whisper large-v3 `text` ("N/A" when empty).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "detwin"))
import common as C  # noqa: E402

DATA = Path(os.environ.get("CLARA_DATA", C.REPO / "data"))  # env override: smoke tests only

RAW = DATA / "clara_raw"
RAT = DATA / "clara_rationale"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_MODEL_LEN = 49152
GEN_TOKENS = 2048


def load_authors():
    p = C.REPO / "third_party/CLARA/data_preprocess/get_raw_fetures/get_rationale_qwen.py"
    spec = importlib.util.spec_from_file_location("clara_get_rationale_qwen", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def text_fields(ds: str) -> dict[str, tuple[str, str]]:
    out = {}
    if ds == "DeHate":
        with (Path.home() / "data/DeHate/DeHate_labels.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                out[r["Video ID"]] = ((r.get("title") or "").strip() or "N/A", (r.get("desc") or "").strip() or "N/A")
    return out


def img_tokens(w: int, h: int) -> int:
    return max(1, round(h / 32)) * max(1, round(w / 32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--chunk", type=int, default=64, help="unused since the KV-aware loop (kept for old commands)")
    ap.add_argument("--kv-tokens", type=int, default=55000, help="KV capacity if vLLM does not report it")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N videos only")
    args = ap.parse_args()
    ds = args.dataset
    log = C.RunLog(C.RUNS / "clara" / ds / "rationale" / "run.log")
    A = load_authors()
    from PIL import Image
    import vllm.multimodal.hasher as H

    def _no_hash(*a, **k):
        raise RuntimeError("content hashing is not allowed (CLAUDE.md hash ban)")
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    sp = C.load_split(ds)
    vids = sorted({x["video_id"] for s in ("train", "val") for x in sp[s]} | {x["video_id"] for x in sp["test"]})
    if args.limit:
        vids = vids[:args.limit]
    index = {v: i for i, v in enumerate(vids)}
    tf = text_fields(ds)
    asr = C.load_asr_text(ds)
    out_dir = RAT / ds / "Qwen"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [v for v in vids if not (out_dir / f"{v}_rationale.json").is_file()]
    log(f"rationale {ds}: {len(vids)} videos, {len(todo)} to do; code {C.code_version()}")
    if not todo:
        log("DONE rationale (nothing to do)")
        return
    processor = AutoProcessor.from_pretrained(MODEL)
    max_len = MAX_MODEL_LEN
    try:
        llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.90,
                  enable_prefix_caching=False, mm_processor_cache_gb=0,
                  limit_mm_per_prompt={"image": 20, "video": 0}, seed=0)
    except Exception as e:  # noqa: BLE001 - one conservative retry (memory / profiling limits)
        log(f"vLLM init failed ({e!r}); retry with max_model_len 32768, eager mode")
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        max_len = 32768
        llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.88,
                  enable_prefix_caching=False, mm_processor_cache_gb=0,
                  limit_mm_per_prompt={"image": 20, "video": 0}, seed=0, enforce_eager=True, max_num_seqs=16)
    log(f"vLLM loaded: {MODEL}, max_model_len {max_len}")
    # Engine start-up profiles memory with synthetic dummy images, which vLLM keys internally; the guard is armed
    # after start-up, so no frame of ours can be hashed (requests carry request-id item ids, caches are off).
    H.MultiModalHasher.hash_kwargs = staticmethod(_no_hash)

    def request(v, images, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": text, "multi_modal_data": {"image": images},
                "multi_modal_uuids": {"image": [f"{ds}/{v}/rationale_frame_{i:03d}" for i in range(len(images))]}}

    def params(step, v):
        return SamplingParams(temperature=0.2, top_p=0.9, top_k=20, repetition_penalty=1.0, max_tokens=2048,
                              seed=1000 * step + index[v])

    # KV-aware admission (2026-10-09). vLLM's scheduler admits requests as long as their *prompts* fit and, when the
    # KV cache later runs out during decoding, preempts a request and recomputes it from scratch (vLLM v1, silent).
    # DeHate frames are often 720p-1080p, so a 20-frame prompt holds 17k-41k image tokens against a 55k-token cache;
    # admitting several such requests made vLLM preempt and recompute them over and over (throughput fell from 2.5k to
    # 0.25k image tokens/s, Slurm 318). Here a request is added only when its prompt + 2048 new tokens fits in the
    # KV cache not yet reserved by running requests, so nothing is ever preempted. Inputs, prompts and sampling are
    # unchanged; step B of a video is queued as soon as its step A finishes.
    from collections import deque
    eng = llm.llm_engine
    try:
        cc = eng.vllm_config.cache_config
        kv_cap = int(cc.num_gpu_blocks) * int(cc.block_size)
    except Exception:  # noqa: BLE001
        kv_cap = 0
    if kv_cap <= 0:
        kv_cap = args.kv_tokens
    kv_cap -= 256
    log(f"KV capacity used for admission: {kv_cap} tokens")
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def plan(v):
        """Everything except the decoded frames: text fields, frame sizes, scale, prompt token counts (cheap)."""
        paths = sorted((RAW / ds / v / "rationale_frames").glob("frame_*.jpg"))[:20]
        if not paths:
            return None
        sizes = []
        for p in paths:
            with Image.open(p) as im:
                sizes.append(im.size)
        title, desc = tf.get(v, ("N/A", "N/A"))
        trans = asr.get(v, "").strip() or "N/A"
        # the step-B prompt (step-A text + up to 2048 step-A tokens) and 2048 new tokens must fit the context;
        # frames are shrunk only when they would not (very large frames or very long transcripts), recorded
        n_text = len(processor.tokenizer(A.prompt_step_a_tagged(title, desc, trans))["input_ids"])
        trunc = False
        while n_text > max_len // 2 and len(trans) > 1000:  # extreme transcripts only; recorded
            trans = trans[: int(len(trans) * 0.8)]
            trunc = True
            n_text = len(processor.tokenizer(A.prompt_step_a_tagged(title, desc, trans))["input_ids"])
        budget = max_len - GEN_TOKENS - (n_text + GEN_TOKENS + 512) - 4 * len(sizes)
        tok = sum(img_tokens(*wh) for wh in sizes)
        scale = 1.0
        if tok > budget:
            scale = math.sqrt(max(budget, 64 * len(sizes)) / tok) * 0.97
            sizes = [(max(32, int(w * scale)), max(32, int(h * scale))) for w, h in sizes]
        inp = {"paths": paths, "sizes": sizes, "title": title, "desc": desc, "trans": trans, "scale": scale,
               "n_frames": len(paths), "trans_truncated": trunc, "img_tokens": sum(img_tokens(*wh) for wh in sizes)}
        inp["need1"] = prompt_tokens(inp, A.prompt_step_a_tagged(title, desc, trans)) + GEN_TOKENS
        return inp

    def load_images(inp):
        imgs = [Image.open(p).convert("RGB") for p in inp["paths"]]
        if inp["scale"] < 1.0:
            imgs = [im.resize(wh) for im, wh in zip(imgs, inp["sizes"])]
        return imgs

    def prompt_tokens(inp, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in inp["paths"]] + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = processor.tokenizer(text)["input_ids"]
        return len(ids) - sum(1 for t in ids if t == pad_id) + inp["img_tokens"] + 64

    t0 = time.time()
    t_last, n_last = t0, 0
    n_ok = 0
    pending = deque(todo)
    ready_b = deque()          # videos whose step A is done
    plans, images, raw_a, step_a, failed = {}, {}, {}, {}, {}
    inflight = {}              # request id -> (video, step, reserved tokens)
    reserved = 0
    WINDOW = 64                # how far down the queue to look for a request that fits

    def finish_video(v, raw_b):
        nonlocal n_ok
        b = A.parse_tagged_step_b(raw_b)
        inp = plans.pop(v)
        images.pop(v, None)
        rec = {"video_id": v, "objective_description": step_a.pop(v),
               "final_decision": {"label": b["final_decision"]["label"],
                                  "explicitness": b["final_decision"]["explicitness"],
                                  "reasons": b.get("reasons", ""), "confidence": b.get("confidence", ""),
                                  "notes": b.get("notes", "")},
               "raw": {"step_a": raw_a.pop(v), "step_b": raw_b},
               "inputs": {"n_frames": inp["n_frames"], "frame_scale": inp["scale"],
                          "transcription_truncated": inp["trans_truncated"],
                          "title": inp["title"] != "N/A", "description": inp["desc"] != "N/A",
                          "transcription_chars": len(inp["trans"]), "image_tokens": inp["img_tokens"]}}
        (out_dir / f"{v}_rationale.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
        n_ok += 1

    def fail(v, why):
        failed[v] = why
        plans.pop(v, None)
        images.pop(v, None)
        (out_dir / f"{v}_rationale.json").write_text(json.dumps({"video_id": v, "error": why}) + "\n")

    def try_admit():
        nonlocal reserved
        admitted = False
        for queue, step in ((ready_b, 2), (pending, 1)):
            for _ in range(min(WINDOW, len(queue))):
                v = queue[0]
                if v not in plans:
                    inp = plan(v)
                    if inp is None:
                        queue.popleft()
                        fail(v, "no rationale frames")
                        continue
                    plans[v] = inp
                inp = plans[v]
                need = inp["need1"] if step == 1 else inp["need2"]
                if reserved + need > kv_cap and inflight:
                    queue.rotate(-1)   # this one waits for room; look at the next one
                    continue
                queue.popleft()
                if v not in images:
                    images[v] = load_images(inp)
                if step == 1:
                    req = request(v, images[v], A.prompt_step_a_tagged(inp["title"], inp["desc"], inp["trans"]))
                else:
                    req = request(v, images[v], A.prompt_step_b_tagged(step_a[v], inp["title"], inp["desc"],
                                                                       inp["trans"]))
                rid = str(next(llm.request_counter))
                try:
                    eng.add_request(rid, req, params(step, v), tokenization_kwargs=None)
                except Exception as e:  # noqa: BLE001 - a bad input for this video only
                    fail(v, f"step {step}: {e!r}")
                    continue
                inflight[rid] = (v, step, need)
                reserved += need
                admitted = True
        return admitted

    while pending or ready_b or inflight:
        while try_admit():
            pass
        if not inflight:
            if pending or ready_b:
                continue
            break
        for o in eng.step():
            if not o.finished:
                continue
            v, step, need = inflight.pop(o.request_id)
            reserved -= need
            text = o.outputs[0].text.strip()
            if step == 1:
                raw_a[v] = text
                step_a[v] = A.parse_tagged_step_a(text)
                inp = plans[v]
                inp["need2"] = prompt_tokens(inp, A.prompt_step_b_tagged(step_a[v], inp["title"], inp["desc"],
                                                                        inp["trans"])) + GEN_TOKENS
                ready_b.append(v)
            else:
                finish_video(v, text)
                if (n_ok - n_last) >= 64:
                    now = time.time()
                    log(f"  {n_ok}/{len(todo)} videos, {now - t0:.0f}s; last {n_ok - n_last}: "
                        f"{(now - t_last) / (n_ok - n_last):.2f} s/video; in flight {len(inflight)}, "
                        f"reserved {reserved}/{kv_cap} tokens")
                    t_last, n_last = now, n_ok
        if len(failed) >= 20 and n_ok == 0:
            raise RuntimeError(f"first {len(failed)} videos failed, e.g. {next(iter(failed.items()))}")
    log(f"DONE rationale: {n_ok} written, {len(failed)} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
