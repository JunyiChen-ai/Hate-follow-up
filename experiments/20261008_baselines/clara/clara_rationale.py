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
    ap.add_argument("--chunk", type=int, default=64)
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

    t0 = time.time()
    n_ok = 0
    for c0 in range(0, len(todo), args.chunk):
        chunk = todo[c0:c0 + args.chunk]
        inputs = {}
        for v in chunk:
            paths = sorted((RAW / ds / v / "rationale_frames").glob("frame_*.jpg"))
            if not paths:
                inputs[v] = None
                continue
            imgs = [Image.open(p).convert("RGB") for p in paths[:20]]
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
            budget = max_len - GEN_TOKENS - (n_text + GEN_TOKENS + 512) - 4 * len(imgs)
            tok = sum(img_tokens(*im.size) for im in imgs)
            scale = 1.0
            if tok > budget:
                scale = math.sqrt(max(budget, 64 * len(imgs)) / tok) * 0.97
                imgs = [im.resize((max(32, int(im.width * scale)), max(32, int(im.height * scale)))) for im in imgs]
            inputs[v] = {"images": imgs, "title": title, "desc": desc, "trans": trans, "scale": scale,
                         "n_frames": len(imgs), "trans_truncated": trunc}
        live = [v for v in chunk if inputs[v] is not None]

        def gen(step, vs, prompt_of):
            """Batched generate; if the batch fails, retry one by one so one bad video cannot stop the rest."""
            try:
                outs = llm.generate([request(v, inputs[v]["images"], prompt_of(v)) for v in vs],
                                    [params(step, v) for v in vs], use_tqdm=False)
                return {v: o.outputs[0].text.strip() for v, o in zip(vs, outs)}
            except Exception as e:  # noqa: BLE001
                log(f"  batch step {step} failed ({e!r}); one by one")
                res = {}
                for v in vs:
                    try:
                        o = llm.generate([request(v, inputs[v]["images"], prompt_of(v))], [params(step, v)],
                                         use_tqdm=False)[0]
                        res[v] = o.outputs[0].text.strip()
                    except Exception as e2:  # noqa: BLE001
                        failed[v] = f"step {step}: {e2!r}"
                return res

        failed = {}
        raw_a = gen(1, live, lambda v: A.prompt_step_a_tagged(inputs[v]["title"], inputs[v]["desc"], inputs[v]["trans"]))
        step_a = {v: A.parse_tagged_step_a(t) for v, t in raw_a.items()}
        live_b = [v for v in live if v in step_a]
        raw_bs = gen(2, live_b, lambda v: A.prompt_step_b_tagged(step_a[v], inputs[v]["title"], inputs[v]["desc"],
                                                                 inputs[v]["trans"]))
        if live and len(failed) == len(live):  # a systematic error, not a bad video: stop instead of writing errors
            raise RuntimeError(f"every video of the chunk failed, e.g. {next(iter(failed.items()))}")
        for v in failed:
            (out_dir / f"{v}_rationale.json").write_text(json.dumps({"video_id": v, "error": failed[v]}) + "\n")
        for v, raw_b in raw_bs.items():
            b = A.parse_tagged_step_b(raw_b)
            rec = {"video_id": v, "objective_description": step_a[v],
                   "final_decision": {"label": b["final_decision"]["label"],
                                      "explicitness": b["final_decision"]["explicitness"],
                                      "reasons": b.get("reasons", ""), "confidence": b.get("confidence", ""),
                                      "notes": b.get("notes", "")},
                   "raw": {"step_a": raw_a[v], "step_b": raw_b},
                   "inputs": {"n_frames": inputs[v]["n_frames"], "frame_scale": inputs[v]["scale"],
                              "transcription_truncated": inputs[v]["trans_truncated"],
                              "title": inputs[v]["title"] != "N/A", "description": inputs[v]["desc"] != "N/A",
                              "transcription_chars": len(inputs[v]["trans"])}}
            (out_dir / f"{v}_rationale.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
            n_ok += 1
        for v in chunk:
            if inputs[v] is None:
                (out_dir / f"{v}_rationale.json").write_text(json.dumps(
                    {"video_id": v, "error": "no rationale frames"}) + "\n")
        el = time.time() - t0
        log(f"  {min(c0 + args.chunk, len(todo))}/{len(todo)} videos, {el:.0f}s ({el / max(n_ok, 1):.2f} s/video)")
    log(f"DONE rationale: {n_ok} written, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
