"""Score unaligned/aligned/shuffled temporal-evidence arms."""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "duplex"))
from score_duplex_probe import (DUPLEX_PROMPT, READER_BLOCKS, SYSTEM_MESSAGE,
                                YOUTUBE_RULES, build_binary_token_ids)
from extract_duplex_readout import MAX_PIXELS, MIN_PIXELS, resolve_frames
from data_utils import load_annotations

RUN = os.path.join(ROOT, "results", "temporal_attribution")
ARMS = ("unaligned", "aligned", "shuffled")


def read_jsonl(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("error") is None:
            out[r["video_id"]] = r
    return out


def load_durations():
    out = {}
    for line in open(os.path.join(ROOT, "results/c2_fullcorpus/audio_meta.jsonl")):
        r = json.loads(line)
        out[r["video_id"]] = float(r.get("wav_duration") or r.get("container_duration") or 0)
    return out


def make_bins(record, duration, n=16):
    bins = [[] for _ in range(n)]
    for c in record["chunks"]:
        start = 0.0 if c["start"] is None else float(c["start"])
        end = start if c["end"] is None else float(c["end"])
        mid = (start + end) / 2
        idx = min(n - 1, max(0, int(mid / max(duration, 1e-6) * n)))
        if c["text"]:
            bins[idx].append(c["text"])
    return [" ".join(x).strip() for x in bins]


def permutation(vid, n=16):
    seed = int(hashlib.sha256(("temporal-pilot:" + vid).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    p = list(range(n))
    while True:
        rng.shuffle(p)
        if all(i != p[i] for i in range(n)):
            return p


def messages(ann, arm, bins):
    tail = DUPLEX_PROMPT.format(
        title=ann.get("title", "") or "",
        transcript="[Speech evidence is provided in the 16 blocks above.]",
        rules=YOUTUBE_RULES, reader_block=READER_BLOCKS["prag"])
    content = []
    if arm == "unaligned":
        content.extend({"type": "image"} for _ in range(16))
        content.extend({"type": "text", "text": f"[Segment {i+1}/16 speech] {bins[i] or '[no speech]'}"}
                       for i in range(16))
    else:
        order = list(range(16)) if arm == "aligned" else permutation(ann["video_id"])
        for i in range(16):
            content.append({"type": "image"})
            text = bins[order[i]]
            content.append({"type": "text", "text": f"[Segment {i+1}/16 speech] {text or '[no speech]'}"})
    content.append({"type": "text", "text": tail})
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content}]


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                if np.isfinite(r["z"]): done.add((r["video_id"], r["arm"]))
            except Exception: pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cohort = json.load(open(os.path.join(RUN, "cohorts.json")))
    cells = {v: c for c, vs in cohort["cohorts"].items() for v in vs}
    ids = sorted(cells)
    if args.limit: ids = ids[:args.limit]
    asr = read_jsonl(os.path.join(RUN, "timestamped_asr.jsonl"))
    durations = load_durations()
    ann = load_annotations("ImpliHateVid")
    out_path = os.path.join(RUN, "probe_scores.jsonl")
    done = load_done(out_path)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    processor = AutoProcessor.from_pretrained(args.model)
    tokids = build_binary_token_ids(processor.tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    yes = torch.tensor(sorted(tokids["Yes"]), device=model.device)
    no = torch.tensor(sorted(tokids["No"]), device=model.device)
    size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    t0, calls = time.time(), 0
    for j, vid in enumerate(ids, 1):
        frames = resolve_frames(vid, "ImpliHateVid", 16)
        if len(frames) != 16 or vid not in asr:
            raise SystemExit(f"{vid}: expected 16 frames and timestamped ASR")
        images = [Image.open(p).convert("RGB") for p in frames]
        bins = make_bins(asr[vid], durations[vid])
        item = dict(ann[vid]); item["video_id"] = vid
        for arm in ARMS:
            if (vid, arm) in done: continue
            msg = messages(item, arm, bins)
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt", size=size).to(model.device)
            with torch.no_grad():
                o = model(**inputs, use_cache=False, logits_to_keep=1)
                logits = o.logits[0, -1, :].float()
                z = float(torch.logsumexp(logits[yes], 0) - torch.logsumexp(logits[no], 0))
            rec = {"video_id": vid, "cell": cells[vid], "arm": arm, "z": z,
                   "n_tokens": int(inputs["input_ids"].shape[1]),
                   "n_nonempty_bins": sum(bool(x) for x in bins)}
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
            del o, logits, inputs
            calls += 1
        for im in images: im.close()
        if j == 1 or j % 10 == 0:
            logging.info("[%d/%d] %s calls=%d %.2fs/call", j, len(ids), vid, calls,
                         (time.time()-t0)/max(calls,1))
    logging.info("done: %d calls in %.1fs", calls, time.time()-t0)


if __name__ == "__main__":
    main()
