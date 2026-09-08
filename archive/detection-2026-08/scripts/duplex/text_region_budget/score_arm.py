#!/usr/bin/env python
"""Stage 3 of the text-region pixel-budget probe: score one visual arm.

Everything that is not the image content is imported from
`src/duplex/extract_duplex_readout.py`, which produced the BASELINE scores:
the prompt builder, the transcript-override logic, the pixel-budget size
arguments and the raw-z readout. This module only replaces the frame list with
the arm manifest and drops the hidden-state dump, which no metric in this probe
reads.

Output: <out-dir>/scores.jsonl, one record per video, resume-safe.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS, MIN_PIXELS, build_messages, resolve_transcript,
)
from score_duplex_probe import (  # noqa: E402
    BILIBILI_RULES, YOUTUBE_RULES, build_binary_token_ids,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r.get("z"), (int, float)) and np.isfinite(r["z"]):
                done.add(r["video_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHClip_EN")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--transcript-override-json", required=True)
    ap.add_argument("--transcript-limit", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    os.makedirs(args.out_dir, exist_ok=True)
    scores_path = os.path.join(args.out_dir, "scores.jsonl")

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.transcript_override_json) as f:
        overrides = json.load(f)

    annotations = load_annotations(args.dataset)
    split_ids = load_clean_split_ids(args.dataset, args.split)
    seen, ids = set(), []
    for v in split_ids:
        if v not in seen:
            seen.add(v)
            ids.append(v)
    rules_text = BILIBILI_RULES if args.dataset == "MHClip_ZH" else YOUTUBE_RULES
    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

    done = load_done(scores_path)
    remaining = [v for v in ids if v not in done]
    logging.info(f"arm manifest={args.manifest} n_ids={len(ids)} "
                 f"done={len(done)} remaining={len(remaining)} size={size_kwarg}")
    if not remaining:
        logging.info("Nothing to do.")
        return

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    label_token_ids = build_binary_token_ids(tokenizer)
    yes_ids = sorted(label_token_ids["Yes"])
    no_ids = sorted(label_token_ids["No"])
    merge_size = processor.image_processor.merge_size

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    yes_idx = torch.tensor(yes_ids, device=model.device)
    no_idx = torch.tensor(no_ids, device=model.device)

    status_path = os.path.join(args.out_dir, "STATUS")
    t0 = time.time()
    n_done = 0
    for i, vid in enumerate(remaining, 1):
        entry = manifest[vid]
        paths = entry["images"]
        ann = annotations[vid]
        images = [Image.open(p).convert("RGB") for p in paths]
        transcript = resolve_transcript(ann, vid, overrides, args.transcript_limit)
        messages = build_messages(ann, paths, rules_text, transcript)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           size=size_kwarg)
        for im in images:
            im.close()
        del images

        grid = inputs["image_grid_thw"]
        per_slot = [int(t * h * w) // (merge_size ** 2) for t, h, w in grid.tolist()]
        observed = sum(per_slot)
        if observed != entry["predicted_image_tokens"]:
            raise SystemExit(
                f"ABORT {vid}: image tokens {observed} != predicted "
                f"{entry['predicted_image_tokens']}")

        inputs = inputs.to(model.device)
        n_tokens_total = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False, logits_to_keep=1)
            last_logits = outputs.logits[0, -1, :].float()
            z = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                      - torch.logsumexp(last_logits[no_idx], dim=0))
        del outputs, last_logits, inputs
        if not np.isfinite(z):
            raise SystemExit(f"ABORT {vid}: non-finite z")

        rec = {"video_id": vid, "z": z,
               "p_yes_renorm": float(1.0 / (1.0 + np.exp(-z))),
               "n_tokens_total": n_tokens_total,
               "n_image_tokens_total": observed,
               "n_image_slots": len(per_slot),
               "no_text": bool(entry["no_text"]),
               "n_transcript_chars": len(transcript),
               "transcript_source": "override" if vid in overrides else "dataset"}
        with open(scores_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n_done += 1
        with open(status_path, "w") as f:
            f.write(f"{len(done) + n_done}/{len(ids)} {time.strftime('%F %T')}\n")
        if n_done % 25 == 0 or i == 1:
            el = time.time() - t0
            logging.info(f"  [{n_done}/{len(remaining)}] z={z:+.3f} "
                         f"img_tok={observed} tokens={n_tokens_total} "
                         f"{el / n_done:.2f}s/video")

    total = len(load_done(scores_path))
    logging.info(f"Done: {n_done} scored this pass, {total}/{len(ids)} on disk, "
                 f"{time.time() - t0:.1f}s")
    if total != len(ids):
        raise SystemExit(f"ABORT: coverage {total} != expected {len(ids)}")
    with open(status_path, "w") as f:
        f.write(f"COMPLETE {total}/{len(ids)} {time.strftime('%F %T')}\n")


if __name__ == "__main__":
    main()
