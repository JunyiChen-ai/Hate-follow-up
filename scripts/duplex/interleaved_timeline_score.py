"""Interleaved-timeline kill test, stage C/D (GPU): the judge under three arms.

One HF-transformers forward pass per video, reading the same raw logit contrast
the frozen judge reads:

    z = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids])

at the final prompt position. Model, dtype, pixel budget, frame count, system
message, platform rules, `prag` reader block, question and Yes/No token-id sets
are imported from the frozen modules and are identical to
`src/duplex/extract_duplex_readout.py`. Hidden states are not stored; this test
needs only the scalar.

The arms differ in one thing: where the transcript sits in the content list.

  baseline  -- sixteen images, then one text block carrying title, whole
               transcript, rules, reader block, question. Reproduces the scores
               already on disk and exists only as a replication check.
  interleaved -- frame i, then the transcript segment assigned to frame i, for
               i = 0..15, then the same trailing block with the transcript field
               replaced by a fixed pointer string.
  misaligned -- identical to interleaved except that frame i receives segment
               (i + 8) mod 16. Same format, same text, wrong pairing.

Empty segments emit no content item, so the two interleaved arms carry the same
number of items for every video (rotation permutes the segments).

Pre-registration: docs/duplex/PREREG_interleaved_timeline_killtest.md.

Output: results/interleaved_timeline/<slug>/<arm>/scores.jsonl.

Usage:
  python scripts/duplex/interleaved_timeline_score.py --corpus mhclip_zh --arm interleaved
"""

import argparse
import glob as globmod
import json
import logging
import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from score_duplex_probe import (  # noqa: E402
    BILIBILI_RULES,
    DUPLEX_PROMPT,
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from data_utils import DATASET_ROOTS, load_annotations, load_clean_split_ids  # noqa: E402
from interleaved_timeline_asr import CORPORA, load_jsonl  # noqa: E402
from interleaved_timeline_build import N_SLOTS, ROTATION  # noqa: E402

# Frozen pixel budget, identical to src/duplex/extract_duplex_readout.py.
MAX_PIXELS = 100352
MIN_PIXELS = 65536
READER = "prag"
NUM_FRAMES = 16

# The one prompt-text change the interleaved arms carry, identical in both of
# them, frozen in the pre-registration.
TRANSCRIPT_POINTER = "(given above, interleaved with the frames)"

BASELINE_JUDGE_DIRS = {
    "mhclip_zh": "results/testruns/mhclip_zh/judge_8b",
    "mhclip_en": "results/testruns/mhclip_en/judge_8b",
    "implihatevid": "results/testruns/implihatevid/judge_8b",
    "hateclipseg": "results/hateclipseg/judge_8b",
}


def resolve_frames(vid, dataset, num_frames=NUM_FRAMES):
    """Frame paths, identical selection rule to the frozen extractor."""
    frame_dir = os.path.join(DATASET_ROOTS[dataset], "frames_16", vid)
    if not os.path.isdir(frame_dir):
        return []
    jpgs = sorted(globmod.glob(os.path.join(frame_dir, "*.jpg")))
    if not jpgs:
        return []
    if len(jpgs) > num_frames:
        indices = np.linspace(0, len(jpgs) - 1, num_frames, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return jpgs


def build_content(arm, n_frames, slots, transcript, title, rules_text):
    """(content list, prompt text) for one video under one arm."""
    if arm == "baseline":
        prompt_text = DUPLEX_PROMPT.format(
            title=title, transcript=transcript, rules=rules_text,
            reader_block=READER_BLOCKS[READER])
        content = [{"type": "image"} for _ in range(n_frames)]
        content.append({"type": "text", "text": prompt_text})
        return content, prompt_text

    shift = 0 if arm == "interleaved" else ROTATION
    prompt_text = DUPLEX_PROMPT.format(
        title=title, transcript=TRANSCRIPT_POINTER, rules=rules_text,
        reader_block=READER_BLOCKS[READER])
    content = []
    for i in range(n_frames):
        content.append({"type": "image"})
        seg = slots[(i + shift) % N_SLOTS].strip() if slots else ""
        if seg:
            content.append({"type": "text", "text": seg})
    content.append({"type": "text", "text": prompt_text})
    return content, prompt_text


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("video_id") and isinstance(r.get("z"), (int, float)) \
                        and np.isfinite(r["z"]):
                    done.add(r["video_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "interleaved", "misaligned"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out-root", default=os.path.join(
        ROOT, "results", "interleaved_timeline"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-baseline", action="store_true",
                    help="baseline arm only: assert every z reproduces the "
                         "frozen judge score exactly")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    dataset, work_rel = CORPORA[args.corpus]
    work = os.path.join(ROOT, work_rel)
    out_dir = os.path.join(args.out_root, args.corpus, args.arm)
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.jsonl")

    platform = "bilibili" if dataset == "MHClip_ZH" else "youtube"
    rules_text = BILIBILI_RULES if platform == "bilibili" else YOUTUBE_RULES

    ann = load_annotations(dataset)
    overrides = json.load(open(os.path.join(work, "c2_overrides.json")))
    segments = {}
    if args.arm != "baseline":
        with open(os.path.join(args.out_root, args.corpus, "segments.json")) as f:
            segments = json.load(f)

    baseline_z = {}
    bl = os.path.join(ROOT, BASELINE_JUDGE_DIRS[args.corpus], "scores.jsonl")
    for line in open(bl):
        if line.strip():
            r = json.loads(line)
            if isinstance(r.get("z"), (int, float)):
                baseline_z[r["video_id"]] = float(r["z"])

    split_ids = load_clean_split_ids(dataset, "test")
    seen, ids = set(), []
    for v in split_ids:
        if v in seen or v not in baseline_z:
            continue
        seen.add(v)
        ids.append(v)

    done = load_done(scores_path)
    remaining = [v for v in ids if v not in done]
    logging.info(f"corpus={args.corpus} dataset={dataset} arm={args.arm} "
                 f"n_ids={len(ids)} done={len(done)} remaining={len(remaining)}")
    if args.limit is not None:
        remaining = remaining[:args.limit]
    if not remaining:
        logging.info("Nothing to do.")
        return

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    label_token_ids = build_binary_token_ids(tokenizer)
    yes_ids = sorted(label_token_ids["Yes"])
    no_ids = sorted(label_token_ids["No"])
    if not yes_ids or not no_ids or (set(yes_ids) & set(no_ids)):
        raise SystemExit(f"bad label token ids: Yes={yes_ids} No={no_ids}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    yes_idx = torch.tensor(yes_ids, device=model.device)
    no_idx = torch.tensor(no_ids, device=model.device)
    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

    t0 = time.time()
    n_done = 0
    n_mismatch = 0
    for i, vid in enumerate(remaining):
        frame_paths = resolve_frames(vid, dataset)
        if len(frame_paths) != NUM_FRAMES:
            raise SystemExit(f"ABORT: {vid} has {len(frame_paths)} frames, "
                             f"expected {NUM_FRAMES}")
        a = ann[vid]
        transcript = overrides.get(vid)
        if transcript is None:
            transcript = a.get("transcript", "") or ""
        if args.arm != "baseline" and vid not in segments:
            raise SystemExit(f"ABORT: {vid} has no segmentation record")
        slots = segments.get(vid, {}).get("slots") if segments else None
        content, _ = build_content(args.arm, len(frame_paths), slots,
                                   transcript, a.get("title", "") or "",
                                   rules_text)
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           size=size_kwarg)
        for im in images:
            im.close()
        del images

        grid = inputs["image_grid_thw"]
        ip = processor.image_processor
        per_frame_pixels = [int(h * w) * ip.patch_size * ip.patch_size
                            for _, h, w in grid.tolist()]
        if max(per_frame_pixels) > MAX_PIXELS:
            raise SystemExit(f"{vid}: pixel cap not honored")

        inputs = inputs.to(model.device)
        n_tokens_total = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True,
                            use_cache=False, logits_to_keep=1)
            last_logits = outputs.logits[0, -1, :].float()
            z = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                      - torch.logsumexp(last_logits[no_idx], dim=0))
        del outputs, last_logits, inputs
        if not np.isfinite(z):
            raise SystemExit(f"ABORT: {vid} produced a non-finite z")

        rec = {"video_id": vid, "z": z, "arm": args.arm,
               "n_tokens_total": n_tokens_total,
               "n_text_items": sum(1 for c in content if c["type"] == "text"),
               "route": (segments.get(vid, {}).get("route") if segments
                         else "baseline")}
        if args.arm == "baseline":
            rec["z_frozen"] = baseline_z[vid]
            rec["reproduces_frozen"] = (z == baseline_z[vid])
            if not rec["reproduces_frozen"]:
                n_mismatch += 1
                logging.error(f"  {vid}: z={z:.6f} != frozen "
                              f"{baseline_z[vid]:.6f}")
        with open(scores_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_done += 1
        if n_done % 50 == 0 or i == 0:
            el = time.time() - t0
            logging.info(f"  [{n_done}/{len(remaining)}] z={z:+.3f} "
                         f"tokens={n_tokens_total} "
                         f"{el / max(n_done, 1):.2f}s/video")

    el = time.time() - t0
    logging.info(f"Done [{args.corpus}/{args.arm}]: {n_done} scored, "
                 f"{el:.1f}s ({el / max(n_done, 1):.2f}s/video)")
    if args.arm == "baseline":
        logging.info(f"baseline replication: {n_done - n_mismatch}/{n_done} "
                     f"exact")
        if args.check_baseline and n_mismatch:
            raise SystemExit(f"ABORT: {n_mismatch} baseline z values did not "
                             "reproduce the frozen judge")


if __name__ == "__main__":
    main()
