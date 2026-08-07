"""
Duplex readout probe: hidden-state extraction.

One HF-transformers forward pass per video under the frozen `prag` reader of
the kill-test, reading two things at the final prompt position (the position
whose next-token distribution yields Yes or No):

  1. the raw, unclipped logit difference
     z = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids]),
     so that sigmoid(z) reproduces the kill-test renormalized P(Yes);
  2. the hidden state at that position for every layer (embedding output plus
     each transformer layer), stored fp16.

Pre-registration: docs/duplex/PREREG_duplex_readout.md. The prompt is imported
from `src/duplex/score_duplex_probe.py`, which is frozen: the reader block, the
prompt skeleton, the rules, the system message, and the Yes/No token-id sets
all come from that module and are never redefined here.

The per-frame pixel budget the kill-test failed to apply under vLLM is applied
here through the parameters the Qwen3-VL image processor actually reads,
`size["shortest_edge"]` and `size["longest_edge"]`. A 1920x1080 frame then
resizes to 416x224 (93,184 pixels, 91 vision tokens) instead of costing 2,040
tokens at native resolution, so the 128 HD videos the kill-test lost to context
overflow re-enter the sample.

Output: results/duplex_readout/<dataset>/scores.jsonl (one record per video)
and results/duplex_readout/<dataset>/hidden/<video_id>.npy of shape
[n_layers + 1, hidden_size], fp16.

Usage:
  python src/duplex/extract_duplex_readout.py --dataset ImpliHateVid --split train
  python src/duplex/extract_duplex_readout.py --video-ids NH_1,EX_1  # smoke test
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
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)

# Frozen prompt material. Imported, never copied: score_duplex_probe.py is the
# kill-test's frozen judge and must not be edited or duplicated.
from score_duplex_probe import (  # noqa: E402
    BILIBILI_RULES,
    DUPLEX_PROMPT,
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from data_utils import DATASET_ROOTS, load_annotations, load_clean_split_ids  # noqa: E402

# Per-frame pixel budget. `longest_edge` is the Qwen3-VL image processor's
# max_pixels; `shortest_edge` is its min_pixels and is left at the value the
# shipped preprocessor_config.json carries, so only the upper bound changes.
MAX_PIXELS = 100352
MIN_PIXELS = 65536

READER = "prag"


def resolve_frames(vid, dataset, num_frames):
    """Return up to `num_frames` frame paths from frames_16/<vid>, evenly spaced.

    Mirrors the frames branch of score_holistic_2b.build_media_content: sorted
    *.jpg, then np.linspace sub-sampling. frames_16 is addressed directly rather
    than through the frames -> frames_16 symlink get_media_path relies on.
    """
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


def resolve_transcript(ann, vid, overrides, transcript_limit):
    """The transcript text the judge sees.

    `overrides` maps video_id to a replacement transcript and is None for the
    frozen default; ids absent from it keep the dataset transcript. A
    `transcript_limit` of 0 or None removes the character cap. With the
    defaults (no overrides, limit 300) this returns the frozen judge's input.
    """
    if overrides is not None and vid in overrides:
        text = overrides[vid] or ""
    else:
        text = ann.get("transcript", "") or ""
    return text[:transcript_limit] if transcript_limit else text


def build_messages(ann, frame_paths, rules_text, transcript):
    """System message plus a user turn of N images followed by the prompt text."""
    prompt_text = DUPLEX_PROMPT.format(
        title=ann.get("title", "") or "",
        transcript=transcript,
        rules=rules_text,
        reader_block=READER_BLOCKS[READER],
    )
    content = [{"type": "image"} for _ in frame_paths]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


def load_done_ids(scores_path, hidden_dir, expected_shape):
    """Video ids that are complete: a finite z in the jsonl AND a loadable .npy
    of the expected shape.

    The kill-test's resume treated any row carrying a video_id as done, so
    null-score rows written during an engine death were skipped forever. Both
    halves of the output are checked here instead.
    """
    finite_z = set()
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = r.get("video_id")
                z = r.get("z")
                if vid and isinstance(z, (int, float)) and np.isfinite(z):
                    finite_z.add(vid)

    done = set()
    for vid in finite_z:
        npy = os.path.join(hidden_dir, f"{vid}.npy")
        if not os.path.isfile(npy):
            continue
        try:
            arr = np.load(npy, mmap_mode="r")
        except Exception:
            continue
        if tuple(arr.shape) == tuple(expected_shape):
            done.add(vid)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ImpliHateVid",
                        choices=["ImpliHateVid", "MHClip_EN", "MHClip_ZH",
                                 "HateMM", "HateClipSeg"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--transcript-limit", type=int, default=300,
                        help="Character cap on the transcript; 0 = uncapped")
    parser.add_argument("--transcript-override-json", default=None,
                        help="JSON mapping video_id -> transcript text that "
                             "replaces the dataset transcript for those ids")
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to results/duplex_readout/<dataset>")
    parser.add_argument("--video-ids", default=None,
                        help="Comma-separated subset of video ids (smoke tests)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many remaining videos")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "results", "duplex_readout", args.dataset)
    hidden_dir = os.path.join(out_dir, "hidden")
    os.makedirs(hidden_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.jsonl")

    platform = "bilibili" if args.dataset == "MHClip_ZH" else "youtube"
    rules_text = BILIBILI_RULES if platform == "bilibili" else YOUTUBE_RULES

    annotations = load_annotations(args.dataset)
    overrides = None
    if args.transcript_override_json:
        with open(args.transcript_override_json) as f:
            overrides = json.load(f)
        logging.info(f"Transcript overrides: {len(overrides)} ids from "
                     f"{args.transcript_override_json}")
    if args.video_ids:
        split_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    else:
        split_ids = load_clean_split_ids(args.dataset, args.split)

    size_kwarg = {"shortest_edge": args.min_pixels, "longest_edge": args.max_pixels}
    logging.info(f"Config: dataset={args.dataset} split={args.split} model={args.model} "
                 f"reader={READER} n_videos={len(split_ids)} frames={args.num_frames} "
                 f"transcript_limit={args.transcript_limit} size={size_kwarg}")
    logging.info(f"Output: {scores_path} + {hidden_dir}/<video_id>.npy")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    config = AutoConfig.from_pretrained(args.model)
    n_layers = config.text_config.num_hidden_layers
    hidden_size = config.text_config.hidden_size
    expected_shape = (n_layers + 1, hidden_size)
    logging.info(f"Model config: n_layers={n_layers} hidden_size={hidden_size} "
                 f"hidden-state array shape {expected_shape}")

    done = load_done_ids(scores_path, hidden_dir, expected_shape)
    remaining = [v for v in split_ids if v not in done]
    logging.info(f"Resume: {len(done)} complete, {len(remaining)} remaining")
    if args.limit is not None:
        remaining = remaining[:args.limit]
        logging.info(f"--limit: processing {len(remaining)}")
    if not remaining:
        logging.info("Nothing to do.")
        return

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    label_token_ids = build_binary_token_ids(tokenizer)
    yes_ids = sorted(label_token_ids["Yes"])
    no_ids = sorted(label_token_ids["No"])
    if not yes_ids or not no_ids:
        raise SystemExit(f"empty label token id set: Yes={yes_ids} No={no_ids}")
    if set(yes_ids) & set(no_ids):
        raise SystemExit(f"overlapping label token id sets: {set(yes_ids) & set(no_ids)}")
    logging.info(f"Yes ids {yes_ids} -> {[tokenizer.decode([t]) for t in yes_ids]}")
    logging.info(f"No  ids {no_ids} -> {[tokenizer.decode([t]) for t in no_ids]}")

    ip = processor.image_processor
    patch_size = ip.patch_size
    merge_size = ip.merge_size
    logging.info(f"Image processor: {type(ip).__name__} patch_size={patch_size} "
                 f"merge_size={merge_size} config_size={dict(ip.size)}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    yes_idx = torch.tensor(yes_ids, device=model.device)
    no_idx = torch.tensor(no_ids, device=model.device)

    from PIL import Image

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    cap_checked = False

    for i, vid in enumerate(remaining):
        t_vid = time.time()
        ann = annotations.get(vid)
        if ann is None:
            logging.warning(f"  {vid}: not in annotations, skipping")
            n_skipped += 1
            continue
        frame_paths = resolve_frames(vid, args.dataset, args.num_frames)
        if not frame_paths:
            logging.warning(f"  {vid}: no frames, skipping")
            n_skipped += 1
            continue

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        transcript = resolve_transcript(ann, vid, overrides, args.transcript_limit)
        messages = build_messages(ann, frame_paths, rules_text, transcript)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           size=size_kwarg)
        for im in images:
            im.close()
        del images

        grid = inputs["image_grid_thw"]
        per_frame = [int(t * h * w) // (merge_size ** 2) for t, h, w in grid.tolist()]
        per_frame_pixels = [int(h * w) * patch_size * patch_size
                            for _, h, w in grid.tolist()]
        if max(per_frame_pixels) > args.max_pixels:
            raise SystemExit(
                f"{vid}: pixel cap not honored: {max(per_frame_pixels)} > {args.max_pixels}")
        if not cap_checked:
            logging.info(f"  cap check on {vid}: grid={grid.tolist()[0]} "
                         f"pixels/frame={per_frame_pixels[0]} tokens/frame={per_frame[0]}")
            cap_checked = True
        n_img_tokens = per_frame[0] if len(set(per_frame)) == 1 else per_frame

        inputs = inputs.to(model.device)
        n_tokens_total = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True,
                            use_cache=False, logits_to_keep=1)
            # Final prompt position: the one whose next-token distribution is
            # the Yes/No answer.
            last_logits = outputs.logits[0, -1, :].float()
            z = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                      - torch.logsumexp(last_logits[no_idx], dim=0))
            hidden = torch.stack(
                [h[0, -1, :] for h in outputs.hidden_states], dim=0
            ).to(torch.float16).cpu().numpy()

        del outputs, last_logits, inputs
        if not np.isfinite(z):
            logging.error(f"  {vid}: non-finite z, skipping")
            n_skipped += 1
            del hidden
            continue
        if hidden.shape != expected_shape:
            raise SystemExit(f"{vid}: hidden shape {hidden.shape} != {expected_shape}")

        np.save(os.path.join(hidden_dir, f"{vid}.npy"), hidden)
        del hidden

        rec = {
            "video_id": vid,
            "z": z,
            "p_yes_renorm": float(1.0 / (1.0 + np.exp(-z))),
            "n_tokens_total": n_tokens_total,
            "n_image_tokens_per_frame": n_img_tokens,
        }
        # Provenance of the transcript, emitted only when an intervention is
        # active so that a default run's records stay byte-identical.
        if overrides is not None or args.transcript_limit != 300:
            rec["n_transcript_chars"] = len(transcript)
            rec["transcript_source"] = (
                "override" if overrides is not None and vid in overrides else "dataset")
        with open(scores_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_done += 1
        dt = time.time() - t_vid
        if n_done % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            logging.info(
                f"  [{n_done}/{len(remaining)}] {vid} z={z:+.3f} "
                f"tokens={n_tokens_total} img_tok/frame={per_frame[0]} "
                f"{dt:.2f}s/video, {rate:.2f} vid/s, "
                f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    elapsed = time.time() - t0
    logging.info(f"Done: {n_done} extracted, {n_skipped} skipped, "
                 f"{elapsed:.1f}s total ({elapsed / max(n_done, 1):.2f}s/video), "
                 f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
