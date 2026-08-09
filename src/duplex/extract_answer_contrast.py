"""Answer-contrast extraction: two forced continuations of the frozen judge.

Pre-registration: docs/duplex/PREREG_answer_contrast_pilot.md.

Minimal fork of src/duplex/extract_duplex_readout.py. The prompt, the reader
block, the rules, the system message, the transcript cap, the frame count and
the per-frame pixel budget are all imported from that module and from
src/duplex/score_duplex_probe.py; nothing about the model's input is redefined
here. The only change is what happens after the prefill.

Per video:
  1. one prefill over the frozen prompt, keeping the key-value cache; the
     final-position logits give the frozen z;
  2. one single-token forward with the first token of "Yes" appended;
  3. the cache is cropped back to the prefill length;
  4. one single-token forward with the first token of "No" appended.

Both appended tokens are corpus-constant. The hidden state at the appended
position is recorded for every layer in both arms.

Output: <out_dir>/scores.jsonl and <out_dir>/arms/<video_id>.npy of shape
[2, n_layers + 1, hidden_size] in float16, row 0 the Yes arm and row 1 the No
arm.

Usage:
  python src/duplex/extract_answer_contrast.py --dataset HateMM --split test
  python src/duplex/extract_answer_contrast.py --dataset HateMM --split test \
      --limit 3 --verify-full 3
"""

import argparse
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

from score_duplex_probe import (  # noqa: E402
    BILIBILI_RULES,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    build_messages,
    resolve_frames,
    resolve_transcript,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402


def load_done_ids(scores_path, arms_dir, expected_shape):
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
                vid, z = r.get("video_id"), r.get("z")
                if vid and isinstance(z, (int, float)) and np.isfinite(z):
                    finite_z.add(vid)
    done = set()
    for vid in finite_z:
        p = os.path.join(arms_dir, f"{vid}.npy")
        if not os.path.isfile(p):
            continue
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:
            continue
        if tuple(arr.shape) == tuple(expected_shape):
            done.add(vid)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HateMM",
                        choices=["ImpliHateVid", "MHClip_EN", "MHClip_ZH",
                                 "HateMM", "HateClipSeg"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0",
                        help="Smoke tests may set cpu; every reported result "
                             "comes from cuda:0.")
    parser.add_argument("--verify-full", type=int, default=0,
                        help="For the first N videos, recompute both arms with "
                             "a full uncached forward and report the maximum "
                             "absolute deviation.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "results", "answer_contrast", args.dataset)
    arms_dir = os.path.join(out_dir, "arms")
    os.makedirs(arms_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.jsonl")

    platform = "bilibili" if args.dataset == "MHClip_ZH" else "youtube"
    rules_text = BILIBILI_RULES if platform == "bilibili" else YOUTUBE_RULES

    annotations = load_annotations(args.dataset)
    split_ids = load_clean_split_ids(args.dataset, args.split)
    seen, ordered = set(), []
    for v in split_ids:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    split_ids = ordered

    size_kwarg = {"shortest_edge": args.min_pixels,
                  "longest_edge": args.max_pixels}
    logging.info(f"Config: dataset={args.dataset} split={args.split} "
                 f"n_videos={len(split_ids)} frames={args.num_frames} "
                 f"transcript_limit={args.transcript_limit} size={size_kwarg}")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    config = AutoConfig.from_pretrained(args.model)
    n_layers = config.text_config.num_hidden_layers
    hidden_size = config.text_config.hidden_size
    expected_shape = (2, n_layers + 1, hidden_size)
    logging.info(f"Arm array shape {expected_shape}")

    done = load_done_ids(scores_path, arms_dir, expected_shape)
    remaining = [v for v in split_ids if v not in done]
    logging.info(f"Resume: {len(done)} complete, {len(remaining)} remaining")
    if args.limit is not None:
        remaining = remaining[:args.limit]
    if not remaining:
        logging.info("Nothing to do.")
        return

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    label_token_ids = build_binary_token_ids(tokenizer)
    yes_ids = sorted(label_token_ids["Yes"])
    no_ids = sorted(label_token_ids["No"])
    if set(yes_ids) & set(no_ids):
        raise SystemExit("overlapping label token id sets")

    # Frozen appended tokens: the first token of the bare strings, asserted to
    # belong to the frozen id sets. Corpus-constant by construction.
    tok_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    tok_no = tokenizer.encode("No", add_special_tokens=False)[0]
    if tok_yes not in yes_ids or tok_no not in no_ids:
        raise SystemExit(f"appended tokens {tok_yes}/{tok_no} are not members "
                         f"of the frozen sets {yes_ids}/{no_ids}")
    logging.info(f"Appended tokens: Yes={tok_yes} "
                 f"({tokenizer.decode([tok_yes])!r}) No={tok_no} "
                 f"({tokenizer.decode([tok_no])!r})")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    yes_idx = torch.tensor(yes_ids, device=model.device)
    no_idx = torch.tensor(no_ids, device=model.device)

    from PIL import Image

    t0, n_done, n_skipped = time.time(), 0, 0
    verify_reports = []

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
        transcript = resolve_transcript(ann, vid, None, args.transcript_limit)
        messages = build_messages(ann, frame_paths, rules_text, transcript)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           size=size_kwarg)
        for im in images:
            im.close()
        del images

        inputs = inputs.to(model.device)
        prefill_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            pre = model(**inputs, use_cache=True, logits_to_keep=1)
            last_logits = pre.logits[0, -1, :].float()
            z = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                      - torch.logsumexp(last_logits[no_idx], dim=0))
            cache = pre.past_key_values

            arms = []
            for tok in (tok_yes, tok_no):
                cache.crop(prefill_len)
                step = model(
                    input_ids=torch.tensor([[tok]], device=model.device),
                    attention_mask=torch.ones(1, prefill_len + 1,
                                              dtype=inputs["attention_mask"].dtype,
                                              device=model.device),
                    cache_position=torch.tensor([prefill_len],
                                                device=model.device),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                arms.append(torch.stack(
                    [h[0, -1, :] for h in step.hidden_states], dim=0
                ).to(torch.float16).cpu().numpy())
                del step

            arr = np.stack(arms, axis=0)

            if len(verify_reports) < args.verify_full:
                ref = []
                for tok in (tok_yes, tok_no):
                    full_ids = torch.cat(
                        [inputs["input_ids"],
                         torch.tensor([[tok]], device=model.device)], dim=1)
                    full_mask = torch.cat(
                        [inputs["attention_mask"],
                         torch.ones(1, 1, dtype=inputs["attention_mask"].dtype,
                                    device=model.device)], dim=1)
                    kw = {k: v for k, v in inputs.items()
                          if k not in ("input_ids", "attention_mask")}
                    out = model(input_ids=full_ids, attention_mask=full_mask,
                                use_cache=False, output_hidden_states=True,
                                logits_to_keep=1, **kw)
                    ref.append(torch.stack(
                        [h[0, -1, :] for h in out.hidden_states], dim=0
                    ).to(torch.float16).cpu().numpy())
                    del out
                ref = np.stack(ref, axis=0)
                a32, r32 = arr.astype(np.float32), ref.astype(np.float32)
                d = np.abs(a32 - r32)
                rel = float(np.linalg.norm(d) / max(np.linalg.norm(r32), 1e-9))
                # The decision-relevant quantity is the layer-27 difference of
                # the two arms, so it is checked directly.
                dc = a32[0, 27] - a32[1, 27]
                dr = r32[0, 27] - r32[1, 27]
                cos = float(dc @ dr / (np.linalg.norm(dc)
                                       * np.linalg.norm(dr) + 1e-9))
                verify_reports.append(
                    {"max_abs_diff": float(d.max()),
                     "mean_abs_diff": float(d.mean()),
                     "mean_abs_value": float(np.abs(r32).mean()),
                     "relative_frobenius_error": rel,
                     "cosine_delta_layer27": cos,
                     "delta_norm_cached": float(np.linalg.norm(dc)),
                     "delta_norm_full": float(np.linalg.norm(dr))})
                logging.info(f"  verify {vid}: rel_fro={rel:.4g} "
                             f"cos(Δh27)={cos:.6f} max|Δ|={d.max():.4g}")

            del pre, cache, last_logits, inputs

        if not np.isfinite(z):
            logging.error(f"  {vid}: non-finite z, skipping")
            n_skipped += 1
            continue
        if arr.shape != expected_shape:
            raise SystemExit(f"{vid}: arm shape {arr.shape} != {expected_shape}")
        if not np.isfinite(arr.astype(np.float32)).all():
            raise SystemExit(f"{vid}: non-finite hidden state")

        np.save(os.path.join(arms_dir, f"{vid}.npy"), arr)
        delta = arr[0].astype(np.float32) - arr[1].astype(np.float32)
        rec = {"video_id": vid, "z": z,
               "n_tokens_total": prefill_len,
               "delta_norm_layer27": float(np.linalg.norm(delta[27])),
               "yes_norm_layer27": float(np.linalg.norm(arr[0, 27].astype(np.float32))),
               "no_norm_layer27": float(np.linalg.norm(arr[1, 27].astype(np.float32)))}
        with open(scores_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        del arr

        n_done += 1
        if n_done % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            logging.info(
                f"  [{n_done}/{len(remaining)}] {vid} z={z:+.3f} "
                f"{time.time() - t_vid:.2f}s/video, "
                f"{n_done / max(elapsed, 1e-9):.2f} vid/s, "
                f"peak_vram="
                f"{torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.2f} GiB")

    elapsed = time.time() - t0
    if verify_reports:
        logging.info(f"Cache-vs-full verification: {json.dumps(verify_reports)}")
    logging.info(f"Done: {n_done} extracted, {n_skipped} skipped, "
                 f"{elapsed:.1f}s total "
                 f"({elapsed / max(n_done, 1):.2f}s/video)")


if __name__ == "__main__":
    main()

