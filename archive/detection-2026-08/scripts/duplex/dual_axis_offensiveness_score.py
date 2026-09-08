"""Offensiveness-axis judge call for the dual-axis kill test.

Pre-registration: docs/duplex/PREREG_dual_axis_killtest.md. The prompt below is
frozen in that file's appendix and was committed before the first judge call.

The call is the frozen joint judge with one substitution: the construct
definition. Everything else is taken from the same frozen modules the joint
judge uses -- the system message, the prompt skeleton, the `prag` judgment-mode
block, the 16-frame media path, the fresh-transcript override logic, the pixel
budget, the Yes/No token-id sets, and the raw readout
z = logsumexp(logits[Yes]) - logsumexp(logits[No]) at the final prompt position.
Hidden states are not stored; this test only needs z.

Output: results/dual_axis/z_off_scores.jsonl
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))

from score_duplex_probe import (  # noqa: E402
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    build_binary_token_ids,
)
from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    READER,
    resolve_frames,
    resolve_transcript,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

# ---------------------------------------------------------------------------
# FROZEN OFFENSIVENESS CONSTRUCT (committed before the first judge call).
#
# The skeleton is score_duplex_probe.DUPLEX_PROMPT with the policy sentence and
# the rule list replaced. The judgment-mode block, the question sentence, and
# the answer instruction are unchanged.
# ---------------------------------------------------------------------------

OFFENSIVENESS_RULES = """1. Insulting or demeaning language aimed at any person, group, or thing
2. Degrading, humiliating, or shaming portrayal of anyone
3. Vulgar, obscene, or crude language, gestures, or imagery
4. Aggressive, hostile, threatening, or abusive speech or behaviour
5. Mockery, ridicule, or contemptuous framing of anyone
6. Profanity, name-calling, or personal attacks
7. Gratuitously shocking, disgusting, or distasteful material"""

OFFENSIVENESS_PROMPT = """You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's offensive-content policy, the following types of content are prohibited:
{rules}

These rules are about offensiveness alone and apply no matter who or what is targeted. Content violates them even when it targets no protected group (such as race, religion, ethnicity, national origin, gender, sexual orientation, or disability), and even when it targets one individual, an institution, or nobody at all.

{reader_block}

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No"."""


def build_messages(ann, frame_paths, transcript):
    prompt_text = OFFENSIVENESS_PROMPT.format(
        title=ann.get("title", "") or "",
        transcript=transcript,
        rules=OFFENSIVENESS_RULES,
        reader_block=READER_BLOCKS[READER],
    )
    content = [{"type": "image"} for _ in frame_paths]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


def load_done_ids(scores_path):
    done = set()
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                z = r.get("z")
                if r.get("video_id") and isinstance(z, (int, float)) and np.isfinite(z):
                    done.add(r["video_id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MHClip_EN")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--transcript-limit", type=int, default=0)
    parser.add_argument("--transcript-override-json",
                        default=os.path.join(PROJECT_ROOT, "results", "testruns",
                                             "mhclip_en", "c2_overrides.json"))
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results",
                                                          "dual_axis"))
    parser.add_argument("--out-name", default="z_off_scores.jsonl")
    parser.add_argument("--status-path", default=None)
    parser.add_argument("--expect", type=int, default=161)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    os.makedirs(args.out_dir, exist_ok=True)
    scores_path = os.path.join(args.out_dir, args.out_name)
    status_path = args.status_path or os.path.join(args.out_dir, "STATUS")

    def status(msg):
        with open(status_path, "w") as f:
            f.write(msg + "\n")

    annotations = load_annotations(args.dataset)
    with open(args.transcript_override_json) as f:
        overrides = json.load(f)
    logging.info(f"Transcript overrides: {len(overrides)} ids from "
                 f"{args.transcript_override_json}")

    split_ids = load_clean_split_ids(args.dataset, args.split)
    logging.info(f"Split ids: {len(split_ids)}")
    if len(split_ids) != args.expect:
        raise SystemExit(f"expected {args.expect} split ids, got {len(split_ids)}")

    # ---- pre-flight: every frame of every video must decode ----------------
    status("preflight")
    t_pre = time.time()
    from PIL import Image
    frame_index = {}
    bad = []
    for vid in split_ids:
        paths = resolve_frames(vid, args.dataset, args.num_frames)
        if len(paths) != args.num_frames:
            bad.append((vid, f"{len(paths)} frames"))
            continue
        for p in paths:
            try:
                im = Image.open(p)
                im.load()
                im.close()
            except Exception as exc:
                bad.append((vid, f"{os.path.basename(p)}: {exc}"))
                break
        else:
            frame_index[vid] = paths
    if bad:
        for vid, why in bad[:20]:
            logging.error(f"  preflight failure {vid}: {why}")
        raise SystemExit(f"ABORT: {len(bad)} videos failed frame pre-flight")
    logging.info(f"Pre-flight OK: {len(frame_index)} videos x {args.num_frames} "
                 f"frames decoded in {time.time() - t_pre:.1f}s")

    done = load_done_ids(scores_path)
    remaining = [v for v in split_ids if v not in done]
    logging.info(f"Resume: {len(done)} done, {len(remaining)} remaining")

    if remaining:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer
        label_token_ids = build_binary_token_ids(tokenizer)
        yes_ids = sorted(label_token_ids["Yes"])
        no_ids = sorted(label_token_ids["No"])
        if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
            raise SystemExit(f"bad label token ids: Yes={yes_ids} No={no_ids}")
        logging.info(f"Yes ids {yes_ids} / No ids {no_ids}")

        ip = processor.image_processor
        patch_size = ip.patch_size

        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        yes_idx = torch.tensor(yes_ids, device=model.device)
        no_idx = torch.tensor(no_ids, device=model.device)

        t0 = time.time()
        n_done = 0
        for i, vid in enumerate(remaining):
            status(f"scoring {i + 1}/{len(remaining)} {vid}")
            ann = annotations.get(vid)
            if ann is None:
                raise SystemExit(f"{vid}: not in annotations")
            frame_paths = frame_index[vid]
            images = [Image.open(p).convert("RGB") for p in frame_paths]
            transcript = resolve_transcript(ann, vid, overrides, args.transcript_limit)
            messages = build_messages(ann, frame_paths, transcript)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt",
                               size=size_kwarg)
            for im in images:
                im.close()
            del images

            grid = inputs["image_grid_thw"]
            per_frame_pixels = [int(h * w) * patch_size * patch_size
                                for _, h, w in grid.tolist()]
            if max(per_frame_pixels) > MAX_PIXELS:
                raise SystemExit(f"{vid}: pixel cap not honored")

            inputs = inputs.to(model.device)
            n_tokens_total = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False, logits_to_keep=1)
                last_logits = outputs.logits[0, -1, :].float()
                z = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                          - torch.logsumexp(last_logits[no_idx], dim=0))
            del outputs, last_logits, inputs
            if not np.isfinite(z):
                raise SystemExit(f"{vid}: non-finite z")

            rec = {
                "video_id": vid,
                "z": z,
                "p_yes_renorm": float(1.0 / (1.0 + np.exp(-z))),
                "n_tokens_total": n_tokens_total,
                "n_transcript_chars": len(transcript),
                "transcript_source": "override" if vid in overrides else "dataset",
            }
            with open(scores_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            n_done += 1
            if n_done % 25 == 0 or i == 0:
                el = time.time() - t0
                logging.info(f"  [{n_done}/{len(remaining)}] z={z:+.3f} "
                             f"{el / n_done:.2f}s/video")
        logging.info(f"Scored {n_done} in {time.time() - t0:.1f}s")

    n_scored = len(load_done_ids(scores_path))
    logging.info(f"Total scored: {n_scored}")
    if n_scored != args.expect:
        status(f"FAILED: n_scored={n_scored} != {args.expect}")
        raise SystemExit(f"ABORT: n_scored={n_scored} != {args.expect}")
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(f"n_scored={n_scored}\n")
    logging.info("DONE")


if __name__ == "__main__":
    main()
