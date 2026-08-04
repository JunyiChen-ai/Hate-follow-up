"""
Duplex reading kill-test probe.

Scores a dataset split under five reader roles (lit_v1, lit_v2, prag,
prag_para, effort) that share the frozen BINARY_PROMPT skeleton from
src/our_method/score_holistic_2b.py, differing only in one inserted
"judgment mode" block. Per reader, extracts the renormalized
P(Yes)/(P(Yes)+P(No)) next-token score exactly as the frozen stage-1
scorer does.

Pre-registration: docs/duplex/PREREG_duplex_killtest.md. Prompt texts are
frozen there; do not edit the READER_BLOCKS below without a new
pre-registration entry.

Output: results/duplex_probe/<dataset>/<split>_<reader>_<slug>.jsonl
with records {"video_id", "score", "reader", "model"}.
"""

import argparse
import json
import logging
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))

from data_utils import get_media_path, load_annotations, load_clean_split_ids  # noqa: E402
from score_holistic_2b import (  # noqa: E402
    YOUTUBE_RULES,
    BILIBILI_RULES,
    build_binary_token_ids,
    build_media_content,
    extract_binary_score,
)

# vLLM will only read media below this path. Must cover the dataset root.
MEDIA_ROOT = os.environ.get("HVD_DATA_ROOT", "/data/jehc223")

# Frozen BINARY_PROMPT skeleton with one extra {reader_block} slot between
# the rules and the question. Byte-identical to score_holistic_2b.BINARY_PROMPT
# otherwise.
DUPLEX_PROMPT = """You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

{reader_block}

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No"."""

READER_BLOCKS = {
    "lit_v1": (
        "Judgment mode: surface reading. Judge only what is literally said, "
        "written, or shown in the video, giving every word and image its "
        "plain, face-value meaning. A rule is violated only if the violation "
        "is explicit on the surface."
    ),
    "lit_v2": (
        "Judgment mode: surface reading. Judge only what is literally said, "
        "written, or shown in the video, giving every word and image its "
        "plain, face-value meaning. Do not infer hidden meanings: coded "
        "references, wordplay, irony, or implications are outside your "
        "scope. A rule is violated only if the violation is explicit on the "
        "surface."
    ),
    "prag": (
        "Judgment mode: full reading. Judge what the video communicates as "
        "a whole: both what is explicitly said or shown and what is conveyed "
        "beyond the surface. Consider coded or euphemistic references, "
        "wordplay and homophones, ironic or mocking framing, how the title "
        "reframes the visuals, the joint effect of images, audio, and text, "
        "and how the referenced group or an informed viewer would receive "
        "the message. A rule is violated if either the explicit content or "
        "the conveyed meaning violates it."
    ),
    "prag_para": (
        "Judgment mode: complete interpretation. Assess the overall message "
        "the video sends: what it states or displays openly, and also what "
        "it suggests indirectly. Take into account euphemisms or coded "
        "expressions, puns and sound-alike words, sarcastic or derisive "
        "presentation, the way the title recasts the footage, how images, "
        "audio, and text work together, and the way an insider or the group "
        "being referenced would understand it. A violation occurs when "
        "either the open content or the suggested meaning breaks a rule."
    ),
    "effort": (
        "Judgment mode: thorough reading. Examine the video with maximum "
        "care: inspect every frame closely, read the title and the entire "
        "transcript word by word, check all on-screen text against each "
        "rule, and weigh every piece of evidence systematically before "
        "deciding. Review the complete material and let nothing escape your "
        "attention. A rule is violated if any part of the carefully reviewed "
        "material violates it."
    ),
}

SYSTEM_MESSAGE = "You are a content moderation analyst. Answer based strictly on observable evidence."


def model_slug(model):
    return model.split("/")[-1].lower()


def out_path_for(dataset, split, reader, slug):
    out_dir = os.path.join(PROJECT_ROOT, "results", "duplex_probe", dataset)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{split}_{reader}_{slug}.jsonl")


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("video_id"):
                            done.add(r["video_id"])
                    except json.JSONDecodeError:
                        pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["ImpliHateVid", "MHClip_EN", "MHClip_ZH", "HateMM"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--readers", default="lit_v1,lit_v2,prag,prag_para,effort",
                        help="Comma-separated subset of reader roles to score")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--no-video", action="store_true",
                        help="Feed frames_16 images instead of the mp4")
    parser.add_argument("--gpu-mem", type=float, default=0.92)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    for r in readers:
        if r not in READER_BLOCKS:
            raise SystemExit(f"unknown reader {r!r}; valid: {sorted(READER_BLOCKS)}")

    slug = model_slug(args.model)
    platform = "bilibili" if args.dataset == "MHClip_ZH" else "youtube"
    rules_text = BILIBILI_RULES if platform == "bilibili" else YOUTUBE_RULES

    annotations = load_annotations(args.dataset)
    split_ids = load_clean_split_ids(args.dataset, args.split)
    logging.info(f"Config: dataset={args.dataset} split={args.split} model={args.model} "
                 f"readers={readers} n_videos={len(split_ids)}")

    # Figure out what remains per reader before paying for engine load.
    plans = []
    for reader in readers:
        path = out_path_for(args.dataset, args.split, reader, slug)
        done = load_done_ids(path)
        remaining = [v for v in split_ids if v not in done]
        logging.info(f"  reader={reader}: {len(done)} done, {len(remaining)} remaining")
        if remaining:
            plans.append((reader, path, remaining))
    if not plans:
        logging.info("All readers complete. Nothing to do.")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=32768,
        limit_mm_per_prompt=({"image": args.num_frames} if args.no_video
                             else {"video": 1, "image": args.num_frames}),
        allowed_local_media_path=MEDIA_ROOT,
        mm_processor_kwargs={"max_pixels": 100352},
    )
    tokenizer = llm.get_tokenizer()
    label_token_ids = build_binary_token_ids(tokenizer)
    all_constrained_ids = set()
    for tids in label_token_ids.values():
        all_constrained_ids.update(tids)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=list(all_constrained_ids),
    )

    for reader, out_path, remaining in plans:
        block = READER_BLOCKS[reader]
        t0 = time.time()
        n_processed = 0
        n_skipped = 0
        logging.info(f"=== reader={reader} n={len(remaining)} -> {out_path} ===")

        for batch_start in range(0, len(remaining), args.batch_size):
            batch_ids = remaining[batch_start:batch_start + args.batch_size]
            batch_messages = []
            batch_vid_ids = []

            for vid_id in batch_ids:
                ann = annotations.get(vid_id)
                if ann is None:
                    logging.warning(f"  {vid_id}: not in annotations, skipping")
                    n_skipped += 1
                    continue
                media = get_media_path(vid_id, args.dataset)
                if media is None:
                    logging.warning(f"  {vid_id}: no media, skipping")
                    n_skipped += 1
                    continue
                media_path, media_type = media
                prompt_text = DUPLEX_PROMPT.format(
                    title=ann.get("title", "") or "",
                    transcript=(ann.get("transcript", "") or "")[:args.transcript_limit],
                    rules=rules_text,
                    reader_block=block,
                )
                media_content = build_media_content(
                    media_path, media_type,
                    no_video=args.no_video, dataset=args.dataset,
                    num_frames=args.num_frames,
                )
                if not media_content:
                    logging.warning(f"  {vid_id}: empty media content, skipping")
                    n_skipped += 1
                    continue
                content = media_content + [{"type": "text", "text": prompt_text}]
                batch_messages.append([
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": content},
                ])
                batch_vid_ids.append(vid_id)

            if not batch_messages:
                continue

            def _record(vid_id, score):
                return {"video_id": vid_id, "score": score,
                        "reader": reader, "model": slug}

            try:
                outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
            except Exception as e:
                logging.error(f"  Batch failed: {e}, falling back to single")
                for i, msgs in enumerate(batch_messages):
                    try:
                        out_single = llm.chat(messages=[msgs], sampling_params=sampling_params)
                        score = extract_binary_score(out_single[0], label_token_ids)
                    except Exception as e2:
                        err2 = str(e2)
                        if "maximum model length" in err2 or "max_model_len" in err2:
                            logging.warning(f"  {batch_vid_ids[i]}: SKIPPED (too long)")
                        else:
                            logging.error(f"  {batch_vid_ids[i]}: single failed: {err2[:200]}")
                        score = None
                        n_skipped += 1
                    rec = _record(batch_vid_ids[i], score)
                    rec["skipped"] = score is None
                    with open(out_path, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    n_processed += 1
                continue

            with open(out_path, "a") as f:
                for i, output in enumerate(outputs):
                    score = extract_binary_score(output, label_token_ids)
                    f.write(json.dumps(_record(batch_vid_ids[i], score), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            n_processed += len(batch_vid_ids)
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            logging.info(f"  [{reader} {n_processed}/{len(remaining)}] {rate:.2f} vid/s")

        logging.info(f"=== reader={reader} done: {n_processed} scored, {n_skipped} skipped ===")

    logging.info("All requested readers complete.")


if __name__ == "__main__":
    main()
