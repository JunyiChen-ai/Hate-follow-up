from __future__ import annotations

import argparse
import json
import logging
import os
import time

try:
    from .common import (
        DATASETS,
        DEFAULT_STAGE1_MODEL,
        PROJECT_ROOT,
        RESULT_ROOT,
        binary_score_from_output,
        binary_token_ids,
        build_image_content,
        model_tag,
        prompt_for_record,
        read_jsonl,
    )
    from .data_utils import PROCESSED_ROOT
except ImportError:
    from common import (
        DATASETS,
        DEFAULT_STAGE1_MODEL,
        PROJECT_ROOT,
        RESULT_ROOT,
        binary_score_from_output,
        binary_token_ids,
        build_image_content,
        model_tag,
        prompt_for_record,
        read_jsonl,
    )
    from data_utils import PROCESSED_ROOT


def output_path(dataset: str, split: str, slug: str) -> os.PathLike:
    return RESULT_ROOT / f"holistic_{slug}" / dataset / f"{split}_binary.jsonl"


def load_done(path) -> set[str]:
    return {r["id"] for r in read_jsonl(path) if r.get("id")}


def score_dataset(args, dataset: str, split: str) -> None:
    in_path = PROCESSED_ROOT / dataset / f"{split}.jsonl"
    rows = read_jsonl(in_path)
    if args.limit:
        rows = rows[: args.limit]
    slug = args.model_slug or model_tag(args.model)
    out_path = output_path(dataset, split, slug)
    done = load_done(out_path)
    remaining = [r for r in rows if r["id"] not in done]
    logging.info("Stage-1 %s/%s: input=%d done=%d remaining=%d out=%s", dataset, split, len(rows), len(done), len(remaining), out_path)
    if not remaining:
        return

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "limit_mm_per_prompt": {"image": 1},
        "allowed_local_media_path": "/data/jehc223",
    }
    if not args.no_mm_kwargs:
        llm_kwargs["mm_processor_kwargs"] = {"max_pixels": args.max_pixels}
    if args.tokenizer_mode:
        llm_kwargs["tokenizer_mode"] = args.tokenizer_mode
    llm = LLM(**llm_kwargs)
    label_ids = binary_token_ids(llm.get_tokenizer())
    allowed = sorted({tid for tids in label_ids.values() for tid in tids})
    params = SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=allowed)

    t0 = time.time()
    processed = 0
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start : start + args.batch_size]
        messages = []
        valid_rows = []
        for row in batch:
            if not os.path.isfile(row["image_path"]):
                logging.warning("missing image: %s", row["id"])
                continue
            content = build_image_content(row["image_path"]) + [{"type": "text", "text": prompt_for_record(row)}]
            messages.append([
                {"role": "system", "content": "You are a content moderation analyst. Answer based strictly on observable evidence."},
                {"role": "user", "content": content},
            ])
            valid_rows.append(row)
        if not messages:
            continue
        try:
            outputs = llm.chat(messages=messages, sampling_params=params)
        except Exception as exc:
            logging.error("batch failed (%s); falling back to single", str(exc)[:300])
            outputs = []
            for msg in messages:
                try:
                    outputs.append(llm.chat(messages=[msg], sampling_params=params)[0])
                except Exception as sub_exc:
                    logging.error("single failed: %s", str(sub_exc)[:300])
                    outputs.append(None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for row, output in zip(valid_rows, outputs):
                score = binary_score_from_output(output, label_ids) if output is not None else None
                rec = {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "label": row["label"],
                    "score": score,
                    "model": args.model,
                    "image_path": row["image_path"],
                    "source_id": row["source_id"],
                    "skipped": score is None,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        processed += len(valid_rows)
        elapsed = max(time.time() - t0, 1e-6)
        logging.info("Stage-1 %s/%s [%d/%d] %.3f sample/s", dataset, split, processed, len(remaining), processed / elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 Judge MLLM scoring for harmful memes")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--split", default="all", choices=("all", "train", "test"))
    parser.add_argument("--model", default=DEFAULT_STAGE1_MODEL)
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--no-mm-kwargs", action="store_true")
    parser.add_argument("--tokenizer-mode", default=None)
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "logs" / "meme_variant"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "score_stage1.log"),
            logging.StreamHandler(),
        ],
    )
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    for ds in datasets:
        for split in splits:
            score_dataset(args, ds, split)


if __name__ == "__main__":
    main()

