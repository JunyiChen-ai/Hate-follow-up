from __future__ import annotations

import argparse
import json
import logging
import os
import time

from .common import (
    DATASETS,
    LOG_ROOT,
    RESULT_ROOT,
    dataset_iter,
    done_ids,
    image_content,
    load_split,
    parse_yes_no,
    safe_text,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_SUBDIR = "naive_2b"
PROMPT = """Is this meme harmful?
Meme text: {text}

Only respond Yes or No."""


def score_dataset(args, dataset: str, llm, sampling_params) -> None:
    rows = load_split(dataset, "test")
    if args.limit:
        rows = rows[: args.limit]
    out_path = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "test_naive.jsonl"
    done = done_ids(out_path)
    remaining = [r for r in rows if r["id"] not in done]
    logging.info("[%s] input=%d done=%d remaining=%d out=%s", dataset, len(rows), len(done), len(remaining), out_path)
    if not remaining:
        return

    t0 = time.time()
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start : start + args.batch_size]
        messages = []
        valid = []
        for row in batch:
            if not os.path.isfile(row["image_path"]):
                logging.error("missing image %s", row["image_path"])
                continue
            prompt = PROMPT.format(text=safe_text(row))
            messages.append([{"role": "user", "content": image_content(row["image_path"]) + [{"type": "text", "text": prompt}]}])
            valid.append(row)
        if not valid:
            continue
        try:
            outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        except Exception as exc:
            logging.error("[%s] batch failed: %s", dataset, str(exc)[:300])
            outputs = []
            for msg in messages:
                try:
                    outputs.append(llm.chat(messages=[msg], sampling_params=sampling_params)[0])
                except Exception as sub_exc:
                    outputs.append(None)
                    logging.error("[%s] single failed: %s", dataset, str(sub_exc)[:300])
        with out_path.open("a", encoding="utf-8") as f:
            for row, output in zip(valid, outputs):
                text = output.outputs[0].text if output is not None else ""
                rec = {
                    "id": row["id"],
                    "dataset": dataset,
                    "label": int(row["label"]),
                    "pred": parse_yes_no(text),
                    "raw_response": text,
                    "model": args.model,
                    "prompt_version": "naive_meme_v1",
                    "image_path": row["image_path"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n += len(valid)
        if n == len(valid) or n % (args.batch_size * 10) == 0:
            rate = n / max(time.time() - t0, 1e-6)
            logging.info("[%s] %d/%d %.2f sample/s", dataset, n, len(remaining), rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive MLLM harmful meme baseline")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=100352)
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_ROOT / "naive_2b.log"), logging.StreamHandler()],
    )
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": args.max_pixels},
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=4)
    for ds in dataset_iter(args.dataset):
        score_dataset(args, ds, llm, sampling_params)


if __name__ == "__main__":
    main()

