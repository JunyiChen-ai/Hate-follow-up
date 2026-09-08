from __future__ import annotations

import argparse
import json
import logging
import os
import time

from .common import DATASETS, LOG_ROOT, RESULT_ROOT, dataset_iter, done_ids, image_content, load_split

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = """Describe the meme image in one concise sentence. Mention visible people, objects, setting, style, and any clearly visible text. Do not classify whether it is harmful."""


def metadata_caption(row: dict) -> str:
    meta = row.get("metadata") or {}
    return meta.get("meme_discription") or meta.get("image_description") or ""


def caption_dataset(args, dataset: str, split: str, llm, sampling_params) -> None:
    rows = load_split(dataset, split)
    if args.limit:
        rows = rows[: args.limit]
    out = RESULT_ROOT / "captions" / dataset / f"{split}_captions.jsonl"
    done = done_ids(out)
    remaining = [r for r in rows if r["id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.info("[%s/%s] captions input=%d done=%d remaining=%d", dataset, split, len(rows), len(done), len(remaining))
    t0 = time.time()
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start : start + args.batch_size]
        records = []
        messages = []
        valid = []
        for row in batch:
            cap = metadata_caption(row)
            if cap and not args.regenerate_metadata:
                records.append({"id": row["id"], "dataset": dataset, "split": split, "caption": cap, "caption_source": "metadata", "model": None})
                continue
            messages.append([{"role": "user", "content": image_content(row["image_path"]) + [{"type": "text", "text": PROMPT}]}])
            valid.append(row)
        outputs = []
        if messages:
            try:
                outputs = llm.chat(messages=messages, sampling_params=sampling_params)
            except Exception as exc:
                logging.error("[%s/%s] caption batch failed: %s", dataset, split, str(exc)[:250])
                for msg in messages:
                    try:
                        outputs.append(llm.chat(messages=[msg], sampling_params=sampling_params)[0])
                    except Exception as sub_exc:
                        logging.error("[%s/%s] caption single failed: %s", dataset, split, str(sub_exc)[:250])
                        outputs.append(None)
        for row, out_obj in zip(valid, outputs):
            cap = out_obj.outputs[0].text.strip() if out_obj is not None else ""
            records.append({"id": row["id"], "dataset": dataset, "split": split, "caption": cap, "caption_source": "vlm", "model": args.model})
        with out.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n = min(start + args.batch_size, len(remaining))
        if n == len(batch) or n % (args.batch_size * 10) == 0:
            logging.info("[%s/%s] %d/%d %.2f sample/s", dataset, split, n, len(remaining), n / max(time.time() - t0, 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate meme captions for Mod-HATE")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--split", default="all", choices=("all", "train", "test"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--regenerate-metadata", action="store_true")
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_ROOT / "captions.log"), logging.StreamHandler()])
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.90, max_model_len=32768, limit_mm_per_prompt={"image": 1}, allowed_local_media_path="/data/jehc223", mm_processor_kwargs={"max_pixels": 100352})
    sampling = SamplingParams(temperature=0, max_tokens=96)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    for ds in dataset_iter(args.dataset):
        for split in splits:
            caption_dataset(args, ds, split, llm, sampling)


if __name__ == "__main__":
    main()

