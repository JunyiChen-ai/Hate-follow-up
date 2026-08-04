from __future__ import annotations

import argparse
import json
import logging
import os
import time

try:
    from .common import DATASETS, PROJECT_ROOT, RESULT_ROOT, build_image_content, model_tag, parse_yes_no, prompt_for_record, read_jsonl, required_judge_ids
    from .data_utils import PROCESSED_ROOT
except ImportError:
    from common import DATASETS, PROJECT_ROOT, RESULT_ROOT, build_image_content, model_tag, parse_yes_no, prompt_for_record, read_jsonl, required_judge_ids
    from data_utils import PROCESSED_ROOT


def output_path(dataset: str, split: str, tag: str):
    return RESULT_ROOT / "judges" / dataset / f"{split}_{tag}.jsonl"


def judge_dataset(args, dataset: str, split: str) -> None:
    rows = read_jsonl(PROCESSED_ROOT / dataset / f"{split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    tag = args.model_tag or model_tag(args.model)
    if split == "test" and not args.all_rows:
        band_path = RESULT_ROOT / "boundary" / dataset / "candidates_entropy_band.jsonl"
        band_rows = read_jsonl(band_path)
        if not band_rows:
            raise RuntimeError(f"{dataset}: missing entropy band file for band-only judging: {band_path}")
        band_ids = {r["id"] for r in band_rows if r.get("in_band") is True}
        rows = [r for r in rows if r["id"] in band_ids]
        logging.info("Judge %s/%s restricted to entropy band: %d rows", dataset, split, len(rows))
        required_ids = required_judge_ids(dataset, tag)
        if required_ids:
            rows = [r for r in rows if r["id"] in required_ids]
            logging.info("Judge %s %s/%s required by sequential policy: %d rows", tag, dataset, split, len(rows))
    out_path = output_path(dataset, split, tag)
    done = {r["id"] for r in read_jsonl(out_path) if r.get("id")}
    remaining = [r for r in rows if r["id"] not in done]
    logging.info("Judge %s %s/%s: input=%d done=%d remaining=%d out=%s", tag, dataset, split, len(rows), len(done), len(remaining), out_path)
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
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    elif args.quantization and args.quantization.lower() == "awq":
        llm_kwargs["dtype"] = "float16"
    llm = LLM(**llm_kwargs)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    effective_batch_size = args.batch_size
    if args.quantization and args.quantization.lower() == "awq" and "32b" in args.model.lower() and effective_batch_size < 4:
        effective_batch_size = 4
        logging.info("Using batch_size=%d for 32B AWQ judge (requested %d)", effective_batch_size, args.batch_size)

    processed = 0
    t0 = time.time()
    for start in range(0, len(remaining), effective_batch_size):
        batch = remaining[start : start + effective_batch_size]
        messages, valid_rows = [], []
        for row in batch:
            if not os.path.isfile(row["image_path"]):
                logging.warning("missing image: %s", row["id"])
                continue
            content = build_image_content(row["image_path"]) + [{"type": "text", "text": prompt_for_record(row, judge=True)}]
            messages.append([
                {"role": "system", "content": "You are a careful harmful meme verifier."},
                {"role": "user", "content": content},
            ])
            valid_rows.append(row)
        if not messages:
            continue
        try:
            outputs = llm.chat(messages=messages, sampling_params=params)
        except Exception as exc:
            logging.error("judge batch failed (%s); falling back to single", str(exc)[:300])
            outputs = []
            for msg in messages:
                try:
                    outputs.append(llm.chat(messages=[msg], sampling_params=params)[0])
                except Exception as sub_exc:
                    logging.error("judge single failed: %s", str(sub_exc)[:300])
                    outputs.append(None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for row, output in zip(valid_rows, outputs):
                raw = output.outputs[0].text if output is not None and output.outputs else ""
                rec = {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "label": row["label"],
                    "pred": parse_yes_no(raw),
                    "raw_response": raw,
                    "model": args.model,
                    "model_tag": tag,
                    "image_path": row["image_path"],
                    "source_id": row["source_id"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        processed += len(valid_rows)
        elapsed = max(time.time() - t0, 1e-6)
        logging.info("Judge %s %s/%s [%d/%d] %.3f sample/s", tag, dataset, split, processed, len(remaining), processed / elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier Judge MLLM for harmful meme variant")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-mm-kwargs", action="store_true")
    parser.add_argument("--tokenizer-mode", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--all-rows", action="store_true", help="Judge the full split instead of only entropy-band candidates")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "logs" / "meme_variant"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model_tag or model_tag(args.model)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"judge_{tag}.log"),
            logging.StreamHandler(),
        ],
    )
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for ds in datasets:
        judge_dataset(args, ds, args.split)


if __name__ == "__main__":
    main()
