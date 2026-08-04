from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

from .common import DATASETS, LOG_ROOT, RESULT_ROOT, dataset_iter, done_ids, load_split, read_jsonl, safe_text

PROJECT_ROOT = "/data/jehc223/EMNLP2"
DEFAULT_BASE_MODEL = "yahma/llama-7b-hf"
YES_TOKEN_ID = 8241
NO_TOKEN_ID = 3782
INSTRUCTION = "Please decide whether the meme is hateful according to its image caption and meme text."

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "mod_hate_repro"))
import lora_compose  # noqa: E402


def load_captions(dataset: str, split: str) -> dict[str, str]:
    path = RESULT_ROOT / "captions" / dataset / f"{split}_captions.jsonl"
    rows = read_jsonl(path)
    return {r["id"]: r.get("caption", "") for r in rows if r.get("id")}


def row_to_example(row: dict, caption: str, output: str | None = None) -> dict:
    text = safe_text(row, 1200)
    ex = {
        "img": row["id"],
        "instruction": INSTRUCTION,
        "input": f"Image caption:{caption}\nMeme text:{text}",
    }
    if output is not None:
        ex["output"] = output
    return ex


def build_support_and_test(dataset: str, num_shots: int, seed: int):
    train = load_split(dataset, "train")
    test = load_split(dataset, "test")
    caps_train = load_captions(dataset, "train")
    caps_test = load_captions(dataset, "test")
    rnd = random.Random(seed)
    train = list(train)
    rnd.shuffle(train)
    counts = {0: 0, 1: 0}
    support = []
    for row in train:
        y = int(row["label"])
        if counts[y] >= num_shots:
            continue
        counts[y] += 1
        support.append(row_to_example(row, caps_train.get(row["id"], ""), "Yes" if y == 1 else "No"))
        if counts[0] >= num_shots and counts[1] >= num_shots:
            break
    test_rows = [row_to_example(r, caps_test.get(r["id"], "")) | {"label": int(r["label"])} for r in test]
    return support, test_rows, counts


def generate_eval_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:\n"""


def generate_train_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:{data_point["instruction"]}

                ### Input:{data_point["input"]}

                ### Response:{data_point["output"]}"""


def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


class SupportDataset:
    def __init__(self, rows, tokenizer, cutoff_len=512, train_on_inputs=False):
        import torch
        self.torch = torch
        self.rows = rows
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs
        self.entries = [self._prep(r) for r in rows]

    def _prep(self, data_point):
        tok = tokenize(generate_train_prompt(data_point), self.tokenizer, self.cutoff_len)
        if not self.train_on_inputs:
            tok_user = tokenize(generate_train_prompt({**data_point, "output": ""}), self.tokenizer, self.cutoff_len, add_eos_token=False)
            user_prompt_len = len(tok_user["input_ids"])
            tok["labels"] = [-100] * user_prompt_len + tok["labels"][user_prompt_len:]
        pad_len = self.cutoff_len - len(tok["input_ids"])
        if pad_len > 0:
            tok["input_ids"] = [0] * pad_len + tok["input_ids"]
            tok["labels"] = [-100] * pad_len + tok["labels"]
            tok["attention_mask"] = [0] * pad_len + tok["attention_mask"]
        return tok

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        t = self.torch
        return {k: t.LongTensor(e[k]) for k in ("input_ids", "labels", "attention_mask")}


def score_one(row, model, tokenizer):
    import torch
    from transformers import GenerationConfig
    prompt = generate_eval_prompt(row)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, generation_config=GenerationConfig(), return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    yes_logit = out["scores"][0][0, YES_TOKEN_ID].item()
    no_logit = out["scores"][0][0, NO_TOKEN_ID].item()
    pred = 1 if yes_logit >= no_logit else 0
    response_tok = torch.argmax(out["scores"][0][0]).item()
    response = tokenizer.decode([response_tok], skip_special_tokens=True)
    return pred, yes_logit, no_logit, response


def run_dataset(dataset: str, shots: int, args) -> None:
    support, test_rows, counts = build_support_and_test(dataset, shots, args.seed)
    if counts[0] < shots or counts[1] < shots:
        logging.error("[%s] insufficient balanced support for %d-shot: %s", dataset, shots, counts)
        return
    lora_paths = lora_compose.build_lora_module_list(lora_root=args.lora_root)
    from transformers import LlamaTokenizer
    tokenizer_tmp = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer_tmp.pad_token_id = 0
    tokenizer_tmp.padding_side = "left"
    support_ds = SupportDataset(support, tokenizer_tmp, cutoff_len=args.cutoff_len)
    del tokenizer_tmp

    weights, model, tokenizer = lora_compose.lorahub_learning(
        lora_module_list=lora_paths,
        base_model=args.base_model,
        support_dataset=support_ds,
        max_inference_step=args.max_inference_step,
        batch_size=args.batch_size,
        load_8bit=args.load_8bit,
        seed=args.seed,
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    out = RESULT_ROOT / "mod_hate" / dataset / f"test_mod_hate_{shots}shot.jsonl"
    done = done_ids(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    remaining = [r for r in test_rows if r["img"] not in done]
    logging.info("[%s] Mod-HATE %d-shot test=%d done=%d remaining=%d", dataset, shots, len(test_rows), len(done), len(remaining))
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(remaining, 1):
            try:
                pred, yes_l, no_l, resp = score_one(row, model, tokenizer)
                rec = {"id": row["img"], "dataset": dataset, "label": int(row["label"]), "pred": int(pred), "raw_response": resp, "model": args.base_model, "prompt_version": "mod_hate_meme_v1", "yes_logit": yes_l, "no_logit": no_l, "lora_weights": list(weights)}
            except Exception as exc:
                logging.error("[%s] %s score failed: %s", dataset, row["img"], str(exc)[:250])
                rec = {"id": row["img"], "dataset": dataset, "label": int(row["label"]), "pred": -1, "raw_response": "", "model": args.base_model, "error": str(exc)[:500]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if i == 1 or i % 20 == 0:
                logging.info("[%s] Mod-HATE %d-shot %d/%d %.3f sample/s", dataset, shots, i, len(remaining), i / max(time.time() - t0, 1e-6))
    del model
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mod-HATE harmful meme runner")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--shots", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-root", default=lora_compose.UPSTREAM_LORA_ROOT)
    parser.add_argument("--max-inference-step", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true", default=True)
    parser.add_argument("--no-load-8bit", dest="load_8bit", action="store_false")
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_ROOT / "mod_hate.log"), logging.StreamHandler()])
    for ds in dataset_iter(args.dataset):
        for k in args.shots:
            run_dataset(ds, k, args)


if __name__ == "__main__":
    main()

