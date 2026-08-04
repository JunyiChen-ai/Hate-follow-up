from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from .common import DATASETS, LOG_ROOT, RESULT_ROOT, dataset_iter, done_ids, image_placeholder_prompt, load_split, parse_harmful_harmless, safe_text

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
OUTPUT_SUBDIR = "lorehm_32b_awq"
POOL_TOPK = 50

BASIC_PROMPT = '''Given the meme, with the Text: "{}" embedded in the image, your task is to determine whether this meme is harmful or not, in order to maintain the harmlessness and integrity of information on the Internet. Please leverage your extensive knowledge to deeply analyze and understand this meme, and give your final judgment.
Your output should stricly follow the format: "Thought: [Your analysis]\\nAnswer: [harmful/harmless].'''

RSA_PROMPT = '''Given the meme, with the Text: "{}" embedded in the image, your task is to determine whether this meme is harmful or not, in order to maintain the harmlessness and integrity of information on the Internet. A classifier that can identify common features among multiple memes has labeled this meme as {}. Please review the classifier's judgment carefully and use your extensive knowledge to analyze and understand this meme before providing your final verdict. If you disagree with the classifier's judgment, you must provide exceptionally thorough and persuasive reasons.
Your output should strictly follow the format: "Thought: [Your analysis]\\nAnswer: [harmful/harmless]."\\n'''

LABELS_STR = ["harmless", "harmful"]

FHM_INSIGHTS = """1. Always consider the potential for double meanings, innuendos, and cultural stereotypes in the text of a meme.
2. Evaluate the context in which the meme could be shared, its intent, and its potential impact on different audiences.
3. Be cautious of memes that could be interpreted as promoting or trivializing inappropriate or harmful behavior.
4. Consider the broader social and cultural implications of the meme's content and text, while also weighing the intent, humor, and factual accuracy behind it.
5. Be sensitive to cultural stereotypes and avoid memes that could perpetuate or reinforce negative stereotypes about any group.
6. Differentiate between provocative or controversial statements and genuinely harmful content, ensuring that the meme's potential to cause real harm is the primary consideration."""

MAMI_INSIGHTS = """1. Always consider if the meme perpetuates stereotypes or objectifies individuals based on gender, race, or other attributes.
2. Assess if the humor or content used in the meme could be interpreted as demeaning, derogatory, or objectifying towards any group.
3. Take into account the broader social context, intent, and potential negative interpretations of the meme, distinguishing between harmful content and harmless humor.
4. Evaluate whether the meme could be perceived as trivializing or making light of serious social issues, even if the intent appears humorous.
5. Consider the use of satire, exaggeration, and wordplay in the meme and whether it is intended to provoke thought or humor rather than to harm."""

TOXICN_ADAPTED_INSIGHTS = """1. Consider both the Chinese meme text and the visual context together; harmful intent may appear only through their combination.
2. Be careful with sarcasm, slang, homophones, and internet catchphrases that can intensify insults or discriminatory implications.
3. Distinguish ordinary teasing or situational humor from content that demeans, abuses, threatens, or discriminates against a person or group.
4. Pay attention to attacks on protected or vulnerable groups, dehumanizing metaphors, humiliating stereotypes, and calls for hostility.
5. If the meme is merely crude, absurd, or personally insulting without broader harmful targeting, avoid over-classifying it as harmful."""

MEME_INSIGHTS = {"FHM": FHM_INSIGHTS, "MAMI": MAMI_INSIGHTS, "ToxiCN_MM": TOXICN_ADAPTED_INSIGHTS}


def _load_jina_clip():
    from sentence_transformers import SentenceTransformer
    import torch
    return SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True, truncate_dim=512, device="cuda" if torch.cuda.is_available() else "cpu")


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def build_features(rows: list[dict], model) -> dict[str, np.ndarray]:
    feats = {}
    for row in rows:
        if not Path(row["image_path"]).is_file():
            continue
        try:
            img = model.encode([row["image_path"]], normalize_embeddings=True)[0]
            txt = model.encode([row.get("text") or ""], normalize_embeddings=True)[0]
            feats[row["id"]] = _norm(0.5 * np.asarray(img) + 0.5 * np.asarray(txt))
        except Exception as exc:
            logging.error("feature failed %s: %s", row["id"], str(exc)[:200])
    return feats


def build_rel_sampl(dataset: str, persist: bool = True) -> dict:
    out_path = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "rel_sampl.json"
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return {k: (v[0], v[1], v[2]) for k, v in payload.items()}
    model = _load_jina_clip()
    train = load_split(dataset, "train")
    test = load_split(dataset, "test")
    logging.info("[%s] building LoReHM rel_sampl features", dataset)
    train_feats = build_features(train, model)
    test_feats = build_features(test, model)
    train_by_id = {r["id"]: r for r in train}
    harmful = [r["id"] for r in train if r.get("label") == 1 and r["id"] in train_feats]
    harmless = [r["id"] for r in train if r.get("label") == 0 and r["id"] in train_feats]
    rel = {}
    train_ids = list(train_feats)
    mat = np.stack([train_feats[i] for i in train_ids]) if train_ids else np.zeros((0, 512), dtype=np.float32)
    for row in test:
        feat = test_feats.get(row["id"])
        if feat is None or len(train_ids) == 0:
            continue
        sims = mat @ feat
        idx = np.argsort(-sims)[:POOL_TOPK]
        examples = [train_ids[i] for i in idx]
        rel[row["id"]] = (examples, harmful, harmless)
    if persist:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({k: [list(v[0]), list(v[1]), list(v[2])] for k, v in rel.items()}, f, ensure_ascii=False)
    return rel


def get_rsa_label(rel_sampl, k: int) -> int:
    examples, harmful_examples, harmless_examples = rel_sampl
    examples = examples[:k]
    count = 0
    for example in examples:
        if example in harmful_examples:
            count += 1
        elif example in harmless_examples:
            count -= 1
    return 1 if count >= 0 else 0


def extract_thought(text: str) -> str:
    if "Thought:" in text and "Answer:" in text:
        return text.split("Thought:", 1)[1].split("Answer:", 1)[0].strip()
    return ""


def run_model(llm, processor, image, query: str, max_tokens: int):
    from vllm import SamplingParams
    prompt = image_placeholder_prompt(processor, query)
    outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": [image]}}, sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens))
    return outputs[0].outputs[0].text.strip()


def score_one(row, rel_sampl, args, llm, processor):
    image = Image.open(row["image_path"]).convert("RGB")
    text = safe_text(row, args.text_limit)
    mia_block = MEME_INSIGHTS.get(row["dataset"]) if args.mia else None
    query = BASIC_PROMPT.format(text)
    if mia_block:
        query += f"\nNote:\n{mia_block}\n"
    basic_response = run_model(llm, processor, image, query, args.max_tokens)
    basic_predict = parse_harmful_harmless(basic_response, default=1)
    final_predict = basic_predict
    final_response = basic_response
    rsa_label = None
    used_rsa = False
    if args.rsa and row["id"] in rel_sampl:
        rsa_label = get_rsa_label(rel_sampl[row["id"]], args.rsa_k)
        if basic_predict != rsa_label:
            reask = RSA_PROMPT.format(text, LABELS_STR[rsa_label])
            if mia_block:
                reask += f"\nNote:\n{mia_block}\n"
            final_response = run_model(llm, processor, image, reask, args.max_tokens)
            final_predict = parse_harmful_harmless(final_response, default=1)
            used_rsa = True
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "label": int(row["label"]),
        "pred": int(final_predict),
        "raw_response": final_response,
        "model": args.model,
        "prompt_version": "lorehm_meme_v1",
        "basic_predict": int(basic_predict),
        "rsa_label": int(rsa_label) if rsa_label is not None else None,
        "used_rsa_reask": bool(used_rsa),
        "mia": bool(args.mia and mia_block),
        "mia_adapted": row["dataset"] == "ToxiCN_MM" and bool(args.mia and mia_block),
        "thought": extract_thought(final_response),
    }


def score_dataset(args, dataset: str, llm, processor) -> None:
    rel = build_rel_sampl(dataset) if args.rsa else {}
    rows = load_split(dataset, "test")
    if args.limit:
        rows = rows[: args.limit]
    out_path = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "test_lorehm.jsonl"
    done = done_ids(out_path)
    remaining = [r for r in rows if r["id"] not in done]
    logging.info("[%s] input=%d done=%d remaining=%d out=%s", dataset, len(rows), len(done), len(remaining), out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(remaining, 1):
            try:
                rec = score_one(row, rel, args, llm, processor)
            except Exception as exc:
                logging.error("[%s] %s failed: %s", dataset, row["id"], str(exc)[:300])
                rec = {"id": row["id"], "dataset": dataset, "label": int(row["label"]), "pred": -1, "raw_response": "", "model": args.model, "error": str(exc)[:500]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if i == 1 or i % 10 == 0:
                logging.info("[%s] %d/%d %.3f sample/s pred=%s", dataset, i, len(remaining), i / max(time.time() - t0, 1e-6), rec.get("pred"))


def main() -> None:
    parser = argparse.ArgumentParser(description="LoReHM harmful meme adaptation")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rsa", action="store_true", default=True)
    parser.add_argument("--no-rsa", dest="rsa", action="store_false")
    parser.add_argument("--mia", action="store_true", default=True)
    parser.add_argument("--no-mia", dest="mia", action="store_false")
    parser.add_argument("--rsa-k", type=int, default=5)
    parser.add_argument("--text-limit", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_ROOT / "lorehm_32b_awq.log"), logging.StreamHandler()])

    from transformers import AutoProcessor
    from vllm import LLM
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.88, max_model_len=65536, limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": 32768}, enforce_eager=True)
    for ds in dataset_iter(args.dataset):
        score_dataset(args, ds, llm, processor)


if __name__ == "__main__":
    main()

