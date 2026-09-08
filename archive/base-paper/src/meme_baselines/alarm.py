from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from .common import DATASETS, LOG_ROOT, RESULT_ROOT, dataset_iter, done_ids, first_token_prob, image_placeholder_prompt, load_split, parse_harmful_harmless, safe_text, token_ids

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_SUBDIR = "alarm_7b"

LABEL_PROMPT = """
Given the meme with its image and the textual element '{text}' embedded in the image, your task is to carefully and critically assess whether this meme is harmful or not, in order to maintain the benignness and integrity of information on the Internet.
Please leverage your extensive knowledge to deeply analyze and understand this meme, and give your final judgment.
Please not that while your primary goal is to provide a judgment after thoughtful analysis, it's important to avoid overgeneralizing or being overly conclusive in cases where ambiguity exists.
You must only return `1' for harmful, or `0' for benign.
            """

EXPERIENCE_PROMPT = """
Meme A Text: {text1}
Meme B Text: {text2}
Given two memes that are visually or structurally similar but belong to distinct categories: Meme i, which is harmful, and Meme j, which is benign. The input contains two images: image 1 is Meme A, image 2 is Meme B. Please complete the following two steps:
Step 1:
Clearly summarize the content of each meme by carefully analyzing its image and textual element accompanying the meme, and considering any implicit or explicit messages it conveys.
Step 2:
Based on the content of two memes, contrast the key differences between them to explain why Meme i is classified as harmful content, while Meme j remains benign.
    """

REFERENCE_PROMPT = """
You have a set of experiences for identifying harmful memes, originally created by comparing similar but contradictory categories of two memes.
Now, a new experience arrives containing the description of one harmful and one similar but benign one, and a summary of the differences between the them.
Your task is to distill new references from the experience and update the existing references by choosing one operation: : ADD, EDIT, UPVOTE, and DOWNVOTE.

Strict Rules:

1. ADD only if:
   - add new references that are very different from exisitng references and relevatnt for other detection.
2. EDIT must:
   - if any existing reference is not general enough or can be enhanced, rewrite and improve it.
3. UPVOTE if:
   - if the existing reference is strongly relevant for current reference
4. DOWNVOTE if:
   - if one exisiing reference is contradictory or similar/duplicated to other existing reference.
5. Maximum {size} references preserved
6. Output only valid JSON

Context:
Current references Set (importance order):
{cur_set_str}

New Coming Experience:
{new_experience}

Processing Steps:

1. Ensure the added and edited references are concise, clear while keeping them 2 or 3 sentences.
2. Ensure the references are concise and easy to follow.
3. Actively downvote references that are vague or hard to understand, and maintain the reference set at {size} items.
4. Try to make every reference useful, make more upvotes, and downvotes.
5. Refine references to emphasize distinct signals that uniquely identify specific harmful patterns.
6. Generalize references to extract universal principles that capture common traits of harmful content.
7. Return only the JSON operations with the below format:
[
  {{
    "operation": "<ADD|EDIT|UPVOTE|DOWNVOTE>",
    "reference": "<index/none>",
    "insight": "<new/revised text>"
  }}
]
"""

INPREDICT_PROMPT = """
Given a meme with its image and textual element '{text}' embedded in the image, your task is to determine whether this meme is harmful or benign by considering its multimodal content.
Moreover, a set of valuable detection references are also provided to serve as additional knowledge, which will help you in determine the class of the given meme.
References set:
{reference_set}.
Please leverage your pre-trained knowledge and the references to analyze and understand this meme, and give your final judgment.
Your output should strictly follow the format: Thought: [Your analysis] Answer: [harmful/harmless].
"""


def run_text_image(llm, processor, images, prompt: str, sampling_params):
    full_prompt = image_placeholder_prompt(processor, prompt, n_images=len(images))
    out = llm.generate({"prompt": full_prompt, "multi_modal_data": {"image": images}}, sampling_params=sampling_params)
    return out[0].outputs[0].text.strip()


def load_image(row: dict):
    return Image.open(row["image_path"]).convert("RGB")


def label_stage(args, dataset: str, llm, processor, sampling_params, label_ids) -> Path:
    from vllm import SamplingParams
    out = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "label.jsonl"
    done = done_ids(out)
    rows = [r for r in load_split(dataset, "train") if r["id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    allowed = sorted(set(label_ids["1"]) | set(label_ids["0"]))
    params = SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=allowed)
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            try:
                prompt = LABEL_PROMPT.format(text=safe_text(row))
                full = image_placeholder_prompt(processor, prompt)
                image = load_image(row)
                output = llm.generate({"prompt": full, "multi_modal_data": {"image": [image]}}, sampling_params=params)[0]
                text = output.outputs[0].text.strip()
                prob1 = first_token_prob(output, label_ids["1"], label_ids["0"])
                pred = 1 if (prob1 if prob1 is not None else 0.0) >= 0.5 else 0
                rec = {"id": row["id"], "dataset": dataset, "label": int(row["label"]), "pred": int(pred), "prob0": float(1 - prob1) if prob1 is not None else 0.5, "prob1": float(prob1) if prob1 is not None else 0.5, "raw_response": text}
            except Exception as exc:
                logging.error("[%s] label %s failed: %s", dataset, row["id"], str(exc)[:250])
                rec = {"id": row["id"], "dataset": dataset, "label": int(row["label"]), "pred": -1, "prob0": 0.5, "prob1": 0.5, "raw_response": "", "error": str(exc)[:500]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if i == 1 or i % 100 == 0:
                logging.info("[%s] label %d/%d", dataset, i, len(rows))
    return out


def _load_jina_clip():
    from sentence_transformers import SentenceTransformer
    import torch
    return SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True, truncate_dim=512, device="cuda" if torch.cuda.is_available() else "cpu")


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def build_embeddings(dataset: str) -> dict[str, np.ndarray]:
    emb_path = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "joint_embed.json"
    if emb_path.exists():
        with emb_path.open(encoding="utf-8") as f:
            return {k: np.asarray(v, dtype=np.float32) for k, v in json.load(f).items()}
    rows = load_split(dataset, "train")
    model = _load_jina_clip()
    feats = {}
    for i, row in enumerate(rows, 1):
        try:
            img = model.encode([row["image_path"]], normalize_embeddings=True)[0]
            txt = model.encode([row.get("text") or ""], normalize_embeddings=True)[0]
            feats[row["id"]] = _norm(0.5 * np.asarray(img) + 0.5 * np.asarray(txt))
        except Exception as exc:
            logging.error("[%s] embed %s failed: %s", dataset, row["id"], str(exc)[:200])
        if i == 1 or i % 500 == 0:
            logging.info("[%s] embeddings %d/%d", dataset, i, len(rows))
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    with emb_path.open("w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in feats.items()}, f)
    return feats


def retrieve_pairs(dataset: str, coverage_rate: float) -> Path:
    out = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "retrieve" / "pairs.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out
    labels = [r for r in [json.loads(x) for x in (RESULT_ROOT / OUTPUT_SUBDIR / dataset / "label.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] if r.get("pred") in (0, 1)]
    labels = sorted(labels, key=lambda r: max(float(r.get("prob0", 0.5)), float(r.get("prob1", 0.5))), reverse=True)
    keep = labels[: max(2, int(len(labels) * coverage_rate))]
    pos = [r["id"] for r in keep if r["pred"] == 1]
    neg = [r["id"] for r in keep if r["pred"] == 0]
    feats = build_embeddings(dataset)
    neg_ids = [i for i in neg if i in feats]
    neg_mat = np.stack([feats[i] for i in neg_ids]) if neg_ids else np.zeros((0, 512), dtype=np.float32)
    pairs = []
    for pid in pos:
        if pid not in feats or len(neg_ids) == 0:
            continue
        sims = neg_mat @ feats[pid]
        nid = neg_ids[int(np.argmax(sims))]
        pairs.append({"id1": pid, "id2": nid, "score": float(np.max(sims))})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logging.info("[%s] retrieved %d pairs", dataset, len(pairs))
    return out


def run_experience(args, dataset: str, llm, processor, sampling_params) -> Path:
    out = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "experience.jsonl"
    pairs_path = retrieve_pairs(dataset, args.coverage_rate)
    done = {(r.get("id1"), r.get("id2")) for r in [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()] } if out.exists() else set()
    train_by_id = {r["id"]: r for r in load_split(dataset, "train")}
    pairs = [json.loads(x) for x in pairs_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for i, pair in enumerate(pairs, 1):
            key = (pair["id1"], pair["id2"])
            if key in done:
                continue
            try:
                a, b = train_by_id[pair["id1"]], train_by_id[pair["id2"]]
                prompt = EXPERIENCE_PROMPT.format(text1=safe_text(a), text2=safe_text(b))
                exp = run_text_image(llm, processor, [load_image(a), load_image(b)], prompt, sampling_params)
                rec = {"id1": pair["id1"], "id2": pair["id2"], "experience": exp}
            except Exception as exc:
                logging.error("[%s] experience %s failed: %s", dataset, key, str(exc)[:250])
                rec = {"id1": pair["id1"], "id2": pair["id2"], "experience": "", "error": str(exc)[:500]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if i == 1 or i % 20 == 0:
                logging.info("[%s] experience %d/%d", dataset, i, len(pairs))
    return out


class ReferenceSet:
    def __init__(self, size=20):
        self.size = size
        self.set = [{"reference": "placeholder", "importance": 2}]

    def cur_set_str(self):
        return "\n".join(f"{i}: {r['reference']}" for i, r in enumerate(self.set[: self.size]))

    def apply_ops(self, ops):
        for op in ops:
            kind = (op.get("operation") or "").upper()
            insight = op.get("insight") or ""
            try:
                idx = int(op.get("reference"))
            except Exception:
                idx = None
            if kind == "ADD" and insight:
                self.set.append({"reference": insight, "importance": 1})
            elif kind == "EDIT" and idx is not None and 0 <= idx < len(self.set) and insight:
                self.set[idx]["reference"] = insight
            elif kind == "UPVOTE" and idx is not None and 0 <= idx < len(self.set):
                self.set[idx]["importance"] = self.set[idx].get("importance", 1) + 1
            elif kind == "DOWNVOTE" and idx is not None and 0 <= idx < len(self.set):
                self.set[idx]["importance"] = self.set[idx].get("importance", 1) - 1
        self.set = sorted(self.set, key=lambda r: r.get("importance", 0), reverse=True)[: self.size]


def parse_ops(text: str):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return [x for x in data if isinstance(x, dict)]


def run_reference(args, dataset: str, llm, sampling_params) -> Path:
    out = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "reference.json"
    if out.exists():
        return out
    exp_path = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "experience.jsonl"
    ref = ReferenceSet(args.reference_size)
    from vllm import SamplingParams
    text_params = SamplingParams(temperature=0, max_tokens=2048)
    for i, row in enumerate([json.loads(x) for x in exp_path.read_text(encoding="utf-8").splitlines() if x.strip()], 1):
        exp = row.get("experience") or ""
        if not exp:
            continue
        prompt = REFERENCE_PROMPT.format(size=args.reference_size, cur_set_str=ref.cur_set_str(), new_experience=exp)
        try:
            out_text = llm.generate(prompt, sampling_params=text_params)[0].outputs[0].text
            ref.apply_ops(parse_ops(out_text))
        except Exception as exc:
            logging.error("[%s] reference failed: %s", dataset, str(exc)[:250])
        if i == 1 or i % 20 == 0:
            logging.info("[%s] reference %d", dataset, i)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(ref.set, f, ensure_ascii=False, indent=2)
    return out


def run_inpredict(args, dataset: str, llm, processor, sampling_params) -> None:
    out = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "test_alarm.jsonl"
    done = done_ids(out)
    refs = json.load((RESULT_ROOT / OUTPUT_SUBDIR / dataset / "reference.json").open(encoding="utf-8"))
    ref_text = "\n".join(f"{i}: {r.get('reference', '')}" for i, r in enumerate(refs))
    rows = load_split(dataset, "test")
    if args.limit:
        rows = rows[: args.limit]
    remaining = [r for r in rows if r["id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(remaining, 1):
            try:
                prompt = INPREDICT_PROMPT.format(text=safe_text(row), reference_set=ref_text)
                resp = run_text_image(llm, processor, [load_image(row)], prompt, sampling_params)
                pred = parse_harmful_harmless(resp, default=0)
                thought = resp.split("Thought:", 1)[1].split("Answer:", 1)[0].strip() if "Thought:" in resp and "Answer:" in resp else ""
                rec = {"id": row["id"], "dataset": dataset, "label": int(row["label"]), "pred": int(pred), "thought": thought, "raw_response": resp, "model": args.model, "prompt_version": "alarm_meme_v1"}
            except Exception as exc:
                logging.error("[%s] inpredict %s failed: %s", dataset, row["id"], str(exc)[:250])
                rec = {"id": row["id"], "dataset": dataset, "label": int(row["label"]), "pred": -1, "raw_response": "", "model": args.model, "error": str(exc)[:500]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if i == 1 or i % 20 == 0:
                logging.info("[%s] inpredict %d/%d %.3f sample/s", dataset, i, len(remaining), i / max(time.time() - t0, 1e-6))


def run_dataset(args, dataset: str, llm, processor, sampling_params, label_ids) -> None:
    label_stage(args, dataset, llm, processor, sampling_params, label_ids)
    build_embeddings(dataset)
    retrieve_pairs(dataset, args.coverage_rate)
    run_experience(args, dataset, llm, processor, sampling_params)
    run_reference(args, dataset, llm, sampling_params)
    run_inpredict(args, dataset, llm, processor, sampling_params)


def main() -> None:
    parser = argparse.ArgumentParser(description="ALARM 7B harmful meme adaptation")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--coverage-rate", type=float, default=0.5)
    parser.add_argument("--reference-size", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=0, help="Cap experience pairs for debugging only; default 0 means no cap.")
    args = parser.parse_args()
    if args.max_pairs == 0:
        args.max_pairs = None
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_ROOT / "alarm_7b.log"), logging.StreamHandler()])
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.88, max_model_len=32768, limit_mm_per_prompt={"image": 2}, mm_processor_kwargs={"max_pixels": 100352}, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    ids = token_ids(llm.get_tokenizer(), ("0", "1"))
    for ds in dataset_iter(args.dataset):
        run_dataset(args, ds, llm, processor, sampling_params, ids)


if __name__ == "__main__":
    main()
