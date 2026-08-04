from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path("/data/jehc223/EMNLP2")
DATA_ROOT = PROJECT_ROOT / "datasets" / "harmful_meme" / "processed"
RESULT_ROOT = PROJECT_ROOT / "results" / "meme_baselines"
LOG_ROOT = PROJECT_ROOT / "logs" / "meme_baselines"
DATASETS = ("FHM", "MAMI", "ToxiCN_MM")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_split(dataset: str, split: str) -> list[dict]:
    return read_jsonl(DATA_ROOT / dataset / f"{split}.jsonl")


def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in read_jsonl(path) if r.get("id")}


def image_content(path: str) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": f"file://{path}"}}]


def image_placeholder_prompt(processor, prompt_text: str, n_images: int = 1) -> str:
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": prompt_text})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def parse_yes_no(text: str | None) -> int:
    if not text:
        return -1
    t = text.strip().lower().lstrip("\"'`#*- \n\t")
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    answer_match = re.search(r"(?im)^\s*answer\s*:\s*(yes|no|harmful|harmless)\b", text)
    if answer_match:
        val = answer_match.group(1).lower()
        return 1 if val in ("yes", "harmful") else 0
    if re.search(r"\bharmless\b", t):
        return 0
    if re.search(r"\bharmful\b|\bhateful\b", t):
        return 1
    return -1


def parse_harmful_harmless(text: str | None, default: int = 1) -> int:
    pred = parse_yes_no(text)
    if pred in (0, 1):
        return pred
    return default


def token_ids(tokenizer, labels: tuple[str, ...]) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {label: set() for label in labels}
    for label in labels:
        variants = (label, f" {label}", label.lower(), f" {label.lower()}", label.upper(), f" {label.upper()}")
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                out[label].add(ids[0])
    return {k: sorted(v) for k, v in out.items()}


def first_token_prob(output, positive_ids: list[int], negative_ids: list[int]) -> float | None:
    if not output or not output.outputs or not output.outputs[0].logprobs:
        return None
    pos0 = output.outputs[0].logprobs[0]
    fallback = -30.0
    p_pos = sum(math.exp(pos0[tid].logprob if tid in pos0 else fallback) for tid in positive_ids)
    p_neg = sum(math.exp(pos0[tid].logprob if tid in pos0 else fallback) for tid in negative_ids)
    total = p_pos + p_neg
    if total <= 0:
        return None
    return p_pos / total


def compute_metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("label") in (0, 1) and r.get("pred") in (0, 1)]
    y = np.array([int(r["label"]) for r in valid])
    yh = np.array([int(r["pred"]) for r in valid])
    if len(valid) == 0:
        return {"n": 0, "acc": None, "mf1": None, "mp": None, "mr": None}
    acc = float((y == yh).mean())
    classes = sorted(set(y.tolist()) | set(yh.tolist()))
    fs, ps, rs = [], [], []
    for c in classes:
        tp = int(((y == c) & (yh == c)).sum())
        fp = int(((y != c) & (yh == c)).sum())
        fn = int(((y == c) & (yh != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return {
        "n": int(len(valid)),
        "n_total": int(len(rows)),
        "n_invalid": int(len(rows) - len(valid)),
        "acc": acc,
        "mf1": float(sum(fs) / len(fs)) if fs else 0.0,
        "mp": float(sum(ps) / len(ps)) if ps else 0.0,
        "mr": float(sum(rs) / len(rs)) if rs else 0.0,
        "n_pos_gt": int(y.sum()),
        "n_pos_pred": int(yh.sum()),
    }


def dataset_iter(selection: str) -> tuple[str, ...]:
    return DATASETS if selection == "all" else (selection,)


def safe_text(row: dict, limit: int = 1200) -> str:
    return (row.get("text") or "")[:limit]
