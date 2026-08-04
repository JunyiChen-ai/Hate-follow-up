from __future__ import annotations

import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path("/data/jehc223/EMNLP2")
RESULT_ROOT = PROJECT_ROOT / "results" / "meme_variant"

DATASETS = ("FHM", "MAMI", "ToxiCN_MM")
DEFAULT_STAGE1_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")

TASK_DEFINITIONS = {
    "FHM": (
        "Positive means the meme is hateful: it attacks, demeans, dehumanizes, "
        "threatens, stereotypes, or incites hostility toward a protected group "
        "or a person because of protected identity."
    ),
    "MAMI": (
        "Positive means the meme is misogynous: it shames, stereotypes, "
        "objectifies, threatens, or promotes hostility or violence toward women."
    ),
    "ToxiCN_MM": (
        "Positive means the meme is toxic or harmful in Chinese multimodal "
        "context: insulting, abusive, discriminatory, threatening, demeaning, "
        "or otherwise harmful toward a person or group."
    ),
}


STAGE1_PROMPT = """You are a content moderation analyst. Examine the meme image and its transcribed text.

Meme text:
{text}

Task definition:
{definition}

Does this meme satisfy the positive class?

Answer "Yes" or "No"."""


JUDGE_PROMPT = """You are a careful verifier for harmful meme detection. Examine the image and the transcribed meme text.

Meme text:
{text}

Task definition:
{definition}

Decide whether the meme satisfies the positive class. Use the image and text together. Do not rely on dataset-specific metadata.

Return exactly two lines:
rationale: one short sentence
answer: Yes or No"""


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_tag(model: str) -> str:
    last = model.split("/")[-1].lower()
    tag = last.replace("-instruct", "").replace("_instruct", "")
    tag = tag.replace("qwen2.5-vl-", "qwen2.5-vl-")
    tag = tag.replace("qwen3-vl-", "qwen3-vl-")
    tag = tag.replace("_", "-")
    if tag == "qwen3-vl-2b":
        return "2b"
    return tag


def build_image_content(image_path: str) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]


def prompt_for_record(row: dict, judge: bool = False) -> str:
    template = JUDGE_PROMPT if judge else STAGE1_PROMPT
    text = row.get("text") or ""
    definition = TASK_DEFINITIONS.get(row["dataset"], TASK_DEFINITIONS["FHM"])
    return template.format(text=text[:1200], definition=definition)


def parse_yes_no(text: str | None) -> int | None:
    if not text:
        return None
    answer_match = re.search(r"(?im)^\s*answer\s*:\s*(yes|no)\b", text)
    if answer_match:
        return 1 if answer_match.group(1).lower() == "yes" else 0
    hits = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if not hits:
        lowered = text.lower()
        if re.search(r"does\s+not\s+(?:satisfy|exhibit|show|contain|include).*?(?:harmful|toxic|hateful|misogyn|positive class)", lowered):
            return 0
        if re.search(r"\bnot\s+(?:harmful|toxic|hateful|misogynous)\b", lowered):
            return 0
        if re.search(r"\b(?:non-harmful|non-toxic|harmless|lighthearted)\b", lowered):
            return 0
        return None
    return 1 if hits[-1].lower() == "yes" else 0


def binary_token_ids(tokenizer) -> dict[str, list[int]]:
    def first_token(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    mapping: dict[str, list[int]] = {}
    for label in ("Yes", "No"):
        tids: set[int] = set()
        for variant in (label, f" {label}", label.lower(), f" {label.lower()}", label.upper(), f" {label.upper()}"):
            tid = first_token(variant)
            if tid is not None:
                tids.add(tid)
        mapping[label] = sorted(tids)
    return mapping


def binary_score_from_output(output, label_token_ids: dict[str, list[int]]) -> float | None:
    if not output or not output.outputs:
        return None
    gen = output.outputs[0]
    if not gen.logprobs:
        return None
    pos0 = gen.logprobs[0]
    fallback = -30.0
    p_yes = sum(math.exp(pos0[tid].logprob if tid in pos0 else fallback) for tid in label_token_ids["Yes"])
    p_no = sum(math.exp(pos0[tid].logprob if tid in pos0 else fallback) for tid in label_token_ids["No"])
    total = p_yes + p_no
    return p_yes / total if total > 0 else None


def entropy(p: float) -> float:
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rho_from_hbar(hbar: float) -> float:
    lo, hi = 0.5, 1.0 - 1e-12
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if entropy(mid) > hbar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def required_judge_ids(dataset: str, tag: str) -> set[str]:
    band_rows = [r for r in read_jsonl(RESULT_ROOT / "boundary" / dataset / "candidates_entropy_band.jsonl") if r.get("in_band") is True]
    if not band_rows:
        return set()
    if tag not in DEFAULT_ORDER:
        return {r["id"] for r in band_rows}
    tag_idx = DEFAULT_ORDER.index(tag)
    if tag_idx == 0:
        return {r["id"] for r in band_rows}

    hbar = float(band_rows[0].get("hbar", sum(float(r["entropy"]) for r in band_rows) / len(band_rows)))
    rho = rho_from_hbar(hbar)
    lam = math.log(rho / (1.0 - rho))
    previous = DEFAULT_ORDER[:tag_idx]
    judges = {prev: {r["id"]: r for r in read_jsonl(RESULT_ROOT / "judges" / dataset / f"test_{prev}.jsonl") if r.get("id")} for prev in previous}

    required: set[str] = set()
    for row in band_rows:
        ell = logit(float(row.get("posterior_hi", 0.5)))
        stopped = False
        for prev in previous:
            jr = judges[prev].get(row["id"], {})
            pred = jr.get("pred")
            if pred not in (0, 1):
                pred = parse_yes_no(jr.get("raw_response"))
            if pred in (0, 1):
                ell += (2 * int(pred) - 1) * lam
                if entropy(sigmoid(ell)) <= hbar:
                    stopped = True
                    break
        if not stopped:
            required.add(row["id"])
    return required


def macro_prf(y: list[int], yh: list[int]) -> tuple[float, float, float]:
    classes = sorted(set(y) | set(yh))
    if not classes:
        return 0.0, 0.0, 0.0
    fs, ps, rs = [], [], []
    for c in classes:
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return sum(fs) / len(fs), sum(ps) / len(ps), sum(rs) / len(rs)
