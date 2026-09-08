from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Optional

from PIL import Image, ImageFile

from .common import DATASETS, LOG_ROOT, RESULT_ROOT, dataset_iter, done_ids, image_placeholder_prompt, load_split, safe_text

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
OUTPUT_SUBDIR = "mars_32b_awq"

MARS_TURN_1 = """You are a content analysis expert. Provide a detailed, objective description of this meme content.

Task: Describe what you observe in the meme image and text:
1. objective_visual_description: Describe what you see - objects, people, actions, settings, text, symbols, colors, expressions, gestures, etc.

Rules:
- Be comprehensive and detailed
- NO analysis, interpretations, or value judgments
- Simply report what you observe
- Return ONLY valid JSON with one key: ["objective_visual_description"]"""

MARS_TURN_2 = """You are a hateful content detection specialist. Analyze the meme image and text under the ASSUMPTION that it contains hateful or offensive content.

HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.
OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech.
IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech.

Task: Identify evidence supporting hateful(offensive) interpretation:
1. evidence: Describe specific evidence supporting the hateful nature of the content
2. reasoning: Explain why it is hateful content.
3. strength: Your confidence in this hate interpretation [0-1]

Focus on:
- Language targeting religion, ethnicity, race, gender, nationality, etc.
- Group-based dehumanizing language or symbols
- Identity-based discriminatory attacks (not personal insults)
- Incitement against identity groups

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""

MARS_TURN_3 = """You are a content analyst. Analyze this meme image and text under the ASSUMPTION that it does NOT constitute hate speech and offensive content.

HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.
OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech.
IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech.

Task: Identify evidence supporting non-hate(non-offensive) interpretation:
1. evidence: Describe specific evidence supporting the non-hateful nature of the content
2. reasoning: Explain why this content does not hateful
3. strength: Your confidence in this non-hate interpretation [0-1]

Consider:
- Is this a personal dispute rather than group targeting?
- Are insults directed at individuals rather than identity groups?
- Is there artistic, satirical, or educational context?
- Does the content lack group-based discriminatory language?

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""


def build_turn_4_prompt(turn_1_parsed, turn_2_parsed, turn_3_parsed) -> str:
    return f"""You are a senior content moderation specialist making the final determination.

OBJECTIVE DESCRIPTION: {json.dumps(turn_1_parsed or {}, ensure_ascii=False)}

HATE ARGUMENT: {json.dumps(turn_2_parsed or {}, ensure_ascii=False)}

NON-HATE ARGUMENT: {json.dumps(turn_3_parsed or {}, ensure_ascii=False)}

Task: Conduct critical analysis and make final determination:

1. evidence_comparison: Compare the quality and strength of both arguments, identifying which evidence is more compelling
2. contextual_analysis: Analyze how the content's context (platform, audience, intent) affects interpretation
3. harm_assessment: Evaluate potential real-world impact and harm to targeted groups
4. final_decision:
   - label: 0 (non-hate) or 1 (hate/offensive)
   - confidence: Overall confidence in decision [0-1]
   - key_factors: The decisive elements that determined your judgment
   - reasoning: 2-3 sentences explaining your decision

Rules:
- Weigh evidence objectively, not just confidence scores
- Consider both explicit and subtle indicators
- Prioritize potential for real-world harm
- Base decision on strongest evidence, not balanced arguments

Return ONLY valid JSON with key "final_decision" containing the above structure"""


def append_meme_text(base_prompt: str, text: str) -> str:
    if (text or "").strip():
        return base_prompt + f"\n\nMeme text: {text}"
    return base_prompt + "\n\nMeme text: No transcribed text available."


def clean_json_response(response_text: str, default_key: Optional[str] = None):
    cleaned = (response_text or "").strip()
    if not cleaned:
        return ({default_key: ""}, True) if default_key else (response_text, False)
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1]), True
                    except json.JSONDecodeError:
                        break
    return ({default_key: cleaned}, True) if default_key else (response_text, False)


def extract_final_label(parsed, raw_text: str) -> int:
    if isinstance(parsed, dict):
        fd = parsed.get("final_decision") if isinstance(parsed.get("final_decision"), dict) else parsed
        try:
            label = int(fd.get("label"))
            if label in (0, 1):
                return label
        except Exception:
            pass
    m = re.search(r'["\']?label["\']?\s*[:=]\s*["\']?([01])', raw_text or "", re.I)
    if m:
        return int(m.group(1))
    tail = (raw_text or "").lower()[-500:]
    if "harmless" in tail or "non-hate" in tail or "not hateful" in tail:
        return 0
    if "harmful" in tail or "hateful" in tail:
        return 1
    return -1


def mars_one(row: dict, image, llm, processor, sampling_params) -> dict:
    text = safe_text(row)
    turns = {}
    for key, prompt, default_key in (
        ("turn_1", append_meme_text(MARS_TURN_1, text), "objective_visual_description"),
        ("turn_2", append_meme_text(MARS_TURN_2, text), "evidence"),
        ("turn_3", append_meme_text(MARS_TURN_3, text), "evidence"),
    ):
        full_prompt = image_placeholder_prompt(processor, prompt)
        out = llm.generate({"prompt": full_prompt, "multi_modal_data": {"image": [image]}}, sampling_params=sampling_params)
        raw = out[0].outputs[0].text
        parsed, ok = clean_json_response(raw, default_key=default_key)
        turns[key] = {"raw": raw, "parsed": parsed if ok else None, "parse_ok": ok}
    p4 = append_meme_text(build_turn_4_prompt(turns["turn_1"]["parsed"], turns["turn_2"]["parsed"], turns["turn_3"]["parsed"]), text)
    out4 = llm.generate({"prompt": image_placeholder_prompt(processor, p4), "multi_modal_data": {"image": [image]}}, sampling_params=sampling_params)
    raw4 = out4[0].outputs[0].text
    parsed4, ok4 = clean_json_response(raw4)
    pred = extract_final_label(parsed4 if ok4 else None, raw4)
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "label": int(row["label"]),
        "pred": int(pred),
        "raw_response": raw4,
        "model": DEFAULT_MODEL,
        "prompt_version": "mars_meme_v1",
        "turn_1_raw": turns["turn_1"]["raw"],
        "turn_2_raw": turns["turn_2"]["raw"],
        "turn_3_raw": turns["turn_3"]["raw"],
        "turn_4_raw": raw4,
        "turn_4_parsed": parsed4 if ok4 else None,
        "parse_ok": {"turn_1": turns["turn_1"]["parse_ok"], "turn_2": turns["turn_2"]["parse_ok"], "turn_3": turns["turn_3"]["parse_ok"], "turn_4": ok4},
    }


def score_dataset(args, dataset: str, llm, processor, sampling_params) -> None:
    rows = load_split(dataset, "test")
    if args.limit:
        rows = rows[: args.limit]
    out_path = RESULT_ROOT / OUTPUT_SUBDIR / dataset / "test_mars.jsonl"
    done = done_ids(out_path)
    remaining = [r for r in rows if r["id"] not in done]
    logging.info("[%s] input=%d done=%d remaining=%d out=%s", dataset, len(rows), len(done), len(remaining), out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(remaining, 1):
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                rec = mars_one(row, image, llm, processor, sampling_params)
            except Exception as exc:
                logging.error("[%s] %s failed: %s", dataset, row["id"], str(exc)[:300])
                rec = {"id": row["id"], "dataset": dataset, "label": int(row["label"]), "pred": -1, "raw_response": "", "model": args.model, "error": str(exc)[:500]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if i == 1 or i % 5 == 0:
                logging.info("[%s] %d/%d %.3f sample/s pred=%s", dataset, i, len(remaining), i / max(time.time() - t0, 1e-6), rec.get("pred"))


def main() -> None:
    parser = argparse.ArgumentParser(description="MARS 32B-AWQ harmful meme adaptation")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_ROOT / "mars_32b_awq.log"), logging.StreamHandler()])
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.88, max_model_len=65536, limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": 32768}, enforce_eager=True, seed=args.seed)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096, seed=args.seed)
    for ds in dataset_iter(args.dataset):
        score_dataset(args, ds, llm, processor, sampling_params)


if __name__ == "__main__":
    main()

