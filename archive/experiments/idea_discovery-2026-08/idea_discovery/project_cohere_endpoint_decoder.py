#!/usr/bin/env python3
"""COHERE endpoint pilot: decode boundaries from visual jumps and text cuts.

The candidate bank is frozen to tight/current/broad endpoints.  Vision owns
inside/outside evidence discontinuity; timestamp language owns whether a
candidate boundary cuts a cohesive discourse edge.  Rank consensus is
noncompensatory and contains no learned or GT-tuned weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATA_DIR = {
    "HateMM": "hatemm",
    "HateClipSeg": "hateclipseg",
    "MHC": "mhclip_en",
    "MHC_zh": "mhclip_zh",
}


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    out = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") == method:
                out[(row["dataset"], row["video_id"])] = row
    return out


def interval(row: dict | None) -> tuple[float, float] | None:
    if not row or not row.get("intervals"):
        return None
    return tuple(map(float, row["intervals"][0][:2]))


def normalized(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)


def local_jump(field: np.ndarray, index: int, side: str) -> float:
    n = len(field)
    width = max(2, min(12, n // 20))
    before = field[max(0, index - width):index]
    after = field[index:min(n, index + width)]
    if not len(before) or not len(after):
        return -np.inf
    delta = float(np.mean(after) - np.mean(before))
    return delta if side == "left" else -delta


def language_cut(features: np.ndarray, second: float) -> float:
    """High when the adjacent timestamp embeddings are weakly cohesive."""
    if len(features) < 2:
        return -np.inf
    index = int(np.clip(round(second), 1, len(features) - 1))
    left, right = features[index - 1], features[index]
    # Missing-ASR zero vectors carry no authority, so leave the endpoint alone.
    if np.linalg.norm(left) < 1e-8 or np.linalg.norm(right) < 1e-8:
        return -np.inf
    return float(1.0 - np.dot(left, right))


def rank(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.zeros(len(values), dtype=float)
    if not np.any(finite):
        return result
    order = np.argsort(values[finite], kind="stable")
    positions = np.empty(np.sum(finite), dtype=float)
    positions[order] = np.arange(1, np.sum(finite) + 1) / np.sum(finite)
    result[finite] = positions
    return result


def select_endpoint(
    seconds: list[float], field: np.ndarray, duration: float,
    text_features: np.ndarray, side: str, current_index: int,
) -> tuple[float, dict]:
    indices = [int(np.clip(round(x / max(duration, 1e-9) * (len(field) - 1)), 0, len(field) - 1))
               for x in seconds]
    visual = [local_jump(field, index, side) for index in indices]
    textual = [language_cut(text_features, second) for second in seconds]
    visual_rank, text_rank = rank(visual), rank(textual)
    consensus = np.sqrt(visual_rank * text_rank)
    # No valid language edge means exact fallback to the current endpoint.
    if not np.any(np.isfinite(textual)):
        selected = current_index
    else:
        selected = int(np.argmax(consensus))
    audit = {
        "seconds": seconds,
        "visual_jump": visual,
        "language_cut": textual,
        "consensus": consensus.tolist(),
        "selected": selected,
    }
    return seconds[selected], audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--field-method", default="route_text_v1")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--tight-method", default="fact_less_t3al_dualgeo_shorter_v5")
    parser.add_argument("--broad-method", default="fact_less_t3al_dualgeo_union_v5")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    fields = load(args.fields, args.field_method)
    tight = load(args.bank, args.tight_method)
    broad = load(args.bank, args.broad_method)
    changed = 0
    eligible = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for key, row in sorted(fields.items()):
            current_i, tight_i, broad_i = interval(row), interval(tight.get(key)), interval(broad.get(key))
            result = dict(row)
            result["method"] = "cohere_geometry_endpoint_v1"
            audit = {"eligible": False}
            feature_path = (args.feature_root / "bert_sentence_1fps" /
                            DATA_DIR[key[0]] / f"{key[1]}.npy")
            if current_i and tight_i and broad_i and feature_path.exists():
                text_features = normalized(np.load(feature_path))
                duration = float(row["duration"])
                # Deduplicate candidates while retaining the current endpoint index.
                left = [tight_i[0], current_i[0], broad_i[0]]
                right = [tight_i[1], current_i[1], broad_i[1]]
                new_left, left_audit = select_endpoint(
                    left, np.asarray(row["score_curve"], float), duration,
                    text_features, "left", 1,
                )
                new_right, right_audit = select_endpoint(
                    right, np.asarray(row["score_curve"], float), duration,
                    text_features, "right", 1,
                )
                # Tight core containment is the only structural safety condition.
                if new_left <= tight_i[0] and new_right >= tight_i[1] and new_left < new_right:
                    result["intervals"] = [[new_left, new_right, 1.0]]
                    eligible += 1
                    changed += int((new_left, new_right) != current_i)
                audit = {"eligible": True, "left": left_audit, "right": right_audit}
            result["raw"] = {
                **result.get("raw", {}),
                "gt_access": False,
                "cohere_endpoint": audit,
                "endpoint_bank": ["tight", "current", "broad"],
                "endpoint_rule": "geometric_rank_consensus_visual_jump_language_cut",
            }
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(fields), "eligible": eligible, "changed": changed}, indent=2))


if __name__ == "__main__":
    main()
