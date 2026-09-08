#!/usr/bin/env python3
"""Parameter-free semantic admission authorized by within-video T/A timing.

Text supplies hostile-proposition evidence; audio supplies no hate score and
only authorizes text according to its exact zero-lag rank among all circular
shifts.  The admission weight is the positive excess over the uniform-null
median mapped to [0,1].  No target labels, dataset thresholds, or gains enter.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.idea_discovery.project_role_orthogonal_transport import DATA_DIR, load, sigmoid


METHODS = ("oas_base_v1", "oas_text_full_v1", "oas_ta_orbit_v1", "oas_ta_reverse_v1")


def chunks(path: Path) -> dict[tuple[str, str], list[dict]]:
    out = defaultdict(list)
    for row in map(json.loads, path.open()):
        out[(row["dataset"], row["video_id"])].append(row)
    return out


def midrank(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=np.float64), method="average") / (len(values) + 1.0)


def resize(values: np.ndarray, n: int) -> np.ndarray:
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(values)), values)


def text_field(records: list[dict], duration: float, n: int) -> np.ndarray | None:
    if not records:
        return None
    null = float(np.median([x.get("config", {}).get("null_log_odds", -12.0) for x in records]))
    values = np.full(n, null, dtype=np.float64)
    times = (np.arange(n) + 0.5) * duration / n
    for row in records:
        mask = (times >= float(row["start"])) & (times < float(row["end"]))
        values[mask] = np.maximum(values[mask], float(row["log_odds"]))
    return midrank(values)


def audio_field(path: Path, n: int) -> np.ndarray | None:
    if not path.exists():
        return None
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        return None
    unit = value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
    change = np.empty(len(value), dtype=np.float64)
    change[1:] = 1.0 - np.sum(unit[1:] * unit[:-1], axis=1)
    change[0] = change[1]
    return resize(midrank(change), n)


def orbit_weight(text: np.ndarray, audio: np.ndarray) -> tuple[float, dict]:
    t = text - text.mean(); a = audio - audio.mean()
    t /= max(np.linalg.norm(t), 1e-12); a /= max(np.linalg.norm(a), 1e-12)
    scores = np.asarray([float(t @ np.roll(a, shift)) for shift in range(len(t))])
    rank = float(rankdata(scores, method="average")[0] / len(scores))
    weight = max(0.0, 2.0 * rank - 1.0)
    return weight, {"orbit_rank": rank, "weight": weight, "best_shift": int(np.argmax(scores))}


def admit(base: np.ndarray, text: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
        return base.copy()
    base_logit = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1.0))
    text_logit = np.log(np.clip(text, 1e-6, 1 - 1e-6) / np.clip(1 - text, 1e-6, 1.0))
    text_logit -= text_logit.mean()
    text_scale = np.median(np.abs(text_logit - np.median(text_logit)))
    base_scale = np.median(np.abs(base_logit - np.median(base_logit)))
    if text_scale <= 1e-12 or base_scale <= 1e-12:
        return base.copy()
    changed = base_logit + weight * text_logit * (base_scale / text_scale)
    target = float(base.mean())
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if float(sigmoid(changed + mid).mean()) < target:
            lo = mid
        else:
            hi = mid
    return sigmoid(changed + (lo + hi) / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    base, chunk_map = load(args.base, args.base_method), chunks(args.chunks)
    with args.out.open("w") as handle:
        for key, row in sorted(base.items()):
            b = np.asarray(row["score_curve"], dtype=np.float64); n = len(b)
            t = text_field(chunk_map.get(key, []), float(row["duration"]), n)
            a = audio_field(args.audio_root / "vggish_1s" / DATA_DIR[key[0]] / f"{key[1]}.npy", n)
            variants = {method: b.copy() for method in METHODS}; audits = {}
            if t is not None:
                variants["oas_text_full_v1"] = admit(b, t, 1.0)
            if t is not None and a is not None:
                factual_weight, factual_audit = orbit_weight(t, a)
                reverse_t = t[::-1]
                reverse_weight, reverse_audit = orbit_weight(reverse_t, a)
                variants["oas_ta_orbit_v1"] = admit(b, t, factual_weight)
                variants["oas_ta_reverse_v1"] = admit(b, reverse_t, reverse_weight)
                audits = {"factual": factual_audit, "reverse": reverse_audit}
            for method in METHODS:
                output = dict(row); output["method"] = method
                output["score_curve"] = variants[method].tolist()
                output["raw"] = {**output.get("raw", {}), "gt_access": False,
                    "module": "orbit_authorized_semantics_v1", "dataset_parameters": 0,
                    "tie_rule": "average_midrank", "video_mean_preserved": True,
                    "intervals_preserved": True, "orbit_audit": audits}
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
