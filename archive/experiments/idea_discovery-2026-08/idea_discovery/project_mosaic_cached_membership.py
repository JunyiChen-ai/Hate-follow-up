#!/usr/bin/env python3
"""Cached MOSAIC endpoint-membership pilot and its fused-field kill control.

The program is label-free.  It uses existing cross-fitted LESS-3V coalition
posteriors to recover three scale-comparable within-video modality residuals,
then ranks local endpoint contrasts against fixed circular temporal orbits.
It is a cache feasibility test, not the final chunk-identity-aware method.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


BANK_METHODS = {
    "tight": "fact_less_t3al_dualgeo_shorter_v5",
    "midpoint": "fact_less_t3al_dualgeo_midpoint_v5",
    "broad": "fact_less_t3al_dualgeo_union_v5",
}
COALITION_METHODS = {
    "full": "less_3v_full_majority_global_aligned_within_video_v2",
    "va": "less_3v_no_language_majority_global_aligned_within_video_v2",
    "vl": "less_3v_no_audio_majority_global_aligned_within_video_v2",
    "la": "less_3v_no_visual_majority_global_aligned_within_video_v2",
}
MODALITIES = ("visual", "audio", "language")
ORBIT_FRACTIONS = tuple(range(1, 8))


def load(path: Path, methods: dict[str, str]) -> dict[str, dict[tuple[str, str], dict]]:
    output = {name: {} for name in methods}
    reverse = {method: name for name, method in methods.items()}
    for row in map(json.loads, path.open()):
        name = reverse.get(row.get("method"))
        if name is not None:
            output[name][(row["dataset"], row["video_id"])] = row
    return output


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def resize(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values
    index = np.minimum((np.arange(length) * len(values) / length).astype(int), len(values) - 1)
    return values[index]


def centered(values: np.ndarray) -> np.ndarray:
    return values - float(np.mean(values))


def modality_fields(rows: dict[str, dict], length: int) -> dict[str, np.ndarray]:
    # Pairwise coalition inversion under the additive cache proxy.  Centering
    # removes independently fitted coalition intercepts before any comparison.
    va = centered(resize(logit(rows["va"]["score_curve"]), length))
    vl = centered(resize(logit(rows["vl"]["score_curve"]), length))
    la = centered(resize(logit(rows["la"]["score_curve"]), length))
    return {
        "visual": centered(0.5 * (va + vl - la)),
        "audio": centered(0.5 * (va + la - vl)),
        "language": centered(0.5 * (vl + la - va)),
    }


def interval(row: dict) -> tuple[float, float] | None:
    if not row.get("intervals"):
        return None
    return float(row["intervals"][0][0]), float(row["intervals"][0][1])


def endpoint_levels(rows: dict[str, dict], side: int) -> list[tuple[str, float]]:
    values = {name: interval(row)[side] for name, row in rows.items() if interval(row) is not None}
    if len(values) != 3:
        return []
    return [
        ("tight", values["tight"]),
        ("tight_mid_half", 0.5 * (values["tight"] + values["midpoint"])),
        ("midpoint", values["midpoint"]),
        ("mid_broad_half", 0.5 * (values["midpoint"] + values["broad"])),
        ("broad", values["broad"]),
    ]


def local_gap(field: np.ndarray, duration: float, endpoint: float, side: int,
              width: float) -> float:
    rate = len(field) / max(duration, 1e-9)
    center = int(round(endpoint * rate))
    cells = max(1, int(round(width * rate)))
    if side == 0:
        inside = field[max(0, center):min(len(field), center + cells)]
        outside = field[max(0, center - cells):max(0, center)]
    else:
        inside = field[max(0, center - cells):max(0, center)]
        outside = field[max(0, center):min(len(field), center + cells)]
    if not len(inside) or not len(outside):
        return float("nan")
    return float(np.mean(inside) - np.mean(outside))


def orbit_rank(field: np.ndarray, duration: float, endpoint: float, side: int,
               width: float) -> tuple[float, float]:
    aligned = local_gap(field, duration, endpoint, side, width)
    if not np.isfinite(aligned) or len(field) < 8:
        return float("nan"), aligned
    nulls = []
    for numerator in ORBIT_FRACTIONS:
        shifted = np.roll(field, max(1, round(len(field) * numerator / 8)))
        value = local_gap(shifted, duration, endpoint, side, width)
        if np.isfinite(value):
            nulls.append(value)
    if not nulls:
        return float("nan"), aligned
    return (1 + sum(value < aligned for value in nulls)) / (1 + len(nulls)), aligned


def choose_endpoint(fields: dict[str, np.ndarray], duration: float,
                    levels: list[tuple[str, float]], side: int,
                    required: tuple[str, ...]) -> tuple[str, float, dict]:
    midpoint = next((name, value) for name, value in levels if name == "midpoint")
    unique = sorted(set(value for _, value in levels))
    width = float(np.median(np.diff(unique))) if len(unique) > 1 else duration / 20
    width = max(0.25, width)
    candidates = []
    audit = {}
    for name, value in levels:
        ranks, gaps = {}, {}
        for modality in required:
            ranks[modality], gaps[modality] = orbit_rank(
                fields[modality], duration, value, side, width)
        valid = all(np.isfinite(ranks[m]) and ranks[m] > 0.5 for m in required)
        strength = min((ranks[m] for m in required), default=float("nan"))
        audit[name] = {"value": value, "ranks": ranks, "gaps": gaps,
                       "valid": bool(valid), "strength": strength}
        if valid:
            # Prefer strongest weakest-view evidence, then the conservative
            # midpoint in an exact tie. No extent preference is introduced.
            candidates.append((strength, name == "midpoint", name, value))
    if not candidates:
        return midpoint[0], midpoint[1], audit
    _, _, name, value = max(candidates)
    return name, value, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--coalitions", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--field-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    bank = load(args.bank, BANK_METHODS)
    coalitions = load(args.coalitions, COALITION_METHODS)
    field_rows = load(args.field, {"field": args.field_method})["field"]
    keys = set.intersection(*(set(rows) for rows in bank.values()),
                            *(set(rows) for rows in coalitions.values()), set(field_rows))
    counts = Counter()
    with args.out.open("w") as handle:
        for key in sorted(keys):
            bank_rows = {name: rows[key] for name, rows in bank.items()}
            base = bank_rows["midpoint"]
            midpoint = interval(base)
            field_row = field_rows[key]
            length = len(field_row["score_curve"])
            duration = float(base["duration"])
            if midpoint is None:
                for method in ("mosaic_cached_vta_v1", "mosaic_fused_kill_v1"):
                    row = dict(field_row)
                    row["method"] = method
                    row["intervals"] = []
                    row["raw"] = {**row.get("raw", {}), "gt_access": False,
                                  "mosaic_cached_proxy": True, "fallback": "empty_midpoint"}
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts["empty"] += 1
                continue
            levels = [endpoint_levels(bank_rows, side) for side in (0, 1)]
            coalition_rows = {name: rows[key] for name, rows in coalitions.items()}
            split_fields = modality_fields(coalition_rows, length)
            fused = {"fused": centered(resize(logit(field_row["score_curve"]), length))}
            for method, fields, required in (
                    ("mosaic_cached_vta_v1", split_fields, MODALITIES),
                    ("mosaic_fused_kill_v1", fused, ("fused",))):
                left_name, left, left_audit = choose_endpoint(
                    fields, duration, levels[0], 0, required)
                right_name, right, right_audit = choose_endpoint(
                    fields, duration, levels[1], 1, required)
                if left >= right:
                    left, right = midpoint
                    left_name = right_name = "midpoint_invalid_fallback"
                changed = abs(left - midpoint[0]) > 1e-8 or abs(right - midpoint[1]) > 1e-8
                counts[f"{method}:changed"] += int(changed)
                row = dict(field_row)
                row["method"] = method
                row["intervals"] = [[left, right, 1.0]]
                row["raw"] = {**row.get("raw", {}), "gt_access": False,
                              "mosaic_cached_proxy": True,
                              "endpoint_levels": 5, "orbit_fractions": list(ORBIT_FRACTIONS),
                              "required_modalities": list(required),
                              "left_choice": left_name, "right_choice": right_name,
                              "left_audit": left_audit, "right_audit": right_audit}
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(keys), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
