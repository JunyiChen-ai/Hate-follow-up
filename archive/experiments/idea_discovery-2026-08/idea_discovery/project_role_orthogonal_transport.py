#!/usr/bin/env python3
"""ROUTE pilot: role-orthogonal multimodal temporal evidence transport.

This is a deterministic, training- and label-free mechanism test.  Visual
evidence remains in the frozen base field, transcript cohesion supplies the
first temporal correction, and audio is admitted only through the component
not linearly explained by the base, transcript cohesion, or a quadratic time
trend.  All transports preserve each video's mean probability exactly.
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
    "HateClipSeg_sealed": "HateClipSeg_sealed",
}
FEATURES = {"A": "vggish_1s", "T": "bert_sentence_1fps"}
METHODS = (
    "route_base_v1",
    "route_text_v1",
    "route_raw_at_v1",
    "route_text_orth_audio_v1",
    "route_concordant_at_v1",
    "route_concordant_shift_a_v1",
    "route_concordant_shift_t_v1",
    "route_concordant_shift_at_v1",
    "route_text_rotated_orth_audio_v1",
)


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    out = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") == method:
                out[(row["dataset"], row["video_id"])] = row
    return out


def first_interval(row: dict | None) -> tuple[float, float] | None:
    if not row or not row.get("intervals"):
        return None
    return tuple(map(float, row["intervals"][0][:2]))


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values))
    return (ranks + 0.5) / max(1, len(values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    scale = np.median(np.abs(values - np.median(values)))
    if not np.isfinite(scale) or scale < 1e-8:
        return np.zeros_like(values)
    return values / scale


def transport(base: np.ndarray, correction: np.ndarray, gain: float = 0.01) -> np.ndarray:
    """Bounded logit transport with exact mean preservation."""
    base = np.asarray(base, dtype=float)
    correction = robust_unit(correction)
    if not np.any(correction):
        return base.copy()
    logits = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1))
    logit_scale = np.median(np.abs(logits - np.median(logits))) + 1e-6
    changed = logits + gain * correction * logit_scale
    target = float(np.mean(base))
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if float(np.mean(sigmoid(changed + mid))) < target:
            lo = mid
        else:
            hi = mid
    return sigmoid(changed + (lo + hi) / 2.0)


def cohesion(features: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, float]:
    features = np.asarray(features, dtype=float)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    lo = max(0, min(len(features) - 1, lo))
    hi = max(lo + 1, min(len(features), hi))
    prototype = features[lo:hi].mean(axis=0)
    prototype /= max(np.linalg.norm(prototype), 1e-9)
    scores = features @ prototype
    return scores, float(np.std(scores))


def resize(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == n:
        return values
    return np.interp(np.linspace(0, len(values) - 1, n), np.arange(len(values)), values)


def residualize_audio(audio: np.ndarray, base: np.ndarray, text: np.ndarray) -> np.ndarray:
    """Remove audio components explained by admitted evidence and absolute time."""
    n = len(audio)
    time = np.linspace(-1.0, 1.0, n)
    design = np.stack(
        [np.ones(n), robust_unit(base), robust_unit(text), time, time * time], axis=1
    )
    target = robust_unit(audio)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return robust_unit(target - design @ coefficients)


def concordance(text: np.ndarray, audio_residual: np.ndarray) -> np.ndarray:
    text = robust_unit(text)
    audio_residual = robust_unit(audio_residual)
    agreed = np.sign(text) == np.sign(audio_residual)
    result = np.zeros_like(text)
    result[agreed] = np.sign(text[agreed]) * np.minimum(
        np.abs(text[agreed]), np.abs(audio_residual[agreed])
    )
    return robust_unit(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--tight-method", default="fact_less_t3al_dualgeo_shorter_v5")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    base_rows = load(args.base, args.base_method)
    tight_rows = load(args.bank, args.tight_method)
    coverage = {method: 0 for method in METHODS}
    mean_errors = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for key, row in sorted(base_rows.items()):
            base = np.asarray(row["score_curve"], dtype=float)
            n = len(base)
            core = first_interval(tight_rows.get(key))
            fields: dict[str, np.ndarray] = {}
            shifted: dict[str, np.ndarray] = {}
            reliabilities: dict[str, float] = {}

            if core is not None:
                lo = int(np.floor(core[0]))
                hi = max(lo + 1, int(np.ceil(core[1])))
                for modality, folder in FEATURES.items():
                    path = args.feature_root / folder / DATA_DIR[key[0]] / f"{key[1]}.npy"
                    if not path.exists():
                        continue
                    features = np.load(path)
                    aligned, reliability = cohesion(features, lo, hi)
                    if reliability <= 1e-5:
                        continue
                    shift = max(1, len(features) // 2)
                    shifted_scores, _ = cohesion(np.roll(features, shift, axis=0), lo, hi)
                    fields[modality] = resize(rank01(aligned), n)
                    shifted[modality] = resize(rank01(shifted_scores), n)
                    reliabilities[modality] = reliability

            variants = {method: base.copy() for method in METHODS}
            variants["route_base_v1"] = base.copy()
            if "T" in fields:
                variants["route_text_v1"] = transport(base, fields["T"])
            if "A" in fields and "T" in fields:
                text = robust_unit(fields["T"])
                audio = robust_unit(fields["A"])
                audio_orth = residualize_audio(audio, base, text)
                variants["route_raw_at_v1"] = transport(base, text + audio)
                variants["route_text_orth_audio_v1"] = transport(base, text + audio_orth)
                variants["route_concordant_at_v1"] = transport(base, concordance(text, audio_orth))

                shift_audio_orth = residualize_audio(shifted["A"], base, text)
                variants["route_concordant_shift_a_v1"] = transport(
                    base, concordance(text, shift_audio_orth)
                )
                shift_text = robust_unit(shifted["T"])
                audio_for_shift_text = residualize_audio(audio, base, shift_text)
                variants["route_concordant_shift_t_v1"] = transport(
                    base, concordance(shift_text, audio_for_shift_text)
                )
                shifted_audio_for_shift_text = residualize_audio(
                    shifted["A"], base, shift_text
                )
                variants["route_concordant_shift_at_v1"] = transport(
                    base, concordance(shift_text, shifted_audio_for_shift_text)
                )
                rotated = np.roll(audio_orth, max(1, n // 2))
                variants["route_text_rotated_orth_audio_v1"] = transport(
                    base, text + rotated
                )

            for method in METHODS:
                scores = variants[method]
                coverage[method] += int(not np.array_equal(scores, base))
                mean_errors.append(abs(float(np.mean(scores)) - float(np.mean(base))))
                result = dict(row)
                result["method"] = method
                result["score_curve"] = scores.tolist()
                # ROUTE is a dense reranker in this pilot; proposal geometry is frozen.
                result["raw"] = {
                    **result.get("raw", {}),
                    "gt_access": False,
                    "route_version": 1,
                    "gain": 0.01,
                    "available_modalities": sorted(fields),
                    "reliability": reliabilities,
                    "transport": method,
                    "video_mean_preserved": True,
                    "intervals_preserved": True,
                }
                output.write(json.dumps(result, separators=(",", ":")) + "\n")

    print(json.dumps({
        "n_videos": len(base_rows),
        "coverage_nonidentical": coverage,
        "max_video_mean_error": max(mean_errors, default=0.0),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
