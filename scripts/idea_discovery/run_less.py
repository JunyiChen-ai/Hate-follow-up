#!/usr/bin/env python3
"""Cross-fitted Label-free Evidence State-Space localization (LESS-3V).

No annotation or metric enters this program.  Three independently oriented
views are combined by a two-state categorical latent label model.  Parameters
are fitted on one deterministic hash fold and applied to the other; a sticky
state decoder then converts frame posteriors into events.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.audit_less_views import (
    hash_fold, imagebind_text_embeddings, load_predictions, resize,
    ternary_text, text_curve)
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


VIEW_NAMES = ("visual", "language", "audio")
CATEGORIES = (-1, 0, 1)


def collect(args) -> dict[tuple[str, str], dict]:
    manifest = load_predictions(args.manifest)
    visual_paths = {"a08": args.a08, "a10": args.a10, "a12": args.a12}
    if args.visual_mode == "a10":
        # A single-detector run must not silently inherit an availability gate
        # from unused detector artifacts.
        visual_sources = [load_predictions(visual_paths["a10"])]
    else:
        visual_sources = [load_predictions(path) for path in
                          (visual_paths["a08"], visual_paths["a10"], visual_paths["a12"])]
    chunks = defaultdict(list)
    for row in map(json.loads, args.text.open()):
        chunks[(row["dataset"], row["video_id"])].append(row)
    text_embeddings = imagebind_text_embeddings()
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    keys = sorted(set(manifest).intersection(*(set(source) for source in visual_sources)))
    output = {}
    for key in keys:
        row = manifest[key]
        length = max(1, int(np.ceil(float(row["duration"]) * 4)))
        bits = [resize(np.asarray(source[key]["score_curve"]), length) > 0
                for source in visual_sources]
        if args.visual_mode == "majority":
            visual = np.where(np.stack(bits).sum(0) >= 2, 1, -1).astype(np.int8)
        elif args.visual_mode == "a10":
            visual = np.where(bits[0], 1, -1).astype(np.int8)
        elif args.visual_mode == "unanimous":
            votes = np.stack(bits).sum(0)
            visual = np.where(votes == 3, 1, np.where(votes == 0, -1, 0)).astype(np.int8)
        else:
            raise ValueError(args.visual_mode)
        language = ternary_text(text_curve(chunks.get(key, []), length))
        if args.text_shift == "half" and length > 1:
            language = np.roll(language, length // 2)
        path = args.audio_dir / key[0] / f"{key[1]}.npy"
        if not path.exists():
            continue
        audio_embeddings = np.load(path).astype(np.float32)
        audio_embeddings /= np.maximum(np.linalg.norm(audio_embeddings, axis=1, keepdims=True), 1e-12)
        similarities = audio_embeddings @ text_embeddings.T
        margin = similarities[:, 1] - similarities[:, 0]
        dense_audio = resize(margin, length)
        if args.audio_mode == "absolute":
            audio = np.where(dense_audio >= 0, 1, -1).astype(np.int8)
        elif args.audio_mode == "within_video":
            # Separate temporal residual from the between-video acoustic level.
            # The median is label-free and equivariant to source score scaling.
            audio = np.where(dense_audio >= np.median(dense_audio), 1, -1).astype(np.int8)
        else:
            raise ValueError(args.audio_mode)
        output[key] = {"duration": float(row["duration"]), "fold": hash_fold(key),
                       "evidence": np.stack([visual, language, audio], axis=1)}
    return output


def initialize_posterior(evidence: np.ndarray) -> np.ndarray:
    # Each view has one vote; abstention has zero vote.  The small scale avoids
    # hard pseudo-labels while fixing the semantic orientation of the mixture.
    return 1.0 / (1.0 + np.exp(-0.8 * evidence.sum(axis=1)))


def fit_label_model(sequences: list[np.ndarray], view_indices=(0, 1, 2),
                    iterations: int = 50) -> dict:
    """Video-balanced EM for a two-state categorical Dawid-Skene model."""
    selected = [sequence[:, view_indices] for sequence in sequences]
    posteriors = [initialize_posterior(sequence) for sequence in selected]
    nviews = len(view_indices)
    theta = np.full((nviews, 2, 3), 1 / 3, dtype=float)
    prior = 0.5
    for _ in range(iterations):
        # Equal total mass per video prevents long clips defining authority.
        class_mass = np.ones(2, dtype=float)
        counts = np.ones((nviews, 2, 3), dtype=float)  # Dirichlet(1)
        for sequence, posterior in zip(selected, posteriors):
            weight = 1.0 / len(sequence)
            class_mass += weight * np.array([(1 - posterior).sum(), posterior.sum()])
            for view in range(nviews):
                for category_index, category in enumerate(CATEGORIES):
                    mask = sequence[:, view] == category
                    counts[view, 0, category_index] += weight * (1 - posterior[mask]).sum()
                    counts[view, 1, category_index] += weight * posterior[mask].sum()
        prior = float(class_mass[1] / class_mass.sum())
        theta = counts / counts.sum(axis=2, keepdims=True)
        # Resolve the unavoidable label permutation using the predeclared
        # semantic polarity shared by the three zero-shot sources.
        orientation = np.mean(theta[:, 1, 2] - theta[:, 0, 2])
        if orientation < 0:
            prior = 1 - prior
            theta = theta[:, ::-1, :]
        new_posteriors = [infer_static(sequence, prior, theta) for sequence in selected]
        change = np.mean([np.mean(np.abs(a - b))
                          for a, b in zip(posteriors, new_posteriors)])
        posteriors = new_posteriors
        if change < 1e-7:
            break
    reliability = []
    for view in range(nviews):
        reliability.append(float(0.5 * (theta[view, 1, 2] + theta[view, 0, 0])))
    return {"prior": prior, "theta": theta, "reliability": reliability,
            "view_indices": tuple(view_indices)}


def infer_static(evidence: np.ndarray, prior: float, theta: np.ndarray,
                 view_weights: np.ndarray | None = None) -> np.ndarray:
    logp = np.empty((len(evidence), 2), dtype=float)
    logp[:, 0] = math.log(max(1 - prior, 1e-9))
    logp[:, 1] = math.log(max(prior, 1e-9))
    if view_weights is None:
        view_weights = np.ones(evidence.shape[1], dtype=float)
    for view in range(evidence.shape[1]):
        indices = evidence[:, view] + 1
        logp[:, 0] += view_weights[view] * np.log(
            np.maximum(theta[view, 0, indices], 1e-9))
        logp[:, 1] += view_weights[view] * np.log(
            np.maximum(theta[view, 1, indices], 1e-9))
    difference = np.clip(logp[:, 1] - logp[:, 0], -40, 40)
    return 1.0 / (1.0 + np.exp(-difference))


def synchrony_authority(evidence: np.ndarray) -> float:
    """Permutation-calibrated authority of timestamped language evidence.

    Zero lag is ranked against seven circular temporal permutations while all
    per-view marginals remain fixed.  Dividing its empirical rank by the null
    expectation (1/2) yields a label-free likelihood exponent in [0.25, 2].
    """
    if len(evidence) < 8:
        return 1.0
    language = evidence[:, 1].astype(float)
    reference = evidence[:, [0, 2]].mean(axis=1)
    aligned = float(np.mean(language * reference))
    null = []
    for numerator in range(1, 8):
        shift = max(1, round(len(language) * numerator / 8))
        null.append(float(np.mean(np.roll(language, shift) * reference)))
    rank = (1 + sum(value <= aligned for value in null)) / 8
    return float(2 * rank)


def transition_from_soft(sequences: list[np.ndarray], model: dict) -> np.ndarray:
    counts = np.ones((2, 2), dtype=float)  # Beta smoothing, no duration labels
    for evidence in sequences:
        posterior = infer_static(evidence[:, model["view_indices"]],
                                 model["prior"], model["theta"])
        weight = 1.0 / max(1, len(evidence) - 1)
        previous, current = posterior[:-1], posterior[1:]
        counts[0, 0] += weight * ((1 - previous) * (1 - current)).sum()
        counts[0, 1] += weight * ((1 - previous) * current).sum()
        counts[1, 0] += weight * (previous * (1 - current)).sum()
        counts[1, 1] += weight * (previous * current).sum()
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi(posterior: np.ndarray, transition: np.ndarray, prior: float) -> np.ndarray:
    emission = np.stack([1 - posterior, posterior], axis=1)
    scores = np.full((len(posterior), 2), -np.inf)
    back = np.zeros((len(posterior), 2), dtype=np.int8)
    scores[0] = np.log(np.maximum([1 - prior, prior], 1e-9)) + np.log(
        np.maximum(emission[0], 1e-9))
    log_transition = np.log(np.maximum(transition, 1e-9))
    for time in range(1, len(posterior)):
        candidates = scores[time - 1][:, None] + log_transition
        back[time] = np.argmax(candidates, axis=0)
        scores[time] = candidates[back[time], np.arange(2)] + np.log(
            np.maximum(emission[time], 1e-9))
    path = np.zeros(len(posterior), dtype=np.int8)
    path[-1] = int(np.argmax(scores[-1]))
    for time in range(len(posterior) - 1, 0, -1):
        path[time - 1] = back[time, path[time]]
    return path


def path_intervals(path: np.ndarray, posterior: np.ndarray, duration: float) -> list[Interval]:
    intervals = []
    start = None
    for index, active in enumerate(np.r_[path, 0]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index
            intervals.append(Interval(start / 4, min(duration, end / 4),
                                      float(posterior[start:end].mean())))
            start = None
    return intervals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path,
                        help="optional frozen unlabeled calibration cohort")
    parser.add_argument("--a08", type=Path)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--a12", type=Path)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--visual-mode", choices=("majority", "a10", "unanimous"),
                        default="majority")
    parser.add_argument("--fit-scope", choices=("global", "dataset"), default="global")
    parser.add_argument("--text-shift", choices=("aligned", "half"), default="aligned")
    parser.add_argument("--audio-mode", choices=("absolute", "within_video"),
                        default="absolute")
    parser.add_argument("--synchrony-calibration", choices=("none", "permutation"),
                        default="none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.visual_mode != "a10" and (args.a08 is None or args.a12 is None):
        parser.error("--a08 and --a12 are required unless --visual-mode=a10")
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    data = collect(args)
    if args.fit_manifest:
        fit_args = copy.copy(args)
        fit_args.manifest = args.fit_manifest
        fit_data = collect(fit_args)
    else:
        fit_data = data
    tag = f"{args.visual_mode}_{args.fit_scope}_{args.text_shift}_{args.audio_mode}"
    if args.synchrony_calibration != "none":
        tag += f"_sync-{args.synchrony_calibration}"
    configurations = {
        f"less_3v_full_{tag}_v2": (0, 1, 2),
        f"less_3v_no_audio_{tag}_v2": (0, 1),
        f"less_3v_no_language_{tag}_v2": (0, 2),
        f"less_3v_no_visual_{tag}_v2": (1, 2),
    }
    audit = {"gt_access": False, "n": len(data), "fit_n": len(fit_data),
             "fit_manifest": str(args.fit_manifest or args.manifest), "models": {}}
    for method, indices in configurations.items():
        fold_models = {}
        scopes = sorted({key[0] for key in data}) if args.fit_scope == "dataset" else ["GLOBAL"]
        for scope in scopes:
            for target_fold in (0, 1):
                training = [row["evidence"] for key, row in fit_data.items()
                            if row["fold"] != target_fold and
                            (scope == "GLOBAL" or key[0] == scope)]
                model = fit_label_model(training, indices)
                transition = transition_from_soft(training, model)
                fold_models[(scope, target_fold)] = (model, transition)
                audit["models"][f"{method}:{scope}:target_fold_{target_fold}"] = {
                    "prior": model["prior"], "reliability": dict(zip(
                        [VIEW_NAMES[i] for i in indices], model["reliability"])),
                    "transition": transition.tolist()}
        for key, row in data.items():
            scope = key[0] if args.fit_scope == "dataset" else "GLOBAL"
            model, transition = fold_models[(scope, row["fold"])]
            evidence = row["evidence"][:, indices]
            weights = np.ones(len(indices), dtype=float)
            sync_authority = 1.0
            if args.synchrony_calibration == "permutation" and 1 in indices:
                sync_authority = synchrony_authority(row["evidence"])
                weights[indices.index(1)] = sync_authority
            posterior = infer_static(evidence, model["prior"], model["theta"], weights)
            path = viterbi(posterior, transition, model["prior"])
            intervals = path_intervals(path, posterior, row["duration"])
            append_jsonl(args.out, Prediction(
                method, key[0], key[1], row["duration"],
                score_curve=posterior.tolist(), intervals=intervals, calls=0,
                modality_evidence={"fold": row["fold"],
                                   "views": [VIEW_NAMES[i] for i in indices],
                                   "reliability": dict(zip(
                                       [VIEW_NAMES[i] for i in indices], model["reliability"])),
                                   "transition": transition.tolist()},
                raw={"gt_access": False, "fit": "opposite_hash_fold",
                     "fit_scope": args.fit_scope, "visual_mode": args.visual_mode,
                     "text_shift": args.text_shift,
                     "audio_mode": args.audio_mode,
                     "synchrony_calibration": args.synchrony_calibration,
                     "language_synchrony_authority": sync_authority,
                     "video_balanced_em": True,
                     "orientation": "predeclared_support_polarity",
                     "decoder": "sticky_two_state_viterbi"}))
    audit_path = args.out.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
