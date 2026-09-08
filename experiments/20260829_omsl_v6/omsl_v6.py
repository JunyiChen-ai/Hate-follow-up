#!/usr/bin/env python3
"""Parameter-free orthogonal propensity/residual hateful-video localizer.

The global frozen MLLM logit is a video intercept.  A separate local scorer
forms all modality coalitions from visual, timestamped-language, and audio
streams; temporal Mobius differences identify contributions relative to the
same fixed scorer.  Calibrated interaction evidence and unimodal language/audio
evidence refine the visual detector's tied plateaus, after which the resulting
rank residual is projected off the constant vector.  Consequently the global
judgment cannot alter temporal ordering.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omsl_io import imagebind_text_embeddings, load_predictions, resize, text_curve  # noqa: E402

CODE_PATH = "experiments/20260829_omsl_v6/omsl_v6.py"
CODE_REVISION = ("2026-09-09 migration of scripts/idea_discovery/"
                 "project_orthogonal_mobius_localizer.py (OMSL-v6, 2026-08-29); "
                 "permutation null seeded with PERMUTATION_SEED instead of a "
                 "data-derived digest (hash ban, CLAUDE.md)")
PERMUTATION_SEED = 0
PERMUTATIONS = 31  # fixed Monte-Carlo resolution, never label selected


PLAYERS = (0, 1, 2)
NAMES = ("visual", "language", "audio")
SUBSETS = [
    frozenset(s)
    for size in range(4)
    for s in itertools.combinations(PLAYERS, size)
]


def centered_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if len(values) <= 1 or np.ptp(values) <= 1e-12:
        return np.zeros_like(values)
    return (rankdata(values, method="average") - 0.5) / len(values) - 0.5


def robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float) - float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values)))
    if scale <= 1e-12:
        scale = float(np.sqrt(np.mean(values * values)))
    return values / scale if scale > 1e-12 else np.zeros_like(values)


def existential_worth(streams: np.ndarray, subset: frozenset[int]) -> np.ndarray:
    """Symmetric frozen coalition scorer; singleton worth is the stream itself."""
    if not subset:
        return np.zeros(len(streams), dtype=float)
    selected = streams[:, sorted(subset)]
    if selected.shape[1] == 1:
        return selected[:, 0]
    maximum = selected.max(axis=1)
    return maximum + np.log(np.mean(np.exp(selected - maximum[:, None]), axis=1))


def logmeanexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.mean(np.exp(values - maximum))))


def decorrelation_block_length(streams: np.ndarray) -> tuple[int, str]:
    """Return the first joint ACF crossing, or an explicit full-video fallback."""
    n = len(streams)
    if n < 4:
        return n, "too_short_full_video"
    prepared = []
    for column in streams.T:
        x = column - column.mean()
        denominator = float(x @ x)
        if denominator <= 1e-12:
            continue
        prepared.append((x, denominator))
    if not prepared:
        return n, "all_streams_constant_full_video"
    for lag in range(1, n):
        common = True
        for x, denominator in prepared:
            acf = float(x[:-lag] @ x[lag:] / denominator)
            if acf > np.exp(-1):
                common = False
                break
        if common:
            return lag, "first_joint_acf_crossing"
    return n, "no_crossing_full_video"


def permute_full_blocks(values: np.ndarray, block: int,
                        generator: np.random.Generator) -> np.ndarray:
    """Reorder complete contiguous blocks; keep the terminal remainder fixed."""
    n_full = len(values) // block
    if n_full <= 1:
        return values.copy()
    prefix = values[:n_full * block].reshape(n_full, block)
    order = generator.permutation(n_full)
    return np.concatenate((prefix[order].reshape(-1), values[n_full * block:]))


def calibrated_mobius_field(
        streams: np.ndarray, observed: dict) -> tuple[np.ndarray, int, str]:
    """Joint nonwrapping block-permutation max calibration of interactions."""
    interaction_sets = [s for s in observed if len(s) >= 2]
    block, block_rule = decorrelation_block_length(streams)
    generator = np.random.default_rng(PERMUTATION_SEED)
    null_maxima = []
    for _ in range(PERMUTATIONS):
        permuted = streams.copy()
        permuted[:, 1] = permute_full_blocks(streams[:, 1], block, generator)
        permuted[:, 2] = permute_full_blocks(streams[:, 2], block, generator)
        values = {subset: existential_worth(permuted, subset) for subset in SUBSETS}
        null_maxima.append(max(
            float(np.max(np.abs(mobius(values, subset))))
            for subset in interaction_sets
        ))
    null_maxima = np.asarray(null_maxima)
    field = np.zeros(len(streams), dtype=float)
    for subset in interaction_sets:
        effect = observed[subset]
        # Conventional inclusive upper-tail Monte-Carlo p-value against the
        # joint maximum null; signed confidence is a bounded evidence display,
        # not a probability of interaction.
        p_value = (
            1 + (null_maxima[:, None] >= np.abs(effect)[None, :]).sum(axis=0)
        ) / (len(null_maxima) + 1)
        field += np.sign(effect) * (1 - p_value)
    return field, block, block_rule


def mobius(values: dict[frozenset[int], np.ndarray], subset: frozenset[int]) -> np.ndarray:
    output = np.zeros_like(next(iter(values.values())))
    members = sorted(subset)
    for size in range(len(members) + 1):
        for inner in itertools.combinations(members, size):
            output += (-1) ** (len(members) - len(inner)) * values[frozenset(inner)]
    return output


def timeline_from_chunks(rows: list[dict], length: int) -> np.ndarray:
    # audit_less_views.text_curve consumes log_odds and timestamps and performs
    # deterministic overlap averaging on the 4-fps grid.
    return text_curve(rows, length)


def load_reservoir(arguments: list[list[str]]) -> dict[tuple[str, str], float]:
    output = {}
    for dataset, path in arguments:
        for row in map(json.loads, Path(path).open()):
            z = row.get("z")
            if isinstance(z, (int, float)) and np.isfinite(z):
                output[(dataset, str(row["video_id"]))] = float(z)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--scores", action="append", nargs=2,
                        metavar=("DATASET", "JSONL"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    # Ablation switches (2026-09-09). Defaults reproduce OMSL-v6 exactly.
    parser.add_argument("--drop", nargs="*", default=[], choices=["language", "audio"],
                        help="zero a stream before the coalition game")
    parser.add_argument("--fusion", default="mobius_calibrated",
                        choices=["mobius_calibrated", "mobius_raw", "mains_only", "none"],
                        help="coalition field: calibrated interactions (v6), raw Mobius "
                             "interactions, language+audio mains only, or no field")
    parser.add_argument("--order", default="lexicographic", choices=["lexicographic", "sum"],
                        help="visual-primary lexicographic order (v6) or equal-weight sum")
    parser.add_argument("--intercept", default="both", choices=["both", "mllm", "occupancy", "none"],
                        help="video intercept: logmeanexp(occupancy, z) (v6), z only, occupancy only, 0")
    parser.add_argument("--permutations", type=int, default=31)
    parser.add_argument("--tag", default="", help="suffix appended to the method name")
    args = parser.parse_args()
    global PERMUTATIONS
    PERMUTATIONS = args.permutations
    method_name = "orthogonal_mobius_semantic_localizer_v6" + (f"__{args.tag}" if args.tag else "")
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    manifest = load_predictions(args.manifest)
    visual = load_predictions(args.visual)
    chunks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in map(json.loads, args.text.open()):
        chunks[(str(row["dataset"]), str(row["video_id"]))].append(row)
    reservoir = load_reservoir(args.scores)
    text_embeddings = imagebind_text_embeddings().astype(np.float32)
    text_embeddings /= np.maximum(np.linalg.norm(text_embeddings, axis=1, keepdims=True), 1e-12)

    keys = sorted(set(manifest) & set(visual) & set(reservoir))
    component_sets = [frozenset((i,)) for i in PLAYERS]
    component_sets += [frozenset(s) for s in itertools.combinations(PLAYERS, 2)]
    component_sets += [frozenset(PLAYERS)]
    audit = {"n": 0, "skipped_audio": [], "max_center_error": 0.0,
             "max_mobius_reconstruction_error": 0.0, "unique_scores": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for key in keys:
            base = manifest[key]
            length = max(1, int(np.ceil(float(base["duration"]) * 4)))
            audio_path = args.audio_dir / key[0] / f"{key[1]}.npy"
            if not audio_path.exists():
                audit["skipped_audio"].append(list(key))
                continue
            visual_curve = resize(np.asarray(visual[key]["score_curve"], float), length)
            language_curve = timeline_from_chunks(chunks.get(key, []), length)
            audio_embeddings = np.load(audio_path).astype(np.float32)
            audio_embeddings /= np.maximum(
                np.linalg.norm(audio_embeddings, axis=1, keepdims=True), 1e-12)
            audio_margin = audio_embeddings @ text_embeddings.T
            audio_curve = resize(audio_margin[:, 1] - audio_margin[:, 0], length)
            if "language" in args.drop:
                language_curve = np.zeros(length)
            if "audio" in args.drop:
                audio_curve = np.zeros(length)
            streams = np.stack([
                centered_rank(visual_curve), centered_rank(language_curve),
                centered_rank(audio_curve),
            ], axis=1)
            values = {subset: existential_worth(streams, subset) for subset in SUBSETS}
            components = {subset: mobius(values, subset) for subset in component_sets}
            reconstruction = sum(components.values())
            target = values[frozenset(PLAYERS)] - values[frozenset()]
            audit["max_mobius_reconstruction_error"] = max(
                audit["max_mobius_reconstruction_error"],
                float(np.max(np.abs(reconstruction - target))),
            )
            # The visual detector is the primary partial order; the derived
            # multimodal coalition field resolves only detector ties.  This
            # lexicographic dominance constraint prevents a weak modality from
            # overturning visually distinguishable frames while still making
            # every tied plateau temporally dense.
            mains = (robust_unit(components[frozenset((1,))])
                     + robust_unit(components[frozenset((2,))]))
            block_length, block_rule = -1, "not_computed"
            if args.fusion == "mobius_calibrated":
                calibrated_interactions, block_length, block_rule = (
                    calibrated_mobius_field(streams, components))
                coalition_field = mains + calibrated_interactions
            elif args.fusion == "mobius_raw":
                coalition_field = mains + sum(
                    robust_unit(components[s]) for s in component_sets if len(s) >= 2)
            elif args.fusion == "mains_only":
                coalition_field = mains
            else:  # "none": visual order only, ties averaged
                coalition_field = np.zeros(length)
            if args.order == "lexicographic":
                order = np.lexsort((coalition_field, visual_curve))
                primary_key, secondary_key = visual_curve, coalition_field
            else:  # equal-weight sum of unit-scaled visual and coalition field
                fused = robust_unit(visual_curve) + coalition_field
                order = np.argsort(fused, kind="stable")
                primary_key, secondary_key = fused, np.zeros(length)
            residual = np.empty(length, dtype=float)
            position = 0
            while position < length:
                end = position + 1
                anchor = order[position]
                while (end < length
                       and primary_key[order[end]] == primary_key[anchor]
                       and secondary_key[order[end]] == secondary_key[anchor]):
                    end += 1
                average_rank = 0.5 * (position + end - 1)
                residual[order[position:end]] = (average_rank + 0.5) / length - 0.5
                position = end
            residual -= residual.mean()  # exact orthogonal projection P_1^perp
            # A Jeffreys-smoothed detector occupancy is the local branch's
            # parameter-free video propensity. Equal-mass existential pooling
            # with the frozen MLLM logit supplies the semantic intercept.
            active = np.asarray(visual_curve > 0, dtype=float)
            occupancy = (float(active.sum()) + 0.5) / (length + 1.0)
            occupancy_logit = float(np.log(occupancy / (1 - occupancy)))
            if args.intercept == "both":
                intercept = logmeanexp(np.asarray([occupancy_logit, reservoir[key]]))
            elif args.intercept == "mllm":
                intercept = float(reservoir[key])
            elif args.intercept == "occupancy":
                intercept = occupancy_logit
            else:
                intercept = 0.0
            score = intercept + residual
            audit["max_center_error"] = max(
                audit["max_center_error"], abs(float(residual.mean())))
            audit["unique_scores"].append(int(len(np.unique(score))))
            output = {
                "schema_version": 1,
                "method": method_name,
                "ablation": {"drop": list(args.drop), "fusion": args.fusion, "order": args.order,
                             "intercept": args.intercept, "permutations": args.permutations},
                "dataset": key[0], "video_id": key[1],
                "duration": float(base["duration"]), "native_rate": 4.0,
                "score_curve": score.tolist(),
                # OMSL is a frame-score localizer.  Do not copy A10 proposals:
                # those intervals are not decoded from this method's timeline.
                "intervals": [],
                "modality_evidence": {
                    "players": list(NAMES),
                    "coalitions": 8,
                    "mobius_components": ["V", "L", "A", "VL", "VA", "LA", "VLA"],
                    "permutation_block_length": block_length,
                    "permutation_block_rule": block_rule,
                    "whole_video_intercept": intercept,
                    "raw_whole_video_mllm_logit": reservoir[key],
                    "visual_occupancy_logit": occupancy_logit,
                },
                "raw": {
                    "gt_access": False,
                    "global_local_noninterference": "score=semantic_intercept+P1_perp(local)",
                    "coalition_scorer": "symmetric_equal_mass_logmeanexp",
                    "fusion": "language_audio_main_plus_signed_joint_max_confidence_then_lexicographic_visual_dominance",
                    "interaction_calibration": "inclusive_upper_tail_p_from_31_nonwrapping_block_permutations_joint_max_over_time_and_VL_VA_LA_VLA",
                    "remaining_ties": "average_rank_no_random_or_temporal_index_break",
                    "normalization": "per_sample_robust_null_unit",
                    "learned_parameters": 0, "dataset_parameters": 0,
                    "runtime_label_selected_parameters": 0,
                    "development_selected_design": True,
                    "eligible_evidence_status": "exploratory_until_untouched_confirmation",
                    "score_domain": "unbounded_logit",
                },
                "code_path": CODE_PATH, "code_revision": CODE_REVISION,
                "calls": 0, "seed": PERMUTATION_SEED, "error": None,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            audit["n"] += 1
    if audit["unique_scores"]:
        audit["unique_score_median"] = float(np.median(audit["unique_scores"]))
        audit["unique_score_min"] = int(min(audit["unique_scores"]))
        audit["unique_score_max"] = int(max(audit["unique_scores"]))
    audit["code_path"] = CODE_PATH
    audit["code_revision"] = CODE_REVISION
    audit["run_date"] = date.today().isoformat()
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
