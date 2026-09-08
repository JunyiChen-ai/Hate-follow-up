#!/usr/bin/env python3
"""TRIARC M2 kill pilot: parameter-free role-disjoint evidence bottlenecks.

V is the frozen semantic field, T is timestamped MLLM hostile-proposition
evidence, and A is acoustic carrier/change support.  Each field is converted to
average-tie within-video empirical ranks.  Fusion is the non-compensatory
minimum, so no modality can dominate by compensating for an absent role.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


DATA_DIR = {"HateMM": "hatemm", "HateClipSeg": "hateclipseg", "MHC": "mhclip_en", "MHC_zh": "mhclip_zh"}
METHODS = ("triarc_v_v1", "triarc_t_v1", "triarc_a_v1", "triarc_vt_v1", "triarc_va_v1", "triarc_ta_v1", "triarc_vta_v1")


def load_base(path: Path, method: str) -> dict[tuple[str, str], dict]:
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, path.open()) if r.get("method") == method}


def load_chunks(path: Path) -> dict[tuple[str, str], list[dict]]:
    out = defaultdict(list)
    for row in map(json.loads, path.open()):
        out[(row["dataset"], row["video_id"])].append(row)
    return out


def midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return rankdata(values, method="average") / (len(values) + 1.0)


def resize(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == n:
        return values
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(values)), values)


def audio_support(path: Path, n: int) -> np.ndarray | None:
    if not path.exists():
        return None
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        return None
    norm = value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
    change = np.empty(len(value), dtype=np.float64)
    change[1:] = 1.0 - np.sum(norm[1:] * norm[:-1], axis=1)
    change[0] = change[1]
    return resize(midrank(change), n)


def text_support(chunks: list[dict], duration: float, n: int) -> np.ndarray | None:
    if not chunks:
        return None
    nulls = [float(x.get("config", {}).get("null_log_odds", -12.0)) for x in chunks]
    field = np.full(n, float(np.median(nulls)), dtype=np.float64)
    times = (np.arange(n) + 0.5) * duration / n
    for row in chunks:
        mask = (times >= float(row["start"])) & (times < float(row["end"]))
        field[mask] = np.maximum(field[mask], float(row["log_odds"]))
    return midrank(field)


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
    base, chunks = load_base(args.base, args.base_method), load_chunks(args.chunks)
    with args.out.open("w") as handle:
        for key, row in sorted(base.items()):
            n = len(row["score_curve"])
            fields = {"V": midrank(np.asarray(row["score_curve"], dtype=np.float64))}
            text = text_support(chunks.get(key, []), float(row["duration"]), n)
            audio = audio_support(args.audio_root / DATA_DIR[key[0]] / f"{key[1]}.npy", n)
            if text is not None:
                fields["T"] = text
            if audio is not None:
                fields["A"] = audio
            variants = {
                "triarc_v_v1": fields["V"],
                "triarc_t_v1": fields.get("T", np.zeros(n)),
                "triarc_a_v1": fields.get("A", np.zeros(n)),
                "triarc_vt_v1": np.minimum(fields["V"], fields["T"]) if "T" in fields else np.zeros(n),
                "triarc_va_v1": np.minimum(fields["V"], fields["A"]) if "A" in fields else np.zeros(n),
                "triarc_ta_v1": np.minimum(fields["T"], fields["A"]) if "T" in fields and "A" in fields else np.zeros(n),
                "triarc_vta_v1": np.minimum.reduce([fields["V"], fields["T"], fields["A"]]) if "T" in fields and "A" in fields else np.zeros(n),
            }
            for method in METHODS:
                output = dict(row)
                output["method"] = method
                output["score_curve"] = variants[method].tolist()
                output["intervals"] = []
                output["raw"] = {
                    "gt_access": False,
                    "module": "role_disjoint_bottleneck_v1",
                    "available_modalities": sorted(fields),
                    "tie_rule": "average_midrank",
                    "dataset_parameters": 0,
                    "fusion": "noncompensatory_minimum",
                    "interval_decoder_executed": False,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
