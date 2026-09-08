#!/usr/bin/env python3
"""Per-video CCA using nearest visual-consensus calibration neighbors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def embedding(root: Path, key: tuple[str, str]) -> np.ndarray:
    value = np.asarray(np.load(root / key[0] / f"{key[1]}.npy"), dtype=np.float32)
    vector = value.mean(axis=0) if value.ndim > 1 else value
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    extent, a08, a12, text = load(args.extent), load(args.a08), load(args.a12), load(args.text)
    keys = sorted(set(extent) & set(a08) & set(a12) & set(text))
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one explicit text null, got {anchors}")
    anchor = anchors.pop()
    vectors = {key: embedding(args.features_root, key) for key in keys}
    consensus_keys = [key for key in keys
                      if bool(a08[key].get("intervals")) == bool(a12[key].get("intervals"))]
    matrix = np.stack([vectors[key] for key in consensus_keys])
    authorized_count = vetoes = 0

    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        z = float(text[key]["log_odds"])
        audit = None; authorized = False
        if not agreement:
            similarities = matrix @ vectors[key]
            order = np.argsort(-similarities, kind="stable")[:min(args.neighbors, len(consensus_keys))]
            neighbors = [consensus_keys[int(index)] for index in order]
            agree = conflict = 0
            for other in neighbors:
                other_veto = float(text[other]["log_odds"]) < anchor
                if not other_veto:
                    continue
                visual_positive = bool(a08[other].get("intervals"))
                if visual_positive:
                    conflict += 1
                else:
                    agree += 1
            posterior_mean = (agree + 1.0) / (agree + conflict + 2.0)
            authorized = posterior_mean > 0.5
            audit = {"neighbors": len(neighbors), "veto_agrees_negative": agree,
                     "veto_conflicts_positive": conflict, "posterior_mean": posterior_mean,
                     "text_veto_authorized": authorized}
            authorized_count += int(authorized)
        keep = e08 if agreement else (z >= anchor if authorized else True)
        veto = not agreement and authorized and z < anchor
        vetoes += int(veto)
        source = extent[key]
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "local_cca_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"visual_agreement": agreement, "a08_nonempty": e08,
                               "a12_nonempty": e12, "text_log_odds": z,
                               "null_anchor": anchor, "local_consensus_audit": audit,
                               "text_veto_authorized": authorized, "text_veto": veto,
                               "keep": keep},
            raw={"gt_access": False, "neighbors": args.neighbors,
                 "neighbor_space": "mean_frozen_clip_l14_cosine",
                 "dataset_id_used_for_calibration": False,
                 "authority_rule": "beta11_posterior_mean_above_half"}))
    print(json.dumps({"n": len(keys), "disagreements": len(keys) - len(consensus_keys),
                      "authorized_disagreements": authorized_count, "vetoes": vetoes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
