#!/usr/bin/env python3
"""CCA: earn disagreement veto rights on the visual-consensus jurisdiction."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorization-confidence", type=float, default=0.95)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    extent, a08, a12, text = load(args.extent), load(args.a08), load(args.a12), load(args.text)
    keys = sorted(set(extent) & set(a08) & set(a12) & set(text))
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one explicit text null, got {anchors}")
    anchor = anchors.pop()

    audits = defaultdict(lambda: {"veto_agrees_negative": 0, "veto_conflicts_positive": 0})
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        text_veto = float(text[key]["log_odds"]) < anchor
        if e08 == e12 and text_veto:
            field = "veto_conflicts_positive" if e08 else "veto_agrees_negative"
            audits[key[0]][field] += 1
    authority = {}
    for dataset, values in audits.items():
        agree, conflict = values["veto_agrees_negative"], values["veto_conflicts_positive"]
        posterior_mean = (agree + 1.0) / (agree + conflict + 2.0)
        probability_above_half = float(beta.sf(0.5, agree + 1, conflict + 1))
        authority[dataset] = {**values, "posterior_mean": posterior_mean,
                              "probability_reliability_above_half": probability_above_half,
                              "authorization_confidence": args.authorization_confidence,
                              "text_veto_authorized": probability_above_half > args.authorization_confidence}

    vetoes = 0
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        z = float(text[key]["log_odds"])
        authorized = authority[key[0]]["text_veto_authorized"]
        if agreement:
            keep = e08
        elif authorized:
            keep = z >= anchor
        else:
            keep = True
        veto = not agreement and authorized and z < anchor
        vetoes += int(veto)
        source = extent[key]
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "cca_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"visual_agreement": agreement, "a08_nonempty": e08,
                               "a12_nonempty": e12, "text_log_odds": z,
                               "null_anchor": anchor, "consensus_veto_audit": authority[key[0]],
                               "text_veto_authorized": authorized,
                               "text_veto": veto, "keep": keep},
            raw={"gt_access": False,
                 "calibration_jurisdiction": "visual_consensus_only",
                 "deployment_jurisdiction": "visual_disagreement_only",
                 "authority_rule": "beta11_probability_reliability_above_half",
                 "text_can_create_or_move_boundary": False}))
    print(json.dumps({"n": len(keys), "authority": authority, "vetoes": vetoes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
