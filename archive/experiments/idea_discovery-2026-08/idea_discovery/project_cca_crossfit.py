#!/usr/bin/env python3
"""Two-fold cross-fitted Consensus-Calibrated Authority."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def fold(key: tuple[str, str]) -> int:
    return int(hashlib.sha256(f"cca-crossfit-v1|{key[0]}|{key[1]}".encode()).hexdigest(), 16) % 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
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
    audits = defaultdict(lambda: {"agree": 0, "conflict": 0})
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        if e08 != e12 or float(text[key]["log_odds"]) >= anchor:
            continue
        # Stored by calibration fold. Deployment fold uses the opposite one.
        field = "conflict" if e08 else "agree"
        audits[(key[0], fold(key))][field] += 1
    authority = {}
    for dataset in {key[0] for key in keys}:
        for deployment_fold in (0, 1):
            values = audits[(dataset, 1 - deployment_fold)]
            posterior = (values["agree"] + 1.0) / (values["agree"] + values["conflict"] + 2.0)
            authority[(dataset, deployment_fold)] = {
                "calibration_fold": 1 - deployment_fold,
                "veto_agrees_negative": values["agree"],
                "veto_conflicts_positive": values["conflict"],
                "posterior_mean": posterior,
                "text_veto_authorized": posterior > 0.5,
            }
    vetoes = 0
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12; deployment_fold = fold(key)
        audit = authority[(key[0], deployment_fold)]
        authorized = audit["text_veto_authorized"]
        z = float(text[key]["log_odds"])
        keep = e08 if agreement else (z >= anchor if authorized else True)
        veto = not agreement and authorized and z < anchor
        vetoes += int(veto)
        source = extent[key]
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "cca_crossfit_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"fold": deployment_fold, "visual_agreement": agreement,
                               "a08_nonempty": e08, "a12_nonempty": e12,
                               "text_log_odds": z, "null_anchor": anchor,
                               "consensus_veto_audit": audit,
                               "text_veto_authorized": authorized,
                               "text_veto": veto, "keep": keep},
            raw={"gt_access": False, "crossfit": "sha256_two_fold",
                 "calibration_excludes_deployment_fold": True,
                 "authority_rule": "beta11_posterior_mean_above_half"}))
    printable = {f"{dataset}/deploy{number}": value
                 for (dataset, number), value in sorted(authority.items())}
    print(json.dumps({"n": len(keys), "authority": printable, "vetoes": vetoes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
