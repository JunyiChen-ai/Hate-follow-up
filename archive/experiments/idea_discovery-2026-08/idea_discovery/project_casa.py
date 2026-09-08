#!/usr/bin/env python3
"""CASA: cohort-adaptive selective authority for label-free localization."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def positive_sign_pvalue(positive: int, negative: int) -> float:
    """One-sided exact sign-test p-value under equal consensus probability."""
    n = positive + negative
    if n == 0 or positive <= negative:
        return 1.0
    return sum(math.comb(n, value) for value in range(positive, n + 1)) / (2.0 ** n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    extent, a08, a12, text = load(args.extent), load(args.a08), load(args.a12), load(args.text)
    keys = sorted(set(extent) & set(a08) & set(a12) & set(text))
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one text null anchor, got {anchors}")
    anchor = anchors.pop()

    consensus = defaultdict(lambda: {"positive": 0, "negative": 0})
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        if e08 == e12:
            consensus[key[0]]["positive" if e08 else "negative"] += 1
    regimes = {}
    for dataset, counts in consensus.items():
        pvalue = positive_sign_pvalue(counts["positive"], counts["negative"])
        regimes[dataset] = {**counts, "pvalue": pvalue,
                            "positive_dominant": pvalue < args.alpha}

    vetoes = 0
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        positive_dominant = regimes[key[0]]["positive_dominant"]
        z = float(text[key]["log_odds"])
        if agreement:
            keep = e08
        elif positive_dominant:
            keep = True
        else:
            keep = z >= anchor
        veto = not agreement and not positive_dominant and z < anchor
        vetoes += int(veto)
        source = extent[key]
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "casa_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"visual_agreement": agreement, "a08_nonempty": e08,
                               "a12_nonempty": e12, "text_log_odds": z,
                               "null_anchor": anchor, "cohort_regime": regimes[key[0]],
                               "text_authority_open": not positive_dominant,
                               "text_veto": veto, "keep": keep},
            raw={"gt_access": False, "cohort_statistic": "visual_consensus_exact_sign_test",
                 "alpha": args.alpha, "multiple_testing_adjustment": None,
                 "positive_dominant_action": "visual_OR",
                 "other_regime_action": "null_referenced_disagreement_arbitration"}))
    print(json.dumps({"n": len(keys), "regimes": regimes, "vetoes": vetoes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
