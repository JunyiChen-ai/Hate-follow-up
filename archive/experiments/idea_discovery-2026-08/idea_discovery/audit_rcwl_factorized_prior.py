#!/usr/bin/env python3
"""GT-free falsification audit for RCWL's factorized proposal prior.

The role-binding head is not treated as a localizer.  This audit asks the
narrower question needed by RCWL: does the temporally aligned pairing of
visual role grounding and transcript assertion contain more local structure
than circularly shifted pairings?  If not, the static head must not be used to
route or veto counterfactual trials.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


CHOICES = ("FULL", "STANCE_FLIP", "ROLE_MISMATCH", "ABSENT", "UNKNOWN")


def load(path: Path):
    rows = {}
    for line in path.open():
        row = json.loads(line)
        rows[(row["dataset"], row["video_id"])] = row
    return rows


def factor(row, modality: str):
    ev = row.get("modality_evidence", {})
    names = tuple(ev.get("binding_choices", CHOICES))
    scores = np.asarray(ev.get("binding_scores", []), dtype=float)
    if scores.ndim != 2 or scores.shape[0] < 2:
        return None
    index = {name: names.index(name) for name in CHOICES if name in names}
    if len(index) != len(CHOICES):
        return None
    eps = 1e-6
    p = np.clip(scores, eps, 1.0)
    if modality == "visual":
        # Visual evidence may ground the relation while leaving stance unknown.
        positive = p[:, index["FULL"]] + p[:, index["STANCE_FLIP"]]
        negative = p[:, index["ROLE_MISMATCH"]] + p[:, index["ABSENT"]]
    else:
        # Transcript evidence must assert/endorse rather than merely quote it.
        positive = p[:, index["FULL"]]
        negative = p[:, index["STANCE_FLIP"]] + p[:, index["ABSENT"]]
    value = np.log((positive + eps) / (negative + eps))
    value *= 1.0 - p[:, index["UNKNOWN"]]
    # Within-video ranks prevent either head's confidence scale dominating.
    return (rankdata(value, method="average") - 1.0) / max(1, len(value) - 1)


def local_witness(v, t, radius=1):
    n = min(len(v), len(t)); v = v[:n]; t = t[:n]
    field = np.zeros(n, dtype=float)
    for i in range(n):
        js = [(i + d) % n for d in range(-radius, radius + 1)]
        field[i] = max(min(v[i], t[j]) for j in js)
    # Concentration rewards a localized common witness, not a global high gate.
    return float(np.quantile(field, .9) - np.median(field)), field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", type=Path, required=True)
    ap.add_argument("--transcript", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=int, default=1)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    visual, transcript = load(args.visual), load(args.transcript)
    records = []
    for key in sorted(set(visual) & set(transcript)):
        vr, tr = visual[key], transcript[key]
        v, t = factor(vr, "visual"), factor(tr, "transcript")
        if v is None or t is None:
            continue
        n = min(len(v), len(t)); v, t = v[:n], t[:n]
        aligned, field = local_witness(v, t, args.radius)
        null = []
        for shift in range(1, n):
            value, _ = local_witness(v, np.roll(t, shift), args.radius)
            null.append(value)
        relation = vr.get("modality_evidence", {}).get("event_relation", {})
        rank = 1 + sum(x >= aligned - 1e-12 for x in null)
        records.append({
            "dataset": key[0], "video_id": key[1], "n_bins": n,
            "stance": relation.get("stance", "unknown"),
            "aligned_stat": aligned,
            "null_median": float(np.median(null)),
            "aligned_minus_null_median": aligned - float(np.median(null)),
            "exact_rotation_rank": rank,
            "exact_p": rank / n,
            "strictly_beats_all_shifts": bool(aligned > max(null) + 1e-12),
            "witness_bins": np.flatnonzero(field >= np.quantile(field, .9)).tolist(),
        })
    by_stance = defaultdict(list)
    for row in records:
        by_stance[row["stance"]].append(row)
    def summarize(rows):
        gains = np.asarray([x["aligned_minus_null_median"] for x in rows])
        return {
            "n": len(rows),
            "aligned_gt_null_median": int(np.sum(gains > 0)),
            "aligned_gt_null_rate": float(np.mean(gains > 0)) if len(rows) else None,
            "median_aligned_minus_null": float(np.median(gains)) if len(rows) else None,
            "strict_all_shift_certificates": int(sum(x["strictly_beats_all_shifts"] for x in rows)),
            "exact_p_le_0p125": int(sum(x["exact_p"] <= .125 for x in rows)),
        }
    report = {
        "audit": "rcwl_factorized_prior_v1",
        "gt_access": False,
        "statistic": "q90-minus-median of local bottleneck rank field",
        "radius": args.radius,
        "overall": summarize(records),
        "stance_counts": dict(Counter(x["stance"] for x in records)),
        "by_stance": {key: summarize(value) for key, value in sorted(by_stance.items())},
        "records": records,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
