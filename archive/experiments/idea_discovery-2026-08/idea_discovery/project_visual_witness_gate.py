#!/usr/bin/env python3
"""Apply a frozen visual shell witness to a candidate prediction file."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(path, method=None):
    out = {}
    for row in map(json.loads, Path(path).open()):
        if method is None or row.get("method") == method:
            out[(row["dataset"], row["video_id"])] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--candidate", required=True)
    ap.add_argument("--candidate-method", required=True); ap.add_argument("--witness", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--method", default="pact_visual_witness")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--mode", choices=("warrant", "ambiguity", "appellate"), default="warrant",
                    help="ambiguity delegates only when the two witness scores have opposite signs or touch zero")
    a = ap.parse_args(); out = Path(a.out)
    if out.exists(): raise RuntimeError(f"refusing existing output: {out}")
    base, candidate = load(a.base), load(a.candidate, a.candidate_method)
    witness = load(a.witness); out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"candidate": 0, "fallback": 0}
    with out.open("x", encoding="utf-8") as handle:
        for key in sorted(set(base) & set(candidate)):
            witness_row = witness.get(key, {})
            scores = witness_row.get("scores")
            if a.mode in {"ambiguity", "appellate"}:
                if not isinstance(scores, list) or len(scores) != 2:
                    passed = False
                elif a.mode == "appellate":
                    # SUPPORT approves the motion; REJECT (two negatives)
                    # denies it; ABSTAIN lets the already full-orbit-certified
                    # language motion carry the appeal.
                    passed = not (float(scores[0]) < 0 and float(scores[1]) < 0)
                else:
                    passed = float(scores[0]) * float(scores[1]) <= 0
            else:
                passed = bool(witness_row.get("visual_warrant"))
            if a.invert and key in witness:
                passed = not passed
            source = candidate[key] if passed else base[key]
            row = dict(source); row["method"] = a.method
            row["modality_evidence"] = {"visual_shell_warrant": passed,
                "witness_scores": witness.get(key, {}).get("scores"),
                "candidate_method": a.candidate_method, "fallback_method": base[key]["method"]}
            if a.mode == "ambiguity":
                rule = "candidate_iff_dual_offset_sign_disagreement_or_zero"
            elif a.mode == "appellate":
                rule = "candidate_iff_visual_support_or_full_orbit_language_appeal_on_abstention"
            else:
                rule = "candidate_iff_visual_shell_warrant"
            row["raw"] = {"gt_access": False, "rule": rule, "mode": a.mode}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["candidate" if passed else "fallback"] += 1
    print(json.dumps(counts))


if __name__ == "__main__": main()
