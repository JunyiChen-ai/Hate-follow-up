#!/usr/bin/env python3
"""Audit cached VideoTGB decisions against an anchored first-answer parser."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    details = []
    for row in map(json.loads, args.predictions.open()):
        answer = str(row.get("raw", {}).get("answer", ""))
        match = re.match(r"^\s*(yes|no)\b", answer, flags=re.IGNORECASE)
        first = match.group(1).lower() if match else None
        predicted_nonempty = bool(row.get("intervals"))
        tail = answer[match.end():] if match else answer
        contradictory_tail = bool(first and re.search(
            r"(?:</s>|\buser\b|\bassistant\b).*\b" +
            ("no" if first == "yes" else "yes") + r"\b", tail,
            flags=re.IGNORECASE | re.DOTALL))
        details.append({
            "dataset": row["dataset"], "video_id": row["video_id"],
            "first_answer": first, "predicted_nonempty": predicted_nonempty,
            "decision_matches_first_answer": (
                first is not None and predicted_nonempty == (first == "yes")),
            "contradictory_tail": contradictory_tail,
        })
    output = {
        "n": len(details),
        "n_parseable": sum(row["first_answer"] is not None for row in details),
        "n_decision_matches": sum(row["decision_matches_first_answer"] for row in details),
        "n_contradictory_tail": sum(row["contradictory_tail"] for row in details),
        "all_decisions_match": all(row["decision_matches_first_answer"] for row in details),
        "detail": details,
    }
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in (
        "n", "n_parseable", "n_decision_matches", "n_contradictory_tail",
        "all_decisions_match")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
