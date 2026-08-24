#!/usr/bin/env python3
"""Refuse frozen-test confirmation until every requested search is complete."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/reproduction/official_val/tuning")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    errors = []
    for corpus in args.corpora:
        for method in args.methods:
            path = root / method / corpus / "best.json"
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            try:
                rec = json.loads(path.read_text())
                if rec.get("method") != method or rec.get("corpus") != corpus:
                    errors.append(f"identity mismatch in {path}")
                if int(rec.get("n_complete", -1)) < args.trials:
                    errors.append(
                        f"incomplete {method}/{corpus}: "
                        f"{rec.get('n_complete')}/{args.trials}")
                if not math.isfinite(float(rec["best_value"])):
                    errors.append(f"non-finite best value in {path}")
                if not isinstance(rec.get("best_params"), dict) or not rec["best_params"]:
                    errors.append(f"missing best params in {path}")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"invalid {path}: {exc}")

    if errors:
        print("official-validation confirmation is not ready:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"ready: {len(args.methods)} methods x {len(args.corpora)} corpora")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
