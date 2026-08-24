#!/usr/bin/env python3
"""Refuse frozen-test confirmation until every requested search is complete."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import optuna
from optuna.trial import TrialState


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
            database = path.parent / "study.sqlite3"
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            if not database.is_file():
                errors.append(f"missing {database}")
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
                study = optuna.load_study(
                    study_name=f"{method}-{corpus}",
                    storage=f"sqlite:///{database.resolve()}")
                complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
                running = sum(t.state == TrialState.RUNNING for t in study.trials)
                if complete != int(rec.get("n_complete", -1)):
                    errors.append(
                        f"study/summary COMPLETE mismatch for {method}/{corpus}: "
                        f"{complete} != {rec.get('n_complete')}")
                if running:
                    errors.append(f"{method}/{corpus} has {running} RUNNING trial(s)")
                if study.best_trial.number != int(rec.get("best_trial", -1)):
                    errors.append(f"best-trial mismatch for {method}/{corpus}")
                if study.best_value != float(rec["best_value"]):
                    errors.append(f"best-value mismatch for {method}/{corpus}")
                if study.best_params != rec["best_params"]:
                    errors.append(f"best-params mismatch for {method}/{corpus}")
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
