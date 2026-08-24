#!/usr/bin/env python3
"""Build canonical JSON/Markdown tables from official-val seed results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from hate_common import data as hdata

METHODS = {
    "vadclip": ("score_mlp", "AAAI 2024"),
    "dsanet": ("score_mlp", "AAAI 2026"),
    "macilsd": ("score_av", "ACM MM 2022"),
    # The independently trained unimodal models use Single_Model and emit its
    # sole upstream inference branch as score_mil.  score_audio/score_visual
    # belong only to the jointly trained AV model.
    "macilsd_audio": ("score_mil", "ACM MM 2022"),
    "macilsd_visual": ("score_mil", "ACM MM 2022"),
    "multihateloc": ("score_fused", "WWW 2026"),
    "cmhkf": ("score_align", "ACL 2025 Long"),
    "fed_wsvad_1client": ("score_align", "AAAI 2025"),
    "fed_wsvad_3client": ("score_align", "AAAI 2025"),
    "vera": ("score_official_postprocessed", "CVPR 2025"),
}
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
TRAIN_SEEDS = {234, 2025, 3407}
VERA_SEEDS = {234}
SUPERVISION = {method: "video-level labels" for method in METHODS if method != "vera"}
SUPERVISION["vera"] = "validation-selected; training-free"


def mean_sd(values):
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/reproduction/official_val/final")
    ap.add_argument("--json-out", default="docs/duplex/official_val_results.json")
    ap.add_argument("--md-out", default="docs/duplex/OFFICIAL_VAL_RESULTS.md")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write an explicitly incomplete preview instead of "
                         "requiring every preregistered seed and corpus")
    args = ap.parse_args(argv)
    root, rows, errors, code_commits = Path(args.root), [], [], set()
    expected_cohorts = {}
    for corpus in CORPORA:
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        expected_cohorts[corpus] = {
            "n_videos": len(gt),
            "within_hate_n": sum(
                labels.get(vid) == 1 and len(np.unique(gold)) == 2
                for vid, gold in gt.items()),
        }
    for method, (branch, venue) in METHODS.items():
        for corpus in CORPORA:
            runs = []
            for path in sorted((root / method / corpus).glob("seed_*/frame_eval.json")):
                seed = int(path.parent.name.split("_")[-1])
                frozen_path = path.parent / "frozen_config.json"
                try:
                    frozen = json.loads(frozen_path.read_text())
                    commit = frozen["code_commit"]
                    source = Path(frozen["source"])
                    frozen_ok = (frozen.get("method") == method and
                                 frozen.get("corpus") == corpus and
                                 frozen.get("seed") == seed and
                                 isinstance(commit, str) and len(commit) == 40 and
                                 all(ch in "0123456789abcdef" for ch in commit) and
                                 source.is_file() and
                                 frozen.get("source_sha256") == file_sha256(source))
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    frozen_ok, commit = False, None
                if not frozen_ok:
                    errors.append(f"invalid frozen config: {frozen_path}")
                    continue
                code_commits.add(commit)
                payload = json.loads(path.read_text())
                if payload.get("corpus") != corpus or payload.get("split") != "test":
                    errors.append(f"identity/split mismatch: {path}")
                score_path = Path(payload.get("scores_file", ""))
                if (not score_path.is_file() or
                        payload.get("scores_sha256") != file_sha256(score_path)):
                    errors.append(f"score fingerprint mismatch: {path}")
                if branch not in payload["results"]:
                    errors.append(f"missing branch {branch}: {path}")
                    continue
                r = payload["results"][branch]
                if r.get("n_videos_missing_from_scores") != 0:
                    errors.append(f"missing scored videos: {path}")
                if r.get("n_videos_not_in_gold") != 0:
                    errors.append(f"scores outside frozen gold: {path}")
                metrics = {"roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                           "video_roc_auc": r["video_level"]["max_roc_auc"],
                           "video_pr_auc": r["video_level"]["max_pr_auc"],
                           "within_hate_auc": r["per_video"]["macro_auc"]}
                if not all(value is not None and math.isfinite(float(value))
                           for value in metrics.values()):
                    errors.append(f"non-finite metric: {path} / {branch}")
                    continue
                runs.append({"seed": seed, "code_commit": commit,
                             **metrics,
                             "n_videos": r["n_videos"],
                             "within_hate_n": r["per_video"]["n_videos_both_classes"]})
            expected = VERA_SEEDS if method == "vera" else TRAIN_SEEDS
            actual = {r["seed"] for r in runs}
            if actual != expected:
                errors.append(
                    f"{method}/{corpus} seeds {sorted(actual)}; "
                    f"expected {sorted(expected)}")
            if not runs:
                continue
            if len({r["n_videos"] for r in runs}) != 1:
                errors.append(f"inconsistent video counts: {method}/{corpus}")
            if len({r["within_hate_n"] for r in runs}) != 1:
                errors.append(f"inconsistent within-hate cohorts: {method}/{corpus}")
            expected = expected_cohorts[corpus]
            if any(r["n_videos"] != expected["n_videos"] for r in runs):
                errors.append(f"wrong frozen video count: {method}/{corpus}")
            if any(r["within_hate_n"] != expected["within_hate_n"] for r in runs):
                errors.append(f"wrong within-hate cohort: {method}/{corpus}")
            rows.append({"method": method, "venue": venue, "corpus": corpus,
                         "supervision": SUPERVISION[method], "branch": branch,
                         "protocol": "official-val",
                         "n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
                         **{k: mean_sd([r[k] for r in runs])
                            for k in ("roc_auc", "pr_auc", "video_roc_auc",
                                      "video_pr_auc", "within_hate_auc")},
                         "within_hate_n": runs[0]["within_hate_n"]})
    if errors and not args.allow_partial:
        raise SystemExit("official-val aggregation refused:\n  - " +
                         "\n  - ".join(errors))
    if len(code_commits) != 1:
        errors.append(f"expected one code commit, found {sorted(code_commits)}")
        if not args.allow_partial:
            raise SystemExit("official-val aggregation refused:\n  - " + errors[-1])
    archive_commit = next(iter(code_commits)) if len(code_commits) == 1 else None
    payload = {"schema_version": 3, "protocol": "official-val",
               "code_commit": archive_commit,
               "complete": not errors, "validation_errors": errors,
               "rows": rows}
    jout = Path(args.json_out)
    atomic_write(jout, json.dumps(payload, indent=2) + "\n")
    lines = ["# Weakly supervised baselines — official validation", "",
             f"Code commit: `{archive_commit or 'mixed/incomplete'}`", "",
             "| Method | Venue | Supervision | Corpus | Seeds | Frame ROC | Frame PR | Video ROC | Video AP | Within-hate ROC |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return f"{x['mean']:.4f}" + (f" ± {x['std']:.4f}" if x["std"] is not None else "")
    for r in rows:
        lines.append(f"| {r['method']} | {r['venue']} | {r['supervision']} | "
                     f"{r['corpus']} | {r['n_seeds']} | "
                     f"{fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} | "
                     f"{fmt(r['video_roc_auc'])} | {fmt(r['video_pr_auc'])} | "
                     f"{fmt(r['within_hate_auc'])} (n={r['within_hate_n']}) |")
    mout = Path(args.md_out)
    atomic_write(mout, "\n".join(lines) + "\n")
    print(f"wrote {jout} and {mout}: {len(rows)} rows")
    return 0


if __name__ == "__main__": raise SystemExit(main())
