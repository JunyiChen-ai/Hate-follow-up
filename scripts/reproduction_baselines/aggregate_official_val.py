#!/usr/bin/env python3
"""Build canonical JSON/Markdown tables from official-val seed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/reproduction/official_val/final")
    ap.add_argument("--json-out", default="docs/duplex/official_val_results.json")
    ap.add_argument("--md-out", default="docs/duplex/OFFICIAL_VAL_RESULTS.md")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write an explicitly incomplete preview instead of "
                         "requiring every preregistered seed and corpus")
    args = ap.parse_args(argv)
    root, rows, errors = Path(args.root), [], []
    for method, (branch, venue) in METHODS.items():
        for corpus in CORPORA:
            runs = []
            for path in sorted((root / method / corpus).glob("seed_*/frame_eval.json")):
                payload = json.loads(path.read_text())
                if payload.get("corpus") != corpus or payload.get("split") != "test":
                    errors.append(f"identity/split mismatch: {path}")
                if branch not in payload["results"]: continue
                r = payload["results"][branch]
                if r.get("n_videos_missing_from_scores") != 0:
                    errors.append(f"missing scored videos: {path}")
                if r.get("n_videos_not_in_gold") != 0:
                    errors.append(f"scores outside frozen gold: {path}")
                runs.append({"seed": int(path.parent.name.split("_")[-1]),
                             "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                             "video_roc_auc": r["video_level"]["max_roc_auc"],
                             "video_pr_auc": r["video_level"]["max_pr_auc"],
                             "within_hate_auc": r["per_video"]["macro_auc"],
                             "within_hate_n": r["per_video"]["n_videos_both_classes"]})
            expected = VERA_SEEDS if method == "vera" else TRAIN_SEEDS
            actual = {r["seed"] for r in runs}
            if actual != expected:
                errors.append(
                    f"{method}/{corpus} seeds {sorted(actual)}; "
                    f"expected {sorted(expected)}")
            if not runs:
                continue
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
    payload = {"schema_version": 2, "protocol": "official-val",
               "complete": not errors, "validation_errors": errors,
               "rows": rows}
    jout = Path(args.json_out)
    atomic_write(jout, json.dumps(payload, indent=2) + "\n")
    lines = ["# Weakly supervised baselines — official validation", "",
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
