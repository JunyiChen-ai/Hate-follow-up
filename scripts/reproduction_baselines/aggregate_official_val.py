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
    "multihateloc": ("score_fused", "WWW 2026"),
    "cmhkf": ("score_align", "ACL 2025 Long"),
    "fed_wsvad_1client": ("score_align", "AAAI 2025"),
    "fed_wsvad_3client": ("score_align", "AAAI 2025"),
    "vera": ("score_official_postprocessed", "CVPR 2025"),
}
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")


def mean_sd(values):
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/reproduction/official_val/final")
    ap.add_argument("--json-out", default="docs/duplex/official_val_results.json")
    ap.add_argument("--md-out", default="docs/duplex/OFFICIAL_VAL_RESULTS.md")
    args = ap.parse_args(argv)
    root, rows = Path(args.root), []
    for method, (branch, venue) in METHODS.items():
        for corpus in CORPORA:
            runs = []
            for path in sorted((root / method / corpus).glob("seed_*/frame_eval.json")):
                payload = json.loads(path.read_text())
                if branch not in payload["results"]: continue
                r = payload["results"][branch]
                runs.append({"seed": int(path.parent.name.split("_")[-1]),
                             "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                             "video_roc_auc": r["video_level"]["max_roc_auc"],
                             "video_pr_auc": r["video_level"]["max_pr_auc"],
                             "within_hate_auc": r["per_video"]["macro_auc"],
                             "within_hate_n": r["per_video"]["n_videos_both_classes"]})
            if not runs: continue
            rows.append({"method": method, "venue": venue, "corpus": corpus,
                         "branch": branch, "protocol": "official-val",
                         "n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
                         **{k: mean_sd([r[k] for r in runs])
                            for k in ("roc_auc", "pr_auc", "video_roc_auc",
                                      "video_pr_auc", "within_hate_auc")},
                         "within_hate_n": runs[0]["within_hate_n"]})
    payload = {"schema_version": 1, "protocol": "official-val", "rows": rows}
    jout = Path(args.json_out); jout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Weakly supervised baselines — official validation", "",
             "| Method | Venue | Corpus | Seeds | Frame ROC | Frame PR | Video ROC | Video AP | Within-hate ROC |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return f"{x['mean']:.4f}" + (f" ± {x['std']:.4f}" if x["std"] is not None else "")
    for r in rows:
        lines.append(f"| {r['method']} | {r['venue']} | {r['corpus']} | {r['n_seeds']} | "
                     f"{fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} | "
                     f"{fmt(r['video_roc_auc'])} | {fmt(r['video_pr_auc'])} | "
                     f"{fmt(r['within_hate_auc'])} (n={r['within_hate_n']}) |")
    mout = Path(args.md_out); mout.parent.mkdir(parents=True, exist_ok=True)
    mout.write_text("\n".join(lines) + "\n")
    print(f"wrote {jout} and {mout}: {len(rows)} rows")
    return 0


if __name__ == "__main__": raise SystemExit(main())
