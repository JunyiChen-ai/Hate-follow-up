"""Audit script: pull boundary-band videos and produce a markdown for evidence-type categorization.

Reads the entropy-band candidate file and the annotation file, samples N in-band
videos with mixed correct/incorrect baseline predictions, and writes a markdown
with title + transcript + ground-truth label + baseline prediction so we can
manually annotate which evidence type drives each case.
"""

import argparse
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path("/data/jehc223/EMNLP2")


def load_band(dataset: str):
    path = PROJECT_ROOT / f"results/boundary_rescue/{dataset}/candidates_entropy_band_2b.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_annotations(dataset: str):
    path = PROJECT_ROOT / f"datasets/{dataset}/annotation(new).json"
    with open(path) as f:
        ann = json.load(f)
    return {item["Video_ID"]: item for item in ann}


def label_to_int(item, dataset: str) -> int:
    if dataset == "HateMM":
        return 1 if item.get("Label", "").lower().startswith("h") else 0
    raw = item.get("Label", "")
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"hateful", "offensive", "1"}:
            return 1
        if s in {"normal", "0"}:
            return 0
    if isinstance(raw, int):
        return 1 if raw >= 1 else 0
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MHClip_EN")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    band = load_band(args.dataset)
    ann = load_annotations(args.dataset)

    in_band = [r for r in band if r.get("in_band")]
    enriched = []
    for r in in_band:
        vid = r["video_id"]
        if vid not in ann:
            continue
        item = ann[vid]
        gt = label_to_int(item, args.dataset)
        if gt < 0:
            continue
        pred = r.get("pred_baseline", -1)
        enriched.append({
            "video_id": vid,
            "score": r["score"],
            "entropy": r["entropy"],
            "pred_baseline": pred,
            "ground_truth": gt,
            "correct": int(pred == gt),
            "title": item.get("Title", "")[:300],
            "transcript": item.get("Transcript", "")[:1500],
        })

    correct = [e for e in enriched if e["correct"] == 1]
    wrong = [e for e in enriched if e["correct"] == 0]
    rng = random.Random(args.seed)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    half = args.n // 2
    pick = wrong[:half] + correct[: args.n - half]
    rng.shuffle(pick)

    out_path = args.out or str(PROJECT_ROOT / f"docs/audit_band_{args.dataset}.md")
    with open(out_path, "w") as f:
        f.write(f"# Boundary band audit — {args.dataset}\n\n")
        f.write(f"In-band total: {len(enriched)}; "
                f"baseline-correct: {len(correct)}; baseline-wrong: {len(wrong)}.\n")
        f.write(f"Sampled {len(pick)} ({half} wrong + {args.n - half} correct).\n\n")
        f.write("Columns: VID | score | entropy | pred | GT | correct\n\n")
        f.write("---\n\n")
        for i, e in enumerate(pick, 1):
            f.write(f"## {i}. `{e['video_id']}`\n\n")
            f.write(f"score={e['score']:.3f}  entropy={e['entropy']:.3f}  "
                    f"pred={e['pred_baseline']}  GT={e['ground_truth']}  "
                    f"correct={'YES' if e['correct'] else 'NO'}\n\n")
            f.write(f"**Title**: {e['title']}\n\n")
            f.write(f"**Transcript** (first 1500 chars):\n\n")
            f.write(f"> {e['transcript']}\n\n")
            f.write("**Evidence type** (to fill): ___\n\n")
            f.write("---\n\n")

    print(f"Wrote {out_path}")
    print(f"In-band: {len(enriched)} | sampled {len(pick)} (correct {args.n - half}, wrong {half})")


if __name__ == "__main__":
    main()
