"""Compute ACC/M-F1/M-P/M-R for each label-free / few-shot baseline across 4 datasets."""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from boundary_rescue.grid_eval_all import load_labels, SKIP_VIDEOS, ld_jsonl

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

ROOT = Path("/data/jehc223/EMNLP2/results")
DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]

BASELINES = {
    "Naive MLLM (Qwen3-VL-2B)": {ds: ROOT / "naive_2b" / ds / "test_naive.jsonl" for ds in DATASETS},
    "Naive (InternVL3-8B)":     {ds: ROOT / "naive_internvl3_8b" / ds / "test_naive.jsonl" for ds in DATASETS},
    "Naive (Gemma-3-12B)":      {ds: ROOT / "naive_gemma3_vl_12b" / ds / "test_naive.jsonl" for ds in DATASETS},
    "Naive (LLaVA-OV-7B)":      {ds: ROOT / "naive_llava_ov_7b" / ds / "test_naive.jsonl" for ds in DATASETS},
    "MARS (Qwen3-VL-2B)":       {ds: ROOT / "mars_2b" / ds / "test_mars.jsonl" for ds in DATASETS},
    "Mod-HATE (8-shot)":        {ds: ROOT / "mod_hate" / ds / "test_mod_hate_8shot.jsonl" for ds in DATASETS},
    "LoReHM":                   {ds: ROOT / "lorehm" / ds / "test_lorehm.jsonl" for ds in DATASETS},
    "ALARM":                    {ds: ROOT / "alarm_backup_7b_20260416" / ds / "test_alarm.jsonl" for ds in DATASETS},
}


def eval_baseline(path, ds):
    if not path.exists():
        return None
    preds = {r["video_id"]: int(r.get("pred")) for r in ld_jsonl(path)
             if isinstance(r.get("pred"), int) or r.get("pred") in (0, 1)}
    labels = load_labels(ds)
    skip = SKIP_VIDEOS.get(ds, set())
    valid = [v for v in preds if v in labels and v not in skip and labels[v] in (0, 1)]
    if not valid:
        return None
    y = np.array([labels[v] for v in valid])
    yh = np.array([preds[v] for v in valid])
    return {
        "n": len(valid),
        "acc": accuracy_score(y, yh) * 100,
        "mf1": f1_score(y, yh, average="macro") * 100,
        "mp":  precision_score(y, yh, average="macro", zero_division=0) * 100,
        "mr":  recall_score(y, yh, average="macro", zero_division=0) * 100,
    }


def main():
    for name, paths in BASELINES.items():
        print(f"\n=== {name} ===")
        for ds in DATASETS:
            m = eval_baseline(paths[ds], ds)
            if m:
                print(f"  {ds:<16s} n={m['n']:>4d}  ACC={m['acc']:5.1f}  M-F1={m['mf1']:5.1f}  M-P={m['mp']:5.1f}  M-R={m['mr']:5.1f}")
            else:
                print(f"  {ds:<16s} (no data)")


if __name__ == "__main__":
    main()
