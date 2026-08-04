#!/usr/bin/env python3
"""Collect validation metrics for produced baseline outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data/jehc223/EMNLP2")
sys.path.insert(0, str(PROJECT_ROOT / "src" / "naive_baseline"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "our_method"))

from eval_generative_predictions import eval_one  # noqa: E402
from data_utils import load_clean_split_ids  # noqa: E402

DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
JUDGES = [
    ("qwen3-vl-8b", "qwen3-vl-8b"),
    ("gemma-3-12b-it", "gemma-3-12b-it"),
    ("gemma-3-27b-it", "gemma-3-27b-it"),
    ("qwen2.5-vl-32b-awq", "qwen2.5-vl-32b-awq"),
    ("qwen2.5-vl-72b-awq", "qwen2.5-vl-72b-awq"),
    ("internvl35-8b", "internvl35-8b"),
    ("llava-onevision-qwen2-7b-ov-hf", "llava-onevision-qwen2-7b-ov-hf"),
    ("minicpm-v-26", "minicpm-v-26"),
]


def candidate_outputs():
    for ds in DATASETS:
        for judge, tag in JUDGES:
            ih = "_ih" if ds == "ImpliHateVid" else ""
            yield {
                "method": "offline_judge",
                "model": judge,
                "dataset": ds,
                "path": PROJECT_ROOT / "results" / "boundary_rescue" / ds / f"offline_validation{ih}_{tag}.jsonl",
            }
        yield {
            "method": "mars_32b_awq",
            "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
            "dataset": ds,
            "path": PROJECT_ROOT / "results" / "mars_32b_awq" / ds / "validation_mars.jsonl",
        }
        yield {
            "method": "alarm_7b",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "dataset": ds,
            "path": PROJECT_ROOT / "results" / "alarm_validation_7b" / ds / "validation_alarm.jsonl",
        }
        yield {
            "method": "lorehm_32b_awq",
            "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
            "dataset": ds,
            "path": PROJECT_ROOT / "results" / "lorehm" / ds / "validation_lorehm.jsonl",
        }
        for k in (4, 8):
            yield {
                "method": f"mod_hate_{k}shot",
                "model": "yahma/llama-7b-hf",
                "dataset": ds,
                "path": PROJECT_ROOT / "results" / "mod_hate" / ds / f"validation_mod_hate_{k}shot.jsonl",
            }


def main() -> int:
    rows = []
    for info in candidate_outputs():
        path = Path(info["path"])
        n_expected = len(load_clean_split_ids(info["dataset"], "validation"))
        row = {
            **info,
            "path": str(path),
            "split": "validation",
            "n_expected": n_expected,
            "exists": path.exists(),
        }
        if path.exists():
            metrics = eval_one(str(path), info["dataset"])
            row.update(metrics)
        rows.append(row)
    out_dir = PROJECT_ROOT / "results" / "validation_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "validation_summary.json"
    out_md = out_dir / "validation_summary.md"
    with out_json.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    lines = ["# Validation Summary", "", "| method | dataset | exists | n_expected | n_eval | acc | mF1 | path |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        acc = "" if r.get("acc") is None else f"{r['acc']:.4f}"
        mf = "" if r.get("mf") is None else f"{r['mf']:.4f}"
        lines.append(
            f"| {r['method']} | {r['dataset']} | {r['exists']} | "
            f"{r.get('n_expected', '')} | {r.get('n_total', '')} | {acc} | {mf} | `{r['path']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
