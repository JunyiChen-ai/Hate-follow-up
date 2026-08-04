from __future__ import annotations

import argparse
import json

from .common import DATASETS, RESULT_ROOT, compute_metrics, read_jsonl


DEFAULT_OUTPUTS = {
    "naive_2b": "test_naive.jsonl",
    "mars_32b_awq": "test_mars.jsonl",
    "alarm_7b": "test_alarm.jsonl",
    "lorehm_32b_awq": "test_lorehm.jsonl",
    "mod_hate": "test_mod_hate_4shot.jsonl",
    "mod_hate_8shot": "test_mod_hate_8shot.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=str(RESULT_ROOT / "summary.json"))
    parser.add_argument("--out-md", default=str(RESULT_ROOT / "summary.md"))
    args = parser.parse_args()

    summary = {}
    lines = ["# Meme Baseline Results", "", "| baseline | dataset | n | ACC | MF1 | MP | MR | invalid |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for baseline, filename in DEFAULT_OUTPUTS.items():
        base_dir = "mod_hate" if baseline == "mod_hate_8shot" else baseline
        summary[baseline] = {}
        for ds in DATASETS:
            path = RESULT_ROOT / base_dir / ds / filename
            rows = read_jsonl(path)
            m = compute_metrics(rows)
            summary[baseline][ds] = {"path": str(path), **m}
            def fmt(x):
                return "--" if x is None else f"{100*x:.2f}"
            lines.append(
                f"| {baseline} | {ds} | {m.get('n', 0)} | {fmt(m.get('acc'))} | "
                f"{fmt(m.get('mf1'))} | {fmt(m.get('mp'))} | {fmt(m.get('mr'))} | {m.get('n_invalid', 0)} |"
            )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

