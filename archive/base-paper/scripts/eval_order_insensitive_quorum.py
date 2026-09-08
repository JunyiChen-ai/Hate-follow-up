#!/usr/bin/env python3
"""Evaluate order sensitivity for the Stage-2 verifier panel.

This script is CPU-only and uses existing Stage-1 scores, entropy-band files,
and offline verifier outputs. It compares the current one-call early stopping
rule with a minimum-two-verifier quorum rule that removes order dependence for
a fixed 3-verifier set.
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3")
OUT = ROOT / "paper" / "analysis"

sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

import build_experiments as exp  # noqa: E402
from grid_eval_all import DS, N_TEST  # noqa: E402


SETS = {
    "G27_Q32_InternVL": (
        "gemma-3-27b-it",
        "qwen2.5-vl-32b-awq",
        "internvl35-8b",
    ),
    "G27_Q32_Q3": (
        "gemma-3-27b-it",
        "qwen2.5-vl-32b-awq",
        "qwen3-vl-8b",
    ),
    "G27_Q32_Q72": (
        "gemma-3-27b-it",
        "qwen2.5-vl-32b-awq",
        "qwen2.5-vl-72b-awq",
    ),
}

PRETTY = {
    "gemma-3-27b-it": "G27",
    "qwen2.5-vl-32b-awq": "Q32",
    "internvl35-8b": "InternVL",
    "qwen3-vl-8b": "Q3",
    "qwen2.5-vl-72b-awq": "Q72",
}


def eval_order(order: tuple[str, str, str], min_calls: int) -> dict:
    cache = exp.EvalCache()
    per = []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for ds in DS:
        labels = cache.load_labels_for(ds)
        base = cache.load_base("2b", ds)
        band = cache.load_band("2b", ds)
        hbar = cache.hbar[("2b", ds)]
        rho = cache.rhod[("2b", ds)]
        lam = math.log(rho / (1.0 - rho))
        judges = [cache.load_judge(j, ds) for j in order]

        y, yh = [], []
        calls = 0
        for v in cache.valid_vids("2b", ds):
            y.append(labels[v])
            s1 = base[v]
            b = band.get(v, {})
            if not bool(b.get("in_band")):
                yh.append(s1)
                counts[0] += 1
                continue

            ell = exp.logit(float(b.get("posterior_hi", 0.5)))
            used = 0
            for table in judges:
                used += 1
                calls += 1
                r = table.get(v, {}).get("pred")
                if r in (0, 1):
                    ell += (2 * int(r) - 1) * lam
                    if used >= min_calls and exp.ent(exp.sigmoid(ell)) <= hbar:
                        break
            counts[used] += 1
            yh.append(1 if exp.sigmoid(ell) >= 0.5 else 0)

        acc = sum(1 for a, b in zip(y, yh) if a == b) / N_TEST[ds]
        mf1, mp, mr = exp.macro_prf(y, yh)
        per.append(
            {
                "ds": ds,
                "acc": acc,
                "mf1": mf1,
                "mp": mp,
                "mr": mr,
                "calls": calls / len(y),
            }
        )

    row = exp.summarize_method("quorum", per)
    row.update({f"counts_{k}": v for k, v in counts.items()})
    return row


def flatten(panel_name: str, order: tuple[str, str, str], min_calls: int, r: dict) -> dict:
    out = {
        "panel": panel_name,
        "order": " > ".join(PRETTY[x] for x in order),
        "min_calls": min_calls,
        "avg_acc": r["avg_acc"],
        "avg_mf1": r["avg_mf1"],
        "avg_calls": r["avg_calls"],
        "counts_0": r["counts_0"],
        "counts_1": r["counts_1"],
        "counts_2": r["counts_2"],
        "counts_3": r["counts_3"],
    }
    for d in r["per_dataset"]:
        out[f"{d['ds']}_acc"] = d["acc"]
        out[f"{d['ds']}_mf1"] = d["mf1"]
        out[f"{d['ds']}_calls"] = d["calls"]
    return out


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_lines = [
        "# Order-Insensitive Quorum Analysis",
        "",
        "CPU-only evaluation using existing offline verifier outputs. "
        "`min_calls=1` is the current one-call early stop; `min_calls=2` "
        "requires a two-verifier quorum before stopping.",
        "",
    ]

    for panel_name, panel in SETS.items():
        summary_lines.append(f"## {panel_name}")
        for min_calls in (1, 2):
            panel_rows = []
            for order in itertools.permutations(panel, 3):
                result = eval_order(order, min_calls=min_calls)
                row = flatten(panel_name, order, min_calls, result)
                rows.append(row)
                panel_rows.append(row)

            accs = [100 * float(r["avg_acc"]) for r in panel_rows]
            mf1s = [100 * float(r["avg_mf1"]) for r in panel_rows]
            calls = [float(r["avg_calls"]) for r in panel_rows]
            best = max(panel_rows, key=lambda r: (float(r["avg_acc"]), float(r["avg_mf1"])))
            summary_lines.extend(
                [
                    f"- min_calls={min_calls}: "
                    f"ACC range {min(accs):.2f}-{max(accs):.2f}; "
                    f"MF1 range {min(mf1s):.2f}-{max(mf1s):.2f}; "
                    f"mean calls {sum(calls) / len(calls):.3f}.",
                    f"  Best/representative: {best['order']} "
                    f"ACC={pct(float(best['avg_acc']))}, "
                    f"MF1={pct(float(best['avg_mf1']))}, "
                    f"calls={float(best['avg_calls']):.3f}.",
                ]
            )
        summary_lines.append("")

    fields = [
        "panel",
        "order",
        "min_calls",
        "avg_acc",
        "avg_mf1",
        "avg_calls",
        "counts_0",
        "counts_1",
        "counts_2",
        "counts_3",
    ]
    for ds in DS:
        fields.extend([f"{ds}_acc", f"{ds}_mf1", f"{ds}_calls"])

    csv_path = OUT / "order_insensitive_quorum.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUT / "order_insensitive_quorum.md"
    md_path.write_text("\n".join(summary_lines) + "\n")

    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
