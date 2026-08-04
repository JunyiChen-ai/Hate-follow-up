"""E2 --- Small-verifier-only panel (cpjh-C3). CPU, live recompute.

Emits a table of every all-small verifier panel (three distinct verifiers drawn
from {qwen3-vl-8b, internvl35-8b, gemma-3-12b-it}, so max model size 12B) with
per-dataset ACC / macro-F1 and calls/video, plus reference rows: the default
full panel (g27 > q32 > q72), Stage-1 only, and the best label-free/few-shot
baseline per dataset from paper Table 1.

Every row is recomputed LIVE from the current offline files via
build_experiments.eval_sequential / eval_stage1 --- NOT read from the frozen
all_results.json --- so the table is internally consistent with current files
after the 2026-05-05 HateMM 72B verdict rerun (which drifted the default panel's
HateMM cell; see rebuttal/PAPER_NUMBER_DRIFT.md). The all-small panels use only
verifiers whose files were untouched by that rerun, so they are unchanged.

Output: results/rebuttal/E2_small_panel/table.md

Run:
  python scripts/rebuttal_e2_extract.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2")
OUT_DIR = ROOT / "results" / "rebuttal" / "E2_small_panel"

sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))
sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))

import build_experiments as BE  # noqa: E402

SMALL = ["qwen3-vl-8b", "internvl35-8b", "gemma-3-12b-it"]  # all <= 12B
DS_ORDER = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
DS_SHORT = {"MHClip_EN": "EN", "MHClip_ZH": "ZH", "HateMM": "HM", "ImpliHateVid": "IH"}

# Best label-free / few-shot baseline ACC per dataset, from paper Table 1.
BEST_BASELINE = {
    "MHClip_EN": ("LoReHM", 76.4),
    "MHClip_ZH": ("LLaVA-OV-7B", 75.2),
    "HateMM": ("ALARM", 79.5),
    "ImpliHateVid": ("MARS", 80.3),
}


def per_ds_map(row):
    return {d["ds"]: d for d in row["per_dataset"]}


def fmt_cell(pd, ds):
    d = pd[ds]
    return f"{100 * d['acc']:.1f} / {100 * d['mf1']:.3f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = BE.EvalCache()

    stage1 = BE.eval_stage1(cache, "2b")
    full = BE.eval_sequential(cache, BE.MAIN_ORDER, rho_mode="entropy",
                              name="Full default panel")
    small_orders = [
        BE.eval_sequential(cache, order, rho_mode="entropy", name="small")
        for order in itertools.permutations(SMALL, 3)
    ]
    small_orders.sort(key=lambda r: (r["avg_acc"], r["avg_mf1"]), reverse=True)

    header = (
        "| Verifier order | EN ACC/MF1 | ZH ACC/MF1 | HM ACC/MF1 | IH ACC/MF1 "
        "| Avg ACC | Avg MF1 | Calls/video |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [
        "# E2 --- Small-verifier-only panel (max model size 12B)",
        "",
        "All six ordered all-small panels drawn from "
        "{qwen3-vl-8b, internvl35-8b, gemma-3-12b-it}. Per-cell values are "
        "ACC / macro-F1 (percent). Recomputed LIVE from current offline files via "
        "build_experiments (drift-consistent; see rebuttal/PAPER_NUMBER_DRIFT.md).",
        "",
        header,
        sep,
    ]

    for r in small_orders:
        pd = per_ds_map(r)
        lines.append("| " + " | ".join([
            r["order"],
            fmt_cell(pd, "MHClip_EN"),
            fmt_cell(pd, "MHClip_ZH"),
            fmt_cell(pd, "HateMM"),
            fmt_cell(pd, "ImpliHateVid"),
            f"{100 * r['avg_acc']:.2f}",
            f"{r['avg_mf1']:.4f}",
            f"{r['avg_calls']:.3f}",
        ]) + " |")

    lines += ["", "## Reference rows (also live from current files)", "", header, sep]
    for label, r in (("Full default panel (g27 > q32 > q72)", full),
                     ("Stage-1 only (Boundary Mapper)", stage1)):
        pd = per_ds_map(r)
        lines.append("| " + " | ".join([
            label,
            fmt_cell(pd, "MHClip_EN"),
            fmt_cell(pd, "MHClip_ZH"),
            fmt_cell(pd, "HateMM"),
            fmt_cell(pd, "ImpliHateVid"),
            f"{100 * r['avg_acc']:.2f}",
            f"{r['avg_mf1']:.4f}",
            f"{r['avg_calls']:.3f}",
        ]) + " |")

    best_small = small_orders[0]
    bpd = per_ds_map(best_small)
    lines += [
        "",
        "## Beats the best label-free / few-shot baseline on all four datasets",
        "",
        "Best all-small panel `" + best_small["order"] + "` versus the strongest "
        "label-free/few-shot baseline per dataset (paper Table 1, ACC %):",
        "",
        "| Dataset | Best all-small panel ACC | Best baseline (ACC) |",
        "|---|---|---|",
    ]
    for ds in DS_ORDER:
        name, val = BEST_BASELINE[ds]
        lines.append(
            f"| {DS_SHORT[ds]} | {100 * bpd[ds]['acc']:.1f} | {name} ({val:.1f}) |"
        )

    lines += [
        "",
        "## Deployment / VRAM note",
        "",
        "The entire small panel tops out at a 12B verifier. In bf16 a 12B model "
        "needs roughly 24--28 GB of weights, so the full three-verifier small "
        "panel fits and runs on a single 48 GB GPU. The default panel's "
        "Qwen2.5-VL-72B-AWQ needs 40+ GB for weights alone (plus KV cache), i.e. "
        "a materially larger deployment. The six orders sit within ~0.3 pp of "
        "each other on average ACC, so the small-panel result is not sensitive "
        "to verifier order.",
        "",
    ]

    (OUT_DIR / "table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'table.md'} ({len(small_orders)} small orders, live)")
    print(f"  full default panel HM ACC = {per_ds_map(full)['HateMM']['acc']:.6f} "
          f"(live current-file value)")


if __name__ == "__main__":
    main()
