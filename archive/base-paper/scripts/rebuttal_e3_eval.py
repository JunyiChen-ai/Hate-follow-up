"""E3 eval --- zero-shot verifier control rows (cpjh-C1, 7rqV-C3). CPU-only.

Merges the frozen offline verdict files with the E3 fills, then scores each
verifier as a plain single-pass zero-shot classifier on all four full test sets
with the canonical eval_one protocol (SKIP_VIDEOS, collapse_label, unparseable
-> normal). The headline row is Qwen2.5-VL-72B-AWQ; other verifiers are emitted
too for context.

Provenance caveat (E3a): eval_one on the frozen offline verdict files reproduces
the paper Table-1 zero-shot rows EXACTLY for ImpliHateVid (ih-prompt files), but
NOT for EN / ZH / HateMM --- those Table-1 rows were produced by a different
zero-shot pass. The rows here are therefore labelled "verifier prompt,
single-pass": internally consistent across verifiers, and Table-1-identical only
for ImpliHateVid.

Output: results/rebuttal/E3_72b_zeroshot/{eval.json, table.md,
merged_<tag>_<ds>.jsonl}

Run:
  python scripts/rebuttal_e3_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3")
OUT_DIR = ROOT / "results" / "rebuttal" / "E3_72b_zeroshot"

sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

from grid_eval_all import judge_path, ld_jsonl  # noqa: E402
from eval_generative_predictions import eval_one  # noqa: E402

DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
DS_SHORT = {"MHClip_EN": "EN", "MHClip_ZH": "ZH", "HateMM": "HM", "ImpliHateVid": "IH"}
DEFAULT_MODELS = [
    "qwen2.5-vl-72b-awq", "qwen2.5-vl-32b-awq", "gemma-3-27b-it",
    "internvl35-8b", "qwen3-vl-8b", "gemma-3-12b-it",
]


def merge(tag: str, ds: str) -> tuple[Path, int, int]:
    """Write merged frozen+fills verdicts to a file; return (path, n_frozen, n_fill)."""
    frozen_p = judge_path(tag, ds)
    frozen = ld_jsonl(frozen_p) if (frozen_p and frozen_p.exists()) else []
    fills_p = OUT_DIR / f"fills_{tag}_{ds}.jsonl"
    fills = ld_jsonl(fills_p) if fills_p.exists() else []
    seen, rows = set(), []
    for r in list(frozen) + list(fills):  # frozen first; fills only add missing
        vid = r.get("video_id")
        if vid is None or vid in seen:
            continue
        seen.add(vid)
        rows.append(r)
    out = OUT_DIR / f"merged_{tag}_{ds}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out, len(frozen), len(fills)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in args.models.split(",") if t.strip()]

    results = {}
    for tag in tags:
        results[tag] = {}
        for ds in DATASETS:
            merged, n_frozen, n_fill = merge(tag, ds)
            m = eval_one(str(merged), ds)
            results[tag][ds] = {
                "n_total": m["n_total"], "n_frozen": n_frozen, "n_fill": n_fill,
                "n_unparseable": m.get("n_unparseable", 0),
                "acc": m["acc"], "mf1": m["mf"],
            }
            # macro P / R from confusion counts (eval_one returns tp/fp/fn/tn).
            tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
            p_pos = tp / (tp + fp) if (tp + fp) else 0.0
            r_pos = tp / (tp + fn) if (tp + fn) else 0.0
            p_neg = tn / (tn + fn) if (tn + fn) else 0.0
            r_neg = tn / (tn + fp) if (tn + fp) else 0.0
            results[tag][ds]["mp"] = 0.5 * (p_pos + p_neg)
            results[tag][ds]["mr"] = 0.5 * (r_pos + r_neg)

    # Table-1-comparable row: EN/ZH/HM from the generic-def zero-shot CoT pass
    # (Fig. resolver_prompt), IH from the merged offline_test_ih verdicts.
    table1 = {}
    for tag in tags:
        row = {}
        for ds in DATASETS:
            if ds == "ImpliHateVid":
                src, n_frozen, n_fill = merge(tag, ds)  # offline_test_ih + fills
                note = "offline_test_ih (IH_DEF)"
            else:
                src = OUT_DIR / f"zeroshot_cot_{tag}_{ds}.jsonl"
                n_frozen, n_fill = (0, 0)
                note = "zeroshot_cot (generic HATEMM_DEF)"
                if not src.exists():
                    row[ds] = {"status": "MISSING zeroshot_cot file --- run rebuttal_e3_zeroshot72b.py",
                               "note": note}
                    continue
            m = eval_one(str(src), ds)
            tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
            p_pos = tp / (tp + fp) if (tp + fp) else 0.0
            r_pos = tp / (tp + fn) if (tp + fn) else 0.0
            p_neg = tn / (tn + fn) if (tn + fn) else 0.0
            r_neg = tn / (tn + fp) if (tn + fp) else 0.0
            row[ds] = {
                "n_total": m["n_total"], "acc": m["acc"], "mf1": m["mf"],
                "mp": 0.5 * (p_pos + p_neg), "mr": 0.5 * (r_pos + r_neg),
                "n_unparseable": m.get("n_unparseable", 0), "source": note,
            }
        table1[tag] = row
    results = {"single_pass_verifier_prompt": results, "table1_comparable": table1}

    (OUT_DIR / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# E3 --- Zero-shot verifier control rows (verifier prompt, single-pass)",
        "",
        "Each verifier scored as a plain single-pass classifier on the full test "
        "sets (frozen offline verdicts + E3 fills), canonical eval_one protocol "
        "with SKIP_VIDEOS. Cells are ACC / M-F1 / M-P / M-R (percent). Headline "
        "control: Qwen2.5-VL-72B-AWQ.",
        "",
        "Provenance (E3a): identical to the paper Table-1 zero-shot rows for "
        "ImpliHateVid (ih-prompt) but not for EN / ZH / HateMM; see the eval "
        "script header.",
        "",
        "| Verifier | EN | ZH | HM | IH | coverage (filled) |",
        "|---|---|---|---|---|---|",
    ]

    def cell(d):
        def pc(x):
            return f"{100 * x:.1f}" if x is not None else "--"
        return f"{pc(d['acc'])} / {pc(d['mf1'])} / {pc(d['mp'])} / {pc(d['mr'])}"

    sp = results["single_pass_verifier_prompt"]
    for tag in tags:
        cov = "; ".join(
            f"{DS_SHORT[ds]} {sp[tag][ds]['n_total']}(+{sp[tag][ds]['n_fill']})"
            for ds in DATASETS
        )
        row = "| " + " | ".join(
            [tag] + [cell(sp[tag][ds]) for ds in DATASETS] + [cov]
        ) + " |"
        lines.append(row)

    # Table-1-comparable section (generic-def zero-shot CoT for EN/ZH/HM).
    lines += [
        "",
        "## Table-1-comparable zero-shot row (Fig. resolver_prompt protocol)",
        "",
        "EN / ZH / HM from the generic-def zero-shot CoT pass "
        "(rebuttal_e3_zeroshot72b.py); IH from offline_test_ih (already "
        "Table-1-matched). Cells ACC / M-F1 / M-P / M-R (percent).",
        "",
        "| Verifier | EN | ZH | HM | IH |",
        "|---|---|---|---|---|",
    ]

    def cell1(d):
        if "status" in d:
            return d["status"]
        def pc(x):
            return f"{100 * x:.1f}" if x is not None else "--"
        return f"{pc(d['acc'])} / {pc(d['mf1'])} / {pc(d['mp'])} / {pc(d['mr'])}"

    for tag in tags:
        r = table1.get(tag, {})
        cells = [cell1(r[ds]) if ds in r else "--" for ds in DATASETS]
        lines.append("| " + " | ".join([tag] + cells) + " |")

    (OUT_DIR / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'eval.json'} and {OUT_DIR / 'table.md'}")


if __name__ == "__main__":
    main()
