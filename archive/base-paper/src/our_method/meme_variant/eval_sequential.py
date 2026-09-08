from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .common import DATASETS, DEFAULT_ORDER, RESULT_ROOT, entropy, logit, macro_prf, parse_yes_no, read_jsonl, rho_from_hbar, sigmoid, write_jsonl
except ImportError:
    from common import DATASETS, DEFAULT_ORDER, RESULT_ROOT, entropy, logit, macro_prf, parse_yes_no, read_jsonl, rho_from_hbar, sigmoid, write_jsonl


def load_by_id(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path) if r.get("id")}


def eval_dataset(dataset: str, order: tuple[str, ...]) -> dict:
    boundary_root = RESULT_ROOT / "boundary" / dataset
    base = load_by_id(boundary_root / "baseline_preds.jsonl")
    band = load_by_id(boundary_root / "candidates_entropy_band.jsonl")
    if not base or not band:
        raise RuntimeError(f"{dataset}: missing boundary files under {boundary_root}")

    band_rows = list(band.values())
    hbar = float(band_rows[0].get("hbar", sum(float(r["entropy"]) for r in band_rows) / len(band_rows)))
    rho = rho_from_hbar(hbar)
    lam = math.log(rho / (1.0 - rho))
    judges = {tag: load_by_id(RESULT_ROOT / "judges" / dataset / f"test_{tag}.jsonl") for tag in order}

    final_rows = []
    y, yh = [], []
    calls_by_tag = {tag: 0 for tag in order}
    stopped_after = {tag: 0 for tag in order}
    invalid_judge = 0

    for sid in sorted(base):
        b0 = base[sid]
        b = band.get(sid, b0)
        ell = logit(float(b.get("posterior_hi", 0.5)))
        trace = []
        in_band = bool(b.get("in_band"))
        if in_band:
            for tag in order:
                calls_by_tag[tag] += 1
                jr = judges[tag].get(sid, {})
                pred = jr.get("pred")
                if pred not in (0, 1):
                    pred = parse_yes_no(jr.get("raw_response"))
                trace.append({"judge": tag, "pred": pred})
                if pred in (0, 1):
                    ell += (2 * int(pred) - 1) * lam
                    if entropy(sigmoid(ell)) <= hbar:
                        stopped_after[tag] += 1
                        break
                else:
                    invalid_judge += 1
        posterior = sigmoid(ell)
        final_pred = int(posterior >= 0.5) if in_band else int(b0["pred_baseline"])
        label = int(b0["label"])
        y.append(label)
        yh.append(final_pred)
        final_rows.append({
            "id": sid,
            "dataset": dataset,
            "label": label,
            "score": float(b0["score"]),
            "pred_baseline": int(b0["pred_baseline"]),
            "in_band": in_band,
            "posterior_final": float(posterior),
            "pred_final": final_pred,
            "trace": trace,
        })

    out_root = RESULT_ROOT / "final" / dataset
    write_jsonl(out_root / "test_sequential.jsonl", final_rows)
    acc = sum(1 for a, p in zip(y, yh) if a == p) / len(y) if y else 0.0
    mf1, mp, mr = macro_prf(y, yh)
    summary = {
        "dataset": dataset,
        "n": len(y),
        "hbar": hbar,
        "rho_D": rho,
        "acc": acc,
        "macro_f1": mf1,
        "macro_precision": mp,
        "macro_recall": mr,
        "n_in_band": sum(1 for r in final_rows if r["in_band"]),
        "calls_by_tag": calls_by_tag,
        "stopped_after": stopped_after,
        "invalid_judge_outputs": invalid_judge,
        "output": str(out_root / "test_sequential.jsonl"),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_summary(summaries: list[dict]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    avg = {}
    for key in ("acc", "macro_f1", "macro_precision", "macro_recall"):
        avg[key] = sum(float(s[key]) for s in summaries) / len(summaries) if summaries else 0.0
    blob = {"datasets": summaries, "average": avg}
    (RESULT_ROOT / "summary.json").write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Meme Variant Sequential Results", ""]
    lines.append("| Dataset | N | In band | ACC | Macro F1 | Macro P | Macro R | rho_D |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s['dataset']} | {s['n']} | {s['n_in_band']} | {100*s['acc']:.2f} | "
            f"{s['macro_f1']:.4f} | {s['macro_precision']:.4f} | {s['macro_recall']:.4f} | {s['rho_D']:.4f} |"
        )
    lines.append(f"| Average | - | - | {100*avg['acc']:.2f} | {avg['macro_f1']:.4f} | {avg['macro_precision']:.4f} | {avg['macro_recall']:.4f} | - |")
    (RESULT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian sequential verifier eval for harmful meme variant")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--order", default=",".join(DEFAULT_ORDER), help="Comma-separated judge tags")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    order = tuple(x.strip() for x in args.order.split(",") if x.strip())
    summaries = [eval_dataset(ds, order) for ds in datasets]
    write_summary(summaries)
    print(json.dumps({"datasets": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
