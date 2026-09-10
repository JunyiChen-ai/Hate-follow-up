#!/usr/bin/env python3
"""Family / size robustness tables from runs/20260910_spvl/mllm/<tag>/<arm>/metrics_*.json (README §11).

Table A (Q1 / Q2): per model, the MLLM alone (whole-video verdict only; per-window independent judgement)
and the full SPVL-r2 pipeline. Table B (Q3): per model, each ablation's change relative to that model's
full run (within-video macro ROC; M3 column = pooled PR change). Writes runs/20260910_spvl/mllm_table.md.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20260910_spvl/mllm"
DS = ["HateMM", "HateClipSeg"]
MODELS = [("q3vl-8b", "Qwen3-VL-8B (current)"), ("q3vl-2b", "Qwen3-VL-2B"), ("q3vl-4b", "Qwen3-VL-4B"),
          ("q3vl-32b", "Qwen3-VL-32B"), ("q25vl-7b", "Qwen2.5-VL-7B"), ("internvl35-8b", "InternVL3.5-8B"),
          ("llava-ov-7b", "LLaVA-OneVision-7B"), ("gemma3-12b", "Gemma-3-12B")]
ABL = [("winonly", "per-window alone"), ("noctx", "− transcript context"), ("noframes", "− frames"),
       ("asr", "ASR segments"), ("nostance", "− stance"), ("joint", "joint branch")]


def load(tag, arm, comp):
    f = R / tag / arm / f"metrics_{comp}.json"
    if not f.exists():
        return None
    return {p["dataset"]: p for p in json.load(open(f))["per_dataset"]}


def cell(m, ds):
    if m is None or ds not in m:
        return "—"
    p = m[ds]
    return f"{p['frame_ROC_AUC']:.4f} / {p['frame_PR_AUC']:.4f} / {p['within_video_macro_ROC_AUC']:.4f}"


def delta(a, b, ds, key):
    if a is None or b is None or ds not in a or ds not in b:
        return "—"
    return f"{a[ds][key] - b[ds][key]:+.3f}"


def extra(tag):
    f = R / tag / "full" / "predictions.jsonl"
    if not f.exists():
        return "—", "—"
    toks, pre = [], []
    for line in open(f):
        r = json.loads(line)
        if r.get("error"):
            continue
        e = r["extra"]
        if e.get("img_tokens"):
            toks.append(e["img_tokens"][0])
        pre.append(e.get("prefix_tokens", 0))
    import statistics as st
    return (str(int(st.median(toks))) if toks else "0"), (str(int(st.median(pre))) if pre else "—")


out = []
out.append("## Table A — the MLLM alone vs the full pipeline (pooled ROC / pooled PR / within)\n")
out.append("| model | tokens/frame | prefix tokens (median) | " + " | ".join(
    f"{d}: verdict only / per-window alone / SPVL-r2" for d in DS) + " |")
out.append("|---|---|---|" + "---|" * len(DS))
for tag, name in MODELS:
    full = load(tag, "full", "izv_plus_mean_rrank")
    only = load(tag, "full", "ispvl_rnone")
    win = load(tag, "winonly", "ispvl_rrank")
    tpf, pre = extra(tag)
    cells = [f"{cell(only, d)} <br> {cell(win, d)} <br> **{cell(full, d)}**" for d in DS]
    out.append(f"| {name} | {tpf} | {pre} | " + " | ".join(cells) + " |")
out.append("\nverdict only = z_video intercept, no residual (within = .5 by construction); per-window alone = joint branch, "
           "rules question, no context, no frames, no stance, z_video intercept + rank residual.\n")
out.append("## Table B — ablations relative to each model's full run (Δ within; last column Δ pooled PR of M3)\n")
out.append("| model | " + " | ".join(f"{d}: " + " / ".join(a[1] for a in ABL) + " / − M3 (ΔPR)" for d in DS) + " |")
out.append("|---|" + "---|" * len(DS))
for tag, name in MODELS:
    full = load(tag, "full", "izv_plus_mean_rrank")
    nom3 = load(tag, "full", "ispvl_rrank")
    cells = []
    for d in DS:
        parts = [delta(load(tag, arm, "izv_plus_mean_rrank" if arm != "winonly" else "ispvl_rrank"), full, d,
                       "within_video_macro_ROC_AUC") for arm, _ in ABL]
        parts.append(delta(nom3, full, d, "frame_PR_AUC"))
        cells.append(" / ".join(parts))
    out.append(f"| {name} | " + " | ".join(cells) + " |")
out.append("\nNegative = removing the part lowers the metric (the part helps). Noise floor: within .01, pooled .005.\n")
text = "\n".join(out)
(ROOT / "runs/20260910_spvl/mllm_table.md").write_text(text)
print(text)
