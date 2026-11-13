#!/usr/bin/env python3
"""Label-free rows of the main table next to the naive-prompt rows (README §1, comparison rule).

Reads only metrics.json written by src/eval/evaluate_four_datasets.py. Published rows are the variants the main table
uses (runs/20261008_baselines/<method>_av, i.e. with speech); the old naive rows are the ones being replaced.
Writes runs/20261113_lf_baselines/naive/compare.{md,json}: per corpus and metric, every row and the best cell among
(a) the published rows, including the re-runs on Qwen3-VL-8B (README §2, primary read-outs only), (b) the new naive
rows, and which group holds the maximum.
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(REPO, "runs", "20261008_baselines")
Q = os.path.join(REPO, "runs", "20261113_lf_baselines")
N = os.path.join(Q, "naive")
DATASETS = ("HateMM", "HateClipSeg", "DeHate")
KEYS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
ROWS = [  # (group, name, metrics path pattern)
    ("published", "ZS-CLIP", B + "/zs_clip_av/{ds}/metrics.json"),
    ("published", "ZS-ImageBind", B + "/zs_imagebind_av/{ds}/metrics.json"),
    ("published", "LAVAD", B + "/lavad_av/{ds}/metrics.json"),
    ("published", "EventVAD", B + "/eventvad_av/{ds}/metrics.json"),
    ("published", "PRISM", B + "/prism_av/{ds}/metrics.json"),
    ("published", "VideoMind", B + "/videomind_av/{ds}/metrics.json"),
    ("old naive (our rules)", "Qwen3-8B, rules prompt", B + "/qwen3_8b_text/{ds}/metrics.json"),
    ("old naive (our rules)", "Qwen2.5-VL-7B, rules prompt, transcript", B + "/qwen25vl_winonly/raw/{ds}/metrics.json"),
    ("new naive", "Qwen3-8B, plain question", N + "/llm/{ds}/metrics.json"),
    ("new naive", "Qwen3-VL-8B, plain question, frames + transcript", N + "/vlm/{ds}/metrics.json"),
    ("new naive (check)", "Qwen3-VL-8B, plain question, transcript only", N + "/vlm_text/{ds}/metrics.json"),
    ("published, Qwen3-VL-8B", "MARS, fp32 label log-odds", Q + "/mars/{ds}/prob/metrics.json"),
    ("published, Qwen3-VL-8B (check)", "MARS, 0/1 label", Q + "/mars/{ds}/label/metrics.json"),
    ("published, Qwen3-VL-8B", "EventVAD, fp32 expected score", Q + "/eventvad/{ds}/prob/metrics.json"),
    ("published, Qwen3-VL-8B (check)", "EventVAD, stated score", Q + "/eventvad/{ds}/text/metrics.json"),
]
PUBLISHED = ("published", "published, Qwen3-VL-8B")


def read(path, ds):
    path = path.format(ds=ds)
    if not os.path.isfile(path):
        return None
    for r in json.load(open(path))["per_dataset"]:
        if r.get("dataset") == ds:
            return [r[k] for k in KEYS], path
    return None


def main():
    table, lines = {}, []
    lines.append("| group | row | " + " | ".join("%s ROC / PR / within" % d for d in DATASETS) + " |")
    lines.append("|---|---|" + "---|" * len(DATASETS))
    for g, name, pat in ROWS:
        cells = []
        for ds in DATASETS:
            got = read(pat, ds)
            table.setdefault(name, {"group": g})[ds] = {"values": got[0], "source": got[1]} if got else None
            cells.append(" / ".join("%.1f" % (100 * x) for x in got[0]) if got else "—")
        lines.append("| %s | %s | %s |" % (g, name, " | ".join(cells)))
    lines.append("")
    lines.append("Best cell per corpus and metric: published rows (main table and re-run on Qwen3-VL-8B, primary read-outs) vs new naive rows (the old naive rows and the check rows are left out).")
    lines.append("")
    best = {}
    for ds in DATASETS:
        for k, key in enumerate(KEYS):
            cand = {}
            for name, d in table.items():
                if d["group"] in PUBLISHED + ("new naive",) and d.get(ds):
                    cand[name] = d[ds]["values"][k]
            if not cand:
                continue
            top = max(cand, key=cand.get)
            pub = max((v, n) for n, v in cand.items() if table[n]["group"] in PUBLISHED)
            nai = max(((v, n) for n, v in cand.items() if table[n]["group"] == "new naive"), default=(None, None))
            best["%s %s" % (ds, key)] = {"top": top, "group": table[top]["group"], "published": pub, "naive": nai}
            lines.append("- %s %s: best published %.1f (%s), best new naive %s; top = %s" % (
                ds, key.replace("frame_", "pooled ").replace("within_video_macro_ROC_AUC", "within"), 100 * pub[0],
                pub[1], ("%.1f (%s)" % (100 * nai[0], nai[1])) if nai[0] is not None else "—", table[top]["group"]))
    os.makedirs(N, exist_ok=True)
    with open(os.path.join(N, "compare.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(N, "compare.json"), "w") as fh:
        json.dump({"rows": table, "best": best}, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
