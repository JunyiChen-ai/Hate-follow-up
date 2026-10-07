#!/usr/bin/env python3
"""Collect the weakly supervised seed summaries into one table (CPU).

Reads runs/20261008_baselines/<method>/<Dataset>/summary.json (written by run_method.py, train_mil.py,
reuse_dehate.py, each from the per-seed metrics.json of the canonical evaluator) and writes
runs/20261008_baselines/weaksup_table.md and weaksup_table.json.  Rows whose summary is missing are listed as such.

    python experiments/20261008_baselines/weaksup_common/collect.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROWS = [("mil_bert", "mil_bert", "BERT + MIL (text)"), ("mil_wav2vec2", "mil_wav2vec2", "wav2vec2 + MIL (audio)"),
        ("mil_clip", "mil_clip", "CLIP + MIL (visual)"), ("vadclip", "vadclip", "VadCLIP"),
        ("vadclip", "vadclip_mlp", "VadCLIP, score_mlp"), ("dsanet", "dsanet", "DSANet"),
        ("dsanet", "dsanet_align", "DSANet, score_align"), ("avadclip", "avadclip", "AVadCLIP (audio-visual)"),
        ("avadclip", "avadclip_mlp", "AVadCLIP, score_mlp"), ("multihateloc", "multihateloc", "MultiHateLoc (reimpl.)")]
KEYS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")


def main():
    table, lines = {}, []
    head = "| method | " + " | ".join(f"{ds} ROC | PR | within" for ds in C.DATASET.values()) + " |"
    lines += [head, "|---|" + "---:|" * 9]
    for run, name, label in ROWS:
        cells = []
        for ds in C.DATASET.values():
            p = C.RUNS / run / ds / "summary.json"
            if not p.is_file():
                cells += ["—"] * 3
                continue
            e = json.loads(p.read_text())["methods"].get(name)
            table.setdefault(name, {})[ds] = {"source": str(p.relative_to(C.REPO)), **(e or {})}
            cells += [f"{e[k]['mean']:.4f} ± {e[k]['sd']:.4f}" if e else "—" for k in KEYS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (C.RUNS / "weaksup_table.md").write_text(
        "Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407; source = runs/20261008_baselines/<method>/<Dataset>/"
        "summary.json, built from each seed's metrics.json.\n\n" + text)
    (C.RUNS / "weaksup_table.json").write_text(json.dumps(table, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
