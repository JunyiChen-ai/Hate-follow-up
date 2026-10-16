"""Shared constants of candidate 46 (README §2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_manifest  # noqa: E402

EXP = ROOT / "runs/20261017_m1_topic_removal"
R6 = ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"
R6_BMA = ROOT / "runs/20260926_twolevel/r6_bma/metrics.json"
DIRECTIONS = ROOT / "data/topic_direction/qwen3vl8b/directions_body.pt"
VALIDITY = EXP / "directions/validity.json"
WINDOW_S = 8.0
NATIVE_FRAMES = 20
N_SHUF = 5
N_RAND = 1
SHUF = lambda k: ROOT / f"data/topic_direction/qwen3vl8b/directions_shuf{k}.pt"  # noqa: E731
ARMS = ["abl"] + [f"shuf{k}" for k in range(N_SHUF)] + [f"rand{k}" for k in range(N_RAND)]
IMPLEMENTATION = "candidate 46 topic-direction removal v2: directional ablation of the synthetic-bank topic direction (non-hostile carriers) at every residual point of the speech branch; 5 label-shuffled controls orthogonalised to the topic direction + 1 Gaussian control; 2026-10-17"


def selected_rows(datasets, smoke):
    rows = load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", datasets)
    if smoke:
        rows = [r for ds in datasets for r in [x for x in rows if x["dataset"] == ds][:2]] + [
            r for r in rows if r["dataset"] == "HateMM" and r["video_id"] == "hate_video_114"]
    return rows


def stored_r6():
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(R6)) if not r.get("error")}
