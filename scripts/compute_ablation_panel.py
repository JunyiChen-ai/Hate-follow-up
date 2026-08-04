"""Compute real numbers for Table 2 (ablation rows R0-R4) and Table 4 (panel robustness).

All rows reuse the frozen Qwen3-VL-2B stage-1 (TrainFit baseline_preds) and the frozen
uncertainty band from candidates_entropy_band_2b.jsonl. Panel variants reuse the
offline_test_<judge>.jsonl files which cover the full test set.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from boundary_rescue.grid_eval_all import (
    baseline_pred_path, entropy_band_path, judge_path,
    load_labels, SKIP_VIDEOS, ld_jsonl,
)

SLUG = "2b"
DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
CRITERION = {"MHClip_EN": "otsu", "MHClip_ZH": "gmm", "HateMM": "li_lee", "ImpliHateVid": "gmm"}

HEADLINE_TRIPLET = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "llava-onevision-qwen2-7b-ov-hf")
# Panel variants for Table 4
PANEL_VARIANTS = {
    "cross_family_headline": ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "llava-onevision-qwen2-7b-ov-hf"),
    "cross_family_swap":     ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "internvl35-8b"),
    "single_strongest":      ("qwen2.5-vl-32b-awq",),
    "same_family_qwen":      ("qwen3-vl-8b", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq"),
}
# All 8 judges for R4 (8-judge majority)
ALL_JUDGES_8 = (
    "gemma-3-12b-it", "gemma-3-27b-it",
    "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq",
    "internvl35-8b", "llava-onevision-qwen2-7b-ov-hf",
    "minicpm-v-26", "qwen3-vl-8b",
)

OFFLINE_DIR = Path("/data/jehc223/EMNLP2/results/boundary_rescue")


def _offline_judge(ds, judge_stem):
    p = judge_path(judge_stem, ds)
    if p is None or not p.exists():
        return {}
    return {r["video_id"]: r.get("pred") for r in ld_jsonl(p)
            if r.get("pred") in (0, 1)}


def _baseline_map(ds):
    crit = CRITERION[ds]
    bp = baseline_pred_path(SLUG, ds, crit)
    if not bp.exists():
        bp = baseline_pred_path(SLUG, ds, "protocol")
    return {r["video_id"]: int(r["pred_baseline"]) for r in ld_jsonl(bp)}


def _band_set(ds):
    rows = ld_jsonl(entropy_band_path(SLUG, ds))
    return {r["video_id"] for r in rows if r.get("in_band")}


def _panel_majority(votes, *, single_ok: bool = False):
    votes = [v for v in votes if v in (0, 1)]
    if not votes:
        return None
    if len(votes) < 2 and not single_ok:
        return None
    c = Counter(votes)
    one = c.get(1, 0)
    zero = c.get(0, 0)
    if one > zero:
        return 1
    if zero > one:
        return 0
    return None  # tie


def _acc(preds, labels, valid):
    hits = sum(1 for v in valid if preds[v] == labels[v])
    return 100.0 * hits / len(valid)


def evaluate_row(ds, *, use_band: bool, judges: tuple[str, ...] | None):
    """Return accuracy for one configuration on one dataset.

    If judges is None → stage-1 only.
    If use_band → apply panel only to band videos; else apply to every video.
    """
    base = _baseline_map(ds)
    labels = load_labels(ds)
    skip = SKIP_VIDEOS.get(ds, set())
    valid = [v for v in base if v not in skip and labels.get(v) in (0, 1)]
    preds = {v: base[v] for v in valid}

    if judges:
        judge_maps = [_offline_judge(ds, j) for j in judges]
        targets = _band_set(ds) if use_band else set(valid)
        single = len(judges) == 1
        for v in valid:
            if v not in targets:
                continue
            votes = [jm.get(v) for jm in judge_maps]
            mv = _panel_majority(votes, single_ok=single)
            if mv is not None:
                preds[v] = mv
    return _acc(preds, labels, valid)


def run_table2_ablation():
    print("\n=== Table 2 Ablation ===")
    print(f"{'Row':<40s} {'HM':>6s} {'EN':>6s} {'ZH':>6s} {'IH':>6s} {'Avg':>6s}")
    rows = [
        ("R0 Full TRIAGE (band + 3-family)",   True,  HEADLINE_TRIPLET),
        ("R1 Stage-1 only",                    False, None),
        ("R2 Band off, panel on all",          False, HEADLINE_TRIPLET),
        ("R3 Band on, 1 strongest judge",      True,  ("qwen2.5-vl-32b-awq",)),
        ("R4 Band on, 8-judge majority",       True,  ALL_JUDGES_8),
    ]
    for label, use_band, judges in rows:
        accs = [evaluate_row(ds, use_band=use_band, judges=judges) for ds in DATASETS]
        avg = sum(accs) / 4
        print(f"{label:<40s} "
              f"{accs[0]:>6.1f} {accs[1]:>6.1f} {accs[2]:>6.1f} {accs[3]:>6.1f} {avg:>6.1f}")


def run_table4_panel():
    print("\n=== Table 4 Panel Robustness (band on) ===")
    print(f"{'Panel':<40s} {'HM':>6s} {'EN':>6s} {'ZH':>6s} {'IH':>6s} {'Avg':>6s}")
    for name, panel in PANEL_VARIANTS.items():
        accs = [evaluate_row(ds, use_band=True, judges=panel) for ds in DATASETS]
        avg = sum(accs) / 4
        print(f"{name:<40s} "
              f"{accs[0]:>6.1f} {accs[1]:>6.1f} {accs[2]:>6.1f} {accs[3]:>6.1f} {avg:>6.1f}")


if __name__ == "__main__":
    run_table2_ablation()
    run_table4_panel()
