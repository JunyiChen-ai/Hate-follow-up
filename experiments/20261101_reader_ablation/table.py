#!/usr/bin/env python3
"""Reader-block table from the analysis outputs (decode_eval.sh): per arm and corpus the absolute pooled ROC / PR /
within, the difference to the adopted method with the 95 % paired-bootstrap interval, and the metrics.json path.
Evaluation only (reads summary.json of analyze_dvd.py, which read the gold); writes runs/20261101_reader_ablation/table.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20261101_reader_ablation"
OLD = {  # paper Table tab:ablation, Reader block (within; r6-era reads)
    "no_transcript": ("transcript context in the prefix", "-.039", "+.001", "-.049 [-.079, -.021]"),
    "no_prefix_frames": ("the 20 frames (speech branches only)", "-.060", "-.109", "-.003"),
    "no_stance": ("stance turn (windows read before the verdict)", "-.014", "-.011", "-.010"),
    "asr_windows": ("fixed 8-s grid (ASR segments as windows)", "-.089", "-.052", "-.065 [-.106, -.026]"),
    "no_isolation": ("branch isolation (branches see each other in order)", "-.014", "+.011", "-.031 [-.060, -.001]"),
    "joint": ("two branches (one joint question per window)", "+.009", "-.074", "+.027"),
}
GROUPS = {"HateMM": "main", "HateClipSeg": "main", "DeHate": "DeHate"}


def metrics_path(arm, ds):
    g = GROUPS[ds]
    if arm == "asr_windows":
        return R / f"compose/asr_{g}/metrics_izv_plus_mean_rrank.json"
    return R / f"decoded/{arm}_{g}/metrics.json"


def main():
    lines = ["| arm | corpus | old within diff (paper) | new pooled ROC diff | new pooled PR diff | new within diff | arm ROC / PR / within | base ROC / PR / within | metrics.json |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, (row, *old) in OLD.items():
        for k, ds in enumerate(GROUPS):
            g = GROUPS[ds]
            summ = json.load(open(R / (f"analysis_asr_{g}" if arm == "asr_windows" else f"analysis_{g}") / "summary.json"))
            name = "spvl_asr" if arm == "asr_windows" else arm
            base = "spvl_fixed" if arm == "asr_windows" else "base"
            a, b = summ[f"{ds}:{name}"], summ[f"{ds}:{base}"]
            d = a["diff_vs_base"]
            ev = next(x for x in json.load(open(metrics_path(arm, ds)))["per_dataset"] if x["dataset"] == ds)
            ev = [ev["frame_ROC_AUC"], ev["frame_PR_AUC"], ev["within_video_macro_ROC_AUC"]]
            gap = max(abs(x - y) for x, y in zip(ev, a["metrics"]))
            assert gap < 1e-6, (arm, ds, ev, a["metrics"])  # bootstrap point estimate = evaluator output
            cell = [f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]" for x, lo, hi in d]
            lines.append(f"| {arm} ({row}) | {ds} | {old[k]} | {cell[0]} | {cell[1]} | {cell[2]} | "
                         + " / ".join(f"{x:.4f}" for x in ev) + " | " + " / ".join(f"{x:.4f}" for x in b["metrics"])
                         + f" | `{metrics_path(arm, ds).relative_to(ROOT)}` |")
    (R / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
