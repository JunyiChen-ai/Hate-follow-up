#!/usr/bin/env python3
"""Candidate 43 analysis on sc474397 (CPU): STAGE=check | evaluate | report  (README §3 gates, §6 checks).

check:    every record of runs/20261014_m1_audio_bound/<name>/records: GT_read False, derangement, the proportional
          text slice present exactly where r6 had a speech read, standalone-vs-cached margins, audio token counts,
          timing and memory. Writes <name>_analysis/checks.json.
evaluate: fixed r6_bma time level (twolevel_r2.py, r6 flags) on every arm -> <name>_decoded/<arm>/metrics.json;
          the raw window reads through the sole evaluator -> <name>_raw/<arm>/metrics.json.
report:   metrics table, the declared gates and the diagnostic readings -> <name>_analysis/summary.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "runs/20261014_m1_audio_bound"
DATASETS = ("HateMM", "HateClipSeg")
ARMS = ("audio", "text_aligned", "text_prop", "audio_swap", "audio_nbr", "r6_inputs")
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
FLOOR = {"frame_ROC_AUC": .005, "frame_PR_AUC": .005, "within_video_macro_ROC_AUC": .01}
R6 = ROOT / "runs/20260926_twolevel/r6_bma/metrics.json"
R6_RAW = ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"


def records(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["dataset"], r["video_id"])] = r
    return out


def metric(path):
    return {d["dataset"]: d for d in json.loads(Path(path).read_text())["per_dataset"]}


def check(name):
    root = EXP / name
    cfg = json.loads((root / "config.json").read_text())
    assert cfg["GT_read"] is False
    r6 = records(R6_RAW)
    out = {ds: dict(videos=0, windows=0, speech_prop_matches_r6=0, aligned_status={}, aligned_differs=0, aligned_windows=0, local_windows=0,
                    standalone=[], audio_tokens=[], seconds={}, peak_GiB=0.0, derangement_ok=0) for ds in DATASETS}
    preds = {arm: records(root / arm / "predictions.jsonl") for arm in ARMS}
    for ds in DATASETS:
        for path in sorted((root / "records" / ds).glob("*.json")):
            b = json.loads(path.read_text())
            key = (ds, path.stem)
            s = out[ds]
            assert b["checks"]["GT_read"] is False and b["implementation"].startswith("candidate 43")
            s["videos"] += 1
            s["derangement_ok"] += bool(b["checks"]["derangement_ok"])
            s["peak_GiB"] = max(s["peak_GiB"], b["checks"]["peak_GiB"])
            for k, v in b["checks"]["times"].items():
                s["seconds"][k] = s["seconds"].get(k, 0.0) + v
            st = b["checks"]["aligned_status"]
            s["aligned_status"][str(st)] = s["aligned_status"].get(str(st), 0) + 1
            ref = r6[key]["extra"]["windows"]
            assert len(ref) == len(b["traces"]) == b["checks"]["windows"]
            for t, w in zip(b["traces"], ref):
                s["windows"] += 1
                s["speech_prop_matches_r6"] += (t["speech_prop"] is not None) == (w.get("z_speech") is not None)
                s["local_windows"] += t["visual_local"] is not None
                if t["text_aligned"] is not None:
                    s["aligned_windows"] += 1
                    s["aligned_differs"] += t["text_aligned"].split() != (t["text_prop"] or "").split()
                for k in ("speech_audio", "speech_swap", "speech_nbr"):
                    s["audio_tokens"].append(t[k]["audio_tokens"])
            for arm in ARMS:
                p = preds[arm][key]
                assert [w["z"] for w in p["extra"]["windows"]] == [w["z"] for w in b["predictions"][arm]["extra"]["windows"]]
            if b["diagnostic"]:
                d = b["diagnostic"]
                s["standalone"].append({k: dict(abs=v["abs_difference"], **({"head_ids_equal": v["head_ids_equal"]} if "head_ids_equal" in v else {}))
                                        for k, v in d.items() if isinstance(v, dict)})
    for ds, s in out.items():
        s["speech_prop_matches_r6_share"] = s["speech_prop_matches_r6"] / max(1, s["windows"])
        s["aligned_differs_share"] = s["aligned_differs"] / max(1, s["aligned_windows"])
        a = np.asarray(s.pop("audio_tokens"))
        s["audio_tokens_mean"] = float(a.mean()) if len(a) else None
        s["audio_tokens_min_max"] = [int(a.min()), int(a.max())] if len(a) else None
        s["standalone_max_abs"] = max((v["abs"] for d in s["standalone"] for v in d.values()), default=None)
        s["standalone_head_ids_equal"] = all(v.get("head_ids_equal", True) for d in s["standalone"] for v in d.values())
    result = dict(PASS=all(s["speech_prop_matches_r6_share"] == 1.0 and s["derangement_ok"] == s["videos"] for s in out.values()),
                  GT_read=False, coverage={ds: out[ds]["videos"] for ds in DATASETS}, datasets=out)
    folder = root.parent / (root.name + "_analysis")
    folder.mkdir(exist_ok=True)
    (folder / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({ds: {k: v for k, v in s.items() if k != "standalone"} for ds, s in out.items()}, indent=1))
    print("CHECK_PASS" if result["PASS"] else "CHECK_FAIL", flush=True)


def evaluate(name, arm):
    root = EXP / name
    assert json.loads((root.parent / (root.name + "_analysis") / "checks.json").read_text())["PASS"] is True
    raw = root.parent / (root.name + "_raw") / arm
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(root / arm / "predictions.jsonl"),
                    "--gt-dir", "data/gt_4fps", "--datasets", *DATASETS, "--out", str(raw / "metrics.json")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "experiments/20260926_twolevel/twolevel_r2.py", "--run", str(root / arm), "--datasets", *DATASETS,
                    "--noleak", "--transform", "nscore", "--key", "calib", "--duration", "bma", "--bma-prior", "length", "--min-windows", "2",
                    "--bma-grid", "6", "--arm", "m2", "--out-root", str(root.parent / (root.name + "_decoded")), "--tag", arm], cwd=ROOT, check=True)


def report(name):
    root = EXP / name
    decoded = root.parent / (root.name + "_decoded")
    m = {"r6_bma": metric(R6)}
    for arm in ARMS:
        m[arm] = metric(decoded / arm / "metrics.json")
    raw = {arm: metric(root.parent / (root.name + "_raw") / arm / "metrics.json") for arm in ARMS}

    def diff(x, y):
        return {ds: {k: m[x][ds][k] - m[y][ds][k] for k in METRICS} for ds in DATASETS}

    def gate_within(d, threshold=.01):
        return all(d[ds]["within_video_macro_ROC_AUC"] >= threshold for ds in DATASETS)

    def no_loss(d):
        return all(d[ds][k] > -FLOOR[k] for ds in DATASETS for k in METRICS)

    steps = dict(performance=diff("audio", "r6_bma"), M1_binding=diff("audio", "audio_swap"), M2_hearing=diff("audio", "text_aligned"),
                 reader_swap=diff("r6_inputs", "r6_bma"), aligned_vs_prop=diff("text_aligned", "text_prop"), audio_vs_prop=diff("audio", "text_prop"),
                 audio_vs_nbr=diff("audio", "audio_nbr"), frames_under_omni=diff("text_prop", "r6_inputs"))
    verdicts = dict(performance=dict(within_both=gate_within(steps["performance"]), no_loss=no_loss(steps["performance"]),
                                     PASS=gate_within(steps["performance"]) and no_loss(steps["performance"])),
                    M1_binding=gate_within(steps["M1_binding"]), M2_hearing=gate_within(steps["M2_hearing"]),
                    aligned_vs_prop=gate_within(steps["aligned_vs_prop"]),
                    reader_swap_below_minus_02=any(steps["reader_swap"][ds]["within_video_macro_ROC_AUC"] < -.02 for ds in DATASETS))
    if verdicts["M1_binding"] and verdicts["M2_hearing"]:
        reading = "mechanism supported as stated (binding and hearing beyond the aligned text)"
    elif verdicts["M1_binding"] and verdicts["aligned_vs_prop"]:
        reading = "binding by alignment only: the aligned-text variant carries the gain, the audio reader is not needed"
    elif verdicts["M1_binding"]:
        reading = "binding supported, hearing not; aligned text does not reproduce it"
    else:
        reading = "binding not supported: audio acts as context"
    checks = json.loads((root.parent / (root.name + "_analysis") / "checks.json").read_text())
    result = dict(scope="development-selected; fixed r6_bma time level; sole evaluator; readings declared in README §3 before the run",
                  metrics={a: {ds: {k: m[a][ds][k] for k in METRICS} for ds in DATASETS} for a in m},
                  raw_window_reads={a: {ds: {k: raw[a][ds][k] for k in METRICS} for ds in DATASETS} for a in ARMS},
                  n_videos_defined={a: {ds: m[a][ds]["n_videos_defined"] for ds in DATASETS} for a in m},
                  sources={"r6_bma": str(R6.relative_to(ROOT)), **{a: str((decoded / a / "metrics.json").relative_to(ROOT)) for a in ARMS}},
                  steps=steps, verdicts=verdicts, reading=reading,
                  binding={ds: {k: checks["datasets"][ds][k] for k in ("videos", "windows", "speech_prop_matches_r6_share", "aligned_differs_share",
                                                                      "aligned_status", "standalone_max_abs", "audio_tokens_mean", "seconds", "peak_GiB")} for ds in DATASETS})
    folder = root.parent / (root.name + "_analysis")
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    f = lambda d, ds: " / ".join(f"{d[ds][k]:+.4f}" for k in METRICS)
    lines = ["arm | HateMM ROC / PR / within | HateClipSeg ROC / PR / within"]
    for a in m:
        lines.append(f"{a:13s}| " + " | ".join(" / ".join(f"{m[a][ds][k]:.4f}" for k in METRICS) for ds in DATASETS))
    lines.append("raw window reads (no time level):")
    for a in ARMS:
        lines.append(f"{a:13s}| " + " | ".join(" / ".join(f"{raw[a][ds][k]:.4f}" for k in METRICS) for ds in DATASETS))
    lines.append("steps (difference, HateMM | HateClipSeg):")
    for k, d in steps.items():
        lines.append(f"{k:17s}| {f(d, 'HateMM')} | {f(d, 'HateClipSeg')}")
    lines.append("verdicts: " + json.dumps(verdicts))
    lines.append("reading: " + reading)
    (folder / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("REPORT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("check", "evaluate", "report"), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm")
    a = ap.parse_args()
    if a.stage == "check":
        check(a.name)
    elif a.stage == "evaluate":
        assert a.arm in ARMS
        evaluate(a.name, a.arm)
    else:
        report(a.name)


if __name__ == "__main__":
    main()
