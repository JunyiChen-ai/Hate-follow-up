#!/usr/bin/env python3
"""Candidate 46 CPU stage (README §2). Stages:
  precheck            D2 / D3 on the records (reads test GT: rule 10) -> <name>_precheck/{summary.json,per_video.json,table.txt}
  predictions --arm   predictions.jsonl of one arm (abl, shuf0..shuf4, rand0): z_speech := the arm's read, z_visual and the key term z from r6
  decode --arm        unified evaluator on the raw curve + fixed r6_bma time level -> <name>_raw/<arm>, <name>_decoded/<arm>
  report              six numbers of every decoded arm against r6_bma with the rule-8 gates -> <name>_report/summary.json
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARMS, EXP, N_RAND, N_SHUF, R6_BMA, ROOT, VALIDITY, WINDOW_S, stored_r6  # noqa: E402
from src.video_inputs import fixed_windows  # noqa: E402
from src.window_level import auc, gold, mean, window_label, within_by_video, within_eligible  # noqa: E402

DATASETS = ("HateMM", "HateClipSeg")
D2_DROP, D3_MARGIN = .05, .02
NOISE_POOLED, NOISE_WITHIN, PROMOTE = .005, .01, .01
TWOLEVEL_FLAGS = ["--noleak", "--transform", "nscore", "--key", "calib", "--duration", "bma", "--bma-prior", "length",
                  "--min-windows", "2", "--bma-grid", "6", "--arm", "m2"]
GROUP_TERMS = ["black", "white", "jew", "jews", "jewish", "muslim", "muslims", "islam", "christian", "christians", "asian", "asians",
               "mexican", "mexicans", "immigrant", "immigrants", "arab", "arabs", "gay", "gays", "trans", "transgender", "women", "woman",
               "female", "disabled", "indian", "indians", "african", "africans", "hispanic", "hispanics", "latino", "latinos"]
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def records(name, ds):
    return {json.loads(p.read_text())["video_id"]: json.loads(p.read_text()) for p in sorted((EXP / name / "records" / ds).glob("*.json"))}


def arm_score(t, arm):
    if arm == "abl":
        return t["z_abl"]
    return t["z_shuf"][int(arm[4:])] if arm.startswith("shuf") else t["z_rand"][int(arm[4:])]


def window_rows(name, ds, gt, eligible_only):
    """One row per speech window with a clean label; eligible_only restricts to within-eligible videos (D3), otherwise all (D2)."""
    rows = []
    best = json.loads(VALIDITY.read_text())["body"]["best_layer"] if VALIDITY.exists() else None
    for vid, rec in records(name, ds).items():
        y = gt.get(vid)
        if y is None or (eligible_only and not within_eligible(y)):
            continue
        for t in rec["traces"]:
            if not t["text"] or "z_abl" not in t:
                continue
            label, frac = window_label(y, t["bounds"][0], t["bounds"][1])
            if label is None:
                continue
            words = _WORD.findall(t["text"].lower())
            row = dict(video_id=vid, i=t["i"], label=label, frac=frac, z_speech=t["z_speech_r6"], abl=t["z_abl"],
                       group_hits=sum(w in GROUP_TERMS for w in words), word_count=len(words),
                       proj_best=None if best is None else t["proj_topic"][best])
            for k in range(N_SHUF):
                row[f"shuf{k}"] = t["z_shuf"][k]
            for k in range(N_RAND):
                row[f"rand{k}"] = t["z_rand"][k]
            rows.append(row)
    return rows


def precheck(name):
    summary, per_video, lines = {}, {}, []
    validity = json.loads(VALIDITY.read_text()) if VALIDITY.exists() else None
    for ds in DATASETS:
        gt = gold(ds)
        rows_all = window_rows(name, ds, gt, eligible_only=False)
        rows = window_rows(name, ds, gt, eligible_only=True)
        scores = ["z_speech", "abl"] + [f"shuf{k}" for k in range(N_SHUF)] + [f"rand{k}" for k in range(N_RAND)]
        per = {k: within_by_video(rows, k) for k in scores}
        w = {k: mean(per[k].values()) for k in scores}
        w["shuf_mean"] = float(np.mean([w[f"shuf{k}"] for k in range(N_SHUF)]))
        w["rand_mean"] = float(np.mean([w[f"rand{k}"] for k in range(N_RAND)]))
        n_def = {k: sum(v is not None for v in per[k].values()) for k in scores}
        pooled = {k: auc([r["label"] for r in rows_all], [r[k] for r in rows_all]) for k in ("z_speech", "abl")}
        pooled["shuf_mean"] = float(np.mean([auc([r["label"] for r in rows_all], [r[f"shuf{k}"] for r in rows_all]) for k in range(N_SHUF)]))
        pooled["rand_mean"] = float(np.mean([auc([r["label"] for r in rows_all], [r[f"rand{k}"] for r in rows_all]) for k in range(N_RAND)]))
        rho = float(spearmanr([r["z_speech"] for r in rows], [r["abl"] for r in rows]).statistic)
        rho_rand = [float(spearmanr([r["z_speech"] for r in rows], [r[f"shuf{k}"] for r in rows]).statistic) for k in range(N_SHUF)]
        proj_ok = all(r["proj_best"] is not None for r in rows)
        rho_proj = float(spearmanr([r["group_hits"] for r in rows_all], [r["proj_best"] for r in rows_all]).statistic) if proj_ok else None
        by_label = {k: {str(l): float(np.mean([r[k] for r in rows if r["label"] == l])) for l in (0, 1)} for k in ("z_speech", "abl")}
        d2 = pooled["abl"] >= pooled["z_speech"] - D2_DROP
        d3a = w["abl"] >= w["z_speech"] + D3_MARGIN
        d3b = w["shuf_mean"] <= w["abl"] - D3_MARGIN
        summary[ds] = dict(videos_eligible=len({r["video_id"] for r in rows}), windows=len(rows), positives=int(sum(r["label"] for r in rows)),
                           windows_all_videos=len(rows_all), within=w, videos_defined=n_def, pooled_window_auc=pooled,
                           spearman_abl_vs_z_speech=rho, spearman_shuf_vs_z_speech=rho_rand, spearman_group_hits_vs_proj=rho_proj,
                           mean_by_label=by_label,
                           gates=dict(D2_judgment_survives=dict(pooled_abl=pooled["abl"], pooled_z_speech=pooled["z_speech"], max_drop=D2_DROP, PASS=bool(d2)),
                                      D3_within=dict(abl=w["abl"], z_speech=w["z_speech"], shuf_mean=w["shuf_mean"], rand_mean=w["rand_mean"], margin=D3_MARGIN,
                                                     beats_judgment=bool(d3a), beats_random=bool(d3b), PASS=bool(d3a and d3b))))
        per_video[ds] = per
        lines.append(f"[{ds}] eligible videos {summary[ds]['videos_eligible']}, windows {len(rows)} (pos {summary[ds]['positives']}); all-video windows {len(rows_all)}")
        for k in scores + ["shuf_mean", "rand_mean"]:
            lines.append(f"  window-level within {k:10s} {w[k]:.4f}" + (f"  (videos {n_def[k]})" if k in n_def else ""))
        lines.append(f"  pooled window AUC: z_speech {pooled['z_speech']:.4f}  abl {pooled['abl']:.4f}  shuf_mean {pooled['shuf_mean']:.4f}  rand_mean {pooled['rand_mean']:.4f}")
        lines.append(f"  Spearman abl vs z_speech {rho:+.3f}; shuf vs z_speech {np.mean(rho_rand):+.3f}; group-term hits vs topic projection {rho_proj if rho_proj is None else round(rho_proj, 3)}")
        lines.append(f"  D2 {'PASS' if d2 else 'FAIL'}; D3 beats judgment {d3a}, beats shuffled-label controls {d3b} -> {'PASS' if d3a and d3b else 'FAIL'}")
    d1 = None if validity is None else validity["D1"]["PASS"]
    decision = bool(d1) and all(summary[ds]["gates"]["D2_judgment_survives"]["PASS"] and summary[ds]["gates"]["D3_within"]["PASS"] for ds in DATASETS)
    summary["decision"] = dict(D1=d1, D1_best_layer=None if validity is None else validity["body"]["best_layer"],
                               D1_best_auc=None if validity is None else validity["body"]["best_auc_test"], PASS=decision, GT_read=True,
                               rule="D1, D2, D3 on both corpora -> abl replaces the speech emission and is decoded with the fixed r6_bma time level")
    out = EXP / (name + "_precheck")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "per_video.json").write_text(json.dumps(per_video, indent=2) + "\n")
    lines.append(f"D1 {d1}; decision PASS={decision}")
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def predictions(name, arm):
    r6 = stored_r6()
    out = EXP / name / arm
    out.mkdir(parents=True, exist_ok=True)
    checks = dict(arm=arm, videos={}, missing=[], replay_max_abs={}, stance_mismatch={})
    with open(out / "predictions.jsonl", "w") as fh:
        for ds in DATASETS:
            recs = records(name, ds)
            expected = [k[1] for k in r6 if k[0] == ds]
            checks["missing"] += [f"{ds}/{v}" for v in expected if v not in recs]
            checks["videos"][ds] = len(recs)
            checks["replay_max_abs"][ds] = max((r["replay_max_abs"] for r in recs.values() if r["replay_max_abs"] is not None), default=None)
            checks["stance_mismatch"][ds] = sum(r["stance"] != r["stance_r6"] for r in recs.values())
            for vid, r in recs.items():
                ref = r6[(ds, vid)]
                ww, raw = [], []
                for t, (t1, t2), w6 in zip(r["traces"], fixed_windows(r["duration"], WINDOW_S), ref["extra"]["windows"]):
                    assert abs(w6["start"] - t1) < 1e-6 and abs(w6["end"] - t2) < 1e-6 and w6["z_visual"] == t["z_visual_r6"], (vid, t["i"])
                    one = dict(i=t["i"], start=t1, end=t2, z_visual=t["z_visual_r6"], z=w6["z"])     # key term stays r6's (README §2)
                    if t["text"]:
                        one["z_speech"] = float(arm_score(t, arm))
                        raw.append(max(one["z_visual"], one["z_speech"]))
                    else:
                        raw.append(one["z_visual"])
                    ww.append(one)
                idx = np.clip(((np.arange(math.ceil(r["duration"] * 4)) + .5) / 4 // WINDOW_S).astype(int), 0, len(ww) - 1)
                fh.write(json.dumps(dict(schema_version=1, method=f"m1_topic_removal_{arm}", dataset=ds, video_id=vid, duration=r["duration"], native_rate=4,
                                         score_curve=np.asarray(raw)[idx].tolist(), intervals=[], error=None, seed=0,
                                         calls=3 + 2 * len(ww), code_path="experiments/20261017_m1_topic_removal/analyze.py",
                                         extra=dict(z_video=ref["extra"]["z_video"], stance=r["stance"], prefix_tokens=r["prefix_tokens"],
                                                    stance_cache_tokens=r["stance_cache_tokens"], windows=ww))) + "\n")
    checks["PASS"] = not checks["missing"]
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=1))


def decode(name, arm):
    root = EXP / name
    assert json.loads((root / arm / "checks.json").read_text())["PASS"] is True, "predictions checks did not pass"
    raw = EXP / (name + "_raw") / arm
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(root / arm / "predictions.jsonl"),
                    "--gt-dir", "data/gt_4fps", "--datasets", *DATASETS, "--out", str(raw / "metrics.json")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "experiments/20260926_twolevel/twolevel_r2.py", "--run", str(root / arm), "--datasets", *DATASETS,
                    *TWOLEVEL_FLAGS, "--out-root", str(EXP / (name + "_decoded")), "--tag", arm], cwd=ROOT, check=True)


def metric(path):
    per = {d["dataset"]: d for d in json.loads(Path(path).read_text())["per_dataset"]}
    return {ds: (float(per[ds]["frame_ROC_AUC"]), float(per[ds]["frame_PR_AUC"]), float(per[ds]["within_video_macro_ROC_AUC"])) for ds in DATASETS}


def report(name):
    base = metric(R6_BMA)
    res, gates = {"r6_bma": base}, {}
    for arm in ARMS:
        p = EXP / (name + "_decoded") / arm / "metrics.json"
        if not p.exists():
            continue
        m = metric(p)
        res[arm] = m
        d = {ds: tuple(m[ds][k] - base[ds][k] for k in range(3)) for ds in DATASETS}
        no_loss = all(d[ds][0] >= -NOISE_POOLED and d[ds][1] >= -NOISE_POOLED and d[ds][2] >= -NOISE_WITHIN for ds in DATASETS)
        gain = any(all(d[ds][k] >= PROMOTE for ds in DATASETS) for k in range(3))
        within_gain = all(d[ds][2] >= PROMOTE for ds in DATASETS)
        gates[arm] = dict(diff=d, no_loss_beyond_noise=no_loss, some_metric_up_01_both=gain, within_up_01_both=within_gain,
                          promotion_gate_rule8=bool(no_loss and gain), mechanism_claim_within=bool(no_loss and within_gain))
    out = EXP / (name + "_report")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(metrics=res, gates=gates, sources={"r6_bma": str(R6_BMA.relative_to(ROOT))}), indent=2) + "\n")
    for k, v in res.items():
        print(k, {ds: tuple(round(x, 4) for x in v[ds]) for ds in DATASETS})
    print(json.dumps(gates, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("precheck", "predictions", "decode", "report"), required=True)
    ap.add_argument("--name", default="main")
    ap.add_argument("--arm", choices=tuple(ARMS))
    a = ap.parse_args()
    {"precheck": lambda: precheck(a.name), "predictions": lambda: predictions(a.name, a.arm),
     "decode": lambda: decode(a.name, a.arm), "report": lambda: report(a.name)}[a.stage]()


if __name__ == "__main__":
    main()
