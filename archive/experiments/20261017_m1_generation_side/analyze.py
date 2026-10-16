#!/usr/bin/env python3
"""Candidate 45 CPU stage (README §2). Stages:
  precheck            window-level pre-check on the within-eligible videos (reads test GT: rule 10) -> precheck/{summary.json,per_video.json,table.txt}
  predictions --arm   predictions.jsonl of one arm: z_speech := the arm's primary score, z_visual from r6, raw z = max(z_visual, s_aff)
  decode --arm        unified evaluator on the raw curve + fixed r6_bma time level -> main_raw/<arm>, main_decoded/<arm>
  report              six numbers of every decoded arm against r6_bma with the rule-8 gates -> report/summary.json
Window text, z_visual and z_speech come from the stored r6 run; the generation-side scores from measure.py records.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import A_SCORES, ARMS, B_SCORES, EXP, R6_BMA, ROOT, WINDOW_S, frame_span, norm_words, stored_r6  # noqa: E402
from src.video_inputs import fixed_windows  # noqa: E402

DATASETS = ("HateMM", "HateClipSeg")
CONTROLS = ("lex_profanity", "lex_hate", "word_count")
GATE_RHO, GATE_MARGIN = .80, .02
NOISE_POOLED, NOISE_WITHIN, PROMOTE = .005, .01, .01
TWOLEVEL_FLAGS = ["--noleak", "--transform", "nscore", "--key", "calib", "--duration", "bma", "--bma-prior", "length",
                  "--min-windows", "2", "--bma-grid", "6", "--arm", "m2"]


def records(name, ds):
    out = {}
    for p in sorted((EXP / name / "records" / ds).glob("*.json")):
        r = json.loads(p.read_text())
        out[r["video_id"]] = r
    return out


def gold(ds):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, int) for v, y, s in zip(g["video_ids"], g["y4"], g["split"]) if str(s) == "test"}


def lexicons():
    prof = [tuple(norm_words(l)) for l in (ROOT / "data/lexicons/ldnoobw/en.txt").read_text().splitlines() if l.strip()]
    hate = []
    for k, l in enumerate((ROOT / "data/lexicons/davidson2017/refined_ngram_dict.csv").read_text().splitlines()):
        if k == 0 or not l.strip():
            continue
        hate.append(tuple(norm_words(l.rsplit(",", 1)[0])))
    return {"lex_profanity": [t for t in prof if t], "lex_hate": [t for t in hate if t]}


def hits(words, grams):
    n = 0
    for g in grams:
        k = len(g)
        n += sum(1 for i in range(len(words) - k + 1) if tuple(words[i:i + k]) == g)
    return n


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, np.asarray(s, float))) if 0 < y.sum() < len(y) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def window_table(name, ds, lex, gt):
    """One row per speech window of a within-eligible video with a clean label (hate-frame share >= .5 or == 0)."""
    rows = []
    for vid, rec in records(name, ds).items():
        y = gt.get(vid)
        if y is None or not (0 < y.sum() < len(y)):
            continue
        for t in rec["traces"]:
            if not t["text"]:
                continue
            a, b = frame_span(t["bounds"][0], t["bounds"][1], len(y))
            if b <= a:
                continue
            frac = float(y[a:b].mean())
            if frac != 0 and frac < .5:
                continue
            words = norm_words(t["text"])
            row = dict(video_id=vid, i=t["i"], label=int(frac >= .5), frac=frac, z_speech=t["z_speech_r6"], word_count=len(words),
                       lex_profanity=hits(words, lex["lex_profanity"]), lex_hate=hits(words, lex["lex_hate"]))
            if "A" in t:
                row.update({k: t["A"][k] for k in A_SCORES})
            if "B" in t:
                row.update({k: t["B"][k] for k in B_SCORES})
                row["unchanged"] = int(norm_words(t["B"]["rewrite"]) == words)
            rows.append(row)
    return rows


def within_by_video(rows, key):
    per = {}
    for vid in sorted({r["video_id"] for r in rows}):
        rr = [r for r in rows if r["video_id"] == vid and r.get(key) is not None]
        per[vid] = auc([r["label"] for r in rr], [r[key] for r in rr]) if rr else None
    return per


def precheck(name):
    lex = lexicons()
    summary, per_video, lines = {}, {}, []
    for ds in DATASETS:
        rows = window_table(name, ds, lex, gold(ds))
        scores = ["z_speech"] + [k for k in (*A_SCORES, *B_SCORES) if any(k in r for r in rows)] + list(CONTROLS)
        per = {k: within_by_video(rows, k) for k in scores}
        rng = np.random.default_rng(0)
        rand = []
        for _ in range(10):
            for r in rows:
                r["_rand"] = float(rng.random())
            rand.append(mean(within_by_video(rows, "_rand").values()))
        w = {k: mean(per[k].values()) for k in scores}
        w["random"] = float(np.mean(rand))
        n_def = {k: sum(v is not None for v in per[k].values()) for k in scores}
        z = np.array([r["z_speech"] for r in rows], float)
        rho = {k: float(spearmanr(z, [r[k] for r in rows]).statistic) for k in scores if k != "z_speech"}
        wc = np.array([r["word_count"] for r in rows], float)
        rho_len = {k: float(spearmanr(wc, [r[k] for r in rows]).statistic) for k in scores if k != "word_count"}
        best_ctrl = max(w[k] for k in CONTROLS)
        gates = {}
        for arm, key in ARMS.items():
            if key not in w:
                continue
            g1, g2, g3 = rho[key] < GATE_RHO, w[key] >= w["z_speech"] + GATE_MARGIN, w[key] >= best_ctrl + GATE_MARGIN
            gates[arm] = dict(primary=key, G1_not_z_renamed=dict(rho=rho[key], threshold=GATE_RHO, PASS=bool(g1)),
                              G2_beats_judgment=dict(within=w[key], z_speech=w["z_speech"], margin=GATE_MARGIN, PASS=bool(g2)),
                              G3_beats_lexicon=dict(within=w[key], best_control=best_ctrl, margin=GATE_MARGIN, PASS=bool(g3)), PASS=bool(g1 and g2 and g3))
        extra = {}
        if any("edit_rate" in r for r in rows):
            extra["B_unchanged_share"] = float(np.mean([r["unchanged"] for r in rows if "unchanged" in r]))
            extra["B_edit_rate_by_label"] = {str(l): float(np.mean([r["edit_rate"] for r in rows if r["label"] == l and "edit_rate" in r])) for l in (0, 1)}
        if any("copy_mean" in r for r in rows):
            extra["A_copy_mean_by_label"] = {str(l): float(np.mean([r["copy_mean"] for r in rows if r["label"] == l and "copy_mean" in r])) for l in (0, 1)}
            extra["A_refuse_mass_gt_log_half_share"] = float(np.mean([r["refuse_mass"] > math.log(.5) for r in rows if "refuse_mass" in r]))
        summary[ds] = dict(videos_eligible=len({r["video_id"] for r in rows}), windows=len(rows), positives=int(sum(r["label"] for r in rows)),
                           within=w, videos_defined=n_def, spearman_vs_z_speech=rho, spearman_vs_word_count=rho_len, gates=gates, **extra)
        per_video[ds] = per
        lines.append(f"[{ds}] eligible videos {summary[ds]['videos_eligible']}, windows {len(rows)} (pos {summary[ds]['positives']})")
        for k in scores + ["random"]:
            lines.append(f"  window-level within {k:14s} {w[k]:.4f}" + ("" if k == "random" else f"  (videos {n_def[k]})") +
                         ("" if k in ("z_speech", "random") else f"  rho(z_speech) {rho[k]:+.3f}"))
        for arm, g in gates.items():
            lines.append(f"  arm {arm} ({g['primary']}): G1 {g['G1_not_z_renamed']['PASS']} G2 {g['G2_beats_judgment']['PASS']} G3 {g['G3_beats_lexicon']['PASS']} -> {'PASS' if g['PASS'] else 'FAIL'}")
    decision = {arm: all(summary[ds]["gates"].get(arm, {}).get("PASS") for ds in DATASETS) for arm in ARMS}
    summary["decision"] = dict(arm_passes_both_corpora=decision, GT_read=True,
                               rule="an arm passing G1-G3 on both corpora replaces z_speech and is decoded with the fixed r6_bma time level")
    out = EXP / (name + "_precheck")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "per_video.json").write_text(json.dumps(per_video, indent=2) + "\n")
    lines.append(f"decision: {decision}")
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def predictions(name, arm):
    key = ARMS[arm]
    r6 = stored_r6()
    out = EXP / name / arm
    out.mkdir(parents=True, exist_ok=True)
    checks = dict(arm=arm, primary=key, videos={}, missing=[], replay_max_abs={}, stance_mismatch={})
    with open(out / "predictions.jsonl", "w") as fh:
        for ds in DATASETS:
            recs = records(name, ds)
            expected = [k[1] for k in r6 if k[0] == ds]
            checks["missing"] += [f"{ds}/{v}" for v in expected if v not in recs]
            s_all = np.array([t[arm][key] for r in recs.values() for t in r["traces"] if t["text"]], float)
            z_all = np.array([t["z_speech_r6"] for r in recs.values() for t in r["traces"] if t["text"]], float)
            aff = dict(s_mean=float(s_all.mean()), s_std=float(s_all.std() or 1.0), z_mean=float(z_all.mean()), z_std=float(z_all.std()))
            checks["videos"][ds] = len(recs)
            checks["replay_max_abs"][ds] = max((r["replay_max_abs"] for r in recs.values() if r["replay_max_abs"] is not None), default=None)
            checks["stance_mismatch"][ds] = sum(r["stance"] != r["stance_r6"] for r in recs.values())
            checks[f"affine_{ds}"] = aff
            for vid, r in recs.items():
                ref = r6[(ds, vid)]
                ww, raw_z = [], []
                for t, (t1, t2), w6 in zip(r["traces"], fixed_windows(r["duration"], WINDOW_S), ref["extra"]["windows"]):
                    assert abs(w6["start"] - t1) < 1e-6 and abs(w6["end"] - t2) < 1e-6 and w6["z_visual"] == t["z_visual_r6"], (vid, t["i"])
                    # README §2: `z` (the video key's window term in twolevel_r2, K = z_video + mean z) stays r6's value, so the
                    # time level differs from r6_bma only in the speech emission; the raw score_curve alone uses max(z_visual, s_aff).
                    one = dict(i=t["i"], start=t1, end=t2, z_visual=t["z_visual_r6"], z=w6["z"])
                    if t["text"]:
                        s = float(t[arm][key])
                        s_aff = (s - aff["s_mean"]) / aff["s_std"] * aff["z_std"] + aff["z_mean"]
                        one["z_speech"] = s
                        one["s_aff"] = s_aff
                        raw_z.append(max(one["z_visual"], s_aff))
                    else:
                        raw_z.append(one["z_visual"])
                    ww.append(one)
                idx = np.clip(((np.arange(math.ceil(r["duration"] * 4)) + .5) / 4 // WINDOW_S).astype(int), 0, len(ww) - 1)
                fh.write(json.dumps(dict(schema_version=1, method=f"m1_generation_side_{arm}", dataset=ds, video_id=vid, duration=r["duration"], native_rate=4,
                                         score_curve=np.asarray(raw_z)[idx].tolist(), intervals=[], error=None, seed=0,
                                         calls=3 + 2 * sum(bool(t["text"]) for t in r["traces"]), code_path="experiments/20261017_m1_generation_side/analyze.py",
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
    res = {"r6_bma": base}
    gates = {}
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
    if a.stage == "precheck":
        precheck(a.name)
    elif a.stage == "predictions":
        predictions(a.name, a.arm)
    elif a.stage == "decode":
        decode(a.name, a.arm)
    else:
        report(a.name)


if __name__ == "__main__":
    main()
