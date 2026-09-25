#!/usr/bin/env python3
"""GLR pilot analysis (CPU). Reads the fixed-ASR SPVL-r2 grid-A run (base) and the GLR measurement run.

1. Primary (declared in README §4): on 8 s windows with speech, within-video window AUC of each GLR variant vs the
   SPVL-r2 speech branch `z_speech` from the base run; paired bootstrap over videos. Window label = at least half
   of the window's 4 fps GT frames positive. Test labels are used here only for evaluation (rule 10).
2. Mechanism: within-video standardized score ~ GT + mention of a protected group (declared word list), OLS.
3. Derived runs for the frame-level evaluation: base windows with `z_speech` replaced by a GLR variant; the
   caller runs experiments/20260922_til/til_infer.py on them (shared evaluator inside).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_asr, window_text  # noqa: E402

H0 = ("H0_discuss", "H0_report", "H0_crude")
VARIANTS = ["full|assistant", "full|document", "none|assistant", "none|document"]
GROUP_WORDS = {  # protected-group names, deliberately without slurs (slurs are the act, not the topic)
    "black", "blacks", "white", "whites", "jew", "jews", "jewish", "muslim", "muslims", "islam", "islamic", "arab",
    "arabs", "asian", "asians", "chinese", "mexican", "mexicans", "latino", "latinos", "hispanic", "hispanics",
    "immigrant", "immigrants", "migrant", "migrants", "refugee", "refugees", "gay", "gays", "lesbian", "lesbians",
    "homosexual", "homosexuals", "lgbt", "trans", "transgender", "women", "woman", "christian", "christians",
    "hindu", "hindus", "indian", "indians", "african", "africans"}
SEED = 0
N_BOOT = 4000


def load(path):
    out = {}
    for line in open(Path(path) / "predictions.jsonl"):
        r = json.loads(line)
        if not r.get("error"):
            out[(r["dataset"], r["video_id"])] = r
    return out


def llr(lp, ntok, variant, h0=H0):
    ctx, fr = variant.split("|")
    num = lp[f"{ctx}|{fr}|H1"]
    den = np.logaddexp.reduce([lp[f"{ctx}|{fr}|{c}"] for c in h0]) - math.log(len(h0))
    return (num - den) / max(ntok[fr], 1)


def boot_ci(d, rng):
    d = np.asarray(d, float)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "runs/20260926_glr/base_gridA"))
    ap.add_argument("--glr", default=str(ROOT / "runs/20260926_glr/glr_pilot"))
    ap.add_argument("--out", default=str(ROOT / "runs/20260926_glr/analysis"))
    ap.add_argument("--derived-root", default=str(ROOT / "runs/20260926_glr/derived"))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    base, glr = load(a.base), load(a.glr)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    signals = ["z_speech"] + VARIANTS + [f"full|assistant|only_{c}" for c in H0]
    summary, lines = {}, []
    for ds in a.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, float) for v, y in zip(g["video_ids"], g["y4"])}
        asr = load_asr(ds)
        per_video, rows_mech = {}, []
        n_mismatch = 0
        for (d, vid), rb in base.items():
            if d != ds or (d, vid) not in glr or vid not in Y:
                continue
            wb, wg = rb["extra"]["windows"], glr[(d, vid)]["extra"]["windows"]
            if len(wb) != len(wg) or any(abs(x["start"] - y["start"]) > 1e-6 for x, y in zip(wb, wg)):
                n_mismatch += 1
                continue
            y = Y[vid]; centers = (np.arange(len(y)) + 0.5) / 4.0
            recs = []
            for x, w in zip(wb, wg):
                if "z_speech" not in x or not w.get("has_speech"):
                    continue
                m = (centers >= x["start"]) & (centers < x["end"])
                if not m.any():
                    continue
                ntok = {"assistant": w["ntok_assistant"], "document": w["ntok_document"]}
                s = {"z_speech": x["z_speech"]}
                for v in VARIANTS:
                    s[v] = llr(w["lp"], ntok, v)
                for c in H0:
                    s[f"full|assistant|only_{c}"] = llr(w["lp"], ntok, "full|assistant", (c,))
                words = set(re.findall(r"[a-z]+", window_text(asr.get(vid, []), x["start"], x["end"]).lower()))
                recs.append({"label": int(y[m].mean() >= 0.5), "mention": int(bool(words & GROUP_WORDS)), **s})
            if len(recs) >= 2 and len({r["label"] for r in recs}) == 2:
                per_video[vid] = {k: roc_auc_score([r["label"] for r in recs], [r[k] for r in recs]) for k in signals}
                for k in ("z_speech", "full|assistant"):
                    v = np.array([r[k] for r in recs]); sd = v.std()
                    if sd > 0:
                        for r, zz in zip(recs, (v - v.mean()) / sd):
                            rows_mech.append((k, zz, r["label"], r["mention"]))
        rng = np.random.default_rng(SEED)
        vids = sorted(per_video)
        res = {"n_videos": len(vids), "n_window_grid_mismatch": n_mismatch, "mean_auc": {}, "delta_vs_z_speech": {}}
        for k in signals:
            res["mean_auc"][k] = float(np.mean([per_video[v][k] for v in vids]))
        for k in signals[1:]:
            res["delta_vs_z_speech"][k] = boot_ci([per_video[v][k] - per_video[v]["z_speech"] for v in vids], rng)
        mech = {}
        for k in ("z_speech", "full|assistant"):
            R = np.array([(zz, lab, men) for kk, zz, lab, men in rows_mech if kk == k])
            X = np.c_[np.ones(len(R)), R[:, 1], R[:, 2]]
            beta = np.linalg.lstsq(X, R[:, 0], rcond=None)[0]
            mech[k] = {"b_gt": float(beta[1]), "b_mention": float(beta[2]), "ratio_mention_over_gt": float(beta[2] / beta[1]),
                       "n_windows": int(len(R)), "share_mention": float(R[:, 2].mean())}
        res["mechanism"] = mech
        summary[ds] = res
        lines.append(f"== {ds}: {len(vids)} videos with both window labels among speech windows")
        lines.append(f"  z_speech (SPVL-r2 speech branch, fixed ASR)   window within {res['mean_auc']['z_speech']:.4f}")
        for k in signals[1:]:
            m_, lo, hi = res["delta_vs_z_speech"][k]
            lines.append(f"  {k:32s} {res['mean_auc'][k]:.4f}   minus z_speech {m_:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        for k, v in mech.items():
            lines.append(f"  mechanism {k:15s} b_gt {v['b_gt']:+.3f}  b_mention {v['b_mention']:+.3f}  ratio {v['ratio_mention_over_gt']:+.2f}  "
                         f"(windows {v['n_windows']}, mention share {v['share_mention']:.2f})")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # derived runs: base windows, z_speech replaced by a GLR variant (intercept z_video + mean raw z unchanged)
    for v in VARIANTS:
        d = Path(a.derived_root) / v.replace("|", "_")
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "predictions.jsonl", "w") as fh:
            for key, rb in base.items():
                if key[0] not in a.datasets or key not in glr:
                    continue
                wg = glr[key]["extra"]["windows"]
                wins = []
                for x, w in zip(rb["extra"]["windows"], wg):
                    nw = {k: x[k] for k in ("i", "start", "end", "z") if k in x}
                    if "z_visual" in x:
                        nw["z_visual"] = x["z_visual"]
                    if "z_speech" in x and w.get("has_speech"):
                        nw["z_speech"] = llr(w["lp"], {"assistant": w["ntok_assistant"], "document": w["ntok_document"]}, v)
                    wins.append(nw)
                rec = {**rb, "method": f"glr_derived_{v}", "extra": {**rb["extra"], "windows": wins}}
                fh.write(json.dumps(rec) + "\n")
        (d / "config.json").write_text(json.dumps({"base": a.base, "glr": a.glr, "variant": v,
                                                   "rule": "z_speech := LLR of this variant; z_visual, z, z_video from base"}, indent=2))


if __name__ == "__main__":
    main()
