#!/usr/bin/env python3
"""Cached proxy for the Codex round-2 proposal, layer-dependent access to context (README.md §"Codex consultation,
round 2"). Reads the gold: analysis only, never part of a method.
Question: does a window read made with less context carry within-video information that the full-context read
lacks? If not, restricting the late layers to local evidence has nothing to recover.
On the Qwen3-VL-8B reads of the family study (older ASR loader, `runs/20260910_spvl/mllm/q3vl-8b/<arm>`):
- window score = normal score (within corpus) of the full-context read + lambda x that of a reduced-context read;
- window-level within AUC against the gold label (hate share >= .5), mean over videos with both labels;
- paired bootstrap over videos.
Reduced-context reads: nostance (no verdict turn), noctx (no transcript context), noframes (no frames), winonly (no
context at all, one joint question).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from continuity_check import share
from headroom import ROOT, boot_diff, gold

ARMS = ("nostance", "noctx", "noframes", "winonly")
LAMBDAS = (.25, .5, 1.0)


def nscores(path, ds):
    """(video, window index) -> normal score of the window read `z` within the corpus."""
    recs = [r for r in map(json.loads, open(path)) if r["dataset"] == ds and not r.get("error")]
    keys = [(r["video_id"], w["i"]) for r in recs for w in r["extra"]["windows"]]
    z = np.array([w["z"] for r in recs for w in r["extra"]["windows"]], float)
    ns = norm.ppf((rankdata(z) - .5) / len(z))
    wins = {r["video_id"]: [(w["i"], w["start"], w["end"]) for w in r["extra"]["windows"]] for r in recs}
    return dict(zip(keys, ns)), wins


def within(score, wins, Y):
    out = {}
    for v, ws in wins.items():
        if v not in Y:
            continue
        lab = np.array([share(Y[v], s, e) for _, s, e in ws])
        keep = ~np.isnan(lab)
        b = lab[keep] >= .5
        if b.all() or not b.any():
            continue
        out[v] = float(roc_auc_score(b, np.array([score[(v, i)] for i, _, _ in ws])[keep]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runs/20260910_spvl/mllm/q3vl-8b"))
    ap.add_argument("--out", default=str(ROOT / "runs/20260928_headroom/context_mix"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {}
    for ds in ("HateMM", "HateClipSeg"):
        Y = gold(ds)
        full, wins = nscores(Path(a.root) / "full/predictions.jsonl", ds)
        base = within(full, wins, Y)
        lines.append(f"== {ds}: window-level within, full context {np.mean(list(base.values())):.4f} (n {len(base)} videos)")
        summary[ds] = {"full": float(np.mean(list(base.values()))), "n": len(base)}
        for arm in ARMS:
            red, _ = nscores(Path(a.root) / f"{arm}/predictions.jsonl", ds)
            alone = within(red, wins, Y)
            d = boot_diff(alone, base)
            row = [f"  {arm:9s} alone {np.mean(list(alone.values())):.4f} ({d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}])"]
            summary[ds][arm] = {"alone": d}
            for lam in LAMBDAS:
                mix = within({k: full[k] + lam * red.get(k, 0.0) for k in full}, wins, Y)
                d = boot_diff(mix, base); summary[ds][arm][f"mix_{lam:g}"] = d
                row.append(f"full + {lam:g} x {arm}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
            lines.append(";  ".join(row))
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
