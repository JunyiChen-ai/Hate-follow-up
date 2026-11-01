#!/usr/bin/env python3
"""Label-free read diagnostics of the reader-ablation arms against the adopted reads (no label file is read).

For every fixed-grid arm and corpus: verdict Spearman with the base (z_video over videos), median per-video Spearman of
the window read z, share of windows whose z changes sign, mean window z shift. For no_isolation also the persistence
of consecutive reads in the order they were asked (visual i, speech i, visual i+1, ...): P(read > 0 | the previous
read in the conversation > 0) and P(read > 0 | previous <= 0), with the same order applied to the base reads.
Writes runs/20261101_reader_ablation/diag_reads.txt.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20261101_reader_ablation"
BASE = {"HateMM": ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl",
        "HateClipSeg": ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl",
        "DeHate": ROOT / "runs/20261031_window_evidence/DeHate/adopted/predictions.jsonl"}
ARMS = ["no_transcript", "no_prefix_frames", "no_stance", "no_isolation", "joint"]


def load(path, ds):
    return {r["video_id"]: r for r in map(json.loads, open(path)) if r["dataset"] == ds and not r.get("error")}


def order(rec):
    seq = []
    for w in rec["extra"]["windows"]:
        seq.append(w["z_visual"])
        if "z_speech" in w:
            seq.append(w["z_speech"])
    return np.array(seq)


def persistence(seqs):
    prev, cur = np.concatenate([s[:-1] for s in seqs if len(s) > 1]), np.concatenate([s[1:] for s in seqs if len(s) > 1])
    return float((cur[prev > 0] > 0).mean()), float((cur[prev <= 0] > 0).mean()), int((prev > 0).sum())


def main():
    lines = []
    for ds in ("HateMM", "HateClipSeg", "DeHate"):
        base = load(BASE[ds], ds)
        lines.append(f"== {ds} ({len(base)} videos)")
        for arm in ARMS:
            run = load(R / ds / arm / "predictions.jsonl", ds)
            vids = sorted(set(run) & set(base))
            zv = spearmanr([base[v]["extra"]["z_video"] for v in vids], [run[v]["extra"]["z_video"] for v in vids])[0]
            rho, flips, shift, n = [], 0, [], 0
            for v in vids:
                a = np.array([w["z"] for w in base[v]["extra"]["windows"]]); b = np.array([w["z"] for w in run[v]["extra"]["windows"]])
                assert len(a) == len(b)
                if len(a) > 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
                    rho.append(spearmanr(a, b)[0])
                flips += int(((a > 0) != (b > 0)).sum()); n += len(a); shift.append(b - a)
            line = (f"  {arm:17s} verdict Spearman {zv:.3f}  window z Spearman median {np.median(rho):.3f}  "
                    f"sign changes {flips / n:.3f} of {n} windows  mean z shift {np.concatenate(shift).mean():+.2f}  "
                    f"share z>0 base {np.mean([w['z'] > 0 for v in vids for w in base[v]['extra']['windows']]):.3f} "
                    f"arm {np.mean([w['z'] > 0 for v in vids for w in run[v]['extra']['windows']]):.3f}")
            if arm == "no_isolation":
                pa, na, ka = persistence([order(run[v]) for v in vids])
                pb, nb, kb = persistence([order(base[v]) for v in vids])
                line += (f"\n{'':19s}persistence P(read>0 | previous read>0) / P(read>0 | previous<=0): "
                         f"no_isolation {pa:.3f} / {na:.3f} (n prev>0 {ka}); base, same order {pb:.3f} / {nb:.3f} (n {kb})")
            lines.append(line)
    (R / "diag_reads.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
