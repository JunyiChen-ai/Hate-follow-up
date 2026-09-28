#!/usr/bin/env python3
"""Reading conditions per window (README §1; label-free, from the inputs of the reading run).

For every window of a reads file: visual condition f1 / f0 (the window holds at least one of the 20 uniform frames
of data/frames_k20, or none) and speech condition w1 / w0 (the number of words the speech branch was given is above
the corpus median, or not). Word counts use src.video_inputs.window_text on the ASR, with the loader the reads used
(--asr-loader fixed = fill_untimed, the 2026-09-26 fix; old = the loader of the 2026-09-10 family-study runs).
Output: a JSON {dataset: {video_id: [[visual_cond, speech_cond], ...]}} (speech_cond null when the window has no
speech read), plus the medians used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import frame_paths, load_asr, window_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True, help="reads directory holding predictions.jsonl")
    ap.add_argument("--asr-loader", choices=["fixed", "old"], default="fixed")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(Path(a.reads) / "predictions.jsonl") if l.strip()]
    out, info = {}, {}
    for ds in a.datasets:
        segs = load_asr(ds, fill_untimed=a.asr_loader == "fixed")
        rows = [r for r in recs if r["dataset"] == ds and not r.get("error")]
        words, fr = {}, {}
        for r in rows:
            v = r["video_id"]
            fr[v] = [t for t, _ in frame_paths(ds, v, 20, "k20")]
            words[v] = [len(window_text(segs.get(v, []), w["start"], w["end"]).split()) if "z_speech" in w else None
                        for w in r["extra"]["windows"]]
        all_words = np.array([x for v in words.values() for x in v if x is not None])
        med = float(np.median(all_words))
        out[ds] = {}
        n_f1 = n_w1 = n_w = n = 0
        for r in rows:
            v = r["video_id"]; conds = []
            for w, nw in zip(r["extra"]["windows"], words[v]):
                f = "f1" if any(w["start"] <= t < w["end"] for t in fr[v]) else "f0"
                s = None if nw is None else ("w1" if nw > med else "w0")
                conds.append([f, s]); n += 1; n_f1 += f == "f1"
                if s is not None:
                    n_w += 1; n_w1 += s == "w1"
            out[ds][v] = conds
        info[ds] = {"videos": len(rows), "windows": n, "share_f1": n_f1 / n, "speech_windows": n_w, "share_w1": n_w1 / max(n_w, 1),
                    "word_median": med, "asr_loader": a.asr_loader}
        print(ds, info[ds])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"conditions": out, "info": info, "reads": a.reads}, open(a.out, "w"))


if __name__ == "__main__":
    main()
