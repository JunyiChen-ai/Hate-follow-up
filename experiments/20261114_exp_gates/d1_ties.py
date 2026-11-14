#!/usr/bin/env python3
"""D1 diagnostic (README §8): saturation and ties of the decoder's cell posterior.

Per corpus and decode: (a) share of cells with cell_prob >= 1 - 1e-12 or <= 1e-12 (the decoder's clip bounds);
(b) share of within-video cell pairs whose cell_prob values are exactly equal; (c) among those tied pairs, the share
whose cells have different GT (one positive, one negative); also the share of cells with min(p, 1 - p) below 1e-6
and 1e-9 (near the clip bound). Cell GT = majority of the cell's 4 fps GT frames (16 frames
per 4 s cell; positive when more than half are positive). Labels are used only for (c).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CELL_FRAMES = 16
DECODES = {"m2": {"main": "runs/20261101_paper_redecode/main/m2", "dehate": "runs/20261101_paper_redecode/dehate/m2"},
           "kap0.3": {"main": "runs/20261114_exp_gates/decoder_scan/main/kap0.3",
                      "dehate": "runs/20261114_exp_gates/decoder_scan/dehate/kap0.3"}}
PARTS = {"HateMM": "main", "HateClipSeg": "main", "DeHate": "dehate"}


def gt(ds):
    z = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, np.int8) for v, s, y in zip(z["video_ids"], z["split"], z["y4"]) if str(s) == "test"}


def stats(path, ds, G):
    n_cells = n_sat = pairs = tied = mixed = near6 = near9 = 0
    for line in open(ROOT / path / "predictions.jsonl"):
        r = json.loads(line)
        if r["dataset"] != ds or r.get("error") or r["video_id"] not in G:
            continue
        p = np.asarray(r["extra"]["cell_prob"], float)
        y = G[r["video_id"]]
        yc = np.array([y[i * CELL_FRAMES:(i + 1) * CELL_FRAMES].mean() > 0.5 if i * CELL_FRAMES < len(y) else False
                       for i in range(len(p))])
        n = len(p)
        n_cells += n
        n_sat += int(np.sum((p >= 1 - 1e-12) | (p <= 1e-12)))
        near6 += int(np.sum(np.minimum(p, 1 - p) < 1e-6)); near9 += int(np.sum(np.minimum(p, 1 - p) < 1e-9))
        pairs += n * (n - 1) // 2
        _, inv = np.unique(p, return_inverse=True)
        for g in range(inv.max() + 1):
            m = inv == g
            k = int(m.sum())
            if k > 1:
                tied += k * (k - 1) // 2
                pos = int(yc[m].sum())
                mixed += pos * (k - pos)
    return {"cells": n_cells, "saturated": n_sat / n_cells, "near_1e-6": near6 / n_cells, "near_1e-9": near9 / n_cells, "tied_pairs": tied / pairs,
            "mixed_among_tied": (mixed / tied) if tied else None, "n_pairs": pairs, "n_tied": tied}


def main():
    out = {}
    for name, parts in DECODES.items():
        for ds, part in PARTS.items():
            out[f"{name} {ds}"] = stats(parts[part], ds, gt(ds))
            s = out[f"{name} {ds}"]
            print(f"{name:7s} {ds:12s} cells {s['cells']:6d}  (a) saturated {100 * s['saturated']:.2f} %  "
                  f"(b) tied pairs {100 * s['tied_pairs']:.2f} %  (c) GT-mixed among tied {100 * (s['mixed_among_tied'] or 0):.2f} %  "
                  f"min(p,1-p)<1e-6 {100 * s['near_1e-6']:.2f} %, <1e-9 {100 * s['near_1e-9']:.2f} %")
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
