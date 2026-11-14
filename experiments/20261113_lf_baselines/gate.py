#!/usr/bin/env python3
"""Nine-cell gate table (README §3, user gate of 2026-11-14): best published label-free cell against the naive
Qwen3-VL-8B row.

Gate: for each of the 9 cells (3 corpora x pooled ROC-AUC, pooled PR-AUC, within-video macro ROC-AUC) take the best
value over all published label-free rows; at least 7 cells must be >= the naive row and every other cell at most
1.0 point below it.

Rows read (only metrics.json written by src/eval/evaluate_four_datasets.py):
- the published rows of the main table and the Qwen3-VL-8B re-runs of §2 (compare.py ROWS, primary read-outs);
- every configuration of the §3 methods, runs/20261113_lf_baselines/<method>/<DS>/<variant>/<config>/metrics.json,
  kept only when the same <variant>/<config> exists for all three corpora (one configuration = one table row; no
  per-corpus choice). DEFAULTS marks the configuration fixed before the run; every other configuration of a §3
  method is a test-selected candidate (favours the baseline) and is marked so.
Prints the table for three row sets: (A) all published rows, (B) paper-default configurations only, (C) peer-reviewed
methods only. Writes runs/20261113_lf_baselines/gate/gate.{md,json}.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as C     # noqa: E402

RUNS = C.Q
NAIVE = C.N + "/vlm/{ds}/metrics.json"
CELLS = [(ds, k) for ds in C.DATASETS for k in range(3)]
KEY_NAMES = ("pooled ROC", "pooled PR", "within")
# method directory -> (display name, peer-reviewed venue or None for a preprint, default <variant>/<config>)
METHODS = {
    "corevad": ("CoReVAD", "ICPR 2026", "u8/prob_lrc1_tau0.05_g9-5_pos1"),
    "anyanomaly": ("AnyAnomaly", "WACV 2026", "u8/prob"),
    "cogvadu": ("Cog-VADU", "TMLR 2026", "s4/prob_default"),
    "lavad": ("LAVAD", "CVPR 2024", "prob_default"),
    "lela": ("LELA", None, "prob_default"),
}
# Run and then excluded as a baseline by the user on 2026-11-15 (README §3.1); not read into any row set.
EXCLUDED_NOTE = ("Probe-VAD (arXiv 2609.17211) was run and then excluded as a baseline by the user on 2026-11-15; "
                 "numbers in README §3.1.")
PEER_REVIEWED_MAIN = {"ZS-CLIP", "ZS-ImageBind", "LAVAD", "EventVAD", "PRISM", "VideoMind",
                      "MARS, fp32 label log-odds", "EventVAD, fp32 expected score"}


def rows():
    """[(name, {ds: [3 values]}, {ds: path}, default?, peer-reviewed?)]"""
    out = []
    for g, name, pat in C.ROWS:
        if g not in C.PUBLISHED:
            continue
        got = {ds: C.read(pat, ds) for ds in C.DATASETS}
        if all(got.values()):
            label = name + (" (main table)" if g == "published" else " (Qwen3-VL-8B)")
            out.append((label, {d: v[0] for d, v in got.items()}, {d: v[1] for d, v in got.items()}, True,
                        name in PEER_REVIEWED_MAIN))
    for m, (disp, venue, default) in METHODS.items():
        base = os.path.join(RUNS, m, C.DATASETS[0])
        for p in sorted(glob.glob(os.path.join(base, "**", "metrics.json"), recursive=True)):
            cfg = os.path.relpath(os.path.dirname(p), base)
            pat = os.path.join(RUNS, m, "{ds}", cfg, "metrics.json")
            got = {ds: C.read(pat, ds) for ds in C.DATASETS}
            if not all(got.values()):
                continue
            is_def = cfg == default
            label = "%s on Qwen3-VL-8B, %s%s" % (disp, cfg, "" if is_def else " [test-selected candidate]")
            out.append((label, {d: v[0] for d, v in got.items()}, {d: v[1] for d, v in got.items()}, is_def,
                        venue is not None))
    return out


def gate(rws, naive):
    cells, n_ok, worst = [], 0, 0.0
    for ds, k in CELLS:
        best = max(rws, key=lambda r: r[1][ds][k]) if rws else None
        nv = 100 * naive[ds][k]
        bv = 100 * best[1][ds][k] if best else float("nan")
        ok = bv >= nv
        n_ok += ok
        if not ok:
            worst = max(worst, nv - bv)
        cells.append({"dataset": ds, "metric": KEY_NAMES[k], "naive": nv, "best": bv, "row": best[0] if best else None,
                      "source": best[2][ds] if best else None, "pass": bool(ok), "margin": bv - nv})
    return {"cells": cells, "n_pass": int(n_ok), "max_shortfall": worst,
            "gate": bool(n_ok >= 7 and worst <= 1.0)}


def fmt(title, g):
    lines = ["### %s" % title, "", "| corpus | metric | naive | best published | row | margin | pass |", "|---|---|---|---|---|---|---|"]
    for c in g["cells"]:
        lines.append("| %s | %s | %.2f | %.2f | %s | %+.2f | %s |" % (
            c["dataset"], c["metric"], c["naive"], c["best"], c["row"], c["margin"], "yes" if c["pass"] else "no"))
    lines.append("")
    lines.append("Cells >= naive: %d of 9; largest shortfall %.2f points; gate %s." % (
        g["n_pass"], g["max_shortfall"], "PASSES" if g["gate"] else "fails"))
    lines.append("")
    return lines


def main():
    naive = {ds: C.read(NAIVE, ds)[0] for ds in C.DATASETS}
    rws = rows()
    sets = [("A. all published rows (main table, Qwen3-VL-8B re-runs, every configuration tried)", rws),
            ("B. configurations fixed before the run only (no test-selected configuration)", [r for r in rws if r[3]]),
            ("C. peer-reviewed methods only (every configuration tried)", [r for r in rws if r[4]]),
            ("D. peer-reviewed methods, configurations fixed before the run", [r for r in rws if r[3] and r[4]])]
    lines, res = [], {}
    for title, sub in sets:
        g = gate(sub, naive)
        res[title[:1]] = g
        lines += fmt(title, g)
    lines.append(EXCLUDED_NOTE)
    lines.append("")
    lines.append("### Rows read")
    lines.append("")
    lines.append("| row | HateMM ROC / PR / within | HateClipSeg ROC / PR / within | DeHate ROC / PR / within | cells >= naive |")
    lines.append("|---|---|---|---|---|")
    lines.append("| naive Qwen3-VL-8B, plain question | %s | — |" % " | ".join(
        " / ".join("%.1f" % (100 * x) for x in naive[ds]) for ds in C.DATASETS))
    for name, vals, _, _, _ in rws:
        n = sum(vals[ds][k] >= naive[ds][k] for ds, k in CELLS)
        lines.append("| %s | %s | %d |" % (name, " | ".join(
            " / ".join("%.1f" % (100 * x) for x in vals[ds]) for ds in C.DATASETS), n))
    out = os.path.join(RUNS, "gate")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gate.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "gate.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
