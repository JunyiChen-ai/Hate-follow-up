#!/usr/bin/env python3
"""Counts over stored gate tables (README §14.2). CPU; reads only runs/20261114_exp_gates/gate_table/tables/<name>.json
written by gate_table.py; no new metric is computed (the cells are the evaluator numbers stored in those files).

For each table name given on the command line:
- the full method's nine numbers;
- cells not below the full method (change > -0.05 point, the convention of gate_table.py) over the counted rows:
  all nine cells per row, and the pooled ROC and within columns only; within cells of Integrator rows are exempt;
- the same count restricted to the 12 rows that exist both with and without the verdict turn (no `Verdict Turn` row,
  no `Linear Evidence` row);
- the same count on corpus-averaged cells (mean of a row's change over the three corpora, per metric). This reading
  was added on 2026-11-17 after the tables were seen (post-hoc; not the gate's definition).

    python experiments/20261114_exp_gates/table_summary.py ns_all__em0__F3 em0__F3 ...
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES = os.path.join(REPO, "runs/20261114_exp_gates/gate_table/tables")
DS = ("HateMM", "HateClipSeg", "DeHate")
METRICS = ("ROC", "PR", "W")
INTEGRATOR = {"w/o Video-Level Term", "w/o Window Aggregate", "w/ Mean over Modalities", "w/o Scope Read"}
APART = {"w/o Scope Read", "w/o Duration Posterior", "w/ EM Evidence", "w/ Verdict Turn", "w/ Linear Evidence (slope 1)"}
NOT_SHARED = {"w/o Verdict Turn", "w/ Linear Evidence"}
THRESHOLD = -0.0005     # -0.05 point


def summary(name):
    res = json.load(open(os.path.join(TABLES, name + ".json")))["res"]
    rows = [r for r in res[DS[0]] if r != "full"]
    out = {"full": {ds: res[ds]["full"]["m"] for ds in DS}}
    for sel, excluded in (("counted", APART), ("shared12", APART | NOT_SHARED)):
        n9 = n_rw = n_mean = n_mean_rw = k = 0
        detail = []
        for r in rows:
            if r in excluded:
                continue
            k += 1
            bad = []
            for i, m in enumerate(METRICS):
                if r in INTEGRATOR and i == 2:
                    continue
                for ds in DS:
                    x = res[ds][r]["d"][i]
                    if x > THRESHOLD:
                        n9 += 1
                        n_rw += i != 1
                        bad.append("%s %s %+.1f" % (ds, m, 100 * x))
                mean = sum(res[ds][r]["d"][i] for ds in DS) / len(DS)
                if mean > THRESHOLD:
                    n_mean += 1
                    n_mean_rw += i != 1
                    bad.append("corpus mean %s %+.2f" % (m, 100 * mean))
            if bad:
                detail.append("%s: %s" % (r, ", ".join(bad)))
        out[sel] = {"rows": k, "cells": n9, "cells_roc_within": n_rw, "mean_cells": n_mean,
                    "mean_cells_roc_within": n_mean_rw, "detail": detail}
    return out


def main():
    for name in sys.argv[1:]:
        s = summary(name)
        print("## %s" % name)
        print("full: " + " | ".join(" / ".join("%.1f" % (100 * x) for x in s["full"][ds]) for ds in DS))
        for sel in ("counted", "shared12"):
            c = s[sel]
            print("%s rows %d: cells not below %d (pooled ROC and within only: %d); corpus-averaged cells not below %d "
                  "(pooled ROC and within only: %d)" % (sel, c["rows"], c["cells"], c["cells_roc_within"],
                                                        c["mean_cells"], c["mean_cells_roc_within"]))
        for line in s["counted"]["detail"]:
            print("  - " + line)
        print()


if __name__ == "__main__":
    main()
