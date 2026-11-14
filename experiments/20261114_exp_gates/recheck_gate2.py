#!/usr/bin/env python3
"""Recompute the gate-2 lines of stored gate tables against the current runs/20261113_lf_baselines/gate/gate.json
(README §5.1; after Probe-VAD left the baseline set, 2026-11-15). Reads the full-method metrics stored in each table's
JSON; writes nothing but stdout."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_table as T  # noqa: E402

for f in sorted(glob.glob(str(T.OUT / "tables" / "*.json"))):
    j = json.load(open(f))
    res = j["res"]
    store = T.Store(["m2"], "prob")
    chk = T.checks(store, j["config"], res)
    a = chk["DeHate"]["vs_best_A"]; c = chk["DeHate"]["vs_best_C"]
    cur = "; ".join(f"{ds} " + " / ".join(f"{100 * x:+.2f}" for x in chk[ds]["vs_current"][:2]) for ds in ("HateMM", "HateClipSeg"))
    print(f"{Path(f).stem:32s} DeHate − best label-free (all published) ROC {100 * a[0][0]:+.1f} ({a[0][1]}), PR {100 * a[1][0]:+.1f} ({a[1][1]}); "
          f"peer-reviewed only ROC {100 * c[0][0]:+.1f}, PR {100 * c[1][0]:+.1f} | vs current {cur}")
