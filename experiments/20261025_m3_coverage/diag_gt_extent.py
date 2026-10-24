#!/usr/bin/env python3
"""DIAGNOSTIC ONLY (reads the test GT into a key; not a method, not an upper bound).

Question: how sensitive are the pooled metrics to the extent term of the r6 video key? The extent term (window mean
in K = z_video + window mean) is replaced by quantities built from the GT, with M2 (the within-video rank term) and
everything else unchanged, through key_eval.py:
  gtcov            s = GT coverage (share of GT-positive frames); all key_eval forms, transform none, scale match.
                   Caution: non-hateful videos all get coverage 0, so this also hands the key the video label.
  gtcov_logit      s = GT coverage, transform logit(clip(s, 1e-4, 1 - 1e-4)); forms replace / add.
  rankperm         the window means of the GT-hateful videos are reassigned among themselves so that their order
                   follows GT coverage (ties broken by the original window mean); non-hateful videos keep theirs.
                   The window mean's distribution among hateful and among non-hateful videos is unchanged, so this
                   isolates the ordering-by-amount information among hateful videos. Form replace, scale raw.
  *_strict         the same with the HateClipSeg hateful-only GT (HateMM keeps the protocol GT); only the
                   HateClipSeg_strict rows of these runs are meaningful.
Output: runs/20261025_m3_coverage/analysis_diag_gt_extent/{values/, <variant>/, summary.txt, summary.json}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_ROOT, ROOT, coverage, gt_sets, load_r6, load_reads, start_run, write_values  # noqa: E402

OUT = OUT_ROOT / "analysis_diag_gt_extent"
VARIANTS = {   # name: (values, extra key_eval args)
    "gtcov": ("gtcov", []),
    "gtcov_logit": ("gtcov", ["--transform", "logit", "--forms", "replace", "add"]),
    "rankperm": ("rankperm", ["--scale", "raw", "--forms", "replace"]),
    "gtcov_strict": ("gtcov_strict", ["--forms", "replace", "product_K_prob"]),
    "rankperm_strict": ("rankperm_strict", ["--scale", "raw", "--forms", "replace"]),
}


def build_values(gts, r6, reads):
    out = {"gtcov": {}, "rankperm": {}, "gtcov_strict": {}, "rankperm_strict": {}}
    for tag, sets in (("", {"HateMM": "HateMM", "HateClipSeg": "HateClipSeg"}),
                      ("_strict", {"HateMM": "HateMM", "HateClipSeg": "HateClipSeg_strict"})):
        for ds, gname in sets.items():
            Y = gts[gname][1]
            ks = sorted(k for k in r6 if k[0] == ds)
            cov = np.array([coverage(Y[k[1]], len(r6[k]["score"])) for k in ks])
            wm = np.array([reads[k]["wm"] for k in ks])
            for k, c in zip(ks, cov):
                out["gtcov" + tag][k] = float(c)
            H = np.flatnonzero(cov > 0)
            perm = wm.copy()
            order = H[np.lexsort((wm[H], cov[H]))]          # hateful videos by coverage, then window mean
            perm[order] = np.sort(wm[H])
            for k, x in zip(ks, perm):
                out["rankperm" + tag][k] = float(x)
    return out


def main():
    log = start_run(OUT, "experiments/20261025_m3_coverage/diag_gt_extent.py", {"variants": VARIANTS})
    log("DIAGNOSTIC: GT-derived keys; not a method and not an upper bound")
    gts, r6, reads = gt_sets(), load_r6(), load_reads()
    vals = build_values(gts, r6, reads)
    for nm, v in vals.items():
        write_values(OUT / "values" / f"DIAGNOSTIC_GT_{nm}.jsonl", v)
    jobs = {}
    for var, (vn, extra) in VARIANTS.items():
        cmd = [sys.executable, str(ROOT / "experiments/20261025_m3_coverage/key_eval.py"),
               "--values", str(OUT / "values" / f"DIAGNOSTIC_GT_{vn}.jsonl"), "--name", var, "--out", str(OUT / var), *extra]
        log(f"launch {var}: {' '.join(cmd[1:])}")
        (OUT / var).mkdir(parents=True, exist_ok=True)
        jobs[var] = subprocess.Popen(cmd, cwd=ROOT, stdout=open(OUT / f"{var}.stdout", "w"), stderr=subprocess.STDOUT)
    for var, p in jobs.items():
        if p.wait() != 0:
            raise SystemExit(f"{var} FAILED (exit {p.returncode}); see {OUT / (var + '.stdout')}")
        log(f"{var} done")
    lines = ["DIAGNOSTIC ONLY: extent term of the r6 key replaced by GT-derived quantities (see the docstring); "
             "d = arm - r6_bma, paired video bootstrap 95% intervals"]
    allres = {}
    for var in VARIANTS:
        s = json.load(open(OUT / var / "summary.json")); allres[var] = s
        for gname, r in s["bootstrap"].items():
            if var.endswith("_strict") and gname != "HateClipSeg_strict":
                continue
            if not var.endswith("_strict") and gname == "HateClipSeg_strict":
                continue
            for a, x in r["delta"].items():
                p = r["point"][a]
                lines.append(f"{var:16s} {gname:19s} {a:16s} {p[0]:.4f} / {p[1]:.4f}   dROC {x['dROC']:+.4f} "
                             f"[{x['ROC_ci95'][0]:+.4f},{x['ROC_ci95'][1]:+.4f}]  dPR {x['dPR']:+.4f} "
                             f"[{x['PR_ci95'][0]:+.4f},{x['PR_ci95'][1]:+.4f}]")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps({v: s["bootstrap"] for v, s in allres.items()}, indent=1))
    print("\n".join(lines))
    log("RUN_DONE")


if __name__ == "__main__":
    main()
