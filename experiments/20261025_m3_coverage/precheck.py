#!/usr/bin/env python3
"""Pre-check of a per-video scalar against GT coverage (CPU, analysis only; GT read for analysis only).

Input: jsonl with dataset, video_id, value (one scalar per test video). For each corpus (and the HateClipSeg
hateful-only diagnostic GT) and two scopes (all test videos; only videos whose GT has a positive frame), reports with
a video bootstrap (resampling videos within the scope):
  - Spearman of the value with GT coverage (share of positive frames), and of the references: window mean (mean of
    the SPVL-r2 window reads z), z_video, K = z_video + window mean, the calibrated r6 key (monotone in K);
  - the paired difference rho(value, coverage) - rho(window mean, coverage);
  - Spearman of the value with z_video, the window mean and K (collapse onto the verdict / onto the window mean);
  - partial Spearman of the value with coverage given the window mean, and given (z_video, window mean);
    reference: partial Spearman of the window mean with coverage given z_video;
  - all-videos scope only: video-level ROC-AUC for "GT has a positive frame".

  python experiments/20261025_m3_coverage/precheck.py --values X.jsonl --name X
  python experiments/20261025_m3_coverage/precheck.py --selftest     # value = window mean, and a random value
Output: runs/20261025_m3_coverage/analysis_precheck_<name>/{summary.json, summary.txt, run.log, config.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATASETS, OUT_ROOT, ci, coverage, gt_sets, load_r6, load_reads, load_values, partial_spearman,  # noqa: E402
                    spearman, start_run, video_auc, write_values)

SCOPES = ("all", "hateful")


def stats(d, idx, scope):
    v, wm, zv, K, lo, c = (d[k][idx] for k in ("value", "wm", "zv", "K", "lo", "cov"))
    s = {"rho_value_cov": spearman(v, c), "rho_wm_cov": spearman(wm, c), "rho_zv_cov": spearman(zv, c),
         "rho_K_cov": spearman(K, c), "rho_key_cov": spearman(lo, c)}
    s["diff_value_minus_wm"] = s["rho_value_cov"] - s["rho_wm_cov"]
    s.update({"rho_value_zv": spearman(v, zv), "rho_value_wm": spearman(v, wm), "rho_value_K": spearman(v, K),
              "rho_zv_wm": spearman(zv, wm),
              "partial_value_cov_given_wm": partial_spearman(v, c, [wm]),
              "partial_value_cov_given_zv": partial_spearman(v, c, [zv]),
              "partial_value_cov_given_zv_wm": partial_spearman(v, c, [zv, wm]),
              "partial_wm_cov_given_zv": partial_spearman(wm, c, [zv])})
    if scope == "all":
        lab = (c > 0).astype(int)
        s.update({"vauc_value": video_auc(v, lab), "vauc_wm": video_auc(wm, lab), "vauc_zv": video_auc(zv, lab),
                  "vauc_K": video_auc(K, lab)})
    return s


def analyse(values, gts, r6, reads, B=2000, seed=0):
    res = {}
    rng = np.random.default_rng(seed)
    for gname, (ds, Y) in gts.items():
        vids = sorted(v for (d, v) in r6 if d == ds and v in Y)
        miss = [v for v in vids if (ds, v) not in values]
        vids = [v for v in vids if (ds, v) in values]
        d = {"value": np.array([values[(ds, v)] for v in vids]),
             "wm": np.array([reads[(ds, v)]["wm"] for v in vids]),
             "zv": np.array([r6[(ds, v)]["zv"] for v in vids]),
             "lo": np.array([r6[(ds, v)]["lo"] for v in vids]),
             "cov": np.array([coverage(Y[v], len(r6[(ds, v)]["score"])) for v in vids])}
        d["K"] = d["zv"] + d["wm"]
        res[gname] = {"n_videos": len(vids), "n_missing_values": len(miss), "missing_examples": miss[:5],
                      "n_hateful": int((d["cov"] > 0).sum())}
        for scope in SCOPES:
            idx = np.arange(len(vids)) if scope == "all" else np.flatnonzero(d["cov"] > 0)
            if len(idx) < 3:                     # too few videos (e.g. a smoke run): undefined, no bootstrap
                names = [k for k, _ in ROWS if scope == "all" or not k.startswith("vauc")]
                res[gname][scope] = {"n": int(len(idx)), **{k: {"point": float("nan"), "ci95": [float("nan")] * 2}
                                                           for k in names}}
                continue
            point = stats(d, idx, scope)
            boot = {k: [] for k in point}
            for _ in range(B):
                bi = idx[rng.integers(0, len(idx), len(idx))]
                sb = stats(d, bi, scope)
                for k in point:
                    boot[k].append(sb[k])
            res[gname][scope] = {"n": int(len(idx)), **{k: {"point": point[k], "ci95": ci(boot[k])} for k in point}}
    return res


ROWS = [("rho_value_cov", "rho(value, coverage)"), ("rho_wm_cov", "rho(window mean, coverage)"),
        ("rho_zv_cov", "rho(z_video, coverage)"), ("rho_K_cov", "rho(K = z_video + window mean, coverage)"),
        ("rho_key_cov", "rho(calibrated key, coverage)"), ("diff_value_minus_wm", "rho(value,cov) - rho(wm,cov)"),
        ("rho_value_zv", "rho(value, z_video)"), ("rho_value_wm", "rho(value, window mean)"),
        ("rho_value_K", "rho(value, K)"), ("rho_zv_wm", "rho(z_video, window mean)"),
        ("partial_value_cov_given_wm", "partial rho(value, cov | wm)"),
        ("partial_value_cov_given_zv", "partial rho(value, cov | z_video)"),
        ("partial_value_cov_given_zv_wm", "partial rho(value, cov | z_video, wm)"),
        ("partial_wm_cov_given_zv", "ref: partial rho(wm, cov | z_video)"),
        ("vauc_value", "video AUC value (GT any positive)"), ("vauc_wm", "video AUC window mean"),
        ("vauc_zv", "video AUC z_video"), ("vauc_K", "video AUC K")]


def table(res, title):
    lines = [title]
    for gname, r in res.items():
        lines.append(f"\n== {gname}: {r['n_videos']} videos ({r['n_hateful']} with a GT-positive frame); "
                     f"values missing for {r['n_missing_values']}")
        lines.append(f"{'':44s} " + "  ".join(f"{s + ' (n=' + str(r[s]['n']) + ')':>26s}" for s in SCOPES))
        for k, lab in ROWS:
            cells = []
            for s in SCOPES:
                x = r[s].get(k)
                cells.append(f"{x['point']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" if x else "")
            lines.append(f"{lab:44s} " + "  ".join(f"{c:>26s}" for c in cells))
    return "\n".join(lines)


def run_one(values_path, name, out, B, seed, log=None, gts=None, r6=None, reads=None):
    gts = gts or gt_sets(); r6 = r6 or load_r6(); reads = reads or load_reads()
    values = load_values(values_path)
    res = analyse(values, gts, r6, reads, B, seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"values": str(values_path), "B": B, "seed": seed, "results": res}, indent=1))
    txt = table(res, f"precheck {name}: values {values_path}; video bootstrap B={B} seed={seed}; "
                     f"HateClipSeg_strict = hateful-only GT, diagnostic")
    (out / "summary.txt").write_text(txt + "\n")
    if log:
        log(f"wrote {out / 'summary.txt'}")
    return res, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", help="jsonl: dataset, video_id, value")
    ap.add_argument("--name", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), help="any of HateMM HateClipSeg DeHate")
    ap.add_argument("--selftest", action="store_true", help="value = window mean (exact checks) and a random value")
    a = ap.parse_args()
    if not a.selftest and not a.values:
        raise SystemExit("--values or --selftest")
    name = "selftest" if a.selftest else (a.name or Path(a.values).stem)
    out = Path(a.out) if a.out else OUT_ROOT / f"analysis_precheck_{name}"
    log = start_run(out, "experiments/20261025_m3_coverage/precheck.py", vars(a))
    gts, r6, reads = gt_sets(a.datasets), load_r6(datasets=a.datasets), load_reads(datasets=a.datasets)
    if not a.selftest:
        res, txt = run_one(a.values, name, out, a.B, a.seed, log, gts, r6, reads)
        print(txt); log("RUN_DONE"); return
    keys = sorted(k for k in r6)
    rng = np.random.default_rng(12345)
    vals = {"wmean": {k: reads[k]["wm"] for k in keys}, "random": {k: float(rng.normal()) for k in keys}}
    ok = True
    for nm, v in vals.items():
        p = out / "selftest_values" / f"{nm}.jsonl"; write_values(p, v)
        res, txt = run_one(p, f"selftest_{nm}", out / nm, a.B, a.seed, log, gts, r6, reads)
        print(txt)
        for gname, r in res.items():
            for s in SCOPES:
                x = r[s]
                if nm == "wmean":
                    checks = {"rho(value,cov) == rho(wm,cov)": abs(x["rho_value_cov"]["point"] - x["rho_wm_cov"]["point"]) < 1e-12,
                              "rho(value,wm) == 1": abs(x["rho_value_wm"]["point"] - 1) < 1e-12,
                              "partial(value,cov|wm) == 0": abs(x["partial_value_cov_given_wm"]["point"]) < 1e-9}
                else:
                    lo_, hi_ = x["rho_value_cov"]["ci95"]
                    checks = {"random: CI of rho(value,cov) contains 0": lo_ <= 0 <= hi_}
                for c, passed in checks.items():
                    log(f"SELFTEST {nm} {gname} {s}: {c}: {'PASS' if passed else 'FAIL'} "
                        f"(rho value/cov {x['rho_value_cov']['point']:+.4f}, wm/cov {x['rho_wm_cov']['point']:+.4f}, "
                        f"partial|wm {x['partial_value_cov_given_wm']['point']:+.4f})")
                    ok &= passed if nm == "wmean" else True
    log(f"SELFTEST exact checks (window mean) {'PASS' if ok else 'FAIL'}; random-value checks are reported, not asserted")
    log("RUN_DONE")


if __name__ == "__main__":
    main()
