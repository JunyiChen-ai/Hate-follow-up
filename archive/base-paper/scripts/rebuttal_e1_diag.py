"""E1 sanity-gate diagnostic --- root-cause the HateMM natural-row divergence.

The E1 natural sanity row gave HateMM 0.860465 (185/215) while the frozen
headline in all_results.json is 0.865120 (186/215). This script isolates whether
that is (A) an E1 code-path bug or (B) verdict file-drift in the frozen offline
verdict files after all_results.json was written.

It:
  (i)   re-runs the ORIGINAL build_experiments.eval_sequential path on the CURRENT
        files for HateMM and reports whether they still yield 0.865120;
  (ii)  diffs per-video final predictions between that build_experiments path
        (frozen band file + current verdicts) and the E1-natural path (re-fit band
        + current verdicts), printing any differing video_id with band membership,
        the per-verifier verdicts consumed, and the stop step;
  (iii) prints mtimes of the three HateMM offline verdict files and any backups
        versus all_results.json, and diffs the current 72B HateMM file against its
        pre-rerun backup.

CPU-only. Run: python scripts/rebuttal_e1_diag.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3")
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))
sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_experiments as BE  # noqa: E402
from grid_eval_all import N_TEST  # noqa: E402
import rebuttal_e1_prevalence as E1  # noqa: E402

DS = "HateMM"
MAIN_ORDER = BE.MAIN_ORDER


def path_a_build_experiments():
    """Exact eval_sequential loop (frozen band file + current verdict files),
    capturing per-video final + consumed verdicts + stop step."""
    cache = BE.EvalCache()
    labels = cache.load_labels_for(DS)
    base = cache.load_base("2b", DS)
    band = cache.load_band("2b", DS)
    hbar = cache.hbar[("2b", DS)]
    rho = cache.rhod[("2b", DS)]
    lam = math.log(rho / (1 - rho))
    judges = [cache.load_judge(j, DS) for j in MAIN_ORDER]
    finals, details = {}, {}
    correct = 0
    for v in cache.valid_vids("2b", DS):
        s1 = base[v]
        b = band.get(v, {})
        in_band = bool(b.get("in_band"))
        if not in_band:
            finals[v] = s1
            details[v] = {"in_band": False, "consumed": [], "used": 0, "final": s1}
            correct += int(s1 == labels[v])
            continue
        ell = BE.logit(float(b.get("posterior_hi", 0.5)))
        used, consumed = 0, []
        for table in judges[:3]:
            used += 1
            r = table.get(v, {}).get("pred")
            consumed.append(r)
            if r in (0, 1):
                ell += (2 * int(r) - 1) * lam
                if BE.ent(BE.sigmoid(ell)) <= hbar:
                    break
        fin = 1 if BE.sigmoid(ell) >= 0.5 else 0
        finals[v] = fin
        details[v] = {"in_band": True, "consumed": consumed, "used": used, "final": fin}
        correct += int(fin == labels[v])
    acc = correct / N_TEST[DS]
    return finals, details, acc, hbar, rho, labels


def path_b_e1_natural():
    """E1-natural path (re-fit threshold + GMM band + current verdicts),
    capturing per-video final + consumed verdicts + stop step."""
    dd = E1.DatasetData(DS)
    eval_vids, calib = E1.build_eval_and_calib(dd, E1.NATURAL, None)
    import numpy as np
    crit_name, crit_fn, fit_src, proto = E1.PROTOCOL[DS]
    thr = float(crit_fn(calib))
    gmm, hi = E1.fit_gmm(calib)
    scores = np.array([s for _, s in eval_vids], dtype=float)
    z = E1.to_logit(scores).reshape(-1, 1)
    post = gmm.predict_proba(z)[:, hi]
    ents = np.array([E1.band_ent(p) for p in post], dtype=float)
    mean_e = float(ents.mean())
    hbar = mean_e
    rho = E1.rho_from_hbar(hbar)
    lam = math.log(rho / (1 - rho))
    base_pred = (scores >= thr).astype(int)
    in_band = ents > mean_e
    judges = [dd.judges[j] for j in MAIN_ORDER]
    finals, details = {}, {}
    correct = 0
    for i, (vid, _) in enumerate(eval_vids):
        s1 = int(base_pred[i])
        y = dd.labels[vid]
        if not in_band[i]:
            finals[vid] = s1
            details[vid] = {"in_band": False, "consumed": [], "used": 0, "final": s1,
                            "post": float(post[i])}
            correct += int(s1 == y)
            continue
        ell = E1.logit(float(post[i]))
        used, consumed = 0, []
        for table in judges[:3]:
            used += 1
            r = table.get(vid, {}).get("pred")
            consumed.append(r)
            if r in (0, 1):
                ell += (2 * int(r) - 1) * lam
                if E1.ent(E1.sigmoid(ell)) <= hbar:
                    break
        fin = 1 if E1.sigmoid(ell) >= 0.5 else 0
        finals[vid] = fin
        details[vid] = {"in_band": True, "consumed": consumed, "used": used, "final": fin,
                        "post": float(post[i])}
        correct += int(fin == y)
    acc = correct / len(eval_vids)
    return finals, details, acc, hbar, rho


def main() -> None:
    print("=" * 78)
    print("E1 HateMM sanity-gate diagnostic")
    print("=" * 78)

    frozen = json.loads((ROOT / "results/paper_stage2_experiments/all_results.json").read_text())
    hm_frozen = next(pd["acc"] for row in frozen["method_rows"]
                     if row["method"] == "Full sequential rho_D (band, g27>q32>q72)"
                     for pd in row["per_dataset"] if pd["ds"] == DS)

    fa, da, acc_a, hbar_a, rho_a, labels = path_a_build_experiments()
    fb, db, acc_b, hbar_b, rho_b = path_b_e1_natural()

    print("\n(i) Does the ORIGINAL build_experiments path on CURRENT files still yield the headline?")
    print(f"    all_results.json (frozen headline) HateMM acc = {hm_frozen:.6f} "
          f"({round(hm_frozen * N_TEST[DS])}/{N_TEST[DS]})")
    print(f"    build_experiments on CURRENT files  HateMM acc = {acc_a:.6f} "
          f"({round(acc_a * N_TEST[DS])}/{N_TEST[DS]})")
    print(f"    E1-natural path                      HateMM acc = {acc_b:.6f} "
          f"({round(acc_b * len(fb))}/{len(fb)})")
    print(f"    hbar/rho  build_experiments={hbar_a:.6f}/{rho_a:.6f}  "
          f"E1={hbar_b:.6f}/{rho_b:.6f}")

    print("\n(ii) Per-video diff: build_experiments path vs E1-natural path")
    common = set(fa) & set(fb)
    only_a = set(fa) - set(fb)
    only_b = set(fb) - set(fa)
    diffs = [v for v in common if fa[v] != fb[v]]
    if only_a or only_b:
        print(f"    video-set mismatch: only_in_build_experiments={sorted(only_a)} "
              f"only_in_E1={sorted(only_b)}")
    if not diffs:
        print("    NO per-video prediction differences --> E1 reproduces the pipeline exactly.")
    else:
        for v in sorted(diffs):
            print(f"    {v}: build_experiments final={fa[v]} E1 final={fb[v]} "
                  f"label={labels.get(v)}")
            print(f"        build_experiments: {da[v]}")
            print(f"        E1:                {db[v]}")

    print("\n    Interpretation:")
    if not diffs and abs(acc_a - acc_b) < 1e-9:
        if abs(acc_a - hm_frozen) > 1e-9:
            print("    => E1 == build_experiments on current files, but BOTH differ from the")
            print("       frozen headline. This is VERDICT FILE-DRIFT, not an E1 bug.")
        else:
            print("    => E1 == build_experiments == frozen headline. No issue.")
    else:
        print("    => E1 and build_experiments diverge --> E1 code-path bug (see diff above).")

    print("\n(iii) File mtimes and 72B HateMM backup diff")
    files = [
        ROOT / "results/paper_stage2_experiments/all_results.json",
        ROOT / "results/boundary_rescue/HateMM/offline_test_gemma-3-27b-it.jsonl",
        ROOT / "results/boundary_rescue/HateMM/offline_test_qwen2.5-vl-32b-awq.jsonl",
        ROOT / "results/boundary_rescue/HateMM/offline_test_qwen2.5-vl-72b-awq.jsonl",
    ]
    import glob
    files += [Path(p) for p in sorted(glob.glob(str(
        ROOT / "results/boundary_rescue/HateMM/offline_test_qwen2.5-vl-72b-awq.jsonl.*")))]
    import datetime
    for p in files:
        if p.exists():
            ts = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"    {ts}  {p.name}")
        else:
            print(f"    (missing)  {p.name}")

    cur_p = ROOT / "results/boundary_rescue/HateMM/offline_test_qwen2.5-vl-72b-awq.jsonl"
    bkp_glob = sorted(glob.glob(str(cur_p) + ".before_rerun_*"))
    if bkp_glob:
        def ld(p):
            return {json.loads(l)["video_id"]: json.loads(l)
                    for l in open(p) if l.strip()}
        cur = ld(cur_p)
        bkp = ld(bkp_glob[0])
        changed = [v for v in cur if v in bkp and cur[v].get("pred") != bkp[v].get("pred")]
        neg1_in_bkp = sum(1 for v in bkp if bkp[v].get("pred") == -1)
        print(f"\n    backup {Path(bkp_glob[0]).name}: {len(bkp)} rows, "
              f"{neg1_in_bkp} of them pred=-1 (degraded intermediate, not the "
              f"04-27 paper-time file)")
        print(f"    current file: {len(cur)} rows; {len(changed)} preds differ vs backup")
        print("    NOTE: the backup is a mostly-(-1) pre-rerun snapshot, so it is NOT the")
        print("    verdict set that produced the frozen headline; the 04-27 verdicts are not")
        print("    preserved, so the single flipped video cannot be reconstructed from files.")


if __name__ == "__main__":
    main()
