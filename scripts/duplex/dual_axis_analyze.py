"""Analysis for the dual-axis kill test.

Pre-registration: docs/duplex/PREREG_dual_axis_killtest.md.

Frozen decision rule, both clauses required for SURVIVES:
  1. Spearman(z_off, z_hate) over the 161 MHClip-EN test videos < 0.95.
  2. AUC(z_off) >= AUC(z_hate) + 0.05 on the blind-coded
     no-protected-target-positive versus shipped-Normal stratum.

Reported outside the verdict: the max-rank composition AUC on the full union
task against the joint baseline 0.785, and AUC(z_off) on the protected-target
stratum against the joint judge's 0.983.

Writes results/dual_axis/results.json.
"""

import glob
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/jehc223/Hate-follow-up"
AV = os.path.join(ROOT, "results", "annotation_validity")
Z_HATE = os.path.join(ROOT, "results", "testruns", "mhclip_en", "judge_8b",
                      "scores.jsonl")
Z_OFF = os.path.join(ROOT, "results", "dual_axis", "z_off_scores.jsonl")
ANN = "/home/jehc223/data/Multihateclip/English/annotation(new).json"

SPEARMAN_CEILING = 0.95
AUC_GAIN_FLOOR = 0.05
JOINT_UNION_AUC = 0.785
JOINT_PROTECTED_AUC = 0.983


def load_z(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r.get("z"), (int, float)) and math.isfinite(r["z"]):
                d[r["video_id"]] = float(r["z"])
    return d


def auc(scores, labels):
    """Mann-Whitney ROC-AUC with midranks for ties (same routine as the audit)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def midranks(s):
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def main():
    z_hate = load_z(Z_HATE)
    z_off = load_z(Z_OFF)
    ids = sorted(set(z_hate) & set(z_off))
    if len(ids) != 161 or len(z_hate) != 161 or len(z_off) != 161:
        raise SystemExit(f"ABORT: n_hate={len(z_hate)} n_off={len(z_off)} "
                         f"n_common={len(ids)}; expected 161 everywhere")

    # ---- strata, reconstructed from the committed blind-coding artifacts ----
    manifest = json.load(open(os.path.join(AV, "manifest.json")))
    codes = {}
    for p in sorted(glob.glob(os.path.join(AV, "coding", "*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                codes[r["item_id"]] = r
    target = [m for m in manifest
              if m["corpus"] == "EN" and m["stratum"] == "target_union_positive"]
    no_pt = [m["video_id"] for m in target
             if codes[m["item_id"]].get("protected_group_targeted") is False]
    pt = [m["video_id"] for m in target
          if codes[m["item_id"]].get("protected_group_targeted") is True]

    shipped = {x["Video_ID"]: x["Label"] for x in json.load(open(ANN))}
    normals = [v for v in ids if shipped.get(v) == "Normal"]
    union_pos = {v for v in ids if shipped.get(v) in ("Hateful", "Offensive")}

    if (len(no_pt), len(pt), len(normals)) != (34, 15, 112):
        raise SystemExit(f"ABORT: strata are {len(no_pt)}/{len(pt)}/{len(normals)}, "
                         "expected 34/15/112")
    for v in no_pt + pt:
        if v not in z_off or v not in z_hate:
            raise SystemExit(f"ABORT: {v} missing a score")

    # ---- clause 1: separability -------------------------------------------
    hv = [z_hate[v] for v in ids]
    ov = [z_off[v] for v in ids]
    rho, p_rho = spearmanr(ov, hv)
    pearson = float(np.corrcoef(ov, hv)[0, 1])
    clause1 = bool(rho < SPEARMAN_CEILING)

    # ---- clause 2: axis validity ------------------------------------------
    def stratum_auc(pos_ids):
        vids = list(pos_ids) + normals
        y = [1] * len(pos_ids) + [0] * len(normals)
        return {
            "n": len(vids), "n_pos": len(pos_ids), "n_neg": len(normals),
            "auc_z_off": auc([z_off[v] for v in vids], y),
            "auc_z_hate": auc([z_hate[v] for v in vids], y),
        }

    no_pt_stratum = stratum_auc(no_pt)
    no_pt_stratum["gain_off_minus_hate"] = (no_pt_stratum["auc_z_off"]
                                            - no_pt_stratum["auc_z_hate"])
    pt_stratum = stratum_auc(pt)
    pt_stratum["gain_off_minus_hate"] = (pt_stratum["auc_z_off"]
                                         - pt_stratum["auc_z_hate"])
    clause2 = bool(no_pt_stratum["gain_off_minus_hate"] >= AUC_GAIN_FLOOR)

    verdict = "SURVIVES" if (clause1 and clause2) else "DEAD"

    # ---- descriptive: composition on the full union task -------------------
    r_off = midranks(ov)
    r_hate = midranks(hv)
    comp = np.maximum(r_off, r_hate)
    y_union = [1 if v in union_pos else 0 for v in ids]
    union = {
        "n": len(ids),
        "n_pos": int(sum(y_union)),
        "auc_z_hate": auc(hv, y_union),
        "auc_z_off": auc(ov, y_union),
        "auc_max_rank_composition": auc(comp.tolist(), y_union),
        "joint_baseline_reference": JOINT_UNION_AUC,
    }
    union["composition_minus_baseline"] = (union["auc_max_rank_composition"]
                                           - JOINT_UNION_AUC)

    # ---- descriptive: agreement of the two axes on hard calls --------------
    off_sign = np.sign(ov)
    hate_sign = np.sign(hv)
    res = {
        "protocol": "docs/duplex/PREREG_dual_axis_killtest.md",
        "corpus": "MHClip_EN test_clean",
        "n_videos": len(ids),
        "clause1_separability": {
            "rule": f"Spearman(z_off, z_hate) < {SPEARMAN_CEILING}",
            "spearman": float(rho),
            "spearman_p": float(p_rho),
            "pearson": pearson,
            "passes": clause1,
        },
        "clause2_axis_validity": {
            "rule": ("AUC(z_off) >= AUC(z_hate) + "
                     f"{AUC_GAIN_FLOOR} on no-protected-target positives "
                     "vs shipped Normals"),
            **no_pt_stratum,
            "passes": clause2,
        },
        "descriptive_protected_target_stratum": {
            "note": ("expected LOWER for z_off than z_hate; a match means the "
                     "axis is not construct-specific"),
            **pt_stratum,
            "joint_reference_quoted_in_prereg": JOINT_PROTECTED_AUC,
            "joint_reference_caveat": (
                "the 0.983 quoted in the prereg body is the ranking autopsy's "
                "narrower 7-video explicit-hostility subset, not this "
                "blind-coded 15-video protected-target stratum; the "
                "like-for-like joint-judge number on this stratum is "
                "auc_z_hate below, which matches the audit's 0.866"),
        },
        "descriptive_full_union_task": union,
        "descriptive_sign_agreement": {
            "frac_same_sign": float(np.mean(off_sign == hate_sign)),
            "n_off_positive": int(np.sum(np.asarray(ov) > 0)),
            "n_hate_positive": int(np.sum(np.asarray(hv) > 0)),
        },
        "verdict": verdict,
    }
    out = os.path.join(ROOT, "results", "dual_axis", "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
