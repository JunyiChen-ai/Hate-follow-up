"""Analysis for the speech-act second call kill test.

Pre-registration: docs/duplex/PREREG_speechact_second_call.md (frozen at commit
8ffe65a). The four decision clauses, the sealed strata, the bootstrap size and
seed, and the descriptive readouts are all fixed there; this script only
evaluates them.

Strata (labels enter here and nowhere else):
  ZH      MENTIONED = MHClip-ZH test items whose harvest title carries the
          keyword markup and whose fine label is Normal (31); ASSERTED = same
          markup, fine label Hateful or Offensive (32).
  HateMM  MENTIONED = the 11 blind code-Q valley false positives of the HateMM
          FP audit; ASSERTED = Hate-labelled test videos (video_id prefix
          hate_video_) with c2 judge z > -2.3455 (83).

Clauses:
  1  AUROC of (-z_sa), mentioned vs asserted, on ZH      >= 0.75
  2  same statistic on HateMM >= 0.70 with bootstrap 95% lower bound > 0.55
  3  residualised (OLS of z_sa on z inside the stratum pool) AUROC >= 0.65 on
     both strata
  4  |Spearman(z_sa, z)| <= 0.80 inside each stratum pool

Output: results/speechact_call/report.json plus a printed summary.
"""

import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

SA_DIR = os.path.join(ROOT, "results", "speechact_call")
SA_ZH = os.path.join(SA_DIR, "scores_mhclip_zh.jsonl")
SA_HM = os.path.join(SA_DIR, "scores_hatemm.jsonl")
SA_EN = os.path.join(SA_DIR, "scores_en3.jsonl")
OUT = os.path.join(SA_DIR, "report.json")

ZH_ITEMS = os.path.join(ROOT, "results", "ranking_autopsy", "zh", "items.json")
HM_CODING = os.path.join(ROOT, "results", "hatemm_fp_audit", "coding.tsv")
HM_ALIAS = os.path.join(ROOT, "results", "hatemm_fp_audit", "alias_map.json")
HM_SCORES = os.path.join(ROOT, "results", "testruns", "hatemm", "judge_8b",
                         "scores.jsonl")
EN_PACKET = os.path.join(ROOT, "results", "ranking_autopsy", "en", "packet.json")

HATEMM_VALLEY_Z = -2.3455
KEYWORD_MARKUP = '<em class="keyword">'

BOOT_N = 10000
BOOT_SEED = 20260810

CLAUSE1_FLOOR = 0.75
CLAUSE2_FLOOR = 0.70
CLAUSE2_LCB = 0.55
CLAUSE3_FLOOR = 0.65
CLAUSE4_CEIL = 0.80

SATURATION_ABS_Z = 13.0


# ------------------------------------------------------------------ helpers --
def auc(scores, labels):
    """Mann-Whitney ROC-AUC with midranks for ties (illocution-audit helper)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
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
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def resid(sv, zv):
    """OLS of sv on zv with intercept; returns residuals and coefficients."""
    amat = np.column_stack([np.ones_like(zv), zv])
    coef, *_ = np.linalg.lstsq(amat, sv, rcond=None)
    return sv - amat @ coef, coef


def load_jsonl(path, key):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["video_id"]] = float(r[key])
    return out


def binary_entropy_bits(z):
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def dist(vals):
    v = np.asarray(vals, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)), "sd": float(v.std(ddof=1))
            if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def saturation(vals):
    v = np.asarray(vals, dtype=float)
    h = binary_entropy_bits(v)
    return {"n": int(v.size),
            "frac_abs_z_gt_13": float(np.mean(np.abs(v) > SATURATION_ABS_Z)),
            "mean_answer_entropy_bits": float(h.mean()),
            "median_answer_entropy_bits": float(np.median(h)),
            "frac_entropy_below_0.01_bits": float(np.mean(h < 0.01)),
            "n_distinct_values": int(np.unique(v).size)}


def stratum_eval(name, ids_ment, ids_asrt, sa_map, z_map, rng):
    ev = list(ids_ment) + list(ids_asrt)
    yy = np.array([1] * len(ids_ment) + [0] * len(ids_asrt))
    sa = np.array([sa_map[v] for v in ev])
    zz = np.array([z_map[v] for v in ev])
    # Predicted direction: mentioned should score LOW on z_sa, so -z_sa ranks
    # mentioned above asserted.
    s = -sa

    a_raw = auc(s, yy)
    r, coef = resid(s, zz)
    a_res = auc(r, yy)
    rho = float(spearmanr(sa, zz)[0])

    i1 = np.where(yy == 1)[0]
    i0 = np.where(yy == 0)[0]
    boots_raw, boots_res = [], []
    for _ in range(BOOT_N):
        b1 = rng.choice(i1, size=len(i1), replace=True)
        b0 = rng.choice(i0, size=len(i0), replace=True)
        bi = np.concatenate([b1, b0])
        vr = auc(s[bi], yy[bi])
        if vr is not None:
            boots_raw.append(vr)
        rb, _ = resid(s[bi], zz[bi])
        vs = auc(rb, yy[bi])
        if vs is not None:
            boots_res.append(vs)
    boots_raw = np.asarray(boots_raw)
    boots_res = np.asarray(boots_res)
    lo_raw, hi_raw = np.percentile(boots_raw, [2.5, 97.5])
    lo_res, hi_res = np.percentile(boots_res, [2.5, 97.5])

    return {
        "name": name,
        "n_mentioned": len(ids_ment), "n_asserted": len(ids_asrt),
        "auroc_neg_zsa": a_raw,
        "auroc_neg_zsa_boot_ci95": [float(lo_raw), float(hi_raw)],
        "auroc_neg_zsa_boot_mean": float(boots_raw.mean()),
        "auroc_residualised_on_z": a_res,
        "auroc_residualised_boot_ci95": [float(lo_res), float(hi_res)],
        "auroc_residualised_boot_mean": float(boots_res.mean()),
        "boot_n": int(len(boots_raw)), "boot_seed": BOOT_SEED,
        "ols_intercept": float(coef[0]), "ols_slope_on_z": float(coef[1]),
        "spearman_zsa_vs_z_in_pool": rho,
        "abs_spearman_zsa_vs_z_in_pool": abs(rho),
        "zsa_mentioned": dist(sa[yy == 1]),
        "zsa_asserted": dist(sa[yy == 0]),
        "z_mentioned": dist(zz[yy == 1]),
        "z_asserted": dist(zz[yy == 0]),
        "saturation_in_pool": saturation(sa),
    }


def main():
    report = {
        "prereg": "docs/duplex/PREREG_speechact_second_call.md",
        "inputs": {"zh_scores": SA_ZH, "hatemm_scores": SA_HM,
                   "en_scores": SA_EN},
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
    }

    sa_zh = load_jsonl(SA_ZH, "z_sa")
    sa_hm = load_jsonl(SA_HM, "z_sa")
    sa_en = load_jsonl(SA_EN, "z_sa")
    if len(sa_zh) != 149:
        raise SystemExit(f"ABORT: ZH z_sa rows {len(sa_zh)} != 149")
    if len(sa_hm) != 215:
        raise SystemExit(f"ABORT: HateMM z_sa rows {len(sa_hm)} != 215")
    if len(sa_en) != 3:
        raise SystemExit(f"ABORT: EN z_sa rows {len(sa_en)} != 3")

    # ---- corpus-level z from the frozen c2 hate judge ----------------------
    zh_items = json.load(open(ZH_ITEMS, encoding="utf-8"))
    z_zh = {x["video_id"]: float(x["z"]) for x in zh_items}
    hm_rows = [json.loads(l) for l in open(HM_SCORES) if l.strip()]
    z_hm = {r["video_id"]: float(r["z"]) for r in hm_rows}

    rng = np.random.default_rng(BOOT_SEED)

    # ---- ZH stratum --------------------------------------------------------
    kw = [x for x in zh_items if KEYWORD_MARKUP in (x.get("title") or "")]
    zh_ment = [x["video_id"] for x in kw if x["fine"] == "Normal"]
    zh_asrt = [x["video_id"] for x in kw
               if x["fine"] in ("Hateful", "Offensive")]
    if (len(zh_ment), len(zh_asrt)) != (31, 32):
        raise SystemExit(f"ABORT: ZH strata {len(zh_ment)}/{len(zh_asrt)}, "
                         "expected 31/32")
    zh = stratum_eval("zh", zh_ment, zh_asrt, sa_zh, z_zh, rng)
    zh["definition"] = ("MHClip-ZH test videos whose harvest title carries the "
                        "keyword markup; MENTIONED = Normal, ASSERTED = "
                        "Hateful or Offensive")
    report["stratum_zh"] = zh

    # ---- HateMM stratum ----------------------------------------------------
    alias = json.load(open(HM_ALIAS))
    qcodes = []
    with open(HM_CODING) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["primary"].strip() == "Q":
                qcodes.append(row["alias"].strip())
    hm_ment = [alias[a] for a in qcodes]
    hm_asrt = [r["video_id"] for r in hm_rows
               if r["video_id"].startswith("hate_video_")
               and float(r["z"]) > HATEMM_VALLEY_Z]
    if len(hm_ment) != 11:
        raise SystemExit(f"ABORT: HateMM code-Q count {len(hm_ment)}, expected 11")
    if len(hm_asrt) != 83:
        raise SystemExit(f"ABORT: HateMM asserted count {len(hm_asrt)}, expected 83")
    if set(hm_ment) & set(hm_asrt):
        raise SystemExit("ABORT: HateMM strata overlap")
    hm = stratum_eval("hatemm", hm_ment, hm_asrt, sa_hm, z_hm, rng)
    hm["definition"] = ("MENTIONED = blind code-Q valley false positives; "
                        "ASSERTED = HateMM test Hate-labelled videos with "
                        f"z > {HATEMM_VALLEY_Z}")
    hm["code_q_aliases"] = qcodes
    report["stratum_hatemm"] = hm

    # ---- decision clauses --------------------------------------------------
    c1 = bool(zh["auroc_neg_zsa"] >= CLAUSE1_FLOOR)
    c2 = bool(hm["auroc_neg_zsa"] >= CLAUSE2_FLOOR
              and hm["auroc_neg_zsa_boot_ci95"][0] > CLAUSE2_LCB)
    c3 = bool(zh["auroc_residualised_on_z"] >= CLAUSE3_FLOOR
              and hm["auroc_residualised_on_z"] >= CLAUSE3_FLOOR)
    c4 = bool(zh["abs_spearman_zsa_vs_z_in_pool"] <= CLAUSE4_CEIL
              and hm["abs_spearman_zsa_vs_z_in_pool"] <= CLAUSE4_CEIL)
    report["clause1_zh_separation"] = {
        "auroc": zh["auroc_neg_zsa"], "floor": CLAUSE1_FLOOR, "passes": c1}
    report["clause2_hatemm_separation"] = {
        "auroc": hm["auroc_neg_zsa"], "floor": CLAUSE2_FLOOR,
        "boot_ci95": hm["auroc_neg_zsa_boot_ci95"], "lcb_floor": CLAUSE2_LCB,
        "passes": c2}
    report["clause3_incremental_over_z"] = {
        "auroc_residualised_zh": zh["auroc_residualised_on_z"],
        "auroc_residualised_hatemm": hm["auroc_residualised_on_z"],
        "floor": CLAUSE3_FLOOR, "passes": c3}
    report["clause4_not_z_renamed"] = {
        "abs_spearman_zh_pool": zh["abs_spearman_zsa_vs_z_in_pool"],
        "abs_spearman_hatemm_pool": hm["abs_spearman_zsa_vs_z_in_pool"],
        "ceiling": CLAUSE4_CEIL, "passes": c4}
    report["survives"] = bool(c1 and c2 and c3 and c4)

    # ---- descriptive: full-corpus correlation and distributions ------------
    def corpus_block(sa_map, z_map, label_of=None):
        ids = [v for v in sa_map if v in z_map]
        sa = np.array([sa_map[v] for v in ids])
        zz = np.array([z_map[v] for v in ids])
        rho, p = spearmanr(sa, zz)
        blk = {"n": len(ids), "spearman_zsa_vs_z": float(rho),
               "spearman_p": float(p),
               "pearson_zsa_vs_z": float(np.corrcoef(sa, zz)[0, 1]),
               "zsa": dist(sa), "z": dist(zz),
               "saturation_zsa": saturation(sa),
               "saturation_z": saturation(zz)}
        if label_of is not None:
            per = {}
            for v, s in zip(ids, sa):
                per.setdefault(label_of(v), []).append(s)
            blk["zsa_by_class"] = {k: dist(vv) for k, vv in sorted(per.items())}
        return blk

    zh_fine = {x["video_id"]: x["fine"] for x in zh_items}
    report["corpus_zh"] = corpus_block(sa_zh, z_zh, lambda v: zh_fine.get(v, "?"))
    report["corpus_hatemm"] = corpus_block(
        sa_hm, z_hm,
        lambda v: "hate" if v.startswith("hate_video_") else "non_hate")

    # ---- descriptive: the three EN incidental cases ------------------------
    packet = json.load(open(EN_PACKET, encoding="utf-8"))
    amap = {x["alias"]: x for x in packet}
    en_cases = {}
    for a in ("EN-FN-02", "EN-FP-07", "EN-FP-16"):
        rec = amap.get(a)
        if rec is None:
            en_cases[a] = {"error": "alias not in packet.json"}
            continue
        v = rec["video_id"]
        en_cases[a] = {"video_id": v, "z": rec.get("z"), "fine": rec.get("fine"),
                       "z_sa": sa_en.get(v)}
    report["en_incidental_cases"] = en_cases

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- print --
    def line(s=""):
        print(s)

    line("=== Speech-act second call ===")
    line(f"clause1 {'PASS' if c1 else 'FAIL'}  ZH AUROC(-z_sa)="
         f"{zh['auroc_neg_zsa']:.4f}  floor {CLAUSE1_FLOOR}  "
         f"CI95={[round(x, 4) for x in zh['auroc_neg_zsa_boot_ci95']]}")
    line(f"clause2 {'PASS' if c2 else 'FAIL'}  HateMM AUROC(-z_sa)="
         f"{hm['auroc_neg_zsa']:.4f}  floor {CLAUSE2_FLOOR}  "
         f"CI95={[round(x, 4) for x in hm['auroc_neg_zsa_boot_ci95']]}  "
         f"LCB floor {CLAUSE2_LCB}")
    line(f"clause3 {'PASS' if c3 else 'FAIL'}  residual AUROC ZH="
         f"{zh['auroc_residualised_on_z']:.4f}  HateMM="
         f"{hm['auroc_residualised_on_z']:.4f}  floor {CLAUSE3_FLOOR}")
    line(f"clause4 {'PASS' if c4 else 'FAIL'}  |rho(z_sa,z)| ZH pool="
         f"{zh['abs_spearman_zsa_vs_z_in_pool']:.4f}  HateMM pool="
         f"{hm['abs_spearman_zsa_vs_z_in_pool']:.4f}  ceiling {CLAUSE4_CEIL}")
    line(f"SURVIVES: {report['survives']}")
    line()
    for key in ("stratum_zh", "stratum_hatemm"):
        b = report[key]
        line(f"[{b['name']}] mentioned n={b['n_mentioned']} "
             f"z_sa mean={b['zsa_mentioned']['mean']:+.3f} "
             f"median={b['zsa_mentioned']['median']:+.3f} "
             f"sd={b['zsa_mentioned']['sd']:.3f}")
        line(f"[{b['name']}] asserted  n={b['n_asserted']} "
             f"z_sa mean={b['zsa_asserted']['mean']:+.3f} "
             f"median={b['zsa_asserted']['median']:+.3f} "
             f"sd={b['zsa_asserted']['sd']:.3f}")
        line(f"[{b['name']}] pool saturation |z_sa|>13: "
             f"{b['saturation_in_pool']['frac_abs_z_gt_13']:.3f}  "
             f"mean entropy={b['saturation_in_pool']['mean_answer_entropy_bits']:.4f} bits")
    line()
    for key, nm in (("corpus_zh", "MHClip-ZH"), ("corpus_hatemm", "HateMM")):
        b = report[key]
        line(f"[{nm} full corpus n={b['n']}] spearman(z_sa,z)="
             f"{b['spearman_zsa_vs_z']:+.4f}  pearson="
             f"{b['pearson_zsa_vs_z']:+.4f}")
        line(f"    z_sa mean={b['zsa']['mean']:+.3f} sd={b['zsa']['sd']:.3f}  "
             f"|z_sa|>13 frac={b['saturation_zsa']['frac_abs_z_gt_13']:.3f}  "
             f"mean entropy={b['saturation_zsa']['mean_answer_entropy_bits']:.4f} bits  "
             f"distinct={b['saturation_zsa']['n_distinct_values']}")
        for k, d in b["zsa_by_class"].items():
            line(f"    class {k:<10} n={d['n']:<4} z_sa mean={d['mean']:+.3f} "
                 f"median={d['median']:+.3f} sd={d['sd']:.3f}")
    line()
    line("EN incidental cases:")
    for a, c in report["en_incidental_cases"].items():
        line(f"    {a} {c.get('video_id')} fine={c.get('fine')} "
             f"z={c.get('z')} z_sa={c.get('z_sa')}")
    line()
    line(f"report -> {OUT}")


if __name__ == "__main__":
    main()
