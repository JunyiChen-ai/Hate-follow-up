"""Interleaved-timeline kill test, stage E (CPU): AUCs, strata, flips, verdict.

Reads the frozen baseline scores and the two new arms, computes full-corpus and
per-stratum ROC-AUC for every arm on every corpus, counts decision flips, and
evaluates the three pre-registered clauses. Labels enter here and nowhere else.

Pre-registration: docs/duplex/PREREG_interleaved_timeline_killtest.md.

Output: results/interleaved_timeline/results.json.

Usage:
  python scripts/duplex/interleaved_timeline_analyze.py
"""

import argparse
import ast
import csv
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from data_utils import load_annotations, load_clean_split_ids  # noqa: E402
from interleaved_timeline_asr import CORPORA  # noqa: E402
from interleaved_timeline_score import BASELINE_JUDGE_DIRS  # noqa: E402

LABEL_MAP = {
    "MHClip_ZH": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    "MHClip_EN": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    "ImpliHateVid": {"Normal": 0, "Hateful": 1},
    "HateClipSeg": {"Normal": 0, "Offensive": 1},
}
ARMS = ["baseline", "interleaved", "misaligned"]

HCS_ANN = os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit",
                       "data", "video_level_annotation.csv")
HCS_IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
HCS_OFFENSIVE = set(HCS_IDX[1:])
ZH_CODING = os.path.join(ROOT, "results", "ranking_autopsy", "zh", "coding_zh.tsv")
ZH_PACKET = os.path.join(ROOT, "results", "ranking_autopsy", "zh", "packet.json")
ZH_BINDING_CODES = {"G", "F", "I"}

CLAUSE1_GAIN = 0.04
CLAUSE1_TOLERANCE = 0.01
CLAUSE2_BOUND = 0.01
CLAUSE3_FRACTION = 0.5


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    s = np.asarray(list(pos) + list(neg), dtype=float)
    y = np.asarray([1] * len(pos) + [0] * len(neg), dtype=int)
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
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def auc_boot(pos, neg, n_boot=2000, seed=0):
    if not len(pos) or not len(neg):
        return None
    rng = np.random.default_rng(seed)
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    vals = [auc(p[rng.integers(0, len(p), len(p))],
                n[rng.integers(0, len(n), len(n))]) for _ in range(n_boot)]
    return [round(float(np.percentile(vals, 2.5)), 6),
            round(float(np.percentile(vals, 97.5)), 6)]


def delta_boot(pos_a, neg_a, pos_b, neg_b, n_boot=2000, seed=0):
    """Paired bootstrap of AUC(b) - AUC(a); the same videos resampled once."""
    rng = np.random.default_rng(seed)
    pa, na = np.asarray(pos_a, float), np.asarray(neg_a, float)
    pb, nb = np.asarray(pos_b, float), np.asarray(neg_b, float)
    vals = []
    for _ in range(n_boot):
        ip = rng.integers(0, len(pa), len(pa))
        ineg = rng.integers(0, len(na), len(na))
        vals.append(auc(pb[ip], nb[ineg]) - auc(pa[ip], na[ineg]))
    return [round(float(np.percentile(vals, 2.5)), 6),
            round(float(np.percentile(vals, 97.5)), 6)]


def load_z(path):
    z = {}
    if not os.path.exists(path):
        return z
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r.get("z"), (int, float)):
                z[r["video_id"]] = float(r["z"])
    return z


def zh_binding_ids():
    alias2id = {r["alias"]: r["video_id"] for r in json.load(open(ZH_PACKET))}
    with open(ZH_CODING) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [alias2id[r["alias"]] for r in rows
            if r["side"] == "FN" and r["code"] in ZH_BINDING_CODES]


def hcs_labels():
    labels = {}
    with open(HCS_ANN) as f:
        for row in csv.DictReader(f):
            labels[row["Video Id"].strip()] = ast.literal_eval(
                row["Video-Level Label"])
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.path.join(
        ROOT, "results", "interleaved_timeline"))
    args = ap.parse_args()

    res = {
        "protocol": "docs/duplex/PREREG_interleaved_timeline_killtest.md",
        "judge": "Qwen3-VL-8B-Instruct, bf16, single forward pass per video",
        "arms": {
            "baseline": "16 images, then title + whole transcript + rules + "
                        "reader block + question (frozen scores on disk)",
            "interleaved": "frame i, segment i, for i=0..15, then the same "
                           "trailing block with a pointer in the transcript field",
            "misaligned": "identical to interleaved, frame i paired with "
                          "segment (i+8) mod 16",
        },
        "frozen_clauses": {
            "clause1_stratum_gain": CLAUSE1_GAIN,
            "clause1_other_stratum_tolerance": CLAUSE1_TOLERANCE,
            "clause2_control_bound": CLAUSE2_BOUND,
            "clause3_misaligned_fraction": CLAUSE3_FRACTION,
        },
        "corpora": {},
        "strata": {},
        "flips": {},
    }
    census_path = os.path.join(args.out_root, "segmentation_census.json")
    if os.path.exists(census_path):
        res["segmentation_census"] = json.load(open(census_path))

    scores = {}
    labels = {}
    for corpus, (dataset, _work) in sorted(CORPORA.items()):
        ann = load_annotations(dataset)
        lmap = LABEL_MAP[dataset]
        base = load_z(os.path.join(ROOT, BASELINE_JUDGE_DIRS[corpus],
                                   "scores.jsonl"))
        seen, ids = set(), []
        for v in load_clean_split_ids(dataset, "test"):
            if v in seen or v not in base:
                continue
            seen.add(v)
            ids.append(v)
        arm_z = {"baseline": base}
        for arm in ARMS[1:]:
            arm_z[arm] = load_z(os.path.join(args.out_root, corpus, arm,
                                             "scores.jsonl"))
        cov = {a: sum(1 for v in ids if v in arm_z[a]) for a in ARMS}
        y = {v: lmap[ann[v]["label"]] for v in ids}
        scores[corpus] = (ids, arm_z)
        labels[corpus] = y

        entry = {"dataset": dataset, "n_videos": len(ids),
                 "n_positive": sum(y.values()),
                 "n_negative": len(ids) - sum(y.values()),
                 "coverage": cov,
                 "full_corpus_auc": {}, "full_corpus_auc_boot95": {},
                 "delta_vs_baseline": {}, "delta_boot95": {}}
        pos_neg = {}
        for a in ARMS:
            if cov[a] != len(ids):
                entry["full_corpus_auc"][a] = None
                continue
            p = [arm_z[a][v] for v in ids if y[v] == 1]
            n = [arm_z[a][v] for v in ids if y[v] == 0]
            pos_neg[a] = (p, n)
            entry["full_corpus_auc"][a] = round(auc(p, n), 6)
            entry["full_corpus_auc_boot95"][a] = auc_boot(p, n)
        for a in ARMS[1:]:
            if a in pos_neg and "baseline" in pos_neg:
                entry["delta_vs_baseline"][a] = round(
                    entry["full_corpus_auc"][a]
                    - entry["full_corpus_auc"]["baseline"], 6)
                entry["delta_boot95"][a] = delta_boot(
                    *pos_neg["baseline"], *pos_neg[a])
        res["corpora"][corpus] = entry

    # ---------------------------------------------------------------- strata
    ann_zh = load_annotations("MHClip_ZH")
    ids_zh, z_zh = scores["mhclip_zh"]
    gfi = [v for v in zh_binding_ids() if v in ids_zh]
    zh_norm = [v for v in ids_zh if ann_zh[v]["label"] == "Normal"]

    hl = hcs_labels()
    ids_hcs, z_hcs = scores["hateclipseg"]
    ins = [v for v in ids_hcs if hl[v] == ["insulting"]]
    clean = [v for v in ids_hcs if not (set(hl[v]) & HCS_OFFENSIVE)]

    stratum_defs = {
        "mhclip_zh_binding_GFI": ("mhclip_zh", gfi, zh_norm,
                                  "ranking-autopsy false negatives coded G "
                                  "(group stereotype), F (fiction/skit) or I "
                                  "(irony), against the corpus's shipped Normals"),
        "hateclipseg_insulting_only": ("hateclipseg", ins, clean,
                                       "shipped insulting-only videos against "
                                       "clean normals"),
    }
    for name, (corpus, pos_ids, neg_ids, note) in stratum_defs.items():
        _ids, arm_z = scores[corpus]
        entry = {"corpus": corpus, "n_positive": len(pos_ids),
                 "n_negative": len(neg_ids), "definition": note,
                 "auc": {}, "auc_boot95": {}, "delta_vs_baseline": {},
                 "delta_boot95": {}}
        pn = {}
        for a in ARMS:
            if not all(v in arm_z[a] for v in pos_ids + neg_ids):
                entry["auc"][a] = None
                continue
            p = [arm_z[a][v] for v in pos_ids]
            n = [arm_z[a][v] for v in neg_ids]
            pn[a] = (p, n)
            entry["auc"][a] = round(auc(p, n), 6)
            entry["auc_boot95"][a] = auc_boot(p, n)
        for a in ARMS[1:]:
            if a in pn and "baseline" in pn:
                entry["delta_vs_baseline"][a] = round(
                    entry["auc"][a] - entry["auc"]["baseline"], 6)
                entry["delta_boot95"][a] = delta_boot(*pn["baseline"], *pn[a])
        res["strata"][name] = entry

    # ----------------------------------------------------------------- flips
    for corpus, (ids, arm_z) in scores.items():
        y = labels[corpus]
        f = {}
        for a in ARMS[1:]:
            if not all(v in arm_z[a] for v in ids):
                f[a] = None
                continue
            changed = [v for v in ids
                       if (arm_z["baseline"][v] > 0) != (arm_z[a][v] > 0)]
            to_yes = [v for v in changed if arm_z[a][v] > 0]
            to_no = [v for v in changed if arm_z[a][v] <= 0]
            f[a] = {
                "n_changed": len(changed),
                "to_yes": len(to_yes),
                "to_no": len(to_no),
                "gained_correct": sum(1 for v in to_yes if y[v] == 1)
                                  + sum(1 for v in to_no if y[v] == 0),
                "lost_correct": sum(1 for v in to_yes if y[v] == 0)
                                + sum(1 for v in to_no if y[v] == 1),
                "mean_abs_z_shift": round(float(np.mean(
                    [abs(arm_z[a][v] - arm_z["baseline"][v]) for v in ids])), 4),
                "spearman_like_rank_corr": None,
            }
            b = np.array([arm_z["baseline"][v] for v in ids])
            n2 = np.array([arm_z[a][v] for v in ids])
            rb = np.argsort(np.argsort(b))
            rn = np.argsort(np.argsort(n2))
            f[a]["rank_corr_vs_baseline"] = round(float(
                np.corrcoef(rb, rn)[0, 1]), 4)
            del f[a]["spearman_like_rank_corr"]
        # How close the two interleaved arms are to each other. If the pairing
        # were the load-bearing part, correct and rotated pairings would
        # disagree more than either disagrees with the baseline.
        if all(v in arm_z["interleaved"] and v in arm_z["misaligned"]
               for v in ids):
            zi = np.array([arm_z["interleaved"][v] for v in ids])
            zm = np.array([arm_z["misaligned"][v] for v in ids])
            zb = np.array([arm_z["baseline"][v] for v in ids])
            rk = lambda a: np.argsort(np.argsort(a))  # noqa: E731
            f["arm_agreement"] = {
                "rank_corr_interleaved_vs_misaligned": round(float(
                    np.corrcoef(rk(zi), rk(zm))[0, 1]), 4),
                "mean_abs_z_interleaved_minus_misaligned": round(float(
                    np.mean(np.abs(zi - zm))), 4),
                "mean_abs_z_interleaved_minus_baseline": round(float(
                    np.mean(np.abs(zi - zb))), 4),
                "n_decision_disagreements": int(np.sum((zi > 0) != (zm > 0))),
            }
        res["flips"][corpus] = f

    # -------------------------------------------------------------- verdicts
    s_zh = res["strata"]["mhclip_zh_binding_GFI"]
    s_hcs = res["strata"]["hateclipseg_insulting_only"]
    d_zh = s_zh["delta_vs_baseline"].get("interleaved")
    d_hcs = s_hcs["delta_vs_baseline"].get("interleaved")
    verdict = {}
    if d_zh is None or d_hcs is None:
        verdict["clause1"] = {"pass": None, "note": "arms incomplete"}
    else:
        carriers = []
        if d_zh >= CLAUSE1_GAIN and d_hcs >= -CLAUSE1_TOLERANCE:
            carriers.append("mhclip_zh_binding_GFI")
        if d_hcs >= CLAUSE1_GAIN and d_zh >= -CLAUSE1_TOLERANCE:
            carriers.append("hateclipseg_insulting_only")
        verdict["clause1"] = {
            "pass": bool(carriers),
            "carrier": carriers[0] if carriers else None,
            "delta_mhclip_zh_binding_GFI": d_zh,
            "delta_hateclipseg_insulting_only": d_hcs,
            "rule": f"one stratum >= +{CLAUSE1_GAIN}, the other not below "
                    f"-{CLAUSE1_TOLERANCE}",
        }

    ihv = res["corpora"]["implihatevid"]["delta_vs_baseline"].get("interleaved")
    verdict["clause2"] = {
        "pass": (None if ihv is None else bool(abs(ihv) < CLAUSE2_BOUND)),
        "delta_implihatevid_full_corpus": ihv,
        "rule": f"|delta| < {CLAUSE2_BOUND} on the binding-irrelevant corpus",
    }

    carrier = verdict["clause1"].get("carrier")
    if carrier is None:
        # No clause-1 carrier: report the control on the stratum with the
        # larger interleaved gain, so clause 3 is still auditable.
        cands = [(d, k) for k, d in
                 (("mhclip_zh_binding_GFI", d_zh),
                  ("hateclipseg_insulting_only", d_hcs)) if d is not None]
        carrier = max(cands)[1] if cands else None
    if carrier is None:
        verdict["clause3"] = {"pass": None, "note": "arms incomplete"}
    else:
        st = res["strata"][carrier]
        gi = st["delta_vs_baseline"].get("interleaved")
        gm = st["delta_vs_baseline"].get("misaligned")
        verdict["clause3"] = {
            "pass": (None if gi is None or gm is None
                     else bool(gm <= CLAUSE3_FRACTION * gi)),
            "stratum": carrier,
            "interleaved_gain": gi,
            "misaligned_gain": gm,
            "rule": f"misaligned gain <= {CLAUSE3_FRACTION} x interleaved gain",
            "note": ("evaluated on the stratum with the larger interleaved "
                     "gain because no stratum met clause 1"
                     if not verdict["clause1"].get("pass") else None),
        }

    passes = [verdict[c].get("pass") for c in ("clause1", "clause2", "clause3")]
    res["verdict"] = verdict
    res["design_verdict"] = ("SURVIVES" if all(p is True for p in passes)
                             else ("INCOMPLETE" if any(p is None for p in passes)
                                   else "DEAD"))

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "results.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({"design_verdict": res["design_verdict"],
                      "verdict": verdict}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
