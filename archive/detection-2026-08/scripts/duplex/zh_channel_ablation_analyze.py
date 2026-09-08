"""Aggregate the MHClip-ZH channel-ablation arms against the frozen baseline."""

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "ranking_autopsy", "zh")
ARMS = os.path.join(OUT, "arms")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    r = np.empty(len(allv), float)
    r[order] = np.arange(len(allv), dtype=float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def oracle(pos, neg):
    best = None
    for t in sorted(set(pos + neg) | {min(pos + neg) - 1}):
        tp = sum(1 for z in pos if z >= t)
        fp = sum(1 for z in neg if z >= t)
        fn, tn = len(pos) - tp, len(neg) - fp
        f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
        m = (f1p + f1n) / 2
        if best is None or m > best[0]:
            best = (m, t, tp, fp, fn, tn)
    return {"macro_f1": round(best[0], 4), "threshold": best[1],
            "tp": best[2], "fp": best[3], "fn": best[4], "tn": best[5]}


def paired_delta_ci(pa, na, pb, nb, n_boot=2000, seed=20260809):
    """Bootstrap the AUC difference with videos resampled jointly across arms."""
    rng = np.random.default_rng(seed)
    pa, na, pb, nb = map(np.asarray, (pa, na, pb, nb))
    d = []
    for _ in range(n_boot):
        ip = rng.integers(0, len(pa), len(pa))
        iN = rng.integers(0, len(na), len(na))
        d.append(auc(pa[ip], na[iN]) - auc(pb[ip], nb[iN]))
    return [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]


def read(path):
    return {json.loads(l)["video_id"]: json.loads(l)["z"]
            for l in open(path) if l.strip()}


def main():
    items = json.load(open(os.path.join(OUT, "items.json")))
    lab = {i["video_id"]: i["binary"] for i in items}
    kw = {i["video_id"]: bool(i["title"] and "keyword" in i["title"]) for i in items}
    ids = [i["video_id"] for i in items]

    arms = {
        "BASELINE": os.path.join(ROOT, "results/testruns/mhclip_zh/judge_8b/scores.jsonl"),
        "NOTITLE": os.path.join(ARMS, "notitle", "scores.jsonl"),
        "NOTRANSCRIPT": os.path.join(ARMS, "notranscript", "scores.jsonl"),
        "NOMARKUP": os.path.join(ARMS, "nomarkup", "scores.jsonl"),
        "FRAMESONLY": os.path.join(ARMS, "framesonly", "scores.jsonl"),
    }
    z = {k: read(v) for k, v in arms.items()}
    base_p = [z["BASELINE"][v] for v in ids if lab[v] == 1]
    base_n = [z["BASELINE"][v] for v in ids if lab[v] == 0]

    res = {"note": "diagnostic ablation of the two text fields in the frozen judge "
                   "prompt; one call per video per arm; frames unchanged",
           "arms": []}
    for name in arms:
        p = [z[name][v] for v in ids if lab[v] == 1]
        n = [z[name][v] for v in ids if lab[v] == 0]
        row = {"arm": name, "n": len(p) + len(n), "auc": round(auc(p, n), 4),
               "oracle": oracle(p, n),
               "median_z_pos": float(np.median(p)), "median_z_neg": float(np.median(n)),
               "delta_auc_vs_baseline": round(auc(p, n) - auc(base_p, base_n), 4)}
        if name != "BASELINE":
            row["delta_auc_paired_ci95"] = paired_delta_ci(p, n, base_p, base_n)
            d = [z[name][v] - z["BASELINE"][v] for v in ids]
            row["median_abs_z_shift"] = float(np.median(np.abs(d)))
            row["mean_z_shift"] = round(float(np.mean(d)), 3)
        # split by whether the shipped title carries the harvester's query term
        for tag, sel in (("keyword_titles", [v for v in ids if kw[v]]),
                         ("plain_titles", [v for v in ids if not kw[v]])):
            pp = [z[name][v] for v in sel if lab[v] == 1]
            nn = [z[name][v] for v in sel if lab[v] == 0]
            row[f"auc_{tag}"] = round(auc(pp, nn), 4)
        res["arms"].append(row)

    # what the negative class does when the title goes away
    hi = [v for v in ids if lab[v] == 0 and z["BASELINE"][v] >= 5.0]
    res["top_scoring_normals_under_notitle"] = {
        "n": len(hi),
        "n_with_keyword_title": sum(1 for v in hi if kw[v]),
        "median_z_baseline": float(np.median([z["BASELINE"][v] for v in hi])),
        "median_z_notitle": float(np.median([z["NOTITLE"][v] for v in hi])),
        "n_falling_below_5": sum(1 for v in hi if z["NOTITLE"][v] < 5.0)}
    lo = [v for v in ids if lab[v] == 1 and z["BASELINE"][v] < 5.0]
    res["low_scoring_positives_under_notitle"] = {
        "n": len(lo),
        "n_with_keyword_title": sum(1 for v in lo if kw[v]),
        "median_z_baseline": float(np.median([z["BASELINE"][v] for v in lo])),
        "median_z_notitle": float(np.median([z["NOTITLE"][v] for v in lo]))}

    with open(os.path.join(OUT, "channel_ablation.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
