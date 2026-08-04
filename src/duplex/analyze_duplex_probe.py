"""
Kill-test analysis for the duplex reading probe.

Reads the per-reader score files produced by score_duplex_probe.py for one
(dataset, split, model) cell and evaluates the pre-registered predictions
in docs/duplex/PREREG_duplex_killtest.md:

  P1 divergence separation   AUC(IM vs EX) and AUC(IM vs NH) using
                             D = z(prag) - z(lit)
  P2 quadrant occupancy      IM enrichment in the (lit-low, prag-high) cell
  P3 incremental value       within the middle tercile of s_prag, AUC of D
                             separating hateful (EX+IM) vs NH
  P4 lit compliance          (a) AUC(EX vs NH) via s_lit alone (detects
                             explicit); (b) suppression: median z_lit - z_prag
                             on IM must be < 0
  P5 controls                placebo: same P1 stats with effort in place of
                             prag; noise floor: P1 stats with D_noise =
                             z(prag) - z(prag_para)

Group labels come from the ImpliHateVid Video_ID prefix (EX/IM/NH); no
model component consumes them. CPU only.
"""

import argparse
import itertools
import json
import math
import os
import sys

PROJECT_ROOT = "/data/jehc223/EMNLP3"

EPS = 1e-4  # clip before logit; probe scores can hit exact 0/1


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def load_scores(dataset, split, reader, slug):
    path = os.path.join(PROJECT_ROOT, "results", "duplex_probe", dataset,
                        f"{split}_{reader}_{slug}.jsonl")
    scores = {}
    if not os.path.exists(path):
        return scores, path
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("score") is not None:
                scores[r["video_id"]] = float(r["score"])
    return scores, path


def auc(pos, neg):
    """Rank-based AUC: P(pos > neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for p, n in itertools.product(pos, neg):
        if p > n:
            wins += 1
        elif p == n:
            ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def cliffs_delta(pos, neg):
    a = auc(pos, neg)
    return None if a is None else 2 * a - 1


def group_of(vid):
    return vid.split("_")[0]  # EX / IM / NH


def d_stats(z_hi, z_lo, groups, label):
    """Divergence D = z_hi - z_lo per video; separation stats across groups."""
    common = set(z_hi) & set(z_lo)
    D = {v: z_hi[v] - z_lo[v] for v in common}
    by_group = {g: [D[v] for v in common if groups.get(v) == g] for g in ("IM", "EX", "NH")}
    out = {
        "pair": label,
        "n": {g: len(by_group[g]) for g in by_group},
        "median_D": {g: median(by_group[g]) for g in by_group},
        "auc_IM_vs_EX": auc(by_group["IM"], by_group["EX"]),
        "auc_IM_vs_NH": auc(by_group["IM"], by_group["NH"]),
        "cliffs_IM_vs_EX": cliffs_delta(by_group["IM"], by_group["EX"]),
        "cliffs_IM_vs_NH": cliffs_delta(by_group["IM"], by_group["NH"]),
    }
    return out, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ImpliHateVid")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-slug", default="qwen3-vl-8b-instruct")
    parser.add_argument("--lit", default="lit_v1", choices=["lit_v1", "lit_v2"],
                        help="Which lit variant to use as the primary axis")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    readers = ["lit_v1", "lit_v2", "prag", "prag_para", "effort"]
    scores = {}
    for r in readers:
        s, path = load_scores(args.dataset, args.split, r, args.model_slug)
        print(f"loaded {r}: {len(s)} scores  ({path})", file=sys.stderr)
        scores[r] = s

    lit = scores[args.lit]
    prag = scores["prag"]
    common = set(lit) & set(prag)
    if not common:
        raise SystemExit("no overlapping videos between lit and prag; run incomplete?")

    groups = {v: group_of(v) for v in common}
    z = {r: {v: logit(s[v]) for v in s} for r, s in scores.items()}

    report = {"dataset": args.dataset, "split": args.split,
              "model": args.model_slug, "lit_variant": args.lit}

    # P1: primary divergence separation
    p1, D_main = d_stats(z["prag"], z[args.lit], groups, f"prag-{args.lit}")
    report["P1_divergence"] = p1

    # P2: quadrant occupancy at per-axis medians over the pool (label-free cutpoints)
    lit_med = median([z[args.lit][v] for v in common])
    prag_med = median([z["prag"][v] for v in common])
    quad = {}
    for g in ("IM", "EX", "NH"):
        vids = [v for v in common if groups[v] == g]
        cells = {"low_low": 0, "low_high": 0, "high_low": 0, "high_high": 0}
        for v in vids:
            lo = "low" if z[args.lit][v] <= lit_med else "high"
            hi = "low" if z["prag"][v] <= prag_med else "high"
            cells[f"{lo}_{hi}"] += 1
        n = max(1, len(vids))
        quad[g] = {k: round(c / n, 3) for k, c in cells.items()}
    report["P2_quadrants"] = {"lit_median": lit_med, "prag_median": prag_med,
                             "occupancy": quad}

    # P3: incremental value of D inside the middle tercile of s_prag
    prag_vals = sorted(scores["prag"][v] for v in common)
    t1 = prag_vals[len(prag_vals) // 3]
    t2 = prag_vals[2 * len(prag_vals) // 3]
    mid = [v for v in common if t1 <= scores["prag"][v] <= t2]
    pos = [D_main[v] for v in mid if groups[v] in ("EX", "IM")]
    neg = [D_main[v] for v in mid if groups[v] == "NH"]
    report["P3_incremental"] = {
        "tercile_bounds": [t1, t2], "n_mid": len(mid),
        "n_hateful": len(pos), "n_normal": len(neg),
        "auc_D_hateful_vs_normal_mid": auc(pos, neg),
        "auc_prag_hateful_vs_normal_mid": auc(
            [scores["prag"][v] for v in mid if groups[v] in ("EX", "IM")],
            [scores["prag"][v] for v in mid if groups[v] == "NH"]),
    }

    # P4: lit compliance, for both variants regardless of --lit choice
    comp = {}
    for lv in ("lit_v1", "lit_v2"):
        if not scores[lv]:
            comp[lv] = None
            continue
        zs = z[lv]
        vids = set(zs) & set(z["prag"])
        ex = [scores[lv][v] for v in vids if group_of(v) == "EX"]
        nh = [scores[lv][v] for v in vids if group_of(v) == "NH"]
        im_gap = [zs[v] - z["prag"][v] for v in vids if group_of(v) == "IM"]
        comp[lv] = {
            "auc_EX_vs_NH_via_lit": auc(ex, nh),
            "median_s_lit_EX": median(ex),
            "median_s_lit_NH": median(nh),
            "median_IM_suppression_z_lit_minus_z_prag": median(im_gap),
        }
    report["P4_compliance"] = comp

    # P5a: placebo — effort in place of prag
    if scores["effort"]:
        p5a, _ = d_stats(z["effort"], z[args.lit], groups, f"effort-{args.lit}")
        report["P5a_placebo"] = p5a
    # P5b: noise floor — prag vs its paraphrase
    if scores["prag_para"]:
        p5b, _ = d_stats(z["prag"], z["prag_para"], groups, "prag-prag_para")
        report["P5b_noise_floor"] = p5b

    print(json.dumps(report, indent=2))
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"written: {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
