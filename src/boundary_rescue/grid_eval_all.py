"""Combined grid evaluation: 4 stage-1 MLLMs × C(8,3)=56 judge triplets × 4 datasets.

For each (stage-1 slug, triplet, dataset):
  1. Load stage-1 baseline_preds (oracle criterion per slug,ds).
  2. Load entropy-above-mean band (from Phase D).
  3. Load 3 judge files (IH-prompt for ImpliHateVid, HATEMM_DEF for others).
  4. Compute: for each band video, majority vote across 3 judges;
     if majority differs from stage-1 pred and ≥2 judges returned a
     valid pred ∈ {0,1}, flip.
  5. Report acc, mF1, Δ stage-1, Δ V1, coverage.

Outputs:
  results/boundary_rescue/grid_eval/grid_raw.jsonl      (one row per cell)
  results/boundary_rescue/grid_eval/grid_summary.json   (rankings)
  results/boundary_rescue/grid_eval/grid_summary_top.md (tables)

No GPU. Expected runtime ~1–2 min for 1120 cells.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "our_method"))
sys.path.insert(0, str(_HERE.parent / "naive_baseline"))

from data_utils import SKIP_VIDEOS  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402
from data_utils import load_annotations  # noqa: E402

PROJECT_ROOT = Path("/data/jehc223/EMNLP2")
OUT_ROOT = PROJECT_ROOT / "results" / "boundary_rescue"
GRID_DIR = OUT_ROOT / "grid_eval"
DS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
N_TEST = {"MHClip_EN": 161, "MHClip_ZH": 149, "HateMM": 215, "ImpliHateVid": 401}

# 8-judge pool (phi-4 excluded: 1-line only).
# canonical stem -> offline_test filename stem (without 'offline_test_')
JUDGES = [
    "qwen3-vl-8b",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "qwen2.5-vl-32b-awq",
    "qwen2.5-vl-72b-awq",
    "internvl35-8b",
    "llava-onevision-qwen2-7b-ov-hf",
    "minicpm-v-26",
]
# For IH, IH-prompt variant filenames differ: some internvl35-8b variants
# may be saved as internvl3_5-8b; we try both.
IH_FILE_CANDIDATES = {
    "qwen3-vl-8b": ["offline_test_ih_qwen3-vl-8b.jsonl"],
    "gemma-3-12b-it": ["offline_test_ih_gemma-3-12b-it.jsonl"],
    "gemma-3-27b-it": ["offline_test_ih_gemma-3-27b-it.jsonl"],
    "qwen2.5-vl-32b-awq": ["offline_test_ih_qwen2.5-vl-32b-awq.jsonl"],
    "qwen2.5-vl-72b-awq": ["offline_test_ih_qwen2.5-vl-72b-awq.jsonl"],
    "internvl35-8b": ["offline_test_ih_internvl35-8b.jsonl",
                      "offline_test_ih_internvl3_5-8b.jsonl"],
    "llava-onevision-qwen2-7b-ov-hf": ["offline_test_ih_llava-onevision-qwen2-7b-ov-hf.jsonl"],
    "minicpm-v-26": ["offline_test_ih_minicpm-v-26.jsonl"],
}


def ld_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def macro_f1(y, yh):
    cls = sorted(set(y))
    if not cls:
        return 0.0
    s = 0.0
    for c in cls:
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        s += 2 * pr * r / (pr + r) if pr + r else 0.0
    return s / len(cls)


def load_oracle_map(slugs):
    """Read threshold_search_summary.json → per (slug, ds) oracle criterion."""
    p = OUT_ROOT / "threshold_search_summary.json"
    if not p.exists():
        return {}
    summary = json.load(open(p))
    m = {}
    for o in summary["oracles"]:
        if o["slug"] in slugs:
            m[(o["slug"], o["ds"])] = o["criterion"]
    return m


def baseline_pred_path(slug, ds, crit):
    if slug == "2b" and crit == "protocol":
        return OUT_ROOT / ds / "baseline_preds_v2.jsonl"
    return OUT_ROOT / ds / f"baseline_preds_v2_{slug}_{crit}.jsonl"


def entropy_band_path(slug, ds):
    return OUT_ROOT / ds / f"candidates_entropy_band_{slug}.jsonl"


def judge_path(judge_stem, ds):
    if ds == "ImpliHateVid":
        for cand in IH_FILE_CANDIDATES[judge_stem]:
            p = OUT_ROOT / ds / cand
            if p.exists():
                return p
        return None
    return OUT_ROOT / ds / f"offline_test_{judge_stem}.jsonl"


def load_labels(ds):
    ann = load_annotations(ds)
    return {v: collapse_label(ds, ann[v]["label"]) for v in ann}


def eval_cell(slug, ds, triplet, oracle_map, caches):
    crit = oracle_map.get((slug, ds), "protocol")
    bp = baseline_pred_path(slug, ds, crit)
    if not bp.exists():
        return None
    base_key = ("base", slug, ds, crit)
    if base_key not in caches:
        caches[base_key] = {r["video_id"]: int(r["pred_baseline"])
                            for r in ld_jsonl(bp)}
    base = caches[base_key]

    ep = entropy_band_path(slug, ds)
    band_rows = caches.get(("band", slug, ds))
    if band_rows is None:
        band_rows = ld_jsonl(ep)
        caches[("band", slug, ds)] = band_rows
    band_set = {r["video_id"] for r in band_rows if r.get("in_band")}

    judge_dicts = []
    for j in triplet:
        key = ("judge", j, ds)
        jd = caches.get(key)
        if jd is None:
            p = judge_path(j, ds)
            if p is None:
                jd = {}
            else:
                jd = {r["video_id"]: r for r in ld_jsonl(p)}
            caches[key] = jd
        judge_dicts.append(jd)

    labels = caches.get(("lab", ds))
    if labels is None:
        labels = load_labels(ds)
        caches[("lab", ds)] = labels

    skip = SKIP_VIDEOS.get(ds, set())
    valid_vids = [v for v in base if v not in skip and labels.get(v) in (0, 1)]
    y = [labels[v] for v in valid_vids]
    yh_s1 = [base[v] for v in valid_vids]
    yh = list(yh_s1)

    n_band = 0
    n_cov = 0
    n_flipped = 0
    n_fixable = 0
    n_harmful = 0
    for i, v in enumerate(valid_vids):
        if v not in band_set:
            continue
        n_band += 1
        preds = []
        for jd in judge_dicts:
            p = jd.get(v, {}).get("pred")
            if p in (0, 1):
                preds.append(p)
        if len(preds) < 2:
            continue
        n_cov += 1
        s1 = sum(1 for p in preds if p == 1)
        s0 = len(preds) - s1
        if s1 >= 2:
            mv = 1
        elif s0 >= 2:
            mv = 0
        else:
            continue
        if mv != yh[i]:
            n_flipped += 1
            s1_right = (yh[i] == y[i])
            if s1_right:
                n_harmful += 1
            else:
                if mv == y[i]:
                    n_fixable += 1
            yh[i] = mv

    correct = sum(1 for a, b in zip(y, yh) if a == b)
    # Stage-1 metrics under same valid set (but acc denominator is N_TEST[ds]
    # for comparability with V1 pinned numbers).
    c_s1 = sum(1 for a, b in zip(y, yh_s1) if a == b)
    acc = correct / N_TEST[ds]
    acc_s1 = c_s1 / N_TEST[ds]
    mf1 = macro_f1(y, yh)
    mf1_s1 = macro_f1(y, yh_s1)
    return {
        "slug": slug, "ds": ds,
        "triplet": list(triplet),
        "criterion": crit,
        "acc": float(acc), "mf1": float(mf1),
        "acc_s1": float(acc_s1), "mf1_s1": float(mf1_s1),
        "delta_acc": float(acc - acc_s1),
        "delta_mf1": float(mf1 - mf1_s1),
        "n_test": N_TEST[ds],
        "n_band": int(n_band),
        "coverage": float(n_cov / n_band) if n_band else 0.0,
        "n_flipped": int(n_flipped),
        "n_fixable": int(n_fixable),
        "n_harmful": int(n_harmful),
    }


def load_v1():
    p = OUT_ROOT / "v2_baseline.json"
    vb = json.load(open(p))
    # V1 from memory / docs (strict-beat pinned targets):
    V1 = {"MHClip_EN": (0.7826, 0.6958),
          "MHClip_ZH": (0.8255, 0.8023),
          "HateMM": (0.8465, 0.8362),
          "ImpliHateVid": (0.8204, 0.8199)}
    return V1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="+",
                    default=["2b", "qwen2.5-vl-7b", "gemma-3-12b-it",
                             "minicpm-v-26", "pixtral-12b-2409", "internvl3-14b"],
                    help="stage-1 slugs to include in the grid")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    oracle_map = load_oracle_map(args.slugs)
    V1 = load_v1()
    triplets = list(itertools.combinations(JUDGES, 3))
    caches = {}
    all_rows = []
    skipped_cells = 0

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = GRID_DIR / "grid_raw.jsonl"

    with open(raw_path, "w") as fout:
        for slug in args.slugs:
            present = any(entropy_band_path(slug, ds).exists() for ds in DS)
            if not present:
                print(f"[skip] slug={slug}: no entropy_band files (Phase D not run)")
                continue
            for triplet in triplets:
                for ds in DS:
                    r = eval_cell(slug, ds, triplet, oracle_map, caches)
                    if r is None:
                        skipped_cells += 1
                        continue
                    v1a, v1m = V1[ds]
                    r["vs_v1_acc"] = r["acc"] - v1a
                    r["vs_v1_mf1"] = r["mf1"] - v1m
                    r["beats_v1"] = bool(r["acc"] >= v1a - 1e-9 and r["mf1"] >= v1m - 1e-9)
                    r["beats_s1"] = bool(r["acc"] >= r["acc_s1"] - 1e-9 and r["mf1"] >= r["mf1_s1"] - 1e-9)
                    all_rows.append(r)
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Cells: {len(all_rows)}  skipped: {skipped_cells}")

    # Aggregate per (slug, triplet)
    agg = {}
    for r in all_rows:
        key = (r["slug"], tuple(r["triplet"]))
        a = agg.setdefault(key, {"slug": r["slug"], "triplet": list(r["triplet"]),
                                 "per_ds": {}, "beats_v1_count": 0,
                                 "beats_s1_count": 0,
                                 "sum_delta_acc": 0.0, "sum_delta_mf1": 0.0,
                                 "sum_vs_v1_acc": 0.0, "sum_vs_v1_mf1": 0.0})
        a["per_ds"][r["ds"]] = {"acc": r["acc"], "mf1": r["mf1"],
                                "delta_acc": r["delta_acc"], "delta_mf1": r["delta_mf1"],
                                "vs_v1_acc": r["vs_v1_acc"], "vs_v1_mf1": r["vs_v1_mf1"],
                                "beats_v1": r["beats_v1"],
                                "beats_s1": r["beats_s1"], "coverage": r["coverage"]}
        if r["beats_v1"]:
            a["beats_v1_count"] += 1
        if r["beats_s1"]:
            a["beats_s1_count"] += 1
        a["sum_delta_acc"] += r["delta_acc"]
        a["sum_delta_mf1"] += r["delta_mf1"]
        a["sum_vs_v1_acc"] += r["vs_v1_acc"]
        a["sum_vs_v1_mf1"] += r["vs_v1_mf1"]
    for a in agg.values():
        n = len(a["per_ds"])
        a["n_ds"] = n
        a["avg_delta_acc"] = a["sum_delta_acc"] / n if n else 0
        a["avg_delta_mf1"] = a["sum_delta_mf1"] / n if n else 0

    # Rankings per slug
    rankings = {"strict_beat_v1": {}, "avg_delta_acc": {}}
    for slug in args.slugs:
        slug_rows = [a for a in agg.values() if a["slug"] == slug and a["n_ds"] == 4]
        strict = sorted(slug_rows,
                        key=lambda a: (-a["beats_v1_count"],
                                       -(a["sum_vs_v1_acc"] + a["sum_vs_v1_mf1"])))
        delta = sorted(slug_rows, key=lambda a: -a["avg_delta_acc"])
        rankings["strict_beat_v1"][slug] = strict[:args.top_k]
        rankings["avg_delta_acc"][slug] = delta[:args.top_k]

    # Cross-slug global top-20 (by strict_beat then sum)
    all_agg = [a for a in agg.values() if a["n_ds"] == 4]
    global_strict = sorted(all_agg,
                           key=lambda a: (-a["beats_v1_count"],
                                          -(a["sum_vs_v1_acc"] + a["sum_vs_v1_mf1"])))[:20]
    global_delta = sorted(all_agg, key=lambda a: -a["avg_delta_acc"])[:20]

    # Stats
    stats = {}
    for slug in args.slugs:
        slug_rows = [a for a in agg.values() if a["slug"] == slug and a["n_ds"] == 4]
        if not slug_rows:
            continue
        # judge frequency in top-10
        top10 = sorted(slug_rows,
                       key=lambda a: (-a["beats_v1_count"],
                                      -(a["sum_vs_v1_acc"] + a["sum_vs_v1_mf1"])))[:10]
        from collections import Counter
        cnt = Counter()
        for a in top10:
            for j in a["triplet"]:
                cnt[j] += 1
        stats[slug] = {
            "n_triplets": len(slug_rows),
            "n_beats_v1_4_of_4": sum(1 for a in slug_rows if a["beats_v1_count"] == 4),
            "n_beats_v1_3_of_4": sum(1 for a in slug_rows if a["beats_v1_count"] >= 3),
            "n_beats_s1_4_of_4": sum(1 for a in slug_rows if a["beats_s1_count"] == 4),
            "n_beats_s1_3_of_4": sum(1 for a in slug_rows if a["beats_s1_count"] >= 3),
            "judges_in_top10": dict(cnt.most_common()),
            "best_triplet": top10[0]["triplet"] if top10 else None,
            "best_triplet_beats_v1": top10[0]["beats_v1_count"] if top10 else 0,
            "best_triplet_avg_delta_acc": slug_rows and max(a["avg_delta_acc"] for a in slug_rows),
        }

    json.dump({"rankings": rankings, "global_strict": global_strict,
               "global_delta": global_delta, "stats": stats,
               "oracle_map": {f"{k[0]}/{k[1]}": v for k, v in oracle_map.items()},
               "V1": V1},
              open(GRID_DIR / "grid_summary.json", "w"), indent=2, default=str)

    # Markdown report
    lines = ["# Grid evaluation — stage-1 MLLM × 3-judge triplet", ""]
    lines += ["## Per-slug summary stats", "",
              "| slug | #triplets | 4/4 beat V1 | ≥3/4 beat V1 | 4/4 beat S1 | ≥3/4 beat S1 | best avg Δacc | best triplet |",
              "|---|---|---|---|---|---|---|---|"]
    for slug, s in stats.items():
        bt = ",".join(s["best_triplet"]) if s["best_triplet"] else "–"
        lines.append(f"| {slug} | {s['n_triplets']} | {s['n_beats_v1_4_of_4']} | "
                     f"{s['n_beats_v1_3_of_4']} | {s['n_beats_s1_4_of_4']} | "
                     f"{s['n_beats_s1_3_of_4']} | {s['best_triplet_avg_delta_acc']:.4f} | {bt} |")
    lines.append("")
    for slug in args.slugs:
        if slug not in rankings["strict_beat_v1"]:
            continue
        lines += [f"## slug = `{slug}` — strict-beat-V1 top-{args.top_k}", "",
                  "| rank | triplet | #V1 | Σ vs-V1 acc+mf1 | per-ds (acc/mf1) |",
                  "|---|---|---|---|---|"]
        for i, a in enumerate(rankings["strict_beat_v1"][slug], 1):
            per = " ; ".join(f"{ds[:2]}:{a['per_ds'][ds]['acc']:.3f}/{a['per_ds'][ds]['mf1']:.3f}"
                             f"{'✓' if a['per_ds'][ds]['beats_v1'] else ''}"
                             for ds in DS if ds in a["per_ds"])
            lines.append(f"| {i} | {','.join(a['triplet'])} | {a['beats_v1_count']} | "
                         f"{a['sum_vs_v1_acc']+a['sum_vs_v1_mf1']:+.4f} | {per} |")
        lines.append("")
        lines += [f"### slug = `{slug}` — avg-Δ-acc top-{args.top_k}", "",
                  "| rank | triplet | avg Δ acc | avg Δ mF1 | #V1 |",
                  "|---|---|---|---|---|"]
        for i, a in enumerate(rankings["avg_delta_acc"][slug], 1):
            lines.append(f"| {i} | {','.join(a['triplet'])} | {a['avg_delta_acc']:+.4f} | "
                         f"{a['avg_delta_mf1']:+.4f} | {a['beats_v1_count']} |")
        lines.append("")

    lines += ["## Global cross-slug top-20 (strict-beat-V1)", "",
              "| rank | slug | triplet | #V1 | Σ vs-V1 | per-ds |",
              "|---|---|---|---|---|---|"]
    for i, a in enumerate(global_strict, 1):
        per = " ; ".join(f"{ds[:2]}:{a['per_ds'][ds]['acc']:.3f}"
                         f"{'✓' if a['per_ds'][ds]['beats_v1'] else ''}"
                         for ds in DS if ds in a["per_ds"])
        lines.append(f"| {i} | {a['slug']} | {','.join(a['triplet'])} | {a['beats_v1_count']} | "
                     f"{a['sum_vs_v1_acc']+a['sum_vs_v1_mf1']:+.4f} | {per} |")
    (GRID_DIR / "grid_summary_top.md").write_text("\n".join(lines))

    print(f"Raw cells → {raw_path}")
    print(f"Summary JSON → {GRID_DIR / 'grid_summary.json'}")
    print(f"Markdown → {GRID_DIR / 'grid_summary_top.md'}")


if __name__ == "__main__":
    main()
