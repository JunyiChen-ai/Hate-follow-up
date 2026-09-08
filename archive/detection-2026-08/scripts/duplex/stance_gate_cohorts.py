"""
Stance-axis gate test: cohort construction (CPU only, no model).

Builds the three diagnostic cohorts per arm from an existing scored run, at
that run's own label-free KDE valley:

  FP   the false positives at the method threshold
  TP   a cue-matched sample of true positives (judge-seen transcript carries a
       surface cue), size-matched to FP where the pool allows, seed 0
  TN   a cue-carrying sample of true negatives, same size rule, seed 0

Labels enter here only to *name* the cohorts. This is error diagnosis: no
threshold, no probe and no decision rule in this test reads a label.

Everything load-bearing is imported rather than redefined: the KDE-valley
recipe and the z loader come from scripts/duplex/c2_fullcorpus_analyze.py, the
surface-cue regexes from scripts/duplex/channel_restoration_analyze.py, and the
judge-seen transcript is reconstructed exactly as the source run's analyser
does (override if the gate accepted one, else the dataset transcript).

Usage:
  python scripts/duplex/stance_gate_cohorts.py --out results/stance_gate/cohorts.json
"""

import argparse
import json
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))

from c2_fullcorpus_analyze import kde_valley, load_z  # noqa: E402
from channel_restoration_analyze import PATS, has_cue  # noqa: E402
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

SEED = 0

# The two source runs. `label` is how the arm's ground truth is read, matching
# each source run's own analyser: ImpliHateVid by id prefix, HateMM by the
# annotation field through the frozen label map.
ARMS = {
    "ihv_train": {
        "dataset": "ImpliHateVid",
        "split": "train",
        "run_dir": "results/c2_fullcorpus",
        "judge_dir": "results/c2_fullcorpus/judge_8b",
        "label": "prefix",
        "source_report": "docs/duplex/reports/c2_fullcorpus_8b.json",
    },
    "hatemm_test": {
        "dataset": "HateMM",
        "split": "test",
        "run_dir": "results/testruns/hatemm",
        "judge_dir": "results/testruns/hatemm/judge_8b",
        "label": "annotation",
        "source_report": "docs/duplex/reports/test_c2_hatemm_8b.json",
    },
}

HATEMM_LABEL_MAP = {"Hate": 1, "Non Hate": 0}


def positive(vid, ann, mode):
    if mode == "prefix":
        return not vid.startswith("NH_")
    return HATEMM_LABEL_MAP[ann[vid]["label"]] == 1


def build_arm(name, cfg):
    run_dir = os.path.join(PROJECT_ROOT, cfg["run_dir"])
    z = load_z(os.path.join(PROJECT_ROOT, cfg["judge_dir"], "scores.jsonl"))

    ov_path = os.path.join(run_dir, "c2_overrides.json")
    with open(ov_path) as f:
        overrides = json.load(f)

    ann = load_annotations(cfg["dataset"])
    split_ids = load_clean_split_ids(cfg["dataset"], cfg["split"])
    scored = [v for v in split_ids if v in z]

    # The run's own label-free threshold, recomputed by the frozen recipe on
    # the same pool the source analyser used.
    thr = kde_valley([z[v] for v in scored])["value"]

    def seen_text(v):
        return overrides.get(v, ann[v].get("transcript", "") or "")

    pos = [v for v in scored if positive(v, ann, cfg["label"])]
    neg = [v for v in scored if not positive(v, ann, cfg["label"])]

    tp_all = [v for v in pos if z[v] >= thr]
    fn_all = [v for v in pos if z[v] < thr]
    fp_all = [v for v in neg if z[v] >= thr]
    tn_all = [v for v in neg if z[v] < thr]

    tp_cue_pool = [v for v in tp_all if has_cue(seen_text(v))]
    tn_cue_pool = [v for v in tn_all if has_cue(seen_text(v))]

    n_target = len(fp_all)
    rng = random.Random(SEED)
    tp_cohort = sorted(rng.sample(sorted(tp_cue_pool), min(n_target, len(tp_cue_pool))))
    rng = random.Random(SEED)
    tn_cohort = sorted(rng.sample(sorted(tn_cue_pool), min(n_target, len(tn_cue_pool))))

    fp_cohort = sorted(fp_all)

    cue_families = {}
    for fam, pat in PATS.items():
        cue_families[fam] = {
            "fp": sum(1 for v in fp_cohort if pat.search(seen_text(v))),
            "tp": sum(1 for v in tp_cohort if pat.search(seen_text(v))),
            "tn": sum(1 for v in tn_cohort if pat.search(seen_text(v))),
        }

    return {
        "arm": name,
        "dataset": cfg["dataset"],
        "split": cfg["split"],
        "judge_dir": cfg["judge_dir"],
        "source_report": cfg["source_report"],
        "label_source": cfg["label"],
        "n_scored": len(scored),
        "valley": thr,
        "confusion_at_valley": {
            "tp": len(tp_all), "fp": len(fp_all),
            "fn": len(fn_all), "tn": len(tn_all),
        },
        "pools": {
            "tp_cue_carrying": len(tp_cue_pool),
            "tn_cue_carrying": len(tn_cue_pool),
            "tp_total": len(tp_all),
            "tn_total": len(tn_all),
        },
        "sample_seed": SEED,
        "sample_target_n": n_target,
        "cohort_sizes": {
            "fp": len(fp_cohort), "tp": len(tp_cohort), "tn": len(tn_cohort),
        },
        "cue_family_counts": cue_families,
        "cohorts": {"fp": fp_cohort, "tp": tp_cohort, "tn": tn_cohort},
        "joint_z": {v: z[v] for v in fp_cohort + tp_cohort + tn_cohort},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stance_gate/cohorts.json")
    args = ap.parse_args()

    out = {"seed": SEED, "arms": {}}
    for name, cfg in ARMS.items():
        arm = build_arm(name, cfg)
        out["arms"][name] = arm
        print(f"[{name}] dataset={arm['dataset']}/{arm['split']} "
              f"n_scored={arm['n_scored']} valley={arm['valley']:.6f}")
        print(f"  confusion  {arm['confusion_at_valley']}")
        print(f"  pools      {arm['pools']}")
        print(f"  cohorts    {arm['cohort_sizes']}")
        print(f"  cue fams   {arm['cue_family_counts']}")

    path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    total = sum(sum(a["cohort_sizes"].values()) for a in out["arms"].values())
    print(f"\nWrote {path}; {total} videos across both arms, "
          f"{total * 3} probe calls at 3 probes each.")


if __name__ == "__main__":
    main()
