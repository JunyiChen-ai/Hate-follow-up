#!/usr/bin/env python3
"""Check and re-evaluate the existing DeHate run of MultiHateLoc (CPU, uoa-lab1).

DSANet's DeHate run was checked the same way and passed, but it is not reused: after the coordinator's instruction of
2026-10-08, DSANet (like VadCLIP and AVadCLIP) runs its upstream XD-Violence preset with a fixed schedule and the last
epoch on all three corpora, while the existing DeHate run was Optuna-tuned.  Its check and evaluation were moved to
runs/20261008_baselines/_superseded/dsanet_DeHate_reused_optuna/; DeHate DSANet is rerun.

Source: Retrieval-hate on uoa-lab2, runs/20260926_dehate_external/baselines/{tuning,final}/<method>/dehate/ (5-trial
Optuna on val video AP, then seeds 234 / 2025 / 3407, launch/baseline.sh there).  The run records (best.json, every
trial's and seed's train_meta.json / train_log.json / frozen_config.json, the seeds' 1 fps scores.jsonl) were copied
read-only to runs/20261008_baselines/<method>/DeHate/source_lab2/.

A run is reused only if all of these hold (otherwise this script stops and the corpus must be rerun):
  * the Optuna objective was the validation video AP (best.json value == the winning trial's selected val AP)
  * every trial and every seed trained on exactly our DeHate train split and selected on exactly our val split
    (DSANet: the id lists in train_meta.json; MultiHateLoc: n_train / n_val, its official manifests), with
    selection by val video AP (`select` = val / val_ap)
  * each seed's frozen configuration is the tuning winner, and seeds are 234 / 2025 / 3407
  * train ∪ val ∩ test cohort = ∅ (checked again here on ids)
  * the scores cover every cohort video with a finite score on every GT frame after the x4 / last-value rule
Branches: DSANet score_mlp (+ score_align as dsanet_align), MultiHateLoc score_fused (the branches of the existing
DeHate table, declared there before its results).

    python experiments/20261008_baselines/weaksup_common/reuse_dehate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

BRANCHES = {"dsanet": (("score_mlp", "dsanet"), ("score_align", "dsanet_align")),
            "multihateloc": (("score_fused", "multihateloc"),)}


def check(method, src, log):
    train, val, test = (set(C.split_ids("dehate", k)) for k in ("train", "val", "test"))
    if (train | val) & test or train & val:
        raise RuntimeError("DeHate split overlap")
    best = json.loads((src / "tuning" / "best.json").read_text())
    report = {"best_trial": best["best_trial"], "best_value": best["best_value"], "n_complete": best["n_complete"],
              "trials": {}, "seeds": {}}
    if best["n_complete"] != 5:
        raise RuntimeError(f"{method}: {best['n_complete']} completed trials, expected 5")

    def meta_of(d):
        if method == "dsanet":
            m = json.loads((d / "train_meta.json").read_text())
            ok_ids = set(m["train_ids"]) == train and set(m["val_ids"]) == val
            return m, ok_ids, m["args"]["select"] == "val", m["args"]["seed"]
        m = json.loads((d / "dehate" / "train_log.json").read_text())
        ok_ids = m["n_train"] == len(train) and m["n_val"] == len(val)
        return m, ok_ids, m["args"]["select"] == "val_ap", m["args"]["seed"]

    winner_ap = None
    for d in sorted((src / "tuning").glob("trial_*")):
        m, ok_ids, ok_sel, seed = meta_of(d)
        report["trials"][d.name] = {"val_ap": m["selected_val_video_ap"], "split_ok": ok_ids, "select_ok": ok_sel}
        if not (ok_ids and ok_sel):
            raise RuntimeError(f"{method} {d.name}: split or selection rule differs")
        if d.name == f"trial_{best['best_trial']:04d}":
            winner_ap = m["selected_val_video_ap"]
    if winner_ap is None or abs(winner_ap - best["best_value"]) > 1e-12:
        raise RuntimeError(f"{method}: best.json value is not the winning trial's validation AP")
    for s in C.SEEDS:
        d = src / "final" / f"seed_{s}"
        m, ok_ids, ok_sel, seed = meta_of(d)
        frozen = json.loads((d / "frozen_config.json").read_text())
        same = frozen["best_trial"] == best["best_trial"] and frozen["seed"] == s == seed
        report["seeds"][str(s)] = {"selected_epoch": m["selected_epoch"], "val_ap": m["selected_val_video_ap"],
                                   "split_ok": ok_ids, "select_ok": ok_sel, "winner_config": same}
        if not (ok_ids and ok_sel and same):
            raise RuntimeError(f"{method} seed {s}: split, selection or configuration differs")
    log(f"{method}: 5 trials and 3 seeds trained on our train split, selected on our val split by val video AP; "
        f"winner trial {best['best_trial']} (val AP {best['best_value']:.4f})")
    return report


def main():
    for method in ("multihateloc",):
        base = C.RUNS / method / "DeHate"
        src = base / "source_lab2"
        log = C.RunLog(base / "run.log")
        log(f"reuse check {method} DeHate, code {C.code_version()}")
        report = check(method, src, log)
        (base / "reuse_check.json").write_text(json.dumps(report, indent=2) + "\n")
        branches, names = zip(*BRANCHES[method])
        for s in C.SEEDS:
            out = base / f"seed{s}"
            out.mkdir(parents=True, exist_ok=True)
            slog = C.RunLog(out / "run.log", mode="w")
            scores = src / "final" / f"seed_{s}" / ("dehate/scores.jsonl" if method == "multihateloc" else "scores.jsonl")
            frozen = json.loads((src / "final" / f"seed_{s}" / "frozen_config.json").read_text())
            config = {"method": method, "dataset": "DeHate", "seed": s, "reused": True,
                      "source": "uoa-lab2 ~/Retrieval-hate/runs/20260926_dehate_external/baselines/final/"
                                f"{method}/dehate/seed_{s}/ (copied to {scores.relative_to(C.REPO)})",
                      "trained_on": f"{frozen.get('host')} {frozen.get('started')}", "params": frozen["params"],
                      "selection": {"hyper-parameters": "5-trial Optuna on val video AP (sampler seed 234)",
                                    "epoch": "best val video AP inside the run"},
                      "checks": report["seeds"][str(s)], "branches": BRANCHES[method],
                      "code": f"experiments/20261008_baselines/weaksup_common/reuse_dehate.py, {C.code_version()}"}
            (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            slog(f"{method} DeHate seed {s}: reused scores {scores.relative_to(C.REPO)}")
            rows, coverage = C.build_predictions("dehate", scores, branches, names, s,
                                                 "experiments/20261008_baselines/weaksup_common/reuse_dehate.py", slog)
            (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
            C.evaluate("dehate", rows, out, slog)
        per = {s: {r["method"]: r for r in json.loads((base / f"seed{s}" / "metrics.json").read_text())["per_dataset"]}
               for s in C.SEEDS}
        summ = {"dataset": "DeHate", "seeds": list(C.SEEDS), "reused": True, "methods": {}}
        for name in names:
            summ["methods"][name] = {}
            for k in ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC"):
                v = np.array([per[s][name][k] for s in C.SEEDS], float)
                summ["methods"][name][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                            "per_seed": {str(s): float(per[s][name][k]) for s in C.SEEDS}}
            e = summ["methods"][name]
            log(f"{name} DeHate: ROC {e['frame_ROC_AUC']['mean']:.4f}±{e['frame_ROC_AUC']['sd']:.4f} "
                f"PR {e['frame_PR_AUC']['mean']:.4f}±{e['frame_PR_AUC']['sd']:.4f} "
                f"within {e['within_video_macro_ROC_AUC']['mean']:.4f}±{e['within_video_macro_ROC_AUC']['sd']:.4f}")
        (base / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")


if __name__ == "__main__":
    main()
