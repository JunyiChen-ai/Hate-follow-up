#!/usr/bin/env python3
"""Oracle test selection, step 3: per method x corpus, the two oracle summaries over seeds, beside the current
(validation- or last-epoch-selected) numbers.

    python summarize.py

(B) single-checkpoint oracle (HEADLINE, user decision 2026-10-09): for each seed, the one candidate (checkpoint x
    branch) with the highest mean of the three test metrics; its three metrics; then mean +- sd over seeds.
(A) per-metric oracle (for the record): for each seed and each metric separately, the maximum over all candidates.
current: the existing row (declared branch, val / last / authors' selection), copied unchanged from
    runs/20261008_baselines/<method>/<DS>/seed<k>/metrics.json.
Also per seed: the retrained run's candidate under the current rule (declared branch) and its difference from the
current number (does the retraining reproduce the current run?).
All metrics come from the canonical evaluator's metrics.json files listed in each seed's candidates.json.
Writes runs/20261008_baselines/oracle_test_selection/{oracle_summary.json, oracle_table.md}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_common as O  # noqa: E402

C = O.C
M3 = O.METRICS
SHORT = {"frame_ROC_AUC": "ROC", "frame_PR_AUC": "PR", "within_video_macro_ROC_AUC": "within"}


def current_metrics(method, ds, seed):
    p = O.current_seed_dir(method, ds, seed) / "metrics.json"
    if not p.is_file():
        return None
    rows = [r for r in json.loads(p.read_text())["per_dataset"] if r["method"] == O.METHODS[method]["current_name"]]
    if len(rows) != 1:
        return None
    return {k: float(rows[0][k]) for k in M3} | {"metrics_json": str(p.relative_to(C.REPO))}


def rule_tag(method, ds, seed):
    """Checkpoint tag the current rule picks inside the retrained run."""
    sd = O.seed_dir(method, ds, seed)
    try:
        if method.startswith("mil_"):
            return f"e{json.loads((sd / 'train_meta.json').read_text())['selected_epoch']:03d}"
        if method in ("vadclip", "dsanet", "avadclip"):
            return "e010"
        if method == "multihateloc":
            return f"e{json.loads((sd / C.CORPUS[ds] / 'train_log.json').read_text())['selected_epoch']:03d}"
        if method == "sage":
            hist = json.loads((sd / "train_history.json").read_text())
            return f"e{[r['epoch'] for r in hist if r.get('saved')][-1]:02d}"
        if method == "clara":
            th = json.loads((sd / "train_history.json").read_text())
            step = int(th["best_checkpoint"].rsplit("-", 1)[1])
            return f"e{int(round([h['epoch'] for h in th['eval'] if h['step'] == step][0])):02d}"
    except (FileNotFoundError, KeyError, IndexError):
        return None
    return None


def mean_sd(vals):
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "per_seed": list(map(float, v))}


def summarize_one(method, ds):
    spec = O.METHODS[method]
    seeds, missing = {}, []
    for s in O.SEEDS:
        cur = current_metrics(method, ds, s)
        cp = O.seed_dir(method, ds, s) / "candidates.json"
        if not cp.is_file():
            missing.append(s)
            seeds[str(s)] = {"current": cur, "status": "pending"}
            continue
        cj = json.loads(cp.read_text())
        valid = [c for c in cj["candidates"] if c["valid"]]
        if not valid:
            missing.append(s)
            seeds[str(s)] = {"current": cur, "status": "no valid candidate"}
            continue
        ident = lambda c: {"checkpoint": c["checkpoint"], "branch": c["branch"], "from": c["from"],
                           "metrics_json": c["metrics_json"]}
        a = {k: {"value": max(c[k] for c in valid),
                 **ident(max(valid, key=lambda c, k=k: c[k]))} for k in M3}
        b_c = max(valid, key=lambda c: sum(c[k] for k in M3) / 3)
        b = {**{k: b_c[k] for k in M3}, "mean_of_three": sum(b_c[k] for k in M3) / 3, **ident(b_c)}
        tag = rule_tag(method, ds, s)
        rr = [c for c in valid if c["checkpoint"] == tag and c["branch"] == spec["declared"]]
        repro = None
        if rr and cur:
            repro = {"checkpoint": tag, **{k: rr[0][k] for k in M3},
                     "delta_vs_current": {k: rr[0][k] - cur[k] for k in M3},
                     "max_abs_delta": max(abs(rr[0][k] - cur[k]) for k in M3)}
        seeds[str(s)] = {"current": cur, "oracle_A": a, "oracle_B": b, "retrain_rule_candidate": repro,
                         "n_candidates": cj["n_candidates"], "n_valid": cj["n_valid"],
                         "n_invalid": cj["n_candidates"] - cj["n_valid"],
                         "invalid": [f"{c['checkpoint']}@{c['branch']}" for c in cj["candidates"] if not c["valid"]],
                         "candidates_json": str(cp.relative_to(C.REPO)), "status": "done"}
    out = {"method": method, "label": spec["label"], "dataset": ds, "declared_branch": spec["declared"],
           "branches": spec["branches"], "current_rule": spec["rule"], "seeds": seeds,
           "complete": not missing, "seeds_pending": missing}
    curs = [seeds[str(s)]["current"] for s in O.SEEDS]
    if all(curs):
        out["current_mean_sd"] = {k: mean_sd([c[k] for c in curs]) for k in M3}
    if not missing:
        out["oracle_B_mean_sd"] = {k: mean_sd([seeds[str(s)]["oracle_B"][k] for s in O.SEEDS]) for k in M3}
        out["oracle_A_mean_sd"] = {k: mean_sd([seeds[str(s)]["oracle_A"][k]["value"] for s in O.SEEDS]) for k in M3}
    return out


def cell(block):
    if not block:
        return "pending"
    return " / ".join(f"{block[k]['mean']:.4f}" for k in M3)


def cell_sd(block):
    if not block:
        return "pending"
    return " / ".join(f"{block[k]['mean']:.4f} ± {block[k]['sd']:.4f}" for k in M3)


def main():
    rows = [summarize_one(m, ds) for m in O.METHODS for ds in O.DATASETS]
    summ = {"what": "oracle test selection of the weakly supervised baselines: checkpoint and output branch chosen on "
                    "TEST labels, for the baselines only, as an upper bound (user decision 2026-10-09); headline = "
                    "oracle B (single checkpoint x branch per seed with the best mean of the three test metrics)",
            "metrics": list(M3), "seeds": list(O.SEEDS), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code": f"experiments/20261008_baselines/oracle_test_selection/summarize.py, {C.code_version()}",
            "rows": rows}
    (O.ORACLE / "oracle_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    L = ["# Weakly supervised baselines: oracle test selection (TEST-label upper bound, baselines only)", "",
         "Checkpoint (epoch) and output branch chosen on the TEST set, separately per seed. This uses test labels and",
         "is an upper bound for the baselines; our own method never reads test labels. Cells: pooled frame ROC-AUC /",
         "pooled frame PR-AUC / within-video macro ROC-AUC, mean over seeds 2025 / 234 / 3407 (sd in the second table",
         "and in `oracle_summary.json`). Source: `runs/20261008_baselines/oracle_test_selection/oracle_summary.json`,",
         "built from each seed's `candidates.json`, whose numbers are the canonical evaluator's `metrics.json` outputs.",
         "",
         "- **oracle B (headline)**: per seed, the one candidate (checkpoint x branch) with the highest mean of the three",
         "  test metrics; its three metrics; mean over seeds.",
         "- oracle A (record only): per seed and per metric separately, the maximum over all candidates (the three",
         "  numbers can come from different candidates).",
         "- current: the existing row (declared branch; val-selected / last epoch / authors' rule), unchanged.", "",
         "| method | corpus | current (val / last) | oracle B (headline) | oracle A (per metric) | candidates per seed |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        ncand = "/".join(str(r["seeds"][str(s)].get("n_valid", "-")) for s in O.SEEDS)
        L.append(f"| {r['label']} | {r['dataset']} | {cell(r.get('current_mean_sd'))} | "
                 f"**{cell(r.get('oracle_B_mean_sd'))}** | {cell(r.get('oracle_A_mean_sd'))} | {ncand} |")
    L += ["", "## Oracle B with sd, and the candidate chosen per seed", "",
          "| method | corpus | oracle B mean ± sd (ROC / PR / within) | chosen candidate per seed (2025; 234; 3407) |",
          "|---|---|---|---|"]
    for r in rows:
        ch = "; ".join(f"{r['seeds'][str(s)]['oracle_B']['checkpoint']} {r['seeds'][str(s)]['oracle_B']['branch']}"
                       if r["seeds"][str(s)].get("oracle_B") else "pending" for s in O.SEEDS)
        L.append(f"| {r['label']} | {r['dataset']} | {cell_sd(r.get('oracle_B_mean_sd'))} | {ch} |")
    L += ["", "## Does the retraining reproduce the current run?", "",
          "Retrained run's candidate under the current rule (declared branch) minus the current number; max |Δ| over",
          "the three metrics, per seed. 0 means bit-identical scores.", "",
          "| method | corpus | max abs Δ per seed (2025; 234; 3407) |", "|---|---|---|"]
    for r in rows:
        d = []
        for s in O.SEEDS:
            rr = r["seeds"][str(s)].get("retrain_rule_candidate")
            d.append(f"{rr['max_abs_delta']:.4f} ({rr['checkpoint']})" if rr else "n/a")
        L.append(f"| {r['label']} | {r['dataset']} | {'; '.join(d)} |")
    (O.ORACLE / "oracle_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[13:13 + 2 + len(rows)]))


if __name__ == "__main__":
    main()
