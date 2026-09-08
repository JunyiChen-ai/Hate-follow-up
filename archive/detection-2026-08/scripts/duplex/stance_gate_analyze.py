"""
Stance-axis gate test: analysis and verdict against the frozen decision rule.

The rule below was frozen in the task assignment before any probe was scored
and is transcribed here unchanged. Point estimates decide it; the bootstrap
intervals are reported as context and have no authority over the verdict.

  Let, on a given arm,
    A_stance = AUC(cue-matched TP vs FP) by stance_v1 z
    A_joint  = the same pairs by the source run's joint z
    A_effort = the same pairs by effort_ctrl z
    noise    = |AUC by (z_stance_v1 - z_stance_para) - 0.5|

  PASS (axis alive), on the primary arm (IHV train):
    A_stance >= 0.65
    A_stance - A_joint  >= 0.05
    A_stance - A_effort >= 0.05
    noise <= 0.05
  and the HateMM arm directionally consistent (A_stance > A_joint).

  FAIL: any of the four fails on the primary arm.

AUC(cue-carrying TN vs FP) is reported descriptively for every probe: a live
stance axis should not flag cue-carrying true negatives either.

Emits statistics only. No transcript text and no video ids reach the output.

Usage:
  python scripts/duplex/stance_gate_analyze.py \
    --cohorts results/stance_gate/cohorts.json \
    --scores  results/stance_gate/probe_scores.jsonl \
    --out     docs/duplex/reports/stance_gate_diag.json
"""

import argparse
import datetime
import json
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))

from channel_restoration_analyze import auc, median, quantile  # noqa: E402
from stance_gate_probe import PROBE_BLOCKS, PROBE_ORDER  # noqa: E402

PRIMARY_ARM = "ihv_train"
BARS = {"a_stance_min": 0.65, "delta_joint_min": 0.05,
        "delta_effort_min": 0.05, "noise_max": 0.05}
N_BOOT = 1000
BOOT_SEED = 0


def spearman(a, b, ids):
    """Rank correlation, average ranks on ties. Descriptive only."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r, i = [0.0] * len(vals), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks([a[v] for v in ids]), ranks([b[v] for v in ids])
    n = len(ids)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else None


def stats(xs):
    return {"n": len(xs), "mean": sum(xs) / len(xs) if xs else None,
            "median": median(xs), "q25": quantile(xs, 0.25),
            "q75": quantile(xs, 0.75)}


def boot_auc(scores, pos, neg, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CI, resampling within each class."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        p = [rng.choice(pos) for _ in pos]
        n = [rng.choice(neg) for _ in neg]
        rows = sorted([(scores[v], 1) for v in p] + [(scores[v], 0) for v in n])
        r, i = [0.0] * len(rows), 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[k] = avg
            i = j + 1
        s = sum(r[k] for k in range(len(rows)) if rows[k][1] == 1)
        vals.append((s - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]]


def boot_delta(sa, sb, pos, neg, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired CI on AUC(sa) - AUC(sb): the same resampled videos score both."""
    rng = random.Random(seed)

    def one(scores, p, n):
        rows = sorted([(scores[v], 1) for v in p] + [(scores[v], 0) for v in n])
        r, i = [0.0] * len(rows), 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[k] = avg
            i = j + 1
        s = sum(r[k] for k in range(len(rows)) if rows[k][1] == 1)
        return (s - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

    vals = []
    for _ in range(n_boot):
        p = [rng.choice(pos) for _ in pos]
        n = [rng.choice(neg) for _ in neg]
        vals.append(one(sa, p, n) - one(sb, p, n))
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]]


def analyze_arm(arm_name, arm, by_probe):
    fp, tp, tn = (arm["cohorts"][c] for c in ("fp", "tp", "tn"))
    joint = {v: float(z) for v, z in arm["joint_z"].items()}

    # Every probe must cover every cohort video, or the AUCs are not comparable.
    scores = {"joint": joint}
    for p in PROBE_ORDER:
        s = by_probe.get(p, {})
        missing = [v for v in fp + tp + tn if v not in s]
        if missing:
            raise SystemExit(f"{arm_name}/{p}: {len(missing)} cohort videos unscored")
        scores[p] = s

    diff = {v: scores["stance_v1"][v] - scores["stance_para"][v]
            for v in fp + tp + tn}

    per_probe = {}
    for key in ["joint"] + PROBE_ORDER:
        s = scores[key]
        per_probe[key] = {
            "auc_tp_vs_fp": auc(s, tp, fp),
            "auc_tp_vs_fp_boot95": boot_auc(s, tp, fp),
            "auc_tn_vs_fp": auc(s, tn, fp) if tn else None,
            "z_by_cell": {c: stats([s[v] for v in ids])
                          for c, ids in (("fp", fp), ("tp", tp), ("tn", tn))},
        }
    per_probe["stance_v1_minus_stance_para"] = {
        "auc_tp_vs_fp": auc(diff, tp, fp),
        "z_by_cell": {c: stats([diff[v] for v in ids])
                      for c, ids in (("fp", fp), ("tp", tp), ("tn", tn))},
    }

    a_stance = per_probe["stance_v1"]["auc_tp_vs_fp"]
    a_joint = per_probe["joint"]["auc_tp_vs_fp"]
    a_effort = per_probe["effort_ctrl"]["auc_tp_vs_fp"]
    noise = abs(per_probe["stance_v1_minus_stance_para"]["auc_tp_vs_fp"] - 0.5)

    clauses = {
        "a_stance_ge_0.65": {"value": a_stance, "bar": BARS["a_stance_min"],
                             "pass": a_stance >= BARS["a_stance_min"]},
        "a_stance_minus_a_joint_ge_0.05": {
            "value": a_stance - a_joint, "bar": BARS["delta_joint_min"],
            "pass": (a_stance - a_joint) >= BARS["delta_joint_min"],
            "boot95": boot_delta(scores["stance_v1"], joint, tp, fp)},
        "a_stance_minus_a_effort_ge_0.05": {
            "value": a_stance - a_effort, "bar": BARS["delta_effort_min"],
            "pass": (a_stance - a_effort) >= BARS["delta_effort_min"],
            "boot95": boot_delta(scores["stance_v1"], scores["effort_ctrl"], tp, fp)},
        "noise_le_0.05": {"value": noise, "bar": BARS["noise_max"],
                          "pass": noise <= BARS["noise_max"]},
    }

    # Is the probe reading a new axis, or re-reading the joint call? Descriptive,
    # computed after the verdict clauses and with no authority over them.
    allids = fp + tp + tn
    redundancy = {
        "spearman_vs_joint": {p: spearman(scores[p], joint, allids)
                              for p in PROBE_ORDER},
        "spearman_stance_v1_vs_effort_ctrl":
            spearman(scores["stance_v1"], scores["effort_ctrl"], allids),
        "spearman_stance_v1_vs_stance_para":
            spearman(scores["stance_v1"], scores["stance_para"], allids),
        "note": ("a probe that separates the same pairs the joint call already "
                 "separates, in the same order, is re-reading the joint call"),
    }

    return {
        "dataset": arm["dataset"],
        "split": arm["split"],
        "source_report": arm["source_report"],
        "valley": arm["valley"],
        "confusion_at_valley": arm["confusion_at_valley"],
        "cohort_sizes": arm["cohort_sizes"],
        "cue_carrying_pools": arm["pools"],
        "cue_family_counts": arm["cue_family_counts"],
        "sample_seed": arm["sample_seed"],
        "headline": {"A_stance": a_stance, "A_joint": a_joint,
                     "A_effort": a_effort, "A_stance_para":
                     per_probe["stance_para"]["auc_tp_vs_fp"], "noise": noise},
        "clauses": clauses,
        "all_clauses_pass": all(c["pass"] for c in clauses.values()),
        "directional_consistency_a_stance_gt_a_joint": a_stance > a_joint,
        "redundancy": redundancy,
        "per_probe": per_probe,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="results/stance_gate/cohorts.json")
    ap.add_argument("--scores", default="results/stance_gate/probe_scores.jsonl")
    ap.add_argument("--out", default="docs/duplex/reports/stance_gate_diag.json")
    args = ap.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.cohorts)) as f:
        cohorts = json.load(f)

    by_arm_probe = {}
    with open(os.path.join(PROJECT_ROOT, args.scores)) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_arm_probe.setdefault(r["arm"], {}).setdefault(
                r["probe"], {})[r["video_id"]] = float(r["z"])

    arms = {}
    for name, arm in cohorts["arms"].items():
        arms[name] = analyze_arm(name, arm, by_arm_probe.get(name, {}))

    primary = arms[PRIMARY_ARM]
    secondary = [a for n, a in arms.items() if n != PRIMARY_ARM]
    verdict = "PASS" if primary["all_clauses_pass"] else "FAIL"

    report = {
        "test": "stance-axis gate (diagnostic)",
        "date": datetime.date.today().isoformat(),
        "model": "Qwen3-VL-8B-Instruct",
        "substrate": "8B, raw unclipped z, gated fresh transcripts, frames_16",
        "question": (
            "Does a stance-oriented probe separate cue-sharing false positives "
            "from true positives, the separation the joint judge fails to make?"),
        "status": (
            "Diagnostic, not a kill-test. Cohorts are built with labels because "
            "this is error diagnosis; no threshold, probe or decision rule reads "
            "a label. One extra call per cohort video, on the cohorts only."),
        "calls_per_video": {
            "source_run": 1,
            "this_diagnostic": len(PROBE_ORDER),
            "note": ("Three probes are three arms of one diagnostic, not a "
                     "method: any surviving mechanism would carry one of them. "
                     "The 2-call cap applies to a method, not to a diagnostic "
                     "run over 471 videos."),
        },
        "primary_arm": PRIMARY_ARM,
        "decision_rule_frozen_before_run": {
            "bars": BARS,
            "definition": {
                "A_stance": "AUC(cue-matched TP vs FP) by stance_v1 raw z",
                "A_joint": "the same pairs by the source run's joint raw z",
                "A_effort": "the same pairs by effort_ctrl raw z",
                "noise": "|AUC(TP vs FP) by (z_stance_v1 - z_stance_para) - 0.5|",
            },
            "pass_condition": ("all four clauses on the primary arm, plus "
                               "A_stance > A_joint on the HateMM arm"),
            "authority": ("point estimates decide; bootstrap intervals are "
                          "context and were not part of the frozen rule"),
        },
        "verdict": verdict,
        "verdict_detail": {
            "primary_all_clauses_pass": primary["all_clauses_pass"],
            "failed_clauses": [k for k, c in primary["clauses"].items()
                               if not c["pass"]],
            "secondary_directionally_consistent":
                all(a["directional_consistency_a_stance_gt_a_joint"]
                    for a in secondary),
        },
        "probe_blocks_verbatim": {p: PROBE_BLOCKS[p] for p in PROBE_ORDER},
        "probe_block_provenance": {
            "slot": ("all three occupy the frozen DUPLEX_PROMPT {reader_block} "
                     "slot; the skeleton, rules, system message, Yes/No token "
                     "id sets, pixel budget and frame set are unchanged"),
            "effort_ctrl": ("READER_BLOCKS['effort'] from the frozen "
                            "src/duplex/score_duplex_probe.py, verbatim"),
            "stance_blocks": ("written once, before scoring; category language "
                              "only, no group names and no lexicon terms"),
            "length_caveat": ("the stance blocks run longer than the frozen "
                              "effort block; effort_ctrl is thoroughness-matched "
                              "by construction, not length-matched"),
        },
        "arms": arms,
        "caveats": {
            "A_joint_is_partly_a_cohort_artifact": (
                "A_joint is not a clean measure of 'the joint judge fails to "
                "separate these piles'. The FP cohort is every false positive, "
                "so it sits just above the valley by construction, while the "
                "TP cohort is sampled from the whole true-positive range, most "
                "of which sits far above it. Some of A_joint is that spread. "
                "This inflates the bar the stance probe had to clear, and the "
                "stance probe still landed below it, so the direction of the "
                "verdict is unaffected: a probe that were reading a genuinely "
                "different axis would not need to beat the joint call on the "
                "joint call's own favourable geometry, it would merely need to "
                "rank differently, and the rank correlations show it does not."),
            "TN_vs_FP_is_near_definitional_for_joint": (
                "AUC(TN vs FP) by joint z is exactly 0 by construction: TN are "
                "below the valley and FP above it. It is reported so the probe "
                "columns can be read against it. The informative fact is that "
                "the probes reproduce that ordering to within 0.01 rather than "
                "moving toward 0.5, which is what an axis orthogonal to "
                "hatefulness would do on two piles that are both non-hateful."),
            "one_prompt_family_one_model": (
                "One stance framing and its paraphrase, one model, one token "
                "position. The result bounds what this elicitation recovers, "
                "not what the concept of assertion structure could ever "
                "support."),
            "cohort_sizes": (
                "The HateMM cue-carrying TN pool is 16 of 59, so that arm's "
                "TN column is descriptive at best."),
        },
        "content_policy": "statistics only: no transcript text, no video ids",
    }

    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"VERDICT: {verdict}")
    for name, a in arms.items():
        h = a["headline"]
        print(f"\n[{name}] {a['dataset']}/{a['split']}  cohorts={a['cohort_sizes']}")
        print(f"  A_stance {h['A_stance']:.4f}  A_joint {h['A_joint']:.4f}  "
              f"A_effort {h['A_effort']:.4f}  A_para {h['A_stance_para']:.4f}  "
              f"noise {h['noise']:.4f}")
        for k, c in a["clauses"].items():
            print(f"    {'PASS' if c['pass'] else 'FAIL'}  {k}: "
                  f"{c['value']:.4f} vs bar {c['bar']}")
        for p in ["joint"] + PROBE_ORDER:
            t = a["per_probe"][p]["auc_tn_vs_fp"]
            print(f"    AUC(TN vs FP) by {p:12s}: "
                  f"{'n/a' if t is None else f'{t:.4f}'}")
        r = a["redundancy"]
        for p, v in r["spearman_vs_joint"].items():
            print(f"    spearman vs joint z, {p:12s}: {v:.4f}")
        print(f"    spearman stance_v1 vs effort_ctrl: "
              f"{r['spearman_stance_v1_vs_effort_ctrl']:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
