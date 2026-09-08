#!/usr/bin/env python
"""Stage 4 of the text-region pixel-budget probe: the pre-registered readout.

Primary metric is ROC-AUC per arm against BASELINE; the secondary is macro-F1
at each arm's own label-free KDE-valley threshold. The three frozen clauses of
`docs/duplex/PREREG_text_region_budget_probe.md` are evaluated verbatim.

The AUC estimator, the KDE-valley recipe and the operating-point arithmetic are
imported from `scripts/duplex/crossbench_analyze.py`, the module that produced
the BASELINE test report, so no statistic is redefined here.

Output: results/text_region_budget/results.json. Statistics only: no video ids,
no transcript text, no recognised strings.
"""

import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from crossbench_analyze import (  # noqa: E402
    LABEL_MAP, auc_boot, auc_np, kde_valley, load_z, operating_point,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

SPEECH_SPLIT_CHARS = 318     # frozen A2 median, MHClip-EN
CLAUSE1_MIN = 0.03
CLAUSE2_MIN = 0.02
ARMS = ("text", "rand", "anti")


def auc_of(z, pos, neg):
    if not pos or not neg:
        return None
    return round(auc_np(np.array([z[v] for v in pos]),
                        np.array([z[v] for v in neg])), 6)


def arm_block(z, ids, lab, tag):
    pos = [v for v in ids if lab[v] == 1]
    neg = [v for v in ids if lab[v] == 0]
    if not pos or not neg:
        return {"stratum": tag, "n": len(ids), "n_pos": len(pos), "n_neg": len(neg),
                "auc": None}
    zp = np.array([z[v] for v in pos])
    zn = np.array([z[v] for v in neg])
    return {"stratum": tag, "n": len(ids), "n_pos": len(pos), "n_neg": len(neg),
            "auc": round(auc_np(zp, zn), 6),
            "auc_boot95": [round(x, 6) for x in auc_boot(zp, zn)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHClip_EN")
    ap.add_argument("--split", default="test")
    ap.add_argument("--baseline-scores", default=os.path.join(
        ROOT, "results/testruns/mhclip_en/judge_8b/scores.jsonl"))
    ap.add_argument("--work-dir", default=os.path.join(
        ROOT, "results/testruns/mhclip_en"))
    ap.add_argument("--arm-dir", default=os.path.join(
        ROOT, "results/text_region_budget"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/text_region_budget/results.json"))
    args = ap.parse_args()

    lmap = LABEL_MAP[args.dataset]
    ann = load_annotations(args.dataset)
    split_ids = load_clean_split_ids(args.dataset, args.split)
    seen, ids = set(), []
    for v in split_ids:
        if v not in seen:
            seen.add(v)
            ids.append(v)
    lab = {v: lmap[ann[v]["label"]] for v in ids}

    with open(os.path.join(args.work_dir, "c2_overrides.json")) as f:
        overrides = json.load(f)
    fresh = {}
    fp = os.path.join(args.work_dir, "fresh_transcripts.jsonl")
    if os.path.exists(fp):
        with open(fp) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    fresh[r["video_id"]] = r

    # Speech availability = the number of characters the judge actually read.
    # For the gated-fresh majority that is the fresh Whisper transcript, which
    # is the quantity the frozen 318-character median was computed on.
    seen_chars = {v: len(overrides[v] if v in overrides
                         else (ann[v]["transcript"] or "")) for v in ids}
    poor = [v for v in ids if seen_chars[v] <= SPEECH_SPLIT_CHARS]
    rich = [v for v in ids if seen_chars[v] > SPEECH_SPLIT_CHARS]

    with open(os.path.join(args.arm_dir, "accounting.json")) as f:
        acct = json.load(f)
    notext = [v for v in ids if acct["per_video"][v]["no_text"]]
    withtext = [v for v in ids if not acct["per_video"][v]["no_text"]]

    z = {"baseline": load_z(args.baseline_scores)}
    for a in ARMS:
        z[a] = load_z(os.path.join(args.arm_dir, f"judge_8b_{a}", "scores.jsonl"))
    for name, d in z.items():
        missing = [v for v in ids if v not in d]
        if missing:
            raise SystemExit(f"ABORT: arm {name} missing {len(missing)} of {len(ids)}")

    out = {
        "probe": "text-region pixel-budget probe, MHClip-EN test_clean",
        "prereg": "docs/duplex/PREREG_text_region_budget_probe.md",
        "judge": "Qwen3-VL-8B-Instruct, bf16, single call, raw z, prompt and "
                 "transcript inputs identical to the BASELINE test run; the arms "
                 "differ only in image content",
        "n_videos": len(ids),
        "label_mapping": lmap,
        "strata": {
            "speech_split_chars": SPEECH_SPLIT_CHARS,
            "n_speech_poor": len(poor), "n_speech_rich": len(rich),
            "n_no_text": len(notext), "n_with_text": len(withtext),
            "no_text_rule": "no frame whose union OCR-box area reaches 1 % of "
                            "the frame, the A2 census substantive-frame threshold",
        },
        "token_accounting": acct["summary"],
    }

    # ---------- determinism check on the no-text stratum ----------
    if notext:
        dz = {a: max(abs(z[a][v] - z["baseline"][v]) for v in notext) for a in ARMS}
        out["no_text_determinism_check"] = {
            "role": "no-text videos are rendered identically in every arm, so "
                    "their raw z must reproduce the BASELINE value",
            "max_abs_delta_vs_baseline": {a: round(dz[a], 8) for a in ARMS},
            "identical": {a: bool(dz[a] == 0.0) for a in ARMS},
        }

    # ---------- primary: AUC ----------
    per_arm = {}
    for name in ("baseline",) + ARMS:
        zz = z[name]
        blocks = {
            "overall": arm_block(zz, ids, lab, "overall"),
            "speech_poor": arm_block(zz, poor, lab, f"seen chars <= {SPEECH_SPLIT_CHARS}"),
            "speech_rich": arm_block(zz, rich, lab, f"seen chars > {SPEECH_SPLIT_CHARS}"),
            "with_text": arm_block(zz, withtext, lab, "text detected"),
            "no_text": arm_block(zz, notext, lab, "no text detected"),
        }
        v = kde_valley([zz[x] for x in ids])
        pos = [x for x in ids if lab[x] == 1]
        neg = [x for x in ids if lab[x] == 0]
        op = (operating_point(zz, pos, neg, v["value"])
              if v["value"] is not None else None)
        per_arm[name] = {
            "auc": blocks,
            "threshold_label_free": {"value": v["value"],
                                     "n_grid_local_maxima": v["n_grid_local_maxima"]},
            "operating_point_at_valley": op,
            "macro_f1_at_valley": round(op["macro_f1"], 6) if op else None,
            "z_mean": round(float(np.mean([zz[x] for x in ids])), 4),
            "z_median": round(float(np.median([zz[x] for x in ids])), 4),
        }
    out["arms"] = per_arm

    def d(a, b, key):
        xa = per_arm[a]["auc"][key]["auc"]
        xb = per_arm[b]["auc"][key]["auc"]
        if xa is None or xb is None:
            return None
        return round(xa - xb, 6)

    out["deltas_auc"] = {
        key: {f"{a}_minus_baseline": d(a, "baseline", key) for a in ARMS}
        for key in ("overall", "speech_poor", "speech_rich", "with_text", "no_text")
    }
    out["deltas_auc"]["overall"]["text_minus_rand"] = d("text", "rand", "overall")
    out["deltas_auc"]["overall"]["text_minus_anti"] = d("text", "anti", "overall")

    c1 = out["deltas_auc"]["overall"]["text_minus_baseline"]
    c2a = out["deltas_auc"]["overall"]["text_minus_rand"]
    c2b = out["deltas_auc"]["overall"]["text_minus_anti"]
    c3p = out["deltas_auc"]["speech_poor"]["text_minus_baseline"]
    c3r = out["deltas_auc"]["speech_rich"]["text_minus_baseline"]
    clauses = {
        "clause_1_text_over_baseline": {
            "rule": f"AUC(TEXT) - AUC(BASELINE) >= +{CLAUSE1_MIN}",
            "observed": c1, "pass": bool(c1 is not None and c1 >= CLAUSE1_MIN)},
        "clause_2_targeting_is_load_bearing": {
            "rule": f"AUC(TEXT) - AUC(RAND) >= +{CLAUSE2_MIN} and "
                    f"AUC(TEXT) - AUC(ANTI) >= +{CLAUSE2_MIN}",
            "observed_vs_rand": c2a, "observed_vs_anti": c2b,
            "pass": bool(c2a is not None and c2b is not None
                         and c2a >= CLAUSE2_MIN and c2b >= CLAUSE2_MIN)},
        "clause_3_gain_concentrates_in_speech_poor_half": {
            "rule": "speech-poor dAUC(TEXT-BASELINE) > speech-rich dAUC",
            "observed_speech_poor": c3p, "observed_speech_rich": c3r,
            "pass": bool(c3p is not None and c3r is not None and c3p > c3r)},
    }
    clauses["VERDICT"] = ("PASS" if all(c["pass"] for k, c in clauses.items()
                                        if k != "VERDICT") else "FAIL")
    out["frozen_decision_rule"] = clauses

    out["macro_f1_secondary"] = {
        a: per_arm[a]["macro_f1_at_valley"] for a in ("baseline",) + ARMS}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({"verdict": clauses["VERDICT"],
                      "auc": {a: per_arm[a]["auc"]["overall"]["auc"]
                              for a in ("baseline",) + ARMS},
                      "clauses": {k: v.get("pass") for k, v in clauses.items()
                                  if k != "VERDICT"},
                      "macro_f1": out["macro_f1_secondary"]}, indent=1))


if __name__ == "__main__":
    main()
