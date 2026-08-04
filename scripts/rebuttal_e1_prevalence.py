"""E1 --- Prevalence-shift robustness (ARuf-C1). CPU-only, offline recompute.

Question: does the label-free TRIAGE pipeline survive when hateful videos are a
severe minority of the sample set?

Design (transductive prevalence resampling; no new MLLM calls):
  For each dataset and target prevalence pi in {0.05, 0.10, 0.20, natural}:
    - Eval set: keep ALL normal test videos, subsample hateful test videos so
      that pos/(pos+neg) reaches pi (n_pos = int(pi * n_neg / (1 - pi)), >= 1).
      Natural = full test set (sanity row).
    - Calibration pool matches the scenario: EN/ZH/IH fit on their train-split
      scores subsampled to the same pi (train labels used ONLY to construct the
      scenario, never inside the method); HateMM's pool is the subsampled test
      split (TF protocol).
    - Full pipeline per replicate: protocol threshold (TR-Otsu EN / TR-GMM ZH /
      TR-GMM IH / TF-li_lee HM) -> logit 2-GMM posterior -> entropy band
      (mean-entropy rule) -> sequential resolver MAIN_ORDER (g27 > q32 > q72)
      with rho_D re-derived from the subsampled band mean entropy, consuming the
      frozen per-video verifier verdict files.
    - Missing-verdict videos: the resolver advances past that verifier without a
      posterior update (identical to the frozen pipeline); counted separately.
    - Seeds: 30 per (dataset, pi); natural is a single deterministic run.
  Comparators on the IDENTICAL subsampled eval sets: (a) Stage-1 mapper only;
  (b) zero-shot Qwen2.5-VL-72B-AWQ; (c) zero-shot Gemma-3-27B-it. The two
  zero-shot rows use the frozen verifier verdicts as plain single-pass
  classifiers (missing verdict -> normal, as in eval_generative_predictions).

The natural-prevalence TRIAGE row is hard-gated to reproduce the LIVE pipeline
(build_experiments.eval_sequential on the current files) exactly, at tolerance
1e-9; a failure exits nonzero with a diagnostic. Any drift between the live
pipeline and the frozen all_results.json headline is reported as a loud warning
(not a failure), because that is a paper-number provenance issue rather than an
E1 bug --- see scripts/rebuttal_e1_diag.py (HateMM 72B verdicts were rerun
2026-05-05, after the 2026-04-27 freeze, dropping HateMM from 186/215 to
185/215).

Outputs: results/rebuttal/E1_prevalence/{summary.csv, raw.csv, README.md}

Run (via sbatch, CPU-only, env SafetyContradiction):
  python scripts/rebuttal_e1_prevalence.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/EMNLP2")
OUT_DIR = ROOT / "results" / "rebuttal" / "E1_prevalence"

sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))
sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))

from grid_eval_all import DS, N_TEST, judge_path, ld_jsonl, load_labels  # noqa: E402
from data_utils import SKIP_VIDEOS  # noqa: E402
from quick_eval_all import load_scores_file, otsu_threshold, gmm_threshold  # noqa: E402
from thresholds import li_lee_threshold  # noqa: E402
from select_bayes_band import to_logit, fit_gmm  # noqa: E402
from select_entropy_band import _ent as band_ent  # noqa: E402
# Pure helpers, imported so the resolver math is byte-identical to the headline.
from build_experiments import ent, logit, sigmoid, rho_from_hbar, macro_prf  # noqa: E402
import build_experiments as BE  # noqa: E402  (live-pipeline reproduction target)

MAIN_ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")
ZS_72B = "qwen2.5-vl-72b-awq"
ZS_G27 = "gemma-3-27b-it"

# Per-dataset threshold protocol (from src/boundary_rescue/baseline_preds_v2.py).
PROTOCOL = {
    "MHClip_EN": ("otsu", otsu_threshold, "train", "TR-Otsu"),
    "MHClip_ZH": ("gmm", gmm_threshold, "train", "TR-GMM"),
    "HateMM": ("li_lee", li_lee_threshold, "test", "TF-li_lee"),
    "ImpliHateVid": ("gmm", gmm_threshold, "train", "TR-GMM"),
}

PI_LIST = [0.05, 0.10, 0.20]
N_SEEDS = 30
NATURAL = "natural"


def _test_score_path(ds: str) -> Path:
    if ds == "MHClip_ZH":
        return ROOT / "results/holistic_2b/MHClip_ZH/test_binary.jsonl.prerepro_20260413"
    return ROOT / f"results/holistic_2b/{ds}/test_binary.jsonl"


def _train_score_path(ds: str) -> Path:
    return ROOT / f"results/holistic_2b/{ds}/train_binary.jsonl"


def _ordered_scored_videos(path: Path, labels: dict, skip: set) -> list[tuple[str, float]]:
    """Return [(vid, score), ...] in file order, keeping only non-skip videos
    with a known binary label. Mirrors the ordering baseline_preds_v2 uses."""
    out, seen = [], set()
    for r in ld_jsonl(path):
        vid = r.get("video_id")
        s = r.get("score")
        if vid is None or s is None or vid in seen:
            continue
        if vid in skip or labels.get(vid) not in (0, 1):
            continue
        seen.add(vid)
        out.append((vid, float(s)))
    return out


class DatasetData:
    """Frozen, per-dataset inputs loaded once."""

    def __init__(self, ds: str):
        self.ds = ds
        self.labels = load_labels(ds)
        skip = SKIP_VIDEOS.get(ds, set())
        self.test = _ordered_scored_videos(_test_score_path(ds), self.labels, skip)
        self.test_pos = [t for t in self.test if self.labels[t[0]] == 1]
        self.test_neg = [t for t in self.test if self.labels[t[0]] == 0]
        # Sorted candidate pools for deterministic subsampling.
        self.test_pos_sorted = sorted(self.test_pos, key=lambda t: t[0])
        fit_src = PROTOCOL[ds][2]
        if fit_src == "train":
            self.train = _ordered_scored_videos(_train_score_path(ds), self.labels, skip)
            self.train_pos_sorted = sorted(
                [t for t in self.train if self.labels[t[0]] == 1], key=lambda t: t[0]
            )
            self.train_neg = [t for t in self.train if self.labels[t[0]] == 0]
        else:
            self.train = None
        # Frozen verifier verdict tables (video_id -> row).
        self.judges = {}
        for j in set(MAIN_ORDER) | {ZS_72B, ZS_G27}:
            p = judge_path(j, ds)
            self.judges[j] = {r["video_id"]: r for r in ld_jsonl(p)} if (p and p.exists()) else {}


def n_pos_for_pi(n_neg: int, pi: float) -> int:
    """Positives needed so pos/(pos+neg) == pi with neg fixed. Truncated to match
    the plan's stated eval-set sizes (EN 12 / ZH 11 / HM 14 / IH 22 at pi=0.10)."""
    return max(1, int(pi * n_neg / (1.0 - pi)))


def build_eval_and_calib(dd: DatasetData, pi, rng):
    """Return (eval_vids, calib_scores). eval_vids = [(vid, score), ...]."""
    if pi == NATURAL:
        eval_vids = list(dd.test)
        if PROTOCOL[dd.ds][2] == "train":
            calib_scores = np.array([s for _, s in dd.train], dtype=float)
        else:
            calib_scores = np.array([s for _, s in eval_vids], dtype=float)
        return eval_vids, calib_scores

    # Eval set: all normals + subsampled hatefuls.
    k = n_pos_for_pi(len(dd.test_neg), pi)
    k = min(k, len(dd.test_pos_sorted))
    idx = rng.choice(len(dd.test_pos_sorted), size=k, replace=False)
    idx.sort()
    sampled_pos = [dd.test_pos_sorted[i] for i in idx]
    eval_vids = list(dd.test_neg) + sampled_pos

    if PROTOCOL[dd.ds][2] == "train":
        kt = n_pos_for_pi(len(dd.train_neg), pi)
        kt = min(kt, len(dd.train_pos_sorted))
        tidx = rng.choice(len(dd.train_pos_sorted), size=kt, replace=False)
        tidx.sort()
        keep_train_pos = {dd.train_pos_sorted[i][0] for i in tidx}
        calib = [t for t in dd.train if (dd.labels[t[0]] == 0 or t[0] in keep_train_pos)]
        calib_scores = np.array([s for _, s in calib], dtype=float)
    else:
        calib_scores = np.array([s for _, s in eval_vids], dtype=float)
    return eval_vids, calib_scores


def run_scenario(dd: DatasetData, eval_vids, calib_scores):
    """Recompute threshold -> GMM band -> rho_D -> sequential resolver, plus the
    three comparators, on one eval pool. Returns a dict of per-method results and
    band-mechanism diagnostics."""
    ds = dd.ds
    crit_name, crit_fn, fit_src, proto = PROTOCOL[ds]
    thr = float(crit_fn(calib_scores))

    gmm, hi = fit_gmm(calib_scores)
    scores = np.array([s for _, s in eval_vids], dtype=float)
    z = to_logit(scores).reshape(-1, 1)
    post = gmm.predict_proba(z)[:, hi]
    ents = np.array([band_ent(p) for p in post], dtype=float)
    mean_e = float(ents.mean())
    hbar = mean_e
    rho = rho_from_hbar(hbar)
    lam = math.log(rho / (1.0 - rho))
    base_pred = (scores >= thr).astype(int)
    in_band = ents > mean_e

    y = [dd.labels[v] for v, _ in eval_vids]
    n_eval = len(eval_vids)
    judges = [dd.judges[j] for j in MAIN_ORDER]

    # --- TRIAGE sequential resolver (byte-identical loop to eval_sequential) ---
    yh_triage = []
    calls = 0
    n_band = 0
    for i, (vid, _) in enumerate(eval_vids):
        s1 = int(base_pred[i])
        if not in_band[i]:
            yh_triage.append(s1)
            continue
        n_band += 1
        ell = logit(float(post[i]))
        for table in judges[:3]:
            calls += 1
            r = table.get(vid, {}).get("pred")
            if r in (0, 1):
                ell += (2 * int(r) - 1) * lam
                if ent(sigmoid(ell)) <= hbar:
                    break
        yh_triage.append(1 if sigmoid(ell) >= 0.5 else 0)

    # --- Comparators on the identical eval pool ---
    yh_stage1 = [int(base_pred[i]) for i in range(n_eval)]

    def zeroshot(model):
        tab = dd.judges[model]
        out = []
        for vid, _ in eval_vids:
            r = tab.get(vid, {}).get("pred")
            out.append(int(r) if r in (0, 1) else 0)
        return out

    yh_72b = zeroshot(ZS_72B)
    yh_g27 = zeroshot(ZS_G27)

    # Band-mechanism diagnostics.
    s1_errors = sum(1 for i in range(n_eval) if base_pred[i] != y[i])
    band_err_cap = sum(1 for i in range(n_eval) if in_band[i] and base_pred[i] != y[i])

    def metrics(yh):
        acc = sum(1 for a, b in zip(y, yh) if a == b) / n_eval
        mf1, mp, mr = macro_prf(y, yh)
        return acc, mf1, mp, mr

    npos = sum(y)
    common = {
        "ds": ds,
        "n_eval": n_eval,
        "n_pos": npos,
        "n_neg": n_eval - npos,
        "thr": thr,
        "rho_D": rho,
        "hbar": hbar,
        "n_band": n_band,
        "band_rate": n_band / n_eval,
        "error_capture": (band_err_cap / s1_errors) if s1_errors else 0.0,
        "s1_errors": s1_errors,
    }
    methods = {}
    for name, yh, cpv in (
        ("TRIAGE", yh_triage, calls / n_eval),
        ("Stage-1 only", yh_stage1, 0.0),
        ("Zero-shot 72B-AWQ", yh_72b, 1.0),
        ("Zero-shot Gemma-3-27B", yh_g27, 1.0),
    ):
        acc, mf1, mp, mr = metrics(yh)
        methods[name] = {"acc": acc, "mf1": mf1, "mp": mp, "mr": mr, "calls": cpv}
    return common, methods


def reproduction_targets() -> tuple[dict, dict]:
    """Return (live_pipeline, frozen_headline) per-dataset TRIAGE ACC.

    live_pipeline = build_experiments.eval_sequential on the CURRENT files (the
    true reproduction target for this reimplementation). frozen_headline = the
    all_results.json snapshot, which may have drifted from current files (e.g.
    the HateMM 72B verdicts were rerun 2026-05-05, after the 2026-04-27 freeze)."""
    r = BE.eval_sequential(BE.EvalCache(), BE.MAIN_ORDER, rho_mode="entropy")
    live = {d["ds"]: d["acc"] for d in r["per_dataset"]}
    frozen = {}
    src = ROOT / "results" / "paper_stage2_experiments" / "all_results.json"
    data = json.loads(src.read_text())
    for row in data["method_rows"]:
        if row["method"] == "Full sequential rho_D (band, g27>q32>q72)":
            frozen = {d["ds"]: d["acc"] for d in row["per_dataset"]}
    return live, frozen


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {ds: DatasetData(ds) for ds in DS}

    raw_rows = []

    # 1) Natural sanity row. Hard-gate E1 against the LIVE pipeline
    # (build_experiments on current files) at 1e-9 --- this is the true
    # reimplementation-faithfulness check. Separately WARN (never abort) when the
    # live pipeline has drifted from the frozen headline, since that is a paper
    # number provenance issue, not an E1 bug.
    live, frozen = reproduction_targets()
    failures, warnings = [], []
    for ds in DS:
        dd = data[ds]
        common, methods = run_scenario(dd, *build_eval_and_calib(dd, NATURAL, None))
        acc = methods["TRIAGE"]["acc"]
        if common["n_eval"] != N_TEST[ds]:
            failures.append(
                f"{ds}: natural n_eval={common['n_eval']} != N_TEST={N_TEST[ds]}"
            )
        if abs(acc - live[ds]) >= 1e-9:
            failures.append(
                f"{ds}: E1 natural TRIAGE {acc:.10f} != live pipeline "
                f"{live[ds]:.10f} (diff {abs(acc - live[ds]):.2e})"
            )
        if ds in frozen and abs(live[ds] - frozen[ds]) >= 1e-9:
            warnings.append(
                f"{ds}: live pipeline {live[ds]:.6f} != frozen headline "
                f"{frozen[ds]:.6f} (diff {abs(live[ds] - frozen[ds]):.2e})"
            )
        for name, m in methods.items():
            raw_rows.append(
                {"dataset": ds, "pi": "natural", "method": name, "seed": -1, **common, **m}
            )

    if warnings:
        print("[WARN] frozen-headline drift --- paper-number provenance, NOT an E1 bug "
              "(run scripts/rebuttal_e1_diag.py):", file=sys.stderr)
        for w in warnings:
            print("  " + w, file=sys.stderr)
    if failures:
        print("SANITY ROW FAILED --- E1 aborted. Diagnostics:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        sys.exit(1)
    print("[sanity] natural TRIAGE row reproduces the LIVE pipeline "
          "(build_experiments) on all 4 datasets.")

    # 2) Prevalence sweep: 30 seeds per (dataset, pi).
    for ds in DS:
        dd = data[ds]
        for pi in PI_LIST:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                eval_vids, calib = build_eval_and_calib(dd, pi, rng)
                common, methods = run_scenario(dd, eval_vids, calib)
                for name, m in methods.items():
                    raw_rows.append(
                        {"dataset": ds, "pi": f"{pi:.2f}", "method": name,
                         "seed": seed, **common, **m}
                    )

    # 3) Write raw.csv.
    raw_fields = [
        "dataset", "pi", "method", "seed", "n_eval", "n_pos", "n_neg",
        "acc", "mf1", "mp", "mr", "calls",
        "thr", "rho_D", "hbar", "n_band", "band_rate", "error_capture", "s1_errors",
    ]
    with (OUT_DIR / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

    # 4) Aggregate summary (mean +/- std over seeds).
    from collections import defaultdict

    groups = defaultdict(list)
    for r in raw_rows:
        groups[(r["dataset"], r["pi"], r["method"])].append(r)

    agg_metrics = ["acc", "mf1", "mp", "mr", "calls", "band_rate", "error_capture"]
    summary_fields = ["dataset", "pi", "method", "n_seeds"]
    for m in agg_metrics:
        summary_fields += [f"{m}_mean", f"{m}_std"]

    def pi_key(k):
        return 99.0 if k == "natural" else float(k)

    summary_rows = []
    for (ds, pi, method), rows in groups.items():
        out = {"dataset": ds, "pi": pi, "method": method, "n_seeds": len(rows)}
        for m in agg_metrics:
            vals = np.array([r[m] for r in rows], dtype=float)
            out[f"{m}_mean"] = float(vals.mean())
            out[f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        summary_rows.append(out)
    summary_rows.sort(key=lambda r: (r["dataset"], pi_key(r["pi"]), r["method"]))

    with (OUT_DIR / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    _write_readme()
    print(f"Wrote E1 outputs to {OUT_DIR}")
    print(f"  raw rows: {len(raw_rows)}  summary rows: {len(summary_rows)}")


def _write_readme() -> None:
    txt = f"""# E1 --- Prevalence-shift robustness (ARuf-C1)

Transductive prevalence resampling on the frozen offline artifacts. No new MLLM
calls; CPU-only.

## What this measures

Whether the label-free TRIAGE pipeline (Boundary Mapper + Adaptive Boundary
Resolver) still helps when hateful videos are a severe minority of the evaluated
pool. For each dataset we sweep the eval-set prevalence
pi in {{0.05, 0.10, 0.20, natural}}.

## Method (per replicate)

1. Eval set: keep ALL normal test videos, subsample hateful test videos so that
   pos/(pos+neg) = pi, with n_pos = int(pi * n_neg / (1-pi)), at least 1 positive.
   ZH respects SKIP_VIDEOS. Natural = the full test set.
2. Calibration pool matched to the scenario:
   - EN / ZH / IH: train-split scores subsampled to the same pi (train labels are
     used ONLY to build the scenario, never inside the method).
   - HateMM: the subsampled test pool itself (test-fit protocol).
3. Re-fit the protocol threshold on the calibration pool
   (TR-Otsu EN / TR-GMM ZH / TR-GMM IH / TF-li_lee HM).
4. Fit a 2-component GMM in logit space on the calibration pool; per eval video
   compute posterior_hi and binary entropy; the entropy band is entropy > mean
   entropy over the eval pool; rho_D = rho_from_hbar(mean entropy).
5. Sequential resolver over MAIN_ORDER (gemma-3-27b-it > qwen2.5-vl-32b-awq >
   qwen2.5-vl-72b-awq), consuming the frozen per-video verdict files, with the
   calibrated Bayesian posterior update and entropy early-stop. A video missing a
   verifier verdict advances past that verifier without a posterior update
   (identical to the frozen headline pipeline).

Comparators on the identical eval pool: Stage-1 mapper only; zero-shot
Qwen2.5-VL-72B-AWQ; zero-shot Gemma-3-27B-it. The zero-shot rows use the frozen
verifier verdicts as plain single-pass classifiers; a video with no verdict is
scored normal (0), matching eval_generative_predictions' unparseable fallback.

## Seeds and reproduction gate

30 seeds per (dataset, pi) via numpy default_rng(seed); natural is a single
deterministic run. The natural TRIAGE row is hard-gated to reproduce the live
pipeline (build_experiments.eval_sequential on current files) at tol 1e-9; on
failure the script exits nonzero. Drift between the live pipeline and the frozen
all_results.json headline is reported as a warning (paper-number provenance, not
an E1 bug): HateMM currently yields 185/215 (0.86047) vs the frozen 186/215
(0.86512) because its 72B verdicts were rerun after the headline was frozen.

## Eval-set sizes at pi=0.10

EN 112 normal + 12 hateful; ZH 104 + 11; HM 129 + 14; IH 201 + 22.

## Files

- summary.csv: per (dataset, pi, method) mean and std over seeds for
  acc, macro-F1, macro-P, macro-R, calls/video, band rate, error capture.
- raw.csv: one row per (dataset, pi, method, seed), including threshold, rho_D,
  mean entropy (hbar), band size, band rate, and error-capture rate.

## Rerun

    conda activate SafetyContradiction
    python scripts/rebuttal_e1_prevalence.py
"""
    (OUT_DIR / "README.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
