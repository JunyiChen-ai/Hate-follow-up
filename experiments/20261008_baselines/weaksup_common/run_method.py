#!/usr/bin/env python3
"""One weakly supervised baseline on one corpus: training at three seeds, test scoring, evaluation.

Two protocols (coordinator instruction of 2026-10-08 for the CLIP-based detectors):

  fixed  (VadCLIP, DSANet, AVadCLIP)  Upstream trains on XD-Violence, tests on the test set after every epoch (DSANet:
         every few steps) and keeps the checkpoint with the best TEST frame AP.  That selection is removed.  Here each
         method uses one documented hyper-parameter set on all three corpora, its upstream XD-Violence preset
         (src/xd_option.py; classes 7 -> 2), trains for the preset's fixed 10 epochs and keeps the LAST epoch.  The
         validation split is not read at all (port.py --no-val; AVadCLIP --select last).
  tuned  (MultiHateLoc, reimplemented; no official code)  The protocol of the existing DeHate MultiHateLoc run
         (Retrieval-hate experiments/20260926_dehate_external/launch/baseline.sh): Optuna TPE, 5 completed trials,
         sampler seed 234, trials trained at seed 234, objective = the trial's selected validation video AP (val split,
         video-level labels; search space = scripts/reproduction_baselines/tune_official_val.suggest with its
         MultiHateLoc guard); the winner is retrained per seed and its epoch is selected on validation video AP.
         Trial checkpoints are deleted.

Seeds 2025 / 234 / 3407.  The selected / last checkpoint scores the exact test cohort at 1 fps; the scores are put on
the 4 fps grid (x4, last-value pad), checked for full coverage, and evaluated by src/eval/evaluate_four_datasets.py.
Test labels and frame labels are never read.

Branches (declared before any result): VadCLIP and AVadCLIP score_align (upstream's reported AP branch), with
score_mlp as `<method>_mlp`; DSANet score_mlp (the branch its earlier DeHate run reported), with score_align as
`dsanet_align`; MultiHateLoc score_fused.

    python experiments/20261008_baselines/weaksup_common/run_method.py --method vadclip --corpus hatemm
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "scripts" / "reproduction_baselines"))
import optuna  # noqa: E402
from optuna.trial import TrialState  # noqa: E402
from tune_official_val import option_args, suggest  # noqa: E402

PORT = HERE / "port.py"
AVAD = HERE.parent / "avadclip" / "avadclip_port.py"
CLIP_ROOT = str(C.REPO / ".cache" / "clip")
BRANCHES = {"vadclip": (("score_align", "vadclip"), ("score_mlp", "vadclip_mlp")),
            "avadclip": (("score_align", "avadclip"), ("score_mlp", "avadclip_mlp")),
            "dsanet": (("score_mlp", "dsanet"), ("score_align", "dsanet_align")),
            "multihateloc": (("score_fused", "multihateloc"),)}
FIXED = ("vadclip", "dsanet", "avadclip")
FIXED_NOTE = {"vadclip": "VadCLIP @ c41067f src/xd_option.py (port defaults): embed 512, width 512, 1 head, 1 layer, "
                         "visual-length 256, attn-window 64, prompt 10/10, 10 epochs, batch 96, lr 1e-5, MultiStepLR "
                         "[3, 6, 10] x0.1, loss3 weight 1e-4",
              "dsanet": "DSANet @ eb335b2 src/xd_option.py (port defaults): embed 512, width 512, 1 head, 1 layer, "
                        "visual-length 256, attn-window 64, prompt 10/10, decoder_depth 8, normal_selection_ratio .8, "
                        "DNP on, 16 prototypes, text_adapt_until 1, t_w .6, loss2_weight 5, temp 1, 10 epochs, batch 96, "
                        "lr 1e-5, warm-up 100",
              "avadclip": "AVadCLIP @ d3f6e16 src/xd_option.py (teacher): embed 512, width 512, 1 head, 2 layers, "
                          "visual/audio length 256, attn-window 4, prompt 10/10, 10 epochs, batch 96, lr 1e-5, "
                          "MultiStepLR [3, 6, 10] x0.1, loss3 weight 1e-4"}


def train_cmd(method, corpus, out, values, seed, run_test=False):
    py = sys.executable
    space = "vadclip" if method == "avadclip" else method
    opts = option_args(values, space)
    if method in FIXED:
        opts = ["--select", "last"] + ([] if method == "avadclip" else ["--no-val"])
    common = ["--corpus", corpus, "--device", "cuda", "--seed", str(seed)]
    if method == "multihateloc":
        return [py, str(PORT), "multihateloc.train", *common, "--out-root", str(out), *opts] + \
            (["--run-test"] if run_test else [])
    if method == "avadclip":
        return [py, str(AVAD), "train", *common, "--out-dir", str(out), "--clip-download-root", CLIP_ROOT, *opts]
    return [py, str(PORT), f"{method}.train", *common, "--out-dir", str(out), "--clip-download-root", CLIP_ROOT,
            *opts]


def infer_cmd(method, corpus, out, values):
    py = sys.executable
    if method == "multihateloc":
        return None
    space = "vadclip" if method == "avadclip" else method
    opts = [] if method in FIXED else option_args(values, space)
    # inference needs the architecture options only; optimisation ones are accepted and ignored by the parsers
    if method == "avadclip":
        return [py, str(AVAD), "infer", "--corpus", corpus, "--device", "cuda", "--out-dir", str(out),
                "--model-path", str(out / "model.pth"), "--clip-download-root", CLIP_ROOT, *opts]
    return [py, str(PORT), f"{method}.infer", "--corpus", corpus, "--device", "cuda", "--out-dir", str(out),
            "--split", "test", "--model-path", str(out / "model.pth"), "--clip-download-root", CLIP_ROOT, *opts]


def run(cmd, logfile):
    with open(logfile, "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=C.REPO, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        raise RuntimeError(f"rc={proc.returncode}; see {logfile}")


def metric_file(method, out, corpus):
    return out / corpus / "train_log.json" if method == "multihateloc" else out / "train_meta.json"


def materialize(params):
    values = dict(params)
    temporal = values.pop("temporal", None)
    if temporal:
        length, window = temporal.split(":")
        values.update(visual_length=int(length), attn_window=int(window))
    return values


def tune(method, corpus, root, n_trials, log):
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best.json"
    if best_path.is_file():
        best = json.loads(best_path.read_text())
        if best.get("n_complete", 0) >= n_trials:
            log(f"tuning already complete: trial {best['best_trial']} val video AP {best['best_value']:.4f}")
            return best
    space = "vadclip" if method == "avadclip" else method
    storage = f"sqlite:///{root / 'study.sqlite3'}"
    name = f"{method}-{corpus}"
    study = optuna.create_study(study_name=name, direction="maximize", storage=storage, load_if_exists=True)
    sampler_seed = 234 + len(study.trials)
    study = optuna.load_study(study_name=name, storage=storage,
                              sampler=optuna.samplers.TPESampler(seed=sampler_seed))

    def objective(trial):
        values = suggest(trial, space, corpus)
        if method == "multihateloc" and values["batch_size"] >= 64:
            raise optuna.TrialPruned("batch_size >= 64 exceeds the available GPU memory (tune_official_val guard)")
        if method == "dsanet":
            if values["lr"] * values["loss2_weight"] > 2.5e-4:
                raise optuna.TrialPruned("unstable DSA-Net lr * loss2_weight region (tune_official_val guard)")
            if values["visual_length"] == 256 and (values["batch_size"] == 96 or
                                                   (values["num_prototypes"] == 32 and values["batch_size"] >= 64)):
                raise optuna.TrialPruned("DSA-Net configuration exceeds GPU memory (tune_official_val guard)")
        if any(t.state == TrialState.COMPLETE and t.params == trial.params
               for t in study.trials if t.number != trial.number):
            raise optuna.TrialPruned("duplicate completed parameter set")
        out = root / f"trial_{trial.number:04d}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            run(train_cmd(method, corpus, out, values, 234), out / "train.log")
        finally:
            for p in list(out.glob("model.p*")) + list(out.glob(f"{corpus}/model.p*")):
                p.unlink()
        meta = json.loads(metric_file(method, out, corpus).read_text())
        score = float(meta["selected_val_video_ap"])
        log(f"trial {trial.number}: val video AP {score:.4f} ({time.time() - t0:.0f}s) {values}")
        return score

    def n_complete():
        return sum(t.state == TrialState.COMPLETE for t in study.trials)

    # Pruned attempts (batch guard, duplicates) cost no training; with the MultiHateLoc batch guard about a third of
    # the suggestions are pruned, so the budget is generous. A rerun resumes the same study.
    attempts = n_trials - n_complete() + 20
    while n_complete() < n_trials and attempts > 0:
        before = len(study.trials)
        study.optimize(objective, n_trials=min(n_trials - n_complete(), attempts), gc_after_trial=True,
                       catch=(RuntimeError,))
        used = len(study.trials) - before
        attempts -= used
        if used == 0:
            break
    if n_complete() < n_trials:
        raise RuntimeError(f"only {n_complete()}/{n_trials} trials completed; see {root}/trial_*/train.log")
    best = {"method": method, "corpus": corpus, "n_trials": len(study.trials), "n_complete": n_complete(),
            "n_failed": sum(t.state == TrialState.FAIL for t in study.trials),
            "n_pruned": sum(t.state == TrialState.PRUNED for t in study.trials),
            "sampler_seed": sampler_seed, "objective": "selected validation video AP (val split, video labels)",
            "best_value": study.best_value, "best_params": study.best_params, "best_trial": study.best_trial.number}
    best_path.write_text(json.dumps(best, indent=2) + "\n")
    log(f"tuning done: best trial {best['best_trial']} val video AP {best['best_value']:.4f}")
    return best


def seed_run(method, corpus, seed, best, base, log):
    ds = C.DATASET[corpus]
    out = base / f"seed{seed}"
    metrics = out / "metrics.json"
    if metrics.is_file():
        log(f"seed {seed}: metrics.json exists, kept")
        return json.loads(metrics.read_text())
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    slog = C.RunLog(out / "run.log", mode="w")
    values = materialize(best["best_params"])
    if method in FIXED:
        selection = {"protocol": "fixed", "hyper-parameters": FIXED_NOTE[method],
                     "epoch": "last of the fixed 10 epochs; validation split not read"}
    else:
        selection = {"protocol": "tuned",
                     "hyper-parameters": f"{best['n_complete']}-trial Optuna on val video AP, best trial "
                                         f"{best['best_trial']} ({best['best_value']:.4f})",
                     "epoch": "best val video AP inside the run (val split, video-level labels)"}
    config = {"method": method, "dataset": ds, "corpus_key": corpus, "seed": seed, "params": values,
              "selection": selection,
              "train_val_test": {k: len(C.split_ids(corpus, k)) for k in ("train", "val", "test")},
              "inputs": "data/weaksup_1fps (see PROVENANCE.md)", "branches": BRANCHES[method],
              "code": f"experiments/20261008_baselines/weaksup_common/run_method.py, {C.code_version()}",
              "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    t0 = time.time()
    slog(f"train {method} {ds} seed {seed} params {values}")
    run(train_cmd(method, corpus, out, values, seed, run_test=True), out / "train.log")
    cmd = infer_cmd(method, corpus, out, values)
    if cmd:
        run(cmd, out / "infer.log")
    scores = out / corpus / "scores.jsonl" if method == "multihateloc" else out / "scores.jsonl"
    meta = json.loads(metric_file(method, out, corpus).read_text())
    slog(f"selected epoch {meta.get('selected_epoch')} val video AP {meta.get('selected_val_video_ap')} "
         f"({time.time() - t0:.0f}s)")
    branches, names = zip(*BRANCHES[method])
    rows, coverage = C.build_predictions(corpus, scores, branches, names, seed,
                                         "experiments/20261008_baselines/weaksup_common/run_method.py", slog)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    result = C.evaluate(corpus, rows, out, slog)
    log(f"seed {seed} done")
    return result


def summarize(method, corpus, base, seeds, log):
    ds = C.DATASET[corpus]
    per = {}
    for seed in seeds:
        m = json.loads((base / f"seed{seed}" / "metrics.json").read_text())
        for r in m["per_dataset"]:
            per.setdefault(r["method"], {})[seed] = r
    out = {"dataset": ds, "seeds": list(seeds), "methods": {}}
    for name, rows in per.items():
        entry = {}
        for k in ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC"):
            v = np.array([rows[s][k] for s in seeds], float)
            entry[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "per_seed": dict(zip(map(str, seeds), map(float, v)))}
        out["methods"][name] = entry
        log(f"{name} {ds}: ROC {entry['frame_ROC_AUC']['mean']:.4f}±{entry['frame_ROC_AUC']['sd']:.4f} "
            f"PR {entry['frame_PR_AUC']['mean']:.4f}±{entry['frame_PR_AUC']['sd']:.4f} "
            f"within {entry['within_video_macro_ROC_AUC']['mean']:.4f}±{entry['within_video_macro_ROC_AUC']['sd']:.4f}")
    (base / "summary.json").write_text(json.dumps(out, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(BRANCHES))
    ap.add_argument("--corpus", required=True, choices=C.CORPORA)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    args = ap.parse_args()
    base = C.RUNS / args.method / C.DATASET[args.corpus]
    log = C.RunLog(base / "run.log")
    log(f"start {args.method} {args.corpus} trials={args.trials} seeds={args.seeds} code {C.code_version()}")
    if args.method in FIXED:
        best = {"best_params": {}}
        log(f"fixed protocol: {FIXED_NOTE[args.method]}; last epoch; no validation")
    else:
        best = tune(args.method, args.corpus, base / "tuning", args.trials, log)
    for seed in args.seeds:
        seed_run(args.method, args.corpus, seed, best, base, log)
    summarize(args.method, args.corpus, base, args.seeds, log)
    log("DONE")


if __name__ == "__main__":
    main()
