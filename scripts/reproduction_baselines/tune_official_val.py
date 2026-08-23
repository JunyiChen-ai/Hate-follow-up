#!/usr/bin/env python3
"""Validation-only Optuna search for official-val WS-VAD ports.

Each trial is an isolated subprocess. Test inference is never requested. Trial
checkpoints are deleted after their validation metric is recorded; metadata,
stdout/stderr and the Optuna SQLite study remain, and the selected settings are
retrained from scratch in the confirmation/final-seed stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import optuna

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PYTHON = "/home/jehc223/miniconda3/envs/HateVideo/bin/python"


def temporal(trial, corpus):
    choices = (["64:8", "64:16", "128:16", "128:32"]
               if corpus.startswith("mhclip") else
               ["128:16", "128:32", "256:32", "256:64"])
    length, window = trial.suggest_categorical("temporal", choices).split(":")
    return int(length), int(window)


def suggest(trial, method, corpus):
    length, window = temporal(trial, corpus)
    common = {"lr": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 96]),
              "max_epoch": trial.suggest_categorical("max_epoch", [10, 20, 30, 50]),
              "visual_length": length, "attn_window": window}
    if method == "vadclip":
        common.update(prompt_prefix=trial.suggest_categorical("prompt_prefix", [5, 10, 20]),
                      prompt_postfix=trial.suggest_categorical("prompt_postfix", [5, 10, 20]),
                      loss3_weight=trial.suggest_float("loss3_weight", 1e-5, 1e-2, log=True))
    elif method == "dsanet":
        common.update(num_prototypes=trial.suggest_categorical("num_prototypes", [8, 16, 32]),
                      normal_selection_ratio=trial.suggest_float("normal_selection_ratio", .6, .9),
                      t_w=trial.suggest_float("t_w", .1, .9),
                      temp=trial.suggest_categorical("temp", [.5, 1., 2., 5.]),
                      loss2_weight=trial.suggest_float("loss2_weight", .5, 8., log=True))
    elif method == "cmhkf":
        common.update(loss_mil=trial.suggest_float("loss_mil", .25, 4., log=True),
                      loss_align=trial.suggest_float("loss_align", .25, 4., log=True),
                      loss_text=trial.suggest_float("loss_text", 1e-5, 1e-2, log=True),
                      prompt_prefix=trial.suggest_categorical("prompt_prefix", [5, 10, 20]),
                      prompt_postfix=trial.suggest_categorical("prompt_postfix", [5, 10, 20]))
    elif method.startswith("fed_wsvad"):
        common.pop("max_epoch")
        common.update(global_rounds=trial.suggest_categorical("global_rounds", [10, 20, 30]),
                      local_epochs=trial.suggest_categorical("local_epochs", [2, 5, 10]),
                      visual_layers=trial.suggest_categorical("visual_layers", [1, 2]),
                      prompt_prefix=trial.suggest_categorical("prompt_prefix", [5, 10, 20]),
                      prompt_postfix=trial.suggest_categorical("prompt_postfix", [5, 10, 20]))
    elif method.startswith("macilsd"):
        common = {"lr": trial.suggest_float("lr", 2e-5, 1e-3, log=True),
                  "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                  "max_epoch": trial.suggest_categorical("max_epoch", [20, 30, 50]),
                  "max_seqlen": trial.suggest_categorical("max_seqlen", [100, 150, 200]),
                  "dropout": trial.suggest_categorical("dropout", [.05, .1, .2]),
                  "lamda_a2b": trial.suggest_float("lamda_a2b", .5, 3., log=True),
                  "lamda_a2n": trial.suggest_float("lamda_a2n", .5, 3., log=True),
                  "lamda_cof": trial.suggest_float("lamda_cof", .03, .3, log=True)}
    elif method == "multihateloc":
        common = {"lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
                  "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                  "max_epoch": trial.suggest_categorical("max_epoch", [30, 50, 100]),
                  "k_proportion": trial.suggest_categorical("k_proportion", [2, 3, 5, 8]),
                  "lambda_smooth": trial.suggest_float("lambda_smooth", .01, .5, log=True),
                  "lambda_contrast": trial.suggest_float("lambda_contrast", .02, 1., log=True),
                  "hidden": trial.suggest_categorical("hidden", [128, 256, 512]),
                  "embed": trial.suggest_categorical("embed", [64, 128, 256]),
                  "dropout": trial.suggest_categorical("dropout", [.05, .1, .2]),
                  "temperature": trial.suggest_categorical("temperature", [.03, .07, .1])}
    return common


def option_args(values, method):
    out = []
    underscore = {"decoder_depth", "normal_selection_ratio", "DNP_use",
                  "num_prototypes", "text_adapt_until", "t_w",
                  "loss2_weight"} if method == "dsanet" else set()
    for key, value in values.items():
        flag = key if key in underscore else key.replace("_", "-")
        out.extend(["--" + flag, str(value)])
    return out


def command(method, corpus, out, values, python):
    if method == "vadclip":
        script = HERE / "train_vadclip_hatemm.py"
    elif method == "dsanet":
        script = HERE / "train_dsanet_hatemm.py"
    elif method == "cmhkf":
        script = HERE / "cmhkf_adapter.py"
    elif method.startswith("fed_wsvad"):
        script = HERE / "fed_wsvad_adapter.py"
    elif method.startswith("macilsd"):
        script = HERE / "train_macilsd_hatemm.py"
    elif method == "multihateloc":
        script = HERE / "train_multihateloc.py"
    else:
        raise ValueError(method)
    out_flag = "--out-root" if method == "multihateloc" else "--out-dir"
    cmd = [python, str(script), "--corpus", corpus, out_flag, str(out),
           "--device", "cuda", "--seed", "234"] + option_args(values, method)
    if method == "fed_wsvad_1client": cmd += ["--clients", "1"]
    if method == "fed_wsvad_3client": cmd += ["--clients", "3", "--partition-seed", "234"]
    if method.startswith("macilsd"):
        modality = "av" if method == "macilsd" else method.removeprefix("macilsd_")
        cmd += ["--modality", modality]
    return cmd


def metric_path(method, out, corpus):
    return (out / corpus / "train_log.json" if method == "multihateloc"
            else out / "train_meta.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=("vadclip", "dsanet", "cmhkf",
                             "fed_wsvad_1client", "fed_wsvad_3client",
                             "macilsd", "macilsd_audio", "macilsd_visual",
                             "multihateloc"))
    ap.add_argument("--corpus", required=True,
                    choices=("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg"))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--root", default=str(REPO / "results" / "reproduction" /
                                            "official_val" / "tuning"))
    args = ap.parse_args(argv)
    root = Path(args.root) / args.method / args.corpus
    root.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{args.method}-{args.corpus}", direction="maximize",
        storage=f"sqlite:///{root / 'study.sqlite3'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=234))

    def objective(trial):
        values = suggest(trial, args.method, args.corpus)
        out = root / f"trial_{trial.number:04d}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = command(args.method, args.corpus, out, values, args.python)
        proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
        (out / "stdout.log").write_text(proc.stdout)
        (out / "stderr.log").write_text(proc.stderr)
        (out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        if proc.returncode != 0:
            raise RuntimeError(f"trial rc={proc.returncode}; see {out}")
        meta = json.loads(metric_path(args.method, out, args.corpus).read_text())
        score = float(meta["selected_val_video_ap"])
        for name in ("model.pth", "model.pt", "checkpoint.pth"):
            path = out / name
            if path.is_file(): path.unlink()
        return score

    study.optimize(objective, n_trials=max(0, args.trials - len(study.trials)),
                   gc_after_trial=True, catch=(RuntimeError,))
    summary = {"method": args.method, "corpus": args.corpus,
               "n_trials": len(study.trials), "best_value": study.best_value,
               "best_params": study.best_params,
               "best_trial": study.best_trial.number}
    (root / "best.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
