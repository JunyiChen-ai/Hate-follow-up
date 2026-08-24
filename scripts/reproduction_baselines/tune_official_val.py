#!/usr/bin/env python3
"""Validation-only Optuna search for official-val WS-VAD ports.

Each trial is an isolated subprocess. Test inference is never requested. Trial
checkpoints are deleted after their validation metric is recorded; metadata,
stdout/stderr and the Optuna SQLite study remain, and the selected settings are
retrained from scratch in the confirmation/final-seed stage.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import optuna
from optuna.trial import TrialState

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PYTHON = "/home/jehc223/miniconda3/envs/HateVideo/bin/python"


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def temporal(trial, corpus):
    choices = (["64:8", "64:16", "128:16", "128:32"]
               if corpus.startswith("mhclip") else
               ["128:16", "128:32", "256:32", "256:64"])
    length, window = trial.suggest_categorical("temporal", choices).split(":")
    return int(length), int(window)


def suggest(trial, method, corpus):
    # These adapters have method-specific spaces and do not consume the
    # CLIP-port temporal arguments.  Build them before the shared space so an
    # Optuna trial never sees the same parameter name with two distributions.
    if method.startswith("macilsd"):
        return {"lr": trial.suggest_float("lr", 2e-5, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "max_epoch": trial.suggest_categorical("max_epoch", [20, 30, 50]),
                "max_seqlen": trial.suggest_categorical("max_seqlen", [100, 150, 200]),
                "dropout": trial.suggest_categorical("dropout", [.05, .1, .2]),
                "lamda_a2b": trial.suggest_float("lamda_a2b", .5, 3., log=True),
                "lamda_a2n": trial.suggest_float("lamda_a2n", .5, 3., log=True),
                "lamda_cof": trial.suggest_float("lamda_cof", .03, .3, log=True)}
    if method == "multihateloc":
        return {"lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "max_epoch": trial.suggest_categorical("max_epoch", [30, 50, 100]),
                "k_proportion": trial.suggest_categorical("k_proportion", [2, 3, 5, 8]),
                "lambda_smooth": trial.suggest_float("lambda_smooth", .01, .5, log=True),
                "lambda_contrast": trial.suggest_float("lambda_contrast", .02, 1., log=True),
                "hidden": trial.suggest_categorical("hidden", [128, 256, 512]),
                "embed": trial.suggest_categorical("embed", [64, 128, 256]),
                "dropout": trial.suggest_categorical("dropout", [.05, .1, .2]),
                "temperature": trial.suggest_categorical("temperature", [.03, .07, .1])}

    # CMHKF's fusion block and Fed-WSVAD's prompt visual projection are fixed
    # at 256 visual tokens.  The shorter MHClip search space used by the other
    # CLIP ports would make every such trial structurally invalid.
    if method.startswith("fed_wsvad") or (
            method == "cmhkf" and corpus.startswith("mhclip")):
        choice = trial.suggest_categorical("temporal", ["256:32", "256:64"])
        length, window = map(int, choice.split(":"))
    else:
        length, window = temporal(trial, corpus)
    # Preserve HateMM's persisted CMHKF distributions, but start new corpus
    # studies directly in the empirically viable region.  Changing a
    # categorical distribution inside an existing Optuna study is forbidden.
    constrained_cmhkf = method == "cmhkf" and corpus != "hatemm"
    batch_choices = [32] if constrained_cmhkf else [16, 32, 64, 96]
    epoch_choices = [10] if constrained_cmhkf else [10, 20, 30, 50]
    common = {"lr": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
              "batch_size": trial.suggest_categorical("batch_size", batch_choices),
              "visual_length": length, "attn_window": window}
    if not method.startswith("fed_wsvad"):
        common["max_epoch"] = trial.suggest_categorical(
            "max_epoch", epoch_choices)
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
        common.update(global_rounds=trial.suggest_categorical("global_rounds", [10, 20, 30]),
                      local_epochs=trial.suggest_categorical("local_epochs", [2, 5, 10]),
                      visual_layers=trial.suggest_categorical("visual_layers", [1, 2]),
                      prompt_prefix=trial.suggest_categorical("prompt_prefix", [5, 10, 20]),
                      prompt_postfix=trial.suggest_categorical("prompt_postfix", [5, 10, 20]))
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
    ap.add_argument(
        "--max-new-attempts", type=int, default=None,
        help=("maximum subprocess attempts in this invocation; defaults to "
              "a method-aware multiple of remaining COMPLETE trials so "
              "failures are replaced without an unbounded loop"))
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--root", default=str(REPO / "results" / "reproduction" /
                                            "official_val" / "tuning"))
    args = ap.parse_args(argv)
    root = Path(args.root) / args.method / args.corpus
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / ".tuner.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"another tuner owns {args.method}/{args.corpus}: "
            f"{root / '.tuner.lock'}")

    def remove_tuning_checkpoints(trial_root):
        paths = [parent / name
                 for parent in (trial_root, trial_root / args.corpus)
                 for name in ("model.pth", "model.pt", "checkpoint.pth")]
        for path in paths:
            if path.is_file():
                path.unlink()

    # Recover the narrow interruption window after an adapter saves its model
    # but before objective() removes it.  Only this study's trial directories
    # are touched; final seed checkpoints live under a different root.
    for trial_root in root.glob("trial_*"):
        if trial_root.is_dir():
            remove_tuning_checkpoints(trial_root)

    storage = f"sqlite:///{root / 'study.sqlite3'}"
    # Optuna persists trials but not the sampler RNG state.  Re-seed from the
    # persisted attempt count so a resumed process does not restart the exact
    # same deterministic suggestion sequence.
    study = optuna.create_study(
        study_name=f"{args.method}-{args.corpus}", direction="maximize",
        storage=storage, load_if_exists=True)
    sampler_seed = 234 + len(study.trials)
    study = optuna.load_study(
        study_name=f"{args.method}-{args.corpus}", storage=storage,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed))

    def objective(trial):
        values = suggest(trial, args.method, args.corpus)
        if args.method == "multihateloc" and values["batch_size"] >= 64:
            # MultiHateLoc's pairwise temporal contrastive objective retains
            # substantially more activations than the other adapters.  On the
            # shared 32 GiB GPU (about 20 GiB is occupied by another process),
            # batch 64 consistently OOMs during backward even for the smallest
            # hidden size.  Keep the persisted categorical distribution stable
            # for Optuna resumes, but prune this empirically infeasible region
            # before launching a subprocess.
            raise optuna.TrialPruned(
                "batch_size >= 64 exceeds the available GPU memory")
        if args.method == "cmhkf":
            if values["visual_length"] != 256:
                # The upstream multimodal fusion block has a fixed 256-step
                # affine dimension and fails for the advertised 128-step
                # variant before the first update.
                raise optuna.TrialPruned(
                    "CMHKF upstream fusion requires visual_length=256")
            if values["batch_size"] != 32:
                # Full 10-epoch probes take roughly 40 minutes at batches 16,
                # 32, and 96: the upstream fusion compute, not update count,
                # dominates.  Batch 96 also reduced validation AP from .8125
                # to .7408, while batch 16 is computationally infeasible.
                raise optuna.TrialPruned(
                    "CMHKF batch_size=32 is the only viable setting")
            if values["max_epoch"] > 10:
                # Two full 10-epoch feasibility runs selected epoch 4 and
                # degraded thereafter, while each run required about 40
                # minutes.  Retain validation-based checkpoint selection but
                # avoid 3--5x longer configurations unsupported by validation.
                raise optuna.TrialPruned(
                    "CMHKF validation peaks before epoch 10")
        if (args.method.startswith("fed_wsvad") and
                values["visual_length"] != 256):
            # Upstream PromptLearner's visual projection is constructed with
            # a fixed 256-wide temporal input and fails on any other length.
            raise optuna.TrialPruned(
                "Fed-WSVAD prompt network requires visual_length=256")
        if args.method == "dsanet":
            # Empirical feasibility guards from the shared 32 GiB GPU.  These
            # combinations either overflow memory before the first update or
            # drive the weighted alignment gradient non-finite within a few
            # epochs.  Prune them before launching an expensive subprocess;
            # viable batch-64 configurations (including the current best)
            # remain in the search space.
            effective_alignment_lr = values["lr"] * values["loss2_weight"]
            if effective_alignment_lr > 2.5e-4:
                raise optuna.TrialPruned(
                    "unstable DSA-Net lr * loss2_weight region")
            if (values["visual_length"] == 256 and
                    (values["batch_size"] == 96 or
                     (values["num_prototypes"] == 32 and
                      values["batch_size"] >= 64))):
                raise optuna.TrialPruned(
                    "DSA-Net configuration exceeds shared-GPU memory")
        if any(t.state == TrialState.COMPLETE and t.params == trial.params
               for t in study.trials if t.number != trial.number):
            raise optuna.TrialPruned("duplicate completed parameter set")
        out = root / f"trial_{trial.number:04d}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = command(args.method, args.corpus, out, values, args.python)
        proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
        (out / "stdout.log").write_text(proc.stdout)
        (out / "stderr.log").write_text(proc.stderr)
        (out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        # A tuning checkpoint is never selected directly: the winner is
        # retrained from its archived parameters for each final seed.  Remove
        # checkpoints from successful and failed subprocesses alike, including
        # MultiHateLoc's corpus-nested output layout.
        remove_tuning_checkpoints(out)
        if proc.returncode != 0:
            raise RuntimeError(f"trial rc={proc.returncode}; see {out}")
        meta = json.loads(metric_path(args.method, out, args.corpus).read_text())
        score = float(meta["selected_val_video_ap"])
        return score

    def n_complete():
        return sum(t.state == TrialState.COMPLETE for t in study.trials)

    # A failed subprocess is evidence, but not a successfully evaluated
    # hyperparameter configuration.  Resume toward the requested number of
    # completed validation trials rather than counting FAIL states as done.
    if args.max_new_attempts is not None:
        attempts_left = args.max_new_attempts
    else:
        remaining = args.trials - n_complete()
        # HateMM's persisted CMHKF study predates the feasibility guards, so
        # its categorical distributions must retain many choices that are now
        # pruned immediately.  Budget attempts by remaining successful trials
        # to prevent a healthy resume from stopping before 40 COMPLETE runs.
        multiplier = 8 if args.method == "cmhkf" and args.corpus == "hatemm" else 2
        attempts_left = max(multiplier * remaining, remaining + 5)
    while n_complete() < args.trials and attempts_left > 0:
        before = len(study.trials)
        batch = min(args.trials - n_complete(), attempts_left)
        study.optimize(objective, n_trials=batch, gc_after_trial=True,
                       catch=(RuntimeError,))
        used = len(study.trials) - before
        attempts_left -= used
        if used == 0:
            break

    complete = n_complete()
    failed = sum(t.state == TrialState.FAIL for t in study.trials)
    pruned = sum(t.state == TrialState.PRUNED for t in study.trials)
    if complete < args.trials:
        raise RuntimeError(
            f"only {complete}/{args.trials} validation trials completed "
            f"({failed} failed); inspect {root}/trial_*/stderr.log")
    summary = {"method": args.method, "corpus": args.corpus,
               "n_trials": len(study.trials), "n_complete": complete,
               "n_failed": failed, "n_pruned": pruned,
               "sampler_seed": sampler_seed, "best_value": study.best_value,
               "best_params": study.best_params,
               "best_trial": study.best_trial.number}
    atomic_write(root / "best.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
