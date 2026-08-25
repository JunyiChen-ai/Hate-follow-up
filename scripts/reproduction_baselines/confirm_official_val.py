#!/usr/bin/env python3
"""Retrain the frozen Optuna winner on three seeds, then touch test once."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

from tune_official_val import DEFAULT_PYTHON, HERE, REPO, option_args

SEEDS = (234, 2025, 3407)


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_code_commit(expected):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, text=True).strip()
    if head != expected or dirty:
        raise RuntimeError(
            f"confirmation code changed: expected {expected}, head={head}, "
            f"tracked_dirty={bool(dirty)}")


def materialize(best):
    values = dict(best)
    temporal = values.pop("temporal", None)
    if temporal:
        length, window = temporal.split(":")
        values.update(visual_length=int(length), attn_window=int(window))
    return values


def train_command(method, corpus, out, values, seed, python):
    common = ["--corpus", corpus, "--device", "cuda", "--seed", str(seed)]
    if method == "vadclip":
        return [python, str(HERE / "train_vadclip_hatemm.py"), *common,
                "--out-dir", str(out), *option_args(values, method)]
    if method == "dsanet":
        return [python, str(HERE / "train_dsanet_hatemm.py"), *common,
                "--out-dir", str(out), *option_args(values, method)]
    if method.startswith("macilsd"):
        modality = "av" if method == "macilsd" else method.removeprefix("macilsd_")
        return [python, str(HERE / "train_macilsd_hatemm.py"), *common,
                "--out-dir", str(out), "--modality", modality,
                *option_args(values, method)]
    if method == "multihateloc":
        return [python, str(HERE / "train_multihateloc.py"), *common,
                "--out-root", str(out), "--run-test",
                *option_args(values, method)]
    if method == "cmhkf":
        return [python, str(HERE / "cmhkf_adapter.py"), *common,
                "--out-dir", str(out), "--run-test",
                *option_args(values, method)]
    if method.startswith("fed_wsvad"):
        clients = "3" if method.endswith("3client") else "1"
        return [python, str(HERE / "fed_wsvad_adapter.py"), *common,
                "--out-dir", str(out), "--clients", clients,
                "--partition-seed", "234", "--run-test",
                *option_args(values, method)]
    raise ValueError(method)


def inference_command(method, corpus, out, values, python):
    if method not in ("vadclip", "dsanet") and not method.startswith("macilsd"):
        return None
    if method.startswith("macilsd"):
        modality = "av" if method == "macilsd" else method.removeprefix("macilsd_")
        return [python, str(HERE / "test_macilsd_hatemm.py"),
                "--corpus", corpus, "--device", "cuda",
                "--out-dir", str(out), "--split", "test",
                "--model-path", str(out / "model.pth"),
                *option_args(values, method), "--modality", modality]
    cmd = [python, str(HERE / method / "infer.py"), "--corpus", corpus,
           "--device", "cuda", "--out-dir", str(out), "--split", "test",
           "--model-path", str(out / "model.pth"), *option_args(values, method)]
    return cmd


def run(cmd, log):
    proc = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log.write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"rc={proc.returncode}; see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--tuning-root", default=str(
        REPO / "results/reproduction/official_val/tuning"))
    ap.add_argument("--final-root", default=str(
        REPO / "results/reproduction/official_val/final"))
    ap.add_argument("--code-commit", default=None,
                    help="Git commit frozen across all confirmation subprocesses")
    args = ap.parse_args()
    code_commit = args.code_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    verify_code_commit(code_commit)
    best_path = Path(args.tuning_root) / args.method / args.corpus / "best.json"
    selected = json.loads(best_path.read_text())
    source_sha256 = file_sha256(best_path)
    values = materialize(selected["best_params"])
    confirmation_root = Path(args.final_root) / args.method / args.corpus
    confirmation_root.mkdir(parents=True, exist_ok=True)
    lock_path = confirmation_root / ".confirmation.lock"
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"another confirmation owns {args.method}/{args.corpus}: {lock_path}")
    for seed in SEEDS:
        if file_sha256(best_path) != source_sha256:
            raise RuntimeError(f"validation selection changed: {best_path}")
        out = confirmation_root / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        frozen = {"method": args.method, "corpus": args.corpus, "seed": seed,
                  "code_commit": code_commit,
                  "source": str(best_path), "source_sha256": source_sha256,
                  "best_trial": selected["best_trial"],
                  "best_validation_ap": selected["best_value"], "params": values}
        frozen_path = out / "frozen_config.json"
        scores = (out / args.corpus / "scores.jsonl"
                  if args.method == "multihateloc" else out / "scores.jsonl")
        evaluation_path = out / "frame_eval.json"
        # A completed seed is immutable: do not retrain it or touch test again
        # when a long multi-method confirmation run is resumed.  Only accept
        # the checkpoint as complete when its frozen selection is byte-for-
        # value identical and both score/evaluation artifacts parse.
        if frozen_path.is_file() and scores.is_file() and evaluation_path.is_file():
            try:
                same = json.loads(frozen_path.read_text()) == frozen
                evaluation_payload = json.loads(evaluation_path.read_text())
                identity_ok = (
                    evaluation_payload.get("corpus") == args.corpus and
                    evaluation_payload.get("split") == "test" and
                    Path(evaluation_payload.get("scores_file", "")).resolve() ==
                    scores.resolve() and
                    evaluation_payload.get("scores_sha256") == file_sha256(scores))
                coverage_ok = all(
                    not result.get("n_videos_missing_from_scores") and
                    not result.get("n_videos_not_in_gold")
                    for result in evaluation_payload.get("results", {}).values())
                score_rows = [json.loads(line) for line in scores.read_text().splitlines()
                              if line.strip()]
                if (same and identity_ok and score_rows and coverage_ok and
                        evaluation_payload.get("results")):
                    print(f"already complete {args.method}/{args.corpus}/seed_{seed}",
                          flush=True)
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        atomic_write(frozen_path, json.dumps(frozen, indent=2) + "\n")
        verify_code_commit(code_commit)
        train = train_command(args.method, args.corpus, out, values, seed, args.python)
        run(train, out / "train.log")
        infer = inference_command(args.method, args.corpus, out, values, args.python)
        if infer:
            run(infer, out / "infer.log")
        evaluation = [args.python, str(HERE / "eval_baseline_scores.py"),
                      "--corpus", args.corpus, "--scores", str(scores),
                      "--split", "test", "--require-full-coverage",
                      "--json-out", str(evaluation_path)]
        run(evaluation, out / "eval.log")
        verify_code_commit(code_commit)
        if file_sha256(best_path) != source_sha256:
            raise RuntimeError(f"validation selection changed: {best_path}")
        print(f"completed {args.method}/{args.corpus}/seed_{seed}", flush=True)


if __name__ == "__main__":
    main()
