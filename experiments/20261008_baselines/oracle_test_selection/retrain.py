#!/usr/bin/env python3
"""Oracle test selection, step 1 for the 1 fps weakly supervised baselines: retrain each seed with the identical
recipe and write the test cohort's scores after EVERY epoch (all output branches).

    python retrain.py mil   --feature {bert,wav2vec2,clip} --corpus {hatemm,hateclipseg,dehate}
    python retrain.py fixed --method {vadclip,dsanet,avadclip} --corpus ...
    python retrain.py mhl   --corpus ...

Recipes are the ones of the current runs, unchanged:
  mil    experiments/20261008_baselines/mil/train_mil.py run_seed (50 epochs, val-AP selection); the test cohort is
         scored after each epoch through train_mil.predict (eval mode, no random draw, so the trajectory is the
         same as without the hook).
  fixed  weaksup_common/run_method.py train_cmd (XD preset, 10 epochs, last epoch) + --save-every-epoch; each
         model_eNN.pth is scored by the port's own inference command (run_method.infer_cmd), then deleted.
  mhl    run_method.train_cmd with the tuned hyper-parameters (HateMM / HateClipSeg: tuning/best.json of the current
         run; DeHate: the 2026-09-26 frozen_config.json) + --save-every-epoch; each epoch_states/eNNN.pt is scored
         with multihateloc.train.predict (all six branches), then deleted.
Outputs: runs/20261008_baselines/oracle_test_selection/<method>/<Dataset>/seed<k>/
  run.log (first line = host), config.json, train logs, epochs/e<NNN>.jsonl.gz (1 fps scores per test video, every
  branch, 6 decimals: the scores.jsonl format), DONE.
The test cohort's labels are never read here; the canonical evaluator reads them in evaluate_candidates.py.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "weaksup_common"))
import common as C  # noqa: E402

ORACLE = C.RUNS / "oracle_test_selection"
CUR = C.RUNS  # current runs (read only)


def write_gz(path: Path, recs) -> None:
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    tmp.rename(path)


def gz_from_jsonl(src: Path, dst: Path) -> int:
    n = 0
    tmp = dst.with_name(dst.name + ".part")
    with open(src) as fi, gzip.open(tmp, "wt") as fo:
        for line in fi:
            if line.strip():
                fo.write(line if line.endswith("\n") else line + "\n")
                n += 1
    tmp.rename(dst)
    return n


def fresh_seed_dir(out: Path) -> bool:
    """True if the seed must run; a finished seed (DONE) is kept."""
    if (out / "DONE").is_file():
        print(f"{out}: DONE exists, kept", flush=True)
        return False
    if out.exists():
        shutil.rmtree(out)
    (out / "epochs").mkdir(parents=True)
    return True


def finish(out: Path, log, n_epochs: int, t0: float) -> None:
    files = sorted((out / "epochs").glob("e*.jsonl.gz"))
    if len(files) != n_epochs:
        raise RuntimeError(f"{out}: {len(files)} epoch files, expected {n_epochs}")
    log(f"{len(files)} per-epoch test score files written ({time.time() - t0:.0f}s)")
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def base_config(method, corpus, seed, recipe, extra=None):
    return {"method": method, "dataset": C.DATASET[corpus], "corpus_key": corpus, "seed": seed,
            "purpose": "oracle test selection (user decision 2026-10-09): every epoch of the identical recipe is "
                       "scored on the test cohort so that the checkpoint and branch can be chosen on TEST labels, "
                       "for the weakly supervised baselines only, as an upper bound",
            "recipe": recipe, "current_run": str((CUR / method / C.DATASET[corpus]).relative_to(C.REPO)),
            "train_val_test": {k: len(C.split_ids(corpus, k)) for k in ("train", "val", "test")},
            "code": f"experiments/20261008_baselines/oracle_test_selection/retrain.py, {C.code_version()}",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}


# ------------------------------------------------------------------------------------------------- MIL ---
def cmd_mil(args):
    sys.path.insert(0, str(BASE / "mil"))
    import train_mil as TM
    method = f"mil_{args.feature}"
    ds = C.DATASET[args.corpus]
    for seed in args.seeds:
        out = ORACLE / method / ds / f"seed{seed}"
        if not fresh_seed_dir(out):
            continue
        log = C.RunLog(out / "run.log", mode="w")
        cfg = base_config(method, args.corpus, seed, "mil/train_mil.py run_seed, unchanged (50 epochs, Adam 1e-4, "
                          "top-k MIL BCE, val video AP selection); test scored after every epoch",
                          {"feature": str(TM.FEATURES[args.feature][0].relative_to(C.REPO)), "branches": ["score_mil"]})
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        log(f"{method} {ds} seed {seed}: retrain with per-epoch test scores")
        t0 = time.time()

        def hook(epoch, model, te, feats, device):
            frames, _ = TM.predict(model, te, feats, device)
            write_gz(out / "epochs" / f"e{epoch:03d}.jsonl.gz",
                     ({"video_id": v, "n_frames": len(frames[v]),
                       "score_mil": [round(float(s), 6) for s in frames[v]]} for v in te))

        ns = argparse.Namespace(feature=args.feature, corpus=args.corpus, epochs=50, seeds=[seed], device="cuda")
        meta = TM.run_seed(ns, seed, out, log, epoch_hook=hook)
        (out / "model.pth").unlink(missing_ok=True)
        log(f"selected epoch (val video AP, as the current run's rule) {meta['selected_epoch']}")
        finish(out, log, 50, t0)


# ---------------------------------------------------------------------------------- VadCLIP / DSANet / AVadCLIP ---
def cmd_fixed(args):
    import run_method as RM
    method, corpus = args.method, args.corpus
    ds = C.DATASET[corpus]
    for seed in args.seeds:
        out = ORACLE / method / ds / f"seed{seed}"
        if not fresh_seed_dir(out):
            continue
        log = C.RunLog(out / "run.log", mode="w")
        tcmd = RM.train_cmd(method, corpus, out, {}, seed) + ["--save-every-epoch"]
        cfg = base_config(method, corpus, seed, RM.FIXED_NOTE[method] + "; 10 epochs; train_cmd of run_method.py "
                          "+ --save-every-epoch; each epoch scored by the port's infer command",
                          {"train_cmd": tcmd[1:]})
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        log(f"{method} {ds} seed {seed}: train {' '.join(tcmd[1:])}")
        t0 = time.time()
        RM.run(tcmd, out / "train.log")
        cks = sorted(out.glob("model_e*.pth"))
        log(f"trained ({time.time() - t0:.0f}s); {len(cks)} epoch checkpoints")
        for ck in cks:
            e = int(ck.stem.split("_e")[1])
            tmp = out / "infer_tmp"
            tmp.mkdir(exist_ok=True)
            icmd = RM.infer_cmd(method, corpus, tmp, {})
            icmd[icmd.index("--model-path") + 1] = str(ck)
            RM.run(icmd, out / "infer_last.log")
            n = gz_from_jsonl(tmp / "scores.jsonl", out / "epochs" / f"e{e:03d}.jsonl.gz")
            if n != C.COHORT_SIZE[corpus]:
                raise RuntimeError(f"epoch {e}: {n} test videos scored")
            ck.unlink()
        shutil.rmtree(out / "infer_tmp", ignore_errors=True)
        (out / "model.pth").unlink(missing_ok=True)  # = the last epoch, already scored as e010
        finish(out, log, 10, t0)


# ------------------------------------------------------------------------------------------------ MultiHateLoc ---
def mhl_params(corpus):
    import run_method as RM
    if corpus == "dehate":
        src = CUR / "multihateloc" / "DeHate" / "source_lab2" / "final"
        ps = [json.loads((src / f"seed_{s}" / "frozen_config.json").read_text())["params"] for s in C.SEEDS]
        if any(p != ps[0] for p in ps):
            raise RuntimeError("DeHate seeds used different hyper-parameters")
        return dict(ps[0]), str((src / "seed_<k>" / "frozen_config.json").relative_to(C.REPO))
    best = json.loads((CUR / "multihateloc" / C.DATASET[corpus] / "tuning" / "best.json").read_text())
    return RM.materialize(best["best_params"]), str((CUR / "multihateloc" / C.DATASET[corpus] / "tuning" /
                                                     "best.json").relative_to(C.REPO))


def mhl_score_epochs(corpus, out, log):
    """Score every epoch_states/eNNN.pt on the test cohort with multihateloc.train.predict (all branches)."""
    import importlib

    import torch
    import torch.utils.data as tdata
    import port
    port.patch(no_val=False)
    MT = importlib.import_module("multihateloc.train")
    MT.mdata.FEATURE_ROOT = str(C.INPUTS)
    tl = json.loads((out / corpus / "train_log.json").read_text())
    a = tl["args"]
    labels = MT.hdata.load_labels(corpus)
    gold = MT.hdata.gt_arrays(corpus, "test")
    test_ids = [v for v in MT.hdata.load_split(corpus, "test") if v in gold]
    if len(test_ids) != C.COHORT_SIZE[corpus]:
        raise RuntimeError(f"{len(test_ids)} test ids")
    loader = tdata.DataLoader(MT.mdata.MultiModalDataset(corpus, test_ids, labels), batch_size=a["batch_size"],
                              shuffle=False, collate_fn=MT.mdata.collate, num_workers=2)
    model = MT.MultiHateLoc({m: MT.mdata.FEATURE_DIMS[m] for m in MT.mdata.MODALITIES}, hidden=a["hidden"],
                            embed=a["embed"], dropout=a["dropout"], k_proportion=a["k_proportion"],
                            temperature=a["temperature"]).to("cuda")
    states = sorted((out / corpus / "epoch_states").glob("e*.pt"))
    for sp in states:
        model.load_state_dict(torch.load(sp, map_location="cuda"))
        frames, _, _ = MT.predict(model, loader, "cuda")
        e = int(sp.stem[1:])
        write_gz(out / "epochs" / f"e{e:03d}.jsonl.gz",
                 ({"video_id": v, **{k: [round(float(x), 6) for x in arr] for k, arr in frames[v].items()}}
                  for v in test_ids))
        sp.unlink()
    (out / corpus / "epoch_states").rmdir()
    log(f"scored {len(states)} epochs (branches {sorted(frames[test_ids[0]])})")
    del model
    gc.collect()
    return len(states)


def cmd_mhl(args):
    import run_method as RM
    corpus = args.corpus
    ds = C.DATASET[corpus]
    values, src = mhl_params(corpus)
    for seed in args.seeds:
        out = ORACLE / "multihateloc" / ds / f"seed{seed}"
        if not fresh_seed_dir(out):
            continue
        log = C.RunLog(out / "run.log", mode="w")
        tcmd = RM.train_cmd("multihateloc", corpus, out, values, seed, run_test=True) + ["--save-every-epoch"]
        cfg = base_config("multihateloc", corpus, seed, "tuned hyper-parameters of the current run (5-trial Optuna on "
                          "val video AP), epochs as tuned, val-AP epoch selection; + --save-every-epoch; every epoch "
                          "scored with multihateloc.train.predict", {"params": values, "params_source": src,
                                                                     "train_cmd": tcmd[1:]})
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        log(f"multihateloc {ds} seed {seed}: train {' '.join(tcmd[1:])}")
        t0 = time.time()
        RM.run(tcmd, out / "train.log")
        n = mhl_score_epochs(corpus, out, log)
        (out / corpus / "model.pt").unlink(missing_ok=True)
        finish(out, log, int(values["max_epoch"]), t0)
        if n != int(values["max_epoch"]):
            raise RuntimeError(f"{n} epochs scored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=("mil", "fixed", "mhl"))
    ap.add_argument("--feature", choices=("bert", "wav2vec2", "clip"))
    ap.add_argument("--method", choices=("vadclip", "dsanet", "avadclip"))
    ap.add_argument("--corpus", required=True, choices=C.CORPORA)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    args = ap.parse_args()
    {"mil": cmd_mil, "fixed": cmd_fixed, "mhl": cmd_mhl}[args.family](args)


if __name__ == "__main__":
    main()
