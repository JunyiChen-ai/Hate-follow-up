#!/usr/bin/env python3
"""CLARA step 5: train as published (run_clara.sh settings), select on val, score every test window.

    python clara_train.py --dataset DS --seed S [--rationale-mode both|none]

The authors' `main.py` pipeline (VideoDataset, CLARACollator, CLARA, GatedVideoTransformer, MoE clip encoder,
CLARATrainer with early stopping and `load_best_model_at_end` on val accuracy) is imported and driven with the
arguments of `run_clara.sh`; only paths, seed and `dataloader_num_workers` (8 -> 4, the job's CPUs) differ.
`--rationale-mode none` is the paper's "w/o Rationale" ablation (`--gvt_rationale_mode none`), run as a check of
how much of the window score comes from the video-level rationale tokens.

Hash ban: the collator seeds each video's contrastive segment sampling with md5(video_id)
(`utils/segment_sampling.py` `stable_hash_str`). Here the seed term is the index of the id in the sorted list of all
sample ids instead; `utils/dataset.py` `_stable_int_from_str` (only used by clip_select=random, not the default
truncate) is replaced by a function that raises.
Test windows carry placeholder label 0 (no test label is read); the trainer's test metrics are meaningless and
discarded; only the window probabilities softmax(logits)[:, 1] are kept.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "detwin"))
import common as C  # noqa: E402

DATA = Path(os.environ.get("CLARA_DATA", C.REPO / "data"))  # env override: smoke tests only

CLARA_DIR = C.REPO / "third_party" / "CLARA"
EMB = DATA / "clara_emb"


def run_clara_args(ds: str, out: Path, seed: int, mode: str) -> list[str]:
    """run_clara.sh defaults (BATCH 32, EPOCHS 50, WARMUP 0.05, EARLY_PATIENCE 10, BEST_METRIC acc, LR 3e-5,
    WD 1e-4, PROJ 256, moe 8 experts top-3, GVT 4 layers 8 heads ffn 4, SEG independent 1 pair 0.2/0.8,
    tau 0.1, CL weight 0.3, BUDGET 40, RAT Qwen, TEXT bert, bf16)."""
    return [
        "--do_train", "--do_eval", "--do_predict", "--bf16",
        "--use_transformer", "true", "--use_contrastive", "true",
        "--dataset_root", str(EMB), "--dataset_name", ds, "--split_json", str(EMB / ds / "split.json"),
        "--fold", "0", "--budget", "40", "--rationale_source", "Qwen", "--map_location", "cpu",
        "--text_emb_model", "bert", "--output_dir", str(out),
        "--learning_rate", "3e-5", "--weight_decay", "1e-4", "--num_train_epochs", "50", "--warmup_ratio", "0.05",
        "--per_device_train_batch_size", "32", "--per_device_eval_batch_size", "32", "--clip_microbatch_size", "0",
        "--early_stopping_patience", "10", "--early_stopping_threshold", "0.0",
        "--proj_out_dim", "256", "--clip_encoder_backend", "moe", "--moe_num_experts", "8", "--moe_top_k", "3",
        "--gvt_num_layers", "4", "--gvt_num_heads", "8", "--gvt_ffn_mult", "4", "--gvt_rationale_mode", mode,
        "--gvt_use_cls", "true", "--gvt_use_mean_pool", "false",
        "--seg_mode", "independent", "--seg_num_pairs_per_video", "1", "--seg_l_ratio", "0.2", "--seg_g_ratio", "0.8",
        "--contrast_tau", "0.1", "--contrastive_weight", "0.3",
        "--dataloader_pin_memory", "true", "--dataloader_num_workers", "4",
        "--eval_strategy", "epoch", "--save_strategy", "epoch", "--save_total_limit", "1",
        "--logging_strategy", "steps", "--logging_steps", "50", "--load_best_model_at_end", "true",
        "--metric_for_best_model", "acc", "--greater_is_better", "true", "--save_last", "false",
        "--remove_unused_columns", "false", "--seed", str(seed), "--report_to", "none",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rationale-mode", default="both", choices=["both", "none"])
    ap.add_argument("--cpu-smoke", action="store_true", help="debug on CPU (bf16 autocast on CPU), 2 epochs")
    ap.add_argument("--cpu", action="store_true", help="train on CPU (bf16 autocast on CPU), all settings unchanged")
    args = ap.parse_args()
    ds, mode = args.dataset, args.rationale_mode
    method_dir = "clara" if mode == "both" else "clara_norationale"
    method = "CLARA_win8" if mode == "both" else "CLARA_win8_norationale"
    rd = C.RUNS / method_dir / ds / f"seed{args.seed}"
    hf_out = rd / "hf"
    rd.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(rd / "run.log")
    log(f"train {method} {ds} seed {args.seed}; device {'cpu' if (args.cpu or args.cpu_smoke) else 'cuda'}; "
        f"code {C.code_version()}")

    sys.path.insert(0, str(CLARA_DIR))
    import utils.segment_sampling as SS
    import utils.collator as COL
    import utils.dataset as DSET

    split = json.loads((EMB / ds / "split.json").read_text())
    all_ids = sorted({x["video_id"] for k in ("train", "valid", "test") for x in split["folds"][0][k]})
    id_index = {v: i for i, v in enumerate(all_ids)}

    def build_video_rng(base_seed, epoch, rank, video_id, epoch_mult=1000003, rank_mult=10007):
        import random
        return random.Random(int(base_seed) + epoch_mult * int(epoch) + rank_mult * int(rank) + id_index[str(video_id)])

    def _no_hash(s):
        raise RuntimeError("hash-derived seeds are not allowed (CLAUDE.md)")

    SS.build_video_rng = build_video_rng
    COL.build_video_rng = build_video_rng
    SS.stable_hash_str = _no_hash
    DSET._stable_int_from_str = _no_hash

    # DeepSpeed is installed in the HateVideo env but has no CUDA toolkit (CUDA_HOME) on the lab nodes; accelerate
    # imports it inside Trainer.__init__ just to recognise wrapped models and the import fails on a GPU node.
    # DeepSpeed is not used here, so accelerate is told it is unavailable.
    import accelerate.accelerator as _AA
    import accelerate.utils as _AU
    import accelerate.utils.imports as _AUI
    import accelerate.utils.other as _AUO
    for _m in (_AA, _AU, _AUI, _AUO):
        if hasattr(_m, "is_deepspeed_available"):
            _m.is_deepspeed_available = lambda: False
    import main as CM
    import torch
    # The packs store features in fp16 (extract_video_emb.py OUT_DTYPE). Under run_clara.sh's --bf16 autocast,
    # torch.cat of two fp16 tensors in clip_encoder.FeatureProjector (text_proj_both) raises "Unexpected floating
    # ScalarType in at::autocast::prioritize". Upcasting the loaded features to fp32 keeps every value and lets the
    # released code run; autocast then computes the layers in bf16 as intended.
    _orig_getitem = DSET.VideoDataset.__getitem__

    def _getitem_fp32(self, idx):
        out = _orig_getitem(self, idx)
        return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}

    DSET.VideoDataset.__getitem__ = _getitem_fp32
    from transformers import HfArgumentParser
    from utils.training_arguments import TrainingArguments, DataArguments
    argv = run_clara_args(ds, hf_out, args.seed, mode)
    if args.cpu or args.cpu_smoke:
        argv = argv + ["--use_cpu", "true"]
    if args.cpu_smoke:
        argv[argv.index("--num_train_epochs") + 1] = "2"
    (rd / "config_snapshot.json").write_text(json.dumps({"argv": argv, "method": method}, indent=1) + "\n")
    training_args, data_args = HfArgumentParser((TrainingArguments, DataArguments)).parse_args_into_dataclasses(argv)
    training_args.raw_textual_emb_dim = CM.infer_text_dim(data_args.text_emb_model)
    hf_out.mkdir(parents=True, exist_ok=True)
    train_ds = CM.VideoDataset(data_args=data_args, split="train")
    eval_ds = CM.VideoDataset(data_args=data_args, split="valid")
    test_ds = CM.VideoDataset(data_args=data_args, split="test")
    log(f"samples: train {len(train_ds)} val {len(eval_ds)} test windows {len(test_ds)}")
    collator = CM.CLARACollator(data_args, seg_cfg=CM.build_seg_cfg(training_args), rank=0)
    model = CM.CLARA(cfg=CM.build_clara_cfg(training_args), clip_encoder=CM.build_clip_encoder(training_args),
                     transformer=CM.build_gvt(training_args) if training_args.use_transformer else None)
    trainer = CM.build_trainer(model=model, training_args=training_args, train_dataset=train_ds,
                               eval_dataset=eval_ds, data_collator=collator, compute_metrics=CM.compute_metrics)
    trainer.add_callback(CM.SetEpochCallback(collator))
    tr = trainer.train()
    ev = trainer.evaluate()
    log(f"train done: {json.dumps(tr.metrics)}")
    log(f"val (best checkpoint, selection metric acc): {json.dumps(ev)}")
    best = trainer.state.best_model_checkpoint
    hist = [h for h in trainer.state.log_history if "eval_acc" in h]
    (rd / "train_history.json").write_text(json.dumps({"best_checkpoint": best, "eval": hist}, indent=1) + "\n")
    pred = trainer.predict(test_ds)
    logits = np.asarray(pred.predictions, dtype=np.float64)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = (p / p.sum(1, keepdims=True))[:, 1]
    ids = [test_ds.samples[i]["video_id"] for i in range(len(test_ds))]
    win = {}
    for sid, pj in zip(ids, p):
        v, j = sid.rsplit("__w", 1)
        win.setdefault(v, {})[int(j)] = float(pj)
    window_scores = {v: [d[j] for j in sorted(d)] for v, d in win.items()}
    (rd / "window_scores.json").write_text(json.dumps(window_scores) + "\n")
    # keep the selected model's weights only; drop the HF checkpoint folders
    trainer.save_model(str(rd / "best_model"))
    for ck in hf_out.glob("checkpoint-*"):
        shutil.rmtree(ck, ignore_errors=True)
    C.finalize(method, ds, args.seed, rd, window_scores,
               code_path="experiments/20261008_baselines/clara/clara_train.py",
               config={"window_seconds": C.WIN, "rationale_mode": mode, "clara_commit": "468a6bc",
                       "best_checkpoint": best, "val_metrics": ev,
                       "inputs": "per window: clips of the window's Whisper segments, frames at the video's own "
                                 "40-frame sampling rate (<= 40), window audio, OCR, video-level rationale"},
               log=log)
    log("DONE clara")


if __name__ == "__main__":
    main()
