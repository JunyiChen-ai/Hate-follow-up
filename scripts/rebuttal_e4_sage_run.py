#!/usr/bin/env python
"""Rebuttal E4b — train + eval SAGE on OUR fixed splits, one (dataset, seed) per call.

Reuses upstream SAGE verbatim (model/Sage.py, CustomLoss, Trainer.train_model,
CustomDataset, utils.load_split_*). This driver only handles seeding, our-split
plumbing, and per-video prediction / metric dumping — no method change.

  python rebuttal_e4_sage_run.py --mode train \
      --config external_repos/SAGE/config/config_ours.yaml \
      --dataset {hatemm,mhclip_yt,mhclip_bl} --seed 0 \
      --out_dir results/rebuttal/E4_sage
  python rebuttal_e4_sage_run.py --mode aggregate --dataset hatemm --out_dir ...
"""
import os
import sys
import json
import argparse
import random

import numpy as np
import yaml
import torch

REPO = "/data/jehc223/EMNLP3"
sys.path.insert(0, os.path.join(REPO, "external_repos/SAGE/model"))
sys.path.insert(0, os.path.join(REPO, "src"))

from utils import (  # noqa: E402
    get_mapping, load_split_hatemm, load_split_mhclip, get_training_setting,
)
from CustomDataset import CustomDataset  # noqa: E402
from Sage import SAGE  # noqa: E402
from Trainer import train_model, collate_fn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)

# dataset_key -> (our dataset name, loader kind, mapping name).
# ImpliHateVid is already binary {Hateful, Normal} -> mhclip loader path, no collapse.
DS_MAP = {
    "hatemm": ("HateMM", "hatemm", "hatemm"),
    "mhclip_yt": ("MHClip_EN", "mhclip", "mhclip"),
    "mhclip_bl": ("MHClip_ZH", "mhclip", "mhclip"),
    "impli": ("ImpliHateVid", "mhclip", "mhclip"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_trans(trans_path):
    with open(trans_path) as f:
        trans_list = json.load(f)
    trans = {}
    for item in trans_list:
        vid = item.get("Video_ID")
        if vid:
            trans[vid] = item
    return trans


def build_df(kind, path, trans, audio_dir, frame_dir, n_frames):
    if kind == "hatemm":
        return load_split_hatemm(path, trans, audio_dir, frame_dir, n_frames)
    return load_split_mhclip(path, trans, audio_dir, frame_dir, n_frames, num_classes=2)


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    recs, y_true, y_pred = [], [], []
    for batch in loader:
        vids = batch["Video_ID"]
        t = batch["text_feat"].to(device)
        a = batch["audio_feat"].to(device)
        v = batch["vision_feat"].to(device)
        labels = batch["label"].to(device)
        _, _, _, logits_fusion, _, _ = model(t, a, v)
        probs = torch.softmax(logits_fusion, dim=1)
        preds = torch.argmax(logits_fusion, dim=1)
        for i, vid in enumerate(vids):
            recs.append({
                "video_id": vid,
                "label": int(labels[i].item()),
                "pred": int(preds[i].item()),
                "prob_pos": float(probs[i, 1].item()),
            })
            y_true.append(int(labels[i].item()))
            y_pred.append(int(preds[i].item()))
    return recs, y_true, y_pred


@torch.no_grad()
def diagnose_experts(model, dataset, device, batch_size, class_names):
    """Reload a trained SAGE and report, on the test set, each expert's own
    accuracy and how much gate mass it receives — to tell whether any single
    expert (e.g. the ZH text expert) is effectively dead or noisy."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    per_video = []
    acc = {"text": 0, "audio": 0, "vision": 0, "fusion": 0}
    gate_sum = {"text": 0.0, "audio": 0.0, "vision": 0.0}
    n = 0
    for batch in loader:
        t = batch["text_feat"].to(device)
        a = batch["audio_feat"].to(device)
        v = batch["vision_feat"].to(device)
        labels = batch["label"].to(device)
        lt, la, lv, lf, gates, _ = model(t, a, v)
        pt, pa, pv, pf = (torch.argmax(x, 1) for x in (lt, la, lv, lf))
        for i, vid in enumerate(batch["Video_ID"]):
            y = int(labels[i].item())
            g = gates[i].tolist()  # [g_T, g_A, g_V]
            row = {"video_id": vid, "label": y,
                   "pred_text": int(pt[i]), "pred_audio": int(pa[i]),
                   "pred_vision": int(pv[i]), "pred_fusion": int(pf[i]),
                   "gate_text": g[0], "gate_audio": g[1], "gate_vision": g[2]}
            per_video.append(row)
            acc["text"] += int(pt[i]) == y
            acc["audio"] += int(pa[i]) == y
            acc["vision"] += int(pv[i]) == y
            acc["fusion"] += int(pf[i]) == y
            gate_sum["text"] += g[0]
            gate_sum["audio"] += g[1]
            gate_sum["vision"] += g[2]
            n += 1
    summary = {"n": n,
               "expert_accuracy": {k: acc[k] / n for k in acc},
               "mean_gate_weight": {k: gate_sum[k] / n for k in gate_sum},
               "note": "expert_accuracy = argmax of each expert's own logits vs label; "
                       "mean_gate_weight = average top-k-renormalised gate mass per modality"}
    return summary, per_video


def compute_metrics(y_true, y_pred, class_names):
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, zero_division=0),
    }


def run_train(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    ds_name, kind, map_name = DS_MAP[args.dataset]
    device = config["device"] if torch.cuda.is_available() else "cpu"
    config["device"] = device
    n_frames = config["vision"]["n_frames"]
    dc = config[args.dataset]

    set_seed(args.seed)
    out = os.path.join(args.out_dir, ds_name, f"seed{args.seed}")
    metrics_path = os.path.join(out, "metrics.json")
    # Collision/resume guard: if a completed run already wrote metrics.json for
    # this (dataset, seed), skip. Lets a shard job and the all-in-one job overlap
    # on a dataset without racing on the same seed dir — whoever finishes first
    # wins, the other skips. (Safe only because they never start the same
    # (dataset, seed) simultaneously; metrics.json is written last, on success.)
    if os.path.exists(metrics_path):
        print(f"[{ds_name} seed{args.seed}] SKIP — {metrics_path} already exists", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    config["train"]["save_path"] = os.path.join(out, "best_model.pth")

    trans = load_trans(dc["trans_path"])
    train_df = build_df(kind, dc["train_path"], trans, dc["audio_path"], dc["frame_path"], n_frames)
    val_df = build_df(kind, dc["val_path"], trans, dc["audio_path"], dc["frame_path"], n_frames)
    test_df = build_df(kind, dc["test_path"], trans, dc["audio_path"], dc["frame_path"], n_frames)
    print(f"[{ds_name} seed{args.seed}] train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)
    print("  train labels:", train_df["Label"].value_counts().to_dict(), flush=True)
    print("  test  labels:", test_df["Label"].value_counts().to_dict(), flush=True)

    mapping = get_mapping(map_name, 2)
    class_names = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    model = SAGE(config["model"]).to(device)
    _, _, _, _, optimizer, criterion, scheduler = get_training_setting(config, model)

    train_data = CustomDataset(config, train_df, train=True)
    valid_data = CustomDataset(config, val_df, train=False)
    train_model(model, config["train"], train_data, valid_data, mapping,
                criterion, optimizer, scheduler, device)

    # free train/val encoder copies before building the test dataset (peak-mem safety)
    del train_data, valid_data
    torch.cuda.empty_cache()

    # reload best-val checkpoint, then dump test predictions
    ckpt = torch.load(config["train"]["save_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    test_data = CustomDataset(config, test_df, train=False)
    recs, y_true, y_pred = predict(model, test_data, device, config["train"]["batch_size"])

    with open(os.path.join(out, "predictions.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
        os.fsync(f.fileno())
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics.update({"dataset": ds_name, "seed": args.seed,
                    "best_val_f1": float(ckpt.get("best_f1", 0.0)),
                    "best_epoch": int(ckpt.get("epoch", -1))})
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    print(f"[{ds_name} seed{args.seed}] ACC={metrics['accuracy']:.4f} "
          f"M-F1={metrics['macro_f1']:.4f} M-P={metrics['macro_precision']:.4f} "
          f"M-R={metrics['macro_recall']:.4f}  (best_val_f1={metrics['best_val_f1']:.4f})", flush=True)


def run_aggregate(args):
    ds_name = DS_MAP[args.dataset][0]
    base = os.path.join(args.out_dir, ds_name)
    seeds = []
    for d in sorted(os.listdir(base)):
        mp = os.path.join(base, d, "metrics.json")
        if d.startswith("seed") and os.path.isfile(mp):
            with open(mp) as f:
                seeds.append(json.load(f))
    if not seeds:
        print(f"[aggregate] no seed metrics under {base}")
        return
    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    agg = {"dataset": ds_name, "n_seeds": len(seeds),
           "seeds": [s["seed"] for s in seeds]}
    for k in keys:
        vals = [s[k] for s in seeds]
        agg[k + "_mean"] = float(np.mean(vals))
        agg[k + "_std"] = float(np.std(vals))
        agg[k + "_per_seed"] = vals
    with open(os.path.join(base, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[aggregate {ds_name}] n={len(seeds)} "
          f"ACC={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f} "
          f"M-F1={agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}", flush=True)


def run_diag(args):
    """Load a trained seed's best_model.pth and dump per-expert accuracy + gate
    weights on the test set. Read-only w.r.t. training artifacts."""
    with open(args.config) as f:
        config = yaml.safe_load(f)
    ds_name, kind, map_name = DS_MAP[args.dataset]
    device = config["device"] if torch.cuda.is_available() else "cpu"
    config["device"] = device
    n_frames = config["vision"]["n_frames"]
    dc = config[args.dataset]
    out = os.path.join(args.out_dir, ds_name, f"seed{args.seed}")
    ckpt_path = os.path.join(out, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[diag {ds_name} seed{args.seed}] no checkpoint at {ckpt_path}", flush=True)
        return
    set_seed(args.seed)
    trans = load_trans(dc["trans_path"])
    test_df = build_df(kind, dc["test_path"], trans, dc["audio_path"], dc["frame_path"], n_frames)
    mapping = get_mapping(map_name, 2)
    class_names = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    model = SAGE(config["model"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    test_data = CustomDataset(config, test_df, train=False)
    summary, per_video = diagnose_experts(model, test_data, device,
                                          config["train"]["batch_size"], class_names)
    with open(os.path.join(out, "diag.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    with open(os.path.join(out, "diag_predictions.jsonl"), "w") as f:
        for r in per_video:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    ea, gw = summary["expert_accuracy"], summary["mean_gate_weight"]
    print(f"[diag {ds_name} seed{args.seed}] n={summary['n']} | "
          f"expert-acc text={ea['text']:.3f} audio={ea['audio']:.3f} vision={ea['vision']:.3f} "
          f"fusion={ea['fusion']:.3f} | gate text={gw['text']:.3f} audio={gw['audio']:.3f} "
          f"vision={gw['vision']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "aggregate", "diag"], default="train")
    ap.add_argument("--config", default=os.path.join(REPO, "external_repos/SAGE/config/config_ours.yaml"))
    ap.add_argument("--dataset", required=True, choices=list(DS_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(REPO, "results/rebuttal/E4_sage"))
    args = ap.parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "aggregate":
        run_aggregate(args)
    else:
        run_diag(args)


if __name__ == "__main__":
    main()
