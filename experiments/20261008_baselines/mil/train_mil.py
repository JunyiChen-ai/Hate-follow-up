#!/usr/bin/env python3
"""Single-feature MIL baselines: BERT + MIL (text), wav2vec2 + MIL (audio), CLIP + MIL (visual).

One head and one training recipe for all three rows; only the instance feature changes (run_plan.md §3 W1-W3,
decision D1 = top-k BCE):
  instances  one row per second (1 fps): bert_sentence_1fps (768, BERT CLS of the Whisper segment covering the second,
             zeros in silence), wav2vec2_base_1s (768), clip_b16_1fps (512)
  head       Sultani et al. MLP d -> 512 -> 32 -> 1, ReLU, dropout 0.6, sigmoid
  loss       top-k MIL with binary cross-entropy (Wu et al., ECCV 2020): video score = mean of the top
             k = floor(T / 16) + 1 instance scores, BCE against the video label
  training   Adam, lr 1e-4, weight decay 5e-4, batch 64 videos padded with a mask, 50 epochs; no hyper-parameter search
  selection  the epoch with the best validation video AP (top-k mean score vs val video labels)
  inference  per-second score of each test-cohort video -> scores.jsonl (score_mil); the 4 fps grid and the evaluation
             are done by weaksup_common.common (x4 repeat, last-value pad)
Seeds 2025 / 234 / 3407; each seed in runs/20261008_baselines/mil_<feature>/<Dataset>/seed<k>/.

    python experiments/20261008_baselines/mil/train_mil.py --feature bert --corpus hatemm
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "weaksup_common"))
import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "scripts" / "duplex"))
import frame_eval_common as fec  # noqa: E402

FEATURES = {"bert": (C.INPUTS / "bert_sentence_1fps", 768),
            "wav2vec2": (C.REPO / "data" / "wav2vec2_base_1s", 768),
            "clip": (C.INPUTS / "clip_b16_1fps", 512)}


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.6), nn.Linear(512, 32), nn.ReLU(),
                                 nn.Dropout(0.6), nn.Linear(32, 1))

    def forward(self, x):                      # (B, T, d) -> (B, T) probabilities
        return torch.sigmoid(self.net(x).squeeze(-1))


def topk_mean(p, lengths):
    out = []
    for i in range(p.shape[0]):
        n = int(lengths[i])
        out.append(torch.topk(p[i, :n], k=n // 16 + 1).values.mean())
    return torch.stack(out)


def load_all(root, corpus, ids):
    return {v: torch.from_numpy(np.load(root / corpus / f"{v}.npy").astype(np.float32)) for v in ids}


def batches(ids, feats, labels, bs, shuffle, rng):
    order = list(ids)
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = order[i:i + bs]
        lengths = torch.tensor([feats[v].shape[0] for v in chunk])
        x = torch.zeros(len(chunk), int(lengths.max()), feats[chunk[0]].shape[1])
        for j, v in enumerate(chunk):
            x[j, :lengths[j]] = feats[v]
        y = torch.tensor([labels[v] for v in chunk], dtype=torch.float32) if labels is not None else None
        yield chunk, x, lengths, y


@torch.no_grad()
def predict(model, ids, feats, device, bs=16):
    model.eval()
    frames, video = {}, {}
    for chunk, x, lengths, _ in batches(ids, feats, None, bs, False, None):
        p = model(x.to(device))
        v = topk_mean(p, lengths)
        for j, vid in enumerate(chunk):
            frames[vid] = p[j, :lengths[j]].cpu().numpy()
            video[vid] = float(v[j])
    return frames, video


def run_seed(args, seed, out, log):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    root, d = FEATURES[args.feature]
    labels = C.train_val_labels(args.corpus)
    tr, va, te = (C.split_ids(args.corpus, k) for k in ("train", "val", "test"))
    feats = load_all(root, args.corpus, tr + va + te)
    for v, f in feats.items():
        if f.ndim != 2 or f.shape[1] != d or f.shape[0] < 1:
            raise ValueError(f"{args.feature}/{args.corpus}/{v}: shape {tuple(f.shape)}")
    device = args.device
    model = Head(d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    best_ap, best_epoch, best_state, history = -1.0, None, None, []
    yv = np.array([labels[v] for v in va])
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for _, x, lengths, y in batches(tr, feats, labels, 64, True, rng):
            p = model(x.to(device))
            loss = F.binary_cross_entropy(topk_mean(p, lengths).clamp(1e-6, 1 - 1e-6), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        _, vs = predict(model, va, feats, device)
        ap = float(fec.average_precision(np.array([vs[v] for v in va]), yv))
        history.append({"epoch": epoch, "loss": tot / n, "val_video_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    log(f"seed {seed}: selected epoch {best_epoch}, val video AP {best_ap:.4f} ({time.time() - t0:.0f}s)")
    frames, _ = predict(model, te, feats, device)
    with (out / "scores.jsonl").open("w") as fh:
        for v in te:
            fh.write(json.dumps({"video_id": v, "n_frames": len(frames[v]),
                                 "score_mil": [round(float(s), 6) for s in frames[v]]}) + "\n")
    torch.save(model.state_dict(), out / "model.pth")
    meta = {"selected_epoch": best_epoch, "selected_val_video_ap": best_ap, "history": history,
            "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True, choices=sorted(FEATURES))
    ap.add_argument("--corpus", required=True, choices=C.CORPORA)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    method = f"mil_{args.feature}"
    ds = C.DATASET[args.corpus]
    base = C.RUNS / method / ds
    for seed in args.seeds:
        out = base / f"seed{seed}"
        if (out / "metrics.json").is_file():
            print(f"{out}/metrics.json exists, kept", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        log = C.RunLog(out / "run.log", mode="w")
        config = {"method": method, "dataset": ds, "corpus_key": args.corpus, "seed": seed,
                  "feature": str(FEATURES[args.feature][0].relative_to(C.REPO)), "feature_dim": FEATURES[args.feature][1],
                  "head": "Linear(d,512)-ReLU-Dropout(.6)-Linear(512,32)-ReLU-Dropout(.6)-Linear(32,1)-sigmoid",
                  "loss": "top-k MIL BCE, k = floor(T/16)+1 (Wu et al. ECCV 2020)",
                  "optim": {"optimizer": "Adam", "lr": 1e-4, "weight_decay": 5e-4, "batch_videos": 64,
                            "epochs": args.epochs},
                  "selection": "epoch with best val video AP (val split, video-level labels)",
                  "train_val_test": {k: len(C.split_ids(args.corpus, k)) for k in ("train", "val", "test")},
                  "code": f"experiments/20261008_baselines/mil/train_mil.py, {C.code_version()}",
                  "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        log(f"{method} {ds} seed {seed}")
        run_seed(args, seed, out, log)
        rows, coverage = C.build_predictions(args.corpus, out / "scores.jsonl", ["score_mil"], [method], seed,
                                             "experiments/20261008_baselines/mil/train_mil.py", log)
        (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        C.evaluate(args.corpus, rows, out, log)
    # seed summary
    per = {s: next(r for r in json.loads((base / f"seed{s}" / "metrics.json").read_text())["per_dataset"]
                   if r["method"] == method) for s in args.seeds}
    summ = {"dataset": ds, "seeds": args.seeds, "methods": {method: {}}}
    for k in ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC"):
        v = np.array([per[s][k] for s in args.seeds], float)
        summ["methods"][method][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                                      "per_seed": {str(s): float(per[s][k]) for s in args.seeds}}
    (base / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    m = summ["methods"][method]
    fmt = lambda e: f"{e['mean']:.4f}" + (f"±{e['sd']:.4f}" if e["sd"] is not None else "")
    print(f"{method} {ds}: ROC {fmt(m['frame_ROC_AUC'])} PR {fmt(m['frame_PR_AUC'])} "
          f"within {fmt(m['within_video_macro_ROC_AUC'])}", flush=True)


if __name__ == "__main__":
    main()
