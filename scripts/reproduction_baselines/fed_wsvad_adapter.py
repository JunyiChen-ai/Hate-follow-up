#!/usr/bin/env python3
"""Fed-WSVAD on the four hateful-video corpora, with validation-only selection.

The released model is imported unchanged from the pinned upstream checkout.
This file replaces its dataset/federation driver: upstream evaluates the test
set every global round, whereas this study selects exclusively on the frozen
validation manifest and touches test only when ``--run-test`` is requested.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UPSTREAM = REPO / "third_party" / "Fed-WSVAD"
if not UPSTREAM.is_dir():
    UPSTREAM = REPO / "third_party" / "fed_wsvad"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "duplex"))

from utils.model import Model  # type: ignore  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from frame_eval_common import average_precision  # noqa: E402

PROMPTS = ["normal content", "hateful content"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mil_loss(logits, labels, lengths):
    bags = []
    for i in range(len(labels)):
        n = int(lengths[i])
        k = max(1, n // 16 + 1)
        bags.append(logits[i, :n].topk(k, dim=0).values.mean(0))
    return F.cross_entropy(torch.stack(bags), labels.long())


def make_loader(corpus, ids, labels, visual_length, batch, seed, shuffle=True):
    ds = hdata.HateVideoDataset(corpus, ids, visual_length, False, labels)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False,
                      num_workers=0, generator=gen)


def score_ids(model, corpus, ids, visual_length, device):
    scores, video_scores = {}, {}
    model.eval()
    with torch.no_grad():
        for vid in ids:
            feat = np.load(hdata.feature_path(corpus, vid)).astype(np.float32)
            n = len(feat)
            blocks, _ = hdata.tools.process_split(feat, visual_length)
            # Upstream process_split returns [T,D] for a short video and
            # [B,T,D] otherwise.  Model always expects the latter, and its
            # attention mask needs each block's real (unpadded) length.
            if blocks.ndim == 2:
                blocks = blocks[None, ...]
            lengths = runtime.chunk_lengths(n, visual_length).to(device)
            logits = model(torch.from_numpy(blocks).to(device), PROMPTS,
                           lengths)
            s = (1.0 - logits.softmax(-1)[..., 0]).reshape(-1)[:n]
            arr = s.float().cpu().numpy()
            scores[vid] = arr
            video_scores[vid] = float(arr.max())
    return scores, video_scores


def val_ap(model, corpus, ids, labels, visual_length, device):
    _, values = score_ids(model, corpus, ids, visual_length, device)
    ordered = sorted(ids)
    return average_precision([values[v] for v in ordered],
                             [labels[v] for v in ordered])


def partition(ids, labels, clients, seed=234):
    if clients == 1:
        return [sorted(ids)]
    rng = random.Random(seed)
    out = [[] for _ in range(clients)]
    for cls in (0, 1):
        members = sorted(v for v in ids if labels[v] == cls)
        rng.shuffle(members)
        for i, vid in enumerate(members):
            out[i % clients].append(vid)
    return [sorted(x) for x in out]


def local_update(model, loader, epochs, lr, device):
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr)
    total, steps = 0.0, 0
    model.train()
    for _ in range(epochs):
        for feat, label, lengths in loader:
            feat, label, lengths = feat.to(device), label.to(device), lengths.to(device)
            loss = mil_loss(model(feat, PROMPTS, lengths), label, lengths)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Fed-WSVAD training loss became non-finite")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
             if p.requires_grad}
    return state, total / max(steps, 1), len(loader.dataset)


def apply_trainable(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(param.device))


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--partition-seed", type=int, default=234)
    ap.add_argument("--global-rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--visual-length", type=int, default=None)
    ap.add_argument("--visual-layers", type=int, default=2)
    ap.add_argument("--attn-window", type=int, default=None)
    ap.add_argument("--prompt-prefix", type=int, default=10)
    ap.add_argument("--prompt-postfix", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-test", action="store_true")
    args = ap.parse_args(argv)

    seed_all(args.seed)
    device = torch.device(args.device)
    visual_length = args.visual_length or runtime.default_visual_length(args.corpus)
    attn_window = args.attn_window or runtime.default_attn_window(args.corpus)
    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(args.corpus, labels)
    client_ids = partition(train_ids, labels, args.clients, args.partition_seed)
    loaders = [make_loader(args.corpus, x, labels, visual_length,
                           args.batch_size, args.seed + i)
               for i, x in enumerate(client_ids)]

    model = Model(512, visual_length, args.prompt_prefix, args.prompt_postfix,
                  512, args.visual_layers, 1, attn_window, str(device)).to(device)
    best_ap, best_round, best_state, history = -1.0, None, None, []
    started = time.time()
    for rnd in range(1, args.global_rounds + 1):
        global_state = {n: p.detach().cpu().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        local_states, losses, weights = [], [], []
        for loader in loaders:
            apply_trainable(model, global_state)
            state, loss, weight = local_update(
                model, loader, args.local_epochs, args.lr, device)
            local_states.append(state); losses.append(loss); weights.append(weight)
        merged = {name: sum(state[name] * (w / sum(weights))
                            for state, w in zip(local_states, weights))
                  for name in global_state}
        apply_trainable(model, merged)
        score = val_ap(model, args.corpus, val_ids, labels, visual_length, device)
        if not np.isfinite(score):
            raise RuntimeError(
                f"Fed-WSVAD validation AP became non-finite at round {rnd}")
        history.append({"round": rnd, "loss": float(np.average(losses,
                        weights=weights)), "val_video_ap": score})
        print(f"round {rnd}/{args.global_rounds} loss={history[-1]['loss']:.4f} val_ap={score:.4f}", flush=True)
        if score > best_ap:
            best_ap, best_round = score, rnd
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Fed-WSVAD produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pth")
    meta = {"method": f"fed_wsvad_{args.clients}client",
            "protocol": "official-val",
            "upstream": "wbfwonderful/Fed-WSVAD@287747f",
            "args": vars(args), "visual_length": visual_length,
            "attn_window": attn_window, "train_ids": train_ids,
            "val_ids": val_ids, "client_ids": client_ids,
            "selected_round": best_round, "selected_val_video_ap": best_ap,
            "history": history, "wall_seconds": time.time() - started}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if args.run_test:
        test_ids = [v for v in hdata.load_split(args.corpus, "test")
                    if v in hdata.gt_arrays(args.corpus)]
        scores, _ = score_ids(model, args.corpus, test_ids, visual_length, device)
        with (out / "scores.jsonl").open("w") as fh:
            for vid in test_ids:
                fh.write(json.dumps({"video_id": vid,
                                     "score_align": scores[vid].tolist()}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
