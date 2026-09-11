#!/usr/bin/env python3
"""SDL: label-free MIL adaptation of the window decision, supervised by the model's own video verdict.

Training (no labels anywhere): for each video, the shared prefix is encoded WITHOUT gradient and its KV
cache is treated as a constant; each 8 s window's Yes/No branch is then run WITH gradient through a rank-8
LoRA on the language layers' attention projections. The loss is multiple-instance:

    y_v = 1[z_video > 0]                      pseudo label from the FROZEN model, fixed before training
    L   = - y_v * log sigmoid(max_i s_i)
          - (1 - y_v) * mean_i log sigmoid(-s_i)
          + lambda * mean_i sigmoid(s_i)

Inference is SPVL-r2 unchanged except that the adapter is active. Nothing here reads a hate label; the
pseudo labels come from a frozen run's predictions.jsonl (extra.z_video).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
CODE_PATH = "experiments/20260912_sdl/sdl.py"
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION  # noqa: E402
from src.video_inputs import (FPS, fixed_windows, frame_paths, load_asr, load_manifest,  # noqa: E402
                              window_text, within_defined_ids)

SEED = 0
FILL_UNCOVERED = -12.0
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


# ------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B(A(x)). Base weights stay frozen; only A and B train."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        dev = base.weight.device
        self.A = nn.Linear(base.in_features, r, bias=False, device=dev, dtype=torch.float32)
        self.B = nn.Linear(r, base.out_features, bias=False, device=dev, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        delta = self.B(self.A(x.to(torch.float32))) * self.scale
        return out + delta.to(out.dtype)


def inject_lora(model, r=8, alpha=16, targets=TARGETS):
    """Wrap every nn.Linear whose attribute name is in `targets` inside the language layers."""
    lang = getattr(model, "language_model", None) or getattr(getattr(model, "model", model), "language_model", None)
    root = lang if lang is not None else model
    n = 0
    for module in root.modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
                n += 1
    return n


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_lora(model, enabled):
    for m in lora_modules(model):
        m.enabled = enabled


# ------------------------------------------------------------------ per video
def build_prefix(judge, row, segments, args):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    wins = fixed_windows(dur, args.window_seconds)
    wtexts = [window_text(segments, a, b) for a, b in wins]
    frames = frame_paths(ds, vid, args.frames, "k20") if args.frames > 0 else []
    if args.frames > 0 and not frames:
        return None
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=args.frames > 0)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    return dict(vid=vid, ds=ds, dur=dur, wins=wins, wtexts=wtexts, msgs=msgs, image_files=image_files,
                prefix_text=prefix_text, enc=enc)


def act_question(i, n, t1, t2, text, kind):
    body = text.strip() if text and text.strip() else "(no speech)"
    head = f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video."
    if kind == "visual":
        ctx = (f"{head} Look only at the frames whose timestamps fall inside this window and judge the visual "
               f"content alone (imagery, gestures, symbols, on-screen text), ignoring the speech.\n\n")
        q = "Is THIS window one of the segments where visual content that violates the above rules occurs?"
    else:
        ctx = f"{head} Judge only what is spoken in this window: {body}\n\n"
        q = "Is THIS window one of the segments where speech that violates the above rules occurs?"
    return ctx + q + '\n\nAnswer "Yes" or "No".'


def branch_specs(judge, P, stance, args):
    """(window index, kind, token ids) for every branch, conditioned on the stance turn."""
    msgs, wins, wtexts = P["msgs"], P["wins"], P["wtexts"]
    b0, b0_text = judge.branch_ids(msgs, VIDEO_QUESTION)
    a0, a0_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance)
    history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]},
               judge.turn("assistant", stance)]
    head = P["prefix_text"] + b0_text + a0_text
    out = []
    for i, ((t1, t2), txt) in enumerate(zip(wins, wtexts)):
        for kind in ("visual", "speech"):
            if kind == "speech" and not (txt and txt.strip()):
                continue
            if kind == "visual" and args.frames == 0:
                continue
            q = act_question(i, len(wins), t1, t2, txt, kind)
            ids, _ = judge.branch_ids(msgs, q, history, head_text=head)
            out.append((i, kind, ids))
    return out, b0, a0


def step_grad(judge, cache, ids):
    """judge._step with gradients: src/mllm_judge.py decorates its own _step with @torch.no_grad()."""
    out = judge.model.model(input_ids=torch.tensor([ids], device=judge.device),
                            past_key_values=cache, use_cache=True)
    h = out.last_hidden_state[0, -1]
    del out
    return h


def crop_cache(cache, n_added):
    """Remove the n_added tokens the branch just appended (negative form; positive is deprecated)."""
    cache.crop(-int(n_added))
    return cache


def branch_margins(judge, cache, prefix_len, specs, grad):
    """Yes/No log-odds for every branch, each read on the prefix cache which is restored afterwards."""
    zs = []
    for _, _, ids in specs:
        if grad:
            with torch.enable_grad():
                h = step_grad(judge, cache, ids)
                zs.append(judge.margins_fp32_t(h[None])[0])
        else:
            h = judge._step(cache, ids)
            zs.append(judge.margins_fp32(h[None])[0])
        crop_cache(cache, len(ids))
    return zs


def window_scores(specs, zs, n_windows):
    """max over the branches of a window; windows with no branch get FILL_UNCOVERED."""
    per = {}
    for (i, _, _), z in zip(specs, zs):
        per[i] = z if i not in per else torch.maximum(per[i], z) if torch.is_tensor(z) else max(per[i], z)
    return [per.get(i, None) for i in range(n_windows)]


def mil_loss(scores, y, lam):
    s = torch.stack([x for x in scores if x is not None])
    if y == 1:
        loss = -torch.nn.functional.logsigmoid(s.max())
    else:
        loss = -torch.nn.functional.logsigmoid(-s).mean()
    return loss + lam * torch.sigmoid(s).mean()


def mil_loss_mean(scores, y, lam):
    """Control: mean-pool on positives, i.e. no within-video contrast."""
    s = torch.stack([x for x in scores if x is not None])
    if y == 1:
        loss = -torch.nn.functional.logsigmoid(s).mean()
    else:
        loss = -torch.nn.functional.logsigmoid(-s).mean()
    return loss + lam * torch.sigmoid(s).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260912_sdl")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pseudo-from", default=str(ROOT / "runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl"),
                    help="frozen run whose extra.z_video supplies the pseudo video labels")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--pool", choices=["max", "mean"], default="max")
    ap.add_argument("--grad-windows", type=int, default=4,
                    help="windows carried with gradient per step (positives use the argmax window only)")
    ap.add_argument("--shuffle-pseudo", action="store_true", help="control: permute the pseudo labels")
    ap.add_argument("--only-within-defined", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--method-name", default=None)
    args = ap.parse_args()
    torch.manual_seed(SEED)

    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    args.method_name = args.method_name or f"sdl_r{args.rank}_{args.pool}_lam{args.lam:g}_e{args.epochs}"
    cfg = dict(vars(args))
    cfg.update({"code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
                "seed": SEED, "targets": list(TARGETS)})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    rows = load_manifest(args.manifest, args.datasets)
    if args.only_within_defined:
        keep = within_defined_ids(args.datasets)
        rows = [r for r in rows if (r["dataset"], r["video_id"]) in keep]
    rows.sort(key=lambda r: (r["dataset"], r["video_id"]))
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}

    pseudo = {}
    for line in open(args.pseudo_from):
        r = json.loads(line)
        if r.get("extra") and r["extra"].get("z_video") is not None:
            pseudo[(r["dataset"], r["video_id"])] = float(r["extra"]["z_video"])
    rows = [r for r in rows if (r["dataset"], r["video_id"]) in pseudo]
    labels = {k: int(v > 0) for k, v in pseudo.items()}
    if args.shuffle_pseudo:
        rng = np.random.default_rng(SEED)
        for ds in args.datasets:
            ks = [k for k in labels if k[0] == ds]
            vals = [labels[k] for k in ks]
            rng.shuffle(vals)
            labels.update(dict(zip(ks, vals)))
    logging.info("%d videos, pseudo-positive rate %.3f", len(rows),
                 float(np.mean([labels[(r['dataset'], r['video_id'])] for r in rows])))

    judge = Judge(model_id=args.model)
    judge.margins_fp32_t = lambda h: (lambda lg, ny: torch.logsumexp(lg[:, :ny], 1) - torch.logsumexp(lg[:, ny:], 1))(
        judge._logits_fp32(h, torch.tensor(judge.yes_ids + judge.no_ids, device=judge.device)), len(judge.yes_ids))
    n_lora = inject_lora(judge.model, args.rank, args.alpha)
    params = [p for m in lora_modules(judge.model) for p in (m.A.weight, m.B.weight)]
    logging.info("LoRA on %d linears, %d trainable tensors, %.2fM params", n_lora, len(params),
                 sum(p.numel() for p in params) / 1e6)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    if args.train:
        step, t0 = 0, time.time()
        for ep in range(args.epochs):
            order = np.random.default_rng(SEED + ep).permutation(len(rows))
            for n, ix in enumerate(order):
                row = rows[int(ix)]
                key = (row["dataset"], row["video_id"])
                P = build_prefix(judge, row, asr[row["dataset"]].get(row["video_id"], []), args)
                if P is None:
                    continue
                with torch.no_grad():
                    set_lora(judge.model, False)
                    cache = judge.prefix_cache(P["enc"])
                    prefix_len = P["enc"]["input_ids"].shape[1]
                    b0, _ = judge.branch_ids(P["msgs"], VIDEO_QUESTION)
                    zv_frozen = judge.cached_margin(cache, b0, in_place=True)
                    stance = "Yes" if pseudo[key] > 0 else "No"
                    a0, _ = judge.answer_ids(P["msgs"], VIDEO_QUESTION, stance)
                    judge.extend_cache(cache, a0)
                    ctx_len = prefix_len + len(b0) + len(a0)
                    set_lora(judge.model, True)
                specs, _, _ = branch_specs(judge, P, stance, args)
                if not specs:
                    del cache
                    continue
                # Pass 1, no gradient: all window scores. Pass 2: the loss is a sum of per-window
                # terms, so each picked window is re-run with gradient and backpropagated on its own; only
                # one branch graph is alive at a time (8 graphs over a 3k-token cache OOM a 32G card).
                with torch.no_grad():
                    zs0 = branch_margins(judge, cache, ctx_len, specs, grad=False)
                s0 = window_scores(specs, zs0, len(P["wins"]))
                have = [i for i, v in enumerate(s0) if v is not None]
                y = labels[key]
                rng_w = np.random.default_rng(SEED * 7919 + step)
                if y == 1 and args.pool == "max":
                    top = max(have, key=lambda i: s0[i])
                    rest = [i for i in have if i != top]
                    k = min(args.grad_windows - 1, len(rest))
                    pick = [top] + (sorted(rng_w.choice(rest, size=k, replace=False).tolist()) if k > 0 else [])
                else:
                    top = None
                    k = min(args.grad_windows, len(have))
                    pick = sorted(rng_w.choice(have, size=k, replace=False).tolist())
                K = len(pick)
                total = 0.0
                for wi in pick:
                    wspecs = [sp for sp in specs if sp[0] == wi]
                    zsw = branch_margins(judge, cache, ctx_len, wspecs, grad=True)
                    sw = zsw[0]
                    for extra in zsw[1:]:
                        sw = torch.maximum(sw, extra)
                    if y == 1 and args.pool == "max":
                        term = args.lam * torch.sigmoid(sw) / K
                        if wi == top:
                            term = term - torch.nn.functional.logsigmoid(sw)
                    else:
                        term = (-torch.nn.functional.logsigmoid(-sw) + args.lam * torch.sigmoid(sw)) / K
                    (term / args.accum).backward()
                    total += float(term.detach())
                    del zsw, sw, term
                loss = total
                del cache, zs0, s0
                step += 1
                if step % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); opt.zero_grad(set_to_none=True)
                if n % 20 == 0:
                    logging.info("ep%d %d/%d %s y=%d zv=%.1f loss=%.4f %.0fs mem=%.1fG", ep, n, len(rows),
                                 row["video_id"], labels[key], zv_frozen, loss,
                                 time.time() - t0, torch.cuda.max_memory_allocated() / 1e9)
        torch.save({("%d" % i): {"A": m.A.weight.detach().cpu(), "B": m.B.weight.detach().cpu()}
                    for i, m in enumerate(lora_modules(judge.model))}, out_dir / "adapter.pt")
        logging.info("TRAINED %d steps in %.0fs", step, time.time() - t0)

    # ---- inference with the adapter active (SPVL-r2 otherwise unchanged)
    set_lora(judge.model, True)
    fh = open(out_dir / "predictions.jsonl", "w")
    t0 = time.time()
    for n, row in enumerate(rows):
        P = build_prefix(judge, row, asr[row["dataset"]].get(row["video_id"], []), args)
        if P is None:
            fh.write(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                                 "error": "no frames", "method": args.method_name}) + "\n")
            continue
        with torch.no_grad():
            cache = judge.prefix_cache(P["enc"])
            prefix_len = P["enc"]["input_ids"].shape[1]
            b0, _ = judge.branch_ids(P["msgs"], VIDEO_QUESTION)
            z_video = judge.cached_margin(cache, b0, in_place=True)
            stance = "Yes" if z_video > 0 else "No"
            a0, _ = judge.answer_ids(P["msgs"], VIDEO_QUESTION, stance)
            judge.extend_cache(cache, a0)
            ctx_len = prefix_len + len(b0) + len(a0)
            specs, _, _ = branch_specs(judge, P, stance, args)
            zs = branch_margins(judge, cache, ctx_len, specs, grad=False)
        per = []
        smax = {}
        for (i, kind, _), z in zip(specs, zs):
            smax.setdefault(i, {})[kind] = float(z)
        for i, (t1, t2) in enumerate(P["wins"]):
            d = smax.get(i, {})
            a = max(d.values()) if d else FILL_UNCOVERED
            per.append({"i": i, "start": t1, "end": t2, "a": a, "z": a,
                        **{f"z_{k}": v for k, v in d.items()}})
        del cache
        L = int(math.ceil(P["dur"] * FPS))
        centers = (np.arange(L) + 0.5) / FPS
        idx = np.minimum((centers // args.window_seconds).astype(int), len(P["wins"]) - 1)
        curve = np.asarray([p["a"] for p in per], float)[idx]
        fh.write(json.dumps({"schema_version": 1, "method": args.method_name, "dataset": row["dataset"],
                             "video_id": row["video_id"], "duration": P["dur"], "native_rate": FPS,
                             "score_curve": [float(x) for x in curve], "intervals": [], "error": None,
                             "calls": 2, "seed": SEED, "code_path": CODE_PATH,
                             "extra": {"z_video": z_video, "stance": stance, "windows": per,
                                       "n_windows": len(P["wins"])}}, ensure_ascii=False) + "\n")
        fh.flush()
        if n % 20 == 0:
            logging.info("infer %d/%d %s %.0fs", n, len(rows), row["video_id"], time.time() - t0)
    fh.close()
    logging.info("DONE inference %d videos in %.0fs", len(rows), time.time() - t0)


if __name__ == "__main__":
    main()
