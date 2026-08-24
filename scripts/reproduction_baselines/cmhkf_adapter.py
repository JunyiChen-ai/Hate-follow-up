#!/usr/bin/env python3
"""CMHKF on hateful-video corpora with official validation selection."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UPSTREAM = REPO / "third_party" / "CMHKF"
if not UPSTREAM.is_dir():
    UPSTREAM = REPO / "third_party" / "cmhkf"
sys.path.insert(0, str(UPSTREAM / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "duplex"))

from model import CMHKF  # type: ignore  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from frame_eval_common import average_precision  # noqa: E402

PROMPTS = ["normal content", "hateful content"]
VGG_ROOT = REPO / "results" / "reproduction" / "features" / "vggish_1s"


class AVData(Dataset):
    def __init__(self, corpus, ids, labels, length):
        self.corpus, self.ids, self.labels, self.length = corpus, list(ids), labels, length
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        vid = self.ids[i]
        v = np.load(hdata.feature_path(self.corpus, vid)).astype(np.float32)
        a = np.load(VGG_ROOT / self.corpus / f"{vid}.npy").astype(np.float32)
        if len(v) != len(a): raise ValueError(f"{vid}: CLIP {len(v)} != VGGish {len(a)}")
        x, n = hdata.tools.process_feat(np.concatenate([v, a], -1), self.length)
        return torch.from_numpy(x), int(self.labels[vid]), n


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def onehot(labels, device):
    return hdata.label_vectors(labels, device)


def reuse_identical_text_prompt_pass(model):
    """Avoid CMHKF's duplicate, mathematically identical CLIP text pass."""
    original_begin = model.encode_textprompt_begin

    def encode_begin(this, text):
        result = original_begin(text)
        this._reproduction_prompt_cache = result
        return result

    def encode_reuse(this, text):
        result = this._reproduction_prompt_cache
        del this._reproduction_prompt_cache
        return result

    model.encode_textprompt_begin = types.MethodType(encode_begin, model)
    model.encode_textprompt = types.MethodType(encode_reuse, model)


def score_ids(model, corpus, ids, length, device, batch_size=32):
    out, blocks_all, lengths_all, owners, sizes = {}, [], [], [], {}
    model.eval()
    for vid in ids:
        v = np.load(hdata.feature_path(corpus, vid)).astype(np.float32)
        a = np.load(VGG_ROOT / corpus / f"{vid}.npy").astype(np.float32)
        n = len(v)
        blocks, _ = hdata.tools.process_split(np.concatenate([v, a], -1), length)
        # process_split returns [T, D] when the video fits in one block,
        # but CMHKF consistently expects a batched [B, T, D] tensor.
        if blocks.ndim == 2:
            blocks = blocks[None, ...]
        lens = runtime.chunk_lengths(n, length).cpu().tolist()
        blocks_all.extend(blocks)
        lengths_all.extend(lens)
        owners.extend([vid] * len(blocks))
        sizes[vid] = n
        out[vid] = {"score_mil": [], "score_align": []}
    with torch.no_grad():
        for start in range(0, len(blocks_all), batch_size):
            stop = min(start + batch_size, len(blocks_all))
            feat = torch.from_numpy(np.stack(blocks_all[start:stop])).to(device)
            lens = torch.tensor(lengths_all[start:stop], device=device)
            _, l1, l2, _, _ = model(feat, None, PROMPTS, lens)
            mil = torch.sigmoid(l1).reshape(stop - start, -1).float().cpu().numpy()
            align = (1 - l2.softmax(-1)[..., 0]).reshape(stop - start, -1).float().cpu().numpy()
            for i, vid in enumerate(owners[start:stop]):
                out[vid]["score_mil"].append(mil[i])
                out[vid]["score_align"].append(align[i])
    for vid in ids:
        for key in ("score_mil", "score_align"):
            out[vid][key] = np.concatenate(out[vid][key])[:sizes[vid]]
    return out


def val_ap(model, corpus, ids, labels, length, device):
    scores = score_ids(model, corpus, ids, length, device)
    ordered = sorted(ids)
    return average_precision([float(scores[v]["score_align"].max()) for v in ordered],
                             [labels[v] for v in ordered])


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epoch", type=int, default=10)
    ap.add_argument("--visual-length", type=int, default=None)
    ap.add_argument("--attn-window", type=int, default=None)
    ap.add_argument("--prompt-prefix", type=int, default=10)
    ap.add_argument("--prompt-postfix", type=int, default=10)
    ap.add_argument("--loss-mil", type=float, default=1.0)
    ap.add_argument("--loss-align", type=float, default=1.0)
    ap.add_argument("--loss-text", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-test", action="store_true")
    args = ap.parse_args(argv)

    seed_all(args.seed); device = torch.device(args.device)
    length = args.visual_length or runtime.default_visual_length(args.corpus)
    window = args.attn_window or runtime.default_attn_window(args.corpus)
    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(args.corpus, labels)
    loader = DataLoader(AVData(args.corpus, train_ids, labels, length),
                        batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    model = CMHKF(2, 512, length, 512, 1, 1, window,
                  args.prompt_prefix, args.prompt_postfix, str(device)).to(device)
    reuse_identical_text_prompt_pass(model)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    sched = MultiStepLR(opt, [3, 6, 10], 0.1)
    best_ap, best_epoch, best_state, history = -1.0, None, None, []
    started = time.time()
    for epoch in range(1, args.max_epoch + 1):
        model.train(); total = 0.0
        for feat, label, lengths in loader:
            feat, label, lengths = feat.to(device), label.to(device), lengths.to(device)
            text, l1, l2, _, _ = model(feat, None, PROMPTS, lengths)
            targets = onehot(label, device)
            loss1 = runtime.CLAS2(l1, targets, lengths, device)
            loss2 = runtime.CLASM(l2, targets, lengths, device)
            normal = text[0] / text[0].norm()
            hateful = text[1] / text[1].norm()
            loss_text = 1 + normal @ hateful
            loss = args.loss_mil * loss1 + args.loss_align * loss2 + args.loss_text * loss_text
            if not torch.isfinite(loss).item():
                raise RuntimeError("CMHKF training loss became non-finite")
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach())
        sched.step()
        score = val_ap(model, args.corpus, val_ids, labels, length, device)
        if not np.isfinite(score):
            raise RuntimeError(
                f"CMHKF validation AP became non-finite at epoch {epoch}")
        history.append({"epoch": epoch, "loss": total / max(len(loader), 1),
                        "val_video_ap": score})
        print(f"epoch {epoch}/{args.max_epoch} loss={history[-1]['loss']:.4f} val_ap={score:.4f}", flush=True)
        if score > best_ap:
            best_ap, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("CMHKF produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pth")
    meta = {"method": "cmhkf", "protocol": "official-val",
            "upstream": "ssp-seven/CMHKF@3b07707", "args": vars(args),
            "visual_length": length, "attn_window": window,
            "train_ids": train_ids, "val_ids": val_ids,
            "selected_epoch": best_epoch, "selected_val_video_ap": best_ap,
            "history": history, "wall_seconds": time.time() - started}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.run_test:
        gt = hdata.gt_arrays(args.corpus)
        test_ids = [v for v in hdata.load_split(args.corpus, "test") if v in gt]
        scores = score_ids(model, args.corpus, test_ids, length, device)
        with (out / "scores.jsonl").open("w") as fh:
            for vid in test_ids:
                fh.write(json.dumps({"video_id": vid,
                                     **{k: x.tolist() for k, x in scores[vid].items()}}) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
