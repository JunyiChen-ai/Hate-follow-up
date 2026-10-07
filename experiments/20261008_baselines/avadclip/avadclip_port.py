#!/usr/bin/env python3
"""AVadCLIP (Wu, Su, Pang et al., IEEE TMM 2026; github.com/RowanSu/AVadCLIP @ d3f6e16) on the hateful-video corpora.

The audio-visual model (`model_t.AVADCLIP`, the paper's audio-visual detector; the repo calls it the teacher) and the
two MIL losses (`CLAS2`, `CLASM` with the learned focal alpha) are imported unchanged from third_party/AVadCLIP/src.
This file replaces only the data plumbing and the training loop's selection, in the same way the repository's
VadCLIP port does (scripts/reproduction_baselines/vadclip/train.py, patches V3-V5):

  inputs   visual = CLIP ViT-B/16 image embeddings, one row per second (data/weaksup_1fps/clip_b16_1fps);
           audio  = Wav2CLIP, one row per second, same rows (data/wav2clip_1s); upstream uses CLIP and Wav2CLIP per
           16-frame snippet, here one row is one second
  classes  two, ["normal content", "hateful content"] (the VadCLIP port's prompts); the text-orthogonality term is
           divided by num_class - 1 instead of the literal 6 (VadCLIP port patch T1)
  V3       upstream evaluates the test set every epoch and keeps the best test AP. That is removed.
           --select last (used for all reported runs): train the preset's fixed epochs, keep the last epoch, and do
           not read the validation split.  --select val (not used): keep the epoch with the best validation video
           AP (bag score = mean of the top floor(T/16)+1 sigmoid(logits1) rows vs video-level val labels).
           Test labels are never read.
  V4       upstream reloads the best checkpoint after every epoch; that reload is removed (with --select val the best
           state is kept in memory and restored once at the end)
  V5       per-epoch log line instead of the step % 4800 print
Scores written per test video (1 fps): score_align = 1 - softmax(logits2)[:, 0] (upstream's AP2, the number it
selects on), score_mlp = sigmoid(logits1) (upstream's AUC1).

    python experiments/20261008_baselines/avadclip/avadclip_port.py train --corpus hatemm --out-dir ... [opts]
    python experiments/20261008_baselines/avadclip/avadclip_port.py infer --corpus hatemm --out-dir ... [opts]
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "weaksup_common"))
import common as C  # noqa: E402

UPSTREAM = C.REPO / "third_party" / "AVadCLIP" / "src"
sys.path.insert(0, str(UPSTREAM))
import model_t  # noqa: E402
from utils import tools as avtools  # noqa: E402
from xd_train_t import CLAS2, CLASM  # noqa: E402

sys.path.insert(0, str(C.REPO / "scripts" / "duplex"))
import frame_eval_common as fec  # noqa: E402

PROMPT_TEXT = ["normal content", "hateful content"]
AUDIO = C.REPO / "data" / "wav2clip_1s"


def feature(kind, corpus, vid):
    root = C.INPUTS / "clip_b16_1fps" if kind == "visual" else AUDIO
    return np.load(root / corpus / f"{vid}.npy").astype(np.float32)


class AVDataset(Dataset):
    def __init__(self, corpus, ids, length, test_mode, labels):
        self.corpus, self.ids, self.length, self.test_mode, self.labels = corpus, list(ids), int(length), test_mode, labels
        for v in self.ids:
            for kind, root in (("visual", C.INPUTS / "clip_b16_1fps"), ("audio", AUDIO)):
                if not (root / corpus / f"{v}.npy").is_file():
                    raise FileNotFoundError(f"{kind} feature missing for {corpus}/{v}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        vis, aud = feature("visual", self.corpus, vid), feature("audio", self.corpus, vid)
        if vis.shape[0] != aud.shape[0]:
            raise ValueError(f"{vid}: {vis.shape[0]} visual rows vs {aud.shape[0]} audio rows")
        proc = avtools.process_split if self.test_mode else avtools.process_feat
        vis, n = proc(vis, self.length)
        aud, _ = proc(aud, self.length)
        item = (torch.from_numpy(np.ascontiguousarray(vis)), torch.from_numpy(np.ascontiguousarray(aud)),
                int(self.labels[vid]), int(n))
        return item + (vid,) if self.test_mode else item


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("train", "infer"))
    p.add_argument("--corpus", required=True, choices=C.CORPORA)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=234, type=int)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--clip-download-root", default=None)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--limit-videos", default=0, type=int, help="debug only")
    p.add_argument("--select", default="last", choices=("last", "val"))
    # architecture: upstream xd_option.py defaults (classes 7 -> 2)
    p.add_argument("--classes-num", default=2, type=int)
    p.add_argument("--embed-dim", default=512, type=int)
    p.add_argument("--visual-length", default=256, type=int, help="rows per block (one row = one second)")
    p.add_argument("--visual-width", default=512, type=int)
    p.add_argument("--visual-head", default=1, type=int)
    p.add_argument("--visual-layers", default=2, type=int)
    p.add_argument("--audio-width", default=512, type=int)
    p.add_argument("--attn-window", default=4, type=int)
    p.add_argument("--prompt-prefix", default=10, type=int)
    p.add_argument("--prompt-postfix", default=10, type=int)
    # optimisation: upstream defaults
    p.add_argument("--max-epoch", default=10, type=int)
    p.add_argument("--batch-size", default=96, type=int)
    p.add_argument("--lr", default=1e-5, type=float)
    p.add_argument("--scheduler-rate", default=0.1, type=float)
    p.add_argument("--scheduler-milestones", default=[3, 6, 10], type=int, nargs="+")
    p.add_argument("--loss3-weight", default=1e-4, type=float)
    return p


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_model(args):
    if args.clip_download_root:
        model_t.clip.load = functools.partial(model_t.clip.load, download_root=args.clip_download_root)
    return model_t.AVADCLIP(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                            args.visual_head, args.visual_layers, args.visual_length, args.audio_width,
                            args.attn_window, args.prompt_prefix, args.prompt_postfix, args.device)


def onehot(labels, device):
    idx = torch.as_tensor(labels, dtype=torch.long)
    if (idx < 0).any():
        raise RuntimeError("a training item has no video label")
    out = torch.zeros(len(idx), 2)
    out[torch.arange(len(idx)), idx] = 1.0
    return out.to(device)


def orthogonality(text_features, device):
    loss = torch.zeros(1).to(device)
    normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        loss = loss + torch.abs(normal @ (text_features[j] / text_features[j].norm(dim=-1, keepdim=True)))
    return loss / max(text_features.shape[0] - 1, 1)


@torch.no_grad()
def val_video_ap(model, loader, device):
    model.eval()
    scores, labels = [], []
    for vis, aud, lab, lengths in loader:
        lengths = lengths.to(device)
        _, _, logits1, _, _ = model(vis.to(device), aud.to(device), None, None, PROMPT_TEXT, lengths)
        probs = torch.sigmoid(logits1).reshape(logits1.shape[0], logits1.shape[1])
        for i in range(probs.shape[0]):
            n = int(lengths[i])
            top, _ = torch.topk(probs[i, :n], k=int(n / 16 + 1))
            scores.append(float(top.mean()))
        labels.extend(int(x) for x in lab)
    model.train()
    if len(set(labels)) < 2:
        return None
    return float(fec.average_precision(np.asarray(scores), np.asarray(labels)))


def train(args):
    device = args.device
    setup_seed(args.seed)
    labels = C.train_val_labels(args.corpus)
    train_ids = C.split_ids(args.corpus, "train")
    val_ids = C.split_ids(args.corpus, "val") if args.select == "val" else []
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train/val overlap")
    if args.limit_videos:
        train_ids, val_ids = train_ids[:args.limit_videos], val_ids[:max(args.limit_videos // 4, 2)]
    train_loader = DataLoader(AVDataset(args.corpus, train_ids, args.visual_length, False, labels),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(AVDataset(args.corpus, val_ids, args.visual_length, False, labels),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ids else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"corpus {args.corpus}: train {len(train_ids)} ({sum(labels[v] for v in train_ids)} hateful), "
          f"val {len(val_ids)}; visual-length {args.visual_length}, attn-window {args.attn_window}", flush=True)
    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    best_ap, best_state, best_epoch, history = -1.0, None, -1, []
    for e in range(args.max_epoch):
        model.train()
        t0, totals, nb = time.time(), np.zeros(3), 0
        for vis, aud, lab, lengths in train_loader:
            vis, aud, lengths = vis.to(device), aud.to(device), lengths.to(device)
            text_labels = onehot(lab, device)
            _, text_features, logits1, logits2, alpha = model(vis, aud, None, None, PROMPT_TEXT, lengths)
            loss1 = CLAS2(logits1, text_labels, lengths, device)
            loss2 = CLASM(logits2, text_labels, lengths, device, alpha)
            loss3 = orthogonality(text_features, device)
            loss = loss1 + loss2 + loss3 * args.loss3_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += [loss1.item(), loss2.item(), loss3.item()]
            nb += 1
        scheduler.step()
        totals /= max(nb, 1)
        ap = val_video_ap(model, val_loader, device) if val_loader is not None else None
        history.append({"epoch": e + 1, "loss1": totals[0], "loss2": totals[1], "loss3": totals[2],
                        "val_video_ap": ap, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {e + 1:2d} | loss1 {totals[0]:.4f} | loss2 {totals[1]:.4f} | loss3 {totals[2]:.4f} | "
              f"val AP {ap if ap is None else round(ap, 4)} | {time.time() - t0:.0f}s", flush=True)
        if not np.all(np.isfinite(totals)):
            print("non-finite loss; stopping and keeping the best validation state", flush=True)
            break
        if args.select == "val" and ap is not None and ap > best_ap:
            best_ap, best_epoch, best_state = ap, e + 1, copy.deepcopy(model.state_dict())
    if args.select == "val":
        if best_state is None:
            raise RuntimeError("no validation-selected state")
        model.load_state_dict(best_state)
        print(f"selected epoch {best_epoch} (val video AP {best_ap:.4f})", flush=True)
    else:
        if not np.all(np.isfinite(totals)):
            raise RuntimeError("last epoch has a non-finite loss")
        best_epoch, best_ap = len(history), None
        print(f"kept the last epoch ({best_epoch}); validation not used", flush=True)
    # the frozen CLIP (clipmodel.*, ~600 MB) is not saved; it is reloaded from ViT-B-16.pt when the model is built
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("clipmodel.")}, out_dir / "model.pth")
    meta = {"method": "avadclip", "upstream": "https://github.com/RowanSu/AVadCLIP @ d3f6e16 (model_t.AVADCLIP)",
            "args": vars(args), "train_ids": train_ids, "val_ids": val_ids, "selected_epoch": best_epoch,
            "select": args.select, "selected_val_video_ap": best_ap, "history": history, "class_prompts": PROMPT_TEXT}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


@torch.no_grad()
def infer(args):
    device = args.device
    ids = C.split_ids(args.corpus, "test")
    if args.limit_videos:
        ids = ids[:args.limit_videos]
    labels = {v: C.TEST_LABEL_PLACEHOLDER for v in ids}
    loader = DataLoader(AVDataset(args.corpus, ids, args.visual_length, True, labels), batch_size=1,
                        shuffle=False, num_workers=args.num_workers)
    model = make_model(args)
    state = torch.load(args.model_path or Path(args.out_dir) / "model.pth", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("clipmodel.") for k in missing):
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.to(device).eval()
    maxlen = args.visual_length
    out = Path(args.out_dir) / "scores.jsonl"
    n = 0
    with out.open("w") as fh:
        for vis, aud, _lab, length, vid in loader:
            len_cur = int(length)
            vis, aud = vis.squeeze(0), aud.squeeze(0)
            if len_cur < maxlen:
                vis, aud = vis.unsqueeze(0), aud.unsqueeze(0)
            lengths = torch.zeros(int(len_cur / maxlen) + 1)          # upstream xd_test_t chunk bookkeeping
            rest = len_cur
            for j in range(len(lengths)):
                if j == 0 and rest < maxlen:
                    lengths[j] = rest
                elif j == 0 and rest > maxlen:
                    lengths[j] = maxlen
                    rest -= maxlen
                elif rest > maxlen:
                    lengths[j] = maxlen
                    rest -= maxlen
                else:
                    lengths[j] = rest
            lengths = lengths.to(int)
            _, _, logits1, logits2, _ = model(vis.to(device), aud.to(device), None, None, PROMPT_TEXT, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])
            s_mlp = torch.sigmoid(logits1[:len_cur].squeeze(-1))
            s_align = 1 - F.softmax(logits2[:len_cur], dim=-1)[:, 0]
            if s_mlp.shape[0] != len_cur:
                raise RuntimeError(f"{vid[0]}: {s_mlp.shape[0]} rows, expected {len_cur}")
            fh.write(json.dumps({"video_id": vid[0], "n_frames": len_cur,
                                 "score_align": [round(float(x), 6) for x in s_align.cpu()],
                                 "score_mlp": [round(float(x), 6) for x in s_mlp.cpu()]}) + "\n")
            n += 1
    print(f"wrote {n} videos to {out}", flush=True)


def main():
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.mode == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
