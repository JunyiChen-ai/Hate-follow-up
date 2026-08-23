"""VadCLIP training loop, ported from VadCLIP @ c41067f src/xd_train.py.

What is kept
    the two MIL losses (CLAS2, CLASM) verbatim, in hate_common.runtime;
    the text-orthogonality term, generalised from /6 to /(num_class - 1);
    the loss sum loss1 + loss2 + 1e-4 * loss3;
    AdamW with MultiStepLR at milestones [3, 6, 10], rate 0.1, lr 1e-5.

What changed
    patch V3  Upstream evaluates the *test* set after every epoch and keeps the
              best-AP checkpoint, i.e. it selects on the test set. This port
              never opens the test split during training. It carves a
              stratified validation subset out of the train split and selects
              on video-level average precision there; --val-frac 0 --select last
              disables selection entirely.
    patch V4  Upstream reloads the best checkpoint from disk at the end of every
              epoch, which throws away the epoch's optimisation state whenever
              the epoch did not improve. That silently turns training into a
              random restart search. The best state is kept in memory here and
              restored once, after the last epoch.
    patch V5  Upstream logs on `step % 4800 == 0` where step is a stale local
              recomputed each iteration; the logging is replaced by a
              per-epoch summary line.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hate_common import data as hdata          # noqa: E402
from hate_common import runtime                # noqa: E402
from vadclip.model import CLIPVAD              # noqa: E402
from vadclip import option                     # noqa: E402

sys.path.insert(0, os.path.join(hdata.REPO_ROOT, "scripts", "duplex"))
import frame_eval_common as fec                # noqa: E402


def evaluate_video_ap(model, loader, prompt_text, device):
    """Video-level average precision of the MIL bag score on a held-out set."""
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for visual, label, lengths in loader:
            visual = visual.to(device)
            lengths = lengths.to(device)
            _, logits1, _ = model(visual, None, prompt_text, lengths)
            scores.extend(runtime.video_scores_from_logits(logits1, lengths))
            labels.extend(int(x) for x in label)
    model.train()
    if not scores or len(set(labels)) < 2:
        return None
    return fec.average_precision(np.asarray(scores), np.asarray(labels))


def train(args):
    device = args.device
    runtime.setup_seed(args.seed)

    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(
        args.corpus, labels, args.val_frac, args.seed)

    if args.limit_videos:
        train_ids = train_ids[:args.limit_videos]
        val_ids = val_ids[:max(args.limit_videos // 4, 2)]

    train_set = hdata.HateVideoDataset(args.corpus, train_ids,
                                       args.visual_length, False, labels)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=False)
    val_loader = None
    if val_ids:
        val_set = hdata.HateVideoDataset(args.corpus, val_ids,
                                         args.visual_length, False, labels)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    out_dir = args.out_dir or os.path.dirname(
        runtime.scores_out_path("vadclip", args.corpus))
    os.makedirs(out_dir, exist_ok=True)

    print("corpus            %s" % args.corpus)
    print("train / val       %d / %d videos (%d hateful in train)"
          % (len(train_ids), len(val_ids),
             sum(labels[v] for v in train_ids)))
    print("visual-length     %d rows = %d s per block, attn-window %d"
          % (args.visual_length, args.visual_length, args.attn_window))
    print("device            %s" % device)
    sys.stdout.flush()

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device, clip_download_root=args.clip_download_root)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones,
                            args.scheduler_rate)
    prompt_text = list(hdata.PROMPT_TEXT)

    best_ap, best_state, best_epoch = -1.0, None, -1
    history = []

    for e in range(args.max_epoch):
        model.train()
        t0 = time.time()
        totals = np.zeros(3)
        n_batches = 0
        for visual_feat, label, feat_lengths in train_loader:
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = hdata.label_vectors(label, device)

            text_features, logits1, logits2 = model(
                visual_feat, None, prompt_text, feat_lengths)

            loss1 = runtime.CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = runtime.CLASM(logits2, text_labels, feat_lengths, device)
            loss3 = runtime.text_orthogonality_loss(text_features, device)
            loss = loss1 + loss2 + loss3 * args.loss3_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += [loss1.item(), loss2.item(), loss3.item()]
            n_batches += 1

        scheduler.step()
        totals /= max(n_batches, 1)
        val_ap = (evaluate_video_ap(model, val_loader, prompt_text, device)
                  if val_loader is not None else None)
        history.append({"epoch": e + 1, "loss1": totals[0], "loss2": totals[1],
                        "loss3": totals[2], "val_video_ap": val_ap,
                        "seconds": round(time.time() - t0, 1)})
        print("epoch %2d | loss1 %.4f | loss2 %.4f | loss3 %.4f | val AP %s "
              "| %.0fs" % (e + 1, totals[0], totals[1], totals[2],
                           ("%.4f" % val_ap) if val_ap is not None else "n/a",
                           time.time() - t0))
        sys.stdout.flush()

        if args.select == "val" and val_ap is not None and val_ap > best_ap:
            best_ap, best_epoch = val_ap, e + 1
            best_state = copy.deepcopy(model.state_dict())

    if args.select == "val" and best_state is not None:
        model.load_state_dict(best_state)
        print("selected epoch %d (val video AP %.4f)" % (best_epoch, best_ap))
    else:
        print("selected the last epoch (no validation selection)")

    model_path = os.path.join(out_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    meta = {
        "method": "vadclip",
        "upstream": "https://github.com/nwpu-zxr/VadCLIP @ c41067f",
        "args": {k: v for k, v in vars(args).items()},
        "train_ids": train_ids,
        "val_ids": val_ids,
        "selected_epoch": best_epoch if args.select == "val" else args.max_epoch,
        "selected_val_video_ap": best_ap if best_ap >= 0 else None,
        "history": history,
        "class_prompts": prompt_text,
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote %s" % model_path)
    return model_path


def main(argv=None):
    args = option.resolve(option.build_parser().parse_args(argv))
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
