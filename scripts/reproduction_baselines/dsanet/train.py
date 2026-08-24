"""DSANet training loop, ported from DSANet @ eb335b2 src/xd_train.py.

What is kept
    all five losses and the reconstruction consistency term, verbatim;
    the loss sum loss1 + 5.0*loss2 + loss3 + loss4 + loss5 + consistency + g;
    two optimisers, StableAdamW on the refiner with a warm-cosine schedule
    (warmup 100 iters, final lr = 0.1 * base) and AdamW with CosineAnnealingLR
    on everything else;
    lr 1e-5, batch size 96, 10 epochs, seed 234.

What changed
    patch V3, V4, V5  the same three changes as the VadCLIP port: validation
                      selection instead of test selection, no per-epoch reload
                      of the best checkpoint, per-epoch logging.
    patch T1          the text-orthogonality term divides by num_class - 1
                      rather than the literal 6.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hate_common import data as hdata          # noqa: E402
from hate_common import runtime                # noqa: E402
from dsanet.model import DSANet                # noqa: E402
from dsanet.StableAdamW import StableAdamW     # noqa: E402
from dsanet import option                      # noqa: E402

sys.path.insert(0, os.path.join(hdata.REPO_ROOT, "scripts", "duplex"))
import frame_eval_common as fec                # noqa: E402


# ------------------------------------------------------- upstream, verbatim
def CLASM_EVENT(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)
    labels_sum = labels.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels_sm = (1 - epsilon) * (labels / labels_sum) + epsilon / num_classes
    labels_sm = labels_sm.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(1),
                            largest=True, dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(
        torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0)


def CLASM_BKG(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    labels2 = torch.full(labels.shape, 0.01, device=labels.device)
    labels2[:, 0] = 1
    labels2_sum = labels2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels2 = (1 - epsilon) * (labels2 / labels2_sum) + epsilon / num_classes
    labels2 = labels2.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(1),
                            largest=True, dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(
        torch.sum(labels2 * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0)


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters,
                 warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)
        iters = np.arange(max(total_iters - warmup_iters, 1))
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= len(self.schedule):
            return [self.final_value for _ in self.base_lrs]
        return [self.schedule[self.last_epoch] for _ in self.base_lrs]


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, logits1, original_features, reconstructed_features,
                lengths):
        recon_error_score = 1.0 - F.cosine_similarity(
            original_features, reconstructed_features, dim=-1)
        recon_error_score = recon_error_score / 2.0
        classifier_prob_score = torch.sigmoid(logits1.squeeze(-1))
        N = logits1.shape[1]
        mask = torch.arange(N, device=logits1.device)[None, :] < lengths[:, None]
        return self.mse_loss(classifier_prob_score[mask],
                             recon_error_score[mask])


consistency_loss_fn = ConsistencyLoss()


# ------------------------------------------------------------------- train
def evaluate_video_ap(model, loader, prompt_text, dnp_use, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for visual, label, lengths in loader:
            visual = visual.to(device)
            lengths = lengths.to(device)
            out = model(visual, None, prompt_text, lengths, dnp_use)
            logits1 = out[1]
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
                              shuffle=True, num_workers=args.num_workers)
    val_loader = None
    if val_ids:
        val_set = hdata.HateVideoDataset(args.corpus, val_ids,
                                         args.visual_length, False, labels)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    out_dir = args.out_dir or os.path.dirname(
        runtime.scores_out_path("dsanet", args.corpus))
    os.makedirs(out_dir, exist_ok=True)

    print("corpus            %s" % args.corpus)
    print("train / val       %d / %d videos (%d hateful in train)"
          % (len(train_ids), len(val_ids), sum(labels[v] for v in train_ids)))
    print("visual-length     %d rows = %d s per block, attn-window %d"
          % (args.visual_length, args.visual_length, args.attn_window))
    print("device            %s | DNP %s" % (device, args.DNP_use))
    sys.stdout.flush()

    model = DSANet(args.classes_num, args.embed_dim, args.visual_length,
                   args.visual_width, args.visual_head, args.visual_layers,
                   args.attn_window, args.prompt_prefix, args.prompt_postfix,
                   args, device, clip_download_root=args.clip_download_root)
    model.to(device)

    refiner_params, main_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (refiner_params if "video_anomaly_refiner" in name
         else main_params).append(param)

    optimizer_refiner = StableAdamW([{"params": refiner_params}], lr=args.lr,
                                    betas=(0.9, 0.999), weight_decay=1e-4,
                                    amsgrad=True, eps=1e-10)
    total_iters = args.max_epoch * max(len(train_loader), 1)
    scheduler_refiner = WarmCosineScheduler(
        optimizer_refiner, base_value=args.lr, final_value=args.lr * 0.1,
        total_iters=total_iters,
        warmup_iters=min(args.warmup_iters, max(total_iters - 1, 0)))
    optimizer_main = torch.optim.AdamW([{"params": main_params}], lr=args.lr)
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=args.max_epoch)

    prompt_text = list(hdata.PROMPT_TEXT)
    dnp_use = bool(args.DNP_use)
    best_ap, best_state, best_epoch = -1.0, None, -1
    history = []
    stopped_nonfinite = False

    for e in range(args.max_epoch):
        model.train()
        t0 = time.time()
        totals = np.zeros(7)
        n_batches = 0
        for visual_feat, label, feat_lengths in train_loader:
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = hdata.label_vectors(label, device)

            out = model(visual_feat, None, prompt_text, feat_lengths, dnp_use)
            if dnp_use:
                text_features, logits1, logits2, logits3, logits4, DNP = out
            else:
                text_features, logits1, logits2, logits3, logits4 = out
                DNP = None

            # Some aggressive configurations remain useful for many epochs
            # before an optimiser update makes the next forward pass NaN.  Do
            # not feed those values into CUDA BCE (which poisons the process
            # with a device-side assert): stop cleanly and retain the best
            # validation-selected state accumulated so far.
            tensors = (text_features, logits1, logits2, logits3, logits4)
            if not all(torch.isfinite(x).all().item() for x in tensors):
                stopped_nonfinite = True
                print("non-finite model output; stopping early and retaining "
                      "the best validation checkpoint")
                sys.stdout.flush()
                break

            loss1 = runtime.CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = runtime.CLASM(logits2, text_labels, feat_lengths, device)
            loss3 = runtime.text_orthogonality_loss(text_features, device)
            loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
            loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)

            if dnp_use:
                closs = consistency_loss_fn(
                    logits1=logits1,
                    original_features=DNP["original_features"],
                    reconstructed_features=DNP["reconstructed_features"],
                    lengths=feat_lengths)
                gloss = DNP["g_loss"]
                loss = (loss1 + loss2 * args.loss2_weight + loss3 + loss4
                        + loss5 + closs + gloss)
            else:
                closs = torch.zeros(())
                gloss = torch.zeros(())
                loss = loss1 + loss2 + loss3 + loss4 + loss5

            if not torch.isfinite(loss).all().item():
                stopped_nonfinite = True
                print("non-finite loss; stopping early and retaining the best "
                      "validation checkpoint")
                sys.stdout.flush()
                break

            optimizer_main.zero_grad()
            optimizer_refiner.zero_grad()
            loss.backward()
            optimizer_main.step()
            optimizer_refiner.step()
            scheduler_refiner.step()

            totals += [loss1.item(), loss2.item(), loss3.item(), loss4.item(),
                       loss5.item(), closs.item(), gloss.item()]
            n_batches += 1

        if stopped_nonfinite:
            break

        scheduler_main.step()
        totals /= max(n_batches, 1)
        val_ap = (evaluate_video_ap(model, val_loader, prompt_text, dnp_use,
                                    device) if val_loader is not None else None)
        history.append({
            "epoch": e + 1, "loss1": totals[0], "loss2": totals[1],
            "loss3": totals[2], "loss4": totals[3], "loss5": totals[4],
            "consistency": totals[5], "g_loss": totals[6],
            "val_video_ap": val_ap, "seconds": round(time.time() - t0, 1)})
        print("epoch %2d | l1 %.4f l2 %.4f l3 %.4f l4 %.4f l5 %.4f "
              "cons %.4f g %.4f | val AP %s | %.0fs"
              % (e + 1, totals[0], totals[1], totals[2], totals[3], totals[4],
                 totals[5], totals[6],
                 ("%.4f" % val_ap) if val_ap is not None else "n/a",
                 time.time() - t0))
        sys.stdout.flush()

        if args.select == "val" and val_ap is not None and val_ap > best_ap:
            best_ap, best_epoch = val_ap, e + 1
            best_state = copy.deepcopy(model.state_dict())

    if stopped_nonfinite and best_state is None:
        raise RuntimeError(
            "DSA-Net became non-finite before producing a valid checkpoint")

    if args.select == "val" and best_state is not None:
        model.load_state_dict(best_state)
        print("selected epoch %d (val video AP %.4f)" % (best_epoch, best_ap))
    else:
        print("selected the last epoch (no validation selection)")

    model_path = os.path.join(out_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(out_dir, "train_meta.json"), "w") as fh:
        json.dump({
            "method": "dsanet",
            "upstream": "https://github.com/lessiYin/DSANet @ eb335b2",
            "args": {k: v for k, v in vars(args).items()},
            "train_ids": train_ids,
            "val_ids": val_ids,
            "selected_epoch": (best_epoch if args.select == "val"
                               else args.max_epoch),
            "selected_val_video_ap": best_ap if best_ap >= 0 else None,
            "stopped_nonfinite": stopped_nonfinite,
            "history": history,
            "class_prompts": prompt_text,
        }, fh, indent=2)
    print("wrote %s" % model_path)
    return model_path


def main(argv=None):
    args = option.resolve(option.build_parser().parse_args(argv))
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
