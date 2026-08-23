"""MACIL-SD training, from MACIL_SD @ c20943f main.py + train.py.

What is kept
    the loss sum `clsloss + lamda_a2b * (a2v_a2b + v2a_a2b)
                          + lamda_a2n * (a2v_a2n + v2a_a2n)`;
    CMAL and its self-guided branch on `mmil_logits > 0.5`, verbatim;
    the two-optimiser alternation, Adam at lr for the audio-visual model and
    lr/5 for the uni-modal partner, both on CosineAnnealingLR(T_max=60);
    the per-epoch lambda ramp `min(lamda, cof * epoch)`;
    the per-epoch EMA that mixes the uni-modal partner into the audio-visual
    model, its `cosine_scheduler(m, 1, epoch, 50)` rate, and its exclusion of
    the audio-specific parameters;
    `seq_len` recovered from the non-zero rows of the padded visual tensor, and
    the batch truncated to the longest real sequence in it.

Patches
    M3, M4, M5   device, batch-of-one squeeze, input widths. See avce_network.
    M7  Upstream evaluates the *test* split after every epoch, and after every
        epoch keeps the checkpoint with the best test AP. That is a
        test-selected baseline and it is not comparable with a method that is
        not test-selected, so this port never opens the test split during
        training: it carves a seeded, label-stratified 10 % validation subset
        out of the train split and selects on video-level average precision
        there. `--val-frac 0 --select last` restores upstream's structure minus
        the test peek. Same rule as the VadCLIP and DSANet ports, patch V3.
    M8  Logging. Upstream logs twice per epoch through a `Prepare_logger`
        file handler rooted at a relative `log/` path. Replaced by one stdout
        summary line per epoch carrying every loss term, the validation AP and
        the wall time.
    M9  `torch.multiprocessing.set_start_method('spawn')` and the
        `os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus` assignment are gone;
        the device comes from `--device` and nothing here touches a GPU unless
        asked to.
    M10 Upstream calls `test()` once before the first epoch to log the
        random-initialisation AP. Dropped: it reads the test split.
    M11 The uni-modal ablations. `Single_Model` is upstream's own uni-modal
        network -- in `main.py` it is the visual partner that the EMA distils
        from -- so `--modality audio` and `--modality visual` train exactly that
        module, at exactly the lr/5 upstream trains it at, on one modality.
        This is what makes the audio-only row an honest comparator rather than
        a new architecture: nothing is added, the audio-visual model's own
        uni-modal component is trained alone.
    M13 `--fix-rep-swap`. Off by default; see the note in avce_network.py.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hate_common import data as hdata          # noqa: E402
from hate_common import runtime                # noqa: E402

from . import option                           # noqa: E402
from .avce_network import AVCE_Model, Single_Model  # noqa: E402
from .CMA_MIL import CMAL                      # noqa: E402
from .dataset import MacilTrainDataset, feature_dim, usable_ids  # noqa: E402
from .utils import cosine_scheduler            # noqa: E402

sys.path.insert(0, os.path.join(hdata.REPO_ROOT, "scripts", "duplex"))
import frame_eval_common as fec                # noqa: E402

UPSTREAM = "https://github.com/JustinYuu/MACIL_SD @ c20943f"


def build_models(args):
    """The audio-visual model and its uni-modal partner, per --modality.

    Returns (model_av, model_uni). For the ablations model_av is None and
    model_uni is the standalone network.
    """
    if args.modality == "av":
        return AVCE_Model(args), Single_Model(args, n_dim=args.v_feature_size)
    return None, Single_Model(args, n_dim=feature_dim(args.modality))


def _scalar(x):
    """Loss value as a plain float, without the detach warning.

    CMAL returns tensors when both of its self-guided banks are populated and
    the literal 0.0 when either is empty, so the accumulator has to take both.
    """
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _stratified_head(ids, labels, n):
    """First `n` ids, alternating hateful and normal. Debug helper only."""
    pos = [v for v in ids if labels[v] == 1]
    neg = [v for v in ids if labels[v] == 0]
    out = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos):
            out.append(pos[i])
        if i < len(neg):
            out.append(neg[i])
        if len(out) >= n:
            break
    return out[:n]


def _seq_len_of(f_v):
    """Upstream's length recovery: rows whose visual feature is not all zero."""
    return torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)


def _bag_scores(args, model_av, model_uni, f_v, f_a, seq_len):
    """Per-video MIL bag score in [0, 1], the quantity the BCE loss reads."""
    if args.modality == "av":
        mmil_logits = model_av(f_a, f_v, seq_len)[0]
        return mmil_logits.reshape(-1)
    feat = f_a if args.modality == "audio" else f_v
    return model_uni(feat, seq_len).reshape(-1)


def evaluate_video_ap(args, model_av, model_uni, loader, device):
    """Video-level average precision of the bag score on the held-out carve."""
    if loader is None:
        return None
    was_training = []
    for m in (model_av, model_uni):
        if m is not None:
            was_training.append((m, m.training))
            m.eval()
    scores, labels = [], []
    with torch.no_grad():
        for f_v, f_a, label in loader:
            seq_len = _seq_len_of(f_v)
            keep = int(torch.max(seq_len))
            f_v = f_v[:, :keep, :].to(device)
            f_a = f_a[:, :keep, :].to(device)
            s = _bag_scores(args, model_av, model_uni, f_v, f_a, seq_len)
            scores.extend(float(x) for x in s.cpu())
            labels.extend(int(x) for x in label)
    for m, flag in was_training:
        m.train(flag)
    if not scores or len(set(labels)) < 2:
        return None
    return fec.average_precision(np.asarray(scores), np.asarray(labels))


def train_one_epoch(args, model_av, model_uni, opt_av, opt_uni, criterion,
                    loader, lamda_a2b, lamda_a2n, device):
    """Upstream avce_train, one epoch. Returns the mean of each loss term."""
    totals = np.zeros(6)
    n_batches = 0
    for f_v, f_a, label in loader:
        seq_len = _seq_len_of(f_v)
        keep = int(torch.max(seq_len))
        f_v = f_v[:, :keep, :].float().to(device)
        f_a = f_a[:, :keep, :].float().to(device)
        label = label.float().to(device)

        if args.modality != "av":
            feat = f_a if args.modality == "audio" else f_v
            logits = model_uni(feat, seq_len).reshape(-1)
            loss_uni = criterion(logits, label)
            opt_uni.zero_grad()
            loss_uni.backward()
            opt_uni.step()
            totals += [_scalar(loss_uni), 0.0, 0.0, 0.0, 0.0,
                       _scalar(loss_uni)]
            n_batches += 1
            continue

        out = model_av(f_a, f_v, seq_len)
        mmil_logits, audio_logits, visual_logits, _av_logits, v_out, a_out = out
        # Upstream binds (audio_rep, visual_rep) = (v_out, a_out), i.e. each
        # representation arrives under the other modality's name while the
        # logits that pick the top-k positions do not. Reproduced by default.
        if args.fix_rep_swap:
            audio_rep, visual_rep = a_out, v_out
        else:
            audio_rep, visual_rep = v_out, a_out

        # PORT PATCH (patch M4): reshape rather than a bare squeeze, so a batch
        # holding a single video keeps its batch axis.
        audio_logits = audio_logits.squeeze(-1)
        visual_logits = visual_logits.squeeze(-1)
        mmil_logits = mmil_logits.reshape(-1)

        clsloss = criterion(mmil_logits, label)
        cma_a2v_a2b, cma_a2v_a2n, cma_v2a_a2b, cma_v2a_a2n = CMAL(
            mmil_logits, audio_logits, visual_logits, seq_len,
            audio_rep, visual_rep)
        total_loss = (clsloss
                      + lamda_a2b * cma_a2v_a2b + lamda_a2b * cma_v2a_a2b
                      + lamda_a2n * cma_a2v_a2n + lamda_a2n * cma_v2a_a2n)

        uni_logits = model_uni(f_v, seq_len).reshape(-1)
        loss_uni = criterion(uni_logits, label)

        # Upstream's two-step alternation, kept including its `requires_grad`
        # assignments -- which set a plain attribute on the Module rather than
        # on its parameters and therefore do nothing. The zero_grad pair before
        # each backward is what actually keeps the two graphs apart, and the
        # two losses share no parameters in any case.
        opt_av.zero_grad()
        opt_uni.zero_grad()
        model_av.requires_grad = True
        model_uni.requires_grad = False
        total_loss.backward()
        opt_av.step()

        opt_av.zero_grad()
        opt_uni.zero_grad()
        model_av.requires_grad = False
        model_uni.requires_grad = True
        loss_uni.backward()
        opt_uni.step()

        totals += [_scalar(clsloss), _scalar(cma_a2v_a2b),
                   _scalar(cma_a2v_a2n), _scalar(cma_v2a_a2b),
                   _scalar(cma_v2a_a2n), _scalar(loss_uni)]
        n_batches += 1
    return totals / max(n_batches, 1), n_batches


def distil_step(args, model_av, model_uni, epoch):
    """Upstream's per-epoch EMA of the uni-modal partner into model_av.

    Verbatim, including the name-matching rules and the exclusion of any
    parameter whose name contains 'sa_a' or 'fc_a' -- the audio-specific
    projection, which has no counterpart in a visual-only partner. Returns the
    mixing rate actually used.
    """
    m = cosine_scheduler(base_value=args.m, final_value=1,
                         curr_epoch=epoch, epochs=args.ema_epochs)
    if m == 1.0:
        return m
    with torch.no_grad():
        for param_av in model_av.named_parameters():
            if 'sa_a' in param_av[0] or 'fc_a' in param_av[0]:
                continue
            for param_v in model_uni.named_parameters():
                if param_av[0] == param_v[0]:
                    param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                    break
                elif param_av[0] == 'att_mmil.fc.weight' and param_v[0] == 'fc.weight':
                    param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                    break
                elif param_av[0] == 'att_mmil.fc.bias' and param_v[0] == 'fc.bias':
                    param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                    break
    return m


def train(args):
    device = args.device
    runtime.setup_seed(args.seed)

    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(
        args.corpus, labels, args.val_frac, args.seed)
    train_ids = usable_ids(args.corpus, train_ids)
    val_ids = usable_ids(args.corpus, val_ids)
    if args.limit_videos:
        # Debug only. Interleaved by class rather than a plain head slice: the
        # ids sort with every hateful video first in all three corpora, so
        # train_ids[:N] would be single-class and the validation AP -- and with
        # it the whole selection path -- would never be exercised by a dry run.
        train_ids = _stratified_head(train_ids, labels, args.limit_videos)
        val_ids = _stratified_head(val_ids, labels,
                                   max(args.limit_videos // 4, 2))

    train_set = MacilTrainDataset(args.corpus, train_ids, labels,
                                  args.max_seqlen, args.grid, args.modality,
                                  args.crop_repeat)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=False)
    val_loader = None
    if val_ids:
        val_set = MacilTrainDataset(args.corpus, val_ids, labels,
                                    args.max_seqlen, args.grid, args.modality,
                                    args.crop_repeat)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.out_dir, exist_ok=True)
    print("method            %s (modality %s, grid %s)"
          % (args.method, args.modality, args.grid))
    print("corpus            %s" % args.corpus)
    print("train / val       %d / %d videos (%d hateful in train), "
          "%d train items at crop-repeat %d"
          % (len(train_ids), len(val_ids), sum(labels[v] for v in train_ids),
             len(train_set), args.crop_repeat))
    print("max-seqlen        %d rows = %.1f s on the %s grid"
          % (args.max_seqlen,
             args.max_seqlen * (16.0 / 24.0 if args.grid == "snippet" else 1.0),
             args.grid))
    print("device            %s" % device)
    sys.stdout.flush()

    model_av, model_uni = build_models(args)
    if model_av is not None:
        model_av.to(device)
    model_uni.to(device)
    n_params = sum(p.numel() for p in model_uni.parameters())
    if model_av is not None:
        n_params += sum(p.numel() for p in model_av.parameters())
    print("parameters        %.3fM" % (n_params / 1e6))

    criterion = torch.nn.BCELoss()
    opt_av = (optim.Adam(model_av.parameters(), lr=args.lr, weight_decay=0.000)
              if model_av is not None else None)
    opt_uni = optim.Adam(model_uni.parameters(),
                         lr=args.lr * args.single_lr_scale, weight_decay=0.000)
    sched_av = (optim.lr_scheduler.CosineAnnealingLR(opt_av,
                                                     T_max=args.sched_tmax,
                                                     eta_min=0)
                if opt_av is not None else None)
    sched_uni = optim.lr_scheduler.CosineAnnealingLR(opt_uni,
                                                     T_max=args.sched_tmax,
                                                     eta_min=0)

    best_ap, best_state, best_epoch = -1.0, None, -1
    history = []
    selected_model = model_av if model_av is not None else model_uni

    for epoch in range(args.max_epoch):
        t0 = time.time()
        if model_av is not None:
            model_av.train()
        model_uni.train()
        lamda_a2b = min(args.lamda_a2b, args.lamda_cof * epoch)
        lamda_a2n = min(args.lamda_a2n, args.lamda_cof * epoch)
        means, n_batches = train_one_epoch(
            args, model_av, model_uni, opt_av, opt_uni, criterion,
            train_loader, lamda_a2b, lamda_a2n, device)
        if sched_av is not None:
            sched_av.step()
        sched_uni.step()

        m = (distil_step(args, model_av, model_uni, epoch)
             if model_av is not None else None)

        val_ap = evaluate_video_ap(args, model_av, model_uni, val_loader,
                                   device)
        history.append({
            "epoch": epoch + 1, "batches": n_batches,
            "cls_loss": means[0], "cma_a2v_a2b": means[1],
            "cma_a2v_a2n": means[2], "cma_v2a_a2b": means[3],
            "cma_v2a_a2n": means[4], "uni_loss": means[5],
            "lamda_a2b": lamda_a2b, "lamda_a2n": lamda_a2n, "ema_m": m,
            "val_video_ap": val_ap, "seconds": round(time.time() - t0, 1),
        })
        print("epoch %2d | cls %.4f | cma %.4f %.4f %.4f %.4f | uni %.4f "
              "| lam %.2f | m %s | val AP %s | %.0fs"
              % (epoch + 1, means[0], means[1], means[2], means[3], means[4],
                 means[5], lamda_a2b, ("%.4f" % m) if m is not None else "n/a",
                 ("%.4f" % val_ap) if val_ap is not None else "n/a",
                 time.time() - t0))
        sys.stdout.flush()

        if args.select == "val" and val_ap is not None and val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch + 1
            best_state = copy.deepcopy(selected_model.state_dict())

    if args.select == "val" and best_state is not None:
        selected_model.load_state_dict(best_state)
        print("selected epoch %d (val video AP %.4f)" % (best_epoch, best_ap))
    else:
        print("selected the last epoch (no validation selection)")

    model_path = os.path.join(args.out_dir, "model.pth")
    torch.save(selected_model.state_dict(), model_path)
    meta = {
        "method": args.method,
        "upstream": UPSTREAM,
        "args": {k: v for k, v in vars(args).items()},
        "train_ids": train_ids,
        "val_ids": val_ids,
        "n_train_items": len(train_set),
        "selected_epoch": best_epoch if args.select == "val" else args.max_epoch,
        "selected_val_video_ap": best_ap if best_ap >= 0 else None,
        "history": history,
        "grid": args.grid,
        "row_seconds": (16.0 / 24.0) if args.grid == "snippet" else 1.0,
    }
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print("wrote %s" % model_path)
    return model_path


def main(argv=None):
    args = option.resolve(option.build_parser().parse_args(argv))
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
