"""Train the MultiHateLoc reimplementation and write per-frame scores.

Published training setup, used verbatim: Adam, learning rate 1e-4, batch size
32, 100 epochs, K = 3 (the top third of frames), lambda_smooth = 0.1,
lambda_contrastive = 0.2.

Protocol, following the VadCLIP / DSANet ports in this directory (PATCHES.md
patch V3): the test split is never opened during training. A seeded,
label-stratified 10 % of the train split is held out and the checkpoint is
selected on its video-level average precision. The published 100-epoch budget
is kept; selection decides which of those 100 epochs is scored.

Output: results/reproduction/baselines/multihateloc_reimpl/<corpus>/
    scores.jsonl   one record per test video, all branches
    train_log.json per-epoch losses and validation AP
    model.pt       the selected state dict

  python train.py --corpus hatemm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.utils.data as tdata

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..")))

import data as mdata                              # noqa: E402
from model import MultiHateLoc                    # noqa: E402
from hate_common import data as hdata             # noqa: E402

OUT_ROOT = os.path.join(hdata.REPO_ROOT, "results", "reproduction",
                        "baselines", "multihateloc_reimpl")


def average_precision(y_true, y_score):
    """Step-wise average precision, no scikit-learn dependency.

    Same convention as scripts/duplex/frame_eval_common: sum of (R_i - R_{i-1})
    * P_i over the ranked list, ties broken by taking the last rank of a tie.
    """
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    npos = y.sum()
    if npos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / npos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def run_epoch(model, loader, device, opt, args):
    model.train()
    agg = {}
    n = 0
    for feats, labels, lengths, mask, _ in loader:
        feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
        labels = labels.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        out = model(feats, mask)
        mil, per_branch = model.mil_loss(out["probs"], mask, lengths, labels)
        smooth = model.smoothness_loss(out["probs"], mask)
        contrast = model.contrastive_loss(out["embeds"], mask)
        loss = mil + args.lambda_smooth * smooth + args.lambda_contrast * contrast
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        terms = {"loss": loss, "mil": mil, "smooth": smooth,
                 "contrast": contrast}
        terms.update({"mil_" + k: v for k, v in per_branch.items()})
        for k, v in terms.items():
            agg[k] = agg.get(k, 0.0) + float(v.detach()) * len(labels)
        n += len(labels)
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def predict(model, loader, device):
    """Per-video branch score arrays and video-level scores."""
    model.eval()
    frames, video, labels_out = {}, {}, {}
    for feats, labels, lengths, mask, vids in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        lengths_d = lengths.to(device)
        mask = mask.to(device)
        out = model(feats, mask)
        probs = out["probs"]
        vid_scores = model.video_scores(probs, mask, lengths_d)
        union = model.union_frames(probs, out["weights"], mask, lengths_d)
        dms = model.dms_frames(probs, out["weights"])
        for i, vid in enumerate(vids):
            L = int(lengths[i])
            rec = {"score_%s" % k: probs[k][i, :L].cpu().numpy()
                   for k in probs}
            rec["score_dms"] = dms[i, :L].cpu().numpy()
            rec["score_union"] = union[i, :L].cpu().numpy()
            frames[vid] = rec
            video[vid] = {k: float(v[i]) for k, v in vid_scores.items()}
            video[vid]["weights"] = [float(x) for x in out["weights"][i]]
            labels_out[vid] = int(labels[i])
    return frames, video, labels_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=list(hdata.CORPORA))
    ap.add_argument("--out-root", default=OUT_ROOT)
    # published settings
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("--k-proportion", type=int, default=3)
    ap.add_argument("--lambda-smooth", type=float, default=0.1)
    ap.add_argument("--lambda-contrast", type=float, default=0.2)
    # inferred settings
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)
    # protocol
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--select", default="val_ap", choices=("val_ap", "last"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-test", action="store_true",
                    help="run frozen-checkpoint inference on the test split; "
                         "leave off during validation tuning")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA unavailable; pass --device cpu to smoke test")

    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)

    labels = hdata.load_labels(args.corpus)
    test_ids = hdata.load_split(args.corpus, "test")
    if args.run_test:
        gold = hdata.gt_arrays(args.corpus, "test")
        test_ids = [vid for vid in test_ids if vid in gold]
    train_ids, val_ids = hdata.load_train_val(
        args.corpus, labels, args.val_frac, args.seed)
    print("multihateloc [%s]: %d train, %d val, %d test  (%d hateful in train)"
          % (args.corpus, len(train_ids), len(val_ids), len(test_ids),
             sum(labels[v] for v in train_ids)), flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = tdata.DataLoader(
        mdata.MultiModalDataset(args.corpus, train_ids, labels),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        collate_fn=mdata.collate, num_workers=4, generator=gen)
    val_loader = (tdata.DataLoader(
        mdata.MultiModalDataset(args.corpus, val_ids, labels),
        batch_size=args.batch_size, shuffle=False, collate_fn=mdata.collate,
        num_workers=2) if val_ids else None)
    test_loader = (tdata.DataLoader(
        mdata.MultiModalDataset(args.corpus, test_ids, labels),
        batch_size=args.batch_size, shuffle=False, collate_fn=mdata.collate,
        num_workers=2) if args.run_test else None)

    model = MultiHateLoc({m: mdata.FEATURE_DIMS[m] for m in mdata.MODALITIES},
                         hidden=args.hidden, embed=args.embed,
                         dropout=args.dropout, k_proportion=args.k_proportion,
                         temperature=args.temperature).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  %.2f M parameters" % (n_params / 1e6), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_ap, best_epoch, best_state = -1.0, None, None
    t0 = time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_loader, device, opt, args)
        val_ap = float("nan")
        if val_loader is not None:
            _, vvid, vlab = predict(model, val_loader, device)
            ids = sorted(vvid)
            val_ap = average_precision([vlab[v] for v in ids],
                                       [vvid[v]["fused"] for v in ids])
        stats["epoch"] = epoch
        stats["val_ap"] = val_ap
        stats["seconds"] = round(time.time() - t0, 1)
        history.append(stats)
        if args.select == "val_ap" and val_loader is not None \
                and val_ap == val_ap and val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print("  epoch %3d  loss %.4f (mil %.4f smooth %.4f contrast "
                  "%.4f)  val AP %.4f  %.0fs"
                  % (epoch, stats["loss"], stats["mil"], stats["smooth"],
                     stats["contrast"], val_ap, stats["seconds"]), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("  selected epoch %d, val video AP %.4f" % (best_epoch, best_ap),
              flush=True)
    else:
        best_epoch, best_ap = args.max_epoch, float("nan")
        print("  selection off: taking the last epoch", flush=True)

    frames, video = {}, {}
    scores_path = os.path.join(out_dir, "scores.jsonl")
    if test_loader is not None:
        frames, video, _ = predict(model, test_loader, device)
        with open(scores_path, "w", encoding="utf-8") as fh:
            for vid in test_ids:
                rec = {"video_id": vid}
                rec.update({k: [round(float(x), 6) for x in v]
                            for k, v in frames[vid].items()})
                fh.write(json.dumps(rec) + "\n")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    log = {
        "corpus": args.corpus,
        "args": vars(args),
        "n_parameters": int(n_params),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "n_test": len(test_ids) if args.run_test else 0,
        "steps_per_epoch": int(np.ceil(len(train_ids) / args.batch_size)),
        "selected_epoch": best_epoch,
        "selected_val_video_ap": best_ap,
        "wall_seconds": round(time.time() - t0, 1),
        "history": history,
        "test_video_scores": video,
        "corpus_summary": {s: mdata.describe(args.corpus, s)
                           for s in ("train", "test")},
    }
    with open(os.path.join(out_dir, "train_log.json"), "w") as fh:
        json.dump(log, fh, indent=1, default=float)
    if args.run_test:
        print("  wrote %s (%d videos), %.0fs total"
              % (scores_path, len(test_ids), time.time() - t0), flush=True)
    else:
        print("  validation-only run; test split not loaded, %.0fs total"
              % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
