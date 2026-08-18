#!/usr/bin/env python
"""CPU-only smoke test for both baseline ports. Never touches a GPU.

What it checks, per method:
  1. the model builds on cpu at a small visual_length;
  2. a synthetic training batch forwards, every logit tensor has the expected
     shape, and every loss in the published sum is finite;
  3. backward runs and every trainable parameter that received a gradient has
     a finite one;
  4. the inference path forwards a `process_split` item and returns exactly
     len_cur rows, for a video shorter than one block and for one longer than
     two blocks;
  5. the real dataset and label plumbing agrees with the frozen gold: feature
     row counts equal gold frame counts, and the labels cover every split id.

The evaluator is checked separately: frame_eval_common's own selftest is run,
then eval_baseline_scores.evaluate_scores is checked against a perfect and an
inverted ranking.

    CUDA_VISIBLE_DEVICES="" python scripts/reproduction_baselines/smoke_cpu.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "duplex")))

from hate_common import data as hdata          # noqa: E402
from hate_common import runtime                # noqa: E402
import frame_eval_common as fec                # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402

DEVICE = "cpu"
VLEN = 32
AWIN = 8
B = 4
D = 512

_failures = []


def check(name, ok, detail=""):
    print("%-62s %s%s" % (name, "OK" if ok else "FAIL",
                          ("  " + detail) if detail else ""))
    if not ok:
        _failures.append(name)
    return bool(ok)


def synthetic_batch(seed=0):
    """A batch shaped like HateVideoDataset's training output."""
    g = torch.Generator().manual_seed(seed)
    # Feature scale matched to the real files: CLIP ViT-B/16 image_embeds have
    # row norms near 11, not 1.
    visual = torch.randn(B, VLEN, D, generator=g) * (11.0 / np.sqrt(D))
    lengths = torch.tensor([VLEN, VLEN // 2, 5, 2])
    class_idx = [1, 0, 1, 0]
    labels = hdata.label_vectors(class_idx)
    return visual, labels, lengths


def smoke_vadclip():
    print("\n--- VadCLIP ---")
    from vadclip.model import CLIPVAD
    torch.manual_seed(0)
    model = CLIPVAD(hdata.NUM_CLASSES, 512, VLEN, D, 1, 1, AWIN, 10, 10,
                    DEVICE)
    model.to(DEVICE).train()
    check("vadclip: builds on cpu", True,
          "%.1fM params" % (sum(p.numel() for p in model.parameters()) / 1e6))

    visual, labels, lengths = synthetic_batch()
    prompt = list(hdata.PROMPT_TEXT)
    text_features, logits1, logits2 = model(visual, None, prompt, lengths)

    check("vadclip: text_features shape (2, 512)",
          tuple(text_features.shape) == (2, 512), str(tuple(text_features.shape)))
    check("vadclip: logits1 shape (B, T, 1)",
          tuple(logits1.shape) == (B, VLEN, 1), str(tuple(logits1.shape)))
    check("vadclip: logits2 shape (B, T, 2)",
          tuple(logits2.shape) == (B, VLEN, 2), str(tuple(logits2.shape)))

    loss1 = runtime.CLAS2(logits1, labels, lengths, DEVICE)
    loss2 = runtime.CLASM(logits2, labels, lengths, DEVICE)
    loss3 = runtime.text_orthogonality_loss(text_features, DEVICE)
    loss = loss1 + loss2 + loss3 * 1e-4
    check("vadclip: losses finite",
          all(torch.isfinite(x).all() for x in (loss1, loss2, loss3, loss)),
          "l1 %.4f l2 %.4f l3 %.4f total %.4f"
          % (loss1, loss2, loss3.item(), loss))

    loss.backward()
    grads = [p.grad for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    check("vadclip: backward produced gradients", len(grads) > 0,
          "%d tensors" % len(grads))
    check("vadclip: gradients finite",
          all(torch.isfinite(g).all() for g in grads))

    smoke_inference("vadclip", model, prompt)


def smoke_dsanet():
    print("\n--- DSANet ---")
    from dsanet.model import DSANet
    from dsanet.train import CLASM_EVENT, CLASM_BKG, consistency_loss_fn
    from dsanet import option

    args = option.resolve(option.build_parser().parse_args(
        ["--corpus", "hatemm", "--device", "cpu",
         "--visual-length", str(VLEN), "--attn-window", str(AWIN)]))
    torch.manual_seed(0)
    model = DSANet(hdata.NUM_CLASSES, 512, VLEN, D, 1, 1, AWIN, 10, 10,
                   args, DEVICE)
    model.to(DEVICE).train()
    check("dsanet: builds on cpu", True,
          "%.1fM params" % (sum(p.numel() for p in model.parameters()) / 1e6))

    visual, labels, lengths = synthetic_batch()
    prompt = list(hdata.PROMPT_TEXT)
    text_features, logits1, logits2, logits3, logits4, DNP = model(
        visual, None, prompt, lengths, True)

    check("dsanet: text_features shape (2, 512)",
          tuple(text_features.shape) == (2, 512), str(tuple(text_features.shape)))
    check("dsanet: logits1 shape (B, T, 1)",
          tuple(logits1.shape) == (B, VLEN, 1), str(tuple(logits1.shape)))
    check("dsanet: logits2 shape (B, T, 2)",
          tuple(logits2.shape) == (B, VLEN, 2), str(tuple(logits2.shape)))
    check("dsanet: logits3/4 shape (B, 1, 2)",
          tuple(logits3.shape) == (B, 1, 2) and tuple(logits4.shape) == (B, 1, 2),
          "%s %s" % (tuple(logits3.shape), tuple(logits4.shape)))
    check("dsanet: reconstruction shape matches features",
          DNP["reconstructed_features"].shape == DNP["original_features"].shape,
          str(tuple(DNP["reconstructed_features"].shape)))

    loss1 = runtime.CLAS2(logits1, labels, lengths, DEVICE)
    loss2 = runtime.CLASM(logits2, labels, lengths, DEVICE)
    loss3 = runtime.text_orthogonality_loss(text_features, DEVICE)
    loss4 = CLASM_EVENT(logits3, labels, lengths, DEVICE)
    loss5 = CLASM_BKG(logits4, labels, lengths, DEVICE)
    closs = consistency_loss_fn(logits1=logits1,
                               original_features=DNP["original_features"],
                               reconstructed_features=DNP["reconstructed_features"],
                               lengths=lengths)
    gloss = DNP["g_loss"]
    loss = (loss1 + loss2 * args.loss2_weight + loss3 + loss4 + loss5
            + closs + gloss)
    parts = (loss1, loss2, loss3, loss4, loss5, closs, gloss, loss)
    check("dsanet: losses finite",
          all(torch.isfinite(x).all() for x in parts),
          "l1 %.4f l2 %.4f l3 %.4f l4 %.4f l5 %.4f cons %.4f g %.4f total %.4f"
          % (loss1, loss2, loss3.item(), loss4, loss5, closs, gloss, loss))

    loss.backward()
    grads = [p.grad for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    check("dsanet: backward produced gradients", len(grads) > 0,
          "%d tensors" % len(grads))
    check("dsanet: gradients finite",
          all(torch.isfinite(g).all() for g in grads))

    refiner = [n for n, p in model.named_parameters()
               if "video_anomaly_refiner" in n and p.grad is not None]
    check("dsanet: refiner parameters received gradients", len(refiner) > 0,
          "%d tensors" % len(refiner))

    smoke_inference("dsanet", model, prompt, dsanet_args=args)


def smoke_inference(method, model, prompt, dsanet_args=None):
    """Feed process_split items of both regimes through the inference path."""
    from hate_common import tools
    model.eval()
    for label, T in (("short", VLEN // 2), ("multi-block", VLEN * 2 + 7)):
        feat = np.random.default_rng(T).normal(
            0, 11.0 / np.sqrt(D), (T, D)).astype(np.float32)
        chunks, length = tools.process_split(feat, VLEN)
        visual = torch.from_numpy(np.ascontiguousarray(chunks)).unsqueeze(0)
        visual, lengths = runtime.prepare_test_item(visual, length, VLEN)
        with torch.no_grad():
            if dsanet_args is not None:
                out = model(visual, None, prompt, lengths,
                            bool(dsanet_args.DNP_use))
            else:
                out = model(visual, None, prompt, lengths)
        logits1 = out[1].reshape(-1, out[1].shape[2])
        logits2 = out[2].reshape(-1, out[2].shape[2])
        s = torch.sigmoid(logits1[0:T].squeeze(-1))
        a = 1 - torch.softmax(logits2[0:T], dim=-1)[:, 0]
        check("%s: %s video (T=%d) yields T scores" % (method, label, T),
              s.shape == (T,) and a.shape == (T,),
              "%s %s" % (tuple(s.shape), tuple(a.shape)))
        check("%s: %s video scores finite and in [0, 1]" % (method, label),
              bool(torch.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
                   and torch.isfinite(a).all()))

        if method == "dsanet":
            from dsanet.infer import refine_scores_hierarchical
            refined = 1 - refine_scores_hierarchical(
                logits1[0:T], logits2[0:T], dsanet_args.temp)[:, 0]
            check("dsanet: %s refined branch == mlp branch (binary collapse)"
                  % label, bool(torch.allclose(refined, s, atol=1e-5)),
                  "max abs diff %.2e" % float((refined - s).abs().max()))
    model.train()


def smoke_data():
    print("\n--- data and gold ---")
    for corpus in hdata.CORPORA:
        labels = hdata.load_labels(corpus)
        gt = hdata.gt_arrays(corpus, "test")
        train_ids = hdata.load_split(corpus, "train")
        test_ids = hdata.load_split(corpus, "test")

        check("%s: every split id has a label" % corpus,
              all(v in labels for v in train_ids + test_ids))
        check("%s: every split id has a feature file" % corpus,
              all(os.path.exists(hdata.feature_path(corpus, v))
                  for v in train_ids + test_ids))

        bad = []
        for vid in gt:
            T = np.load(hdata.feature_path(corpus, vid), mmap_mode="r").shape[0]
            if T != len(gt[vid]):
                bad.append((vid, T, len(gt[vid])))
        check("%s: feature rows == gold frames for all %d gold videos"
              % (corpus, len(gt)), not bad, str(bad[:3]))

        tr, va = hdata.split_train_val(train_ids, labels, 0.1, 234)
        tr2, va2 = hdata.split_train_val(list(reversed(train_ids)), labels,
                                         0.1, 234)
        check("%s: validation carve is deterministic and disjoint" % corpus,
              tr == tr2 and va == va2 and not (set(tr) & set(va))
              and len(tr) + len(va) == len(train_ids),
              "train %d val %d (%d hateful in val)"
              % (len(tr), len(va), sum(labels[v] for v in va)))

        ds = hdata.HateVideoDataset(corpus, test_ids[:3],
                                    runtime.default_visual_length(corpus),
                                    True, labels)
        feat, lab, length, vid = ds[0]
        check("%s: dataset item well formed" % corpus,
              feat.dtype == torch.float32 and lab in (0, 1) and length > 0
              and vid == test_ids[0],
              "feat %s length %d" % (tuple(feat.shape), length))


def smoke_eval():
    print("\n--- evaluator ---")
    check("frame_eval_common selftest", fec.selftest())

    gt = {"a": np.array([0, 0, 1, 1], dtype=np.uint8),
          "b": np.array([0, 1, 1, 0], dtype=np.uint8),
          "c": np.zeros(4, dtype=np.uint8)}
    perfect = {"a": np.array([0.0, 0.1, 0.9, 1.0]),
               "b": np.array([0.0, 0.9, 1.0, 0.1]),
               "c": np.zeros(4)}
    inverted = {k: -v for k, v in perfect.items()}
    hate = {"a", "b"}

    r = evaluate_scores(perfect, gt, hate)
    check("evaluate_scores: perfect ranking, pooled AUC 1.0",
          r["roc_auc"] == 1.0, "%.4f" % r["roc_auc"])
    check("evaluate_scores: perfect ranking, within-hate macro 1.0",
          r["per_video"]["macro_auc"] == 1.0
          and r["per_video"]["n_videos_both_classes"] == 2,
          "%.4f over %d" % (r["per_video"]["macro_auc"],
                            r["per_video"]["n_videos_both_classes"]))
    r = evaluate_scores(inverted, gt, hate)
    check("evaluate_scores: inverted ranking, pooled AUC 0.0",
          r["roc_auc"] == 0.0, "%.4f" % r["roc_auc"])
    r = evaluate_scores({k: v for k, v in perfect.items() if k != "c"}, gt, hate)
    check("evaluate_scores: reports videos missing from scores",
          r["n_videos_missing_from_scores"] == 1
          and r["videos_missing_from_scores"] == ["c"])
    try:
        evaluate_scores({"a": np.zeros(3)}, {"a": np.zeros(4, dtype=np.uint8)})
        ok = False
    except ValueError:
        ok = True
    check("evaluate_scores: rejects a length mismatch", ok)


def main():
    if torch.cuda.is_available():
        print("NOTE: a GPU is visible; this script pins everything to cpu "
              "anyway and issues no cuda calls.")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    smoke_data()
    smoke_eval()
    smoke_vadclip()
    smoke_dsanet()
    print("")
    if _failures:
        print("SMOKE FAILED: %d check(s)" % len(_failures))
        for name in _failures:
            print("  %s" % name)
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
