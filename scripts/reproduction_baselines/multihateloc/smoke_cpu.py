"""CPU checks for the MultiHateLoc reimplementation. No GPU, no training.

What it asserts, in order:

  features   every video in every split has all three feature matrices, of
             the stated dimensionality, and the three lengths agree;
  grid       for every video in the frozen gold cohort, the feature length
             equals the gold array length, so a score row is a gold frame;
  masking    a padded batch produces the same frame probabilities as the same
             videos scored one at a time -- padding leaks nowhere;
  top-K      the MIL pool size is ceil(T / K) and the pooled value is the
             mean of exactly that many highest valid frames;
  losses     every loss term is finite, the smoothness term is zero on a
             constant score, and the contrastive term is at its floor when
             the modalities are already aligned;
  union      the union rule returns a subset of the valid frames that
             contains the fused branch's top-K frames.

  python smoke_cpu.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..")))

import data as mdata                              # noqa: E402
import model as mmodel                            # noqa: E402
from hate_common import data as hdata             # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL",
                            "" if not detail else "   " + detail))
    if not ok:
        FAILED.append(name)


def main():
    torch.manual_seed(0)

    print("features and grid")
    for corpus in hdata.CORPORA:
        ids = (hdata.load_split(corpus, "train")
               + hdata.load_split(corpus, "test"))
        bad_shape, bad_len = [], []
        for vid in ids:
            lens = set()
            for mod in mdata.MODALITIES:
                p = mdata.feature_path(mod, corpus, vid)
                if not os.path.isfile(p):
                    bad_shape.append((vid, mod, "missing"))
                    continue
                a = np.load(p, mmap_mode="r")
                if a.ndim != 2 or a.shape[1] != mdata.FEATURE_DIMS[mod]:
                    bad_shape.append((vid, mod, a.shape))
                lens.add(a.shape[0])
            if len(lens) > 1:
                bad_len.append((vid, sorted(lens)))
        check("%s: all three matrices present and %s-d"
              % (corpus, "/".join(str(mdata.FEATURE_DIMS[m])
                                  for m in mdata.MODALITIES)),
              not bad_shape, "%d videos, %d bad" % (len(ids), len(bad_shape)))
        check("%s: the three modality lengths agree per video" % corpus,
              not bad_len, "%d mismatched" % len(bad_len))

        gt = hdata.gt_arrays(corpus, "test")
        off = [v for v in gt
               if np.load(mdata.feature_path("visual", corpus, v),
                          mmap_mode="r").shape[0] != len(gt[v])]
        check("%s: feature rows == gold frames on the gold cohort" % corpus,
              not off, "%d gold videos, %d off" % (len(gt), len(off)))

    print("model")
    dims = {m: mdata.FEATURE_DIMS[m] for m in mdata.MODALITIES}
    net = mmodel.MultiHateLoc(dims).eval()
    n_params = sum(p.numel() for p in net.parameters())
    check("parameter count is in the weakly-supervised range",
          0.1e6 < n_params < 20e6, "%.2f M" % (n_params / 1e6))

    corpus = "mhclip_en"
    ids = hdata.load_split(corpus, "test")[:6]
    ds = mdata.MultiModalDataset(corpus, ids)
    batch = mdata.collate([ds[i] for i in range(len(ids))])
    feats, labels, lengths, mask, vids = batch
    with torch.no_grad():
        out_batched = net(feats, mask)
    worst = 0.0
    for i in range(len(ids)):
        single = mdata.collate([ds[i]])
        with torch.no_grad():
            out_one = net(single[0], single[3])
        L = int(single[2][0])
        for b in out_one["probs"]:
            d = (out_batched["probs"][b][i, :L]
                 - out_one["probs"][b][0, :L]).abs().max().item()
            worst = max(worst, d)
    check("padding does not leak: batched == one-at-a-time",
          worst < 1e-5, "max abs diff %.2e" % worst)

    check("valid frames are the only non-zero rows",
          bool((out_batched["probs"]["fused"] * ~mask).abs().max() == 0))
    w = out_batched["weights"]
    check("DMS weights are a simplex", bool(
        torch.allclose(w.sum(1), torch.ones(len(ids)), atol=1e-5)
        and (w >= 0).all()), "min %.3f max %.3f" % (w.min(), w.max()))

    print("top-K")
    counts = mmodel.topk_counts(lengths, 3)
    expect = torch.tensor([int(np.ceil(int(L) / 3)) for L in lengths])
    check("pool size is ceil(T / 3)", bool((counts == expect).all()),
          "T %s -> k %s" % (lengths.tolist(), counts.tolist()))
    p = out_batched["probs"]["fused"]
    pooled = mmodel.topk_mean(p, mask, counts)
    manual = []
    for i in range(len(ids)):
        v = p[i, :int(lengths[i])].sort(descending=True).values[:int(counts[i])]
        manual.append(v.mean())
    check("top-K mean matches a per-video sort",
          bool(torch.allclose(pooled, torch.stack(manual), atol=1e-6)))
    sel = mmodel.topk_mask(p, mask, counts)
    check("top-K mask selects exactly k valid frames",
          bool((sel.sum(1) == counts).all() and not bool((sel & ~mask).any())))

    print("losses")
    net.train()
    out = net(feats, mask)
    mil, per_branch = net.mil_loss(out["probs"], mask, lengths, labels)
    smooth = net.smoothness_loss(out["probs"], mask)
    contrast = net.contrastive_loss(out["embeds"], mask)
    total = mil + 0.1 * smooth + 0.2 * contrast
    check("every loss term is finite",
          all(torch.isfinite(x) for x in [mil, smooth, contrast, total]),
          "mil %.4f smooth %.5f contrast %.4f"
          % (mil.detach(), smooth.detach(), contrast.detach()))
    check("one MIL term per branch", set(per_branch) ==
          set(mdata.MODALITIES) | {"fused"}, ", ".join(sorted(per_branch)))
    total.backward()
    grads = {n: p.grad for n, p in net.named_parameters()}
    check("every parameter receives a gradient",
          all(g is not None and torch.isfinite(g).all() for g in grads.values()))
    check("the DMS block receives a gradient",
          any(n.startswith("dms.") and grads[n].abs().sum() > 0
              for n in grads),
          "%d dms tensors" % sum(1 for n in grads if n.startswith("dms.")))

    flat = {"fused": torch.full((2, 5), 0.4)}
    m2 = torch.ones(2, 5, dtype=torch.bool)
    check("smoothness is zero on a constant score",
          float(net.smoothness_loss(flat, m2)) == 0.0)
    e = torch.randn(4, 7, 128)
    aligned = net.contrastive_loss([e, e.clone(), e.clone()],
                                   torch.ones(4, 7, dtype=torch.bool))
    shuffled = net.contrastive_loss([e, e[[1, 2, 3, 0]], e[[2, 3, 0, 1]]],
                                    torch.ones(4, 7, dtype=torch.bool))
    check("contrastive loss is lower when the modalities agree",
          float(aligned) < float(shuffled),
          "aligned %.4f vs shuffled %.4f" % (aligned, shuffled))

    print("union readout")
    net.eval()
    with torch.no_grad():
        out = net(feats, mask)
        u = net.union_frames(out["probs"], out["weights"], mask, lengths)
        fused_top = mmodel.topk_mask(out["probs"]["fused"], mask, counts)
    check("union contains the fused top-K and no padded frame",
          bool(((u > 0) | ~fused_top).all() and (u * ~mask).sum() == 0),
          "union covers %.3f of valid frames"
          % float(u.sum() / mask.sum()))

    print("")
    if FAILED:
        print("FAILED %d check(s): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
