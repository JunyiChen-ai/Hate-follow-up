#!/usr/bin/env python
"""CPU-only smoke test for the MACIL-SD port. Never touches a GPU.

Separate from smoke_cpu.py, which covers the VadCLIP and DSANet ports and is
being executed by another run; nothing here imports or modifies it.

What it checks:

  1. `resample_intervals` is exact on cases with a known answer: an identity
     grid, a two-into-one average, a one-into-two split, and the hold-last rule
     past the end of the source coverage.
  2. `snippet_index_for_seconds` maps gold second i to the snippet containing
     i + 0.5, and clamps the dropped-tail seconds to the last snippet.
  3. On real files, `aligned_pair` returns equal-length visual and audio on
     both grids, and `scores_to_gold_grid` returns exactly the gold length.
  4. Both models build on cpu, a synthetic training batch forwards, every
     logit tensor has the expected shape, and every loss in the published sum
     is finite -- at batch sizes 4 and 1, since upstream's bare `squeeze()`
     breaks at 1.
  5. Backward runs and every parameter that received a gradient has a finite
     one, for the audio-visual model and for both uni-modal ablations.
  6. The EMA distillation step runs and moves the parameters it should while
     leaving the audio-specific projection alone.
  7. The inference path forwards a five-crop item and returns exactly
     `n_seconds` rows per branch.
  8. The real plumbing agrees with the frozen gold: every gold video has both
     feature files, and its VGGish row count equals its gold array length.

    CUDA_VISIBLE_DEVICES="" python scripts/reproduction_baselines/smoke_cpu_macilsd.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "duplex")))

from hate_common import data as hdata          # noqa: E402
from macilsd import align                      # noqa: E402
from macilsd import option                     # noqa: E402
from macilsd.avce_network import AVCE_Model, Single_Model  # noqa: E402
from macilsd.CMA_MIL import CMAL               # noqa: E402
from macilsd.dataset import MacilTestDataset, MacilTrainDataset  # noqa: E402
from macilsd.train import (build_models, distil_step, evaluate_video_ap,
                           train_one_epoch, _seq_len_of)  # noqa: E402

DEVICE = "cpu"
_failures = []


def check(name, ok, detail=""):
    print("%-64s %s%s" % (name, "OK" if ok else "FAIL",
                          ("  " + detail) if detail else ""))
    if not ok:
        _failures.append(name)
    return bool(ok)


def base_args(**over):
    args = option.build_parser().parse_args([])
    args.device = DEVICE
    for k, v in over.items():
        setattr(args, k, v)
    return args


# ------------------------------------------------------------------ alignment
def smoke_align():
    print("\n--- alignment ---")
    # Identity: same grid in, same values out.
    src = np.arange(12, dtype=np.float32).reshape(4, 3)
    b = align.second_bounds(4)
    out = align.resample_intervals(src, b, b)
    check("resample: identity grid is exact", np.allclose(out, src))

    # Two source seconds into one two-second destination: plain mean.
    src = np.array([[0.0], [4.0]], dtype=np.float32)
    out = align.resample_intervals(src, align.second_bounds(2),
                                   np.array([[0.0, 2.0]]))
    check("resample: two seconds into one window is their mean",
          np.allclose(out, [[2.0]]), "%.3f" % out[0, 0])

    # One source second into two half-second destinations: both take its value.
    src = np.array([[7.0]], dtype=np.float32)
    out = align.resample_intervals(src, np.array([[0.0, 1.0]]),
                                   np.array([[0.0, 0.5], [0.5, 1.0]]))
    check("resample: splitting one row copies it to both halves",
          np.allclose(out, [[7.0], [7.0]]))

    # Overlap weighting: a window covering 2/3 of row 0 and 1/3 of row 1.
    src = np.array([[0.0], [3.0]], dtype=np.float32)
    out = align.resample_intervals(src, align.second_bounds(2),
                                   np.array([[0.0, 1.5]]))
    check("resample: weights are overlap length, not row count",
          abs(float(out[0, 0]) - 1.0) < 1e-6, "%.6f, want 1.0" % out[0, 0])

    # Past the end of the source: hold the last row.
    src = np.array([[1.0], [2.0]], dtype=np.float32)
    out = align.resample_intervals(src, align.second_bounds(2),
                                   np.array([[5.0, 6.0]]))
    check("resample: a window past the source holds the last row",
          np.allclose(out, [[2.0]]), "%.3f" % out[0, 0])

    # Snippet lookup: 0.666667 s snippets, gold seconds 0, 1, 2.
    snip = np.array([[i * 2.0 / 3.0, (i + 1) * 2.0 / 3.0] for i in range(5)])
    idx = align.snippet_index_for_seconds(snip, 4)
    # midpoints 0.5, 1.5, 2.5, 3.5 -> snippets 0, 2, 3, 4 (4 is the clamp)
    check("snippet lookup: gold second takes the snippet at its midpoint",
          list(idx) == [0, 2, 3, 4], str(list(idx)))
    idx = align.snippet_index_for_seconds(snip[:2], 4)
    check("snippet lookup: the dropped tail holds the last snippet",
          list(idx) == [0, 1, 1, 1], str(list(idx)))


def smoke_align_real(corpus="hatemm", n=6):
    print("\n--- alignment on real files ---")
    gt = hdata.gt_arrays(corpus, "test")
    ids = [v for v in hdata.load_split(corpus, "test") if v in gt][:n]
    ok_pair = ok_len = ok_map = True
    detail = ""
    for grid in align.GRIDS:
        for vid in ids:
            v, a, n_sec, snip = align.aligned_pair(corpus, vid, grid)
            ok_pair &= (v.shape[0] == a.shape[0])
            ok_len &= (n_sec == len(gt[vid]))
            fake = np.arange(v.shape[0], dtype=float)
            lifted = align.scores_to_gold_grid(fake, snip, n_sec, grid)
            ok_map &= (lifted.shape[0] == n_sec)
            if grid == "snippet" and vid == ids[0]:
                detail = ("%s: %d snippets, %d gold seconds"
                          % (vid, v.shape[0], n_sec))
    check("aligned_pair: visual and audio have equal length on both grids",
          ok_pair, detail)
    check("aligned_pair: n_seconds equals the gold array length", ok_len)
    check("scores_to_gold_grid: output length is the gold length", ok_map)

    # The crop-wise fast path must equal the full-array reference path.
    vid = ids[0]
    same = True
    for grid in align.GRIDS:
        v_ref, _a, n_sec, snip = align.aligned_pair(corpus, vid, grid)
        for c in range(align.N_CROPS):
            v_c = align.aligned_visual_crop(corpus, vid, c, grid, n_sec, snip)
            same &= np.allclose(v_c, v_ref[:, c, :], atol=1e-5)
    check("aligned_visual_crop matches the full-array path on both grids", same)


# --------------------------------------------------------------------- models
def synthetic_batch(batch, seqlen, v_dim=1024, a_dim=128, seed=0):
    """A batch shaped like MacilTrainDataset's output, with real padding."""
    g = torch.Generator().manual_seed(seed)
    f_v = torch.randn(batch, seqlen, v_dim, generator=g)
    f_a = torch.randn(batch, seqlen, a_dim, generator=g)
    lengths = [seqlen, max(seqlen // 2, 20), 32, 17][:batch]
    for i, n in enumerate(lengths):
        f_v[i, n:] = 0.0
        f_a[i, n:] = 0.0
    label = torch.tensor([1.0, 0.0, 1.0, 0.0][:batch])
    return f_v, f_a, label


def smoke_models():
    print("\n--- models ---")
    seqlen = 64
    for batch in (4, 1):
        args = base_args(modality="av")
        torch.manual_seed(0)
        model_av = AVCE_Model(args)
        model_uni = Single_Model(args, n_dim=args.v_feature_size)
        n = (sum(p.numel() for p in model_av.parameters())
             + sum(p.numel() for p in model_uni.parameters()))
        if batch == 4:
            check("av: builds on cpu", True, "%.3fM params" % (n / 1e6))

        f_v, f_a, label = synthetic_batch(batch, seqlen)
        seq_len = _seq_len_of(f_v)
        keep = int(torch.max(seq_len))
        f_v, f_a = f_v[:, :keep], f_a[:, :keep]

        out = model_av(f_a, f_v, seq_len)
        mmil, a_log, v_log, av_log, v_out, a_out = out
        shapes_ok = (tuple(mmil.shape) == (batch,)
                     and tuple(a_log.shape) == (batch, keep, 1)
                     and tuple(v_log.shape) == (batch, keep, 1)
                     and tuple(av_log.shape) == (batch, keep, 1)
                     and tuple(v_out.shape) == (batch, keep, args.hid_dim))
        check("av: forward shapes at batch %d" % batch, shapes_ok,
              "mmil %s av_logits %s" % (tuple(mmil.shape), tuple(av_log.shape)))

        crit = torch.nn.BCELoss()
        mmil = mmil.reshape(-1)
        clsloss = crit(mmil, label)
        terms = CMAL(mmil, a_log.squeeze(-1), v_log.squeeze(-1), seq_len,
                     v_out, a_out)
        finite = torch.isfinite(clsloss) and all(
            (torch.isfinite(t) if torch.is_tensor(t) else np.isfinite(t))
            for t in terms)
        check("av: every published loss term is finite at batch %d" % batch,
              bool(finite),
              "cls %.4f cma %s" % (clsloss,
                                   " ".join("%.3f" % float(t) for t in terms)))

        total = clsloss + 1.5 * (terms[0] + terms[2]) + 1.5 * (terms[1] + terms[3])
        total.backward()
        grads = [p.grad for p in model_av.parameters() if p.grad is not None]
        check("av: backward gives finite gradients at batch %d" % batch,
              len(grads) > 0 and all(torch.isfinite(g).all() for g in grads),
              "%d tensors with a gradient" % len(grads))


def smoke_cmal():
    """CMAL with both of its self-guided branches populated.

    `CMAL` returns four literal 0.0 floats unless the batch contains at least
    one video the model currently scores above 0.5 and at least one it scores
    below, because the abnormal and normal banks are filled by that test. A
    freshly initialised model puts every video on one side, so the loss terms
    in smoke_models() are the short circuit, not the formula. This drives the
    formula directly with a bag-score vector that straddles the threshold.
    """
    print("\n--- CMAL with both branches populated ---")
    args = base_args(modality="av")
    torch.manual_seed(0)
    model = AVCE_Model(args)
    batch, seqlen = 4, 64
    f_v, f_a, _label = synthetic_batch(batch, seqlen)
    seq_len = _seq_len_of(f_v)
    keep = int(torch.max(seq_len))
    out = model(f_a[:, :keep], f_v[:, :keep], seq_len)
    _mmil, a_log, v_log, _av, v_out, a_out = out
    straddle = torch.tensor([0.9, 0.1, 0.8, 0.2])

    terms = CMAL(straddle, a_log.squeeze(-1), v_log.squeeze(-1), seq_len,
                 v_out, a_out)
    nonzero = all(torch.is_tensor(t) for t in terms)
    finite = all(torch.isfinite(t).all() for t in terms if torch.is_tensor(t))
    check("CMAL: all four InfoNCE terms are computed, not short-circuited",
          nonzero, " ".join("%.4f" % float(t) for t in terms))
    check("CMAL: all four terms are finite", bool(finite))

    total = sum(terms)
    total.backward(retain_graph=True)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("CMAL: backward gives finite gradients",
          len(grads) > 0 and all(torch.isfinite(g).all() for g in grads),
          "%d tensors with a gradient" % len(grads))

    # --fix-rep-swap must change the value; if it did not, the swap would be
    # cosmetic and the note in avce_network.py would be wrong.
    swapped = CMAL(straddle, a_log.squeeze(-1), v_log.squeeze(-1), seq_len,
                   a_out, v_out)
    differs = any(abs(float(x) - float(y)) > 1e-6
                  for x, y in zip(terms, swapped))
    check("CMAL: --fix-rep-swap changes the loss, so the swap is load bearing",
          differs,
          "upstream %.4f vs fixed %.4f" % (float(sum(terms)),
                                           float(sum(swapped))))

    # An all-normal batch must hit the documented short circuit.
    zeros = CMAL(torch.zeros(batch), a_log.squeeze(-1), v_log.squeeze(-1),
                 seq_len, v_out, a_out)
    check("CMAL: an all-normal batch returns the four-zero short circuit",
          all((not torch.is_tensor(t)) and t == 0.0 for t in zeros))


def smoke_unimodal():
    print("\n--- uni-modal ablations ---")
    seqlen = 64
    for modality, dim in (("audio", 128), ("visual", 1024)):
        args = base_args(modality=modality)
        torch.manual_seed(0)
        _av, model = build_models(args)
        check("%s: build_models returns Single_Model alone" % modality,
              _av is None and isinstance(model, Single_Model),
              "%.3fM params" % (sum(p.numel() for p in model.parameters()) / 1e6))
        check("%s: input width is %d" % (modality, dim),
              model.fc_v.in_features == dim, str(model.fc_v.in_features))

        f_v, f_a, label = synthetic_batch(4, seqlen)
        feat = f_a if modality == "audio" else f_v
        seq_len = _seq_len_of(f_v)
        keep = int(torch.max(seq_len))
        logits = model(feat[:, :keep], seq_len).reshape(-1)
        loss = torch.nn.BCELoss()(logits, label)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        check("%s: bag score in [0,1] and finite backward" % modality,
              bool((logits >= 0).all() and (logits <= 1).all()
                   and torch.isfinite(loss)
                   and all(torch.isfinite(g).all() for g in grads)),
              "loss %.4f" % loss)

        # Inference path: no seq_len, per-frame logits kept.
        out = model(feat[:1, :keep], seq_len=None)
        check("%s: inference path returns per-frame logits" % modality,
              tuple(out.shape) == (1, keep, 1), str(tuple(out.shape)))


def smoke_distil():
    print("\n--- self-distillation ---")
    args = base_args(modality="av")
    torch.manual_seed(0)
    model_av = AVCE_Model(args)
    model_uni = Single_Model(args, n_dim=args.v_feature_size)
    with torch.no_grad():
        for p in model_uni.parameters():
            p.fill_(1.0)
    before = {k: v.clone() for k, v in model_av.state_dict().items()}
    m = distil_step(args, model_av, model_uni, epoch=0)
    after = model_av.state_dict()
    moved = [k for k in before if not torch.equal(before[k], after[k])]
    check("distil: the mixing rate is the published cosine value",
          abs(m - args.m) < 1e-9, "m=%.4f at epoch 0" % m)
    check("distil: shared parameters move", len(moved) > 0,
          "%d tensors changed" % len(moved))
    check("distil: the audio projection fc_a is left alone",
          not any(k.startswith("fc_a") for k in moved))
    check("distil: the MIL head is mixed through the name remap",
          "att_mmil.fc.weight" in moved)
    # cosine_scheduler reaches 1.0 at the final epoch, where upstream skips.
    m_last = distil_step(args, model_av, model_uni, epoch=args.ema_epochs)
    check("distil: the rate reaches 1.0 at the last epoch and is a no-op",
          abs(m_last - 1.0) < 1e-12, "m=%.6f" % m_last)


# ------------------------------------------------------------ real dry run
def smoke_dataset(corpus="hatemm", n_videos=4):
    print("\n--- dataset on real files ---")
    labels = hdata.load_labels(corpus)
    gt = hdata.gt_arrays(corpus, "test")
    ids = [v for v in hdata.load_split(corpus, "train")
           if align.has_features(corpus, v)][:n_videos]

    ds = MacilTrainDataset(corpus, ids, labels, 200, "snippet", "av", 5)
    check("train dataset: 5 items per video, upstream's five-crop list",
          len(ds) == 5 * len(ids), "%d items for %d videos" % (len(ds), len(ids)))
    f_v, f_a, label = ds[0]
    check("train dataset: item is (max_seqlen, 1024) and (max_seqlen, 128)",
          tuple(f_v.shape) == (200, 1024) and tuple(f_a.shape) == (200, 128),
          "%s %s label %.0f" % (tuple(f_v.shape), tuple(f_a.shape), label))
    # The five crops of one video must differ in the visual and agree in audio.
    v0, a0, _ = ds[0]
    v4, a4, _ = ds[4]
    check("train dataset: crops differ visually, share one audio array",
          not torch.equal(v0, v4) and torch.equal(a0, a4))

    ds_a = MacilTrainDataset(corpus, ids, labels, 200, "snippet", "audio", 5)
    fv, fa, _ = ds_a[0]
    check("train dataset: audio-only keeps the 5N index space",
          len(ds_a) == 5 * len(ids) and tuple(fa.shape) == (200, 128),
          "%d items, visual placeholder %s" % (len(ds_a), tuple(fv.shape)))

    tids = [v for v in hdata.load_split(corpus, "test") if v in gt][:n_videos]
    ts = MacilTestDataset(corpus, tids, 200, "snippet", "av")
    f_v, f_a, idx, n_sec, vid = ts[0]
    check("test dataset: five crops stacked, audio repeated to match",
          (f_v.shape[0] == 5 and f_a.shape[0] == 5
           and f_v.shape[1] == f_a.shape[1]),
          "%s %s" % (tuple(f_v.shape), tuple(f_a.shape)))
    check("test dataset: index map has one entry per gold frame",
          int(n_sec) == len(gt[vid]) == idx.shape[0],
          "%s: %d gold frames" % (vid, int(n_sec)))

    ts_a = MacilTestDataset(corpus, tids, 200, "snippet", "audio")
    f_v, f_a, idx, n_sec, vid = ts_a[0]
    check("test dataset: audio-only forwards one crop, not five",
          f_a.shape[0] == 1, str(tuple(f_a.shape)))


def smoke_gold_cohort():
    print("\n--- gold cohort ---")
    for corpus in hdata.CORPORA:
        gt = hdata.gt_arrays(corpus, "test")
        missing = [v for v in gt if not align.has_features(corpus, v)]
        bad = []
        for vid in gt:
            if vid in missing:
                continue
            n_rows = np.load(align.audio_path(corpus, vid),
                             mmap_mode="r").shape[0]
            if n_rows != len(gt[vid]):
                bad.append(vid)
        check("%s: every gold video has an I3D and a VGGish file" % corpus,
              not missing, "%d gold videos" % len(gt))
        check("%s: VGGish rows equal gold frames for every gold video" % corpus,
              not bad, "%d checked" % (len(gt) - len(missing)))


def smoke_epoch(corpus="hatemm", n_videos=6):
    """One real training epoch on a handful of videos, both model families."""
    print("\n--- one real epoch on cpu ---")
    from torch.utils.data import DataLoader
    labels = hdata.load_labels(corpus)
    ids = [v for v in hdata.load_split(corpus, "train")
           if align.has_features(corpus, v)]
    pos = [v for v in ids if labels[v] == 1][:n_videos // 2]
    neg = [v for v in ids if labels[v] == 0][:n_videos // 2]
    ids = pos + neg

    for modality in ("av", "audio", "visual"):
        args = base_args(modality=modality, max_seqlen=64, batch_size=4,
                         num_workers=0)
        torch.manual_seed(0)
        model_av, model_uni = build_models(args)
        opt_av = (torch.optim.Adam(model_av.parameters(), lr=args.lr)
                  if model_av is not None else None)
        opt_uni = torch.optim.Adam(model_uni.parameters(),
                                   lr=args.lr * args.single_lr_scale)
        ds = MacilTrainDataset(corpus, ids, labels, args.max_seqlen,
                               args.grid, modality, args.crop_repeat)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0)
        means, n_batches = train_one_epoch(
            args, model_av, model_uni, opt_av, opt_uni, torch.nn.BCELoss(),
            loader, 0.0, 0.0, DEVICE)
        check("%s: a real epoch runs and every mean loss is finite" % modality,
              n_batches > 0 and bool(np.isfinite(means).all()),
              "%d batches, cls %.4f uni %.4f" % (n_batches, means[0], means[5]))
        ap = evaluate_video_ap(args, model_av, model_uni, loader, DEVICE)
        check("%s: video-level val AP is computable" % modality,
              ap is None or (0.0 <= ap <= 1.0),
              "AP %s" % ("%.4f" % ap if ap is not None else "n/a"))


def smoke_infer(corpus="hatemm", n_videos=3):
    print("\n--- inference path on real files ---")
    gt = hdata.gt_arrays(corpus, "test")
    ids = [v for v in hdata.load_split(corpus, "test") if v in gt][:n_videos]
    for modality in ("av", "audio", "visual"):
        args = base_args(modality=modality)
        torch.manual_seed(0)
        model_av, model_uni = build_models(args)
        model = model_av if model_av is not None else model_uni
        model.eval()
        ts = MacilTestDataset(corpus, ids, args.max_seqlen, args.grid, modality)
        ok = True
        detail = ""
        with torch.no_grad():
            for i in range(len(ts)):
                f_v, f_a, idx, n_sec, vid = ts[i]
                n_sec = int(n_sec)
                if modality == "av":
                    out = model(f_a, f_v, seq_len=None)
                    vec = torch.sigmoid(out[3].squeeze(-1)).mean(0)
                else:
                    feat = f_a if modality == "audio" else f_v
                    vec = torch.sigmoid(model(feat, seq_len=None).squeeze(-1)).mean(0)
                lifted = np.asarray(vec)[idx.numpy()]
                ok &= (lifted.shape[0] == n_sec == len(gt[vid])
                       and np.isfinite(lifted).all()
                       and (0.0 <= lifted).all() and (lifted <= 1.0).all())
                if i == 0:
                    detail = ("%s: %d model rows -> %d gold frames"
                              % (vid, vec.shape[0], n_sec))
        check("%s: inference lifts scores onto the gold grid" % modality, ok,
              detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="hatemm", choices=list(hdata.CORPORA))
    ap.add_argument("--skip-real", action="store_true",
                    help="synthetic checks only")
    a = ap.parse_args()

    if torch.cuda.is_available():
        print("NOTE: a GPU is visible; this script pins everything to cpu "
              "anyway and issues no cuda calls.")
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    smoke_align()
    smoke_models()
    smoke_cmal()
    smoke_unimodal()
    smoke_distil()
    if not a.skip_real:
        smoke_align_real(a.corpus)
        smoke_dataset(a.corpus)
        smoke_gold_cohort()
        smoke_epoch(a.corpus)
        smoke_infer(a.corpus)

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
