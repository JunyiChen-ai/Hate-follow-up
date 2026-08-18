"""MACIL-SD inference, from MACIL_SD @ c20943f test.py + infer.py.

Emits per-video frame-level score arrays on the 1 fps gold grid, one JSON
object per line, to
results/reproduction/baselines/<method>/<corpus>/scores.jsonl.

Branches written for `--modality av`, all upstream formulas:

    score_av      mean over the five crops of sigmoid(av_logits). This is
                  upstream's `pred`, the number MACIL-SD reports.
    score_audio   mean over the five crops of the audio-branch frame
                  probability, `sigmoid(frame_prob[:, :, 0, :])`.
    score_visual  the same for the visual branch, column 1.

For `--modality audio` and `--modality visual` there is one branch,
`score_mil`, the sigmoid of `Single_Model`'s per-frame logit -- upstream's
`pred3`.

Patches
    M4  `squeeze(-1)` rather than `squeeze()`. The audio-only path forwards a
        single crop, and a bare squeeze would flatten the crop axis away and
        turn the crop mean into a scalar.
    M12 `from tSNE import batch_tsne` and the `if i == 10000:` block it guards
        are removed. The branch is unreachable -- `i` is a batch index over a
        few hundred videos -- and the import pulls in matplotlib and
        scikit-learn for nothing.
    M14 Snippet-to-second remapping replaces upstream's `np.repeat(pred, 16)`.
        Upstream lifted one snippet score onto 16 frames of a 24 fps grid. The
        target grid here is the study's 1 fps gold grid, so the ratio is
        1 s / 0.666667 s, not an integer, and the map is a midpoint lookup with
        the tail held. macilsd/align.py states the rule and argues for it;
        `--grid second` makes it the identity.
    M15 Scoring is separated from inference. Upstream's `avce_test` computes
        `precision_recall_curve` inline and returns an AP. This port writes
        score arrays and does no scoring; eval_baseline_scores.py reads the
        jsonl and calls scripts/duplex/frame_eval_common.py, so every method in
        the study is scored by one implementation. It also removes the
        scikit-learn dependency.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hate_common import data as hdata          # noqa: E402

from . import option                           # noqa: E402
from .dataset import MacilTestDataset, usable_ids  # noqa: E402
from .train import build_models                # noqa: E402


def _to_gold(vec, index_map):
    """Apply the row -> gold-second lookup produced by the test dataset."""
    return np.asarray(vec, dtype=np.float64)[np.asarray(index_map)]


def infer(args, model_path, split="test"):
    device = args.device
    gt = hdata.gt_arrays(args.corpus, split)
    # Score exactly the cohort the gold covers, as the other two ports do.
    ids = [v for v in usable_ids(args.corpus, hdata.load_split(args.corpus, split))
           if v in gt]
    if args.limit_videos:
        ids = ids[:args.limit_videos]

    dataset = MacilTestDataset(args.corpus, ids, args.max_seqlen, args.grid,
                               args.modality)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    model_av, model_uni = build_models(args)
    model = model_av if model_av is not None else model_uni
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    out_path = os.path.join(args.out_dir, "scores.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_written = 0
    with torch.no_grad(), open(out_path, "w") as fh:
        for f_v, f_a, index_map, n_seconds, vid in loader:
            vid = vid[0]
            n_seconds = int(n_seconds)
            index_map = index_map[0].numpy()
            f_v = f_v[0].to(device)
            f_a = f_a[0].to(device)

            if args.modality == "av":
                out = model(f_a, f_v, seq_len=None)
                _mmil, a_logits, v_logits, av_logits, _v_out, _a_out = out
                # PORT PATCH (patch M4): squeeze(-1), then the crop mean over
                # dim 0 -- upstream's `torch.mean(av_logits, 0)` with its
                # batch_size=5 test loader, which is one video's five crops.
                av = torch.sigmoid(av_logits.squeeze(-1)).mean(0)
                au = a_logits.squeeze(-1).mean(0)
                vi = v_logits.squeeze(-1).mean(0)
                branches = {
                    "score_av": _to_gold(av.cpu(), index_map),
                    "score_audio": _to_gold(au.cpu(), index_map),
                    "score_visual": _to_gold(vi.cpu(), index_map),
                }
            else:
                feat = f_a if args.modality == "audio" else f_v
                logits = model(feat, seq_len=None)
                mil = torch.sigmoid(logits.squeeze(-1)).mean(0)
                branches = {"score_mil": _to_gold(mil.cpu(), index_map)}

            for name, arr in branches.items():
                if arr.shape[0] != n_seconds:
                    raise RuntimeError("video %s / %s: produced %d rows, "
                                       "expected %d"
                                       % (vid, name, arr.shape[0], n_seconds))
            if len(gt[vid]) != n_seconds:
                raise RuntimeError("video %s: %d gold frames but %d audio rows"
                                   % (vid, len(gt[vid]), n_seconds))

            rec = {"video_id": vid, "n_frames": n_seconds}
            for name, arr in branches.items():
                rec[name] = [round(float(x), 6) for x in arr]
            fh.write(json.dumps(rec) + "\n")
            n_written += 1

    print("wrote %d videos to %s" % (n_written, out_path))
    return out_path


def main(argv=None):
    parser = option.build_parser()
    parser.add_argument("--model-path", default=None,
                        help="default <out-dir>/model.pth")
    parser.add_argument("--split", default="test")
    args = option.resolve(parser.parse_args(argv))
    model_path = args.model_path or os.path.join(args.out_dir, "model.pth")
    infer(args, model_path, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
