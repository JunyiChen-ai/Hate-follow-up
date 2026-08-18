"""VadCLIP inference, ported from VadCLIP @ c41067f src/xd_test.py.

Emits per-video frame-level score arrays on the 1 fps grid, one JSON object
per line, to results/reproduction/baselines/vadclip/<corpus>/scores.jsonl.

Two branches are written for every video, both verbatim upstream formulas:

    score_mlp    sigmoid(logits1), the classification branch (upstream's ap1)
    score_align  1 - softmax(logits2)[:, 0], the visual-language alignment
                 branch (upstream's ap2, the number VadCLIP reports as its AP)

Upstream then does `np.repeat(scores, 16)` to lift snippet scores onto the
frame grid. That step is dropped here (patch V6): one feature row is one
second is one gold frame, so a length-T score vector already sits on the grid
the gold arrays use. The emitted array length equals the feature row count,
which hate_common.data has checked equals the gold array length for every test
video in all three corpora.

Scoring is not done here. eval_baseline_scores.py reads the jsonl and scores it
through scripts/duplex/frame_eval_common.py, so every method in the study goes
through one evaluator.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hate_common import data as hdata          # noqa: E402
from hate_common import runtime                # noqa: E402
from vadclip.model import CLIPVAD              # noqa: E402
from vadclip import option                     # noqa: E402


def infer(args, model_path, split="test"):
    device = args.device
    labels = hdata.load_labels(args.corpus)
    gt = hdata.gt_arrays(args.corpus, split)
    # Score exactly the cohort the gold covers: the split ids whose media was
    # present when results/reproduction/gt was built.
    ids = [v for v in hdata.load_split(args.corpus, split) if v in gt]
    if args.limit_videos:
        ids = ids[:args.limit_videos]

    dataset = hdata.HateVideoDataset(args.corpus, ids, args.visual_length,
                                     True, labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device, clip_download_root=args.clip_download_root)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    prompt_text = list(hdata.PROMPT_TEXT)
    out_path = args.out_dir and os.path.join(args.out_dir, "scores.jsonl")
    out_path = out_path or runtime.scores_out_path("vadclip", args.corpus)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_written = 0
    with torch.no_grad(), open(out_path, "w") as fh:
        for visual, _label, length, vid in loader:
            vid = vid[0]
            len_cur = int(length)
            visual, lengths = runtime.prepare_test_item(
                visual, len_cur, args.visual_length)
            visual = visual.to(device)

            _, logits1, logits2 = model(visual, None, prompt_text,
                                        lengths.to(device))
            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])

            score_mlp = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            score_align = 1 - F.softmax(logits2[0:len_cur], dim=-1)[:, 0]

            if score_mlp.shape[0] != len_cur:
                raise RuntimeError("video %s: produced %d rows, expected %d"
                                   % (vid, score_mlp.shape[0], len_cur))
            if len(gt[vid]) != len_cur:
                raise RuntimeError("video %s: %d feature rows but %d gold "
                                   "frames" % (vid, len_cur, len(gt[vid])))

            fh.write(json.dumps({
                "video_id": vid,
                "n_frames": len_cur,
                "score_mlp": [round(float(x), 6) for x in score_mlp.cpu()],
                "score_align": [round(float(x), 6) for x in score_align.cpu()],
            }) + "\n")
            n_written += 1

    print("wrote %d videos to %s" % (n_written, out_path))
    return out_path


def main(argv=None):
    parser = option.build_parser()
    parser.add_argument("--model-path", default=None,
                        help="default <out-dir>/model.pth")
    parser.add_argument("--split", default="test")
    args = option.resolve(parser.parse_args(argv))
    out_dir = args.out_dir or os.path.dirname(
        runtime.scores_out_path("vadclip", args.corpus))
    model_path = args.model_path or os.path.join(out_dir, "model.pth")
    args.out_dir = out_dir
    infer(args, model_path, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
