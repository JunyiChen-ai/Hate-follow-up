"""DSANet inference, ported from DSANet @ eb335b2 src/xd_test.py.

Emits per-video frame-level score arrays on the 1 fps grid to
results/reproduction/baselines/dsanet/<corpus>/scores.jsonl.

Three branches per video:

    score_mlp        sigmoid(logits1 / temp), the classification branch
                     (upstream's ap1)
    score_refined    1 - refine_scores_hierarchical(...)[:, 0], upstream's ap2
    score_align      1 - softmax(logits2)[:, 0], VadCLIP's alignment formula

Why the third branch exists. `refine_scores_hierarchical` splits the total
abnormal probability sigmoid(logits1) across the non-normal columns of the
alignment softmax, in proportion to those columns. With two classes there is
exactly one non-normal column, so the proportion is 1 and the refinement
returns sigmoid(logits1) unchanged: score_refined and score_mlp are the same
vector under the binary collapse, up to floating point. That is a property of
the collapse, not a porting error, and it means DSANet's headline branch
carries no alignment signal here. score_align recovers an alignment-only
reading of logits2 so the text branch can still be inspected. All three are
written; eval_baseline_scores.py picks one with --branch.

The `np.repeat(scores, 16)` upsampling is dropped for the reason given in
vadclip/infer.py: one row is one second is one gold frame.
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
from dsanet.model import DSANet                # noqa: E402
from dsanet import option                      # noqa: E402


def refine_scores_hierarchical(logits_mlp, logits_align, temp=1.0):
    """Verbatim upstream (DSANet @ eb335b2 src/xd_test.py)."""
    epsilon = 1e-12
    total_abnormal_prob = torch.sigmoid(logits_mlp / temp)
    total_normal_prob = 1.0 - total_abnormal_prob
    p_align = F.softmax(logits_align / temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p_align_abnormal = p_align_abnormal_only.sum(dim=1, keepdim=True)
    abnormal_distribution = p_align_abnormal_only / (sum_p_align_abnormal
                                                     + epsilon)
    final_abnormal_probs = total_abnormal_prob * abnormal_distribution
    return torch.cat([total_normal_prob, final_abnormal_probs], dim=1)


def infer(args, model_path, split="test"):
    device = args.device
    labels = hdata.load_labels(args.corpus)
    gt = hdata.gt_arrays(args.corpus, split)
    ids = [v for v in hdata.load_split(args.corpus, split) if v in gt]
    if args.limit_videos:
        ids = ids[:args.limit_videos]

    dataset = hdata.HateVideoDataset(args.corpus, ids, args.visual_length,
                                     True, labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    model = DSANet(args.classes_num, args.embed_dim, args.visual_length,
                   args.visual_width, args.visual_head, args.visual_layers,
                   args.attn_window, args.prompt_prefix, args.prompt_postfix,
                   args, device, clip_download_root=args.clip_download_root)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    prompt_text = list(hdata.PROMPT_TEXT)
    dnp_use = bool(args.DNP_use)
    out_path = args.out_dir and os.path.join(args.out_dir, "scores.jsonl")
    out_path = out_path or runtime.scores_out_path("dsanet", args.corpus)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_written = 0
    with torch.no_grad(), open(out_path, "w") as fh:
        for visual, _label, length, vid in loader:
            vid = vid[0]
            len_cur = int(length)
            visual, lengths = runtime.prepare_test_item(
                visual, len_cur, args.visual_length)
            visual = visual.to(device)

            out = model(visual, None, prompt_text, lengths.to(device), dnp_use)
            logits1, logits2 = out[1], out[2]
            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])

            refined = refine_scores_hierarchical(
                logits1[0:len_cur], logits2[0:len_cur], args.temp)
            score_refined = 1 - refined[:, 0]
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
                "score_refined": [round(float(x), 6) for x in score_refined.cpu()],
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
        runtime.scores_out_path("dsanet", args.corpus))
    model_path = args.model_path or os.path.join(out_dir, "model.pth")
    args.out_dir = out_dir
    infer(args, model_path, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
