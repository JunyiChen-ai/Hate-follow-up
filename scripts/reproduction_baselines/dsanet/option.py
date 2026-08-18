"""DSANet hyperparameters.

Defaults are the published XD-Violence preset (DSANet @ eb335b2
src/xd_option.py), for the same reason VadCLIP's port takes the XD preset:
XD-Violence is scored as a binary anomaly task, UCF-Crime as 13-way
classification. DSANet's UCF preset differs on eight values
(visual-layers 2, attn-window 8, lr 7e-5, batch-size 64, text_adapt_until 3,
t_w 0.1, temp 5.0, loss2_weight 1.1) and is not used here.

Published XD values kept verbatim
    embed-dim 512, visual-width 512, visual-head 1, visual-layers 1,
    prompt-prefix 10, prompt-postfix 10, max-epoch 10, batch-size 96,
    lr 1e-5, seed 234, decoder_depth 8, normal_selection_ratio 0.8,
    DNP_use True, num_prototypes 16, loss2_weight 5.0, temp 1.0,
    text_adapt_until 1, t_w 0.6

Changed, with reasons
    classes-num  7 -> 2       one normal class, one hateful class
    visual-length 256 -> per corpus, see hate_common.runtime.default_visual_length
    attn-window   64 -> per corpus, same place
"""

import argparse

from hate_common import runtime


def build_parser():
    parser = argparse.ArgumentParser(
        description="DSANet (AAAI'26) on the hateful-video corpora",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    runtime.add_common_args(parser)

    # --- architecture (published XD values) ---
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=1, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=2, type=int,
                        help="published XD value 7; 2 after the binary collapse")

    # --- temporal window, re-read in seconds ---
    parser.add_argument("--visual-length", default=None, type=int)
    parser.add_argument("--attn-window", default=None, type=int)

    # --- self-guided normality modelling (published XD values) ---
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--normal_selection_ratio", default=0.8, type=float)
    parser.add_argument("--DNP_use", default=True, type=lambda s:
                        str(s).lower() not in ("0", "false", "no"))
    parser.add_argument("--num_prototypes", default=16, type=int)

    # --- text adapter and losses (published XD values) ---
    parser.add_argument("--text_adapt_until", default=1, type=int)
    parser.add_argument("--t_w", default=0.6, type=float)
    parser.add_argument("--loss2_weight", default=5.0, type=float)
    parser.add_argument("--temp", default=1.0, type=float)

    # --- optimisation (published XD values) ---
    parser.add_argument("--max-epoch", default=10, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--warmup-iters", default=100, type=int,
                        help="published value for the refiner's cosine schedule")

    # --- model selection (this port) ---
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--select", default="val", choices=["val", "last"])

    # --- io ---
    parser.add_argument("--out-dir", default=None,
                        help="default results/reproduction/baselines/dsanet/<corpus>")
    parser.add_argument("--clip-download-root", default=None)
    return parser


def resolve(args):
    if args.visual_length is None:
        args.visual_length = runtime.default_visual_length(args.corpus)
    if args.attn_window is None:
        args.attn_window = runtime.default_attn_window(args.corpus)
    args.device = runtime.resolve_device(args.device)
    return args
