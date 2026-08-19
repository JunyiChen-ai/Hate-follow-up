"""VadCLIP hyperparameters.

Defaults are the published XD-Violence preset (VadCLIP @ c41067f
src/xd_option.py) with three documented changes. The XD preset, not the
UCF-Crime one, is the right ancestor: XD-Violence is scored as a binary
anomaly task with a handful of anomaly subclasses, which is the shape of the
hateful/normal collapse here, whereas UCF-Crime is a 13-way classification
benchmark and its preset (visual-layers 2, attn-window 8, lr 2e-5,
milestones [4, 8]) is tuned for that.

Published XD values kept verbatim
    embed-dim 512, visual-width 512, visual-head 1, visual-layers 1,
    prompt-prefix 10, prompt-postfix 10, max-epoch 10, batch-size 96,
    lr 1e-5, scheduler-rate 0.1, scheduler-milestones [3, 6, 10], seed 234

Changed, with reasons
    classes-num  7 -> 2       one normal class, one hateful class
    visual-length 256 -> per corpus, see hate_common.runtime.default_visual_length
    attn-window   64 -> per corpus, same place
"""

import argparse

from hate_common import runtime


def build_parser():
    parser = argparse.ArgumentParser(
        description="VadCLIP (AAAI'24) on the hateful-video corpora",
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
    parser.add_argument("--visual-length", default=None, type=int,
                        help="rows per block; one row is one second. "
                             "Default 256 for hatemm and hateclipseg, 64 for mhclip_*")
    parser.add_argument("--attn-window", default=None, type=int,
                        help="local attention window in rows. "
                             "Default 64 for hatemm and hateclipseg, 16 for mhclip_*")

    # --- optimisation (published XD values) ---
    parser.add_argument("--max-epoch", default=10, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--scheduler-rate", default=0.1, type=float)
    parser.add_argument("--scheduler-milestones", default=[3, 6, 10],
                        type=int, nargs="+")
    parser.add_argument("--loss3-weight", default=1e-4, type=float,
                        help="published XD weight on the text orthogonality term")

    # --- model selection (this port; see hate_common.data.split_train_val) ---
    parser.add_argument("--val-frac", default=0.1, type=float,
                        help="stratified holdout carved from the train split; "
                             "0 reproduces upstream's no-selection behaviour")
    parser.add_argument("--select", default="val",
                        choices=["val", "last"],
                        help="'val' keeps the best video-level val AP epoch, "
                             "'last' keeps the final epoch")

    # --- io ---
    parser.add_argument("--out-dir", default=None,
                        help="default results/reproduction/baselines/vadclip/<corpus>")
    parser.add_argument("--clip-download-root", default=None,
                        help="where clip.load looks for ViT-B-16.pt "
                             "(default ~/.cache/clip)")
    return parser


def resolve(args):
    """Fill in the per-corpus defaults left as None."""
    if args.visual_length is None:
        args.visual_length = runtime.default_visual_length(args.corpus)
    if args.attn_window is None:
        args.attn_window = runtime.default_attn_window(args.corpus)
    args.device = runtime.resolve_device(args.device)
    return args
