"""MACIL-SD hyperparameters, from MACIL_SD @ c20943f option.py.

PORT PATCH (patch O2). Every published value is kept verbatim. What changed is
only what had to: the six XD-Violence list/ground-truth/checkpoint paths are
gone, because this port reads the study's frozen manifests, and the flags that
name this port's own choices are added.

Published values kept verbatim
    lr 4e-4, batch-size 128, max_seqlen 200, max-epoch 50, m 0.91,
    lamda_a2b 1.5, lamda_a2n 1.5, lamda_cof 0.1, hid_dim 128, ffn_dim 128,
    nhead 4, dropout 0.1, num_classes 1, a_feature_size 128,
    v_feature_size 1024, seed 2333, single-lr-scale 1/5, sched-tmax 60,
    ema-epochs 50

Two of those deserve a note, because they look like transcription errors and
are not.

`--sched-tmax 60` against `--max-epoch 50`. Upstream builds
`CosineAnnealingLR(..., T_max=60)` while running 50 epochs, so the cosine never
reaches its trough: the run ends at lr * (1 + cos(50pi/60)) / 2 = 0.033 * lr.
It is left as published and exposed so the truncation is visible.

`--ema-epochs 50` against `--max-epoch 50`. The teacher-student mixing rate
comes from `cosine_scheduler(base_value=m, final_value=1, curr_epoch=e,
epochs=50)` with the 50 written as a literal, independent of `--max-epoch`.
Changing the epoch budget without this flag would silently reshape the
distillation schedule, so it is its own argument.

Not kept
    `--modality` upstream defaults to the string 'MIX2' and is read by
    `avce_dataset.Dataset.__init__` and then never used again -- no branch in
    the repository tests it. The name is reused here for a live choice between
    the audio-visual model and the two uni-modal ablations.
    `--num_stages 3` is likewise dead: nothing constructs a stack of stages.
    Dropped rather than carried as a decoration.
    `--workers 8` becomes `--num-workers` from hate_common.runtime, so all
    three ports in this study take the same common flags.

What `--max-seqlen 200` means here is argued in macilsd/utils.py: this study's
I3D features share XD-Violence's 24 fps decode and 16-frame snippet, so a row
is 0.666667 s in both places and 200 rows is 133.3 s in both places. Unlike the
VadCLIP and DSANet ports, whose 1 fps CLIP features forced `--visual-length` to
be re-read per corpus, nothing here needs re-reading.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hate_common import runtime                # noqa: E402

from .align import GRIDS                       # noqa: E402
from .dataset import MODALITIES                # noqa: E402

# Output directory stem per modality, so the ablation rows do not overwrite the
# headline run.
METHOD_NAME = {"av": "macilsd",
               "audio": "macilsd_audio",
               "visual": "macilsd_visual"}


def build_parser():
    parser = argparse.ArgumentParser(
        description="MACIL-SD (ACM MM'22) on the hateful-video corpora",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    runtime.add_common_args(parser)
    parser.set_defaults(seed=2333)  # upstream setup_seed(2333) in main.py

    # --- architecture (published values) ---
    parser.add_argument("--hid-dim", default=128, type=int)
    parser.add_argument("--ffn-dim", default=128, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num-classes", default=1, type=int,
                        help="published value 1: MACIL-SD is already a binary "
                             "MIL scorer, so the hateful/normal collapse needs "
                             "no change here")
    parser.add_argument("--a-feature-size", default=128, type=int,
                        help="VGGish width")
    parser.add_argument("--v-feature-size", default=1024, type=int,
                        help="I3D RGB width")

    # --- optimisation (published values) ---
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--max-epoch", default=50, type=int)
    parser.add_argument("--max-seqlen", default=200, type=int,
                        help="training rows per item; one row is one 0.666667 s "
                             "I3D snippet, as upstream")
    parser.add_argument("--sched-tmax", default=60, type=int,
                        help="CosineAnnealingLR T_max; upstream literal 60 "
                             "against 50 epochs")
    parser.add_argument("--single-lr-scale", default=0.2, type=float,
                        help="upstream trains the uni-modal partner at lr/5")

    # --- self-distillation and the contrastive terms (published values) ---
    parser.add_argument("--m", default=0.91, type=float,
                        help="teacher/student mixing base value")
    parser.add_argument("--ema-epochs", default=50, type=int,
                        help="upstream literal in cosine_scheduler; see the "
                             "module docstring")
    parser.add_argument("--lamda-a2b", default=1.5, type=float)
    parser.add_argument("--lamda-a2n", default=1.5, type=float)
    parser.add_argument("--lamda-cof", default=0.1, type=float,
                        help="per-epoch ramp: lamda = min(lamda_*, cof * epoch)")

    # --- this port: modality, grid, five-crop bookkeeping ---
    parser.add_argument("--modality", default="av", choices=list(MODALITIES),
                        help="'av' is MACIL-SD proper (AVCE_Model with the "
                             "self-distilled partner); 'audio' and 'visual' "
                             "train upstream's own Single_Model on one "
                             "modality alone")
    parser.add_argument("--grid", default="snippet", choices=list(GRIDS),
                        help="'snippet' trains on the 0.666667 s I3D grid with "
                             "VGGish resampled onto it and maps scores back to "
                             "the 1 fps gold grid; 'second' trains on the 1 fps "
                             "grid with I3D pooled onto it and VGGish native. "
                             "macilsd/align.py argues for the default")
    parser.add_argument("--crop-repeat", default=5, type=int,
                        help="crop slots visited per video per epoch; 5 is "
                             "upstream's five-crop training list")
    parser.add_argument("--fix-rep-swap", action="store_true",
                        help="pair each modality's representation with its own "
                             "logits in the contrastive loss. Upstream does "
                             "not; off by default, see PATCHES.md patch M13")

    # --- model selection (this port; see hate_common.data.split_train_val) ---
    parser.add_argument("--val-frac", default=0.1, type=float,
                        help="stratified holdout carved from the train split; "
                             "0 reproduces upstream's lack of a clean holdout")
    parser.add_argument("--select", default="val", choices=["val", "last"],
                        help="'val' keeps the best video-level val AP epoch, "
                             "'last' keeps the final epoch")

    # --- io ---
    parser.add_argument("--out-dir", default=None,
                        help="default results/reproduction/baselines/"
                             "<method>/<corpus>, method per --modality")
    return parser


def resolve(args):
    args.device = runtime.resolve_device(args.device)
    args.method = METHOD_NAME[args.modality]
    if args.out_dir is None:
        args.out_dir = os.path.dirname(
            runtime.scores_out_path(args.method, args.corpus))
    return args
