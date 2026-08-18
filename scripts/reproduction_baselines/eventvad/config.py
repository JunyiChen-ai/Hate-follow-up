#!/usr/bin/env python
"""EventVAD hyperparameters, with the paper as the authority.

Upstream ships `src/event_seg/config.py`. Two of its values contradict the
paper, and the released pipeline cannot run at all (`graph_propagation` is
imported but defined nowhere -- see DESIGN_EVENTVAD.md, gap G1), so the
released config was never executed and cannot be treated as a tested preset.
The paper's published values are therefore the defaults here and the config
literals are reachable through `--preset upstream`.

    quantity                paper            config.py        default here
    alpha  (semantic/motion) 0.75            clip_weight 0.8   0.75
    gamma  (time decay)      0.6             time_decay 0.05   0.6
    savgol window w          60 @ 30 fps     ema_window 2.0 s  2.0 s
    MAD multiplier k         3               3.0               3.0
    GAT iterations           1               gat_iters 1       1
    projection dim k         64              ortho_dim 64      64
    fused dim d              640             feature_dim 640   640
    RAFT iterations          --              raft_iters 20     20
    decode FPS               30              native            min(native, 30)

`ema_window` and `min_segment_gap` are already written in seconds upstream and
multiplied by fps at the point of use, so they carry over untouched. `gamma`
is the one that does not: it multiplies a frame-index difference, so its
physical meaning depends on the decode rate. See `gamma_per_frame`.
"""

from __future__ import annotations

import dataclasses

#: The rate the paper's gamma was tuned at ("with FPS = 30", section 4.1).
REFERENCE_FPS = 30.0


@dataclasses.dataclass
class EventVADConfig:
    # ---------------------------------------------------------- features
    #: Longest side / shortest side cap, upstream `max_resolution`.
    max_width: int = 1280
    max_height: int = 720
    #: Frames pushed through CLIP in one batch, upstream `chunk_size`.
    chunk_size: int = 500
    #: RAFT GRU iterations, upstream `raft_iters`.
    raft_iters: int = 20
    #: Decode rate cap. The paper fixes FPS = 30; upstream reads every frame.
    #: A video slower than this keeps its own rate -- upsampling would insert
    #: duplicate frames, and a duplicate frame has exactly zero optical flow,
    #: which is a boundary signal the video does not contain.
    max_fps: float = 30.0

    # ------------------------------------------------------ dynamic graph
    #: Paper's alpha, the semantic-motion fusion coefficient (Eq. 3, Eq. 4).
    alpha: float = 0.75
    #: Paper's gamma, the temporal decay factor (Eq. 4), in units of
    #: 1 / frame at REFERENCE_FPS.
    gamma: float = 0.6
    #: "per_second" rescales gamma so the decay per *second* is the one the
    #: paper tuned; "per_frame" uses gamma literally against the frame index,
    #: which is what upstream's code does.
    gamma_mode: str = "per_second"
    #: Eq. (1) L2-normalises CLIP before Eq. (3) fuses it, so the node feature
    #: Eq. (9) differences is built from unit vectors. Upstream normalises only
    #: inside the similarity and leaves raw CLIP magnitudes in the node.
    clip_norm_in_nodes: bool = True
    #: kNN fan-out before the linear decay upstream applies over blocks.
    init_k: int = 5
    #: Block side for the O(n^2) similarity sweep, upstream `graph_block_size`.
    graph_block_size: int = 200

    # -------------------------------------------------- graph propagation
    #: Paper's k, the orthogonal projection dimension (Eq. 5).
    ortho_dim: int = 64
    #: Paper: "the graph attention propagation is only a single iteration".
    gat_iters: int = 1
    #: Reconstruction choice, see DESIGN_EVENTVAD.md G1-c.
    gat_edge_term: str = "weight"
    #: Seed for the fixed orthogonal Q/K/V. The paper fixes none; upstream
    #: seeds its flow projection with 42, so 42 is used here too.
    ortho_seed: int = 42

    # -------------------------------------------------- boundary detection
    #: Savitzky-Golay window in seconds; 2.0 s x 30 fps = the paper's w = 60.
    ema_window: float = 2.0
    #: Savitzky-Golay polynomial order.
    savgol_polyorder: int = 2
    #: Paper's k in M = median(r) + k x MAD(r) (Eq. 14).
    mad_multiplier: float = 3.0
    #: Minimum spacing between accepted boundaries, seconds.
    min_segment_gap: float = 2.0
    #: "upstream" is the released trailing-window arithmetic and its index
    #: offset; "trailing_aligned" keeps the trailing window and fixes the
    #: index; "centered" is Eq. (11) as written, which measurably fails to
    #: detect the changes it is meant to. See boundary.py and DESIGN G7.
    ma_mode: str = "upstream"

    # ------------------------------------------------------------ scoring
    #: Frames VideoLLaMA2.1-7B-16F consumes per event.
    frames_per_event: int = 16

    # ---------------------------------------------------------------- api
    def gamma_per_frame(self, fps: float) -> float:
        """The gamma to multiply a frame-index difference by, at `fps`.

        The paper's penalty is `1 + gamma * |i - j|` with gamma = 0.6 at
        30 fps, i.e. a decay of 18 per second of separation. Reproducing that
        decay at another rate needs `gamma * REFERENCE_FPS / fps`.
        """
        if self.gamma_mode == "per_frame":
            return self.gamma
        if self.gamma_mode == "per_second":
            if fps <= 0:
                raise ValueError("fps must be positive, got %r" % (fps,))
            return self.gamma * REFERENCE_FPS / fps
        raise ValueError("unknown gamma_mode %r" % (self.gamma_mode,))

    def savgol_window(self, fps: float) -> int:
        """Upstream's `max(1, int(fps * ema_window))`."""
        return max(1, int(fps * self.ema_window))

    def output_size(self, width: int, height: int):
        """Upstream's resize rule, verbatim: cap, then force even sides."""
        ratio = min(self.max_width / width, self.max_height / height)
        if ratio < 1:
            size = (int(width * ratio), int(height * ratio))
        else:
            size = (width, height)
        return (size[0] // 2 * 2, size[1] // 2 * 2)

    def as_dict(self):
        return dataclasses.asdict(self)


#: The literals in upstream `src/event_seg/config.py`, for `--preset upstream`.
UPSTREAM_OVERRIDES = {
    "alpha": 0.8,          # config.clip_weight
    "gamma": 0.05,         # config.time_decay
    "gamma_mode": "per_frame",
    "ma_mode": "upstream",
    "clip_norm_in_nodes": False,
}


def build_config(preset="paper", **overrides):
    cfg = EventVADConfig()
    if preset == "upstream":
        for key, value in UPSTREAM_OVERRIDES.items():
            setattr(cfg, key, value)
    elif preset != "paper":
        raise ValueError("unknown preset %r (expected paper or upstream)"
                         % (preset,))
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise ValueError("unknown config field %r" % (key,))
        setattr(cfg, key, value)
    return cfg


def add_config_args(ap):
    """Attach the config overrides every stage shares."""
    ap.add_argument("--preset", default="paper",
                    choices=("paper", "upstream"),
                    help="paper: the published alpha/gamma and Eq. (11) as "
                         "written. upstream: the literals in the released "
                         "config.py and its trailing-window moving average.")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gamma-mode", default=None,
                    choices=("per_second", "per_frame"))
    ap.add_argument("--ma-mode", default=None,
                    choices=("upstream", "trailing_aligned", "centered"))
    ap.add_argument("--gat-iters", type=int, default=None)
    ap.add_argument("--gat-edge-term", default=None,
                    choices=("weight", "indicator"))
    ap.add_argument("--mad-multiplier", type=float, default=None)
    ap.add_argument("--max-fps", type=float, default=None)
    ap.add_argument("--raft-iters", type=int, default=None)
    return ap


def config_from_args(args):
    return build_config(
        preset=args.preset,
        alpha=getattr(args, "alpha", None),
        gamma=getattr(args, "gamma", None),
        gamma_mode=getattr(args, "gamma_mode", None),
        ma_mode=getattr(args, "ma_mode", None),
        gat_iters=getattr(args, "gat_iters", None),
        gat_edge_term=getattr(args, "gat_edge_term", None),
        mad_multiplier=getattr(args, "mad_multiplier", None),
        max_fps=getattr(args, "max_fps", None),
        raft_iters=getattr(args, "raft_iters", None),
    )
