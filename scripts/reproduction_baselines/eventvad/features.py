#!/usr/bin/env python
"""CLIP and RAFT frame features, streamed.

Ports upstream `src/event_seg/feature_extractor.py`. Three changes, all
recorded in PATCHES.md.

**LAVIS is dropped (patch E2).** Upstream calls
`lavis.models.load_model_and_preprocess(name="clip", model_type="ViT-B-16")`.
LAVIS's `clip_vit_base16.yaml` resolves that to `pretrained: openai`, i.e. the
OpenAI CLIP ViT-B/16 checkpoint this repository already caches at
`~/.cache/clip/ViT-B-16.pt` (sha256 `5806e77c...`), and its `clip_image_eval`
processor at image_size 224 is

    Resize(224, BICUBIC) -> CenterCrop(224) -> convert("RGB")
        -> ToTensor() -> Normalize((0.48145466, 0.4578275, 0.40821073),
                                   (0.26862954, 0.26130258, 0.27577711))

step for step the transform `hate_common/clip/clip.py:_transform(224)` builds.
The vendored CLIP this study already uses for VadCLIP and DSANet therefore
produces the same 512-d embeddings from the same weights through the same
preprocessing, and LAVIS -- which pins an old transformers and pulls in spacy,
open3d and a jupyter stack -- buys nothing. `smoke_cpu_eventvad.py` asserts the
transform equality rather than leaving it as a claim in a comment.

**The RAFT checkpoint path is resolved (patch E3).** Upstream hard-codes
`model='/path/raft-things.pth'`, a placeholder. The file is the `raft-things`
entry of the `models.zip` that `princeton-vl/RAFT`'s own `download_models.sh`
fetches; it lives at `/home/jehc223/data/checkpoints/raft/raft-things.pth`,
sha256 `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`.

**Extraction streams (patch E4).** Upstream runs CLIP over every frame, then
RAFT over every adjacent pair, holding the entire decoded video in RAM in
between. CLIP is a per-frame function and RAFT a per-adjacent-pair function,
so interleaving them in one pass produces the same two feature arrays while
holding one chunk of frames instead of all of them. See `video_io` for why
that matters here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASELINES))
if BASELINES not in sys.path:
    sys.path.insert(0, BASELINES)

RAFT_CORE = os.path.join(PROJECT_ROOT, "third_party", "RAFT", "core")
DEFAULT_RAFT_CKPT = "/home/jehc223/data/checkpoints/raft/raft-things.pth"
RAFT_THINGS_SHA256 = \
    "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"
CLIP_CACHE = os.path.expanduser("~/.cache/clip")
CLIP_NAME = "ViT-B/16"

CLIP_DIM = 512
FLOW_DIM = 128


def init_random_ortho(in_dim, out_dim, seed=42):
    """Upstream `FeatureExtractor._init_random_ortho`, verbatim.

    The rows of an orthogonal matrix are orthonormal, so `P` has orthonormal
    rows and `u -> u @ P` is an isometry of R^2 into R^128: the flow branch
    carries exactly the two degrees of freedom of the spatially averaged flow
    vector, lifted without distortion. That is worth knowing before reading
    the paper's ablation, where RAFT alone is worth +1.42 AUC.
    """
    np.random.seed(seed)
    side = max(in_dim, out_dim)
    q_mat, _ = np.linalg.qr(np.random.randn(side, side))
    return q_mat[:in_dim, :out_dim].astype(np.float32)


class FeatureExtractor:
    """CLIP ViT-B/16 semantics and RAFT mean-flow motion, per frame."""

    def __init__(self, device="cuda", raft_ckpt=DEFAULT_RAFT_CKPT,
                 clip_cache=CLIP_CACHE, raft_iters=20, chunk_size=500):
        self.device = torch.device(device)
        self.raft_iters = raft_iters
        self.chunk_size = chunk_size

        from hate_common import clip as clip_pkg
        model, preprocess = clip_pkg.load(
            CLIP_NAME, device=self.device, jit=False,
            download_root=clip_cache)
        # Upstream: `model_info[0].float()`. clip.load leaves the weights fp16
        # on a CUDA device; Config.fp16_enabled is False.
        self.clip_model = model.float().eval()
        self.preprocess = preprocess

        if RAFT_CORE not in sys.path:
            sys.path.insert(0, RAFT_CORE)
        from argparse import Namespace
        from raft import RAFT
        args = Namespace(small=False, mixed_precision=False,
                         alternate_corr=False, dropout=0.0)
        raft = RAFT(args)
        state = torch.load(raft_ckpt, map_location="cpu", weights_only=True)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        raft.load_state_dict(state, strict=True)
        self.raft_model = raft.to(self.device).eval().float()

        self.flow_proj = init_random_ortho(2, FLOW_DIM)

    # ------------------------------------------------------------- pieces
    @torch.no_grad()
    def _clip_chunk(self, frames):
        from PIL import Image
        batch = torch.stack([self.preprocess(Image.fromarray(f))
                             for f in frames]).to(self.device).float()
        return self.clip_model.encode_image(batch).cpu().numpy()

    @torch.no_grad()
    def _mean_flow(self, prev, curr):
        """Upstream: RAFT's final flow, averaged over H and W. Plus the pad
        upstream omits and RAFT does not survive without (patch E10).

        `RAFT.forward` sizes its coordinate grid as `H // 8, W // 8` but its
        encoder produces `ceil(H / 8), ceil(W / 8)`, so on any side that is not
        a multiple of 8 the two disagree and `bilinear_sampler` raises:

            RuntimeError: grid_sampler(): expected grid and input to have same
            batch size, but got input with sizes [6420, 1, 60, 107] and grid
            with sizes [6360, 9, 9, 2]

        which is 854x480 -- a HateMM video, and the most common resolution in
        that corpus. Upstream calls `self.raft_model(prev_frame, curr_frame)`
        with neither pad nor resize, so it raises on those files. RAFT's own
        `demo.py` wraps every call in `InputPadder`, which replicate-pads up to
        the next multiple of 8 and crops the flow back afterwards; that is done
        here, so the averaged field covers exactly the original frame. This is
        not a choice between readings -- without it the stage cannot run.
        """
        from utils.utils import InputPadder
        padder = InputPadder(prev.shape)
        p, c = padder.pad(prev, curr)
        flow = self.raft_model(p, c, iters=self.raft_iters)[-1]
        flow = padder.unpad(flow)
        return torch.mean(flow, dim=[2, 3]).cpu().numpy().reshape(-1)

    def _to_tensor(self, frame):
        return (torch.from_numpy(np.ascontiguousarray(frame))
                .permute(2, 0, 1).unsqueeze(0).to(self.device).float())

    # -------------------------------------------------------------- main
    def extract(self, frame_iter):
        """Consume an RGB uint8 frame iterator, return (clip, flow) arrays.

        `clip` is (n, 512) float32 post-projection embeddings, unnormalised --
        Eq. (1)'s normalisation happens in `graph.py`, where upstream also
        applies it. `flow` is (n, 128) float32 with row 0 all zeros, as
        upstream sets it: there is no frame before the first.
        """
        clip_out, flow_out = [], []
        buf = []
        prev = None
        for frame in frame_iter:
            buf.append(frame)
            curr = self._to_tensor(frame)
            if prev is None:
                flow_out.append(np.zeros(2, dtype=np.float32))
            else:
                flow_out.append(self._mean_flow(prev, curr))
            prev = curr
            if len(buf) >= self.chunk_size:
                clip_out.append(self._clip_chunk(buf))
                buf = []
        if buf:
            clip_out.append(self._clip_chunk(buf))
        if not clip_out:
            raise RuntimeError("no frames reached the feature extractor")

        clip = np.concatenate(clip_out).astype(np.float32)
        flow_raw = np.stack(flow_out).astype(np.float32)
        if clip.shape[0] != flow_raw.shape[0]:
            raise AssertionError("clip %d rows vs flow %d rows"
                                 % (clip.shape[0], flow_raw.shape[0]))
        flow = (flow_raw @ self.flow_proj).astype(np.float32)
        return clip, flow


def check_raft_checkpoint(path=DEFAULT_RAFT_CKPT):
    """Return (exists, sha256_matches, sha)."""
    import hashlib
    if not os.path.isfile(path):
        return False, False, None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    sha = h.hexdigest()
    return True, sha == RAFT_THINGS_SHA256, sha
