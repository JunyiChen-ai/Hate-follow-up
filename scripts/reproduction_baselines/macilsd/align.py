"""Temporal alignment between the I3D snippet grid and the VGGish second grid.

This module is new; it has no upstream counterpart. It exists because
MACIL-SD's one structural assumption about its inputs does not hold for this
study's features out of the box, and the resolution has to be written down
rather than buried in a dataloader.

The assumption
--------------
`Att_MMIL.forward` does

    x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2)

so the audio and visual sequences must have **the same number of rows**, row
for row, and row `t` of one must describe the same instant as row `t` of the
other. On XD-Violence that is free: the released RGB and VGGish arrays are
distributed pre-paired at one row each per 16-frame snippet, and upstream's
dataloader simply loads both and never checks.

What this study has
-------------------
Two grids, both honest, neither equal to the other:

    visual  results/reproduction/features/i3d_rgb_5crop/<corpus>/<id>.npy
            (n_snippets, 5, 1024) float32. Snippet j covers
            [j*16/24, (j+1)*16/24) s = 0.666667 s, recorded exactly in
            <id>.times.json. Frames past the last whole snippet are dropped by
            the extractor.

    audio   results/reproduction/features/vggish_1s/<corpus>/<id>.npy
            (T, 128) float32. Row i is second i, i.e. [i, i+1).

The second grid is the gold grid: `results/reproduction/gt/<corpus>_test.npz`
arrays have length exactly T for all 214 + 158 + 153 gold videos, checked in
smoke_cpu_macilsd.py. The two grids also do not cover the same span. Because
the extractor drops the tail frames that do not fill a whole snippet, audio
outlives visual in 1042 / 790 / 808 of the 1066 / 792 / 814 videos, by at most
5.33 s (hatemm `non_hate_video_149`), 1.67 s and 2.00 s. Visual outlives audio
in 8 / 0 / 3 videos, by at most 2.67 s.

The decision
------------
**Train on the I3D snippet grid; resample VGGish onto it; map the snippet
scores back onto the second grid at scoring time.** Not the other way round.

Three reasons, in order of weight.

1. *The snippet grid is upstream's grid, physically.* This study's I3D
   features were extracted at 24 fps with 16-frame snippets -- the same decode
   rate and the same snippet length XD-Violence used. One row is 0.666667 s
   here and 0.666667 s there. Every hyperparameter MACIL-SD counts in rows
   therefore keeps the physical meaning it was tuned with, and none has to be
   re-read: `--max-seqlen 200` is 133.3 s in both places, and the MIL top-k
   `int(seq_len // 16 + 1)` reads the same ~6 % of the same physical window.
   Training on the 1 s grid would silently rescale both.

2. *Resampling should degrade the coarser signal, not the finer one.* Pushing
   I3D down to 1 s throws away a third of the visual temporal resolution the
   extraction run paid for. Lifting VGGish up to 0.667 s invents no
   information but loses none either: each snippet window falls inside one or
   two second-long VGGish windows, and the resampled row is the time-average
   of the piecewise-constant VGGish signal over that window.

3. *The back-map is exact where it matters and explicit where it is not.*
   Scores live on the second grid at evaluation, and mapping 0.667 s scores up
   to 1 s is a lookup, not an average: gold second `i` takes the score of the
   snippet containing its midpoint `i + 0.5`. Seconds past the end of the
   visual coverage -- the dropped-tail seconds above -- hold the last snippet's
   score. Both rules are stated here and asserted in the smoke test, so the
   one lossy step in the chain is a named two-line rule rather than an
   interpolation nobody can reconstruct.

`--grid second` runs the mirror image: visual mean-pooled onto the 1 s grid,
audio native, scores already on the gold grid with no back-map at all. It is
not the default, but it is what the audio-only ablation should be read against
if anyone suspects the audio resampling of doing work, since on that grid the
VGGish rows are consumed exactly as they were written.

Everything here operates on time intervals read from `.times.json`; no
constant 0.666667 is hard-coded in the resampling itself.
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VISUAL_ROOT = os.path.join(REPO_ROOT, "results", "reproduction",
                           "features", "i3d_rgb_5crop")
AUDIO_ROOT = os.path.join(REPO_ROOT, "results", "reproduction",
                          "features", "vggish_1s")

V_DIM = 1024
A_DIM = 128
N_CROPS = 5
GRIDS = ("snippet", "second")


# --------------------------------------------------------------- file access
def visual_path(corpus, video_id):
    return os.path.join(VISUAL_ROOT, corpus, "%s.npy" % video_id)


def times_path(corpus, video_id):
    return os.path.join(VISUAL_ROOT, corpus, "%s.times.json" % video_id)


def audio_path(corpus, video_id):
    return os.path.join(AUDIO_ROOT, corpus, "%s.npy" % video_id)


def has_features(corpus, video_id):
    return (os.path.exists(visual_path(corpus, video_id))
            and os.path.exists(times_path(corpus, video_id))
            and os.path.exists(audio_path(corpus, video_id)))


def load_visual(corpus, video_id):
    """(n_snippets, 5, 1024) float32, as written by the extraction run."""
    feat = np.load(visual_path(corpus, video_id)).astype(np.float32)
    if feat.ndim != 3 or feat.shape[1] != N_CROPS or feat.shape[2] != V_DIM:
        raise ValueError("%s/%s: expected (T, %d, %d), got %s"
                         % (corpus, video_id, N_CROPS, V_DIM, feat.shape))
    return feat


def load_visual_crop(corpus, video_id, crop):
    """(n_snippets, 1024) float32 for one spatial crop.

    Memory-mapped and sliced, so a training item reads a fifth of the bytes of
    the full five-crop array. `times.json` records the crop order as
    top_left, top_right, bottom_left, bottom_right, centre; upstream indexes
    crops by the `__0` .. `__4` suffix of five separate files and never names
    them, so the correspondence is positional either way.
    """
    if not 0 <= crop < N_CROPS:
        raise ValueError("crop must be in [0, %d), got %r" % (N_CROPS, crop))
    feat = np.load(visual_path(corpus, video_id), mmap_mode="r")
    if feat.ndim != 3 or feat.shape[1] != N_CROPS or feat.shape[2] != V_DIM:
        raise ValueError("%s/%s: expected (T, %d, %d), got %s"
                         % (corpus, video_id, N_CROPS, V_DIM, feat.shape))
    return np.ascontiguousarray(feat[:, crop, :], dtype=np.float32)


def load_audio(corpus, video_id):
    """(T, 128) float32, row i = second i."""
    feat = np.load(audio_path(corpus, video_id)).astype(np.float32)
    if feat.ndim != 2 or feat.shape[1] != A_DIM:
        raise ValueError("%s/%s: expected (T, %d), got %s"
                         % (corpus, video_id, A_DIM, feat.shape))
    return feat


def snippet_bounds(corpus, video_id, n_snippets=None):
    """(n_snippets, 2) float64 of [start, end) seconds, from .times.json."""
    with open(times_path(corpus, video_id)) as fh:
        meta = json.load(fh)
    bounds = np.asarray(meta["times"], dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("%s/%s: malformed times.json" % (corpus, video_id))
    if n_snippets is not None and bounds.shape[0] != n_snippets:
        raise ValueError("%s/%s: %d snippets in the feature but %d in "
                         "times.json" % (corpus, video_id, n_snippets,
                                         bounds.shape[0]))
    return bounds


def second_bounds(n_seconds):
    """(n_seconds, 2) of [i, i+1), the grid the VGGish rows and the gold sit on."""
    i = np.arange(n_seconds, dtype=np.float64)
    return np.stack([i, i + 1.0], axis=1)


# ----------------------------------------------------------------- resampling
def resample_intervals(src, src_bounds, dst_bounds):
    """Time-average a piecewise-constant signal onto a new interval grid.

    `src` is (n_src, ...) with row k constant over `src_bounds[k]`. The output
    row for `dst_bounds[j]` is the overlap-length-weighted mean of every source
    row that intersects it, which is exactly the average value of the
    piecewise-constant signal over the destination interval.

    A destination interval that intersects nothing -- it lies past the end of
    the source coverage, or before its start -- takes the nearest source row
    whole. That is the hold-last-row rule the module docstring names; for this
    study it fires on the dropped-tail seconds and nowhere else.

    Both grids here are regular and short (a destination interval meets at most
    two source rows either way round), so the straightforward loop is fast
    enough and stays readable. The weights are exact, not nearest-neighbour, so
    a snippet straddling a second boundary gets both seconds in proportion.
    """
    src = np.asarray(src)
    src_bounds = np.asarray(src_bounds, dtype=np.float64)
    dst_bounds = np.asarray(dst_bounds, dtype=np.float64)
    n_src = src_bounds.shape[0]
    n_dst = dst_bounds.shape[0]
    if src.shape[0] != n_src:
        raise ValueError("src has %d rows but %d bounds"
                         % (src.shape[0], n_src))
    if n_src == 0:
        raise ValueError("empty source grid")

    starts = src_bounds[:, 0]
    ends = src_bounds[:, 1]
    out = np.empty((n_dst,) + src.shape[1:], dtype=np.float32)

    # First source row whose end is strictly past the destination start, and
    # first source row whose start is at or past the destination end.
    lo_all = np.searchsorted(ends, dst_bounds[:, 0], side="right")
    hi_all = np.searchsorted(starts, dst_bounds[:, 1], side="left")

    for j in range(n_dst):
        a, b = dst_bounds[j]
        lo = int(lo_all[j])
        hi = int(hi_all[j])
        if hi <= lo:
            # No overlap at all: hold the nearest source row.
            out[j] = src[min(max(lo, 0), n_src - 1)]
            continue
        w = np.minimum(ends[lo:hi], b) - np.maximum(starts[lo:hi], a)
        np.clip(w, 0.0, None, out=w)
        total = w.sum()
        if total <= 0.0:
            out[j] = src[min(max(lo, 0), n_src - 1)]
            continue
        w = (w / total).astype(np.float32)
        block = src[lo:hi]
        out[j] = np.tensordot(w, block, axes=(0, 0))
    return out


def snippet_index_for_seconds(snip_bounds, n_seconds):
    """Gold second i -> index of the snippet covering its midpoint i + 0.5.

    Clamped at both ends, so seconds past the last snippet hold the last
    snippet's score. This is the only lossy step between the model's grid and
    the evaluation grid, and it is a lookup: no score is averaged or
    interpolated on the way out.
    """
    mid = np.arange(n_seconds, dtype=np.float64) + 0.5
    idx = np.searchsorted(np.asarray(snip_bounds)[:, 0], mid, side="right") - 1
    return np.clip(idx, 0, snip_bounds.shape[0] - 1).astype(np.int64)


# ------------------------------------------------------------- the pair, aligned
def aligned_pair(corpus, video_id, grid="snippet"):
    """(visual, audio, n_seconds, snip_bounds) with visual and audio the same length.

    grid="snippet"  visual native (n_snippets, 5, 1024);
                    audio resampled to (n_snippets, 128).
    grid="second"   visual resampled to (n_seconds, 5, 1024);
                    audio native (n_seconds, 128).

    `n_seconds` is the VGGish row count, which equals the gold array length for
    every gold video in all three corpora.
    """
    if grid not in GRIDS:
        raise ValueError("grid must be one of %s, got %r" % (GRIDS, grid))
    visual = load_visual(corpus, video_id)
    audio = load_audio(corpus, video_id)
    snip = snippet_bounds(corpus, video_id, visual.shape[0])
    n_seconds = audio.shape[0]

    if grid == "snippet":
        audio_g = resample_intervals(audio, second_bounds(n_seconds), snip)
        visual_g = visual
    else:
        visual_g = resample_intervals(visual, snip, second_bounds(n_seconds))
        audio_g = audio

    if visual_g.shape[0] != audio_g.shape[0]:
        raise AssertionError("%s/%s: %d visual rows vs %d audio rows on the "
                             "%s grid" % (corpus, video_id, visual_g.shape[0],
                                          audio_g.shape[0], grid))
    return (np.ascontiguousarray(visual_g), np.ascontiguousarray(audio_g),
            n_seconds, snip)


def aligned_audio(corpus, video_id, grid="snippet"):
    """Audio on the chosen grid, plus (n_seconds, snip_bounds).

    Split out of `aligned_pair` because the dataset precomputes this once per
    video -- it is the same array for all five crops -- while the visual side
    is read per crop.
    """
    audio = load_audio(corpus, video_id)
    n_seconds = audio.shape[0]
    snip = snippet_bounds(corpus, video_id)
    if grid == "snippet":
        audio = resample_intervals(audio, second_bounds(n_seconds), snip)
    elif grid != "second":
        raise ValueError("grid must be one of %s, got %r" % (GRIDS, grid))
    return np.ascontiguousarray(audio), n_seconds, snip


def aligned_visual_crop(corpus, video_id, crop, grid, n_seconds, snip_bounds_):
    """One crop's visual features on the chosen grid, (n_rows, 1024)."""
    visual = load_visual_crop(corpus, video_id, crop)
    if visual.shape[0] != snip_bounds_.shape[0]:
        raise ValueError("%s/%s: %d snippets in the feature but %d in "
                         "times.json" % (corpus, video_id, visual.shape[0],
                                         snip_bounds_.shape[0]))
    if grid == "second":
        visual = resample_intervals(visual, snip_bounds_,
                                    second_bounds(n_seconds))
    return np.ascontiguousarray(visual)


def scores_to_gold_grid(scores, snip_bounds, n_seconds, grid="snippet"):
    """Lift a length-`n_rows` model score vector onto the 1 fps gold grid.

    On the second grid this is the identity, and the length is asserted. On the
    snippet grid it is the midpoint lookup, with the tail held.

    This replaces upstream's `np.repeat(pred, 16)`, which lifted snippet scores
    onto XD-Violence's 24 fps *frame* grid. The target grid here is 1 fps, so
    the factor is neither 16 nor an integer, and a lookup is the honest form.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if grid == "second":
        if scores.shape[0] != n_seconds:
            raise ValueError("second grid: %d scores for %d gold frames"
                             % (scores.shape[0], n_seconds))
        return scores
    if scores.shape[0] != snip_bounds.shape[0]:
        raise ValueError("snippet grid: %d scores for %d snippets"
                         % (scores.shape[0], snip_bounds.shape[0]))
    return scores[snippet_index_for_seconds(snip_bounds, n_seconds)]
