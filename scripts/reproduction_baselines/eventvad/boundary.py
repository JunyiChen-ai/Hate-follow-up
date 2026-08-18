#!/usr/bin/env python
"""Statistical boundary detection, and the events it cuts the video into.

Ports upstream `src/event_seg/boundary_detection.py` and the boundary-to-frame
arithmetic at the top of `src/event_seg/video_processing.py`.

The divergence metric, Eq. (9), is one place where the released code and the
paper agree exactly:

    s_i = ||f_{i+1} - f_i||^2 + (1 - cos(f_i, f_{i+1}))

The smoothing chain is where they part company, and the disagreement decides
whether the detector fires at all. Eq. (11) defines the moving average as
**centred** on i, from i - floor(w/2) to i + floor(w/2), and Eq. (12) divides
the smoothed signal by it at the same index. The released code instead computes
`np.convolve(s_smoothed, ones(w)/w, mode='valid')` and pairs
`s_smoothed[w-1 + j]` with `ema[j]`, a window *trailing* the sample, then adds
`w // 2` back to the detected index.

Measured, on a 400-frame synthetic with regime changes planted at frames 149
and 269 (`selftest` reproduces it): the raw divergence peaks at exactly 149 and
269, so the signal is there. A Savitzky-Golay filter of width 60 spreads that
one-frame peak into a bump of width 60, and a **centred** 60-wide average then
covers the bump it is being compared against -- the ratio tops out at 1.194
against a threshold of 1.281 and nothing is detected. The trailing window
compares the bump against the quiet stretch preceding it, reaches 1.602 against
a threshold of 1.577, and fires. A centred normaliser cannot detect a change
whose width is its own window; Eq. (11) as printed is not what produced the
paper's numbers.

`ma_mode` therefore selects among three readings, defaulting to the one the
released code implements:

    "upstream" (default)  trailing window, index reported at j + w // 2.
    "trailing_aligned"    trailing window, index at j + w, which is the frame
                          the ratio at j actually describes. Upstream's w // 2
                          is a partial correction for the w - 1 shift the
                          `mode='valid'` convolution introduces and leaves
                          boundaries reported about w/2 frames -- a second at
                          30 fps -- early. This mode separates the trailing
                          comparison, which works, from the index arithmetic,
                          which is an artefact.
    "centered"            Eq. (11) as written, kept so the claim above stays
                          checkable.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def divergence(features):
    """Eq. (9), for consecutive rows. Returns a length n-1 array."""
    feats = np.asarray(features, dtype=np.float64)
    if feats.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    diffs = np.diff(feats, axis=0)
    magnitude = np.linalg.norm(diffs, axis=1) ** 2
    a, b = feats[:-1], feats[1:]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6
    cos = np.einsum("ij,ij->i", a, b) / denom
    return magnitude + (1.0 - cos)


def _centred_mean(x, window):
    """Eq. (11) with the window clipped at the ends.

    The paper's sum runs from i - floor(w/2) to i + floor(w/2) and divides by
    w, which is neither the number of terms it sums (2*floor(w/2)+1) nor
    defined at the ends of the signal. Dividing by the number of terms
    actually inside the signal is the reading that leaves the ratio r_i = 1
    for a constant signal at every index, which is what makes the MAD
    threshold in Eq. (14) mean the same thing at the edges as in the middle.
    """
    half = window // 2
    csum = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half + 1, n)
    return (csum[hi] - csum[lo]) / (hi - lo)


def detect_boundaries(features, fps, cfg):
    """Return (boundary_frame_indices, diagnostics).

    The indices are frame numbers on the decoded stream: a boundary at index b
    means frame b is the first frame of a new event.
    """
    n = len(features)
    s = divergence(features)
    window = cfg.savgol_window(fps)
    diag = {"n_frames": n, "window": window, "n_raw": 0, "n_merged": 0,
            "threshold": None, "median": None, "mad": None,
            "reason": None}

    if len(s) == 0:
        diag["reason"] = "fewer than two frames"
        return np.zeros(0, dtype=int), diag
    # Upstream's own guard, kept verbatim: too short to smooth, no boundaries.
    if len(s) < window * 2:
        diag["reason"] = ("signal %d shorter than 2x window %d"
                          % (len(s), window))
        return np.zeros(0, dtype=int), diag
    if window <= cfg.savgol_polyorder:
        diag["reason"] = ("window %d not above polyorder %d"
                          % (window, cfg.savgol_polyorder))
        return np.zeros(0, dtype=int), diag

    smoothed = savgol_filter(s, window_length=window,
                             polyorder=cfg.savgol_polyorder)

    if cfg.ma_mode in ("centred", "centered"):
        mu = _centred_mean(smoothed, window)
        ratio = smoothed / (mu + 1e-6)
        # Divergence index i is the transition between frames i and i+1, so
        # the new event starts at frame i+1.
        offset = 1
    elif cfg.ma_mode in ("upstream", "trailing_aligned"):
        mu = np.convolve(smoothed, np.ones(window) / window, mode="valid")
        ratio = smoothed[window - 1:] / (mu + 1e-6)
        # ratio index j is smoothed index j + w - 1, i.e. divergence index
        # j + w - 1, i.e. boundary frame j + w.
        offset = window // 2 if cfg.ma_mode == "upstream" else window
    else:
        raise ValueError("unknown ma_mode %r" % (cfg.ma_mode,))

    median = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - median)))          # Eq. (13)
    threshold = median + cfg.mad_multiplier * mad           # Eq. (14)
    raw = np.flatnonzero(ratio > threshold) + offset
    raw = raw[(raw > 0) & (raw < n)]

    merged = _merge(raw, cfg.min_segment_gap * fps)

    diag.update({"n_raw": int(len(raw)), "n_merged": int(len(merged)),
                 "threshold": threshold, "median": median, "mad": mad})
    return np.asarray(merged, dtype=int), diag


def _merge(boundaries, min_gap):
    """Upstream's merge: a run of boundaries closer than `min_gap` collapses
    to its **last** member. Reproduced exactly, including that choice."""
    merged = []
    prev = boundaries[0] if len(boundaries) else None
    for b in boundaries[1:]:
        if prev is not None and (b - prev) < min_gap:
            prev = b
        else:
            if prev is not None:
                merged.append(prev)
            prev = b
    if prev is not None:
        merged.append(prev)
    return merged


def events_from_boundaries(boundaries, n_frames, fps, cfg):
    """Cut [0, n_frames) into events, upstream's arithmetic then a gap close.

    Upstream turns merged boundaries into segments `(b_i, b_{i+1})` with a
    final `(b_last, b_last + min_segment_gap * fps)`, drops any segment
    shorter than two frames, prepends `(0, first_start)` when the first
    segment does not start at zero and appends `(last_end, n-1)` when the last
    does not reach the end. Two departures, both recorded in the returned
    diagnostics:

    * the drop rule can in principle leave a hole in the middle, and a hole is
      a stretch of video no event scores. Any hole is closed by extending the
      previous event, and `n_gaps_closed` counts it. Merging already forces
      boundaries `min_segment_gap * fps` apart, so this should never fire;
      it is a guarantee, not a correction.
    * upstream's ends are inclusive (`total_frames - 1`) and its writer copies
      frames `start..end` inclusive. Half-open `[start, end)` is used here so
      the events form an exact partition of `range(n_frames)`, which is what
      the 1 fps rasteriser needs.
    """
    diag = {"n_gaps_closed": 0, "n_dropped": 0}
    if n_frames <= 0:
        return [], diag
    bounds = [int(b) for b in boundaries if 0 < int(b) < n_frames]
    if not bounds:
        return [(0, n_frames)], diag

    spans = []
    for i, b in enumerate(bounds):
        if i + 1 < len(bounds):
            end = bounds[i + 1]
        else:
            end = int(round(b + cfg.min_segment_gap * fps))
        end = min(end, n_frames)
        if end - b > 1:
            spans.append((b, end))
        else:
            diag["n_dropped"] += 1

    if not spans:
        return [(0, n_frames)], diag

    if spans[0][0] > 0:
        spans.insert(0, (0, spans[0][0]))
    if spans[-1][1] < n_frames:
        spans.append((spans[-1][1], n_frames))

    closed = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = closed[-1]
        if start > prev_end:
            closed[-1] = (prev_start, start)
            diag["n_gaps_closed"] += 1
        closed.append((start, end))

    if closed[0][0] != 0 or closed[-1][1] != n_frames:
        raise AssertionError("events do not cover [0, %d): %s"
                             % (n_frames, closed[:2] + closed[-2:]))
    for (_, a_end), (b_start, _) in zip(closed, closed[1:]):
        if a_end != b_start:
            raise AssertionError("events are not contiguous at %d/%d"
                                 % (a_end, b_start))
    return closed, diag
