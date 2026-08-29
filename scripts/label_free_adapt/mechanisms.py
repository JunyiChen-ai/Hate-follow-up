#!/usr/bin/env python3
"""Reusable training-free mechanisms for the first adaptation wave."""
from __future__ import annotations

import json
import math
import re
from statistics import fmean, pstdev
from typing import Iterable, Sequence

import numpy as np

from .schema import Interval, normalize_intervals

GENERIC_QUERY = (
    "Locate every interval containing an attack, dehumanization, threat, exclusion, "
    "or hateful stereotype directed at a person or group. Distinguish endorsement "
    "from quotation, reporting, counterspeech, satire, and condemnation."
)

QUERY_FACETS = (
    "a person or social group is explicitly or implicitly identified",
    "the identified target is attacked, dehumanized, threatened, excluded, or stereotyped",
    "the harmful position is endorsed rather than merely quoted, reported, or condemned",
    "visual, spoken, written, and acoustic context jointly establish the event",
)


def numbered_timeline_prompt(frame_ids: Sequence[int], duration: float,
                             query: str = GENERIC_QUERY, transcript: str = "") -> str:
    ids = ", ".join(str(i) for i in frame_ids)
    return (
        f"Frames are displayed in temporal order and visibly numbered [{ids}]. "
        f"The video duration is {duration:.3f} seconds. The available whole-video "
        f"speech transcript is: {json.dumps(transcript[:8000], ensure_ascii=False)}. "
        "Raw audio is unavailable in this call; do not invent acoustic evidence. "
        f"{query} "
        "Return strict JSON: {\"intervals\":[{\"start_frame\":int,"
        "\"end_frame\":int,\"confidence\":number}],\"evidence\":{"
        "\"visual\":[],\"speech\":[],\"text\":[],\"audio\":[]}}. "
        "Use an empty interval list when evidence is insufficient."
    )


def dense_evidence_prompt(frame_ids: Sequence[int], duration: float,
                          transcript: str = "", nbins: int = 16,
                          timed_bins: Sequence[str] | None = None) -> str:
    """Non-abstaining candidate prompt; produces a ranking even under uncertainty."""
    ids = ", ".join(str(i) for i in frame_ids)
    return (
        f"Analyze every numbered frame [{ids}] from a {duration:.3f}s video. "
        + (f"Timestamp-aligned speech per bin: "
           f"{json.dumps(list(timed_bins), ensure_ascii=False)}. "
           if timed_bins is not None else
           f"Whole-video transcript (not timestamp aligned): "
           f"{json.dumps(transcript[:8000], ensure_ascii=False)}. ") +
        f"Divide the ordered frames into {nbins} consecutive equal-sized time bins. "
        "For EACH BIN, estimate evidence on 0-4 ordinal scales: target (a person/group "
        "is referenced), harm (attack, dehumanization, threat, exclusion, or stereotype), "
        "endorsement (speaker/video endorses rather than quotes, reports, condemns, or jokes), "
        "and visual_text (visible imagery or on-screen text supports the event). The transcript "
        "may establish whole-video semantic context but MUST NOT by itself assign a time to a "
        "frame. Do not abstain and do not omit frames: uncertainty should produce middle or low "
        "scores, not missing entries. Return only strict compact JSON containing four integer "
        f"arrays, each of EXACTLY {nbins} entries: "
        '{"target":[...],"harm":[...],"endorsement":[...],"visual_text":[...]}. '
        "Use the full 0-4 scale and rank the strongest candidate bins above neutral bins."
    )


def parse_dense_scores(text: str, nframes: int) -> list[float]:
    obj = parse_json_object(text)
    if all(name in obj for name in ("target", "harm", "endorsement", "visual_text")):
        raw_arrays = [[float(x) for x in obj[name]]
                  for name in ("target", "harm", "endorsement", "visual_text")]
        if any(len(x) < 2 for x in raw_arrays):
            raise ValueError("dense arrays have fewer than two entries")
        if any(not 0 <= value <= 4 for array in raw_arrays for value in array):
            raise ValueError("dense score outside 0-4 range")
        arrays = [np.interp(np.linspace(0, len(x) - 1, nframes),
                            np.arange(len(x)), np.asarray(x) / 4.0).tolist()
                  for x in raw_arrays]
        return [min(arrays[0][i], arrays[1][i]) *
                (0.5 + 0.25 * arrays[2][i] + 0.25 * arrays[3][i])
                for i in range(nframes)]
    values: dict[int, float] = {}
    for row in obj.get("frames", []):
        if not isinstance(row, dict):
            continue
        try:
            frame_id = int(row["frame"])
            components = [float(row[name]) / 4.0
                          for name in ("target", "harm", "endorsement", "visual_text")]
            if not 0 <= frame_id < nframes or any(not 0 <= x <= 1 for x in components):
                continue
            # Harm without a target is not hate; endorsement and visual evidence
            # modulate rather than veto the semantic conjunction.
            semantic = min(components[0], components[1])
            score = semantic * (0.5 + 0.25 * components[2] + 0.25 * components[3])
            values[frame_id] = max(0.0, min(1.0, score))
        except (KeyError, TypeError, ValueError):
            continue
    if len(values) < max(2, nframes // 2):
        raise ValueError(f"only {len(values)}/{nframes} frame scores parsed")
    # Missing individual entries are linearly interpolated, never silently zeroed.
    known = sorted(values)
    return [float(np.interp(i, known, [values[j] for j in known])) for i in range(nframes)]


def self_correction_prompt(initial: Sequence[Interval], duration: float,
                           evidence: dict) -> str:
    spans = [x.as_list() for x in initial]
    return (
        "Audit the proposed harmful-event intervals against evidence immediately "
        "inside and outside each boundary. Check missed disjoint events, quoted or "
        "condemned speech, and unsupported modality assumptions. "
        f"duration={duration:.3f}; initial={json.dumps(spans)}; "
        f"evidence={json.dumps(evidence, ensure_ascii=False)}. "
        "Return strict JSON with corrected `intervals` and a `changes` list."
    )


def parse_json_object(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    loose = re.search(r"\{.*\}", text, re.S)
    if loose:
        candidates.append(loose.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (TypeError, json.JSONDecodeError):
            pass
    raise ValueError("no parseable JSON object")


def parse_numbered_intervals(text: str, frame_times: Sequence[float],
                             duration: float) -> tuple[list[Interval], dict]:
    obj = parse_json_object(text)
    values = []
    for row in obj.get("intervals", []):
        if not isinstance(row, dict):
            continue
        try:
            i0, i1 = int(row["start_frame"]), int(row["end_frame"])
            if not (0 <= i0 < len(frame_times) and 0 <= i1 < len(frame_times)):
                continue
            a, b = frame_times[min(i0, i1)], frame_times[max(i0, i1)]
            step = (frame_times[1] - frame_times[0]) if len(frame_times) > 1 else duration
            values.append([a, min(duration, b + step), float(row.get("confidence", 1.0))])
        except (KeyError, TypeError, ValueError):
            continue
    return normalize_intervals(values, duration), obj.get("evidence", {})


def consistency_curve(curves: Sequence[Sequence[float]], penalty: float = 0.5) -> list[float]:
    """Mean evidence minus perturbation uncertainty; no target labels involved."""
    if not curves:
        raise ValueError("at least one curve is required")
    n = len(curves[0])
    if n == 0 or any(len(x) != n for x in curves):
        raise ValueError("all curves must have the same non-zero length")
    out = []
    for values in zip(*curves):
        vals = [float(x) for x in values]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("curves contain non-finite values")
        out.append(max(0.0, min(1.0, fmean(vals) - penalty * pstdev(vals))))
    return out


def boundary_distributions(samples: Iterable[Sequence[Interval]], duration: float,
                           fps: float = 4.0, bandwidth_seconds: float = 1.0
                           ) -> tuple[list[float], list[float]]:
    """Convert repeated predictions into smooth start/end distributions."""
    n = max(1, int(math.floor(duration * fps)))
    starts, ends = [0.0] * n, [0.0] * n
    sigma = max(1e-6, bandwidth_seconds * fps)
    for sample in samples:
        for interval in sample:
            for target, value in ((starts, interval.start), (ends, interval.end)):
                center = value * fps
                for i in range(n):
                    target[i] += math.exp(-0.5 * ((i - center) / sigma) ** 2)
    for target in (starts, ends):
        total = sum(target)
        if total > 0:
            target[:] = [x / total for x in target]
    return starts, ends
