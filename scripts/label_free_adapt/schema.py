#!/usr/bin/env python3
"""Prediction interchange format shared by every adaptation.

Inference receives a sanitized manifest with media metadata only.  Ground truth
is deliberately absent and is consumed later by the isolated evaluator.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

FPS = 4.0


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    score: float = 1.0

    def validate(self, duration: float) -> None:
        vals = (self.start, self.end, self.score)
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("interval contains a non-finite value")
        if self.start < 0 or self.end <= self.start or self.end > duration + 1e-6:
            raise ValueError(f"invalid interval {self} for duration={duration}")

    def as_list(self) -> list[float]:
        return [self.start, self.end, self.score]


@dataclass
class Prediction:
    method: str
    dataset: str
    video_id: str
    duration: float
    native_rate: float = FPS
    score_curve: list[float] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)
    modality_evidence: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    calls: int = 0
    seed: int = 20250819
    error: str | None = None

    def validate(self) -> None:
        if not self.method or not self.dataset or not self.video_id:
            raise ValueError("method, dataset and video_id are required")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be positive")
        if not math.isfinite(self.native_rate) or self.native_rate <= 0:
            raise ValueError("native_rate must be positive")
        if not isinstance(self.calls, int) or self.calls < 0:
            raise ValueError("calls must be a non-negative integer")
        if self.error is None and not self.score_curve and not self.intervals:
            raise ValueError("successful prediction needs a curve or intervals")
        if self.score_curve and not all(math.isfinite(float(x)) for x in self.score_curve):
            raise ValueError("score curve contains a non-finite value")
        for interval in self.intervals:
            interval.validate(self.duration)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "method": self.method,
            "dataset": self.dataset,
            "video_id": self.video_id,
            "duration": self.duration,
            "native_rate": self.native_rate,
            "score_curve": self.score_curve,
            "intervals": [x.as_list() for x in self.intervals],
            "modality_evidence": self.modality_evidence,
            "raw": self.raw,
            "calls": self.calls,
            "seed": self.seed,
            "error": self.error,
        }


def normalize_intervals(values: Iterable[Iterable[float]], duration: float,
                        merge_gap: float = 0.0) -> list[Interval]:
    """Clip, sort and merge model-produced intervals deterministically."""
    clean: list[Interval] = []
    for value in values:
        row = list(value)
        if len(row) < 2:
            continue
        a, b = sorted((float(row[0]), float(row[1])))
        score = float(row[2]) if len(row) > 2 else 1.0
        a, b = max(0.0, a), min(duration, b)
        if math.isfinite(a) and math.isfinite(b) and math.isfinite(score) and b > a:
            clean.append(Interval(a, b, score))
    clean.sort(key=lambda x: (x.start, x.end))
    merged: list[Interval] = []
    for item in clean:
        if merged and item.start <= merged[-1].end + merge_gap:
            prev = merged[-1]
            merged[-1] = Interval(prev.start, max(prev.end, item.end),
                                  max(prev.score, item.score))
        else:
            merged.append(item)
    return merged


def intervals_to_curve(intervals: Iterable[Interval], duration: float,
                       fps: float = FPS) -> list[float]:
    n = max(1, int(math.floor(duration * fps)))
    curve = [0.0] * n
    for interval in intervals:
        i0 = max(0, int(math.ceil(interval.start * fps - 1e-9)))
        i1 = min(n, int(math.ceil(interval.end * fps - 1e-9)))
        for i in range(i0, i1):
            curve[i] = max(curve[i], interval.score)
    return curve


def curve_to_intervals(curve: Iterable[float], duration: float, threshold: float,
                       fps: float = FPS) -> list[Interval]:
    """Threshold a dense score curve into half-open scored intervals."""
    values = [float(x) for x in curve]
    output = []
    start = None
    for index, score in enumerate(values + [float("-inf")]):
        active = score >= threshold
        if active and start is None:
            start = index
        elif not active and start is not None:
            segment = values[start:index]
            output.append(Interval(start / fps, min(duration, index / fps), max(segment)))
            start = None
    return output


def append_jsonl(path: Path, prediction: Prediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
