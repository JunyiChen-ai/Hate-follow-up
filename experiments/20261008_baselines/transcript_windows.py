#!/usr/bin/env python3
"""Whisper transcript of a time span, shared by the audio-visual (transcript) variants of the label-free baselines
(user decision 2026-10-09): `lavad/lavad_av.py`, `zs_clip/zs_clip_av.py`, `zs_imagebind_audio/zs_imagebind_av.py`.

Segments: `data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl`, read by `qwen3_text/text_llm.py`
`load_segments` (imported unchanged, so the rule is the one the Qwen3-8B text baseline used): untimed chunks are
kept, a missing end becomes the next chunk's start (if later) or the video duration of the manifest, a missing start
the previous chunk's end (0 for the first chunk).

Span text: `src/video_inputs.py` `window_text` (the Reader's helper): a segment inside the span enters whole, a
segment that overlaps it partly is cut proportionally at word boundaries; the pieces are joined with spaces.
An empty span gives "" here. What an empty span means is the method's choice (LAVAD-AV writes "(no speech)" in the
prompt; ZS-CLIP-AV and ZS-ImageBind-AV leave the transcript modality out for that window).

No labels are read.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/20261008_baselines/qwen3_text"))
from src.video_inputs import window_text  # noqa: E402
from text_llm import load_segments  # noqa: E402  (qwen3_text/text_llm.py; imports only numpy and exact_cohort)

ASR_SOURCE = "data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl"
EMPTY = "(no speech)"


class Transcripts:
    """Per-corpus segment table; `span(vid, a, b)` = transcript text inside [a, b) seconds ("" if none)."""

    def __init__(self, ds: str):
        self.ds = ds
        self.segs = load_segments(ds)

    def span(self, vid: str, a: float, b: float) -> str:
        return " ".join(window_text(self.segs.get(vid, []), float(a), float(b)).split())

    def n_segments(self, vid: str) -> int:
        return len(self.segs.get(vid, []))
