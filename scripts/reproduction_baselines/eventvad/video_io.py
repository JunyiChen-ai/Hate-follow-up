#!/usr/bin/env python
"""Decoding for the EventVAD port: one ffmpeg path for every codec.

Upstream `src/event_seg/utils.py:video_to_frames` opens the file with
`cv2.VideoCapture`, reads **every** frame into one list, and returns
`np.array(frames)`. Two things stop that being usable here.

**AV1.** The OpenCV build in this environment has no AV1 decoder and a large
minority of the MultiHateClip files are AV1; `cap.read()` returns False on the
first call and upstream would hand the graph an empty array. The rest of this
study solves that with an ffmpeg fallback beside an OpenCV main path
(`scripts/duplex/extract_clip_features.py`, `vadr1/run_vadr1_inference.py`).
Here there is no reason to keep two paths: the segmentation stage wants a
whole decoded stream rather than a handful of indexed frames, and the system
ffmpeg reads H.264, HEVC and AV1 alike. One path also removes the question of
whether the two decoders agree.

**Memory.** `np.array(frames)` for a 1000 s 720p video at 30 fps is 30000
frames x 1280 x 720 x 3 bytes = 83 GB. The features derived from those frames
are (n, 640) float32, i.e. 77 MB, so the frames are the only thing that does
not fit. `iter_frames` therefore streams: ffmpeg writes rawvideo to a pipe and
the caller consumes one frame at a time. Nothing downstream needs random
access -- CLIP is per frame and RAFT is per adjacent pair -- so the streamed
features are the features upstream's whole-array code would have produced.

Matching upstream's pixels
    Colour: upstream does `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`; `rgb24`
    out of ffmpeg is that same channel order.
    Resize: upstream does `cv2.resize(..., new_size)` with the default
    `INTER_LINEAR`, so the scale filter is pinned to `flags=bilinear` rather
    than ffmpeg's bicubic default.
    Size: `EventVADConfig.output_size` is upstream's rule character for
    character -- cap to (1280, 720) by the smaller ratio, then round both
    sides down to even.

Rate
    Upstream reads every frame; the paper fixes FPS = 30. `max_fps` caps
    rather than resamples: a 60 fps file is decoded at 30, a 25 fps file stays
    at 25. Upsampling 25 to 30 would insert duplicated frames, and a duplicate
    frame has exactly zero optical flow and cosine dissimilarity zero, which
    the boundary detector would read as a stretch of perfect event continuity
    that the video does not contain.
"""

from __future__ import annotations

import json
import subprocess


class VideoProbe:
    """What ffprobe reports about a file, plus the decode plan."""

    def __init__(self, path, width, height, native_fps, duration,
                 out_width, out_height, decode_fps):
        self.path = path
        self.width = width
        self.height = height
        self.native_fps = native_fps
        self.duration = duration
        self.out_width = out_width
        self.out_height = out_height
        self.decode_fps = decode_fps

    def as_dict(self):
        return {
            "width": self.width, "height": self.height,
            "native_fps": self.native_fps, "duration": self.duration,
            "out_width": self.out_width, "out_height": self.out_height,
            "decode_fps": self.decode_fps,
        }

    def __repr__(self):
        return ("VideoProbe(%dx%d @%.3f -> %dx%d @%.3f, %.2fs)"
                % (self.width, self.height, self.native_fps,
                   self.out_width, self.out_height, self.decode_fps,
                   self.duration))


def _parse_rate(text):
    if not text or text == "0/0":
        return 0.0
    if "/" in text:
        num, den = text.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    return float(text)


def probe(path, cfg):
    """ffprobe the file and resolve the output size and decode rate."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("ffprobe failed on %s: %s"
                           % (path, res.stderr.strip()[:300]))
    info = json.loads(res.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError("no video stream in %s" % path)
    st = streams[0]
    width, height = int(st["width"]), int(st["height"])
    fps = _parse_rate(st.get("avg_frame_rate")) or _parse_rate(
        st.get("r_frame_rate"))
    if fps <= 0:
        raise RuntimeError("ffprobe reports no frame rate for %s" % path)
    duration = float((info.get("format") or {}).get("duration") or 0.0)

    out_w, out_h = cfg.output_size(width, height)
    decode_fps = min(fps, cfg.max_fps) if cfg.max_fps else fps
    return VideoProbe(path, width, height, fps, duration,
                      out_w, out_h, decode_fps)


def iter_frames(pr, cfg=None):
    """Yield RGB uint8 frames of shape (out_height, out_width, 3).

    `pr` is a VideoProbe. The generator owns the ffmpeg process and tears it
    down on exhaustion or on an exception raised by the consumer.
    """
    import numpy as np

    filters = []
    # Only ever downsample. `probe` already clamped decode_fps to the native
    # rate, so this fires exactly when the file is faster than the cap.
    if pr.decode_fps < pr.native_fps - 1e-6:
        filters.append("fps=%.10g" % pr.decode_fps)
    if (pr.out_width, pr.out_height) != (pr.width, pr.height):
        filters.append("scale=%d:%d:flags=bilinear"
                       % (pr.out_width, pr.out_height))

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", pr.path]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    stride = pr.out_width * pr.out_height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=stride * 4)
    n = 0
    try:
        while True:
            buf = proc.stdout.read(stride)
            if not buf:
                break
            if len(buf) < stride:
                # A torn final frame is a truncated file, not a frame.
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                pr.out_height, pr.out_width, 3)
            n += 1
            yield frame
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        err = proc.stderr.read() if proc.stderr is not None else b""
        if proc.stderr is not None:
            proc.stderr.close()
        proc.wait()
        if n == 0:
            raise RuntimeError("ffmpeg decoded 0 frames from %s: %s"
                               % (pr.path, err.decode("utf-8", "ignore")[:300]))


def read_frame_range(pr, indices):
    """Pull specific 0-based frame indices of the *decoded* stream.

    Used by the scoring stage, which needs 16 frames out of one event rather
    than the whole video. The indices are indices into the same stream
    `iter_frames` produces -- same rate, same size -- so an event's frame
    numbers mean the same thing in both stages.
    """
    import numpy as np

    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return []
    keep = set(wanted)
    last = wanted[-1]
    out = {}
    for i, frame in enumerate(iter_frames(pr)):
        if i in keep:
            out[i] = np.ascontiguousarray(frame)
        if i >= last:
            break
    if not out:
        raise RuntimeError("no frames decoded from %s for indices %s"
                           % (pr.path, wanted[:5]))
    # Hold the last available frame for indices past the end of the stream,
    # which happens when ffprobe's duration over-reports the decoded length.
    highest = max(out)
    return [out.get(int(i), out[highest]) for i in indices]
