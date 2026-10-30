"""Window evidence for the Contextual Reader (adopted by the user 2026-10-31): each window branch carries its own
moment's evidence, placed after the shared verdict-conditioned context and right before its question.

- Visual branch: the window's 0.5 fps frames, labelled "[t=..s]" like the prefix frames, then the r6 visual
  question, as one user turn on the shared cache. Promoted from experiments/20261007_m1_streamingtom/
  placement_controls.py (`adjacent_local`, `suffix_margin`).
- Speech branch: the r6 speech question with the words whose forced-aligned midpoint lies in the window; videos
  without an alignment keep the proportional slice. Promoted from experiments/20261014_m1_audio_bound/
  measure_audio.py (`aligned_words`, `aligned_slice`, `prediction`).
- Frame sampler: the first decoded frame at/after every 1 / 0.5 s target, saved as PNG with its decoded PTS time,
  as experiments/20261006_m1_rekv/source_frames.py. Cache `data/temporal_retrieved_kv/<dataset>/<video_id>/`.
No labels are read anywhere in this module.
"""
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.actual_video_frames import resolve_video
from src.stance_cache import positions
from src.video_inputs import fixed_windows

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FPS = 0.5
WINDOW_S = 8.0
FRAME_CACHE = ROOT / "data/temporal_retrieved_kv"
ALIGNED = ROOT / "data/asr_whisper_aligned"
NATIVE_LABEL = "[t={time:.1f}s]\n"


# ---------------------------------------------------------------- 0.5 fps frames
def frame_targets(duration):
    assert duration > 0
    return [i / SOURCE_FPS for i in range(math.ceil(duration * SOURCE_FPS))]


def acquire_frames(row, generator):
    """Decode the video once; keep the first frame at/after every target (PNG). Writes metadata.json last."""
    import av
    folder = FRAME_CACHE / row["dataset"] / row["video_id"]
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    duration = float(row["duration"])
    targets = frame_targets(duration)
    source = resolve_video(row)
    entries, selected, saved = [], [], set()
    cursor, decoded, origin, previous = 0, 0, None, -math.inf
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        origin = float(container.start_time / av.time_base) if container.start_time is not None else None
        for index, frame in enumerate(container.decode(stream)):
            decoded += 1
            assert frame.pts is not None, "decoded frame has no PTS"
            absolute = float(frame.pts * frame.time_base)
            if origin is None:
                origin = absolute
            t = absolute - origin
            assert t > previous, "PTS not monotone"
            previous = t
            if not 0 <= t < duration:
                continue
            entries.append(dict(index=index, pts=int(frame.pts), time=t, absolute=absolute,
                                time_base=[frame.time_base.numerator, frame.time_base.denominator], width=frame.width, height=frame.height))
            while cursor < len(targets) and t >= targets[cursor]:
                selected.append(index)
                cursor += 1
                if index not in saved:
                    image = frame.to_image().convert("RGB")
                    try:
                        image.save(folder / "frames" / f"frame_{index:08d}.png")
                    finally:
                        image.close()
                    saved.add(index)
    selected.extend([None] * (len(targets) - cursor))
    src = dict(input_video=str(source), manifest_video_path=row["video_path"], duration=duration, origin=origin, decoded_frames=decoded,
               entries=entries, source_fps=SOURCE_FPS, targets=targets, target_indices=selected, selected_indices=sorted(saved),
               uncovered_targets=[i for i, v in enumerate(selected) if v is None], decode_seconds=time.perf_counter() - start)
    assert all((folder / "frames" / f"frame_{i:08d}.png").exists() for i in src["selected_indices"])
    meta = dict(version="window-evidence 0.5 fps frames; 2026-10-31", spec=dict(source_fps=SOURCE_FPS), GT_read=False, host=socket.gethostname(),
                date=time.strftime("%Y-%m-%d"), dataset=row["dataset"], video_id=row["video_id"], generator=generator, source=src)
    tmp = folder / "metadata.json.partial"
    tmp.write_text(json.dumps(meta) + "\n")
    tmp.replace(folder / "metadata.json")
    return meta


def local_frames(row):
    """[(time, path)] of the cached 0.5 fps frames of one video."""
    folder = FRAME_CACHE / row["dataset"] / row["video_id"]
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["GT_read"] is False and (meta["dataset"], meta["video_id"]) == (row["dataset"], row["video_id"])
    src = meta["source"]
    assert src["source_fps"] == SOURCE_FPS and float(src["duration"]) == float(row["duration"])
    entries = {e["index"]: e for e in src["entries"]}
    out = []
    for i in src["selected_indices"]:
        p = folder / "frames" / f"frame_{i:08d}.png"
        assert p.exists(), p
        out.append((float(entries[i]["time"]), p))
    return out


def in_window(frames, t1, t2):
    return [(t, p) for t, p in frames if t1 <= t < t2]


# ---------------------------------------------------------------- aligned words
def aligned_words(dataset):
    out = {}
    for line in open(ALIGNED / dataset / "words.jsonl"):
        r = json.loads(line)
        out[r["video_id"]] = r
    return out


def aligned_slice(rec, t1, t2, fallback):
    """Words whose midpoint lies in [t1, t2); videos without an alignment keep the proportional slice."""
    if rec is None or rec["status"] != "aligned":
        return fallback
    return " ".join(w["w"] for w in rec["words"] if t1 <= (w["s"] + w["e"]) / 2 < t2)


# ---------------------------------------------------------------- frames next to the question
def _open_images(paths):
    images = []
    try:
        for p in paths:
            with Image.open(p) as original:
                images.append(original.convert("RGB"))
        return images
    except Exception:
        for image in images:
            image.close()
        raise


def _forward_margin(j, enc, position_ids, cache):
    kwargs = j.model_inputs(enc)
    kwargs.pop("attention_mask", None)  # the processor's mask covers the suffix only; the model builds the causal mask over cache + suffix
    kwargs["position_ids"] = position_ids
    out = j.model.model(**kwargs, past_key_values=cache, use_cache=True)
    hidden = out.last_hidden_state[0, -1].clone()
    del out
    return j.margins_fp32(hidden[None])[0]


@torch.no_grad()
def frames_margin(j, cache, ctx, frames, question):
    """The window's frames with "[t=..s]" labels, then the question, as one user turn on the shared cache."""
    labelled = [(NATIVE_LABEL.format(time=t), p) for t, p in frames]
    assert labelled and cache.get_seq_length() == ctx["stance_cache_tokens"]
    content = []
    for label, path in labelled:
        content += [dict(type="text", text=label), dict(type="image", image=str(path))]
    content.append(dict(type="text", text=question))
    full = j.render(ctx["msgs"] + ctx["history"] + [dict(role="user", content=content)], True)
    assert full.startswith(ctx["head"])
    text = full[len(ctx["head"]):]
    assert all(label in text for label, _ in labelled) and text.count(question) == 1
    images = _open_images([p for _, p in labelled])
    try:
        enc = j.encode(text, images)
    finally:
        for image in images:
            image.close()
    ids = enc["input_ids"].to(j.device)
    grid = enc["image_grid_thw"].to(j.device)
    assert len(grid) == len(labelled)
    counts = [int(torch.tensor(g).prod()) // j.processor.image_processor.merge_size ** 2 for g in grid.tolist()]
    assert int((ids == j.image_token_id).sum()) == sum(counts)
    relative, _ = positions(j, ids, grid)
    n = cache.get_seq_length()
    j.model.model.rope_deltas = ctx["rope"].clone()
    try:
        return _forward_margin(j, enc, relative + ctx["stance_cache_logical_start"], cache)
    finally:
        cache.crop(n)
        j.model.model.rope_deltas = ctx["rope"].clone()


# ---------------------------------------------------------------- output record
def window_prediction(row, ctx, visual, speech, method, seconds, code_path):
    """Prediction record in the schema the decoder reads (extra.windows with z_visual / z_speech, extra.z_video)."""
    ww = []
    for w, v, s in zip(fixed_windows(float(row["duration"]), WINDOW_S), visual, speech):
        one = dict(i=len(ww), start=w[0], end=w[1], z_visual=v, z=max(v, s) if s is not None else v)
        if s is not None:
            one["z_speech"] = s
        ww.append(one)
    idx = np.clip(((np.arange(math.ceil(float(row["duration"]) * 4)) + .5) / 4 // WINDOW_S).astype(int), 0, len(ww) - 1)
    return dict(schema_version=1, method=method, dataset=row["dataset"], video_id=row["video_id"], duration=float(row["duration"]),
                native_rate=4, score_curve=np.asarray([w["z"] for w in ww])[idx].tolist(), intervals=[], error=None, seed=0,
                calls=3 + len(ww) + sum(s is not None for s in speech), code_path=code_path,
                extra=dict(z_video=ctx["global_margin"], stance=ctx["stance"], prefix_tokens=ctx.get("prefix_tokens"),
                           stance_cache_tokens=ctx["stance_cache_tokens"], windows=ww, standalone_seconds=seconds))
