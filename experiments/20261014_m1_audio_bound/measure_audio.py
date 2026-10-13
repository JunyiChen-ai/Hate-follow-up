#!/usr/bin/env python3
"""Candidate 43 measurement (README §2): one Qwen3-Omni prefix per video (r6 layout), then per 8 s window the
branches visual_native, visual_local, speech_prop, speech_aligned, speech_audio, speech_swap, speech_nbr on the
same cache. Records per video under runs/20261014_m1_audio_bound/<name>/records/<dataset>/<video_id>.json and the
per-arm predictions (r6 schema) under <name>/<arm>/predictions.jsonl. No label file is read.

Arms (README §2): audio = visual_local + speech_audio; text_aligned = visual_local + speech_aligned;
text_prop = visual_local + speech_prop; audio_swap = visual_local + speech_swap; audio_nbr = visual_local +
speech_nbr (reviewer's extra control); r6_inputs = visual_native + speech_prop.
"""
import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omni_judge import MODEL, SR, OmniJudge, build, media_margin, standalone, text_margin, tick  # noqa: E402
from src.mllm_judge import yesno_question  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402

EXP = ROOT / "runs/20261014_m1_audio_bound"
IMPLEMENTATION = "candidate 43 audio-bound speech evidence; Qwen3-Omni thinker; 2026-10-14"
WINDOW_S = 8.0
NATIVE_FRAMES = 20
SOURCE_FPS = 0.5
NATIVE_LABEL = "[t={time:.1f}s]\n"
ARMS = {"audio": ("visual_local", "speech_audio"), "text_aligned": ("visual_local", "speech_aligned"),
        "text_prop": ("visual_local", "speech_prop"), "audio_swap": ("visual_local", "speech_swap"),
        "audio_nbr": ("visual_local", "speech_nbr"), "r6_inputs": ("visual_native", "speech_prop")}
BRANCHES = ("visual_native", "visual_local", "speech_prop", "speech_aligned", "speech_audio", "speech_swap", "speech_nbr")
MIN_CLIP = 400  # samples: one analysis window of the feature extractor


def selected_rows(datasets, smoke):
    rows = load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", datasets)
    if smoke:  # the fixed smoke subset of experiments/20261007_m1_streamingtom/inputs.py, restricted to `datasets`
        rows = [r for ds in datasets for r in [x for x in rows if x["dataset"] == ds][:2]] + [
            r for r in rows if r["dataset"] == "HateMM" and r["video_id"] == "hate_video_114"]
    return rows


def local_frames(row):
    folder = ROOT / "data/temporal_retrieved_kv" / row["dataset"] / row["video_id"]
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


def aligned_words(dataset):
    path = ROOT / "data/asr_whisper_aligned" / dataset / "words.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["video_id"]] = r
    return out


def aligned_slice(rec, t1, t2):
    if rec is None or rec["status"] != "aligned":
        return None
    return " ".join(w["w"] for w in rec["words"] if t1 <= (w["s"] + w["e"]) / 2 < t2)


def derangement(n, video_index):
    if n < 2:
        return list(range(n))
    rng = np.random.default_rng([1, video_index])
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return [int(x) for x in p]


def neighbour(n):
    return [min(i + 1, n - 1) if i < n - 1 else max(i - 1, 0) for i in range(n)]


def load_wave(row):
    sr, wav = wavfile.read(ROOT / "data/audio16k" / row["dataset"] / (row["video_id"] + ".wav"))
    assert sr == SR and wav.ndim == 1 and wav.dtype == np.int16
    return wav.astype(np.float32) / 32768.0


def clip(wav, t1, t2):
    a, b = int(round(t1 * SR)), int(round(t2 * SR))
    x = wav[a:b]
    if len(x) < MIN_CLIP:
        x = np.pad(x, (0, MIN_CLIP - len(x)))
    return x


def speech_audio_parts(i, n, t1, t2):
    head = f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video."
    q = "Is THIS window one of the segments where speech that violates the above rules occurs?"
    return [{"type": "text", "text": f"{head} Judge only what is spoken in this window; its audio follows.\n\n"},
            {"type": "audio"},
            {"type": "text", "text": "\n\n" + q + '\n\nAnswer "Yes" or "No".'}]


def visual_local_parts(labelled, question):
    parts = []
    for label, _ in labelled:
        parts += [{"type": "text", "text": label}, {"type": "image"}]
    parts.append({"type": "text", "text": question})
    return parts


def prediction(row, ctx, visual, speech, name, seconds):
    ww = []
    for w, v, s in zip(fixed_windows(float(row["duration"]), WINDOW_S), visual, speech):
        one = dict(i=len(ww), start=w[0], end=w[1], z_visual=v, z=max(v, s) if s is not None else v)
        if s is not None:
            one["z_speech"] = s
        ww.append(one)
    idx = np.clip(((np.arange(math.ceil(float(row["duration"]) * 4)) + .5) / 4 // WINDOW_S).astype(int), 0, len(ww) - 1)
    return dict(schema_version=1, method=f"m1_audio_{name}", dataset=row["dataset"], video_id=row["video_id"], duration=float(row["duration"]),
                native_rate=4, score_curve=np.asarray([w["z"] for w in ww])[idx].tolist(), intervals=[], error=None, seed=0,
                calls=3 + len(ww) + sum(s is not None for s in speech), code_path="experiments/20261014_m1_audio_bound/measure_audio.py",
                extra=dict(z_video=ctx["global_margin"], stance=ctx["stance"], prefix_tokens=ctx["prefix_tokens"],
                           stance_cache_tokens=ctx["stance_cache_tokens"], windows=ww, standalone_seconds=seconds))


@torch.no_grad()
def read(j, row, video_index, segments, words, verify):
    begin = tick(j)
    first = (j.forward_calls, j.audio_calls, j.vision_calls)
    torch.cuda.reset_peak_memory_stats()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    assert 0 < len(native) <= NATIVE_FRAMES
    frames = local_frames(row)
    wav = load_wave(row)
    stamp = tick(j)
    cache, ctx = build(j, native, segments)
    times = {"prefix": tick(j) - stamp, **{b: 0.0 for b in BRANCHES}}
    wins = fixed_windows(float(row["duration"]), WINDOW_S)
    n = len(wins)
    swap, nbr = derangement(n, video_index), neighbour(n)
    values = {b: [] for b in BRANCHES}
    traces = []
    for i, (t1, t2) in enumerate(wins):
        body = window_text(segments, t1, t2)
        abody = aligned_slice(words, t1, t2)
        local = [(NATIVE_LABEL.format(time=t), p) for t, p in frames if t1 <= t < t2]
        trace = dict(i=i, bounds=[t1, t2], text_prop=body, text_aligned=abody, local_times=[t for t, _ in frames if t1 <= t < t2],
                     swap=swap[i], nbr=nbr[i])
        vq = yesno_question(i, n, t1, t2, body, "visual")
        stamp = tick(j); z, d = text_margin(j, cache, ctx, vq); times["visual_native"] += tick(j) - stamp
        values["visual_native"].append(z); trace["visual_native"] = dict(z=z, **d)
        if local:
            stamp = tick(j); z, d, _ = media_margin(j, cache, ctx, visual_local_parts(local, vq), [p for _, p in local]); times["visual_local"] += tick(j) - stamp
            values["visual_local"].append(z); trace["visual_local"] = dict(z=z, **d)
        else:
            values["visual_local"].append(values["visual_native"][-1]); trace["visual_local"] = None
        for name, text in (("speech_prop", body), ("speech_aligned", abody)):
            if text is not None and text.strip():
                stamp = tick(j); z, d = text_margin(j, cache, ctx, yesno_question(i, n, t1, t2, text, "speech")); times[name] += tick(j) - stamp
                values[name].append(z); trace[name] = dict(z=z, **d)
            else:
                values[name].append(None); trace[name] = None
        for name, k in (("speech_audio", i), ("speech_swap", swap[i]), ("speech_nbr", nbr[i])):
            a, b = wins[k]
            stamp = tick(j); z, d, _ = media_margin(j, cache, ctx, speech_audio_parts(i, n, t1, t2), (), [clip(wav, a, b)]); times[name] += tick(j) - stamp
            values[name].append(z); trace[name] = dict(z=z, source_window=k, **d)
        traces.append(trace)
    checks = dict(GT_read=False, host=socket.gethostname(), times=times, windows=n, derangement_ok=all(swap[i] != i for i in range(n)) or n < 2,
                  actual_forwards=j.forward_calls - first[0], actual_audio=j.audio_calls - first[1], actual_vision=j.vision_calls - first[2],
                  peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, pipeline_seconds=tick(j) - begin, audio_samples=int(len(wav)),
                  aligned_status=None if words is None else words["status"])
    expected = 1 + 2 + n * 3 + sum(t["visual_local"] is not None for t in traces) + sum(t["speech_prop"] is not None for t in traces) + sum(t["speech_aligned"] is not None for t in traces)
    assert checks["actual_forwards"] == expected, (checks["actual_forwards"], expected)
    assert checks["actual_audio"] == 3 * n and checks["actual_vision"] == 1 + sum(t["visual_local"] is not None for t in traces)
    diagnostic = None
    if verify:  # standalone forwards (no cache) for the first window with local frames and speech, every branch kind
        stamp = tick(j)
        k = next((t["i"] for t in traces if t["visual_local"] is not None and t["speech_prop"] is not None), 0)
        t1, t2 = wins[k]; body = traces[k]["text_prop"] or ""
        local = [(NATIVE_LABEL.format(time=t), p) for t, p in frames if t1 <= t < t2]
        vq = yesno_question(k, n, t1, t2, body, "visual")
        probes = {}
        z, ids = standalone(j, ctx, [{"type": "text", "text": vq}]); probes["visual_native"] = dict(standalone=z, cached=values["visual_native"][k], ids=ids)
        if local:
            z, ids = standalone(j, ctx, visual_local_parts(local, vq), [p for _, p in local]); probes["visual_local"] = dict(standalone=z, cached=values["visual_local"][k], ids=ids)
        if body.strip():
            z, ids = standalone(j, ctx, [{"type": "text", "text": yesno_question(k, n, t1, t2, body, "speech")}]); probes["speech_prop"] = dict(standalone=z, cached=values["speech_prop"][k], ids=ids)
        z, ids = standalone(j, ctx, speech_audio_parts(k, n, t1, t2), (), [clip(wav, t1, t2)]); probes["speech_audio"] = dict(standalone=z, cached=values["speech_audio"][k], ids=ids)
        # token seam: the standalone ids must end with the cached suffix ids of the same branch
        prefix_ids = j.encode_text_with_media(ctx["prefix_text"], ctx["files"])["input_ids"][0].tolist()
        head_len = ctx["stance_cache_tokens"]
        for name, p in probes.items():
            p["abs_difference"] = abs(p["standalone"] - p["cached"])
            p["tokens"] = len(p.pop("ids")) if name != "speech_audio" else len(p["ids"])
            if name == "speech_audio":
                ids = p.pop("ids"); p["head_ids_equal"] = ids[:len(prefix_ids)] == prefix_ids; p["audio_tokens"] = int(sum(1 for x in ids if x == j.audio_token_id))
        probes["head_tokens"] = head_len; probes["prefix_tokens"] = len(prefix_ids)
        probes["seconds"] = tick(j) - stamp
        diagnostic = dict(window=k, **probes)
    predictions = {}
    for arm, (vb, sb) in ARMS.items():
        secs = times["prefix"] + times[vb] + times[sb]
        p = prediction(row, ctx, values[vb], values[sb], arm, secs)
        predictions[arm] = p
    native = dict(global_margin=ctx["global_margin"], stance=ctx["stance"], stance_cache_tokens=ctx["stance_cache_tokens"],
                  prefix_tokens=ctx["prefix_tokens"], image_counts=ctx["image_counts"], frame_times=[t for t, _ in native])
    return dict(implementation=IMPLEMENTATION, model=MODEL, checks=checks, diagnostic=diagnostic, native=native, predictions=predictions,
                local_frames=[(t, str(p.relative_to(ROOT))) for t, p in frames], traces=traces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", required=True, help="run folder under runs/20261014_m1_audio_bound/")
    ap.add_argument("--verify-all", action="store_true", help="standalone checks on every video (smoke); default: first video per dataset")
    a = ap.parse_args()
    out = EXP / a.name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    import transformers
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, implementation=IMPLEMENTATION, arms=ARMS, branches=BRANCHES,
               window_seconds=WINDOW_S, native_frames=NATIVE_FRAMES, source_fps=SOURCE_FPS, GT_read=False, smoke=a.smoke, datasets=a.datasets,
               torch=torch.__version__, transformers=transformers.__version__, command="python -u " + " ".join(sys.argv),
               source="experiments/20261014_m1_audio_bound/{measure_audio,omni_judge}.py; 2026-10-14")
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert all(old[k] == cfg[k] for k in cfg if k not in ("date", "command", "host")), "run folder holds a different configuration"
    else:
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    rows = selected_rows(a.datasets, a.smoke)
    all_rows = load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", ("HateMM", "HateClipSeg"))
    index = {(r["dataset"], r["video_id"]): k for k, r in enumerate(all_rows)}
    asr = {ds: load_asr(ds) for ds in a.datasets}
    words = {ds: aligned_words(ds) for ds in a.datasets}
    number = 1
    while (out / f"pipeline_attempt_{number:04d}.json").exists():
        number += 1
    audit_path = out / f"pipeline_attempt_{number:04d}.json"
    audit = dict(host=cfg["host"], completed=False, GT_read=False, videos=[])
    start = time.perf_counter()
    torch.manual_seed(0)
    stamp = time.perf_counter()
    j = OmniJudge(MODEL)
    audit["model_load_seconds"] = time.perf_counter() - stamp
    results = {arm: [] for arm in ARMS}
    seen = set()
    try:
        for ordinal, row in enumerate(rows, 1):
            stamp = time.perf_counter()
            path = out / "records" / row["dataset"] / (row["video_id"] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            reused = path.exists()
            if reused:
                bundle = json.loads(path.read_text())
            else:
                verify = a.verify_all or row["dataset"] not in seen
                bundle = read(j, row, index[(row["dataset"], row["video_id"])], asr[row["dataset"]].get(row["video_id"], []),
                              words[row["dataset"]].get(row["video_id"]), verify)
                partial = path.with_suffix(".partial")
                partial.write_text(json.dumps(bundle, allow_nan=False) + "\n")
                partial.replace(path)
            seen.add(row["dataset"])
            for arm in ARMS:
                results[arm].append(bundle["predictions"][arm])
            d = bundle["diagnostic"]
            audit["videos"].append(dict(dataset=row["dataset"], video_id=row["video_id"], reused=reused, elapsed_seconds=time.perf_counter() - stamp,
                                        windows=bundle["checks"]["windows"], peak_GiB=bundle["checks"]["peak_GiB"],
                                        standalone_max_abs=max((v["abs_difference"] for k, v in d.items() if isinstance(v, dict)), default=None) if d else None))
            audit["elapsed_seconds"] = time.perf_counter() - start
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
            logging.info("%d/%d %s/%s windows=%d peak=%.1fGiB %.0fs standalone_max_abs=%s", ordinal, len(rows), row["dataset"], row["video_id"],
                         bundle["checks"]["windows"], bundle["checks"]["peak_GiB"], audit["videos"][-1]["elapsed_seconds"], audit["videos"][-1]["standalone_max_abs"])
        for arm, rr in results.items():
            folder = out / arm
            folder.mkdir(exist_ok=True)
            (folder / "config.json").write_text(json.dumps({**cfg, "arm": arm}, indent=2) + "\n")
            (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
        audit["completed"] = True
        logging.info("DONE coverage=%d", len(rows))
    finally:
        audit["elapsed_seconds"] = time.perf_counter() - start
        audit["actual_forwards"] = j.forward_calls
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
