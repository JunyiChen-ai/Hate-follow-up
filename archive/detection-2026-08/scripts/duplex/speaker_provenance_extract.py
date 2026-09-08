#!/usr/bin/env python3
"""Restore coarse speaker ownership and visible-speaker roles for the probe."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "speaker_provenance"
MODEL_ID = "microsoft/wavlm-base-plus-sv"
SEED = 20260808
MIN_WINDOW = 1.5
MAX_WINDOW = 3.0
MIN_CLUSTER_WINDOWS = 2
MIN_TWO_WINDOWS = 4
SILHOUETTE_BAR = 0.20


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def media_index() -> dict[str, Path]:
    out = {}
    for p in glob.glob("/home/jehc223/data/ImpliHateVid_video*/*.mp4"):
        out[Path(p).stem] = Path(p)
    return out


def repair_chunks(chunks: list[dict], duration: float) -> list[dict]:
    clean = []
    prev_end = 0.0
    for i, c in enumerate(chunks):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        start, end = c.get("start"), c.get("end")
        start = float(start) if isinstance(start, (int, float)) else prev_end
        if not isinstance(end, (int, float)) or float(end) <= start:
            future = [x.get("start") for x in chunks[i + 1:]
                      if isinstance(x.get("start"), (int, float)) and float(x["start"]) > start]
            end = min(future) if future else min(duration, start + max(2.0, len(text) / 13.0))
        end = min(float(end), duration)
        start = max(0.0, min(start, end))
        if end - start < 0.15:
            end = min(duration, start + min(2.0, max(0.25, len(text) / 13.0)))
        clean.append({"start": start, "end": end, "text": text})
        prev_end = max(prev_end, end)
    return clean


def make_windows(chunks: list[dict]) -> list[dict]:
    windows = []
    for ci, c in enumerate(chunks):
        length = c["end"] - c["start"]
        if length <= 0:
            continue
        n = max(1, int(math.ceil(length / MAX_WINDOW)))
        edges = np.linspace(c["start"], c["end"], n + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < MIN_WINDOW:
                mid = 0.5 * (a + b)
                a, b = mid - MIN_WINDOW / 2, mid + MIN_WINDOW / 2
            windows.append({"start": max(0.0, float(a)), "end": float(b), "chunk": ci})
    return windows


def cosine_kmeans2(x: np.ndarray, iters: int = 50) -> np.ndarray:
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    i0, i1 = np.unravel_index(np.argmin(sim), sim.shape)
    centers = np.stack([x[i0], x[i1]])
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ centers.T, axis=1)
        if np.array_equal(new, labels) and _:
            break
        labels = new
        for k in (0, 1):
            if np.any(labels == k):
                centers[k] = np.mean(x[labels == k], axis=0)
                centers[k] /= max(np.linalg.norm(centers[k]), 1e-12)
    return labels


def cosine_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    vals = []
    for i in range(len(x)):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]
        other = np.where(labels != labels[i])[0]
        if not len(same) or not len(other):
            vals.append(0.0)
            continue
        a, b = np.mean(d[i, same]), np.mean(d[i, other])
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals))


def cluster_embeddings(x: np.ndarray) -> tuple[np.ndarray, float, bool]:
    if len(x) < MIN_TWO_WINDOWS:
        return np.zeros(len(x), dtype=int), 0.0, False
    labels = cosine_kmeans2(x)
    counts = np.bincount(labels, minlength=2)
    sil = cosine_silhouette(x, labels)
    use_two = bool(np.min(counts) >= MIN_CLUSTER_WINDOWS and sil >= SILHOUETTE_BAR)
    return (labels if use_two else np.zeros(len(x), dtype=int)), sil, use_two


def visible_face_fraction(video: Path, start: float, end: float, detector) -> tuple[float | None, int]:
    if end <= start or not video.exists():
        return None, 0
    cap = cv2.VideoCapture(str(video))
    times = np.linspace(start, end, 5, endpoint=False) + (end - start) / 10
    seen = 0
    valid = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        valid += 1
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(h, w))
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        if faces is not None and len(faces):
            seen += 1
    cap.release()
    return (seen / valid if valid else None), valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", type=Path, default=ROOT / "results/temporal_attribution/cohorts.json")
    ap.add_argument("--asr", type=Path, default=ROOT / "results/temporal_attribution/timestamped_asr.jsonl")
    ap.add_argument("--wav-dir", type=Path, default=ROOT / "results/c2_fullcorpus/wav")
    ap.add_argument("--output", type=Path, default=RUN / "provenance.jsonl")
    args = ap.parse_args()

    cohort_obj = json.loads(args.cohorts.read_text())
    ids = sorted({v for values in cohort_obj["cohorts"].values() for v in values})
    asr = {r["video_id"]: r for r in read_jsonl(args.asr) if r.get("video_id") in set(ids)}
    media = media_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = {r["video_id"] for r in read_jsonl(args.output)} if args.output.exists() else set()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioXVector.from_pretrained(MODEL_ID).to(device).eval()
    face_model = RUN / "models" / "face_detection_yunet_2026may.onnx"
    if not face_model.exists():
        raise SystemExit(f"Missing frozen YuNet model: {face_model}")
    face = cv2.FaceDetectorYN_create(
        str(face_model), "", (320, 320), score_threshold=0.8,
        nms_threshold=0.3, top_k=5000)

    for vi, vid in enumerate(ids, 1):
        if vid in done:
            continue
        wav, sr = torchaudio.load(str(args.wav_dir / f"{vid}.wav"))
        wav = torch.mean(wav, dim=0).numpy()
        duration = len(wav) / sr
        chunks = repair_chunks(asr.get(vid, {}).get("chunks", []), duration)
        windows = make_windows(chunks)
        embs = []
        valid_windows = []
        for w in windows:
            a, b = int(w["start"] * sr), min(len(wav), int(w["end"] * sr))
            audio = wav[max(0, a):b]
            if len(audio) < int(0.25 * sr):
                continue
            inp = extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.inference_mode():
                emb = model(**inp).embeddings[0]
                emb = torch.nn.functional.normalize(emb, dim=0)
            embs.append(emb.cpu().float().numpy())
            valid_windows.append(w)
        if embs:
            labels, sil, use_two = cluster_embeddings(np.stack(embs))
        else:
            labels, sil, use_two = np.zeros(0, dtype=int), 0.0, False

        for w, lab in zip(valid_windows, labels):
            w["speaker"] = int(lab)
        assigned_duration = [0.0, 0.0]
        for ci, c in enumerate(chunks):
            labs = [w["speaker"] for w in valid_windows if w["chunk"] == ci]
            if labs:
                c["speaker_raw"] = int(np.bincount(labs, minlength=2).argmax())
                assigned_duration[c["speaker_raw"]] += c["end"] - c["start"]
            else:
                c["speaker_raw"] = None
        primary = int(np.argmax(assigned_duration))
        role_counts = {"onscreen": 0, "voiceover": 0, "unknown": 0}
        rendered = []
        for c in chunks:
            frac, n_frames = visible_face_fraction(media.get(vid, Path()), c["start"], c["end"], face)
            c["face_visible_fraction"] = frac
            c["visual_frames_checked"] = n_frames
            visual = "ONSCREEN" if frac is not None and frac >= 0.4 else "VOICEOVER" if frac is not None else "UNKNOWN"
            role_counts[visual.lower()] += 1
            if c["speaker_raw"] is None:
                tag = "SPEAKER_UNKNOWN"
            else:
                owner = "PRIMARY" if c["speaker_raw"] == primary else "SECONDARY"
                tag = f"{owner}_{visual}"
            c["tag"] = tag
            rendered.append(f"[{tag}]: {c['text']}")
        rec = {
            "video_id": vid,
            "duration": duration,
            "n_chunks": len(chunks),
            "n_windows": len(valid_windows),
            "two_speakers": use_two,
            "silhouette": sil,
            "cluster_window_counts": np.bincount(labels, minlength=2).tolist() if len(labels) else [0, 0],
            "primary_raw_cluster": primary,
            "assigned_duration": assigned_duration,
            "role_counts": role_counts,
            "chunks": chunks,
            "provenance_transcript": "\n".join(rendered),
        }
        with args.output.open("a") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if vi == 1 or vi % 10 == 0:
            print(f"[{vi}/{len(ids)}] {vid} chunks={len(chunks)} windows={len(valid_windows)} "
                  f"two={use_two} sil={sil:.3f}", flush=True)


if __name__ == "__main__":
    main()
