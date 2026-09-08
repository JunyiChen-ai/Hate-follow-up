#!/usr/bin/env python3
"""Extract the independent ImageBind audio view for LESS-3V.

This runner is deliberately manifest-driven and resumable.  It never reads a
label or a ground-truth file; each 2 s native audio clip becomes one embedding.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
IB_DIR = ROOT / "third_party/lavad/libs/ImageBind"
IB_CKPT = ROOT / "results/label_free_adapt/assets/imagebind/imagebind_huge.pth"
SR = 16_000
CLIP_SECONDS = 2.0
NUM_MEL_BINS = 128
TARGET_LENGTH = 204
AUDIO_MEAN = -4.268
AUDIO_STD = 9.138


def load_wav(path: Path) -> np.ndarray | None:
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
               "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
               "-f", "f32le", "-"]
    process = subprocess.run(command, capture_output=True)
    if process.returncode or not process.stdout:
        return None
    return np.frombuffer(process.stdout, dtype=np.float32).copy()


def melspec(waveform: torch.Tensor) -> torch.Tensor:
    import torchaudio

    waveform = waveform - waveform.mean()
    bank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=SR, use_energy=False,
        window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0,
        frame_length=25, frame_shift=10)
    bank = bank.transpose(0, 1)
    pad = TARGET_LENGTH - bank.size(1)
    if pad > 0:
        bank = torch.nn.functional.pad(bank, (0, pad), value=0)
    elif pad < 0:
        bank = bank[:, :TARGET_LENGTH]
    return bank.unsqueeze(0)


def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open()]
    if args.limit:
        rows = rows[:args.limit]
    sys.path.insert(0, str(IB_DIR))
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    device = "cuda"
    dtype = torch.float16
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(IB_CKPT, map_location="cpu"))
    model = model.to(device, dtype=dtype).eval()

    statistics = {"requested": len(rows), "written": 0, "skipped": 0,
                  "no_audio": [], "errors": []}
    started = time.time()
    for index, row in enumerate(rows, 1):
        output = args.out_dir / row["dataset"] / f'{row["video_id"]}.npy'
        if output.exists():
            statistics["skipped"] += 1
            continue
        waveform = load_wav(Path(row["video_path"]))
        if waveform is None:
            statistics["no_audio"].append([row["dataset"], row["video_id"]])
            continue
        window = int(CLIP_SECONDS * SR)
        clips = max(1, int(np.ceil(len(waveform) / window)))
        features = []
        for clip in range(clips):
            segment = waveform[clip * window:(clip + 1) * window]
            if len(segment) < 400:
                segment = np.pad(segment, (0, 400 - len(segment)))
            features.append((melspec(torch.from_numpy(segment)[None]) -
                             AUDIO_MEAN) / AUDIO_STD)
        embeddings = []
        try:
            with torch.no_grad():
                for start in range(0, len(features), args.batch_size):
                    inputs = torch.stack(features[start:start + args.batch_size]).to(
                        device, dtype)
                    encoded = model({ModalityType.AUDIO: inputs})[ModalityType.AUDIO]
                    embeddings.append(encoded.float().cpu().numpy())
            atomic_save(output, np.concatenate(embeddings).astype(np.float16))
            statistics["written"] += 1
        except Exception as error:  # preserve progress and record exact failures
            statistics["errors"].append(
                [row["dataset"], row["video_id"], type(error).__name__, str(error)])
        if index % 10 == 0 or index == len(rows):
            elapsed = time.time() - started
            rate = (statistics["written"] + statistics["skipped"]) / max(elapsed, 1e-9)
            print(f"PROGRESS {index}/{len(rows)} written={statistics['written']} "
                  f"rate={rate:.2f} videos/s", flush=True)

    statistics["wall_seconds"] = round(time.time() - started, 2)
    report = args.out_dir / "extract_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(statistics, indent=2))
    print(json.dumps(statistics), flush=True)
    return 0 if not statistics["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
