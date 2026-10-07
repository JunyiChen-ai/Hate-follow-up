#!/usr/bin/env python3
"""ZS-ImageBind (audio channel) on DeHate with the exact setup of the HateMM / HateClipSeg rows
(runs/20261008_baselines/zs_imagebind_audio/).

Port of Retrieval-hate `scripts/repro_campaign/extract_imagebind.py` (audio channel only) and the scoring of
`eval_frame.imagebind_curves`; uoa-lab2 `~/Retrieval-hate` is read only.

  audio     ffmpeg `-map 0:a:0 -ac 1 -ar 16000 -f f32le` from the video file; 2 s clips tiling the track
            (stride 2 s, a clip shorter than 400 samples is zero-padded to 400); ImageBind `waveform2melspec`
            numerics (kaldi fbank, 128 mel bins, target length 204, 25 ms / 10 ms), normalised with mean -4.268,
            std 9.138; ImageBind-Huge in float16, batch 16; embeddings stored float16 (T, 1024).
  score     softmax over (normal, hateful) of the unit-normalised audio embedding times the raw text embeddings
            of ["normal", "hateful"] (the campaign's own `imagebind_text_normal_hateful.npy`, copied to
            data/retrieval_hate_repro/imagebind_audio/); score = p(hateful). Native rate 0.5 fps, piecewise
            constant on the 4 fps grid; frames past the last clip hold its value.
  F1        a video without an audio stream gets the embedding of digital silence of the video's duration
            (run_plan.md §1.3; the campaign dropped such videos instead). These embeddings are kept in the run
            directory, not in the feature cache.

Stages: verify (5 HateMM test videos vs the campaign's cached embeddings), extract (DeHate, all 1341 test videos
-> data/imagebind_audio_2s/DeHate/), score (+ exact-cohort finalisation -> runs/20261008_baselines/
zs_imagebind_audio/DeHate/). No labels are read.
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402

CODE_PATH = "experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_audio.py"
IB_DIR = REPO / "third_party/lavad/libs/ImageBind"
IB_CKPT = REPO / "data/assets/imagebind/imagebind_huge.pth"
TEXT_EMB = REPO / "data/retrieval_hate_repro/imagebind_audio/imagebind_text_normal_hateful.npy"
FEAT_DIR = REPO / "data/imagebind_audio_2s/DeHate"
OUT = ec.OUT_ROOT / "zs_imagebind_audio" / "DeHate"
SILENCE_DIR = OUT / "silence_embeddings"
RH = Path.home() / "Retrieval-hate"          # read only (uoa-lab2), verify stage only
HMM_VIDEO = Path.home() / "data/HateMM/video"

SR = 16000
CLIP_S = 2.0
NUM_MEL_BINS = 128
TARGET_LENGTH = 204
AUD_MEAN, AUD_STD = -4.268, 9.138
RATE = 1.0 / CLIP_S


def load_wav(path: Path):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def melspec(waveform):
    """imagebind.data.waveform2melspec, verbatim numerics (as in the campaign script)."""
    import torch
    import torchaudio
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=SR, use_energy=False,
        window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0,
        frame_length=25, frame_shift=10)
    fbank = fbank.transpose(0, 1)
    p = TARGET_LENGTH - fbank.size(1)
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, :TARGET_LENGTH]
    return fbank.unsqueeze(0)


class Encoder:
    def __init__(self, batch: int):
        import torch
        sys.path.insert(0, str(IB_DIR))
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        self.torch, self.MT = torch, ModalityType
        self.model = imagebind_model.imagebind_huge(pretrained=False)
        self.model.load_state_dict(torch.load(IB_CKPT, map_location="cpu"))
        self.model = self.model.to("cuda", dtype=torch.float16).eval()
        self.batch = batch

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        torch = self.torch
        win = int(CLIP_S * SR)
        nclip = max(1, int(np.ceil(len(wav) / win)))
        mels = []
        for k in range(nclip):
            seg = wav[k * win:(k + 1) * win]
            if len(seg) < 400:
                seg = np.pad(seg, (0, 400 - len(seg)))
            m = melspec(torch.from_numpy(seg.copy())[None])
            mels.append((m - AUD_MEAN) / AUD_STD)
        out = []
        with torch.no_grad():
            for i in range(0, len(mels), self.batch):
                x = torch.stack(mels[i:i + self.batch]).to("cuda", torch.float16)
                e = self.model({self.MT.AUDIO: x})[self.MT.AUDIO]
                out.append(e.float().cpu().numpy())
        return np.concatenate(out, 0).astype(np.float16)


def atomic_save(d: Path, vid: str, arr: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{vid}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, d / f"{vid}.npy")


def scores(e: np.ndarray, text: np.ndarray) -> np.ndarray:
    """Retrieval-hate eval_frame.imagebind_curves (as in convert_rh_baselines.imagebind_audio_loader)."""
    e = e.astype(np.float32)
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
    logits = e @ text.T
    m = logits.max(axis=1, keepdims=True)
    q = np.exp(logits - m)
    return (q[:, 1] / q.sum(axis=1)).astype(np.float64)


def stage_verify(args, log) -> int:
    enc = Encoder(args.batch)
    ids = ec.cohort("HateMM")[:5]
    res, ok = {}, True
    text = np.load(TEXT_EMB)
    for v in ids:
        wav = load_wav(HMM_VIDEO / f"{v}.mp4")
        ref_p = RH / f"data/CLIP_Embedding/HateMM/imagebind_audio/{v}.npy"
        if wav is None or not ref_p.exists():
            res[v] = {"skipped": "no audio or no campaign embedding"}
            continue
        mine = enc(wav).astype(np.float32)
        ref = np.load(ref_p).astype(np.float32)
        n = min(len(mine), len(ref))
        cos = float(np.mean(np.sum(mine[:n] * ref[:n], 1) /
                            (np.linalg.norm(mine[:n], axis=1) * np.linalg.norm(ref[:n], axis=1) + 1e-12)))
        res[v] = {"T_mine": len(mine), "T_campaign": len(ref), "mean_cos": cos,
                  "max_abs_score_diff": float(np.abs(scores(mine[:n], text) - scores(ref[:n], text)).max())}
        ok &= len(mine) == len(ref) and cos > 0.999
        log(f"VERIFY {v} {json.dumps(res[v])}")
    ok &= sum("mean_cos" in r for r in res.values()) >= 3
    (OUT / "verify.json").write_text(json.dumps({"ok": ok, "videos": res}, indent=2) + "\n")
    log(f"VERIFY {'PASSED' if ok else 'FAILED'} (same clip count and mean cosine > 0.999 on HateMM test videos)")
    return 0 if ok else 4


def stage_extract(args, log) -> int:
    enc = Encoder(args.batch)
    man = ec.manifest_rows("DeHate")
    stats = {"n": 0, "noaudio": [], "missing_file": [], "oom": []}
    t0 = time.time()
    todo = [(v, r) for v, r in sorted(man.items())
            if not (FEAT_DIR / f"{v}.npy").exists() and not (SILENCE_DIR / f"{v}.npy").exists()]
    log(f"extract: {len(todo)} of {len(man)} videos to do")
    for n, (v, r) in enumerate(todo, 1):
        p = Path(r["video_path"])
        if not p.exists():
            stats["missing_file"].append(v)
            log(f"[MISS] {v}")
            continue
        wav = load_wav(p)
        target = FEAT_DIR
        if wav is None:
            # F1: digital silence of the same length (run_plan.md §1.3)
            wav = np.zeros(int(round(float(r["duration"]) * SR)), dtype=np.float32)
            target = SILENCE_DIR
            stats["noaudio"].append(v)
            log(f"[NOAUDIO] {v} -> F1 silence embedding ({r['duration']} s)")
        try:
            e = enc(wav)
        except enc.torch.cuda.OutOfMemoryError:
            enc.torch.cuda.empty_cache()
            stats["oom"].append(v)
            log(f"[OOM] {v}")
            continue
        atomic_save(target, v, e)
        stats["n"] += 1
        if n % 50 == 0:
            el = time.time() - t0
            log(f"PROGRESS {n}/{len(todo)} elapsed={el:.0f}s")
    stats["wall_seconds"] = round(time.time() - t0, 1)
    (OUT / "extract_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"extract done: {stats['n']} videos, no audio {len(stats['noaudio'])}, missing {len(stats['missing_file'])}, "
        f"OOM {len(stats['oom'])}, {stats['wall_seconds']} s")
    return 0 if not stats["missing_file"] else 5


def stage_score(args, log) -> int:
    text = np.load(TEXT_EMB)
    dur = ec.durations("DeHate")
    curves, extra, fail = {}, {}, {}
    n_tail, n_f1 = 0, 0
    for v in ec.cohort("DeHate"):
        p, f1 = FEAT_DIR / f"{v}.npy", False
        if not p.exists():
            p, f1 = SILENCE_DIR / f"{v}.npy", True
        if not p.exists():
            fail[v] = "no embedding"
            continue
        s = scores(np.load(p), text)
        T = ec.curve_length(dur[v])
        tail = ec.tail_frames(len(s), RATE, T)
        n_tail += tail
        n_f1 += f1
        curves[v] = ec.broadcast_to_4fps(s, RATE, T)
        extra[v] = {"native_samples": int(len(s)), "tail_hold_frames": int(tail)}
        if f1:
            extra[v]["fallback"] = {"code": "F1", "reason": "no audio stream: digital-silence embedding"}
    rep = ec.finalize("zs_imagebind_audio", "DeHate", OUT, curves, native_rate=RATE, code_path=CODE_PATH, log=log,
                      extra=extra, failures=fail,
                      notes={"tail_hold_frames_total": n_tail, "f1_silence_videos": n_f1},
                      config={"variant": "audio channel, text ['normal', 'hateful']",
                              "backbone": "ImageBind-Huge (third_party/lavad/libs/ImageBind, "
                                          "data/assets/imagebind/imagebind_huge.pth), float16",
                              "features": str(FEAT_DIR.relative_to(REPO)),
                              "text_embeddings": str(TEXT_EMB.relative_to(REPO)),
                              "native_rate": "0.5 fps (2 s audio clips)",
                              "score": "softmax over (normal, hateful); p(hateful)",
                              "grid": "frame i <- clip floor(i/4 * 0.5); frames past the last clip hold its value",
                              "fallback_F1": "no audio stream -> embedding of digital silence of the duration"})
    return 0 if rep.get("exact_test_set") else 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify", "extract", "score"])
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = ec.RunLog(OUT / "run.log", append=True)
    (OUT / "run.pid").write_text(f"{os.getpid()}\n")
    log(f"stage {a.stage}; code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
    rc = {"verify": stage_verify, "extract": stage_extract, "score": stage_score}[a.stage](a, log)
    log(f"stage {a.stage} {'DONE' if rc == 0 else f'FAILED rc={rc}'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
