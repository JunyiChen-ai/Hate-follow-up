#!/usr/bin/env python3
"""1 fps audio features for the weakly supervised baselines (GPU, Slurm job on uoa-lab1).

For every id in the train / val / test(cohort) splits of data/weaksup_1fps/splits, one row per second, with as many
rows T as the video's clip_b16_1fps feature (row i = audio [i, i+1) s), so audio and visual rows line up:

  data/wav2vec2_base_1s/<corpus>/<id>.npy  (T, 768)  facebook/wav2vec2-base (pre-trained, not ASR-fine-tuned), last
      hidden layer, mean over the frames of each second; the waveform is fed in 20 s pieces (feature-extractor
      normalisation per piece), frame j of a piece belongs to second floor(j * 320 / 16000)
  data/wav2clip_1s/<corpus>/<id>.npy       (T, 512)  Wav2CLIP (descriptinc/lyrebird-wav2clip v0.1.0-alpha checkpoint,
      encoder + audio_transform, frozen) on 1 s frames, i.e. wav2clip.get_model(frame_length=16000, hop_length=16000)
      applied to the whole waveform; the spectrogram normalisation spans all frames of the video, as in that call

Input: 16 kHz mono wavs in Retrieval-hate data/AV2A_wav/<Dataset>/ (read-only, uoa-lab1). The audio is padded with
zeros or cut to T seconds. A video without a wav (no audio stream) gets digital silence of T seconds (run_plan §1.3 F1);
such ids are listed in the run log and in PROVENANCE.md.

    python experiments/20261008_baselines/weaksup_common/extract_audio.py --corpora hatemm hateclipseg dehate
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

WAV_ROOT = Path("/home/jehc223/Retrieval-hate/data/AV2A_wav")
SR = 16000
PIECE_S = 20
W2C_CKPT = C.REPO / ".cache/torch/hub/checkpoints/Wav2CLIP.pt"
W2C_PKG = C.REPO / "third_party/wav2clip_pkg"


def load_wav2clip(device):
    sys.path.insert(0, str(W2C_PKG))
    try:
        import librosa  # noqa: F401
    except ImportError:
        # wav2clip.model.encoder imports librosa only for its frame-splitting helper, which this script does not
        # call (framing is done below). A placeholder module keeps the import from failing.
        import types
        sys.modules["librosa"] = types.ModuleType("librosa")
    from wav2clip.model.encoder import ResNetExtractor
    ckpt = torch.load(W2C_CKPT, map_location="cpu")
    model = ResNetExtractor(checkpoint=ckpt, scenario="frozen", transform=True)
    return model.to(device).eval()


@torch.inference_mode()
def wav2clip_rows(model, audio, T, device, chunk=256):
    frames = torch.from_numpy(audio.reshape(T, SR)).to(device)
    enc = model.encoder
    specs = [torch.log(enc.spectrogram(frames[i:i + chunk].unsqueeze(1)) + 1e-7) for i in range(0, T, chunk)]
    allx = torch.cat(specs, 0)
    mean, std = torch.mean(allx), torch.std(allx)
    out = []
    for i in range(0, T, chunk):
        x = (allx[i:i + chunk] - mean) / (std + 1e-9)
        x = enc.maxpool(enc.relu(enc.bn1(enc.conv1(x))))
        x = enc.avgpool(enc.layer4(enc.layer3(enc.layer2(enc.layer1(x)))))
        out.append(model.transform(x.reshape(x.size(0), -1)))
    return torch.cat(out, 0).float().cpu().numpy()


@torch.inference_mode()
def wav2vec2_rows(model, audio, T, device):
    piece = PIECE_S * SR
    out = np.zeros((T, 768), dtype=np.float32)
    for start in range(0, T * SR, piece):
        x = audio[start:start + piece]
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)       # Wav2Vec2FeatureExtractor do_normalize
        h = model(torch.from_numpy(x).to(device)[None]).last_hidden_state[0].float().cpu().numpy()
        sec = np.arange(h.shape[0]) * 320 // SR
        n_sec = len(x) // SR
        base = start // SR
        for s in range(n_sec):
            m = sec == s
            if m.any():
                out[base + s] = h[m].mean(0)
            else:                                         # cannot happen for whole seconds; kept explicit
                out[base + s] = h[min(s * 50, h.shape[0] - 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=list(C.CORPORA))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(C.REPO / ".cache/hf"))
    from transformers import Wav2Vec2Model
    log = C.RunLog(C.RUNS / "audio_features" / "run.log")
    log(f"start corpora={args.corpora} code {C.code_version()}")
    device = args.device
    w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    w2c = load_wav2clip(device)
    report = {}
    for corpus in args.corpora:
        ds = C.DATASET[corpus]
        ids = C.split_ids(corpus, "train") + C.split_ids(corpus, "val") + C.split_ids(corpus, "test")
        d_v = C.REPO / "data/wav2vec2_base_1s" / corpus
        d_c = C.REPO / "data/wav2clip_1s" / corpus
        d_v.mkdir(parents=True, exist_ok=True)
        d_c.mkdir(parents=True, exist_ok=True)
        silent, done, t0 = [], 0, time.time()
        for k, vid in enumerate(ids):
            fv, fc = d_v / f"{vid}.npy", d_c / f"{vid}.npy"
            T = np.load(C.INPUTS / "clip_b16_1fps" / corpus / f"{vid}.npy", mmap_mode="r").shape[0]
            wav = WAV_ROOT / ds / f"{vid}.wav"
            if not wav.is_file():
                silent.append(vid)
            if fv.is_file() and fc.is_file():
                continue
            if wav.is_file():
                a, sr = sf.read(str(wav), dtype="float32", always_2d=True)
                if sr != SR:
                    raise RuntimeError(f"{wav}: sample rate {sr}")
                a = a.mean(1)
            else:
                a = np.zeros(0, dtype=np.float32)
            audio = np.zeros(T * SR, dtype=np.float32)
            n = min(len(a), T * SR)
            audio[:n] = a[:n]
            np.save(fv, wav2vec2_rows(w2v, audio, T, device))
            np.save(fc, wav2clip_rows(w2c, audio, T, device))
            done += 1
            if done % 200 == 0:
                log(f"{corpus}: {k + 1}/{len(ids)} ({time.time() - t0:.0f}s)")
        report[corpus] = {"n_ids": len(ids), "extracted_now": done, "no_wav_digital_silence": silent}
        log(f"{corpus}: {len(ids)} ids done ({done} extracted now, {time.time() - t0:.0f}s); "
            f"no wav -> digital silence: {silent}")
    for name, what in (("wav2vec2_base_1s", "facebook/wav2vec2-base last hidden layer, mean per second, (T, 768)"),
                       ("wav2clip_1s", "Wav2CLIP v0.1.0-alpha encoder + audio_transform on 1 s frames, (T, 512)")):
        prov = C.REPO / "data" / name / "PROVENANCE.md"
        prev = prov.read_text() if prov.is_file() else f"# data/{name} — provenance\n\n"
        prev += (f"\n## {datetime.date.today().isoformat()} on {socket.gethostname()}\n\n"
                 f"- generator: `experiments/20261008_baselines/weaksup_common/extract_audio.py` ({C.code_version()})\n"
                 f"- content: {what}; T rows = rows of `data/weaksup_1fps/clip_b16_1fps/<corpus>/<id>.npy`\n"
                 f"- input: 16 kHz mono wavs, `{WAV_ROOT}/<Dataset>/<id>.wav` (Retrieval-hate, read-only); "
                 f"ids = `data/weaksup_1fps/splits/<corpus>_{{train,val,test}}.txt`\n"
                 f"- per corpus: {json.dumps(report)}\n")
        prov.write_text(prev)
    log("DONE")


if __name__ == "__main__":
    main()
