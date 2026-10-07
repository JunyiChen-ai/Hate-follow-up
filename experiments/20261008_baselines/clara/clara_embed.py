#!/usr/bin/env python3
"""CLARA step 4 (`extract_video_emb.py`), GPU, torch venv (.cache/envs/hv_cv2).

    python clara_embed.py --dataset DS

One `.pt` per sample in the authors' flat format, for the configuration the model trains on (run_clara.sh:
BUDGET 40, TEXT_EMB_MODEL bert, RAT_SOURCE Qwen):
`whisper` [T,1280] (Whisper large-v3 encoder, mean over the 30-s padded clip), `vit_40` [T,768] (ViT-B/16 in21k CLS
mean over the clip's frames), `text_bert` / `ocr_bert_40` [T,768] (BERT CLS of the clip transcript / mean over OCR
items), `rationale_Qwen_bert` [8,768], and their masks. The authors' encoding functions are imported unchanged;
Whisper clips of one sample go through the encoder in batches (same per-clip arithmetic). The Qwen3-Embedding text
variants are not computed (not used by the bert configuration).

Samples: a train / val video = the whole video (as published; a video without a wav, clips or rationale is skipped,
as the authors' extractor does). A test window `<vid>__wJJJ` = the window's clips (`clara_prep.py`), audio sliced
from the video's wav at the window's absolute times, and the video's rationale (shared by all windows).
Empty-input rule for test windows (run_plan.md §1.3 F1): a video without an audio stream gets digital silence; a
video whose rationale failed gets the all-blank rationale (zero rows, mask off).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "detwin"))
import common as C  # noqa: E402

DATA = Path(os.environ.get("CLARA_DATA", C.REPO / "data"))  # env override: smoke tests only

RAW = DATA / "clara_raw"
RAT = DATA / "clara_rationale"
EMB = DATA / "clara_emb"


def load_authors():
    p = C.REPO / "third_party/CLARA/data_preprocess/extract_video_emb.py"
    spec = importlib.util.spec_from_file_location("clara_extract_video_emb", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame_key(p: Path) -> int:
    m = re.findall(r"\d+", p.stem)
    return int(m[0]) if m else 0


@torch.inference_mode()
def whisper_batch(E, proc, model, clips_audio, device, bs=16):
    """`whisper_clip_meanpool` for several clips at once (padding to 30 s per clip, mean over encoder frames)."""
    out = []
    for i in range(0, len(clips_audio), bs):
        xs = [a.detach().cpu().float().numpy() for a in clips_audio[i:i + bs]]
        feats = proc(xs, sampling_rate=16000, return_tensors="pt", padding="max_length", truncation=True,
                     max_length=30 * 16000)
        enc = model.encoder(input_features=feats.input_features.to(device)).last_hidden_state
        out.append(enc.detach().cpu().float().mean(dim=1))
    return torch.cat(out, 0) if out else torch.zeros((0, 1280))


class Models:
    def __init__(self, E, device):
        from transformers import WhisperProcessor, WhisperModel, ViTImageProcessor, ViTModel, BertTokenizer, BertModel
        self.E, self.device = E, device
        self.wproc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-large-v3").to(device).eval()
        self.vproc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
        self.btok = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device).eval()

    def rationale(self, rjson: Path):
        E = self.E
        try:
            r = json.loads(rjson.read_text())
            if "error" in r:
                raise ValueError(r["error"])
            texts8 = E.extract_text_fields_8(r)
        except Exception as e:  # noqa: BLE001
            return None, None, repr(e)
        mask8 = torch.tensor([not E.is_blank_text(s) for s in texts8], dtype=torch.bool)
        rb = E.bert_encode_cls_batch(self.btok, self.bert, texts8, self.device)
        rb = E.apply_blank_mask_and_zero_rows(rb, mask8)
        return E.cast_fp16(rb), mask8, None

    def sample(self, clips, frame_dirs, wav, sr, offset, rat, rat_mask):
        """clips: clip dicts (times relative to offset); frame_dirs[i]: the clip's frame folder."""
        E = self.E
        T = len(clips)
        whisper_out = torch.zeros((T, 1280), dtype=E.OUT_DTYPE)
        whisper_mask = torch.zeros((T,), dtype=torch.bool)
        vit_out = torch.zeros((T, 768), dtype=E.OUT_DTYPE)
        vit_mask = torch.zeros((T,), dtype=torch.bool)
        text_out = torch.zeros((T, 768), dtype=E.OUT_DTYPE)
        text_mask = torch.zeros((T,), dtype=torch.bool)
        ocr_out = torch.zeros((T, 768), dtype=E.OUT_DTYPE)
        ocr_mask = torch.zeros((T,), dtype=torch.bool)
        audios, a_idx = [], []
        for i, c in enumerate(clips):
            a = E.slice_audio(wav, sr, offset + float(c["start"]), offset + float(c["end"]))
            if a.numel() > 0:
                audios.append(a)
                a_idx.append(i)
        if audios:
            w = whisper_batch(E, self.wproc, self.whisper, audios, self.device)
            for j, i in enumerate(a_idx):
                whisper_out[i] = E.cast_fp16(w[j])
                whisper_mask[i] = True
        for i, c in enumerate(clips):
            fdir = frame_dirs[i]
            frames = sorted(fdir.glob("frame_*.jpg"), key=frame_key) if fdir.is_dir() else []
            if frames:
                v = E.vit_clip_meanpool(self.vit, self.vproc, frames, self.device, batch_size=32)
                vit_out[i] = E.cast_fp16(v)
                vit_mask[i] = True
            has_text = E.transcript_exists_from_clip(c)
            text_mask[i] = bool(has_text)
            if has_text:
                text_out[i] = E.cast_fp16(E.bert_encode_one_cls(self.btok, self.bert, str(c.get("transcript", "")).strip(),
                                                                self.device))
            ocr_json = fdir.parent.parent / "ocr_text" / "ocr_clip.json"
            items = []
            if ocr_json.is_file():
                items = [w.strip() for w in json.loads(ocr_json.read_text()).get("clip_words_keep", []) if w.strip()]
            elif frames:
                raise FileNotFoundError(f"OCR missing for {fdir}")
            ocr_mask[i] = bool(items)
            if items:
                mat = E.bert_encode_cls_batch(self.btok, self.bert, items, self.device)
                ocr_out[i] = E.cast_fp16(mat.mean(dim=0))
        return {"num_clips": T, "whisper": whisper_out, "whisper_mask": whisper_mask, "text_mask": text_mask,
                "vit_40": vit_out, "vit_mask_40": vit_mask, "text_bert": text_out, "ocr_mask_40": ocr_mask,
                "ocr_bert_40": ocr_out, "rationale_Qwen_bert": rat, "rationale_mask_Qwen": rat_mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ds = args.dataset
    log = C.RunLog(C.RUNS / "clara" / ds / "embed" / "run.log")
    E = load_authors()
    device = torch.device(args.device)
    M = Models(E, device)
    sp = C.load_split(ds)
    out_dir = EMB / ds / "video_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"embed {ds}; code {C.code_version()}")
    t0 = time.time()
    split_out = {"train": [], "valid": [], "test": []}
    skipped, notes = {}, {}
    zero_rat = (torch.zeros((8, 768), dtype=E.OUT_DTYPE), torch.zeros((8,), dtype=torch.bool))
    for split, key in (("train", "train"), ("val", "valid"), ("test", "test")):
        rows = sp[split]
        for k, x in enumerate(rows):
            v = x["video_id"]
            vdir = RAW / ds / v
            wavp = C.wav_path(ds, v)
            rat, rat_mask, rerr = M.rationale(RAT / ds / "Qwen" / f"{v}_rationale.json")
            if split != "test":
                pt = out_dir / f"{v}.pt"
                if not pt.is_file():
                    ci = json.loads((vdir / "clipinfo.json").read_text()) if (vdir / "clipinfo.json").is_file() else {}
                    clips = ci.get("clips", [])
                    if not wavp.is_file() or not clips or rerr:
                        skipped[v] = "no wav" if not wavp.is_file() else ("no clips" if not clips else f"rationale {rerr}")
                        continue
                    wav, sr = E.wav_load_mono_16k(wavp, 16000)
                    fdirs = [vdir / f"clip_{int(c['clip_idx']):03d}" / "frames" / "frame_40" for c in clips]
                    pack = M.sample(clips, fdirs, wav, sr, 0.0, rat, rat_mask)
                    torch.save({"video_id": v, **pack}, pt)
                split_out[key].append({"video_id": v, "label": int(x["label"]), "note": "na"})
            else:
                if rerr:
                    notes.setdefault("rationale_blank_F1", []).append(v)
                    rat, rat_mask = zero_rat
                if wavp.is_file():
                    wav, sr = E.wav_load_mono_16k(wavp, 16000)
                else:
                    notes.setdefault("silent_audio_F1", []).append(v)
                    wav, sr = torch.zeros(int(round(float(x["duration"]) * 16000)) + 1), 16000
                for j, _ in enumerate(C.windows(float(x["duration"]))):
                    sid = f"{v}__w{j:03d}"
                    pt = out_dir / f"{sid}.pt"
                    if not pt.is_file():
                        wdir = vdir / f"win_{j:03d}"
                        ci = json.loads((wdir / "clipinfo.json").read_text())
                        clips = ci["clips"]
                        fdirs = [wdir / f"clip_{int(c['clip_idx']):03d}" / "frames" / "frame_w" for c in clips]
                        pack = M.sample(clips, fdirs, wav, sr, float(ci["offset"]), rat, rat_mask)
                        torch.save({"video_id": sid, **pack}, pt)
                    # test labels are never read: placeholder 0, ignored downstream
                    split_out[key].append({"video_id": sid, "label": 0, "note": "test_window"})
            if (k + 1) % 100 == 0:
                log(f"  {split} {k + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    (EMB / ds / "split.json").write_text(json.dumps({"dataset": ds, "n_splits": 1, "folds": [split_out]}) + "\n")
    rep = {"n": {k: len(v) for k, v in split_out.items()}, "skipped_train_val": skipped, "test_notes": notes,
           "seconds": round(time.time() - t0, 1)}
    (C.RUNS / "clara" / ds / "embed" / "embed_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    log(f"DONE embed: {rep['n']}; skipped {len(skipped)}; notes { {k: len(v) for k, v in notes.items()} }")


if __name__ == "__main__":
    main()
