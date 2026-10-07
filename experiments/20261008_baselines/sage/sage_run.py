#!/usr/bin/env python3
"""SAGE (ACL 2026, github.com/XinLiao04/SAGE @3545f6d) trained as published, scored per 8-s window.

    python sage_run.py prep  --dataset DS            # 16 kHz wav + 16 sampled frames of every train/val video
    python sage_run.py train --dataset DS --seed S   # train on train, select the epoch on val macro-F1
    python sage_run.py infer --dataset DS            # every test window of the cohort, all trained seeds

Model, loss, optimiser, schedule, encoders and hyper-parameters are the authors' (`third_party/SAGE/model/*.py`,
`config/config.yaml`), imported unchanged. What this file adds (see README.md):
- inputs for our corpora: the authors' 16-frame sampler (`preprocess/frame-extract.py` logic, frames JPEG-encoded
  at cv2's default quality and read back with torchvision `read_image` semantics), 16 kHz mono wav (config `sr`),
  Whisper large-v3 transcript text (HateMM recipe: transcript only);
- the frozen encoders are run batched in the collate step instead of per sample; each sample's arithmetic is the
  authors' (`VideoTransform`, `TextEncoder`, `AudioEncoder`);
- two bug fixes needed to train at all (`Trainer.evaluate` reads `batch_size` before assignment and is handed a
  DataLoader instead of a dataset); val and test use the eval branch of `VideoTransform` (the release passes the
  single config flag `train: True` to all three datasets);
- per-window inference: a window is fed as a short video (16 frames sampled by the same rule inside the window, the
  window's audio, the window's transcript).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "detwin"))
import common as C  # noqa: E402

SAGE_DIR = C.REPO / "third_party" / "SAGE"
sys.path.insert(0, str(SAGE_DIR / "model"))
METHOD = "SAGE_win8"
N_FRAMES = 16
FRAME_DIR = C.REPO / "data" / "sage_frames16"  # <ds>/<vid>.npy uint8 [16,3,224,224] (train/val videos)
WIN_DIR = C.REPO / "data" / "sage_frames16_win8"  # <ds>/<vid>.npy uint8 [n_win,16,3,224,224] + <vid>.json


def load_config() -> dict:
    cfg = yaml.safe_load((SAGE_DIR / "config" / "config.yaml").read_text())
    return cfg


# ---------------------------------------------------------------- sampling ---
def authors_pick(n: int, num_frames: int = N_FRAMES) -> list[int]:
    """`frame-extract.py` `sample_frames_uniform` on a stream of n readable frames: local indices picked."""
    if n <= 0:
        return []
    k = min(num_frames, n)
    interval = n / k
    picks, nxt = [], 0.0
    for i in range(n):
        if i + 1 >= nxt:
            picks.append(i)
            nxt += interval
        if len(picks) == k:
            break
    return picks


def decode_resized(jpeg_bytes: bytes) -> torch.Tensor:
    """read_image(path) of the saved JPEG, then the first op of VideoTransform: F.resize(frame, (224, 224)).
    Resizing a 224x224 frame again is the identity, so caching this tensor leaves the transform unchanged."""
    import torchvision.transforms.functional as F
    from torchvision.io import decode_jpeg
    t = decode_jpeg(torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8))
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    return F.resize(t, (224, 224))


def video_frames16(path: Path) -> tuple[np.ndarray | None, str]:
    """Whole-video 16 frames exactly as the authors: CAP_PROP_FRAME_COUNT frames, picks by the sampler; the authors'
    loader drops a video with fewer than 16 saved frames."""
    import cv2
    fps, n = C.probe_video(path)
    picks = authors_pick(n)
    if len(picks) < N_FRAMES:
        return None, f"frame_count {n} < 16"
    got, n_dec = C.read_frames(path, set(picks), transform=lambda f: decode_resized(C.jpeg_roundtrip(f)))
    if n_dec <= max(picks):
        return None, f"stream ended at {n_dec} before frame {max(picks)}"
    arr = np.stack([got[i].numpy() for i in picks])
    return arr, "ok"


def window_frame_plan(fps: float, n_est: int, duration: float):
    """Per window: absolute frame indices (frames with timestamp i / fps inside [a, b)), then the authors' picks."""
    plan = []
    for a, b in C.windows(duration):
        i0 = int(math.ceil(a * fps - 1e-9))
        i1 = min(int(math.ceil(b * fps - 1e-9)), n_est)
        idx = list(range(i0, max(i0, i1)))
        plan.append([idx[j] for j in authors_pick(len(idx))])
    return plan


# -------------------------------------------------------------------- prep ---
def _prep_one(task):
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    ds, vid, split = task
    out = FRAME_DIR / ds / f"{vid}.npy"
    res = {"video_id": vid, "split": split}
    mp4 = C.video_path(ds, vid, split)
    if mp4 is None:
        res["status"] = "no_video"
        return res
    wav = C.wav_path(ds, vid)
    if not wav.is_file():
        err = C.extract_wav(mp4, wav)
        res["wav_error"] = err
    if split in ("train", "val") and not out.is_file():
        arr, why = video_frames16(mp4)
        if arr is None:
            res["status"] = why
            return res
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, arr)
    if split == "test":
        wout = WIN_DIR / ds / f"{vid}.npy"
        if not wout.is_file():
            res.update(window_frames(ds, vid, mp4, wout))
    res["status"] = "ok"
    return res


def window_frames(ds: str, vid: str, mp4: Path, wout: Path) -> dict:
    """Test video: the authors' 16-frame sampler inside each 8-s window. Windows with fewer than 16 readable frames
    (a last window shorter than ~16 frames) are flagged and get the F1 tail rule at inference."""
    dur = {x["video_id"]: float(x["duration"]) for x in C.load_split(ds)["test"]}[vid]
    fps, n_cap = C.probe_video(mp4)
    n_est = n_cap if n_cap > 0 else int(math.ceil(dur * fps))
    plan = window_frame_plan(fps, n_est, dur)
    got, n_dec = C.read_frames(mp4, {i for p in plan for i in p},
                               transform=lambda f: decode_resized(C.jpeg_roundtrip(f)).numpy())
    ok = [len(p) == N_FRAMES for p in plan]
    arr = np.zeros((len(plan), N_FRAMES, 3, 224, 224), dtype=np.uint8)
    for j, p in enumerate(plan):
        if ok[j]:
            arr[j] = np.stack([got[i] for i in p])
    meta = {"video_id": vid, "duration": dur, "fps": fps, "n_frames_cap": n_cap, "n_decoded": n_dec,
            "n_windows": len(plan), "window_ok": ok, "picks": plan,
            "clamped_frame_picks": sum(1 for p in plan for i in p if i >= n_dec)}
    wout.parent.mkdir(parents=True, exist_ok=True)
    np.save(wout, arr)
    wout.with_suffix(".json").write_text(json.dumps(meta) + "\n")
    return {"n_windows": len(plan), "clamped": meta["clamped_frame_picks"]}


def cmd_prep(args):
    log = C.RunLog(C.RUNS / "sage" / args.dataset / "prep" / "run.log")
    sp = C.load_split(args.dataset)
    tasks = [(args.dataset, x["video_id"], s) for s in ("train", "val") for x in sp[s]]
    tasks += [(args.dataset, x["video_id"], "test") for x in sp["test"]]
    log(f"prep {args.dataset}: {len(tasks)} videos, {args.workers} workers")
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_prep_one, tasks, chunksize=2)):
            rows.append(r)
            if (k + 1) % 200 == 0:
                log(f"  {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    status = {}
    for r in rows:
        status.setdefault(r["status"], []).append(r["video_id"])
    wav_err = {r["video_id"]: r["wav_error"] for r in rows if r.get("wav_error")}
    rep = {"counts": {k: len(v) for k, v in status.items()},
           "not_ok": {k: v for k, v in status.items() if k != "ok"}, "wav_errors": wav_err,
           "seconds": round(time.time() - t0, 1)}
    (C.RUNS / "sage" / args.dataset / "prep" / "prep_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    log(f"prep done: {rep['counts']}; wav errors {len(wav_err)}")
    missing_test = [r["video_id"] for r in rows if r["split"] == "test" and r["status"] == "no_video"]
    if missing_test:
        log(f"FAILED: test videos without media: {missing_test}")
        sys.exit(1)


# ---------------------------------------------------------------- encoders ---
class Encoders:
    """The authors' frozen encoders (CustomDataset.py), applied to a batch."""

    def __init__(self, cfg, device):
        from CustomDataset import TextEncoder, AudioEncoder, VideoTransform
        from transformers import VideoMAEModel
        cfg = dict(cfg)
        cfg["device"] = device
        self.cfg = cfg
        self.device = device
        self.text = TextEncoder(cfg)
        self.audio = AudioEncoder(cfg)
        self.vmae = VideoMAEModel.from_pretrained(cfg["vision"]["model_name"]).eval().to(device)
        self.tf_train = VideoTransform(cfg["vision"], True)
        self.tf_eval = VideoTransform(cfg["vision"], False)

    @torch.no_grad()
    def vision(self, frame_stacks: list[torch.Tensor], train: bool) -> torch.Tensor:
        """frame_stacks: list of uint8 [16,3,224,224]; per sample VideoTransform (list of frames), batched VideoMAE."""
        tf = self.tf_train if train else self.tf_eval
        x = torch.stack([tf([f for f in st]) for st in frame_stacks]).to(self.device)
        return self.vmae(x).last_hidden_state  # [B,1568,768]

    @torch.no_grad()
    def texts(self, texts: list[str]) -> torch.Tensor:
        enc = self.text.tokenizer(texts, truncation=True, padding="max_length", max_length=self.text.max_length,
                                  return_tensors="pt").to(self.device)
        return self.text.model(**enc).last_hidden_state  # [B,128,768]

    @torch.no_grad()
    def audio_from_path(self, path: str) -> torch.Tensor:
        return self.audio(path)  # authors' AudioEncoder.forward (load, mono, MFCC, cut/pad to 2048)

    @torch.no_grad()
    def audio_from_wave(self, wave: np.ndarray | None) -> torch.Tensor:
        """AudioEncoder.forward on an in-memory 16 kHz mono waveform (a window's samples)."""
        a = self.audio
        if wave is None or len(wave) == 0:
            return torch.zeros((a.max_audio_len, a.cfg["n_mfcc"]), dtype=torch.float32, device=self.device)
        w = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).unsqueeze(0).to(self.device)
        feat = a.mfcc_transform(w).squeeze(0).T
        L = feat.size(0)
        if L > a.max_audio_len:
            feat = feat[:a.max_audio_len]
        elif L < a.max_audio_len:
            feat = torch.cat([feat, torch.zeros((a.max_audio_len - L, feat.size(1)), device=feat.device,
                                                dtype=feat.dtype)], 0)
        return feat


# ------------------------------------------------------------------- train ---
def no_attention_weights(model):
    """GED calls nn.MultiheadAttention with the default need_weights=True and discards the weights. With 3744 keys
    and batch 16 the materialised weights do not fit a 32 GB GPU (OOM, job 293). need_weights=False computes the same
    attention output (PyTorch's fused path, same dropout) without returning the discarded weights."""
    import functools
    for mod in model.modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.forward = functools.partial(mod.forward, need_weights=False)
    return model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_val_items(ds: str, split: str, log) -> list[dict]:
    text = C.load_asr_text(ds)
    items, dropped = [], []
    for x in C.samples(ds, split):
        v = x["video_id"]
        fr, wav = FRAME_DIR / ds / f"{v}.npy", C.wav_path(ds, v)
        # authors' load_split_*: keep a video only if all 16 frames and the audio file exist
        if fr.is_file() and wav.is_file():
            items.append({"video_id": v, "label": int(x["label"]), "frames": fr, "wav": str(wav),
                          "text": text.get(v, "")})
        else:
            dropped.append(v)
    log(f"{split}: {len(items)} usable, {len(dropped)} dropped (no 16 frames or no audio file): {dropped[:20]}")
    return items


def batches(items, bs, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i + bs]]


class _Prefetch(torch.utils.data.Dataset):
    """CPU part of the authors' CustomDataset.__getitem__ (frames + VideoTransform, torchaudio.load of the wav),
    run in DataLoader workers so the GPU part does not wait for disk."""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torchaudio
        b = self.items[i]
        st = torch.from_numpy(np.load(b["frames"]))
        frames = self.tf([f for f in st])
        try:
            w, _ = torchaudio.load(b["wav"])
            if w.shape[0] > 1:
                w = torch.mean(w, dim=0, keepdim=True)
        except Exception:  # noqa: BLE001 - AudioEncoder's failure path gives zeros
            w = None
        return frames, w, b["text"], b["label"]


def loader(items, bs, shuffle, seed, tf, workers=3):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(_Prefetch(items, tf), batch_size=bs, shuffle=shuffle, generator=g,
                                       num_workers=workers, collate_fn=lambda x: list(zip(*x)),
                                       persistent_workers=False, prefetch_factor=4)


def encode_prefetched(enc: Encoders, batch):
    frames, waves, texts, labels = batch
    with torch.no_grad():
        v = enc.vmae(torch.stack(frames).to(enc.device)).last_hidden_state
    t = enc.texts(list(texts))
    a = torch.stack([enc.audio_from_wave(None if w is None else w[0].numpy()) for w in waves]).to(enc.device)
    y = torch.tensor(list(labels), dtype=torch.long, device=enc.device)
    return t, a, v, y


def encode_batch(enc: Encoders, batch, train: bool):
    frames = [torch.from_numpy(np.load(b["frames"])) for b in batch]
    v = enc.vision(frames, train)
    t = enc.texts([b["text"] for b in batch])
    a = torch.stack([enc.audio_from_path(b["wav"]) for b in batch])
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long, device=enc.device)
    a = a.to(enc.device)
    return t, a, v, y


@torch.no_grad()
def evaluate(model, enc, items, bs):
    """Trainer.evaluate: argmax of logits_fusion; macro-F1 (the selection metric), accuracy."""
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader(items, bs, False, 0, enc.tf_eval):
        t, a, v, y = encode_prefetched(enc, batch)
        _, _, _, logits, _, _ = model(t, a, v)
        preds += logits.argmax(1).tolist()
        labels += y.tolist()
        probs += torch.softmax(logits.float(), 1)[:, 1].tolist()
    return {"acc": float(accuracy_score(labels, preds)), "macro_f1": float(f1_score(labels, preds, average="macro")),
            "n": len(labels)}


def cmd_train(args):
    from Sage import SAGE
    from utils import get_training_setting
    out = C.RUNS / "sage" / args.dataset / (f"smoke_seed{args.seed}" if args.smoke else f"seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(out / "run.log")
    cfg = load_config()
    cfg["device"] = args.device
    if args.smoke:  # debug only: tiny subset, 1 epoch, separate output dir
        cfg["train"]["epochs"] = 1
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg))
    set_seed(args.seed)
    log(f"train SAGE {args.dataset} seed {args.seed}; code {C.code_version()}")
    train = train_val_items(args.dataset, "train", log)
    val = train_val_items(args.dataset, "val", log)
    if args.smoke:
        train, val = train[:4], val[:4]
    enc = Encoders(cfg, args.device)
    model = no_attention_weights(SAGE(cfg["model"]).to(args.device))
    _, _, _, _, optimizer, criterion, scheduler = get_training_setting(cfg, model)
    tc = cfg["train"]
    bs, epochs = int(tc["batch_size"]), int(tc["epochs"])
    rng = random.Random(args.seed)
    best, hist = 0.0, []
    ckpt = out / "sage_best_model.pth"
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in loader(train, bs, True, args.seed * 1000 + ep, enc.tf_train):
            t, a, v, y = encode_prefetched(enc, batch)
            optimizer.zero_grad()
            lt, la, lv, lf, gates, gmask = model(t, a, v)
            loss, _ = criterion(lt, la, lv, lf, gates, gmask, y)
            loss.backward()
            optimizer.step()
            tot += loss.item()
            n += 1
        m = evaluate(model, enc, val, bs)
        rec = {"epoch": ep + 1, "train_loss": tot / max(1, n), "val_macro_f1": m["macro_f1"], "val_acc": m["acc"],
               "lr": optimizer.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        if m["macro_f1"] > best:  # Trainer.train_model: strictly better val macro-F1 -> save
            best = m["macro_f1"]
            torch.save({"epoch": ep + 1, "model_state_dict": model.state_dict(), "best_f1": best}, ckpt)
            rec["saved"] = True
        log(json.dumps(rec))
        scheduler.step()
    (out / "train_history.json").write_text(json.dumps(hist, indent=1) + "\n")
    log(f"DONE train: best val macro-F1 {best:.4f} at epoch {max(hist, key=lambda r: r['val_macro_f1'])['epoch']}")


# ------------------------------------------------------------------- infer ---
def cmd_infer(args):
    from Sage import SAGE
    ds = args.dataset
    out = C.RUNS / "sage" / ds / "infer"
    out.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(out / "run.log")
    cfg = load_config()
    cfg["device"] = args.device
    seeds = [s for s in C.SEEDS if (C.RUNS / "sage" / ds / f"seed{s}" / "sage_best_model.pth").is_file()]
    log(f"infer SAGE {ds}, seeds {seeds}; code {C.code_version()}")
    enc = Encoders(cfg, args.device)
    models = {}
    for s in seeds:
        m = no_attention_weights(SAGE(cfg["model"]).to(args.device))
        ck = torch.load(C.RUNS / "sage" / ds / f"seed{s}" / "sage_best_model.pth", map_location=args.device,
                        weights_only=False)  # own checkpoint
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        models[s] = m
    test = C.load_split(ds)["test"]
    dur = {x["video_id"]: float(x["duration"]) for x in test}
    segs = C.load_asr(ds, dur)
    part = out / "window_scores.jsonl"
    done = {}
    if part.is_file():
        for line in part.open():
            r = json.loads(line)
            done[r["video_id"]] = r
    fh = part.open("a")
    t0 = time.time()
    for k, x in enumerate(test):
        v = x["video_id"]
        if v in done:
            continue
        meta = json.loads((WIN_DIR / ds / f"{v}.json").read_text())
        warr = np.load(WIN_DIR / ds / f"{v}.npy", mmap_mode="r")
        wave = C.load_wav(C.wav_path(ds, v))
        wins = C.windows(dur[v])
        ok_w = [j for j, okj in enumerate(meta["window_ok"]) if okj]
        scores = {s: [None] * len(wins) for s in seeds}
        for i0 in range(0, len(ok_w), args.batch):
            js = ok_w[i0:i0 + args.batch]
            frames = [torch.from_numpy(np.array(warr[j])) for j in js]
            vfe = enc.vision(frames, train=False)
            tfe = enc.texts([C.window_text(segs.get(v, []), *wins[j]) for j in js])
            afe = torch.stack([enc.audio_from_wave(None if wave is None else
                                                   wave[int(round(wins[j][0] * 16000)):int(round(wins[j][1] * 16000))])
                               for j in js])
            with torch.no_grad():
                for s, m in models.items():
                    _, _, _, lf, _, _ = m(tfe, afe, vfe)
                    p = torch.softmax(lf.float(), 1)[:, 1].tolist()
                    for j, pj in zip(js, p):
                        scores[s][j] = pj
        # F1 tail rule: a window with fewer than 16 readable frames (only a last window shorter than ~0.6 s) takes
        # the previous window's value; a video with no scorable window is left to F2.
        f1 = [j for j in range(len(wins)) if j not in ok_w]
        for s in seeds:
            for j in range(len(wins)):
                if scores[s][j] is None and j > 0 and scores[s][j - 1] is not None:
                    scores[s][j] = scores[s][j - 1]
        row = {"video_id": v, "n_windows": len(wins), "fps": meta["fps"], "n_frames_cap": meta["n_frames_cap"],
               "n_decoded": meta["n_decoded"], "clamped_frame_picks": meta["clamped_frame_picks"], "f1_windows": f1,
               "has_wav": wave is not None,
               "scores": {str(s): scores[s] for s in seeds}}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        done[v] = row
        if (k + 1) % 20 == 0:
            log(f"  {k + 1}/{len(test)} videos ({time.time() - t0:.0f}s)")
    fh.close()
    for s in seeds:
        ws, f1w, fails = {}, {}, {}
        for v, r in done.items():
            sc = r["scores"][str(s)]
            if any(z is None for z in sc):
                fails[v] = "unscored windows"
            else:
                ws[v] = sc
                f1w[v] = r["f1_windows"]
        rd = C.RUNS / "sage" / ds / f"seed{s}"
        rl = C.RunLog(rd / "run.log")
        C.finalize(METHOD, ds, s, rd, ws, code_path="experiments/20261008_baselines/sage/sage_run.py",
                   config={"window_seconds": C.WIN, "checkpoint": str(rd / "sage_best_model.pth"),
                           "sage_commit": "3545f6d", "inputs": "16 frames/window (authors' sampler), window "
                           "audio MFCC, window transcript (Whisper large-v3, proportional word slicing)"},
                   log=rl, f1_windows=f1w, failures=fails)
    log("DONE infer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "infer"])
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="debug: 4 train / 4 val videos, 1 epoch")
    args = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    main()
