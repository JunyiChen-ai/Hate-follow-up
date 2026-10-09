#!/usr/bin/env python3
"""ZS-ImageBind-AV: zero-shot ImageBind on three modalities, vision + audio + speech transcript (user decision
2026-10-09). The audio-only run (`zs_imagebind_audio.py`, runs/20261008_baselines/zs_imagebind_audio/) is unchanged.

Every modality is scored against the same text pair as the audio-only run, ImageBind text embeddings of
["normal", "hateful"] (the campaign's `imagebind_text_normal_hateful.npy`; the text postprocessor carries the
learned logit scale, |t| = 100): p = softmax over (normal, hateful) of unit(e) @ t.T, take p(hateful)
(`zs_imagebind_audio.scores`, = Retrieval-hate `eval_frame.imagebind_curves`).

  vision      per-frame ImageBind VISION embeddings at 4 fps, the Retrieval-hate campaign's `image` channel
              (`extract_imagebind.py`: ffmpeg fps=4, short side 224 bicubic + centre crop 224, rgb24, /255, CLIP
              mean/std, ImageBind-Huge float16, batch 16, stored float16).
                HateMM / HateClipSeg: the campaign's cache (uoa-lab2 `data/CLIP_Embedding/<DS>/imagebind_image/`),
                  cohort files copied to data/retrieval_hate_repro/imagebind_image/<DS>/.
                DeHate: extracted with the same pipeline by the `extract-image` stage below (all 1341 test videos)
                  -> data/imagebind_image_4fps/DeHate/.
              4 fps frame i <- embedding i; frames past the last decoded frame hold its value.
  audio       the existing 2 s clip AUDIO embeddings of the audio-only run (HateMM / HateClipSeg:
              data/retrieval_hate_repro/imagebind_audio/<DS>/, DeHate: data/imagebind_audio_2s/DeHate/); frame i <-
              clip floor(i / 8); frames past the last clip hold its value. Same p as the audio-only run.
  transcript  ImageBind TEXT embedding of the Whisper transcript of the 8 s window containing the frame (windows
              [8w, 8w + 8) s, frame i -> window floor(i / 32)), text from `transcript_windows.Transcripts`. CLIP BPE
              (ImageBind's SimpleTokenizer) with OpenAI-CLIP truncation to 77 tokens: start token + the first 75 text
              tokens + end token (ImageBind's own tokenizer would cut the end token of a long text, and its EOS
              pooling then reads an arbitrary token). A window without speech has no transcript modality.
  score       mean of the probabilities of the modalities available at the frame (vision, audio always; transcript
              only in a window with speech). No corpus statistic, no label.

Stages
  verify-image   GPU: re-extract the image channel of 5 HateMM cohort videos, compare with the campaign cache
  extract-image  GPU: DeHate test videos (manifest, 1341) -> --out-dir (default data/imagebind_image_4fps/DeHate)
  text           CPU or GPU: transcript-window embeddings -> p(hateful) per window, all three corpora
                 (runs/20261008_baselines/zs_imagebind_av/<DS>/transcript_windows.jsonl)
  score          CPU: combine + exact-cohort finalisation -> runs/20261008_baselines/zs_imagebind_av/<DS>/
                 (`--modalities vision` or `vision,audio`: comparison rows -> <DS>/modalities_<set>/)
No labels are read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402
from transcript_windows import Transcripts  # noqa: E402

CODE_PATH = "experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_av.py"
IB_DIR = REPO / "third_party/lavad/libs/ImageBind"
sys.path.insert(0, str(IB_DIR))
IB_CKPT = REPO / "data/assets/imagebind/imagebind_huge.pth"
BPE = IB_DIR / "bpe/bpe_simple_vocab_16e6.txt.gz"
TEXT_EMB = REPO / "data/retrieval_hate_repro/imagebind_audio/imagebind_text_normal_hateful.npy"
IMAGE_DIR = {"HateMM": REPO / "data/retrieval_hate_repro/imagebind_image/HateMM",
             "HateClipSeg": REPO / "data/retrieval_hate_repro/imagebind_image/HateClipSeg",
             "DeHate": REPO / "data/imagebind_image_4fps/DeHate"}
AUDIO_DIR = {"HateMM": REPO / "data/retrieval_hate_repro/imagebind_audio/HateMM",
             "HateClipSeg": REPO / "data/retrieval_hate_repro/imagebind_audio/HateClipSeg",
             "DeHate": REPO / "data/imagebind_audio_2s/DeHate"}
OUT_ROOT = ec.OUT_ROOT / "zs_imagebind_av"
DATASETS = ("HateMM", "HateClipSeg", "DeHate")

FPS = 4.0
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
AUDIO_RATE = 0.5
WIN_S = 8.0
CONTEXT = 77


# ------------------------------------------------------------------ scoring ---
def p_hateful(e: np.ndarray, text: np.ndarray) -> np.ndarray:
    """zs_imagebind_audio.scores (Retrieval-hate eval_frame.imagebind_curves), for any modality."""
    e = np.asarray(e, dtype=np.float32)
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
    logits = e @ text.T
    m = logits.max(axis=1, keepdims=True)
    q = np.exp(logits - m)
    return (q[:, 1] / q.sum(axis=1)).astype(np.float64)


# ------------------------------------------------------------------ model ---
def load_model(device: str, dtype_name: str):
    import torch
    from imagebind.models import imagebind_model
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(IB_CKPT, map_location="cpu"))
    return model.to(device, dtype=getattr(torch, dtype_name)).eval()


def decode_frames(path: Path):
    """Retrieval-hate extract_imagebind.decode_frames, unchanged: all frames at 4 fps, short side 224 bicubic +
    centre crop 224, uint8 RGB."""
    vf = (f"fps={FPS:g},scale=w={SIZE}:h={SIZE}:force_original_aspect_ratio=increase:"
          f"flags=bicubic,crop={SIZE}:{SIZE}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True)
    nb = SIZE * SIZE * 3
    if not p.stdout or len(p.stdout) < nb:
        return None
    n = len(p.stdout) // nb
    return np.frombuffer(p.stdout[: n * nb], dtype=np.uint8).reshape(n, SIZE, SIZE, 3)


class VisionEncoder:
    def __init__(self, batch: int):
        import torch
        from imagebind.models.imagebind_model import ModalityType
        self.torch = torch
        self.model = load_model("cuda", "float16")
        self.MT = ModalityType
        self.mean = torch.tensor(MEAN, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
        self.batch = batch

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch):
                x = torch.from_numpy(np.ascontiguousarray(frames[i:i + self.batch])).to("cuda")
                x = x.permute(0, 3, 1, 2).to(torch.float16).div_(255.0)
                x = (x - self.mean) / self.std
                e = self.model({self.MT.VISION: x})[self.MT.VISION]
                out.append(e.float().cpu().numpy())
        return np.concatenate(out, 0).astype(np.float16)


def find_video(root: Path, vid: str):
    for ext in (".mp4", ".webm", ".mkv", ".avi"):
        p = root / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def atomic_save(d: Path, vid: str, arr: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{vid}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, d / f"{vid}.npy")


def stage_verify_image(a, log) -> int:
    enc = VisionEncoder(a.batch)
    text = np.load(TEXT_EMB)
    root = Path(a.video_root)
    res, ok = {}, True
    for v in ec.cohort("HateMM")[:5]:
        p, ref_p = find_video(root, v), IMAGE_DIR["HateMM"] / f"{v}.npy"
        if p is None or not ref_p.exists():
            res[v] = {"skipped": "no video or no campaign embedding"}
            continue
        mine = enc(decode_frames(p)).astype(np.float32)
        ref = np.load(ref_p).astype(np.float32)
        n = min(len(mine), len(ref))
        cos = np.sum(mine[:n] * ref[:n], 1) / (np.linalg.norm(mine[:n], axis=1) * np.linalg.norm(ref[:n], axis=1))
        res[v] = {"T_mine": len(mine), "T_campaign": len(ref), "mean_cos": float(cos.mean()),
                  "min_cos": float(cos.min()),
                  "max_abs_p_diff": float(np.abs(p_hateful(mine[:n], text) - p_hateful(ref[:n], text)).max())}
        ok &= len(mine) == len(ref) and float(cos.mean()) > 0.99
        log(f"VERIFY {v} {json.dumps(res[v])}")
    ok &= sum("mean_cos" in r for r in res.values()) >= 3
    out = OUT_ROOT / "verify_image.json"
    out.write_text(json.dumps({"ok": bool(ok), "host": os.uname().nodename, "videos": res}, indent=2) + "\n")
    log(f"VERIFY {'PASSED' if ok else 'FAILED'} (same frame count and mean cosine > 0.99 on HateMM cohort videos)")
    return 0 if ok else 4


def stage_extract_image(a, log) -> int:
    enc = VisionEncoder(a.batch)
    out_dir = Path(a.out_dir)
    root = Path(a.video_root)
    man = ec.manifest_rows("DeHate")
    todo = [v for v in sorted(man) if not (out_dir / f"{v}.npy").exists()]
    log(f"extract-image: {len(todo)} of {len(man)} DeHate test videos to do -> {out_dir}")
    stats = {"n": 0, "missing_file": [], "no_frames": [], "oom": []}
    t0 = time.time()

    def load(v):
        p = find_video(root, v)
        return v, p, (decode_frames(p) if p is not None else None)

    def prefetched(pool):
        """Decoded videos in order, at most 2 x decoders ahead of the GPU (bounded memory)."""
        q, it = [], iter(todo)
        for v in it:
            q.append(pool.submit(load, v))
            if len(q) >= 2 * a.decoders:
                break
        while q:
            r = q.pop(0).result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(pool.submit(load, nxt))
            yield r

    with ThreadPoolExecutor(max_workers=a.decoders) as pool:
        for k, (v, p, frames) in enumerate(prefetched(pool), 1):
            if p is None:
                stats["missing_file"].append(v)
                log(f"[MISS] {v}")
                continue
            if frames is None:
                stats["no_frames"].append(v)
                log(f"[NOFRAMES] {v}")
                continue
            try:
                e = enc(frames)
            except enc.torch.cuda.OutOfMemoryError:
                enc.torch.cuda.empty_cache()
                stats["oom"].append(v)
                log(f"[OOM] {v}")
                continue
            atomic_save(out_dir, v, e)
            stats["n"] += 1
            if k % 50 == 0 or k == len(todo):
                el = time.time() - t0
                log(f"PROGRESS {k}/{len(todo)} elapsed={el:.0f}s eta={(len(todo) - k) * el / k:.0f}s")
    stats["wall_seconds"] = round(time.time() - t0, 1)
    (OUT_ROOT / "extract_image_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"extract-image done: {stats['n']} videos, missing {len(stats['missing_file'])}, "
        f"no frames {len(stats['no_frames'])}, OOM {len(stats['oom'])}, {stats['wall_seconds']} s")
    return 0 if not (stats["missing_file"] or stats["oom"]) else 5


# --------------------------------------------------------------- transcript ---
def tokenize(tok, texts: list[str]):
    """OpenAI-CLIP truncation: [SOT] + first 75 tokens + [EOT], zero padded to 77."""
    import torch
    sot, eot = tok.encoder["<|startoftext|>"], tok.encoder["<|endoftext|>"]
    out = torch.zeros(len(texts), CONTEXT, dtype=torch.long)
    n_cut = 0
    for i, t in enumerate(texts):
        ids = tok.encode(t)
        n_cut += len(ids) > CONTEXT - 2
        ids = [sot] + ids[: CONTEXT - 2] + [eot]
        out[i, : len(ids)] = torch.tensor(ids)
    return out, n_cut


def stage_text(a, log) -> int:
    import torch
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    dtype = "float16" if dev == "cuda" else "float32"
    model = load_model(dev, dtype)
    from imagebind.models.imagebind_model import ModalityType
    from imagebind.models.multimodal_preprocessors import SimpleTokenizer
    tok = SimpleTokenizer(bpe_path=str(BPE))
    text = np.load(TEXT_EMB)

    def embed(texts: list[str]) -> tuple[np.ndarray, int]:
        out, n_cut = [], 0
        with torch.no_grad():
            for i in range(0, len(texts), a.batch):
                t, nc = tokenize(tok, texts[i:i + a.batch])
                n_cut += nc
                e = model({ModalityType.TEXT: t.to(dev)})[ModalityType.TEXT]
                out.append(e.float().cpu().numpy())
        return (np.concatenate(out, 0) if out else np.zeros((0, 1024), np.float32)), n_cut

    # pipeline check: the campaign's "normal"/"hateful" embeddings re-encoded here
    chk, _ = embed(["normal", "hateful"])
    cos = np.sum(chk * text, 1) / (np.linalg.norm(chk, axis=1) * np.linalg.norm(text, axis=1))
    log(f"text check vs campaign normal/hateful embeddings: cosine {cos.tolist()}, norms "
        f"{np.linalg.norm(chk, axis=1).tolist()} vs {np.linalg.norm(text, axis=1).tolist()} ({dev}, {dtype})")
    if cos.min() < 0.999:
        log("FAILED text pipeline does not reproduce the campaign text embeddings")
        return 7
    for ds in a.datasets:
        out_dir = OUT_ROOT / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        tr = Transcripts(ds)
        lens = ec.gt_lengths(ds)
        dur = ec.durations(ds)
        rows = []
        for v in ec.cohort(ds):
            T = max(lens[v], ec.curve_length(dur[v]))
            for w in range(int(np.ceil(T / (WIN_S * FPS)))):
                s = tr.span(v, w * WIN_S, (w + 1) * WIN_S)
                if s:
                    rows.append({"video_id": v, "window": w, "start": w * WIN_S, "end": (w + 1) * WIN_S, "text": s})
        t0 = time.time()
        E, n_cut = embed([r["text"] for r in rows])
        p = p_hateful(E, text) if len(rows) else np.zeros(0)
        with (out_dir / "transcript_windows.jsonl").open("w") as fh:
            for r, x in zip(rows, p):
                fh.write(json.dumps({**r, "p_hateful": float(x)}) + "\n")
        log(f"text {ds}: {len(rows)} windows with speech, {n_cut} cut at 77 tokens, {time.time() - t0:.0f}s ({dev})")
    return 0


def load_text_windows(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            out.setdefault(r["video_id"], {})[int(r["window"])] = float(r["p_hateful"])
    return out


# -------------------------------------------------------------------- score ---
def stage_score(a, log) -> int:
    text = np.load(TEXT_EMB)
    rc = 0
    use = set(a.modalities.split(","))
    full = use == {"vision", "audio", "transcript"}
    for ds in a.datasets:
        out = OUT_ROOT / ds
        tw = load_text_windows(out / "transcript_windows.jsonl") if "transcript" in use else {}
        if not full:   # comparison rows (one modality or a pair), kept apart from the reported AV row
            out = out / ("modalities_" + "_".join(m for m in ("vision", "audio", "transcript") if m in use))
            out.mkdir(parents=True, exist_ok=True)
        dlog = ec.RunLog(out / "run.log", append=True)
        (out / "run.pid").write_text(f"{os.getpid()}\n")
        dlog(f"stage score; code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
        dur, lens = ec.durations(ds), ec.gt_lengths(ds)
        curves, extra, fail = {}, {}, {}
        tot = {"frames": 0, "frames_with_transcript": 0, "videos_without_speech": 0, "videos_without_vision": 0,
               "vision_tail_hold": 0, "audio_tail_hold": 0}
        for v in ec.cohort(ds):
            T = max(lens[v], ec.curve_length(dur[v]))
            mods, ex = [], {}
            vp = IMAGE_DIR[ds] / f"{v}.npy"
            if "vision" not in use:
                pass
            elif vp.exists() and np.load(vp, mmap_mode="r").shape[0] > 0:
                pv = p_hateful(np.load(vp), text)
                mods.append(ec.broadcast_to_4fps(pv, FPS, T))
                ex["vision_samples"] = int(len(pv))
                ex["vision_tail_hold"] = ec.tail_frames(len(pv), FPS, T)
                tot["vision_tail_hold"] += ex["vision_tail_hold"]
            else:
                tot["videos_without_vision"] += 1
                ex["vision"] = "absent (no decodable video stream)"
            ap = AUDIO_DIR[ds] / f"{v}.npy"
            if "audio" not in use:
                pass
            elif ap.exists():
                pa = p_hateful(np.load(ap), text)
                mods.append(ec.broadcast_to_4fps(pa, AUDIO_RATE, T))
                ex["audio_samples"] = int(len(pa))
                ex["audio_tail_hold"] = ec.tail_frames(len(pa), AUDIO_RATE, T)
                tot["audio_tail_hold"] += ex["audio_tail_hold"]
            else:
                ex["audio"] = "absent"
            num = np.sum(mods, axis=0) if mods else np.zeros(T)
            cnt = np.full(T, float(len(mods)))
            win = tw.get(v, {})
            if win:
                w_idx = np.arange(T) // int(WIN_S * FPS)
                pt = np.array([win.get(int(w), np.nan) for w in w_idx])
                has = np.isfinite(pt)
                num[has] += pt[has]
                cnt[has] += 1
                ex["frames_with_transcript"] = int(has.sum())
                tot["frames_with_transcript"] += int(has.sum())
            else:
                tot["videos_without_speech"] += 1
                ex["frames_with_transcript"] = 0
            tot["frames"] += T
            if not (cnt > 0).all():   # only possible without vision and audio: frames with no modality
                fail[v] = "frames with no available modality"
                continue
            curves[v] = num / cnt
            extra[v] = ex
        rep = ec.finalize("zs_imagebind_av" + ("" if full else "_" + "_".join(sorted(use))), ds, out, curves, native_rate=FPS, code_path=CODE_PATH, log=dlog,
                          extra=extra, failures=fail, notes=tot,
                          config={"variant": "ImageBind-Huge zero-shot, vision + audio + transcript, text pair "
                                             "['normal', 'hateful']",
                                  "text_embeddings": str(TEXT_EMB.relative_to(REPO)),
                                  "vision": str(IMAGE_DIR[ds].relative_to(REPO)) + " (4 fps, centre crop 224)",
                                  "audio": str(AUDIO_DIR[ds].relative_to(REPO)) + " (2 s clips, 0.5 fps)",
                                  "transcript": "Whisper large-v3 text of the 8 s window [8w, 8w+8) containing the "
                                                "frame; ImageBind text encoder, OpenAI-CLIP truncation to 77 tokens; "
                                                "windows: transcript_windows.jsonl",
                                  "modalities": sorted(use),
                                  "score": "mean over available modalities of p(hateful) = softmax(unit(e) @ t.T)",
                                  "grid": "native 4 fps; audio clip floor(i/8); window floor(i/32); tails hold"})
        rc |= 0 if rep.get("exact_test_set") else 6
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify-image", "extract-image", "text", "score"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decoders", type=int, default=6, help="extract-image: parallel ffmpeg decoders")
    ap.add_argument("--video-root", default=str(Path.home() / "data/DeHate/test"),
                    help="directory holding <video_id>.mp4 (extract-image: DeHate; verify-image: HateMM)")
    ap.add_argument("--out-dir", default=str(IMAGE_DIR["DeHate"]), help="extract-image output directory")
    ap.add_argument("--cpu", action="store_true", help="text stage: force CPU")
    ap.add_argument("--modalities", default="vision,audio,transcript",
                    help="score stage: modalities to average (default all three = the reported row; any other set "
                         "is a comparison row written to <DS>/modalities_<set>/)")
    a = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = ec.RunLog(OUT_ROOT / "run.log", append=True)
    (OUT_ROOT / "run.pid").write_text(f"{os.getpid()}\n")
    log(f"stage {a.stage}; code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
    rc = {"verify-image": stage_verify_image, "extract-image": stage_extract_image, "text": stage_text,
          "score": stage_score}[a.stage](a, log)
    log(f"stage {a.stage} {'DONE' if rc == 0 else f'FAILED rc={rc}'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
