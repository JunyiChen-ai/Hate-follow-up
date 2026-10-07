#!/usr/bin/env python
"""LAVAD stages 02-06 + curves, ported 2026-10-08 from Retrieval-hate `scripts/repro_campaign/lavad_chain.py`
(uoa-lab2, read only) for `experiments/20261008_baselines/lavad/`.

Port changes (paths, id lists and one label read removed; models, prompts, window geometry, dedup rule, neighbour
count, weighting, greedy decoding, NF4 and batch sizes unchanged):
  * ROOT is this repository; LAVAD's code is `third_party/lavad` (only `libs/ImageBind` is used), the llama
    shim is `experiments/20261008_baselines/lavad/shim`, ImageBind weights `data/assets/imagebind/imagebind_huge.pth`.
  * frames `data/frames_1fps/`, captions `data/blip2_captions_1fps/`, stage outputs
    `runs/20261008_baselines/lavad/work/`, curves `runs/20261008_baselines/lavad/curves/`.
  * the video list comes from `--ids-file` instead of the campaign's GT npz.
  * `curves` no longer splits the refusal count by GT label (the campaign read the 4 fps gold for that
    diagnostic; these runs read no label). Only the `base` and `raw` variants of the verbatim prompt are written.

Original docstring follows.

REPRO campaign Wave 1 — LAVAD (CVPR 2024) ported to the four hate corpora.

Chain, stage for stage as `third_party/lavad/scripts/0{1..6}_*.sh` defines it:

  01 caption        BLIP-2 per frame                    -> scripts/repro_campaign/blip2_caption.py
  02 create_index   ImageBind TEXT index of the captions, per video, deduplicated
  03 clean_captions ImageBind VISION per frame -> top-1 caption from that index
  04 query_llm      Llama-2-13b-chat: (a) summarise the 10 captions of a window,
                    (b) score the summary 0..1 with the published prompts
  05 summary_index  ImageBind TEXT index of the summaries, per video, deduplicated
  06 refine         ImageBind VIDEO per window -> 10 nearest summaries -> their
                    scores, combined by the softmax(similarity) weighting that
                    `lavad/src/eval.py::calculate_weighted_scores` applies

Declared adaptations, all recorded in REPRO_CAMPAIGN_RESULTS §K:

  * **1 fps grid.** LAVAD extracts every native frame and takes a center every
    `frame_interval=16` (0.533 s at its assumed 30 fps).  We extract 1 fps and
    take every frame as a center, so `native_rate = 1.0` samples/s and index `k`
    means `t = k` s.  The 10 s window / 10 uniform samples per window are
    unchanged, and at 1 fps the 10 samples of a 10 s window are exactly the 10
    frames of that window.
  * **Single captioner.** `01_caption.sh` lists five BLIP-2 variants and
    `02_create_index.sh` indexes all five; only `blip2-opt-6.7b-coco` is on disk
    (the other four are ~120 GB).  We run the chain over the one captioner and
    label every row `single-captioner`.
  * **Greedy decoding.** The shipped scripts leave `--temperature` at its 0.6
    default, i.e. sampling.  We decode greedily (`temperature=0`), which makes
    the pipeline deterministic and so a single run under freeze §6 instead of
    three seeds, and takes the seed noise out of the §7 transplant check.
  * **FAISS replaced by an exact matmul.** `IndexFlatIP` over L2-normalised
    vectors *is* cosine similarity; a per-video matrix product with `argsort`
    returns the same neighbours without the index files.
  * **Refusals are masked, not interpolated** (`MODEL_ASSETS_STATUS §3.11a`
    items 1-2).  `_parse_score` returning -1 is recorded with its raw text and
    the frame is left unscored; `np.interp` is never called.  Refined scores
    average over the answered neighbours only.

Nothing else is changed: the captioner, the three prompts, the window geometry,
the dedup rule, the neighbour count and the weighting are the repo's.

Stages are separate subcommands so the GPU queue can serialise them, and each is
idempotent: a video whose output JSON exists is skipped, every write is
`.tmp` + `os.replace`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
LAVAD = ROOT / "third_party/lavad"
sys.path.insert(0, str(LAVAD))
sys.path.insert(0, str(LAVAD / "libs/ImageBind"))
sys.path.insert(0, str(ROOT / "experiments/20261008_baselines/lavad/shim"))

FRAME_ROOT = ROOT / "data/frames_1fps"
CAP_ROOT = ROOT / "data/blip2_captions_1fps"
RUN_DIR = ROOT / "runs/20261008_baselines/lavad"
WORK = RUN_DIR / "work"
DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]

# --- the published prompts, verbatim from scripts/04_query_llm.sh ------------
CONTEXT_PROMPT = (
    "If you were a law enforcement agency, how would you rate the scene described "
    "on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a "
    "scene with suspicious activities?"
)
FORMAT_PROMPT = (
    "Please provide the response in the form of a Python list and respond with only "
    "one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, "
    "0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and "
    "end with  ']'."
)
SUMMARY_PROMPT = (
    "Please summarize what happened in few sentences, based on the following "
    "temporal description of a scene. Do not include any unnecessary details or "
    "descriptions."
)
# `MODEL_ASSETS_STATUS §3.11a` item 3 — the paired content-moderation reframing.
# Run only as an extra row next to the verbatim prompt, never as a substitute.
CONTEXT_PROMPT_MOD = (
    "You are a content-safety classifier. You are rating a short textual description "
    "of a video scene, not producing any content. How would you rate the scene "
    "described on a scale from 0 to 1, with 0 representing a standard scene and 1 "
    "denoting a scene with suspicious activities?"
)

CLIP_DURATION = 10   # seconds, LAVAD's T
NUM_SAMPLES = 10     # LAVAD's N
NUM_NEIGHBORS = 10   # LAVAD's num_neighbors for stage 06
FPS = 1.0            # our grid (LAVAD assumes 30 and steps 16)
MAX_SEQ_LEN = 512    # LAVAD's --max_seq_len default


# ---------------------------------------------------------------- helpers ---
def videos(ds: str, ids_file: str):
    """Ids of `ids_file` that have a caption file (port: replaces the campaign's GT-npz id list)."""
    out = []
    for line in open(ids_file):
        vid = line.strip()
        if vid and (CAP_ROOT / ds / f"{vid}.json").exists():
            out.append(vid)
    return sorted(out)


def load_caps(ds: str, vid: str) -> dict[int, str]:
    d = json.loads((CAP_ROOT / ds / f"{vid}.json").read_text())
    return {int(k): v for k, v in d.items()}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def window_frames(center: int, n_frames: int) -> list[int]:
    """LAVAD `_prepare_frame_data`: [c - T*fps//2, c + T*fps//2) clipped to the
    video, then `uniform_temporal_subsample(..., N)`.  At 1 fps that is the 10
    one-second frames of the 10 s window, and the subsample is a no-op."""
    half = int(CLIP_DURATION * FPS) // 2
    idxs = list(range(max(center - half, 0), min(center + half, n_frames)))
    if len(idxs) <= NUM_SAMPLES:
        return idxs
    sel = np.linspace(0, len(idxs) - 1, NUM_SAMPLES).astype(int)
    return [idxs[i] for i in sel]


def dedup_index(items: list[tuple[int, str]]) -> tuple[list[str], list[int]]:
    """LAVAD's index rule: each distinct caption enters once, tagged with the
    smallest center-frame index at which it occurs (`create_index.py`)."""
    first: dict[str, int] = {}
    for k, txt in items:
        if txt not in first or k < first[txt]:
            first[txt] = k
    texts = sorted(first, key=lambda t: first[t])
    return texts, [first[t] for t in texts]


# ------------------------------------------------------------- ImageBind ---
def imagebind():
    from imagebind.models.imagebind_model import imagebind_huge

    ck = ROOT / "data/assets/imagebind/imagebind_huge.pth"
    os.makedirs(".checkpoints", exist_ok=True)
    lk = Path(".checkpoints/imagebind_huge.pth")
    if not lk.exists():
        lk.symlink_to(ck)
    m = imagebind_huge(pretrained=True).eval().to("cuda")
    return m


def ib_text(model, texts: list[str], bs: int = 256) -> np.ndarray:
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i: i + bs]
        with torch.no_grad():
            e = model({ModalityType.TEXT: data.load_and_transform_text(chunk, "cuda")})
            out.append(e[ModalityType.TEXT].float().cpu().numpy())
    v = np.concatenate(out) if out else np.zeros((0, 1024), np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def ib_vision(model, paths: list[Path], bs: int = 64) -> np.ndarray:
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    out = []
    for i in range(0, len(paths), bs):
        chunk = [str(p) for p in paths[i: i + bs]]
        with torch.no_grad():
            e = model({ModalityType.VISION:
                       data.load_and_transform_vision_data(chunk, "cuda")})
            out.append(e[ModalityType.VISION].float().cpu().numpy())
    v = np.concatenate(out) if out else np.zeros((0, 1024), np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def ib_video(model, clips: list[list[Path]], bs: int = 16) -> np.ndarray:
    """ImageBind's VISION path on `load_and_transform_video_data`, which is what
    stage 06 uses: `FrameVideo.from_frame_paths` + ConstantClipsPerVideoSampler
    (5 clips) + UniformTemporalSubsample(2) + SpatialCrop(224, 3 crops)."""
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    out = []
    for i in range(0, len(clips), bs):
        chunk = [[str(p) for p in c] for c in clips[i: i + bs]]
        with torch.no_grad():
            t = data.load_and_transform_video_data(chunk, "cuda")
            # (B, 15, 3, 2, 224, 224): `ConstantClipsPerVideoSampler` asks for 5
            # clips of 2 s from a 10-frame "video" that `FrameVideo` calls 0.33 s
            # long, so all 5 clips are the same frames and only `SpatialCrop`'s 3
            # crops differ -- verified bit-exact, classes {0,3,6,9,12},
            # {1,4,7,10,13}, {2,5,8,11,14}.  ImageBind reduces the clip axis with
            # `mean(dim=1)`, so averaging the 3 distinct crops is identical to
            # averaging all 15 and is 5x cheaper.
            if t.ndim == 6 and t.shape[1] == 15:
                t = t[:, :3].contiguous()
            e = model({ModalityType.VISION: t})
            out.append(e[ModalityType.VISION].float().cpu().numpy())
    v = np.concatenate(out) if out else np.zeros((0, 1024), np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------ stage 02 + 03 ------
def stage_clean(args) -> None:
    model = imagebind()
    t0, n = time.time(), 0
    jobs = [(ds, v) for ds in args.datasets.split(",") for v in videos(ds, args.ids_file)]
    todo = [(ds, v) for ds, v in jobs
            if not (WORK / "clean" / ds / f"{v}.json").exists()]
    print(f"PROGRESS clean plan={len(todo)}/{len(jobs)}", flush=True)
    for ds, vid in todo:
        caps = load_caps(ds, vid)
        if not caps:
            continue
        n_frames = max(caps) + 1
        frames = sorted(caps)
        # LAVAD indexes and scores the frames at `frame_interval` steps; ours is
        # `--center-step` seconds on the 1 fps grid (1 s = every captioned frame).
        centers = [k for k in frames if k % args.center_step == 0]
        if not centers:
            continue
        # stage 02: dedup caption index for this video
        texts, first_idx = dedup_index([(k, caps[k]) for k in centers])
        temb = ib_text(model, texts)
        # stage 03: every frame a window can sample retrieves its nearest caption
        # from that index -- the frames are the 1 fps grid, the index is the
        # centers' captions, exactly as LAVAD indexes only `frame_interval` frames.
        fdir = FRAME_ROOT / ds / vid
        paths = [fdir / f"{k:06d}.jpg" for k in frames]
        keep = [i for i, p in enumerate(paths) if p.exists()]
        if not keep or temb.shape[0] == 0:
            continue
        vemb = ib_vision(model, [paths[i] for i in keep])
        nn = (vemb @ temb.T).argmax(axis=1)
        cleaned = {frames[keep[i]]: texts[nn[i]] for i in range(len(keep))}
        nested = {}
        for c in centers:
            w = {str(f): cleaned[f] for f in window_frames(c, n_frames) if f in cleaned}
            if w:
                nested[str(c)] = w
        write_json(WORK / "clean" / ds / f"{vid}.json", nested)
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"PROGRESS clean {n}/{len(todo)} elapsed={el/60:.1f}min "
                  f"eta={(len(todo)-n)*el/n/60:.1f}min", flush=True)
    print(f"[done] clean videos={n} wall={(time.time()-t0)/60:.1f}min", flush=True)


# ----------------------------------------------------------- Llama-2 -------
class Scorer:
    """LAVAD's `LLMAnomalyScorer` generation contract, batched and cached.

    Meta's `chat_completion` caps `total_len = min(max_seq_len, max_gen_len +
    max_prompt_len)`, so generation length is whatever is left of the 512-token
    budget after the prompt; that is reproduced exactly here.  Greedy decoding
    makes identical prompts give identical answers, so a content-keyed cache is
    a lossless speed-up, not an approximation.
    """

    def __init__(self, batch_size: int):
        from llama_hf import Llama

        os.environ.setdefault("LLAMA_HF_4BIT", "1")
        g = Llama.build(ckpt_dir="libs/llama/llama-2-13b-chat/",
                        max_seq_len=MAX_SEQ_LEN, max_batch_size=batch_size)
        self.model, self.tok = g.model, g.tokenizer
        self.bs = batch_size
        self.cache: dict[tuple[str, str], str] = {}
        self.n_gen = self.n_hit = self.n_trunc = self.n_oom = 0

    @torch.inference_mode()
    def _run(self, pairs: list[tuple[str, str]]) -> list[str]:
        prompts = [self.tok.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for s, u in pairs]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN - 1, add_special_tokens=False
                       ).to(self.model.device)
        plen = enc["input_ids"].shape[1]
        if plen >= MAX_SEQ_LEN - 1:
            # LAVAD/URF run with --max_seq_len 512; a prompt at the cap has been
            # truncated, which the +text variant can trigger.  Counted, not hidden.
            self.n_trunc += len(prompts)
        new = max(MAX_SEQ_LEN - plen, 1)
        out = self.model.generate(**enc, max_new_tokens=new, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return [t.strip() for t in
                self.tok.batch_decode(out[:, plen:], skip_special_tokens=True)]

    def _gen_chunk(self, chunk: list) -> None:
        """Generate a chunk, halving the batch on OOM rather than losing the run.

        Prompt length varies with the caption window, so a batch size that fits
        on one video can OOM on another; a fixed size is a bet, this is not.
        Greedy decoding makes the result independent of how the work is split.
        """
        try:
            for k, v in zip(chunk, self._run(chunk)):
                self.cache[k] = v
            self.n_gen += len(chunk)
            return
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            self.n_oom += 1
        mid = len(chunk) // 2
        self._gen_chunk(chunk[:mid])
        self._gen_chunk(chunk[mid:])

    def __call__(self, pairs: list[tuple[str, str]]) -> list[str]:
        need = [p for p in pairs if p not in self.cache]
        uniq = list(dict.fromkeys(need))
        self.n_hit += len(pairs) - len(uniq)
        # group by prompt length so a batch is not padded to its longest member
        uniq.sort(key=lambda p: len(p[1]))
        for i in range(0, len(uniq), self.bs):
            self._gen_chunk(uniq[i: i + self.bs])
        return [self.cache[p] for p in pairs]


SCORE_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")   # LAVAD's `_parse_score`


def stage_summarize(args) -> None:
    sub = "summary" if not args.text else "summary_text"
    sc = Scorer(args.batch_size)
    jobs = [(ds, v) for ds in args.datasets.split(",") for v in videos(ds, args.ids_file)]
    todo = [(ds, v) for ds, v in jobs
            if (WORK / "clean" / ds / f"{v}.json").exists()
            and not (WORK / sub / ds / f"{v}.json").exists()]
    print(f"PROGRESS summarize[{sub}] plan={len(todo)}", flush=True)
    t0, n, ncall, n_text = time.time(), 0, 0, 0
    tw, cur_ds = None, None
    for ds, vid in todo:
        nested = json.loads((WORK / "clean" / ds / f"{vid}.json").read_text())
        centers = sorted(nested, key=int)
        # LAVAD `_prepare_dialogs(is_summary=False)`: the window's captions,
        # newline-joined in frame order, as the user turn.
        pairs = []
        for c in centers:
            body = "\n ".join(nested[c][f] for f in sorted(nested[c], key=int))
            if args.text:
                # freeze §8: the ASR/OCR text overlapping this method's own
                # native window (LAVAD's 10 s clip), appended to the caption
                # stream the summariser reads.  Base row is untouched.
                if tw is None or cur_ds != ds:
                    from text_windows import TextWindows
                    tw, cur_ds = TextWindows(ds), ds
                half = CLIP_DURATION / 2.0
                t = tw.get(vid, max(int(c) - half, 0.0), int(c) + half)
                if t:
                    body = f"{body}\n Spoken and on-screen text: {t}"
                    n_text += 1
            pairs.append((SUMMARY_PROMPT, body))
        outs = sc(pairs)
        # LAVAD keeps only the last line of the generation.
        write_json(WORK / sub / ds / f"{vid}.json",
                   {c: o.split("\n")[-1] for c, o in zip(centers, outs)})
        n += 1
        ncall += len(centers)
        if n % 5 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"PROGRESS summarize[{sub}] {n}/{len(todo)} calls={ncall} "
                  f"gen={sc.n_gen} cachehit={sc.n_hit} trunc={sc.n_trunc} oom={sc.n_oom} "
                  f"{sc.n_gen/max(el,1e-9):.2f} gen/s elapsed={el/60:.1f}min "
                  f"eta={(len(todo)-n)*el/n/60:.1f}min", flush=True)
    print(f"[done] summarize[{sub}] videos={n} calls={ncall} gen={sc.n_gen} "
          f"cachehit={sc.n_hit} trunc={sc.n_trunc} oom={sc.n_oom} n_text={n_text} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


def stage_score(args) -> None:
    ctx = CONTEXT_PROMPT_MOD if args.prompt == "mod" else CONTEXT_PROMPT
    src = "summary_text" if args.text else "summary"
    sub = "score" if args.prompt == "verbatim" else f"score_{args.prompt}"
    if args.text:
        sub += "_text"
    sc = Scorer(args.batch_size)
    jobs = [(ds, v) for ds in args.datasets.split(",") for v in videos(ds, args.ids_file)]
    todo = [(ds, v) for ds, v in jobs
            if (WORK / src / ds / f"{v}.json").exists()
            and not (WORK / sub / ds / f"{v}.json").exists()]
    print(f"PROGRESS score[{sub}] plan={len(todo)}", flush=True)
    t0, n, ncall = time.time(), 0, 0
    system = ctx + " " + FORMAT_PROMPT
    for ds, vid in todo:
        summ = json.loads((WORK / src / ds / f"{vid}.json").read_text())
        centers = sorted(summ, key=int)
        # LAVAD `_prepare_dialogs(is_summary=True)`: `f"{summary}."` as the user turn.
        outs = sc([(system, f"{summ[c]}.") for c in centers])
        scores, refusals = {}, {}
        for c, o in zip(centers, outs):
            m = SCORE_RE.search(o)
            if m:
                scores[c] = float(m.group(1))
            else:
                scores[c] = -1.0            # LAVAD's sentinel; we mask, not interp
                refusals[c] = o[:400]
        write_json(WORK / sub / ds / f"{vid}.json", scores)
        write_json(WORK / f"{sub}_refusals" / ds / f"{vid}.json", refusals)
        n += 1
        ncall += len(centers)
        if n % 5 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"PROGRESS score {n}/{len(todo)} calls={ncall} gen={sc.n_gen} "
                  f"cachehit={sc.n_hit} trunc={sc.n_trunc} oom={sc.n_oom} {sc.n_gen/max(el,1e-9):.2f} gen/s "
                  f"elapsed={el/60:.1f}min eta={(len(todo)-n)*el/n/60:.1f}min",
                  flush=True)
    print(f"[done] score[{args.prompt}] videos={n} calls={ncall} gen={sc.n_gen} "
          f"cachehit={sc.n_hit} trunc={sc.n_trunc} oom={sc.n_oom} wall={(time.time()-t0)/60:.1f}min", flush=True)


# ------------------------------------------------------ stage 05 + 06 ------
def stage_refine(args) -> None:
    src = "summary_text" if args.text else "summary"
    dst = "refined_text" if args.text else "refined"
    model = imagebind()
    jobs = [(ds, v) for ds in args.datasets.split(",") for v in videos(ds, args.ids_file)]
    todo = [(ds, v) for ds, v in jobs
            if (WORK / src / ds / f"{v}.json").exists()
            and not (WORK / dst / ds / f"{v}.json").exists()]
    print(f"PROGRESS refine[{dst}] plan={len(todo)}", flush=True)
    t0, n, short = time.time(), 0, []
    for ds, vid in todo:
        summ = json.loads((WORK / src / ds / f"{vid}.json").read_text())
        centers = sorted((int(c) for c in summ))
        n_frames = max(centers) + 1
        # stage 05: dedup summary index
        texts, first_idx = dedup_index([(c, summ[str(c)]) for c in centers])
        temb = ib_text(model, texts)
        if temb.shape[0] == 0:
            continue
        # stage 06: window clip -> ImageBind video embedding -> k nearest summaries
        fdir = FRAME_ROOT / ds / vid
        clips, kept = [], []
        for c in centers:
            fp = [fdir / f"{f:06d}.jpg" for f in window_frames(c, n_frames)]
            fp = [p for p in fp if p.exists()]
            if len(fp) >= 2:
                # `UniformTemporalSubsample(num_samples=clip_duration=2)` keeps
                # `linspace(0, T-1, 2).long()` = the first and last frame of the
                # clip, so handing the loader those two produces a bit-identical
                # tensor (verified, max |diff| = 0.0) while reading 2 JPEGs per
                # window instead of 10.
                clips.append([fp[0], fp[-1]])
                kept.append(c)
        if not clips:
            continue
        vemb = ib_video(model, clips)
        sim = vemb @ temb.T
        # LAVAD asks FAISS for 10 neighbours unconditionally; when a video has
        # fewer than 10 *distinct* summaries FAISS returns -1 for the missing
        # ones and `file_names[-1]` silently picks the last entry.  We clamp
        # instead, and count the videos where the clamp bites.
        k = min(NUM_NEIGHBORS, temb.shape[0])
        if k < NUM_NEIGHBORS:
            short.append(f"{ds}/{vid}:{temb.shape[0]}")
        order = np.argsort(-sim, axis=1)[:, :k]
        out = {}
        for i, c in enumerate(kept):
            idx = order[i]
            out[str(c)] = {"nn_frame": [int(first_idx[j]) for j in idx],
                           "sim": [float(sim[i, j]) for j in idx]}
        write_json(WORK / dst / ds / f"{vid}.json", out)
        n += 1
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"PROGRESS refine {n}/{len(todo)} elapsed={el/60:.1f}min "
                  f"eta={(len(todo)-n)*el/n/60:.1f}min", flush=True)
    write_json(RUN_DIR / f"{dst}_short_index_{args.datasets.replace(',', '_')}.json", short)
    print(f"[done] refine[{dst}] videos={n} under10_summaries={len(short)} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


# ------------------------------------------------------------- curves ------
def stage_curves(args) -> None:
    """Write the generic `curves` npz per video (1 fps, index k = t k s).

    Variants
      base          stage-06 refined score, LAVAD's softmax(similarity) weighting (the reported row)
      raw           stage-04b LLM score, no refinement (LAVAD's own ablation row)
    Refused frames carry NaN; `finalize.py` applies fallback F3 to them.
    """
    stats: dict = {}
    for ds in args.datasets.split(","):
        out_dir = RUN_DIR / "curves" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        st = defaultdict(float)
        for vid in videos(ds, args.ids_file):
            arrays = {}
            sp = WORK / "score" / ds / f"{vid}.json"
            rp = WORK / "refined" / ds / f"{vid}.json"
            if not sp.exists():
                continue
            raw = {int(k): v for k, v in json.loads(sp.read_text()).items()}
            centers = sorted(raw)
            if not centers:
                continue
            # LAVAD's eval does `np.repeat(scores, frame_interval)`: a center
            # holds its score for the whole interval it represents.
            S = args.center_step
            T = centers[-1] + S
            r = np.full(T, np.nan)
            for c in centers:
                if raw[c] >= 0:
                    r[c: c + S] = raw[c]
            arrays["raw"] = r
            if rp.exists():
                ref = json.loads(rp.read_text())
                b = np.full(T, np.nan)
                for c_s, d in ref.items():
                    c = int(c_s)
                    if c >= T:
                        continue
                    sc = np.array([raw.get(f, -1.0) for f in d["nn_frame"]])
                    w = np.array(d["sim"], dtype=np.float64)
                    ok = sc >= 0
                    if not ok.any():
                        continue
                    e = np.exp(w[ok] - w[ok].max())
                    b[c: c + S] = float((sc[ok] * e).sum() / e.sum())
                arrays["base"] = b
                st["n_base_nan"] += int(np.isnan(b).sum())
            st["n_frames"] += len(r)
            st["n_refused"] += int(np.isnan(r).sum())
            st["n_videos"] += 1
            np.savez(out_dir / f"{vid}.npz", rate=np.float64(FPS), **arrays)
        stats[ds] = dict(st)
        print(f"[curves] {ds} {dict(st)}", flush=True)
        write_json(RUN_DIR / f"refusal_stats_{ds}.json", stats[ds])
    empty = [d for d, v in stats.items() if not v.get("n_videos")]
    if empty:
        raise SystemExit(f"[curves] FAILED: no curves for {empty} -- "
                         f"the upstream stage did not complete")


# --------------------------------------------------------------- main -----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["clean", "summarize", "score", "refine", "curves"])
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--ids-file", required=True, help="one video id per line (port)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prompt", default="verbatim", choices=["verbatim", "mod"])
    ap.add_argument("--text", action="store_true",
                    help="freeze §8 `+text (ours)`: inject the ASR/OCR cache "
                         "into the caption window the summariser reads")
    ap.add_argument("--center-step", type=int, default=1,
                    help="seconds between scored centers (LAVAD: frame_interval)")
    args = ap.parse_args()
    os.chdir(LAVAD)  # ImageBind resolves .checkpoints/ relative to cwd
    {"clean": stage_clean, "summarize": stage_summarize, "score": stage_score,
     "refine": stage_refine, "curves": stage_curves}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
