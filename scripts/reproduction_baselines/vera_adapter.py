#!/usr/bin/env python3
"""VERA (CVPR 2025) adapter for the four hateful-video corpora.

The adapter keeps VERA's released InternVL2-8B backbone, eight-frame input,
binary initial segment score, visual-neighbour propagation and Gaussian
smoothing.  Prompt-bank selection is video-level and validation-only.  Test
inference is a separate explicit stage so tuning cannot touch test media.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from hate_common import data as hdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_ID = "OpenGVLab/InternVL2-8B"
FEATURE_ROOT = REPO / "results/reproduction/features/clip_b16_1fps"

VIDEO_DIRS = {
    "hatemm": Path("/home/jehc223/data/HateMM/video"),
    "mhclip_en": Path("/home/jehc223/data/Multihateclip/English/video_mp4"),
    "mhclip_zh": Path("/home/jehc223/data/Multihateclip/Chinese/video"),
    "hateclipseg": Path("/home/jehc223/data/HateClipSeg/videos"),
}
EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv")

PROMPT_BANK = {
    "ucf_released": [
        "Are any people outside typical positions or doing activities inconsistent with usual behavior?",
        "Are any vehicles outside typical positions or used inconsistently with their usual function?",
        "Are any objects outside typical positions or used inconsistently with their usual function?",
        "Is there visible damage or unusual movement indicating an anomaly?",
        "Are there unusual sounds or noises suggesting an anomaly?",
    ],
    "xd_released": [
        "Does any person or object perform an unusual action that does not fit the video's usual pattern?",
        "Does any person or object behave in a clearly abnormal or unexpected way?",
        "Is there a clear behavioral anomaly, such as unexpected object movement or an unusual human action?",
        "Is a person or object out of place or inconsistent with the video's usual pattern?",
        "Is the normal flow disrupted, for example by a sudden camera change or loss of video quality?",
    ],
    "hate_domain": [
        "Does anyone attack, threaten, demean, or promote hostility toward a person or protected group?",
        "Are hateful symbols, slurs, insults, or dehumanizing claims visibly communicated?",
        "Does the scene endorse violence, exclusion, discrimination, or harm against a person or group?",
        "Do gestures, displayed text, or interactions convey targeted hateful or abusive meaning?",
        "Taken in context, is hateful or severely offensive content present in this video segment?",
    ],
}


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def video_path(corpus, vid):
    for ext in EXTS:
        path = VIDEO_DIRS[corpus] / (vid + ext)
        if path.is_file():
            return path
    raise FileNotFoundError(f"media not found: {corpus}/{vid}")


def transform_images(images):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    tfm = T.Compose([
        T.Lambda(lambda x: x.convert("RGB")),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    return __import__("torch").stack([tfm(x) for x in images])


def read_frames(path, start, window, count=8):
    import decord
    vr = decord.VideoReader(str(path), num_threads=2)
    fps = float(vr.get_avg_fps())
    duration = len(vr) / max(fps, 1e-6)
    end = min(duration, start + window)
    times = np.linspace(start, max(start, end - 1 / max(fps, 1)), count)
    indices = np.clip(np.rint(times * fps).astype(int), 0, len(vr) - 1)
    batch = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(x) for x in batch], duration


def load_model(required_backend=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    flash_available = importlib.util.find_spec("flash_attn") is not None
    if required_backend == "flash_attention_2" and not flash_available:
        raise RuntimeError("selected VERA prompt used flash_attention_2, but "
                           "flash_attn is unavailable for frozen inference")
    use_flash = (flash_available if required_backend is None else
                 required_backend == "flash_attention_2")
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, use_flash_attn=use_flash).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backend = "flash_attention_2" if use_flash else "torch_attention"
    print(f"VERA attention backend: {backend}", flush=True)
    return model, tok, backend


def question(prompt_lines, n_frames=8):
    prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(n_frames))
    guides = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(prompt_lines))
    return (prefix + "Analyze the video using these guiding questions:\n" + guides +
            "\nBased on the analysis, output exactly `Output: 1` if hateful or "
            "severely offensive content is present, otherwise `Output: 0`.")


def predict(model, tok, images, prompts):
    import torch
    pixels = transform_images(images).to("cuda", dtype=torch.bfloat16)
    cfg = dict(num_beams=1, max_new_tokens=256, do_sample=False)
    with torch.inference_mode():
        response = model.chat(tok, pixels, question(prompts, len(images)), cfg)
    hits = re.findall(r"Output\s*:\s*([01])", response, flags=re.I)
    if hits:
        return int(hits[-1]), response
    return int("yes" in response.lower() and "no, there" not in response.lower()), response


def valid_selection(path, corpus):
    try:
        rec = json.loads(Path(path).read_text())
        return (rec.get("corpus") == corpus and
                rec.get("selection_split") == "official-val" and
                rec.get("selected") in PROMPT_BANK and
                rec.get("prompts") == PROMPT_BANK[rec["selected"]] and
                set(rec.get("scores", {})) == set(PROMPT_BANK) and
                all(math.isfinite(float(x)) for x in rec["scores"].values()) and
                rec.get("backbone") == MODEL_ID and
                rec.get("attention_backend") in
                ("flash_attention_2", "torch_attention"))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def completed_validation_rows(path, valid_ids):
    rows, dirty = {}, False
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if (rec.get("video_id") not in valid_ids or
                        rec.get("score") not in (0, 1) or
                        not isinstance(rec.get("response"), str)):
                    raise ValueError("invalid validation row")
                rows[rec["video_id"]] = rec
            except (ValueError, TypeError, json.JSONDecodeError):
                dirty = True
    if dirty:
        atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                   for r in rows.values()))
    return rows


def select(args):
    from sklearn.metrics import average_precision_score
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
    selected_path = root / "selected_prompt.json"
    if valid_selection(selected_path, args.corpus):
        print(f"already selected {args.corpus}: {selected_path}")
        return
    model, tok, backend = load_model()
    labels = hdata.load_labels(args.corpus)
    ids = hdata.load_split(args.corpus, "val")
    valid_ids = set(ids)
    scores = {}
    for name, prompts in PROMPT_BANK.items():
        pred = []
        log = root / f"val_{name}.jsonl"
        done = completed_validation_rows(log, valid_ids)
        with log.open("a") as fh:
            for vid in ids:
                if vid in done:
                    pred.append(done[vid]["score"]); continue
                _, duration = read_frames(video_path(args.corpus, vid), 0, 1, 8)
                images, _ = read_frames(video_path(args.corpus, vid), 0, duration, 8)
                score, response = predict(model, tok, images, prompts)
                rec = {"video_id": vid, "score": score, "response": response}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                pred.append(score)
        scores[name] = float(average_precision_score([labels[v] for v in ids], pred))
    best = max(scores, key=scores.get)
    payload = {"corpus": args.corpus, "selection_split": "official-val",
               "metric": "video_average_precision", "scores": scores,
               "selected": best, "prompts": PROMPT_BANK[best],
               "backbone": MODEL_ID, "attention_backend": backend}
    atomic_write(selected_path, json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def valid_raw_result(path, vid, expected_windows):
    try:
        rec = json.loads(path.read_text())
        duration = float(rec["duration"])
        segments = rec["segments"]
        return (rec.get("video_id") == vid and math.isfinite(duration) and
                duration > 0 and len(segments) == expected_windows and
                expected_windows > 0 and
                all(x.get("score") in (0, 1) and
                    math.isfinite(float(x["start"])) and
                    math.isfinite(float(x["end"])) and
                    isinstance(x.get("response"), str) for x in segments))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def infer(args):
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.prompt_json).read_text())
    model, tok, _backend = load_model(selection["attention_backend"])
    prompts = selection["prompts"]
    gt = hdata.gt_arrays(args.corpus, args.split)
    ids = [vid for vid in hdata.load_split(args.corpus, args.split) if vid in gt]
    for vi, vid in enumerate(ids, 1):
        out = root / f"{vid}.json"
        starts = np.arange(0, len(gt[vid]), args.stride)
        if valid_raw_result(out, vid, len(starts)):
            continue
        path = video_path(args.corpus, vid)
        _, duration = read_frames(path, 0, 1, 8)
        records = []
        for start in starts:
            images, _ = read_frames(path, float(start), args.window, 8)
            score, response = predict(model, tok, images, prompts)
            records.append({"start": float(start), "end": min(duration, start + args.window),
                            "score": score, "response": response})
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"video_id": vid, "duration": duration,
                                   "segments": records}, ensure_ascii=False))
        os.replace(tmp, out)
        print(f"[{vi}/{len(ids)}] {vid}: {len(records)} windows", flush=True)


def softmax(x):
    z = np.exp(x - np.max(x)); return z / z.sum()


def postprocess(args):
    from scipy.ndimage import gaussian_filter1d
    raw_root, out = Path(args.raw_dir), Path(args.out)
    rows = []
    gt = hdata.gt_arrays(args.corpus, args.split)
    ids = [vid for vid in hdata.load_split(args.corpus, args.split) if vid in gt]
    for vid in ids:
        rec = json.loads((raw_root / f"{vid}.json").read_text())
        raw = np.asarray([x["score"] for x in rec["segments"]], dtype=float)
        visual = np.load(FEATURE_ROOT / args.corpus / f"{vid}.npy")
        if len(raw) != len(gt[vid]) or len(visual) != len(gt[vid]):
            raise ValueError(
                f"{args.corpus}/{vid}: raw={len(raw)}, visual={len(visual)}, "
                f"gold={len(gt[vid])}; refusing silent temporal truncation")
        n = len(raw)
        visual = visual / np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
        similarity = visual @ visual.T
        top_n = max(1, int(.15 * n))
        propagated = np.empty(n)
        for i in range(n):
            ix = np.argsort(similarity[i])[-top_n:]
            propagated[i] = softmax(similarity[i, ix] * 10) @ raw[ix]
        neighbor = gaussian_filter1d(propagated, sigma=10, radius=7, mode="nearest")
        x = np.arange(n); sigma = max(n / 2, 1)
        center_weight = np.exp(-.5 * ((x - n // 2) / sigma) ** 2)
        official = neighbor * center_weight
        rows.append({"video_id": vid, "score_raw": raw.tolist(),
                     "score_neighbor": neighbor.tolist(),
                     "score_official_postprocessed": official.tolist()})
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, "".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {out}: {len(rows)} videos")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("stage", choices=("select", "infer", "postprocess"))
    ap.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    ap.add_argument("--out-dir", default="results/reproduction/official_val/vera")
    ap.add_argument("--prompt-json")
    ap.add_argument("--raw-dir")
    ap.add_argument("--out")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--window", type=float, default=10.)
    ap.add_argument("--stride", type=float, default=1.)
    args = ap.parse_args()
    if args.stage == "select": select(args)
    elif args.stage == "infer":
        if not args.prompt_json: ap.error("infer requires --prompt-json")
        infer(args)
    else:
        if not args.raw_dir or not args.out: ap.error("postprocess requires --raw-dir and --out")
        postprocess(args)


if __name__ == "__main__":
    main()
