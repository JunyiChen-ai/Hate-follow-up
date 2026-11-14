#!/usr/bin/env python3
"""AnyAnomaly (WACV 2026, arXiv 2503.04504, github.com/SkiddieAhn/Paper-AnyAnomaly, `vad_proposed_Qwen_vllm.py`) on
Qwen3-VL-8B with frames + Whisper words. README §3.3.

AnyAnomaly is training-free customizable VAD: for every clip and a user-defined anomaly text (keyword), CLIP builds
three images and an LVLM scores each with a 0-1 "does the image include <keyword>" prompt (context-aware VQA):
  1. key frame: KFS 4 (CLIP similarity to the keyword, then grouping) picks the most similar frame and three more
     frames spaced by a quarter clip (functions/key_func.py key_frame_selection_clip_grouping);
  2. position context (WinCLIP attention): the key frame masked by the multi-scale patch similarity to the keyword's
     template-ensembled CLIP text embedding (functions/attn_func.py);
  3. temporal context: the patch (over three scales) most similar to the keyword across the four key frames, shown as
     a 2x2 grid in time order (functions/grid_func.py), with the temporal-context prompt.
The three scores are combined as alpha * key + beta * wa + gamma * tc (alpha .6, beta .3, gamma .1), Gaussian-smoothed
(sigma 15 frames) and min-max normalised.

Kept: the three prompts (functions/qwen_vllm_func.py make_instruction, avenue's out_prompt "Output"), the system
message, KFS 4, the WinCLIP attention and the grid construction with the release's avenue scales (img 240; patches
48 / 80 / 240, stride = size; the default dataset of the release), template ensemble on, CLIP ViT-B/32, the weights
alpha .6 / beta .3 / gamma .1 and sigma 15 frames (the release's non-grid-search constants).
Changed (fixed before any run; README §3.3 lists each):
- backbone Qwen3-VL-8B (release default Qwen2.5-VL-3B-Instruct); greedy decoding, at most 50 new tokens (release:
  temperature .1, top_p .001, repetition penalty 1.05, 50 tokens); every image resized to the naive row's image size
  rule (smart_resize, src/mllm_judge.py) so generation and the fp32 read-out see the same pixels;
- the clip's Whisper words are added to each prompt ("Transcript: <words>" before the instruction);
- keyword "hateful content";
- clip: 8 s window with 24 frames from a 3 fps decode (release: 24 consecutive native frames, about 1 s at 25 fps), for
  cost; a last window with fewer frames uses the frames it has (the release drops an incomplete last clip);
- class adaptation off: the release's softmax over the dataset's anomaly classes is constant 1 with one keyword, which
  makes its attention normalisation divide by zero; the release's `class_adaption False` path is used;
- CLIP ViT-B/32 from transformers (openai/clip-vit-base-patch32, the same weights as the `clip` package); CLIP
  preprocessing for KFS as `clip.load`'s (resize 224 bicubic, centre crop, CLIP normalisation); patches go to the
  image encoder unnormalised, as in the release;
- read-outs: `prob` = expected score under the fp32 distribution at the position where the answer writes its first
  number (vllm_common.expected_score; if the answer has no number, after the answer + "\\nScore: "); `text` = the
  release's extract_score (first number in the answer, 0 if none);
- the Gaussian (sigma 15 native frames = 0.6 s at 25 fps, i.e. 2.4 frames at 4 fps) is applied per video on the 4 fps
  curve (the release smooths the concatenated test set); the global min-max normalisation is omitted (monotone, no
  effect on any metric); no grid search over sigma / weights (the release's default searches them on test labels).

Stages: `infer` (GPU, vLLM env: CLIP context images, stored as JPEG under <run>/ctx/ until the read-out, and the three
generations) -> units.jsonl; `readout` (GPU, HateVLM env) -> readout.jsonl; `raster` (CPU, HateVideo env). No label is
read. `clean` removes the context images after the read-out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_common as Q         # noqa: E402
from vllm_common import L       # noqa: E402

KEYWORD = "hateful content"
UNIT = 8.0
CLIP_LEN = 24
DECODE_FPS = CLIP_LEN / UNIT
CLIP_ID = "openai/clip-vit-base-patch32"
IMG = 240
SCALES = [((48, 48), (48, 48)), ((80, 80), (80, 80)), ((240, 240), (240, 240))]   # (kernel, stride): sml, mid, lge
ALPHA, BETA, GAMMA = 0.6, 0.3, 0.1
SIGMA_S = 15 / 25.0
SYSTEM = "You are a vision anomaly detector."
NO_SPEECH = "(no speech)"
OUT_PROMPT = "Output"
TEMPLATES = ["a cropped photo of the {}.", "a close-up photo of a {}.", "a close-up photo of the {}.",
             "a bright photo of a {}.", "a bright photo of the {}.", "a dark photo of the {}.", "a dark photo of a {}.",
             "a jpeg corrupted photo of the {}.", "a jpeg corrupted photo of the {}.", "a blurry photo of the {}.",
             "a blurry photo of a {}.", "a photo of a {}.", "a photo of the {}.", "a photo of a small {}.",
             "a photo of the small {}.", "a photo of a large {}.", "a photo of the large {}.",
             "a photo of the {} for visual inspection.", "a photo of a {} for visual inspection.",
             "a photo of the {} for anomaly detection.", "a photo of a {} for anomaly detection."]


def instructions(keyword):
    """functions/qwen_vllm_func.py make_instruction, verbatim (avenue out_prompt)."""
    ins = f"""\
- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1.
A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all.
For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.
- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.
- **{OUT_PROMPT}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**.
"""
    tc = f"""\
- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1.
A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all.
For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.
- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.
- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.
- **{OUT_PROMPT}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**.
"""
    return ins, tc


def user_text(instruction, words):
    return "Transcript: %s\n\n%s" % (words or NO_SPEECH, instruction)


def out_dir(ds):
    return os.path.join(Q.RUNS, "anyanomaly", ds, "u8")


def load_records(path):
    recs = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                recs[r["video_id"]] = r
    return recs


def extract_score(text):
    """utils.py extract_score: the first number in the answer, 0 if none."""
    m = re.search(r'(\d+(\.\d+)?)', text or "")
    return float(m.group(0)) if m else 0.0


def number_prefix(text):
    """The answer up to its first number (the position where the method writes its score), or None."""
    m = re.search(r'\d', text or "")
    return text[:m.start()] if m else None


def reader_size(img):
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from src.mllm_judge import MAX_PIXELS, MIN_PIXELS
    h, w = smart_resize(img.height, img.width, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return img if (h, w) == (img.height, img.width) else img.resize((w, h), Image.BICUBIC)


# ------------------------------------------------------------------------------------------------ CLIP context
class Context:
    def __init__(self, device="cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.device = torch, device
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)
        with torch.no_grad():
            self.kw = self._text([KEYWORD])                                 # KFS / grid: the bare keyword
            self.kw_tmpl = self._text([t.format(KEYWORD) for t in TEMPLATES]).mean(0, keepdim=True)   # WA embedding

    def _text(self, texts):
        enc = self.proc(text=texts, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(self.device)
        return self.model.text_projection(self.model.text_model(**enc).pooler_output).float()

    def _image(self, pixel_values):
        return self.model.visual_projection(self.model.vision_model(pixel_values=pixel_values).pooler_output).float()

    def _sim(self, img_feat, txt):
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return (txt @ img_feat.T).cpu().numpy()[0]

    def _patches(self, path, kernel, stride):
        """grid_func / attn_func split_*_with_unfold: 240 x 240 RGB in [0, 1], unfold, each patch resized to 224."""
        import cv2
        F = self.torch.nn.functional
        im = cv2.resize(cv2.imread(path), (IMG, IMG)).astype("float32")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32") / 255
        t = self.torch.tensor(im).permute(2, 0, 1).unsqueeze(0)
        p = F.unfold(t, kernel_size=kernel, stride=stride)
        p = p.permute(0, 2, 1).reshape(-1, 3, kernel[0], kernel[1])
        n_r = (IMG - kernel[0]) // stride[0] + 1
        n_c = (IMG - kernel[1]) // stride[1] + 1
        return F.interpolate(p, size=(224, 224), mode="bilinear"), (n_r, n_c)

    def build(self, files):
        """files: the clip's frames (JPEG paths). Returns (key file, WA PIL image, grid PIL image, info)."""
        import cv2
        from PIL import Image
        torch, F = self.torch, self.torch.nn.functional
        n = len(files)
        with torch.no_grad():
            pix = self.proc(images=[Image.open(f).convert("RGB") for f in files], return_tensors="pt")["pixel_values"]
            s = self._sim(self._image(pix.to(self.device)), self.kw)
            max_idx = int(np.argmax(s))
            g = max(1, n // 4)
            d = max_idx % g
            keys = [min(d + k * g, n - 1) for k in range(4)]
            # position context (class adaptation off): mean over scales of the upsampled patch similarity maps
            maps = []
            for kernel, stride in SCALES:
                p, (nr, nc) = self._patches(files[max_idx], kernel, stride)
                sim = torch.tensor(self._sim(self._image(p.to(self.device)), self.kw_tmpl)).view(1, 1, nr, nc)
                maps.append(F.interpolate(sim, size=(224, 224), mode="bilinear").squeeze(0))
            m = (sum(maps) / len(maps)).squeeze(0).numpy()
            m = (m - m.min()) / (m.max() - m.min()) if m.max() > m.min() else np.ones_like(m)
            img = cv2.cvtColor(cv2.resize(cv2.imread(files[max_idx]), (224, 224)), cv2.COLOR_BGR2RGB)
            wa = Image.fromarray((img * np.repeat(m[:, :, None], 3, axis=2)).astype(np.uint8))
            # temporal context: grouped patches over the four key frames, the one most similar to the keyword
            groups = []
            for kernel, stride in SCALES:
                per = [self._patches(files[k], kernel, stride)[0] for k in keys]
                groups += [torch.stack([pp[i] for pp in per]) for i in range(per[0].shape[0])]
            best, bi = -1e9, 0
            for i, gp in enumerate(groups):
                v = float(np.max(self._sim(self._image(gp.to(self.device)), self.kw)))
                if v > best:
                    best, bi = v, i
            gp = groups[bi]
            grid = np.zeros((3, 224, 224), dtype=np.float32)
            for q, (r0, c0) in enumerate(((0, 0), (0, 112), (112, 0), (112, 112))):
                grid[:, r0:r0 + 112, c0:c0 + 112] = F.interpolate(gp[q].unsqueeze(0), size=(112, 112), mode="bilinear").numpy()
            grid = Image.fromarray((np.transpose(grid * 255, [1, 2, 0])).astype(np.uint8))
        return files[max_idx], wa, grid, {"key": max_idx, "keys": keys, "grid_patch": bi}


# ------------------------------------------------------------------------------------------------ stages
def cmd_infer(args):
    from PIL import Image
    from src.video_inputs import fixed_windows
    ds = args.dataset
    dest = out_dir(ds)
    L.start_run_log(dest)
    log = Q.log_to(os.path.join(dest, "infer.log"))
    path = os.path.join(dest, "units.jsonl")
    man = L.manifest(ds)
    vids = L.cohort(ds)
    if args.limit:
        vids = vids[:args.limit]
    done = set(load_records(path))
    todo = [v for v in vids if v not in done]
    log("anyanomaly %s: %d cohort videos, %d to do; code %s" % (ds, len(vids), len(todo), L.git_version()))
    if not todo:
        log("DONE infer (nothing to do)")
        return
    segs = L.transcript_segments(ds, {v: man[v]["duration"] for v in vids})
    ctx = Context()
    eng = Q.Engine(max_model_len=4096, max_images=1, log=log, gpu_mem=0.80)
    ins, tc = instructions(KEYWORD)
    with open(os.path.join(dest, "prompt.txt"), "w") as fh:
        for t in (ins, tc):
            fh.write(eng.request([None], user_text(t, "<words>"), "x", system=SYSTEM)["prompt"] + "\n\n=====\n\n")
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        try:
            return v, Q.decode_frames(man[v]["video_path"], DECODE_FPS, "aa_%s_%s" % (ds, v)), None
        except Exception as e:  # noqa: BLE001
            return v, [], repr(e)

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    fut = pool.map(decode, chunks[0])
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        decoded = list(fut)
        if ci + 1 < len(chunks):
            fut = pool.map(decode, chunks[ci + 1])
        reqs, meta, failed = [], [], {}
        for v, frames, err in decoded:
            if err or not frames:
                failed[v] = err or "no frames decoded"
                continue
            cdir = os.path.join(dest, "ctx", v)
            os.makedirs(cdir, exist_ok=True)
            for i, (t1, t2) in enumerate(fixed_windows(man[v]["duration"], UNIT)):
                files = Q.uniform(Q.frames_in(frames, t1, t2), CLIP_LEN)
                key, wa, grid, info = ctx.build(files)
                wa_p, grid_p = os.path.join(cdir, "w%04d_wa.jpg" % i), os.path.join(cdir, "w%04d_tc.jpg" % i)
                wa.save(wa_p, quality=95); grid.save(grid_p, quality=95)
                key_p = os.path.join(cdir, "w%04d_key.jpg" % i)
                shutil.copyfile(key, key_p)
                words = L.span_text(segs.get(v, []), t1, t2)
                imgs = [reader_size(Image.open(p).convert("RGB")) for p in (key_p, wa_p, grid_p)]
                for kind, im, text in (("key", imgs[0], ins), ("wa", imgs[1], ins), ("tc", imgs[2], tc)):
                    reqs.append(eng.request([im], user_text(text, words), "%s/%s/w%04d/%s" % (ds, v, i, kind), system=SYSTEM))
                meta.append((v, i, t1, t2, len(files), words, info))
        outs = eng.generate(reqs, args.max_tokens) if reqs else []
        per = {}
        for j, (v, i, t1, t2, nf, words, info) in enumerate(meta):
            o = outs[3 * j:3 * j + 3]
            per.setdefault(v, []).append({"i": i, "start": t1, "end": t2, "n_frames": nf, "words": words, **info,
                                          "answers": {k: x.text for k, x in zip(("key", "wa", "tc"), o)},
                                          "finish": {k: x.finish_reason for k, x in zip(("key", "wa", "tc"), o)}})
        with open(path, "a") as fh:
            for v in chunk:
                fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"],
                                     "error": failed.get(v), "units": per.get(v, [])}) + "\n")
        for v, _, _ in decoded:
            Q.drop_frames("aa_%s_%s" % (ds, v))
        log("chunk %d/%d: %d videos, %d windows, %d failed videos, %.0f s" % (
            ci + 1, len(chunks), len(chunk), len(meta), len(failed), time.time() - t0))
    log("DONE infer %s" % ds)


def cmd_readout(args):
    from PIL import Image
    ds = args.dataset
    dest = out_dir(ds)
    log = Q.log_to(os.path.join(dest, "readout.log"))
    recs = load_records(os.path.join(dest, "units.jsonl"))
    path = os.path.join(dest, "readout.jsonl")
    done = set(load_records(path))
    todo = [v for v in recs if v not in done]
    log("anyanomaly readout %s: %d videos, %d to do; code %s" % (ds, len(recs), len(todo), L.git_version()))
    if not todo:
        log("DONE readout (nothing to do)")
        return
    sc = Q.Scorer()
    ins, tc = instructions(KEYWORD)
    t0, n, no_num = time.time(), 0, 0
    for vi, v in enumerate(todo):
        out = []
        for u in recs[v]["units"]:
            cdir = os.path.join(dest, "ctx", v)
            row = {}
            for kind, text in (("key", ins), ("wa", ins), ("tc", tc)):
                im = reader_size(Image.open(os.path.join(cdir, "w%04d_%s.jpg" % (u["i"], kind))).convert("RGB"))
                ans = u["answers"][kind]
                pre = number_prefix(ans)
                if pre is None:
                    pre = ans.rstrip() + "\nScore: "; no_num += 1
                lp1, lp2 = sc.score_digits([im], user_text(text, u["words"]), pre, system=SYSTEM)
                row[kind] = Q.expected_score(lp1, lp2)
            out.append(row); n += 1
        with open(path, "a") as fh:
            fh.write(json.dumps({"video_id": v, "scores": out}) + "\n")
        if (vi + 1) % 50 == 0:
            log("%d / %d videos, %d windows, %.0f s" % (vi + 1, len(todo), n, time.time() - t0))
    log("DONE readout %s: %d windows, %d answers without a number, %.0f s" % (ds, n, no_num, time.time() - t0))


def cmd_raster(args):
    from scipy.ndimage import gaussian_filter1d
    ds = args.dataset
    dest = out_dir(ds)
    recs = load_records(os.path.join(dest, "units.jsonl"))
    prob = {r["video_id"]: r["scores"] for r in load_records(os.path.join(dest, "readout.jsonl")).values()}
    stats = {"videos": len(recs), "windows": sum(len(r["units"]) for r in recs.values()),
             "finish_length": sum(f == "length" for r in recs.values() for u in r["units"] for f in u["finish"].values()),
             "answers_without_number": sum(number_prefix(a) is None for r in recs.values() for u in r["units"]
                                           for a in u["answers"].values())}
    for readout in ("prob", "text"):
        rows = {}
        method = "anyanomaly_q3vl_" + readout
        for v, r in recs.items():
            n = L.n_frames(r["duration"])
            us = r["units"]
            if r.get("error") or not us:
                rows[v] = L.row(method, ds, v, r["duration"], [], None, __file__, error=r.get("error") or "no windows")
                continue
            if readout == "prob":
                sc = [(s["key"], s["wa"], s["tc"]) for s in prob[v]]
            else:
                sc = [tuple(extract_score(u["answers"][k]) for k in ("key", "wa", "tc")) for u in us]
            val = [ALPHA * a + BETA * b + GAMMA * c for a, b, c in sc]
            curve = L.units_to_4fps([u["start"] for u in us], None, val, n)
            curve = gaussian_filter1d(np.asarray(curve, float), sigma=SIGMA_S * L.RATE)
            rows[v] = L.row(method, ds, v, r["duration"], curve, 1.0 / UNIT, __file__)
        med, failed = L.apply_f2(rows, ds)
        stats["f2_videos_" + readout] = failed
        L.finalize(ds, list(rows.values()), os.path.join(dest, readout), method)
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


def cmd_clean(args):
    shutil.rmtree(os.path.join(out_dir(args.dataset), "ctx"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("infer", "readout", "raster", "clean"))
    ap.add_argument("--dataset", required=True, choices=L.DATASETS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8, help="videos per vLLM batch")
    ap.add_argument("--workers", type=int, default=4, help="ffmpeg decoders")
    ap.add_argument("--max-tokens", type=int, default=50)
    args = ap.parse_args()
    {"infer": cmd_infer, "readout": cmd_readout, "raster": cmd_raster, "clean": cmd_clean}[args.stage](args)


if __name__ == "__main__":
    main()
