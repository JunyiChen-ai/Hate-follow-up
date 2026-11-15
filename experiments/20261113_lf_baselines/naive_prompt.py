#!/usr/bin/env python3
"""Naive-prompt LLM / VLM baselines (README §1, user request 2026-11-13).

The earlier naive rows (experiments/20261008_baselines/{qwen3_text,qwen25vl_winonly}) asked with the nine policy
rules of our Reader and our question wording, and the VLM row also carried our reading instruction. Here the model
is asked one plain question with nothing else: no system message beyond the chat template's default, no rules, no
definition, no reading instruction, no whole-video context.

Arms:
  vlm       Qwen/Qwen3-VL-8B-Instruct (the backbone of our method), one call per window of the fixed 8 s grid:
            the window's 1 fps frames (at most 8; the Reader's pixel cap, src/mllm_judge.py MAX_PIXELS) + its
            Whisper words,
            'Transcript: <words>\\n\\nIs this video clip hateful? Answer "Yes" or "No".'
  vlm_internvl  the `vlm` arm with OpenGVLab/InternVL3_5-8B-HF (2026-11-15, experiments/20261115_rh_hcs_split/README.md):
            same windows, frames, words, question and read-out; one 448 tile per frame (src/mllm_judge.py
            FAMILY_IMAGE_KW, as experiments/20261101_backbones)
  vlm_tax   the `vlm` arm with the question replaced by the HateClipSeg taxonomy question (2026-11-15, brief 5 K3;
            fairness reference for the taxonomy scope read, README §1 reference row):
            'Transcript: <words>\n\nDoes this video clip contain content of any of these kinds: hateful, insulting,
            sexual, violence, or self-harm? Answer "Yes" or "No".'  Everything else identical to `vlm`.
  vlm_text  the same model and question without frames (the input of the old Qwen2.5-VL row), to separate the
            effect of the prompt from that of the frames
  llm       Qwen/Qwen3-8B (thinking off), one call per Whisper large-v3 segment, as the old Qwen3-8B row:
            'Transcript: <segment text>\\n\\nIs this speech hateful? Answer "Yes" or "No".'
Words: Whisper large-v3 segments (src/lf_baselines.py transcript_segments, untimed chunks kept), sliced
proportionally to the window (src/video_inputs.py window_text); "(no speech)" when empty. The forced-aligned words of
our Reader are not used.
Score = log P(Yes) - log P(No) at the first answer position, from one ordinary forward pass in HF transformers
(bf16 weights, fp32 output layer over the Yes / No token ids, as the old rows and the Reader: src/mllm_judge.py
`Judge.margins_fp32`; the text LLM uses the same arithmetic on its causal-LM head).
4 fps: a frame takes the score of the window containing its centre (vlm, vlm_text); for llm, the maximum over the
segments containing its centre, and the score of the same question with "(no speech)" outside every segment (the
old row's rule F1).

Stages: `infer` (GPU, HateVLM env) -> runs/20261113_lf_baselines/naive/<arm>/<DS>/units.jsonl (one line per video,
resumable); `raster` (CPU, HateVideo env) -> predictions.jsonl + canonical evaluator. No label is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_common as Q         # noqa: E402
from vllm_common import L       # noqa: E402

WINDOW = 8.0
FPS = 1.0
MAX_FRAMES = 8
NO_SPEECH = "(no speech)"
VLM_Q = 'Is this video clip hateful? Answer "Yes" or "No".'
LLM_Q = 'Is this speech hateful? Answer "Yes" or "No".'
TAX_Q = ('Does this video clip contain content of any of these kinds: hateful, insulting, sexual, violence, or '
         'self-harm? Answer "Yes" or "No".')
MODELS = {"vlm": "Qwen/Qwen3-VL-8B-Instruct", "vlm_text": "Qwen/Qwen3-VL-8B-Instruct", "llm": "Qwen/Qwen3-8B",
          "vlm_internvl": "OpenGVLab/InternVL3_5-8B-HF", "vlm_tax": "Qwen/Qwen3-VL-8B-Instruct"}
FRAME_ARMS = ("vlm", "vlm_internvl", "vlm_tax")     # arms that see the window's frames


def prompt(arm, text):
    q = LLM_Q if arm == "llm" else TAX_Q if arm == "vlm_tax" else VLM_Q
    return "Transcript: %s\n\n%s" % (text or NO_SPEECH, q)


def out_dir(arm, ds):
    return os.path.join(Q.RUNS, "naive", arm, ds)


def done_ids(path):
    if not os.path.isfile(path):
        return set()
    with open(path) as fh:
        return {json.loads(x)["video_id"] for x in fh if x.strip()}


def windows(duration):
    from src.video_inputs import fixed_windows
    return fixed_windows(duration, WINDOW)


class Reader:
    """One ordinary full forward per prompt; Yes/No log-odds with an fp32 output layer."""

    def __init__(self, arm):
        import torch
        self.torch = torch
        self.arm = arm
        if arm == "llm":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(MODELS[arm])
            self.model = AutoModelForCausalLM.from_pretrained(MODELS[arm], dtype=torch.bfloat16, device_map="cuda:0",
                                                              attn_implementation="sdpa").eval()
            ids = lambda w: sorted({self.tok.encode(v, add_special_tokens=False)[0]
                                    for v in (w, " " + w, w.lower(), " " + w.lower(), w.upper(), " " + w.upper())})
            self.yes_ids, self.no_ids = ids("Yes"), ids("No")
            self.device = self.model.device
        else:
            from src.mllm_judge import Judge
            self.j = Judge(MODELS[arm])

    def render(self, text, n_images):
        if self.arm == "llm":
            return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        msgs = [{"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": text}]}]
        return self.j.render(msgs, add_generation_prompt=True)

    def margin(self, text, image_files=()):
        torch = self.torch
        full = self.render(text, len(image_files))
        with torch.inference_mode():
            if self.arm == "llm":
                ids = self.tok(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
                h = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1:]
                W = self.model.get_output_embeddings().weight
                lg = h.float() @ W[self.yes_ids + self.no_ids].float().T
                ny = len(self.yes_ids)
                return float(torch.logsumexp(lg[0, :ny], 0) - torch.logsumexp(lg[0, ny:], 0))
            images = Q.open_images(image_files)
            enc = self.j.encode(full, images)
            out = self.j.model.model(**self.j.model_inputs(enc), use_cache=False)
            z = self.j.margins_fp32(out.last_hidden_state[0, -1:])[0]
            del out
            return float(z)


def cmd_infer(args):
    arm, ds = args.arm, args.dataset
    dest = out_dir(arm, ds)
    L.start_run_log(dest)
    log = Q.log_to(os.path.join(dest, "infer.log"))
    path = os.path.join(dest, "units.jsonl")
    man = L.manifest(ds)
    vids = L.cohort(ds)
    if args.limit:
        vids = vids[:args.limit]
    done = done_ids(path)
    todo = [v for v in vids if v not in done]
    log("naive %s %s: model %s, %d cohort videos, %d to do; code %s" % (
        arm, ds, MODELS[arm], len(vids), len(todo), L.git_version()))
    if not todo:
        log("DONE infer (nothing to do)")
        return
    segs = L.transcript_segments(ds, {v: man[v]["duration"] for v in vids})
    rd = Reader(arm)
    with open(os.path.join(dest, "prompt.txt"), "w") as fh:
        fh.write(rd.render(prompt(arm, "<words>"), 2 if arm in FRAME_ARMS else 0))

    if arm == "llm":
        t0 = time.time()
        z0 = rd.margin(prompt(arm, NO_SPEECH))
        with open(path, "a") as fh:
            for v in todo:
                us = []
                for s, e, t in segs.get(v, []):
                    t = " ".join((t or "").split())
                    us.append({"start": s, "end": e, "text": t, "z": rd.margin(prompt(arm, t)), "status": "ok"})
                fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"], "error": None,
                                     "z_no_speech": z0, "z_no_speech_status": "ok", "units": us}) + "\n")
        log("%d videos in %.0f s; no-speech z %.4f" % (len(todo), time.time() - t0, z0))
        log("DONE infer %s %s" % (arm, ds))
        return

    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        if arm == "vlm_text":
            return v, [], None
        try:
            return v, Q.decode_frames(man[v]["video_path"], FPS, "naive_%s_%s_%s" % (arm, ds, v)), None
        except Exception as e:  # noqa: BLE001
            return v, [], repr(e)

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    fut = pool.map(decode, chunks[0])
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        decoded = list(fut)
        if ci + 1 < len(chunks):
            fut = pool.map(decode, chunks[ci + 1])
        nwin, failed = 0, 0
        with open(path, "a") as fh:
            for v, frames, err in decoded:
                if arm in FRAME_ARMS and (err or not frames):
                    fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"],
                                         "error": err or "no frames decoded", "units": []}) + "\n")
                    failed += 1
                    Q.drop_frames("naive_%s_%s_%s" % (arm, ds, v))
                    continue
                us = []
                for i, (t1, t2) in enumerate(windows(man[v]["duration"])):
                    text = L.span_text(segs.get(v, []), t1, t2)
                    files = Q.uniform(Q.frames_in(frames, t1, t2), MAX_FRAMES) if arm in FRAME_ARMS else []
                    us.append({"i": i, "start": t1, "end": t2, "n_frames": len(files), "text": text,
                               "z": rd.margin(prompt(arm, text), files), "status": "ok"})
                nwin += len(us)
                fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"], "error": None,
                                     "units": us}) + "\n")
                Q.drop_frames("naive_%s_%s_%s" % (arm, ds, v))
        log("chunk %d/%d: %d videos, %d windows, %d failed videos, %.0f s" % (
            ci + 1, len(chunks), len(chunk), nwin, failed, time.time() - t0))
    log("DONE infer %s %s" % (arm, ds))


def cmd_raster(args):
    arm, ds = args.arm, args.dataset
    dest = out_dir(arm, ds)
    recs = {}
    with open(os.path.join(dest, "units.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["video_id"]] = r
    stats = {"videos": len(recs), "units": 0, "bounded": 0, "videos_without_units": 0}
    rows = {}
    for v, r in recs.items():
        n = L.n_frames(r["duration"])
        us = r["units"]
        stats["units"] += len(us)
        stats["bounded"] += sum(u["status"] != "ok" for u in us)
        if r.get("error"):
            rows[v] = L.row("naive_" + arm, ds, v, r["duration"], [], None, __file__, error=r["error"])
            continue
        if arm == "llm":
            curve = np.full(n, r["z_no_speech"], dtype=np.float64)
            hit = np.zeros(n, dtype=bool)
            centres = (np.arange(n) + 0.5) / L.RATE
            for u in us:
                m = (centres >= u["start"]) & (centres < u["end"])
                curve[m] = np.where(hit[m], np.maximum(curve[m], u["z"]), u["z"])
                hit |= m
            stats["videos_without_units"] += not us
            native = None
        else:
            if not us:
                rows[v] = L.row("naive_" + arm, ds, v, r["duration"], [], None, __file__, error="no windows")
                continue
            curve = L.units_to_4fps([u["start"] for u in us], [u["end"] for u in us], [u["z"] for u in us], n)
            native = 1.0 / WINDOW
        rows[v] = L.row("naive_" + arm, ds, v, r["duration"], curve, native, __file__)
    med, failed = L.apply_f2(rows, ds)
    stats["f2_videos"] = failed
    L.finalize(ds, list(rows.values()), dest, "naive_" + arm)
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("infer", "raster"))
    ap.add_argument("--arm", required=True, choices=tuple(MODELS))
    ap.add_argument("--dataset", required=True, choices=L.DATASETS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8, help="videos decoded ahead")
    ap.add_argument("--workers", type=int, default=4, help="ffmpeg decoders")
    args = ap.parse_args()
    {"infer": cmd_infer, "raster": cmd_raster}[args.stage](args)


if __name__ == "__main__":
    main()
