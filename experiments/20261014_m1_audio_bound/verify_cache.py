#!/usr/bin/env python3
"""Candidate 43, follow-up of README §6 check 1. The smoke run showed cached-branch vs standalone margins differing by
0.04-0.55 log-odds on every branch kind (text and audio alike), far above the declared 1e-3. This job separates a
plumbing error from bf16 / mixture-of-experts numerics by reading the same windows four ways on two smoke videos:

  A   cached branch on the normal prefix cache (the measurement path);
  B   cached branch on a prefix cache prefilled in two chunks (split right after the last image token): same logical
      computation as A, different kernel shapes -> the numeric floor of the cached path;
  S   standalone full forward (the model's own rope index), as in the smoke diagnostic;
  S2  standalone full forward with the sequence duplicated along the batch dimension (row 0 read): same logical
      computation as S, different kernel shapes -> the numeric floor of the standalone path;
  R   A repeated (determinism of the cached path).

If |A-S| is of the same size as |A-B| and |S-S2|, the smoke difference is numerics and the cached path is faithful;
if |A-S| is much larger than both floors, the cached path is wrong. Also reports the Spearman correlation and sign
agreement between A and S across the windows of each video. GPU only inside Slurm; no label file is read.
Output: runs/20261014_m1_audio_bound/verify_cache/{<dataset>_<video_id>.json, summary.json, run.log}.
"""
import json
import logging
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_audio import NATIVE_FRAMES, WINDOW_S, clip, load_wave, speech_audio_parts  # noqa: E402
from omni_judge import MODEL, SR, OmniJudge, build, media_margin, standalone, text_margin, tick  # noqa: E402
from src.mllm_judge import VIDEO_QUESTION, yesno_question  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402

VIDEOS = (("HateMM", "hate_video_1"), ("HateClipSeg", "bit_0EHvMSiEHVoc"))
OUT = ROOT / "runs/20261014_m1_audio_bound/verify_cache"


@torch.no_grad()
def build_split(j, frames, segments):
    """`omni_judge.build` with the prefix prefilled in two chunks (split after the last image token)."""
    msgs, files = j.prefix_messages(frames, segments)
    text = j.render(msgs, add_generation_prompt=False)
    enc = j.encode_text_with_media(text, files)
    ids = enc["input_ids"][0]
    k = int(torch.nonzero(ids == j.image_token_id).max()) + 1
    pos, _ = j.positions(enc)
    kw = j.model_inputs(enc)
    media = {kk: v for kk, v in kw.items() if kk not in ("input_ids", "attention_mask")}
    j.model.rope_deltas = None
    j._hidden = None
    out = j.model(input_ids=kw["input_ids"][:, :k], attention_mask=kw["attention_mask"][:, :k], position_ids=pos[:, :, :k],
                  past_key_values=None, use_cache=True, use_audio_in_video=False, **media)
    cache = out.past_key_values
    del out
    assert cache.get_seq_length() == k, (cache.get_seq_length(), k)
    out = j.model(input_ids=kw["input_ids"][:, k:], position_ids=pos[:, :, k:], past_key_values=cache, use_cache=True, use_audio_in_video=False)
    del out
    prefix_tokens = int(ids.shape[0])
    assert cache.get_seq_length() == prefix_tokens
    start = int(pos.max()) + 1
    qcontent = [{"type": "text", "text": VIDEO_QUESTION}]
    qid, qtext = j.branch_ids(text, msgs, qcontent)
    h, start2 = j.text_step(cache, qid, start)
    z = j.margins_fp32(h[None])[0]
    stance = "Yes" if z > 0 else "No"
    aid, atext = j.answer_ids(msgs, qcontent, stance)
    _, start3 = j.text_step(cache, aid, start2)
    history = [{"role": "user", "content": qcontent}, j.turn("assistant", stance)]
    return cache, dict(msgs=msgs, files=files, history=history, head=text + qtext + atext, logical_start=start3, global_margin=z, stance=stance,
                       stance_cache_tokens=cache.get_seq_length(), prefix_tokens=prefix_tokens, split_at=k, prefix_text=text)


@torch.no_grad()
def batched_standalone(j, ctx, parts, image_files=(), audios=()):
    """`omni_judge.standalone` with the whole sequence duplicated along the batch dimension; row 0 is read."""
    full = j.render(ctx["msgs"] + ctx["history"] + [{"role": "user", "content": parts}], True)
    files = list(ctx["files"]) + list(image_files)
    images = j.open_images(files + files)
    try:
        kw = dict(text=[full, full], images=images, return_tensors="pt", add_special_tokens=False, use_audio_in_video=False, **j.img_kw)
        if audios:
            kw["audio"] = [np.asarray(a, dtype=np.float32) for a in list(audios) * 2]
            kw["sampling_rate"] = SR
        enc = j.processor(**kw)
    finally:
        for im in images:
            im.close()
    assert enc["input_ids"].shape[0] == 2 and bool(enc["attention_mask"].all())
    j.model.rope_deltas = None
    h = j.forward_hidden(enc)
    return j.margins_fp32(h[None])[0]


def stats(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = np.abs(x - y)
    return dict(mean_abs=float(d.mean()), max_abs=float(d.max()), spearman=float(spearmanr(x, y)[0]) if len(x) > 2 else None,
                sign_agreement=float(np.mean(np.sign(x) == np.sign(y))))


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    rows = {(r["dataset"], r["video_id"]): r for r in load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", ("HateMM", "HateClipSeg"))}
    asr = {ds: load_asr(ds) for ds in ("HateMM", "HateClipSeg")}
    torch.manual_seed(0)
    j = OmniJudge(MODEL)
    summary = {}
    for ds, vid in VIDEOS:
        row = rows[(ds, vid)]
        segments = asr[ds].get(vid, [])
        native = frame_paths(ds, vid, NATIVE_FRAMES)
        wav = load_wave(row)
        cacheA, ctxA = build(j, native, segments)
        cacheB, ctxB = build_split(j, native, segments)
        wins = fixed_windows(float(row["duration"]), WINDOW_S)
        n = len(wins)
        per = {k: {"text": [], "audio": []} for k in ("A", "B", "S", "S2", "R")}
        stamp = tick(j)
        for i, (t1, t2) in enumerate(wins):
            body = window_text(segments, t1, t2)
            vq = yesno_question(i, n, t1, t2, body, "visual")
            tparts = [{"type": "text", "text": vq}]
            aparts, aud = speech_audio_parts(i, n, t1, t2), [clip(wav, t1, t2)]
            per["A"]["text"].append(text_margin(j, cacheA, ctxA, vq)[0])
            per["A"]["audio"].append(media_margin(j, cacheA, ctxA, aparts, (), aud)[0])
            per["B"]["text"].append(text_margin(j, cacheB, ctxB, vq)[0])
            per["B"]["audio"].append(media_margin(j, cacheB, ctxB, aparts, (), aud)[0])
            per["S"]["text"].append(standalone(j, ctxA, tparts)[0])
            per["S"]["audio"].append(standalone(j, ctxA, aparts, (), aud)[0])
            per["S2"]["text"].append(batched_standalone(j, ctxA, tparts))
            per["S2"]["audio"].append(batched_standalone(j, ctxA, aparts, (), aud))
            per["R"]["text"].append(text_margin(j, cacheA, ctxA, vq)[0])
            per["R"]["audio"].append(media_margin(j, cacheA, ctxA, aparts, (), aud)[0])
            logging.info("%s/%s window %d/%d A=%.3f/%.3f B=%.3f/%.3f S=%.3f/%.3f S2=%.3f/%.3f", ds, vid, i + 1, n, per["A"]["text"][-1], per["A"]["audio"][-1],
                         per["B"]["text"][-1], per["B"]["audio"][-1], per["S"]["text"][-1], per["S"]["audio"][-1], per["S2"]["text"][-1], per["S2"]["audio"][-1])
        comp = {kind: {"A_vs_S": stats(per["A"][kind], per["S"][kind]), "A_vs_B": stats(per["A"][kind], per["B"][kind]),
                       "S_vs_S2": stats(per["S"][kind], per["S2"][kind]), "A_vs_R": stats(per["A"][kind], per["R"][kind]),
                       "B_vs_S": stats(per["B"][kind], per["S"][kind])} for kind in ("text", "audio")}
        rec = dict(dataset=ds, video_id=vid, windows=n, stance=dict(A=ctxA["stance"], B=ctxB["stance"]), global_margin=dict(A=ctxA["global_margin"], B=ctxB["global_margin"]),
                   prefix_tokens=ctxA["prefix_tokens"], split_at=ctxB["split_at"], values=per, comparisons=comp, seconds=tick(j) - stamp,
                   peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, GT_read=False, host=socket.gethostname(), model=MODEL, date=time.strftime("%Y-%m-%d"))
        (OUT / f"{ds}_{vid}.json").write_text(json.dumps(rec, indent=1) + "\n")
        summary[f"{ds}/{vid}"] = dict(windows=n, stance=rec["stance"], comparisons=comp, seconds=rec["seconds"], peak_GiB=rec["peak_GiB"])
        logging.info("%s/%s done: %s", ds, vid, json.dumps({k: {c: round(v["mean_abs"], 4) for c, v in comp[k].items()} for k in comp}))
        del cacheA, cacheB
        torch.cuda.empty_cache()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    logging.info("VERIFY_CACHE_DONE")


if __name__ == "__main__":
    main()
