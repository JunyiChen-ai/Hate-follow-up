#!/usr/bin/env python3
"""VideoMind (ICLR 2026, arXiv 2503.13444; github.com/yeliudev/VideoMind) as a zero-shot hate localizer.

The released VideoMind-7B checkpoint (Qwen2-VL-7B-Instruct + grounder / verifier LoRA roles) is run exactly as the
authors' temporal-grounding evaluation (`videomind/eval/infer_auto.py`, run by `scripts/evaluation/eval_auto_7b.sh`
for Charades-STA, ActivityNet-Captions, ...): no planner call (`--auto_rephrasing` / `--auto_planning` are off in
that script), grounder over the whole video (1 fps, at most 150 frames, 36..64 x 28 x 28 pixels), up to 100
proposals after the model's own NMS, verifier on the top 5 proposals (2 fps, at most 64 frames, the proposal
widened by half its length on each side, <|seg_start|>/<|seg_end|> inserted, score = sigmoid(logit Yes - logit No)).
The grounder/verifier code below is copied from infer_auto.py with the dataset plumbing removed.

Query (hate_query.md section 2): the short positive query, used as the grounding query and the verifier query.

Frame score (task brief, 2026-10-08): the maximum verifier score over the verified proposals that cover the frame
centre, 0 where no verified proposal covers it. One deviation from infer_auto.py: when the grounder returns a single
proposal, infer_auto.py skips the verifier (`len(pred) > 1`); here the verifier is run on it as well, so that every
video has verifier scores. The count of such videos is reported.

Decoding: VideoMind reads videos with decord. A file decord cannot open is re-encoded to H.264 (same rate and size)
once, read, and the copy deleted (lf_common.transcode_h264); such videos are listed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, BASE)
import lf_common as L                                            # noqa: E402

VM = os.path.join(REPO, "third_party", "VideoMind")
METHOD = "videomind"
OUT_ROOT = os.path.join(REPO, "runs", "20261008_baselines", "videomind")
CODE_PATH = "experiments/20261008_baselines/videomind/videomind_hate.py"
QUERY = "hateful content that attacks or demeans people based on protected attributes"   # hate_query.md 2


def load_done(path):
    out = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r.get("error") and r["video_id"] in out and not out[r["video_id"]].get("error"):
                        continue
                    out[r["video_id"]] = r
    return out


def run_video(model, processor, device, video_path, query, question):
    """infer_auto.py grounder + verifier for one video (VTG setting). Returns a dict."""
    import torch
    from videomind.constants import GROUNDER_PROMPT, VERIFIER_PROMPT
    from videomind.dataset.utils import process_vision_info
    from videomind.utils.io import get_duration
    from videomind.utils.parser import parse_query, parse_span

    duration = get_duration(video_path, num_threads=1)
    query = parse_query(query)
    messages = [{'role': 'user', 'content': [{
        'type': 'video', 'video': video_path, 'num_threads': 1, 'min_pixels': 36 * 28 * 28,
        'max_pixels': 64 * 28 * 28, 'max_frames': 150, 'fps': 1.0}, {
        'type': 'text', 'text': GROUNDER_PROMPT.format(query)}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)
    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    model.set_adapter('grounder')
    output_ids = model.generate(**data, do_sample=False, temperature=None, top_p=None, top_k=None,
                                repetition_penalty=None, max_new_tokens=256)
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    success = len(model.reg) > 0
    if success:
        blob = model.reg[0].cpu().float()
        pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()
        pred = pred.clamp(min=0, max=duration)
        unit = 0.001
        pred = torch.round(pred / unit).long() * unit
        inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
        pred[inds] = pred[inds].roll(1)
        pred = pred.tolist()
    else:
        pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
        conf = [0] * 5

    probs = []
    for cand in pred[:5]:
        s0, e0 = parse_span(cand, duration, 2)
        offset = (e0 - s0) / 2
        s1, e1 = parse_span([s0 - offset, e0 + offset], duration)
        s = (s0 - s1) / (e1 - s1)
        e = (e0 - s1) / (e1 - s1)
        messages = [{'role': 'user', 'content': [{
            'type': 'video', 'video': video_path, 'num_threads': 1, 'video_start': s1, 'video_end': e1,
            'min_pixels': 36 * 28 * 28, 'max_pixels': 64 * 28 * 28, 'max_frames': 64, 'fps': 2.0}, {
            'type': 'text', 'text': VERIFIER_PROMPT.format(question)}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        video_grid_thw = data['video_grid_thw'][0]
        num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
        assert num_frames * window * 4 == data['pixel_values_videos'].size(0)
        pos_s, pos_e = round(s * num_frames), round(e * num_frames)
        pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
        assert pos_s <= pos_e, (num_frames, s, e)
        base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
        pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2
        input_ids = data['input_ids'][0].tolist()
        input_ids.insert(pos_s, model.config.seg_s_token_id)
        input_ids.insert(pos_e, model.config.seg_e_token_id)
        data['input_ids'] = torch.LongTensor([input_ids])
        data['attention_mask'] = torch.ones_like(data['input_ids'])
        data = data.to(device)
        model.base_model.disable_adapter_layers()
        model.base_model.enable_adapter_layers()
        model.set_adapter('verifier')
        with torch.inference_mode():
            logits = model(**data).logits[0, -1].softmax(dim=-1)
        probs.append((logits[9454] - logits[2753]).sigmoid().item())   # Qwen2-VL vocab: 9454 Yes, 2753 No
    return {"duration_decoded": float(duration), "grounder_response": response, "grounder_success": bool(success),
            "n_proposals": len(pred), "pred": pred, "conf": conf, "verified": pred[:5], "probs": probs}


def cmd_infer(args):
    import torch
    from transformers import AutoConfig
    sys.path.insert(0, VM)
    import nncore
    from videomind.model.builder import build_model

    ds = args.dataset
    dest = os.path.join(OUT_ROOT, ds)
    L.start_run_log(dest)
    raw_path = os.path.join(dest, "raw.jsonl")
    done = {v for v, r in load_done(raw_path).items() if not r.get("error")}
    ids = L.cohort(ds)
    if args.limit:
        ids = ids[:args.limit]
    todo = [v for v in ids if v not in done]
    man = L.manifest(ds)
    print("%s: %d cohort, %d done, %d to do" % (ds, len(ids), len(done), len(todo)), flush=True)
    if not todo:
        return 0
    config = AutoConfig.from_pretrained(args.model)
    config.base_model_path = args.base_model
    print("loading", args.model, "base", args.base_model, flush=True)
    model, processor = build_model(args.model, config=config, device="cuda")
    model.load_adapter(nncore.join(args.model, 'verifier'), adapter_name='verifier')
    device = next(model.parameters()).device
    with open(os.path.join(dest, "infer_config.json"), "w") as fh:
        json.dump({"dataset": ds, "model": args.model, "base_model": args.base_model, "query": QUERY,
                   "roles": ["grounder", "verifier"], "planner": "not called (authors' VTG evaluation)",
                   "grounder_video": {"fps": 1.0, "max_frames": 150, "min_pixels": 36 * 28 * 28,
                                      "max_pixels": 64 * 28 * 28},
                   "verifier_video": {"fps": 2.0, "max_frames": 64, "top_k": 5},
                   "dtype": "float16 (build_model default)", "code_version": L.git_version(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    t_start = time.time()
    with open(raw_path, "a") as out:
        for k, vid in enumerate(todo, 1):
            src = man[vid]["video_path"]
            rec = {"video_id": vid, "video_path": src}
            t0 = time.time()
            tmp = None
            try:
                path = src
                if not L.decord_ok(src):
                    tmp = os.path.join(dest, "_transcode", vid + ".mp4")
                    path = L.transcode_h264(src, tmp)
                    rec["transcoded"] = True
                    if not L.decord_ok(path):
                        raise RuntimeError("decord cannot read the file nor its H.264 copy")
                rec.update(run_video(model, processor, device, path, QUERY, QUERY))
            except Exception as exc:                                       # noqa: BLE001
                rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                torch.cuda.empty_cache()
            finally:
                if tmp and os.path.isfile(tmp):
                    os.remove(tmp)
            rec["wall_s"] = round(time.time() - t0, 2)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            el = time.time() - t_start
            print("[%4d/%4d] %-22s %5.1fs props=%s probs=%s eta %.1f min%s" % (
                k, len(todo), vid, rec["wall_s"], rec.get("n_proposals"),
                [round(p, 3) for p in rec.get("probs", [])], (len(todo) - k) * el / k / 60,
                ("  ERROR " + rec["error"][:200]) if rec.get("error") else ""), flush=True)
    print("INFER_DONE %s %.1f min" % (ds, (time.time() - t_start) / 60), flush=True)
    return 0


def cmd_raster(args):
    ds = args.dataset
    dest = os.path.join(OUT_ROOT, ds)
    L.start_run_log(dest)
    man = L.manifest(ds)
    raw = load_done(os.path.join(dest, "raw.jsonl"))
    rows, stats = {}, {"n_proposals": [], "n_single_proposal": 0, "n_grounder_failed": 0, "transcoded": [],
                       "frac_frames_covered": []}
    for vid in L.cohort(ds):
        dur = man[vid]["duration"]
        n = L.n_frames(dur)
        r = raw.get(vid)
        if r is None or r.get("error"):
            rows[vid] = L.row(METHOD, ds, vid, dur, [], "intervals", CODE_PATH,
                              error=(r or {}).get("error", "no raw row"))
            continue
        centres = (np.arange(n) + 0.5) / L.RATE
        curve = np.zeros(n)
        covered = np.zeros(n, dtype=bool)
        for (s, e), p in zip(r["verified"], r["probs"]):
            m = (centres >= s) & (centres <= e)
            curve[m] = np.maximum(curve[m], p)
            covered |= m
        stats["n_proposals"].append(r["n_proposals"])
        stats["n_single_proposal"] += int(r["n_proposals"] == 1)
        stats["n_grounder_failed"] += int(not r["grounder_success"])
        stats["frac_frames_covered"].append(float(covered.mean()))
        if r.get("transcoded"):
            stats["transcoded"].append(vid)
        intervals = [[float(s), float(e), float(p)] for (s, e), p in zip(r["verified"], r["probs"])]
        rows[vid] = L.row(METHOD, ds, vid, dur, curve, "intervals", CODE_PATH, intervals=intervals,
                          extra={"n_proposals": r["n_proposals"], "n_verified": len(r["probs"]),
                                 "grounder_success": r["grounder_success"], "transcoded": bool(r.get("transcoded"))},
                          calls=1 + len(r["probs"]))
    med, failed = L.apply_f2(rows, ds)
    npr = np.array(stats["n_proposals"])
    summary = {"n_videos": len(rows), "n_f2": len(failed), "f2": failed, "f2_median": med,
               "proposals_per_video_mean": float(npr.mean()) if len(npr) else None,
               "proposals_per_video_median": float(np.median(npr)) if len(npr) else None,
               "proposals_per_video_min": int(npr.min()) if len(npr) else None,
               "proposals_per_video_max": int(npr.max()) if len(npr) else None,
               "n_single_proposal": stats["n_single_proposal"], "n_grounder_failed": stats["n_grounder_failed"],
               "frac_frames_covered_mean": float(np.mean(stats["frac_frames_covered"])),
               "transcoded": stats["transcoded"]}
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(dest, "run.log"), "a") as fh:
        fh.write("raster: %s\n" % json.dumps({k: v for k, v in summary.items() if k != "f2"}))
        for vid, why in failed:
            fh.write("F2 %s: %s\n" % (vid, why))
    print(json.dumps({k: v for k, v in summary.items() if k != "f2"}, indent=1))
    if len(failed) > 0.01 * len(rows):
        print("STOP: %d F2 videos > 1%% of %d" % (len(failed), len(rows)))
        return 1
    with open(os.path.join(dest, "config.json"), "w") as fh:
        json.dump({"method": METHOD, "dataset": ds, "code": CODE_PATH, "code_version": L.git_version(),
                   "query": QUERY, "cohort": "runs/20261008_baselines/cohort/%s.txt" % ds,
                   "infer_config": "infer_config.json",
                   "frame_score": "max verifier probability over the verified (top-5) proposals covering the 4 fps "
                                  "frame centre; 0 elsewhere",
                   "intervals_field": "[start, end, verifier prob] of the verified proposals",
                   "fallback_F2": "median frame score of scored videos", "date": time.strftime("%Y-%m-%d")},
                  fh, indent=2)
    L.finalize(ds, list(rows.values()), dest, METHOD)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("infer", "raster"))
    ap.add_argument("--dataset", choices=L.DATASETS, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="VideoMind-7B snapshot directory")
    ap.add_argument("--base-model", default=None, help="Qwen2-VL-7B-Instruct snapshot directory")
    args = ap.parse_args()
    return {"infer": cmd_infer, "raster": cmd_raster}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
