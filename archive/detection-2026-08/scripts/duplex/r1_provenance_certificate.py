#!/usr/bin/env python3
"""R1 frozen provenance-certificate scorer and analyzer."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/jehc223/Hate-follow-up")
sys.path.insert(0, str(ROOT / "src/duplex"))
sys.path.insert(0, str(ROOT / "src/our_method"))

from extract_duplex_readout import MAX_PIXELS, MIN_PIXELS, build_messages
from score_duplex_probe import (BILIBILI_RULES, YOUTUBE_RULES,
                                build_binary_token_ids)

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
OUT = ROOT / "results/r1_provenance_certificate"
SCORES = OUT / "scores.jsonl"
REPORT = OUT / "report.json"
STATUS = OUT / "STATUS"
DONE = OUT / "DONE"
VIEWS = ("full", "payload_only", "no_payload", "source_shift", "nuisance")
FRAME_ROOTS = {
    "HateMM": Path("/home/jehc223/data/HateMM/frames_16"),
    "MHC_zh": Path("/home/jehc223/data/Multihateclip/Chinese/frames_16"),
}


def strip_html(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(text or ""))).strip()


def load_hatemm_payloads():
    best = {}
    path = ROOT / "results/hatemm_localization/per_chunk.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        vid, z = r["video_id"], float(r["z_isolated"])
        key = (z, -int(r["chunk_index"]))
        if vid not in best or key > best[vid][0]:
            best[vid] = (key, r.get("text", ""))
    return {k: v[1] for k, v in best.items()}


def build_items():
    items = []
    amap = json.loads((ROOT / "results/hatemm_fp_audit/alias_map.json").read_text())
    with (ROOT / "results/hatemm_fp_audit/coding.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["primary"] in {"Q", "D"}:
                items.append({"dataset": "HateMM", "video_id": amap[r["alias"]],
                              "stratum": "mention" if r["primary"] == "Q" else "asserted",
                              "audit_code": r["primary"]})

    packet = {r["alias"]: r for r in json.loads(
        (ROOT / "results/ranking_autopsy/zh/packet.json").read_text())}
    with (ROOT / "results/ranking_autopsy/zh/coding_zh.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            keep_assert = r["side"] == "FN" and r["code"] in {"G", "I"} and r["protected"] in {"explicit", "implicit"}
            keep_mention = r["side"] == "FP" and r["code"] in {"Q", "R", "P"}
            if keep_assert or keep_mention:
                p = packet[r["alias"]]
                items.append({"dataset": "MHC_zh", "video_id": p["video_id"],
                              "stratum": "asserted" if keep_assert else "mention",
                              "audit_code": r["code"], "override": p.get("judge_text", "")})
    return items


def load_manifest_annotations():
    out = {}
    path = ROOT / "results/label_free_adapt/manifests/all_test.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["dataset"], r["video_id"])] = {
            "title": r.get("title", ""), "transcript": r.get("transcript", "")}
    # Ranking packet retains the exact title/transcript seen by the frozen ZH judge.
    for r in json.loads((ROOT / "results/ranking_autopsy/zh/packet.json").read_text()):
        out[("MHC_zh", r["video_id"])] = {
            "title": r.get("title", ""), "transcript": r.get("judge_text", "")}
    return out


def frame_paths(dataset, video_id, n=16):
    paths = sorted((FRAME_ROOTS[dataset] / video_id).glob("*.jpg"))
    if len(paths) > n:
        idx = np.linspace(0, len(paths) - 1, n, dtype=int)
        paths = [paths[i] for i in idx]
    return [str(p) for p in paths]


def remove_once(text, payload):
    if not payload or payload not in text:
        return text
    return text.replace(payload, "", 1).strip()


def nuisance_text(full, payload):
    context = remove_once(full, payload)
    n = min(len(payload), len(context))
    if n == 0:
        return full
    # Fixed opposite-end deletion: payload location chooses the other end.
    if full.find(payload) < len(full) / 2:
        trimmed = context[:-n]
    else:
        trimmed = context[n:]
    return (payload + " " + trimmed).strip()


def prepare_views(item, ann, payloads):
    title = ann.get("title", "") or ""
    transcript = item.get("override") or ann.get("transcript", "") or ""
    if item["dataset"] == "HateMM":
        payload = payloads.get(item["video_id"], "")
        full_text = transcript
        title_payload = False
    else:
        clean_title = strip_html(title)
        payload = clean_title or transcript[len(transcript)//4:3*len(transcript)//4]
        full_text = transcript
        title_payload = bool(clean_title)

    base = {"title": title, "transcript": full_text}
    po = {"title": title if title_payload else "", "transcript": "" if title_payload else payload}
    npv = {"title": "" if title_payload else title,
           "transcript": full_text if title_payload else remove_once(full_text, payload)}
    nui = {"title": title if title_payload else "",
           "transcript": "" if title_payload else nuisance_text(full_text, payload)}
    return payload, {"full": base, "payload_only": po, "no_payload": npv,
                     "source_shift": base, "nuisance": nui}


def score_one(model, processor, yes_idx, no_idx, ann, frames, transcript, rules):
    messages = build_messages(ann, frames, rules, transcript)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [Image.open(p).convert("RGB") for p in frames]
    inputs = processor(text=[text], images=images, return_tensors="pt",
                       size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}).to(model.device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        z = float(torch.logsumexp(logits[yes_idx], 0) - torch.logsumexp(logits[no_idx], 0))
        hidden = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
    for im in images:
        im.close()
    return z, hidden


def completed():
    done = set()
    if SCORES.exists():
        for line in SCORES.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["dataset"], r["video_id"], r["view"]))
    return done


def run(limit=0):
    OUT.mkdir(parents=True, exist_ok=True)
    items = build_items()
    if limit:
        items = items[:limit]
    payloads = load_hatemm_payloads()
    anns = load_manifest_annotations()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    ids = build_binary_token_ids(processor.tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True).eval()
    yes_idx = torch.tensor(sorted(ids["Yes"]), device=model.device)
    no_idx = torch.tensor(sorted(ids["No"]), device=model.device)
    done = completed(); total = len(items) * len(VIEWS); n = len(done); t0 = time.time()
    for item in items:
        ann0 = anns[(item["dataset"], item["video_id"])]
        payload, views = prepare_views(item, ann0, payloads)
        base_frames = frame_paths(item["dataset"], item["video_id"], 16)
        if len(base_frames) < 2:
            raise RuntimeError(f"missing frames: {item}")
        rules = BILIBILI_RULES if item["dataset"] == "MHC_zh" else YOUTUBE_RULES
        for view in VIEWS:
            key = (item["dataset"], item["video_id"], view)
            if key in done:
                continue
            v = views[view]
            frames = base_frames[8:] + base_frames[:8] if view == "source_shift" else base_frames
            ann = dict(ann0); ann["title"] = v["title"]
            z, hidden = score_one(model, processor, yes_idx, no_idx, ann, frames, v["transcript"], rules)
            hp = OUT / "hidden" / item["dataset"] / item["video_id"]
            hp.mkdir(parents=True, exist_ok=True)
            np.save(hp / f"{view}.npy", hidden.astype(np.float16))
            rec = {**item, "view": view, "z": z, "payload_chars": len(payload),
                   "transcript_chars": len(v["transcript"]), "title_chars": len(v["title"])}
            with SCORES.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
            n += 1
            STATUS.write_text(f"{n}/{total} {item['dataset']} {item['video_id']} {view} z={z:+.3f}\n")
            print(STATUS.read_text().strip(), flush=True)
    DONE.write_text("ok\n")
    print(f"complete in {(time.time()-t0)/60:.1f} min")


def loo_residual(z, c):
    out = np.zeros_like(c, dtype=float)
    for i in range(len(c)):
        m = np.arange(len(c)) != i
        X = np.c_[np.ones(m.sum()), z[m]]
        b = np.linalg.lstsq(X, c[m], rcond=None)[0]
        out[i] = c[i] - np.array([1.0, z[i]]) @ b
    return out


def analyze():
    rows = [json.loads(x) for x in SCORES.read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["video_id"]), {"meta": r})[r["view"]] = r["z"]
    report = {"preregistered": True, "corpora": {}, "decision": {}}
    survive = True
    for ds in ("HateMM", "MHC_zh"):
        rr = [v for (d, _), v in by.items() if d == ds and all(k in v for k in VIEWS)]
        y = np.array([1 if v["meta"]["stratum"] == "asserted" else 0 for v in rr])
        full = np.array([v["full"] for v in rr])
        context = full - np.array([v["payload_only"] for v in rr])
        binding = full - np.array([v["source_shift"] for v in rr])
        nuisance = np.abs(full - np.array([v["nuisance"] for v in rr]))
        cert = context + binding - nuisance
        deletion = full - np.array([v["no_payload"] for v in rr])
        residual = loo_residual(full, cert)
        metrics = {"n": len(rr), "n_asserted": int(y.sum()), "n_mention": int((1-y).sum())}
        for name, score in [("original", full), ("context", context), ("binding", binding),
                            ("deletion", deletion), ("certificate", cert),
                            ("certificate_residual", residual)]:
            metrics[f"auc_{name}"] = float(roc_auc_score(y, score))
        metrics["pass_residual_ge_070"] = metrics["auc_certificate_residual"] >= .70
        metrics["pass_certificate_beats_deletion"] = metrics["auc_certificate"] > metrics["auc_deletion"]
        report["corpora"][ds] = metrics
        survive &= metrics["pass_residual_ge_070"] and metrics["pass_certificate_beats_deletion"]
    report["decision"] = {"survive": bool(survive), "verdict": "SURVIVE" if survive else "KILL",
                          "rule": "both residual AUC>=0.70 and certificate beats deletion"}
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["run", "analyze"], default="run")
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    run(a.limit) if a.mode == "run" else analyze()


if __name__ == "__main__":
    main()
