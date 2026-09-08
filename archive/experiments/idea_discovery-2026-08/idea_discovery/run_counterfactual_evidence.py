#!/usr/bin/env python3
"""Stage-A counterfactual multimodal evidence localization.

Candidate generation is label-free.  A frozen Qwen3-VL judge scores the whole
video and actual keep/remove/shuffle interventions.  The selected interval is
therefore a prediction, never ground truth or a model-produced explanation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import (Interval, Prediction, append_jsonl,
                                             curve_to_intervals, intervals_to_curve)

GT = Path("/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps")
MANIFEST = ROOT / "results/label_free_adapt/manifests/all_test.jsonl"
COHORT = ROOT / "results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl"
ASR = {
    dataset: ROOT / f"results/idea_discovery/paradigm_adapt/asr_inference_only/{dataset}.jsonl"
    for dataset in ("HateMM", "HateClipSeg", "MHC", "MHC_zh")
}
STAMP = {
    "HateClipSeg": ROOT / "results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl",
    "MHC": ROOT / "results/interleaved_timeline/mhclip_en/timestamped_chunks.jsonl",
    "MHC_zh": ROOT / "results/interleaved_timeline/mhclip_zh/timestamped_chunks.jsonl",
}
DATASETS = tuple(ASR)


def rows(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def ecdf(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x), float)
    out[order] = (np.arange(len(x)) + .5) / max(1, len(x))
    return out


def transcript_rows():
    out = {}
    stamped = {}
    for dataset, path in STAMP.items():
        for row in rows(path):
            stamped[(dataset, row["video_id"])] = row.get("chunks", [])
    for dataset, path in ASR.items():
        for row in rows(path):
            forbidden = set(row) - {"video_id", "chunk_index", "span", "text", "z_masked", "z_isolated"}
            if forbidden:
                raise RuntimeError(f"label/GT-bearing ASR fields rejected: {forbidden}")
            if not row.get("text"):
                source = stamped.get((dataset, row["video_id"]), [])
                index = int(row.get("chunk_index", -1))
                row["text"] = source[index].get("text", "") if 0 <= index < len(source) else ""
            out.setdefault((dataset, row["video_id"]), []).append(row)
    for value in out.values():
        value.sort(key=lambda x: tuple(map(float, x["span"])))
    return out


def freeze_text_units(items, max_chars=6000):
    """Freeze one budgeted ASR sequence before any intervention partition."""
    units = []
    remaining = max_chars
    for row in items:
        text = str(row.get("text", "")).strip()
        if not text or remaining <= 0:
            continue
        text = text[:remaining]
        a, b = map(float, row["span"])
        units.append((a, b, text)); remaining -= len(text) + 1
    return units


def partition_text(units, start, end):
    inside, outside = [], []
    for a, b, text in units:
        (inside if start <= (a + b) / 2 < end else outside).append(text)
    return " ".join(inside), " ".join(outside)


def sample_frame_grid(path, duration, n):
    """Decode one frozen timestamp grid; interventions only subset this grid."""
    times = np.linspace(0, duration, n + 2)[1:-1]
    cap = cv2.VideoCapture(str(path)); output = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = cap.read()
        if ok:
            output.append((float(t), Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return output


def intervention_frames(grid, start, end, inside):
    return [frame for timestamp, frame in grid if ((start <= timestamp < end) == inside)]


def snap_candidates(spans, grid, text_units, duration):
    """Snap proposals to frozen visual/ASR unit cuts and guarantee visual support."""
    times = np.asarray([timestamp for timestamp, _ in grid], float)
    cuts = {0.0, float(duration)}
    for a, b, _ in text_units:
        cuts.update((max(0., min(duration, a)), max(0., min(duration, b))))
    if len(times):
        cuts.update(((times[:-1] + times[1:]) / 2).tolist())
    cuts = np.asarray(sorted(cuts), float)
    output = []
    for start, end in spans:
        a = float(cuts[np.argmin(abs(cuts - start))]); b = float(cuts[np.argmin(abs(cuts - end))])
        if b <= a or not np.any((times >= a) & (times < b)):
            center = (start + end) / 2
            index = int(np.argmin(abs(times - center)))
            a = 0.0 if index == 0 else float((times[index - 1] + times[index]) / 2)
            b = duration if index == len(times) - 1 else float((times[index] + times[index + 1]) / 2)
        if not np.any((times >= a) & (times < b)):
            raise RuntimeError(f"candidate has no frozen visual sample: {(a, b)}")
        if (a, b) not in output:
            output.append((a, b))
    return output


def binary_ids(tokenizer):
    result = {"Yes": set(), "No": set()}
    for label in result:
        for form in (label, " " + label, label.lower(), " " + label.lower()):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                result[label].add(ids[0])
    if not result["Yes"] or not result["No"] or result["Yes"] & result["No"]:
        raise RuntimeError(f"invalid Yes/No token ids: {result}")
    return result


class Judge:
    def __init__(self, model_name):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True,
                                                       max_pixels=448 * 448)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True).eval()
        ids = binary_ids(self.processor.tokenizer)
        self.yes = torch.tensor(sorted(ids["Yes"]), device=self.model.device)
        self.no = torch.tensor(sorted(ids["No"]), device=self.model.device)
        self.calls = 0

    @torch.inference_mode()
    def score(self, frames, transcript, condition):
        policy = (
            "Hateful content means an asserted or endorsed attack, dehumanization, threat, "
            "exclusion, or hateful stereotype directed at a person or protected/social group. "
            "Quotation, reporting, counterspeech, satire, and condemnation are not hateful "
            "unless the video endorses the hostility."
        )
        question = (f"Evidence condition: {condition}. Timestamp-aligned speech: "
                    f"{json.dumps(transcript, ensure_ascii=False)}\n{policy}\n"
                    "Does the supplied evidence contain hateful content? Answer Yes or No.")
        content = [{"type": "image", "image": x} for x in frames]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=frames or None,
                                return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        self.calls += 1
        return float(torch.logsumexp(logits[self.yes], 0) -
                     torch.logsumexp(logits[self.no], 0))


def candidates(curves, duration, top_k):
    proposals = set()
    for curve in curves:
        x = ecdf(curve); n = len(x)
        peaks = np.argsort(x)[::-1][:max(6, top_k * 3)]
        for center in peaks:
            for frac in (.10, .20, .35):
                width = max(2, int(round(n * frac)))
                lo = max(0, center - width // 2); hi = min(n, lo + width)
                lo = max(0, hi - width)
                proposals.add((lo / n * duration, hi / n * duration))
    scored = []
    for start, end in proposals:
        means = []
        for curve in curves:
            lo = int(start / duration * len(curve)); hi = max(lo + 1, int(math.ceil(end / duration * len(curve))))
            means.append(float(ecdf(curve)[lo:hi].mean()))
        scored.append((np.mean(means) - .10 * (end - start) / duration, start, end))
    return [(a, b) for _, a, b in sorted(scored, reverse=True)[:top_k]]


def poset_curve(visual, chunks, duration, margin=.4, visual_budget=.05):
    n = len(visual); spans = []; confidence = []
    for row in chunks:
        lo = max(0, min(n - 1, int(float(row["span"][0]) / duration * n)))
        hi = min(n, max(lo + 1, int(math.ceil(float(row["span"][1]) / duration * n))))
        spans.append((lo, hi))
        z = float(row.get("z_masked", row.get("z_isolated", -20)))
        confidence.append(1 / (1 + math.exp(-np.clip(z / 4, -30, 30))))
    if not spans:
        return ecdf(visual)
    confidence = np.asarray(confidence); total = np.zeros(n); count = np.zeros(n)
    for value, (lo, hi) in zip(confidence, spans):
        total[lo:hi] += value; count[lo:hi] += 1
    base = np.where(count > 0, total / np.maximum(count, 1), .5) + visual_budget * (ecdf(visual) - .5)
    edges = [(i, j) for i in range(len(spans)) for j in range(len(spans))
             if confidence[i] - confidence[j] >= margin]
    if not edges:
        return base
    membership = np.zeros((n, len(spans))); overlap = np.zeros(n)
    for k, (lo, hi) in enumerate(spans): membership[lo:hi, k] = 1; overlap[lo:hi] += 1
    membership /= np.maximum(overlap[:, None], 1)
    op = np.stack([membership[lo:hi].mean(0) for lo, hi in spans])
    means = np.asarray([base[lo:hi].mean() for lo, hi in spans])
    edge_op = np.stack([op[i] - op[j] for i, j in edges]); edge_base = np.asarray([means[i] - means[j] for i, j in edges])
    fit = minimize(lambda d: .5 * float(d @ d), np.zeros(len(spans)), jac=lambda d: d,
                   constraints={"type": "ineq", "fun": lambda d: edge_base + edge_op @ d,
                                "jac": lambda d: edge_op}, method="SLSQP")
    return base + membership @ fit.x if fit.success else base


def fuse(base, evidence):
    base = ecdf(base); out = .75 * base
    for item in evidence:
        lo = int(item["start"] / item["duration"] * len(out))
        hi = max(lo + 1, int(math.ceil(item["end"] / item["duration"] * len(out))))
        out[lo:hi] += .25 * item["score"]
    return np.clip(out, 0, 1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--per-dataset", type=int, default=2)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--top-k", type=int, default=3); ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--t3al-curves", type=Path,
                    default=ROOT / "results/idea_discovery/paradigm_adapt/stage_a_t3al_clean")
    ap.add_argument("--topology-curves", type=Path,
                    default=ROOT / "results/idea_discovery/paradigm_adapt/stage_a_topology_clean")
    ap.add_argument("--seed", type=int, default=20260826); args = ap.parse_args()
    torch.manual_seed(args.seed)
    manifest = {(x["dataset"], x["video_id"]): x for x in rows(MANIFEST)}
    cohort_ids = {d: [] for d in DATASETS}
    for x in rows(args.cohort):
        if len(cohort_ids[x["dataset"]]) < args.per_dataset:
            cohort_ids[x["dataset"]].append(x["video_id"])
    chunks = transcript_rows(); judge = Judge(args.model); output = Path(args.out)
    fingerprint_payload = {"model": args.model, "top_k": args.top_k, "frames": args.frames,
                           "seed": args.seed, "cohort": args.cohort.read_text(),
                           "t3al_curves": str(args.t3al_curves.resolve()),
                           "topology_curves": str(args.topology_curves.resolve())}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    existing = rows(output) if output.exists() else []
    bad = [x for x in existing if x.get("raw", {}).get("run_fingerprint") != fingerprint]
    if bad:
        raise RuntimeError(f"output contains {len(bad)} records from a different run configuration")
    done = {(x["dataset"], x["video_id"], x["method"]) for x in existing}
    expected = {(dataset, video_id, method) for dataset in DATASETS
                for video_id in cohort_ids[dataset]
                for method in ("cf_evidence_only", "t3al_cf", "poset_cf")}
    for dataset in DATASETS:
        for video_id in cohort_ids[dataset]:
            variant_names = {"cf_evidence_only", "t3al_cf", "poset_cf"}
            if all((dataset, video_id, name) in done for name in variant_names):
                continue
            row = manifest[(dataset, video_id)]; duration = float(row["duration"]); media = row["video_path"]
            fixed_path = args.t3al_curves / dataset / f"{video_id}.npy"
            topology_path = args.topology_curves / dataset / f"{video_id}.npy"
            if not fixed_path.exists() or not topology_path.exists():
                raise FileNotFoundError(f"missing required curves: {fixed_path}, {topology_path}")
            t3al = np.load(fixed_path); topology = np.load(topology_path)
            poset = poset_curve(topology, chunks.get((dataset, video_id), []), duration)
            grid = sample_frame_grid(media, duration, args.frames)
            if len(grid) != args.frames:
                raise RuntimeError(f"decoded {len(grid)}/{args.frames} frozen frames for {dataset}/{video_id}")
            text_units = freeze_text_units(chunks.get((dataset, video_id), []))
            spans = snap_candidates(candidates((t3al, poset), duration, args.top_k),
                                    grid, text_units, duration)
            all_frames = [frame for _, frame in grid]
            all_text = " ".join(text for _, _, text in text_units)
            z_full = judge.score(all_frames, all_text, "complete video evidence")
            evidence = []
            for start, end in spans:
                inside_frames = intervention_frames(grid, start, end, True)
                outside_frames = intervention_frames(grid, start, end, False)
                inside_text, outside_text = partition_text(text_units, start, end)
                z_only = judge.score(inside_frames, inside_text, "candidate interval only")
                z_remove = judge.score(outside_frames, outside_text, "candidate interval removed")
                z_visual = judge.score(inside_frames, "", "candidate interval, visual evidence only")
                z_speech = judge.score([], inside_text, "candidate interval, speech evidence only")
                reversed_frames = list(reversed(inside_frames))
                z_shuffle = judge.score(reversed_frames, inside_text, "candidate interval with reversed visual time order")
                sig = lambda z: 1 / (1 + math.exp(-np.clip(z / 4, -30, 30)))
                suff = sig(z_only); necessity = max(0., sig(z_full) - sig(z_remove))
                alignment = max(0., suff - sig(z_shuffle)); balance = min(sig(z_visual), sig(z_speech))
                compact = (end - start) / duration
                score = float(np.clip(.40 * suff + .35 * necessity + .15 * alignment + .10 * balance - .10 * compact, 0, 1))
                evidence.append({"start": start, "end": end, "duration": duration, "score": score,
                                 "sufficiency": suff, "necessity": necessity, "alignment": alignment,
                                 "modality_balance": balance, "z": {"full": z_full, "only": z_only,
                                 "remove": z_remove, "visual": z_visual, "speech": z_speech,
                                 "reverse": z_shuffle}})
            evidence.sort(key=lambda x: x["score"], reverse=True)
            best = evidence[:1]
            evidence_intervals = [Interval(x["start"], x["end"], x["score"]) for x in best]
            pure = intervals_to_curve(evidence_intervals, duration)
            variants = {"cf_evidence_only": pure, "t3al_cf": fuse(t3al, best), "poset_cf": fuse(poset, best)}
            used_calls = 1 + 5 * len(spans)
            for method, curve in variants.items():
                if (dataset, video_id, method) in done: continue
                intervals = (evidence_intervals if method == "cf_evidence_only" else
                             curve_to_intervals(curve, duration, threshold=.75))
                pred = Prediction(method, dataset, video_id, duration, score_curve=curve,
                                  intervals=intervals, modality_evidence={"candidates": evidence},
                                  raw={"base": "none" if method == "cf_evidence_only" else method.split("_")[0],
                                       "run_fingerprint": fingerprint},
                                  calls=used_calls, seed=args.seed)
                append_jsonl(output, pred)
            print(json.dumps({"dataset": dataset, "video_id": video_id,
                              "calls": used_calls, "best": best[0] if best else None}), flush=True)
    final = {(x["dataset"], x["video_id"], x["method"]) for x in rows(output)
             if x.get("raw", {}).get("run_fingerprint") == fingerprint}
    missing = expected - final
    if missing:
        raise RuntimeError(f"incomplete run: {len(missing)} prediction records missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
