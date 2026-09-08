#!/usr/bin/env python3
"""A08: adapt the released VideoTGB/LSTP temporal bridge to hate localization."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .run_numpro import load_done
from .schema import Prediction, append_jsonl, intervals_to_curve, normalize_intervals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sampler-model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20250819)
    args = ap.parse_args()

    from eval.utils.builder_utils import get_frames, load_pretrained_model
    model, processor, sampler_processor = load_pretrained_model(
        args.checkpoint, args.base_model, args.sampler_model, "cuda:0", False)
    model = model.to("cuda:0", dtype=torch.bfloat16).eval()
    # The released RAFT path disables autocast and constructs fp32 coordinate
    # tensors internally; keep the temporal bridge in fp32 to match it.
    model.of_extractor.float()
    model.temporal_encoder.float()
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    method = "A08_VideoTGB"
    done = load_done(Path(args.out))
    question = (
        "Does this video contain hateful or demeaning content toward a person or protected group? "
        "Answer yes or no and briefly identify the relevant content."
    )
    torch.manual_seed(args.seed)
    for row in rows:
        if (row["dataset"], row["video_id"], method) in done:
            continue
        pred = Prediction(method, row["dataset"], row["video_id"],
                          float(row["duration"]), seed=args.seed)
        try:
            frames, flow = get_frames(row["video_path"], fps=2)
            # Video embeddings are prepended explicitly inside LSTP.generate;
            # the current HF processor maps literal <video> beyond the released
            # checkpoint's 32001-token embedding table.
            prompt = "USER:\n" + question + " ASSISTANT: "
            text = processor(text=prompt, padding="longest", truncation=True,
                             max_length=128, return_tensors="pt").to("cuda:0")
            sample_text = sampler_processor(text=question, padding="longest",
                                            truncation=True, max_length=128,
                                            return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                output, indices = model.generate(
                    frames.to("cuda:0", dtype=torch.bfloat16),
                    flow.unsqueeze(0).to("cuda:0", dtype=torch.float32),
                    4, text, sample_text, do_sample=False, max_new_tokens=96,
                    use_cache=False)
            answer = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
            # VideoTGB can continue generating memorized dialogue after its
            # first EOS marker. Only the anchored first answer is admissible;
            # contradictory tail text must never alter the existence decision.
            match = re.match(r"^\s*(yes|no)\b", answer, flags=re.IGNORECASE)
            if match is None:
                raise RuntimeError(f"unparseable first answer: {answer[:160]!r}")
            first_answer = match.group(1).lower()
            negative = first_answer == "no"
            values = []
            selected = [int(x) for x in indices.detach().cpu().tolist()]
            if not negative and selected:
                # `indices` addresses the bridge's 32 uniformly spaced candidate frames.
                start = min(selected) / 32.0 * pred.duration
                end = (max(selected) + 1) / 32.0 * pred.duration
                values = [[start, end, 1.0]]
            pred.intervals = normalize_intervals(values, pred.duration)
            pred.score_curve = intervals_to_curve(pred.intervals, pred.duration)
            pred.raw = {"answer": answer, "first_answer": first_answer,
                        "answer_parser": "anchored_yes_no_v1", "sampled_indices": selected,
                        "candidate_grid": 32, "official_nframe": 4}
        except Exception as exc:
            pred.error = f"{type(exc).__name__}: {exc}"
        append_jsonl(Path(args.out), pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
