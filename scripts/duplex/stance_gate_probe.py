"""
Stance-axis gate test: probe scoring (one GPU pass over the cohorts).

Diagnostic, not a method. Question: on the current substrate (8B, raw z,
restored transcripts), does a stance-oriented probe separate cue-sharing false
positives from true positives -- the separation the joint judge fails to make?

Three probes, each one extra call per cohort video, greedy, raw-z readout:

  stance_v1    asks what the video itself asserts or endorses, explicitly
               distinguishing asserting from mentioning/quoting/reporting/
               countering
  stance_para  a semantically equivalent paraphrase of stance_v1, giving the
               prompt-sensitivity noise floor
  effort_ctrl  the frozen thoroughness placebo, READER_BLOCKS["effort"],
               reused verbatim: maximal-care wording, zero stance content

This is a wrapper. Nothing frozen is edited: the prompt skeleton, the system
message, the rules, the Yes/No token-id sets, the pixel budget, the frame
resolution and the transcript resolution are all imported from
src/duplex/score_duplex_probe.py and src/duplex/extract_duplex_readout.py. The
only new text is the two stance blocks, and they occupy the same
{reader_block} slot the frozen readers occupy, so the question line, the
answer position and the readout are byte-identical to the source runs.

The joint-z comparison arm needs no call: it is read from the source runs.

Usage:
  python scripts/duplex/stance_gate_probe.py --cohorts results/stance_gate/cohorts.json
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))

from score_duplex_probe import (  # noqa: E402
    BILIBILI_RULES,
    DUPLEX_PROMPT,
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    resolve_frames,
    resolve_transcript,
)
from data_utils import load_annotations  # noqa: E402

# ---------------------------------------------------------------------------
# The probe blocks. Frozen at authoring time, before any probe was scored.
# Category language only: no group names and no lexicon terms appear, so the
# probe cannot smuggle in the surface cues whose confusion it is measuring.
# ---------------------------------------------------------------------------
PROBE_BLOCKS = {
    "stance_v1": (
        "Judgment mode: stance attribution. Judge only the position the video "
        "itself takes toward hostility directed at a group defined by a "
        "protected characteristic. The video asserts that hostility when its "
        "own voice -- its narration, its on-screen text, its title, or its "
        "framing -- puts it forward, endorses it, or celebrates it, including "
        "when the editing lends approval to a speaker. The video merely "
        "mentions that hostility when the material is quoted, reported, "
        "documented, examined, argued against, or attributed to someone else. "
        "A rule is violated only if the video asserts or endorses the "
        "hostility itself; mentioning, quoting, reporting, or countering it "
        "violates no rule, however hostile the referenced material is."
    ),
    "stance_para": (
        "Judgment mode: attribution of viewpoint. Assess only where the video "
        "stands on hostility aimed at a group defined by a protected "
        "characteristic. It advances that hostility when it speaks in its own "
        "voice -- through narration, text on screen, the title, or its "
        "framing -- to advocate it, support it, or applaud it, including when "
        "the edit signals approval of a speaker. It only refers to that "
        "hostility when the material is cited, relayed, recorded, analysed, "
        "rebutted, or ascribed to another party. A rule is broken only where "
        "the video advances or supports the hostility itself; citing, "
        "relaying, reporting, or rebutting it breaks no rule, no matter how "
        "hostile the referenced material is."
    ),
    "effort_ctrl": READER_BLOCKS["effort"],
}

PROBE_ORDER = ["stance_v1", "stance_para", "effort_ctrl"]


def build_messages(ann, n_frames, rules_text, transcript, reader_block):
    """The frozen user turn with one probe block in the frozen reader slot."""
    prompt_text = DUPLEX_PROMPT.format(
        title=ann.get("title", "") or "",
        transcript=transcript,
        rules=rules_text,
        reader_block=reader_block,
    )
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            z = r.get("z")
            if r.get("video_id") and r.get("probe") and \
                    isinstance(z, (int, float)) and np.isfinite(z):
                done.add((r["video_id"], r["probe"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="results/stance_gate/cohorts.json")
    ap.add_argument("--out", default="results/stance_gate/probe_scores.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most this many cohort videos (smoke test)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    cpath = os.path.join(PROJECT_ROOT, args.cohorts)
    with open(cpath) as f:
        cohorts = json.load(f)

    # Flatten to a work list of (arm, dataset, video_id, cell), one entry per
    # video; the three probes are run back to back on the same decoded frames.
    work = []
    arm_cfg = {}
    for arm, a in cohorts["arms"].items():
        arm_cfg[arm] = a
        for cell in ("fp", "tp", "tn"):
            for vid in a["cohorts"][cell]:
                work.append((arm, a["dataset"], vid, cell))
    logging.info(f"Cohorts: {len(work)} videos, {len(work) * len(PROBE_ORDER)} probe calls")
    if args.limit is not None:
        work = work[:args.limit]
        logging.info(f"--limit: {len(work)} videos")

    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = load_done(out_path)
    logging.info(f"Resume: {len(done)} probe calls already complete")

    # Per-arm inputs: annotations and the judge-seen transcript map, exactly as
    # the source run assembled them.
    ann_by_ds, ov_by_arm = {}, {}
    for arm, a in arm_cfg.items():
        ds = a["dataset"]
        if ds not in ann_by_ds:
            ann_by_ds[ds] = load_annotations(ds)
        run_dir = os.path.join(PROJECT_ROOT, os.path.dirname(a["judge_dir"]))
        with open(os.path.join(run_dir, "c2_overrides.json")) as f:
            ov_by_arm[arm] = json.load(f)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    ids = build_binary_token_ids(tokenizer)
    yes_ids, no_ids = sorted(ids["Yes"]), sorted(ids["No"])
    if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
        raise SystemExit(f"bad label token id sets: Yes={yes_ids} No={no_ids}")
    logging.info(f"Yes ids {yes_ids}  No ids {no_ids}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    yes_idx = torch.tensor(yes_ids, device=model.device)
    no_idx = torch.tensor(no_ids, device=model.device)

    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    t0, n_calls, n_skipped = time.time(), 0, 0

    for i, (arm, ds, vid, cell) in enumerate(work):
        todo = [p for p in PROBE_ORDER if (vid, p) not in done]
        if not todo:
            continue
        ann = ann_by_ds[ds].get(vid)
        frame_paths = resolve_frames(vid, ds, args.num_frames)
        if ann is None or not frame_paths:
            logging.warning(f"  {vid}: missing annotation or frames, skipping")
            n_skipped += 1
            continue
        rules_text = BILIBILI_RULES if ds == "MHClip_ZH" else YOUTUBE_RULES
        # transcript_limit 0: uncapped, matching the C2 source runs.
        transcript = resolve_transcript(ann, vid, ov_by_arm[arm], 0)
        images = [Image.open(p).convert("RGB") for p in frame_paths]

        for probe in todo:
            messages = build_messages(ann, len(frame_paths), rules_text,
                                      transcript, PROBE_BLOCKS[probe])
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images,
                               return_tensors="pt", size=size_kwarg)
            inputs = inputs.to(model.device)
            n_tokens = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                out = model(**inputs, use_cache=False, logits_to_keep=1)
                last = out.logits[0, -1, :].float()
                z = float(torch.logsumexp(last[yes_idx], dim=0)
                          - torch.logsumexp(last[no_idx], dim=0))
            del out, last, inputs
            if not np.isfinite(z):
                logging.error(f"  {vid}/{probe}: non-finite z, skipping")
                n_skipped += 1
                continue
            rec = {"video_id": vid, "arm": arm, "cell": cell, "probe": probe,
                   "z": z, "n_tokens_total": n_tokens,
                   "n_transcript_chars": len(transcript),
                   "transcript_source": ("override" if vid in ov_by_arm[arm]
                                         else "dataset")}
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            n_calls += 1

        for im in images:
            im.close()
        del images

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            logging.info(f"  [{i + 1}/{len(work)}] videos, {n_calls} calls, "
                         f"{el / max(n_calls, 1):.2f}s/call, "
                         f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    el = time.time() - t0
    logging.info(f"Done: {n_calls} calls, {n_skipped} skipped, {el:.1f}s "
                 f"({el / max(n_calls, 1):.2f}s/call)")


if __name__ == "__main__":
    main()
