"""Spec-displacement pilot: the union arm and the off-construct arm, with
per-layer hidden-state dumps.

Pre-registration: docs/duplex/PREREG_spec_displacement_pilot.md.

Two corpora (MHClip-EN test_clean, HateClipSeg test_clean) times two policy
arms (union offensiveness, off-construct spam/copyright) = 1,110 forward passes
behind one model load. The strict arm already exists on disk and is never
recomputed.

Everything except the policy sentence, the rule list and the scope sentence is
the frozen judge: the prompt skeleton, the `prag` judgment-mode block and the
system message come from src/duplex/score_duplex_probe.py, the frame grid, the
pixel budget and the transcript resolution come from
src/duplex/extract_duplex_readout.py, and the union policy text is imported
verbatim from scripts/duplex/dual_axis_offensiveness_score.py rather than
retyped. The off-construct policy text is frozen in the pre-registration.

Per video it writes the raw readout z = logsumexp(logits[Yes]) -
logsumexp(logits[No]) at the final prompt position and the fp16 hidden state at
that position for all 37 rows.

Output: results/spec_displacement/<corpus>/<arm>/scores.jsonl and
        results/spec_displacement/<corpus>/<arm>/hidden/<video_id>.npy

Usage:
  python scripts/duplex/spec_displacement_score.py
"""

import json
import logging
import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "duplex"))
sys.path.insert(0, _THIS_DIR)
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from score_duplex_probe import (  # noqa: E402
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    build_binary_token_ids,
)
from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    READER,
    resolve_frames,
    resolve_transcript,
)
from dual_axis_offensiveness_score import (  # noqa: E402
    OFFENSIVENESS_PROMPT,
    OFFENSIVENESS_RULES,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NUM_FRAMES = 16
TRANSCRIPT_LIMIT = 0
EXPECTED_ROWS = 37
EXPECTED_DIM = 4096

OUT_ROOT = os.path.join(ROOT, "results", "spec_displacement")

# ---------------------------------------------------------------------------
# FROZEN OFF-CONSTRUCT CONSTRUCT (pre-registration, committed before the first
# call of this script). Same skeleton as the union arm: policy sentence, rule
# list, scope sentence. Rule count, word count and character count are matched
# to the union list; no word denotes hostility, insult, offence, obscenity or a
# protected attribute.
# ---------------------------------------------------------------------------

SPAM_RULES = """1. Duplicated re-uploads of one recording posted repeatedly
2. Reposting footage owned by another uploader without authorisation
3. Thumbnails or titles that misdescribe the recording
4. Redirects to off-platform storefronts, affiliate links, or downloads
5. Artificial inflation of view counts, likes, or subscriber numbers
6. Undisclosed paid promotion, sponsorship, or placement
7. Auto-generated or bulk-scraped material republished as original"""

SPAM_PROMPT = """You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's spam and copyright policy, the following types of content are prohibited:
{rules}

These rules are about distribution and ownership alone and apply no matter what the video is about. Content violates them even when the subject matter is entirely ordinary (such as music, cooking, travel, sport, news, gaming, or classroom teaching), and even when it was uploaded by one person, an organisation, or nobody at all.

{reader_block}

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No"."""

ARMS = {
    "union": (OFFENSIVENESS_PROMPT, OFFENSIVENESS_RULES),
    "spam": (SPAM_PROMPT, SPAM_RULES),
}

CORPORA = [
    # slug, dataset, n expected, transcript-override map of the committed run
    ("mhclip_en", "MHClip_EN", 161,
     os.path.join(ROOT, "results", "testruns", "mhclip_en", "c2_overrides.json")),
    ("hateclipseg", "HateClipSeg", 394,
     os.path.join(ROOT, "results", "hateclipseg", "c2_overrides.json")),
]


def status(msg):
    with open(os.path.join(OUT_ROOT, "STATUS"), "w") as f:
        f.write(msg + "\n")


def build_messages(prompt, rules, ann, n_frames, transcript):
    text = prompt.format(
        title=ann.get("title", "") or "",
        transcript=transcript,
        rules=rules,
        reader_block=READER_BLOCKS[READER],
    )
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


def load_done_ids(scores_path, hidden_dir):
    """Ids with a finite z AND a loadable hidden array of the expected shape."""
    finite = set()
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                z = r.get("z")
                if r.get("video_id") and isinstance(z, (int, float)) and np.isfinite(z):
                    finite.add(r["video_id"])
    done = set()
    for vid in finite:
        p = os.path.join(hidden_dir, vid + ".npy")
        if not os.path.isfile(p):
            continue
        try:
            a = np.load(p, mmap_mode="r")
        except Exception:
            continue
        if tuple(a.shape) == (EXPECTED_ROWS, EXPECTED_DIM):
            done.add(vid)
    return done


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    from PIL import Image

    # ---------------- pre-flight: annotations, overrides, every frame -------
    status("preflight")
    t_pre = time.time()
    plan = []
    for slug, ds, expect, ovr_path in CORPORA:
        ann = load_annotations(ds)
        with open(ovr_path) as f:
            overrides = json.load(f)
        ids = load_clean_split_ids(ds, "test")
        if len(ids) != expect:
            raise SystemExit(f"{slug}: expected {expect} split ids, got {len(ids)}")
        frames = {}
        bad = []
        for vid in ids:
            if ann.get(vid) is None:
                bad.append((vid, "not in annotations"))
                continue
            paths = resolve_frames(vid, ds, NUM_FRAMES)
            if len(paths) != NUM_FRAMES:
                bad.append((vid, f"{len(paths)} frames"))
                continue
            for p in paths:
                try:
                    im = Image.open(p)
                    im.load()
                    im.close()
                except Exception as exc:
                    bad.append((vid, f"{os.path.basename(p)}: {exc}"))
                    break
            else:
                frames[vid] = paths
        if bad:
            for vid, why in bad[:20]:
                logging.error(f"  preflight failure {slug}/{vid}: {why}")
            raise SystemExit(f"ABORT: {slug}: {len(bad)} videos failed pre-flight")
        logging.info(f"Pre-flight OK: {slug} {len(frames)} videos x {NUM_FRAMES} frames")
        plan.append((slug, ds, ids, ann, overrides, frames))
    logging.info(f"Pre-flight total {time.time() - t_pre:.1f}s")

    # ---------------- what remains ------------------------------------------
    cells = []
    for slug, ds, ids, ann, overrides, frames in plan:
        for arm in ARMS:
            out_dir = os.path.join(OUT_ROOT, slug, arm)
            hidden_dir = os.path.join(out_dir, "hidden")
            os.makedirs(hidden_dir, exist_ok=True)
            scores_path = os.path.join(out_dir, "scores.jsonl")
            done = load_done_ids(scores_path, hidden_dir)
            remaining = [v for v in ids if v not in done]
            logging.info(f"cell {slug}/{arm}: {len(done)} done, {len(remaining)} remaining")
            cells.append(dict(slug=slug, ds=ds, arm=arm, ids=ids, ann=ann,
                              overrides=overrides, frames=frames,
                              scores_path=scores_path, hidden_dir=hidden_dir,
                              remaining=remaining))

    if not any(c["remaining"] for c in cells):
        logging.info("All four cells complete.")
    else:
        status("loading model")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
        processor = AutoProcessor.from_pretrained(MODEL)
        tokenizer = processor.tokenizer
        label_token_ids = build_binary_token_ids(tokenizer)
        yes_ids = sorted(label_token_ids["Yes"])
        no_ids = sorted(label_token_ids["No"])
        if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
            raise SystemExit(f"bad label token ids: Yes={yes_ids} No={no_ids}")
        logging.info(f"Yes ids {yes_ids} / No ids {no_ids}")
        patch_size = processor.image_processor.patch_size

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        yes_idx = torch.tensor(yes_ids, device=model.device)
        no_idx = torch.tensor(no_ids, device=model.device)

        for cell in cells:
            remaining = cell["remaining"]
            if not remaining:
                continue
            prompt, rules = ARMS[cell["arm"]]
            tag = f"{cell['slug']}/{cell['arm']}"
            logging.info(f"=== cell {tag}: {len(remaining)} videos ===")
            t0 = time.time()
            for i, vid in enumerate(remaining):
                status(f"{tag} {i + 1}/{len(remaining)}")
                ann_v = cell["ann"][vid]
                frame_paths = cell["frames"][vid]
                images = [Image.open(p).convert("RGB") for p in frame_paths]
                transcript = resolve_transcript(ann_v, vid, cell["overrides"],
                                                TRANSCRIPT_LIMIT)
                messages = build_messages(prompt, rules, ann_v,
                                          len(frame_paths), transcript)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=images,
                                   return_tensors="pt", size=size_kwarg)
                for im in images:
                    im.close()
                del images

                grid = inputs["image_grid_thw"]
                per_frame_pixels = [int(h * w) * patch_size * patch_size
                                    for _, h, w in grid.tolist()]
                if max(per_frame_pixels) > MAX_PIXELS:
                    raise SystemExit(f"{vid}: pixel cap not honored")

                inputs = inputs.to(model.device)
                n_tokens_total = int(inputs["input_ids"].shape[1])
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True,
                                use_cache=False, logits_to_keep=1)
                    last_logits = out.logits[0, -1, :].float()
                    z = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                              - torch.logsumexp(last_logits[no_idx], dim=0))
                    hidden = torch.stack(
                        [h[0, -1, :] for h in out.hidden_states], dim=0
                    ).to(torch.float16).cpu().numpy()
                del out, last_logits, inputs

                if not np.isfinite(z):
                    raise SystemExit(f"{tag}/{vid}: non-finite z")
                if hidden.shape != (EXPECTED_ROWS, EXPECTED_DIM):
                    raise SystemExit(f"{tag}/{vid}: hidden shape {hidden.shape}")
                np.save(os.path.join(cell["hidden_dir"], vid + ".npy"), hidden)
                del hidden

                rec = {
                    "video_id": vid,
                    "z": z,
                    "p_yes_renorm": float(1.0 / (1.0 + np.exp(-z))),
                    "n_tokens_total": n_tokens_total,
                    "n_transcript_chars": len(transcript),
                    "transcript_source": ("override" if vid in cell["overrides"]
                                          else "dataset"),
                    "arm": cell["arm"],
                }
                with open(cell["scores_path"], "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                if (i + 1) % 50 == 0 or i == 0:
                    el = time.time() - t0
                    logging.info(f"  [{tag} {i + 1}/{len(remaining)}] z={z:+.3f} "
                                 f"tokens={n_tokens_total} {el / (i + 1):.2f}s/video "
                                 f"peak_vram="
                                 f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            logging.info(f"=== cell {tag} done in {time.time() - t0:.1f}s ===")

    # ---------------- coverage asserts --------------------------------------
    status("coverage")
    cov = {}
    for cell in cells:
        n = len(load_done_ids(cell["scores_path"], cell["hidden_dir"]))
        cov[f"{cell['slug']}/{cell['arm']}"] = n
        expect = len(cell["ids"])
        if n != expect:
            status(f"FAILED: {cell['slug']}/{cell['arm']} coverage {n} != {expect}")
            raise SystemExit(f"ABORT: {cell['slug']}/{cell['arm']} {n} != {expect}")
    logging.info(f"Coverage OK: {cov}")
    with open(os.path.join(OUT_ROOT, "coverage.json"), "w") as f:
        json.dump(cov, f, indent=1)
    status("SCORING_DONE")
    logging.info("SCORING_DONE")


if __name__ == "__main__":
    main()
