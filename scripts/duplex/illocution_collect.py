"""Illocution transfer audit: activation collection on the researcher-authored
speech-act instrument.

Pre-registration: docs/duplex/PREREG_illocution_transfer_audit.md.

One HF-transformers forward per instrument sentence under the frozen `prag`
reader of the duplex judge. The prompt skeleton, reader block, system message
and rules text are imported from src/duplex/score_duplex_probe.py, exactly as
src/duplex/extract_duplex_readout.py imports them; nothing about the prompt is
redefined here. The instrument sentence goes in the transcript slot, the title
slot is empty, and the visual channel carries 16 IDENTICAL mid-gray frames
(the same PIL object repeated, constant across every item), so no contrast in
the collected states can come from the images.

Output, mirroring the testruns dump format:
  results/illocution/hidden/<item_id>.npy   (37, 4096) fp16
  results/illocution/manifest.jsonl         one record per item
  results/illocution/STATUS                 progress, rewritten during the run
  results/illocution/DONE                   sentinel written at completion

Usage:
  python scripts/duplex/illocution_collect.py
"""

import json
import logging
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = "/home/jehc223/Hate-follow-up"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))

# Frozen prompt material, imported never copied.
from score_duplex_probe import (  # noqa: E402
    BILIBILI_RULES,
    DUPLEX_PROMPT,
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    YOUTUBE_RULES,
    build_binary_token_ids,
)

# Same per-frame pixel budget as extract_duplex_readout.py.
MAX_PIXELS = 100352
MIN_PIXELS = 65536
READER = "prag"
NUM_FRAMES = 16
GRAY_SIDE = 448
GRAY_VALUE = 128
MODEL = "Qwen/Qwen3-VL-8B-Instruct"

BANK_DIR = os.path.join(PROJECT_ROOT, "scripts", "duplex", "illocution_bank")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "illocution")
HIDDEN_DIR = os.path.join(OUT_DIR, "hidden")
MANIFEST = os.path.join(OUT_DIR, "manifest.jsonl")
STATUS = os.path.join(OUT_DIR, "STATUS")
DONE = os.path.join(OUT_DIR, "DONE")

# Rules text is the one the corresponding corpus is scored under: the English
# bank sits in the youtube-rules prompt, the Chinese bank in the bilibili-rules
# prompt, matching MHClip-EN/HateMM and MHClip-ZH respectively.
RULES_FOR_LANG = {"en": YOUTUBE_RULES, "zh": BILIBILI_RULES}


def load_bank():
    items = []
    for lang in ("en", "zh"):
        path = os.path.join(BANK_DIR, f"{lang}.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["language"] != lang:
                    raise SystemExit(f"ABORT: {r['id']} language {r['language']} "
                                     f"in {lang} bank")
                items.append(r)
    ids = [r["id"] for r in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("ABORT: duplicate item ids in the banks")
    return items


def write_status(text):
    with open(STATUS, "w") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_ids(expected_shape):
    done = set()
    if not os.path.exists(MANIFEST):
        return done
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = r.get("id")
            if not iid:
                continue
            p = os.path.join(HIDDEN_DIR, f"{iid}.npy")
            if not os.path.isfile(p):
                continue
            try:
                a = np.load(p, mmap_mode="r")
            except Exception:
                continue
            if tuple(a.shape) == tuple(expected_shape):
                done.add(iid)
    return done


def main():
    os.makedirs(HIDDEN_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if os.path.exists(DONE):
        os.remove(DONE)

    items = load_bank()
    logging.info(f"Bank: {len(items)} items "
                 f"({sum(1 for r in items if r['language'] == 'en')} en / "
                 f"{sum(1 for r in items if r['language'] == 'zh')} zh)")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    config = AutoConfig.from_pretrained(MODEL)
    n_layers = config.text_config.num_hidden_layers
    hidden_size = config.text_config.hidden_size
    expected_shape = (n_layers + 1, hidden_size)
    logging.info(f"Model config: n_layers={n_layers} hidden_size={hidden_size} "
                 f"shape {expected_shape}")

    done = load_done_ids(expected_shape)
    remaining = [r for r in items if r["id"] not in done]
    logging.info(f"Resume: {len(done)} complete, {len(remaining)} remaining")
    if not remaining:
        write_status(f"nothing to do; {len(done)} complete")
        with open(DONE, "w") as f:
            f.write("ok\n")
        return

    processor = AutoProcessor.from_pretrained(MODEL)
    tokenizer = processor.tokenizer
    label_token_ids = build_binary_token_ids(tokenizer)
    yes_ids = sorted(label_token_ids["Yes"])
    no_ids = sorted(label_token_ids["No"])
    if not yes_ids or not no_ids or (set(yes_ids) & set(no_ids)):
        raise SystemExit(f"ABORT: label token ids Yes={yes_ids} No={no_ids}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    yes_idx = torch.tensor(yes_ids, device=model.device)
    no_idx = torch.tensor(no_ids, device=model.device)

    # One constant mid-gray frame, repeated 16 times, shared by every item.
    gray = Image.new("RGB", (GRAY_SIDE, GRAY_SIDE),
                     (GRAY_VALUE, GRAY_VALUE, GRAY_VALUE))
    images = [gray] * NUM_FRAMES
    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    logging.info(f"Visual channel: {NUM_FRAMES} identical "
                 f"{GRAY_SIDE}x{GRAY_SIDE} RGB({GRAY_VALUE}) frames, "
                 f"size={size_kwarg}")

    t0 = time.time()
    n_done = 0
    cap_checked = False
    for i, item in enumerate(remaining):
        prompt_text = DUPLEX_PROMPT.format(
            title="",
            transcript=item["text"],
            rules=RULES_FOR_LANG[item["language"]],
            reader_block=READER_BLOCKS[READER],
        )
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           size=size_kwarg)
        grid = inputs["image_grid_thw"]
        ip = processor.image_processor
        per_frame_pixels = [int(h * w) * ip.patch_size * ip.patch_size
                            for _, h, w in grid.tolist()]
        if max(per_frame_pixels) > MAX_PIXELS:
            raise SystemExit(f"{item['id']}: pixel cap not honored: "
                             f"{max(per_frame_pixels)} > {MAX_PIXELS}")
        if not cap_checked:
            logging.info(f"  cap check: grid={grid.tolist()[0]} "
                         f"pixels/frame={per_frame_pixels[0]}")
            cap_checked = True

        inputs = inputs.to(model.device)
        n_tokens_total = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True,
                            use_cache=False, logits_to_keep=1)
            last_logits = outputs.logits[0, -1, :].float()
            z = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                      - torch.logsumexp(last_logits[no_idx], dim=0))
            hidden = torch.stack(
                [h[0, -1, :] for h in outputs.hidden_states], dim=0
            ).to(torch.float16).cpu().numpy()
        del outputs, last_logits, inputs

        if hidden.shape != expected_shape:
            raise SystemExit(f"{item['id']}: hidden shape {hidden.shape} "
                             f"!= {expected_shape}")
        if not np.isfinite(z):
            raise SystemExit(f"{item['id']}: non-finite z")
        np.save(os.path.join(HIDDEN_DIR, f"{item['id']}.npy"), hidden)
        del hidden

        rec = dict(item)
        rec["z"] = z
        rec["n_tokens_total"] = n_tokens_total
        with open(MANIFEST, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_done += 1
        if n_done % 25 == 0 or i == 0:
            el = time.time() - t0
            msg = (f"[{n_done}/{len(remaining)}] {item['id']} z={z:+.3f} "
                   f"tokens={n_tokens_total} {n_done / max(el, 1e-9):.2f} it/s")
            logging.info("  " + msg)
            write_status(msg)

    el = time.time() - t0
    logging.info(f"Done: {n_done} items in {el:.1f}s "
                 f"({el / max(n_done, 1):.2f}s/item)")
    write_status(f"complete: {n_done} items, {el:.1f}s")
    with open(DONE, "w") as f:
        f.write(f"{n_done}\n")
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
