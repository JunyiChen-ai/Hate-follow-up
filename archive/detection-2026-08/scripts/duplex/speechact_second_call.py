"""Speech-act (attribution) second call for the speech-act kill test.

Pre-registration: docs/duplex/PREREG_speechact_second_call.md (frozen at commit
8ffe65a, before the first call was scored). The question text below is copied
verbatim from that file.

The call is the frozen c2 hate judge with exactly one substitution: the
judgment question. Everything else is taken from the same frozen modules the c2
judge used -- the system message (score_duplex_probe.SYSTEM_MESSAGE), the prompt
skeleton (score_duplex_probe.DUPLEX_PROMPT), the `prag` reader block, the
per-corpus rules block (YOUTUBE_RULES for HateMM and MHClip_EN, BILIBILI_RULES
for MHClip_ZH), the 16 frames from frames_16, the uncapped fresh-transcript
overrides from results/testruns/<corpus>/c2_overrides.json, the title, the
{shortest_edge, longest_edge} pixel budget, the Yes/No token-id sets, and the
raw readout at the final prompt position

    z_sa = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids]).

High z_sa = the hostile content is asserted/endorsed by the author. Hidden
states are not stored; this test only needs z_sa.

Implementation note (the one ambiguity in the prereg, resolved here and
reported): the frozen question text ends with its own answer instruction
("Answer Yes or No."), so it replaces both the skeleton's question sentence and
the skeleton's trailing 'Answer "Yes" or "No".' line rather than being stacked
on top of it. Every frozen word of the question appears verbatim; no answer
instruction is duplicated. Nothing above the question changes.

Outputs (results/speechact_call/):
  scores_hatemm.jsonl    215 HateMM test videos
  scores_mhclip_zh.jsonl 149 MHClip_ZH test videos
  scores_en3.jsonl       the 3 incidental MHClip_EN cases
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

# Frozen prompt material. Imported, never copied.
from score_duplex_probe import (  # noqa: E402
    BILIBILI_RULES,
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    READER,
    resolve_frames,
    resolve_transcript,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

# ---------------------------------------------------------------------------
# FROZEN SPEECH-ACT QUESTION -- verbatim from the pre-registration.
# ---------------------------------------------------------------------------

SPEECHACT_QUESTION = (
    "Consider the hostile, offensive, or negative statements that appear in "
    "this video (in its speech, its title, or its on-screen text). Does the "
    "video's author or narrator assert or endorse any of these statements "
    "themselves, rather than only quoting, reporting, criticizing, or mocking "
    "them as someone else's words? Answer Yes or No."
)

# score_duplex_probe.DUPLEX_PROMPT with its question sentence and answer
# instruction replaced by {question}. Byte-identical above that point.
SPEECHACT_PROMPT = """You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

{reader_block}

{question}"""

# Corpus conventions of the frozen c2 run (scripts/duplex/run_testrun.sh:
# --split test --transcript-limit 0 --transcript-override-json c2_overrides.json).
CORPORA = {
    "hatemm": {
        "dataset": "HateMM",
        "overrides": os.path.join(PROJECT_ROOT, "results", "testruns",
                                  "hatemm", "c2_overrides.json"),
        "out_name": "scores_hatemm.jsonl",
        "expect": 215,
        "video_ids": None,
    },
    "mhclip_zh": {
        "dataset": "MHClip_ZH",
        "overrides": os.path.join(PROJECT_ROOT, "results", "testruns",
                                  "mhclip_zh", "c2_overrides.json"),
        "out_name": "scores_mhclip_zh.jsonl",
        "expect": 149,
        "video_ids": None,
    },
    "en3": {
        "dataset": "MHClip_EN",
        "overrides": os.path.join(PROJECT_ROOT, "results", "testruns",
                                  "mhclip_en", "c2_overrides.json"),
        "out_name": "scores_en3.jsonl",
        "expect": 3,
        # EN-FN-02, EN-FP-07, EN-FP-16 of results/ranking_autopsy/en/packet.json
        "video_ids": ["6EL3gDMlve4", "j_foVftOOs4", "TfPv2KiLOrs"],
    },
}


def build_messages(ann, n_frames, rules_text, transcript):
    prompt_text = SPEECHACT_PROMPT.format(
        title=ann.get("title", "") or "",
        transcript=transcript,
        rules=rules_text,
        reader_block=READER_BLOCKS[READER],
        question=SPEECHACT_QUESTION,
    )
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


def load_done_ids(scores_path):
    done = set()
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                z = r.get("z_sa")
                if r.get("video_id") and isinstance(z, (int, float)) and np.isfinite(z):
                    done.add(r["video_id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", default="hatemm,mhclip_zh,en3")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--transcript-limit", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results",
                                                          "speechact_call"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    names = [c.strip() for c in args.corpora.split(",") if c.strip()]
    for c in names:
        if c not in CORPORA:
            raise SystemExit(f"unknown corpus {c!r}; valid: {sorted(CORPORA)}")

    os.makedirs(args.out_dir, exist_ok=True)
    status_path = os.path.join(args.out_dir, "STATUS")

    def status(msg):
        with open(status_path, "w") as f:
            f.write(msg + "\n")

    logging.info("Frozen question: %s", SPEECHACT_QUESTION)

    from PIL import Image

    # ---- plan every corpus before paying for the model load ----------------
    status("preflight")
    plans = []
    for name in names:
        cfg = CORPORA[name]
        ds = cfg["dataset"]
        annotations = load_annotations(ds)
        with open(cfg["overrides"]) as f:
            overrides = json.load(f)
        ids = cfg["video_ids"] or load_clean_split_ids(ds, "test")
        if len(ids) != cfg["expect"]:
            raise SystemExit(f"{name}: expected {cfg['expect']} ids, got {len(ids)}")
        rules_text = BILIBILI_RULES if ds == "MHClip_ZH" else YOUTUBE_RULES

        frame_index = {}
        bad = []
        for vid in ids:
            if vid not in annotations:
                bad.append((vid, "not in annotations"))
                continue
            paths = resolve_frames(vid, ds, args.num_frames)
            if len(paths) != args.num_frames:
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
                frame_index[vid] = paths
        if bad:
            for vid, why in bad[:20]:
                logging.error(f"  preflight failure {name}/{vid}: {why}")
            raise SystemExit(f"ABORT: {len(bad)} {name} videos failed pre-flight")

        scores_path = os.path.join(args.out_dir, cfg["out_name"])
        done = load_done_ids(scores_path)
        remaining = [v for v in ids if v not in done]
        logging.info(f"{name}: {len(ids)} ids, overrides={len(overrides)}, "
                     f"{len(done)} done, {len(remaining)} remaining -> {scores_path}")
        plans.append({
            "name": name, "dataset": ds, "annotations": annotations,
            "overrides": overrides, "rules": rules_text, "ids": ids,
            "frames": frame_index, "path": scores_path, "remaining": remaining,
            "expect": cfg["expect"],
        })

    if any(p["remaining"] for p in plans):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer
        label_token_ids = build_binary_token_ids(tokenizer)
        yes_ids = sorted(label_token_ids["Yes"])
        no_ids = sorted(label_token_ids["No"])
        if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
            raise SystemExit(f"bad label token ids: Yes={yes_ids} No={no_ids}")
        logging.info(f"Yes ids {yes_ids} / No ids {no_ids}")

        ip = processor.image_processor
        patch_size = ip.patch_size

        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        yes_idx = torch.tensor(yes_ids, device=model.device)
        no_idx = torch.tensor(no_ids, device=model.device)

        for plan in plans:
            remaining = plan["remaining"]
            if not remaining:
                continue
            logging.info(f"=== {plan['name']}: scoring {len(remaining)} ===")
            t0 = time.time()
            n_done = 0
            for i, vid in enumerate(remaining):
                status(f"{plan['name']} {i + 1}/{len(remaining)} {vid}")
                ann = plan["annotations"][vid]
                frame_paths = plan["frames"][vid]
                images = [Image.open(p).convert("RGB") for p in frame_paths]
                transcript = resolve_transcript(ann, vid, plan["overrides"],
                                                args.transcript_limit)
                messages = build_messages(ann, len(frame_paths), plan["rules"],
                                          transcript)
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
                    outputs = model(**inputs, use_cache=False, logits_to_keep=1)
                    last_logits = outputs.logits[0, -1, :].float()
                    z_sa = float(torch.logsumexp(last_logits[yes_idx], dim=0)
                                 - torch.logsumexp(last_logits[no_idx], dim=0))
                del outputs, last_logits, inputs
                if not np.isfinite(z_sa):
                    raise SystemExit(f"{vid}: non-finite z_sa")

                rec = {
                    "video_id": vid,
                    "z_sa": z_sa,
                    "p_yes_renorm": float(1.0 / (1.0 + np.exp(-z_sa))),
                    "n_tokens_total": n_tokens_total,
                    "n_transcript_chars": len(transcript),
                    "transcript_source": ("override" if vid in plan["overrides"]
                                          else "dataset"),
                }
                with open(plan["path"], "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_done += 1
                if n_done % 25 == 0 or i == 0:
                    el = time.time() - t0
                    logging.info(f"  [{plan['name']} {n_done}/{len(remaining)}] "
                                 f"z_sa={z_sa:+.3f} {el / n_done:.2f}s/video")
            logging.info(f"=== {plan['name']}: {n_done} scored in "
                         f"{time.time() - t0:.1f}s ===")

    totals = {}
    for plan in plans:
        n = len(load_done_ids(plan["path"]))
        totals[plan["name"]] = n
        logging.info(f"{plan['name']}: total scored {n}/{plan['expect']}")
        if n != plan["expect"]:
            status(f"FAILED: {plan['name']} n={n} != {plan['expect']}")
            raise SystemExit(f"ABORT: {plan['name']} n={n} != {plan['expect']}")
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(json.dumps(totals) + "\n")
    logging.info("DONE %s", totals)


if __name__ == "__main__":
    main()
