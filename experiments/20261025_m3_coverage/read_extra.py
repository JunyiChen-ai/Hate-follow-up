#!/usr/bin/env python3
"""Extra whole-video question on the r6 prefix, read as a distribution over option labels (M3 candidate reads).

Per video: the r6 prefix (system message, 20 uniform frames with timestamps, the full timestamped Whisper transcript,
the rules and the reader instruction; Qwen/Qwen3-VL-8B-Instruct; identical to experiments/20260922_til/til_measure.py
and runs/20260926_glr/base_gridA) -> the r6 whole-video verdict z_video (Yes/No log-odds, the model's own answer is
appended as the stance turn, src/stance_cache.build) -> one extra question, asked
  direct : right after the prefix (no verdict turn; the cache is cropped back to the prefix), and/or
  stance : after the r6 verdict turn (question + "Yes"/"No" answer, as the r6 window branches see it).
At the first answer position the log-probability of each option's first token (option label, with space / case
variants that are single tokens, as the Yes/No read of src/mllm_judge.label_ids) is read under the softmax
restricted to the union of the option tokens, i.e. a distribution over the options. Also kept: the full-vocabulary
probability mass of the option tokens and the full-vocabulary top-1 token (does the model want to answer with an
option at all). No label file is read. GPU inside Slurm only (launch/lab1_read.sbatch); --dry-run is CPU only.

  python experiments/20261025_m3_coverage/read_extra.py --run-name <name> --question-file <q.txt> --options A B C D E F \
      --mode both [--datasets HateMM HateClipSeg DeHate] [--limit N] [--verify]
  python experiments/20261025_m3_coverage/read_extra.py --dry-run --question-file <q.txt> --options A B C D E F
      (CPU: option-token check, rendered branch text, token seam of both modes and prefix length vs r6, no weights)

--limit N: first N videos of every selected corpus (manifest order). --verify: on the first 2 videos of every corpus,
the cached reads are compared with one ordinary full call each (no cache): |dz_video|, max |d log p| per mode, token
seam; the differences go to run.log and verify.jsonl, and the run stops (exit != 0) when a gate below is crossed.
Output: runs/<exp-id>/<run-name>/{reads.jsonl, config.json, run.log, run.pid, verify.jsonl}; one line per video.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mllm_judge import FAMILY_IMAGE_KW, MODEL, VIDEO_QUESTION, Judge  # noqa: E402
from src.stance_cache import build  # noqa: E402
from src.video_inputs import frame_paths, load_asr, load_manifest  # noqa: E402

CODE_PATH = "experiments/20261025_m3_coverage/read_extra.py"
IMPLEMENTATION = ("M3 extra-question option read v1: r6 prefix (rules + 20 uniform frames + timestamped transcript, "
                  "Qwen3-VL-8B), one extra question read as the distribution over option first tokens, asked right "
                  "after the prefix (direct) and/or after the r6 verdict turn (stance); 2026-10-25")
SEED = 0
NATIVE_FRAMES = 20
MANIFEST = {"HateMM": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl",
            "HateClipSeg": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl",
            "DeHate": ROOT / "data/manifests/DeHate_test.jsonl"}
REFERENCE = {"HateMM": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl",          # r6 reads
             "HateClipSeg": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl",
             "DeHate": ROOT / "runs/20260927_dehate_external/reads_gridA/predictions.jsonl"}
MODES = {"direct": ("direct",), "stance": ("stance",), "both": ("stance", "direct")}
N_VERIFY = 2
GATE_DZ = 3.0      # verdict, cache vs plain call (til_measure.py uses the same 3.0 nat gate; bf16 noise stays < 1)
GATE_DLOGP = 2.0   # option log-probabilities, cache vs plain call (bounded in (-inf, 0]; noise is a fraction of a nat)
CHUNK = 32768      # vocabulary rows per fp32 block in the full-vocabulary pass


# ----------------------------------------------------------------------------------------------- options

def option_token_sets(tok, labels, variants="case_space"):
    """Per option label, the first-token ids it may start with. The bare label must be a single token. case_space adds
    ' label' and the lower / upper case forms (the src/mllm_judge.label_ids convention), keeping only forms that are a
    single token themselves (a multi-token form such as ' 5' -> ' ', '5' would put a shared token into every option).
    Raises when a bare label is not one token or two options share a token."""
    sets, report = [], []
    for lab in labels:
        bare = tok.encode(lab, add_special_tokens=False)
        if len(bare) != 1:
            raise SystemExit(f"option {lab!r} is {len(bare)} tokens {bare}; option labels must be single tokens")
        forms = [lab] if variants == "bare" else list(dict.fromkeys(
            [lab, f" {lab}", lab.lower(), f" {lab.lower()}", lab.upper(), f" {lab.upper()}"]))
        kept, dropped = {}, []
        for f in forms:
            ids = tok.encode(f, add_special_tokens=False)
            if len(ids) == 1:
                kept.setdefault(ids[0], f)
            else:
                dropped.append([f, ids])
        sets.append(sorted(kept))
        report.append({"label": lab, "ids": sorted(kept), "forms": [kept[i] for i in sorted(kept)],
                       "decoded": [tok.decode([i]) for i in sorted(kept)], "dropped_multi_token_forms": dropped})
    seen = {}
    for lab, ids in zip(labels, sets):
        for i in ids:
            if i in seen:
                raise SystemExit(f"options {seen[i]!r} and {lab!r} share token {i} ({tok.decode([i])!r})")
            seen[i] = lab
    return sets, report


def labels_in_question(question, labels):
    return [lab for lab in labels if lab not in question]


# ----------------------------------------------------------------------------------------------- reads

def full_vocab(j, h, union):
    """log of the full-vocabulary probability mass of `union`, and the top-1 token (fp32 lm_head, blockwise)."""
    W = j.model.get_output_embeddings().weight
    hf = h.float()
    lse, best, best_i = [], -math.inf, -1
    for a in range(0, W.shape[0], CHUNK):
        lg = W[a:a + CHUNK].float() @ hf
        if j.softcap:
            lg = torch.tanh(lg / j.softcap) * j.softcap
        lse.append(torch.logsumexp(lg, 0))
        m, i = lg.max(0)
        if float(m) > best:
            best, best_i = float(m), a + int(i)
        del lg
    total = float(torch.logsumexp(torch.stack(lse), 0))
    opt = j._logits_fp32(h[None], torch.tensor(union, device=j.device))[0]
    return float(torch.logsumexp(opt, 0)) - total, best_i, best - total


def summarize(j, h, labels, sets):
    lp = j.choice_logprobs(h, sets)
    union = [t for s in sets for t in s]
    mass, top_i, top_lp = full_vocab(j, h, union)
    return {"logp": {lab: x for lab, x in zip(labels, lp)}, "option_mass_full_vocab_log": mass,
            "top_token": j.tok.decode([top_i]), "top_token_id": top_i, "top_token_logp_full_vocab": top_lp}


@torch.no_grad()
def cached_read(j, cache, ctx, ids):
    """Last hidden row after `ids` on the cache; the cache and the rope offsets are restored afterwards."""
    n = cache.get_seq_length()
    j.model.model.rope_deltas = ctx["rope"].clone()
    try:
        return j._step(cache, ids)
    finally:
        cache.crop(n)
        j.model.model.rope_deltas = ctx["rope"].clone()


def stance_history(j, stance):
    return [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, j.turn("assistant", stance)]


def seam_pieces(j, msgs, files, question, mode, stance):
    """Token ids the cache path feeds (prefix + [verdict question + answer] + branch) and the ids of the same
    conversation rendered and encoded in one piece; equal when the seams are clean. CPU only (processor)."""
    from PIL import Image
    _, enc = j.encode_prefix(msgs, files)
    ids = enc["input_ids"][0].tolist()
    head, history = j._prefix_text, None
    if mode == "stance":
        qid, qtext = j.branch_ids(msgs, VIDEO_QUESTION)
        aid, atext = j.answer_ids(msgs, VIDEO_QUESTION, stance)
        ids, head, history = ids + qid + aid, head + qtext + atext, stance_history(j, stance)
    bids, btext = j.branch_ids(msgs, question, history, head_text=head)
    full = j.render(j.conv(msgs, question, history), add_generation_prompt=True)
    images = [Image.open(p).convert("RGB") for p in files]
    enc_full = j.encode(full, images)
    for im in images:
        im.close()
    return ids + bids, enc_full, btext, history


@torch.no_grad()
def plain_choices(j, enc_full, sets):
    out = j.model.model(**j.model_inputs(enc_full), use_cache=False)
    lp = j.choice_logprobs(out.last_hidden_state[0, -1], sets)
    del out
    return lp


@torch.no_grad()
def read_video(j, row, segments, args, sets, verify):
    ds, vid = row["dataset"], row["video_id"]
    frames = frame_paths(ds, vid, args.frames, "k20")
    if not frames:
        raise RuntimeError("no frames cached")
    cache, ctx = build(j, frames, segments)
    msgs, files = ctx["msgs"], ctx["files"]
    reads = {}
    for mode in args.modes:            # stance first: the direct read crops the cache back to the prefix
        if mode == "stance":
            ids, _ = j.branch_ids(msgs, args.question, ctx["history"], head_text=ctx["head"])
        else:
            cache.crop(ctx["prefix_tokens"])
            j.model.model.rope_deltas = ctx["rope"].clone()
            ids, _ = j.branch_ids(msgs, args.question)   # head = the prefix string set by encode_prefix inside build
        reads[mode] = summarize(j, cached_read(j, cache, ctx, ids), args.options, sets)
        reads[mode]["branch_tokens"] = len(ids)
    del cache
    rec = {"z_video": ctx["global_margin"], "stance": ctx["stance"], "prefix_tokens": ctx["prefix_tokens"],
           "n_frames": len(frames), "reads": reads}
    if verify:
        chk = {}
        for mode in args.modes:
            expected, enc_full, _, _ = seam_pieces(j, msgs, files, args.question, mode, ctx["stance"])
            chk[f"seam_ok_{mode}"] = enc_full["input_ids"][0].tolist() == expected
            lp = plain_choices(j, enc_full, sets)
            chk[f"max_abs_dlogp_{mode}"] = max(abs(a - reads[mode]["logp"][lab]) for a, lab in zip(lp, args.options))
            chk[f"logp_plain_{mode}"] = dict(zip(args.options, lp))
        z_ref = j.plain_margin(msgs, files, VIDEO_QUESTION)
        chk["dz_video_cache_vs_plain"] = abs(ctx["global_margin"] - z_ref)
        chk["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
        rec["verify"] = chk
    return rec


# ----------------------------------------------------------------------------------------------- CPU dry run

def processor_judge(model_id):
    """Judge without weights (processor, tokenizer, template handling only), for the CPU dry run."""
    from transformers import AutoConfig, AutoProcessor
    j = Judge.__new__(Judge)
    j.model_id = model_id
    j.processor = AutoProcessor.from_pretrained(model_id)
    j.tok = j.processor.tokenizer
    cfg = AutoConfig.from_pretrained(model_id)
    j.family = cfg.model_type
    j.img_kw = dict(FAMILY_IMAGE_KW[j.family])
    j.image_token_id = getattr(cfg, "image_token_id", None)
    j.list_content = j.family == "llava_onevision"
    j.loose_stance_seam = False
    try:
        j.render([{"role": "user", "content": [{"type": "text", "text": "a"}]},
                  {"role": "user", "content": [{"type": "text", "text": "b"}]}], True)
        j.same_turn = False
    except Exception:
        j.same_turn = True
    return j


def dry_run(args, rows, asr, ref):
    j = processor_judge(args.model)
    sets, report = option_token_sets(j.tok, args.options, args.variants)
    print("OPTION TOKENS", json.dumps(report, ensure_ascii=False))
    miss = labels_in_question(args.question, args.options)
    print("labels not found verbatim in the question:", miss or "none")
    shown = False
    for row in rows:
        ds, vid = row["dataset"], row["video_id"]
        frames = frame_paths(ds, vid, args.frames, "k20")
        msgs, files = j.prefix_messages(frames, asr[ds].get(vid, []))
        r = ref.get((ds, vid), {})
        line = {"dataset": ds, "video_id": vid, "n_frames": len(frames)}
        for mode in args.modes:
            expected, enc_full, btext, _ = seam_pieces(j, msgs, files, args.question, mode, r.get("stance", "Yes"))
            line[f"seam_ok_{mode}"] = enc_full["input_ids"][0].tolist() == expected
            if not shown:
                print(f"---- {mode} branch text (after {'prefix + verdict turn' if mode == 'stance' else 'prefix'}):\n"
                      f"{btext}\n----")
        shown = True
        _, enc = j.encode_prefix(msgs, files)
        line["prefix_tokens"] = int(enc["input_ids"].shape[1])
        line["prefix_tokens_r6"] = r.get("prefix_tokens")
        print("DRY", json.dumps(line))


# ----------------------------------------------------------------------------------------------- main

def load_reference(datasets):
    out = {}
    for path in sorted({REFERENCE[ds] for ds in datasets}):
        for line in open(path):
            r = json.loads(line)
            if r.get("error") or r["dataset"] not in datasets:
                continue
            e = r["extra"]
            out[(r["dataset"], r["video_id"])] = {"z_video": e["z_video"], "stance": e["stance"],
                                                  "prefix_tokens": e.get("prefix_tokens")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name")
    ap.add_argument("--exp-id", default="20261025_m3_coverage")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"], choices=sorted(MANIFEST))
    ap.add_argument("--question-file", help="text file holding the question exactly as it is asked (one user turn)")
    ap.add_argument("--question", help="the question text (instead of --question-file)")
    ap.add_argument("--options", nargs="+", required=True, help="option labels, e.g. A B C D E F or 0 1 2 3 4 5")
    ap.add_argument("--mode", choices=sorted(MODES), default="both",
                    help="direct: right after the prefix; stance: after the r6 verdict turn; both: both in one pass")
    ap.add_argument("--variants", choices=["case_space", "bare"], default="case_space")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=NATIVE_FRAMES)
    ap.add_argument("--limit", type=int, default=0, help="first N videos of every selected corpus")
    ap.add_argument("--verify", action="store_true", help="cache vs plain call on the first 2 videos of every corpus")
    ap.add_argument("--dry-run", action="store_true", help="CPU only, no weights, no output files")
    args = ap.parse_args()
    if bool(args.question) == bool(args.question_file):
        raise SystemExit("give exactly one of --question / --question-file")
    args.question = args.question if args.question else Path(args.question_file).read_text().strip("\n")
    args.modes = MODES[args.mode]
    rows = []
    for ds in args.datasets:
        rr = load_manifest(MANIFEST[ds], [ds])
        rows += rr[:args.limit] if args.limit else rr
    asr = {ds: load_asr(ds) for ds in args.datasets}
    ref = load_reference(args.datasets)
    if args.dry_run:
        dry_run(args, rows, asr, ref)
        return
    if not args.run_name:
        raise SystemExit("--run-name is required")
    torch.manual_seed(SEED)
    out = ROOT / "runs" / args.exp_id / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    j = Judge(model_id=args.model)
    sets, report = option_token_sets(j.tok, args.options, args.variants)
    miss = labels_in_question(args.question, args.options)
    if miss:
        logging.warning("option labels not found verbatim in the question: %s", miss)
    import transformers
    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"host": socket.gethostname(), "date": time.strftime("%Y-%m-%d"), "implementation": IMPLEMENTATION,
                "code": f"{CODE_PATH}; working copy, {time.strftime('%Y-%m-%d')}", "command": "python -u " + " ".join(sys.argv),
                "family": j.family, "same_turn": j.same_turn, "transformers": transformers.__version__,
                "torch": torch.__version__, "seed": SEED, "GT_read": False, "video_question": VIDEO_QUESTION,
                "option_tokens": report, "manifests": {ds: str(MANIFEST[ds].relative_to(ROOT)) for ds in args.datasets},
                "reference_reads": {ds: str(REFERENCE[ds].relative_to(ROOT)) for ds in args.datasets},
                "verify_gates": {"dz_video": GATE_DZ, "max_abs_dlogp": GATE_DLOGP, "seam": "exact token equality"},
                "n_videos": len(rows)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    logging.info("videos %d  modes %s  options %s  option tokens %s", len(rows), args.modes, args.options,
                 [r["ids"] for r in report])
    path = out / "reads.jsonl"
    done = set()
    if path.exists():
        done = {(r["dataset"], r["video_id"]) for r in map(json.loads, open(path)) if not r.get("error")}
    verify_ids = {(r["dataset"], r["video_id"]) for ds in args.datasets
                  for r in [x for x in rows if x["dataset"] == ds][:N_VERIFY]} if args.verify else set()
    n_err, t0, dzs, agree, n_new = 0, time.time(), [], 0, 0
    with open(path, "a") as fh:
        for n, row in enumerate(rows):
            key = (row["dataset"], row["video_id"])
            if key in done:
                continue
            base = {"dataset": key[0], "video_id": key[1], "host": socket.gethostname(), "question": args.question,
                    "options": args.options, "modes": list(args.modes)}
            try:
                begin = time.time()
                rec = read_video(j, row, asr[key[0]].get(key[1], []), args, sets, key in verify_ids)
                r6 = ref.get(key)
                rec.update({"z_video_r6": r6 and r6["z_video"], "stance_r6": r6 and r6["stance"],
                            "dz_r6": None if r6 is None else abs(rec["z_video"] - r6["z_video"]),
                            "prefix_tokens_r6": r6 and r6["prefix_tokens"], "seconds": time.time() - begin})
                if r6 is not None:
                    dzs.append(rec["dz_r6"]); agree += rec["stance"] == r6["stance"]
                n_new += 1
                if "verify" in rec:
                    v = rec["verify"]
                    logging.info("VERIFY %s/%s %s", *key, json.dumps(v))
                    with open(out / "verify.jsonl", "a") as vf:
                        vf.write(json.dumps({"dataset": key[0], "video_id": key[1], **v}) + "\n")
                    bad = (v["dz_video_cache_vs_plain"] >= GATE_DZ or not all(v[f"seam_ok_{m}"] for m in args.modes)
                           or max(v[f"max_abs_dlogp_{m}"] for m in args.modes) >= GATE_DLOGP)
                    if bad:
                        raise SystemExit(f"VERIFY GATE FAILED {key}: {json.dumps(v)}")
                fh.write(json.dumps({**base, "error": None, **rec}, ensure_ascii=False) + "\n")
                fh.flush()
                logging.info("%d/%d %s/%s z_video %.3f (r6 %s) stance %s  %s", n + 1, len(rows), *key, rec["z_video"],
                             None if r6 is None else round(r6["z_video"], 3), rec["stance"],
                             "  ".join(f"{m}: " + " ".join(f"{lab} {math.exp(x):.2f}" for lab, x in rd["logp"].items())
                                       + f" mass {math.exp(rd['option_mass_full_vocab_log']):.3f} top {rd['top_token']!r}"
                                       for m, rd in rec["reads"].items()))
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                n_err += 1
                logging.error("OOM %s %s", key[1], str(exc)[:200])
                fh.write(json.dumps({**base, "error": "OOM"}) + "\n")
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                logging.exception("FAILED %s: %s", key[1], exc)
                fh.write(json.dumps({**base, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            if (n + 1) % 20 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(rows), (time.time() - t0) / max(1, n_new), n_err)
    if dzs:
        s = sorted(dzs)
        logging.info("z_video vs stored r6 reads: median |dz| %.4f  max %.4f  stance agreement %d/%d",
                     s[len(s) // 2], s[-1], agree, len(s))
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
