"""Several whole-video questions on one r6 prefix, each in an isolated branch (shared reading code, 2026-10-25).

The prefix is the r6 one, built by src/stance_cache.build (system message, 20 uniform frames with timestamps, the full
timestamped Whisper transcript, the rules and the reader instruction; the r6 verdict z_video is read and the model's own
"Yes"/"No" answer is appended as the stance turn). Every question is then asked in its own branch:
  after="stance" : after the verdict turn (as the r6 window branches see it);
  after="direct" : right after the prefix (the cache is cropped back to the prefix; no verdict turn).
Branches never see each other: each one runs on the cache and the cache is cropped back afterwards. One prefix encoding
per video serves all questions. Two read-outs:
  kind="yesno"   : Yes/No log-odds with the token sets and the fp32 read of the r6 verdict (Judge.yes_ids / no_ids,
                   Judge.margins_fp32), i.e. exactly how z_video is read;
  kind="options" : the distribution over option labels, as experiments/20261025_m3_coverage/read_extra.py: per option
                   the first tokens it may start with (bare label, ' label', lower / upper case; only forms that are a
                   single token), log-probabilities under the softmax restricted to the union of the option tokens.
Both also keep the full-vocabulary probability mass of the answer tokens and the full-vocabulary top-1 token.

verify=True compares every cached branch with one ordinary full call (no cache) and checks the token seams; the gates
are those of read_extra.py (|dz| < 3 nats, max |d log p| < 2 nats, exact token equality); `verify_failed` applies them.

Question specs: {"name", "text", "kind": "yesno" | "options", "after": "direct" | "stance", "options": [labels]}; the
text is asked verbatim (include the answer instruction). `load_questions` reads a .json list of specs or a .txt file
of blocks, each opened by a header line
    @ name=<name> kind=<yesno|options> after=<direct|stance> [options=A,B,C]
followed by the question text (lines starting with '#' are comments).

Subset prefixes (per-segment reads): equal_segments, frames_in, transcript_in (the r6 window-text slicing rule),
subset_inputs and read_subset_verdict(j, frames, pieces, verify) read VIDEO_QUESTION on a prefix that holds only a
time span's frames and transcript pieces, with its own prefix cache (absolute timestamps; prompt material unchanged).

CPU check without weights (rendering, token seams of every question, Yes/No and option tokens, prefix length vs r6):
    python -m src.prefix_questions --dry-run --questions Q.txt [--datasets HateMM HateClipSeg DeHate] [--limit N]
CPU check of the read path itself (cache, crop, isolation, verify) on a small random Qwen3-VL, no weights, no files:
    python -m src.prefix_questions --selftest
GPU smoke (inside a Slurm job only; prints the reads and the verify checks, writes no file):
    python -m src.prefix_questions --smoke --questions Q.txt --limit 2
No label file is read here.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

from src.mllm_judge import FAMILY_IMAGE_KW, MODEL, VIDEO_QUESTION, Judge
from src.stance_cache import build
from src.video_inputs import frame_paths, load_asr, load_manifest

ROOT = Path(__file__).resolve().parents[1]
NATIVE_FRAMES = 20
KINDS = ("yesno", "options")
AFTER = ("stance", "direct")       # read order inside a video: stance branches first, then the cache is cropped
GATE_DZ = 3.0
GATE_DLOGP = 2.0
CHUNK = 32768
MANIFEST = {"HateMM": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl",
            "HateClipSeg": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl",
            "DeHate": ROOT / "data/manifests/DeHate_test.jsonl"}
REFERENCE = {"HateMM": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl",
             "HateClipSeg": ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl",
             "DeHate": ROOT / "runs/20260927_dehate_external/reads_gridA/predictions.jsonl"}


# ----------------------------------------------------------------------------------------------- question specs

def load_questions(path):
    path = Path(path)
    if path.suffix == ".json":
        specs = json.loads(path.read_text())
    else:
        specs, cur, body = [], None, []
        for line in path.read_text().splitlines() + ["@ END"]:
            if line.startswith("#"):
                continue
            if line.startswith("@ "):
                if cur is not None:
                    cur["text"] = "\n".join(body).strip("\n")
                    specs.append(cur)
                if line.strip() == "@ END":
                    break
                cur, body = dict(kv.split("=", 1) for kv in line[2:].split()), []
                if "options" in cur:
                    cur["options"] = cur["options"].split(",")
            elif cur is not None:
                body.append(line)
            elif line.strip():
                raise SystemExit(f"{path}: text before the first '@ ' header: {line!r}")
    return validate(specs)


def validate(specs):
    names = [s.get("name") for s in specs]
    if not specs or None in names or len(set(names)) != len(names):
        raise SystemExit(f"question names must be present and unique: {names}")
    for s in specs:
        if s.get("kind") not in KINDS or s.get("after") not in AFTER or not s.get("text", "").strip():
            raise SystemExit(f"bad question spec {s}")
        if (s["kind"] == "options") != bool(s.get("options")):
            raise SystemExit(f"{s['name']}: options are required for kind=options and only there")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", s["name"]):
            raise SystemExit(f"{s['name']}: name must be [A-Za-z0-9_.-]")
    return specs


def option_token_sets(tok, labels, variants="case_space"):
    """Per option label, the first-token ids it may start with (same rule as read_extra.py): the bare label must be a
    single token; case_space adds ' label' and lower / upper case forms, keeping only single-token forms. Raises when a
    bare label is not one token or two options share a token."""
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
        report.append({"label": lab, "ids": sorted(kept), "decoded": [tok.decode([i]) for i in sorted(kept)],
                       "dropped_multi_token_forms": dropped})
    seen = {}
    for lab, ids in zip(labels, sets):
        for i in ids:
            if i in seen:
                raise SystemExit(f"options {seen[i]!r} and {lab!r} share token {i} ({tok.decode([i])!r})")
            seen[i] = lab
    return sets, report


def prepare(j, specs, variants="case_space"):
    """Token sets per question (options) and a report of every answer token; Yes/No questions use j.yes_ids / no_ids."""
    report = {"yes_ids": j.yes_ids, "no_ids": j.no_ids, "yes_decoded": [j.tok.decode([i]) for i in j.yes_ids],
              "no_decoded": [j.tok.decode([i]) for i in j.no_ids], "options": {}}
    if set(j.yes_ids) & set(j.no_ids):
        raise SystemExit("Yes and No token sets overlap")
    sets = {}
    for s in specs:
        if s["kind"] == "options":
            sets[s["name"]], report["options"][s["name"]] = option_token_sets(j.tok, s["options"], variants)
            missing = [lab for lab in s["options"] if lab not in s["text"]]
            if missing:
                report["options"][s["name"]].append({"labels_not_in_text": missing})
    return sets, report


# ----------------------------------------------------------------------------------------------- reads

def stance_history(j, stance):
    return [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, j.turn("assistant", stance)]


def full_vocab(j, h, union):
    """log full-vocabulary probability mass of `union`, top-1 token id and its log-probability (fp32, blockwise)."""
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
    sub = j._logits_fp32(h[None], torch.tensor(union, device=j.device))[0]
    return float(torch.logsumexp(sub, 0)) - total, best_i, best - total


def answer_read(j, h, spec, sets):
    """The read-out of one question from the hidden row at its first answer position."""
    if spec["kind"] == "yesno":
        out = {"z": j.margins_fp32(h[None])[0]}
        union = list(j.yes_ids) + list(j.no_ids)
    else:
        out = {"logp": dict(zip(spec["options"], j.choice_logprobs(h, sets[spec["name"]])))}
        union = [t for s in sets[spec["name"]] for t in s]
    mass, top_i, top_lp = full_vocab(j, h, union)
    out.update({"answer_mass_full_vocab_log": mass, "top_token": j.tok.decode([top_i]), "top_token_id": top_i,
                "top_token_logp_full_vocab": top_lp})
    return out


@torch.no_grad()
def cached_hidden(j, cache, ctx, ids):
    """Last hidden row after `ids` on the cache; the cache and the rope offsets are restored afterwards."""
    n = cache.get_seq_length()
    j.model.model.rope_deltas = ctx["rope"].clone()
    try:
        return j._step(cache, ids)
    finally:
        cache.crop(n)
        j.model.model.rope_deltas = ctx["rope"].clone()


def branch(j, msgs, spec, ctx_or_stance):
    """(branch ids, branch text, history) of a question; for after=stance the verdict turn comes first."""
    if spec["after"] == "stance":
        if isinstance(ctx_or_stance, dict):
            history, head = ctx_or_stance["history"], ctx_or_stance["head"]
        else:   # CPU dry run: the stance string, head rebuilt from the template
            qid, qtext = j.branch_ids(msgs, VIDEO_QUESTION)
            aid, atext = j.answer_ids(msgs, VIDEO_QUESTION, ctx_or_stance)
            history, head = stance_history(j, ctx_or_stance), j._prefix_text + qtext + atext
        ids, text = j.branch_ids(msgs, spec["text"], history, head_text=head)
        return ids, text, history
    ids, text = j.branch_ids(msgs, spec["text"])   # head = the prefix string set by encode_prefix
    return ids, text, None


def seam_checks(j, msgs, files, specs, stance):
    """Per question: the token ids the cache path feeds (prefix + [verdict question + answer] + branch) against the ids
    of the same conversation rendered and encoded in one piece. Returns {name: (ok, encoded full conversation, branch
    text)}. Processor only (CPU)."""
    from PIL import Image
    _, enc = j.encode_prefix(msgs, files)
    prefix_ids = enc["input_ids"][0].tolist()
    qid, _ = j.branch_ids(msgs, VIDEO_QUESTION)
    aid, _ = j.answer_ids(msgs, VIDEO_QUESTION, stance)
    images = [Image.open(p).convert("RGB") for p in files]
    out = {}
    try:
        for s in specs:
            bids, btext, history = branch(j, msgs, s, stance)
            fed = prefix_ids + (qid + aid if s["after"] == "stance" else []) + bids
            enc_full = j.encode(j.render(j.conv(msgs, s["text"], history), add_generation_prompt=True), images)
            out[s["name"]] = (enc_full["input_ids"][0].tolist() == fed, enc_full, btext)
    finally:
        for im in images:
            im.close()
    return out


@torch.no_grad()
def plain_hidden(j, enc_full):
    out = j.model.model(**j.model_inputs(enc_full), use_cache=False)
    h = out.last_hidden_state[0, -1].clone()
    del out
    return h


@torch.no_grad()
def read_prefix_questions(j, frames, segments, specs, sets, verify=False):
    """All questions of `specs` on one video. frames: list of (t, path) (src.video_inputs.frame_paths); sets: from
    prepare(). Returns {"z_video", "stance", "prefix_tokens", "stance_cache_tokens", "n_frames", "questions": {name:
    read}, ["verify": checks]}."""
    cache, ctx = build(j, frames, segments)
    msgs, files = ctx["msgs"], ctx["files"]
    reads = {}
    for after in AFTER:
        todo = [s for s in specs if s["after"] == after]
        if not todo:
            continue
        if after == "direct":
            cache.crop(ctx["prefix_tokens"])
            j.model.model.rope_deltas = ctx["rope"].clone()
        for s in todo:
            ids, _, _ = branch(j, msgs, s, ctx)
            reads[s["name"]] = {"kind": s["kind"], "after": after, "branch_tokens": len(ids),
                                **answer_read(j, cached_hidden(j, cache, ctx, ids), s, sets)}
    del cache
    rec = {"z_video": ctx["global_margin"], "stance": ctx["stance"], "prefix_tokens": ctx["prefix_tokens"],
           "stance_cache_tokens": ctx["stance_cache_tokens"], "n_frames": len(frames), "questions": reads}
    if verify:
        chk, seams = {}, seam_checks(j, msgs, files, specs, ctx["stance"])
        for s in specs:
            ok, enc_full, _ = seams[s["name"]]
            ref = answer_read(j, plain_hidden(j, enc_full), s, sets)
            got = reads[s["name"]]
            d = abs(ref["z"] - got["z"]) if s["kind"] == "yesno" else \
                max(abs(ref["logp"][lab] - got["logp"][lab]) for lab in s["options"])
            chk[s["name"]] = {"seam_ok": ok, ("dz" if s["kind"] == "yesno" else "max_abs_dlogp"): d}
        chk["dz_video_cache_vs_plain"] = abs(ctx["global_margin"] - j.plain_margin(msgs, files, VIDEO_QUESTION))
        chk["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        j.model.model.rope_deltas = None
        rec["verify"] = chk
    return rec


def verify_failed(chk):
    """Reasons the verify gates fail (empty list = pass)."""
    bad = []
    if chk["dz_video_cache_vs_plain"] >= GATE_DZ:
        bad.append(f"z_video |dz| {chk['dz_video_cache_vs_plain']:.3f}")
    for name, c in chk.items():
        if not isinstance(c, dict):
            continue
        if not c["seam_ok"]:
            bad.append(f"{name}: token seam")
        if c.get("dz", 0.0) >= GATE_DZ or c.get("max_abs_dlogp", 0.0) >= GATE_DLOGP:
            bad.append(f"{name}: cache vs plain {c}")
    return bad


# ----------------------------------------------------------------------------------------------- subset prefixes
# A prefix built from a time subset of the video's inputs (added 2026-10-25 for the per-segment reads): the r6 prompt
# material unchanged (system message, intro, rules, reader instruction, Judge.prefix_messages) with only the frames and
# transcript pieces whose time falls in [t1, t2). Timestamps stay absolute (seconds from the video start), as in r6.

def equal_segments(duration, n):
    """n equal-duration segments [k D / n, (k + 1) D / n) of [0, D)."""
    return [(k * duration / n, (k + 1) * duration / n) for k in range(n)]


def frames_in(frames, t1, t2, last=False):
    """frames ((t, path) list) with t1 <= t < t2; last=True closes the right end (t >= t2 also kept)."""
    return [(t, p) for t, p in frames if t >= t1 and (last or t < t2)]


def transcript_in(segments, t1, t2):
    """Transcript pieces of [t1, t2) as (start, end, text) with clipped times. A chunk inside the span is kept whole;
    a chunk crossing a boundary is cut at word boundaries in proportion to time, with exactly the rounding of
    src.video_inputs.window_text (the r6 window-text rule), so " ".join of the texts equals window_text(segments, t1,
    t2) and adjacent spans split a chunk's words without loss or overlap."""
    out = []
    for s, e, text in segments:
        lo, hi = max(s, t1), min(e, t2)
        if hi <= lo or not text:
            continue
        if s >= t1 and e <= t2:
            out.append((s, e, text))
            continue
        words = text.split()
        if not words:
            continue
        a = int(round((lo - s) / (e - s) * len(words)))
        b = int(round((hi - s) / (e - s) * len(words)))
        piece = " ".join(words[a:b]).strip()
        if piece:
            out.append((lo, hi, piece))
    return out


def subset_inputs(frames, segments, t1, t2, last=False, transcript_end=None):
    """(frames, transcript pieces) of the span; for the last span the transcript bound is extended to `transcript_end`
    (e.g. the latest chunk end) so that speech timed past the container duration is not lost."""
    hi = max(t2, transcript_end) if (last and transcript_end is not None) else t2
    return frames_in(frames, t1, t2, last), transcript_in(segments, t1, hi)


@torch.no_grad()
def read_subset_verdict(j, frames, pieces, verify=False):
    """The r6 whole-video question (VIDEO_QUESTION) on a prefix holding only `frames` and transcript `pieces`, read as
    Yes/No log-odds exactly as z_video (own prefix cache, Judge.cached_margin, fp32 Yes/No sets). No cache is shared
    with any other prefix. verify=True: token seam and one ordinary full call. Returns {z, prefix_tokens,
    branch_tokens, n_frames, [verify]}."""
    j.model.model.rope_deltas = None
    msgs, files = j.prefix_messages(frames, pieces)
    _, enc = j.encode_prefix(msgs, files)
    cache = j.prefix_cache(enc)
    qid, _ = j.branch_ids(msgs, VIDEO_QUESTION)
    z = j.cached_margin(cache, qid, in_place=True)
    del cache
    rec = {"z": z, "prefix_tokens": int(enc["input_ids"].shape[1]), "branch_tokens": len(qid), "n_frames": len(frames)}
    if verify:
        ok = subset_seam_ok(j, msgs, files, enc["input_ids"][0].tolist(), qid)
        j.model.model.rope_deltas = None
        rec["verify"] = {"seam_ok": ok, "dz": abs(z - j.plain_margin(msgs, files, VIDEO_QUESTION))}
        j.model.model.rope_deltas = None
    return rec


def subset_seam_ok(j, msgs, files, prefix_ids, qid):
    """prefix ids + question ids equal the ids of the conversation encoded in one piece (processor only)."""
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in files]
    try:
        full = j.encode(j.render(j.conv(msgs, VIDEO_QUESTION), add_generation_prompt=True), images)
    finally:
        for im in images:
            im.close()
    return full["input_ids"][0].tolist() == prefix_ids + qid


# ----------------------------------------------------------------------------------------------- CPU dry run

def processor_judge(model_id=MODEL):
    """Judge without weights (processor, tokenizer, chat template, Yes/No ids), for CPU checks."""
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
    j.yes_ids, j.no_ids = j.label_ids("Yes"), j.label_ids("No")
    return j


@torch.no_grad()
def selftest():
    """CPU check of the read path on a small random Qwen3-VL (src/qwen_synthetic_fixture: real Qwen3VLModel, fixture
    tokenizer / renderer; nothing pretrained, nothing saved). Asserts, in float32: every cached branch equals the plain
    call (verify path), the direct Yes/No copy of the verdict question reproduces z_video after the crop, reading the
    questions in reverse order gives the same reads (branch isolation), all token seams are clean. In bfloat16: the
    verify gates pass. Returns the worst differences."""
    import tempfile
    import numpy as np
    from PIL import Image
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from src.qwen_synthetic_fixture import fixture
    torch.manual_seed(0)
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=128, max_position_embeddings=8192,
                         rope_parameters={"rope_type": "default", "rope_theta": 5000000., "mrope_section": [24, 20, 20],
                                          "mrope_interleaved": True}),
        vision_config=dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=2, spatial_merge_size=2,
                           temporal_patch_size=2, out_hidden_size=64, num_position_embeddings=16,
                           deepstack_visual_indexes=[0, 1, 2]),
        image_token_id=127, video_token_id=124, vision_start_token_id=125, vision_end_token_id=126)
    cfg._attn_implementation = cfg.text_config._attn_implementation = cfg.vision_config._attn_implementation = "sdpa"
    specs = validate([
        {"name": "verdict_copy", "kind": "yesno", "after": "direct", "text": VIDEO_QUESTION},
        {"name": "ladder_direct", "kind": "yesno", "after": "direct", "text": "Does it take up more than half ? Yes or No"},
        {"name": "ladder_stance", "kind": "yesno", "after": "stance", "text": "Does it take up more than half ? Yes or No"},
        {"name": "opts_stance", "kind": "options", "after": "stance", "options": ["A", "B", "C"], "text": "Pick A B or C"},
        {"name": "opts_direct", "kind": "options", "after": "direct", "options": ["A", "B", "C"], "text": "Pick A B or C"}])
    worst = {}
    with tempfile.TemporaryDirectory() as tmp:
        native = Path(tmp) / "frame.png"
        Image.fromarray(np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(native)
        for dtype in (torch.float32, torch.bfloat16):
            model = Qwen3VLModel(cfg).eval().to(dtype)
            model.requires_grad_(False)
            j, frames, hooks = fixture(model, 6, dtype, native)
            j.tok.decode = lambda ids: " ".join(map(str, ids))
            try:
                sets = {s["name"]: [[j.tok(lab)["input_ids"][0]] for lab in s["options"]] for s in specs if s["kind"] == "options"}
                segs = [(0.0, 6.0, "fixture speech")]
                rec = read_prefix_questions(j, frames, segs, specs, sets, verify=True)
                rev = read_prefix_questions(j, frames, segs, specs[::-1], sets)
                q, chk = rec["questions"], rec["verify"]
                d_order = max(abs(q[n]["z"] - rev["questions"][n]["z"]) if q[n]["kind"] == "yesno" else
                              max(abs(q[n]["logp"][k] - rev["questions"][n]["logp"][k]) for k in q[n]["logp"]) for n in q)
                d_copy = abs(q["verdict_copy"]["z"] - rec["z_video"])
                d_plain = max(c.get("dz", c.get("max_abs_dlogp", 0.0)) for c in chk.values() if isinstance(c, dict))
                # subset prefixes: the full inputs reproduce z_video; a 2-frame span and a frame-less span read and verify
                full = read_subset_verdict(j, frames, segs, verify=True)
                spans = [read_subset_verdict(j, frames_in(frames, 2.0, 4.0), transcript_in(segs, 2.0, 4.0), verify=True),
                         read_subset_verdict(j, [], transcript_in(segs, 4.0, 6.0), verify=True)]
                assert [s["n_frames"] for s in spans] == [2, 0], spans
                d_subset = max([full["verify"]["dz"]] + [s["verify"]["dz"] for s in spans])
                tag = str(dtype).split(".")[-1]
                worst[tag] = {"cache_vs_plain": d_plain, "verdict_copy_vs_z_video": d_copy, "order": d_order,
                              "z_video_cache_vs_plain": chk["dz_video_cache_vs_plain"],
                              "subset_full_vs_z_video": abs(full["z"] - rec["z_video"]), "subset_cache_vs_plain": d_subset,
                              "seams_ok": all(c["seam_ok"] for c in chk.values() if isinstance(c, dict))
                              and full["verify"]["seam_ok"] and all(s["verify"]["seam_ok"] for s in spans),
                              "verify_failed": verify_failed(chk), "stance": rec["stance"]}
                assert worst[tag]["seams_ok"] and not worst[tag]["verify_failed"] and d_subset < GATE_DZ, worst[tag]
                assert worst[tag]["subset_full_vs_z_video"] < 1e-6, worst[tag]
                if dtype == torch.float32:
                    assert max(d_plain, d_copy, d_order, d_subset, chk["dz_video_cache_vs_plain"]) < 1e-3, worst[tag]
                    assert abs(sum(math.exp(x) for x in q["opts_stance"]["logp"].values()) - 1) < 1e-5
            finally:
                for h in hooks:
                    h.remove()
    return worst


def reference_reads(datasets):
    out = {}
    for path in sorted({REFERENCE[ds] for ds in datasets}):
        for line in open(path):
            r = json.loads(line)
            if not r.get("error") and r["dataset"] in datasets:
                out[(r["dataset"], r["video_id"])] = {"z_video": r["extra"]["z_video"], "stance": r["extra"]["stance"],
                                                      "prefix_tokens": r["extra"].get("prefix_tokens")}
    return out


def selected_rows(datasets, limit):
    rows = []
    for ds in datasets:
        rr = load_manifest(MANIFEST[ds], [ds])
        rows += rr[:limit] if limit else rr
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", help=".json list of specs or .txt blocks (see the module doc)")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"], choices=sorted(MANIFEST))
    ap.add_argument("--limit", type=int, default=2, help="first N videos of every corpus (0 = all)")
    ap.add_argument("--variants", choices=["case_space", "bare"], default="case_space")
    ap.add_argument("--model", default=MODEL)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="CPU, no weights")
    g.add_argument("--smoke", action="store_true", help="GPU (Slurm job only): reads + verify, printed, no files")
    g.add_argument("--selftest", action="store_true", help="CPU: read path on a small random Qwen3-VL (no weights)")
    a = ap.parse_args()
    if a.selftest:
        print("SELFTEST", json.dumps(selftest()))
        print("SELFTEST_PASS")
        return
    if not a.questions:
        raise SystemExit("--questions is required")
    specs = load_questions(a.questions)
    rows = selected_rows(a.datasets, a.limit)
    asr = {ds: load_asr(ds) for ds in a.datasets}
    ref = reference_reads(a.datasets)
    j = processor_judge(a.model) if a.dry_run else Judge(model_id=a.model)
    sets, report = prepare(j, specs, a.variants)
    print("ANSWER TOKENS", json.dumps(report, ensure_ascii=False))
    print("QUESTIONS", json.dumps([{k: v for k, v in s.items() if k != "text"} for s in specs]))
    shown = False
    for row in rows:
        ds, vid = row["dataset"], row["video_id"]
        frames = frame_paths(ds, vid, NATIVE_FRAMES, "k20")
        segments = asr[ds].get(vid, [])
        r6 = ref.get((ds, vid), {})
        if a.dry_run:
            msgs, files = j.prefix_messages(frames, segments)
            seams = seam_checks(j, msgs, files, specs, r6.get("stance", "Yes"))
            if not shown:
                for s in specs:
                    print(f"---- {s['name']} ({s['kind']}, after {s['after']}) branch text:\n{seams[s['name']][2]}----")
                shown = True
            _, enc = j.encode_prefix(msgs, files)
            print("DRY", json.dumps({"dataset": ds, "video_id": vid, "n_frames": len(frames),
                                     "prefix_tokens": int(enc["input_ids"].shape[1]), "prefix_tokens_r6": r6.get("prefix_tokens"),
                                     "seam_ok": {n: v[0] for n, v in seams.items()}}))
            continue
        rec = read_prefix_questions(j, frames, segments, specs, sets, verify=True)
        rec["z_video_r6"], rec["stance_r6"] = r6.get("z_video"), r6.get("stance")
        print("SMOKE", json.dumps({"dataset": ds, "video_id": vid, **rec}, ensure_ascii=False))
        bad = verify_failed(rec["verify"])
        if bad:
            raise SystemExit(f"VERIFY GATE FAILED {ds}/{vid}: {bad}")
    print("DONE prefix_questions", "dry-run" if a.dry_run else "smoke", len(rows))


if __name__ == "__main__":
    main()
