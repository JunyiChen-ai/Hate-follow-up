"""Entropy-adaptation kill test on HateMM test (215 videos).

Pre-registration: docs/duplex/PREREG_entropy_tta_killtest.md.

TENT-style test-time adaptation of Qwen3-VL-8B-Instruct. Only normalization
gains are trainable. The objective is the Shannon entropy of the renormalized
two-way {Yes, No} answer distribution at the final prompt position. No labels
of any kind enter training.

Every piece of the judge input -- prompt, reader block, frames, transcript
overrides, pixel budget, and the z readout -- is imported from
`src/duplex/extract_duplex_readout.py` and is never redefined here.

Subcommands:
  score  --model-state <none|path>   one forward pass per video, writes scores.jsonl
  train  --arm <real|placebo>        one epoch of entropy minimization, saves norm gains

Both write a STATUS file and a DONE marker so a detached run can be polled.
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
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS, MIN_PIXELS, build_messages, resolve_frames, resolve_transcript,
)
from score_duplex_probe import (  # noqa: E402
    YOUTUBE_RULES, build_binary_token_ids,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

DATASET = "HateMM"
SPLIT = "test"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
WORK = os.path.join(ROOT, "results", "testruns", "hatemm")
OVERRIDES_JSON = os.path.join(WORK, "c2_overrides.json")
OUT_ROOT = os.path.join(ROOT, "results", "entropy_tta")
SEED = 20260808


# --------------------------------------------------------------- bookkeeping

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])


def write_status(path, text):
    with open(path, "w") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------- inputs

def corpus():
    """The 215 test ids, their annotations, and the frozen transcript overrides."""
    ann = load_annotations(DATASET)
    ids = load_clean_split_ids(DATASET, SPLIT)
    seen, out = set(), []
    for v in ids:
        if v not in seen:
            seen.add(v)
            out.append(v)
    with open(OVERRIDES_JSON) as f:
        overrides = json.load(f)
    return out, ann, overrides


def derangement(n, seed):
    """A permutation of range(n) with no fixed point, by rejection sampling."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p.tolist()


def preflight(ids, num_frames=16):
    """Decode every frame of every video before the GPU pass starts."""
    from PIL import Image
    bad = []
    for vid in ids:
        paths = resolve_frames(vid, DATASET, num_frames)
        if len(paths) != num_frames:
            bad.append((vid, f"{len(paths)} frames"))
            continue
        try:
            for p in paths:
                with Image.open(p) as im:
                    im.convert("RGB").load()
        except Exception as e:  # noqa: BLE001
            bad.append((vid, repr(e)))
    if bad:
        raise SystemExit(f"pre-flight decode failed for {len(bad)} videos: {bad[:5]}")
    logging.info(f"pre-flight: decoded {num_frames} frames for all {len(ids)} videos")


# -------------------------------------------------------------------- model

def load_model(processor_only=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor_only:
        return processor, None
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    return processor, model


def norm_gain_params(model, include_vision=True):
    """Named parameters that are normalization gains.

    The language tower uses Qwen3VLTextRMSNorm (a bare `weight` gain, no bias).
    The vision tower uses nn.LayerNorm, which also carries a bias; only its
    gain is adapted, matching the pre-registration's wording.
    """
    names = []
    for mod_name, mod in model.named_modules():
        cls = type(mod).__name__
        if "RMSNorm" not in cls and not isinstance(mod, torch.nn.LayerNorm):
            continue
        is_vision = ".visual." in f".{mod_name}." or mod_name.startswith("visual.") \
            or ".model.visual" in mod_name
        if is_vision and not include_vision:
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        names.append(f"{mod_name}.weight")
    name_set = set(names)
    return [(n, p) for n, p in model.named_parameters() if n in name_set]


def build_inputs(processor, ann, vid, frame_paths, transcript):
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    messages = build_messages(ann, frame_paths, YOUTUBE_RULES, transcript)
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt",
                       size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})
    for im in images:
        im.close()
    return inputs


def readout_z(logits, yes_idx, no_idx):
    """z = logsumexp(Yes logits) - logsumexp(No logits) at the final position."""
    last = logits[0, -1, :].float()
    return torch.logsumexp(last[yes_idx], dim=0) - torch.logsumexp(last[no_idx], dim=0)


def binary_entropy(z):
    """Shannon entropy, in nats, of the renormalized {Yes, No} distribution."""
    p = torch.sigmoid(z)
    sp = torch.nn.functional.softplus
    return p * sp(-z) + (1.0 - p) * sp(z)


def label_token_idx(processor, device):
    ids = build_binary_token_ids(processor.tokenizer)
    yes_ids, no_ids = sorted(ids["Yes"]), sorted(ids["No"])
    if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
        raise SystemExit(f"bad label token sets Yes={yes_ids} No={no_ids}")
    return (torch.tensor(yes_ids, device=device),
            torch.tensor(no_ids, device=device))


# ------------------------------------------------------------------- score

def cmd_score(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    status = os.path.join(out_dir, "STATUS")
    scores_path = os.path.join(out_dir, "scores.jsonl")
    if os.path.exists(scores_path):
        os.remove(scores_path)

    ids, ann, overrides = corpus()
    write_status(status, f"score: pre-flight over {len(ids)} videos")
    preflight(ids)

    processor, model = load_model()
    yes_idx, no_idx = label_token_idx(processor, model.device)

    if args.model_state and args.model_state != "none":
        sd = torch.load(args.model_state, map_location="cpu")
        missing = load_norm_state(model, sd)
        logging.info(f"loaded {len(sd)} adapted norm tensors from {args.model_state} "
                     f"(unmatched: {missing})")

    t0 = time.time()
    with open(scores_path, "a") as fh:
        for i, vid in enumerate(ids):
            frame_paths = resolve_frames(vid, DATASET, 16)
            transcript = resolve_transcript(ann[vid], vid, overrides, 0)
            inputs = build_inputs(processor, ann[vid], vid, frame_paths, transcript)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                out = model(**inputs, use_cache=False, logits_to_keep=1)
                z = float(readout_z(out.logits, yes_idx, no_idx))
            del out, inputs
            if not np.isfinite(z):
                raise SystemExit(f"{vid}: non-finite z")
            fh.write(json.dumps({"video_id": vid, "z": z,
                                 "p_yes_renorm": float(1.0 / (1.0 + np.exp(-z)))}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if (i + 1) % 25 == 0 or i == 0:
                write_status(status, f"score: {i + 1}/{len(ids)}")
                logging.info(f"  [{i + 1}/{len(ids)}] {vid} z={z:+.3f} "
                             f"{(time.time() - t0) / (i + 1):.2f}s/video")

    n = sum(1 for _ in open(scores_path))
    if n != len(ids):
        raise SystemExit(f"expected {len(ids)} scored rows, got {n}")
    write_status(status, "DONE")
    with open(os.path.join(out_dir, "DONE"), "w") as f:
        f.write(f"n_scored={n} seconds={time.time() - t0:.1f}\n")
    logging.info(f"scored {n} videos in {time.time() - t0:.1f}s")


def load_norm_state(model, sd):
    own = dict(model.named_parameters())
    unmatched = []
    with torch.no_grad():
        for k, v in sd.items():
            if k not in own:
                unmatched.append(k)
                continue
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    return unmatched


# ------------------------------------------------------------------- train

def cmd_train(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    status = os.path.join(out_dir, "STATUS")
    traj_path = os.path.join(out_dir, "entropy_trajectory.jsonl")
    if os.path.exists(traj_path):
        os.remove(traj_path)

    ids, ann, overrides = corpus()
    write_status(status, f"train[{args.arm}]: pre-flight over {len(ids)} videos")
    preflight(ids)

    # Fixed shuffled visiting order, and -- for the placebo arm -- a fixed
    # derangement pairing each video's frames with another video's transcript.
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(len(ids)))
    der = derangement(len(ids), SEED) if args.arm == "placebo" else None

    processor, model = load_model()
    yes_idx, no_idx = label_token_idx(processor, model.device)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = norm_gain_params(model, include_vision=not args.no_vision_norms)
    for _, p in trainable:
        p.requires_grad_(True)
    n_par = sum(p.numel() for _, p in trainable)
    logging.info(f"trainable norm gains: {len(trainable)} tensors, {n_par} scalars "
                 f"(vision tower {'frozen' if args.no_vision_norms else 'adapted'})")

    # Gradient checkpointing in this transformers version is gated on
    # module.training, so the model is put in train mode and every dropout is
    # asserted inert -- there is then no stochasticity in the forward pass.
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout) and mod.p > 0:
            raise SystemExit(f"active dropout {mod} would make training stochastic")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)

    t0 = time.time()
    with open(traj_path, "a") as fh:
        for step, k in enumerate(order):
            vid = ids[k]
            frame_paths = resolve_frames(vid, DATASET, 16)
            t_vid = ids[der[k]] if der is not None else vid
            transcript = resolve_transcript(ann[t_vid], t_vid, overrides, 0)
            inputs = build_inputs(processor, ann[vid], vid, frame_paths, transcript)
            inputs = inputs.to(model.device)

            out = model(**inputs, use_cache=False, logits_to_keep=1)
            z = readout_z(out.logits, yes_idx, no_idx)
            loss = binary_entropy(z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            rec = {"step": step, "entropy_nats": float(loss.detach()),
                   "z_before_update": float(z.detach())}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            del out, inputs, z, loss

            if (step + 1) % 10 == 0 or step == 0:
                write_status(status, f"train[{args.arm}]: {step + 1}/{len(order)}")
            if (step + 1) % 25 == 0 or step == 0:
                logging.info(f"  [{step + 1}/{len(order)}] H={rec['entropy_nats']:.4f} "
                             f"z={rec['z_before_update']:+.2f} "
                             f"{(time.time() - t0) / (step + 1):.2f}s/step "
                             f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")

    state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in trainable}
    torch.save(state, os.path.join(out_dir, "norm_gains.pt"))
    write_status(status, "DONE")
    with open(os.path.join(out_dir, "DONE"), "w") as f:
        f.write(f"steps={len(order)} seconds={time.time() - t0:.1f}\n")
    logging.info(f"trained {len(order)} steps in {time.time() - t0:.1f}s; "
                 f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--model-state", default="none")
    s.set_defaults(func=cmd_score)

    t = sub.add_parser("train")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--arm", choices=["real", "placebo"], required=True)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--no-vision-norms", action="store_true",
                   help="memory lever (b): adapt language-tower norms only")
    t.set_defaults(func=cmd_train)

    args = ap.parse_args()
    torch.manual_seed(SEED)
    args.func(args)


if __name__ == "__main__":
    main()
