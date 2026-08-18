#!/usr/bin/env python
"""Vad-R1 zero-shot inference over the reproduction study's three test splits.

Upstream: https://github.com/wbfwonderful/Vad-R1 @ 8536296 (NeurIPS 2025),
released checkpoint https://huggingface.co/wbfwonderful/Vad-R1, base model
Qwen2.5-VL-7B-Instruct. The generation path here is upstream's
`inference/inference-vllm.py` with four changes and nothing else:

  1. the video list comes from this study's frozen split manifests rather than
     `Vad-Reasoning-SFT-test.jsonl`;
  2. the output record carries `video_id` plus the parse of `<which>` and
     `<when>`, so the rasteriser downstream never re-reads free text;
  3. the run is resumable, because 534 videos over three corpora is a long
     single job;
  4. frames are pulled with ffmpeg on the videos OpenCV cannot decode. The
     OpenCV build in this environment has no AV1 decoder, and 37 of the 158
     MultiHateClip English test videos and 8 of the 153 Chinese ones are AV1;
     upstream's loop returns an empty array on those files. The *index set* is
     unchanged -- it is upstream's linspace over the container's frame count,
     chosen before any decoding -- and on the H.264 videos both decoders were
     checked to return bit-identical frames, so the fallback closes a codec
     gap rather than changing the sample. `--dry-run` re-runs that check on
     every OpenCV-readable file it touches.

The prompt, the system message, the frame index policy, the frame compressor,
the vLLM engine arguments and the sampling parameters are byte-for-byte
upstream. `--verify-prompt` re-derives the two prompt strings from a local
clone of the upstream repository and compares them to the copies embedded
below; the SHA256 pins are checked on every run regardless.

What the model is asked
-----------------------
The released prompt asks whether the video is "abnormal or normal" and, for an
abnormal video, for the time range of the anomaly in normalised units. It never
mentions hate. That is the point of the primary arm: Vad-R1 is evaluated here
exactly as released, so the number it produces is the number a reader gets by
downloading the checkpoint and pointing it at hateful video. Running it under a
substituted vocabulary would answer a different question and would no longer be
a zero-shot reading of the released system.

`--arm hateful` implements that different question as a *term-adaptation
ablation*: the same prompt with "abnormal" rewritten to "hateful" and "normal"
to "non-hateful" throughout, including the `<which>` vocabulary the parser then
expects. It exists so the substitution is specified rather than improvised, and
it is not part of the zero-shot result. **Do not run it without owner approval**
-- it is a second condition on the same test split, and whether it is a fair
comparison or a second bite at the benchmark is a call for the owner, not for
this script.

Environment
-----------
vLLM 0.11.0 and transformers 4.57.1, the versions already in
/home/jehc223/venvs/SafetyContradiction. Upstream's `setup.sh` pins vLLM 0.7.2
and its README ships `transformers-main.zip`; that zip is an unmodified
snapshot of transformers main at 4.49.0.dev0, which is simply the state of the
tree before Qwen2.5-VL support was released, and it carries no Vad-R1 edits
(the string "vad" does not appear in its `modeling_qwen2_5_vl.py`). vLLM 0.11.0
requires transformers>=4.55.2 and implements Qwen2.5-VL itself, so installing
the bundled snapshot would break the engine and buy nothing. It is not
installed. transformers is used here only for `AutoProcessor`, i.e. the chat
template and the preprocessor config.

Two details of that stack were checked against the upstream call, because both
could have silently changed the model's input:

  * `mm_processor_kwargs={"fps": 1}` reaches `Qwen2_5_VLProcessor.__call__`,
    which sets `second_per_grid_ts = temporal_patch_size / fps = 2.0`. Same as
    upstream.
  * `Qwen2VLVideoProcessor.do_sample_frames` defaults to False in 4.57.1, so
    the 16 frames sampled here are the 16 frames the vision tower sees. The
    processor does not re-sample them.

A bare `(T, H, W, C)` ndarray is a valid `multi_modal_data["video"]` item in
vLLM 0.11.0, and Qwen2.5-VL is not one of the models that demand video
metadata, so upstream's ndarray hand-off works unchanged.

Output
------
results/reproduction/baselines/vadr1/<corpus>/generations.jsonl, one object per
video, and run_meta.json beside it. Scoring is not done here; see
rasterize_and_eval.py.

Usage
-----
    python run_vadr1_inference.py --corpus hatemm \
        --model /home/jehc223/data/checkpoints/vad_r1

    python run_vadr1_inference.py --corpus hatemm --dry-run 5   # CPU only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASELINES))
sys.path.insert(0, BASELINES)

from hate_common import data as hdata          # noqa: E402

DEFAULT_MODEL = "/home/jehc223/data/checkpoints/vad_r1"
UPSTREAM_CLONE = os.path.join(PROJECT_ROOT, "third_party", "Vad-R1")
UPSTREAM_SHA = "8536296b748d389dfca2d8f81a9703aa57404bc2"

VIDEO_DIRS = {
    "hatemm": "/home/jehc223/data/HateMM/video",
    "mhclip_en": "/home/jehc223/data/Multihateclip/English/video_mp4",
    "mhclip_zh": "/home/jehc223/data/Multihateclip/Chinese/video",
}
VIDEO_EXT = ".mp4"

# ------------------------------------------------------- upstream constants
NUM_FRAMES = 16
MAX_PIXELS = 128 * 28 * 28
ALIGN_FACTOR = 28

# Verbatim from inference/inference-vllm.py and inference-vllm-single.py, which
# carry identical copies. Do not reflow: the pins below are over these bytes.
BINARY_PROMPT = (
    'Your task is to analyze whether the given video is abnormal or normal. Think before answering, and generate:\n'
    '1. A structured reasoning process enclosed in <think></think> tags\n'
    '2. A final explanation enclosed in <answer></answer> tags\n'
    '\n'
    'For abnormal videos, the reasoning should be based on a structured 4-step process:\n'
    '<think> must include the following four steps enclosed in corresponding tags:\n'
    '<step1>: Scene Description — Provide an objective overview of the environment and normal behaviors, without mentioning any abnormal activity or speculation.\n'
    '<step2>: Abnormal Event Description — Describe the abnormal event and its approximate spatial location (e.g., bottom left of the frame), without explaining why it is abnormal.\n'
    '<step3>: Abnormal Event Recognition — Explain why this event is considered abnormal compared to normal patterns or expectations.\n'
    '<step4>: Causal Reasoning and Social Norms — Analyze potential negative consequences and explain how this behavior violates social norms or expectations.\n'
    '\n'
    '<answer> must be a single, coherent paragraph in natural language, which includes exactly the following five tags:\n'
    '<which>: Define the video as "Abnormal."\n'
    '<what>: What happened (describe the anomalous event)\n'
    '<when>: When it happened, in normalized frame indices (e.g., <when>[0.25, 0.45]</when>)\n'
    '<where>: Where it happened (use approximate spatial descriptions)\n'
    '<why>: Why it is considered abnormal\n'
    '<how>: How this behavior could cause harm or violate norms\n'
    '\n'
    'Example Output for Abnormal Videos:\n'
    '<think> \n'
    '<step1>The video shows ...</step1>\n'
    '<step2>Next, we observe an abnormal event ...</step2> \n'
    '<step3>Based on these observations ...</step3> \n'
    '<step4>As a result, this behavior ...</step4> \n'
    '</think> \n'
    '<answer>\n'
    'The video is classified as <which>Abnormal</which>. In this video, <what>a pedestrian ...</what>, occurring during the time range <when>[0.121, 0.826]</when>. The event takes place approximately in the <where>lower-left area of the frame</where>. This is considered abnormal because <why>pedestrians are expected to ...</why>. As a result, <how>such behavior could ...</how>.\n'
    '</answer>\n'
    '\n'
    'For normal videos, the reasoning should be simplified to just two steps:\n'
    '<think> must include only the following two steps:\n'
    '<step1>: Scene and Object Description — Provide a concise and objective overview of the environment and typical behaviors, without mentioning anomalies.\n'
    '<step2>: Normal Event Explanation — Explain why the video is considered normal.\n'
    '\n'
    '<answer> must be a single, coherent paragraph in natural language, which includes the following three tags:\n'
    '<which>: Define the video as "Normal."\n'
    '<what>: A concise description of the event in the video.\n'
    '<why>: Why it is considered normal.\n'
    '\n'
    'Example Output for Normal Videos:\n'
    '<think> \n'
    '<step1>The video shows scenes ...</step1> \n'
    '<step2>Based on the described scenes ...</step2> \n'
    '</think> \n'
    '<answer>\n'
    'The video is classified as <which>Normal</which>. In this video, <what>people are ...</what>. This is considered normal because <why>...</why>\n'
    '</answer>\n'
    ''
)

SYSTEM_PROMPT = ("You are a multimodal reasoning assistant for "
                 "understanding anomalies in videos.")

BINARY_PROMPT_SHA = \
    "cb673111d1b01d00eb4094b4aac51bd4da54d40fc8b785402ef1eee31be8caac"
SYSTEM_PROMPT_SHA = \
    "7bf05ce3b7d793963696dc54b019903d844223a3e6ff78802b53a9c9d4b63b01"

SAMPLING = {"temperature": 0.1, "top_p": 0.9, "max_tokens": 512}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_prompt_pins():
    """Fail loudly if the embedded prompts have drifted from the pins."""
    if _sha(BINARY_PROMPT) != BINARY_PROMPT_SHA:
        raise SystemExit("ABORT: BINARY_PROMPT sha256 is %s, pinned %s"
                         % (_sha(BINARY_PROMPT), BINARY_PROMPT_SHA))
    if _sha(SYSTEM_PROMPT) != SYSTEM_PROMPT_SHA:
        raise SystemExit("ABORT: SYSTEM_PROMPT sha256 is %s, pinned %s"
                         % (_sha(SYSTEM_PROMPT), SYSTEM_PROMPT_SHA))


def verify_against_clone(clone=UPSTREAM_CLONE):
    """Re-derive both prompts from the upstream clone and compare.

    third_party/ is gitignored, so the clone is not guaranteed to be present;
    this is an opt-in check, not a precondition of a run.
    """
    import ast
    ok = True
    for name in ("inference-vllm.py", "inference-vllm-single.py"):
        path = os.path.join(clone, "inference", name)
        if not os.path.isfile(path):
            print("  %-28s MISSING (clone absent?)" % name)
            ok = False
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        found = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id in ("BINARY_PROMPT", "SYSTEM_PROMPT")):
                    found[target.id] = ast.literal_eval(node.value)
        for key, ours in (("BINARY_PROMPT", BINARY_PROMPT),
                          ("SYSTEM_PROMPT", SYSTEM_PROMPT)):
            same = found.get(key) == ours
            ok &= same
            print("  %-28s %-14s %s" % (name, key, "OK" if same else "DIFFERS"))
    return ok


# ------------------------------------------------------------------- arms
def _to_hateful_terms(text):
    """Literal term substitution for the term-adaptation ablation.

    Order matters: "Abnormal" contains "normal", so the abnormal forms are
    rewritten before the normal ones.
    """
    pairs = (
        ("Abnormal", "Hateful"), ("abnormal", "hateful"),
        ("Anomalous", "Hateful"), ("anomalous", "hateful"),
        ("Anomalies", "Hate"), ("anomalies", "hate"),
        ("Anomaly", "Hate"), ("anomaly", "hate"),
        ("Normal", "Non-hateful"), ("normal", "non-hateful"),
    )
    for old, new in pairs:
        text = text.replace(old, new)
    return text


ARMS = {
    # name: (binary prompt, system prompt, positive which tokens,
    #        negative which tokens)
    "anomaly": (BINARY_PROMPT, SYSTEM_PROMPT, ("abnormal",), ("normal",)),
    "hateful": (_to_hateful_terms(BINARY_PROMPT),
                _to_hateful_terms(SYSTEM_PROMPT),
                ("hateful",), ("non-hateful", "nonhateful")),
}


def arm_spec(arm):
    if arm not in ARMS:
        raise SystemExit("ABORT: unknown arm %r" % (arm,))
    return ARMS[arm]


# ---------------------------------------------------------------- parsing
# Both regexes are upstream's, from evaluation/1-evaluate_detection.py.
_WHEN_RE = re.compile(r"<when>\[([0-9.]+)\s*,\s*([0-9.]+)\]</when>", re.DOTALL)
_WHICH_RE = re.compile(r"<which>(.*?)</which>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def extract_when(text):
    """[start, end] as normalised fractions, or None when absent.

    Upstream returns [0.0, 0.0] on a miss, which is indistinguishable from a
    model that literally emitted [0.0, 0.0]. None keeps the two apart so the
    rasteriser can count parse failures.
    """
    m = _WHEN_RE.search(text or "")
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2))]


def extract_which(text):
    """Raw <which> contents, or None when the tag is absent."""
    m = _WHICH_RE.search(text or "")
    return m.group(1) if m else None


def normalise_which(raw, positive, negative):
    """Map raw <which> text to 'positive' / 'negative' / None.

    Tags, whitespace, quotes and a trailing period are stripped, because the
    prompt's own instruction line reads `Define the video as "Abnormal."` and
    the model does sometimes copy the punctuation.
    """
    if raw is None:
        return None
    text = _TAG_RE.sub("", raw).replace("\n", " ").strip()
    text = text.strip().strip('"').strip("'").strip().rstrip(".").strip()
    text = text.lower()
    if text in [t.lower() for t in positive]:
        return "positive"
    if text in [t.lower() for t in negative]:
        return "negative"
    return None


def parse_generation(text, arm="anomaly"):
    """Full parse of one generation into the fields the rasteriser reads.

    ``verdict`` is the model's own class call; ``when`` is the normalised
    interval it gave, if any. ``status`` names what the rasteriser will do:

        negative              model called the video normal -> all-zero
        positive_interval     model called it abnormal and gave an interval
        positive_no_interval  called it abnormal, no parseable <when>
        unparsed_interval     no usable <which>, but a <when> was present
        unparsed              neither tag usable
    """
    _, _, positive, negative = arm_spec(arm)
    raw_which = extract_which(text)
    verdict = normalise_which(raw_which, positive, negative)
    when = extract_when(text)
    if verdict == "negative":
        status = "negative"
    elif verdict == "positive":
        status = "positive_interval" if when else "positive_no_interval"
    else:
        status = "unparsed_interval" if when else "unparsed"
    return {
        "which_raw": raw_which,
        "verdict": verdict,
        "when": when,
        "parse_status": status,
        "which_nonstandard": raw_which is not None and verdict is None,
    }


# ------------------------------------------------------------------ video
def compress_frame(frame, max_pixels=MAX_PIXELS, factor=ALIGN_FACTOR):
    """Verbatim upstream."""
    import cv2
    h, w = frame.shape[:2]
    if (h * w <= max_pixels) and (h % factor == 0) and (w % factor == 0):
        return frame
    scale = (max_pixels / (h * w)) ** 0.5
    new_h = max(int((h * scale) // factor) * factor, factor)
    new_w = max(int((w * scale) // factor) * factor, factor)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_video(path, num_frames=NUM_FRAMES):
    """Upstream's sampler, with an ffmpeg fallback for AV1.

    The index set is upstream's and is chosen before any decoding happens:
    `np.linspace(0, total - 1, num_frames)` over the frame count OpenCV
    reports. Which decoder then fetches those indices does not change which
    frames the model sees.

    The fallback exists because OpenCV in this environment is built without an
    AV1 decoder, and a quarter of the MultiHateClip English test split and a
    handful of the Chinese one are AV1. On those files `cap.read()` returns
    False on the first call, so upstream's loop would hand the model an empty
    array. `cv2.CAP_PROP_FRAME_COUNT` is still read correctly from the
    container, so the fallback selects exactly the indices upstream would have
    selected and pulls them with ffmpeg's `select` filter. H.264 files never
    reach the fallback.

    Returns (frames, short, decoder).
    """
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError("Cannot read video: %s" % path)
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = compress_frame(frame)
            frames.append(frame)
    cap.release()
    decoder = "opencv"
    if not frames:
        frames = _sample_frames_ffmpeg(path, idxs)
        decoder = "ffmpeg"
    if not frames:
        raise RuntimeError("decoded 0 frames from %s" % path)
    short = len(frames) != num_frames
    return np.stack(frames), short, decoder


def _sample_frames_ffmpeg(path, idxs):
    """Pull the given 0-based frame indices with ffmpeg, RGB, compressed.

    rgb24 out of ffmpeg is the same channel order `cv2.cvtColor(..., BGR2RGB)`
    produces, and `compress_frame` is then applied unchanged, so a frame
    reaching the model through this path is the frame the OpenCV path would
    have produced had the codec been supported.
    """
    import json as _json
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError("ffprobe failed on %s: %s"
                           % (path, probe.stderr.strip()[:200]))
    streams = _json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise RuntimeError("no video stream in %s" % path)
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])

    wanted = sorted({int(i) for i in idxs})
    expr = "+".join(r"eq(n\,%d)" % i for i in wanted)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", "select=%s" % expr,
         "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed on %s: %s"
                           % (path, proc.stderr.decode("utf-8", "ignore")[:200]))
    stride = width * height * 3
    n = len(proc.stdout) // stride
    if n == 0:
        return []
    block = np.frombuffer(proc.stdout[:n * stride], dtype=np.uint8)
    block = block.reshape(n, height, width, 3)

    # Upstream's index list repeats an index when the video is shorter than
    # num_frames; ffmpeg's select emits each matching frame once, so the
    # repeats are restored here against the deduplicated list.
    by_index = {idx: block[k] for k, idx in enumerate(wanted[:n])}
    out = []
    for idx in idxs:
        frame = by_index.get(int(idx))
        if frame is not None:
            out.append(compress_frame(np.ascontiguousarray(frame)))
    return out


def video_path(corpus, video_id):
    return os.path.join(VIDEO_DIRS[corpus], video_id + VIDEO_EXT)


# -------------------------------------------------------------------- run
def out_dir(corpus, arm, root=None):
    root = root or os.path.join(PROJECT_ROOT, "results", "reproduction",
                                "baselines", "vadr1")
    return os.path.join(root, corpus if arm == "anomaly"
                        else "%s_%s" % (corpus, arm))


def already_done(path):
    """Video ids that already have a *successful* record in the jsonl.

    A record carrying an `error` field is not counted, so a rerun retries the
    videos that failed to decode. The retry appends a second line for that id;
    the rasteriser keys on video_id and keeps the last one, so the good record
    wins.
    """
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                done.discard(rec["video_id"])
            else:
                done.add(rec["video_id"])
    return done


def cohort(corpus, split, limit=None):
    """Split ids that have gold, in manifest order.

    The split manifest is the cohort of record; the gold npz drops the handful
    of videos build_gt_arrays.py excluded (one HateMM positive with no usable
    span, and the MultiHateClip videos without local media). Generating for a
    video with no gold would produce a record nothing can score, so the
    intersection is taken here and the drop is reported.
    """
    gt = hdata.gt_arrays(corpus, split)
    ids = [v for v in hdata.load_split(corpus, split) if v in gt]
    dropped = [v for v in hdata.load_split(corpus, split) if v not in gt]
    if limit:
        ids = ids[:limit]
    return ids, dropped


STALE_IMAGE_PROCESSOR = "Qwen2_5_VLImageProcessor"
CURRENT_IMAGE_PROCESSOR = "Qwen2VLImageProcessor"


def ensure_processor_compat(model_path, apply=True):
    """Repoint the checkpoint's stale image-processor class name.

    The released checkpoint was saved under transformers 4.49.0.dev0, whose
    `preprocessor_config.json` names `Qwen2_5_VLImageProcessor`. That class was
    a duplicate of `Qwen2VLImageProcessor` and no longer exists in 4.57.1,
    where the auto mapping for `qwen2_5_vl` resolves to `Qwen2VLImageProcessor`
    for both the slow and, via `Qwen2VLImageProcessorFast`, the fast path. So
    `AutoProcessor.from_pretrained` raises "Unrecognized image processor" on
    the checkpoint as shipped, and vLLM raises it too, because it loads the
    same processor.

    Rewriting the class name to the one the checkpoint's own `model_type`
    already resolves to is a rename, not a substitution: the weights, the
    preprocessing constants and the chat template are untouched. The original
    file is kept as preprocessor_config.json.orig.
    """
    path = os.path.join(model_path, "preprocessor_config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg.get("image_processor_type") != STALE_IMAGE_PROCESSOR:
        return None
    if not apply:
        return ("stale image_processor_type %r in %s; rerun without "
                "--no-fix-processor-config" % (STALE_IMAGE_PROCESSOR, path))
    backup = path + ".orig"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
    cfg["image_processor_type"] = CURRENT_IMAGE_PROCESSOR
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    return ("rewrote image_processor_type %s -> %s in %s (original kept at "
            "%s)" % (STALE_IMAGE_PROCESSOR, CURRENT_IMAGE_PROCESSOR, path,
                     backup))


def build_prompt(model_path, arm, fix_processor_config=True):
    from transformers import AutoProcessor
    binary, system, _, _ = arm_spec(arm)
    note = ensure_processor_compat(model_path, apply=fix_processor_config)
    if note:
        print("  processor compat: %s" % note)
    processor = AutoProcessor.from_pretrained(model_path,
                                              trust_remote_code=True)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "video", "video": "dummy_path.mp4",
             "nframes": NUM_FRAMES},
            {"type": "text", "text": binary},
        ]},
    ]
    return processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=list(hdata.CORPORA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arm", default="anomaly", choices=sorted(ARMS),
                    help="anomaly = the released prompt, verbatim, the "
                         "zero-shot arm. hateful = term-adaptation ablation, "
                         "OWNER APPROVAL REQUIRED before running.")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-fix-processor-config", action="store_true",
                    help="do not repoint the checkpoint's stale "
                         "image_processor_type; the run will then fail the "
                         "way the released checkpoint fails under "
                         "transformers 4.57")
    ap.add_argument("--seed", type=int, default=0,
                    help="vLLM engine seed; the released sampling params are "
                         "stochastic (temperature 0.1), so a run is only "
                         "reproducible with the seed recorded")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                    help="CPU only: sample N videos and build the prompt, "
                         "load no model, write nothing")
    ap.add_argument("--verify-prompt", action="store_true",
                    help="compare the embedded prompts to the upstream clone")
    ap.add_argument("--selftest", action="store_true",
                    help="parser unit checks on synthetic generations")
    args = ap.parse_args(argv)

    check_prompt_pins()

    if args.selftest:
        return 0 if selftest() else 1

    if args.verify_prompt:
        print("upstream clone %s (pinned %s)" % (UPSTREAM_CLONE,
                                                 UPSTREAM_SHA[:7]))
        ok = verify_against_clone()
        print("prompt verification %s" % ("PASSED" if ok else "FAILED"))
        if not args.corpus:
            return 0 if ok else 1

    if not args.corpus:
        ap.error("--corpus is required unless --selftest/--verify-prompt")

    ids, dropped = cohort(args.corpus, args.split, args.limit_videos)
    print("%s/%s: %d videos with gold (%d split ids dropped for lack of gold)"
          % (args.corpus, args.split, len(ids), len(dropped)))

    if args.dry_run:
        return dry_run(args, ids[:args.dry_run])

    if args.arm != "anomaly":
        print("WARNING: arm %r is a term-adaptation ablation, not the "
              "zero-shot result. Proceeding because it was asked for "
              "explicitly." % args.arm)

    from vllm import LLM, SamplingParams

    dest = out_dir(args.corpus, args.arm, args.out_root)
    os.makedirs(dest, exist_ok=True)
    gen_path = os.path.join(dest, "generations.jsonl")
    done = already_done(gen_path)
    todo = [v for v in ids if v not in done]
    print("  %d already generated, %d to do" % (len(done), len(todo)))
    if not todo:
        print("  nothing to do")
        return 0

    prompt = build_prompt(args.model, args.arm,
                          not args.no_fix_processor_config)
    binary, system, _, _ = arm_spec(args.arm)

    meta = {
        "upstream": "https://github.com/wbfwonderful/Vad-R1",
        "upstream_commit": UPSTREAM_SHA,
        "model": args.model,
        "arm": args.arm,
        "corpus": args.corpus,
        "split": args.split,
        "num_frames": args.num_frames,
        "max_pixels": MAX_PIXELS,
        "min_pixels": ALIGN_FACTOR * ALIGN_FACTOR,
        "processor_fps": 1,
        "sampling": dict(SAMPLING),
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "binary_prompt_sha256": _sha(binary),
        "system_prompt_sha256": _sha(system),
        "chat_prompt_sha256": _sha(prompt),
        "n_videos_in_cohort": len(ids),
        "split_ids_without_gold": dropped,
    }
    try:
        import vllm as _vllm
        import transformers as _tf
        meta["vllm_version"] = _vllm.__version__
        meta["transformers_version"] = _tf.__version__
    except Exception:  # pragma: no cover - version reporting only
        pass
    with open(os.path.join(dest, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        mm_processor_kwargs={
            "min_pixels": ALIGN_FACTOR * ALIGN_FACTOR,
            "max_pixels": MAX_PIXELS,
            "fps": 1,
        },
        limit_mm_per_prompt={"video": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    sampling = SamplingParams(**SAMPLING)

    t_start = time.time()
    with open(gen_path, "a", encoding="utf-8") as out:
        for n, vid in enumerate(todo, 1):
            path = video_path(args.corpus, vid)
            rec = {"video_id": vid, "video_path": path, "arm": args.arm}
            try:
                frames, short, decoder = sample_video(
                    path, args.num_frames)
            except Exception as exc:
                rec.update({"error": "%s: %s" % (type(exc).__name__, exc),
                            "output": None, "parse_status": "decode_error"})
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print("  [%d/%d] %s DECODE ERROR %s" % (n, len(todo), vid, exc))
                continue
            rec["n_frames_sampled"] = int(frames.shape[0])
            rec["frame_hw"] = [int(frames.shape[1]), int(frames.shape[2])]
            rec["short_sample"] = bool(short)
            rec["decoder"] = decoder

            t0 = time.time()
            result = llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"video": frames}}],
                sampling_params=sampling)
            text = result[0].outputs[0].text
            rec["gen_seconds"] = round(time.time() - t0, 3)
            rec["output"] = text
            rec.update(parse_generation(text, args.arm))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(todo):
                print("  [%d/%d] %s  %s  (%.1f min elapsed)"
                      % (n, len(todo), vid, rec["parse_status"],
                         (time.time() - t_start) / 60.0))

    print("wrote %s" % gen_path)
    return 0


def dry_run(args, ids):
    """CPU-only rehearsal: prompt construction and frame sampling."""
    print("DRY RUN -- no model is loaded, nothing is written")
    binary, system, positive, negative = arm_spec(args.arm)
    print("  arm %r  positive=%s negative=%s" % (args.arm, positive, negative))
    print("  binary prompt sha256 %s (%d chars)" % (_sha(binary), len(binary)))
    print("  system prompt        %r" % system)
    if os.path.isdir(args.model):
        prompt = build_prompt(args.model, args.arm,
                              not args.no_fix_processor_config)
        print("  chat prompt sha256   %s (%d chars)"
              % (_sha(prompt), len(prompt)))
        for token in ("<|vision_start|>", "<|video_pad|>", "<|vision_end|>"):
            print("  chat prompt contains %-18s %s"
                  % (token, token in prompt))
        print("  ---- chat prompt head ----")
        print(prompt[:400])
        print("  ---- chat prompt tail ----")
        print(prompt[-200:])
    else:
        print("  model dir %s absent, skipping chat-template check"
              % args.model)
    for vid in ids:
        path = video_path(args.corpus, vid)
        t0 = time.time()
        try:
            frames, short, decoder = sample_video(path,
                                                  args.num_frames)
        except Exception as exc:
            print("  %-20s DECODE ERROR %s" % (vid, exc))
            continue
        print("  %-20s frames=%s dtype=%s short=%s decoder=%s  %.2fs"
              % (vid, frames.shape, frames.dtype, short, decoder,
                 time.time() - t0))
        if frames.shape[1] % ALIGN_FACTOR or frames.shape[2] % ALIGN_FACTOR:
            print("     note: %dx%d is not a multiple of %d; the video was "
                  "already under max_pixels so upstream leaves it alone and "
                  "smart_resize handles the alignment"
                  % (frames.shape[1], frames.shape[2], ALIGN_FACTOR))
        if decoder == "opencv":
            # Cross-check the fallback against the decoder upstream uses, on a
            # file both can read. If the two ever disagree, the AV1 videos are
            # not getting the frames the H.264 videos get.
            import cv2
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            idxs = np.linspace(0, total - 1, args.num_frames, dtype=int)
            alt = np.stack(_sample_frames_ffmpeg(path, idxs))
            same = alt.shape == frames.shape and np.array_equal(alt, frames)
            print("     ffmpeg fallback reproduces the OpenCV frames: %s"
                  % ("yes, exactly" if same else "NO -- investigate"))
    return 0


# --------------------------------------------------------------- selftest
def _check(name, ok, detail=""):
    print("%-62s %s%s" % (name, "OK" if ok else "FAIL",
                          ("  " + detail) if detail else ""))
    return bool(ok)


def selftest():
    ok = True
    ok &= _check("BINARY_PROMPT sha256 matches pin",
                 _sha(BINARY_PROMPT) == BINARY_PROMPT_SHA)
    ok &= _check("SYSTEM_PROMPT sha256 matches pin",
                 _sha(SYSTEM_PROMPT) == SYSTEM_PROMPT_SHA)

    abnormal = ("<think><step1>x</step1></think>\n<answer>The video is "
                "classified as <which>Abnormal</which>. In this video, "
                "<what>a man shouts</what>, occurring during the time range "
                "<when>[0.121, 0.826]</when>. ...</answer>")
    p = parse_generation(abnormal)
    ok &= _check("abnormal + interval -> positive_interval",
                 p["parse_status"] == "positive_interval"
                 and p["when"] == [0.121, 0.826])

    normal = ("<think></think><answer>The video is classified as "
              "<which>Normal</which>. In this video, <what>people talk"
              "</what>.</answer>")
    p = parse_generation(normal)
    ok &= _check("normal -> negative, no interval",
                 p["parse_status"] == "negative" and p["when"] is None)

    p = parse_generation("<which>Abnormal.</which> no time given")
    ok &= _check("trailing period on <which> still reads as abnormal",
                 p["verdict"] == "positive"
                 and p["parse_status"] == "positive_no_interval")

    p = parse_generation('<which>"Abnormal"</which>')
    ok &= _check("quoted <which> still reads as abnormal",
                 p["verdict"] == "positive")

    p = parse_generation("<which>Abnormal</which> <when>[0.2,0.4]</when>")
    ok &= _check("<when> without a space after the comma parses",
                 p["when"] == [0.2, 0.4])

    p = parse_generation("<which>Abnormal</which> <when>[0.2 , 0.4]</when>")
    ok &= _check("<when> with padded comma parses", p["when"] == [0.2, 0.4])

    p = parse_generation("<which>Abnormal</which> the anomaly is at 0.2-0.4")
    ok &= _check("prose time range is not a <when>", p["when"] is None)

    p = parse_generation("<which>Abnormal</which> <when>[0.0, 0.0]</when>")
    ok &= _check("literal [0.0, 0.0] is kept, not read as a miss",
                 p["when"] == [0.0, 0.0]
                 and p["parse_status"] == "positive_interval")

    p = parse_generation("the video looks fine to me")
    ok &= _check("no tags at all -> unparsed",
                 p["parse_status"] == "unparsed" and p["verdict"] is None)

    p = parse_generation("<when>[0.3, 0.5]</when> but no verdict tag")
    ok &= _check("interval without <which> -> unparsed_interval",
                 p["parse_status"] == "unparsed_interval")

    p = parse_generation("<which>Suspicious</which>")
    ok &= _check("unknown <which> flagged nonstandard",
                 p["which_nonstandard"] and p["verdict"] is None)

    p = parse_generation("<which>Hateful</which> <when>[0.1, 0.2]</when>",
                         arm="hateful")
    ok &= _check("hateful arm reads <which>Hateful</which> as positive",
                 p["parse_status"] == "positive_interval")
    p = parse_generation("<which>Non-hateful</which>", arm="hateful")
    ok &= _check("hateful arm reads <which>Non-hateful</which> as negative",
                 p["parse_status"] == "negative")
    p = parse_generation("<which>Abnormal</which>", arm="hateful")
    ok &= _check("hateful arm does not accept the anomaly vocabulary",
                 p["verdict"] is None)

    hateful_prompt = ARMS["hateful"][0]
    ok &= _check("hateful arm leaves no 'abnormal' behind",
                 "abnormal" not in hateful_prompt.lower())
    ok &= _check("hateful arm leaves no bare 'normal' behind",
                 not re.search(r"(?<!-)\bnormal\b", hateful_prompt,
                               re.IGNORECASE))
    ok &= _check("hateful arm is the same length structure (tags intact)",
                 hateful_prompt.count("<when>") == BINARY_PROMPT.count("<when>")
                 and hateful_prompt.count("<which>")
                 == BINARY_PROMPT.count("<which>"))

    # Upstream agreement: on well-formed output our parse must agree with
    # evaluation/1-evaluate_detection.py's two extractors.
    for text in (abnormal, normal, "<which>Abnormal</which>"):
        up_which = extract_which(text)
        up_when = extract_when(text) or [0.0, 0.0]
        mine = parse_generation(text)
        agree = (mine["which_raw"] == up_which
                 and (mine["when"] or [0.0, 0.0]) == up_when)
        ok &= _check("upstream extractors agree on %r..." % text[:24], agree)

    print("")
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(main())
