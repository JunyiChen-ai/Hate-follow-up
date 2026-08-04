"""Offline judge: run the boundary-rescue 8B judge prompt on ALL test
samples with an arbitrary MLLM. Identical prompt, definition, system
message as rescue_8b.py — the only difference is the input source
(full test split, not boundary candidates).

Media input: video_url first; if model doesn't support video or mp4 is
missing, fallback to 16 frames from frames_16/<vid>/. Never single image.

Output: results/boundary_rescue/<dataset>/offline_test_<model_tag>.jsonl
Each line: {video_id, dataset, pred, label, verdict, rationale, raw_response}

For Qwen2.5-VL models, uses llm.generate() with multi_modal_data to
avoid the vLLM 0.11.0 llm.chat() hang (docs/bugs_encountered.md Bug 1).
"""

import argparse
import glob as globmod
import json
import logging
import math
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))

from data_utils import (  # noqa: E402
    DATASET_ROOTS,
    get_media_path,
    load_annotations,
    load_clean_split_ids,
)

import os as _os
NUM_FRAMES = int(_os.environ.get("NUM_FRAMES", 16))

# Videos that trigger unrecoverable CUDA device-side assert on
# certain AWQ models via vLLM. Skip them to avoid crashing the engine.
# NH_295, NH_912, NH_566 added 2026-04-19 after 72B AWQ IH-prompt
# crashes with "CUDA: an illegal memory access" on these three.
SKIP_VIDEOS = {"NH_219", "IM_326", "hKwgFaE7fbQ", "U1RWRsmPCcg",
               "NH_295", "NH_912", "NH_566",
               "IM_59", "EX_144", "EX_412"}

_EXTRA_SKIP_FILE = _os.environ.get("EXTRA_SKIP_VIDEOS_FILE")
if _EXTRA_SKIP_FILE and os.path.isfile(_EXTRA_SKIP_FILE):
    with open(_EXTRA_SKIP_FILE) as _sf:
        SKIP_VIDEOS.update(line.strip() for line in _sf if line.strip())

_EXTRA_SKIP_VIDEOS = _os.environ.get("EXTRA_SKIP_VIDEOS")
if _EXTRA_SKIP_VIDEOS:
    SKIP_VIDEOS.update(v.strip() for v in _EXTRA_SKIP_VIDEOS.split(",") if v.strip())

ONLY_VIDEO_ID = _os.environ.get("ONLY_VIDEO_ID", "").strip()

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

# ---- Prompts & definitions: byte-for-byte from rescue_8b.py ----

HATEMM_DEF = (
    "A video is HATEFUL if it directly mocks, demeans, threatens, "
    "stereotypes, or dehumanizes an identifiable group based on race, "
    "ethnicity, religion, gender, sexuality, disability, nationality, "
    "or any similar protected characteristic. A video that merely shows "
    "or discusses hate for the purpose of reporting, critiquing, "
    "parodying, or educating is NOT hateful."
)

# ImpliHateVid-tailored definition. The dataset targets IMPLICIT hate —
# coded language, stereotyping via framing, selective blame, dog-whistles,
# often packaged as news, parliament, sports, or religious commentary. The
# generic HATEMM_DEF's "DIRECTLY mocks" + "reporting/critiquing → NOT
# hateful" carve-outs structurally under-flag this distribution.
IH_DEF = (
    "A video is HATEFUL if, as a whole, it demeans, stereotypes, "
    "ridicules, or spreads hostility toward an identifiable group "
    "based on race, ethnicity, religion, gender, sexuality, disability, "
    "nationality, caste, or a similar protected characteristic. "
    "Hate is often IMPLICIT — flag as hateful when the video "
    "(a) associates a protected group with crime, decay, disease, "
    "danger, parasitism, or moral failure; (b) uses coded language, "
    "stereotypes, or dog-whistles to mock or disparage a group; "
    "(c) attacks a group's institutions, traditions, leaders, or "
    "culture as a proxy for attacking the group itself; or "
    "(d) selectively blames a group for societal problems. "
    "The video being framed as news, political commentary, "
    "parliamentary speech, religious debate, sports analysis, "
    "historical discussion, or 'just asking questions' does NOT "
    "neutralize the hate — evaluate the underlying message the content "
    "transmits about the group. Exclude ONLY content whose purpose "
    "is clearly to critique or analyze hate itself (e.g., a documentary "
    "about racism) and straight factual reporting free of stereotyping "
    "framing."
)

MHCLIP_DEF = (
    "A video is HATEFUL in this dataset if it (a) mocks, ridicules, "
    "demeans, or stereotypes an identifiable real person or group, OR "
    "(b) contains slurs, dehumanizing language, or hostile framing of "
    "any identity, gender, race, ethnicity, religion, sexuality, "
    "disability, nationality, or shared characteristic. The mockery "
    "or hostility must be aimed at a specific identifiable target — "
    "real or representative of a real group. "
    "IMPORTANT: the following are NOT hateful in this dataset, even "
    "if they contain shock value or strong language: fictional "
    "dialogue or scripted dramatic conflict between fictional "
    "characters; action-movie or video-game violence or threats; "
    "generic vulgar, crude, or sexual humor that does not target an "
    "identity; LGBTQ+ creators simply being themselves; news, "
    "documentary, parody, or commentary that depicts hate in order "
    "to critique or report on it."
)

PROMPT_TEMPLATE = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

Examine the video carefully. Then output your decision in EXACTLY this two-field format on two lines, with no additional fields, no extra prose, and no markdown:

rationale: <one paragraph (3–6 sentences) describing what you observed in the video and why you reached your verdict>
answer: <Yes if the video is hateful per the definition above, No otherwise>"""

# IH-tailored template: pushes the judge to look beneath news/commentary
# packaging and evaluate the message transmitted about the target group,
# rather than describing surface content. Output fields (rationale +
# answer) are byte-identical to PROMPT_TEMPLATE so downstream parsing and
# band-routing code do not need a new branch.
PROMPT_TEMPLATE_IH = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

Look beneath the surface framing: news, parliament, sports, religious-debate, and commentary formats can still transmit hateful messages through implicit cues (stereotyping, selective blame, coded language, dog-whistles, associating a group with crime or decay). Ask yourself: if a listener accepted the video's framing at face value, would they come away with a more hostile, suspicious, or contemptuous view of a protected identity group?

Output your decision in EXACTLY this two-field format on two lines, with no additional fields, no extra prose, and no markdown:

rationale: <one paragraph (3–6 sentences) identifying which protected group (if any) the video is about, how the video portrays that group (neutral / supportive / stereotyping / hostile), and whether implicit cues push the overall message into hateful territory>
answer: <Yes if the video is hateful per the definition above, No otherwise>"""

PROMPT_TEMPLATE_DUAL = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

Respond with EXACTLY three lines, in this fixed order, no additional fields, no markdown:

initial_answer: <Yes or No — your first impression, based only on observing the video, title, and transcript, before writing any detailed reasoning>
rationale: <one paragraph (3–6 sentences) explaining what you observed and your reasoning>
final_answer: <Yes or No — your final decision after the rationale>"""

# Multi-Aspect Definition Probe (MADP): forces the judge to commit on three
# definition-derived sub-questions in one call, then a final verdict. The
# decision rule is a fixed Boolean combination of the sub-answers, applied
# the same way to every judge.
PROMPT_TEMPLATE_EAA = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

Per the definition above, evaluate the video by answering FOUR questions in EXACTLY this format, on four separate lines, English only, no extra prose, no markdown:

target_real_group: <Yes if the video targets a specific identifiable real-world person or group with a shared identity (race, religion, sexuality, gender, nationality, disability, ethnicity, etc.); No if there is no such target or the target is fictional, a brand, a product, or generic individuals>
hostile_framing: <Yes if the framing toward the identified target is hostile, mocking, dehumanizing, demeaning, or uses slurs; No if framing is neutral, supportive, or merely informative>
protected_context: <Yes if the content is news/documentary/parody/satire/educational/critical commentary that depicts the topic in order to report or critique rather than promote hate; No otherwise>
answer: <Yes if the video is hateful per the definition above, No if normal>"""

PROMPT_TEMPLATE_RTG = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

Output your decision in EXACTLY this three-line format, no extra prose, no markdown:

rationale: <one paragraph (2-4 sentences) describing what you observed>
verdict: <hateful if the video is hateful per the definition above, normal otherwise>
real_target_identified: <Yes if the video actually attacks a real-world protected identity group (race, religion, sexuality, gender, nationality, disability, ethnicity); No if the perceived target is fictional, a brand/product, a celebrity/individual not attacked as group representative, or no clear group target>"""

PROMPT_TEMPLATE_DV = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

You must give TWO verdicts, same judgment criterion but different sensitivity, in EXACTLY this three-line format:

rationale: <one paragraph (2-4 sentences)>
verdict_lenient: <hateful or normal — apply definition leniently, flag as hateful even when hostile framing is indirect or implied>
verdict_strict: <hateful or normal — apply definition strictly, flag as hateful ONLY when a specific real-world identity group is unambiguously attacked>"""

PROMPT_TEMPLATE_CONS = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

STRICT APPLICATION RULES before reaching a verdict:
- Fictional characters, cartoons, brands, products, and game characters are NOT protected identity groups. Slurs or insults directed at them are NOT hateful.
- Medical, educational, documentary, and critical-commentary contexts that discuss sensitive topics for informational purposes are NOT hateful.
- Crude, sexual, or vulgar humor that does not target a specific real-world protected identity group is NOT hateful.
- Only label as hateful when there is clear attack on a real-world identity group (race, religion, sexuality, gender, nationality, disability, ethnicity).

Output EXACTLY this two-line format:

rationale: <one paragraph (2-4 sentences) describing what you observed and your reasoning>
verdict: <hateful or normal>"""

SYSTEM_MSG = "You are a content moderation analyst. Answer based strictly on observable evidence."


def get_definition(dataset, ih_prompt=False):
    if dataset == "ImpliHateVid" and ih_prompt:
        return IH_DEF
    if dataset in {"HateMM", "ImpliHateVid"}:
        return HATEMM_DEF
    return MHCLIP_DEF


def model_tag(model_id):
    """Derive a short tag from model ID for output filenames."""
    name = model_id.lower().split("/")[-1]
    name = name.replace("-instruct", "").replace("-instruct-", "")
    # e.g. qwen2.5-vl-32b-awq, qwen3-vl-8b
    name = re.sub(r"[^a-z0-9.\-]", "", name)
    return name


def _resolve_frames(media_path, media_type, dataset, vid):
    """Return list of frame jpg paths (up to NUM_FRAMES).
    For video (mp4): look for pre-extracted frames_16/<vid>/.
    For frames dir: use directly.
    Returns empty list on failure.
    """
    if media_type == "video":
        vid_name = os.path.basename(media_path).rsplit(".", 1)[0]
        parent = os.path.dirname(media_path)
        root = os.path.dirname(parent) if os.path.basename(parent) in ("video", "video_mp4") else parent
        frame_dir = os.path.join(root, "frames_16", vid_name)
        if not os.path.isdir(frame_dir):
            # Try dataset root
            frame_dir = os.path.join(DATASET_ROOTS[dataset], "frames_16", vid_name)
        if os.path.isdir(frame_dir):
            jpgs = sorted(globmod.glob(os.path.join(frame_dir, "*.jpg")) +
                          globmod.glob(os.path.join(frame_dir, "*.jpeg")) +
                          globmod.glob(os.path.join(frame_dir, "*.png")))
            if len(jpgs) > NUM_FRAMES:
                indices = np.linspace(0, len(jpgs) - 1, NUM_FRAMES, dtype=int)
                jpgs = [jpgs[i] for i in indices]
            return jpgs
        return []  # no frames_16 dir
    else:
        # frames directory
        jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")) +
                      globmod.glob(os.path.join(media_path, "*.jpeg")) +
                      globmod.glob(os.path.join(media_path, "*.png")))
        if len(jpgs) > NUM_FRAMES:
            indices = np.linspace(0, len(jpgs) - 1, NUM_FRAMES, dtype=int)
            jpgs = [jpgs[i] for i in indices]
        return jpgs


def load_split_ids(dataset, split):
    return load_clean_split_ids(dataset, split)


def load_done_ids(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r.get("video_id"):
                    done.add(r["video_id"])
            except (json.JSONDecodeError, Exception):
                pass
    return done


def parse_verdict(text):
    """Extract the decisive answer from response. Returns (verdict_str, pred_int).
    Yes/No format: 'answer:' (New) / 'final_answer:' (Dual). Legacy: 'verdict:'.
    EAA mode also accepts 'answer: Unsure/Unclear/Uncertain' → ("abstain", -2).
    pred == -2 means 'abstain' (distinct from -1 = parse failure)."""
    if not text:
        return "", -1
    # Prefer final_answer if present (dual-verdict mode)
    t_low = text.lower()
    if "final_answer:" in t_low:
        idx = t_low.rfind("final_answer:")
        tail = text[idx + len("final_answer:"):].strip().lower().rstrip(".").split("\n", 1)[0].strip()
        if tail.startswith("yes"):
            return "hateful", 1
        if tail.startswith("no"):
            return "normal", 0
        if tail.startswith(("unsure", "unclear", "uncertain")):
            return "abstain", -2
    for line in text.strip().split("\n"):
        line_lower = line.strip().lower()
        if line_lower.startswith("answer:"):
            v = line_lower.split("answer:", 1)[1].strip().rstrip(".")
            if v.startswith("yes"):
                return "hateful", 1
            if v.startswith("no"):
                return "normal", 0
            if v.startswith(("unsure", "unclear", "uncertain")):
                return "abstain", -2
        elif line_lower.startswith("verdict:"):
            v = line_lower.split("verdict:", 1)[1].strip().rstrip(".")
            if "hateful" in v:
                return "hateful", 1
            if "normal" in v:
                return "normal", 0
            if "unsure" in v or "unclear" in v or "uncertain" in v:
                return "abstain", -2
    # Fallback: search anywhere
    t = text.lower()
    if "answer: yes" in t or "answer:yes" in t:
        return "hateful", 1
    if "answer: no" in t or "answer:no" in t:
        return "normal", 0
    if "answer: unsure" in t or "answer:unsure" in t:
        return "abstain", -2
    if "verdict: hateful" in t or "verdict:hateful" in t:
        return "hateful", 1
    if "verdict: normal" in t or "verdict:normal" in t:
        return "normal", 0
    return "", -1


def is_qwen25(model_id):
    """Check if model is Qwen2.5-VL (needs generate() path)."""
    return "qwen2.5" in model_id.lower() or "qwen2-vl" in model_id.lower()


def _extract_rationale(text):
    """Strip the rationale paragraph from the full response, handling
    both 'answer:' (new) and 'verdict:' (legacy) delimiters."""
    low = (text or "").lower()
    for marker in ("\nanswer:", "answer:", "\nverdict:", "verdict:"):
        idx = low.find(marker)
        if idx >= 0:
            body = text[:idx]
            return body.replace("rationale:", "", 1).strip()
    return ""


def get_yes_no_token_ids(tokenizer):
    """Return (yes_id, no_id) — the single-token ids for ' Yes' and ' No'.
    Raises if either is not a single token."""
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            f"Expected single-token ' Yes'/' No', got {yes_ids}/{no_ids}"
        )
    return yes_ids[0], no_ids[0]


def _lp_at(lp_dict, tid):
    """exp of logprob for tid in lp_dict, 0 if missing."""
    if not lp_dict or tid not in lp_dict:
        return 0.0
    o = lp_dict[tid]
    v = float(o.logprob if hasattr(o, "logprob") else o)
    return math.exp(v)


def extract_verdict_logprobs(output, yes_id, no_id):
    """Single-verdict mode: find last token id in {yes_id, no_id}, read
    its top-k logprob dict, return P(Yes), P(No). Returns dict or None.
    """
    if not output.outputs:
        return None
    out0 = output.outputs[0]
    if not out0.logprobs or not out0.token_ids:
        return None
    tokens = out0.token_ids
    lps = out0.logprobs
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] != yes_id and tokens[i] != no_id:
            continue
        lp = lps[i]
        if lp is None:
            continue
        return {
            "pos": i,
            "chosen_class": "hateful" if tokens[i] == yes_id else "normal",
            "p_hateful": _lp_at(lp, yes_id),
            "p_normal": _lp_at(lp, no_id),
        }
    return None


def extract_full_logprobs(output, tokenizer):
    """Dump (token_id, token_text, logprob_of_chosen) per generated token.
    Used for TokenSAR: need per-token NLL over the full generation span.
    """
    if not output.outputs:
        return None
    out0 = output.outputs[0]
    if not out0.logprobs or not out0.token_ids:
        return None
    tokens = out0.token_ids
    lps = out0.logprobs
    seq = []
    for i, tid in enumerate(tokens):
        lp = lps[i] if i < len(lps) else None
        chosen_lp = None
        if lp is not None and tid in lp:
            o = lp[tid]
            chosen_lp = float(o.logprob if hasattr(o, "logprob") else o)
        try:
            tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            tok_str = ""
        seq.append({"tid": int(tid), "tok": tok_str, "lp": chosen_lp})
    return seq


def extract_dual_verdict_logprobs(output, yes_id, no_id):
    """Dual-verdict mode: find the FIRST yes/no token (initial) and the
    LAST yes/no token (final). If they coincide, return None.
    Returns a dict with 'initial' and 'final' sub-dicts.
    """
    if not output.outputs:
        return None
    out0 = output.outputs[0]
    if not out0.logprobs or not out0.token_ids:
        return None
    tokens = out0.token_ids
    lps = out0.logprobs
    first_pos = None
    for i in range(len(tokens)):
        if tokens[i] == yes_id or tokens[i] == no_id:
            first_pos = i
            break
    last_pos = None
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == yes_id or tokens[i] == no_id:
            last_pos = i
            break
    if first_pos is None or last_pos is None or first_pos == last_pos:
        return None
    def _pack(idx):
        lp = lps[idx] if lps[idx] is not None else {}
        return {
            "pos": idx,
            "chosen_class": "hateful" if tokens[idx] == yes_id else "normal",
            "p_hateful": _lp_at(lp, yes_id),
            "p_normal": _lp_at(lp, no_id),
        }
    return {"initial": _pack(first_pos), "final": _pack(last_pos)}


def main():
    parser = argparse.ArgumentParser(
        description="Offline judge: run rescue prompt on all test samples"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"])
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-mem", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--no-video", action="store_true",
                        help="Force 16-frame image input even when mp4 exists")
    parser.add_argument("--band-only", action="store_true",
                        help="Restrict work to band candidate video_ids "
                             "(results/boundary_rescue/<ds>/candidates_bayes_band_rate.jsonl)")
    parser.add_argument("--logprobs", type=int, default=0,
                        help="If > 0, request top-k token logprobs during decode "
                             "and record verdict-token logprob dist in output")
    parser.add_argument("--save-full-logprobs", action="store_true",
                        help="Save full per-token (tid, tok, lp) list for TokenSAR "
                             "(requires --logprobs > 0)")
    parser.add_argument("--dual-verdict", action="store_true",
                        help="Use PROMPT_TEMPLATE_DUAL (initial_answer + "
                             "rationale + final_answer); record logprobs at "
                             "both verdict positions")
    parser.add_argument("--eaa-mode", action="store_true",
                        help="Use Evidence-Anchored Abstention template "
                             "(rationale + evidence + Yes/No/Unsure answer). "
                             "Records with Unsure are written with pred=-2.")
    parser.add_argument("--rtg-mode", action="store_true",
                        help="Use Real-Target-Gate template "
                             "(rationale + real_target_identified Yes/No + verdict hateful/normal)")
    parser.add_argument("--cons-mode", action="store_true",
                        help="Use Conservative template with explicit exclusion reminders")
    parser.add_argument("--dv2-mode", action="store_true",
                        help="Use Dual-Verdict-2 template (lenient + strict verdicts in one call)")
    parser.add_argument("--ih-prompt", action="store_true",
                        help="For ImpliHateVid only: swap in IH_DEF + "
                             "PROMPT_TEMPLATE_IH that target implicit/coded "
                             "hate packaged as news/commentary. No-op for "
                             "other datasets. Adds 'ih' tag to output file.")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Provide --dataset or --all")

    datasets = ALL_DATASETS if args.all else [args.dataset]
    tag = model_tag(args.model)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(
        f"judge_offline: model={args.model} tag={tag} datasets={datasets} "
        f"batch={args.batch_size} gpu_mem={args.gpu_mem}"
    )

    # Build work plan
    work = []  # (dataset, vid, ann)
    out_paths = {}
    annotations_cache = {}

    for ds in datasets:
        annotations_cache[ds] = load_annotations(ds)
        split_ids = load_split_ids(ds, args.split)
        out_dir = os.path.join(OUT_ROOT, ds)
        os.makedirs(out_dir, exist_ok=True)
        if args.band_only:
            band_path = os.path.join(out_dir, "candidates_bayes_band_rate.jsonl")
            band_vids = set()
            if os.path.isfile(band_path):
                with open(band_path) as bf:
                    for line in bf:
                        try:
                            band_vids.add(json.loads(line.strip())["video_id"])
                        except (json.JSONDecodeError, KeyError):
                            pass
            before = len(split_ids)
            split_ids = [v for v in split_ids if v in band_vids]
            logging.info(f"  {ds}: --band-only filter: {before} -> {len(split_ids)}")
        fname_parts = [f"offline_{args.split}"]
        if args.band_only:
            fname_parts.append("band")
        if args.logprobs > 0:
            fname_parts.append("lp")
        if args.save_full_logprobs:
            fname_parts.append("tsar")
        if args.dual_verdict:
            fname_parts.append("dv")
        if args.eaa_mode:
            fname_parts.append("eaa")
        if args.rtg_mode:
            fname_parts.append("rtg")
        if args.cons_mode:
            fname_parts.append("cons")
        if args.dv2_mode:
            fname_parts.append("dv2")
        if args.ih_prompt and ds == "ImpliHateVid":
            fname_parts.append("ih")
        fname_parts.append(tag)
        out_path = os.path.join(out_dir, "_".join(fname_parts) + ".jsonl")
        out_paths[ds] = out_path
        done_ids = load_done_ids(out_path)
        new_ids = [v for v in split_ids if v not in done_ids and v not in SKIP_VIDEOS]
        if ONLY_VIDEO_ID:
            new_ids = [v for v in new_ids if v == ONLY_VIDEO_ID]
        skipped_vids = [v for v in split_ids if v in SKIP_VIDEOS and v not in done_ids]
        if skipped_vids:
            logging.warning(f"  {ds}: skipping {len(skipped_vids)} known-crash videos: {skipped_vids}")
        logging.info(f"  {ds}: {len(split_ids)} {args.split}, {len(done_ids)} done, {len(new_ids)} to process")
        for vid in new_ids:
            ann = annotations_cache[ds].get(vid)
            if ann is not None:
                work.append((ds, vid, ann))

    if not work:
        logging.info("All test samples already processed. Nothing to do.")
        return

    # Init vLLM
    from vllm import LLM, SamplingParams

    use_generate = is_qwen25(args.model)
    use_video = not args.no_video and not use_generate
    logging.info(
        f"Inference path: {'llm.generate()' if use_generate else 'llm.chat()'} "
        f"video_input={use_video} frames={NUM_FRAMES}"
    )

    # mm_processor_kwargs: max_pixels is Qwen-specific
    is_qwen = "qwen" in args.model.lower()
    mm_kwargs = {"max_pixels": 100352} if is_qwen else {}

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1, "image": NUM_FRAMES},
        enforce_eager=True,
    )
    if mm_kwargs:
        llm_kwargs["mm_processor_kwargs"] = mm_kwargs
    if not use_generate:
        llm_kwargs["allowed_local_media_path"] = "/data/jehc223"

    llm = LLM(**llm_kwargs)

    processor = None
    if use_generate:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    tokenizer = None
    yes_id = no_id = None
    if args.logprobs > 0:
        if processor is not None and hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            try:
                tokenizer = llm.get_tokenizer()
            except Exception:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model, trust_remote_code=True
                )
        yes_id, no_id = get_yes_no_token_ids(tokenizer)
        logging.info(f"verdict token ids: yes={yes_id} no={no_id}")

    sp_kwargs = dict(temperature=0, max_tokens=args.max_tokens)
    if args.logprobs > 0:
        sp_kwargs["logprobs"] = args.logprobs
    sampling_params = SamplingParams(**sp_kwargs)

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    consecutive_fails = 0
    for batch_start in range(0, len(work), args.batch_size):
        batch = work[batch_start: batch_start + args.batch_size]
        batch_inputs = []
        batch_meta = []

        for ds, vid, ann in batch:
            media = get_media_path(vid, ds)
            if media is None:
                n_skipped += 1
                continue
            media_path, media_type = media
            title = ann.get("title", "") or ""
            transcript = (ann.get("transcript", "") or "")[:args.transcript_limit]
            if args.eaa_mode:
                tpl = PROMPT_TEMPLATE_EAA
            elif args.rtg_mode:
                tpl = PROMPT_TEMPLATE_RTG
            elif args.cons_mode:
                tpl = PROMPT_TEMPLATE_CONS
            elif args.dv2_mode:
                tpl = PROMPT_TEMPLATE_DV
            elif args.dual_verdict:
                tpl = PROMPT_TEMPLATE_DUAL
            elif args.ih_prompt and ds == "ImpliHateVid":
                tpl = PROMPT_TEMPLATE_IH
            else:
                tpl = PROMPT_TEMPLATE
            prompt_text = tpl.format(
                title=title,
                transcript=transcript,
                definition=get_definition(ds, ih_prompt=args.ih_prompt),
            )

            # --- Resolve frames: video first, fallback to frames_16 (16 frames) ---
            frame_jpgs = _resolve_frames(media_path, media_type, ds, vid)
            if not frame_jpgs:
                logging.warning(f"  {ds}/{vid}: no video or frames, skipping")
                n_skipped += 1
                continue

            if use_generate:
                from PIL import Image, ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                pil_frames = [Image.open(p).convert("RGB") for p in frame_jpgs]
                n_img = len(pil_frames)
                user_content = [{"type": "image"} for _ in range(n_img)]
                user_content.append({"type": "text", "text": prompt_text})
                tpl_msgs = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": user_content},
                ]
                prompt_str = processor.apply_chat_template(
                    tpl_msgs, add_generation_prompt=True, tokenize=False
                )
                batch_inputs.append({
                    "prompt": prompt_str,
                    "multi_modal_data": {"image": pil_frames},
                })
            else:
                if use_video and media_type == "video":
                    media_content = [{"type": "video_url",
                                      "video_url": {"url": f"file://{media_path}"}}]
                else:
                    media_content = [
                        {"type": "image_url",
                         "image_url": {"url": f"file://{p}"}}
                        for p in frame_jpgs
                    ]
                content = media_content + [{"type": "text", "text": prompt_text}]
                messages = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": content},
                ]
                batch_inputs.append(messages)

            batch_meta.append((ds, vid, ann))

        if not batch_inputs:
            continue

        try:
            if use_generate:
                outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            else:
                outputs = llm.chat(messages=batch_inputs, sampling_params=sampling_params)
        except Exception as e:
            logging.error(f"  Batch failed: {str(e)[:200]}, falling back to single")
            outputs = []
            for i, inp in enumerate(batch_inputs):
                try:
                    if use_generate:
                        out_one = llm.generate([inp], sampling_params=sampling_params)
                    else:
                        out_one = llm.chat(messages=[inp], sampling_params=sampling_params)
                    outputs.append(out_one[0])
                except Exception as e2:
                    logging.error(f"  {batch_meta[i][1]}: single failed: {str(e2)[:200]}")
                    outputs.append(None)

        for (ds, vid, ann), out in zip(batch_meta, outputs):
            if out is None:
                response_text = ""
            else:
                response_text = out.outputs[0].text or ""

            verdict_str, pred = parse_verdict(response_text)
            raw_label = ann.get("label", "")
            # Map string labels to binary: Hateful/Offensive→1, Normal→0
            if isinstance(raw_label, str):
                label_int = 1 if raw_label.lower() in ("hateful", "offensive") else 0
            else:
                label_int = int(raw_label)

            # Don't write parse failures — skip them so resume can retry.
            # pred == -2 means EAA 'abstain' — valid, write it.
            if pred == -1:
                consecutive_fails += 1
                logging.warning(f"  {ds}/{vid}: empty response (engine may be broken), not writing")
                if consecutive_fails >= 3:
                    logging.error(
                        f"  {consecutive_fails} consecutive failures — EngineCore likely crashed. "
                        f"Exiting early. Re-run to resume from here with a fresh engine."
                    )
                    logging.info(f"Done (early exit). processed={n_processed}, skipped={n_skipped}")
                    return
                continue

            consecutive_fails = 0
            rec = {
                "video_id": vid,
                "dataset": ds,
                "pred": pred,
                "label": label_int,
                "verdict": verdict_str,
                "rationale": _extract_rationale(response_text),
                "raw_response": response_text,
                "num_frames_requested": NUM_FRAMES,
                "num_frames_resolved": len(frame_jpgs),
                "media_input": "video" if use_video and media_type == "video" else "frames",
            }
            if args.logprobs > 0 and out is not None and yes_id is not None:
                try:
                    if args.dual_verdict:
                        lp_info = extract_dual_verdict_logprobs(out, yes_id, no_id)
                        if lp_info is not None:
                            rec["dual_verdict_logprobs"] = lp_info
                    else:
                        lp_info = extract_verdict_logprobs(out, yes_id, no_id)
                        if lp_info is not None:
                            rec["verdict_logprobs"] = lp_info
                except Exception as e:
                    logging.warning(f"  {ds}/{vid}: verdict_logprobs extract failed: {e}")
            if args.save_full_logprobs and args.logprobs > 0 and out is not None:
                try:
                    full = extract_full_logprobs(out, tokenizer)
                    if full is not None:
                        rec["full_logprobs"] = full
                except Exception as e:
                    logging.warning(f"  {ds}/{vid}: full_logprobs extract failed: {e}")
            with open(out_paths[ds], "a") as fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
            n_processed += 1

        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        if n_processed % 10 == 0 or n_processed <= 5:
            logging.info(f"  [{n_processed}/{len(work)}] {rate:.2f} vid/s")

    logging.info(f"Done. processed={n_processed}, skipped={n_skipped}")


if __name__ == "__main__":
    main()
