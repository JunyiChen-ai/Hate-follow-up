"""Shared constants and helpers of candidate 45 (README §2). Imported by measure.py (GPU) and analyze.py (CPU)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_manifest  # noqa: E402

EXP = ROOT / "runs/20261017_m1_generation_side"
R6 = ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"
R6_BMA = ROOT / "runs/20260926_twolevel/r6_bma/metrics.json"
WINDOW_S = 8.0
NATIVE_FRAMES = 20
MAX_NEW_CAP = 256
REFUSAL_STARTERS = ("I", "I'm", "I’m", "Sorry", "As", "Unfortunately", "Apolog")
ARMS = {"A": "copy_mean", "B": "edit_rate"}          # arm -> primary score (README §2)
A_SCORES = ("copy_mean", "copy_max", "first_loss", "refuse_mass")
B_SCORES = ("edit_rate", "edit_lev")
IMPLEMENTATION = "candidate 45 generation-side reading v3: whitespace-normalised body, teacher-forced repetition, minimal compliant rewrite with LCS edit_rate; 2026-10-17"


def head(i, n, t1, t2, body):
    return f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video. The speech in this window is: {body}"


def question_repeat(i, n, t1, t2, body):
    return (head(i, n, t1, t2, body) + "\n\nRepeat this speech verbatim, word for word, speaking in the first person as the speaker. "
            "Output only the repeated speech: no preamble, no quotation marks, no commentary.")


def question_rewrite(i, n, t1, t2, body):
    return (head(i, n, t1, t2, body) + "\n\nRewrite this speech so that it no longer violates any of the above rules, changing as few "
            "words as possible and keeping every other word exactly as it is. If nothing in it violates the rules, output it unchanged. "
            "Output only the rewritten speech: no preamble, no quotation marks, no commentary.")


def selected_rows(datasets, smoke):
    """All test videos of `datasets`; smoke = the fixed five-video subset used since experiments/20261007_m1_streamingtom."""
    rows = load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", datasets)
    if smoke:
        rows = [r for ds in datasets for r in [x for x in rows if x["dataset"] == ds][:2]] + [
            r for r in rows if r["dataset"] == "HateMM" and r["video_id"] == "hate_video_114"]
    return rows


def stored_r6():
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(R6)) if not r.get("error")}


_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def norm_words(text):
    """Lower-cased alphanumeric words (apostrophe-joined contractions kept), punctuation dropped."""
    return _WORD.findall((text or "").lower().replace("’", "'"))


def levenshtein(a, b):
    """Word-level edit distance between two word lists."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm_body(text):
    """The window text as used in both the question and the forced answer: runs of whitespace collapsed to one space,
    ends stripped (README §2; the smoke run showed double spaces alone cost several nats under teacher forcing)."""
    return " ".join((text or "").split())


def lcs_len(a, b):
    """Length of the longest common subsequence of two word lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def edit_scores(body, output):
    """README §2 arm B. edit_rate (primary) = share of the body's words that are removed or replaced in the output,
    1 - LCS / len(body words): insertions (appended commentary, continued captions) do not count. edit_lev (secondary)
    = word-level Levenshtein / len(body words), clipped to [0, 1]. Identical word lists -> 0; empty output for a
    non-empty body -> 1; a body without words -> 0."""
    wb, wo = norm_words(body), norm_words(output)
    if wb == wo or not wb:
        return dict(edit_rate=0.0, edit_lev=0.0)
    if not wo:
        return dict(edit_rate=1.0, edit_lev=1.0)
    return dict(edit_rate=1.0 - lcs_len(wb, wo) / len(wb), edit_lev=min(1.0, levenshtein(wb, wo) / len(wb)))


def edit_rate(body, output):
    return edit_scores(body, output)["edit_rate"]


def frame_span(t1, t2, length):
    """4 fps frame indices of window [t1, t2) (candidate 44 convention), clipped to the GT length."""
    a, b = int(round(4 * t1)), int(round(4 * t2))
    return max(0, min(a, length)), max(0, min(b, length))
