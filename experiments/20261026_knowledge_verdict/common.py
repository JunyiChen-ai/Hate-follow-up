"""Shared helpers of candidate 51: subset choice, manifests, pack loading and the prompt inserts."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = "20261026_knowledge_verdict"
MANIFESTS = {"HateMM": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "HateClipSeg": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "DeHate": "data/manifests/DeHate_test.jsonl"}
SUBSET_N = 40
AGENT_DIR = ROOT / "runs" / EXP / "agent"

PREFIX_HEAD = ("Background notes (retrieved automatically; they may be irrelevant, and a term may also have an "
               "ordinary meaning):")
JUDGMENT_HEAD = "An external analyst's assessment of this video (it may be wrong):"
WINDOW_HEAD = "Notes for this moment:"
MAX_WINDOW_NOTES = 2
WITHHELD = "(note withheld)"


def manifest_rows(ds):
    return [r for r in (json.loads(l) for l in open(ROOT / MANIFESTS[ds]) if l.strip()) if r["dataset"] == ds]


def subset_rows(ds, n=SUBSET_N):
    """n test videos of corpus ds: default_rng(0).choice over the sorted test ids (no label is read)."""
    rows = sorted(manifest_rows(ds), key=lambda r: r["video_id"])
    pick = np.random.default_rng(0).choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(pick)]


def load_pack(ds, vid):
    p = AGENT_DIR / ds / f"{vid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def note_line(n):
    return f"[{float(n['t_start']):.1f}s-{float(n['t_end']):.1f}s] {n['clue']} ({n['source_title']}): {n['note']}"


def mark_line(n):
    """Kmark: the note's time span with its content withheld."""
    return f"[{float(n['t_start']):.1f}s-{float(n['t_end']):.1f}s] {WITHHELD}"


def shifted_notes(notes, dur):
    """Kshift: every note moved by half the video duration, circularly with its length kept: the part past the end
    continues at 0 as a second piece of the same note. A note covering the whole video stays as it is. Pieces carry the
    note's index (_idx) so that window_notes counts a note once per window."""
    out = []
    for i, n in enumerate(notes or []):
        a, b = float(n["t_start"]), float(n["t_end"])
        if dur <= 0 or b - a >= dur:
            out.append({**n, "_idx": i})
            continue
        a2 = (a + dur / 2) % dur
        b2 = a2 + (b - a)
        out.append({**n, "t_start": a2, "t_end": min(b2, dur), "_idx": i})
        if b2 > dur:
            out.append({**n, "t_start": 0.0, "t_end": b2 - dur, "_idx": i})
    return out


def rescaled_pack(pack, dur):
    """Kother: another video's pack with its note times scaled to this video's duration."""
    if not pack:
        return pack
    src = float(pack.get("duration") or 0)
    f = dur / src if src > 0 else 1.0
    notes = [{**n, "t_start": min(float(n["t_start"]) * f, dur), "t_end": min(float(n["t_end"]) * f, dur)}
             for n in pack.get("notes") or []]
    return {**pack, "notes": notes}


def prefix_insert(pack, with_notes, judgment_key=None):
    """Text appended to the end of the r6 prefix (after the policy rules, before the verdict question).
    judgment_key: None, "judgment" (written with the notes) or "judgment0" (written without them)."""
    parts = []
    notes = (pack or {}).get("notes") or []
    if with_notes and notes:
        parts.append(PREFIX_HEAD + "\n" + "\n".join(note_line(n) for n in notes))
    j = (pack or {}).get(judgment_key) if judgment_key else None
    if j:
        times = ", ".join(f"{float(a):.1f}s-{float(b):.1f}s" for a, b in (j.get("times") or []))
        parts.append(JUDGMENT_HEAD + "\n" +
                     f"Violates the rules: {j.get('violates')} (confidence {float(j.get('confidence', 0)):.2f}). "
                     f"Target: {j.get('target') or 'none'}. Intent: {j.get('intent') or 'none'}. "
                     f"Means: {j.get('means') or 'none'}. Where: {times or 'none'}. "
                     f"Explanation: {j.get('explanation') or ''}")
    return ("\n" + "\n\n".join(parts) + "\n") if parts else ""


def window_notes(notes, a, b, last=False):
    """Notes whose [t_start, t_end] overlaps the window [a, b) ([a, b] for the last window), at most MAX_WINDOW_NOTES,
    in pack order."""
    out, seen = [], set()
    for k, n in enumerate(notes or []):
        if (float(n["t_start"]) < b or (last and float(n["t_start"]) <= b)) and float(n["t_end"]) >= a:
            idx = n.get("_idx", k)
            if idx not in seen:
                seen.add(idx)
                out.append(n)
    return out[:MAX_WINDOW_NOTES]


def window_insert(notes, a, b, last=False, mark=False):
    sel = window_notes(notes, a, b, last)
    line = mark_line if mark else note_line
    return (WINDOW_HEAD + "\n" + "\n".join(line(n) for n in sel) + "\n\n") if sel else ""
