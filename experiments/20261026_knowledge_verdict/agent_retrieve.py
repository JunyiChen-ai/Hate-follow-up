#!/usr/bin/env python3
"""Candidate 51, API side (runs on uoa-lab2, CPU): per video, clues -> Wikipedia -> self-reflection notes -> judgment,
plus a second judgment written without the notes (judgment0, arm J0 and the strong-model control of README §3).

Calls qwen3-vl-plus on the DashScope OpenAI-compatible endpoint (temperature 0, seed 0, no thinking). Inputs are the
same 20 frames (data/frames_k20) and the same timestamped transcript the 8B reader sees. Labels are never read.
Output: runs/20261026_knowledge_verdict/agent/<dataset>/<video_id>.json (one file per video, skipped if present),
run.log and usage.jsonl next to it. The API key is read from ~/.dashscope_api_key at call time and never written.
"""
import argparse
import base64
import io
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AGENT_DIR, ROOT, subset_rows  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.video_inputs import frame_paths, load_asr, transcript_block  # noqa: E402

CODE_PATH = "experiments/20261026_knowledge_verdict/agent_retrieve.py"
MODEL = "qwen3-vl-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PRICE_IN, PRICE_OUT = 0.002, 0.008          # yuan per 1k tokens (Retrieval-hate record; may be outdated)
MAX_QUERIES, HITS_PER_QUERY, SUMMARY_SENTENCES, MAX_NOTES = 12, 3, 5, 8
UA = {"User-Agent": "HateFollowUp-research/0.1 (academic research; contact via repository owner)"}
# policy text: identical to the rules the 8B reader sees (src/mllm_judge.py YOUTUBE_RULES; copied to avoid importing torch)
RULES = """1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events"""

CLUE_PROMPT = """You are given {n} frames sampled from an online video, in order, each with its timestamp, and the video's timestamped transcript.
Identify the social knowledge a viewer would need to understand what the video refers to:
1. social_entities: named people, organisations, places, dates, events or media titles referred to in the frames or the speech.
2. social_groups: social groups shown or referred to (for example by religion, ethnicity, nationality, gender, sexual orientation, disability, immigration status, or political movement).
3. symbols: flags, logos, gestures, emblems, numbers, on-screen text, meme formats or images that may carry a meaning beyond their literal appearance.
4. coded_terms: words or phrases in the speech or on-screen text that may be slang, euphemisms, dog whistles or references whose meaning is not literal.
For every item give its text, the time span where it appears (seconds; a single moment has t_start = t_end), and at most two short English Wikipedia search queries that would retrieve background knowledge about it. Give at most 10 items, most informative first. Also give the social domain of the video in a few words.
Return JSON only, in this form:
{{"items": [{{"kind": "social_entities|social_groups|symbols|coded_terms", "text": "...", "t_start": 0.0, "t_end": 0.0, "queries": ["..."]}}], "social_domain": "..."}}

Transcript:
{transcript}"""

REFLECT_PROMPT = """You are a cultural knowledge verifier for English-language online culture.
Below are the timestamped transcript of a video, the items a first reader identified in it, and numbered Wikipedia summaries retrieved for those items.
For each summary decide:
(1) cultural consciousness: is this information known in mainstream English-language online discourse?
(2) relevance: does it directly explain an item that appears in this video, rather than being only topically adjacent?
Keep only summaries that pass both. For each kept summary write one note of at most 60 words stating the neutral encyclopaedic fact that explains the item. Do not judge or describe the video itself in the note.
Return at most {max_notes} notes, most relevant first, with the time span of the item they explain. Return JSON only, in this form:
{{"notes": [{{"summary_id": 1, "clue": "...", "t_start": 0.0, "t_end": 0.0, "source_title": "...", "note": "..."}}]}}
If nothing passes, return {{"notes": []}}.

Transcript:
{transcript}

Items:
{items}

Summaries:
{summaries}"""

JUDGE_PROMPT = """You are given {n} frames sampled from an online video, in order, each with its timestamp, the video's timestamped transcript, and background notes retrieved for things that appear in it (the notes may be irrelevant).
Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

Decide whether the video contains content that violates any of these rules, either explicitly or through its conveyed meaning. If it does, name the targeted group, the intent, the means, and the time spans where it occurs.
Return JSON only, in this form:
{{"violates": "yes|no", "confidence": 0.0, "target": "...", "intent": "...", "means": "...", "times": [[0.0, 0.0]], "explanation": "at most 80 words"}}

Transcript:
{transcript}

Background notes:
{notes}"""

JUDGE0_PROMPT = """You are given {n} frames sampled from an online video, in order, each with its timestamp, and the video's timestamped transcript.
Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

Decide whether the video contains content that violates any of these rules, either explicitly or through its conveyed meaning. If it does, name the targeted group, the intent, the means, and the time spans where it occurs.
Return JSON only, in this form:
{{"violates": "yes|no", "confidence": 0.0, "target": "...", "intent": "...", "means": "...", "times": [[0.0, 0.0]], "explanation": "at most 80 words"}}

Transcript:
{transcript}"""

_lock = threading.Lock()
_spent = {"in": 0, "out": 0}


def image_part(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((512, 512))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def media_content(frames, text):
    parts = []
    for t, f in frames:
        parts.append({"type": "text", "text": f"[frame at t={t:.1f}s]"})
        parts.append(image_part(f))
    parts.append({"type": "text", "text": text})
    return parts


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def call(client, content, max_tokens, usage_log, tag, vid, meta):
    """One chat call; returns (parsed JSON or None, raw text, error). Retries once on a parse failure. The model name
    the endpoint reports is added to meta["model_returned"]."""
    msgs = [{"role": "user", "content": content}]
    raw, err = "", None
    for attempt in range(2):
        try:
            r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0, seed=0, max_tokens=max_tokens,
                                               extra_body={"enable_thinking": False})
        except Exception as exc:  # DashScope input moderation refusals arrive as 400 errors
            return None, raw, f"{type(exc).__name__}: {str(exc)[:300]}"
        u = r.usage
        if r.model and r.model not in meta.setdefault("model_returned", []):
            meta["model_returned"].append(r.model)
        with _lock:
            _spent["in"] += u.prompt_tokens
            _spent["out"] += u.completion_tokens
            usage_log.write(json.dumps({"video_id": vid, "call": tag, "attempt": attempt, "in": u.prompt_tokens,
                                        "out": u.completion_tokens, "model_returned": r.model}) + "\n")
            usage_log.flush()
        raw = r.choices[0].message.content or ""
        try:
            return parse_json(raw), raw, None
        except Exception as exc:
            err = f"parse: {exc}"
            msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": "Return only the JSON object, valid JSON, nothing else."}]
    return None, raw, err


def first_sentences(text, k):
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(sents[:k])


def wiki_search(query):
    r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=20,
                     params={"action": "query", "list": "search", "srsearch": query, "srlimit": HITS_PER_QUERY,
                             "format": "json"})
    r.raise_for_status()
    return [h["title"] for h in r.json().get("query", {}).get("search", [])]


def wiki_summary(title):
    """Returns (first SUMMARY_SENTENCES sentences or None, HTTP status)."""
    r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")),
                     headers=UA, timeout=20)
    if r.status_code != 200:
        return None, r.status_code
    return first_sentences(r.json().get("extract", ""), SUMMARY_SENTENCES), 200


def clamp_time(x, dur):
    try:
        return min(max(float(x), 0.0), dur)
    except (TypeError, ValueError):
        return None


def as_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_judgment(jd, dur):
    if not jd or str(jd.get("violates", "")).lower() not in ("yes", "no"):
        return None
    try:
        conf = min(max(float(jd.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.5
    times = []
    for t in jd.get("times") or []:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            a, b = clamp_time(t[0], dur), clamp_time(t[1], dur)
            if a is not None and b is not None:
                times.append([min(a, b), max(a, b)])
    return {"violates": str(jd["violates"]).lower(), "confidence": conf, "target": jd.get("target"),
            "intent": jd.get("intent"), "means": jd.get("means"), "times": times,
            "explanation": " ".join(str(jd.get("explanation", "")).split()[:80])}


def process(client, row, asr, usage_log):
    ds, vid, dur = row["dataset"], row["video_id"], float(row["duration"])
    today = time.strftime("%Y-%m-%d")
    out = {"dataset": ds, "video_id": vid, "duration": dur, "model": MODEL, "code_path": CODE_PATH,
           "date": today, "wikipedia_retrieved": today, "host": socket.gethostname(), "errors": {}}
    frames = frame_paths(ds, vid, 20, "k20")
    transcript = transcript_block(asr.get(vid, []))
    # 1. clues
    clues, raw, err = call(client, media_content(frames, CLUE_PROMPT.format(n=len(frames), transcript=transcript)),
                           1500, usage_log, "clues", vid, out)
    out["clues_raw"] = raw
    if err:
        out["errors"]["clues"] = err
    items = []
    for it in (clues or {}).get("items", [])[:10]:
        a, b = clamp_time(it.get("t_start"), dur), clamp_time(it.get("t_end"), dur)
        if a is None or b is None or not str(it.get("text", "")).strip():
            continue
        items.append({"kind": it.get("kind"), "text": str(it["text"]).strip(), "t_start": min(a, b), "t_end": max(a, b),
                      "queries": [str(q) for q in (it.get("queries") or [])[:2] if str(q).strip()]})
    out["items"], out["social_domain"] = items, (clues or {}).get("social_domain")
    # 2. retrieval
    queries = []
    for it in items:
        for q in it["queries"]:
            if q not in queries:
                queries.append(q)
    queries = queries[:MAX_QUERIES]
    summaries, seen = [], set()
    for q in queries:
        try:
            titles = wiki_search(q)
        except Exception as exc:
            out["errors"].setdefault("wikipedia", []).append(f"search {q!r}: {type(exc).__name__}")
            continue
        for t in titles:
            if t in seen:
                continue
            seen.add(t)
            try:
                s, status = wiki_summary(t)
            except Exception as exc:
                out["errors"].setdefault("wikipedia", []).append(f"summary {t!r}: {type(exc).__name__}")
                continue
            if status != 200:
                out["errors"].setdefault("wikipedia", []).append(f"summary {t!r}: HTTP {status}")
            if s:
                summaries.append({"id": len(summaries) + 1, "query": q, "title": t, "summary": s})
    out["queries"], out["summaries"] = queries, summaries
    # 3. self-reflection
    notes = []
    if summaries:
        rp = REFLECT_PROMPT.format(max_notes=MAX_NOTES, transcript=transcript,
                                   items="\n".join(f"- [{i['t_start']:.1f}s-{i['t_end']:.1f}s] ({i['kind']}) {i['text']}" for i in items),
                                   summaries="\n".join(f"{s['id']}. {s['title']}: {s['summary']}" for s in summaries))
        ref, raw, err = call(client, [{"type": "text", "text": rp}], 1500, usage_log, "reflect", vid, out)
        out["reflect_raw"] = raw
        if err:
            out["errors"]["reflect"] = err
        by_id = {s["id"]: s for s in summaries}
        for n in (ref or {}).get("notes", [])[:MAX_NOTES]:
            a, b = clamp_time(n.get("t_start"), dur), clamp_time(n.get("t_end"), dur)
            src = by_id.get(as_int(n.get("summary_id")))
            if a is None or b is None or src is None or not str(n.get("note", "")).strip():
                continue
            notes.append({"summary_id": src["id"], "clue": str(n.get("clue", "")).strip(), "t_start": min(a, b),
                          "t_end": max(a, b), "source_title": src["title"],
                          "note": " ".join(str(n["note"]).split()[:60])})
    out["notes"] = notes
    # 4. judgment
    note_text = "\n".join(f"[{n['t_start']:.1f}s-{n['t_end']:.1f}s] {n['clue']} ({n['source_title']}): {n['note']}"
                          for n in notes) or "(none)"
    jd, raw, err = call(client, media_content(frames, JUDGE_PROMPT.format(n=len(frames), rules=RULES, transcript=transcript,
                                                                          notes=note_text)), 600, usage_log, "judge", vid, out)
    out["judge_raw"] = raw
    if err:
        out["errors"]["judge"] = err
    out["judgment"] = parse_judgment(jd, dur)
    # 5. judgment without the notes (same frames, transcript and rules)
    jd0, raw, err = call(client, media_content(frames, JUDGE0_PROMPT.format(n=len(frames), rules=RULES, transcript=transcript)),
                         600, usage_log, "judge0", vid, out)
    out["judge0_raw"] = raw
    if err:
        out["errors"]["judge0"] = err
    out["judgment0"] = parse_judgment(jd0, dur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg", "DeHate"])
    ap.add_argument("--limit", type=int, default=0, help="first N subset videos per corpus (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-yuan", type=float, default=20.0)
    args = ap.parse_args()
    from openai import OpenAI
    key = open(os.path.expanduser("~/.dashscope_api_key")).read().strip()
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=180, max_retries=2)
    del key
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(AGENT_DIR / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s  code %s  model %s", socket.gethostname(), CODE_PATH, MODEL)
    (AGENT_DIR / "run.pid").write_text(str(os.getpid()))
    usage_log = open(AGENT_DIR / "usage.jsonl", "a")
    todo = []
    for ds in args.datasets:
        rows = subset_rows(ds)
        if args.limit:
            rows = rows[:args.limit]
        asr = load_asr(ds)
        (AGENT_DIR / ds).mkdir(exist_ok=True)
        todo += [(r, asr) for r in rows if not (AGENT_DIR / ds / f"{r['video_id']}.json").exists()]
    logging.info("videos to do %d", len(todo))
    stop = threading.Event()

    def work(item):
        row, asr = item
        if stop.is_set():
            return
        t0 = time.time()
        try:
            out = process(client, row, asr, usage_log)
        except Exception as exc:
            logging.exception("FAILED %s %s: %s", row["dataset"], row["video_id"], exc)
            return
        p = AGENT_DIR / row["dataset"] / f"{row['video_id']}.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        with _lock:
            cost = _spent["in"] / 1000 * PRICE_IN + _spent["out"] / 1000 * PRICE_OUT
        logging.info("%s %s items %d summaries %d notes %d judgment %s judgment0 %s errors %s %.0fs  spent ~%.2f yuan",
                     row["dataset"], row["video_id"], len(out["items"]), len(out["summaries"]), len(out["notes"]),
                     (out["judgment"] or {}).get("violates"), (out["judgment0"] or {}).get("violates"),
                     list(out["errors"]), time.time() - t0, cost)
        if cost > args.budget_yuan:
            logging.error("BUDGET reached (%.2f > %.2f yuan): stopping", cost, args.budget_yuan)
            stop.set()

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    logging.info("DONE spent_in=%d spent_out=%d approx_yuan=%.2f", _spent["in"], _spent["out"],
                 _spent["in"] / 1000 * PRICE_IN + _spent["out"] / 1000 * PRICE_OUT)


if __name__ == "__main__":
    main()
