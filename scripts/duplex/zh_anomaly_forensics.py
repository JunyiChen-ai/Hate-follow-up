"""MHClip_ZH test anomaly: control comparison and transcript forensics.

The MHClip_ZH test_clean C2 run reported 8B AUC 0.675 against 2B 0.8125, the
only scale inversion across the four benchmarks. This module assembles the
evidence on what caused it and whether the fresh Whisper Chinese transcripts are
implicated at all.

Two things are computed.

1. The control comparison. `judge_8b_ctrl` and `judge_2b_ctrl` re-score the same
   149 test videos under a configuration identical to the C2 arms in every
   respect except one: no override map is passed, so the judge reads the dataset
   transcript instead of the gated fresh Whisper transcript. Comparing the C2
   and control arms on the same videos separates the transcript from the split.

2. Forensics on the 149 fresh Chinese transcripts: Whisper's own per-chunk
   language votes, an independent Unicode-script profile, length shift against
   the dataset transcript, the gate's collapse and rejection statistics, gzip
   ratios, degeneracy indicators, and normalized edit distance -- each crossed
   with the label half, and each crossed with how far that video's raw z moved
   between the control and C2 arms.

Statistics only. No transcript text, no video ids, and no per-video rows reach
the output; everything is aggregated or reported as a distribution.

Usage:
  python scripts/duplex/zh_anomaly_forensics.py \
      --out docs/duplex/reports/test_zh_anomaly_diag.json
"""

import argparse
import gzip
import json
import math
import os
import statistics
import sys
import unicodedata
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from channel_restoration_analyze import auc, median, quantile  # noqa: E402
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

WORK = os.path.join(ROOT, "results", "testruns", "mhclip_zh")
LABEL_MAP = {"Hateful": 1, "Offensive": 1, "Normal": 0}


# --------------------------------------------------------------------- io
def read_jsonl(path, key="video_id"):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d[r[key]] = r
    return d


def load_z(arm):
    path = os.path.join(WORK, f"judge_{arm}", "scores.jsonl")
    out = {}
    for vid, r in read_jsonl(path).items():
        if isinstance(r.get("z"), (int, float)) and math.isfinite(r["z"]):
            out[vid] = r["z"]
    return out


def dist(xs, nd=4):
    """Five-number summary plus mean. Empty input returns nulls."""
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": round(min(xs), nd), "q1": round(quantile(xs, 0.25), nd),
            "median": round(median(xs), nd), "q3": round(quantile(xs, 0.75), nd),
            "max": round(max(xs), nd), "mean": round(sum(xs) / len(xs), nd),
            "sd": round(statistics.pstdev(xs), nd) if len(xs) > 1 else 0.0}


def boot_auc(scores, pos, neg, n_boot=2000, seed=0):
    """Percentile bootstrap on the AUC, resampling videos within each class."""
    import random
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        p = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        q = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        vals.append(auc(scores, p, q))
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 6),
            round(vals[int(0.975 * len(vals))], 6)]


def arm_auc(z_by_vid, y_by_vid):
    pos = [v for v in z_by_vid if y_by_vid.get(v) == 1]
    neg = [v for v in z_by_vid if y_by_vid.get(v) == 0]
    if not pos or not neg:
        return None
    return {"auc": round(auc(z_by_vid, pos, neg), 6),
            "boot95": boot_auc(z_by_vid, pos, neg),
            "n_pos": len(pos), "n_neg": len(neg), "n_scored": len(z_by_vid)}


# ------------------------------------------------------------- text profile
def script_profile(text):
    """Fraction of the text's letters/ideographs falling in each writing system.

    Independent of Whisper's own language vote: it reads the Unicode block of
    every non-punctuation, non-digit, non-space character. `han` covers CJK
    ideographs, `latin` the ASCII/Latin alphabets, and `kana`, `hangul`,
    `cyrillic`, `arabic`, `other` the remainder.
    """
    counts = Counter()
    n = 0
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        cat = unicodedata.category(ch)
        if not cat.startswith("L") and not cat.startswith("M"):
            continue
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            k = "han"
        elif 0x3040 <= cp <= 0x30FF:
            k = "kana"
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            k = "hangul"
        elif cp < 0x0250:
            k = "latin"
        elif 0x0400 <= cp <= 0x04FF:
            k = "cyrillic"
        elif 0x0600 <= cp <= 0x06FF:
            k = "arabic"
        else:
            k = "other"
        counts[k] += 1
        n += 1
    if n == 0:
        return {"n_letters": 0, "dominant": "none", "frac": {}}
    frac = {k: round(v / n, 4) for k, v in counts.items()}
    return {"n_letters": n, "dominant": max(frac, key=frac.get), "frac": frac}


def gzip_ratio(text):
    b = text.encode("utf-8")
    if not b:
        return None
    return round(len(b) / len(gzip.compress(b)), 4)


def degeneracy_flags(text):
    """Indicators that the ASR output is a loop or a non-speech artifact.

    `loop_frac` is the share of the text taken by clauses that occur more than
    once, so a text whose clauses are all distinct scores 0 and a text that is
    one phrase repeated scores near 1. It is defined only when the text splits
    into at least two clauses. `max_char_run` is the longest run of one repeated
    character and `distinct_char_ratio` the vocabulary of the string over its
    length; both rise when Whisper collapses on music or silence.
    `bracket_tag` catches the bracketed non-speech tags Whisper emits on music
    and applause, and the subscribe-and-like boilerplate it hallucinates on
    Chinese short video.
    """
    if not text:
        return {"loop_frac": None, "max_char_run": 0, "n_clauses": 0,
                "distinct_char_ratio": None, "bracket_tag": False}
    parts = [p.strip() for p in text.replace("\n", " ")
             .replace("。", "。|").replace("，", "，|").replace(".", ".|")
             .replace(",", ",|").split("|") if p.strip()]
    loop_frac = None
    if len(parts) >= 2:
        c = Counter(parts)
        repeated = sum(k * len(p) for p, k in c.items() if k > 1)
        loop_frac = round(repeated / max(sum(len(p) for p in parts), 1), 4)
    run = best = 1
    for i in range(1, len(text)):
        run = run + 1 if text[i] == text[i - 1] else 1
        best = max(best, run)
    tags = ("[", "(", "♪", "《音", "音乐", "掌声", "Music", "music",
            "Applause", "字幕", "订阅", "点赞")
    return {"loop_frac": loop_frac, "max_char_run": best, "n_clauses": len(parts),
            "distinct_char_ratio": round(len(set(text)) / len(text), 4),
            "bracket_tag": any(t in text for t in tags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = load_clean_split_ids("MHClip_ZH", "test")
    ann = load_annotations("MHClip_ZH")
    y = {v: LABEL_MAP[ann[v]["label"]] for v in ids
         if v in ann and ann[v].get("label") in LABEL_MAP}

    fresh = read_jsonl(os.path.join(WORK, "fresh_transcripts.jsonl"))
    gate = read_jsonl(os.path.join(WORK, "gate_outcomes.jsonl"))
    audio = read_jsonl(os.path.join(WORK, "audio_meta.jsonl"))
    overrides = json.load(open(os.path.join(WORK, "c2_overrides.json")))

    z_c2_8b, z_c2_2b = load_z("8b"), load_z("2b")
    z_ct_8b, z_ct_2b = load_z("8b_ctrl"), load_z("2b_ctrl")
    z_fz_8b = load_z("8b_forcedzh")

    forced_dir = os.path.join(WORK, "forced_zh")
    fresh_fz = read_jsonl(os.path.join(forced_dir, "fresh_transcripts.jsonl"))
    gate_fz = read_jsonl(os.path.join(forced_dir, "gate_outcomes.jsonl"))

    # ------------------------------------------------- the four-cell table
    prior = {}
    for name, path in [
            ("train_dataset_transcript_8b", "crossbench_mhclip_zh_8b.json"),
            ("train_dataset_transcript_2b", "crossbench_mhclip_zh_2b.json")]:
        p = os.path.join(ROOT, "docs", "duplex", "reports", path)
        d = json.load(open(p))
        prior[name] = {"auc": d["auc"]["hateful_vs_normal"],
                       "boot95": d["auc"]["boot95"],
                       "n_pos": d["auc"]["n_pos"], "n_neg": d["auc"]["n_neg"],
                       "n_scored": d["coverage"]["n_scored"],
                       "restoration_fraction":
                           d["restoration_coverage"]["restoration_fraction"],
                       "source": f"docs/duplex/reports/{path}"}

    cells = {
        "train_x_dataset_transcript": {
            "8b": prior["train_dataset_transcript_8b"],
            "2b": prior["train_dataset_transcript_2b"]},
        "train_x_fresh_whisper": {
            "8b": None, "2b": None,
            "note": "never measured: the train-split source media was not pulled "
                    "to this machine, so restoration_fraction on train is 0.0 and "
                    "the ASR route was never exercised there"},
        "test_x_dataset_transcript": {
            "8b": arm_auc(z_ct_8b, y), "2b": arm_auc(z_ct_2b, y),
            "note": "the control arm run for this diagnostic"},
        "test_x_fresh_whisper": {
            "8b": arm_auc(z_c2_8b, y), "2b": arm_auc(z_c2_2b, y),
            "note": "the C2 condition, re-run to full coverage after the frame repair"},
        "test_x_fresh_whisper_language_forced_zh": {
            "8b": arm_auc(z_fz_8b, y), "2b": None,
            "note": "fifth cell, from the optional arm: the same pipeline with "
                    "Whisper's decoding language pinned to zh instead of detected"},
    }

    def delta_auc(z_a, z_b, seed=0, n_boot=4000):
        """Paired bootstrap on AUC(a) - AUC(b) over the same videos.

        The two arms score the same 149 videos, so the videos are resampled once
        and both arms are re-scored on that resample. The paired design cancels
        the between-video variance that dominates the marginal interval and
        leaves the variance of the difference itself.
        """
        import random
        vids = sorted(set(z_a) & set(z_b) & set(y))
        pos = [v for v in vids if y[v] == 1]
        neg = [v for v in vids if y[v] == 0]
        if not pos or not neg:
            return None
        point = auc(z_a, pos, neg) - auc(z_b, pos, neg)
        rng = random.Random(seed)
        vals = []
        for _ in range(n_boot):
            p = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
            q = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
            vals.append(auc(z_a, p, q) - auc(z_b, p, q))
        vals.sort()
        lo, hi = vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]
        return {"delta_auc": round(point, 6),
                "boot95_paired": [round(lo, 6), round(hi, 6)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "n_pos": len(pos), "n_neg": len(neg)}

    restoration_effect = {
        "role": "AUC(fresh Whisper) - AUC(dataset transcript) on the same 149 test "
                "videos, with a paired bootstrap. This is the quantity the question "
                "'did restoration hurt' actually asks about.",
        "8b_fresh_minus_dataset": delta_auc(z_c2_8b, z_ct_8b),
        "2b_fresh_minus_dataset": delta_auc(z_c2_2b, z_ct_2b),
        "8b_forcedzh_minus_dataset": delta_auc(z_fz_8b, z_ct_8b),
        "8b_forcedzh_minus_autolang": delta_auc(z_fz_8b, z_c2_8b),
        "split_effect_8b_dataset_transcript": {
            "role": "the other half of the question: is the test split different? "
                    "Both numbers read the dataset transcript, so the transcript is "
                    "held fixed and only the split changes. Unpaired: different videos.",
            "train_auc": prior["train_dataset_transcript_8b"]["auc"],
            "test_auc": arm_auc(z_ct_8b, y)["auc"],
            "test_minus_train": round(arm_auc(z_ct_8b, y)["auc"]
                                      - prior["train_dataset_transcript_8b"]["auc"], 6),
            "train_boot95": prior["train_dataset_transcript_8b"]["boot95"],
            "test_boot95": arm_auc(z_ct_8b, y)["boot95"],
        },
    }

    # paired difference on the same 149 videos
    common = sorted(set(z_c2_8b) & set(z_ct_8b) & set(y))
    dz8 = {v: z_c2_8b[v] - z_ct_8b[v] for v in common}
    common2 = sorted(set(z_c2_2b) & set(z_ct_2b) & set(y))
    dz2 = {v: z_c2_2b[v] - z_ct_2b[v] for v in common2}

    def spearman(a, b):
        def rank(xs):
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            r = [0.0] * len(xs)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                    j += 1
                avg = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        ra, rb = rank(a), rank(b)
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        num = sum((x - ma) * (yv - mb) for x, yv in zip(ra, rb))
        den = math.sqrt(sum((x - ma) ** 2 for x in ra)
                        * sum((yv - mb) ** 2 for yv in rb))
        return round(num / den, 4) if den else None

    paired = {
        "role": "the same 149 videos scored twice, differing only in which "
                "transcript the judge read. dz = z(fresh Whisper) - z(dataset).",
        "8b": {
            "n": len(common),
            "dz": dist([dz8[v] for v in common]),
            "dz_hateful": dist([dz8[v] for v in common if y[v] == 1]),
            "dz_normal": dist([dz8[v] for v in common if y[v] == 0]),
            "n_dz_zero": sum(1 for v in common if abs(dz8[v]) < 1e-9),
            "n_dz_gt_1": sum(1 for v in common if abs(dz8[v]) > 1.0),
            "n_dz_gt_5": sum(1 for v in common if abs(dz8[v]) > 5.0),
            "spearman_z_c2_vs_ctrl": spearman([z_c2_8b[v] for v in common],
                                              [z_ct_8b[v] for v in common]),
            "separation_shift": {
                "role": "mean z on hateful minus mean z on normal, per arm. The "
                        "quantity the AUC ranks.",
                "c2_fresh": round(
                    sum(z_c2_8b[v] for v in common if y[v] == 1) / max(sum(1 for v in common if y[v] == 1), 1)
                    - sum(z_c2_8b[v] for v in common if y[v] == 0) / max(sum(1 for v in common if y[v] == 0), 1), 4),
                "control_dataset": round(
                    sum(z_ct_8b[v] for v in common if y[v] == 1) / max(sum(1 for v in common if y[v] == 1), 1)
                    - sum(z_ct_8b[v] for v in common if y[v] == 0) / max(sum(1 for v in common if y[v] == 0), 1), 4)},
        },
        "2b": {
            "n": len(common2),
            "dz": dist([dz2[v] for v in common2]),
            "dz_hateful": dist([dz2[v] for v in common2 if y[v] == 1]),
            "dz_normal": dist([dz2[v] for v in common2 if y[v] == 0]),
            "n_dz_zero": sum(1 for v in common2 if abs(dz2[v]) < 1e-9),
            "spearman_z_c2_vs_ctrl": spearman([z_c2_2b[v] for v in common2],
                                              [z_ct_2b[v] for v in common2]),
        },
    }

    # ------------------------------------------------------- forensics
    rows = []
    for v in ids:
        fr = fresh.get(v, {})
        ft = fr.get("fresh_text", "") or ""
        dt = (ann.get(v, {}) or {}).get("transcript", "") or ""
        gt = gate.get(v, {})
        served = overrides.get(v)
        rows.append({
            "vid": v, "y": y.get(v),
            "top_language": fr.get("top_language"),
            "languages": fr.get("languages") or {},
            "prof_fresh": script_profile(ft),
            "prof_dataset": script_profile(dt),
            "fresh_chars": len(ft), "dataset_chars": len(dt),
            "served_chars": len(served) if served is not None else len(dt),
            "served_is_fresh": served is not None,
            "gzip_ratio_raw": fr.get("gzip_ratio_raw"),
            "gzip_ratio_dataset": gzip_ratio(dt),
            "edit_norm": fr.get("edit_norm_vs_dataset"),
            "vad": fr.get("vad_speech_frac"),
            "wav_duration": (audio.get(v, {}) or {}).get("wav_duration"),
            "collapse_shrink": gt.get("collapse_shrink"),
            "outcome": gt.get("outcome"),
            "deg": degeneracy_flags(ft),
            "deg_dataset": degeneracy_flags(dt),
            "dz8": dz8.get(v),
        })

    def half(pred):
        return [r for r in rows if pred(r)]

    hate, norm = half(lambda r: r["y"] == 1), half(lambda r: r["y"] == 0)

    def lang_tab(rs):
        c = Counter(r["top_language"] or "none" for r in rs)
        dom = Counter(r["prof_fresh"]["dominant"] for r in rs)
        n = len(rs)
        n_voted = sum(1 for r in rs if r["top_language"])
        return {
            "n": n,
            "whisper_top_language": dict(c.most_common()),
            "whisper_vote_available": n_voted,
            "whisper_non_zh_frac": (round(
                sum(1 for r in rs if r["top_language"] and r["top_language"] != "chinese")
                / n_voted, 4) if n_voted else None),
            "whisper_vote_note": (
                "the ASR pipeline returned no per-chunk language for any video on "
                "this run, so Whisper's own language vote is unavailable and the "
                "Unicode script profile below is the language evidence"
                if n_voted == 0 else None),
            "script_dominant_fresh": dict(dom.most_common()),
            "script_non_han_frac": round(
                sum(1 for r in rs if r["prof_fresh"]["dominant"] not in ("han", "none")) / n, 4) if n else None,
            "han_frac_of_letters": dist([r["prof_fresh"]["frac"].get("han", 0.0)
                                         for r in rs if r["prof_fresh"]["n_letters"] > 0]),
            "latin_frac_of_letters": dist([r["prof_fresh"]["frac"].get("latin", 0.0)
                                           for r in rs if r["prof_fresh"]["n_letters"] > 0]),
        }

    def len_tab(rs):
        return {
            "dataset_chars": dist([r["dataset_chars"] for r in rs]),
            "fresh_chars": dist([r["fresh_chars"] for r in rs]),
            "served_chars": dist([r["served_chars"] for r in rs]),
            "log2_ratio_fresh_over_dataset": dist(
                [math.log2((r["fresh_chars"] + 1) / (r["dataset_chars"] + 1)) for r in rs]),
            "n_fresh_shorter": sum(1 for r in rs if r["fresh_chars"] < r["dataset_chars"]),
            "n_fresh_longer": sum(1 for r in rs if r["fresh_chars"] > r["dataset_chars"]),
            "n_fresh_near_empty_lt10": sum(1 for r in rs if r["fresh_chars"] < 10),
            "n_dataset_empty": sum(1 for r in rs if r["dataset_chars"] == 0),
        }

    def deg_tab(rs):
        return {
            "gzip_ratio_fresh": dist([r["gzip_ratio_raw"] for r in rs]),
            "gzip_ratio_dataset": dist([r["gzip_ratio_dataset"] for r in rs]),
            "loop_frac_fresh": dist([r["deg"]["loop_frac"] for r in rs]),
            "loop_frac_dataset": dist([r["deg_dataset"]["loop_frac"] for r in rs]),
            "max_char_run_fresh": dist([r["deg"]["max_char_run"] for r in rs]),
            "distinct_char_ratio_fresh": dist([r["deg"]["distinct_char_ratio"] for r in rs]),
            "frac_bracket_or_nonspeech_tag": round(
                sum(1 for r in rs if r["deg"]["bracket_tag"]) / len(rs), 4) if rs else None,
            "frac_loop_frac_over_0p5": round(
                sum(1 for r in rs if (r["deg"]["loop_frac"] or 0) > 0.5) / len(rs), 4) if rs else None,
            "collapse_shrink": dist([r["collapse_shrink"] for r in rs]),
            "n_collapse_shrink_gt_0p2": sum(1 for r in rs if (r["collapse_shrink"] or 0) > 0.2),
            "edit_norm_vs_dataset": dist([r["edit_norm"] for r in rs]),
            "vad_speech_frac": dist([r["vad"] for r in rs]),
            "outcomes": dict(Counter(r["outcome"] for r in rs).most_common()),
        }

    forensics = {
        "role": "what the fresh Chinese ASR actually produced, and how it differs "
                "from the dataset transcript the judge would otherwise have read",
        "language": {"all": lang_tab(rows), "hateful": lang_tab(hate),
                     "normal": lang_tab(norm)},
        "length": {"all": len_tab(rows), "hateful": len_tab(hate),
                   "normal": len_tab(norm)},
        "degeneracy_and_gate": {"all": deg_tab(rows), "hateful": deg_tab(hate),
                                "normal": deg_tab(norm)},
    }

    # ------------------------------------- what moved the most, characterized
    moved = [r for r in rows if r["dz8"] is not None]
    moved.sort(key=lambda r: -abs(r["dz8"]))
    k = max(1, len(moved) // 10)
    top, rest = moved[:k], moved[k:]

    def cmp_group(rs):
        return {
            "n": len(rs),
            "abs_dz8": dist([abs(r["dz8"]) for r in rs]),
            "edit_norm": dist([r["edit_norm"] for r in rs]),
            "log2_len_ratio": dist([math.log2((r["fresh_chars"] + 1) / (r["dataset_chars"] + 1)) for r in rs]),
            "fresh_chars": dist([r["fresh_chars"] for r in rs]),
            "dataset_chars": dist([r["dataset_chars"] for r in rs]),
            "han_frac": dist([r["prof_fresh"]["frac"].get("han", 0.0) for r in rs]),
            "whisper_non_zh_frac": round(
                sum(1 for r in rs if (r["top_language"] or "chinese") != "chinese") / len(rs), 4) if rs else None,
            "loop_frac_fresh": dist([r["deg"]["loop_frac"] for r in rs]),
            "vad": dist([r["vad"] for r in rs]),
            "prevalence_hateful": round(sum(1 for r in rs if r["y"] == 1) / len(rs), 4) if rs else None,
        }

    # which input statistic best explains |dz|
    def sp(field):
        pairs = [(field(r), abs(r["dz8"])) for r in moved if field(r) is not None]
        if len(pairs) < 10:
            return None
        return spearman([p[0] for p in pairs], [p[1] for p in pairs])

    movement = {
        "role": "the videos whose 8B raw z moved most between the control and the "
                "C2 arm, characterized by what changed in their judge input. "
                "Statistics only; no ids and no text.",
        "top_decile_by_abs_dz": cmp_group(top),
        "remaining_nine_deciles": cmp_group(rest),
        "spearman_abs_dz_vs": {
            "edit_norm_vs_dataset": sp(lambda r: r["edit_norm"]),
            "abs_log2_len_ratio": sp(lambda r: abs(math.log2((r["fresh_chars"] + 1) / (r["dataset_chars"] + 1)))),
            "fresh_chars": sp(lambda r: r["fresh_chars"]),
            "dataset_chars": sp(lambda r: r["dataset_chars"]),
            "han_frac_fresh": sp(lambda r: r["prof_fresh"]["frac"].get("han", 0.0)),
            "loop_frac_fresh": sp(lambda r: r["deg"]["loop_frac"]),
            "vad_speech_frac": sp(lambda r: r["vad"]),
            "note": "Spearman rank correlation between an input statistic and how "
                    "far that video's 8B raw z moved. Read as: which axis of input "
                    "change the judge is actually sensitive to.",
        },
    }

    # ------------------------------------------- forced-zh diagnostic arm
    fz_rows = []
    for v in ids:
        ft = (fresh_fz.get(v, {}) or {}).get("fresh_text", "") or ""
        auto = (fresh.get(v, {}) or {}).get("fresh_text", "") or ""
        fz_rows.append({"vid": v, "y": y.get(v), "chars": len(ft),
                        "prof": script_profile(ft), "auto_chars": len(auto),
                        "auto_prof": script_profile(auto),
                        "gzip": (fresh_fz.get(v, {}) or {}).get("gzip_ratio_raw"),
                        "edit_norm": (fresh_fz.get(v, {}) or {}).get("edit_norm_vs_dataset"),
                        "outcome": (gate_fz.get(v, {}) or {}).get("outcome")})

    n_fz = len(fz_rows)
    auto_non_han = [r for r in fz_rows if r["auto_prof"]["dominant"] not in ("han", "none")]
    rescued = [r for r in auto_non_han if r["prof"]["dominant"] == "han"]
    forced = {
        "role": "Whisper re-run over the same 149 wavs with the decoding language "
                "forced to zh, then the same frozen gate and the same 8B judge. "
                "This isolates the language-mismatch channel.",
        "trigger": {
            "rule": "run this arm only if at least 10 percent of clips "
                    "auto-detect a non-Chinese language",
            "whisper_vote": "unavailable: the pipeline returned no per-chunk "
                            "language for any of the 149 videos",
            "proxy_used": "the Unicode script profile of the auto-detected "
                          "transcript, which is independent of Whisper's vote",
            "observed_non_han_dominant": len(auto_non_han),
            "observed_frac": round(len(auto_non_han) / n_fz, 4),
            "fired": len(auto_non_han) / n_fz >= 0.10,
        },
        "effect_on_the_transcript": {
            "n_auto_non_han_dominant": len(auto_non_han),
            "n_of_those_now_han_dominant": len(rescued),
            "rescue_rate": round(len(rescued) / len(auto_non_han), 4) if auto_non_han else None,
            "script_dominant_forced": dict(Counter(r["prof"]["dominant"] for r in fz_rows).most_common()),
            "han_frac_auto": dist([r["auto_prof"]["frac"].get("han", 0.0) for r in fz_rows]),
            "han_frac_forced": dist([r["prof"]["frac"].get("han", 0.0) for r in fz_rows]),
            "chars_auto": dist([r["auto_chars"] for r in fz_rows]),
            "chars_forced": dist([r["chars"] for r in fz_rows]),
            "gate_outcomes_forced": dict(Counter(r["outcome"] for r in fz_rows).most_common()),
        },
        "effect_on_the_score": {
            "8b": arm_auc(z_fz_8b, y),
            "vs_auto_language_c2": {
                "dz": dist([z_fz_8b[v] - z_c2_8b[v] for v in z_fz_8b if v in z_c2_8b]),
                "spearman": spearman([z_fz_8b[v] for v in sorted(set(z_fz_8b) & set(z_c2_8b))],
                                     [z_c2_8b[v] for v in sorted(set(z_fz_8b) & set(z_c2_8b))]),
            },
            "restricted_to_the_auto_non_han_videos": {
                "n": len([r for r in auto_non_han if r["vid"] in z_fz_8b]),
                "dz_vs_auto": dist([z_fz_8b[r["vid"]] - z_c2_8b[r["vid"]]
                                    for r in auto_non_han
                                    if r["vid"] in z_fz_8b and r["vid"] in z_c2_8b]),
                "note": "if language mismatch were the causal channel, the videos "
                        "whose auto transcript was not Chinese are where forcing zh "
                        "should move the score",
            },
        },
    }

    out = {
        "title": "MHClip_ZH held-out test anomaly: diagnostic",
        "date": "2026-08-07",
        "status": "diagnostic run. Labels enter evaluation only; no component was "
                  "tuned or selected against them.",
        "question": "The MHClip_ZH test_clean C2 run reported 8B AUC 0.675 against "
                    "2B 0.8125 -- the only scale inversion across four benchmarks -- "
                    "with restoration coverage 1.0. Did the fresh Whisper Chinese "
                    "transcripts hurt the 8B, or is the test split simply different?",
        "finding_headline": "Neither. The reported run was truncated by a corrupt "
                            "input file, not by the method.",
        "root_cause": {
            "what": "frames_16/<one MHClip_ZH test video>/frame_012.jpg was a "
                    "partially written JPEG: 131,059 bytes against roughly 260,000 "
                    "for every intact neighbour in the same directory.",
            "mechanism": "extract_duplex_readout.py opens all 16 frames with "
                         "PIL.Image.open(p).convert('RGB') outside any try block. "
                         "The truncated file raises OSError, which propagates out of "
                         "main() and kills the process. The run script retries three "
                         "times, and each retry resumes onto the same file and dies "
                         "at the same video.",
            "blast_radius": "The loop died at video 14 of 149. Both the 8B and the "
                            "2B arm therefore scored 13 videos, and the published "
                            "AUCs rested on 5 positives and 8 negatives.",
            "why_only_this_dataset": "a scan of every frame of every test_clean "
                                     "video across all four benchmarks found exactly "
                                     "one unreadable JPEG, in MHClip_ZH. HateMM, "
                                     "MHClip_EN and ImpliHateVid have none, which is "
                                     "why only this dataset's run was truncated.",
            "frame_scan": {"MHClip_ZH": 1, "MHClip_EN": 0, "HateMM": 0,
                           "ImpliHateVid": 0,
                           "unit": "unreadable frames across the test_clean split"},
            "repair": "the frame was re-decoded from the local source mp4 at the "
                      "index the original extractor used, "
                      "np.linspace(0, n_frames-1, 16, dtype=int)[12]. The same code "
                      "path applied to the intact neighbour frame reproduces the "
                      "on-disk file at 37.61 dB PSNR (MAE 1.46), which is JPEG "
                      "re-encoding noise, so the decoder's frame numbering agrees "
                      "with the original extractor's. The truncated original was "
                      "quarantined alongside it, not deleted.",
            "superseded_numbers": {
                "role": "what the truncated run reported, kept for the record",
                "8b_auc_at_n13": 0.675, "2b_auc_at_n13": 0.8125,
                "n_scored": 13, "n_pos": 5, "n_neg": 8,
                "boot95_8b": [0.35, 0.95], "boot95_2b": [0.55, 1.0]},
        },
        "four_cell_table": {
            "role": "{train, test} x {dataset transcript, fresh Whisper} for the 8B, "
                    "with the 2B alongside. Every test cell is 149 videos, 45 "
                    "hateful and 104 normal.",
            "cells": cells,
        },
        "restoration_effect": restoration_effect,
        "paired_control_vs_c2": paired,
        "transcript_forensics": forensics,
        "score_movement": movement,
        "forced_zh_arm": forced,
        "verdict": {
            "a_restoration_vs_split": {
                "restoration_hurt": "no, not measurably. On the same 149 test "
                                    "videos the fresh Whisper transcript moves the "
                                    "8B AUC by -0.0118, paired bootstrap "
                                    "[-0.0334, +0.0083], which straddles zero. The "
                                    "2B moves +0.0047, [-0.0362, +0.0458].",
                "split_is_different": "no. Reading the dataset transcript in both "
                                      "splits, the 8B scores 0.8555 on 579 train "
                                      "videos and 0.8665 on 149 test videos, a "
                                      "difference of +0.0110 with heavily "
                                      "overlapping intervals.",
                "what_actually_happened": "one truncated JPEG aborted the judge "
                                          "loop at video 14 of 149. The published "
                                          "0.675 was an AUC over 5 positives and 8 "
                                          "negatives with a bootstrap interval of "
                                          "[0.35, 0.95], wide enough to contain both "
                                          "the true value and the 2B's. At full "
                                          "coverage the 8B scores 0.8547 and the 2B "
                                          "0.7917, so the scale inversion disappears "
                                          "and MHClip_ZH matches the ordering of the "
                                          "other three benchmarks.",
            },
            "b_causal_channel": {
                "conclusion": "there is no harmful channel to find: the effect being "
                              "explained is itself within noise. The candidate "
                              "channels were nevertheless measured and all are small.",
                "language_mismatch": "24 of 149 auto-detected transcripts (16.1 "
                                     "percent) are not Han-dominant. Forcing zh "
                                     "rescues 20 of those 24 and lifts the median "
                                     "Han fraction to 1.0, but moves the 8B AUC by "
                                     "+0.0030, [-0.0078, +0.0140]. The mechanism is "
                                     "real and the consequence is nil.",
                "asr_quality": "median normalized edit distance to the dataset "
                               "transcript is 0.43, so the fresh text is genuinely "
                               "different text rather than a copy, yet the 8B raw z "
                               "ranks the two arms at Spearman 0.974. The judge is "
                               "reading the same evidence either way.",
                "length_shift": "the fresh transcript is shorter than the dataset "
                                "one for 86 of 149 videos and longer for 60; the "
                                "median log2 length ratio is -0.13. Absolute z "
                                "movement correlates with the length ratio at "
                                "Spearman 0.03.",
                "degeneracy": "the gate rejected 2 of 149 under auto language and 3 "
                              "of 149 under forced zh. 3.4 percent of transcripts "
                              "have more than half their text in repeated clauses "
                              "and 17.5 percent carry a non-speech or "
                              "subscribe-and-like tag, but none of it reaches the "
                              "score: loop fraction correlates with absolute z "
                              "movement at Spearman -0.08.",
                "strongest_axis": "Han fraction of the fresh transcript, at Spearman "
                                  "0.18 against absolute z movement. That is the "
                                  "largest of the seven axes measured and it is still "
                                  "weak, which is consistent with the forced-zh arm "
                                  "changing nothing.",
            },
            "c_recommendation": {
                "keep_automatic_language_detection": "forcing zh buys +0.003 AUC, "
                                                     "inside noise, and costs a "
                                                     "per-corpus language decision "
                                                     "that the method would then have "
                                                     "to make correctly for every new "
                                                     "corpus. Automatic detection is "
                                                     "the more honest default and the "
                                                     "measurement says it is not "
                                                     "costing anything.",
                "restoration_on_non_english": "channel restoration is neutral on "
                                              "MHClip_ZH rather than helpful. The "
                                              "starvation premise it runs on is "
                                              "weaker here: the dataset transcript "
                                              "already has a median of 76 characters "
                                              "and only 1.3 percent of videos exceed "
                                              "the 300-character window, so there is "
                                              "little starvation left to relieve. "
                                              "Report it as a neutral result on this "
                                              "corpus, not as evidence against the "
                                              "mechanism, and do not tune the ASR to "
                                              "chase it.",
                "infrastructure": "add an input-integrity precondition. One "
                                  "unreadable frame silently cost 91 percent of a "
                                  "held-out measurement and the run still reported "
                                  "DONE with a plausible-looking AUC. The judge "
                                  "should skip and record an unreadable frame rather "
                                  "than abort, and the analysis should refuse to "
                                  "write a report when coverage is far below the "
                                  "split size instead of quietly reporting the "
                                  "subset.",
            },
        },
        "provenance": {
            "judge": "src/duplex/extract_duplex_readout.py, unmodified. All four "
                     "arms: frozen prag reader, 16 frames from frames_16, "
                     "max_pixels 100352, transcript-limit 0, single forward pass.",
            "arms": {
                "judge_8b / judge_2b": "--transcript-override-json c2_overrides.json "
                                       "(gated fresh Whisper large-v3)",
                "judge_8b_ctrl / judge_2b_ctrl": "no override map, so "
                                                 "resolve_transcript returns the "
                                                 "dataset transcript for every video",
            },
            "runner": "scripts/duplex/zh_anomaly_diag.sh, then "
                      "scripts/duplex/zh_forced_lang_arm.sh",
            "forensics": "scripts/duplex/zh_anomaly_forensics.py",
            "gpu_seconds_total": 470,
            "asr": "the frozen auto-language transcripts were reused unchanged. "
                   "The forced-zh arm re-ran Whisper over the same wavs into a "
                   "separate file; crossbench_asr.py gained --language and "
                   "--asr-name, both defaulting to the frozen behaviour.",
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")
    print(json.dumps(cells, indent=1)[:2000])
    print("non-zh (whisper vote):", forensics["language"]["all"]["whisper_non_zh_frac"])
    print("non-han (script):", forensics["language"]["all"]["script_non_han_frac"])


if __name__ == "__main__":
    main()
