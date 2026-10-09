#!/usr/bin/env python
"""LAVAD-AV: LAVAD with the clip's speech transcript in the scoring prompt and the hate rules in place of the
law-enforcement question (user decision 2026-10-09). Visual-only LAVAD (`lavad_chain.py`, `finalize.py`) is not
changed; this file only adds a new scoring stage, the curves and the finalisation of the new variant.

What changes against the visual-only port
  * Stage 04b (LLM scoring) only. Llama-2-13b-chat (NF4, greedy, max_seq_len 512, batch 48; `lavad_chain.Scorer`,
    unchanged) now reads
      system  "Based on this platform's hate speech policy, the following types of content are prohibited:" + the
              9 rules of hate_query.md section 1 + LAVAD's rating question with the anomaly description replaced
              ("... and 1 denoting a scene with content that violates any of the above rules?") + LAVAD's verbatim
              format instruction (one number of the list [0, 0.1, ..., 1.0], begin with '[' and end with ']').
      user    "<temporal summary of the clip>.\n\nSpeech in this clip: <Whisper text of the clip span>"
              (the summary turn is LAVAD's f"{summary}."; an empty span gives "(no speech)").
    The clip span of centre c is LAVAD's 10 s window [max(c - 5, 0), min(c + 5, n)) seconds on the 1 fps grid
    (`lavad_chain.window_frames`), n = last centre + 1. Span text: `transcript_windows.Transcripts`.
  * Reused unchanged from the visual-only run of the same corpus (their inputs do not contain the scoring prompt):
    BLIP-2 captions, ImageBind caption cleaning, the Llama-2 temporal summaries (stage 04a; the summary prompt sees
    captions only) and the stage-05/06 summary index and neighbours (`refined`: for every centre the 10 nearest
    summaries of the same video and their ImageBind similarities).
      HateMM       the Retrieval-hate campaign's work files (uoa-lab2 RTX 5090, 2026-08; the visual-only HateMM row
                   is that campaign's curve), copied to data/retrieval_hate_repro/repro_lavad_work/HateMM/
      HateClipSeg  runs/20261008_baselines/lavad/work/{summary,refined}/HateClipSeg/ (uoa-campus2 A100, Slurm 24602)
      DeHate       runs/20261008_baselines/lavad/work/{summary,refined}/DeHate/ (uoa-campus2 A100, Slurm 24603)
  * Refined score (LAVAD's reported curve, `base`): for each centre, softmax(similarity)-weighted mean of the NEW
    scores of its 10 neighbour summaries, refusals masked (the arithmetic of `lavad_chain.stage_curves`).
  * Prompt budget: LAVAD's max_seq_len is 512 and `Scorer` truncates longer prompts from the right (which would cut
    the [/INST] tag; the official Meta loader would stop with an assertion). A prompt longer than 496 tokens (512
    minus 16 tokens left for the answer) keeps the first words of the speech text that fit, followed by " ..."; if
    it is still too long with the speech cut to "...", the summary is cut the same way. Counts are logged and
    written to `prompt_stats_<DS>.json`. The rules make the system turn about 150 tokens longer than LAVAD's.

Stages
  check-reuse  CPU: recompute the visual-only refined curve from the reused visual score + refined files with this
               file's arithmetic and compare it with the stored visual-only curve (must be identical)
  lengths      CPU: prompt token lengths with the Llama-2 tokenizer, no model (dry run of the prompt builder)
  score        GPU: stage 04b with the AV prompt -> runs/20261008_baselines/lavad_av/work/score/<DS>/<vid>.json
  curves       CPU: 1 fps `base` / `raw` curves -> runs/20261008_baselines/lavad_av/curves/<DS>/<vid>.npz
  finalize     CPU: F3 (nearest scored sample), 1 fps -> 4 fps, exact-cohort check, canonical evaluator
               -> runs/20261008_baselines/lavad_av/<DS>/
No labels are read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments/20261008_baselines/lavad"
sys.path.insert(0, str(ROOT / "experiments/20261008_baselines"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "shim"))   # llama_hf (the Llama-2 shim of the visual-only port)
import exact_cohort as ec  # noqa: E402
from transcript_windows import EMPTY, Transcripts  # noqa: E402

CODE_PATH = "experiments/20261008_baselines/lavad/lavad_av.py"
RUN_DIR = ROOT / "runs/20261008_baselines/lavad_av"
WORK = RUN_DIR / "work"
VISUAL_RUN = ROOT / "runs/20261008_baselines/lavad"
CAMPAIGN_WORK = ROOT / "data/retrieval_hate_repro/repro_lavad_work"
CAMPAIGN_CURVES = Path("/home/jehc223/Retrieval-hate/archive/idea-stage/repro_lavad/curves")  # read only, lab1
DATASETS = ("HateMM", "HateClipSeg", "DeHate")

CLIP_DURATION = 10      # lavad_chain.CLIP_DURATION (LAVAD's T, seconds)
FPS = 1.0               # lavad_chain.FPS (1 fps grid, centre k = k s)
MAX_SEQ_LEN = 512       # lavad_chain.MAX_SEQ_LEN (LAVAD's --max_seq_len)
ANSWER_RESERVE = 16     # tokens left for the answer when a prompt is shortened

# LAVAD's format instruction, verbatim (lavad_chain.FORMAT_PROMPT, scripts/04_query_llm.sh)
FORMAT_PROMPT = (
    "Please provide the response in the form of a Python list and respond with only "
    "one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, "
    "0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and "
    "end with  ']'."
)
# hate_query.md section 1, verbatim (= experiments/20260910_spvl/spvl.py YOUTUBE_RULES)
YOUTUBE_RULES = """1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events"""
# LAVAD's rating question with the law-enforcement persona and "suspicious activities" replaced by the hate rules
CONTEXT_PROMPT_HATE = (
    "Based on this platform's hate speech policy, the following types of content are prohibited:\n"
    f"{YOUTUBE_RULES}\n\n"
    "How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 "
    "denoting a scene with content that violates any of the above rules?"
)
SYSTEM_PROMPT = CONTEXT_PROMPT_HATE + " " + FORMAT_PROMPT   # LAVAD joins context and format with one space
SPEECH_LABEL = "Speech in this clip:"


def user_turn(summary: str, speech: str) -> str:
    return f"{summary}.\n\n{SPEECH_LABEL} {speech if speech else EMPTY}"


def cut(words: list[str], k: int) -> str:
    """First k words followed by ' ...' (marks text removed to fit the prompt budget)."""
    return (" ".join(words[:k]) + " ...").strip()


# ------------------------------------------------------------------ paths ---
def visual_file(ds: str, stage: str, vid: str) -> Path:
    """Reused visual-only stage output (summary / refined / score) of the corpus."""
    if ds == "HateMM":
        return CAMPAIGN_WORK / ds / stage / f"{vid}.json"
    return VISUAL_RUN / "work" / stage / ds / f"{vid}.json"


def visual_curve(ds: str, vid: str) -> Path:
    if ds == "HateMM":
        return CAMPAIGN_CURVES / ds / f"{vid}.npz"
    return VISUAL_RUN / "curves" / ds / f"{vid}.npz"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def clip_span(c: int, n: int) -> tuple[float, float]:
    """lavad_chain.window_frames at 1 fps: frames [max(c - 5, 0), min(c + 5, n)) = seconds of the same range."""
    half = int(CLIP_DURATION * FPS) // 2
    return float(max(c - half, 0)), float(min(c + half, n))


# -------------------------------------------------------------- prompts ---
def load_tokenizer():
    """Tokenizer exactly as the llama_hf shim sets it up (chat template fallback, left padding)."""
    from transformers import AutoTokenizer
    from llama_hf import LLAMA2_CHAT_TEMPLATE
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-13b-chat-hf")
    if tok.chat_template is None:
        tok.chat_template = LLAMA2_CHAT_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


class PromptBuilder:
    def __init__(self, tok):
        self.tok = tok
        self.limit = MAX_SEQ_LEN - ANSWER_RESERVE
        self.stats = Counter()

    def n_tokens(self, system: str, user: str) -> int:
        p = self.tok.apply_chat_template([{"role": "system", "content": system},
                                          {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        return len(self.tok(p, add_special_tokens=False)["input_ids"])

    def _fit(self, words: list[str], render) -> int:
        """Largest k such that render(first k words + ' ...') fits the budget (0 if none does)."""
        lo, hi = 0, len(words) - 1      # k = len(words) (nothing cut) was already tried by the caller
        while lo < hi:
            k = (lo + hi + 1) // 2
            if self.n_tokens(SYSTEM_PROMPT, render(cut(words, k))) <= self.limit:
                lo = k
            else:
                hi = k - 1
        return lo

    def __call__(self, summary: str, speech: str) -> tuple[tuple[str, str], int, int]:
        """-> ((system, user), prompt tokens, speech words dropped).

        Over the budget: the speech text keeps its first words that fit and ends in " ..."; only if the prompt
        with the speech cut to "..." is still too long, the summary is cut the same way (speech stays "...")."""
        user = user_turn(summary, speech)
        n = self.n_tokens(SYSTEM_PROMPT, user)
        self.stats["prompts"] += 1
        self.stats["with_speech"] += bool(speech)
        if n <= self.limit:
            return (SYSTEM_PROMPT, user), n, 0
        words = speech.split()
        dropped = 0
        if words:
            k = self._fit(words, lambda s: user_turn(summary, s))
            user = user_turn(summary, cut(words, k))
            dropped = len(words) - k
            self.stats["speech_shortened"] += 1
            self.stats["speech_words_dropped"] += dropped
        if self.n_tokens(SYSTEM_PROMPT, user) > self.limit:
            sw = summary.split()
            sp = cut(words, 0) if words else ""
            j = self._fit(sw, lambda s: user_turn(s.rstrip("."), sp)) if sw else 0
            user = user_turn(cut(sw, j).rstrip("."), sp)
            self.stats["summary_shortened"] += 1
            self.stats["summary_words_dropped"] += len(sw) - j
        n2 = self.n_tokens(SYSTEM_PROMPT, user)
        if n2 > self.limit:
            self.stats["over_limit_after_shortening"] += 1
        return (SYSTEM_PROMPT, user), n2, dropped


def cohort_with_summary(ds: str) -> list[str]:
    return [v for v in ec.cohort(ds) if visual_file(ds, "summary", v).exists()]


def build_pairs(ds: str, vid: str, tr: Transcripts, pb: PromptBuilder):
    summ = json.loads(visual_file(ds, "summary", vid).read_text())
    centers = sorted(summ, key=int)
    n = int(centers[-1]) + 1 if centers else 0
    pairs, speech, ntok, dropped = [], {}, [], 0
    for c in centers:
        a, b = clip_span(int(c), n)
        s = tr.span(vid, a, b)
        pair, nt, dr = pb(summ[c], s)
        pairs.append(pair)
        speech[c] = s
        ntok.append(nt)
        dropped += dr
    return centers, pairs, speech, ntok, dropped


# --------------------------------------------------------------- stages ---
def refined_curve(scores: dict[int, float], refined: dict, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """lavad_chain.stage_curves arithmetic: raw = LLM score per centre (refusal -> NaN); base = softmax(sim)-weighted
    mean of the answered neighbours' scores (NaN if all 10 refused)."""
    centers = sorted(scores)
    T = centers[-1] + step
    r = np.full(T, np.nan)
    for c in centers:
        if scores[c] >= 0:
            r[c: c + step] = scores[c]
    b = np.full(T, np.nan)
    for c_s, d in refined.items():
        c = int(c_s)
        if c >= T:
            continue
        sc = np.array([scores.get(f, -1.0) for f in d["nn_frame"]])
        w = np.array(d["sim"], dtype=np.float64)
        ok = sc >= 0
        if not ok.any():
            continue
        e = np.exp(w[ok] - w[ok].max())
        b[c: c + step] = float((sc[ok] * e).sum() / e.sum())
    return r, b


def stage_check_reuse(a) -> int:
    rc = 0
    for ds in a.datasets:
        res = {"n": 0, "identical": 0, "missing": 0, "max_abs_diff": 0.0, "nan_pattern_diff": 0, "mismatch": []}
        for v in ec.cohort(ds):
            sp, rp, cp = visual_file(ds, "score", v), visual_file(ds, "refined", v), visual_curve(ds, v)
            if not (sp.exists() and rp.exists() and cp.exists()):
                res["missing"] += 1
                continue
            scores = {int(k): x for k, x in json.loads(sp.read_text()).items()}
            _, b = refined_curve(scores, json.loads(rp.read_text()))
            ref = np.asarray(np.load(cp)["base"], dtype=np.float64)
            res["n"] += 1
            if len(ref) != len(b) or not np.array_equal(np.isnan(ref), np.isnan(b)):
                res["nan_pattern_diff"] += 1
                res["mismatch"].append(v)
                continue
            m = ~np.isnan(b)
            d = float(np.abs(ref[m] - b[m]).max()) if m.any() else 0.0
            res["max_abs_diff"] = max(res["max_abs_diff"], d)
            res["identical"] += d <= 1e-12
            if d > 1e-12:
                res["mismatch"].append(v)
        ok = res["n"] == len(ec.cohort(ds)) and res["identical"] == res["n"]
        res["ok"] = ok
        print(f"CHECK-REUSE {ds} {json.dumps(res)[:600]}", flush=True)
        write_json(RUN_DIR / f"reuse_check_{ds}.json", res)
        rc |= 0 if ok else 8
    return rc


def stage_lengths(a) -> int:
    tok = load_tokenizer()
    for ds in a.datasets:
        tr, pb = Transcripts(ds), PromptBuilder(tok)
        ids = cohort_with_summary(ds)[: a.limit or None]
        allt, drop = [], 0
        for v in ids:
            _, _, _, nt, dr = build_pairs(ds, v, tr, pb)
            allt += nt
            drop += dr
        allt = np.array(allt)
        out = {"dataset": ds, "videos": len(ids), "prompts": int(len(allt)), **dict(pb.stats),
               "tokens_p50": float(np.percentile(allt, 50)), "tokens_p99": float(np.percentile(allt, 99)),
               "tokens_max": int(allt.max()), "limit": pb.limit}
        print("LENGTHS " + json.dumps(out), flush=True)
    return 0


def stage_score(a) -> int:
    from lavad_chain import SCORE_RE, Scorer
    sc = Scorer(a.batch_size)
    pb = PromptBuilder(sc.tok)
    print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT, flush=True)
    for ds in a.datasets:
        tr = Transcripts(ds)
        ids = cohort_with_summary(ds)
        missing = sorted(set(ec.cohort(ds)) - set(ids))
        if missing:
            print(f"!! {ds}: {len(missing)} cohort videos have no visual summary: {missing[:5]}", flush=True)
        todo = [v for v in ids if not (WORK / "score" / ds / f"{v}.json").exists()]
        print(f"PROGRESS score_av {ds} plan={len(todo)}/{len(ids)}", flush=True)
        t0, n, ncall, nref = time.time(), 0, 0, 0
        for vid in todo:
            centers, pairs, speech, ntok, dropped = build_pairs(ds, vid, tr, pb)
            if n == 0:
                print("EXAMPLE USER TURN:\n" + pairs[len(pairs) // 2][1], flush=True)
            outs = sc(pairs)
            scores, refusals = {}, {}
            for c, o in zip(centers, outs):
                m = SCORE_RE.search(o)
                if m:
                    scores[c] = float(m.group(1))
                else:
                    scores[c] = -1.0            # LAVAD's sentinel; masked, not interpolated
                    refusals[c] = o[:400]
            write_json(WORK / "speech" / ds / f"{vid}.json", speech)
            write_json(WORK / "score_refusals" / ds / f"{vid}.json", refusals)
            write_json(WORK / "score" / ds / f"{vid}.json", scores)
            n += 1
            ncall += len(centers)
            nref += len(refusals)
            if n % 5 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"PROGRESS score_av {ds} {n}/{len(todo)} calls={ncall} refusals={nref} gen={sc.n_gen} "
                      f"cachehit={sc.n_hit} trunc={sc.n_trunc} oom={sc.n_oom} "
                      f"{sc.n_gen / max(el, 1e-9):.2f} gen/s elapsed={el / 60:.1f}min "
                      f"eta={(len(todo) - n) * el / n / 60:.1f}min speech_shortened={pb.stats['speech_shortened']} summary_shortened={pb.stats['summary_shortened']}", flush=True)
        st = {"dataset": ds, "videos_scored_now": n, "calls": ncall, "refusals": nref, **dict(pb.stats),
              "gen": sc.n_gen, "cachehit": sc.n_hit, "trunc": sc.n_trunc, "oom": sc.n_oom,
              "wall_min": round((time.time() - t0) / 60, 1)}
        write_json(RUN_DIR / f"prompt_stats_{ds}.json", st)
        print(f"[done] score_av {ds} {json.dumps(st)}", flush=True)
        pb.stats.clear()
    return 0


def stage_curves(a) -> int:
    rc = 0
    for ds in a.datasets:
        out_dir = RUN_DIR / "curves" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        st = Counter()
        for v in ec.cohort(ds):
            sp, rp = WORK / "score" / ds / f"{v}.json", visual_file(ds, "refined", v)
            if not sp.exists():
                st["no_score"] += 1
                continue
            scores = {int(k): x for k, x in json.loads(sp.read_text()).items()}
            if not scores:
                st["no_centre"] += 1
                continue
            refined = json.loads(rp.read_text()) if rp.exists() else {}
            st["no_refined"] += not rp.exists()
            r, b = refined_curve(scores, refined)
            np.savez(out_dir / f"{v}.npz", rate=np.float64(FPS), raw=r, base=b)
            st["videos"] += 1
            st["samples"] += len(r)
            st["refused_centres"] += int(np.isnan(r).sum())
            st["base_nan"] += int(np.isnan(b).sum())
        st = dict(st)
        write_json(RUN_DIR / f"refusal_stats_{ds}.json", st)
        print(f"[curves] {ds} {st}", flush=True)
        rc |= 0 if st.get("videos") else 9
    return rc


def stage_finalize(a) -> int:
    from finalize import fill_nearest     # lavad/finalize.py: F3 rule of the visual-only run
    rc = 0
    for ds in a.datasets:
        out = RUN_DIR / ds
        out.mkdir(parents=True, exist_ok=True)
        log = ec.RunLog(out / "run.log")
        (out / "run.pid").write_text(f"{os.getpid()}\n")
        log(f"code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
        dur = ec.durations(ds)
        curves, extra, fail = {}, {}, {}
        n_f3, n_f3_videos, n_tail = 0, 0, 0
        for v in ec.cohort(ds):
            p = RUN_DIR / "curves" / ds / f"{v}.npz"
            if not p.exists():
                fail[v] = "no curve (no visual summary or scoring not run)"
                continue
            c = np.asarray(np.load(p)["base"], dtype=np.float64)
            if not np.isfinite(c).any():
                fail[v] = "no scored 1 fps sample (all neighbours refused)"
                continue
            c, nf = fill_nearest(c)
            T = ec.curve_length(dur[v])
            tail = ec.tail_frames(len(c), FPS, T)
            n_f3 += nf
            n_f3_videos += nf > 0
            n_tail += tail
            curves[v] = ec.broadcast_to_4fps(c, FPS, T)
            extra[v] = {"native_samples": int(len(c)), "tail_hold_frames": int(tail)}
            if nf:
                extra[v]["fallback"] = {"code": "F3", "n_native_samples_filled": nf,
                                        "rule": "nearest scored 1 fps sample of the same video"}
                log(f"F3 {ds}/{v}: {nf} unscored 1 fps samples filled from the nearest scored sample")
        stats = json.loads((RUN_DIR / f"refusal_stats_{ds}.json").read_text())
        reuse = {"HateMM": "Retrieval-hate campaign summaries + refined neighbours (uoa-lab2 RTX 5090, 2026-08), "
                           "data/retrieval_hate_repro/repro_lavad_work/HateMM/",
                 "HateClipSeg": "runs/20261008_baselines/lavad/work/{summary,refined}/HateClipSeg/ "
                                "(uoa-campus2 A100, Slurm 24602)",
                 "DeHate": "runs/20261008_baselines/lavad/work/{summary,refined}/DeHate/ "
                           "(uoa-campus2 A100, Slurm 24603)"}[ds]
        rep = ec.finalize("lavad_av_base", ds, out, curves, native_rate=FPS, code_path=CODE_PATH, log=log,
                          extra=extra, failures=fail,
                          notes={"f3_native_samples": n_f3, "f3_videos": n_f3_videos,
                                 "tail_hold_frames_total": n_tail, "refusal_stats": stats},
                          config={"variant": "LAVAD-AV base (stage-06 refined score; scoring prompt = hate rules, "
                                             "user turn = temporal summary + speech transcript of the clip span)",
                                  "system_prompt": SYSTEM_PROMPT,
                                  "user_turn": user_turn("<temporal summary>", "<Whisper text of [max(c-5,0), "
                                                         "min(c+5,n)) s, or (no speech)>"),
                                  "reused_from_visual_run": reuse,
                                  "transcripts": "data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl, "
                                                 "qwen3_text load_segments + src/video_inputs.window_text",
                                  "llm": "NousResearch/Llama-2-13b-chat-hf NF4 (bitsandbytes, double quant, bf16 "
                                         "compute), greedy, max_seq_len 512, lavad_chain.Scorer",
                                  "native_rate": "1 fps", "grid": "frame i <- sample floor(i/4); tail holds last",
                                  "fallback_F3": "unscored 1 fps sample <- nearest scored sample of the same video"})
        log("DONE" if rep.get("exact_test_set") else "FAILED not evaluated")
        rc |= 0 if rep.get("exact_test_set") else 6
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check-reuse", "lengths", "score", "curves", "finalize"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="lengths stage: first N videos")
    a = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"lavad_av {a.stage} host {os.uname().nodename} {time.strftime('%Y-%m-%d %H:%M:%S')} code {CODE_PATH} "
          f"({ec.code_version()}) datasets {a.datasets}", flush=True)
    return {"check-reuse": stage_check_reuse, "lengths": stage_lengths, "score": stage_score,
            "curves": stage_curves, "finalize": stage_finalize}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
