# Candidate 51: agent-retrieved knowledge for the M1 verdict (and the window reads)

Declared 2026-10-26, before any run. Development host uoa-lab1 (sc474397). API calls run on uoa-lab2 (sc474399), which
holds the DashScope key (`~/.dashscope_api_key`, read at call time, never written anywhere). GPU reads on uoa-lab1,
Slurm partition `local-sc474397`. Every number this candidate produces is development-selected (rule 10).

## 1. Why
- User request (2026-10-26): borrow the method of CHER (Ailneni & Harabagiu, "Discerning Culturally-Aware Hate Framings
  for Hate Speech Identification in Memes", EMNLP 2026 main, https://aclanthology.org/2026.emnlp-main.2538/) to make
  the video-level verdict more accurate. Design given by the user: a stronger agent retrieves the knowledge; the
  current 8B reader then reads the video with it; if it works, try a smaller reader of the same family.
- CHER: a ReAct agent infers social entities, groups, symbols and the social domain of a meme, retrieves Wikipedia
  summaries and other cultural sources, filters them by self-reflection ("known in mainstream discourse" and "directly
  explains this meme"), then identifies target / intent / act and decides by contrasting hateful and non-hateful
  explanations. Its ablation: removing the knowledge package costs the most (GPT-5 US 82.3 → 79.3, IN 78.9 → 69.9).
  Its limitations section lists extension to video as future work.
- What this repo already closed and is NOT repeated here: target / intent / means decomposition (PP-v2, MADP, DVD, HVL
  TARGET/FORM, RAVEL: new questions collapse to the verdict, ρ .89–.99), opposite-hypothesis reasoning (MARS
  reproduction, candidate 4 Marginalizer stopped against MARS / RAMF). External knowledge was never tried.
- Oracle (`experiments/20261026_verdict_oracle/README.md`, rule 10): r6's wrong verdicts are almost all false
  positives (HateMM 64 of 67, HateClipSeg 10 of 15, DeHate 518 of 556). Fixing half of the wrong verdicts would give
  pooled +.013 / +.016 (HateMM) and +.024 / +.011 (HateClipSeg); within cannot move through the verdict alone.

## 2. Mechanism
Per video, three API calls to `qwen3-vl-plus` (DashScope OpenAI-compatible endpoint, temperature 0, seed 0,
`enable_thinking=false`), then the r6 read with the result inserted:

1. **Clues (vision call).** Input: the same 20 frames the 8B reader sees (`data/frames_k20`, long side ≤ 512 px, JPEG
   q80) with their timestamps, and the same timestamped transcript (`src/video_inputs.transcript_block`). Output JSON:
   social entities (people, organisations, places, dates, events), visible social groups, symbols (flags, logos,
   gestures, on-screen text, meme formats), coded or unusual terms in the speech, and the social domain. Each clue has
   the time(s) where it appears and at most two Wikipedia search queries.
2. **Retrieval (no model).** English Wikipedia API: for each query (at most 12 per video) the top 3 search hits with the
   lead section of each, in one request (`generator=search`, `prop=extracts`, `exintro`, plain text), cut to its first 5
   sentences (CHER keeps the first five lines). Requests are serialised with 1 s between them and retried on HTTP 429
   (the first smoke test, 4 threads in parallel, was rate-limited after the first query of each video). Only entity
   names and search words leave the machine; no video content is sent to Wikipedia.
3. **Self-reflection (text call).** Input: transcript, clues, retrieved summaries. Keep only entries that pass both
   CHER criteria (known in mainstream English-language online discourse; directly explains something in this video).
   Output: at most 8 notes `{t_start, t_end, clue, source_title, note}`, each note at most 60 words restating only what
   its summary says, each summary used at most once (the first smoke test showed notes with facts from the model's
   own memory attributed to an unrelated summary).
4. **Judgment (vision call).** Input: frames, transcript, the kept notes, the same 9 policy rules the 8B reader uses.
   Output JSON: `violates` (yes / no), `confidence` (0–1), target, intent, means, times `[[s, e], ...]`, explanation
   (at most 80 words). The user ruled (2026-10-26) that the reader may take this judgment as an input; arm J and arm KJ
   test it, and arm K keeps the facts alone.
4b. **Judgment without the notes (vision call, control).** The same call with the same frames, transcript and rules but
   no notes (`judgment0`). It measures how much of a judgment-arm gain is the stronger model itself rather than the
   knowledge (rule-4 review advisory 1; rule 14(f) asks for this number).
5. **8B read** (`kv_measure.py`, a copy of `experiments/20260922_til/til_measure.py` `score_video` with the inserts; the
   frozen prompt material in `src/mllm_judge.py` is not edited). The block is appended to the end of the prefix, after
   the policy rules and before the verdict question:
   `Background notes (retrieved automatically; they may be irrelevant, and a term may also have an ordinary meaning):`
   followed by one line per note `[t_start–t_end] clue (source_title): note`. For arms with the window insert, each window
   question gets, before its first sentence, `Notes for this moment:` and the notes whose interval overlaps the window
   (at most 2, in pack order; the last window also takes notes that start at its end). Windows without such notes get
   no insert. Everything after the read is r6 unchanged
   (`experiments/20260926_twolevel/twolevel_r2.py --key raw`).

## 3. Arms (all on the same subset)
| arm | prefix insert | window insert |
|---|---|---|
| base | none | none (must reproduce the stored r6 z_video within bf16 noise) |
| K | notes | notes of the window |
| Kp | notes | none |
| KJ | notes + judgment | notes of the window |
| J | judgment | none |
| J0 | judgment written without the notes | none |
| Kother | notes of another video of the same corpus (fixed derangement, seed 0), times scaled to this video's duration | that video's notes mapped to windows by the scaled time |
| Kshift | as K | the video's own notes, each moved by half the video duration (circular, length kept) |
| Kmark | as K | at the windows K would note: the notes' time spans with the content replaced by "(note withheld)" |
| A (CPU) | none; z_video := the agent's judgment probability, put on the base z_video scale by rank within the corpus subset | base window reads |
| A0 (CPU) | as A, from the judgment written without the notes | base window reads |

Kshift and Kmark share K's prefix, so their z_video equals K's; they differ from K only in the window reads. They test
whether the window insert helps because the right facts sit at the right window (rule-4 review advisory 2: the earlier
StreamingTOM controls showed that material placed next to the window question raises within by itself, and the note
times are themselves a localization signal from the stronger model).

## 4. Subset, scale-up and gates (fixed before any run)
- **Subset:** 40 test videos per corpus (HateMM, HateClipSeg, DeHate), `numpy.random.default_rng(0).choice` over the
  sorted test ids of each corpus. No label is used to pick them. 6 of the 40 DeHate ids are not in
  `data/gt_4fps/DeHate.npz` (hateful videos the GT builder dropped for having no usable span), so DeHate is evaluated
  on 34 videos; the GT-free G0 counts use all 40. Base false positives / false negatives on the subset (rule-6 review):
  HateMM 11 / 1, HateClipSeg 3 / 2, DeHate 17 / 0; videos with a within-video ROC: 15, 35, 6.
- **Decoding:** arm reads replace the r6 reads of the subset videos; all other videos keep their r6 reads; the full
  corpus is decoded with the r6 settings, and metrics are computed on the 40 subset videos with the repo evaluator.
- **G0 (GT-free):** at least 60 % of videos get ≥ 1 note; Spearman(z_video of K, z_video of base) < .9 on at least two
  corpora.
- **G1 (development, on the subset):** for K or KJ: (a) pooled ROC and PR both above base on HateMM and on HateClipSeg;
  (b) at least 40 % of base false positives become verdict-No, counted over the three corpora together (31 base false
  positives; HateClipSeg alone has 3); (c) at most 1 new false negative per corpus; (d) gain ≥ 2 × max(Kother gain, 0),
  where gain = mean of ΔROC and ΔPR over HateMM and HateClipSeg (the content, not "more text", carries it); and for KJ
  also (e) KJ gain > J0 gain (the notes add to what the stronger model's own judgment gives). New false positives and
  the A / A0 numbers are reported next to it. Computed in `decode_eval.py` (`summary.json` → `verdict`).
- **G2 (decides whether the window insert is part of the claim):** within(K) above within(Kp), within(Kshift) and
  within(Kmark) on both HateMM and HateClipSeg, reported with the number of within videos (15 and 35). If G1 passes and
  G2 fails, the claim is the verdict part only (close to Yadav & Singh, WWW Companion 2026; see the rule-4 review).
- Gates amended 2026-10-26 after the rule-4 and rule-6 reviews (`docs/reviews/20261026_knowledge_verdict_novelty.md`,
  `docs/reviews/20261026_knowledge_verdict_code.md`), before any API or GPU run.
- **Scale-up (user ruling 2026-10-26):** only if G1 passes, run the full test set of at most two corpora (HateMM and
  HateClipSeg first). A smaller reader of the same family (Qwen3-VL-4B, then 2B) is tried only after that.

## 5. Cost
- API: 4 calls per video (3 with 20 frames; the fourth, judgment0, is a control only), roughly 35k input and 3k output
  tokens per video. At the price recorded in Retrieval-hate (¥0.002 / 1k in, ¥0.008 / 1k out; may be outdated) about
  ¥0.09 per video, ¥11 for the subset. Budget cap for the subset: ¥20. Wikipedia: at most 12 searches + 36 summary
  requests per video. Each pack records the model name the endpoint returned and the Wikipedia retrieval date.
- GPU: the r6 read plus about 650 prefix tokens and about 100 tokens per noted window; 9 arms × 120 videos, about
  30 min on one RTX 5090.
- For a new video at deployment: 3 API calls + the r6 read; no added 8B calls.

## 5b. How to run
1. uoa-lab2 (API, CPU only, detached): `bash experiments/20261026_knowledge_verdict/launch/lab2_agent.sh --limit 2`
   (smoke), then without `--limit`. Log `runs/20261026_knowledge_verdict/agent/run.log`.
2. rsync `runs/20261026_knowledge_verdict/agent/` back to uoa-lab1.
3. uoa-lab1: `RUN=smoke LIMIT=1 sbatch experiments/20261026_knowledge_verdict/launch/lab1_read.sbatch`, then
   `sbatch experiments/20261026_knowledge_verdict/launch/lab1_read.sbatch`.
4. uoa-lab1 (CPU): `python experiments/20261026_knowledge_verdict/decode_eval.py`.

## 6. Error analysis log (rule 10)
- 2026-10-26: `experiments/20261026_verdict_oracle/` (verdict flips, test GT) set the target "fix about half of the
  false positives". No other test file was read for this design.

## 7. Status
Rule-4 review PASS, rule-6 review one blocker (DeHate ids missing from GT in the gate code), fixed together with the
review advisories (arms J0, Kshift, Kmark, A, A0; gates computed in code). Waiting for the agent run.
