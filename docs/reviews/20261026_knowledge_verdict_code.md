# Code review: candidate 51 (agent-retrieved knowledge for the verdict), rule 6

Date: 2026-10-26. Reviewer: independent agent. Scope (rule 6): only bugs that would change what the experiment observes or concludes. Files: `experiments/20261026_knowledge_verdict/` (all scripts and the sbatch; untracked at `68cb14e`), plus the shared code they call. Nothing was edited. CPU checks only.

## VERDICT: one BLOCKER (only in `decode_eval.py`; the API and GPU runs can start)

**B1. The DeHate gate loop crashes, so no `summary.json` is written** (`decode_eval.py:78`, `:82`, `:109`).
- 6 of the 40 DeHate subset ids are in the manifest but not in `data/gt_4fps/DeHate.npz`: `6D2rdGNRujCP, 7427143342787579139, JxKwEhoX1hZh, Ottt1b7n8sdH, cK0K8fIKklwW, tRSm6XkbcGVB`. The GT builder drops these hateful videos because they have no usable span.
- `gt[v]` then raises `KeyError` after all decoding. I reproduced this in a scratch copy.
- Fix:
  - Line 78: `ids = [v for v in subset[ds] if v in gt and all(...)]`.
  - Compute the GT-free G0 share over all 40 ids.
  - Record `n_not_in_gt`.
  - Write in README §4 that DeHate is evaluated on 34 videos (the evaluator overlap is 34).
  - Keep the subset selection as it is.
- With the fix, the gate section runs. Base FP / FN on the subset: HateMM 11/1, HCS 3/2, DeHate 17/0.

## Non-blocking notes

1. **No check that base reproduces r6.** README §3 requires that base reproduces the stored r6 `z_video`, but nothing computes it. Add the largest |Δ z_video| and |Δ window z| between `read_base` and `BASE_READS`.
2. **G1 pass/fail is not computed** (`decode_eval.py:97-106`). Before reading results, state in code:
   - which gain the "≥ 2 × Kother" rule uses (ROC and PR separately, per corpus);
   - what happens when the Kother gain is ≤ 0.

   HCS has only 3 negative videos in the subset, so "40 % of FPs fixed" means 2 of 3.
3. **Kother gets fewer window notes** when the other video has a different length (`common.py:59-62`). This follows README "mapped by time", but it makes the 2× control easier to pass. Report `windows_with_notes` for K and Kother, or rescale the times to this video's duration.
4. **Notes at t = duration reach no window** (`common.py:61`: `t_start < b` fails on the last window). Seen on `hate_video_1`. Fix: use `<=` on the last window, or clamp to `duration − 1e-3` (`agent_retrieve.py:174`).
5. **Silent note loss.**
   - A string `summary_id` drops the note (`agent_retrieve.py:238`). Fix: coerce with `int(float(·))`.
   - Wikipedia non-200 responses are not recorded (`:167-168`).
6. **Moderation refusals** (`:132`) leave the pack empty, so the prompt stays as base and no label leaks. Refusals likely correlate with hate; report their counts per corpus.
7. **Smoke tests** with `LIMIT` must use `RUN≠read`. Otherwise resume keeps those rows (`kv_measure.py:176-180`), and their Kother uses a shorter derangement (`:170`).
8. **Before the agent run:** uoa-lab2 is at `683b57e`. Commit, push and pull first. `agent_retrieve.py` has no launcher.

## Checked and found correct

- **Inserts.** I ran `kv_measure.score_video` with the real Qwen3-VL processor and template (HateVLM env, transformers 5.15.1, CPU) and a stub forward, on all six arms of `hate_video_1` with a synthetic pack.
  - The prefix insert sits after `READER_PRAG`, before the prefix `<|im_end|>`, the verdict question and the stance.
  - In K, KJ and Kother, `Notes for this moment:` precedes "Consider only window…" in both the visual and the speech branch. Kp, J and base have no window insert.
  - The token-seam check passes for all arms.
- **Base = r6.**
  - A line-by-line diff with `til_measure.score_video` shows only the inserts, an empty-by-default stance override, and logging.
  - On all 120 subset videos these match the stored `base_gridA` / `reads_gridA`: prefix tokens, window bounds, branch count, speech-branch set, duration and curve length.
  - Library versions also match.
- **No labels on the scoring path.** The scoring path reads only manifest `dataset/video_id/duration`, the frames and the ASR. The prompts carry no id or path; images go as base64.
- **Alignment.**
  - Times are in seconds everywhere.
  - The same 120 ids come out under numpy 1.26.4 and 2.5.2 and on lab2.
  - lab2 has identical frame lists, file sizes, ASR chunks and durations for all 120.
  - The derangement has no fixed points.
- **Decoder.**
  - `R6` matches every field of both r6 configs.
  - Scratch `decode_eval`, with the stored reads as arms, reproduces `final_rawkey` (.8955/.6883/.7508, .7137/.6666/.6373) and `r6_rawkey` (.6994/.1568/.6431) exactly.
  - Its subset metrics equal the stored r6 predictions restricted to the subset (ROC/PR/within, videos, within videos): HateMM .9081/.6560/.6575 (40, 15); HCS .6921/.6441/.6667 (40, 35); DeHate .7902/.1443/.5867 (34, 6).

  - The splice replaces exactly the subset rows that have a non-error read.
  - The evaluator is the unmodified repo file.

## Re-review (2026-10-26, after the fixes for B1, notes 1–8 and the rule-4 advisories)

Scope: only the changes in commits `f199e74` and `adc4986`. These cover `common.py`, `kv_measure.py`, `agent_retrieve.py`, `decode_eval.py`, README §2–§7 and `launch/`. No candidate file was edited. CPU only.

### VERDICT: one BLOCKER

**B2. Kshift does not keep the note length, which biases G2** (`common.py:48-55`).
- README §3 says the shift is "circular, length kept". The code shifts the start circularly but clips the end at the video duration.
- So a note that crosses the midpoint loses coverage after the shift. A whole-video note `[0, dur]` becomes `[dur/2, dur]`.
- Checked on CPU with `hate_video_1` (12 windows): the whole-video note reaches 12 windows in K but only 7 (windows 5–11) in Kshift.
- Effect on the control:
  - In K, such a note is uniform across all windows and carries no position.
  - In Kshift, the note becomes a half-video position signal, and the windows get less text.
- G2 is a strict "within(K) > within(Kshift)" test on 15 and 35 videos. It would partly measure note coverage, not note placement.
- Fix: wrap the part that runs past the end, and count each note once per window. Sketch:
  - If `b − a ≥ dur`, keep `[0, dur]`.
  - Otherwise, with `a2 = (a + dur/2) % dur` and `b2 = a2 + (b − a)`:
    - if `b2 ≤ dur`, emit `[a2, b2]`;
    - if `b2 > dur`, emit two pieces, `[a2, dur]` and `[0, b2 − dur]`. Tag both with the note's index `_k`.
  - In `window_notes`, deduplicate by `_k` before applying the 2-note cap.

### Non-blocking notes

1. **G0 strictness** (`decode_eval.py:209`). The code requires the ≥ 60 % notes share on all three corpora. The README sentence can also be read as one share over all videos. Fix the reading in README §4.
2. **A corpus with no judgment empties that corpus's gates** (`decode_eval.py:161`). If a whole corpus has no judgment (e.g. all judge calls refused), `cpu_arm_reads` returns nothing for it. Then `ids` is empty for every arm on that corpus, and gate (c) passes by default (`:217`, `.get(..., 0)`). Fix: build `ids` from the GPU arms only, or give the CPU arms the base read in that case.
3. **`base_reproduction` has no declared tolerance and is not part of `verdict`.** Declare a bound, or at least report it next to G1.
4. **Confidence semantics of A / A0** (`decode_eval.py:48`). The code assumes that `confidence` refers to the given answer. If the model sometimes gives the probability of violation instead, a "no" with confidence < .5 lands on the wrong side. Report how many answers have confidence < .5.

### Checked and found correct

- **Inserts for all nine arms** (stub-model CPU render, with the token-seam check on every arm):
  - J0 shows `judgment0`, and J shows `judgment`.
  - Kshift and Kmark use K's prefix, with the original times.
  - Kmark notes exactly K's windows, with the text "(note withheld)".
  - Kother times are rescaled: `[15, 30]` of a 30 s video becomes `[47.5, 95.0]` of 95 s, so the covered fraction is kept.
  - A note at `t = duration` now reaches the last window (old note 4).
  - `--limit` together with `--run-name read` exits.
- **`agent_retrieve.py`:**
  - `summary_id` is coerced with `as_int`.
  - Wikipedia non-200 statuses are recorded.
  - The model name returned by the endpoint is recorded.
  - The JUDGE0 prompt has no notes, and `parse_judgment` is shared by both judgment calls.
  - The `DataInspectionFailed` count works: the DashScope compatible-mode 400 message contains `<400> InternalError.Algo.DataInspectionFailed` within its first 300 characters (seen in the Retrieval-hate logs).
- **`decode_eval.py`:**
  - B1 is fixed: GT-missing ids are dropped from the GT counts, G0 is counted over all 40, and `n_not_in_gt` is recorded.
  - The G1 code matches README §4:
    - (a) both pooled deltas > 0 on HateMM and HCS;
    - (b) the FP share is pooled over the three corpora (31 base FPs);
    - (c) new FN ≤ 1 per corpus;
    - (d) gain ≥ 2 × max(Kother gain, 0), with gain = the mean of ΔROC and ΔPR over HateMM and HCS;
    - (e) KJ gain > J0 gain;
    - an arm passes only if every condition is `True`, so a missing comparison counts as a fail.
  - G2 equals within(K) − within(x) > 0 for Kp, Kshift and Kmark on both corpora.
  - The A / A0 mapping, tested with synthetic packs:
    - Spearman(mapped z, agent probability) = 1;
    - the mapped values are base z values of the same subset videos (tied probabilities share an interpolated value);
    - videos without a judgment keep their base read;
    - window reads are unchanged, and the base reads are not mutated.
- **`launch/lab2_agent.sh`** uses lab2's HateVideo env, which has openai 2.30, and runs detached with `setsid nohup`.

### Addendum: `adc4986` (Wikipedia requests and reflection notes)

**Checked and found correct**, with a mocked HTTP layer (no network):
- `wiki_get` serialises all requests over the threads with 1 s spacing.
- On 429 or 5xx it retries, waiting `Retry-After` or else 2, 4, 8, 16 s. On any other 4xx it stops and returns `(None, status)`, and the failure is written to `errors["wikipedia"]`.
- `wiki_search_summaries` returns the hits in search order (sorted by `index`), each cut to 5 sentences. A hit without an extract is skipped. A query with no hits returns an empty list, not an error.
- At most one note is kept per `summary_id`.
- The smoke packs from the rate-limited run were moved out of `agent/<dataset>/`. The full run therefore regenerates them instead of skipping them.

**Non-blocking note 5.** Nothing in the code checks that a note restates its own summary. The prompt now asks for this, but a note written from the model's memory and filed under an unused `summary_id` still passes. Add a check that uses no labels, for example the share of the note's content words that occur in its summary or clue. Run it on the new smoke packs before the full run, and report the share next to G0. If notes still come from memory, K measures the agent's own knowledge, not retrieved knowledge. That changes the CHER-style claim, though not the gates.

The verdict stays: **one BLOCKER (B2, Kshift clipping in `common.py:48-55`, unchanged in `adc4986`)**. B2 affects only the reader arms and G2. The agent run on uoa-lab2 can go ahead.
