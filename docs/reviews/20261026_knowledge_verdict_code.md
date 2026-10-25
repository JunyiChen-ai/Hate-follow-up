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
