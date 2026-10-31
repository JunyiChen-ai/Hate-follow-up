# Free-form per-video detection plan in the adopted reader (declared 2026-11-01, before any run)

Development host uoa-lab1 (sc474397). Every number here is development-selected (rule 10). No label is read on any
scoring path; `diagnose.py` reads video-level labels post hoc only.

## 1. Why

PANDA (NeurIPS 2025, video anomaly detection) looks at the whole video first, writes a scene description and a
detection plan, then checks each clip with that plan in its prompt. Our earlier hypothesis step (HVL,
`experiments/20260911_hvl/README.md` §7–§8) forced a fixed format (`TARGET / FORM / EVIDENCE(start-end: note)`); it
lost to the base, its cited spans localized near chance, and a permuted (another video's) hypothesis scored as well as
the model's own, so its content was not used. The user's point: the fixed fields removed the freedom of a plan that
differs from video to video. This experiment tests whether a **free-form** plan, written by the same reader in its own
words after the whole-video verdict, improves localization when every window is then read with the plan in the shared
context. Candidate 51 (`archive/experiments/20261026_knowledge_verdict/README.md`) is the warning: notes inserted into
the prefix lifted the window reads of non-hateful videos that mention hate topics.

Mechanism hypothesis: a plan that names what could be hateful in this particular video and what to look for gives
each isolated window branch video-specific criteria that the generic rules and the one-bit verdict do not carry;
if the content is used, the model's own plan should beat another video's plan, and the gain should sit in the
within-video order (the plan is the same for every window of a video, so it can only change window reads relative to
each other).

## 2. Method (`read_plan.py`; shared code `src/stance_cache.py` `generate_turn` / `append_turn`, `src/window_evidence.py`)

Per video, Qwen3-VL-8B-Instruct frozen, one shared KV cache:

1. The adopted shared context: r6 prefix (20 uniform frames with timestamps, timestamped Whisper transcript, the 9
   rules, the reader instruction) and the whole-video verdict turn (`src/stance_cache.build`; z_video, stance written as
   the model's own answer). Unchanged.
2. **Plan turn (new).** One user question, `PLAN_QUESTION`:

   > This video will now be checked moment by moment. Before that, write a short plan in your own words for checking
   > this video: what in this video could violate the rules above, or why nothing in it does, and what to look and
   > listen for in each moment. Keep it to one short paragraph of plain prose, about 100 words.

   (The last sentence is the one prompt fix allowed before the full run; see the rule-10 log, §6.)

   Greedy decoding, at most 256 new tokens, stopped at end-of-turn. No fields, no format, no request for timestamps.
   The generated text is written back onto the shared context as the model's own assistant turn (the cache is cropped
   to the verdict turn and the plan is re-tokenised through the chat template, the same path for any source text, so
   `own_plan` and `other_plan` differ only in the plan text). One extra generation per video; no extra calls per window.
3. **Window reads.** Exactly the adopted reader (`experiments/20261031_window_evidence/`): 8 s windows, each an isolated
   branch of the shared cache (cropped back after every read, no memory of earlier window answers); visual question
   with the window's 0.5 fps frames placed right before it (`frames_margin`), native visual question when the window
   has no cached frame; speech question with the forced-aligned word slice (`aligned_slice`, proportional slice for
   videos without an alignment), skipped when the slice is empty. Same questions, frames and word slices as the
   adopted run; the only difference is the plan turn in the shared context before them.
4. Decoder and evaluation unchanged: `experiments/20260926_twolevel/twolevel_r2.py` with the adopted settings
   (`--noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`,
   EM fitted on the whole unlabeled corpus), then the single evaluator (`src/eval/evaluate_four_datasets.py`, called by
   the decoder). `decode_eval.sh` runs it for every arm.

### Arms

| arm | plan in the shared context | reads |
|---|---|---|
| `base` | none | the stored adopted reads `runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl` (not re-run; `--smoke` re-reads 2 + 2 videos with the plan off and checks every value against them within 1e-3) |
| `own_plan` | the video's own plan | new |
| `other_plan` | the plan of another video of the same corpus (fixed derangement of the sorted corpus video ids, `numpy.random.RandomState(0)`, rejection sampling until no fixed point; ids only, no labels) | new |

### Constants (both corpora identical)

20 prefix frames, 8 s windows, 0.5 fps in-window frames, pixel caps as the adopted reader, `PLAN_QUESTION` above,
256 max plan tokens, greedy decoding, seed 0, derangement seed 0, decoder settings above. Nothing is tuned on test.

### Gate (declared before running)

`own_plan` beats `base` on within-video macro ROC-AUC on both HateMM and HateClipSeg (by more than the .01 noise
floor, rule 7), AND `own_plan` beats `other_plan` on within-video on both corpora. Pooled ROC-AUC and pooled PR-AUC
are reported for every arm. The `other_plan` control separates "the plan content is used" from "any extra turn in
the context changes the reads" (the HVL permutation control and candidate 51's withheld-content control both showed
the latter).

### Diagnostics (post hoc, `diagnose.py`; the only label use)

Plan length (tokens, words; share hitting the 256 cap); share of plans that mention times anyway (regex); share of
plans on non-hateful videos that still name a target group (keyword regex, listed in the script); mean change of the
window reads (own_plan minus base; other_plan minus base) on non-hateful vs hateful videos, visual and speech; mean
within-video Spearman between own and base visual reads; four example plans (2 hateful, 2 non-hateful).

## 3. Inputs and cost

Caches reused: `data/frames_k20`, `data/asr_whisper_large_v3`, `data/temporal_retrieved_kv` (0.5 fps frames),
`data/asr_whisper_aligned` (forced-aligned words). No new preprocessing. Per video: prefix forward + verdict (as
adopted) + one greedy generation of ≤ 256 tokens + the adopted window reads, for the own arm; a second prefix forward
+ plan write + window reads for the other arm. Expected about 10–15 s per video per corpus on a 5090 for both arms
(the adopted read alone was 2.9 s per video).

## 4. How to run

- lab1 (`local-sc474397`): `SCOPE=smoke sbatch experiments/20261101_video_plan/launch/lab1.sbatch`, then
  `SCOPE=HateMM sbatch experiments/20261101_video_plan/launch/lab1.sbatch`.
- lab-server (`local-sc448960`, account junyi; 0.5 fps frames and aligned words rsynced from lab1 first):
  `SCOPE=HateClipSeg sbatch experiments/20261101_video_plan/launch/lab_server.sbatch`, then rsync
  `runs/20261101_video_plan/HateClipSeg/` back to lab1.
- Decode and evaluate (CPU, lab1): `bash experiments/20261101_video_plan/decode_eval.sh` → `runs/20261101_video_plan/decoded/<arm>/metrics.json`.
- Diagnostics: `python experiments/20261101_video_plan/diagnose.py` → `runs/20261101_video_plan/diagnostics.json`.

## 5. Results

(filled in after the runs)

## 6. Rule-10 log (test files read, what was found, what changed)

- 2026-11-01 smoke 1 (lab1 Slurm 1307, `runs/20261101_video_plan/smoke_prompt_v1/`, 2 HateMM + 2 HateClipSeg videos):
  plan-off reads equal the stored adopted reads on every value (max |Δ| 0.0 for z_video, visual, speech). Plans with
  the first prompt (without the last sentence): 4 of 4 hit the 256-token cap and stop mid-sentence; all four are
  markdown with headings and numbered sections; in 3 of 4 the "what to look for in each moment" part is cut off
  after its heading; one (hate_video_1) writes its own `t=..s` ranges. No test label or metric was read. Change: one
  sentence added to the prompt asking for one short paragraph of plain prose of about 100 words (the cap stays 256).
  Smoke 2 re-run with the new prompt before the full runs.
