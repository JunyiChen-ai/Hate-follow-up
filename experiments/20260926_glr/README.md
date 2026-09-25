# GLR: generative likelihood ratio of what is said (2026-09-26, pilot)

Hosts are written in the first line of each `runs/20260926_glr/<run>/run.log`. Code: this directory,
`src/mllm_judge.py` (`cached_logprobs`, `token_logprobs`, added today), `src/video_inputs.py` (ASR fix, today).
Evaluator: `src/eval/evaluate_four_datasets.py` (unchanged), reached through `experiments/20260922_til/til_infer.py`.

Process note: the user asked on 2026-09-26 to run this pilot directly, without the rule-4 proposal review and the
rule-6 code review agents. The plumbing checks of §5 replace the code review for this pilot.

## 1. Problem

Every label-free localizer so far (LAVAD, LELA, T3AL, Probe-VAD, NOVA, CSI-VAD, SPVL) asks a model a question
about each segment. For hate this measures the wrong thing: a hateful video talks about the same group
throughout, so a per-segment judge answers "is this about the target / part of a hateful video" rather than "is
hate expressed here". Evidence in this repo: mentioning the target group moves the SPVL window log-odds as much as
the window being GT-positive (count-weighted 1.15x HateMM, 1.47x HCS; `experiments/20260912_tad/README.md` §1);
about 60 % of GT-negative windows in HateMM videos with hate are judged violating
(`experiments/20260912_pwc/README.md` §7c); fitting every per-window feature with real labels does not beat the raw
read (TAD §5d). Nine variants of the judge (question, read-out, adaptation, post-processing) stayed at that level.

## 2. Mechanism

The frozen MLLM is used as a conditional language model. For each 8 s window with speech, compare how probable the
words actually spoken in that window are under two kinds of speaker:

- H1, a speaker who violates the hate policy (the nine `YOUTUBE_RULES` items paraphrased as one description);
- H0, a compliant speaker in three variants: discusses people or topics without hate; reports, quotes or
  criticizes others' hate without endorsing it; crude or profane without attacking a protected group.

Evidence of window i: `LLR_i = [log p(x_i | H1) - log mean_k p(x_i | H0_k)] / n_tokens(x_i)`.
A group name is about as probable under both kinds of speaker, so the topic cancels by construction; filler in a
hateful video is also about as probable under both, so there is no leak from the video's stance. Quotation, news
and condemnation are in H0, so they are explained by the compliant side.

The context is the 20 timestamped frames plus (variant `full`) the transcript of the earlier windows, or (variant
`none`) no transcript. The window's own words never appear before the point where they are scored: otherwise the
model copies them and both probabilities are close to 1.

Two framings are read (declared scan): `assistant` (the condition is an instruction, "In the part of the video
from a s to b s, the speaker <description>. Write exactly what is said in that part.", and the words are scored as
the assistant's reply) and `document` (the condition is a note on the window's transcript line inside the user
turn, "[a s-b s] (in this part the speaker <description>)", followed by the words). Reason for the second: the
chat model's safety training may suppress hateful words in its own replies under H1, which would shrink the ratio
for exactly the windows that matter; text inside the user turn is closer to plain language modelling.

Visual evidence is not covered (images have no likelihood here); the visual branch of SPVL-r2 and the duration
prior of TIL are kept for the frame-level composition.

Prior art seen in a quick search: sentence-level generative toxicity classification with GPT-2 (arXiv 2205.12390,
macro-F1 .54–.60); noisy-channel prompting for text classification (ACL 2022). Not found: use for temporal
localization, video context, or a compliant same-topic speaker as the counter-hypothesis.

## 3. Inputs and cost

- Reused caches: `data/frames_k20`, `data/asr_whisper_large_v3`, `data/omsl_v6_inputs/manifests/all_test.jsonl`.
- ASR fix (2026-09-26, `src/video_inputs.py` `load_asr`): the last Whisper chunk of 42 HateMM / 95 HCS records has
  no end (sometimes no start) timestamp and was dropped by every run since SPVL; it is now kept (missing start =
  previous end, missing end = audio duration). `fill_untimed=False` reproduces the old behaviour. The baseline is
  therefore re-measured: `runs/20260926_glr/base_gridA` = `experiments/20260922_til/til_measure.py` unchanged
  (SPVL-r2 grid A: verdict, stance, isolated visual and speech branches) on the fixed transcript.
- GLR cost per video: one prefix (20 frames, about 2.1k tokens), plus per speech window 16 teacher-forced
  branches (2 contexts x 2 framings x 4 conditions) of about 60–110 condition tokens + the window's words (capped
  at 256), plus one short extension per window. The pilot reads every combination; a final method would read one
  context and one framing (4 branches per speech window, the same order as SPVL-r2's branches).

## 4. Constants and decision rule (declared before any run)

| constant | value |
|---|---|
| model | Qwen/Qwen3-VL-8B-Instruct, bf16, fp32 lm_head over the full vocabulary, teacher forcing, seed 0 |
| frames | 20 uniform `k20` frames, timestamped, Qwen pixel cap 100352 |
| windows | fixed 8 s from 0 (identical to SPVL-r2 grid A); words = `window_text` of the fixed ASR, whitespace-collapsed |
| conditions | the four strings in `glr_measure.py` `CONDITIONS` |
| H0 mixture | uniform over the three H0 variants, at sequence level |
| contexts | `full` (earlier windows' lines `[a s-b s] words`), `none` (`(not shown)`) |
| framings | `assistant`, `document` (§2) |
| word cap | 256 tokens per window |
| window label (analysis only) | at least half of the window's 4 fps GT frames positive |
| group-word list (mechanism only) | `GROUP_WORDS` in `glr_analyze.py` (group names, no slurs) |

Primary comparison: speech windows of the test videos that have both window labels among their speech windows;
within-video window AUC; `LLR[full|assistant]` minus the SPVL-r2 speech branch `z_speech` (base run); paired
bootstrap over videos (4000, seed 0). **Pass**: the 95 % interval is above 0 on both corpora. The three other
variants are a declared scan; if only one of them passes, it is reported as development-selected. Mechanism
check: OLS of the within-video standardized score on GT and group-word mention; the prediction is a smaller
mention/GT ratio for the LLR than for `z_speech`.

Secondary (frame level, shared evaluator): the base run and derived runs in which `z_speech` is replaced by an LLR
variant, composed by `til_infer.py --model average --fusion max` with `--dwell 80` (SPVL-r2 + duration prior) and
`--dwell 0`; all three metrics on both corpora.

## 5. Plumbing checks (first video of every run, `verify.json`)

- token seam: prefix ids + earlier lines + condition + words equal the processor's tokenization of the whole
  string, for both framings (else the run stops);
- cache path vs one plain forward over the whole string: |difference of summed log-probability| per token < 0.25;
- in-place branch + `DynamicCache.crop` vs a deep-copied branch: |difference| < 0.01.

## 6. How to run

```
# lab (base, then GLR if the same machine), HateVLM env
setsid nohup bash experiments/20260926_glr/launch/run_base.sh > runs/20260926_glr/launch_base.out 2>&1 &
setsid nohup bash experiments/20260926_glr/launch/run_glr.sh  > runs/20260926_glr/launch_glr.out 2>&1 &
# campus: sbatch experiments/20260926_glr/launch/campus_glr.sbatch
# CPU, after both runs are in runs/20260926_glr/ on uoa-lab1
bash experiments/20260926_glr/launch/run_analysis.sh
```

## 7. Test-read log (rule 10)

- 2026-09-26, before this design: `runs/20260922_til/{gridA,infer/A1_gridA_prior80,ablation/*}/predictions.jsonl`,
  `data/gt_4fps*/*.npz`, `data/asr_whisper_large_v3/*/timestamped_chunks.jsonl`. Found: within-video standard error
  .028 / .021; the inverted HateMM videos are mostly GT fragments of all-hateful videos, visual-only hate and the
  ASR bug; 9 of the 12 worst HCS videos have no Hateful-category frames; the top-scored HateMM non-hate videos
  contain slurs in what read like songs, satire or historical clips. Changed: the ASR loader; the choice to test a
  generative read-out with a compliant same-topic counter-hypothesis.

## 8. Results

(filled after the runs)
