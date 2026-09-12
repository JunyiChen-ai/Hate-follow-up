# CVA — counterfactual attribution of the video verdict to time intervals

Status: **2026-09-12 proposal.** Not yet reviewed, not yet run.

## 1. What the 2026-09-12 measurements leave standing

Five candidates were archived today (PWC, TAD, BND, SDL, NGA). All five kept three things fixed and
changed something downstream of them:

1. the frozen Qwen3-VL-8B;
2. the decision unit is a fixed 8-second grid cell;
3. every per-window read is an **absolute** judgement made on one shared global prefix.

Two measurements say the problem is in (3), not in the model:

**(a) The per-window absolute read is dominated by a video-constant term.** `experiments/20260912_pwc/`
§7c: mentioning the target group moves the window log-odds by +8.1 (HateMM) / +7.0 (HCS), while the
window actually being GT-positive moves it by +6.3 / +4.7. That nuisance term is near-constant inside a
video, because a hateful video talks about the same group throughout. `experiments/20260912_tad/` tried
to remove it by subtracting a *different* read (an auxiliary topic question) and made things
monotonically worse, because the topic read also carries signal.

**(b) The visual branch is not reading the window.** Measured today
(`diagnose_frame_coverage.py`, `runs/20260912_cva/diag/`): the shared prefix holds 20 uniformly sampled
frames, so at an 8-second grid **24.2 % (HateMM) / 34.1 % (HCS) of windows contain no frame at all**.
Paired within the same videos, the visual branch orders *better* on the windows where it has no frame:

| | HateMM (23 videos) | HCS (77 videos) |
|---|---|---|
| visual branch, windows containing a frame | .5710 | .5394 |
| visual branch, windows containing no frame | **.7338** (+.163) | **.5664** (+.027) |

Its median within-video Spearman with the speech branch is only +.154 / +.184, so on frameless windows
it is not simply re-reading the speech branch — it is producing a judgement from the global context.
When a frame *is* present the model looks at it, and a single uniformly sampled still from a hateful
video is usually unremarkable, so the local evidence pushes the answer the wrong way.

Together: the absolute per-window read is mostly a global prior plus a topic term, and the one channel
that is supposed to be local is worse when it is actually local.

## 2. Mechanism

Do not ask the model to judge a window. Ask it to judge **the video** twice, and attribute the
difference to the interval.

```
z_video      = log-odds of "does this video violate the rules"           (1 read, already computed)
z_excl(i)    = the same question, with the model instructed to disregard interval i
s_i          = z_video - z_excl(i)
```

`s_i` is how much interval *i* contributes to the model's own video-level verdict. The video-constant
nuisance — the target group, the channel, the speaker, the genre — is present in **both** terms of the
difference and cancels by construction, because both are answers to the *same question* on the *same*
context. This is the thing TAD could not do: it subtracted a different question, so nothing cancelled.

It also sidesteps (b): no read is asked to judge an 8-second window on its own evidence. Every read
still sees the whole video; only what it is told to disregard changes.

**Why this is specific to hateful video.** Hatefulness is a property of a claim about a group, asserted
across a video whose topic never changes. The quantity a localizer needs is therefore not "is this
segment hateful in isolation" — most segments of a hateful video look and sound like the rest of it —
but "does the verdict survive without this segment". That is a counterfactual question, and it is
exactly the one the absolute-read paradigm cannot ask.

**Source.** Leave-one-out / occlusion attribution is standard in interpretability. It has not been used
as a label-free localization method: LAVAD and T3AL score segments absolutely, MultiHateLoc (2512.10408)
uses MIL with video-level labels, TANDEM (2601.11178) fine-tunes LoRA on an MLLM. Neither the attribution
framing nor the cancellation argument appears in hateful video localization. Rule 4 review will verify.

## 3. What is computed, and cost

Per video, on the **same** cached prefix SPVL-r2 already builds (20 frames + full transcript + the rules):

| read | count | note |
|---|---|---|
| whole-video verdict `z_video` | 1 | identical to SPVL-r2 |
| exclusion read `z_excl(i)` | N | one per 8-second window, on the cached prefix |

SPVL-r2 costs 1 + 1 (stance) + up to 2N branch reads. CVA costs 1 + N. **Cheaper than the current
method**, on the same cache, with no new inputs, no new encoder and no new extraction. Median N is 15
(HateMM) / 30 (HCS). Estimated 40 min (HateMM) / 70 min (HCS) on one 5090, from the SPVL-r2 timings.

No new cache. Reuses `data/frames_k20/`, `data/asr_whisper_large_v3/`.

## 4. Constants, declared before running (rule 7)

Identical for both corpora (rule 13).

| constant | value |
|---|---|
| model | `Qwen/Qwen3-VL-8B-Instruct`, greedy, seed 0 |
| prefix | unchanged from SPVL-r2: 20 timestamped `k20` frames + full Whisper transcript + the platform rules |
| window length | 8.0 s, fixed grid, `fixed_windows(duration, 8.0)` |
| `z_video` question | unchanged from SPVL-r2 `VIDEO_QUESTION` |
| exclusion question | `"Now disregard everything between {t1:.1f}s and {t2:.1f}s of this video — both what is shown and what is said in that interval — and judge only the remaining parts. Considering only those remaining parts, does this video violate any of the above rules?"` |
| stance turn | removed (there is no per-window branch to condition) |
| score | `s_i = z_video - z_excl(i)` |
| frame score | unchanged compose (`cva_compose.py`, a byte copy of `experiments/20260910_spvl/compose.py`): `intercept + centred rank of s over frames`, intercept = `z_video + mean_i s_i` (`--intercept zv_plus_mean --residual rank`, the promoted SPVL-r2 setting) |
| noise floor | pooled .005, within .01 (rule 7) |

**No constant scan is planned.** If one becomes necessary the grid goes into this README before it runs.

## 5. Declared controls (run in the same pass, rule 14g)

| arm | extra GPU cost | what it tests |
|---|---|---|
| shuffled | none (analysis-time permutation of the same reads) | interval *i* is scored with a *different* interval's exclusion read, under a fixed permutation (seed 0). If the ordering survives, the reads do not depend on which interval was excluded and the mechanism is fictional. |
| `--score raw` | none | uses `-z_excl(i)` directly instead of the difference. Isolates whether the **cancellation** is what helps, or just the reworded question. This is the ablation that carries the novelty claim (rule 14g). |
| `--read keeponly` | N reads, separate run | "disregard everything **except** interval *i*". The positive counterpart: a direct read of the interval in isolation, with no cancellation. Tests the same interval from the other side. |

**Parity check before any arm is believed** (this is what caught the resampling bug in
`experiments/20260912_tad/`): `cva_compose.py` is run on `runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl`
and must reproduce SPVL-r2's authoritative row exactly.
**Run 2026-09-12: passes.** `metrics_paritychk.json` is identical to
`runs/20260910_spvl/mllm/q3vl-8b/full/metrics_izv_plus_mean_rrank.json` —
HateMM `.8920 / .6825 / .6968` (n=84), HCS `.7132 / .6675 / .6020` (n=99). Those are the baseline
numbers CVA is compared against; the `.8938 / .6863 / .6783` row in
`experiments/20260910_spvl/README.md` §7 is an earlier run and is not the comparison baseline.

## 6. Predicted outcome and the failure signal

pooled ROC / PR come almost entirely from `z_video`, which is unchanged, so both are predicted to match
SPVL-r2 within the noise floor. The target is **within-video macro ROC-AUC**; the gate (rule 8) needs
≥ .01 on both corpora with no other metric dropping more than the noise floor.

Clean failure signal: if the model does not honour the exclusion instruction, `z_excl(i) ≈ z_video` for
all *i*, `s` is constant, its centred rank is 0, and within collapses to .5. That is unambiguous and
distinguishes "the mechanism is wrong" from "the mechanism is right but weak".

## 7. How to run

```
python experiments/20260912_cva/cva.py --datasets HateMM      --run-name e0_hatemm
python experiments/20260912_cva/cva.py --datasets HateClipSeg --run-name e0_hcs
python src/eval/evaluate_four_datasets.py --pred runs/20260912_cva/<run>/predictions.jsonl
```

## 8. Diagnosis that motivated this (rule 10 log)

Read: `runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl` (per-window `z`, `z_visual`, `z_speech`),
`data/gt_4fps/{HateMM,HateClipSeg}.npz`, `data/frames_k20/` filenames for frame timestamps.
Found: §1(b) above — frame coverage of the 8-second grid, and the paired framed/frameless split of the
visual branch. Changed: the decision moved from an absolute per-window read to a counterfactual
difference of two video-level reads. Numbers in §1(b) are development-selected.
