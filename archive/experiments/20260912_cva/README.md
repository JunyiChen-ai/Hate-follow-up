# CVA — counterfactual attribution of the video verdict to time intervals


**归档原因（2026-09-12）**：E0 六项指标全部低于 SPVL-r2，规则 9 无提升即归档。机制本身成立（打乱对照掉到 .508/.514，说明模型确实按区间响应），但反事实差分比直接问窗口弱；§2 声称的"抵消视频内常数项"对 within 指标是空操作，因为残差本来就是视频内秩。
Status: **2026-09-12 archived as a negative result** (rule 9: no metric improved on any corpus).
Rule-4 review 放行, rule-6 code review clean. Development-selected numbers only (rule 10).

The reviewer verified criterion 1 by searching the literature: leave-one-out / occlusion attribution
appears as LLM text-context attribution (AttriBoT 2411.15102), as post-hoc explanation of trained
classifiers (occlusion sensitivity 2207.12859, meme modality ablation 2410.13488) and as a training-time
erasure regulariser in weakly supervised video anomaly detection (DEN 2312.01764), but not as an
inference-time label-free localization score. In hateful video specifically: LELA (2602.09637) scores
segments absolutely (`max` over per-modality caption scores, no erasure), MultiHateLoc (2512.10408) is
MIL with video-level labels, TANDEM (2601.11178) fine-tunes LoRA. Reviewer also flagged six factual
errors in this README; all six are corrected in the text below.

## 1. What the 2026-09-12 measurements leave standing

Five candidates were archived today (PWC, TAD, BND, SDL, NGA). All five kept three things fixed and
changed something downstream of them:

1. the frozen Qwen3-VL-8B;
2. the decision unit is a fixed 8-second grid cell;
3. every per-window read is an **absolute** judgement made on one shared global prefix.

Two measurements say the problem is in (3), not in the model:

**(a) The per-window absolute read is dominated by a video-constant term.** `experiments/20260912_pwc/`
§7c: mentioning the target group moves the window log-odds by +8.1 (HateMM) / +7.0 (HCS), while the
window actually being GT-positive moves it by +4.3 on both (main effects, averaged over the other
factor; count-weighted the nuisance/signal ratio is 1.15x on HateMM and 1.47x on HCS). That nuisance term is near-constant inside a
video, because a hateful video talks about the same group throughout. `experiments/20260912_tad/` tried
to remove it by subtracting a *different* read (an auxiliary topic question) and made things
monotonically worse, because the topic read also carries signal.

**(b) The visual branch is not reading the window.** Measured today
(`diagnose_frame_coverage.py`, output in `runs/20260912_cva/diag/frame_coverage.txt`, on the STATUS-authoritative
run `runs/20260910_spvl/full2_dual_evid_stance/`): the shared prefix holds 20 uniformly sampled frames,
so at an 8-second grid **24.2 % (HateMM) / 34.1 % (HCS) of windows contain no frame at all**. Paired
within the same videos, the visual branch orders *better* on the windows where it has no frame:

| | HateMM (23 videos) | HCS (77 videos) |
|---|---|---|
| visual branch, windows containing a frame | .5663 | .5417 |
| visual branch, windows containing no frame | **.7251** (+.159) | **.5622** (+.020) |

Its median within-video Spearman with the speech branch is only +.111 / +.176, so on frameless windows
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

SPVL-r2 costs 1 prefix forward + up to 2N branch reads (its stance turn is a cache extension, not a
scored read). CVA costs 1 prefix forward + N + 1 branch reads: 3983 reads on HateMM and 3709 on HCS,
about half of SPVL-r2's. **Cheaper than the current method**, on the same cache, with no new inputs, no
new encoder and no new extraction. Median N is 15 (HateMM) / 30 (HCS).
`runs/20260910_spvl/full2_dual_evid_stance/run.log` finished all 333 videos of both corpora in 523 s on
one 5090, so the estimate here is roughly 5 minutes per corpus, not tens of minutes.

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
**Run 2026-09-12: passes on both SPVL-r2 runs.** `metrics_paritychk.json` is byte-for-byte the same
metrics as the run's own `metrics_izv_plus_mean_rrank.json`:

| run | HateMM | HCS |
|---|---|---|
| `full2_dual_evid_stance/` (the STATUS row, **the comparison baseline**) | .8919 / .6831 / .6976 | .7119 / .6664 / .6001 |
| `mllm/q3vl-8b/full/` (same method, separate run) | .8920 / .6825 / .6968 | .7132 / .6675 / .6020 |

The two runs differ by at most .0019 pooled and .0019 within, inside the noise floor. The
`.8938 / .6863 / .6783` row in `experiments/20260910_spvl/README.md` §7 is an earlier configuration and
is not a baseline.

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


---

## 9. Result (`runs/20260912_cva/e0/`, one run, both corpora, 333 videos, 0 errors)

`metrics_izv_plus_mean_rrank.json`, pooled ROC / pooled PR / within (n videos with both classes):

| | HateMM | HateClipSeg |
|---|---|---|
| SPVL-r2 (baseline) | .8919 / .6831 / **.6976** (84) | .7119 / .6664 / **.6001** (99) |
| CVA | .8834 / .6572 / .6747 (84) | .6724 / .6215 / .5779 (99) |
| delta | −.0085 / −.0259 / −.0229 | −.0395 / −.0449 / −.0222 |

All six numbers below baseline. No metric improves on either corpus, so rule 9 archives this without a
second round.

Runtime: HateMM 215 videos in 186 s, HCS 118 in 173 s, one 5090 (`uoa-lab3`) — as predicted in §3, and
about half of SPVL-r2's read count. The cost claim held; the accuracy claim did not.

## 10. What the controls say

Window-level within AUC on the same windows (74 HateMM / 96 HCS videos with both window classes):

| curve | HateMM | HCS |
|---|---|---|
| SPVL-r2's direct window read `z` | **.7563** | **.6190** |
| CVA `s = z_video − z_excl` | .7183 | .5908 |
| `−z_excl` alone (the `--score raw` control) | .7183 | .5908 |
| `s` shuffled within each video (seed 0) | .5079 | .5143 |

**(a) The model does honour the exclusion instruction.** Shuffling `s` inside a video drops it to chance,
so the read genuinely depends on *which* interval was excluded. The mechanism is not fictional: it
produces a real, interval-specific localization signal at .718 / .591. It is simply weaker than asking
about the window directly.

**(b) The cancellation argument in §2 is void for the within metric, and the control proves it.**
`rank(s) == rank(−z_excl)` on **100 %** of videos, so the two rows above are identical by construction:
inside one video `z_video` is a constant, and the residual is a *within-video centred rank*, which
already removes any within-video constant. Subtracting a video-constant term therefore cannot change the
within-video ordering at all. The nuisance that §2 promised to cancel was already cancelled by the rank
residual before CVA was written.

This is the substantive error in the proposal. TAD's subtraction was not equivalent to this: it
subtracted a *per-window varying* topic read, which does move the ranks (and moved them the wrong way).
A cancellation argument can only help here if the term being removed varies inside the video.

**(c) The interval-specific part of the read is small.** `s` has mean +3.57 (HateMM) / +3.98 (HCS) — the
exclusion clause costs about 4 log-odds no matter which interval it names — while the within-video
standard deviation of `s` is only 0.70 / 0.51, against 6.20 / 5.69 for SPVL-r2's direct window read. The
clause's own effect is roughly six times the interval's.

## 11. Where this leaves the question

CVA answers a question the earlier archived rounds did not: *is "would the verdict survive without
interval i" a better per-window quantity than "is interval i violating"?* Measured: no, .718 / .591
against .756 / .619. Asking the model about the whole video and differencing loses more than the
absolute per-window read loses to the topic confound.

The §1(b) diagnosis that motivated this — 24.2 % / 34.1 % of windows contain no prefix frame, and the
visual branch orders better on exactly those windows — is **not** addressed by CVA and remains open. CVA
changed what is asked; it did not change the decision unit, which is still the same fixed 8-second grid.

## 12. Code review notes carried forward (rule 6, non-blocking)

- The declared exclusion string in §4 uses an em dash and omits the trailing `Answer "Yes" or "No".`;
  `cva.py:47` uses `--` and appends the answer instruction. The run used the code's string.
- §5's `--score raw` arm has no `--score` flag in `cva.py`; it was computed post hoc from the persisted
  `extra.windows[].z_excl`, which is exact (see §10). `run_e0.sh` references a `run_keeponly.sh` that was
  never written — the keeponly arm was not run, because rule 9 archives the candidate on the E0 result.
- Error rows are written with `"method": "cva"` while good rows use `"method": "cva_exclude"`. There were
  0 errors, so this never triggered.
