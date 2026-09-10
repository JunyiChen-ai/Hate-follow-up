# SPVL — single-pass verdict-and-evidence localization (label-free)

Status: 2026-09-10 promoted to current method (SPVL-r2, §10) after rounds 1–2; round-3 ablations complete (uoa-lab3). Development-selected numbers only
(test split, 4 fps, `RESEARCH_ITERATION_RULES.md` rule 10); every test-read is logged in §7.

## 1. Mechanism problems this experiment targets

Diagnosis from the 2026-09-09 ablations of OMSL-v6 (`experiments/20260829_omsl_v6/README.md`):
pooled ROC/PR come entirely from the whole-video Qwen3-VL-8B log-odds z; within-video order comes
from the three-stream rank fusion and is weak (HateMM .649, HateClipSeg .547, chance .5).

| problem (what the current computation lacks) | predicted failure | module |
|---|---|---|
| **A. Statement without target.** The language stream scores each Whisper segment with only that segment's text (`run_unified_chunk_scorer.py`, 2026-08). Hate is a relation between a statement and a target; the target is often named outside the segment or referred to by a pronoun. | segments whose target is not inside the segment score low; explicit slurs score fine | **M1 context-shared isolated scoring**: one shared prefix (rules + timestamped frames + full timestamped transcript); N window queries that attend to the prefix and to themselves only |
| **B. Localization never sees the picture.** Within-video visual evidence is produced by Vid-Group, a Charades-trained moment retriever given a hate-definition query it was never trained on (piecewise-constant curve, half the frames exactly 0). The MLLM only sees frames in the whole-video call. | visually carried hate (symbols, gestures, on-screen text, violent imagery) is not localized | M1's prefix contains the frames; **M2 fixed time windows covering the whole video** (Whisper segments are up to 146 s long and leave silent parts unscored) |
| **C. Verdict and evidence never correct each other.** z is weak on HateClipSeg (.67) and the old run gave the HCS judge no transcript at all. | a mis-ranked video cannot be rescued by correct evidence | **M3 evidence feedback** (only if E4 passes and HCS pooled < .70): z' = z + g(window statistics), same forward, no extra call |

Kept from OMSL-v6 (mechanism verified): video intercept + zero-mean rank residual (cross-video
order delegated to the verdict, within-video order to the evidence ranks). Removed (ablation-inert):
Möbius interactions, permutation calibration, occupancy intercept. Vid-Group and ImageBind stay in
the ablation table; they are dropped from the method only if the ablation shows they add nothing.

Why not "ask the model for the target group and condition on it": two preregistered 2026-08 tests
showed that any second verbal question to this judge returns z renamed (Spearman .92–.98,
`archive/detection-2026-08/docs/duplex/{DUAL_AXIS_KILLTEST_NOTE,SPEECHACT_SECOND_CALL_NOTE}.md`).
M1 therefore injects context as input, never as the model's own verbalized hypothesis.

## 2. Method

One Qwen3-VL-8B forward per video (`spvl.py`):

```
PREFIX  : <system> + user[ intro, ([t=..s] frame)×K, "Transcript:" timestamped Whisper segments,
          YOUTUBE_RULES, reading instruction (prag) ]                      (closed with <|im_end|>)
BRANCH_0: user[ "Does this video contain content that violates any of the above rules? Answer Yes or No." ] + assistant header  -> z_video
BRANCH_i: user[ "Consider only window i of N, from t1 s to t2 s. Transcript in this window: ...
          Does THIS window contain content that violates any of the above rules? Answer Yes or No." ] + assistant header -> z_i
```

Attention mask: prefix causal; branch tokens see the whole prefix and their own past, never another
branch (block-diagonal). Positions: the model's own mrope index for the prefix; every branch restarts
at max(prefix position)+1. Each branch is therefore exactly one independent three-turn call; the
packing only removes the K× re-encoding of the frames. Read-out: z = logsumexp(Yes ids) − logsumexp(No ids)
at the last position of each branch (same token sets as the 2026-08 judge).

Windows (M2): fixed S = 8 s, N = ceil(duration / 8); a window's transcript is the part of the
Whisper segments that overlaps it (segments crossing a boundary are sliced at word boundaries in
proportion to time). Windows without speech are still scored with "(no speech)".

Composition (`compose.py`): final(t) = z_video + centred-rank residual of the window curve inside the
video (frame t takes the z of its window). No smoothing, no thresholds, no learned parameters.

Prompt parity with the cached 2026-08 inputs: same system message, same `YOUTUBE_RULES`, same
reading instruction, same Yes/No token sets, same 100352-pixel frame cap. Differences are only the
timestamp tags, the Whisper transcript in the prefix, and the window questions. The legacy per-chunk
prompt is reproduced as a separate arm (`--legacy-chunk-arm`) to confirm the pipeline reproduces the
cached log-odds before any comparison is made.

## 3. Inputs and cost

| input | source | provenance |
|---|---|---|
| 20 frames / video, native resolution | `data/frames_k20/<ds>/<id>/` | `data/frames_k20/PROVENANCE.md` (ffmpeg, uoa-lab1, 2026-09-10) |
| Whisper large-v3 segments with timestamps | `data/asr_whisper_large_v3/<ds>/timestamped_chunks.jsonl` | copied from 2026-08 runs, see PROVENANCE there |
| Qwen/Qwen3-VL-8B-Instruct, bf16, greedy (log-odds read-out, no generation) | HF cache | — |

Cost per new video: Whisper once (already cached), 20 ffmpeg seeks, **one MLLM forward** of about
20×91 image tokens + transcript (≈ 0.3–1.5k) + N×~60 tokens (N ≈ 8–30). The current OMSL-v6 pipeline
uses one multimodal call plus 11–15 text calls per video and three extra encoders (CLIP-L/14 features,
Vid-Group, ImageBind). Sequences above `--max-tokens 9000` (proposal said 12000; §4 has the value actually run) are split into branch groups, each with the
full prefix. Estimated 3–6 s per video on one RTX 5090; 643 videos < 1 h.

## 4. Constants (declared before any run)

K = 20 frames (uniform, t_k = (k+0.5)·D/20); S = 8 s windows; pixel cap 100352 / min 65536; rules =
`YOUTUBE_RULES` for both corpora; reading instruction = `prag`; Yes/No token sets = first tokens of
{Yes, " Yes", yes, " yes", YES, " YES"} and the No analogues; uncovered frames in ASR-window mode =
−12 (as in OMSL-v6 `text_curve`); seed 0; `--max-tokens 9000` (branch groups above it, each with the full prefix); additive bf16 block mask via sdpa (memory-efficient kernel). Both corpora
use exactly the same constants and prompt (rule 13).

Ablation switches: `--frames K` (0 = no frames), `--no-transcript-context`, `--windows {fixed,asr}`,
`--window-seconds S`, `--mask {block,causal}`, `compose.py --intercept {spvl,legacy,none}
--residual {rank,none} --visual-primary`.

## 5. Plumbing checks (must pass before any number is read)

Run on the first video of every run and written to `verify.json`; findings from
`plumbing_debug.py` on lab3 (2026-09-10, hate_video_1, T = 2372–3168):
- position ids: my prefix-from-`get_rope_index` + per-branch restart equals the model's own
  `compute_3d_position_ids` exactly (max diff 0);
- with the model's own causal path (no explicit mask) my positions reproduce the default logits
  exactly (max |Δlogit| 0);
- any explicit 4D mask changes the sdpa kernel (flash → math or memory-efficient) and moves
  bf16 logits (per-run `verify.json`: causal-4D vs no-mask max |Δlogit| 0.8–5.1 over the
  vocabulary, |Δz| for the video question 0.25–0.41, five windows packed vs plain max |Δz|
  0.16–0.75, Spearman .97–1.0); under the *same* kernel the all-causal 4D mask equals no-mask
  exactly (0.0000) and packed-block vs independent calls differ by ≤ 0.24 log-odds with the
  fp32 read-out. Packing is therefore exact up to bf16 kernel rounding, which affects every
  arm (sequential calls included) equally; |Δz| of this size can swap adjacent windows whose
  scores are within ~0.5 of each other, which is the precision floor of bf16 inference;
- Yes/No log-odds are read with the lm_head applied in fp32 to the kept hidden states;
- peak memory 18.1 GB at T = 3168 (bf16 weights 17.5 GB); `--max-tokens 9000`, sequences above it
  are split into branch groups with the full prefix repeated.
Per-run `verify.json` records |Δz| for BRANCH_0 packed vs plain, causal-4D vs no-mask, five
windows packed vs plain (max |Δz| and Spearman), and peak memory.

## 6. Plan and gates

E3 pilot on the within-defined subset (HateMM 84, HateClipSeg 99 videos; selection uses the GT only to
pick videos with both classes, never to score): arms a0 (legacy chunk replica), a (new prompt, no
frames, no context, ASR windows), b (+context), d (+context, fixed windows, no frames),
c (+context, fixed windows, 20 frames). Go to E4 if b or c beats OMSL-v6 within by ≥ .01 on both
corpora without R1 (window scores collapsing onto z_video) or R2 (frames adding nothing).
E4: full 643-video run, ablation table, rule-8 gates (comparison gate vs T3AL; promotion gate vs
OMSL-v6: no metric drops beyond noise pooled .005 / within .01, at least one metric +.01 on both
corpora), rule-14g per-module ablation ≥ .01.

## 7. Test-read log (rule 10)

- 2026-09-10: `data/gt_4fps/{HateMM,HateClipSeg}.npz` read only to list videos whose grid has both
  classes (84 / 99) for the pilot subset. No scores were inspected.
- 2026-09-10 (`analyze_intercept.py`, after E4): read test GT to compute video-level ROC/AP and the
  positive-frame fraction per video for `runs/20260910_spvl/full`. Finding: on HateClipSeg z_video ranks
  hateful videos better than the old judge (video ROC .804 vs .685) but its top-ranked videos have a
  lower positive-frame fraction (top-10: 69 % vs 77 %), which is what pooled PR measures. Design
  decision: M3 intercept = z_video + mean window z (extent from the same forward). Also read
  `~/data/HateClipSeg/annotation(new).json` labels for the top-12 videos (all Offensive, 11/12 strict
  Hateful) to confirm the over-ranking is about extent, not wrong verdicts.

## 7b. Pilot (E3, 2026-09-10, uoa-lab3, within-defined subset HateMM 84 / HateClipSeg 99)

Source: `runs/20260910_spvl/pilot_<arm>/metrics_ispvl_rrank.json` (a0: `metrics_ilegacy_rrank.json`), reference
`runs/20260910_spvl/reference_subset/metrics_v6_*.json` (OMSL-v6 restricted to the same videos; within is
identical to the full-set value by construction). Pooled numbers on this subset are not comparable to
full-set numbers and are not read.

| arm | frames | context | windows | HateMM within | HateClipSeg within |
|---|---|---|---|---|---|
| OMSL-v6 full (reference) | Vid-Group | — | — | .6494 | .5473 |
| a0 legacy chunk replica (batched, right padding — see note) | 0 | no | ASR | .5831 | .5084 |
| a | 0 | no | ASR | .5602 | .5132 |
| b | 0 | yes | ASR | .5691 | .5241 |
| d | 0 | yes | fixed 8 s | .6500 | .5701 |
| c | 20 | yes | fixed 8 s | **.6783** | **.5806** |

Reading: context alone adds +.009 / +.011 (b vs a); fixed windows covering the whole video add
+.081 / +.046 (d vs b); frames add +.028 / +.011 (c vs d). Arm c beats OMSL-v6 by +.029 / +.034, both
above the +.01 gate. Risk checks (`diagnose.json`): window scores are not copies of the verdict
(0–1 videos with a constant window curve, median within-video std of window z ≈ 5.5); frames change
the result, so R2 is not triggered. Spearman(z_video, 2026-08 z) = .82 HateMM / .52 HateClipSeg
(the old HateClipSeg judge saw no transcript).
Note on a0: the batched legacy replica used the tokenizer's default (right) padding, so shorter prompts
were read at a pad position; median per-video Spearman against the cached chunk log-odds was only
.46 / .26. Fixed to left padding for the full-set parity run.

## 7c. Code review (rule 6, 2026-09-10, independent agent)

No blocking bug. Fixed before reading E4 numbers: (1) resume logic re-scores rows that carry
`error` (previously they were treated as done and the evaluator would silently drop them);
(2) legacy replica now passes the raw Whisper text (no `.strip()`) like the 2026-08 scorer.
Noted: prediction grid is ceil(duration·4) from the manifest, one frame longer than the GT grid
for 211/215 and 114/118 videos; the evaluator truncates to the common length, indices align from
t = 0. README constants updated to the values actually run (additive mask, max-tokens 9000).

## 8. Results (E4, 2026-09-10, uoa-lab3; test split, 4 fps; development-selected)

Source files: `runs/20260910_spvl/<run>/metrics_<compose>.json`; table generated by `summarize.py`
(`runs/20260910_spvl/ablation_table.md`). Cells: pooled ROC / pooled PR / within (n videos).

| variant | HateMM | HateClipSeg |
|---|---|---|
| OMSL-v6 (current method) | .8507 / .5781 / .6494 | .6692 / .6622 / .5473 |
| T3AL rerun (label-free comparator) | .6091 / .3096 / .5068 | .6246 / .5645 / .5003 |
| MultiHateLoc rerun (weakly supervised) | .7535 / .4880 / .6070 | .5062 / .4925 / .5128 |
| **SPVL** (K=20, context, fixed 8 s, block mask; intercept z_video) | .8853 / .6502 / .6783 | .6758 / .6204 / .5806 |
| **SPVL + M3** (intercept z_video + mean window z) | **.8938 / .6863 / .6783** | **.6882 / .6506 / .5806** |
| − transcript context | .6586 / .3980 / .6257 | .6814 / .6424 / .5905 |
| − frames | .8524 / .6134 / .6500 | .6240 / .6080 / .5701 |
| − context − frames (windows only) | .5391 / .2641 / .6286 | .5327 / .5004 / .5749 |
| causal mask (branches see each other) | .8854 / .6506 / .6641 | .6755 / .6192 / .5916 |
| ASR segments instead of fixed windows | .8833 / .6439 / .5890 | .6734 / .6174 / .5284 |
| S = 4 s / S = 16 s (within) | .6573 / .6362 | .5714 / .5730 |
| K = 8 frames | .8913 / .6754 / .6608 | .6506 / .5971 / .5759 |
| intercept = 2026-08 z (old judge) | .8541 / .5883 / .6783 | .6724 / .6611 / .5806 |
| intercept only (no residual) | .8829 / .6427 / .5000 | .6718 / .6157 / .5000 |
| + Vid-Group visual primary order | .8834 / .6471 / .6847 | .6742 / .6205 / .5678 |
| legacy chunk replica (2026-08 text scorer + old z) | .8521 / .5847 / .5588 | .6712 / .6589 / .5246 |

### Reading

- **M2 fixed windows** is the largest within-video effect: +.089 / +.052 over ASR segments (rule 14g passes on both corpora).
- **M1 frames**: +.028 / +.011 within (passes); on HateClipSeg frames also carry the verdict (pooled ROC .624 → .676).
- **M1 transcript context**: HateMM within +.053 and the verdict depends on it (pooled ROC .659 → .885); HateClipSeg within −.010 and pooled PR −.022 with context. The context part of M1 passes rule 14g on HateMM only.
- **Isolation (block vs causal mask)**: HateMM within +.014, HateClipSeg −.011. Not a supported novelty component; kept as the packing that makes one forward equal N independent calls.
- **S = 8 s** is best on both corpora; K = 20 vs 8: HateClipSeg prefers 20, HateMM pooled prefers 8 (within prefers 20).
- **Vid-Group** as primary order: +.006 / −.013 → not useful; ImageBind not tested (its OMSL-v6 ablation was already inert).
- **M3 (verdict + extent)**: the whole-video verdict alone over-ranks HateClipSeg videos whose hateful part is short (top-10 videos by z_video have 69 % positive frames vs 77 % for the old judge; §7 test read). Adding the mean window log-odds from the same forward raises pooled PR on both corpora (+.036 / +.030 over z_video alone) and HateMM ROC +.009. Among four constant-free extent estimators (mean, median, logit of mean probability, top-half mean) the spread is ≤ .013; mean is kept as the simplest. This choice was made on test (development-selected).

### Gate check (rule 8, noise floor pooled .005 / within .01) — SPVL + M3 vs OMSL-v6

| | HateMM ROC / PR / within | HateClipSeg ROC / PR / within |
|---|---|---|
| Δ vs OMSL-v6 | +.043 / +.108 / +.029 | +.019 / **−.012** / +.033 |

Comparison gate vs T3AL: passed on all six numbers. Promotion gate: five of six numbers improve beyond
the noise floor; HateClipSeg pooled PR is .012 below OMSL-v6 (beyond the .005 floor). Per rule 9 the
method stays in modification rounds (round 1 = M3); promotion needs a user ruling or a further round.

### Cost (rule 14j)
One Qwen3-VL-8B forward per video (prefix ≈ 20 × 91 image tokens + transcript; N ≈ 8–30 window
branches; mean 0.9 s/video on an RTX 5090, 333 videos in 5 min per arm). Preprocessing: Whisper once
(cached) + 20 ffmpeg seeks. Removed relative to OMSL-v6: 11–15 text calls/video, CLIP-L/14 features,
Vid-Group, ImageBind.

### Rule-14 checklist
(a) single full run of the final code, both corpora — yes; (b) noise floor stated, one number below —
yes; (c) one constant set and prompt for both corpora — yes; (d) constants declared in §4 before the run,
M3 intercept chosen on test among four variants — declared here; (e) no ensemble (one model, one
forward), no post-processing, no per-corpus branch — yes; (f) same model as the 2026-08 judge; the
comparator T3AL uses CLIP, LELA (GPT-4o-mini) not rerun — open; (g) per-module ablation ≥ .01 on both
corpora: M2 yes, M1-frames yes, M1-context HateMM only, M3 pooled PR yes on both; (h) evaluator, split,
GT, 4 fps unchanged — yes; (i) both corpora, all three metrics — yes; (j) cost — above.

## 9. Round 2 (2026-09-10): tailoring the mechanism to the task

User direction: refine the working mechanism into task-specific modules (novelty first, gains second).
Three task characteristics, each with a module and a falsifiable prediction:

| task characteristic (evidence in this repo) | module | prediction |
|---|---|---|
| **Stance is global, evidence is local.** Hate = statement + target + endorsement; target and stance are set by the whole video. Segment-independent localizers (LAVAD, EventVAD, T3AL, LELA) have within ≈ .50; adding the full transcript to the window judge gave HateMM within +.053. | **Stance conditioning**: the model first answers the whole-video question; its own answer (Yes/No) is appended to the shared context, and every window is then asked whether it is *one of the segments where the violating content occurs* (evidence question) — two forwards, still no extra encoding of frames. `--stance verdict --window-question evidence` | windows in hateful videos separate better (within ↑); non-hateful videos unaffected; risk: windows collapse onto the verdict (checked by within-video spread) |
| **Hate is not speech-only and not short.** 12 % (HateMM) / 20 % (HCS) of positive frames lie outside any speech segment; 11/85 and 16/104 hateful videos have most of their hate in silent parts; extent per video ranges .11–.95. Speech-driven localizers leave silent parts unscored (their within on the silent-hate subset is *below* chance: .42 / .39). | **Modality-attributed evidence**: every window gets a visual branch (frames inside the window only) and a speech branch (spoken words only) in the same forward; window score = max; the branch scores are the modality attribution. `--branches dual` | silent-hate videos gain most; the two branches are not copies of each other |
| **Verdict ≠ extent.** Pooled metrics measure how much of the video is hateful; the MLLM answers whether it is hateful (top-10 HCS videos by verdict: 69 % positive frames vs 77 % for the old judge). | **Extent-aware verdict** (M3): video score = verdict log-odds + mean window log-odds from the same forward | pooled PR ↑ on both corpora without touching within |

### Pilot (within-defined subset; within HateMM / HCS; source `runs/20260910_spvl/pilot2_*/metrics_ispvl_rrank.json`)

| arm | HateMM | HCS |
|---|---|---|
| round-1 SPVL (joint branch, rules question) | .6783 | .5806 |
| joint + evidence question | .6746 | .5688 |
| joint + rules + stance | .6828 | .5827 |
| joint + evidence + stance | .6997 | .5718 |
| dual + rules | .6886 | .5977 |
| **dual + evidence + stance** | **.6976** | **.6001** |
| triple + evidence + stance | .6931 | .6003 |

### Full set (333 videos; pooled ROC / PR / within; all with the M3 intercept z_video + mean window z; source `runs/20260910_spvl/<run>/metrics_izv_plus_mean_rrank.json`)

| variant | HateMM | HateClipSeg |
|---|---|---|
| OMSL-v6 | .8507 / .5781 / .6494 | .6692 / .6622 / .5473 |
| SPVL round 1 + M3 | .8938 / .6863 / .6783 | .6882 / .6506 / .5806 |
| **SPVL-r2** = dual + evidence + stance + M3 | **.8919 / .6831 / .6976** | **.7119 / .6664 / .6001** |
| r2 − stance | .8945 / .6852 / .6899 | .7023 / .6579 / .5881 |
| r2 with rules question | .8913 / .6753 / .6954 | .7070 / .6598 / .5998 |
| r2 with joint branch (no modality split) | .8907 / .6838 / .6997 | .6881 / .6511 / .5718 |
| r2 triple (joint + visual + speech) | .8944 / .6862 / .6931 | .7060 / .6637 / .6003 |

### Reading

- **Gate vs OMSL-v6** (noise floor pooled .005 / within .01): all six numbers improve — HateMM
  +.041 / +.105 / +.048, HateClipSeg +.043 / +.004 / +.053 (HCS PR now within noise instead of −.012).
  Comparison gate vs T3AL passed. **Promotion gate passed.**
- **Modality split** (dual vs joint): HCS within +.028, pooled ROC +.024, PR +.015; HateMM within −.002.
  The two branches are nearly independent inside a video (median Spearman(visual, speech) .11 HateMM /
  .16 HCS); the visual branch is the larger score in 29 % / 46 % of windows with speech.
- **Stance conditioning**: within +.008 / +.012, pooled PR −.002 / +.009. Passes the .01 floor on HCS
  only; it is the mechanism that carries the evidence framing (evidence question without stance is
  slightly worse than the rules question).
- **Extent intercept**: with dual branches the mean window log-odds is a better extent estimate — HCS
  pooled PR .622 → .666 (z_video alone → M3), now above OMSL-v6.
- **Silent-hate subset** (26 videos with > 50 % of positive frames outside speech; `subset_eval.py`):
  within HateMM .694 (round 1: .745), HCS .446 (round 1: .472). The prediction that silent-hate videos
  gain most from the visual branch is **not** supported; HCS silent-hate videos stay below chance.
  Diagnosis: with 20 uniform frames a long video has no frame inside most 8-second windows, so the
  visual branch of those windows sees nothing. Round 3 tests one frame per window (`data/frames_w8`).
- Modification rounds used (rule 9): round 1 = M3, round 2 = stance + modality split.

## 10. Round 3 (per-window frames) and the final configuration

`data/frames_w8`: one frame at the centre of every 8-second window (avg 22 frames/video), so the visual
branch of every window has a frame inside it. Source `runs/20260910_spvl/full3_dual_evid_stance_w8/`.

| variant | HateMM ROC / PR / within | HateClipSeg ROC / PR / within | cost |
|---|---|---|---|
| SPVL-r2 (20 uniform frames) + M3 | .8919 / .6831 / .6976 | .7119 / .6664 / .6001 | 1.5 s/video, 1 group |
| SPVL-r2 with per-window frames + M3 | .8936 / .6786 / .6885 | .7199 / .6798 / .6004 | 7–11 s/video, median 2 groups (≈ 10.7k tokens) |
| joint branch + evidence + stance, per-window frames + M3 | .8940 / .6801 / .6797 | .6994 / .6697 / .5823 | 4–6 s/video |
| joint branch + rules question, no stance, per-window frames + M3 | .8935 / .6788 / .6592 | .7003 / .6670 / .5826 | 4–6 s/video |
| silent-hate subset within (26 videos): r2 → r2 per-window frames | .694 → .723 | .446 → .435 | |
| silent-hate subset within: joint + evidence + stance / joint + rules, per-window frames | .698 / .721 | .467 / .511 | |

Per-window frames raise HateClipSeg pooled PR (+.013) and HateMM silent-hate within (+.029) but cost
5× and lower HateMM within by .009 (noise floor .01). With per-window frames the modality split is
still the part that carries HateClipSeg within (dual .6004 vs joint .5823, +.018) and stance
conditioning still carries HateMM within (joint + stance .6797 vs joint no-stance .6592, +.021; on
HateClipSeg ±0). So the round-2 attribution (split helps HCS, stance helps HateMM) holds under a
second frame layout. Sources: `runs/20260910_spvl/abl3_joint_evid_stance_w8/` and
`abl3_joint_rules_none_w8/`, `metrics_izv_plus_mean_rrank.json`. HateClipSeg silent-hate videos stay below chance in
every variant: their GT positives are the offensive union (sexual, violent, harmful, insulting), while the
prompt's rules are hate rules, so silent sexual/violent visuals are not what the judge is asked for. This
is a label-definition mismatch, recorded, not tuned around (rule 13: one prompt for both corpora).

**Final configuration (SPVL-r2)**: 20 uniform timestamped frames + full timestamped Whisper transcript as
the shared context; whole-video verdict; the model's own verdict appended to the context; per 8-second
window a visual branch and a speech branch (evidence question), window score = max; frame score =
(z_video + mean window z) + centred-rank residual. Two forwards per video (verdict, then windows), frames
encoded twice, no other model. The round-3 arms (joint branch with per-window frames) are kept as ablations in the table above.

### Gate (rule 8) — SPVL-r2 + M3 vs OMSL-v6 (noise floor pooled .005 / within .01)
HateMM +.041 / +.105 / +.048; HateClipSeg +.043 / +.004 / +.053. Passed; comparison gate vs T3AL passed.

### Rule-14 checklist for SPVL-r2
(a) single full run of the final code, both corpora — `runs/20260910_spvl/full2_dual_evid_stance/`;
(b) noise floor stated; HCS PR +.004 is reported as "no change"; (c) one prompt, one constant set for both
corpora; (d) constants declared in §4 (K=20, S=8, evidence question, stance verdict, dual branches, M3 mean)
— dual/evidence/stance were chosen among 6 pilot arms and M3 among 4 estimators on test
(development-selected); (e) one model, no ensemble, no post-processing, no per-corpus branch; (f) same
Qwen3-VL-8B as the 2026-08 judge; T3AL uses CLIP; LELA (GPT-4o-mini) not rerun — open; (g) per-module
ablation (both corpora ≥ .01 within): M2 fixed windows yes; M1 frames yes; M1 context HateMM only; modality
split HCS only (+.028; HateMM −.002); stance HCS only (+.012; HateMM +.008); M3 pooled PR yes on both;
the round-2 combination (dual + evidence + stance) vs round 1: within +.019 / +.020 — passes as a
combination, its parts do not individually pass on both corpora; (h) evaluator, split, GT, 4 fps unchanged;
(i) both corpora, three metrics; (j) cost: 2 forwards/video (verdict + windows), ≈ 1.5 s on a 5090, 333
videos in 8 min; preprocessing Whisper + 20 ffmpeg seeks.

## 11. MLLM family / size robustness study (2026-09-10, started)

**Question** (user, 2026-09-10): is the result specific to Qwen3-VL-8B? Three sub-questions: Q1 what another MLLM
gives on its own (whole-video verdict; per-window independent judgement); Q2 whether the same SPVL-r2 pipeline
on another MLLM is comparable; Q3 whether the ablations point the same way on other MLLMs.

**Models** (all native in transformers 5.15.1, non-thinking; chosen from what hateful-video and video-understanding
papers actually use: arXiv 2601.15115, 2606.11953, 2608.15905, 2602.21854, 2508.18265):

| tag | HF id | role | machine |
|---|---|---|---|
| q3vl-8b | Qwen/Qwen3-VL-8B-Instruct | current method; also the cache-path consistency check | uoa-lab3 |
| q3vl-2b / q3vl-4b | Qwen/Qwen3-VL-{2B,4B}-Instruct | same family, smaller | uoa-lab3 |
| q3vl-32b | Qwen/Qwen3-VL-32B-Instruct | same family, larger | uoa-campus1 (A100 80G, Slurm) |
| q25vl-7b | Qwen/Qwen2.5-VL-7B-Instruct | previous generation, the most common baseline in hateful-video papers | uoa-campus2 (Slurm) |
| internvl35-8b | OpenGVLab/InternVL3_5-8B-HF | other family (InternViT + Qwen3 LLM) | uoa-lab2 |
| llava-ov-7b | llava-hf/llava-onevision-qwen2-7b-ov-hf | other family (SigLIP + Qwen2) | uoa-lab2 |
| gemma3-12b | google/gemma-3-12b-it | other family (SigLIP + Gemma), gated | lab-server |

**Code path.** `--isolation cache` (`Judge.prefix_cache / extend_cache / cached_branch`): the prefix is run once
with a KV cache; every branch (whole-video question; per-window visual / speech questions) is one short forward
on a deep copy of that cache, so each branch is exactly an independent call. The stance turn (question +
the model's own Yes/No + end-of-turn, rendered by the model's chat template) is appended to the cache once.
Branch text is the suffix of the chat-template rendering, so no ChatML string is hard-coded; the string and
token seam checks of §5 still run on the first video of every run. Per-family image settings
(`FAMILY_IMAGE_KW`): Qwen pixel cap as before (91 tokens/frame), InternVL one 448 tile, Gemma 3 without
pan-and-scan, LLaVA-OneVision base 384 only (no AnyRes grid); tokens/frame and prefix length are recorded per run.
Everything else (K=20, S=8, rules, reader, questions, Yes/No token sets, seed, greedy read-out) is unchanged.

Cache path vs independent plain calls on Qwen3-VL-8B (`runs/20260910_spvl/mllm/q3vl-8b/verify_cache/verify.json`):
whole-video Δz .033, windows max Δz .228, Spearman 1.000 — the same bf16 kernel-difference magnitude as the
packed path (§5). Full-set consistency of the cache path against the mask path: see the table below.

**Arms per model** (`launch/run_mllm.sh`; Slurm: `launch/campus_mllm.sbatch`): full (SPVL-r2), winonly (joint
branch, rules question, no context, no frames, no stance = the MLLM judging each window on its own), noctx,
noframes, asr (ASR segments instead of fixed windows), nostance, joint; compose variants of full: intercept only
(= the MLLM's whole-video verdict alone), z_video intercept without M3. Table: `summarize_mllm.py` →
`runs/20260910_spvl/mllm_table.md`.

**Reading rule** (declared before the runs): an ablation counts as "design holds across MLLMs" if it has the same
sign and exceeds the noise floor (within .01 / pooled .005) on at least 5 of 7 models; models where the sign flips
are listed with the likely reason (image tokens per frame, model capability).

Results: pending (runs launched 2026-09-10 21:00 on five machines).
