# SPVL — single-pass verdict-and-evidence localization (label-free)

Status: proposal written 2026-09-10 (uoa-lab1); pilot pending. Development-selected numbers only
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
Vid-Group, ImageBind). Sequences above `--max-tokens 12000` are split into branch groups, each with the
full prefix. Estimated 3–6 s per video on one RTX 5090; 643 videos < 1 h.

## 4. Constants (declared before any run)

K = 20 frames (uniform, t_k = (k+0.5)·D/20); S = 8 s windows; pixel cap 100352 / min 65536; rules =
`YOUTUBE_RULES` for both corpora; reading instruction = `prag`; Yes/No token sets = first tokens of
{Yes, " Yes", yes, " yes", YES, " YES"} and the No analogues; uncovered frames in ASR-window mode =
−12 (as in OMSL-v6 `text_curve`); seed 0; `--max-tokens 12000`; bool block mask via sdpa. Both corpora
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
  bf16 logits by up to 0.3–0.7; under the *same* kernel the all-causal 4D mask equals no-mask
  exactly (0.0000) and packed-block vs independent calls differ by ≤ 0.24 log-odds with the
  fp32 read-out. Packing is therefore exact up to bf16 kernel rounding, which affects every
  arm (sequential calls included) equally;
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

## 8. Results

(E4 pending)
