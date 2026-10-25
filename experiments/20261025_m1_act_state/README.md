# Candidate 50: act-state reading, mask form A1 (falsifier pre-check)

Declared 2026-10-25, before any run. No GPU job has been submitted and no read of this candidate exists yet.

- Development host: uoa-lab1 (sc474397). Compute host: uoa-lab1, Slurm partition `local-sc474397`, one RTX 5090.
- Implementation tag `c50-a1-v1` (2026-10-25). Code: `measure.py`, `act_state.py`, `stage0.py`, `stage1.py`,
  `smoke_cpu.py`, `launch/lab1_stage0.sbatch`, `launch/lab1_stage1.sbatch` in this directory.
- Status: implemented; CPU smoke test passed (§10). Waiting for the rule-6 code review. Then stage 0.
- Every number this candidate produces is development-selected (read on test, rule 10).

## 1. What is tested

**Thesis.** The video-level act reaches the moment through the verdict state, not through the rest of the video.

**Mechanism (A1).** Each window branch keeps Grounder's `all_local` access set:
- the scaffold;
- its own window's prefix frames and transcript tokens;
- its own query tokens.

In addition, at all 36 text layers, it sees exactly one more span: the verdict question, the model's verdict answer
and the one-word stance turn. The rest of the video can then reach the window read only through two routes:
- the verdict-span states, which were computed with attention to the whole prefix;
- the window's own prefix states, which were computed with attention to everything before them. So A1 is not "only
  the moment" (review_m1.md §4).

A1 restricts direct attention access, not all information flow (Grounder's wording). The claim is a route: the
reader's own decision about the whole video is the one direct channel through which the rest of the video reaches each
moment's read. The difference between A1 and `all_local` measures what the decision span contributes. It does not show
that the moment is read "from its own evidence alone".

**A1 is a falsifier, not the claim.** A1 is one more cell of Grounder's mask factorial: closed family M1-f, with an
access set that was never run. Its two neighbouring cells were run:
- `verdict_only` (verdict blocked in the last 9 layers, media visible): equal to base;
- `all_local` (verdict and other media blocked at every layer): −.046 / −.049 within.

If A1 passes stage 1 and stage 2, the claimable form is the transfer form **A2**. Its settings were declared in
review_m1.md §5.3 (copied in §3 below): layer 18 of 36, scale 1, and the designer's content-free baseline. That
baseline is the act state minus the same position's state on a content-free prefix (design_m1.md proposal 3). A2 is
run only after A1 passes. If A1 is killed, neither A2 nor design_m1 #3 is run.

**Open before any A2 run.** The two design notes describe two different A2 forms (rule-4 review, advisory 3):
- design_m1 #3, the ICV / function-vector form: add (act state − content-free state) at the question's last token,
  at one layer;
- design_story #2, the Patchscopes form: patch the verdict-position states into a placeholder at every layer.

review_m1.md §5.3's settings (one layer, 18 of 36, scale 1, content-free baseline) belong to the first form. The
form, the positions and the content-free prefix must be declared in this README before any A2 run. A2 costs one short
local pass per window and modality, about 1.5× the r6 reading cost (design notes). The designer estimates about
10 GPU min for HateMM + HCS and 20 min for DeHate.

**How the outcomes will be read** (rule-4 review, advisory 5):
- A1 above `all_local` but not above A1_other means the decision span acts as a per-video level of the reads. P10 and
  the HVL permutation result found the same. Archive under that heading.
- A1 at or below `all_local` means the decision span adds nothing direct. `verdict_only` = base already showed this
  while the media were visible.

**Prior.** About 5% (review_m1.md §5.3). The CPU bound in §3 shows why. Restoring each video's per-modality mean and sd
of the `all_local` reads to r6's own leaves the decoded within .018 / .020 below r6. So A1 can pass only if the
verdict turn carries window-specific information. P10, HVL and Grounder `verdict_only` all say it does not.

**Background.**
- `design_m1.md` proposal 3 (act-state transfer) and `design_story.md` proposal 2 (act-state reading), in the
  2026-10-25 M1 / M3 novelty team folder;
- the adversarial review `review_m1.md` §2–§5 (verdict: RUN PRE-CHECK, mask form A1 only);
- `history.md` §5: the r6 prompts and code paths that every arm here keeps.

**Proposal review (rule 4): PASS**, `docs/reviews/20261025_m1_act_state_novelty.md` (2026-10-25). It covers A1, and
A2 if A1 passes. Its advisories are taken up in §1, §2, §4 and §8.

## 2. Sources

**Mask factorial.** Grounder (`archive/experiments/20261002_m1_grounder/`, 2026-10-02): `base`, `late`,
`verdict_only`, `all_local`, `shifted`, `early`. A1 adds the cell "own-window media + verdict span, all layers".

**Mask form.**
- Gist tokens (NeurIPS 2023, arXiv 2304.08467): later tokens see the rest of the prompt only through a summary state.
  It is trained there; here it is training-free.
- "Language Models Can Control Their Own Attention" (Dynamic Attention, arXiv 2609.02737). This is the closest
  topology. It is training-free: the model attends to a chunk it names, the scaffold and its own running response.
  Its use is long-context text QA. There the model chooses the chunk; here the chunk is fixed by the window's time,
  and the visible own text is a decision about the whole input.
- Neighbouring access patterns: Star Attention (ICML 2025), StreamingLLM (ICLR 2024), LM-Infinite, Longformer /
  BigBird. In these the global part is a position (an attention sink or anchor), not content.

**Blocking edges and donor interchange.**
- Attention knockout: Geva et al. 2023; Zhang et al., "Cross-modal Information Flow in MLLMs", CVPR 2025. These block
  attention between token groups to trace information flow; here the blocking is part of the scoring path.
- A1_other is an interchange intervention with a donor input, a standard form in causal mediation work.

**Transfer-form precedents (A2).** These transfer a task from demonstrations, or steer style:
- I2CL (ICLR 2025, arXiv 2405.14660);
- in-context vectors, ICV (ICML 2024);
- task vectors (Findings of EMNLP 2023);
- function vectors (ICLR 2024);
- Multimodal Task Vectors (NeurIPS 2024);
- Patchscopes (ICML 2024).

What A1 / A2 would add: the transferred state is the same instance's own decision, used to condition parts of the
same input for localisation. Only this difference can be claimed, not the mask or the vector by itself.

**Closest work in neighbouring tasks.**
- FBHM (arXiv 2605.31349): hateful memes. It learns one steering vector and one scale per layer from 500 labelled
  samples, and classifies whole memes. It has no mask, no transfer from the same instance, and no localisation.
- SteerVAD (ICLR 2026, arXiv 2602.24021): video anomaly detection. A frozen MLLM's first-token state drives a
  meta-controller that scales 4 attention heads. The meta-controller is trained with segment labels. It does no
  attention routing and copies no state between passes.

**Hate work.** No use of masks, KV re-placement, task vectors or state patching in hateful video detection or
localisation was found (review_m1.md §3c; the rule-4 review's 32 searches). Searched: LELA, MultiHateLoc, HVGuard,
MARS, RAMF, SafeLens, FBHM.
- Rule 4 does not STOP A1 (review_m1.md §2).
- Rule 5 does not apply: no new input.
- Rule 3: same model, no ensemble.

## 3. Protocol, copied verbatim from review_m1.md §5.3 (thresholds final)

The block below is review_m1.md lines 309–367, unchanged (written by the M1-side adversarial reviewer, 2026-10-25). Section numbers inside it refer to review_m1.md.

### 5.3 design_m1 #3 / design_story #2: RUN PRE-CHECK, mask form A1 only

**CPU prior, computed** (`act_state_bound.py`):
- Grounder `all_local` reads decode (raw key) to .8940 / .6875 / .7052 and .6980 / .6527 / .5887.
- Within-video Spearman with r6: visual .39 / .32, speech .83 / .86. The context reorders visual reads; it does not just
  scale them.
- Restoring each video's per-modality mean and sd to r6's own (gain plus offset, label-free) reaches .7326 / .6178.
  That is −.018 / −.020 below r6.
- So a verdict that acts only as a per-video gain cannot bring A1 to r6. A1 must restore window order. Probability of
  passing: about 5%.

**Stage 0: GT-free, first 20 videos by sorted id per corpus (HateMM, HCS, DeHate), about 5 GPU min including loading.**
- Arms: `all_local`, A1, A1-other.
  - A1 = `all_local` plus the verdict question, answer and stance-turn tokens visible at all 36 layers.
  - A1-other = A1 with the verdict-turn keys and values of the donor video, re-rotated to this video's positions with
    `src/pre_rotary_memory.py`. Donor: the next video, in sorted id order in the same corpus, with the same verdict sign
    (same tokens).
- Stop if any of these fails, on any corpus:
  - median within-video sd of the visual change (A1 − `all_local`) ≥ .5 logit;
  - median per-video constant share of that change, n·mean² / Σ change², ≤ .8;
  - median within-video correlation between (A1 − `all_local`) and (A1-other − `all_local`) ≤ .9.

**Stage 1: full HateMM 215 + HCS 118, four arms, about 30 GPU min on one lab 5090.** The estimate comes from Grounder:
2596 s for six arms on sc474399.
- Arms:
  - base replica, which must equal `base_gridA` exactly (Grounder's did);
  - `all_local` replica, within 1e-3 logit of `runs/20261002_m1_grounder/r1_full/all_local`;
  - A1;
  - A1-other.
- Pass, all of:
  - (a) A1 within ≥ r6 − .005: ≥ .7458 / .6323;
  - (b) A1 pooled ROC / PR ≥ .8905 / .6833 and .7087 / .6616;
  - (c) A1 − `all_local` ≥ +.01 within on both. This is the 14g ablation of the "act-state conduit".
  - (d) A1 − A1-other ≥ +.01 within on both, with the paired CI lower bound > 0;
  - (e) A1 − (its per-video-gain reconstruction) ≥ +.01 within on both. The reconstruction is `all_local` reads mapped per
    video and modality to A1's own per-video mean and sd, then decoded.
  - (f) raw max-curve within of A1 ≥ .670 / .600 (r6 raw − .01).
- Kill: (a) or (d) fails on either corpus → archive. A2 and design_m1 #3 are not run.

**Stage 2: only if stage 1 passes. DeHate 1341 videos, arms A1 and `all_local`, about 60 GPU min.** The r6 read took
1699 s.
- Pass: within ≥ .6331, pooled ≥ .6944 / .1518.

**Cost at deployment**: the same forwards as r6. Grounder measured branch time .991× base, with no added calls.

**Code**: Grounder's mask plumbing (`archive/experiments/20261002_m1_grounder/`) with one more allowed span. A rule-6
code review is needed before stage 1.

**A2 / design_m1 #3, only after A1 passes.**
- Settings, declared before running: layer 18 of 36, scale 1, and the content-free baseline the designer wrote. No
  layer scan unless it is declared and reported in full.
- Arms: local read without a patch, own-video patch, donor-video patch (same rule as A1-other).
- Pass:
  - own − local ≥ .01 within on both corpora;
  - own − donor ≥ .01 within on both;
  - own ≥ r6 − .005 within and pooled on both;
  - DeHate floors as in stage 2.
- Cost: about 10 GPU min for HateMM + HCS and 20 min for DeHate (the designer's estimate).
- Novelty if it passes: high for the task. Cite I2CL, ICV, task and function vectors, MTV, Patchscopes and Gist.

## 4. Arms and how each is built

All arms share the per-video computation of r6 (history.md §5.1, `experiments/20260922_til/til_measure.py`):
- the prefix (system message, 20 timestamped frames, timestamped transcript, nine rules, reader instruction);
- the verdict z_video;
- the stance turn, which is the single word `Yes` / `No` plus its end-of-turn tokens.

Every arm then reads the same isolated visual branch and speech branch per 8 s window, with the r6 questions.

**Mask plumbing.** Every arm uses the same explicit bf16 attention mask at all 36 text layers. The plumbing is copied
from Grounder `QueryAccess`; see the header of `act_state.py` for what changed.

| arm | columns of the cache a window branch may attend to (its own query tokens are always visible) | role |
|---|---|---|
| `base` | everything | plumbing replica of `runs/20260926_glr/base_gridA` (must be exact) |
| `all_local` | scaffold + this window's prefix frame tokens + this window's transcript tokens | replica of Grounder `all_local` (within 1e-3) |
| `A1` | `all_local` + the verdict span | candidate |
| `A1_other` | `A1`'s columns, on a cache copy whose verdict span holds the donor video's states | control (d) |
| `A1_recon` | CPU only: `all_local` reads mapped per video and modality to A1's per-video mean and sd | control (e) |

**Token regions.** They come from `src/window_token_regions.py`, unchanged; this is Grounder's mapping.
- Frame tokens: each frame's `[t=..s]\n` line through `<|vision_end|>`.
- Transcript tokens: the lines between `Transcript:` and `Based on this platform`.
- Scaffold: every other prefix token (system message, intro, headers, rules, reader instruction, chat-template tokens).
  This is 290 tokens in every video checked (§10).
- A window's local tokens: the frames whose timestamp lies in [start, end). Plus, for each overlapping transcript
  chunk, its `[s-e] ` header and the words chosen by `window_text`'s proportional rule.
- Windows with no sampled frame keep zero visual tokens. Nothing is filled in.

**The A1 span.** It is located by position in the cache, not by text search. With P = prefix length, the verdict
span is the cache positions [P, P + 32):
- the verdict question turn: `<|im_start|>user\nDoes this video contain content that violates any of the above
  rules?\n\nAnswer "Yes" or "No".<|im_end|>\n<|im_start|>assistant\n`;
- then `Yes<|im_end|>\n` or `No<|im_end|>\n`.

It is 32 tokens for both stances. These are exactly the tokens appended after the prefix by the verdict forward and
the stance forward (`Judge.branch_ids` + `Judge.answer_ids`).
- `window_keep(..., "A1")` equals `window_keep(..., "all_local")` on [0, P) and is True on [P, P + 32).
- `all_local` is False there.
- The smoke test checks that the two keep vectors differ on exactly these positions (§10).

**A1_other: how the donor's states are built.**
1. Donor rule: the next video in sorted id order, cyclic, of the same corpus with the same verdict sign. The corpus is
   the whole test split, also in stage 0. The signs come from the cached r6 reads (`base_gridA`; DeHate
   `reads_gridA`).
   - Same sign means identical span tokens, including the one-word stance token (asserted in the job).
   - A matching donor exists for every video of all three corpora. Yes / No verdicts: HateMM 146 / 69, HCS 108 / 10,
     DeHate 809 / 532. The smallest group is HCS "No" with 10 videos.
   - Fallback: none. If a sign group had only one video, `measure.py` stops instead of substituting another donor.
   - Every A1_other record names its donor (`extra.donor`); so does `checks.jsonl`.
2. The donor's prefix, verdict and stance are computed exactly as in r6. During its verdict forward and its stance
   forward, `src/pre_rotary_memory.observe` captures the span's pre-rotary keys (`k_norm` output) and values
   (`v_proj` output), at all 36 layers.
3. The keys are rotated with `src/pre_rotary_memory.rotate` to THIS video's span positions. These are M-RoPE text
   positions, all three axes equal: (P .. P + 31) + δ_v.
   - δ_v is this video's rope delta: the largest prefix position + 1 − P, which is negative because image tokens
     share positions. Example: hate_video_1 has P = 2343 and δ = −1560, so the span sits at positions 783–814 (§10).
   - The positions are computed by `get_rope_index` (`src/stance_cache.positions`) and checked against the incremental
     rule the model uses for cached forwards.
   - The donor's own positions never enter, because its keys are captured before rotation.
4. A deep copy of this video's cache gets, at [P, P + 32) of every layer, the re-rotated donor keys and the donor
   values. The A1_other branches run on that copy with A1's mask. Everything else is this video's own: the prefix,
   the branch questions and their positions.
5. Videos are processed in descending sorted-id order, so each donor's states come from its own main pass at no
   extra cost.
   - A donor that was not processed in the session gets one donor pass: prefix, verdict and stance only. This
     happens at the cyclic wrap, at the end of a `--first` subset, and after a resume.
   - Donor passes: stage 0 has 1 / 2 / 2 (HateMM / HCS / DeHate); stage 1 has 2 / 2.
   - The donor's own z_video must equal its r6 read exactly.
   - States stay in GPU memory, about 4 MB per donor, at most a few at a time. They are never written to disk.

**A1_recon (gate e).** Per video and modality, over the windows that have that modality:
- the rule: y = (x_al − mean_al) / sd_al × sd_A1 + mean_A1, with population sd;
- with fewer than 2 such windows, or sd_al = 0: offset only, y = x_al + (mean_A1 − mean_al);
- then z = max over modalities, and the curve is repainted.

This is the convention of the reviewer's `act_state_bound.py` (`AL_aff`), with A1 as the target instead of r6. It
uses no labels.

## 5. Implementation reading of the protocol (declared before any run)

Where §3 leaves a choice open, the implementation fixes it as follows.

**Stage 0** (`stage0.py`, no GT):
- "First 20 videos by sorted id": Python `sorted()` of the video id strings of the corpus's test manifest.
- The three stop statistics are computed on the **visual** change d = z_visual(A1) − z_visual(`all_local`). Every
  window has a visual branch.
- S3 correlates d with d_o = z_visual(A1_other) − z_visual(`all_local`) and uses the Pearson correlation.
- Diagnostics, not gates: the same statistics on the speech change, and the Spearman form of S3.
- sd is the population sd.
- S1 and S2 use videos with at least 2 windows. S3 uses videos with at least 3 windows and a non-zero sd of both
  changes.
- A video whose change is exactly zero in every window counts as constant share 1.
- Stop if any of S1 (≥ .5), S2 (≤ .8), S3 (≤ .9) fails on any of the three corpora.

**Stage 1** (`stage1.py`):
- (a), (b), (c), (e), (f) compare corpus-level numbers from the `metrics.json` that the evaluator writes.
- (c), (d), (e) use this run's own `all_local`, `A1_other` and `A1_recon`, decoded with the same decoder.
- The decoder is `experiments/20260926_twolevel/twolevel_r2.py`, unchanged, with the flags of the r6 raw-key run:
  `--noleak --transform nscore --key raw --arm m2 --duration bma --bma-prior length --min-windows 2 --bma-grid 6`.
  `stage1.py` checks these against `runs/20260926_twolevel/final_rawkey/config.json` key by key, and checks every
  decoded arm's `config.json`.
- Decoder parameters are refit label-free per arm, as for every read set.
- (d)'s CI: paired video bootstrap over the within-eligible videos (84 / 99), with 10,000 resamples, seed 0. It is the
  95% percentile interval of the mean per-video within difference.
- Per-video within values come from `src/eval/evaluate.within_video_macro`, called on one video at a time. Their mean
  reproduces the corpus number exactly (asserted).
- (f) uses the undecoded read curve: per window max(z_visual, z_speech), painted on the 4 fps grid and scored by
  `src/eval/evaluate_four_datasets.py`.
- Verdicts: **PASS** = (a)–(f) all hold on both corpora. **FAIL_KILL** = (a) or (d) fails on either corpus: archive,
  and A2 / design_m1 #3 are not run. **FAIL** = any other failure.
- Reported but not gates:
  - paired CIs for A1 − r6, A1 − `all_local`, A1 − A1_recon, `all_local` − r6, base − r6;
  - the raw-curve comparisons;
  - the mean difference without the 3 largest-gain videos.
- The base replica, decoded, must reproduce `final_rawkey` (§6). `stage1.py` reports the absolute difference.

**Stage 2** (`stage1.py --stage 2`):
- arms `all_local` and A1 on the 1341 DeHate test videos, the same decoder;
- pass = A1 within ≥ .6331, pooled ROC ≥ .6944 and pooled PR ≥ .1518.

## 6. Reference numbers (r6, raw key; development-selected)

Order: pooled ROC / pooled PR / within (within N = 84 / 99 / 222).

| corpus | r6 raw key | source |
|---|---|---|
| HateMM | .8955 / .6883 / .7508 | `runs/20260926_twolevel/final_rawkey/metrics.json` |
| HateClipSeg | .7137 / .6666 / .6373 | same file |
| DeHate | .6994 / .1568 / .6431 | `runs/20260927_dehate_external/r6_rawkey/metrics.json` |

- Reads: `runs/20260926_glr/base_gridA` and `runs/20260927_dehate_external/reads_gridA`.
- Neighbouring cells:
  - Grounder `all_local`, decoded with the raw key: .8940 / .6875 / .7052 and .6980 / .6527 / .5887 (review_m1.md §5.3);
  - the reviewer's gain-plus-offset bound on `all_local`: within .7326 / .6178.
- Floors from §3:
  - (a) .7458 / .6323;
  - (b) .8905 / .6833 and .7087 / .6616;
  - (f) .670 / .600;
  - stage 2: .6331, .6944 / .1518.

## 7. In-job checks (`measure.py`; any failure raises `CHECK_FAILED` and the job prints FAILED)

| check | tolerance |
|---|---|
| z_video of every video (and of every donor) vs the r6 read | exact |
| `base` replica: every z_visual / z_speech vs `runs/20260926_glr/base_gridA` (DeHate: `reads_gridA`) | exact |
| `all_local` replica vs `runs/20261002_m1_grounder/r1_full/all_local` (HateMM, HCS) | 1e-3 logit |
| same branch set (which windows have a speech branch) as the reference reads | exact |
| own span pre-rotary keys re-rotated to own positions vs the cached keys | max abs diff / max abs key ≤ 1e-2 (expected 0) |
| own span captured values vs cached values | exact |
| verdict-span positions from get_rope_index = text positions (cache index + rope delta, 3 axes equal) | exact |
| donor span tokens and stance = this video's | exact |
| first video per corpus: A1 on a cache whose span holds its own re-rotated states vs A1 | 1e-3 logit |
| first video per corpus: in-place + crop vs deep copy, every arm | 1e-2 logit (Grounder's) |
| first video per corpus: explicit-mask base vs the ordinary SDPA branch | .25 logit (Grounder's) |
| first video per corpus: prefix + question token seam (`seam_check_tokens`) | exact |

All values are written per video to `checks.jsonl`.

## 8. Cost

| step | arms × videos | estimate |
|---|---|---|
| stage 0 | 3 × 60 (20 per corpus) | about 5 GPU min including model loading; 5 donor passes |
| stage 1 | 4 × 333 | about 30 GPU min on one lab 5090 |
| stage 2 | 2 × 1341 (DeHate) | about 60 GPU min |
| stage-1 CPU analysis | 5 arms decoded + raw scoring | 89 s on lab1 for HateMM + HCS (smoke test, §10) |

- The stage-1 estimate comes from Grounder: 2596 s for six arms on sc474399. Four arms plus 4 donor passes plus one
  cache copy per video gives about 30 min.
- The stage-2 estimate: the r6 DeHate read took 1699 s.
- **Deployment cost of A1: the same forwards as r6.** These are 1 prefix, 1 verdict, 1 stance extension, and one
  branch per window and observed modality. The mask adds no forward; Grounder measured branch time .991× base.
- A1_other is a control only and has no deployment cost.
- No new input, no new cache, nothing written besides reads, checks and metrics.

## 9. How to run

All commands are from the repo root `/home/jehc223/Hate-follow-up` on uoa-lab1.

**GPU environment.** `.cache/envs/HateVLM`: torch 2.11.0+cu128, transformers 5.15.1, the r6 reader's versions. It is
activated by `PATH`, the same way as `experiments/20261025_m3_coverage/launch/lab1_read.sbatch`.

**CPU analysis environment.** `~/miniconda3/envs/HateVideo/bin/python`, the environment of every r6 decode.

```bash
# 0. CPU smoke test (no GPU; dry-run outputs go to the given directory, not under runs/)
HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .cache/envs/HateVLM/bin/python experiments/20261025_m1_act_state/smoke_cpu.py --out <scratch dir>

# 1. stage 0 (GT-free), after the rule-6 code review
mkdir -p runs/20261025_m1_act_state
sbatch experiments/20261025_m1_act_state/launch/lab1_stage0.sbatch
#    -> runs/20261025_m1_act_state/stage0/stage0_summary.json, "STAGE0_VERDICT CONTINUE|STOP" in the Slurm log

# 2. stage 1 reads (only after CONTINUE), then the CPU analysis
sbatch experiments/20261025_m1_act_state/launch/lab1_stage1.sbatch
setsid nohup ~/miniconda3/envs/HateVideo/bin/python -u experiments/20261025_m1_act_state/stage1.py --stage 1 \
  --reads runs/20261025_m1_act_state/stage1 > runs/20261025_m1_act_state/stage1_analysis.out 2>&1 &
#    -> runs/20261025_m1_act_state/stage1_analysis/{summary.json, table.tsv, per_video.csv, decoded/<arm>/metrics.json}
#       "STAGE1_VERDICT PASS|FAIL|FAIL_KILL"

# 3. stage 2 (only after stage-1 PASS)
RUN=stage2 DATASETS=DeHate ARMS="all_local A1" sbatch experiments/20261025_m1_act_state/launch/lab1_stage1.sbatch
setsid nohup ~/miniconda3/envs/HateVideo/bin/python -u experiments/20261025_m1_act_state/stage1.py --stage 2 \
  --reads runs/20261025_m1_act_state/stage2 > runs/20261025_m1_act_state/stage2_analysis.out 2>&1 &
```

**Monitoring.** Watch for the job leaving `squeue`, then for `C50_STAGE0_DONE` / `C50_READS_DONE` versus
`FAILED` / `CHECK_FAILED` / `Traceback` in `runs/20261025_m1_act_state/slurm_<job>.out`.

**Resume.** `measure.py` appends and skips finished videos. Donor states lost with the process are rebuilt by donor
passes.

## 10. CPU smoke test (2026-10-25, uoa-lab1, before any GPU run)

Command as in §9. HateVLM python for parts 1–3; HateVideo python for the analysis dry runs. Log of the final run:
`smoke_cpu.py --out <scratch>`, exit 0, `SMOKE_CPU_DONE`.

1. **Arguments and plans** (`measure.py --plan-only`). The processing order is descending sorted id. Donor passes:
   - stage 0: HateMM 1 (hate_video_206), HCS 2, DeHate 2;
   - stage 1: 2 per corpus (the cyclic wrap of each sign group);
   - stage 2: none (no A1_other).
   Without `--run-name` the script stops with an argparse error.
2. **Mask spans on real tokenisations** (processor only, first 3 videos of each corpus):
   - The prefix length equals the r6 reads' `prefix_tokens` for all 9 videos.
   - The verdict span is 32 tokens for both stances, `<|im_start|>user\nDoes this video contain ...<|im_end|>\n<|im_start|>assistant\n` + `Yes<|im_end|>\n` / `No<|im_end|>\n`. It is identical across videos of the same stance.
   - `get_rope_index` places the span on plain text positions (e.g. hate_video_1: P = 2343, positions 783–814, δ = −1560).
   - The scaffold is 290 tokens in every video.
   - The A1 and `all_local` keep vectors differ exactly on [P, P + 32).
   - For HateMM and HCS, the per-window kept visual / speech token counts and the cached length equal Grounder's
     `all_local` records.
3. **Plumbing with stand-ins.**
   - The pre-hook injects the arm's mask at all 36 layers, by keyword and by position. Visible columns equal the keep
     vector, and the query part is causal.
   - `swapped_cache` replaces exactly the span at every layer and leaves the source cache unchanged.
   - `rotate_states` equals the model's `apply_rotary_pos_emb` exactly.
   - `donor_map` is the cyclic next same-sign video.
4. **stage0.py dry run** on Grounder reads, with stand-ins A1 := `base` and A1_other := `verdict_only`. It runs and
   prints the three statistics per corpus and a verdict. The numbers mean nothing: these two stand-ins are nearly
   identical, so S3 ≈ .998 → STOP, as intended.
5. **stage1.py dry run** on Grounder reads, with stand-ins A1 := `late` and A1_other := `shifted`.
   - Raw scoring plus decoding of 5 arms took 89 s on lab1 (5 parallel decodes, 2 BLAS threads each).
   - Grounder `base`, decoded, equals `runs/20260926_twolevel/final_rawkey` exactly on all six numbers.
   - `all_local` decodes to .8940 / .6875 / .7052 and .6980 / .6527 / .5887. These are the reviewer's raw-key numbers.
     The within values equal Grounder's README values (.705164 / .588674), because the key is constant within a video.
   - The gates, the kill rule and the paired CIs print as designed. The verdict of a dry run means nothing.
6. **stage1.py --stage 2 dry run**, run separately, with the r6 DeHate reads `runs/20260927_dehate_external/reads_gridA`
   as the stand-in for both arms. They decode to exactly `runs/20260927_dehate_external/r6_rawkey`
   (.6994 / .1568 / .6431). Raw scoring plus 2 decodes took 182 s on lab1.

Not testable on CPU: the GPU forwards themselves. This covers the hooks on the real attention modules, the KV capture
during the verdict / stance forwards, the in-job replica checks and the timing. The in-job checks of §7 cover these on
the first video of each corpus (some on every video).

## 11. Rule-10 test-read log

**2026-10-25, implementation, before any run.**
- Read the cached r6 reads (`base_gridA`, DeHate `reads_gridA`): the z_video signs, to define donors, and the prefix
  lengths, for the smoke test. These are model outputs, not labels.
- Read Grounder's archived reads `runs/20261002_m1_grounder/r1_full/{base,all_local,late,shifted,verdict_only}`, for
  the replica check and as stand-ins in the dry runs.
- The stage-1 dry run of the smoke test read the test GT (`data/gt_4fps/{HateMM,HateClipSeg}.npz`), through the
  evaluator only, on Grounder's archived reads. Those numbers are already in the Grounder README.
- The stage-2 dry run read `data/gt_4fps/DeHate.npz` the same way, on the r6 DeHate reads. That number is already in
  `runs/20260927_dehate_external/r6_rawkey/metrics.json`.
- No design decision was made from these reads.
- Arms, thresholds and definitions come from review_m1.md §5.3. That reviewer's test reads are logged in
  review_m1.md §7.

**Later entries:** one per stage, written when the stage is analysed.

## 12. Results

None yet.
