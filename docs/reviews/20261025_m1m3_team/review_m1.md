# Adversarial review of the M1 proposals (2026-10-25)

Written by the M1-side adversarial reviewer. I wrote none of the proposals.
- No GPU job, no repo edit, no hash or checksum.
- Scripts and outputs are only under `work_review_m1/` (full path
  `/tmp/claude-135258174/-home-jehc223-Hate-follow-up/abae8735-0dba-4355-b09e-130e58ba97bd/scratchpad/m1m3_novelty/work_review_m1/`).
- Every number is development-selected (read on test).
- Decoding uses the unchanged r6 decoder (`experiments/20260926_twolevel/twolevel_r2.py`, `--key raw`). Frame metrics and
  per-video AUCs come from `src/eval/evaluate.py` (`within_video_macro`, called per video for paired bootstraps).
  No evaluator logic was copied.
- My r6 decode reproduces the charter exactly: HateMM .8955 / .6883 / .7508, HCS .7137 / .6666 / .6373
  (pooled ROC / pooled PR / within). DeHate reference .6994 / .1568 / .6431. Deltas are written HateMM ; HCS.

## 0. Verdicts

| proposal | verdict | main reason |
|---|---|---|
| design_m1 #1, evidence-bound branches (re-show the window's frames after the verdict; aligned words) | **DROP as an M1 novelty claim.** As an input change it is now well supported; adopting it is the user's decision. | The binding is real on both corpora (§1). But the operation is "put the window's evidence next to its question". Rule 5 names word timestamps as an input, and re-showing frames is prompt layout with direct precedents (§2, §3). The user already filed both halves as input changes (10-10, 10-14). |
| design_m1 #2 = design_story #1, re-address cached frame states next to the question | **DROP** | Copying cached keys and values while leaving the values unchanged is exactly a re-weighting of attention over the same values (§2). That is the closed family M1-f / M1-g (VGA +.004; Grounder late arm moved logits .06). The copied frame states were computed before the transcript, the rules and the verdict, so the story's sentence "states contextualised by the whole video" is false. ReKV (ICLR 2025) already places cached video keys and values consecutively before the question. |
| design_m1 #3 = design_story #2, act-state reading (own moment + verdict state only) | **RUN PRE-CHECK, mask form A1 only, low prior (about 5%).** Run A2 / the transfer form only if A1 passes. | It is the only proposal whose mechanism is not more or better input. But a new CPU bound (§5.3) shows that restoring each video's level and spread of the reads leaves the all_local branch .018 / .020 within below r6. A1 must therefore carry window-specific information from the verdict turn, and P10, HVL and Grounder `verdict_only` all say it does not. |

No proposal is STRONG.

## 1. Task 1: the two binding tests, reconciled

### What each script did

- **M1 designer, value permutation** (`work_m1/binding_controls.py`, `perm0-2`). It permutes the re-shown read values `a`
  among the covered windows of each video. This destroys two things at once:
  - the binding of the change to its window;
  - the window information the native read already had, which `a` shares.

  `binding_permv.py` applies the same permutation to the native reads. On HCS that alone costs −.030 within (on HateMM
  −.004). So the −.031 / −.052 is a mixture, and on HCS more than half of it is not about the re-show at all.
- **Story designer, change permutation** (`work_story/p4_binding_controls.py`, arm `shuffle`). It keeps the native read
  `v` in place and moves the change `d = a − v` to another changed window of the same video. This is the right test of
  "is the gain tied to its window": it keeps the per-video set of changes and the native order, and breaks only the
  change-to-window assignment. But it was run with **one permutation seed**.

### Re-run (`build_controls.py`, `extra_seeds.py`, `analyze_controls.py`, `seed_distribution.py`)

Arms and seeds:
- change shuffle: 30 seeds for the visual branch, 10 for speech and for the composite;
- per-video mean shift;
- spread-only noise: v + per-video mean(d) + Gaussian noise with sd(d), 5 seeds;
- the M1 value permutation for re-shown and for native reads, 5 seeds each;
- the same change test for the aligned speech slice (`vl_r6_aligned`) and for the composite.

Each arm was decoded. Paired per-video bootstrap over the 84 / 99 within-eligible videos.

| decoded within | HateMM | HCS |
|---|---|---|
| r6 | .7508 | .6373 |
| re-shown frames (`adjacent_native`) | .7732 | .6501 |
| change shuffle, 30 seeds: mean (sd) [min, max] | .7539 (.009) [.736, .770] | .6299 (.007) [.612, .645] |
| story designer's single seed | .7526 | .6479 (above all 30 of my seeds; about 2.6 sd) |
| per-video mean shift (P4 `shift`) | .7468 | .6392 |
| spread-only noise, 5 seeds | .7478 | .6214 |
| **re-shown − change shuffle (30-seed mean), paired [95% CI]** | **+.019 [+.002, +.039]** | **+.020 [+.003, +.039]** |
| same, without the top 3 videos | +.007 | +.010 |
| re-shown − r6 [95% CI] | +.022 [+.003, +.045] | +.013 [−.006, +.034] |
| re-shown − r6, without the top 3 videos | +.0095 | +.002 |
| M1 value test: (re-shown − its permutation) − (r6 − its permutation) | +.026 [+.009, +.047] | +.020 [+.001, +.042] |

Raw window order, before the decoder (max curve, sole evaluator):
- r6 .680 / .610; re-shown .662 / .605; change shuffle .647–.664 / .578–.597.
- Visual curve: r6 .614 / .547; re-shown .611 / .548; shuffle .565–.577 / .514–.535.

Aligned speech slice and the composite:

| decoded within | HateMM | HCS |
|---|---|---|
| aligned − r6 | +.017 [−.004, +.041] | +.029 [+.008, +.051] |
| aligned − its change shuffle (10 seeds) | +.017 [−.000, +.036] | +.023 [+.008, +.038] |
| composite − r6 | +.028 [−.000, +.058] | +.042 [+.014, +.073] |
| **composite − composite change shuffle (10 seeds)** | **+.032 [+.008, +.060]** | **+.045 [+.021, +.068]** |
| same, without the top 3 videos | +.018 | +.035 |

### Which test answers the question, and the answer

**The question is: is the gain window-specific binding, or a per-video shift?** The change shuffle answers it, run
with many seeds, with the per-video mean-shift and spread-only arms next to it. The value permutation answers a
different question: does the re-shown read carry window information at all?

**Answer, HateMM.** The gain is tied to its window.
- The re-shown read beats all 30 change shuffles: +.019 [+.002, +.039].
- The per-video mean shift loses (−.004), and spread alone loses (−.003).
- Two weaknesses:
  - the gain is fragile: +.0095 over r6 without the top 3 videos, with hate_video_329 the largest;
  - it exists only after the time model: raw max within falls .680 → .662.

**Answer, HCS.** The gain is also tied to its window, which corrects the story designer's "83% survives".
- That number came from one seed at the top of the shuffle distribution. Over 30 seeds the shuffle mean is −.0075 below
  r6, and the re-shown read beats every seed: +.020 [+.003, +.039].
- Spread alone loses (−.016, CI below 0), so the gain is not a read-distribution effect either.
- But the gain over r6 on HCS is itself not distinguishable from zero: +.013 [−.006, +.034], and +.002 without the
  top 3 videos.

**One more fact, true on both corpora.** The real changes order windows better than shuffled changes at the raw level,
but no better than the native read.
- They carry window-specific signal and about as much noise.
- The time model turns them into within gain. A plausible reason is that the re-shown visual read is far less smooth
  across neighbouring windows (lag-1 .41 / .54 → .19 / .22).

**Composite.** Frames plus aligned words pass the change-shuffle test on both corpora with CIs above zero:
+.032 / +.045. This is the strongest binding evidence on disk. It does not exist for DeHate, which has neither read
set.

**Method lesson for every future control.** One shuffle seed has an sd of .007–.009 in within. That is the size of the
.01 gate, so single-seed controls (P4, and P6 in `design_story.md`) cannot be read at that level.

**Consequence for design_m1 #1's control (c).** "The visual branch re-encodes another covered window's frames" is a
value swap on the GPU. It would pass on HCS even without any binding, because swapping native reads already costs
−.030. Replace it with the change-shuffle criterion in §5.1.

## 2. Task 2: rule 5 and rule 4 exposure, and whether a non-input version exists

**#1, evidence-bound branches: high exposure.**
- The speech half is forced-aligned word times. Rule 5 lists "词级 ASR 时间戳" by name as an input that is not novelty.
- The visual half shows frames that are already in the prefix a second time, next to the question. That is a prompt
  layout change. It sits in rule 4's "pure engineering" class, even though "layout" is not literally on the list.
- Bundling two input operations under one name ("binding") so that 14g passes jointly is what the story designer itself
  lists as renaming.
- A reviewer would write: "they repeat the relevant frames and use word timestamps". Prompt Repetition (arXiv
  2512.14982) makes the exact argument design_m1 uses, that a causal model's early tokens cannot see later ones.
- The binding evidence of §1 makes it a sound input change. It does not make it a mechanism.

**#2, re-address cached frame states: no input exposure, but it is attention steering.**
- No new content enters. But the copied entries carry the *same value vectors* as the originals, with only their keys
  re-rotated. So at every layer the question's output is a mixture of the same values, where each own-window frame
  token's weight becomes exp(s) + exp(s′) instead of exp(s).
- That is a query-dependent positive attention bias on the window's frame tokens: PASTA / PAI / VGA in form, and RoPE
  distance reduction (CCA) in motive.
- The values are also the least useful ones. In the r6 prefix the frames come before the transcript, the rules and the
  verdict (history §5.1). So the copies are blind to the speech and to the policy, and F3 shows the visual read depends
  on the transcript.
- A reviewer would see "attention re-weighting to the window's frames". The repo would see M1-f / M1-g.

**#3, act-state reading: no input exposure.**
- It removes context and keeps a state. Rule 5 does not apply.
- Rule 4 does not STOP it: no hate work uses it (§3c), and it is not ensemble, calibration or engineering.
- Its risk is performance, not rules.

**Is there a binding mechanism that is not "more or better input"?** I looked for one concretely:
- KV copying: attention re-weighting, closed (above).
- Addressing by position: give the window's question the time coordinate of its window, so RoPE does the lookup. This
  is RoTE (candidate 29), which bound speech keys and the speech question by interval rotary, and lost on all metrics.
- Re-encoding after the act: prompt repetition, an input layout.
- A contrast between the read with its own evidence and the read with that evidence hidden: VCD R2, closed M1-g.

So within "binding" no non-input mechanism remains outside a closed family. The only non-input mechanism in scope
is #3. It tests a different thesis: that the act reaches the moment through the verdict state.

One framing could lift #1 above "input" in the story, without changing its rule-5 status. M2's emission model treats
the reads of neighbouring windows as independent given the state. The timestamp-addressed visual read violates that,
because it is a smooth whole-video read (lag-1 .41 / .54). Binding restores it (.19 / .22), and the gain needs the
change on its own window. This is a story about what M1 must deliver to M2. The operation is still re-encoding.

## 3. Task 3: novelty search (what I searched and found)

Searches (WebSearch, standard mode, 2026-10-25):
- MemVR / re-injecting visual tokens;
- concentric causal attention and RoPE decay;
- ReKV positions;
- Prompt Repetition;
- "duplicate visual tokens closer to the question";
- video LLM "look again / re-watch" frames;
- vision tokens lost in the middle;
- position-independent KV caching (EPIC, KVLink, APE);
- attention sorting;
- MLLM re-reading of image tokens;
- Take-along Visual Conditioning;
- Look-Back;
- multimodal task vectors;
- steering or task vectors for hate speech or memes;
- MLLM internals in hateful video;
- gist tokens;
- Star Attention;
- global-context vector injection in video LLMs;
- Superposition Prompting;
- KV cache steering;
- I2CL;
- hateful-meme detection with probing or intervention;
- training-free hateful video localisation in 2026.

ReKV's method page was read for how positions are handled.

**(a) Re-placing or re-attending cached visual tokens next to a query.**

- **ReKV** (ICLR 2025, arXiv 2503.00540; method page read). Retrieved video KV caches are the context of the question
  and are "treated as regular consecutive tokens", so they sit right before the question with new positions. This is the
  closest published form of #2.
  - Differences: selection by the window's time instead of by similarity, and the whole context stays visible.
  - The repo's candidate 37 already built this machinery (LOCAL blocks packed after the prefix with re-rotated pre-RoPE
    keys) with new 0.5 fps frames. Its LOCAL-only arms L0 / HL0 carried all its gain.
- **Prompt Repetition** (Leviathan, Kalman, Matias, Google, arXiv 2512.14982, Dec 2025, no venue). Repeat the input so
  that each token can attend to every other in a causal model. This is the same reason design_m1 gives for re-encoding
  frames after the verdict. Closest to #1's visual half and to #2's "encoded after the act" cell.
- **RE2** (EMNLP 2024): re-reading the question.
- **Take-along Visual Conditioning** (ACL 2025, arXiv 2503.13360): re-introduces the image at later reasoning stages;
  its test-time part is training-free.
- **MemVR** (ICML 2025, arXiv 2410.03577): re-injects visual tokens as FFN key-value memory at a middle layer when the
  model is uncertain ("look twice").
- **Concentric Causal Attention** (NeurIPS 2024, arXiv 2410.15926): reorders visual tokens to shorten the RoPE distance
  to the instruction, because long-term decay causes hallucination (trained).
- **FarSight** (arXiv 2505.16652) and **DIPE** (arXiv 2603.10863): visual-text RoPE distance bias.
- **Attention sorting** (arXiv 2310.01427): reorder the context by attention so the relevant parts sit near the query.
- **Found in the Middle** (ACL Findings 2024): positional attention calibration.
- **PASTA** (ICLR 2024): post-hoc attention steering toward chosen spans. This is the family #2 reduces to.
- **Systems precedents** for re-rotated KV reuse: EPIC (ICML 2025), CacheBlend, APE, Block-Attention.

**(b) Masking a branch to a local span plus a global summary state, and state transfer.**

- **Gist tokens** (NeurIPS 2023, arXiv 2304.08467): a mask after which later tokens see only the gist positions, so the
  rest of the prompt reaches them only through a summary state. This is A1's mask idea, trained there and training-free
  here.
- **Star Attention** (ICML 2025, arXiv 2411.17116): local blocks each prefixed by a shared anchor block, training-free.
- **StreamingLLM** (ICLR 2024): attention sink plus local window.
- **Longformer / BigBird**: local plus global tokens.
- **Parallel Context Windows** (ACL 2023) and **Superposition Prompting** (ICML 2024): isolated branches over a shared
  prefix. These are already the novelty risk for the current M1.
- **State transfer**: task vectors (Findings of EMNLP 2023), function vectors (ICLR 2024), ICV (ICML 2024),
  **I2CL** (ICLR 2025, arXiv 2405.14660: a context vector injected into the query pass's residual stream),
  **Multimodal Task Vectors** (NeurIPS 2024, arXiv 2406.15334), "VLMs create cross-modal task representations"
  (ICML 2025, patching), **KV cache steering** (arXiv 2507.08799), Patchscopes (ICML 2024).
- What #3 / A2 would add: the transferred state is the same instance's own decision, used to condition parts of the same
  input for localisation. All the above transfer a task from demonstrations or steer style.

**(c) Use in hateful video detection or localisation.** None found.
- Hate localisers and detectors found again: LELA (arXiv 2602.09637), MultiHateLoc (WWW 2026), HVGuard (EMNLP 2025),
  MARS, RAMF, SafeLens.
- They use prompting, captions, chain-of-thought, debate or training. None uses KV re-placement, attention masks, task
  vectors or state patching.
- Rule-4 STOP case 1 does not trigger for any proposal. The risk for #1 and #2 is rule 4 case 4 and closed families,
  not prior hate use.

## 4. Task 4: closed-family check against history.md

**#1, re-shown frames plus aligned words.**
- Not inside a closed family. History §2.3 / O1 lists "binding" as open.
- But both halves were already run and filed: `adjacent_native` (10-10) and `vl_r6_aligned` (10-14), both "input
  change, user decision".
- Not M1-j: nothing remote is added.

**#2, KV re-address.**
- **Inside M1-f / M1-g by construction.** It re-weights attention over unchanged values (§2).
- Nearest repo results:
  - VGA / Highlighter (candidate 15) added salience-weighted in-window image values in layers 4–17: +.004;
  - Grounder `late` restricted the last 9 layers to own-window media, with the same pre-act states: logits moved .06,
    and `late` equalled `shifted`;
  - RoTE lost all metrics with positional addressing.
- The KV packing code is candidate 37's.

**#3, mask form A1.**
- Inside M1-f, but a never-run access set. Precisely:
  - Grounder `all_local` lets a branch attend at every layer to the scaffold, its own window's prefix frames and
    transcript tokens (proportional rule), and its own query tokens.
  - It blocks other media and the global question/answer turn.
  - A1 unblocks exactly that one span (the verdict question, the model's answer and the one-word stance turn) at all
    36 layers. Everything else is identical.
- Two neighbouring cells were run:
  - `verdict_only` blocks only the verdict, in the last 9 layers, with all media visible. It equals base, so late
    direct access to the verdict does nothing while the media are visible.
  - `all_local` blocks both: −.046 / −.049.
- So A1 is one more cell of Grounder's mask factorial. **A1 can be a falsifier, never the claim.**
- Two cautions:
  - the own-window prefix states A1 sees were themselves computed with attention to everything before them, so A1 is
    not "only the moment";
  - the stance turn is the model's own one-word answer, already in r6, so it does not reopen M1-b.

**#3, transfer form A2 / design_m1 #3.**
- Not run in any form. Nearest closed results:
  - M1-h: the dominant judge-state direction is the verdict itself, Spearman .992;
  - M1-b: a permuted, other-video hypothesis works as well as the video's own;
  - story P10: the own stance acts on window reads as a per-video gain, R² .90–.92.
- All three predict that the state carries a scalar.

## 5. Task 5: verdicts and pre-check protocols

### 5.1 design_m1 #1: DROP as a novelty claim. If the user adopts it as an input change, run this confirmation

All thresholds are written now:
- **Arms**, one GPU job on one lab 5090:
  - r6 replica, which must reproduce `base_gridA` exactly;
  - composite: re-shown frames after the verdict, plus the aligned word slice;
  - on HateMM, HCS and DeHate.
  - DeHate needs MMS_FA word times first (under 15 GPU min for HateMM + HCS; DeHate is about 4× the videos).
- **Controls on CPU** from the run's own reads (no extra GPU):
  - change shuffle, 30 seeds, visual and speech changes moved within the video;
  - per-video mean shift;
  - spread-only noise.
- **Pass, all of:**
  - composite − r6 within ≥ +.01 on HateMM and HCS;
  - pooled ROC and PR ≥ r6 − .005 on all three corpora (HateMM .8905 / .6833, HCS .7087 / .6616, DeHate .6944 / .1518);
  - DeHate within ≥ .6331;
  - composite − (30-seed change-shuffle mean) ≥ .01 within on HateMM and HCS, with the paired CI lower bound > 0;
  - flag, not a gate: without the top 3 videos ≥ +.005.
- **Do not use** a GPU "another window's frames" arm as the binding test. If one is run anyway, its threshold must be
  (composite − wrong) − (r6 − r6 with native reads permuted) ≥ .01.
- **Cost**: the designer's estimate, about 45 GPU min including DeHate alignment.
- **Novelty if it passes**: none claimable beyond application. Cite Prompt Repetition, RE2, TVC and ReKV.

### 5.2 design_m1 #2 / design_story #1: DROP

No protocol is offered. If the lead runs the 2 × 2 anyway as a mechanism question for #1, two points:
- its GT-free gate (lag-1 < .30) is good;
- its binding control must be the change shuffle, not a value swap.

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

## 6. Other errors found in the proposals

- **design_story #1, mechanism text**: "copied states ... as the whole video contextualised them". False: in the r6
  prefix the frames precede the transcript, the rules and the verdict. design_m1 #2 states this correctly.
- **design_story P4**: a single-seed shuffle, reported as "83% of the HCS gain survives". Over 30 seeds, none survives
  (§1).
- **design_m1 F2**: the identical attention across windows (r = .9996) is measured pre-RoPE, where the queries of
  different windows differ only in a few number tokens. It is partly true by construction and does not show that
  positional lookup fails. The valid lookup evidence is C41 probe A (36% / 44%).
- **design_m1 #1, 14g**: "binding" is claimed as one component made of two different operations, one of which (the
  visual half) does not pass alone at the CI level. A reviewer will split them.
- **design_m1 #1, control (c)**: not diagnostic on HCS (§1).

## 7. Rule-10 log and files

- **Read**:
  - the reads of `runs/20260926_glr/base_gridA`, `runs/20261007_m1_streamingtom/placement_controls_main/adjacent_native`,
    `runs/20261014_m1_audio_bound/vl_aligned/vl_r6_aligned` and `runs/20261002_m1_grounder/r1_full/{base,all_local}`;
  - `work_story/decoded/shuffle/metrics.json`;
  - test GT `data/gt_4fps/{HateMM,HateClipSeg}.npz`, only through `src/window_level.gold` and the evaluator.
- **No design decision of the method** was made from these. They changed the verdicts:
  - #1's binding is supported;
  - the HCS conflict is resolved;
  - A1's prior is lowered.
- **Scripts**: `build_controls.py` (arms), `extra_seeds.py` (20 more visual shuffle seeds), `analyze_controls.py`,
  `seed_distribution.py`, `act_state_bound.py`.
- **Outputs**: `out/controls_summary.json`, `out/seed_distribution.json`, `out/act_state_bound.json`.
- **Decoded runs**: `decoded/<arm>/metrics.json`.
- **Composed read sets**: `reads/<arm>/` (about 750 MB together with `decoded/`; can be deleted after the lead's
  synthesis).
