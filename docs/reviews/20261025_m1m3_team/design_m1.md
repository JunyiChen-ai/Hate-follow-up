# M1 designer: redesign proposals for the window read (2026-10-25)

Written by the M1 designer. No repo file was edited, no GPU job was started, no hash or checksum was computed.
Scripts and outputs are in `scratchpad/m1m3_novelty/work_m1/` (full path
`/tmp/claude-135258174/-home-jehc223-Hate-follow-up/abae8735-0dba-4355-b09e-130e58ba97bd/scratchpad/m1m3_novelty/work_m1/`).
Every number is development-selected (read on test). Frame metrics come from `src/eval/evaluate.py` (imported); window
labels come from `src/window_level.py`. Decoding uses the unchanged r6 decoder `experiments/20260926_twolevel/twolevel_r2.py`
with `--key raw` (the paper version). My decode of the r6 reads reproduces the charter reference exactly:
HateMM .8955 / .6883 / .7508, HateClipSeg .7137 / .6666 / .6373 (`work_m1/decoded/binding_r6/metrics.json`).
"Within" = within-video macro ROC on 84 / 99 videos. Deltas are HateMM ; HateClipSeg.

## 0. What I ran on CPU, and what it changed

Rule-10 log. Files read: `data/gt_4fps/{HateMM,HateClipSeg}.npz` (only through the evaluator and `src/window_level.py`);
`runs/20260926_glr/base_gridA/predictions.jsonl`; `runs/20261007_m1_streamingtom/placement_controls_main/records/`;
`runs/20261003_m1_contraster/r1_main/checks.jsonl`; `runs/20261003_m1_explorer/r1_main/details/`;
`runs/20260910_spvl/mllm/q3vl-8b/{full,nostance,winonly}`; `runs/20260928_infer/main/reads_{noctx,noframes}`;
`runs/20261014_m1_audio_bound/vl_aligned/vl_r6_aligned`; `runs/20260926_twolevel/final_rawkey/predictions.jsonl`.
No design was chosen by error type. The findings below changed the ranking.

**F1. Re-showing a window's own frames does not give a better raw read. The decoded gain depends on putting each read
on its own window.** (`binding_check.py`, `binding_controls.py`, `binding_permv.py`, `binding_bootstrap.py`)

The arm is `adjacent_native`: the window's own prefix frames are shown again next to its question, with no new image.

| quantity | HateMM | HateClipSeg |
|---|---|---|
| raw visual frame within, r6 → adjacent | .614 → .611 | .546 → .548 |
| raw max(V, S) frame within, r6 → adjacent | .680 → .662 | .610 → .605 |
| decoded within, adjacent − r6 [95% paired CI] | +.022 [+.004, +.045] | +.013 [−.006, +.034] |
| same, without the top 3 videos | +.0095 | +.002 |
| per-video shift only (native order, adjacent level) − r6 | −.004 | +.002 |
| adjacent reads permuted within each video − r6 (mean of 3 seeds) | −.009 | −.039 |
| adjacent, each read moved to the next covered window − r6 | +.002 | −.002 |
| **native** visual reads permuted within each video − r6 (3 seeds) | **−.003** | −.029 |
| lag-1 autocorrelation of visual reads, native → adjacent (median) | .41 → .19 | .54 → .22 |

Reading:
- The re-shown read does not rank windows better on its own.
- It is window-specific. Its value to the time model depends on the right window: wrong assignment costs .031 / .052
  against correct assignment. Moving each read by one window loses the gain.
- The native visual read is close to a whole-video read. On HateMM, shuffling it across a video's windows changes the
  decoded within by only −.003, and neighbouring windows read alike (lag-1 .41 / .54).
- hate_video_329 (raw gain 0, decoded +.60) carries .007 of the HateMM gain. This is trap 1 again.

**F2. The visual question does not address its window.** (`attention_binding.py`, on Explorer's cached attention of the
visual question's last token over the 20 prefix frames, pre-RoPE, mean over heads and layers)
- The attention vectors of different windows of one video correlate at median .9996 / .9993.
- The window's own frames get less than chance mass: median lift .69 / .73, above chance in only 20% / 23% of windows.
- An own frame is the most-attended frame in 7.9% / 5.1% of windows.
- After subtracting the video's mean query (an ICR-style calibration), a window-specific part remains in 88% / 89% of
  windows, but it is about .001 to .002 of the attention mass.
- This matches C41 probe A (frames named correctly 36% / 44%).
- Caveat: the cache excludes the positional (RoPE) term.

**F3. The visual branch is mostly a context read.** (`channel_pure.py`) On a frames-only prefix the visual window-level
within falls from .665 to .537 (HateMM) and from .570 to .512 (HCS).

**F4. Word binding for speech is a raw-read gain.** (`speech_binding.py`; `vl_r6_aligned` is r6 with a forced-aligned
word slice in the speech branch)
- Raw max frame within: .680 → .703 and .610 → .644.
- Decoded (raw key): +.002 / +.002 / +.017 ; −.002 / +.011 / +.029.
- Speech reads were already window-specific: lag-1 .22 / .26.

**F5. Both bindings together, with no new frames.** (`bound_composite.py`: visual from `adjacent_native`, speech from
`vl_r6_aligned`; both were read on the same r6 prefix with isolated branches)

| | pooled ROC / PR / within, HateMM | HateClipSeg |
|---|---|---|
| decoded composite | .8974 / .6906 / .7783 | .7196 / .6826 / .6796 |
| composite − r6 | +.002 / +.002 / **+.028** [.000, +.058] | +.006 / +.016 / **+.042** [+.013, +.073] |
| without the top 3 videos (within) | +.011 | +.028 |
| composite − aligned only (the visual part) | +.011 [−.009, +.034] | +.013 [−.006, +.035] |
| composite − adjacent only (the speech part) | +.005 [−.015, +.027] | +.030 [+.006, +.054] |

All six deltas are ≥ 0, and within is ≥ .01 on both corpora. DeHate has neither read set, so DeHate is not run.

**F6. The depth trajectory adds nothing.** (`depth_check.py`, logit lens of every r6 branch, 18 depths; the final row
matches r6 to 3e-4)
- Window-level within of max(V, S): final .766 / .636.
- Best intermediate depth: L24 .777 on HateMM but .613 on HCS; L28 .759 / .638.
- Commitment depth: .633 / .608.
- Frame raw within at L24: .685 / .580, against the final .680 / .610.
- No depth beats the final layer on both corpora. Lead O2 is closed.

**F7. The whole-video context barely changes the order of speech windows.** (`context_order.py`, old-ASR q3vl-8b,
identical window text)
- Window-level within: full .730 / .610; context-free text-only .727 / .590; no stance .733 / .596.
- Within-video Spearman between full and context-free: .75 / .77.
- 45% / 31% of the full-minus-context-free difference is a per-video constant.

**F8. Few within-eligible videos get a "No" verdict.** (`stance_groups.py`) Only 3 / 84 and 5 / 99 do. A read conditioned
on "act present" instead of the model's own verdict can change at most those videos' within order.

## 1. Novelty risk of the current M1 form (Superposition Prompting, Parallel Context Windows)

From `lit_m1.md` C7, method sections read by the scout:

**Superposition Prompting** (ICML 2024, arXiv 2404.06910) already has the full pattern of a cached shared preamble, one
mutually invisible fork per unit, and a score per fork.
- Each retrieved document forks into its own path, with the query duplicated on each path.
- Each path gets a likelihood-based saliency score.

**Parallel Context Windows** (ACL 2023, arXiv 2212.10947) encodes chunks in parallel without cross-attention.

The systems precedents Hydragen and Prompt Cache cover prefix sharing. Consequences:
- The current M1 cannot claim the topology ("shared prefix + isolated branches"). Superposition must be cited next to
  Hydragen and Prompt Cache in the implementation paragraph.
- What remains claimable is what the shared part holds and what a branch reads under it: the whole video plus the model's
  own verdict, that is, the act.
- In r6 the visual fork holds only a question that names its moment by timestamps. The visual fork reads no unit evidence
  at all (F1, F2), so r6 is not even Superposition's "unit in its own path" form for vision.
- Proposals 1 and 2 move M1 to Superposition's topology for the unit, with the act in the shared part. The difference from
  Superposition must be stated in those terms: the shared part carries the whole video and the reader's own verdict; the
  unit is a time span of the same video, not an independent document; and the read is a decision under the act, not a
  likelihood saliency.

## 2. Proposals, ranked

### Proposal 1: evidence-bound branches (each branch carries its own moment's evidence, encoded after the verdict)

**Mechanism, one sentence.** Every window branch receives the evidence of its own moment inside the branch, after the
verdict turn:
- the visual branch re-encodes the prefix frames whose time lies in the window, right before its question;
- the speech branch receives the words spoken inside the window, by their word times.

The act stays in the shared prefix; the moment moves into the branch.

**Replaces.** Two parts of r6:
- the visual question "look only at the frames whose timestamps fall inside this window" (address by timestamp);
- the proportional word-count slice of long Whisper segments (`src/video_inputs.window_text`).

No new frames. The only new derived quantity is word timing: `data/asr_whisper_aligned` for HateMM and HCS, by
MMS_FA forced alignment. DeHate does not have it yet.

**Named failure answered: the moment is addressed only by a description.** Evidence:
- The visual question's attention over the frames is the same for every window (r = .9996 / .9993), and own frames get
  less than chance (F2).
- The model names its window's frames correctly in only 36% / 44% of windows (C41,
  `runs/20261010_m1_grounded_adjacent/probe_analysis/summary.json`).
- The visual read is a whole-video read: shuffling it within a video costs only −.003 on HateMM, and lag-1 is .41 / .54
  (F1). Without the transcript it is near chance (F3).
- In a causal prefix the frames are encoded before the transcript and the verdict, so visual evidence is never encoded
  under the act. Speech evidence is, because the branch re-tokenises it after the verdict. The proposal makes the two
  branches the same.
- The proportional slice assigns words to the wrong windows at segment boundaries. Aligned words raise the raw read: raw
  frame within +.023 / +.034 (F4).

**Closest prior work, from `lit_m1.md`.**
- Superposition Prompting (ICML 2024): the unit in its own fork. Difference: our shared part holds the whole video and
  the reader's own verdict, and the fork re-presents a span of that same video. It is not used for hate.
- TG-Heads (arXiv 2605.21954, venue unverified, Qwen3-VL-8B): it re-runs the model on the interval found by attention.
  Difference: our interval is the fixed window, and the read is conditioned on the act. It is not used for hate.
- VERA / LAVAD (CVPR 2025 / 2024): they add context to a local read. We add the local evidence to a context read, the
  opposite direction.
- No hate localiser (LELA, MultiHateLoc, SafeLens) has an act-bearing shared context.
- Not in `lit_m1.md` and unverified, for the reviewer to check: Re-Reading (RE2, EMNLP 2024) and Prompt Repetition
  (arXiv 2512.14982, abstract only in `novelty_eval/M1_report.md`).

**Closest past attempts.**
- StreamingTOM placement controls (`adjacent_native`, `adjacent_local`, `prefix_local`; 10-10). They were run as input
  controls and recorded as "input change, user decision".
- C43 follow-up (`vl_r6_aligned`, `vl_text_aligned`; the latter adds new 0.5 fps frames).
- Neither was run as one mechanism, neither had a wrong-binding control, and the combination without new frames was never
  decoded. It is decoded here (F5).

Why outside every closed family:
- M1-f covers restricting or re-weighting access (Grounder, Selector, VGA, RoTE). The re-presentation form is listed there
  as not covered.
- M1-j covers remote content; nothing remote is added here.
- M1-d covers per-window choice of whether to take an extra read; here every window is treated alike.
- M1-k (token compression) and M1-a (question wording) do not apply.

**Rule check.**
- Rule 3: one reader, one read per branch, no ensemble, no per-corpus constant.
- Rule 4: not used in hateful-video work. The STOP risk is case 4, "only prompt layout". The defence is that the evidence
  is unchanged and only its binding to the branch changes, and that the wrong-binding control shows the binding is what
  matters (F1).
- Rule 5: no new frames. Word times are a derived input, and DeHate needs them computed.
- Rule 13: one rule for all corpora. A window with no prefix frame keeps the plain visual question (27% / 34% of windows),
  and a window with no aligned word has no speech branch.
- Rule 14g: claimed as one component, "binding", its removal (back to r6) costs −.028 / −.042 within, and both are ≥ .01.
  Split into parts:
  - the visual part (−.011 / −.013) passes only at the mean (CIs include 0; without the top 3 videos −.001 / +.003);
  - the speech part fails on HateMM (−.005).
  So only the joint binding can be claimed.

**Cost.** No added calls: the same 2 branches per window. Per covered visual branch, +91 tokens per re-shown frame
(1.1 / 0.66 frames per window on average). Measured +4.7 GPU min for HateMM + HCS (StreamingTOM README). Speech re-read
with aligned words takes 6 min, and the one-time forced alignment takes under 15 GPU min for both corpora
(`experiments/20261014_m1_audio_bound/README.md` §7, §11). DeHate (1341 videos): about 20 min of reads plus the
alignment, both estimates.

**Pre-check, written before running.** The CPU proxy was computed (F1, F5). The GPU part is one job that reads both bound
branches together on HateMM, HCS and DeHate. Pass criteria:
- (a) The single-run numbers reproduce the CPU composite within .005 per metric.
- (b) On both main corpora: within ≥ +.01 against r6 and no pooled loss beyond .005. On DeHate: no pooled loss beyond
  .005 and within ≥ r6 − .01.
- (c) Wrong-binding controls in the same job, budget-matched: the visual branch re-encodes the frames of another covered
  window of the same video, and the speech branch gets the neighbouring window's aligned words. Each must decode ≥ .01
  within below the bound arm on both corpora. The CPU proxy for the visual control is −.031 / −.052.
- (d) Fragility flag, reported and not a gate: the gain without the top 3 videos must stay ≥ +.005 on both corpora
  (CPU: +.011 / +.028).

Kill rules:
- If (c) fails, the gain is not binding.
- If (b) fails on DeHate, the method is not uniform-safe.

**Story paragraph (paradigm_framing §3 pattern).**
> A naive act-conditioned reader puts the whole video and its own verdict in one shared context and asks, in an
> isolated branch per moment, whether that moment is part of the act. The branch names its moment only by timestamps,
> and the reader does not resolve them: its visual question attends to the video's frames identically for every
> moment, and it names the frames inside its window correctly for only a third of the windows. The visual read is
> therefore a read of the video, not of the moment, and the frames it should judge were encoded before the reader had
> heard the speech or decided the act. We bind each branch to its own evidence: the moment's frames are encoded again
> inside its branch, after the verdict, and its words are those spoken within it. The act stays in the shared context;
> the moment moves into the branch. Removing the binding costs .028 / .042 within-video AUC; giving each visual
> branch another moment's frames costs .031 / .052.

**Honest estimate.**
- Novelty: moderate as a story, weak as a technique. It is the paper's own thesis, "the act in the context, the moment
  in the read", made literal, with measured lookup-failure evidence. The operation itself is simple, and Superposition is
  close in topology.
- Chance of passing 14g on both corpora as one component: about 60%. The CPU composite already passes; the risks are
  DeHate and a single-run reproduction.
- Main risks:
  - the user or reviewer may rule it an input or layout change (rule 4 case 4, rule 5);
  - the HateMM gain still goes through the time model (raw max within .684 against .680), and the visual part is
    fragile;
  - it needs word times on DeHate.

### Proposal 2: evidence re-addressing in the cache (the model-internal form of proposal 1's visual binding)

**Mechanism, one sentence.** The visual branch reads its window's frames from the shared cache by reference, without
re-encoding them: the cached keys and values of those frame tokens, at all 36 layers, are re-addressed to positions just
before the window question. Pre-rotary keys are re-rotated to the new positions (`src/pre_rotary_memory.py`, token spans
from `src/window_token_regions.py`). Replaces the same visual lookup as proposal 1.

**Named failure.** The same as proposal 1, address by timestamp (F2). It adds one question proposal 1 cannot answer:
does the gain come from binding (where the evidence sits relative to the question) or from encoding the frames after the
act?
- Re-addressed states were computed before the transcript and the verdict.
- Re-encoded frames (`adjacent_native`) are computed after both.

A 2 × 2 design answers it:

| | addressed by timestamp | bound to the question |
|---|---|---|
| encoded before the act | r6 (cached) | **proposal 2, KV re-address (new GPU arm)** |
| encoded after the act | **all 20 frames re-encoded once after the verdict turn, shared (new GPU arm)** | `adjacent_native` (cached) |

**Closest prior work.**
- Position-independent KV reuse (APE, arXiv 2502.05431, and Block-Attention, ICLR 2025, both listed as systems
  precedents in history §6.1.3; Prompt Cache) moves document KV to new positions for speed. Difference: here the copy
  duplicates part of an already-present context next to one question, to bind it.
- Found in the Middle (ACL Findings 2024) recalibrates positional attention globally; here only the window's own span
  moves.
- None was used for hate or video localisation.

**Closest past attempts.**
- Grounder late arm: the same pre-act states with access restricted in the last 9 layers. Branch logits moved only .06,
  and within was −.0006 / −.0001.
- The difference here: no masking, all layers, and the states are moved next to the question, where the RoPE distance is
  small, instead of about 2,000 tokens away.
- The shared post-verdict frame block is close to the Integrator (future-aware frame rows, closed M1-f). It is only a
  control cell here, not a proposal.
- Outside closed families for the same reasons as proposal 1. No new input.

**Rule check.**
- Rules 3 and 13: as proposal 1.
- Rule 4: precedents exist only in systems work. The STOP risk "engineering trick" is answered by the 2 × 2, which turns
  the result into a mechanism finding.
- Rule 5: no input change.
- 14g: removing the re-address (back to r6) must cost ≥ .01 within on both corpora.

**Cost.** No re-encoding. Per covered window, about 100–270 extra cached entries visible to the roughly 90 question
tokens. About the native visual read time (3.6 min for HateMM + HCS). The shared post-verdict block adds about 1,820
tokens of prefill per video, about 3 min. Total under 10 GPU min for both corpora.

**Pre-check, written before running.**
- GT-free first: the lag-1 autocorrelation of the re-addressed visual reads must be < .30 on both corpora (native .41 /
  .54, re-encoded .19 / .22). If not, the copies are not read as the window's evidence; stop.
- With labels, decoded with r6:
  - binding suffices if KV − r6 ≥ .01 within on both corpora;
  - encoding after the act is needed if adjacent − KV ≥ .01 on both;
  - binding is not needed if post-verdict block − r6 ≥ .01 on both;
  - wrong-window KV (another covered window's frames) must fall ≥ .01 below the KV arm.

**Story paragraph.**
> The reader holds every frame of the video in its context, yet a question about one moment cannot find that moment's
> frames: they sit two thousand positions back, and the question's attention over frames is the same whichever moment
> it names. We re-address the moment's cached evidence to its question: the frame states already computed for the
> whole-video read are placed again next to the question that is about them, so the moment is read by reference,
> without new encoding and without new input.

**Honest estimate.**
- Novelty: moderate. It is a model-internal operation, new for hate and for localisation, but its technical relatives
  are systems papers.
- Chance of passing 14g on both corpora: about 25%. Grounder showed that access to the same pre-act states moves logits
  little. If the act-conditioned encoding is what matters, this arm fails by design.
- Main risk: it ties with r6. Even then the run is useful, because it settles which mechanism proposal 1 can claim.

### Proposal 3: act-state transfer (the window read conditioned on the video's own internal act state, not on the whole prefix)

**Mechanism, one sentence.**
- Take the residual-stream state at the verdict answer position of the r6 prefix pass at one middle layer: the "act
  state" of that video.
- Read each window in a short local pass: rules + the window's own prefix frames + its words + its question, with no
  whole-video context.
- At the question's last token, add the act state minus the same position's state on a content-free prefix. This is the
  task-vector / function-vector / ICV form.

Replaces: the whole-prefix attention and the one-word stance turn in the window branch.

**Named failure.**
- The window read inside the full prefix is a read of the video. The visual read is smooth and window-unspecific (F1,
  F2), and the stance turn shifts every window by +2.4 logits with order Spearman .93 (history O4).
- A context-free read loses the act: the 8-MLLM `winonly` arm costs −.021 / −.071 within and −.035 / −.080 pooled ROC.

**Closest prior work, from `lit_m1.md` C4.**
- Task vectors (Findings EMNLP 2023), Function Vectors (ICLR 2024), ICV (ICML 2024), Patchscopes (ICML 2024).
- All of them transfer an in-context task from demonstrations, in text. None is used for video or hate.
- Difference: the transferred state is the video's own verdict state, and it conditions moments of the same video.

**Closest past attempts.**
- Reinforcer (VISTA steering inside the same pass), candidate 46 (a synthetic-text topic direction removed in the branch),
  and the 2026-08 judge-state directions (M1-h).
- None moves the same video's state into another pass.
- Grounder `all_local` (own-window media only, no verdict, all layers) is the no-act control: −.046 / −.049.
- Outside M1-h (direction removal or readout on judge states) and outside M1-f (no mask; a different pass).

**Rule check.**
- Rule 3: same model.
- Rule 4: not used for hate.
- Rule 5: no new input.
- Rule 13: layer and scale declared once. Proposed: layer 18 of 36, scale 1. A 3-point scan {12, 18, 24} is allowed only
  if declared and reported in full.
- 14g: the patched read must beat the unpatched local read by ≥ .01 on one main metric on both corpora.

**Cost.**
- The act state comes from the existing prefix pass (a hook, no extra pass).
- The content-free state is computed once per corpus.
- Each window gets a short local pass of about 250 cached rule tokens + 91 × k frame tokens + about 90 tokens. That is
  about the cost of `adjacent_native`, with no long-prefix attention.
- About 10 GPU min for HateMM + HCS, and about 20 min for DeHate.

**Pre-check, written before running.**
- Stage 0, GT-free, 20 videos per corpus, about 5 min:
  - the patch must change reads window-specifically: within-video sd of (patched − local) > .5 logit, and the per-video
    constant share of that change < .8;
  - the own-video act state must differ from another video's: correlation of the two changes < .9.
  If either fails, the state acts as a verdict scalar; stop.
- Stage 1, full corpora:
  - patched − local ≥ .01 within or pooled on both;
  - patched not below r6 beyond the floors;
  - own act state − another video's act state ≥ .01.

CPU proxy, computed, and it weighs against the proposal:
- For speech, the whole-video context adds little window order (F7: +.003 / +.020 window-level).
- For vision, a read without the transcript is near chance (F3).
- So the act state would have to carry the transcript's information for vision, and has little within-video order to
  restore for speech.

**Story paragraph.**
> A moment read alone does not know the act; a moment read inside the whole video is read as the video. We carry the
> act into the moment as the reader's own internal state at its verdict: each moment is read from its own evidence, in a
> short pass that receives the video's act state, so the act conditions how the moment is read without the rest of the
> video being read again.

**Honest estimate.**
- Novelty: the strongest of the three, with no precedent in hate, VAD or temporal grounding.
- Chance of passing 14g on both corpora: about 10%.
- Main risks:
  - in 2026-08 the dominant judge-state direction was the verdict itself (Spearman .992), so the transferred state may be
    a scalar "Yes" that only shifts reads;
  - the local visual read lacks the transcript the visual branch depends on (F3).
- Run only stage 0 unless it passes.

## 3. Suggested order for the lead

1. Proposal 1 (CPU composite already passes). GPU: one job on HateMM + HCS + DeHate with the wrong-binding controls,
   plus DeHate alignment. About 45 GPU min in total.
2. Proposal 2's two new arms in the same job if possible (under 10 min). They decide which mechanism proposal 1 claims
   (binding vs encoding after the act). This is the part that answers "is it more than layout".
3. Proposal 3 stage 0 only (about 5 min), and stop unless it passes.

## 4. Dropped ideas

| idea | why dropped (closed family, rule, or CPU number) |
|---|---|
| Calibrated attention read of the verdict (ICR, ICLR 2025; TG-Heads) | Inside M1-e: window score from the verdict's dependence on the window. The 08-12 pilot already read verdict attention, which equalled token density (.5252 vs .5182). The only new step is a calibration (rule 4, "pure calibration"). The F2 attention of window questions is window-unspecific (r .9996), and TG-Heads already uses attention with blank-video debiasing on Qwen3-VL-8B, so the novelty would be only "for hate". |
| Act-specific benign counterpart contrast (CEAVAD) | Inside M1-a: TAD r1 topic subtraction (.6159 / .5793), TAD r2 five-way act collapsed to "attack" (80% / 56%), DVD E inert. It is also chosen from an error type (topic confusion), against the user's criterion. Cost: at least 2 reads per window, up to 18 with per-rule counterparts. |
| Binding IDs for "who is targeted" | No CPU check possible, and binding IDs are unshown in VLMs or on noisy ASR. Its neighbours are closed: C44 speaker binding (identity ceiling +.036), DVD T (moves toward the hate definition, loses HCS protocol GT), C24 / C25 reference graphs. |
| Lookback-ratio decomposition of the branch read | A readout or diagnostic of the existing read, not a redesign (the user's no-analysis rule). Next to M1-f (Grounder, Selector). |
| Depth trajectory (fixed layer, depth mean, commitment depth) | F6: no depth beats the final layer on both corpora. DoLa (closed) is the source. |
| Act-hypothesis read (always condition windows on "Yes") | F8: only 3 / 84 and 5 / 99 within-eligible videos have a "No" verdict, so the within room is at most about .01. The two-verdict average is the Marginalizer STOP (MARS, RAMF), and verdicts are saturated. |
| Channel-pure act conditioning (each branch on a single-modality prefix with that prefix's own verdict) | `channel_pure.py`: visual window within .665 → .537 and .570 → .512. Decoded HateMM .8909 / .6833 / .7629 (ROC and PR −.005), HCS .7198 / .6706 / .6360 (within −.001). Fails, and it is near PP-v4 and the modality gate (M3-g). |
| Own-evidence ablation contrast inside the branch (read − read with own evidence hidden) | This is VCD R2 (own-window noise; closed M1-g): R2 HateMM .7537, HCS .6307, raw visual −.070 / −.016. |
| KV-form act transfer (branch sees only its own window plus the verdict-turn KV) | A mask, so inside M1-f. Grounder `all_local` is the same mask without the verdict turn (−.046 / −.049). |
| Ordinal threshold ladder per window (Probe-VAD) | A readout change, inside M1-a, at about 10 Yes/No reads per window (cost rule). |
| Window pointer read (next-token distribution over window indices after the verdict) | Inside M1-c, the model's own statements about where: HVL cited spans .571 / .506; probe A 36% / 44%. |
| Self-anchored re-read (the video's own extreme windows as in-context examples) | Inside M1-b and M1-a: HVL confirmation, a permuted hypothesis works as well, and pairwise C42 equals pointwise (`novelty_eval/M1_report.md` §3). |
| Self-consistency, debate, CoVe | Rule 3 (ensemble) and M1-b. Answers are saturated (2026-08: 82% have entropy ≈ 0). |
| PCW-style shared positions; SimToM past-only context | Branch visibility is closed (DeHate −.031). Integrator covers the future-aware direction. |
| ContextCite / PECoRe attribution | Inside M1-e, and the 09-12 context contrast (.467 / .494). |

## 5. Files

- Scripts: `work_m1/{common,binding_check,binding_controls,binding_permv,binding_pervideo,binding_bootstrap,bound_composite,speech_binding,attention_binding,channel_pure,depth_check,context_order,stance_groups}.py`
- Outputs: `work_m1/out/*.json`, decoded runs `work_m1/decoded/<tag>/metrics.json`, composed read sets `work_m1/reads/<tag>/predictions.jsonl` (127 MB in total; can be deleted after review).
