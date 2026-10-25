# Cross-module and story design: M1 / M3 (2026-10-25)

Written by the cross-module / story designer of the M1 / M3 novelty team (charter.md in this folder).
- No repo edits and no GPU jobs. No hashes or checksums were computed.
- Every number is development-selected (read on test).
- All comparisons are against the charter's current-method numbers, r6 with the raw key:
  - HateMM .8955 / .6883 / .7508;
  - HateClipSeg (HCS) .7137 / .6666 / .6373;
  - DeHate .6994 / .1568 / .6431.
  - Format: pooled ROC / pooled PR / within.
- New CPU pre-checks: P1–P10, scripts and JSON outputs in `work_story/` (paths in §8). Metrics come from
  `src/eval/evaluate.py` (`pooled`, `within_video_macro`) or from `src/eval/evaluate_four_datasets.py` through the
  unchanged `twolevel_r2.py`. No evaluator logic was copied.

## 0. Summary

**Ranked proposals** (§3). Both are M1 proposals that cross a module boundary. Neither has a CPU result that predicts a pass, so both are offered as cheap GPU falsifiers.

| rank | proposal | module that gains a claim | boundary | GPU pre-check | chance to pass 14g on HateMM and HCS with its control | novelty if it passes |
|---|---|---|---|---|---|---|
| 1 | Moment re-addressing: the visual branch gets its own window's cached frame states again, next to its question | M1 | M1 → M2 (the gain is realised through the time level) | about 15 min | about 15% with the wrong-window control; about 40% without it | low to moderate |
| 2 | Act-state reading: a window branch sees only its own moment plus the verdict turn | M1 | M3 → M1 (the video-level decision becomes the only route by which the act reaches a moment) | about 15 min (mask form) | about 10% | high |

**M3.** No cross-module redesign of M3 survives on the current reads. All three examples in the brief were checked (§2):
- the video level as M2's top latent;
- M3 replaced by an M2 quantity;
- a counter-hypothesis key.

Each one loses beyond the noise floor on at least one main corpus.

**New facts from this round** that change the story (§1, §4):
1. **The model's own "Yes" acts on its moment reads as a per-video gain and offset.** In verdict-Yes videos, z(w | own stance) is close to 1.2 · z(w | no stance) + c_v.
   - Median within-video R² is .92 / .90 / .92 (HateMM / HCS / DeHate) (P10).
   - The lift does rank windows (AUC .72 / .59 / .62, P1), but a within-video shuffled lift does as well (P3).
   - So the stance turn adds no window-specific information.
2. **The whole-video context carries moment-specific information on HateMM only** (P6; old-ASR joint-branch reads).
   - HateMM raw within rises from .628 (window alone) to .700 (with context). A control that adds only the video's mean context effect gets .628.
   - On HCS, raw within is .576 alone vs .573 with context. Decoded within is .563 alone, .579 with context, and .584 with the offset-only control: the HCS context gain is a per-video offset acting through the time level.
3. **Re-showing a window's prefix frames next to its question (`adjacent_native`) does not improve the read itself** (P4, P5).
   - Raw frame-level within: HateMM −.018 [−.044, +.003], HCS −.005.
   - Its decoded gain (+.022 / +.013 within) appears only through the time level.
   - On HateMM the gain is lost when the read changes are moved to other windows of the same video (+.002). On HCS 83% of it survives that move (+.0106 of +.0128).

## 1. Target claim sentences (question 1)

Write the claim first, then ask what would make it true.

**M1 target claim.**
> Each moment is read as a step of the act: the reader judges a window with the whole video's act in view, so a
> window that only refers back to the act is read as part of it.

What would make it true: the act context must change moment reads one by one, not only through a per-video level or gain. Current evidence:

| route by which the act reaches a moment | evidence | moment-specific? |
|---|---|---|
| transcript + frames in the shared prefix | HateMM raw within .628 → .700; offset-only control .628 (P6) | yes, on HateMM |
| same, on HCS | raw .576 → .573; decoded offset-only .584 ≥ context .579 (P6) | no: per-video offset through M2 |
| the model's own verdict (stance turn) | per-video gain 1.2 + offset, R² .90–.92; shuffled-lift control equals the real lift (P3, P10) | no |
| re-showing the window's own frames | raw within −.018 / −.005; HCS decoded gain survives shuffling (P4, P5) | HateMM only, and only through M2 |

So the claim is true on HateMM, carried by the transcript context. On HCS it is false, and the verdict turn plays no part in it on either corpus. A mechanism that makes the sentence true on both corpora is not yet known. Proposals 1 and 2 are the two remaining candidates.

The paradigm_framing M1 ablation row needs a caution. The single-window row (−.021 / −.071) compares dual-branch full reads with text-only joint-branch window reads. It therefore mixes branch type and frames with context. In the joint-vs-joint comparison (P6), the HCS context effect is +.015 decoded, and an offset-only control reproduces it. Report this to the user before the row goes into a paper.

**M3 target claim.**
> Videos are ordered by whether they carry the act and by how much of their running time it covers.

What would make it true: the video term must measure coverage, and it must do so with a mechanism rather than a mean.
- Today the verdict says "whether".
- The window mean says "how much" weakly: Spearman with the GT share among hateful videos is .416 / .533.
- Every model-based alternative tried loses (§2.1, §2.3):
  - posteriors of the segment model;
  - share reads;
  - prevalence estimators;
  - products.

So the sentence is true only in its naive form, classify-and-count. On the current reads, no mechanism-level M3 exists.

## 2. Cross-boundary redesigns (question 2)

### 2.1 The video level as the top latent of M2's model, with joint inference

Repo record:
- Round-1 joint model: HateMM pooled PR .5499 vs .6888.
- Bayes video posterior as the key: .8230 / .5576 and .7041 / .6311.
- ICC-effective reads: PR −.013 / −.029.
- Revisable shared random effect: pooled .569 / .360 and .514 / .490; the random effect absorbed 80% / 68% of the verdict variance.
- Segment-model Bayes factor key: pooled −.034 / −.107.

New this round, two forms declared before running:
- **P8. M2's own posterior, with each modality's reads counted as one testimony.** The window term of the log-odds is divided by that modality's read count. Rationale: the reads of one video share its context.
  - Pooled HateMM .8930 / .6567, HCS .7077 / .6484, DeHate .6779 / .1543 (PR −.032 / −.018 / −.003).
  - Video AUC is higher on HateMM (.933 vs .927), but frame pooled PR is lower.
  - Spearman with the raw key is .977–.983.
- **P9. One probability for the frame: P(V = 1) × P(h_t | V = 1), with P(V) from P8.** This needs no calibration.
  - HateMM .8686 / .6503 (−.027 / −.038), HCS .7371 / .6681 (+.023 / +.0015), DeHate .6766 / .1375 (−.023 / −.019).
  - This is the old product pattern again: HCS ROC gains, HateMM loses.
- Plumbing: the imported EM reproduces the logged r6 log-likelihoods exactly: −8621.3714 / −8107.3928 / −40969.7841 (`work_story/p8_em_check.log`).

Verdict: closed. Any likelihood-based aggregation either overcounts the reads or, once tempered, re-weights the verdict away from the 1:1 sum that works. The joint form would give M3 a sentence ("one model for both levels"), but it costs .02–.04 pooled PR. Claiming the current K under that name would be renaming.

### 2.2 The window read conditioned on an act state the model itself infers

- **Text form: closed** (history M1-b).
  - HVL confirmation: 88% "present" inside the model's own cited spans vs 44% outside.
  - A permuted (other video's) hypothesis works as well as the video's own.
- **Verdict-as-context form: closed this round** (P1–P3, P10).
  - The own stance is a per-video gain.
  - A context-aware-decoding style contrast, z(own) + α(z(own) − z(none)), lowers window AUC as α grows: HateMM .781 → .764 at α = 1, HCS .631 → .627, DeHate .669 → .664.
  - The shuffled-lift control matches the real lift.
- **Hidden-state form: untested.**
  - Grounder's `all_local` arm is the nearest run. The branch sees only its own contextualised window tokens, at all layers, with the verdict turn blocked. It loses −.046 / −.049 within.
  - A form that keeps the verdict turn visible to the local branch was never run. This is Proposal 2.
  - P10 and the HVL permutation predict that it adds little beyond a gain.

### 2.3 M3 replaced by a quantity M2 already infers

Already in the repo:
- expected-hate-fraction key: .8632 / .6180;
- M2 implied share: ρ .207 / .418;
- M2 posterior mean share: ρ .136 / .542;
- decoder probability as the frame score: HateMM ROC −.041.

New: P8 and P9 above. Closed.

### 2.4 Counter-hypothesis key (new, the M3 side of the stance contrast)

P1 found a real video-level signal.
- Among verdict-Yes videos, the mean stance lift is larger in benign videos than in hateful ones.
- −(mean lift) ranks hateful above benign at AUC .661 / .731 / .585.
- In repo terms: on HateMM, false-positive videos are lifted +3.16 vs +2.01 for true positives.

But every way to use it without a fitted weight loses on HCS (P2):

| key | HateMM | HCS | DeHate |
|---|---|---|---|
| verdict + mean read without the stance turn | +.002 / +.002 | −.011 [−.021, −.003] / −.010 [−.017, −.001] | −.001 / +.004 |
| verdict + mean of a linear forced-"No" proxy | +.003 / +.010 | −.022 / −.023 (CI excludes 0) | −.005 / +.004 |

Deltas are pooled ROC / PR against the raw key on the same reads. HateMM and HCS use the old-ASR pair (`q3vl-8b/full` vs `nostance`, raw-key reference .8922 / .6844 and .7135 / .6674), because no fixed-ASR no-stance read exists for them. DeHate uses fixed ASR.

The reason is that the window mean without the stance turn orders HCS hateful videos by share worse (.464 vs .520). Closed under the HCS protocol GT, the same pattern as DVD T and candidate 49.

## 3. Proposals

### Proposal 1 (rank 1): Moment re-addressing

**Mechanism.** Each visual window branch gets its own span of the shared context again.
- The cached keys and values of the window's prefix frame tokens are copied next to its question, with the positions re-assigned. These are the states the whole-video pass already computed.
- No image is encoded again, and nothing new enters the input.
- The question therefore reads its moment as the whole video contextualised it.
- The speech branch already works this way: its question carries the window's words.

**Module that gains a claim: M1.** The claim would be: "the act is read once from the whole video; each moment is read where it is asked, with its own evidence at the question."

**Named failure answered.** The visual question names a moment by its timestamps and asks the model to look among 20 prefix frames for "the frames whose timestamps fall inside this window".
- The model names the right frames in only 36% / 44% of covered windows (C41 probe A, `runs/20261010_m1_grounded_adjacent/probe_main/`).
- 31% of windows hold no prefix frame at all (2263 / 7359, P7). For those windows the visual read relies on context, and dropping these reads costs within −.0025 / −.0088 (P7).

**Evidence on disk** (`adjacent_native`: re-show by re-encoding; raw key, P4):

| arm | HateMM | HCS |
|---|---|---|
| r6 | .8955 / .6883 / .7508 | .7137 / .6666 / .6373 |
| re-show (`adjacent_native`) | .8952 / .6879 / .7732 | .7221 / .6724 / .6501 |
| same per-video mean change on the same windows | .8949 / .6882 / .7468 | .7211 / .6702 / .6392 |
| the read changes shuffled among the video's changed windows | .8945 / .6878 / .7526 | .7209 / .6694 / .6479 |

- The re-show passes 14g as a component: within +.022 / +.013, with no pooled loss.
- Raw frame-level within (before M2) does not improve (P5): HateMM .680 → .662 (−.018 [−.044, +.003]); HCS .610 → .605.
- HateMM: the gain needs the change on its own window (shuffle +.002). This is a time-level effect that depends on which window changed.
- HCS: the gain mostly survives the shuffle. This is a read-distribution effect, not tied to the moment.
- On HCS, the extent term of the key rises (ρ with the share .533 → .575), but the mean-shift control keeps the same rise (.572). So the HCS pooled gain is a per-video level effect.

**Closest prior work and the difference.**
- RE2, "Re-Reading Improves Reasoning in LLMs" (EMNLP 2024 main, arXiv 2309.06275), and Prompt Repetition (arXiv 2512.14982, no venue) repeat the question or the prompt.
- Superposition Prompting (ICML 2024, arXiv 2404.06910) duplicates the query onto per-document paths. The documents live only in their paths.
- Prompt Cache (MLSys 2024, arXiv 2311.04934) and CacheBlend (arXiv 2405.16444; venue not verified this session) reuse cached KV at new positions for speed.
- The difference here: each isolated query gets a copy of its own span of a context that stays shared and visible. These are states contextualised by the whole video, not raw content. Nothing in hate, VAD or TAL does this (lit_m1 §F).

**Closest repo attempts, and why this is outside the closed families.**
- `adjacent_native`: the re-encoding form. It was only ever an input-placement control and was never claimed.
- Highlighter / VGA (candidate 15) added salience-weighted in-window values in layers 4–17: +.004. That re-weights values (closed family M1-g); it does not re-present them.
- Grounder (candidate 1) restricted access (closed family M1-f).
- History §2.3 and §3.1 O1 record re-presentation as open.

**Rule check.**
- Rule 3: no ensemble, no post-processing.
- Rule 4: the main risk is case 4 ("prompt layout"). The KV form is not a prompt change, but the claim survives review only if the wrong-window control passes.
- Rule 5: no new input.
- Rule 13: one uniform rule.
- Rule 14g: needs ≥ .01 on both corpora for the component. A mechanism claim also needs the wrong-window control.

**Cost.**
- No new model calls.
- No image encoding.
- About 100 added branch tokens per covered window, for the copied frame states. `adjacent_native`, which re-encodes the frames, cost +4.7 min for both corpora.
- Positions are re-assigned with `src/pre_rotary_memory.py`. Token spans come from `src/window_token_regions.py`.

**Falsifiable pre-check.**
- CPU (done): P4 and P5 above. They predict that the wrong-window control fails on HCS.
- GPU (about 15 min on a lab 5090, 333 videos), three arms:
  - (a) KV re-addressing;
  - (b) wrong-window re-addressing: window i's question gets the copied states of a covered window at least 3 windows away, with the same token count;
  - (c) r6 replica.
- Pass, all of:
  - (a) − (c) within ≥ .01 on both corpora, with no pooled loss beyond .005;
  - (a) − (b) within ≥ .01 on both corpora;
  - raw frame-level within of (a) not below (c) on either corpus.
- Kill: (a) − (b) < .01 on either corpus. The claim would then be layout, not binding.

**Story paragraph.**
> The act is read once, from the whole video; a moment has to be read where it is asked. A question that names a
> moment by its timestamps leaves the reader to find that moment among two thousand tokens of video, and it names the
> right frames in fewer than half of the windows. Each visual branch therefore re-addresses its own moment: the states
> of the window's frames, as the whole video contextualised them, are placed again next to its question, the way the
> speech question already carries the window's words. Removing re-addressing costs within on both corpora;
> re-addressing another window's frames does not give the gain.

The last sentence is the claim the pre-check has to earn. On HCS the CPU proxy says it will not.

**Honest estimate.**
- Novelty: low to moderate. It sits near re-reading and prompt repetition; the KV form and per-branch addressing are the difference.
- Chance that (a) reaches within ≥ .01 on both corpora: about 40%.
- Chance that (a) − (b) ≥ .01 on both corpora: about 15%.
- Main risk: the HCS gain is a read-distribution effect, so the claim reduces to layout, which the user calls engineering.
- If only HateMM passes the control, the component can stay in the method, but no M1 novelty can be claimed for it.

### Proposal 2 (rank 2): Act-state reading

**Mechanism.** At every layer, a window branch may attend only to:
- its own window's prefix states (its frames and words, as the whole-video pass contextualised them);
- the policy scaffold;
- the verdict question-and-answer turn.

The rest of the video reaches the moment only through the model's own act decision.

**Two forms.**
- A1, mask form: run on the cached prefix with Grounder's mask code. This is `all_local` plus the verdict turn.
- A2, transfer form: run only if A1 passes. The window is read in a fresh local pass (rules, its own frames and words, the question), and the verdict-position states are patched into one placeholder position at every layer, Patchscopes-style.

**Module that gains a claim: M1.** The M3 decision becomes the only conduit of the act into M1, which is a cross-module role for the verdict. The claim would be: "the act is defined once, by the model's own decision; a moment is read against that decision, not against the content of every other moment".

**Named failure answered.** The full-context read takes the act from the whole video, including other moments' content:
- 55–81% of the top false alarms in hateful videos are more than 8 s from gold (`experiments/20260927_error_analysis/README.md`);
- a topic mention adds +8.1 / +7.0 logits, against +4.3 / +4.3 for the GT (topic cross-tab, history §1.2).

Meanwhile, the decision itself reaches the moment only as a gain (P10). A read without context loses the act (`all_local` −.046 / −.049).

**Closest prior work and the difference.**
- Patchscopes (ICML 2024, arXiv 2401.06102).
- Task vectors (Findings of EMNLP 2023, arXiv 2310.15916).
- Function vectors (ICLR 2024, arXiv 2310.15213).
- In-context vectors (ICML 2024, arXiv 2311.06668).

These patch states to inspect them or to transfer a task. Here, a same-video decision state is the only context of a per-moment read, for localisation. None of hate, VAD or TAL work does this (lit_m1 §C4, §F).

**Closest repo attempts, and why this is outside the closed families.**
- Grounder `all_local` (verdict turn blocked: −.046 / −.049) and `verdict_only` (verdict blocked late, context kept: equal to base). Source: `archive/experiments/20261002_m1_grounder/README.md`.
- HVL's text hypotheses (closed family M1-b).
- A1 belongs to the access-restriction family (M1-f), with an access set that was never run. It is a falsifier, not the claim.
- A2 is in no closed family.

**Rule check.**
- Rule 3: no ensemble, no post-processing.
- Rule 4: the source is interpretability work, not used in hate.
- Rule 5: no new input.
- Rule 13: one uniform rule.
- Rule 14g: removing the act state (giving `all_local`) must cost ≥ .01 on both corpora, and so must using another video's act state.

**Cost.**
- A1: no added calls. Grounder's runs took about 5 min per corpus pair.
- A2: per-window local passes, about 1.5× the r6 reading cost.
- The states stay in memory. No tensors are saved (memory rule).

**Falsifiable pre-check.** There is no CPU proxy for the state itself. The CPU evidence against it is P10 (the decision acts as a gain) and the HVL permutation result. GPU, A1, about 15 min, arms:
- r6 replica;
- `all_local` replica;
- A1;
- A1-other: the verdict-turn states of another video of the same corpus with the same verdict sign.

Pass, all of:
- A1 within ≥ r6 − .005 on both corpora;
- A1 − `all_local` ≥ .01 on both corpora;
- A1 − A1-other ≥ .01 on both corpora.

Kill: A1 more than .01 below r6 on either corpus. A2 is not run in that case.

**Story paragraph.**
> The act is defined once, by the reader's decision about the whole video. A moment should be read against that
> decision, not against the content of every other moment, which is where far-away false alarms come from. Each window
> is therefore read from its own frames and words and from the decision turn alone; the rest of the video reaches it
> only through the decision's internal state. Blocking the decision costs X on both corpora; giving the moment another
> video's decision costs Y.

**Honest estimate.**
- Novelty: high if A2 passes.
- Chance that A1 reaches r6 − .005 on both corpora: about 10%.
- Main risk: P10, the HVL permutation result and Grounder `verdict_only` all say the decision carries little more than a per-video gain. A1 would then sit near `all_local` (−.046 / −.049).

## 4. Novelty risk of the current M1 framing (question 3)

**What the structure is.** In M1, one cached context holds the whole video, the rules and the model's verdict. Isolated questions branch off that context, one per window and modality. Each question addresses a time span of the same context. The per-branch Yes/No values form the output curve.

**Method-level precedents:**
- Superposition Prompting (ICML 2024, verified): shared preamble KV, per-document paths with the query duplicated, path saliency and pruning.
- Parallel Context Windows (ACL 2023): context chunks encoded in parallel, with task tokens attending to all of them.
- Prefix forking (Hydragen, SGLang, Prompt Cache).
- Per-segment Yes/No reads (Probe-VAD, arXiv 2608.08315, CEAVAD).
- Global-then-local (T3AL, SeViLA, GtS).

M1 differs from Superposition only in use. Its evidence lives in the shared context, not in the paths, and its branches are the output rather than a means to prune. A reviewer can fairly call M1 "forked prompting over video windows".

**Can claim** (application level, with evidence):
1. In hate localisation a moment must be read with the whole video in view. On HateMM this holds moment by moment: raw within .628 → .700, and the offset-only control recovers none of it (P6).
2. The combination of a whole-video context, the model's own verdict and isolated per-window questions is unoccupied in hate and in training-free localisation, as far as searched (final review §6.1).

**Cannot claim:**
- isolation or prefix caching (systems precedents);
- per-window Yes/No;
- global-then-local;
- "the verdict turn makes moments more discriminative": it is a per-video gain (P10);
- "the context carries the act into HCS moments": on HCS it is a per-video offset (P6);
- "re-showing frames binds the moment" on HCS: the decoded gain survives shuffling (P4).

**Minimal mechanism change that would make a claim defensible.** The change must alter how a branch relates to the shared context, and it must pass three tests: an ablation of ≥ .01 on both corpora, a control that isolates the moment-specific part, and no pooled loss. The two candidates are Proposals 1 and 2.

Until one passes, the advice is:
- Write M1 as LAVAD's temporal summary and VERA's scene context are written: as the answer to a named failure ("a moment does not know which act it belongs to").
- Name Superposition, PCW, Hydragen and Prompt Cache as borrowed in the implementation paragraph. paradigm_framing.md currently names only Hydragen and Prompt Cache.
- Claim only items 1–2 above.

## 5. Framing patterns (question 4)

These refer to the rules in paradigm_framing.md §3.

**Patterns that can carry M1 once a mechanism is added: rule 2** (TFVTG's form: naive baseline, named failure, module, ablation row).
- Proposal 1:
  - naive baseline: ask about window i by its timestamps over the whole video;
  - failure: lookup (36% / 44%);
  - module: re-addressing;
  - ablation rows: removal, and wrong window.
- Proposal 2:
  - naive baseline: read each moment against the whole video;
  - failure: other moments' content leaks into the read, while the decision itself arrives only as a gain;
  - module: act-state reading;
  - ablation rows: decision blocked, and another video's decision.
- Rule 6 (distinguish by setting or cost) handles Superposition: "the branch adds no new content and no encoding".
- Rule 7 (SeViLA's two roles for one frozen model) supports the framing that "one reader decides the act and reads its moments". It is framing only.

**M3: no pattern carries it without a mechanism.**
- T3AL and STPN make "classify, then localise" standard.
- LAVAD, VERA and VADTree have no cross-video term at all.
- Following rule 5, write M3 as a simple module with one ablation row each:
  - video term: pooled ROC −.325 / −.147;
  - extent term, verdict only: −.0135 / −.046 and −.037 / −.048 (against the calibrated key, m3_coverage §12).
- Credit the borrowing: "Following T3AL / STPN".

**What the user would see as renaming:**
- calling the current shared prefix an "act-conditioned read" and claiming it as a mechanism;
- calling K an "act coverage key", or "the posterior of V", without changing the computation;
- the final review's "scoring rule matched to the two metric families" for M3;
- moving parts between modules (OR into M3, M3 into M1).

**Analysis framings to avoid:** do not present P6 or P10 as contributions ("we find the verdict acts as a gain"). Use them only as the named failure in a method section.

## 6. Dropped ideas (with the reason)

1. **Hypothesis-paired reading, using the forced-Yes / forced-No contrast as window evidence** (history O4; scout M3 item 2).
   - The own-stance lift is an amplifier: shuffled-lift control equal, R² .90–.92 (P3, P10).
   - O4's label-free test (within-video sd > .5 logit) is necessary but not sufficient: the own-stance lift passes it (sd ratio .38) and still carries nothing beyond the read.
   - A forced-"No" pass (4–9 GPU min) would close O4 for good. It is not proposed.
2. **Verdict susceptibility (the mean stance lift) as an M3 term.** The only weight-free use is the key without the stance turn, which loses on HCS (−.011 / −.010, §2.4).
3. **The video level as M2's top latent; M3 replaced by an M2 quantity.** P8 and P9 plus history (§2.1, §2.3).
4. **Treating a visual read as evidence only where the moment was seen** (dropping frameless-window visual reads in M2). Within −.0025 / −.0088 (P7). A matched random-drop control loses less (−.0004 / +.005).
5. **M2's inferred segments fed back to M1 as context or as a hypothesis** (iterative act-scoped reading).
   - This is HVL confirmation and persistence again.
   - Neighbour transcripts cost −.017.
   - Remote context never reached .01 (family M1-j).
6. **Re-showing the act's anchor moments (the strongest windows) next to every question.** Family M1-j (remote evidence) and the HVL anchor-example rejection (novelty_eval M1_report §3).
7. **Context dependence z(context) − z(alone) as a window type for M2.**
   - The context contrast scores .467 / .494 (history).
   - On HCS the context effect is an offset (P6).
8. **Re-partitioning modules** (merge M3 into M1 as "the act decision"; move the channel OR into M3). This is renaming.
9. **A derivation of K as "the V posterior with exchangeable reads".** The derived computation is P8, which loses. Keeping K under that name is renaming.

## 7. What this means for the lead

- **The cheapest decisive step** is the two GPU falsifiers: Proposal 1 arms (a)–(c) and Proposal 2's A1 arms, about 30 min in total on one lab GPU. If both fail, M1 has no mechanism claim on these reads.
- **M3 has no remaining room from re-expression.** Its only levers are new reads (the M3 designer's threshold ladder, history P1) or better M1 reads. The latter currently move the HCS extent term only as a per-video level.
- **If both proposals fail,** the honest paper keeps two claims: the problem form (act inference) and M2. M1 and M3 are then written per §4 and §5 as answers to named failures. The user should hear this before more rounds are spent.
- **For the M1 designer:** P4 and P5 show that the binding gain on disk is mediated by the time level on HateMM and is not tied to the moment on HCS. Any binding claim needs a real wrong-window read.

## 8. Pre-check log (rule 10)

Scripts and outputs are in
`/tmp/claude-135258174/-home-jehc223-Hate-follow-up/abae8735-0dba-4355-b09e-130e58ba97bd/scratchpad/m1m3_novelty/work_story/`.

| check | script → output | inputs read (test GT read in every row: `data/gt_4fps/*.npz`, analysis and evaluator only) | found |
|---|---|---|---|
| P1 stance lift statistics | `p1_contrast_stats.py` → `.json` | `runs/20260910_spvl/mllm/q3vl-8b/{full,nostance}` (old ASR), `runs/20260927_dehate_external/reads_gridA`, `runs/20260928_infer/dehate/reads_nostance` | lift ranks windows .72 / .59 / .62; −(mean lift) separates benign Yes videos .66 / .73 / .59 |
| P2 key swap | `p2_key_swap.py` → `.json` | the reads above + `runs/20261002_verdict_analysis/main_full`, `runs/20260927_dehate_external/r6_rawkey` (rank terms) | counter-hypothesis keys lose on HCS (CI excludes 0) |
| P3 contrast-weight grid | `p3_contrast_window.py` → `.json` | as P1 | AUC falls with the weight; shuffled control equal |
| P4 re-show controls, decoded | `p4_binding_controls.py` → `.json`, `reads/`, `decoded/` | `runs/20260926_glr/base_gridA`, `runs/20261007_m1_streamingtom/placement_controls_main/adjacent_native` | HateMM gain needs its window; HCS gain survives shuffle |
| P5 re-show controls, raw | `p5_raw_frame_within.py` → `.json` | `work_story/reads/` | raw within −.018 / −.005 |
| P6 context controls | `p6_context_controls.py` → `.json`, `reads_ctx/`, `decoded_ctx/` | `runs/20260910_spvl/mllm/q3vl-8b/{joint,winonly}` | context is moment-specific on HateMM, an offset on HCS |
| P7 frameless visual reads | `p7_unseen_visual.py` → `.json`, `reads_seen/`, `decoded_seen/` | `base_gridA`, `data/frames_k20` file names | dropping them costs −.0025 / −.0088 |
| P8 M2 video posterior as key | `p8_m2_video_posterior.py` → `.json`, `p8_em_check.log` | `base_gridA`, DeHate `reads_gridA`, `runs/20260926_twolevel/final_rawkey`, DeHate `r6_rawkey` | PR −.032 / −.018 / −.003 |
| P9 one-probability frame score | `p9_one_probability.py` → `.json` | as P8 (`extra.cell_prob`) | HateMM −.027 / −.038, HCS +.023 / +.002, DeHate −.023 / −.019 |
| P10 lift vs read (label-free) | `p10_lift_vs_read.py` → `.json` | as P1 (no GT) | slope 1.2, R² .90–.92 |

Design changes made from these reads: Proposal 1 was given a wrong-window control and a raw-read pass condition. The hypothesis-paired idea was moved to the dropped list. No test label enters any score in these scripts; GT is used only by the evaluator and in the analysis AUC and Spearman values.
