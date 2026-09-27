# DVD: definition-conjunctive video verdict (2026-09-27, concern K3)

Status: proposal, declared before any run. Hosts are in the first line of each `runs/20260927_dvd/<run>/run.log`.

## 1. Problem (from `experiments/20260927_error_analysis/README.md`, test-read)

The video level of the current method rests on one question: "Does this video contain content that violates any of
the above rules?" (`src/mllm_judge.py` `VIDEO_QUESTION`, with the hate-policy rules and a "full reading" instruction
in the prefix). It answers Yes for many non-hateful videos: 49 % on HateMM, 67 % on HateClipSeg, 57 % on DeHate.

On DeHate, where 20 % of the videos are hateful, a perfect video level would raise pooled PR from .158 to .351. The
top-ranked non-hateful videos fall into three groups:
- abuse aimed at individuals, officials or groups not defined by a protected attribute;
- slurs that are used but not endorsed: songs, satire, historical clips, in-group use;
- reports and discussion of hate: news, a hearing on antisemitism.

Hate, as the rules define it, needs three things at once:
- an attack or contempt;
- aimed at a group defined by a protected attribute, or at people because of it;
- endorsed by the video, not quoted, reported or mocked.

Each group of false positives lacks one of the last two. One question that asks for "any violation" folds the three
together.

## 2. Mechanism

**The video-level judgment becomes a conjunction of the definition's conditions.** Each condition is read
separately, and the conditions are combined by AND.
- The existing verdict read stays as the "violation / attack" condition.
- Two new reads are added:
  - T asks whether the hostility is aimed at a protected group;
  - E asks whether the video endorses the message.
- Each read is an isolated Yes/No branch from the same cached prefix (rules + 20 frames + transcript), asked before
  and independently of the verdict. The existing verdict, stance turn and window reads are unchanged.
- Label-free combination:
  - Each condition's read gets its own two-component 1-D Gaussian mixture on the corpus. This is the same
    `key_calibration` as the current key: logit P(condition | read) = a·read + b.
  - The video probability is P(V = 1) = P(violation | K) · P(T | z_T) · P(E | z_E). It is a noisy AND of
    independent conditions, where K is the current key (verdict + mean window read).
  - The new key logit P(V = 1) replaces the calibrated key in the current composition (key + centred within rank).
    The time level is untouched.
- **Why this is a mechanism and not a prompt edit:** it changes what the video level infers, from one judgment to a
  conjunction of the definition's parts inferred separately. Each part can fail independently, and each has its own
  label-free calibration.

## 3. Reads and cost

New script `dvd_measure.py`. It uses `src/mllm_judge.Judge`, the same model (Qwen3-VL-8B-Instruct, bf16, greedy
Yes/No log-odds), the same prefix builder and the same 20 frames and transcripts. Per video it adds three branches
on the cached prefix, deep-copied, so it never touches the verdict:

- **T:** Is the hostile or demeaning content in this video aimed at a group of people defined by a protected
  attribute (race, ethnicity, national origin, religion, caste, sex or gender, gender identity, sexual orientation,
  disability, or immigration status), or at people because they belong to such a group?
  - Answer "No" if it is aimed only at specific individuals, public figures, officials, organizations, or groups not
    defined by a protected attribute, or if the video has no hostile content.
- **E:** Does the creator or speaker of this video endorse or promote the hateful message themselves?
  - Answer "No" if the video reports on, quotes, condemns, satirizes, or discusses hateful content without endorsing
    it, or if slurs appear only as in-group or reclaimed usage (for example in music) without hostility toward the
    group, or if the video has no hateful message.
- **A (diagnostic arm only):** Does this video attack, demean, dehumanize, threaten, or express contempt for people?
  - Answer "No" for neutral discussion, news reporting, or criticism of ideas or actions that does not show contempt
    for people.

Each question ends with `Answer "Yes" or "No".`, as the existing questions do.

**Cost:** one prefix encode plus 3 short branches per video, about 1.3 s per video on a 5090. For a new video in the
final method this is 2 extra branches (T, E) on the prefix the method already encodes; the time is dominated by the
window branches. The reads are cached by run; the window reads of `runs/20260926_glr/base_gridA` (HateMM, HateClipSeg)
and `runs/20260927_dehate_external/reads_gridA` (DeHate) are reused unchanged.

## 4. Arms (CPU after the reads; the time level is the current one when this runs)

| arm | video key |
|---|---|
| `dvd` (primary) | logit[P(viol \| K) P(T) P(E)] |
| `dvd_noT` (ablation) | logit[P(viol \| K) P(E)] |
| `dvd_noE` (ablation) | logit[P(viol \| K) P(T)] |
| `dvd_ATE` (diagnostic) | logit[P(A) P(T) P(E)], the verdict replaced by the attack read |
| `base` | the current key (control; must reproduce the current method) |

## 5. Decision rule

Gate, on HateMM and HateClipSeg against `base`:
- no metric may drop by more than the noise floor (pooled .005, within .01);
- at least one main metric must rise by .01 or more on both corpora (rule 8 promotion; rule 14g for a novelty claim).

Each condition must pull its weight: dropping T or E must cost at least .01 on one main metric on both corpora, or
that condition is removed.

Concern check, reported and not a gate:
- the share of non-hateful videos with P(V = 1) > .5, under `base` and under `dvd`, on all three corpora;
- video AUC / AP of the key;
- DeHate external: all metrics, interval precision of the `full` arm.

Routing (rule 9): if the gate fails but one metric rises by .01 or more on one corpus, revise from error analysis, at
most 3 rounds; otherwise archive and close K3 with the evidence.

## 6. Process

- Rule-4 proposal review: one independent agent, literature check (§7).
- Rule-6 code review: one independent agent, before the full run. Done 2026-09-27 on commits b0429ab..cc8c159: no bug.
  - The re-read `z_video` equals the cached `z_video` on all 333 HateMM / HateClipSeg videos (difference 0.0000).
  - Reads align with the manifests: 0 missing, 0 extra.
  - The noisy-AND key matches logit(∏p) to 1e-14; no video reaches the clamp.
  - Without `--dvd-conds` the old key path runs unchanged.
- Test-read log: §8.

Launch: `experiments/20260927_dvd/launch/run_arms.sh` (HateMM, HateClipSeg; the time level is `r3_m2`, the current
method when the arms run) and `run_dehate.sh` (DeHate, external). Output `runs/20260927_dvd/arms/`,
`runs/20260927_dvd/dehate/`; analysis `runs/20260927_dvd/analysis*/`.

The `full` arm (intervals) runs as `base_full` and `dvd_full`, for the interval-precision check in §5.

## 7. Proposal review

**Verdict: PASS.** Rule-4 review, 2026-09-27. No STOP case applies.

- Case 1: no hateful video detection or localization method found reads the definition's conditions separately and combines them by AND. HVGuard ([EMNLP 2025](https://aclanthology.org/2025.emnlp-main.456/)) splits reasoning by modality. LELA ([arXiv 2602.09637](https://arxiv.org/abs/2602.09637)) and MARS ([arXiv 2601.15115](https://arxiv.org/abs/2601.15115)) make one overall hate judgment. MoRE, CLARA, IARE, TANDEM and MultiHateLoc (trained with labels) do not either.
- Cases 2 to 4: T and E are new questions about different conditions, not a second model, a rewording, or a rescaling.

Closest prior work, all outside hateful video:
- Hypothesis Engineering, TRAC@COLING 2022 ([arXiv 2210.00910](https://arxiv.org/abs/2210.00910)). Zero-shot text NLI. Separate hypotheses for protected target, support for quoted hate, and self-directed slurs are combined by fixed AND / AND-NOT rules. It was built from the same three false-positive types. Same mechanism, hard thresholds.
- xList-Hate, arXiv 2026 ([2602.05874](https://arxiv.org/abs/2602.05874)). An LLM answers ten questions separately, including protected target and endorsement versus quotation. A decision tree trained on labels combines them.
- CLUE, arXiv 2025 ([2501.00192](https://arxiv.org/abs/2501.00192)). Label-free MLLM image-safety judge. A rule is violated only if all separately read preconditions hold.

New claim allowed: first use of this conjunction for hateful video and for label-free temporal localization, as a soft noisy-AND of MLLM condition reads, each calibrated on the corpus without labels. Reading the conditions separately and AND-combining them without labels is not new; cite it as a transfer from text. The mixture calibration is the existing key calibration, not a contribution. Rule 14g applies to T and E each.

## 8. Test-read log

- 2026-09-27: `runs/20260927_error_analysis/`; the 12 top-ranked non-hateful videos per corpus (transcripts, DeHate
  titles). This led to the three-condition design.
