# Adversarial review of the M3 proposals (2026-10-25)

Reviewer on the M1 / M3 novelty team. I did not write any proposal. I edited no repo file, ran no GPU job, and computed
no hash or checksum. My own CPU checks are in `work_review_m3/` (R1–R4, scripts `*.py` with `*.json` outputs). Metrics
come only from `src/eval/evaluate.py` (`pooled`, `within_video_macro`). Every number is development-selected: it was
read on test, and test GT was read for analysis only (rule 10, log in §8). The only comparison point is the current
method (pooled ROC / pooled PR / within): HateMM .8955 / .6883 / .7508, HCS .7137 / .6666 / .6373, DeHate .6994 /
.1568 / .6431.

## 0. Verdicts

| proposal | verdict | deciding evidence |
|---|---|---|
| #1 act-span verdicts, key arm | **DROP** (no GPU) | Two of its 14 reads are already on disk with identical inputs (candidate 49's first and last quarter). With the declared formulas they fix the extent at ≥ .75 for 62 % / 69 % of verdict-Yes videos (HateMM / HCS), whatever the other 12 reads say (R1). |
| #1 act-span verdicts, joint arm | **DROP** (no GPU) | A rule with no reads at all, lowering the first and last eighth of every video, gives HateMM within +.007 to +.029 and HCS −.006 to +.0005 (R4). The arm's real-read stand-in (+.0065 / +.0019) is below that rule, so on HateMM it adds nothing beyond the annotation layout. HCS has almost nothing at its edges to lower. |
| #2 counter-verdict reading | **DROP** | Rule-4 STOP is likely: the Marginalizer review already ruled on "opposite verdict conditions" (MARS, RAMF). On HCS the additive proxy loses −.022 / −.023 with a CI that excludes 0. The product arm needs the calibration the user removed: without it, the product loses −.027 / −.048 on HateMM (R2). |
| #3 segment-pooled video node | **DROP** (confirmed) | The designer's own declared check fails: C1 on HCS .783 vs .803, and C3 on HateMM −.013 / −.017. The story designer's P8 replicates the failure independently (PR −.032 / −.018 / −.003). It sits in closed family M3-e. |

**No M3 proposal deserves GPU time.** On the current reads I found no mechanism that gives HateMM a pooled gain (§2).
If the lead wants act-span information anyway, §6 gives the only check worth running: about 5 GPU minutes, with gates
fixed now.

## 1. Act-span verdicts (#1): closed-family check

### 1.1 What is already on disk decides most of it (R1, `work_review_m3/r1_edge_bound.py`)

- **Two reads already exist with identical inputs.**
  - The proposal's beginning [0, D/4] uses the same builder as candidate 49's first quarter read: `subset_inputs`, the
    same 20 r6 frames (t < D/4), the same transcript slice, the same question and no stance turn.
  - Its ending [3D/4, D] is the same as candidate 49's last quarter read.
- **The declared formulas then fix most of the extent.** The design takes running maxima of the beginning and ending
  reads. So for parts 2–7 of 8, a_k ≥ z(Q1) and b_{k−1} ≥ z(Q4). Since σ(a_J) and σ(b_0) are at most 1:
  - **E ≥ (6/8) · σ(z_Q1) · σ(z_Q4)**, whatever the other 12 reads say;
  - for such videos only the first and last eighth can still change E.
  - The reads are close to saturated (|z| ≈ 10–18), so in practice E takes one of three values: .75, .875 or 1.

| (lower bound ≥ .70 = "confined") | HateMM | HCS | DeHate |
|---|---|---|---|
| confined, share of verdict-Yes videos | .62 | .69 | .40 |
| confined, share of GT-hateful videos | .78 | .69 | .44 |
| confined, share of GT-non-hateful verdict-Yes videos | .39 (of 64) | .40 (of 10) | .35 (of 518) |
| GT share of confined hateful videos, 10/50/90 % quantiles | .13 / .80 / .97 | .15 / .64 / .96 | .04 / .19 / 1.0 |
| confined hateful videos with GT share < .5 | 33 % | 32 % | 66 % |
| ρ(share), window mean, within the confined hateful videos | .289 | .463 | .227 |

What this means for the key arm:
- **Most hateful videos get an extent of at least .75**, although a third of them are less than half hateful.
- **Inside that group the window mean orders videos by share** at .29 / .46. The act-span term can order them only by
  two outer-eighth reads.
- **False-Yes videos are lifted.** 39 % of HateMM's 64 false-Yes videos also get E ≥ .75. That lifts the videos whose
  separation from the hateful ones HateMM pooled depends on.
- **The key term tracks read confidence, not extent.** For E near 1, logit E ≈ −log(mean of the 1 − π_k), and that is
  set by the least saturated of the 14 reads. The candidate-49 stand-in confirms this: ρ(logit E, z_video) .74 / .79,
  and 63 % / 72 % of verdict-Yes videos have E ≥ .95. The scale rule s = sd(window mean) / sd(logit E) is then set by
  these saturated values.
- **The designer's Stage-0 gate PB1 does not test this.** PB1 asks whether the outer eighths are both Yes in ≤ 60 % of
  verdict-Yes videos. It can pass while E stays confined for the majority, because the confinement comes from the
  quarter cuts, which are already read.

### 1.2 Against the earlier attempts

- **Candidate 49 (context-free quarters).** Its failure: ρ(share) .278 / .419, partial ρ given the window mean ≤ 0, and
  edge quarters read Yes in 78 % / 79 % of hateful videos against a GT rate of 66 % / 73 %.
  - The act-span design reuses those two edge reads verbatim, so it inherits their over-Yes.
  - Its endings have no earlier context, which is exactly the context candidate 49 lacked.
  - Growing beginnings do keep earlier context. So the proposal answers candidate 49 for beginnings only, and only
    before the first cut that reads Yes.
- **Candidates 47 / 48 (collapse onto the verdict).** The risk is real, and the design builds it in. Once a beginning
  contains the act (or only its topic), the read is the verdict: ρ .965 / .904 in candidate 47. The running maximum
  then holds Yes for every later cut. All the extent information is therefore in where the first Yes falls.
- **That flip point is likely the first topic mention, not the act's onset.**
  - BND (`experiments/20260912_bnd/README.md`) says the per-window judgement is largely topic detection, and that the
    topic is close to constant inside a hateful video. A topic mention adds +8.1 / +7.0 logits, against +4.3 for GT
    (history §1.2).
  - So a truncated verdict says Yes as soon as the group is named. That fits the edge over-Yes above and the 39 % of
    confined false-Yes videos.
- **BND (2026-09-12, archived).** It asked directly per window whether violating content begins or ends here. Begin
  scored .6466 / .5605 against the act read's .7581 / .6212. The model's onset sense was above chance (first GT window
  23 % / 16 % vs ≈ 5 %) but far from usable. Act-span asks a presence question on truncated input instead; the
  quantity it targets (onset, offset) is the same, and the reader has already been shown to be weak at it.
- **HVL persistence.** The sequential chain answered "continue" in 77–82 % of windows. Act-span imposes the same
  persistence by construction: one false Yes on an early beginning fixes the onset there for good.
- **2026-08 sentinel read in one causal pass** (`archive/detection-2026-08/docs/duplex/SENTINEL_LOCALIZATION_NOTE.md`).
  This is the nearest past reading under growing earlier context. The margins drifted up with position (mean Spearman
  +.384) and carried no within-video signal (.4994). Reads under a growing beginning rise with the amount seen.
- **Other neighbours** (2026-08 P3 coarse-to-fine controller search, stopped; AMBI bisection over event membership,
  matched by time-only controls; CVA; Eraser). Each truncates, removes or zooms differently. None was a verdict on
  prefixes and suffixes. They are not the same family, but none gave a usable boundary signal either.
- **Conclusion.** The proposal is not literally inside a closed family. Its two decisive reads are, though: they are
  candidate 49's reads. The rest of it inherits candidate 47's collapse through the running maximum.

### 1.3 The "perfect verdict" bound

The .757 / .730 extent bound and the +.092 / +.067 within bound come from an oracle that answers Yes exactly when the
piece holds a GT-hateful frame. Per the user's rule, I did not use them. All verdicts above rest on real reads (R1,
the stand-in) and on read-free controls (R4).

## 2. Where could a HateMM gain come from?

- **The key's extent term is worth .013 / .041 on HateMM.** With the key cut back to z_video alone, HateMM is
  .8828 / .6471 (−.0127 / −.0412) and HCS −.035 / −.045 (R3).
  - Inside verdict-Yes videos, presence is ranked by the verdict (AUC .921), not by the window mean (.824).
  - So the window mean helps HateMM through share order among its 82 verdict-Yes hateful videos, at a small presence
    cost.
- **A pooled gain therefore needs a share order above ρ .416 that does not lift the 64 false-Yes videos.** No read in
  the record does that:
  - C47 .254, C48 .195–.264, C49 .278;
  - M2 shares .136–.295, P8 .346;
  - act-span stand-in .305, and R1's confinement.
  - Presence moves are blocked too: the counter-verdict proxy gives +.003 / +.010 on HateMM and loses HCS.
- **The product composition gains HateMM PR only with the removed calibration** (§5).
- **The only HateMM within lever is the edge negatives, and a read-free rule already takes it** (R4: demote the first
  and last eighth of every video, +.007 to +.029). That is HateMM's annotation layout. Using it would be per-corpus
  error targeting, and it does not hold on HCS (−.006 to +.0005).

**So, plainly:** on the current reads no M3 mechanism in the record gives HateMM a pooled gain, and no read-based
mechanism gives it a within gain beyond a read-free edge prior.

How that maps onto the gates:
- **Joint arm and promotion gate (rule 8).** Both need a gain on HateMM and HCS. They cannot pass.
- **Correction to the brief: replacement key arm.** Rule 14g, as used for candidate 49, removes the new term down to
  z_video alone. A tie with the current method would therefore pass 14g without any HateMM gain. The gate that binds is
  G1 (no loss on HateMM), and R1 predicts that act-span fails it.
- **Cost.** A tie at three to five times the per-video GPU cost also fails the CLAUDE.md cost rule (§4).

## 3. Novelty search (2026-10-25)

Searches run, with what each returned:

| query (short) | closest finding |
|---|---|
| VLM temporal grounding by binary search / "has the event occurred" yes/no | FV-Action, "Your VLM Already Knows When" (arXiv 2608.08315): coarse-to-fine yes/no scan of clips, peak, fixed width. ReVisionLLM (arXiv 2411.14901): recursive coarse-to-fine. No prefix or suffix verdicts. |
| streaming / partial-output harmful content detection | Li et al., "From Judgment to Interference", NeurIPS 2025 (arXiv 2506.09996). Its partial-detection baselines (ProtectAI, GuardrailsAI) apply a full-output moderator to each growing prefix of a response. That is the unchanged verdict on growing beginnings, for harmful-content onset. The paper reports a mismatch between moderators trained on complete outputs and prefix inputs. Text, LLM outputs, not video. |
| early event detection on partial events | MMED (Hoai & De la Torre, CVPR 2012 / IJCV 2014): detectors trained on partial events, with a monotonic score on prefixes. Ma, Sigal & Sclaroff (CVPR 2016): a ranking loss so the score does not decrease as more of the activity is seen. The running maximum here is that monotonicity, imposed at inference. Both are trained. |
| early classification of time series | ECTS (Xing, Pei & Yu, KAIS 2011), with a minimum prediction length on prefixes. Wu, Der & Keogh critique. |
| noisy bisection / twenty questions | Horstein's probabilistic bisection; Jedynak, Frazier & Sznitman (J. Appl. Prob. 2012); Waeber, Frazier & Henderson (2013); Frazier et al. (arXiv 1612.03964). This is the principled estimator of a change point from a noisy monotone oracle. The proposal is a cruder, non-adaptive grid version of it. |
| perturbation / extremal-mask explanations for video | Deletion and insertion curves (RISE, BMVC 2018, recalled); EP-3D (arXiv 2005.00375); STEP (arXiv 2109.00222); ERASER sufficiency and comprehensiveness (DeYoung et al., ACL 2020). Classifier-on-kept-input tests, used for explanation. |
| training-free MLLM start / end search | Wake et al. (arXiv 2408.17422): iterative narrowing of start and end frames with frame-tile prompts. "MLLMs Know When Before Speaking" (arXiv 2605.21954): read-then-regenerate on a cropped region. |
| hateful video with prefix truncation | Not found. LELA (arXiv 2602.09637) and MultiHateLoc (arXiv 2512.10408) do not do it. Rule-4 STOP case 1 does not apply as far as searched. |
| online VAD with growing context | MoniTor (NeurIPS 2025): streaming VLM scoring with a score queue. Not prefix verdicts for extent. |

**Novelty judgement.**
- As far as searched, no paper intersects verdicts on prefixes and suffixes to estimate an act's span.
- The ingredients are standard, though: prefix moderation, monotone early detection, and noisy bisection. A reviewer
  would call it "prefix moderation plus bisection".
- Its defensible form would be probabilistic bisection over the onset and the offset, but that does not remove the
  confinement in §1.1.
- Rating: low to moderate.

## 4. Cost (CLAUDE.md)

- **Act-span at J = 8.**
  - 14 extra MLLM calls per video, about 7 full-prefix prefill equivalents.
  - The KV cache cannot be shared: the r6 prefix puts all 20 frames and then the full transcript, so a beginning is not
    a token prefix of the full prefix. The frame features could be reused.
  - About +2.8 s per video on a 5090, at candidate 49's measured rate. That is three to five times the current
    per-video reading cost (.6–1.6 s).
  - The noisy-read evidence does not show this cost is needed, so it fails the CLAUDE.md rule against many repeated
    calls without a clear need.
  - J = 4 needs 4 new reads per video. It is dead outright: E = 1 for every video whose two edge quarters are Yes.
- **Counter-verdict.** One extra short branch per window and modality, about +40–60 % of read time. The cost is
  acceptable; the proposal fails on rule 4 and on HCS instead.
- **Segment-pooled node.** CPU only.

## 5. Side finding: the two product numbers, reconciled (R2, `work_review_m3/r2_product_reconcile.py`)

Both products use the same frame term, log p_t, where p_t = `cell_prob` = P(h_t = 1 | V = 1, reads): the OR of the
chains' hate posteriors under V = 1. The within metric is therefore identical to the current method in both. They
differ only in P(V):

- **design_m3 §1.2.** The stored run `runs/20260926_twolevel/final_full` (`--arm full --key calib`):
  - score = log σ(a·K + b) + log p_t, with K = z_video + mean window z;
  - (a, b) come from `key_calibration`, the label-free two-Gaussian mixture on K, fitted per corpus:
    a = .357 / .341 / .374 and b = .399 / 3.508 / 1.792 (HateMM / HCS / DeHate, `params.json`);
  - **this is the calibration the user removed on 2026-10-07**, so this number requires it.
- **design_story P9.**
  - score = log σ(lo_one) + log p_t, where lo_one is the verdict term of M2's EM model plus each modality's chain
    log-likelihood ratio divided by its read count;
  - there is no key and no key calibration; it uses M2's own label-free EM parameters;
  - its video order differs from K (ρ .977–.983).

Decomposition without any fitted constant (difference from the current method, pooled ROC / PR; within unchanged):

| frame score | HateMM | HCS | DeHate |
|---|---|---|---|
| K + log p_t (only the within term widened) | +.000 / −.003 | +.006 / +.002 | +.002 / +.001 |
| log σ(K) + log p_t (product with the uncalibrated key) | **−.027 / −.048** | +.019 / .000 | −.015 / −.017 |
| log σ(z_video) + log p_t | −.029 / −.046 | +.018 / −.003 | −.020 / −.015 |
| P9, log σ(lo_one) + log p_t (story) | −.027 / −.038 | +.023 / +.002 | −.023 / −.019 |
| §1.2 final_full (needs the removed calibration; the designer's number) | −.005 / +.025 | +.025 / +.011 | −.005 / −.001 |

**Reconciliation.**
- The two findings agree. The product helps HCS pooled ROC whatever P(V) is used.
- On HateMM (and DeHate) the product loses, unless P(V) is the calibrated one:
  - the fitted slope of about .35 softens P(V), so that log p_t can order frames across videos;
  - with the uncalibrated key, 60 % of HateMM videos have P(V) > .99 and the product loses.
- So the §1.2 HateMM PR gain exists only through the removed calibration. It is also a hurdle model with no novelty,
  and under rule 4 it is pure calibration. It is not an M3 option unless the user reopens the calibration ruling.

## 6. If the lead still wants act-span data: the one check worth running (not recommended)

The question it answers: do the outer eighths carry read information beyond a read-free edge prior?

- **Reads.**
  - Per video: the beginning [0, D/8] and the ending [7D/8, D], with the same builder and question as candidate 49 and
    no stance turn.
  - For the videos R1 does not confine (≈ 31–38 % of verdict-Yes): also the remaining 10 cuts.
- **Corpora and cost.** HateMM, HCS and DeHate. About 1 GPU minute for the 333 main videos plus about 4 for DeHate, on a
  lab 5090.
- **Arms.**
  - (a) the joint arm as declared;
  - (b) a read-free edge control at matched strength: every video's outer eighths lowered by the corpus mean of the
    arm's log π on those parts;
  - (c) the same per-video demotions shuffled across videos of the same corpus;
  - (d) the current method.
- **Pass, all of these, fixed now:**
  - (a) − (d) within ≥ .01 on HateMM and on HCS;
  - (a) − (b) ≥ .01 and (a) − (c) ≥ .01 within on both;
  - no pooled loss beyond .005 on any of the three corpora; DeHate is reported.
- **Kill.** Any one of these fails. Prediction: it fails on HCS. In confined HCS videos only 7.7 % of the negatives lie
  in an outer eighth with no hateful frame, while 23 % of the positives lie in the outer eighths (R3).

## 7. Process notes

- **Proposal 3's criteria file.** `pG_declared_criteria.txt` was saved at 19:52:19, before the script's last change
  (19:53:22) and the output (19:55:09). The file says it was written while the script was running. The verdict is
  negative, so this does not matter here; the M3 designer should record the edit.
- **The read-free edge prior (R4) is not a method.** It uses HateMM's annotation layout, works with no reads, and does
  not hold on HCS. It is recorded so that no later candidate counts that HateMM within gain as a reading mechanism.

## 8. Test-read log (rule 10)

- **GT read** (test split only, for analysis and inside the evaluator): `data/gt_4fps/{HateMM,HateClipSeg,DeHate}.npz`.
- **Runs read:**
  - `runs/20261025_m3_coverage/{seg_main,seg_dehate}/segments.jsonl`;
  - `runs/20261025_m3_coverage/keys/{r6_key,r6_key_dehate}.jsonl`;
  - `runs/20260926_twolevel/{final_rawkey,final_full}/{predictions.jsonl,config.json,params.json,run.log}`;
  - `runs/20260926_twolevel/final_dehate/final_full/*`;
  - `runs/20260927_dehate_external/r6_rawkey/predictions.jsonl`;
  - all `work_m3/` and `work_story/` scripts and outputs.
- **Docs read:**
  - `experiments/20261025_m3_coverage/README.md`, `experiments/20260912_bnd/README.md`;
  - the 2026-08 sentinel note and its pre-registration, and the 2026-08 paradigm-adaptation results;
  - `docs/reviews/20261003_m1_marginalizer_proposal.md`.
- **What it changed:**
  - proposal #1 is dropped on R1 (key arm) and R4 (joint arm);
  - the side finding is reconciled by R2;
  - no design decision of the method was changed, and no test label entered any score.
