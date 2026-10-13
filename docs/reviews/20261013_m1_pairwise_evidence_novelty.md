# Proposal review (rule 4): candidate 42, evidence-bound pairwise comparison of windows, 2026-10-13

Reviewer: one independent agent (same model as the main session), before any GPU run. Commit: `6abcb5c`
(branch `follow-up`). Read: `RESEARCH_ITERATION_RULES.md` rule 4 (and rule 3 for the ensemble check); the proposal
`experiments/20261013_m1_pairwise_evidence/README.md` ("Evidence and hypothesis", "Mechanism", "Difference from
earlier attempts"); `experiments/20260912_pwc/README.md` §1, §2 (5a–5e), §7b, §8; `research-wiki/DIRECTIONS.md` A2
(PWC row); `experiments/20261002_m1_iteration/README.md` (Pairwise Ranking Prompting located, not selected after the
PWC failure); the earlier proposal-review records `docs/reviews/20261010_m1_grounded_adjacent_proposal.md` and
`docs/reviews/20260927_final_novelty_review.md`. No separate rule-4 record of PWC exists under `docs/reviews/`; the
PWC README §5e records that review's one correction (pooled metrics can move) inline. This file is the only file
written by this review. No hashes were computed.

## What was searched (2026-10-13, WebSearch, 14 queries; two abstracts opened with WebFetch)

| query | closest hit (one line each) |
|---|---|
| pairwise comparison video temporal localization MLLM | no pairwise method; Bridge-STG 2604.08014 samples positive/negative frames for a query (training, not pairwise ranking); 2605.21954 recovers grounding from attention |
| pairwise ranking prompting video moment retrieval LLM | 2607.19027 re-ranks top-5 spans with a per-frame Yes-logit (pointwise); Moment-GPT 2501.07972 and VeRVE 2601.12193 are not pairwise |
| hateful video localization pairwise comparison | only MultiHateLoc 2512.10408 (weakly supervised), LELA 2602.09637 (training-free, per-frame scores), HateClipSeg 2508.01712, DeHate; none compares segments |
| relative judgment VLM temporal grounding 2025 2026 | evaluation papers only (AoT-PsyPhyBENCH 2510.26241, TimeCatch 2608.23474): VLMs near chance when asked to order or compare events in time |
| pairwise comparison VLM Bradley-Terry ranking video segments highlight | X-CoT 2509.21559: LLM compares candidate *videos* pairwise, Bradley–Terry fit, text-to-video retrieval; GenArena: VLM pairwise judging of generated images with Elo / BT; Yao CVPR 2016 and Saquil ICCV 2021: trained pairwise rankers for highlights (no LLM) |
| comparative prompting MLLM "which clip" video anomaly training-free | none; LAVAD (captions + LLM), Probe-VAD 2609.17211 (ordinal likelihood over severity options per clip, pointwise), 2603.13374 |
| hateful video detection multimodal LLM 2026 segment-level localization | LELA 2602.09637, SafeLens AAAI 2026 (tuned policy LLM per segment), MultiHateLoc; all absolute per-segment outputs |
| HodgeRank least-squares aggregation LLM pairwise judgments ranking frames | HodgeRank background (Jiang et al. 2011; least squares on a graph Laplacian); used to rank LLMs (2603.10960); no application to LLM-judged video frames |
| Compare2Score pairwise LMM image quality MAP | Compare2Score NeurIPS 2024 (2405.19298): LMM answers comparative questions against anchor images, probability matrix converted to a score by MAP; image quality, trained |
| VLM pairwise frame comparison "which frame" anomaly two clips side by side | 2607.11957 compares a manual video against an expert video by VLM descriptions; CoReVAD 2605.23116 is pointwise; no side-by-side pairwise prompt |
| hate speech "pairwise comparison" LLM ranking toxicity which is more hateful | Oak et al. 2501.13977 (opened, HTML): LLM sees items A and B (YouTube video *descriptions / titles / transcripts*, Jigsaw comments, DLab hate-speech posts), answers which is more harmful or NONE, all pairs in both orders within a 20-item feed, +1 per win, sort; adapts Qin et al. PRP; text only, whole items, no video input, no temporal localization |
| arXiv 2608.21854 (the within-metric source) | Song & Lee, "Look Inside Each Video": a VAD *evaluation* paper (Within-AUROC); detectors output absolute per-frame scores; not a hateful-video method and not a pairwise method |
| pairwise comparison prompting candidate moments temporal grounding MLLM zero-shot 2026 | none; 2509.15178 (sub-query decomposition), 2510.17023 (query enrichment), 2407.06157 |
| LLM-as-a-judge pairwise video clips zero-shot segment ranking | VideoJudge 2509.21451 compares two *answers* about a video; X-CoT sliding-window pairwise swaps over retrieval candidates; LLMVS summarization is pointwise |

Also known from earlier repository records and not re-searched: Pairwise Ranking Prompting (Qin et al. 2023, text
re-ranking), Bradley–Terry / HodgeRank aggregation (PWC §8), LELA's explicit absence of cross-segment calibration
(PWC §8), MATCH / CLARA video-level verifiers (candidate 41 review).

Result of the search: no paper was found that asks a frozen MLLM to compare two windows of the *same* video, with
both windows' frames in the prompt, and aggregates the relative answers into a within-video ordering, for hateful
video or for any training-free temporal localization task. The three nearest items are (1) Oak et al. 2501.13977,
pairwise LLM harm judgments over whole text items, including YouTube video metadata, aggregated by win counts;
(2) X-CoT, pairwise LLM comparison of candidate videos with a Bradley–Terry fit; (3) Compare2Score, an LMM's
comparative answers against anchors turned into scores. All three compare different items, not segments inside one
item, and none is a localization method.

## Repository matches

The proposal's "Difference from earlier attempts" checks out against the READMEs: PWC (`experiments/20260912_pwc`)
is the same estimator family with frames referenced by timestamp and transcripts in the prompt; its content-blind
control (.546 / .511) and the 2026-10-10 placement probe (frames found by timestamp in 36 % / 44 % of windows) support
the proposal's diagnosis that PWC's comparison was carried by the transcripts. Candidate 41 and its revision, HVL,
ReKV, StreamingTOM, ordered slots and Explorer 18 ask no relative question. PRP was located on 2026-10-02 and
dropped because of the PWC failure (`experiments/20261002_m1_iteration/README.md`), which is the same mechanism
this candidate reopens with a changed evidence presentation. Nothing else in `experiments/` or `archive/experiments/`
compares two windows in one prompt.

## STOP checks

1. **Source method already used for hateful video detection / localization: no.** Pairwise LLM harm judgments
   exist for ranking whole text items (Oak et al. 2501.13977, including YouTube video descriptions and a hate-speech
   post dataset), i.e. cross-item, video-level, text-only exposure re-ranking. No hateful-video detection or
   localization paper found compares segments within a video; LELA, MultiHateLoc, SafeLens, HateClipSeg and DeHate
   baselines all produce absolute per-segment or per-frame outputs. The transfer of pairwise ranking prompting into
   within-video window ordering is allowed under rule 4 (transfer from another task).
2. **Pure ensemble: no.** One frozen model (Qwen3-VL-8B); the comparison forwards are repeated reads of that model
   aggregated by a fixed deterministic rule (the user's 2026-09-12 ruling in PWC §5d applies). The compositions
   combine the pointwise level `mean(l)` or `l_i` with `u_i` from the same model; rule 3 counts outputs of one MLLM
   as one model. The comparison stage is an explicit component with `full_l` declared as its removal, as rule 3
   requires.
3. **Pure calibration / post-processing / smoothing: no.** A new measurement is taken (2 × edges comparison forwards
   per video with the windows' frames in context); the Laplacian least squares is the estimator of the quantity
   those forwards measure, not a transform of existing scores. No labels enter steps 1–6.
4. **Pure engineering trick: no.** Relative to PWC the change is not wording only: the evidence the answer is
   conditioned on changes (frames of both windows shown versus referenced by timestamp, which probe A shows the
   model cannot resolve), the question is restricted to one modality so the output slots into the V channel the
   decoder already reads, and the proposal states a falsifiable hypothesis (H1 versus H2 = `sign(l_i − l_j)`), a
   same-frames pointwise control (`adjacent_local`), six permutation seeds, and a mechanism gate separate from the
   performance gate. No encoder swap, constant tuning or resolution change is involved.

Not grounds for STOP under rule 4 and therefore not used: whether `u` degenerates to `l`, identifiability of the
ring graph, or whether the answer is a position or length shortcut; the declared controls test these.

## Relation to PWC

Same estimator: A/B/C first-token log-probabilities, `d = log((pA + pC/2)/(pB + pC/2))`, swap antisymmetrisation,
a seeded ring graph (distances 1, 2 here; 1, 2, 3 in PWC), Laplacian pseudo-inverse. Different evidence and
different placement in the method: PWC put both transcripts in the question and referenced frames by timestamp;
this candidate shows each window's LOCAL frames next to the question, asks a visual-only question, and composes
into the visual channel of r6 (PWC composed a centred rank into SPVL-r2's residual). So this is a revision of PWC
driven by a later diagnosis, not a new estimator. That is consistent with rule 4 (the candidate source may be a
mechanism redesigned from error analysis) and with rule 9 as long as the proposal is honest that the comparison
family was reopened on new evidence; the README is.

Advisory, not a STOP: the question form kept here ("which window more clearly shows visual content that violates
the above rules") is the "generic comparison" arm of PWC E0, which did slightly worse than the carrier/context arm
(−.003 / −.015 versus −.010 / +.005 against the baseline sign). PWC §8 said the hate-specific claim had to be the
carrier/context distinction; this candidate drops it. If the mechanism gate passes, the claim therefore rests on the
evidence binding, not on the question's content.

## What the novelty claim could be if the gates pass

- Could claim: (i) the diagnosis chain — a frozen MLLM does not resolve timestamp references to frames (probe A),
  so PWC's relative judgments were transcript-carried (content-blind control), and relative reading only carries
  within-video ordering when the compared evidence is bound into the prompt; (ii) evidence-bound pairwise reading
  as the source of within-video ordering in label-free hateful video localization, supported by the ablation
  `cmp_order` versus `full_l` (same frames, pointwise) and by the permutation band — i.e. that the ordering
  information comes from the relative question and not from the extra frames. (iii) That this is, as far as this
  search found, the first within-video pairwise MLLM comparison used for temporal localization of hate.
- Could not claim: pairwise ranking prompting (Qin et al. 2023), Bradley–Terry / HodgeRank / least-squares
  aggregation on a graph, swap symmetrisation, A/B/C logit read-outs with ties (Compare2Score, X-CoT, GenArena),
  first use of pairwise LLM harm judgments (Oak et al. 2501.13977, text items), or that showing in-window frames
  next to the question is new (the repository's own placement result of 2026-10-10 and standard frame-grounded
  prompting). If the performance gate passes but the mechanism gate does not, the gain belongs to the input change
  already measured by `adjacent_local`, and the comparison cannot be named as a component.

VERDICT: PASS — the source method is not used for hateful video detection or localization in the literature
searched; the candidate is neither a pure ensemble, nor pure post-processing, nor a wording-only change; it is a
diagnosed revision of PWC whose novelty, if any, lies in the evidence binding and the within-video relative reading,
not in the estimator.
