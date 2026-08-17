# Pre-registration — Masked parallel isolation pilot (direction 3, HateMM)

**Frozen:** 2026-08-18, before any masked forward pass is run. Novelty
check completed BEFORE this prereg (IDEA_REPORT.md, commit f0deb1f):
NOVEL-with-caveats — the mask primitive is conceded to SingGuard
(2606.22873), InvariRank (2604.27599) and T3S (2511.17945); the claim
is the contamination finding + isolation as its falsifiable remedy +
temporal-localization output. This pilot tests exactly those two
load-bearing facts.
**Compute:** one RTX 5090; 212 packed forwards (Arm M) + 212 packed
forwards (Arm C); text-only, sequences ≤ ~4k tokens; minutes of GPU
time. Reference scores already on disk.

## Phenomenon (measured, this project)

A frozen Qwen3-VL-8B judge forms ONE global verdict per video. Every
in-context per-segment probe returns that verdict smeared over the
timeline (sentinel readout 0.4994, packed questions 0.5186, gradient
attribution 0.5252 ≈ token-density control — all at chance
within-video). The same judge, given each transcript chunk in ISOLATION
(one call per chunk), recovers per-segment signal: HateMM span chunks
vs non-hate-video chunks AUC 0.720, within-hate-video 0.624
(`results/hatemm_localization/per_chunk.jsonl`, scripts 455f669).

## Mechanism

Cross-chunk attention is the contamination channel. A block-diagonal
attention mask over a packed sequence — shared rules prefix attended by
all branches; each chunk branch attends only to the prefix and itself —
reproduces the isolated-call computation exactly inside one forward
pass, provided each branch's position IDs restart at len(prefix). If
the token sequence prefix+branch_k equals the sequential isolated
prompt for chunk k, the branch logits are mathematically identical to
the sequential call's logits; the packed pass is N isolated judgments
at one-pass cost. Falsifiable: if contamination were NOT carried by
cross-chunk attention (e.g., purely positional), removing the mask
would change nothing.

## Frozen protocol

- Cohort: the 212 HateMM test videos / 2281 chunks of the isolated-chunk
  diagnostic, unchanged. Reference = the stored sequential per-chunk z.
- **Arm M (masked):** one packed forward per video. Tokenization: the
  shared prefix (system + rules + everything before chunk-specific
  text in the sequential prompt, split at a clean newline boundary) is
  tokenized once; each branch (chunk text + question + answer cue) is
  tokenized separately. Runtime assertion required: for every chunk,
  concat(prefix_ids, branch_ids) == the sequential isolated prompt's
  token ids (byte-identical prompts to `hatemm_isolated_chunk_diag.py`).
  Custom 4D attention mask (branch → prefix + own branch only);
  position IDs restart at len(prefix) for every branch; bf16, SDPA or
  eager (whichever accepts the 4D mask), no generation.
- Readout: z_k = logsumexp(Yes set) − logsumexp(No set) at each
  branch's final position — token sets identical to the frozen judge.
- **Arm C (counterfactual, the story's ablation):** identical packed
  token sequence and readout positions, FULL causal attention (no
  mask), standard sequential position IDs. Same readout.
- Statistics, computed with the diagnostic's frozen machinery
  (`rank_auc`, same gold, same contrasts): within-hate-video pooled +
  macro AUC; span chunks vs non-hate-video chunks; per-chunk
  Spearman/max-|Δz| of Arm M vs the sequential reference.
- Efficiency accounting (descriptive, no bar): wall-clock per video for
  Arm M vs (i) N sequential isolated calls, (ii) plain left-padded
  batch of N isolated prompts. Reported as the honest three-regime
  decomposition; the claim is "one forward, prefix computed once", not
  a fabricated 1/N cost.

## Frozen decision rule

Direction 3 **SURVIVES** only if BOTH hold:

1. **Fidelity (implementation equivalence):** Spearman(Arm M z,
   sequential z) ≥ 0.99 over all 2281 chunks, AND every frozen AUC
   contrast within ±0.01 of the sequential reference (0.624
   within-pooled, 0.720 cross-video).
2. **Counterfactual gap (contamination story):** Arm C within-hate-video
   macro AUC ≤ Arm M within-hate-video macro AUC − 0.05. Prediction:
   Arm C collapses toward the packed-questions result (≈0.52).

**DIES** if clause 2 fails with clause 1 passing: the mask is not
load-bearing, the contamination story is false, and no rescue variant
may be tried. Clause 1 failure alone is an implementation defect, not a
hypothesis result: ONE bounded debugging pass (position IDs / seam
tokenization / mask dtype) is permitted; if fidelity still fails, the
direction is recorded as implementation-infeasible on this stack.

## Interpretation boundaries

- Survival licenses the method-stage prereg ONLY: LAVAD-convention
  pooled frame-level AUC on HateMM vs LELA's published numbers (GPT-4o
  Mini 72.6 at 12–16 calls/frame; best open 7B 64.7), extension
  corpora, multimodal-prefix ablation, label-free thresholding, and an
  honesty section reporting the within-video boundary (0.624) that the
  pooled metric does not measure. No number from this pilot is a paper
  claim.
- Deployment shape: detection call (frozen multimodal judge) +
  locator call (this packed pass) = 2 calls per video, distinct roles,
  within the ≤2-call cap.
- Span gold is evaluation-only; no label touches any computation.
