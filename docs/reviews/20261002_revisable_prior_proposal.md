# Revisable prior proposal review — PASS

Date: 2026-10-02. Independent reviewer instance; proposal review only, before candidate implementation/results.

Reviewed proposal: `experiments/20261002_revisable_prior/README.md`.
Scope: exactly the four STOP grounds in `RESEARCH_ITERATION_RULES.md` rule 4. No code or proposal was edited by this reviewer. PASS permits implementation and testing; it does not establish effectiveness, publication novelty, or promotion.

## Decision

**PASS.** The proposal replaces conditional independence of repeated reads with a jointly fitted latent shared offset inside the temporal observation model. The offset is marginalized alongside the video regime and temporal states. This is an adaptation of mixed-effects/segmental hidden-state modeling, not a new statistical model family. The searched primary sources did not establish prior application of this mechanism to hateful video detection/localization. The new candidate is sufficiently distinct from this repository's earlier attempts to justify the declared cached pilot.

| Rule-4 STOP ground | Finding |
|---|---|
| Source method already applied to hateful video detection/localization | Not found in the search below. Random-effects segmental HMMs themselves are established in waveform modeling. Domain transfer remains a qualified, search-limited novelty claim. |
| Pure ensemble | No. A single MLLM supplies global and local observations; integration over one model's latent variables is not a combination of independently trained predictors. |
| Pure calibration / postprocessing / smoothing | No as specified. The shared offset enters the likelihood and EM sufficient statistics and changes conditional local state inference. The fixed rank-based output is inherited; neither that output nor normal-score transformation should be claimed as the new mechanism. |
| Pure engineering technique | No. This is a changed dependency structure and fitted generative inference model, not only prompt wording, encoder replacement, constants, or resolution. |

## What actually changes relative to prior attempts

Read: `experiments/20260911_hvl/README.md` §8; `experiments/20260926_twolevel/README.md` §§1–2, 10.2; candidate README.

- **Versus HVL:** no generated TARGET/FORM/EVIDENCE hypothesis, sequential answer continuation, or second self-revision prompt. Local queries use the original multimodal context without the appended global Q/A. Revisability occurs in statistical inference, not another self-generated narrative. Existing HVL failures therefore remain relevant warnings about prompting, but do not make this the same implementation.
- **Versus twolevel r1:** r1 already had a latent video variable, global Gaussian observation, local state emissions and joint Bayesian inference. Therefore **“make the verdict soft” and “local evidence can revise the video estimate” are not new relative to r1**. The substantive change is the common random effect that induces dependence among global and window observations. Conditional on that effect, the chains are tractable; after marginalization, their evidence is no longer counted as independent repetitions. Explicit durations, duration integration, normal scores, removal of the at-least-one constraint, and the output composition are inherited later changes, not the new contribution.
- **Versus previous centering and ICC pilots:** a posterior over a latent offset is not identical to subtracting the observed per-video mean or multiplying likelihood terms by a fitted temperature. The model can preserve uncertainty about whether a high mean comes from states or the shared offset. Whether it does so usefully is an empirical question; there is no guaranteed correction of previous PR losses.

The defensible mechanism description is **joint inference with shared observation variability**. Calling the latent offset specifically “context bias” is a hypothesis: observational reads alone do not prove that context caused the fitted offset. The global log odds are a noisy observation in this model, not themselves a calibrated probability or literally the generative prior on V.

## Primary-source search

Searches were actually run on 2026-10-02. Queries included `"hateful video" "random effects"`, `"hateful video" "random effect"`, `"hateful" "video" "random-effects"`, `"hateful video" "Bayesian" localization`, `"HateMM" "Bayesian"`, `"hateful video" "hidden" "Markov"`, `"hate" "localization" "hidden Markov"`, and `"hidden Markov" "random effects" model paper`. Exact conjunctions did not identify a hateful-video implementation of the candidate dependency structure. This is absence of evidence in the search, not proof of universal absence.

Sources opened and inspected:

1. [Kim and Smyth, *Segmental Hidden Markov Models with Random Effects for Waveform Modeling*, JMLR 2006](https://www.jmlr.org/papers/v7/kim06a.html). A primary source for combining random effects, segmental hidden states and EM in waveform classification/segmentation. This must be acknowledged as methodological ancestry; the proposed statistical ingredients are not invented here. [Full paper](https://www.jmlr.org/papers/volume7/kim06a/kim06a.pdf).
2. [*MultiHateLoc*, §3](https://arxiv.org/html/2512.10408v1). Modality-specific temporal encoders, cross-modal alignment/fusion and video-label MIL. Its described mechanism does not marginalize a shared Gaussian observation offset with a soft video regime.
3. [*LELA*, §3](https://arxiv.org/html/2602.09637v1). Training-free modality captions, composition matching, multi-stage prompting and per-frame modality score aggregation. This is close in application and use of LLM readings, but does not describe the proposed random-effect temporal likelihood.
4. [*CLARA*, §3](https://arxiv.org/html/2608.15905v1). Utterance-aligned clips, MoE fusion, local-global contrastive learning and rationale-gated temporal integration. Broad “global context helps local evidence” language would overlap with this work; its implemented mechanism differs from the proposed joint random-effect model.
5. [*Reasoning-Aware Multimodal Fusion for Hateful Video Detection*, §3](https://arxiv.org/html/2512.02743v1). Objective descriptions plus hateful/non-hateful assumed reasoning and trained hierarchical fusion. It already uses contrasting interpretations; the candidate should not claim that merely considering both verdicts is novel. It does not describe the proposed shared-offset marginalization.

Only primary papers support the comparisons above. Search-engine summaries and third-party reviews were not treated as evidence for a method's contents.

## Mathematical implementation points to resolve/check

These are repairable specification/implementation issues, **not additional STOP conditions**.

1. **Integrate the shared latent variable after modality fusion.** Given V and u, the visual and speech chains factorize; after integrating u they generally do not. At each u, integrate each chain's duration variable, compute the conditional OR probability, then average using the joint posterior over u. Taking OR of already u-marginalized modality probabilities is incorrect.
2. **Use the correct conditional weights.** The rank branch requires `P(h_t=1 | V=1, reads)`, so node weights there must normalize conditional on V=1. Unconditional frame probabilities additionally multiply by `P(V=1 | reads)`. Do not count the regime probability twice.
3. **EM statistics must preserve dependence.** Mean/variance updates need joint expectations over u, V and each observation's OR state, including their cross-products. The global likelihood occurs exactly once per video, not once per window or modality. Start-state statistics should be weighted by V=1 responsibilities. The V=0 likelihood still includes all observed local reads with state zero.
4. **Quadrature convention and the tau update must match.** For physicists' Hermite nodes, use the square-root-of-two node rescaling and normalize weights by the square root of pi; an already standard-normal convention needs neither. Fixed standard-normal nodes with `b=tau*u` support the declared weighted least-squares update. Constrain tau nonnegative or consistently account for the symmetry of a signed coefficient; blindly clipping a fitted coefficient is not automatically the EM maximizer. Check marginal likelihood behavior.
5. **Class semantics need an explicit check.** If fitted global or local class means cross, the positive state must not silently become the negative class. Initialization alone does not guarantee ordered fitted means. Record and handle this consistently without labels; do not quietly relabel V while leaving the V=0 all-zero constraint unchanged.
6. **Coarse quadrature can also create a false negative.** Seven fixed nodes may poorly represent a concentrated random-effect posterior for long videos. The proposed 15-node frozen-parameter check is useful; apply an accuracy check before attributing a bad result to the mechanism as well as before claiming success. Exactness applies to the chain computation conditional on a node, not to the continuous integral approximated with seven nodes.
7. **Retain the declared brute-force tests.** Include a two-modality case with nonzero tau so an incorrect post-integration OR is detectable, a no-speech case, truncated final windows, and tau=0 equivalence to the candidate's own independent arm. This arm is not expected to reproduce historical r1 numerically because other inherited model choices differ.

## Risks to diagnose after implementation

No risk below is used to block the candidate: the common offset can absorb persistent real hate as well as shared bias; equal loadings across transformed global/visual/speech reads may be restrictive; the latent active regime allows all-zero paths and is not literally an OR label; joint fitting can still overstate evidence; and ordinal normal-score confidence need not correspond to semantic certainty. Report the declared ablations, original global-error subgroups, contradictory-global perturbations, fitted parameters and three metrics. Restrict claims to this older matched Reader family until a later matched latest-family run exists.

There are no new GPU calls in the cached pilot. The extra quadrature cost and all numerical failures should be reported, including unsuccessful arms. The current paper and method remain unchanged unless the repository's empirical gates pass.
