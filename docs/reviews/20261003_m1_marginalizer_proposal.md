# M1 Marginalizer: independent proposal review

Date: 2026-10-03. Decision: **STOP — rule 4, source mechanism already used in hateful video detection.**

Reviewed `experiments/20261003_m1_marginalizer/README.md` and `RESEARCH_ITERATION_RULES.md`. This is the single independent proposal review for this candidate, before implementation or performance inspection. No code, declaration, predictions or GT were changed. Attributor's pending results play no role in this decision.

## Decision and its scope

The operative mechanism is to read the same video under both hateful and non-hateful assumptions, then combine those reads instead of committing to one interpretation. That mechanism already appears in **MARS** and **RAMF**, both explicitly for hateful video detection. Under rule 4, detection is included; moving the existing mechanism to the current window reader does not establish a new transfer to the target task.

The proposed computation is not an exact reproduction of either paper. Its distinctive implementation uses canonical assistant Yes/No continuations, reads local class probabilities, and takes their fixed arithmetic mean. MARS generates competing evidence and uses a further model call to synthesize it; RAMF encodes competing reasoning into a trained fusion model. **The literature search did not establish that either paper uses the exact proposed probability average.** The novelty judgment here is that replacing the aggregation of an already-used dual-assumption mechanism with a uniform mean does not constitute a new source mechanism under this repository's gate. This is a mechanism-level judgment, not a claim of identical equations or code.

The symmetry argument does not change that judgment: enumerating both conditions removes dependence on selecting just one condition by construction. It does not add evidence selection, evidence verification, or a new interaction with the temporal observations. The existing window reader and decoder supply those observations and temporal structure unchanged. Calling the average counterfactual marginalization does not by itself turn this variation into a distinct method transferred from outside hateful-video research.

## Primary literature actually checked

1. **MARS: Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning**, Yang, Zhang and Fu. [Primary full text](https://arxiv.org/html/2601.15115v1), especially §2.2.2–2.2.4. The model considers a hateful assumption and a non-hateful assumption using the same frames and transcription, obtains supporting evidence and reasoning for both, and then synthesizes a decision. This is a training-free target-task precedent, so the training-free distinction cannot avoid it. Its final synthesis is a model judgment, not an arithmetic probability average.

2. **RAMF: Reasoning-Aware Multimodal Fusion for Hateful Video Detection**. [Primary full text](https://arxiv.org/html/2512.02743v1), especially §3.2–3.3. It generates objective, hate-assumed and non-hate-assumed descriptions, then encodes the reasoning for multimodal fusion. The stated motivation includes compensating for a flawed reasoning path using the complementary perspective. This directly precedes the broad claim that opposing assumptions mitigate dependence on a mistaken single interpretation. Its fusion and supervision differ from this candidate.

3. **Self-Consistency Improves Chain of Thought Reasoning in Language Models**. [Primary paper](https://arxiv.org/abs/2203.11171). Sampling reasoning paths and aggregating answers is relevant general background. The candidate instead enumerates two fixed conditioning answers without sampling reasoning. This reference does not establish novelty over MARS or RAMF. Searches also returned target-task papers citing self-consistency; a citation alone is not evidence of its use as their inference algorithm.

4. **IARE: Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful Videos**. [Primary full text](https://arxiv.org/html/2606.11953v1). Checked the self-consistency citation and its body reference: it appears in §2.3 as general LLM reasoning background. I do not count that citation as evidence that IARE implements the proposed enumeration or probability averaging.

5. **Counterfactual Fairness**. [Primary paper](https://arxiv.org/abs/1703.06856). Its causal modeling framework is not instantiated by this reader. The README correctly disclaims that guarantee. Changing an assistant answer is an intervention on dialogue conditioning; the average is not demonstrated to be a posterior over the true video label.

Additional discovery hit: [LEAF primary PDF](https://aclanthology.org/2026.findings-acl.604.pdf), read through PDF text extraction. Its bibliography cites self-consistency. This was not used as evidence of exact target-task probability marginalization.

Searches on 2026-10-03 included combinations of `hateful video` / `hate video` / `HateMM` with `marginalization`, `marginalisation`, `probability averaging`, `self-consistency`, `self consistency`, `counterfactual conditioning`, `counterfactual reasoning`, and `hate-assumed non-hate`, plus the exact MARS title. No exact fixed-uniform local Yes/No mixture was verified. This absence is a search boundary, not a novelty guarantee. The STOP rests on the verified dual-assumption target-task antecedents and the limited remaining methodological difference.

## Four STOP conditions

| Rule 4 condition | Assessment |
|---|---|
| Source mechanism already used in hateful video detection/localization | **Triggered.** Dual opposing assumptions followed by combined judgment already occur in MARS and RAMF; the remaining distinction is a simpler aggregation inside the existing local reader. |
| Pure ensemble | Not independently triggered. The same frozen model is reused; rule 3 defines the prohibited ensemble using multiple independent models. |
| Pure calibration / postprocessing / smoothing | Not independently triggered. It actually obtains additional conditional model reads; it is not solely a transformation of the current saved scores. |
| Pure engineering | No separate STOP is needed. Cache reuse is engineering, but the decision above concerns the already-used source mechanism. I do not equate every structured prompting method with mere wording changes. |

## Existing experiments and falsifiability

The reviewed local records do not show a completed experiment with this exact two-answer probability mixture. GLR (`experiments/20260926_glr/README.md`) scores spoken-word likelihoods under speaker hypotheses; the earlier revisable-prior candidate models scalar bias outside the MLLM. They are not identical experiments, and their failures are not grounds for this STOP. Grounder and Selector failures are likewise irrelevant to the novelty decision.

The proposed fixed-Yes, fixed-No, no-verdict, mean-logit, time-misalignment and shift controls are useful and could falsify a performance explanation. In particular, equal performance from one fixed answer or no verdict would defeat the enumeration claim; surviving only video offsets would defeat a localization claim. These are sound empirical checks, but cannot establish that the source mechanism was previously unused in the target task. I am not stopping because of anticipated collapse, shortcuts, uncertainty, or likely low gains.

## Disposition

Archive this proposal as stopped before implementation under rule 4. No additional pilot or generalized review is required. Do not represent the exact probability formula as previously published, and do not represent dual-assumption reading as newly introduced to hateful-video analysis. A rename, another averaging domain, or a revised prompt would not resolve the cited overlap. A subsequent candidate needs a different substantive operation on evidence beyond enumerating and aggregating opposite verdict conditions.
