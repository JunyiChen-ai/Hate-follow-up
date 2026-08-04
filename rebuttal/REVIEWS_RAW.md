# Raw Reviews — ACL ARR 2026 May, Submission 9795

**Paper**: TRIAGE: Label-Free Hateful Video Detection with Boundary Mapping and Resolution
**Venue**: ACL ARR 2026 May (OpenReview, per-reviewer threads)
**Reviews received**: 3 (Reviewer ARuf, Reviewer cpjh, Reviewer 7rqV)
**Captured**: 2026-07-09, verbatim from OpenReview.

---

## Reviewer ARuf (06 Jul 2026)

**Paper Summary:**
The paper proposes an efficient label-free hateful video detection method that employs both a lightweight and a relatively stronger MLLM. A boundary mapper, which utilizes a lightweight MLLM, is prompted to determine whether a certain video segment contains hateful content with a simple yes/no prompt. Then, the authors extract the hate-saliency score from the normalized next token probability of "Yes." Samples with higher entropy are additionally routed to the adaptive boundary resolver, which is based on a set of stronger, larger MLLMs. This phase sequentially adapts the probability that the boundary regions contains hateful material, until sufficient evidence has been obtained or the panel of MLLMs is exhausted.

**Summary Of Strengths:**
- The explored problem of label-free hate video detection is a practically relevant and interesting field.
- The authors present thorough analyses of each proposed technical component to separately verify its effect.
- The authors validate both the effectiveness and efficiency of the proposed method.

**Summary Of Weaknesses:**
1. Can the proposed method be extended to settings where hate video constitutes severe minority, which is probably a more realistic setting (e.g., <10% of the sample set is hateful, and the reset are innocuous)?
2. Although I acknowledge that methodological novelty is not always tied to the quality of research, I do the feel the need to point out that many of the technical concepts are borrowed from existing literature. In some sense, I believe some people could view this paper as an engineering application or extension of previously proposed methods. For a similar reason, the related works section could be improved by discussing a broader range of works that leverages multi-(M)LLMs for specific, interesting applications.
3. Also, this paper is not entirely irrelevant to the *ACL community, in that it uses MLLMs, but the target task seems a bit more suitable for more CV-oriented venues. Therefore, I am doubtful whether this work would be of much interest to the *ACL attendees even if it were to be accepted. (I humbly note that this decision should be made by ACs or SACs)

**Scores:** Confidence 3 / Soundness 4 / Excitement 3 / Overall 3 (Findings) / Reproducibility 5 / Software 5

---

## Reviewer cpjh (02 Jul 2026)

**Paper Summary:**
This paper proposes a label-free hate video detection (HVD) framework called TRIAGE. It addresses two key issues: 1) although supervised HVD methods demonstrate excellent performance, they require labour-intensive manual annotation, which can be costly and potentially have adverse effects; 2) existing MLLM-based label-free or few-shot methods often perform time-consuming and resource-intensive inference on each video, whilst still underperforming compared to supervised methods. To address these issues, TRIAGE first identifies uncertain cases and then allocates additional computational resources solely to these cases, thereby alleviating the trade-off between effectiveness and efficiency.

Specifically, a boundary mapper uses a lightweight MLLM to generate 'yes/no' review probabilities, and normalises the probability of the next token being 'yes' to serve as a hate salience score; subsequently, the paper utilises the score distribution of all samples in the unlabelled video pool to perform unsupervised calibration on these raw scores, converting them into more reliable posterior probabilities. Samples with lower scores are classified directly, whilst those with higher scores are assigned to a borderline region. For samples within the borderline region, the adaptive borderline resolver sequentially invokes more powerful MLLM verifiers and integrates the binary judgements of each verifier via Bayesian odds updates. When the posterior uncertainty falls below a threshold, or after all verifiers in the group have been utilised, the system ceases inference and provides a final prediction.

**Summary Of Strengths:**
- By using a lightweight model to identify uncertain samples in borderline regions, and then applying a more powerful model only to these cases, the trade-off between performance and efficiency is mitigated.
- TRIAGE combines next-token probability extraction, unsupervised calibration, entropy-based selection, sequential MLLM verification, and Bayesian posterior updates. With modular components, the system can be implemented using different mapper models, validator models, calibration schemes or backend detectors.
- Across four datasets in label-free/few-shot scenarios, TRIAGE achieves higher accuracy than supervised baseline models and others label-free methods.

**Summary Of Weaknesses:**
1. TRIAGE is a multi-MLLM collaborative system that utilises models of varying sizes, with default configurations ranging from 2B to 72B. Whilst the use of smaller models for simple cases forms part of the scheme's efficiency design, the system as a whole can still invoke 72B-scale models when processing difficult samples. The authors need to clarify whether the label-free or few-shot baseline models were also evaluated using the same maximum model size of 72B.
2. The set of supervised model baselines should include more recent detection models. The lack of recent supervised learning baselines makes it difficult to substantiate the reliability of the claim that 'systems based on label-free MLLMs can rival or surpass supervised learning methods'.
3. The paper primarily discusses efficiency in terms of inference time; however, actual deployment costs also depend on the hardware required to host large models (deployment of 72B-scale models). It would be of significant value for the authors to discuss and report on TRIAGE's performance when the validator pool is restricted to smaller models (e.g. only 7B/8B-scale MLLMs).

**Scores:** Confidence 5 / Soundness 3 / Excitement 3 / Overall 3.5 (Borderline Conference) / Reproducibility 4 / Software 3

---

## Reviewer 7rqV (01 Jul 2026)

**Paper Summary:**
This paper addresses the challenges of high annotation costs in hateful video detection tasks by proposing TRIAGE, a label-free framework based on MLLMs. The method employs a pipeline comprising a Boundary Mapper and an Adaptive Boundary Resolver, utilizing a set of MLLM verifiers to iteratively update posterior probabilities for video samples. The authors conduct extensive experiments on four public datasets to validate the effectiveness and robustness of the proposed method.

**Summary Of Strengths:**
1. Critical and Relevant Problem: The paper tackles the significant challenges of high annotation costs and the topic is relevant to the ACL venue.
2. Promising Performance Achievements: competitive and superior performance compared to some supervised and label-free baselines.
3. Comprehensive Evaluation: detection performance, ablation studies, efficiency analyses, and evaluations across different MLLM backbones.

**Summary Of Weaknesses:**
1. The methodological novelty is limited, as the framework primarily combines existing MLLMs via a coarse-to-fine pipeline, which is a relatively common paradigm in the literature.
2. The baseline selection is insufficient, particularly as it omits comparisons with post-training based LLM methods and recent strong baselines like SAGE (ACL 2026).
3. Some counter-intuitive results, such as the label-free approach outperforming certain supervised methods, are not adequately explained or analyzed. Is it unclear whether the performance improvement to the supervised approaches is primarily attributed to the strong backbone MLLMs, rather than the proposed method itself.

**Scores:** Confidence 4 / Soundness 3.5 / Excitement 2.5 / Overall 3 (Findings) / Reproducibility 3 / Software 3
