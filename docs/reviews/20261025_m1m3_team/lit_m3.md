# M3 literature scout (2026-10-25)

Task: find work on how a video-level (global) decision and local segment scores are combined, and how scores are made
comparable across videos, without labels. Then say which mechanisms could change what the M3 video level *is*,
not just re-weight it. Written by the M3 literature scout of the M1/M3 novelty team (charter.md in this folder).
No repo edits were made.

## 0. How to read this file

**Verification tags** (how far each entry was checked in this session):
- **[read]**: I read the method section or the relevant equations in this session.
- **[abs]**: I checked the arXiv / proceedings / publisher page for title, authors, venue and abstract. Any mechanism
  detail beyond the abstract is marked "(recalled)".
- **[repo]**: an earlier repo document already verified it (path given). I did not reopen it.
- **[unverified]**: named only. Do not cite without checking.

**What M3 does now.** M3 forms the key K = z_video + mean window z and calibrates it without labels, using a two-Gaussian
mixture, into logit P(V=1|K). The frame score is the calibrated key plus the within-video centred rank of the M2
posterior.

**Repo facts that rule out the textbook form of several mechanisms below:**
- Most read variance is between videos: visual 86% / 75%, speech 69% / 49% (HateMM / HCS).
- EM treats each window as independent evidence and overstates it 5–14×.
  Source: `experiments/20260926_twolevel/README.md` §10.2 items 1–2.
- Round 1 was a joint model with a video latent V gating the chains and corpus-shared emissions. It lost pooled
  (HateMM pooled PR .5499 against .6888). Source: twolevel §1 and §10.2 item 6; DIRECTIONS row "09-26 第 1 轮".
- Five principled video levels lost at least .013 pooled PR on at least one corpus. These were the joint posterior, ICC-effective
  reads, a two-view mixture, the product form and the expected hate fraction (twolevel §10.2 item 6, §11.1).
- The 39 M3 composition variants gave at most +.003 to +.005. The per-video proportion MLE has ρ .295 / .516 with the GT share.
  Window mean: .416 / .533. The segment-model Bayes factor key lost .034 / .107 pooled
  (`experiments/20261025_m3_coverage/README.md` §13).
- Re-asking at the video level collapses onto the verdict (ρ .965 / .904). Context-free segment reads lose the
  share information (same README §8–§12).
- An idea-stage cross-view Dawid–Skene model with a video intercept (LESS, 2026-08) was tried. An unbalanced-OT alignment
  was proposed but never tested (`research-wiki/DIRECTIONS.md` §C).

---

## A. Training-free / zero-shot video anomaly detection and temporal localization: what each does at the video level

**A1. T3AL: Test-Time Zero-Shot Temporal Action Localization.** CVPR 2024, arXiv 2404.05426. [repo:
`research-wiki/paradigm_framing.md` §3, `docs/reviews/20260927_final_novelty_review.md`]
- Mechanism:
  - A video-level pseudo-label is the argmax of the mean frame feature against class text.
  - Frames are scored against that class, thresholded at the video's own mean score, and smoothed by a moving average.
  - Test-time adaptation runs per video.
- Named failure it answers: noise from non-informative frames; a fixed threshold.
- Video level: it only selects the class. It is never a score, and nothing makes scores comparable across videos. The
  per-video mean threshold is centring within the video.
- Map to M3: the repo's stated difference already rests on this ("ours enters every window's read").
- Used for hate: no.

**A2. FreeZAD: Training-Free Zero-Shot Temporal Action Detection with Vision-Language Models.** arXiv 2501.13795
(Jan 2025; no venue stated on arXiv). [read §3.1, §5.2]
- Mechanism:
  - A video-level pseudo-label works as in T3AL.
  - Each segment's score is p · s. Here p is LogOIC = mean inside − 0.2 × log-decay-weighted outside + max, and s is
    sigmoid(FFT energy of the segment's features), the "actionness calibration".
  - The threshold is "the average similarity score across the entire video".
- Video level: class selection only. Each segment is bounded by its own sigmoid, and nothing normalizes across videos.
- Map to M3: none beyond T3AL. It confirms that zero-shot TAL has no cross-video term.
- Used for hate: no.

**A3. LAVAD: Harnessing LLMs for Training-free Video Anomaly Detection.** CVPR 2024, arXiv 2404.01014. [repo:
paradigm_framing §3]
- Mechanism:
  - The LLM gets a temporal summary as global context.
  - Scores are refined by a softmax-weighted average over the K = 10 nearest frames of the same video.
- Video level: context only. There is no video score.
- Map to M3: none (it is an M1-type precedent).
- Used for hate: no.

**A4. VERA.** CVPR 2025, arXiv 2412.01095. [repo: paradigm_framing §3]
- Mechanism: scene-context ensembling over the top-K similar segments of the same video, Gaussian smoothing and
  position weights.
- Video level: none. Nothing makes scores comparable across videos.
- Used for hate: no.

**A5. VADTree.** arXiv 2510.22693. NeurIPS 2025 per paradigm_framing; the arXiv page shows only a NeurIPS checklist.
[read, first 100k characters]
- Mechanism:
  - It builds a hierarchical tree per video by splitting at high-confidence boundaries.
  - k = 2 means over that video's boundary confidences splits nodes into coarse and fine.
  - Frame score = parent/child fusion weighted by the normalised within-cluster variance of the children's scores
    (Eq. 9–10).
- Video level: no video score. The clustering is inside one video, and no cross-video normalisation is described.
- Map to M3: none. Its "two-cluster" step is within a video.
- Used for hate: no.

**A6. EventVAD: Training-Free Event-Aware Video Anomaly Detection.** arXiv 2504.13092. Listed as ACM MM 2025 in its own
reference list; the arXiv PDF header is a template, so the venue is not confirmed from the ACM DL. [abs]
- Mechanism: dynamic spatiotemporal graph → event boundaries by signal-ratio thresholding → the MLLM scores each event
  with hierarchical prompting.
- Video level: none.
- Used for hate: no.

**A7. Anomaly-Guided Observation and Two-Level Semantic Feedback for Training-Free Video Anomaly Analysis.**
arXiv 2610.16324 (12 Oct 2026; no venue). [read: the feedback equations]
- Mechanism:
  - A frozen video-LM writes one whole-video description.
  - A lexical parser maps it to normalish / unclear / mixed / anomalyish.
  - That state chooses one video-level operator on the existing frame curve:
    - suppress: × 0.40;
    - keep;
    - sharpen: frames above max(0.45, the video's own 0.70-quantile) are floored at 0.85, and the others are × 0.80.
  - A second, local level raises event windows after visual checking.
  - The authors state: "The transformed scores are not interpreted as calibrated anomaly probabilities."
- Named failure: the frame curve ignores what the whole video is about.
- Map to M3: the video decision acts on the *shape* of the within-video distribution, not as an additive offset.
  The closest repo test is the per-video rank width 4P(1−P), which lost .003–.010 (m3_coverage §13). Its constants are
  fixed by hand, and the per-video quantile makes scores less comparable across videos.
- Used for hate: no.

**A8. A VLM Answer Is Not an Anomaly Score: Rank Compression Across Image and Video Anomaly Detection.** arXiv
2608.21244 (Aug 2026; no venue). [abs]
- Mechanism: scoring the decoded answer creates ties ("decoded-answer rank compression"). A probability-weighted score
  over all answers recovers ≥ 95% of the gap, with mean gains of 7.66–19.95 points.
- Named failure: ties destroy ranking across inputs.
- Map to M3: it supports reading log-odds rather than decoded answers, which r6 already does. It gives no new video level.
- Used for hate: no.

**A9. C2FPL: A Coarse-to-Fine Pseudo-Labeling Framework for Unsupervised Video Anomaly Detection.** WACV 2024,
arXiv 2310.17650, DOI 10.1109/WACV57701.2024.00665. [read: method]
- Mechanism:
  - **Coarse step:** each unlabeled video is summarised as [mean, std] of its segments' feature magnitudes. Divisive GMM
    clustering splits off anomalous videos.
  - **Fine step:** a Gaussian null is fitted on the segments of the *pseudo-normal videos*.
  - Each pseudo-anomalous video is scored against that shared null. The window of length ⌈0.2 × video length⌉ with the
    lowest average density is labelled anomalous.
- Named failure: no labels at all, so there is no reference for "normal".
- Map to M3: the only label-free video method found where the video-level partition defines a *shared null* that
  local scores are tested against. That gives cross-video comparability by construction (see top-5 item 3).
  The fixed 20% window is a hand constant that we would not copy.
- Used for hate: no. WACV is outside the listed venues.

**A10. AnomalyRuler.** ECCV 2024, arXiv 2407.10299. [repo]
- Mechanism: rule induction and reasoning, plus exponential majority smoothing.
- Video level: none.

**Summary of family A.**
- No training-free VAD or TAL method combines a calibrated video-level probability with local scores.
- None of them makes frame scores comparable across videos. The video level is either class selection (T3AL, FreeZAD)
  or a hand-set operator (A7).
- The label-free shared-null idea exists only in unsupervised VAD (C2FPL), where it produces training pseudo-labels.

## B. Weakly supervised fusion rules (mechanism sources only: all use video labels)

**B1. STPN: Weakly Supervised Action Localization by Sparse Temporal Pooling Network.** CVPR 2018, arXiv 1712.05080.
[repo]
- Mechanism: attention-weighted pooling gives video class scores. Localization uses T-CAMs weighted by the video-level
  class score ("classify, then localize").
- Map to M3: the product / gating form; c_full in the repo is its probabilistic version.

**B2. W-TALC: Weakly-supervised Temporal Activity Localization and Classification.** ECCV 2018, arXiv 1807.10418.
[abs; k-max pooling recalled]
- Mechanism: the video score is the mean of the top-k frame activations (k-max MIL), plus a co-activity similarity loss.
- Map to M3: the "upper mean" key form was tried and is closed.

**B3. RTFM: Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning.** ICCV 2021,
arXiv 2101.10030. [abs; top-k magnitude recalled]
- Mechanism: the video is treated as a bag. The mean of the top-k snippet feature magnitudes separates normal from
  abnormal videos.
- Map to M3: the same family as B2.

**B4. Attention-based Deep Multiple Instance Learning.** ICML 2018, arXiv 1802.04712. [abs]
- Mechanism: a permutation-invariant attention pooling of instance embeddings gives the bag label. The attention
  weights act as instance scores.
- Map to M3: a weighted window mean. Without labels, the weights have no target.

**B5. MILBoost: Multiple Instance Boosting for Object Detection.** NeurIPS 2005, pp. 1417–1424. [read: §2]
- Mechanism: noisy-OR, P(bag) = 1 − Π_j (1 − p_ij). The gradient weight on an instance shrinks as its bag's probability
  approaches its target.
- Map to M3: the classic way to make the bag probability and the instance probabilities *one* model instead of two scores.
  In r6, the OR across modality chains is the same algebra within a frame. Across windows, noisy-OR overcounts
  correlated windows, which is the same failure as round 1.

**B6. MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos.** WWW 2026, arXiv
2512.10408. [abs]
- Mechanism: a modality-aware MIL objective under video-level labels, plus cross-modal contrastive alignment.
- Used for hate: yes (HateMM, MultiHateClip), weakly supervised.
- Map to M3: a comparator, not a mechanism source.

**Summary of family B.** Every fusion rule here is learned or tuned with video labels. The noisy-OR algebra is the only one
that defines both levels in one probability model, and it assumes independent instances.

## C. Two-level probabilistic models where a global latent gates local states, inferred jointly

**C1. Structured Models for Fine-to-Coarse Sentiment Analysis.** McDonald, Hannan, Neylon, Wells, Reynar. ACL 2007,
P07-1055. [read: §1–§2]
- Mechanism:
  - The document label node connects to every sentence label in a linear-chain model ("the label of each sentence is
    dependent on the labels of its neighbouring sentences plus the label of the document").
  - Inference loops over the document labels, runs Viterbi for each and keeps the best ("constrained Viterbi").
    Forward–backward works the same way for marginals.
- Named failure (their words): "a cascaded approach would also be insufficient. Valuable information is passed in both
  directions."
- Map to M3: the cleanest statement of the M3 problem. r6 is a cascade:
  - the verdict goes into the prefix;
  - windows are read under it;
  - the windows come back only through their mean.
  The repo's round-1 model was the generative version of C1 and lost because of evidence overcounting. See top-5 items
  1 and 2 for what must change.
- Training: supervised in the paper, but the inference is label-free.
- Used for hate: no.

**C2. Discovering Fine-Grained Sentiment with Latent Variable Structured Prediction Models.** Täckström & McDonald.
ECIR 2011. [abs]
- Mechanism: a hidden CRF treats sentence labels as latent and learns them from document-level labels only.
- Map to M3: the coarse-to-fine direction. It needs document labels.

**C3. Clustering Sequences with Hidden Markov Models.** Smyth. NIPS 1996 (Advances in NeurIPS 9, 1997), pp. 648–654.
[abs]
- Mechanism: a mixture of HMMs. A sequence-level latent picks which HMM generated the whole sequence, and its posterior
  uses the full forward likelihood. It is fitted by EM, with cross-validated likelihood to pick the number of clusters.
- Map to M3: the generative form of "V gates the chain". It is closed in the repo as round 1 (twolevel §1).

**C4. Sharing Features among Dynamical Systems with Beta Processes (BP-AR-HMM).** Fox, Sudderth, Jordan, Willsky.
NeurIPS 2009, pp. 549–557. Journal-style version arXiv 1111.4226. [abs]
- Mechanism:
  - A shared library of behaviours serves all sequences.
  - Each sequence has a binary feature vector that says which behaviours it may visit (beta-process / IBP prior).
  - Exact MCMC samples the features and the state paths jointly. Acceptance uses sum-product over each sequence's chain.
- Named failure: each series exhibits only a subset of behaviours, and a single HMM forces all series to share all states.
- Map to M3: the video level becomes "is the violating behaviour switched on for this video, and how much of the
  video does it use". Emissions are shared across the corpus, which keeps scores comparable.
- Closest closed: round 1 (one fixed library, V = f_v), the three-layer reads (09-28) and the per-video π MLE.
  It is only new if combined with a per-video random effect (top-5 item 1).
- Used for hate: no.

**C5. The Hierarchical Hidden Markov Model.** Fine, Singer, Tishby. Machine Learning 32(1):41–62, 1998, DOI
10.1023/A:1007469218079. [abs] Also: **Unsupervised discovery of multilevel statistical video structures using
hierarchical HMMs.** Xie, Chang, Divakaran, Sun. ICME 2003. [abs]
- Mechanism: recursive HMMs whose top-level states emit whole sub-HMMs. Xie et al. fit them to soccer video *without
  labels*: they learn both the parameters and the structure, and find play / break segments slightly better than a
  supervised baseline.
- Map to M3: the video and segment levels as one generative hierarchy. A video-domain, label-free precedent, but old
  and not top-venue (ICME).

**C6. Variational Learning for Switching State-Space Models.** Ghahramani & Hinton. Neural Computation 12(4):831–864,
2000, DOI 10.1162/089976600300015619. [abs]
- Mechanism: a discrete switch selects among linear-Gaussian regimes. Variational inference couples HMM recursions
  (for the switch) with Kalman recursions (for the continuous state).
- Map to M3: a continuous per-video level (a Kalman state) under a discrete hate switch. This is the dynamic version of
  item 1 in the top-5.
- Used for hate: no.

**C7. Mixed Hidden Markov Models: An Extension of the HMM to the Longitudinal Data Setting.** Altman. JASA
102(477):201–210, 2007, DOI 10.1198/016214506000001086. [abs]
- Mechanism: many sequences, one HMM. Per-sequence random effects (and covariates) enter the emissions and/or
  transitions, inside the generalised-linear-mixed-model framework. It is estimated by marginal likelihood.
- Named failure: independent-sequence HMMs ignore that sequences differ systematically. Pooled HMMs ignore that
  observations within a sequence share a sequence-level effect.
- Map to M3: this is exactly the repo's documented failure. 86% / 75% of read variance is between videos, and the
  round-1 EM overstated per-window evidence 5–14×. See top-5 item 1.
- Used for hate or video: no.

**C8. Latent Dirichlet Allocation.** Blei, Ng, Jordan. JMLR 3:993–1022, 2003. [abs]
- Mechanism: a per-document proportion θ (Dirichlet prior) is inferred jointly with per-word topic assignments.
  Topics are shared across the corpus.
- Map to M3: the video level as a *proportion* (share) jointly inferred with window states, under shared emissions.
  The repo's per-video π MLE is the no-pooling, per-video-emission version (ρ .295 / .516), which differs.

**C9. Multi-Level VAE: Learning Disentangled Representations from Grouped Observations.** Bouchacourt, Tomioka, Nowozin.
AAAI 2018, arXiv 1705.08841. [abs; the product-of-Gaussians group posterior is recalled] Also **Towards a Neural
Statistician.** Edwards & Storkey. ICLR 2017, arXiv 1606.02185. [abs]
- Mechanism: a group-level latent (content / context) shared by all members, inferred by pooling evidence from all
  members, plus per-member latents.
- Map to M3: the deep analogue of C7. It is useful as modern top-venue wording for "a video-level latent shared by its
  windows".
- Used for hate: no.

**C10. Unsupervised Learning of Action Classes with Continuous Temporal Embedding.** Kukleva, Kuehne, Sener, Gall.
CVPR 2019, arXiv 1904.04189, DOI 10.1109/CVPR.2019.01234. [read §3.5]
- Mechanism:
  - When video content is unknown, each video gets a soft bag-of-words over embedded frame clusters.
  - Videos are clustered into K′ sets, label-free.
  - "For each video set, we then separately infer clusters for subactions and assign them to each video frame" with
    Viterbi.
- Map to M3: a top-venue, label-free video precedent in which a video-level grouping gates the local models.
  The middle ground between per-video EM (M2 now) and corpus EM (round 1) is per-video-set emissions.

**C11. Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation (ASOT).** Xu & Gould.
CVPR 2024 (oral), arXiv 2404.01518. [read: formulation]
- Mechanism:
  - Per video, a fused Gromov–Wasserstein OT problem assigns frames to actions, with a temporal-consistency structure.
  - The frame marginal is hard.
  - The action marginal is soft: λ · KL(Tᵀ1 ‖ q) toward a uniform prior q, so "the distribution of actions observed
    within a video" is decided by the data (λ = 0.01 at inference).
- Named failure: balanced OT (TOT, SwAV-style) forces equal action proportions.
- Map to M3: the share becomes a *solved variable* (the column mass of the plan), determined jointly with the
  temporally consistent assignment. See top-5 item 4.
- Used for hate: no. In the repo, "unbalanced OT alignment" was proposed in 2026-08 but never tested.

**C12. Posterior Regularization for Structured Latent Variable Models.** Ganchev, Graça, Gillenwater, Taskar.
JMLR 11:2001–2049, 2010. [abs]
- Mechanism: constraints on the expectations of latent variables are imposed on the posterior during EM, without labels.
- Map to M3: the label-free way to impose the proportion constraints that 3C-Net and ARMS learn with labels. The
  video-level belief would enter M2's posterior as an expectation constraint rather than an additive offset.
  The constraint value has no label-free source except the verdict, so this is weak.

## D. Cross-instance score calibration, prevalence estimation and ranking without labels

**D1. Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure.** Saerens, Latinne,
Decaestecker. Neural Computation 14(1):21–41, 2002, DOI 10.1162/089976602753284446. [abs]
- Mechanism: EM re-estimates the class prior of a new unlabeled set and rescales the posteriors.
- Map to M3: run per video, it gives a per-video share. This is closed in the repo as SLD (ρ .389 / .437 against the window
  mean .416 / .533; novelty_eval/M3_report.md).

**D2. Maximum Likelihood with Bias-Corrected Calibration is Hard-to-Beat at Label Shift Adaptation.** Alexandari,
Kundaje, Shrikumar. ICML 2020, arXiv 1901.06852. [abs]
- Mechanism: calibrate first (BCTS), then run D1's EM. Miscalibration is what breaks EM prevalence estimates.
- Map to M3: this names the failure of D1 on our reads. The window reads are badly calibrated as evidence (twolevel §10.2
  item 2), so per-video prevalence EM inherits that error.

**D3. Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection.** Perini, Bürkner, Klami.
ICML 2023, PMLR 202:27668–27679, arXiv 2210.10487. [abs]
- Mechanism: the scores of several detectors are the input to a Bayesian mixture, which yields a posterior over the
  share of anomalies without labels. Its mean sets the threshold.
- Map to M3: a per-video contamination posterior would give a share with uncertainty. The prior scout tested the
  related BBE / PCC forms, which lost.

**D4. Testing for Outliers with Conformal p-values.** Bates, Candès, Lei, Romano, Sesia. Annals of Statistics
51(1):149–178, 2023, DOI 10.1214/22-AOS2244, arXiv 2104.08279. [abs]
- Mechanism: the rank of a test score among a reference set of inliers gives a valid p-value. The p-values are
  positively dependent, so BH controls FDR. It needs a clean reference set.
- Map to M3: with C2FPL's pseudo-benign videos as the reference, every window gets a p-value on a scale common to all
  videos (top-5 item 3). M2's corpus rank-normal transform is the version with a contaminated reference.

**D5. Simultaneous Testing of Grouped Hypotheses: Finding Needles in Multiple Haystacks.** Cai & Sun. JASA
104(488):1467–1481, 2009, DOI 10.1198/jasa.2009.tm08415. [abs] Also **D6. Simultaneous Inference: When Should
Hypothesis Testing Problems Be Combined?** Efron. Annals of Applied Statistics 2(1):197–223, 2008, DOI
10.1214/07-AOAS141. [abs]
- Mechanism:
  - Cai & Sun: the oracle ranking across groups uses each group's own non-null proportion and density (the conditional
    local fdr). It uniformly beats both pooled and separate analysis.
  - Efron: separate-class fdr, and "enrichment" (is this subclass richer in non-nulls?).
- Map to M3: the formal statement that the cross-video frame order should mix a group-level (video) quantity with a
  within-group density. The repo's c_full and decoder-probability arms are this form. They lost because EM
  overstated the evidence (scout_report3, twolevel §23), so this form is closed. It is useful as a citation for
  *why* a video term is needed at all.

**D7. Modeling Score Distributions for Combining the Outputs of Search Engines.** Manmatha, Rath, Feng. SIGIR 2001,
DOI 10.1145/383952.384005. [abs]
- Mechanism: per query, a mixture is fitted to the unlabeled scores, with an exponential for non-relevant documents and
  a Gaussian for relevant ones. Scores are converted to probabilities of relevance, which makes them comparable across
  queries and engines.
- Map to M3: the IR precedent for M2's per-video two-class fit and M3's calibration. Its known weak point is a query with
  no relevant documents, which still gets a fitted relevant component. Our benign videos are that case.

**D8. Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering.** Zhou et al. ICLR 2024,
arXiv 2309.17249. [abs]
- Mechanism: the contextual bias is estimated as the mean output over the unlabeled batch and removed, with no labels.
- Map to M3: corpus-level centring. r6's calibration already does more, so there is nothing to transfer.

**D9. Ranking and Combining Multiple Predictors without Labeled Data (Spectral Meta-Learner).** Parisi, Strino,
Nadler, Kluger. PNAS 111(4):1253–1258, 2014, DOI 10.1073/pnas.1219097111, arXiv 1303.3257. [abs]
- Mechanism: under conditional independence, the off-diagonal covariance of the predictors is rank one. Its leading
  eigenvector ranks the predictors and gives the ensemble weights.
- Map to M3: label-free weights for verdict, visual mean and speech mean. This is pure re-weighting, and the repo's
  two-view mixture and LESS (Dawid–Skene) are the same idea. Closed.

**D10. Whose Vote Should Count More (GLAD).** Whitehill, Ruvolo, Wu, Bergsma, Movellan. NeurIPS 2009. [abs]
- Mechanism: the true label of each item, the expertise of each labeler and the *difficulty of each item* are
  inferred jointly, with no labels.
- Map to M3: a per-video "readability" latent that scales how much the window reads count, which addresses the
  speech-driven false highs. Closest closed: LESS (2026-08) and the modality gate (09-30: the per-video gate could not
  pick the right modality, ROC ≈ .5). Weak.

**D11. Brümmer & Garcia-Romero, unsupervised score calibration.** ICASSP 2014, arXiv 1311.0707. [repo]
- Mechanism: identical to r6's two-Gaussian key calibration. Not claimable.

**Summary of family D.**
- Every label-free calibration or prevalence method found is either corpus-level (D7, D8, D11 ≈ r6 now) or per-instance
  prevalence EM (D1–D3), which is closed on our reads.
- Grouped-hypothesis ranking (D5, D6) is the theory that justifies a video term. Its plug-in form is closed.

## E. LLM / MLLM confidence aggregation from local to global

**E1. FActScore.** Min et al. EMNLP 2023, arXiv 2305.14251. [abs]
- Mechanism: the response is split into atomic facts. The score is the fraction of facts supported (classify-and-count).

**E2. SelfCheckGPT.** Manakul, Liusie, Gales. EMNLP 2023, arXiv 2303.08896. [abs; passage score = mean of sentence
scores, recalled]
- Mechanism: sentence-level consistency against sampled responses, averaged into a passage score.

**E3. LUQ: Long-text Uncertainty Quantification for LLMs.** Zhang, Liu, Basaldella, Collier. EMNLP 2024 (main), arXiv
2403.20279. [read: equations]
- Mechanism: an NLI entailment probability per sentence. It is averaged over sentences, then over samples, then over
  responses. The method is entirely means.

**E4. Graph-based Uncertainty Metrics for Long-form LLM Outputs.** Jiang, Ruan, Sattigeri, Roukos, Hashimoto. NeurIPS
2024 (spotlight), arXiv 2410.20783. [abs]
- Mechanism: a bipartite graph links generations and claims. Claim uncertainty is a graph centrality (closeness
  beats the degree used by self-consistency).

**E5. Atomic Calibration of LLMs in Long-Form Generations.** Zhang et al. Findings of IJCNLP-AACL 2025, arXiv
2410.13246. [abs]
- Mechanism: per-claim confidence, with fusion of discriminative and generative confidence methods.

**E6. ConCoRD: Enhancing Self-Consistency and Performance of Pre-Trained LMs through NLI.** Mitchell et al. EMNLP 2022,
arXiv 2211.11875. [abs] Also **Maieutic Prompting.** Jung et al. EMNLP 2022, arXiv 2205.11822. [abs]
- Mechanism: the model's own beliefs (with confidences) and the logical relations between them form a factor graph.
  Weighted MaxSAT picks the most probable consistent assignment, so the global answer is *revised* by the local beliefs.
  No training.
- Map to M3: an LLM-side precedent for joint decoding of the verdict with the window beliefs (C1). Without new reads, its
  constraint "V = 1 ⇔ some segment of ≥ 2 windows" is the round-1 model, which is closed. With reads under both verdicts,
  see top-5 item 2.

**Summary of family E.**
- All response-level scores built from claim-level scores are means or fractions, which is classify-and-count. That is
  r6's window mean, so this family gives a citation for "the naive form of M3 is the standard one" and no better
  mechanism.
- Only E6 does joint inference, and its relational form maps to C1.

## F. Hateful content detection with frozen models: what is used at the video level

**F1. LELA: Towards Training-free Multimodal Hate Localisation with LLMs.** arXiv 2602.09637 (no venue found). [read:
fusion]
- Mechanism: per-frame, per-modality LLM scores. "We select the maximum score across all modalities". τ = 0.5 binarises
  the result.
- Video level: none ("can support video-level aggregation", with no rule given). No normalisation within or across
  videos is described.

**F2. MARS: Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning.** Yang,
Zhang, Fu. ICASSP 2026, arXiv 2601.15115. [abs]
- Mechanism: the VLM gathers the strongest evidence for and against hate, then synthesises one decision.
- Video level: video-level only, with no localisation and no stated score form.

**F3. Pro-Cap.** ACM MM 2023, arXiv 2308.08088. [repo]
- Mechanism: zero-shot probing of a frozen VLM, for hateful memes.

**F4. MultiHateLoc** (B6). WWW 2026. Weakly supervised MIL.

**Summary of family F.**
- No frozen-model hate work found combines a video-level score with local scores, and none makes scores comparable
  across videos.
- M3's slot is empty in hate, so there is no precedent to borrow, but the slot also earns no novelty by itself.

---

## G. Literature mechanisms whose textbook form is already closed in this repo

| Mechanism (source) | Repo attempt | Result | Source |
|---|---|---|---|
| V gates chains, joint posterior (Smyth 1996; McDonald 2007 generative form; noisy-OR) | round-1 two-level model | HateMM pooled PR .5499 vs .6888; EM overstates evidence 5–14× | twolevel §1, §10.2 |
| Grouped-hypothesis ranking (Cai & Sun 2009; Efron 2008) | c_full; decoder probability | mixed / lost | twolevel §11.5, §23 |
| Per-video prevalence EM (Saerens 2002; BBE; PCC) | SLD / PCC shares | ρ below window mean | novelty_eval/M3_report.md |
| Per-video proportion, no pooling (LDA without sharing) | per-video π MLE | ρ .295 / .516 | m3_coverage §13 |
| Design-effect discount (crude random effect) | ICC-effective reads | HateMM .8939 / .6756 vs .8956 / .6888 | twolevel §10.2 item 6 |
| Fixed video effect | centring before EM | within .7030 / .6138 | twolevel §10.2 item 3 |
| Latent class over views (Dawid–Skene; SML) | LESS (2026-08); two-view mixture | lost | DIRECTIONS §C; twolevel §10.2 |
| Video decision reshapes the curve (A7) | per-video rank width 4P(1−P) | −.003 to −.010 | m3_coverage §13 |
| Direct share read (G-Eval-style) | candidates 47, 48 | collapses to verdict | m3_coverage §8–§10 |
| Hypothesis-conditioned local reads (HVL) | HVL | 83% of windows positive under a hypothesis; another video's hypothesis works equally | DIRECTIONS A2 |

---

## H. Most transferable to M3 (top 5)

The ranking criteria were: (a) it answers a failure the repo has already documented; (b) its form is not closed in G;
(c) it has a top-venue or classic mechanism source; (d) a cheap pre-check exists. None was found in hate. All five
change what the video level is: a latent inside the segment model, a reference set, or a solved share. None of them is
a new weight.

**1. The video level as a per-video random effect inside the segment model, counted once.**
- Sources: Altman JASA 2007 (mixed HMM). Modern wording: ML-VAE (AAAI 2018) and Neural Statistician (ICLR 2017).
  Discrete middle ground: Kukleva CVPR 2019 video sets.
- Why it fits: the repo's principled joint video level failed for a documented reason. 86% / 75% of read variance is
  between videos, and per-window evidence was overstated 5–14× because windows that share a video-level shift were
  treated as independent.
- What changes: a mixed HMM makes the shift a latent b_v whose prior depends on V. The level then enters the evidence
  for V once, and the windows add evidence only through their pattern relative to b_v. The key stops being a separate
  module (verdict + mean, calibrated afterwards). It becomes the posterior of the video latent of the M2 model, and frame
  posteriors share one corpus scale.
- Closest closed attempts:
  - round-1 joint model: no random effect;
  - ICC-effective reads: a discount, not a model;
  - fixed-effect centring: throws the level away.
- Risk: if the within-video pattern carries little evidence about V, it reduces to today's calibrated K and gains
  nothing (the +.003–.005 room).
- Its claim would rest on "one model for both levels" and on its ablation, not on gains.
- Pre-check: CPU on `runs/20260926_glr/base_gridA`. Compare the ρ of E[share | reads] and of P(V | reads) with the GT
  share against the window mean (.416 / .533), and pooled against r6.
- Cost: no GPU.

**2. Joint (two-way) decoding of the verdict with the window sequence, using reads under both verdict values.**
- Sources: McDonald ACL 2007 (constrained Viterbi over document labels), ConCoRD / Maieutic EMNLP 2022 as the LLM-side
  precedent.
- Why it fits: McDonald names the exact failure of r6's cascade: "Valuable information is passed in both
  directions". The round-1 model already had to give non-hate windows of violating videos their own mean (μ_10 vs μ_00,
  the "stance leak", twolevel §1), because the verdict in the prefix shifts every window read. It modelled that shift
  but did not remove it.
- What changes: on the shared prefix, fork the stance turn as forced "Yes" and forced "No", and read the windows under each.
  Then decode the label with the M2 chain run under each (C1's algorithm). The video level becomes "the global
  reading under which the window sequence is coherent". The window evidence becomes the contrast z(w | Yes) − z(w | No),
  which removes the shared confirmation shift.
- Closest closed attempts:
  - HVL: confirmation bias, with reads taken at face value;
  - stance conditioning: kept, but one-sided;
  - CVA: an attribution difference on the verdict;
  - candidate 48.
- Risk: the HVL result suggests that forced-Yes reads rise everywhere, which would leave the contrast flat.
- Pre-check: 20 videos. Is the within-video std of the contrast above .5 logit? What is its ρ with the GT share?
- Cost: one extra set of window branches per video, with the prefix reused. That is about one M1 window pass of GPU time
  per corpus.
- This is the only one of the five that adds information, which matters given that "current reads lack the
  information".

**3. A shared null from pseudo-benign videos: windows tested against it, share = per-video discovery rate.**
- Sources: C2FPL (WACV 2024), conformal p-values (Annals of Statistics 2023), Efron AoAS 2008 (separate-class).
- Why it fits: M2's per-video two-class EM fits a hate class in every video, so its posterior means "high for this
  video". That forces M3 to add a separate key (the same weak point as Manmatha's per-query fit on queries with no
  relevant documents).
- What changes: the video level becomes the partition that defines the reference null. A window's evidence is a
  conformal p-value against the windows of videos the corpus calls benign, which puts every window of every video on one
  scale. A per-video estimate 1 − π̂0 from those p-values gives the share.
- Closest closed attempts:
  - CC / PCC / SLD shares: ρ at or below the window mean;
  - M2's corpus rank: its reference set is contaminated.
- Risks: classify-and-count estimators have lost to the window mean before. The pseudo-benign partition is itself built
  from the verdict, which is circular.
- Pre-check: CPU. Compare the ρ of 1 − π̂0_v with the GT share against the window mean.
- Cost: no GPU.

**4. The share as a solved, corpus-regularised variable (unbalanced OT).**
- Source: ASOT (CVPR 2024 oral). Posterior regularization (JMLR 2010) is the probabilistic counterpart.
- Why it fits: the framing says "how long the act runs decides video ordering", but r6's video level is binary.
- What changes: in ASOT the per-video action proportions are the column marginal of a transport plan, decided jointly
  with a temporally consistent assignment and pulled toward a prior by λ·KL. For M3:
  - the prior q is set from the corpus (cross-video coupling);
  - the cost comes from the reads;
  - the column mass is the share key.
- Closest closed attempts:
  - M2's expected-hate-fraction key (twolevel §11.1: HateMM .8632 / .6180, below r6);
  - candidates 47–49.
- Differences: unbalanced OT was proposed in 2026-08 and never tested. The share is a decision variable regularised
  across the corpus, not a posterior mean of a per-video fit.
- Risk: with the current reads it may reproduce the M2 share.
- Pre-check: CPU, via a POT-style Sinkhorn on cached reads.
- Cost: no GPU.

**5. Video sets gate local models (intermediate pooling).**
- Sources: Kukleva CVPR 2019. BP-AR-HMM (NeurIPS 2009) is the sparse-feature version.
- Why it fits: the repo has tried both extremes. One corpus-wide emission model (round 1) overcounts. Per-video EM (M2)
  makes scores incomparable across videos.
- What changes: cluster videos label-free into a few sets by their read distributions, and fit the segment emissions per
  set. In the BP-AR-HMM form, each video carries a binary indicator of whether it may visit the violating behaviour. The
  video level becomes set membership or the indicator. It is inferred together with the window states, and frames within
  a set share one scale.
- Closest closed attempts:
  - round 1 (one set);
  - M2 (one set per video);
  - the three-layer reads (09-28);
  - the reading-condition groups (09-28, M2: all six numbers within noise, and a shuffled-group control matched them).
- Risk: few videos per set, and the verdict may again decide the sets.
- Pre-check: CPU, with K′ ∈ {2, 3} fixed in advance.
- Cost: no GPU.

**Honest overall reading.**
- Items 1, 3, 4 and 5 re-express the current reads. The 2026-10-25 round showed that the current reads hold little more
  video-level information (composition room +.003–.005). Their value for M3 is mainly that they give a claimable
  mechanism: one model for both levels, or a common null. Their gains would likely be small.
- Item 2 is the only one that changes what is measured. It is also the one closest to a known failure (HVL confirmation
  bias).
- I found no hateful-video work, and no training-free VAD or TAL work, that uses any of the five.

---

## I. Not verified / not opened (do not cite without checking)

- Learning-from-label-proportions works (Quadrianto et al. JMLR 2009; ∝SVM, Yu et al. ICML 2013): [unverified]. Named
  only as the supervised counterpart of item 4.
- 3C-Net (ICCV 2019) and ARMS (IEEE TMM 2024): abstract-only, from the earlier M3 scout (`scratchpad/m3_team/scout_report.md` §4).
- Recurrent switching LDS (Linderman et al., arXiv 1610.08466): the arXiv page gives no venue. I believe it is AISTATS 2017, but
  I did not confirm that.
- Forman 2008 (classify-and-count / ACC / PCC): a textbook source, not opened.
- Zero-shot Action Localization via the Confidence of LVLMs (arXiv 2410.14340): no venue stated. Its video-level details
  were not checked.
- LAVIDA (arXiv, 2026): seen only in a search snippet; it uses pseudo-anomaly training data.
- TDA (Karmanov et al., CVPR 2024, a test-time cache of VLM predictions): the search failed in this session. [unverified]
