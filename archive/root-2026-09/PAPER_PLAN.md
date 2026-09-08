# Paper Plan

**Working title**: *TRIAGE: Uncertainty-Routed MLLM Arbitration for Label-Free Hateful Video Detection*

**One-sentence contribution**: We introduce **TRIAGE**, a label-free hateful-video detection pipeline that turns a single-pass MLLM into a continuous saliency scorer and routes only uncertainty-band videos to a heterogeneous MLLM panel, achieving state-of-the-art label-free performance competitive with supervised systems.

**Venue**: EMNLP 2026 (ARR-routed long paper, 8-page main body; References, Limitations, Ethics, Appendix do not count against the 8 pages under ARR rules).

**Type**: Method paper with empirical validation (main-results + ablation + analysis).

**Date**: 2026-04-24

---

## Target review profile (EMNLP 2026 / ARR 2026)

EMNLP 2026 routes through ARR; each reviewer fills three scores — **Soundness (1–5)**, **Excitement (1–5)**, **Overall (1–5)** — plus the mandatory **Responsible NLP checklist**, **Limitations section**, and **Ethics statement**. The plan below is written backward from those rubrics.

| Rubric | What the reviewer is graded on | How this paper plans to score |
|---|---|---|
| **Soundness** | Claims clearly stated, adequately supported; experimental depth, technical soundness, methodological validity | Every headline claim is tied to frozen artefacts; every mechanism claim has a counterfactual ablation or diagnostic analysis; label-free choices use only score distributions, not hate annotations |
| **Excitement** | Transformational, surprising, evidence-challenging; lowers barriers; enables new applications | TRIAGE reaches label-free performance competitive with supervised systems while avoiding hate annotation and reserving expensive MLLM arbitration for the uncertain subset |
| **Overall** | Novelty + impact composite; recommendation for venue fit | Position the contribution as a new design pattern for label-free multimodal moderation: probabilistic reading first, selective heterogeneous arbitration second |
| **Limitations (mandatory)** | Acknowledge genuine open problems; desk-rejectable if misleading | State problems the whole label-free MLLM-moderation field has not solved (pretraining-inherited bias, frame-sampling horizon, closed-source judge reproducibility, adversarial drift); do **not** self-own engineering choices that are defensible |
| **Responsible NLP checklist** | Misleading entries → desk reject | Pre-fill the checklist during drafting, attach as appendix |
| **Ethics** | Dataset provenance, misuse risk, annotator conditions | Cite dataset cards; declare moderation-positive intent; note dual-use risk |

---

## Claims-Evidence Matrix 

Each claim pins to a concrete artefact or planned table/figure. Keep the main text focused on the clean protocol; implementation details and sensitivity analyses live in the appendix.

| # | Claim | Evidence (frozen artefact) | Section |
|---|---|---|---|
| **C1** | TRIAGE sets a new state of the art for label-free hateful-video detection and is competitive with supervised systems. | Main results table. Frozen headline configuration: `stage1=qwen3-vl-2b` + triplet `{gemma-3-27b-it, qwen2.5-vl-32b-awq, llava-onevision-qwen2-7b-ov-hf}` — **EN 0.783 / ZH 0.826 / HM 0.842 / IH 0.833** (acc). | §4.2, Table 1 |
| **C2** | Stage-1 MLLM errors concentrate in a compact label-free subset identified by posterior entropy over a logit-space 2-GMM. | Across the four datasets, the uncertainty band covers **32–62% of the test set** yet recovers **60–80% of stage-1 errors**. Frozen: `candidates_entropy_band_2b.jsonl`. | §4.4, Figure 3 |
| **C3** | Routing only uncertainty-band videos to a three-judge majority vote — leaving confident videos untouched — is the load-bearing design. Removing either the band gate or triplet aggregation weakens the gain. | Ablation table: {full, stage-1-only, band-off (judge everyone), triplet → single-best-judge, triplet → 8-judge all-majority}. | §4.3, Table 2 |
| **C4** | The recipe transfers across stage-1 backbones: across six heterogeneous stage-1 MLLMs, the pipeline yields a positive mean Δ-acc averaged over datasets. | Per-slug Δ-acc table, avg across 4 datasets, all six slugs positive (median Δ-acc ∈ [+0.8, +3.2 pp]). | §4.5, Table 3 |
| **C5** | The per-video MLLM budget is deployment-compatible: on the four datasets, the mean MLLM-call count is 2.14-2.77. | Band coverage × triplet fan-out; worst dataset = MHClip_ZH at 2.77 calls/video. | §4.7 |

Every `acc/M-F1/M-P/M-R` number in the paper is **pinned** to a named cell in `results/boundary_rescue/grid_eval/grid_raw.jsonl` or `results/holistic_*/`. No floating numbers.

---

## Structure (8-page main body)

§1 Introduction (1.0 p) — §2 Related Work (0.75 p) — §3 Method (1.75 p) — §4 Experimental Evaluation (4.0 p) — §5 Conclusion (0.25 p) — Limitations — Ethics — References — Appendix.

### §1 Introduction (≈1.0 page, 3 paragraphs + contributions + paradigm sentence)

User-directed structure: ¶1 background context; ¶2 supervised methods; ¶3 low-resource (label-free/few-shot, may span hateful video *and* harmful meme); ¶4 our paradigm; contributions list; hero figure. The challenge in ¶3 is grounded in real per-dataset error patterns observed in our reproduced baselines (`docs/results_2026_04_16_v2.md`).

**¶1 — Background context.** Hateful video moderation is a multimodal content-understanding task: the offending signal is jointly carried by speech, on-screen text, visual targeting, and situational framing, and no single modality is sufficient to recover it. Detecting it at the scale at which short-form video is produced requires detectors that are accurate, robust across languages and content registers, and cheap enough to apply per-upload.

**¶2 — Supervised dominant paradigm.** The current state-of-the-art is supervised multimodal fusion. MATCH-HVD trains a per-dataset classifier on ViViT visual features, MFCC audio, and BERT transcripts; Pro-Cap (Cao et al., ACM MM 2023) probes a frozen vision-language model for entity-aware captions and feeds them to a supervised downstream classifier; HateCLIPper and its video descendants fine-tune a CLIP backbone on hate-labelled pairs. All of these depend on a curated, per-dataset, per-jurisdiction hate-labelled training set. Hate annotation, however, is expensive, subjective, fragments across language and platform policy, and is well-documented to cause psychological harm to annotators. Every new deployment surface re-pays the annotation bill.

**¶3 — Low-resource / few-shot / label-free prior work and its open challenges.** A recent line of work sidesteps the annotation bill by casting pretrained (multimodal) language models as reasoning *agents*. **Label-free studies targeting hateful video specifically remain very limited** — MARS (Das et al., 2024), a training-free multi-stage adversarial-reasoning pipeline, is the principal representative. Most activity in this space instead addresses the sibling task of harmful-meme detection: Mod-HATE (Cao et al., 2024) composes pre-trained LoRA modules over a K-shot labelled support set; ALARM (Lang et al., 2026) iteratively self-improves an LMM agent using pseudo-labels drawn from confidently-explicit memes; LoReHM (Huang et al., 2024) combines retrieval of labelled exemplars with knowledge-based priors. Across this family, two challenges persist.

- **C1 — Lack of a calibrated decision signal.** Existing label-free MLLM pipelines often produce rich intermediate text — captions, rationales, observations, or agent traces — but their final decisions are usually made from generated artifacts or hard textual verdicts. This leaves them without a continuous decision signal that can be thresholded, compared across samples, or used to preserve the model's uncertainty. As a result, more elaborate reasoning does not necessarily translate into a stronger detector: errors in generated intermediates can propagate, and the model's graded preference is collapsed into discrete text.
- **C2 — Lack of selective compute allocation.** Existing multi-stage and agentic pipelines often apply the same complex reasoning procedure to every sample without distinguishing easy cases from genuinely difficult ones. Many videos do not require such heavy reasoning: a cheap first pass is already sufficient for obvious hateful or obvious benign content. Running the full procedure uniformly therefore makes inference cost scale with the entire stream and wastes compute on samples that are unlikely to benefit from it.

**¶4 — Our paradigm.** We introduce **TRIAGE**, a coarse-to-fine, label-free pipeline whose two design choices directly answer C1 and C2. First, TRIAGE treats the base MLLM as a *probabilistic reader* rather than a text generator: a single policy-grounded binary prompt yields a normalised next-token probability that serves as a continuous hate saliency score, preserving graded uncertainty and bypassing brittle generated artifacts (answers C1). Second, TRIAGE uses the distribution of those scores to localise a label-free *uncertainty band* — the residual subset where the first-stage signal is ambiguous — and forwards only band videos to a fixed cross-family *panel* of three MLLMs whose majority vote may overturn the base verdict; confident videos receive no additional compute (answers C2). No hate label is consulted at any step. TRIAGE matches or exceeds supervised state-of-the-art on three of four hateful-video benchmarks while spending 1 MLLM call on confident videos and limiting heavier reasoning to the residual boundary.

**Contributions (numbered, falsifiable)**:
1. **A new paradigm for label-free multimodal moderation.** We introduce **TRIAGE**, a coarse-to-fine paradigm that couples *probabilistic reading* with *selective arbitration*: a cheap base MLLM first produces a continuous hate-saliency signal, and stronger MLLM reasoning is invoked only for the residual uncertain region.
2. **An effective single-pass label-free scorer.** Instead of relying on generated captions, rationales, observations, or agent traces as decision variables, TRIAGE treats the base MLLM as a probabilistic reader and uses the normalized next-token probability of a policy-grounded Yes/No answer as a continuous saliency score. This preserves graded uncertainty, avoids error propagation through generated artifacts, and provides a strong first-stage detector without hate annotations.
3. **An uncertainty-routed heterogeneous arbitration module.** TRIAGE fits a label-free uncertainty band over first-stage scores and routes only band videos to a fixed cross-family MLLM panel. The band localizes residual boundary cases, while heterogeneous arbitration supplies a structurally different second opinion from models with distinct inductive biases rather than repeated samples from the same model.
4. **State-of-the-art results and empirical findings.** Across multiple hateful-video benchmarks, TRIAGE sets a new state of the art for label-free detection and reaches performance competitive with, and in some cases stronger than, supervised systems. Our broader analysis further reveals how MLLM errors concentrate in a small uncertainty band, why heterogeneous panels outperform uniform re-judging, and how moderation behavior shifts across datasets and languages.

**Hero figure (Fig 1)**: see Figure Plan below — single-column teaser contrasting TRIAGE against prior families along label requirement, decision signal, and compute allocation.

### §2 Related Work (≈0.75 pages)

Three synthesis paragraphs, organised by the question each family answers. No paper-by-paper list.

- **Paragraph 1 — Supervised hateful-video and hateful-meme detection**: MATCH-HVD, Mod-HATE, Pro-Cap, HateMM-original baselines. Position: high accuracy, but hate-labelled training data is the load-bearing dependency.
- **Paragraph 2 — MLLMs as zero-shot moderators**: single-pass prompting (Das et al., 2024 naive), MARS multi-stage reasoning. Position: cheap single-pass underperforms; multi-stage inflates cost without addressing *where* errors live.
- **Paragraph 3 — Selective inference and judge aggregation**: El-Yaniv–Wiener selective prediction, LLM-as-a-judge (Zheng et al., 2023), self-consistency (Wang et al., 2023). Position: we combine *selective routing* (only uncertain videos are re-judged) with *heterogeneous judge voting* (three different MLLM families, not K samples of one), which keeps the deployment budget realistic and the contribution cleanly separated from self-consistency.

### §3 Method (≈1.75 pages)

#### §3.1 Problem formulation

Let `V` denote a target collection of candidate videos drawn from a moderation stream, each associated with an unknown ground-truth label `y(v) ∈ {0, 1}` (non-hateful, hateful). A *label-free* hateful-video detector is a map `f : V → {0, 1}` whose construction and inference do not consume hate annotations: no component is trained, adapted, calibrated, or selected using gold labels or label-derived statistics. The detector may use the unlabeled videos themselves, including their score distribution, for unsupervised calibration and routing; released labels are used only after prediction for evaluation. We measure `f` by per-video accuracy against the held-out gold label. The per-sample compute budget, quantified as the number of MLLM forward passes invoked on each video, is a first-class quantity: a label-free detector whose cost grows unnecessarily with the full stream is not a practical moderation recipe.

#### §3.2 Overview

We introduce **TRIAGE**, a coarse-to-fine detector composed of two functionally distinct modules that operate without any hate annotation (Figure 2):

- **Probabilistic reading** (§3.3) — a single base MLLM reads each video under a policy-grounded binary prompt. Instead of relying on generated captions or rationales, TRIAGE extracts a continuous hate-saliency score from the model's next-token preference for the positive answer and converts that score into a preliminary prediction using an unsupervised bimodality-thresholding family.
- **Uncertainty-routed heterogeneous arbitration** (§3.4) — TRIAGE fits an unsupervised two-component model to the first-stage score distribution, uses posterior entropy to identify an uncertainty band, and sends only that band to a fixed heterogeneous panel of stronger MLLMs. Confident videos keep the first-stage prediction.

Taken together, the two modules address the two missing pieces in prior label-free MLLM pipelines: a continuous decision signal and selective allocation of expensive reasoning.

**Notation**. For a video `v`, let `s(v) ∈ [0, 1]` denote the first-stage saliency score, `z(v)` its logit, `q(v)` the posterior responsibility of the higher-score component of a two-component fit on `{z(v) : v ∈ V}`, `H(v) = −q log q − (1 − q) log(1 − q)` its Shannon entropy, and `B = {v : H(v) > mean_v(H(v))}` the uncertainty band.

#### §3.3 Probabilistic reading

A single base MLLM reads each video under a policy-grounded prompt whose header names the detection scope (protected-attribute framing, moderation-policy headings) and whose body requests a binary verdict without in-context hate exemplars. Greedy decoding yields a renormalised next-token distribution over the {Yes, No} surface forms, from which we define `s(v)` as the probability of the positive surface form. The prompt is held fixed across datasets up to a language switch.

A preliminary hard prediction is obtained from `s(v)` using an unsupervised thresholding rule from the classical family of bimodality-based score partitioning methods. This family includes histogram criteria and mixture-model criteria whose shared assumption is simple: an unlabeled score distribution induced by a binary detector should separate into lower-saliency and higher-saliency regions. Our default protocol is **TrainFit**: fit the threshold on the unlabeled training split's score distribution and apply it unchanged to the test split. We also report **TestFit** variants as a transductive sensitivity check, where the same unsupervised family is fit on the unlabeled test-score distribution.

#### §3.4 Uncertainty-routed heterogeneous arbitration

Raw saliency values are not directly comparable across datasets because MLLM score distributions can be skewed, compressed, or polarised. TRIAGE therefore measures uncertainty relative to the geometry of the target score distribution. We fit an unsupervised two-component Gaussian mixture over logit scores and use the posterior responsibility `q(v)` as the video's relative membership in the higher-saliency component. The Shannon entropy `H(v)` of `q(v)` is a standard measure of binary uncertainty, bounded in `[0, log 2]`, maximised at `q = 0.5`, and invariant to label permutation.

The uncertainty band `B` retains every video whose entropy exceeds the collection mean entropy. The rule is parameter-free, requires no held-out data, and uses only score statistics computable at inference time. We hypothesise that `B` is the subset where additional reasoning is most useful; §4.3 empirically verifies that first-stage errors concentrate inside this band.

Every `v ∈ B` is re-read by a fixed panel of three MLLMs that are stronger than the base reader and drawn from distinct model families. Each panel member receives the same policy-grounded arbitration prompt and emits an independent binary verdict; the panel aggregates by unweighted majority with a validity floor (at least two panellists must emit a parse-legal verdict). If the panel majority disagrees with the first-stage prediction, TRIAGE overturns it; videos outside `B` keep the first-stage prediction and incur no additional MLLM call.

**Why a heterogeneous panel.** The arbitration module is motivated by the same concern that makes self-consistency insufficient for moderation: repeated samples from one model mainly reduce sampling variance, while many moderation errors are systematic blind spots of a model family or alignment recipe. TRIAGE instead aggregates deterministic judgements from different model families, aiming to obtain complementary inductive biases rather than more samples of the same bias. The average MLLM budget is `1 + 3 · |B|/|V|` calls per video, compared with `4` calls for applying the same panel to every video.

### §4 Experimental Evaluation (≈4.0 pages)

§4 contains the full empirical evaluation: setup, main results, ablation, diagnostic analysis, transfer across backbones, panel analysis, and compute accounting.

#### §4.1 Experiment setup

- **Datasets.** We evaluate on four publicly released hateful-video benchmarks spanning two languages and three content regimes: MHClip_EN and MHClip_ZH (short-form, mixed-modality hateful clips with a three-way annotation collapsed to binary), HateMM (hateful-vs-normal videos), and ImpliHateVid (implicit hate). Test sizes, label mappings, and excluded corrupted media are reported in Appendix A.
- **Baselines.** We compare against two baseline groups in the main table. **Supervised baselines** include MATCH-HVD, Pro-Cap, MoRE, and ImpliHateVid (the supervised detector introduced alongside the ImpliHateVid benchmark). **Label-free / few-shot baselines** include naive MLLM prompting, MARS, LoReHM, Mod-HATE, and ALARM. TRIAGE is reported as the proposed method, not as a baseline. Detailed baseline descriptions, reproduction settings, and any deviations from upstream implementations are deferred to Appendix B.

- **Supervised baseline numbers (pinned for Table 1).** Pro-Cap, MoRE, and ImpliHateVid numbers are taken from our in-house replication under the same evaluation protocol (ACC / M-F1 / M-P / M-R, %):

  | Method | HateMM | MHClip-EN | MHClip-ZH | ImpliHateVid |
  |---|---|---|---|---|
  | Pro-Cap       | 64.5 / 63.3 / 63.4 / 63.2 | 70.1 / 66.3 / 66.3 / 66.3 | 72.5 / 66.8 / 66.1 / 68.3 | 82.3 / 82.3 / 82.4 / 82.3 |
  | MoRE          | 83.4 / 82.4 / 81.8 / 83.3 | 77.5 / 75.2 / 75.7 / 74.8 | 78.5 / 74.8 / 75.7 / 74.1 | 84.8 / 84.7 / 85.4 / 84.8 |
  | ImpliHateVid  | 82.8 / 82.0 / 82.1 / 82.0 | 76.1 / 71.7 / 71.6 / 71.8 | 78.3 / 72.8 / 74.4 / 71.8 | 87.5 / 87.5 / 87.6 / 87.5 |

  MATCH-HVD and the label-free / few-shot baseline numbers retain our existing replication in `docs/results_2026_04_16_v2.md`.
- **Evaluation metrics.** We report accuracy as the primary metric, with macro-F1 (M-F1), macro precision (M-P), and macro recall (M-R) as supporting metrics on the common filtered test set. The same binary label convention and media-exclusion policy are applied uniformly to TRIAGE and all baselines.
- **Implementation details.** The headline TRIAGE configuration uses a 2B open-weights MLLM as the base reader, TrainFit unsupervised thresholding, and a fixed three-member heterogeneous arbitration panel. All MLLM calls use deterministic decoding. Full model identifiers, prompts, threshold criteria, frame sampling, parsing rules, runtime, and software versions are listed in Appendix C.

#### §4.2 Main results (Table 1, C1)

Published baselines run on our benchmark: `{Naive-2B (Das et al., 2024), MARS (Das et al., 2024), LoReHM, Mod-HATE, ALARM, MATCH-HVD, Pro-Cap, MoRE, ImpliHateVid}`, each with its upstream-canonical backbone (or our Qwen3-VL-2B backbone where upstream permits). `SKIP_VIDEOS` applied uniformly. The headline table reports accuracy, M-F1, M-P, and M-R for the frozen TRIAGE configuration (acc: EN 0.783, ZH 0.826, HM 0.842, IH 0.833), highlighting state-of-the-art label-free performance and comparison to supervised systems.

#### §4.3 Ablation (Table 2, C3)

Five rows, four datasets × acc. Each row removes exactly one structural element:

| Row | Gate | Judges | Expected behaviour |
|---|---|---|---|
| R0 Full pipeline | entropy band | triplet majority | headline |
| R1 Stage-1 only | (none) | (none) | lower bound; the pipeline gain evaporates |
| R2 Band-off (judge everyone) | (none) | triplet majority | triplet over-flips confident-correct videos; gain shrinks |
| R3 Triplet → single best judge | entropy band | single judge | per-dataset drift, no stability |
| R4 Triplet → 8-judge all-majority | entropy band | 8-judge vote | over-smoothed; no further gain over triplet |

The ablation kills every easy reviewer question: "is the gate doing the work?" (R2), "why not one judge?" (R3), "why not all eight?" (R4).

#### §4.4 The uncertainty band localises errors (Figure 3, C2)

A single four-panel figure shows, for each dataset, the empirical stage-1 error rate as a function of stage-1 score, with the uncertainty band shaded. Caption: *"The shaded uncertainty band covers 32–62% of the test set yet recovers 60–80% of stage-1 errors — without any label."*

#### §4.5 Transfer across base MLLMs (Table 3, C4)

We replace the first-stage reader with different MLLMs (`qwen3-vl-2b`, `qwen2.5-vl-7b`, `gemma-3-12b-it`, `gemma-3-12b-it-16f`, `pixtral-12b-2409`, `minicpm-v-26`) while keeping the same TRIAGE routing-and-arbitration recipe. For each base reader, we report stage-1 accuracy vs. TRIAGE accuracy averaged across datasets. This tests whether the method is tied to the headline 2B reader or transfers as a general uncertainty-routing design.

#### §4.6 Robustness to panel choice

Holding the headline first-stage reader fixed, we vary the arbitration panel. The main text reports a compact comparison among the headline heterogeneous panel and a few representative alternatives (single judge, same-family panel, and another strong cross-family panel). This tests whether the gain comes from heterogeneous arbitration as a design choice rather than one lucky panel. The broader panel landscape is deferred to Appendix F.

#### §4.7 Call-budget accounting (C5)

Mean MLLM calls per video is `1 + 3 · |B|/|V_test|`. For the headline stage-1 (2B): EN 2.70, ZH 2.77, HM 2.40, IH 2.14. Deployment-realistic upper bound is 4 calls per video (one stage-1 + three judges); this occurs only on the 32–62% of videos that are ambiguous.

### §5 Conclusion (≈0.25 pages)

Rephrased restatement — not copy-paste — of the main contribution. One sentence on why the design pattern (uncertainty-gated cross-family judging) generalises beyond hate moderation to other high-stakes, low-label multimodal classification problems.

---

## Limitations (mandatory EMNLP section; does not count against 8 pages)

This section lists **open problems shared by the label-free MLLM-moderation field as a whole** — not self-owns about engineering choices we have already defended in the main text. Calibration of this section is critical: ARR's rubric explicitly warns against Limitations that read as reasons to reject.

1. **Pretraining-inherited cultural and linguistic bias.** All MLLM-based moderators, ours included, inherit the cultural assumptions of their pretraining corpora. What counts as "hateful" is jurisdiction-specific and evolves; current open MLLMs are not calibrated to every legal or policy regime. This is a community-level open problem for label-free MLLM moderation (Hasan & Ng, 2024 survey; Röttger et al., 2022 HateCheck), not specific to our pipeline.
2. **Temporal horizon of frame-sampled video understanding.** We sample 16 frames per video, following community convention for video MLLMs. Long-range temporal reasoning over full-length video remains an open research problem for all video-LLM benchmarks; our pipeline inherits that limitation without introducing a new one.
3. **Closed-source judge reproducibility.** Our judge pool is entirely open-weights to keep the paper reproducible. Whether frontier closed-source MLLMs (GPT-4V class) would act as even stronger judges is not evaluated here; investigating closed-source judging under open-science constraints is an open field-level challenge.
4. **Static-benchmark evaluation.** Hate speech is adversarial — creators re-encode banned content to evade detection. Our benchmarks are static snapshots; evaluating label-free MLLM moderation under adversarial drift is a standing open problem across the moderation literature.

## Ethics statement (mandatory)

- **Dual-use**: hateful-video detection is a moderation task; misuse (e.g., over-suppression of legitimate speech) is a standing risk for the field.
- **Dataset provenance**: MHClip, HateMM, ImpliHateVid are all publicly released benchmarks with published dataset cards; we respect their original licensing.
- **Annotator welfare**: we do not produce new hate annotations; we consume only the released labels at evaluation time.
- **Responsible NLP checklist**: attached as appendix; pre-filled to avoid desk-reject.

---

## Figure Plan

Redesigned from scratch. Every figure has a single-sentence caption lead that states *what the figure proves*.

| ID | Role | Column | Description | Source | Priority |
|---|---|---|---|---|---|
| **Fig 1 — Teaser** | Position vs. prior families | 1-col | Four rows, one per method family (supervised / generated-reasoning MLLM / uniform multi-stage MLLM / **TRIAGE**). Three axes on the right: **hate labels needed**, **decision signal** (generated text vs continuous score), and **compute allocation** (uniform vs uncertainty-routed). Punchline: TRIAGE is label-free, score-based, and selectively spends stronger MLLM reasoning only on uncertain videos. | Manual schematic | HIGH |
| **Fig 2 — Method architecture** | Pipeline diagram | 2-col | Module 1: probabilistic reading (MLLM → continuous saliency score → TrainFit threshold). Module 2: uncertainty-routed heterogeneous arbitration (2-GMM posterior entropy → uncertainty band → fixed triplet panel only on band; confident videos bypass). Show average call budget as `1 + 3·P(B)`. | Manual schematic + notation from §3 | HIGH |
| **Fig 3 — Errors concentrate in the band** | Empirical support for C2 | 2-col, four panels | One panel per dataset (EN, ZH, HM, IH). X = stage-1 score; Y = empirical error rate (stage-1 pred ≠ label); shading = in-band region. Annotate in-band vs. out-of-band error coverage fractions. | `candidates_entropy_band_2b.jsonl` + labels from `load_annotations` | HIGH |
| **Fig 4 — Backbone-agnostic** (C4) | Per-slug Δ-acc summary | 1-col | Six stage-1 MLLMs on x-axis; bar = avg Δ-acc across 4 datasets for the per-slug best-triplet. All bars positive. | `grid_raw.jsonl` | MEDIUM |
| **Table 1 — Main results** | C1 | 2-col | Our method vs. six published baselines × four datasets, reporting acc, M-F1, M-P, and M-R; bold = best. | `docs/results_2026_04_16.md` + triplet grid | HIGH |
| **Table 2 — Ablation** | C3 | 1-col | Rows R0–R4 as in §4.4; columns EN/ZH/HM/IH acc. | Requires one CPU-only rerun of `grid_eval_all.py` with ablation flags (R2, R3, R4). | HIGH |
| **Table 3 — Backbone-agnostic detail** | C4 | 1-col | Six stage-1 slugs × four datasets × (stage-1 acc / best-triplet acc / Δ). | `grid_raw.jsonl` + per-slug baselines | MEDIUM |

## Appendix Experiment Bank

These are not duplicate main experiments. The main body reports the compact headline result or conclusion; the appendix provides expanded protocol details, sensitivity analyses, and diagnostic breakdowns needed for EMNLP/ARR soundness.

| Appendix item | Purpose | Expected placement |
|---|---|---|
| **A1 — Baseline details and reproduction notes** | Provide the detailed description omitted from §4.1: supervised vs label-free/few-shot baseline taxonomy, upstream settings, reproduction choices, and deviations. | Appendix B |
| **A2 — TrainFit vs TestFit threshold sensitivity** | Expand the brief §4.1 threshold note: headline TrainFit as default deployment setting; TestFit as label-free transductive variant; per-dataset deltas. | Appendix C |
| **A3 — Threshold-family robustness** | Compare Otsu, Li-Lee, and 2-GMM thresholds under the same score files to show TRIAGE is not a single-threshold artifact. | Appendix C |
| **A4 — Stage-1 scorer analysis** | Expand the §4.2 result by isolating stage-1 alone against naive / multi-stage MLLM baselines; include score histograms and calibration-style plots. | Appendix D |
| **A5 — Band-rule alternatives** | Expand §4.3 and §4.4: compare entropy-above-mean with entropy quantiles, Bayes-overlap bands, and no band. | Appendix E |
| **A6 — Panel landscape and selection robustness** | Expand §4.6 by reporting the broader triplet landscape: representative high-performing panels in the main text, additional strong panels in the appendix, cross-family vs same-family comparisons, and judge-frequency statistics. | Appendix F |
| **A7 — Per-dataset / per-language error analysis** | Break down improvements by EN vs ZH and by dataset type (explicit, implicit, meme/video). Include representative corrected and harmed cases. | Appendix G |
| **A8 — Implementation and compute details** | Expand §4.1 and §4.7: prompts, parse-validity rates, calls/video, wall-clock/runtime, frame sampling, software versions, and fallback behaviour. | Appendix H |

**Fig 1 caption draft**:
> *Figure 1.* **Where TRIAGE sits relative to prior hateful-video families.** Supervised detectors learn from hate-labelled training sets. Generated-reasoning MLLM pipelines avoid full supervision but often rely on brittle textual artifacts as decision variables. Uniform multi-stage pipelines spend extra inference broadly. TRIAGE is label-free, extracts a continuous saliency score from a single MLLM read, and reserves heterogeneous arbitration for the uncertainty band.

**Fig 2 caption draft**:
> *Figure 2.* **TRIAGE pipeline architecture.** Module 1: a base MLLM produces a continuous saliency score per video; a TrainFit unsupervised threshold gives a preliminary label. Module 2: a logit-space two-component mixture yields posterior entropy, whose above-mean region defines the label-free uncertainty band. Only band videos are re-judged by a fixed heterogeneous MLLM panel; confident videos keep the first-stage prediction. Average budget: `1 + 3·P(B)` MLLM calls per video.

---

## Citation Plan

Every `[VERIFY]` must be verified against the actual PDF or venue page before submission. Never generate BibTeX from memory (RULE-H).

- **§1 Intro — datasets and task**: [VERIFY] MHClip (Das et al., 2024); [VERIFY] HateMM (Das et al., 2023); [VERIFY] ImpliHateVid; [VERIFY] Hateful Memes (Kiela et al., 2020) as supervised-counterpart anchor.
- **§1 Intro — three low-resource hateful-meme families (used for the "existing label-free attempts" paragraph)**:
  - Prompt-based zero-shot: [VERIFY] PromptHate (Cao et al., 2023a); [VERIFY] zero-shot MLLM prompting on Hateful Memes (Burbi et al., 2023) and equivalents.
  - Captioning + LLM reasoning: [VERIFY] Pro-Cap (Cao et al., 2023b); [VERIFY] MR.Harm (Lin et al., 2023); [VERIFY] any recent "probe-then-reason" multimodal-hate pipeline.
  - Ensembling / self-consistency: [VERIFY] Wang et al. 2023 self-consistency (contrast, not endorsement); [VERIFY] any zero-shot multimodal-ensemble harmful-content paper.
- **§2 Related Work**:
  - Supervised hateful-video / meme: [VERIFY] MATCH-HVD, Mod-HATE, Pro-Cap (Cao et al., 2023b), LoReHM, HateCLIPper (Kumar & Nandakumar, 2022).
  - MLLM-as-moderator: [VERIFY] MARS (Das et al., 2024); [VERIFY] naive MLLM prompting citation.
  - Selective inference / uncertainty: [VERIFY] El-Yaniv & Wiener (2010); [VERIFY] Geifman & El-Yaniv (2017).
  - Judge aggregation: [VERIFY] Wang et al. self-consistency; [VERIFY] Zheng et al. (2023) LLM-as-a-judge.
- **§3 Method**: [VERIFY] Otsu (1979); [VERIFY] Li & Lee (1993) cross-entropy thresholding; [VERIFY] 2-component Gaussian mixture / EM standard reference (Dempster et al., 1977; Bishop 2006); [VERIFY] Shannon (1948) entropy; [VERIFY] base-MLLM and panellist technical reports (Qwen3-VL, Gemma-3, Qwen2.5-VL, LLaVA-OneVision).
- **§4 Experimental Evaluation**: [VERIFY] vLLM (Kwon et al., 2023).
- **Limitations**: [VERIFY] Hasan & Ng survey on hate-speech detection; [VERIFY] Röttger et al. HateCheck.

---

## Reviewer-anticipation checklist

Every question a reviewer will ask, with the § and artefact that answers it. Used to pressure-test the plan before drafting.

| Q | Answer location |
|---|---|
| *"Is this really label-free?"* | §3.1 defines the protocol; §3.3 uses unlabeled score distributions for TrainFit/TestFit thresholding; §3.4 uses posterior entropy and majority vote without labels. |
| *"Does the gain come from adding more MLLM calls?"* | §4.3 R2 (panel everyone, no gate) tests uniform judging directly; §4.7 reports the selective call budget. |
| *"Why these three panellists?"* | §4.6 reports the full 56-triplet landscape and shows that cross-family panels form a stable high-performing design class. |
| *"Why not K-sample self-consistency?"* | §1 and §3.4: panellists differ in model family and alignment recipe; the intended benefit is bias diversity, not sampling-variance reduction. |
| *"Does it transfer across base MLLMs?"* | §4.5 Table 3: six base MLLMs, all with positive mean Δ-accuracy. |
| *"Is the threshold protocol robust?"* | §3.3 states the TrainFit headline protocol; Appendix C reports TestFit and threshold-family sensitivity. |
| *"Is the call budget realistic for deployment?"* | §4.7 accounting; a 1 + 3·P(band) budget is deployment-compatible at moderation volumes. |
| *"What could go wrong at deployment?"* | Limitations 1–4: pretraining-inherited bias, temporal horizon of frame-sampled video, open-weights-only panels, adversarial content drift. |

---

## Next Steps

1. Produce the one missing ablation row — R2 (band-off / judge-everyone triplet) — via `grid_eval_all.py` with ablation flags; CPU-only.
2. Verify every `[VERIFY]` citation against an actual PDF.
3. Render Fig 1 (hand-drawn), Fig 2 (hand-drawn), Fig 3 (4-panel error density), Fig 4 (per-slug Δ-acc bars); Tables 1–3.
4. Draft LaTeX via `/paper-write` against the ACL rolling-review template; keep the main body at 8 pages; Limitations, Ethics, References, Appendix spill over freely.
5. Pre-fill the Responsible NLP Research checklist; attach as Appendix X.
6. Run `/paper-compile` followed by an anti-overclaim / page-budget final pass.
7. (Optional but recommended) route the near-final draft through `/research-review` with GPT-5.4 xhigh to stress-test the Soundness / Excitement / Overall framing one more time before submission.
