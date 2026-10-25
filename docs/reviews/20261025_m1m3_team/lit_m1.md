# M1 literature scout: how a frozen (M)LLM produces moment-level evidence in context (2026-10-25)

Scope: mechanisms that could redesign how M1 reads a window given the video-level act. M1 today is a frozen Qwen3-VL-8B. It reads a shared cached prefix (rules + 20 frames + full timestamped transcript), gives a video verdict and a stance turn, and then asks isolated visual and speech Yes/No branches per 8 s window. The window score is the Yes/No log-odds.

## How things were checked

- **arXiv IDs.** Every arXiv ID below was looked up in the arXiv API this session, and the returned title matched. Where no arXiv ID is given, I say so. Helper scripts are in `scratchpad/m1m3_novelty/lit_tools/`.
- **Venues.** Each venue comes from one of three places: the arXiv comment field, the proceedings or Anthology page found this session, or a search result that names the venue page. "Venue unverified" means none of these confirmed it.
- **Mechanism text.** For key works I read the method section: TG-Heads, ICR, Superposition Prompting, CEAVAD, Kang et al. localisation heads, and Probe-VAD (abstract only). For the rest, the description comes from the abstract plus prior knowledge, and the entry says so where it matters.
- **Tags.**
  - **[CITED]**: already cited in `research-wiki/paradigm_framing.md`.
  - **[KNOWN]**: appears in earlier repo or scratchpad notes (M1_report, M3 scout, experiment READMEs), but not in paradigm_framing.
- **"Closest repo attempt"** names the nearest thing already tried. Sources: `research-wiki/DIRECTIONS.md`, `archive/experiments/2026100*_m1_*`, `archive/experiments/20260912_cva`, `archive/detection-2026-08/docs/duplex/`, `experiments/20261002_m1_iteration/README.md`. The 36+ archived M1 directories cover far more than the charter's list. The October 2-6 round alone tried about 30 reader mechanisms, including attention steering, contrastive decoding, attribution, KV retrieval, trees, programs and tools.

---

## A. Training-free / zero-shot video anomaly detection with VLMs and LLMs

**A1. LAVAD**: Harnessing LLMs for Training-free Video Anomaly Detection. CVPR 2024. arXiv 2404.01014. **[CITED]**
- Mechanism: a captioner writes frame captions. An LLM summarises a temporal window of captions and scores that summary. Scores are then refined by kNN over video-text similarity.
- Named failure: "frame-level caption lacks details about the global context".
- M1 mapping: the precedent for "give the local read some context". Its context is a local temporal window, not the whole video.
- Hate use: none.
- Distinct from repo? No. It is the precedent M1 already follows.

**A2. AnomalyRuler**: Follow the Rules: Reasoning for VAD with LLMs. ECCV 2024. arXiv 2407.10299. **[CITED]**
- Mechanism: an LLM induces normality rules from a few normal frames. It then reasons per frame against those rules, and the decisions are smoothed (Exponential Majority Smoothing).
- M1 mapping: M1's policy rules play the same role.
- Hate use: none.
- Distinct? No.

**A3. VERA**: Explainable VAD via Verbalized Learning of VLMs. CVPR 2025. arXiv 2412.01095. **[CITED]**
- Mechanism: guiding questions are learned in words, not weights. The frozen VLM answers them per segment. Scene-context ensembling then adds a context-weighted score.
- Named failure: "only examine a short moment ... without considering any context".
- Distinct? No.

**A4. VADTree**: Explainable Training-Free VAD via Hierarchical Granularity-Aware Tree. NeurIPS 2025. arXiv 2510.22693. **[CITED]**
- Mechanism: a tree is built by recursively splitting the video at confidence-ranked boundaries. LAVAD-style scoring is applied per node, and coarse and fine nodes are fused.
- Closest repo attempt: candidate 21 (semantic cluster tree, `archive/experiments/20261004_m1_tree`), archived.
- Distinct? No.

**A5. EventVAD**: Training-Free Event-Aware Video Anomaly Detection. ACM MM 2025. arXiv 2504.13092. **[KNOWN]** (M3 scout)
- Mechanism: a dynamic spatiotemporal graph with time decay finds event boundaries without supervision (signal-ratio thresholding). An MLLM then reasons per event under a hierarchical prompt.
- Named failure: training-free methods "face challenges in localizing fine-grained visual transitions and diverse events".
- M1 mapping: makes the reading unit an event instead of a fixed window.
- Closest repo attempts: ASR segments lost to the fixed 8 s grid (+.089 / +.052 for the grid); TextTiling (candidate 38) and TIL were also tried. Low novelty for M1.

**A6. PANDA**: Towards Generalist VAD via Agentic AI Engineer. NeurIPS 2025. arXiv 2509.26386.
- Mechanism: an agent loop with scene-aware strategy planning, goal-driven reasoning, tool-augmented self-reflection and a chain-of-memory (mechanism from the abstract).
- M1 mapping: weak. It costs many calls per video, which conflicts with the CLAUDE.md cost rule.
- Closest repo attempts: candidates 18 (Explorer) and 35 (VTimeCoT), tool and acquisition loops, both archived.

**A7. CEAVAD**: Beyond Hazard Resemblance: Contrastive Event Adjudication for Training-Free VAD. arXiv 2608.09908 (2026-08). Venue: none stated. **[CITED by name only, in the "do not claim" line, with no ID]**
- Mechanism (method section read):
  - It holds a fixed library of 8 hazard mechanisms. Each has a hazard statement and a mechanism-specific benign counterpart.
  - A frozen CLIP stage gives a signed margin: the hazard bank minus the larger of the generic-normal bank and the specific-benign bank.
  - Frozen Qwen3-VL-8B then reads each 2 s target interval with up to two past and two future intervals and all 8 hypothesis rows. It compares the next-token logits of "abnormal" and "normal".
  - No whole-video context is used.
- Named failure: "hazard resemblance". Richer anomaly descriptions capture what looks like danger without deciding whether the event is abnormal.
- M1 mapping: this is the VAD version of the repo's D5 law: questions measure the topic (a group is mentioned), not the act (an attack). It contrasts each moment against a benign account of the same mechanism, rather than against "normal" in general. In hate, the counterpart would share the target and topic but carry a different act: quoting, reporting, condemning, or the victim's reply.
- Hate use: none.
- Closest repo attempts: TAD five-way speech-act choice (attack / report / quote / condemn / irrelevant). It collapsed: 80% / 56% of windows answered "attack". The DVD "endorsement" sub-question also had no effect. MARS / RAMF use competing hate / non-hate reasoning at video level.
- Distinct? Partly. The untried part is a counterpart that is specific to the video's own act and matched to its topic, read per window as a two-hypothesis contrast. The tried versions used generic options.

**A8. Probe-VAD**: Ordinal Likelihood Probing for Training-Free VAD. arXiv 2609.17211. Under review. **[CITED by name only]**
- Mechanism (abstract): ten ordered severity thresholds, each asked as a constrained Yes/No. The normalised preferences form a cumulative severity profile; isotonic projection makes it monotone, and tail evidence becomes the score.
- M1 mapping: an ordinal readout of the window instead of one binary question.
- Closest repo attempt: candidate 47 used an ordinal read at video level (collapsed to the verdict). Not tried per window, but this is only a readout change.

**A9. GtS (Glance then Scrutinize)**: VAGU & GtS. arXiv 2507.21507. The authors' 2026 journal extension (arXiv 2608.11260) says the conference version is AAAI 2026; I did not open the AAAI page. **[KNOWN]** (M1_report saw the 2608.11260 abstract)
- Mechanism: a coarse global glance proposes suspicious intervals, then a dense local scrutiny of those intervals. The extension makes one MLLM run the global-to-local loop with a video-crop tool, trained with RL.
- M1 mapping: another "global then local" precedent. The global-then-local space is getting crowded.
- Distinct? No.

**A10. Others, for completeness.**
- **Vad-R1**: NeurIPS 2025, arXiv 2505.19877. Trained with SFT + RL. Not label-free.
- **AnyAnomaly**: arXiv 2503.04504, WACV 2026. Not on the venue list. Customisable text-defined anomaly with an LVLM.
- **2610.16324**, Anomaly-guided observation with two-level semantic feedback (arXiv): a video-level description sharpens or suppresses local scores. **[KNOWN]**
- **2608.08315**, "Your VLM Already Knows When: Training-Free Temporal Grounding by Asking Yes or No" (arXiv): per-clip Yes/No with no whole-video context. **[KNOWN]** It is another precedent for M1's readout.

---

## B. Training-free temporal grounding, moment retrieval, action localization, highlights

**B1. TFVTG**: Training-free VTG using Large-scale Pre-trained Models. ECCV 2024. arXiv 2408.16219. **[CITED]**
- Mechanism: an LLM decomposes the query into ordered sub-events. A VLM scores frames per sub-event with static and dynamic transition scores, then proposals are assembled by enumerating combinations.
- Named failures: boundaries of multiple events; dynamic transitions.
- M1 mapping: decomposing the act into ordered stages.
- Closest repo attempts: BND begin/end questions; M2's 3-phase "topic / attack" layers; DVD T/E decomposition. All failed.

**B2. Zero-shot STVG**: Unleashing the Potential of MLLMs for Zero-Shot Spatio-Temporal Video Grounding. NeurIPS 2025. arXiv 2509.15178. **[CITED]**
- Mechanism:
  - The MLLM assigns "grounding tokens" whose attention localises the target.
  - DSTH splits the query into attribute and action sub-queries.
  - LRA learns latent spatial and temporal prompts at test time by regularising each sub-query's token predictions.
  - TAS keeps results consistent under temporal augmentation.
- M1 mapping: an attention-based readout plus test-time latent prompts.
- Closest repo attempts: Latents (candidate 19, continuous state optimisation) failed. Attention readouts are covered in C1.

**B3. SeViLA**: Self-Chained Image-Language Model for Video Localization and QA. NeurIPS 2023. arXiv 2305.06988. **[CITED]**
- Mechanism: the same frozen BLIP-2 is prompted twice. As a localiser it gives per-frame "is this frame relevant" Yes probabilities; as an answerer it answers the question.
- Distinct? No. It is the precedent for "same model, verdict then window reads".

**B4. T3AL**: Test-Time Zero-Shot TAL. CVPR 2024. arXiv 2404.05426. **[CITED]**
- Mechanism: a video-level pseudo-label, then per-frame scores, a video-mean threshold and test-time self-supervised refinement.
- Distinct? No.

**B5. Moment-GPT**: Zero-shot Video Moment Retrieval via Off-the-shelf MLLMs. AAAI 2025. arXiv 2501.07972.
- Mechanism: LLaMA-3 rewrites the query to reduce language bias. MiniGPT-v2 proposes spans, and Video-ChatGPT with a span scorer picks among them.
- Named failure: language bias in queries.
- M1 mapping: weak. M1's question is fixed.

**B6. NumPro**: Number it: Temporal Grounding Videos like Flipping Manga. CVPR 2025. arXiv 2411.10332.
- Mechanism: frame numbers are drawn onto the frames so the Video-LLM can name "when" by reading numbers. It works training-free or fine-tuned. Its analysis aggregates attention from query tokens to each frame.
- M1 mapping: an input change (rule 5), not a mechanism.

**B7. CASCADE**: Decompose and Conquer: Compositional Reasoning for Zero-Shot TAL. AAAI 2026, DOI 10.1609/aaai.v40i11.37900. arXiv ID not found.
- Mechanism: an MLLM filters and captions actions. An LLM splits each into ordered stages, the MLLM scores frames per stage, and hierarchical merging rebuilds instances.
- M1 mapping: same family as B1, already failing in the repo.

**B8. Search and agent readers.**
- **VideoAgent** (Wang et al.): ECCV 2024, arXiv 2403.10517. An LLM agent iteratively retrieves frames until its self-rated confidence is high.
- **T\***: CVPR 2025, arXiv 2504.02259. Temporal search treated as spatial search over a frame grid.
- **VideoTree**: CVPR 2025, arXiv 2405.19209. Query-adaptive cluster tree. **[KNOWN]**, read in October.
- Closest repo attempts: Explorer, Tree, Spatial search (V*), VTimeCoT, Temporal Search (read), ReKV / MuKV / StreamingTOM. All archived or no qualifying gain. Not distinct.

**B9. ZEAL**: Zero-shot Action Localization via the Confidence of LVLMs. arXiv 2410.14340. Venue unverified.
- Mechanism: an LLM writes start and end descriptions of the action. The LVLM's per-frame confidence for each description gives the boundaries.
- Closest repo attempt: BND (begin/end questions) failed.

**B10. TF-SELECTOR** (in SVHighlights). KDD 2026 Datasets & Benchmarks track. arXiv 2606.06926.
- Mechanism: an LLM assigns a saliency score to each shot-merged segment from captions, transcript and loudness.
- Training-free highlight detection is thin at top venues; I found no MLLM-internal method there.
- M1 mapping: none beyond what M1 already does.

---

## C. Reading internal states of a frozen model

### C1. Attention as the score (prefill attention, re-ranking, localisation heads)

**C1a. ICR (In-Context Re-ranking)**: Attention in LLMs Yields Efficient Zero-Shot Re-Rankers. ICLR 2025. arXiv 2410.02642.
- Mechanism (method read):
  - All candidate documents and the query sit in one prompt.
  - A document token's score is the attention from the query tokens, summed over all layers and heads and averaged over query tokens.
  - The same quantity is computed with a content-free query ("N/A") and subtracted, following "Calibrate before use". Abnormally low tokens are dropped, and the rest are summed per document.
  - Two forward passes rank N documents. No generation, no labels.
- Named failure: attention is biased by position (start and end of the input) and by token type (titles, entities, punctuation), whatever the query.
- M1 mapping: windows are the documents and the verdict question is the query. Every window's evidence would be read from the attention the act judgment pays to that window's frames and words, minus a content-free act query.
  - Cost: one extra forward pass per video, instead of 2 branches per window.
- Hate use: none.
- Closest repo attempts:
  - August 2026-08-12 pilot (`PREREG_temporal_attribution_pilot.md`): raw last-position attention and grad×input of the verdict. The curve equalled token density (HCS segment AUC .525 / .482 vs a density control .518).
  - Selector (10-02): restricted the heads, within −.042 / −.015.
  - Attributor (10-02): signed cached-value attribution, within −.130 / −.070.
  - CVA (09-12): exclusion attribution.
- Distinct? Partly. The family is three times failed. The specific untried piece is the content-free-query calibration, which answers exactly the named failure of the August pilot (attention equals token density or position). The rest of the family is closed.

**C1b. MLLMs Know When Before Speaking (TG-Heads)**: Revealing and Recovering Temporal Grounding via Attention Cues. arXiv 2605.21954 (v2 2026-09-27). Venue unverified. **[KNOWN]** (used as the prior in the August pilot; cited in the candidate 42 novelty review)
- Mechanism (method read):
  - A sparse set of "Temporal Grounding Heads" focuses query-to-frame attention on the right interval during prefill; the signal is lost when the answer is decoded.
  - Per-frame relevance comes from entropy-weighted query tokens.
  - It is debiased by a positive log-ratio against a blank (all-zero) video pass, which removes the attention sink.
  - The model is then re-run on the cropped interval.
  - Backbones include Qwen3-VL-8B.
- Caveat: heads are chosen with a knockout score on 500 labelled examples, so the head choice is not label-free.
- Named failure: "perception-generation gap". The model knows when during prefill but drifts toward salient segments while answering.
- M1 mapping: same as C1a, with a blank-media control instead of a content-free text query. It also gives direct evidence that, on this backbone, prefill attention carries "when".
- Distinct? Same family as C1a. Label-free head choice would be needed.

**C1c. Localization heads**: Your LVLM Only Needs A Few Attention Heads for Visual Grounding. CVPR 2025. arXiv 2503.06287.
- Mechanism (method read): heads are kept if their attention to image tokens is high, ranked by the spatial entropy of their binarised maps, and picked by how often they are among the lowest-entropy heads. The criteria use no GT boxes, though the samples come from RefCOCO. The chosen heads' text-to-image attention gives the box.
- M1 mapping: a label-free head-selection recipe that could replace TG-Heads' labelled selection.
- Closest repo attempt: Selector (window-selective heads). It restricted attention rather than reading it.

**C1d. Retrieval Head Mechanistically Explains Long-Context Factuality.** ICLR 2025 (oral). arXiv 2404.15574.
- Mechanism: under 5% of heads copy from the context. Masking them breaks retrieval.
- M1 mapping: background for which heads to read.
- Distinct? Background only.

**C1e. Lookback Lens**: Detecting and Mitigating Contextual Hallucinations Using Only Attention Maps. EMNLP 2024. arXiv 2407.07071.
- Mechanism: per head, the "lookback ratio" is the attention on context divided by attention on context plus generated tokens. A linear classifier (trained with labels) on these ratios flags hallucinated spans.
- M1 mapping: a label-free per-branch quantity at the answer position: how much the window's Yes / No draws on that window's own tokens versus the rest of the prefix (the act). It is a decomposition of the existing read, not a new read.
- Closest repo attempts: Grounder (late access restriction) and Selector are interventions on the same attention.
- Distinct? Distinct as a readout. The family is near-closed.

**C1f. PASTA**: Post-hoc Attention Steering for LLMs. ICLR 2024. arXiv 2311.02262. **C1g. Found in the Middle**: Calibrating Positional Attention Bias. ACL Findings 2024. arXiv 2406.16008.
- Mechanism: steering (PASTA) or position-debiasing (Found in the Middle) of attention.
- Closest repo attempts: the October intervention round (PAI, VAR, VISTA, ACG, VGA, MAD-RAG, PAS, FutureMask) applied all of these. No qualifying gain. Closed family.

### C2. Context attribution by ablation, and context-reliance contrast

**C2a. ContextCite**: Attributing Model Generation to Context. NeurIPS 2024. arXiv 2409.00729.
- Mechanism: random subsets of context sources are ablated, and a sparse linear surrogate is fit to the response log-probability. Its weights attribute the response to sources.
- M1 mapping: windows as sources; the attribution of the verdict per window.
- Closest repo attempts: CVA (leave-one-out exclusion; the sentence's own effect was about 6× the interval effect), Eraser (actual media deletion, within −.131 / −.131), Allocator (Kernel SHAP; stopped because SHAP is already used in hateful-video detection).
- Distinct? No. Closed.

**C2b. PECoRe**: Quantifying the Plausibility of Context Reliance. ICLR 2024. arXiv 2310.01188.
- Mechanism: contrastive metrics with and without context find context-sensitive output tokens. Contrastive gradient attribution then finds which context cues drove them.
- Closest repo attempt: 09-12 "context contrast", z(full) − z(window only), within .467 / .494.
- Distinct? Closed.

### C3. Contrastive decoding (input or layer ablated, then contrasted)

- **VCD**: CVPR 2024, arXiv 2311.16922. Contrasts against a distorted image.
- **Context-aware decoding (CAD)**: NAACL 2024 (short), arXiv 2305.14739. Contrasts with and without the context.
- **DoLa**: ICLR 2024, arXiv 2309.03883. Contrasts a late layer against an early layer.
- **Contrastive Region Guidance**: arXiv 2403.02325, venue unverified. Contrasts with and without a region prompt.
- **Video Contrastive Decoding** (Qi et al.): NeurIPS 2025 (poster 120105), arXiv ID not found. Corrupts temporal consistency of key frames.
- **EventHallusion's TCD**: arXiv 2409.16597, venue unverified.
- **SEASON**: CVPR 2026 (virtual poster page), arXiv 2512.04643. Per-token self-diagnosis, then adaptive temporal and spatial negatives.

Common mechanism: the output under the real input minus the output under a corrupted input, at decoding time.

Repo status: closed.
- visual_contrast R1 / R2: VCD-style. The R1 gain failed promotion; the R2 local corruption lost it.
- Contraster: DoLa-style. All branches picked layer 2; +.0001 / +.0002.
- 09-12 context contrast and 08-10 answer contrast: dead.

None is used for hate.

### C4. Patching hidden states / task and function vectors (a representation of the act moved into another pass)

**C4a. Patchscopes**: A Unifying Framework for Inspecting Hidden Representations. ICML 2024. arXiv 2401.06102.
- Mechanism: the hidden state from position i, layer l of a source pass is patched into a target "inspection" prompt. The model then decodes in words what that state encodes. Logit lens and similar tools are special cases.

**C4b. In-Context Learning Creates Task Vectors.** Findings of EMNLP 2023. arXiv 2310.15916.
- Mechanism: in-context demonstrations compress into one vector at the separator position of a middle layer. Patching it into a query-only pass reproduces in-context learning.

**C4c. Function Vectors in LLMs.** ICLR 2024. arXiv 2310.15213.
- Mechanism: a few heads carry a compact task representation, found by causal mediation. Adding their sum to the residual stream of a zero-shot pass triggers the task.

**C4d. In-context Vectors (ICV).** ICML 2024 (PMLR v235). arXiv 2311.06668.
- Mechanism: a latent vector built from a forward pass over demonstrations shifts the latent states of a new query, in place of putting demonstrations in the prompt.

Mapping and status for the C4 group:
- M1 mapping: carry the act into the window read as an internal state, not as 3k+ prefix tokens the branch attends to.
  - Take the hidden state at the verdict and stance positions of the video's own prefix pass: the "act state".
  - Patch it into a window read that sees only that window's local frames and aligned words.
  - Or patch each window's local state into one "is this part of the act" inspection prompt.
- Named failure it could answer: inside the full prefix, the branch read is driven by whole-video content, and window order follows topic cues (D5). Changes on the reader side barely move that order: candidate 46 vs r6 Spearman .98; with vs without the stance turn .93 / .92 (M1_report §1). A read without context loses the act (single-window read −.021 / −.071).
- Hate use: none found.
- Closest repo attempts:
  - Reinforcer (VISTA residual steering on visual tokens), Factorizer, Integrator: all reshape attention or the residual stream inside the same full-context pass.
  - Candidate 46 removed a topic direction built from synthetic text.
  - August "text-concept direction" family (RepE) was excluded because directions elicited from text did not transfer to video.
- Distinct? Plausibly yes. No repo attempt moved the video's own act state into another pass. The text-to-video transfer problem does not arise, because the vector comes from the same video's forward pass.

### C5. Binding and reference inside the model ("who is targeted")

**C5a. How do Language Models Bind Entities in Context?** ICLR 2024. arXiv 2310.17191.
- Mechanism: causal interventions show that LMs tag each entity and its attributes with "binding ID" vectors. A query is answered by finding the attribute that shares the queried entity's binding ID. The vectors form a subspace and transfer across tasks.
- M1 mapping: the Figure 1 case has 37 hateful windows, of which only 12 name the group. This is the first-class failure: the target is named earlier, and later windows refer to it.
  - A window read could test whether the window's referents (pronouns, "you", "they") carry the binding of the act's target, as found in the prefix.
  - This is a model-internal version of "who is targeted".
- Hate use: none.
- Closest repo attempts: Quote graph (candidate 24) and Provenance graph (candidate 25), both prompt-level reference structures; candidate 44, speaker-bound reading by diarisation. None read the model's internal binding.
- Distinct? Plausibly yes.
- Caveat: binding IDs were shown on synthetic text with Pythia / LLaMA. Transfer to noisy ASR inside Qwen3-VL is untested.

### C6. Label-free probing of hidden states

**C6a. CCS**: Discovering Latent Knowledge in LMs Without Supervision. ICLR 2023. arXiv 2212.03827.
- Mechanism: finds a direction where p(statement) ≈ 1 − p(negation), using consistency and confidence losses, with no labels.

**C6b. ICM**: Unsupervised Elicitation of Language Models. arXiv 2506.10139. Venue unverified (an Anthropic-hosted PDF exists; no proceedings page found).
- Mechanism: searches for a labelling of the whole corpus that maximises mutual predictability (each label predicted in context from the others) plus logical consistency. Then fine-tunes on it.

Repo status for C6:
- `archive/detection-2026-08/docs/duplex/LABELFREE_PHILOSOPHY_MAP.md` judged CCS "mostly dead": unsupervised structure finds the most prominent axis (topic). It listed ICM as untried.
- ICM needs many in-context calls per label update, so it conflicts with the cost rule. It also ends in fine-tuning, which the repo's no-label self-training history (SDL / NGA) argues against.
- Low priority.

**C6c. LLMs Know More Than They Show.** ICLR 2025. arXiv 2410.02707.
- Mechanism: truthfulness signal concentrates at the exact answer tokens. The probes are supervised.
- M1 mapping: M1 already reads at the answer token. Background.

### C7. Forked shared-prefix structures (method precedents for M1's current form)

**C7a. Superposition Prompting**: Improving and Accelerating RAG. ICML 2024 (per Apple's ICML 2024 listing; PMLR page not opened). arXiv 2404.06910.
- Mechanism (method read):
  - A shared preamble's KV cache is reused.
  - Each retrieved document forks into its own path, with the query duplicated onto each path (the "ForkJoin" topology); paths do not attend to each other.
  - Each path is scored by a Bayesian path saliency built from length-normalised LM likelihoods. Low-saliency paths are pruned.
- Not cited in paradigm_framing, which names only the systems precedents Hydragen and Prompt Cache. This is a closer method-level precedent for M1's "shared prefix + isolated per-unit branches + per-branch score", and it should be cited.
- Its readout is a likelihood of the document under the query, close to GLR. GLR failed: spoken words are too unpredictable (.01 nats/token difference).

**C7b. Parallel Context Windows.** ACL 2023. arXiv 2212.10947.
- Mechanism: context chunks are encoded in parallel with shared position IDs and no cross-attention; task tokens attend to all chunks.
- A precedent for isolation. The repo's branch mutual-visibility test (09-10) was mixed.

---

## D. Context-dependent judgment in text with LLMs (implicit hate, use vs mention, pragmatics)

**D1. NLP Systems That Can't Tell Use from Mention Censor Counterspeech, but Teaching the Distinction Helps.** NAACL 2024. arXiv 2404.01651. **[KNOWN]** (August brainstorm bundle)
- Mechanism: shows that LLMs confuse using a slur with mentioning (quoting) it, which censors counter-speech. Prompts that teach the use / mention distinction reduce the errors.
- M1 mapping: endorsement versus quotation inside the act.
- Closest repo attempts: TAD five-way (report / quote / condemn options collapsed to "attack"); DVD E; GLR; candidate 45 (the model copies slurs without hesitation). Prompt-level versions are closed.

**D2. Don't Go To Extremes**: Revealing the Excessive Sensitivity and Calibration Limitations of LLMs in Implicit Hate Speech Detection. ACL 2024. arXiv 2402.11406.
- Finding: "LLMs display excessive sensitivity towards groups or topics ... misclassifying benign statements as hate speech". Confidence also concentrates in a fixed range regardless of the dataset.
- M1 mapping: not a mechanism. It is the published name for the repo's D5 law, which helps the story: per-window reads measure "the group is mentioned". Any M1 redesign that targets this can cite it as the named failure.

**D3. Pragmatic Inference Chain (PIC).** EMNLP 2025 (main; ACL Anthology 2025.emnlp-main.296 per search). arXiv 2503.01539.
- Mechanism: a prompting chain modelled on pragmatic inference steps for implicit toxic language. It beats direct and chain-of-thought prompting on GPT-4o and Llama-3.1-70B.
- Closest repo attempts: HVL (a generated hypothesis in context gave confirmation bias: 83% of windows "Yes") and the stance turn.
- Distinct? Low.

**D4. HARE**: Explainable Hate Speech Detection with Step-by-Step Reasoning. Findings of EMNLP 2023. arXiv 2311.00321.
- Mechanism: LLM-generated rationales fill gaps in human rationales and are used to fine-tune a detector. Not label-free.

**D5. The Goldilocks of Pragmatic Understanding**: Implicature Resolution by LLMs. NeurIPS 2023 (spotlight). arXiv 2210.14986.
- Finding: base and instruction-tuned LLMs often fail binary implicature; tuning on example-level instructions helps.
- Background for "a moment means what it means because of the conversation". Not a mechanism.

**D6. Think Twice (SimToM)**: Perspective-Taking Improves LLMs' Theory-of-Mind. ACL 2024. arXiv 2311.10227.
- Mechanism: two stages. First the context is filtered to what a given character knows; then the question is answered from that perspective.
- M1 mapping: read window t from "what has been established by time t". The context becomes past-only and is chosen by the model.
- Closest repo attempts: Integrator (future-aware memory, the opposite direction); causal visibility between branches (09-10, DeHate −.031). Weak.

Pre-2023 context works (Yu et al. NAACL 2022; CAD NAACL 2021; Latent Hatred EMNLP 2021; HateCheck ACL 2021) are already **[CITED]** in §5. I found no 2023-2026 top-venue LLM method that models conversation-thread context for hate. The nearest are a NAACL 2025 demo (SafeSpeech) and a WOAH-at-EMNLP workshop paper (arXiv 2510.15685).

---

## E. Self-consistency, verification, debate, query decomposition

- **Self-Consistency.** ICLR 2023. arXiv 2203.11171. Majority over sampled reasoning paths.
  - Repo: rule 3 bans ensembles. Answer distributions also saturate (August: 82% of posteriors have entropy ≈ 0).
- **Chain-of-Verification.** Findings of ACL 2024 (Anthology 2024.findings-acl.212). arXiv 2309.11495. **[KNOWN]**, in the candidate 27 sources.
  - Mechanism: plan verification questions, answer them independently, revise.
  - Closest repo attempts: candidate 27 (factored reobservation) and the HVL review loop, both archived. The repeated review only repeated the verdict (Spearman .76–.84).
- **Multiagent Debate.** ICML 2024 (per author repo and Paper Digest; PMLR page not opened). arXiv 2305.14325. Several LLM instances argue over rounds.
  - Hate precedents are in F. The repo stopped Marginalizer because MARS / RAMF already use opposing hate / non-hate reasoning.
- **Decomposition**: TFVTG (B1), Zero-shot STVG (B2), CASCADE (B7). Closed in the repo, as noted there.

---

## F. Hateful meme / hateful video detection with frozen models (what hate work already uses)

- **Pro-Cap.** ACM MM 2023. arXiv 2308.08088. **[CITED]** A frozen BLIP-2 answers hand-written probing questions; the answers are captions for a classifier.
- **Explainable Harmful Meme Detection through Multimodal Debate between LLMs (ExplainHM).** WWW 2024. arXiv 2401.13298. LLMs argue harmless versus harmful explanations; a fine-tuned small LM judges. Uses labels.
- **Towards Low-Resource Harmful Meme Detection with LMM Agents.** EMNLP 2024. arXiv 2411.05383. Retrieves related memes with labels ("relative sample augmentation") and uses knowledge revision. Few-shot labelled.
- **MARS**: Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning. ICASSP 2026 (not on the venue list). arXiv 2601.15115. **[KNOWN]** Neutral description, then competing hate and non-hate reasoning, then synthesis. Video level only.
- **RAMF**: Reasoning-Aware Multimodal Fusion for Hateful Video Detection. TMLR. arXiv 2512.02743. **[KNOWN]** Local-global fusion of generated reasoning with modality features. Trained.
- **HVGuard.** EMNLP 2025 (Anthology 2025.emnlp-main.456; no arXiv found). **[KNOWN]** MLLM chain-of-thought plus a trained Mixture-of-Experts fusion. Detection only.
- **LEAF.** Findings of ACL 2026 (Anthology 2026.findings-acl.604; arXiv not checked). **[KNOWN]** Self-grounding chain-of-thought from a large MLLM, distilled into a small model. Detection only.
- **SafeLens**: Segment-Level Hate Speech Detection in Online Videos. AAAI 2026 demo track (OJS 42390; arXiv not checked). Segment-level labels with a parameter-efficiently tuned policy LLM; cites HateClipSeg. Trained.
- **LELA**: Towards Training-free Multimodal Hate Localisation with LLMs. arXiv 2602.09637, no venue. **[CITED]** Per-frame five-modality prompting with composition matching, then per-modality max and a threshold. The only training-free hate localiser found again this session.
- **MultiHateLoc.** WWW 2026. arXiv 2512.10408. **[KNOWN]**; CLAUDE.md lists it as a weakly supervised comparator. Uses video-level labels.
- **IARE** (arXiv 2606.11953) and **CLARA** (arXiv 2608.15905): trained with rationales. **[KNOWN]**

What hate work already uses with frozen models: probing questions, chain-of-thought rationales, competing hate / non-hate reasoning or debate, retrieval of labelled neighbours, per-frame prompting. None reads internal states (attention, patched states, binding) for hate, and none localises with them. This search found no hate work that uses attention re-ranking, task or function vectors, Patchscopes or binding IDs.

---

## Most transferable to M1 (top 5)

Feasibility is for the designer. Each item states the M1 redesign, the failure it answers, why it is plausibly new here, and the nearest closed family.

1. **Act-state transfer: the window is read under the video's own act representation** (task vectors, Findings EMNLP 2023; function vectors, ICLR 2024; ICV, ICML 2024; Patchscopes, ICML 2024).
   - Redesign: the act enters the window read as the hidden state of the verdict and stance positions, not through attention over the whole prefix.
   - Read A: patch the act state into a window-local pass.
   - Read B: patch each window's local state into one act-inspection prompt.
   - Failure answered: a full-context read is pulled by whole-video content and topic; a context-free read loses the act.
   - Why new: no repo attempt and no hate work moves a video's own internal state between passes. The vector comes from the same video, so the August text-to-video transfer failure does not apply.
   - Nearest closed family: steering and direction removal (Reinforcer, candidate 46).

2. **Calibrated attention read of the act judgment** (ICR, ICLR 2025; TG-Heads, arXiv 2605.21954, same backbone; label-free head criteria from Kang et al., CVPR 2025).
   - Redesign: window evidence = attention from the verdict question to the window's tokens, minus the same under a content-free act query or a blank-media pass.
   - Cost: about one extra forward pass per video instead of 2 branches per window.
   - Failure answered: the August pilot's named failure, where raw attention equalled token density and position. That is exactly what ICR's calibration removes.
   - Why it fits the story: moments are read from the act decision itself.
   - Risk: the attribution family has failed four times (08-12 pilot, CVA, Attributor, Allocator stop); only the calibration step is new.

3. **Two-hypothesis window adjudication with a counterpart specific to the video's own act** (CEAVAD, arXiv 2608.09908, with Qwen3-VL-8B; use vs mention, NAACL 2024; "excessive sensitivity", ACL 2024, as the named failure).
   - Redesign: per window, contrast "this moment carries the video's act" against a counterpart with the same topic and target but a different act (quote, report, condemn, reply). Both hypotheses are written once per video from the verdict turn.
   - Failure answered: the repo's D5 law, which the literature names "excessive sensitivity to groups".
   - Difference from closed attempts: TAD and DVD used generic options; MARS / RAMF contrast at video level only.
   - Risk: HVL-style confirmation bias and TAD-style collapse to "attack".

4. **Internal reference binding for "who is targeted"** (binding IDs, ICLR 2024).
   - Redesign: a window counts as part of the act if its referents carry the binding of the act's target, as found in the prefix pass.
   - Failure answered: the Figure 1 first-class case, where the target is named once and later windows refer to it (only 12 of 37 hateful windows name the group).
   - Why new: nothing in the repo or in hate work reads binding inside the model. Quote and provenance graphs were prompt-level.
   - Risk: highest. Binding is unshown on noisy ASR or in VLMs.

5. **Lookback-ratio decomposition of the existing branch read** (Lookback Lens, EMNLP 2024; used label-free, without its classifier).
   - Redesign: at each branch's answer token, split the read into "drawn from this window's tokens" versus "drawn from the act context".
   - Failure answered: a speech-driven false high is a branch answering from the act prior without local support.
   - Cost: zero added calls.
   - Weakest of the five on novelty: it sits next to Grounder and Selector (attention interventions) and may count as an analysis rather than a mechanism.

**Must-cite precedents found in passing (novelty risk for the current M1):**
- Superposition Prompting (ICML 2024) and Parallel Context Windows (ACL 2023) are method-level precedents for "shared prefix + isolated per-unit branches + per-branch score". paradigm_framing.md currently cites only the systems precedents Hydragen and Prompt Cache.
- GtS (AAAI 2026), 2608.08315, Probe-VAD and CEAVAD further crowd "global then local" and "per-segment Yes/No" in 2025-2026.
