# Proposal review (rule 4): candidate 46, topic-direction removal at read time

Date: 2026-10-17

VERDICT: PASS

Reviewer: one independent agent (same model as the main session), before any implementation run or GPU job. Commit
`0b8ca9c` (branch `follow-up`); the proposal directory `experiments/20261017_m1_topic_removal/` is untracked at review time.
Read: `RESEARCH_ITERATION_RULES.md` ("目标" and rule 4; rules 3, 5, 7, 10 for the advisories); the proposal
`experiments/20261017_m1_topic_removal/README.md` (all sections), `topic_bank.py` (every carrier and term) and
`directions.py` (to resolve position / split questions); `research-wiki/DIRECTIONS.md` (all of §A–§D); the first lines of
every `archive/experiments/*/README.md` and `experiments/*/README.md`; `archive/experiments/20261003_m1_reinforcer/README.md`,
`20261003_m1_projector/README.md`, `20261003_m1_contraster/README.md`, `20261003_m1_eraser/README.md` and
`20261004_m1_latents/README.md` (mechanism sections); `archive/detection-2026-08/docs/duplex/READOUT_BOTTLENECK_NOTE.md`
and `ILLOCUTION_TRANSFER_NOTE.md`; `experiments/20260912_tad/README.md` (mechanism lines); `src/mllm_judge.py`
(`yesno_question`, `prefix_messages`). Repository grep: eraser / factorizer / directional ablation / concept erasure /
abliterat / project(ed|ion) out / nullspace / INLP / LEACE / steer / direction over `experiments`, `archive/experiments`,
`research-wiki`, `docs`. This file is the only file written. No hashes computed. No GPU job run.

## Why none of the four STOP cases applies

- **Case 1 (source mechanism already used for hateful video detection / localization).** 27 web searches (25 standard,
  2 extended) and 2 full-text fetches, listed below. No published hateful-video detection or localization method removes
  a concept direction from a frozen model's residual stream at inference, and none intervenes on hidden states at all:
  LELA (multi-stage prompting per frame), MultiHateLoc (weakly supervised MIL), HVGuard (chain-of-thought + trained
  mixture-of-experts fusion), MARS 2601.15115 (multi-stage adversarial reasoning prompts), SafeLens AAAI 2026 (agentic
  pipeline with a parameter-efficiently tuned policy LM), LEAF (distillation), SAGE (mixture of experts), HateClipSeg
  (dataset). The nearest hate-domain representation intervention is FBHM / LSV (arXiv 2605.31349, hateful memes): layer-wise
  steering vectors **learned by gradient** from 500 labelled samples with a cross-entropy term on the hateful / not-hateful
  label plus a KL term, **added** to the hidden state (h' = h + c·α·v) of a frozen VLM at inference; the full text confirms
  no direction is projected out, no target-community concept is removed, no mean-difference between communities is
  computed, and no video or temporal component exists. That is a different operation (addition of a learned, label-supervised
  vector) on a different task (image classification). Refusal-direction work in multimodal models (MARS 2606.31876,
  SARSteer 2510.17633, V-DEAL 2607.21151) adds or ablates a refusal direction to make the model refuse or not refuse; it is a
  safety behaviour method, not a detector or localizer. Training-time target-group debiasing for text hate-speech
  classifiers (INLP in the intersectional-bias study 2109.10441, adversarial demotion 2005.12246, explanation
  regularisation on group identifiers 2005.02439, counterfactual augmentation 2205.04238, hierarchical adversarial
  correction WASSA 2024, NLPGuard 2407.01697) aims at the same confound (group mention vs. hate) but removes it from a
  **trained text classifier** during **training**, with labelled attributes, for a fairness objective. That is not the
  same mechanism as inference-time directional ablation inside a frozen MLLM during a window-level judgment read, and
  none of it is hateful video. Within this repository, no archived candidate removed a concept direction (see
  "Repository overlap"). Case 1 is not met.
- **Case 2 (pure ensemble).** One frozen model; one speech read per window replaces `z_speech`; `z_visual`, the video key
  and the decode are unchanged. The five random-direction reads are controls, not combined into the score. Not an ensemble.
- **Case 3 (pure calibration / post-processing / smoothing).** The change happens inside the forward pass before the
  Yes / No read; it is a new measurement, not a transformation of existing scores. The decode path is the fixed r6_bma.
  Not post-processing.
- **Case 4 (pure engineering trick).** Prompt wording, encoder, resolution and constants are all unchanged. The proposal
  defines a construction procedure (synthetic bank, per-layer mean-difference directions), a held-out validity test (D1),
  an intervention with a stated mechanism hypothesis (the model keeps the judgment but loses the "a protected group is
  mentioned" feature), two empirical gates (D2, D3) and controls. That is a complete method, not a trick.

Rule 4 forbids stopping on pre-implementation reasoning (collapse of the judgment, identifiability, shortcuts); none is
used here. The gates D1–D3 and the test numbers decide.

## Searches performed (2026-10-17; WebSearch is US-only; "std" = standard, "ext" = extended; "fetch" = WebFetch full text)

| # | query | tool / mode | top relevant hits | relevance to the four cases |
|---|---|---|---|---|
| 1 | "hateful video" directional ablation OR "concept erasure" OR "activation steering" OR "representation engineering" | WebSearch std | Representation Surgery https://arxiv.org/pdf/2402.09631 ; CASteer (ICLR 2026) https://iclr.cc/virtual/2026/poster/10011392 ; control-theory steering of text-to-video models https://arxiv.org/abs/2602.19631 | Generation-side steering / erasure (diffusion, T2V). Nothing on hateful video detection. Case 1 not met |
| 2 | hate speech video localization activation steering refusal direction multimodal LLM | WebSearch std | LELA https://arxiv.org/abs/2602.09637 ; MultiHateLoc https://arxiv.org/abs/2512.10408v2 ; MARS (refusal transfer) https://arxiv.org/abs/2606.31876 ; SARSteer https://arxiv.org/pdf/2510.17633 ; SafeSteer https://www.emergentmind.com/topics/safesteer | Localizers use prompting or weak supervision; refusal steering papers make models refuse, not detect. Case 1 not met |
| 3 | concept ablation toxicity detection residual stream direction projection inference-time | WebSearch std | SAE toxicity steering https://arxiv.org/pdf/2505.14536v1.pdf ; harmful intent geometrically recoverable https://www.alphaxiv.org/abs/2604.18901.md ; abliteration explainer https://vauban.readthedocs.io/en/latest/class/part1_what_is_abliteration/ | Toxicity steering of generation; 2604.18901 reports projecting out one harm direction leaves detection under another pooling intact (background for D2, not a STOP ground). Not hateful video |
| 4 | LEACE concept erasure hate speech detection target group identity | WebSearch std | LEACE https://arxiv.org/pdf/2306.03819 ; Yoder et al. target-identity generalisation https://arxiv.org/abs/2210.10839v2 ; knowledge-grounded target recognition https://oro.open.ac.uk/92895 | LEACE evaluated on POS and gender in BERT; no hate-speech or video application found. Case 1 not met |
| 5 | steering vector hateful memes multimodal hate detection arXiv 2605.31349 | WebSearch std | FBHM / LSV https://arxiv.org/pdf/2605.31349 ; Hateful Memes Challenge https://proceedings.neurips.cc/paper/2020/hash/1b84c4cee2b8b3d823b30e2d604b1878-Abstract.html | FBHM is the closest hate-domain steering work; details checked by fetch (rows 16, 22). Memes, not video |
| 6 | topic-controlled hate speech detection target group debiasing representation adversarial removal identity terms | WebSearch std | Schäfer et al. WASSA 2024 https://aclanthology.org/2024.wassa-1.4 ; Kennedy et al. 2020 https://arxiv.org/pdf/2005.02439v2 | Training-time debiasing of text classifiers (adversarial correction, explanation regularisation). Same confound, different mechanism and task |
| 7 | HateMM HateClipSeg MultiHateLoc LELA HVGuard MARS SafeLens hateful video representation intervention inference | WebSearch std | HateClipSeg https://arxiv.org/abs/2508.01712 ; MultiHateLoc https://arxiv.org/html/2512.10408v2 | Dataset and weakly supervised localizer; no representation intervention |
| 8 | "hateful video" OR "hate video" abliteration OR "refusal direction" OR "steering vector" OR "activation patching" detection | WebSearch std | Video unlearning via low-rank refusal vector (ICLR 2026) https://arxiv.org/html/2506.07891v1 ; HVGuard https://preview.aclanthology.org/setup/2025.emnlp-main.456 ; what drives representation steering https://www.alphaxiv.org/abs/2604.08524 | Refusal vector embedded in a video *generator*'s weights to suppress content; HVGuard is prompting + trained fusion. Case 1 not met |
| 9 | iterative nullspace projection INLP hate speech toxicity classifier identity bias removal embeddings | WebSearch std | INLP https://aclanthology.org/2020.acl-main.647.pdf ; intersectional-bias study using INLP on hate speech classification https://www.arxiv.org/pdf/2109.10441 ; https://arxiv.org/pdf/2310.18458 | INLP applied to trained hate-speech text classifiers with labelled demographic attributes, training time, fairness objective. Not the proposal's mechanism; not video |
| 10 | counterfactual data augmentation identity term swap hate speech detection bias | WebSearch std | Sen et al. NAACL 2022 https://aclanthology.org/2022.naacl-main.347 | Training-data augmentation for text classifiers. Different mechanism |
| 11 | arXiv 2601.15115 MARS hateful video detection multimodal | WebSearch std | MARS (training-free multi-stage adversarial reasoning) https://arxiv.org/abs/2601.15115v1 ; https://arxiv.org/html/2601.15115v1 | Prompting pipeline on HateMM / MultiHateClip; no hidden-state intervention. Case 1 not met |
| 12 | SafeLens hateful video detection; HVGuard; LELA hidden states steering | WebSearch std | LELA https://arxiv.org/pdf/2602.09637 ; HVGuard https://underline.io/lecture/130208-hvguard-utilizing-multimodal-large-language-models-for-hateful-video-detection | Confirms LELA is prompting / captioning only; HVGuard CoT + MoE. No steering |
| 13 | "hate speech" OR toxicity classifier "linear erasure" OR "concept erasure" OR "projecting out" protected attribute | WebSearch std | NLPGuard https://arxiv.org/pdf/2407.01697 ; demoting racial bias https://arxiv.org/pdf/2005.12246 ; GetFair target-aware fairness https://publications.hnu.de/4843/ ; procedural fairness https://arxiv.org/abs/2210.10689v1 | Data-level removal, adversarial training; no projection-based erasure in hate classifiers found |
| 14 | refusal direction ablation effect on harm classification probe after ablation | WebSearch std | abliteration audit https://arxiv.org/pdf/2607.01854 ; extended-refusal defence https://arxiv.org/pdf/2505.19056 ; harmfulness propagation https://arxiv.org/pdf/2609.13534 ; harm-keyed routing https://arxiv.org/pdf/2609.14759 | Background on whether a harm classifier survives directional ablation (mixed). Not hateful video |
| 15 | hateful video temporal localization MLLM frozen inference-time intervention residual stream steering ablation erasure target group direction | WebSearch ext | MultiHateLoc https://arxiv.org/html/2512.10408v1 ; LELA https://arxiv.org/html/2602.09637 ; temporal label noise https://arxiv.org/html/2508.04900 ; HAS attention steering for video summarisation https://arxiv.org/pdf/2607.17994 ; Buffalo capstone https://cse.buffalo.edu/tech-reports/2025-01.pdf | Extended search finds no hateful-video residual-stream intervention. Case 1 not met |
| 16 | FBHM abstract page | WebFetch https://arxiv.org/abs/2605.31349 | abstract only: "learnable steering vectors", 500 steering samples, causal intervention objective | Insufficient; full text fetched in row 22 |
| 17 | SafeLens hateful video detection large language model | WebSearch std | SafeLens AAAI 2026 https://ojs.aaai.org/index.php/AAAI/article/view/42390 ; SafeLens video guardrails https://www.alphaxiv.org/abs/2605.17610 | Segment-level agentic pipeline with a tuned policy LM; fast-and-slow guardrail. No representation intervention |
| 18 | "activation steering" OR "steering vectors" improve hate speech OR toxicity classification zero-shot LLM content moderation | WebSearch std | UniSteer https://arxiv.org/pdf/2605.30076 ; SafeSteer https://arxiv.org/pdf/2506.04250 ; steering for synthetic moderation data https://arxiv.org/pdf/2605.28664 ; activation engineering https://arxiv.org/pdf/2308.10248 | Steering used for generation control or data generation; UniSteer classifies by activation reconstruction energy. None erases a topic direction for a judgment read |
| 19 | hateful memes VLM representation intervention inference-time target direction ablation OR erasure | WebSearch std | FBHM https://arxiv.org/pdf/2605.31349 ; counterfactually neutral augmentation https://arxiv.org/html/2508.11808v2 ; qualitative VLM analysis https://arxiv.org/pdf/2608.26143 | Only FBHM intervenes on representations (additive, learned). Memes |
| 20 | arXiv 2606.31876 MARS refusal direction multimodal activation re-centering video | WebSearch std | Harnessing textual refusal directions for multimodal safety https://arxiv.org/pdf/2606.31876 ; V-DEAL https://arxiv.org/pdf/2607.21151 ; cross-architecture audit of direction-based defences https://arxiv.org/pdf/2607.27910 ; ASRU https://arxiv.org/pdf/2605.15687 | Refusal-direction steering for MLLM safety incl. video jailbreak benchmarks; purpose is refusal, not detection / localization |
| 21 | LLM hidden representation identity / target group mention direction hate speech probe spurious "mention" vs "attack" | WebSearch std | Who Speaks Matters https://arxiv.org/html/2410.20490v1 ; implicit hate embeddings https://arxiv.org/abs/2508.20750v1 | Behavioural label-flip studies; no direction-level work found |
| 22 | FBHM full text | WebFetch https://arxiv.org/html/2605.31349 | LSV: layer-wise learned vectors, loss = KL + λ·CE on hateful / not-hateful label at the first decoded token, h' = h + c·α·v, frozen base model, no projection / ablation, no community erasure, no mean-difference between communities, no video | Different operation, labelled, memes. Case 1 not met |
| 23 | concept erasure OR directional ablation OR steering vector used to improve hate speech / toxicity / harmful content detection classification | WebSearch ext | concept-suppression survey https://arxiv.org/html/2505.19398 ; CHaRS https://arxiv.org/pdf/2603.02237 ; Representation Surgery https://arxiv.org/pdf/2402.09631 ; steering vectors as attack surface https://arxiv.org/pdf/2606.05958 | All generation-side; 2402.09631 argues erasing the toxicity concept removes discrimination (background for D2). No detection use found |
| 24 | ROBOSHOT zero-shot robustification projecting out spurious concept direction | WebSearch std | ROBOSHOT (ICLR 2024) https://arxiv.org/pdf/2309.04344 ; SAGE spuriousness-aware prompts https://arxiv.org/html/2511.13005v1 | Closest in form: text-described spurious directions projected out of CLIP embeddings at inference for zero-shot classification. Not video, not hate, not a decoder residual stream |
| 25 | inference-time "project out" bias / nuisance direction LLM hidden states zero-shot classification / LLM-as-a-judge | WebSearch std | FairSteer https://arxiv.org/pdf/2504.14492 ; Two Birds One Projection https://arxiv.org/pdf/2603.14825 ; no-free-lunch bias projection https://arxiv.org/pdf/2511.18635 ; non-contrastive visual attribute steering https://arxiv.org/html/2411.12590v3 | Inference-time bias projection / steering for generation fairness and LVLM safety; none scores content for detection |
| 26 | Wiegand 2019 topic bias abusive language detection biased datasets | WebSearch std | Wiegand et al. NAACL 2019 https://aclanthology.org/N19-1060/ ; Razo & Kübler ALW 2020 https://aclweb.org/anthology/2020.alw-1.9.pdf | Dataset topic bias diagnosis; no representation method |
| 27 | Zou 2023 representation engineering (first attempt; returned only name collisions) | WebSearch std | none relevant | repeated as row 28 |
| 28 | "Representation Engineering: A Top-Down Approach to AI Transparency" fairness bias harmfulness | WebSearch std | RepE https://arxiv.org/abs/2310.01405 | Source for mean-difference reading / control; harmlessness and fairness sections concern model behaviour, not hateful video detection |
| 29 | Arditi "Refusal in Language Models Is Mediated by a Single Direction" directional ablation | WebSearch std | https://arxiv.org/pdf/2406.11717 ; NeurIPS 2024 https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract.html | Source mechanism confirmed: difference-in-means direction, ablated from the residual stream; concept = refusal; task = behaviour control |

## Closest prior work

1. Arditi et al. 2024, NeurIPS (https://arxiv.org/abs/2406.11717): difference-in-means "refusal direction" ablated from the
   residual stream at every layer; the model stops refusing. The proposal reuses the operation and swaps the concept to
   "a protected group is referred to", applied only to the window branch of a judgment read; this is the declared source,
   not prior use on hateful video.
2. Zou et al. 2023, RepE (https://arxiv.org/abs/2310.01405): mean-difference directions from stimulus pairs for reading and
   controlling concepts, including harmlessness and fairness sections. Behaviour reading / control on text LLMs; no
   detection or localization task.
3. FBHM / LSV, arXiv 2605.31349 (https://arxiv.org/html/2605.31349): hateful-meme detection with layer-wise steering
   vectors learned from 500 labelled samples (CE on the hate label + KL), added to a frozen VLM's hidden states with an
   inference-time scale. Additive, label-supervised, images; no projection, no community-concept removal, no video.
4. MARS 2606.31876 (https://arxiv.org/pdf/2606.31876) and SARSteer 2510.17633 (https://arxiv.org/pdf/2510.17633): transfer
   or safe-subspace ablation of a refusal direction in multimodal / audio LMs so that they refuse harmful inputs. Safety
   behaviour, not scoring content; the ablated concept is refusal, not topic.
5. ROBOSHOT, ICLR 2024 (https://arxiv.org/pdf/2309.04344): at inference, project out "harmful" (spurious) concept
   directions obtained from language-model-written descriptions out of CLIP embeddings to improve zero-shot classification.
   Closest in form (text-defined nuisance direction removed at inference, no labels). Operates on a single CLIP embedding
   for image / NLP classification; not a decoder residual stream, not video, not hate.
6. INLP (Ravfogel et al. 2020, https://aclanthology.org/2020.acl-main.647.pdf), LEACE (Belrose et al. 2023,
   https://arxiv.org/abs/2306.03819), and the intersectional-bias study applying INLP to hate-speech classification
   (https://www.arxiv.org/pdf/2109.10441): linear erasure of a labelled protected attribute from a trained classifier's
   representations, for fairness. Same family of linear operation, but training-time, label-supervised, text classifier,
   fairness objective; not inference-time ablation in a frozen MLLM for window scoring.
7. Training-time identity debiasing for text hate classifiers: Kennedy et al. 2020 (https://arxiv.org/pdf/2005.02439v2),
   Xia et al. 2020 (https://arxiv.org/pdf/2005.12246), Sen et al. 2022 (https://aclanthology.org/2022.naacl-main.347),
   Schäfer et al. 2024 (https://aclanthology.org/2024.wassa-1.4), NLPGuard 2024 (https://arxiv.org/pdf/2407.01697). They
   target the same confound (group mention predicts the hate label) through data, regularisers or adversaries while
   training a classifier. Different mechanism and task.
8. Hateful-video methods: LELA (https://arxiv.org/abs/2602.09637), MultiHateLoc (https://arxiv.org/abs/2512.10408),
   HVGuard (https://preview.aclanthology.org/setup/2025.emnlp-main.456), MARS 2601.15115
   (https://arxiv.org/abs/2601.15115v1), SafeLens AAAI 2026 (https://ojs.aaai.org/index.php/AAAI/article/view/42390).
   Prompting pipelines, trained fusion or weak supervision; none intervenes on the model's representations.

Background only (not a STOP ground): Representation Surgery (https://arxiv.org/pdf/2402.09631) argues that erasing the
toxicity concept removes the ability to tell toxic from non-toxic text, and 2604.18901 reports that projecting out one harm
direction can leave harm detection intact under another pooling. The proposal's gate D2 is the empirical test of exactly
this, so the question is left to the run.

## Repository overlap

- **TAD (DIRECTIONS §A2, 09-12; `experiments/20260912_tad/README.md`)** targets the same confound (topic vs. act) but
  subtracts a separate *topic read* from the *act score* outside the model (`r = a − βt`); every β lowered within on both
  corpora. The proposal removes a direction inside the forward pass and makes no second read. Same target, different
  mechanism; not a repetition.
- **PCE (DIRECTIONS §B, 08-08, proposal only)** replaced group names in the transcript with generic words. Input editing,
  never run. Different level.
- **Readout bottleneck (08-09)** looked for unsupervised axes in final-token hidden states that *read* "has a protected
  target"; none did. Reading, not removal; different.
- **Illocution transfer (08-10)** is the closest in *construction*: a researcher-authored carrier-sentence bank with
  minimal pairs, one mean-difference direction per layer, leave-family-out validity (AUROC 1.0), then a transfer test. Its
  concept was speech act (asserted vs. attributed) and its use was reading the direction on video-evidence states, where
  the text-fitted geometry did not transfer. The proposal's concept is group reference and the use is ablation on the
  window-branch tokens; DIRECTIONS §C lists "mention vs. assertion direction" as proposed-only and the 08-10 result as
  DEAD for *reading*. Not the same mechanism; the 08-10 negative result is the nearest empirical warning, and D1–D3 are
  the proposal's answer to it.
- **Reinforcer (candidate 13, VISTA)** *adds* the image-minus-no-image direction with norm preservation to the visual
  branch; **Projector (14, ACG)** adds the component of an attention-output correction that is orthogonal to the masked
  output; **Contraster (7, DoLa)** reads a depth contrast; **Highlighter / Amplifier / Recycler / Preserver / Stabilizer**
  edit attention or value vectors on image tokens; **Latents (19)** optimises continuous visual input vectors; **Eraser
  (5)** deletes media from the input and re-encodes (name only). None removes a concept direction from the residual
  stream, and all act on the visual branch; the proposal acts on the speech branch.
- **Candidate 45** shares only the integration recipe (replace `z_speech`, keep the r6 video key and r6_bma decode).
- No DIRECTIONS row and no archived candidate is the same mechanism. The repository's prior residual-stream work is
  addition or reading; this is the first removal.

## Advisory ambiguities (not grounds for STOP; for the implementer and the code reviewer)

- **What the fit / test split tests.** `directions.py` splits by carrier parity; all 32 group terms occur in both
  splits. D1 therefore measures generalisation across sentence frames, not across group terms, and the README's
  "奇数序号载体句留作检验" could be read as held-out terms. State it. Also note that the 16 protected terms are polite
  group nouns; transcripts carry slurs, nationalities not in the list and pronouns, which D1 does not touch (D3 is the
  only check on them).
- **Direction positions vs. ablation positions.** The direction is the mean residual over the sentence tokens of the
  *question copy* (`frame()` takes the last occurrence of the text, after the one-line transcript) in a prompt whose whole
  transcript is that one sentence; the ablation is applied to every new token of the window branch (window header, window
  text, question, assistant header) after a full-video cached prefix, at all 37 points including layers whose D1 AUC may be
  near .5. Both scripts must hook the same modules (`embed_tokens` and each `lm.layers[i]` output, as in `directions.py`).
  Write the two position sets and the hook points once in §2 so the measurement script and the code review use the same
  definition.
- **Random control.** Five independent Gaussian unit vectors per layer have almost no overlap with the residual stream of
  a 4096-dimensional model, so this control is expected to reproduce r6 whatever the mechanism does, and the second clause
  of D3 then adds little beyond the first. A matched control with the same construction (protected / non-protected labels
  shuffled across the 32 terms, same carriers, same norm statistics) is more informative. Decide before the run which is
  the declared control; both can be recorded.
- **Bank: carriers that change violation status.** The eight attack carriers (indices 12–19; 12, 14, 16, 18 are in the
  fit set) and the joke template 36 ("Three {G} walk into a bar") become policy violations only when the slot is a
  protected group, so the fit-set mean difference contains "this is now a violation" along with "a protected group is
  referred to". State that this is intended, or report as a diagnostic (not a gate) the direction fitted on the 24
  non-attack carriers and its per-layer cosine with the full-bank direction.
- **Bank: terms.** "teenagers" is a weak non-protected control because age is a protected characteristic in common
  platform hate policies (check the rule text in the judge prompt); 5 of 16 protected terms are two-word "X people" phrases
  while all 16 non-protected terms are single words, so the body mean also carries a "people" token difference; "Indians"
  is ambiguous between nationality and Native Americans but both readings are protected. Grammar is correct in all 40
  carriers; the mixed capitalisation "Black people" / "white people" is a spelling convention, not an error.
- **Scale of the raw curve.** `score_curve = max(z_visual, z_speech_abl)` assumes the two logits share a scale; ablation
  can shift the speech baseline, so the raw table may move for reasons unrelated to ordering. The decoded path is safe
  (`nscore` ranks per chain). Say in §2 that the raw max is a diagnostic and the comparison of record is the decoded
  r6_bma path plus the window-level within of D3.
