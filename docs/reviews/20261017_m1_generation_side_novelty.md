# Proposal review (rule 4): candidate 45, generation-side reading (verbatim-copy resistance and minimal compliant rewrite)

Date: 2026-10-17

VERDICT: PASS

Reviewer: one independent agent (same model as the main session), before any implementation or GPU run. Commit
`c4769ea` (branch `follow-up`; the proposal directory is untracked at review time). Read: `RESEARCH_ITERATION_RULES.md`
("目标" and rule 4, plus rules 3, 5, 7, 10 for the advisories); the proposal `experiments/20261017_m1_generation_side/README.md`
(all five sections); `research-wiki/DIRECTIONS.md` (all of §A–§D); the first non-empty line of every
`archive/experiments/*/README.md` and `experiments/*/README.md`; `experiments/20260926_glr/README.md` §3–§4 and §8;
`archive/experiments/20261005_m1_rote/README.md` lines 1–40 (name check only; it is a RoPE time-encoding candidate, unrelated);
`experiments/20261008_baselines/README.md` lines 33–143 and `experiments/20261008_baselines/lavad/README.md` lines 24–41,
207–239 (how refusals were handled in the baselines); the two previous novelty records
`docs/reviews/20261013_m1_pairwise_evidence_novelty.md` and `docs/reviews/20261014_m1_audio_bound_novelty.md`.
Repository check: a case-insensitive grep for refus / teacher forc / verbatim / 复述 / 改写 / rewrite / detox / 去毒 /
edit distance / Levenshtein / 编辑距离 / copy loss / copy task over `experiments/*/README.md`, `archive/experiments/*/README.md`,
`research-wiki/`, `docs/` (result summarised under "Repository overlap"). This file is the only file written by this review.
No hashes were computed. No GPU job was run.

## Why none of the four STOP cases applies

- **Case 1 (source mechanism already used for hateful video detection / localization).** Eighteen web searches and five
  abstracts opened (table below). No published hateful-video detection or localization method scores content by refusal
  probability, by the teacher-forced likelihood of reproducing the content, or by how much a model changes it when asked
  to rewrite it. The LLM-based hateful-video methods found (LELA, HVGuard, MARS, SafeLens, TANDEM, LEAF) all ask a judgment
  question or train a classifier. The nearest hate-domain use of refusal is a hateful-memes study that assigns the label
  "hateful" with fixed confidence .9 whenever GPT-4o mini refuses; that is a label-filling heuristic on images, not a
  probability read, not video, and not localization. Refusal-probability detectors exist for text prompts (MULI, Prefix
  Probing) and detoxification-by-rewriting exists for text (ParaDetox family, MaRCo), which is exactly the "transferred
  from another task" situation that rule 4 and the "目标" section count as novel if it works.
- **Case 2 (pure ensemble).** One frozen model; one of the two read-outs replaces `z_speech`, the other is only reported;
  the README states they are not combined. `z_visual` is unchanged. Not an ensemble.
- **Case 3 (pure calibration / post-processing / smoothing).** The proposal changes what is measured per window (a new
  forward pass or generation per window), not how existing scores are transformed. The per-corpus affine `s_aff` is for
  the raw-curve table only; the decoded path uses the existing `nscore` rank-normalisation of r6_bma.
- **Case 4 (pure engineering trick).** This is not a wording change to the same Yes / No question: arm A reads the
  per-token negative log-probability of the window's own words under a copy instruction (teacher forcing), arm B reads a
  model-external word-level edit count between the model's rewrite and the original. Both are different measured
  quantities with a stated mechanism (alignment-induced reluctance to produce attacking words, bound to the words
  themselves), stated target errors (speech-driven false highs on set-up / reported / mention-only windows), and declared
  controls (lexicon counts, word count, random, Spearman against `z_speech`). No encoder, resolution or constant-only
  change is involved.

Rule 4 forbids stopping on pre-implementation reasoning; none is used here. The gates G1–G3 and the test results decide.

## Searches performed (2026-10-17; WebSearch is US-only; "std" = standard mode, "ext" = extended mode)

| # | query | tool / mode | top relevant hits | relevance to the four cases |
|---|---|---|---|---|
| 1 | hateful video localization refusal probability safety alignment signal | WebSearch std | LELA https://arxiv.org/abs/2602.09637 ; V-DEAL https://arxiv.org/abs/2607.21151 ; Video-SafetyBench https://arxiv.org/abs/2505.11842 ; VMDT https://arxiv.org/abs/2511.05682 | LELA scores frames by multi-stage prompting; the other three measure refusal as a per-item evaluation outcome of video LLMs, not as a localization score. Case 1 not met |
| 2 | "hateful video" detection teacher forcing repeat verbatim likelihood LLM | WebSearch std | HVGuard (EMNLP 2025) https://preview.aclanthology.org/setup/2025.emnlp-main.456 ; LEAF https://aclanthology.org/2026.findings-acl.604.pdf ; LELA | Chain-of-thought judgment + trained fusion; distillation ("teacher" = teacher model). No teacher-forced copy likelihood. Case 1 not met |
| 3 | refusal probability as classifier moderation signal LLM "refusal rate" toxicity detection zero-shot | WebSearch std | MULI "Toxicity Detection for Free" (NeurIPS 2024) https://arxiv.org/abs/2405.18822 | Text prompts; refusal probability approximated by first-response-token logits; a sparse logistic regression is fitted on labelled examples. Same family as the proposal's report-only `first_loss` / `refuse_mass`; not video, not localization, not label-free |
| 4 | LLM detoxification edit distance as toxicity score minimal edit rewrite hate speech detection | WebSearch std | MaRCo https://arxiv.org/abs/2212.10543 ; AraDetox https://arxiv.org/abs/2608.22894 ; MICo https://aclanthology.org/2024.findings-naacl.110 ; SoCD https://arxiv.org/abs/2604.19124 | Minimal-edit rewriting as a generation goal; word-level edit rate reported only as a diagnostic of rewrites. Nobody uses the edit amount as a detection score. Case 1 not met |
| 5 | "copy" task language model reluctance to reproduce toxic text verbatim probability harmful content detection | WebSearch std | Schick et al. self-diagnosis https://arxiv.org/abs/2103.00453 ; Xu et al. AAAI 2022 https://ojs.aaai.org/index.php/AAAI/article/view/21406 ; Proximal Decoding (ICML 2026) https://icml.cc/virtual/2026/poster/65462 | Self-diagnosis is a judgment prompt about the model's own output; verbatim-reproduction work is about copyright. No "reluctance to copy" detector found |
| 6 | "repeat after me" OR "repeat the following" verbatim harmful text refusal benchmark over-refusal LLM copy request | WebSearch std | OR-Bench https://arxiv.org/abs/2405.20947 ; DDOR https://www.alphaxiv.org/abs/2606.03601 ; XSB / MS-XSB https://arxiv.org/html/2603.27518 | Over-refusal benchmarks; no copy-task likelihood read, no video |
| 7 | perplexity OR likelihood under safety-aligned LLM as harmful content detector toxic text log-probability | WebSearch std | Prefix Probing https://arxiv.org/abs/2512.16650 ; clean-data curation https://arxiv.org/abs/2405.19358 ; ASTPrompter https://arxiv.org/abs/2407.09447 (cites the Alon & Kamfonas perplexity filter) | Prefix Probing compares log-probabilities of comply- vs refuse-style probe prefixes (text, black-box); perplexity filters flag adversarial prompts. Background for arm A; none is hateful video |
| 8 | rewrite-then-compare toxicity detection LLM rewrite difference detector "edit rate" OR "edit distance" harmful span localization | WebSearch std | span-guided detox comparison https://arxiv.org/abs/2607.26795 ; YOPO prompt learning (toxic span F1) ; SoCD | No rewrite-and-diff detector found; span detection there is a direct tagging prompt |
| 9 | hateful video detoxification rewriting transcript counter-speech MLLM HateMM MultiHateClip | WebSearch std | ParaDeHate https://arxiv.org/abs/2506.01484 ; MultiHateClip https://arxiv.org/abs/2408.03468 ; MM-HSD https://arxiv.org/abs/2508.20546 ; LELA | Text detoxification datasets; hateful-video detectors trained on features. No rewriting of hateful-video transcripts for detection. Case 1 not met |
| 10 | LAVAD OR T3AL OR "Vad-R1" OR EventVAD OR MultiHateLoc OR "MACIL-SD" OR DSANet refusal rewriting repetition likelihood anomaly score | WebSearch std | LAVAD https://arxiv.org/abs/2404.01014 ; EventVAD https://arxiv.org/abs/2504.13092 ; Vad-R1 https://arxiv.org/abs/2505.19877 | Caption + LLM scoring, event segmentation, RL reasoning. T3AL / MultiHateLoc / MACIL-SD / DSANet not returned; from this repository's reruns (`experiments/20261008_baselines/README.md`) they are CLIP self-adaptation and weakly supervised classifiers. None generation-side |
| 11 | log-likelihood ratio aligned chat model vs base model toxicity detection harmful text zero-shot | WebSearch std | "Blink of an eye" https://arxiv.org/abs/2502.00921 ; LAPD alignment imprint (AI-text detection) https://www.themoonlight.io/en/review/alignment-imprint-zero-shot-ai-generated-text-detection-via-provable-preference-discrepancy ; MULI | Aligned-vs-unaligned likelihood ratio tested on jailbreak prompts and on AI-text detection; not hate, not video |
| 12 | LLM refuses to transcribe OR translate OR quote offensive slurs verbatim study | WebSearch std | CISPA study https://arxiv.org/abs/2603.11914 ; weakly aligned summarization (ACL 2024) https://aclanthology.org/2024.acl-long.461 ; Selective Safety Trap https://arxiv.org/abs/2601.04389 | Behavioural studies: models sometimes refuse benign tasks (translation) on harmful material. The premise of arm A exists as an observation; no detector was built from it |
| 13 | toxicity measured as distance to detoxified paraphrase OR "minimal counterfactual" edit hate speech classifier-free score | WebSearch std | CF-Detox https://arxiv.org/abs/2405.09948 ; CausalDetox / PARATOX https://www.alphaxiv.org/abs/2604.14602 ; SRD self-reflective detox https://arxiv.org/abs/2601.11776 ; MaRCo | Counterfactual rewriting evaluated with a toxicity classifier as oracle; no classifier-free edit-distance score |
| 14 | video MLLM first-token refusal logit "Sorry" probability harmful video segment scoring unsafe content detection | WebSearch std | first-token refusal steering (a different "MARS", 2606.31876) https://papers.cool/arxiv/2606.31876 ; SafeWatch https://arxiv.org/html/2412.06878v1 ; video-driven prompting (CVPR 2026) https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Breaking_Multimodal_LLM_Safety_via_Video-Driven_Prompting_CVPR_2026_paper.html | Refusal-token steering / decoding and trained guardrails; nobody reads refusal probability per segment as a score |
| 15 | hateful video temporal localization using refusal or repetition likelihood or rewriting of transcript as frame-level hate score, training-free MLLM | WebSearch ext | LELA html https://arxiv.org/html/2602.09637 ; UG-Ground https://arxiv.org/pdf/2510.00705 ; Yes/No scanning grounding https://arxiv.org/pdf/2608.08315 | Only LELA is on-task; it is prompting for scores. Case 1 not met |
| 16 | hateful memes detection zero-shot refusal probability OR refusal rate VLM safety alignment as hatefulness signal | WebSearch std | GPT-4o mini hateful memes https://arxiv.org/abs/2509.13608 ; ARCADE intent shifts https://arxiv.org/abs/2603.21298 ; zero-shot VLM memes https://arxiv.org/abs/2402.12198 | Closest hate-domain use: refusals labelled "hateful" at fixed confidence .9 (a fill-in heuristic, images, no probability, no localization); ARCADE reports refusals cluster on explicit hate and on counter-speech. Not video, not a scoring method |
| 17 | video LLM per-segment refusal rate used to localize unsafe or harmful segment timestamps | WebSearch ext | Failures to Surface Harmful Contents (AAAI 2025) https://arxiv.org/abs/2508.10974 ; V-DEAL ; From Evaluation to Defense https://arxiv.org/html/2505.16643v2 | Harm-omission rate and defence success rate are per item; no per-segment refusal signal for localization |
| 18 | teacher-forced negative log-likelihood of harmful text under safety-aligned model higher than base model | WebSearch std | Emulated Disalignment https://arxiv.org/abs/2402.12343 ; Qi et al. shallow alignment https://arxiv.org/abs/2406.05946 ; debiasing https://arxiv.org/abs/2502.02153 | Aligned-minus-base log-likelihood on harmful responses used as a proxy safety reward; the gap concentrates in the first tokens. Background that the quantity arm A reads is known to move with alignment; not a detector, not hate video |

Abstracts opened with WebFetch: LELA 2602.09637 (prompting scheme over five modalities; no generation-side read),
MULI 2405.18822 (first-response-token logits + sparse logistic regression, prompts only), Prefix Probing 2512.16650
(conditional log-probabilities of probe prefixes, text), MARS hateful video 2601.15115 (training-free pro / contra
reasoning merged into one video-level decision; no refusal / rewrite / likelihood read stated), CISPA study 2603.11914
(behavioural, nine benign tasks, no detector).

## Closest prior work and the exact difference

1. **MULI, "Toxicity Detection for Free" (NeurIPS 2024, 2405.18822).** Detects toxic *prompts* from the aligned LLM's
   first-response-token logits (refusal-start tokens such as "Sorry"), with a sparse logistic regression fitted on
   labelled prompts. Difference: the proposal's primary scores are the mean teacher-forced loss over the window's own words
   under a copy instruction (A) and a model-external edit count (B); the first-token quantities `first_loss` / `refuse_mass`
   are report-only. MULI classifies the user's request, needs labels, has no window, no video, no localization.
2. **Prefix Probing (2512.16650).** Black-box harmful-content detection by comparing conditional log-probabilities of
   comply-style vs refuse-style probe prefixes, text only. Same instrument family (log-probabilities under the aligned
   model) but it scores a prompt's harmfulness through fixed probe prefixes, not the likelihood of reproducing given content.
3. **Emulated Disalignment (2402.12343) and shallow safety alignment (Qi et al. 2406.05946).** Establish that an aligned
   model's log-likelihood of harmful continuations drops relative to the base model and that the drop concentrates in the
   first tokens. The proposal relies on the first fact and reads the quantity per window; neither paper builds a detector
   or touches hateful video.
4. **GPT-4o mini hateful memes (2509.13608) and ARCADE (2603.21298).** The nearest hate-domain contact with refusal: one
   assigns "hateful" with confidence .9 to every refusal to keep its tables complete; the other reports that GPT-5's
   refusals concentrate on explicitly hateful and on counter-speech samples. Neither reads a probability, neither is video,
   neither localizes. They are evidence that refusal is a confounded proxy, which the proposal's G1–G3 gates will test.
5. **ParaDetox family / ParaDeHate (2506.01484), MaRCo (2212.10543), CF-Detox (2405.09948), AraDetox (2608.22894).**
   Minimal-edit detoxification of text; edit rate appears as a diagnostic of how much rewrites change. The proposal turns
   the edit amount into the per-window score and applies it to video transcripts cut on the 8 s grid. The rewrite side of
   the proposal is a transfer from this task, which rule 4 counts as novel if it works.
6. **CISPA study (2603.11914) and weakly aligned summarization (ACL 2024, 2024.acl-long.461).** Document that aligned
   models sometimes refuse or hesitate on benign tasks (translation) over harmful user-supplied material. This is the
   observation behind arm A, stated as a safety gap, never used as a measurement.
7. **LELA (2602.09637), HVGuard (EMNLP 2025), MARS (2601.15115), SafeLens (AAAI 2026), TANDEM (2601.11178).** The LLM-based
   hateful-video detectors / localizers. All obtain scores from judgment prompts, chain-of-thought rationales, or trained
   heads; none reads generation-side behaviour. LELA remains the strongest published training-free comparator for the task.
8. **LAVAD (2404.01014), EventVAD (2504.13092), Vad-R1 (2505.19877), T3AL, MultiHateLoc, MACIL-SD, DSANet.** Caption + LLM
   scoring, event segmentation, RL reasoning, CLIP self-adaptation, weakly supervised classifiers. In this repository's
   LAVAD rerun refusals are masked as missing values, not used as a signal (`experiments/20261008_baselines/lavad/README.md`
   lines 33–34, 196, 207–239).

## Repository overlap

- **GLR, `experiments/20260926_glr` (DIRECTIONS §A2, 09-26).** Nearest in instrument: Qwen3-VL-8B, fp32 lm_head,
  teacher-forced log-probability of the window's words, seed 0, same prefix and grid. Different in the measured quantity:
  GLR scored the words' predictability under "violating" vs "compliant" speaker conditions with the words *not* in the
  context, and failed because speech is unpredictable (about .01 nats / token difference). Candidate 45 puts the words in
  the context so the copy baseline is near zero and reads the residual loss as reluctance. This is a distinct hypothesis
  (willingness to say, not ability to predict), not a rerun of GLR. The README states this difference correctly.
- **Second question / answer contrast (DIRECTIONS §B, 08-09 / 08-10) and the judgment-question family (TAD five-way,
  BND, HVL state chain, DVD).** All ask the model to judge; the proposal asks it to produce. G1 (Spearman < .80 against
  `z_speech`) is the declared test that the new read is not the same quantity renamed.
- **HVL citation matching and candidate 41 (frame references), candidate 42 (pairwise).** Model names where the evidence
  is; model compares windows. Neither reads reproduction likelihood or edit amount.
- **Candidate 43 (audio-bound) and the in-window frames result (10-10).** Input changes under the same judgment read;
  unrelated mechanism.
- **Teacher forcing elsewhere in the repository:** `experiments/20261006_m1_texttiling` uses teacher-forced Whisper
  decoding for word timestamps; `experiments/20261002_m1_iteration` mentions character teacher forcing for the same
  purpose. Timestamping, not scoring.
- **Refusal in the repository:** only as missing data in the LAVAD and URF-HVAA baselines (masked / NaN windows). No
  experiment or baseline here has used refusal probability, copy loss or rewrite distance as a hate score. The grep over
  experiments, archive, research-wiki and docs found no other hit for 复述 / 改写 / detox / edit distance / Levenshtein
  as a mechanism (the Levenshtein hit in `archive/experiments/20261004_m1_lattice` is an alignment anchor).
- **`archive/experiments/20261005_m1_rote`** is "RoTE" (interval rotary time encoding), not rote repetition; unrelated.

## Advisory ambiguities (not part of the verdict; the author may pin these down before running)

- The refusal-start list `{"I","I'm","I’m","Sorry","As","Unfortunately","Apolog"}` contains the bare pronoun "I". Many
  transcript windows begin with "I" (and arm B's rule says an output beginning with a listed starter scores `edit_rate` =
  1.0), so an unchanged, correct rewrite of a window that starts with "I think…" would be scored as a refusal. The same
  token is also the natural first token of such windows in arm A's `refuse_mass`. State whether "I" alone counts, or
  whether the test is on a longer refusal phrase, and apply the same rule in both arms.
- Arm A's forced assistant turn: specify the exact token sequence that is scored, i.e. whether `{body}` follows the chat
  template's assistant header directly (no thinking block for the Instruct model), whether a leading space / newline is
  part of the first scored token, and whether the end-of-turn token is included in `copy_mean`. Short windows make the
  mean sensitive to these choices, and `first_loss` depends entirely on them.
- The body appears twice in arm A (in the question and as the forced answer) and once in arm B. State the normalisation
  applied before both uses (whitespace collapse, casing, the `fill_untimed` slice) so the copy target is byte-identical to
  the shown text, and whether there is a token cap on the body in the prompt or in the forced answer (arm B has a
  `max_new_tokens` rule; arm A has none written).
- `edit_rate` word tokenisation: define "lowercase, strip punctuation" precisely (apostrophes, hyphens, numbers), and
  whether a rewrite that appends commentary after the speech is scored as-is or trimmed. Two implementers would differ here.
- G1–G3 are to be met "on both corpora"; say explicitly that the Spearman and each window-level within are computed per
  corpus, over the same set of windows (windows with text, positive ≥ .5 / negative = 0), for all arms including the
  lexicon, word-count and random controls.
- If both arms pass, the choice "higher mean window-level within over the two corpora" is a test-selected decision; mark
  the chosen arm as development-selected (rule 10) in §5 and STATUS, and state whether the mean over corpora is unweighted.
