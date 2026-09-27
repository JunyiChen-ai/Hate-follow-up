# Final novelty review of the label-free localization method (r3_m2 / r4_bma), 2026-09-27

Reviewer: an independent agent, under `RESEARCH_ITERATION_RULES.md` rule 4 (novelty) and rule 14g (a component
claimed as novel must cost ≥ .01 on one main metric on both main corpora when removed). This file is the only file the
review wrote. No hashes were computed.

About the numbers:
- HateMM and HateClipSeg (HCS) numbers are development-selected (rule 10), because the designs were chosen while
  reading test results.
- DeHate numbers are external, and nothing was chosen on DeHate.
- Every number was re-read from the evaluator output named next to it.
- Metric order is always pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC ("within").

Literature: about 60 papers from 2023 to September 2026, plus the older statistics sources the modules rely on, were
checked in four groups:
- hateful video;
- training-free video anomaly detection (VAD), temporal action localization and temporal grounding;
- temporal statistical models (hidden semi-Markov models, duration models, model averaging);
- score transformation, calibration, prefix sharing and verdict conditioning.

A link is given only for a paper that was actually opened. "Abstract only" and "not verified" are marked. Some quotes
passed through a summarizing fetch tool; every quote must be checked against the PDF before it goes into a paper
(§8, item 8).

---

## 0. Verdict in brief

- **Pipeline.** The pipeline is new as a whole for this task. No hateful-video paper, and no training-free
  localization paper, combines:
  - window reads made against a shared whole-video context that includes the model's own verdict;
  - an explicit-duration latent-state model fitted without labels;
  - a two-level composition.
- **Parts.** Every part has close prior work. The paper can claim the task-specific design and its evidence. It cannot
  claim new statistical or systems machinery.
- **Claims that meet rule 14g on both corpora:**
  - M1 as a whole: reading windows against the shared context, versus reading each window alone;
  - M3 time coupling;
  - M3 minimum segment length of two reading windows (in the r4 model, where lengths are free);
  - the stance turn, at its point estimate only.
- **Components that fail rule 14g:** M2 normal scores, M3 averaging over lengths, M4 calibrated key, dual branches,
  transcript context, isolation mask. They stay in the method as cited design choices, not contributions.
- **Main risks at review:**
  1. **M3 against simpler persistence.**
     - On the main model it performs like the earlier hand-set persistence chain.
     - It is within .007 of the best Gaussian smoother on HCS.
     - On DeHate it adds only +.013 within over no time model.
  2. **Same-backbone comparator (rule 14f).**
     - No same-backbone training-free comparator has been run.
     - The label-free comparators so far are T3AL and ZS-ImageBind, both on CLIP-class encoders.
  3. **Transductive fitting.** EM, the rank transform and the key calibration are fitted on the unlabeled test corpus.
  4. **Round-4 gate.** It is not yet settled (§1).

## 1. What was reviewed

The method has four modules, M1–M4.

**M1, reads (SPVL-r2, `experiments/20260910_spvl/README.md`).** A frozen Qwen3-VL-8B reads one shared prefix:
policy rules, 20 timestamped frames and the full timestamped Whisper transcript. From it the model gives:
- a whole-video Yes/No verdict `z_video`;
- then, with its own verdict appended as an assistant turn (the stance turn), two isolated Yes/No branches for every
  fixed 8 s window, one visual and one speech.

All branches are decoded from the cached prefix and cannot see each other. Cost: one encoding of the video, then
short branches. That is about 1.4–1.5 s per video on one RTX 5090.

**M2, reads to evidence (`experiments/20260926_twolevel/README.md` §14).**
- Each modality's window reads become normal scores of their rank in the corpus.
- Two Gaussian emissions (hate / non-hate) are fitted by EM without labels.

**M3, time level (§10, §14, §16).**
- One explicit-duration two-state chain per modality, with negative-binomial segment lengths.
- OR fusion: a moment is hateful if either chain is in the hate state.
- In round 4 (`r4_bma`):
  - the shape is k = 2 windows × 8 s / 4 s cells = 4;
  - the mean hate length and the mean gap length are averaged per video over a 6×6 log grid on
    [two windows, video length], with posteriors weighted by likelihood.
- In round 3 (`r3_m2`) the shape is the same k = 4, and both mean lengths are fixed at 80 s.

**M4, composition (§11).**
- Video key K = z_video + mean window read, calibrated to logit P(V=1|K) = aK + b. The calibration is a label-free
  two-component 1-D Gaussian mixture with shared variance.
- Frame score = key + centred within-video rank of the time-level posterior.
- Intervals come from P(V=1|K) × P(hate at t | V=1) ≥ .5.

Results (`runs/20260926_twolevel/{r4_bma,r3_m2}/metrics.json`; DeHate `runs/20260927_dehate_external/summary/table.txt`):

| | HateMM | HCS | DeHate (external) |
|---|---|---|---|
| r4_bma | .8970 / .6943 / .7542 | .7169 / .6714 / .6384 | not yet run |
| r3_m2 | .8971 / .6953 / .7525 | .7168 / .6705 / .6397 | .7009 / .1578 / .6539 |
| strongest weakly supervised on DeHate | — | — | Fed-WSVAD .7007 / .1752 / .5055; best within MultiHateLoc .5420 |

**Status of round 4 at review time.** Sources: `runs/20260926_twolevel/analysis_r4/table.txt`,
`runs/20260926_twolevel/robust/<model>_{r3,r4}/metrics.json`.
- The no-drop check on the main corpora passes: every difference to r3_m2 is ≤ .002.
- The coupling ablation passes.
- The 8-MLLM rule (r4 not below r3 in within by more than .01 on ≥ 7 of 8 models per corpus) is still running; 6 of
  8 models are done.
- On HateMM, Qwen3-VL-2B is at −.0101 and Qwen3-VL-4B at −.0156.
- Unless −.0101 is read as at the floor, HateMM cannot reach 7/8. Round 4 would then fail, and r3_m2 stays.
- The verdicts below cover both variants. Claims that exist only for r4 are marked "r4 only".

**Correction to the brief.** "r3_m2 vs c_m2" compares normal scores against EM on the raw reads (the round-2 time
level). It does not compare against corpus-std scaling; the corpus-std arm is `current` (TIL).

**Files read.**
- `CLAUDE.md` and `RESEARCH_ITERATION_RULES.md`.
- `research-wiki/STATUS.md` and `research-wiki/DIRECTIONS.md`.
- `experiments/20260926_twolevel/README.md` (all sections) and `experiments/20260910_spvl/README.md`.
- `experiments/20260922_til/README.md` §8–§10, `experiments/20260927_dvd/README.md`,
  `experiments/20260927_dehate_external/README.md` and `experiments/20260927_error_analysis/README.md`.
- `docs/reviews/20260922_til_proposal_review.md` and `docs/protocol_1fps_legacy/LOCALIZATION_LANDSCAPE.md`.
- `archive/detection-2026-08/docs/duplex/PAPER_OUTLINE_v1.md`: the earlier concession of the mask primitive to
  SingGuard, InvariRank and T3S.
- The evaluator outputs listed in §4.

---

## 2. Verdict per module

| module | closest prior work | may be claimed | must be cited | must not be claimed | rule 14g |
|---|---|---|---|---|---|
| **M1** reads | LELA (per-frame, per-modality LLM reads, max fusion); ESOM (policy prefix cached for every window, Qwen3-VL-8B); T3AL / FreeZAD (localize under the model's own video-level prediction); SingGuard (isolated rule branches on a shared content prefix); Hydragen, Prompt Cache, PCW, APE; ReKV / STTM (video KV cache reused across questions) | every fixed window is read against one cached whole-video context that includes the model's own verdict, in isolated per-modality Yes/No branches; none found for hateful video or for training-free video localization | prefix sharing, isolation masks, position restart, Yes/No token read-out, class-conditional localization, per-modality reads with either-modality fusion | isolation or KV reuse as new; dual branches, transcript context or isolation as gains; "the stance turn makes windows more discriminative" | package: **yes** on all three metrics; stance: point threshold only; dual, transcript, isolation: **no** |
| **M2** reads → evidence | rank INT (McCaw et al. 2020); nonparanormal (Liu et al. 2009); Efron's two-groups model (2004); Sun &amp; Cai 2009; Wang &amp; Wang 2024 | no novelty; only "the time model uses only the order of each modality's reads, so it does not depend on each MLLM's logit scale" (cross-model count) | all of the left | rank or copula emissions as new; invariance to monotone rescaling as a new property | **no**: removing normal scores raises within by .005 / .003 on the main run |
| **M3** time level | Wang &amp; Wang 2024 (label-free HSMM with shifted negative-binomial durations over normal-scale statistics); Sun &amp; Cai 2009; Hayashi et al. 2017 (one chained-state HMM per event over network scores); NN-Viterbi, Richard 2016/2017; Yu 2010; RJaCGH (averaging HMM posteriors over models); BOCPD, Wilson 2010, Fox 2008, Johnson &amp; Willsky 2013; Kang et al. 2025 (HMM over zero-shot GPT-4o labels, fitted on labels); Ragu &amp; Jonelagadda 2026 (label-free sticky HMM over foundation-model outputs) | first label-free explicit-duration latent-state model over frozen-MLLM window reads for hateful-video (and training-free) localization, as an application; the minimum-length mechanism finding; r4 only: no time constant in seconds | HSMM; negative binomial from chained phases; length models over frame scores; HMM / HSMM over normal-scale scores; averaging over model settings; log-uniform prior; independent per-stream chains; LELA's any-modality rule | HSMM, negative binomial, model averaging, EM or OR as new; "better than the previous persistence chain"; "better than smoothing on both corpora"; "durations learned from data"; "helps short or sparse hate"; "averaging improves accuracy" | coupling: **yes**; minimum length: **yes** in r4 (r3: point estimate only); averaging: **no**; OR: yes, but measured in round 2 only |
| **M4** composition | Gao &amp; Tan 2006 and Brümmer &amp; Garcia-Romero 2014 (identical calibration); Prototypical Calibration; STPN, UntrimmedNets (classify, then localize); T3AL (video decision, then within-video centring) | no novelty; present it as the scoring rule matched to the two metric families | all of the left, plus arXiv 2608.21854 for the within metric | calibration as new or as a gain; the composition or the interval rule as new | calibration: **no** (PR +.007 / +.004); key and within term: yes, but by construction |

---

## 3. Literature per module

### 3.1 M1: reads

**Hateful video (all opened unless marked).**
- **LELA, "Towards Training-free Multimodal Hate Localisation with Large Language Models"**
  ([arXiv 2602.09637](https://arxiv.org/abs/2602.09637), [HTML](https://arxiv.org/html/2602.09637v1)). This is the only
  training-free hateful-video localizer.
  - Per frame and per modality, captioners feed an LLM, which writes a rationale and then a 0–1 score. The captioners
    are BLIP-2, OCR, Whisper, music captions and video captions.
  - Fusion is the max over modalities ("if any modality signals hatefulness, the frame is flagged"), then a fixed .5
    threshold.
  - It has no temporal model, no video-level decision and no calibration. It reports pooled metrics only, and its
    protocol is not stated.
  - It uses 12–16 LLM calls per frame (verified from the full text in
    `docs/protocol_1fps_legacy/LOCALIZATION_LANDSCAPE.md`).
  - Same as ours: per-modality reads per time unit, "either modality suffices", the transcript as context.
  - Different from ours: one MLLM reads a shared multimodal prefix once; each window gets cached, isolated Yes/No
    branches; there is a verdict and a stance turn; the cost is one video encoding per video.
- **MARS** ([arXiv 2601.15115](https://arxiv.org/abs/2601.15115), ICASSP 2026). Training-free, video-level detection
  only. It chains the model's own intermediate outputs (description, hate hypothesis, non-hate hypothesis, synthesis).
  This is "own output in context" at video level, with no temporal output.
- **CLARA** ([arXiv 2608.15905](https://arxiv.org/abs/2608.15905)). A Qwen3-VL-8B rationale per video is fused into a
  trained model over utterance clips, for video classification only. Global context conditions local units, but the
  model is trained.
- **SafeLens** (AAAI-26 demo, [OJS](https://ojs.aaai.org/index.php/AAAI/article/view/42390)). Segment decisions come
  from a LoRA model trained on HateClipSeg. The video summary is built from the segment decisions, the opposite
  direction to ours.
- **MultiHateLoc** ([arXiv 2512.10408](https://arxiv.org/abs/2512.10408), WWW 2026) and **TANDEM**
  ([arXiv 2601.11178](https://arxiv.org/abs/2601.11178)). Both are trained. MultiHateLoc has per-modality streams;
  TANDEM feeds another model's output into its prompt during training. Neither conditions on its own verdict at
  inference.
- **SafeWatch** ([arXiv 2412.06878](https://arxiv.org/abs/2412.06878)), a trained video guardrail. It encodes policy
  chunks in parallel, which is the isolation primitive applied to policy text in video moderation.
- **Detection only:** HVGuard ([EMNLP 2025](https://aclanthology.org/2025.emnlp-main.456/)), RAMF
  ([2512.02743](https://arxiv.org/abs/2512.02743)), MM-HSD ([2508.20546](https://arxiv.org/abs/2508.20546)),
  ImpliHateVid ([ACL 2025](https://aclanthology.org/2025.acl-long.842/)), IARE
  ([2606.11953](https://arxiv.org/abs/2606.11953)) and SCANNER ([2602.00132](https://arxiv.org/abs/2602.00132)).
  MoRE was opened on [GitHub](https://github.com/Jian-Lang/MoRE) only.
- **Result of the search:** no hateful-video paper uses shared-prefix isolated per-segment branches, or conditions
  segment queries on the model's own verdict.
- **Not verified:** the WWW'26 paper "An Interpretable Agentic Framework for Multimodal Hate Video Analysis with
  Explicit Evidence Attribution" (doi 10.1145/3774905.3796488). The ACM page returned 403 and no arXiv copy was found.

**Training-free VAD, action localization and grounding.**
- **ESOM** ([arXiv 2604.07772](https://arxiv.org/html/2604.07772)). Checked directly by this reviewer. Streaming VAD
  on Qwen3-VL-8B.
  - The system prompt and the anomaly definitions are cached once as a KV prefix and reused for every sliding window.
  - Each window sees its own frames plus a text memory of earlier windows. There is no whole-video verdict.
  - It is the closest in structure. The difference is what the prefix holds: no video content and no verdict.
- **T3AL** ([HTML](https://arxiv.org/html/2404.05426), CVPR 2024), **FreeZAD**
  ([HTML](https://arxiv.org/html/2501.13795)) and **Liberatori et al. 2026**
  ([HTML](https://arxiv.org/html/2605.22201)). Localization is conditioned on the model's own video-level prediction:
  the pseudo-label chooses the text prompt. This is the closest precedent for the stance turn's idea: predict for the
  video, then localize under that prediction.
- **Yes/No token probabilities per clip:**
  - Probe-VAD ([2609.17211](https://arxiv.org/html/2609.17211)) also reuses one visual encoding across its 10
    questions.
  - "Your VLM Already Knows When" ([2608.08315](https://arxiv.org/html/2608.08315)) notes that ranking inside a video
    is immune to scale drift across videos.
  - "A VLM Answer Is Not an Anomaly Score" ([2608.21244](https://arxiv.org/html/2608.21244)).
  - These are precedents for the Yes/No log-odds read-out.
- **Segment-local readers:**
  - LAVAD ([2404.01014](https://arxiv.org/abs/2404.01014)): captions plus an LLM per 10 s window.
  - VERA ([2412.01095](https://arxiv.org/html/2412.01095)): questions learned from video labels.
  - VADTree ([2510.22693](https://arxiv.org/abs/2510.22693)), EventVAD
    ([2504.13092](https://arxiv.org/html/2504.13092)), AnyAnomaly ([2503.04504](https://arxiv.org/abs/2503.04504)),
    CoReVAD ([2605.23116](https://arxiv.org/abs/2605.23116)).
  - CEAVAD ([2608.09908](https://arxiv.org/html/2608.09908)): Qwen3-VL-8B two-token posteriors with past/future local
    context only and no temporal model (checked directly).
  - None of them reads a segment against the whole video.
- **PANDA** ([2509.26386](https://arxiv.org/html/2509.26386v1)). A global plan built from keyframes conditions every
  clip. That is global context, not the model's own verdict.

**Systems mechanism (prefix sharing and isolation).**
- **Exact shared-prefix computation for independent requests:**
  - Hydragen ([2402.05099](https://arxiv.org/abs/2402.05099));
  - SGLang RadixAttention ([2312.07104](https://arxiv.org/abs/2312.07104));
  - ChunkAttention ([2402.15220](https://arxiv.org/abs/2402.15220));
  - Prompt Cache ([2311.04934](https://arxiv.org/abs/2311.04934));
  - [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/), which covers image inputs.
- **Isolated parallel encodings:** PCW ([2212.10947](https://arxiv.org/abs/2212.10947); isolated contexts, the reverse
  of our layout) and APE ([2502.05431](https://arxiv.org/abs/2502.05431)).
- **SingGuard** ([2606.22873](https://arxiv.org/html/2606.22873), June 2026). A rule-isolation mask checks policy rules
  in parallel on a shared content prefix, in content moderation (images and text). It is the closest to our branch
  isolation.
- **Also related:**
  - InvariRank ([2604.27599](https://arxiv.org/abs/2604.27599)): shared positional framing under RoPE, which matches
    our branch position restart.
  - T3S ([2511.17945](https://arxiv.org/html/2511.17945)): a block-diagonal mask over video subsequences.
  - arXiv 2605.20194: processes text chunks separately to avoid carryover.
  - Free-MoRef ([2508.02134](https://arxiv.org/abs/2508.02134)).
- **Encode the video once, ask many questions:** ReKV (ICLR 2025, [2503.00540](https://arxiv.org/abs/2503.00540)) and
  STTM ([2507.07990](https://arxiv.org/html/2507.07990)), both checked directly.
- **Chain-of-Verification** ([2309.11495](https://arxiv.org/abs/2309.11495), abstract only). It answers verification
  questions independently "so the answers are not biased by other responses". That is the opposite choice to the
  stance turn; cite it as a contrast.

**Verdict on M1.**
- **New, as far as found in all four groups:**
  - one cached prefix holding the whole video (timestamped frames, timestamped transcript, policy);
  - the model's own verdict appended as an assistant turn;
  - then isolated visual and speech Yes/No branches for every fixed window.
- **What to claim:** each window is read against the whole video's context, including the model's own verdict.
  - Evidence (package ablation, §4): reading windows alone costs within .021 / .071 and pooled ROC .035 / .080. That
    is ≥ .01 on every metric on both corpora.
  - The package removes the frames, the transcript context, the stance turn and the dual branches, and uses the rules
    question.
  - The external result comes from M1. On DeHate, SPVL-r2 (M1 with the old composition and no time level) already
    exceeds the best weakly supervised within by +.099 [.054, .142].
- **Stance turn.** It meets the point threshold on within under r3 (−.014 / −.011), with weaknesses:
  - no bootstrap interval;
  - across MLLMs it goes beyond noise on 5/7 models on HateMM and 3/7 on HCS (older composition, SPVL §11);
  - its main effect is a uniform shift of all windows in videos with a Yes verdict: +2.39 / +2.08 logits, and the
    within-video order barely changes (Spearman .93 / .92 with vs without; twolevel §15.1 item 3).
  - Claim it only as a component with a small, model-dependent effect. Cite T3AL, FreeZAD and STPN for localizing
    under the model's own class decision, and CoVe as the contrast.
- **Do not claim:**
  - the isolation mask, prefix caching or position restart (systems prior art; removing isolation changed within by
    −.014 / +.011, and this was not re-run under r3 or r4);
  - dual branches (removing them raises HateMM within by .009);
  - transcript context (HCS within +.001, pooled −.007 / −.008);
  - frames (an input);
  - fixed 8 s windows (standard; their gain of −.064 / −.058 when replaced by ASR segments is real, but not a
    novelty).

### 3.2 M2: reads to evidence

- **Sun &amp; Cai 2009, "Large-scale multiple testing under dependence"**, JRSS-B 71(2)
  ([PDF](http://www-stat.wharton.upenn.edu/~tcai/paper/FDR-HMM.pdf)).
  - A two-state HMM over z-values, fitted by EM, with the posterior of the null at each position.
  - The transform toward normality is parametric (log); ranks are not used.
- **Wang &amp; Wang 2024, "Large-scale dependent multiple testing via hidden semi-Markov models"**, Computational
  Statistics 39:1093–1126 ([Springer](https://link.springer.com/article/10.1007/s00180-023-01367-z); abstract only).
  - A hidden semi-Markov chain over test statistics with shifted negative-binomial durations, fitted by EM with
    forward–backward.
  - It is the closest statistical precedent for M2 and M3 together.
- **Liu, Lafferty &amp; Wasserman 2009, "The Nonparanormal"** ([JMLR PDF](https://www.jmlr.org/papers/volume10/liu09a/liu09a.pdf)).
  Each variable is replaced by Φ⁻¹ of its empirical distribution before a Gaussian model (a Gaussian copula).
- **McCaw et al. 2020** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8643141/)). Rank-based inverse normal
  transform; our Φ⁻¹((rank − .5)/N) is its c = 1/2 case.
- **Efron 2004** ([PDF](https://www.stat.cmu.edu/~jiashun/Teaching/F08STAT756/Lectures/Efron.pdf)). z = Φ⁻¹(p), a
  two-group mixture, and the local false discovery rate.
- **Chidambaram, McGill &amp; Perona 2019** ([arXiv 2001.00057](https://arxiv.org/pdf/2001.00057)). CNN frame scores are
  binned into equal-probability quantiles as the emissions of a two-state video HMM, whose parameters are counted from
  labelled videos.
- **Nothing in any VAD, localization or hate paper applies rank normal scores to MLLM reads before a mixture or
  HMM.** The nearest ideas:
  - 2608.08315 uses only the ranking inside a video;
  - 2608.21244 keeps answer probabilities for the ordering;
  - FUSE ([2604.18547](https://arxiv.org/pdf/2604.18547)) combines LLM-judge scores on different scales without
    labels, with a different estimator.

**Verdict on M2.**
- It is a textbook transform plus textbook label-free mixture emissions, and it fails rule 14g: on the main run,
  removing it raises within by .005 / .003.
- Keep it as a design choice, justified by the cross-model count:
  - with normal scores, within is not below the previous time level on 15 of 16 corpus–model pairs;
  - with EM on the raw reads it is 12 of 16;
  - the mean within change is +.021 / +.025 (`runs/20260926_twolevel/robust/table_r3.txt`, twolevel §12.3).
- **Caveat:** the rank is taken over the whole test corpus, which is transductive. A single new video needs a stored
  reference set of reads.

### 3.3 M3: time level

- **Closest statistics:**
  - Wang &amp; Wang 2024 (§3.2): a label-free HSMM with negative-binomial durations over normal-scale statistics.
  - Sun &amp; Cai 2009: the HMM version.
- **Closest structure: Hayashi et al., ICASSP 2017** ([PDF](https://www.jonathanleroux.org/pdf/Hayashi2017ICASSP03.pdf)).
  - Sound-event detection with one independent HMM per event. Each event has chained active states over BLSTM
    outputs, so it has a minimum duration.
  - The transitions come from labels.
- **Length models over network frame scores (all supervised):**
  - Richard et al. 2017, fine-to-coarse sub-action chains ([arXiv 1703.08132](https://arxiv.org/abs/1703.08132));
  - Richard &amp; Gall 2016 ([PDF](https://alexanderrichard.github.io/publications/pdf/richard_temporal_action_detection.pdf));
  - NN-Viterbi 2018, a Poisson length model whose mean is re-estimated
    ([arXiv 1805.06875](https://arxiv.org/pdf/1805.06875)).
- **Negative binomial with minimum k from k chained phases:**
  - Yu 2010, "Hidden semi-Markov models", §4.1.3 ([PDF](https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf));
  - Johnson 2005 ([PDF](https://speechlab.eece.mu.edu/johnson/papers/johnson_spl05.pdf)).
  - Our shape k = 4 with leave probability 4k / D per cell is exactly this case.
- **Durations or persistence integrated out, and averaging over models:**
  - BOCPD, Adams &amp; MacKay 2007 ([arXiv 0710.3742](https://arxiv.org/abs/0710.3742)): the run length is integrated
    and the hazard is fixed.
  - Turner et al. 2009 ([PDF](https://mlg.eng.cam.ac.uk/pub/pdf/TurSaaRas09.pdf)): the hazard is fitted.
  - Wilson, Nassar &amp; Gold 2010 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2966286/)): the hazard is integrated.
  - Sticky HDP-HMM, Fox et al. 2008 ([PDF](https://ics.uci.edu/~sudderth/papers/icml08.pdf)).
  - HDP-HSMM, Johnson &amp; Willsky 2013 ([arXiv 1203.1365](https://arxiv.org/abs/1203.1365)).
  - **RJaCGH**, Rueda &amp; Díaz-Uriarte 2007 ([PLOS Comp Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030122)):
    "we incorporate model uncertainty through Bayesian model averaging" over HMMs, for segmentation.
  - Jaynes 1968 ([PDF](https://bayes.wustl.edu/etj/articles/prior.pdf)): the scale-invariant 1/σ prior, which a
    log-spaced grid of lengths implements.
- **Temporal models over zero-shot foundation-model outputs:**
  - **Kang et al., JMIR AI 2025** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12757708/), checked directly).
    GPT-4o gives zero-shot frame labels, and an HMM with Viterbi decoding cleans them up. Transitions and emissions
    come from 50 hand-labelled videos, and there is no duration model.
  - **Ragu &amp; Jonelagadda 2026** ([arXiv 2605.12838](https://arxiv.org/abs/2605.12838), abstract checked directly).
    A sticky factorial HDP-HMM over per-modality valence–arousal outputs, for emotion regimes. It has no explicit
    durations and does no localization.
- **Training-free VAD and localization use no HMM, HSMM or duration model.** They use:
  - Gaussian, EMA or moving-average smoothing;
  - similarity averaging and position weights;
  - event segmentation;
  - fixed threshold state machines (OZ-TAL, [2605.09976](https://arxiv.org/html/2605.09976));
  - window widths set per dataset (2608.08315).
- **Hateful video has no temporal model at inference.** MultiHateLoc has a training-time smoothness loss. arXiv
  2508.04900 argues for "temporal continuity" but proposes no model.
- **Multiple chains:**
  - factorial HMM, Ghahramani &amp; Jordan 1997 ([PDF](https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf));
  - coupled chains with negative-binomial durations, Touloupou et al. 2019 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7455056/)).
  - Our OR over two independent posteriors is Hayashi's independent-chain design applied across modalities, with
    LELA's "either modality" rule at the level of the latent state.

**What is the same.** Every statistical part:
- HSMM with negative-binomial durations from chained phases;
- EM-fitted Gaussian emissions on normal-scale scores;
- a posterior at each position;
- averaging over model settings with a scale-invariant grid;
- independent per-stream chains.

**What is different.**
- The observations are frozen-MLLM Yes/No reads of 8 s windows, each observing a pair of 4 s cells.
- The minimum length is set by the reading grid (two windows).
- The mean hate length and mean gap length are averaged per video, on a log grid bounded by that video's own length.
  This uses no labels but is data-dependent, which a referee may ask about.
- Two modality chains are combined by OR.
- EM pools many short videos.
- The output is a within-video rank plus a video key.

**Verdict on M3.**
- **May claim (an application, under rule 4's transfer definition):** the first label-free explicit-duration
  latent-state model over frozen-MLLM window reads for hateful-video localization. None was found in training-free
  video localization in general either.
- **Mechanism claim (meets 14g in r4).**
  - With emissions fitted without labels, a chain without a minimum length lets one strong window form its own
    segment. Requiring two reading windows restores the persistence.
  - Removing the coupling costs within −.1175 / −.0582, and removing the minimum length −.1035 / −.0424. All
    intervals exclude 0.
  - In r3, where the mean is fixed at 80 s, the minimum length alone costs −.014 / −.010 and the intervals include 0.
    So the claim holds for r4: the minimum length matters because the lengths are free.
- **r4 only: no time constant in seconds.**
  - Per-video averaging matches the test-selected 80 s mean (+.0017 / −.0013).
  - It avoids the worst scanned mean: under r3, a 40 s mean costs −.024 / −.020.
  - This is a property claim, not a novelty claim; it fails 14g as a gain.
- **OR fusion.** Per-modality chains + OR beat one shared chain by −.016 / −.045, but only in round 2. Re-measure it
  under the final model before stating it, and do not present it as new (LELA).
- **Must not claim:**
  - "better than the previous persistence prior": r4 minus the old chain is −.0004 / +.0019 within
    (`runs/20260926_twolevel/current`);
  - "not reproducible by smoothing" on both corpora: the best Gaussian (σ 8 s, chosen on test among five widths) is
    .067 below on HateMM but only .007 below on HCS (`runs/20260926_glr/infer/base_gridA_gauss8`);
  - "durations learned from data" (§15 is negative);
  - "helps short or sparse hate": K5 gives +.032 / −.019, with intervals including 0.
- **External evidence is weaker.** On DeHate the whole time level adds +.013 within over the SPVL-r2 composition
  (.6406 → .6539). On the main corpora it adds +.074 / +.028 (`base_gridA_spvlr2` → `r4_bma`).
- **Rule-4 case (3) exposure.** Functionally, M3 is an inference-time model over classifier outputs. It does more than
  smoothing on HateMM, but not demonstrably on HCS. Present it as an explicit probabilistic model, and report the
  smoothing control next to the no-coupling arm.

### 3.4 M4: composition

- **Gao &amp; Tan, ICDM 2006** ([PDF](https://cse.buffalo.edu/~jing/doc/ICDM06.pdf)). The labels are hidden variables
  fitted by EM. With equal-variance Gaussians the posterior is p = 1/(1 + exp(−Af − B)), which is our
  logit P(V=1|K) = aK + b.
- **Brümmer &amp; Garcia-Romero, ICASSP 2014** ([arXiv 1311.0707](https://arxiv.org/pdf/1311.0707)). Unsupervised score
  calibration with two common-variance Gaussians fitted as a GMM, giving an affine map from score to log-likelihood
  ratio. This is identical to ours apart from the prior term.
- **Prototypical Calibration**, Han et al., ICLR 2023 ([arXiv 2205.10183](https://arxiv.org/pdf/2205.10183)). A GMM
  fitted to LLM outputs without labels.
- **Other label-free calibration** (abstracts only): Batch Calibration ([2309.17249](https://arxiv.org/abs/2309.17249)),
  Contextual Calibration ([2102.09690](https://arxiv.org/abs/2102.09690)) and Generative Calibration
  ([Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.152/)).
- **STPN**, CVPR 2018 ([PDF](https://arxiv.org/pdf/1712.05080)). Video-level classification selects the classes, then
  a per-segment weight is multiplied by the class score: classify, then localize, with product gating. UntrimmedNets
  ([1703.03329](https://arxiv.org/abs/1703.03329), abstract only) is similar.
- **T3AL.** A video-level pseudo-label, then a threshold at the video's own mean, i.e. centring within the video.

**Verdict on M4.**
- The calibration is identical to prior work, and the composition is standard.
- As a gain it fails 14g: PR +.007 / +.004.
- The "no key" and "no within term" ablations hold by construction: removing either term removes the only source of
  that metric's ordering.
- Do not claim M4. Present it as the scoring rule matched to the two metric families (pooled = the order between
  videos; within = the order inside a video; 2608.21854), with the product as the source of intervals.

### 3.5 The whole pipeline

**Closest work.**
- Same task: LELA. It is training-free, makes many calls per frame, has no temporal model, and reports pooled metrics
  only.
- Same paradigm:
  - T3AL: video-level decision, localization under it, centring within the video, smoothing;
  - ESOM: Qwen3-VL-8B with a cached prefix over streaming windows;
  - LAVAD, VERA, VADTree: segment scores followed by refinement.
- No paper found in the four groups combines the three modules.

**Strength.** Moderate as a method paper. Each part has a citable source; the contribution is the task-specific
design and its evidence. There are also two measured findings that the literature states only as open problems.
- **In-context state tracking fails; explicit inference over isolated reads works.**
  - Asking the MLLM to carry persistence across windows in context lost within .03 / .04–.07 (HVL).
  - An explicit prior over isolated reads gained .05 / .02 (TIL §8; measured before the 2026-09-26 ASR fix).
- **The window read tracks topic, not only hate.**
  - The per-window read responds to a protected-group mention about as much as to hate (error analysis, all three
    corpora).
  - No readout of the cached reads fitted on test labels beats the unfitted read: .742 / .626 against .758 / .621
    (TAD §5d).
  - The field states the same limit as "definition blindness" (arXiv 2607.20780, [abs](https://arxiv.org/abs/2607.20780))
    and "hazard resemblance" (CEAVAD).

**The within metric is not ours.**
- It is Within-AUROC / Macro-AUROC in [arXiv 2608.21854](https://arxiv.org/abs/2608.21854), and within-video AUC in
  [arXiv 2608.11985](https://arxiv.org/abs/2608.11985). Both are August 2026 VAD papers, checked directly.
- Macro AUC has long been used in one-class VAD ([arXiv 2211.15597](https://arxiv.org/html/2211.15597v4), which
  credits Georgescu et al.).
- What is new is only its use in hateful video: no hateful-video paper reports it.

---

## 4. Rule-14g evidence (evaluator outputs)

Each difference is the variant minus the full method (pooled ROC / pooled PR / within). The noise floor is pooled
.005 and within .01. Arm names are directories under `runs/20260926_twolevel/` unless another path is given.

| component (claim candidate) | ablation arm | HateMM | HCS | meets 14g? |
|---|---|---|---|---|
| M1 package: windows read alone instead of against the shared context | `robust/winonly_r3` vs `robust/full_r3` (cache path of the family study, pre-fix ASR) | −.035 / −.036 / −.021 | −.080 / −.035 / −.071 | yes, all three metrics |
| M1 frames in the prefix | `robust/noframes_r3` | −.029 / −.025 / −.060 | −.082 / −.051 / −.109 | yes (an input, not a novelty) |
| M1 transcript context | `robust/noctx_r3` | −.147 / −.186 / −.039 | −.007 / −.008 / +.001 | no (HCS) |
| M1 stance turn | `robust/nostance_r3` | +.002 / +.005 / −.014 | −.010 / −.013 / −.011 | on within, point estimates only |
| M1 dual branches | `robust/joint_r3` | −.002 / −.003 / +.009 | −.026 / −.017 / −.074 | no (HateMM) |
| M1 isolation (block vs causal mask) | SPVL E4, round-1 composition, no time level (`experiments/20260910_spvl/README.md` §8) | within −.014 | within +.011 | no; not re-run |
| M1 fixed 8 s windows vs ASR segments | TIL §9, old chain | within −.064 | within −.058 | yes; fixed windows are standard |
| M2 normal scores vs EM on raw reads | `c_m2` vs `r3_m2` | .000 / +.001 / +.005 | .000 / .000 / +.003 | no |
| M3 time coupling | `r4_nocoupling` | −.001 / −.006 / **−.1175** | −.003 / −.003 / **−.0582** | yes; intervals exclude 0 |
| M3 minimum two windows (lengths averaged) | `r4_k1` | −.001 / −.011 / **−.1035** | −.001 / −.004 / **−.0424** | yes; intervals exclude 0 |
| M3 minimum length with fixed 80 s mean | `r3_k1` vs `r3_m2` | .000 / −.003 / −.014 | .000 / −.001 / −.010 | point estimate only; intervals include 0 |
| M3 averaged lengths vs fixed 80 s | `r3_m2` vs `r4_bma` | .000 / +.001 / −.002 | .000 / −.001 / +.001 | no (a constant removed without loss) |
| M3 per-modality chains + OR vs one shared chain | `r2nl_carrier` vs `r2_noleak` (round 2) | .000 / −.001 / −.016 | −.001 / −.001 / −.045 | yes, but round 2 only |
| M2+M3 vs the old geometric chain with corpus-std scaling | `current` vs `r4_bma` | −.001 / −.006 / .000 | −.003 / −.004 / −.002 | no gain on the main model |
| M3 vs best Gaussian smoothing (σ 8 s) | `runs/20260926_glr/infer/base_gridA_gauss8` vs `r4_bma` | −.002 / −.008 / −.067 | −.003 / −.004 / −.007 | HateMM only |
| M4 calibrated key vs raw key | `r2_noleak` vs `c_m2` | −.002 / −.007 / 0 | −.003 / −.004 / 0 | no |
| M4 video key | `c_nokey` vs `c_m2` | −.327 / −.418 / 0 | −.147 / −.147 / 0 | yes, by construction |
| M4 within term | `c_norank` vs `c_m2` | −.004 / −.015 / −.258 | −.006 / −.008 / −.143 | yes, by construction |

**Sensitivity.** Grid resolution 4 or 10 instead of 6 changes nothing beyond ±.001 (`r4_bma_g4`, `r4_bma_g10`).

**Cross-model counts.** Within not below the old chain, over 8 MLLMs:
- r3 time level: 7/8 on HateMM and 8/8 on HCS;
- EM on raw reads: 5/8 and 7/8.

r4 against r3 is still running (§1).

**External (DeHate).** Nothing was chosen on it. Source: `runs/20260927_dehate_external/summary/table.txt`.
- r3_m2 − MultiHateLoc within: +.112 [.062, .159].
- SPVL-r2 alone − MultiHateLoc within: +.099 [.054, .142].
- Pooled ROC is level with Fed-WSVAD (+.0001).
- Pooled PR is below it: −.017 [−.078, +.038].

**Where the component ablations were run.** The component ablations of M1 come from the cache-path runs of the
family study, which predate the ASR fix of 2026-09-26. The main numbers use the fixed reads (`base_gridA`). The M4
ablations were run under the round-2 time level.

---

## 5. Rule-4 STOP check

| case | finding |
|---|---|
| 1. The source method is already used for hateful video | No. LELA shares per-modality reads per time unit and either-modality fusion, which are cited, not claimed. No hateful-video paper has the shared-context verdict-conditioned reads, the temporal model or the composition. |
| 2. Pure ensemble | No. One MLLM; its two modality branches are allowed by rule 3. |
| 3. Pure calibration, post-processing or smoothing | M4's calibration alone would be case 3 (and it equals Gao &amp; Tan and Brümmer &amp; Garcia-Romero). It is not claimed. M2+M3 alone are at the edge of case 3 (§3.3). The method as a whole passes: M1 is a reading design with its own 14g evidence, and M3 is an explicit latent-state model whose effect linear smoothing does not reproduce on HateMM. |
| 4. Pure engineering | The shared prefix and the isolation are engineering (cost) and are not claimed. |

**Result: PASS**, with claims restricted as in §6.

---

## 6. Claims

### 6.1 Allowed (suggested wording)

1. **Setting.**
   - Label-free hateful-video temporal localization: no hate labels for training, adaptation, calibration or
     thresholds.
   - A frozen open MLLM, one video encoding per video.
   - Output: frame scores and intervals.
   - State that EM, ranks and key calibration are fitted on the unlabeled test corpus.
2. **M1.** "Each fixed window is read against one cached context that holds the whole video (timestamped frames,
   timestamped transcript, policy) and the model's own verdict, in isolated per-modality Yes/No branches."
   - Evidence: reading windows alone costs .021 / .071 within and .035 / .080 pooled ROC.
   - "To our knowledge, first for hateful video and for training-free video localization."
3. **M3.** "Persistence enters as an explicit-duration latent-state model outside the MLLM, fitted without labels. A
   segment spans at least two reading windows."
   - r4 only: "The mean lengths are averaged per video, so no time constant in seconds is set by hand."
   - Evidence: coupling −.118 / −.058; minimum length −.104 / −.042; the averaging matches the test-selected 80 s.
   - If r4 fails its gate, drop the r4-only sentences. Report the r3 coupling (−.116 / −.060) and state the 80 s mean
     as a declared prior.
4. **Across MLLMs.** It works with 8 MLLMs from 4 families (15/16 corpus–model pairs not below the previous time
   level).
5. **Results.**
   - Within beats weakly supervised baselines on HateMM and HCS (development-selected).
   - On DeHate (external) within is higher by .09 to .11, and the intervals exclude 0.
   - Pooled ROC on DeHate is level with Fed-WSVAD; pooled PR is lower (−.017, interval includes 0).
6. **Findings, reported as analysis, not modules:**
   - explicit persistence outside the model, against persistence carried in context;
   - the topic confound and its ceiling;
   - HateClipSeg's protocol GT counts non-hate offensive content (`experiments/20260927_dvd/README.md` §9.3).

### 6.2 Must be cited

**M1**
- LELA; ESOM; T3AL / FreeZAD / STPN; CoVe.
- Hydragen, RadixAttention, Prompt Cache, PCW / APE, SingGuard, InvariRank, T3S, ReKV / STTM.
- Probe-VAD, arXiv 2608.08315 and 2608.21244 (Yes/No read-out).

**M2**
- McCaw et al., the nonparanormal, Efron 2004, Sun &amp; Cai 2009, Wang &amp; Wang 2024.

**M3**
- Yu 2010, Johnson 2005, Richard 2016 / 2017, NN-Viterbi, Hayashi 2017.
- Sun &amp; Cai 2009, Wang &amp; Wang 2024.
- BOCPD, Wilson 2010, Fox 2008, Johnson &amp; Willsky 2013, RJaCGH, Jaynes 1968.
- Ghahramani &amp; Jordan 1997.
- Kang et al. 2025, Ragu &amp; Jonelagadda 2026.

**M4**
- Gao &amp; Tan 2006, Brümmer &amp; Garcia-Romero 2014, Prototypical Calibration, STPN.

**Metric and task**
- arXiv 2608.21854, 2608.11985, 2211.15597.
- HateMM, HateClipSeg, DeHate, MultiHateLoc.
- arXiv 2508.04900.

### 6.3 Drop or do not claim

1. The isolation mask, prefix caching and position restart as contributions.
2. Dual visual and speech branches as a gain (HateMM: +.009 when removed).
3. Transcript context as a gain on both corpora (HCS ≈ 0).
4. Normal scores as a contribution. Keep them as a choice for robustness across models.
5. The calibrated key as a contribution, or as a gain.
6. The key + centred-rank composition, and the product interval rule, as contributions.
7. "Durations learned from data"; "better than smoothing" without the HCS caveat; "helps short or sparse hate";
   "model averaging improves accuracy"; "better than the previous persistence chain".
8. The stance turn as the reason windows discriminate better inside a video. It mainly shifts all windows.
9. "First training-free hateful-video localization" (LELA exists). "First within-video metric" (2608.21854,
   2608.11985, macro AUC in VAD).
10. "State of the art among label-free methods" until a same-backbone training-free comparator is run (§8, item 1).
11. The definition-conjunctive verdict (DVD). It failed its gate and is not part of the method.
12. "First interval output for hateful video". HateClipSeg's ActionFormer baseline, SafeLens and TANDEM output
    segments.

---

## 7. Paper story (challenges as the field states them)

**Setting.** No hate labels are used anywhere. A frozen open MLLM produces frame scores. Results are reported with
pooled frame metrics and the within-video macro AUROC of arXiv 2608.21854.

Quotes below must be checked against the PDFs (§8, item 8).

**Challenge 1: whether a segment is hateful depends on context the segment itself may not show.**
- How the field states it:
  - LELA: videos "convey hate in implicit, context-dependent ways".
  - HateClipSeg ([2508.01712](https://arxiv.org/abs/2508.01712)): "Short clips often lack broader context needed to
    disambiguate subtle or implicit hate speech".
  - arXiv 2508.04900 ([abs](https://arxiv.org/abs/2508.04900), checked directly): the "inherent context dependency and
    temporal continuity of hate speech expression".
  - arXiv 2609.00206 ([abs](https://arxiv.org/abs/2609.00206)): "videos composed of seemingly benign components can
    convey harmful meaning when interpreted as a whole".
  - SafeLens: off-the-shelf VLMs "struggle to separate hateful content from nearby benign context".
- What prior work does: training-free localizers read each frame or segment from local content (LELA, LAVAD, VERA,
  VADTree, EventVAD, CEAVAD).
- **Module: M1**, reading every window against the whole video and the model's own verdict.
- **Evidence:**
  - reading windows alone costs .021 / .071 within and .035 / .080 pooled ROC;
  - on DeHate, M1 without any time model already beats the best weakly supervised within by .099;
  - the stance turn adds a small, model-dependent part (−.014 / −.011 when removed).

**Challenge 2: hate persists in time, and this must be imposed without labels or tuned widths.**
- How the field states it:
  - arXiv 2508.04900: "temporal continuity"; hate expressions involve "gradual semantic transitions".
  - TANDEM: hate "may only appear briefly within a long video".
  - VADTree: "fixed-length temporal window sampling approaches struggle to accurately capture anomalies with varying
    temporal spans".
  - EventVAD: "prediction inconsistency across frames".
  - PANDA: training-free methods "depend heavily on manual engineering … and post-processing".
  - FreeZAD: training-free methods lack "task-specific priors".
- What prior work does:
  - training-free methods smooth with fixed widths (T3AL moving average; VERA, AnyAnomaly and CoReVAD Gaussian;
    AnomalyRuler EMA; per-dataset widths in 2608.08315);
  - hateful-video methods have no temporal model at inference.
- **Module: M3 (with M2)**, an explicit-duration latent-state model per modality, fitted without labels, with a
  minimum of two reading windows and (r4) mean lengths averaged per video.
- **Evidence:**
  - coupling −.118 / −.058;
  - minimum length −.104 / −.042;
  - averaging matches the test-selected 80 s;
  - 15/16 corpus–model pairs not below the previous chain.
- **Caveats to state:**
  - HCS is within .007 of the best Gaussian;
  - DeHate gains only +.013 from the time level.

**Challenge 3: the task has two levels, and the usual pooled metrics mostly measure the video level.**
- How the field states it:
  - arXiv 2608.21854: only "0.071–0.388%" of anomalous–normal frame comparisons fall within one video, and
    video-constant outputs reach "81.40–97.18 Micro-AUROC";
  - arXiv 2608.11985: "pooled AUC does not reliably predict within-video anomaly localization".
- What prior work does: hateful-video localizers report only pooled metrics (LELA, MultiHateLoc).
- **Module: M4**, a video key for the order between videos plus the time-level posterior for the order inside a
  video, and within reported as a main metric.
- **Evidence:**
  - dropping either term removes its metric's ordering (by construction);
  - the result is within beating weakly supervised baselines on all three corpora.
- This is a design that matches the metrics, not a novelty claim.

**Challenge 4 (cost only): per-frame, multi-call LLM pipelines are expensive.**
- How the field states it:
  - LELA uses 12–16 calls per frame;
  - PANDA makes several calls per clip;
  - ESOM: agent or multi-model pipelines "fail to meet real-time requirements".
- **Module: M1's cached prefix.**
- **Evidence:**
  - one video encoding per video, about 1.5 s per video on one RTX 5090;
  - branches equal independent calls (Spearman 1.000; `experiments/20260910_spvl/README.md` §11).
- No accuracy claim; the mechanism is cited.

**Limitations to state (aligned with field statements).**
- The window read tracks the topic ("definition blindness", 2607.20780).
- HateClipSeg's protocol GT counts non-hate offensive content.
- Short or sparse hate is weak (K5).
- 20 uniform frames leave 24–34% of windows without a frame of their own. arXiv 2508.10974 reports that sparse
  uniform sampling misses harmful content.
- EM, ranks and key calibration are fitted transductively on the test corpus.

---

## 8. Gaps to close before writing (for the main agent; not gates)

1. **Rule 14f.** Add a same-backbone training-free comparator. Either or both of:
   - (a) LELA re-implemented with Qwen3-VL-8B or an open 7B model (the LELA paper reports open 7B results);
   - (b) Qwen3-VL-8B per-window Yes/No reads alone plus Gaussian smoothing (the recipe of VERA, Probe-VAD and
     2608.08315; the `winonly` reads already exist).
2. **Re-run the component ablations of M1 on the final reads.** Use `base_gridA` with the fixed ASR, the final time
   level and paired bootstrap intervals, for stance, transcript, frames, dual branches and isolation. Isolation has
   only a round-1 number.
3. **Re-measure OR fusion** (per-modality chains against one shared chain) under the final time level.
4. **Report the smoothing controls in the ablation table.** Gaussian at several widths and a T3AL-style moving average,
   next to the no-coupling arm.
5. **Run r4 on DeHate** (pending in `experiments/20260926_twolevel/launch/run_r4.sh`) and report the gain of the time
   level there, with an interval.
6. **Describe the transductive fitting** and the procedure for a single new video (a stored reference set for ranks,
   EM and key calibration).
7. **Fix a citation.** arXiv 2602.21854 is FewMMBench, a few-shot MLLM benchmark, not a hateful-video paper. It is
   listed as one in `experiments/20260910_spvl/README.md` §11.
8. **Check every quote in §7 against the PDF.** Most came through a summarizing fetch tool. The checked-directly ones
   are 2508.04900, 2608.11985, 2608.21854 (abstract), ESOM and CEAVAD.
9. **Read the WWW'26 "Explicit Evidence Attribution" hateful-video paper** (doi 10.1145/3774905.3796488; not
   accessible here).

---

## 9. Sources opened (grouped)

**Hateful video**
- [LELA](https://arxiv.org/abs/2602.09637), [MultiHateLoc](https://arxiv.org/abs/2512.10408),
  [TANDEM](https://arxiv.org/abs/2601.11178), [HateClipSeg](https://arxiv.org/abs/2508.01712),
  [SafeLens](https://ojs.aaai.org/index.php/AAAI/article/view/42390), [MARS](https://arxiv.org/abs/2601.15115).
- [HVGuard](https://aclanthology.org/2025.emnlp-main.456/), [RAMF](https://arxiv.org/abs/2512.02743),
  [MM-HSD](https://arxiv.org/abs/2508.20546), [ImpliHateVid](https://aclanthology.org/2025.acl-long.842/),
  [HateMM](https://arxiv.org/abs/2305.03915), [MultiHateClip](https://arxiv.org/abs/2408.03468).
- [DeHate (GitHub)](https://github.com/yuchen-zhang-essex/DeHate), [CLARA](https://arxiv.org/abs/2608.15905),
  [IARE](https://arxiv.org/abs/2606.11953), [SCANNER](https://arxiv.org/abs/2602.00132),
  [temporal label noise](https://arxiv.org/abs/2508.04900), [distributed implicit harm](https://arxiv.org/abs/2609.00206).
- [harmful-content omission in Video LLMs](https://arxiv.org/abs/2508.10974), [SafeWatch](https://arxiv.org/abs/2412.06878),
  [MoRE (GitHub)](https://github.com/Jian-Lang/MoRE).

**Evaluation**
- [2608.21854](https://arxiv.org/abs/2608.21854), [2608.11985](https://arxiv.org/abs/2608.11985),
  [2211.15597](https://arxiv.org/html/2211.15597v4).

**Training-free VAD, localization and grounding**
- [LAVAD](https://arxiv.org/abs/2404.01014), [AnomalyRuler](https://arxiv.org/html/2407.10299),
  [VERA](https://arxiv.org/html/2412.01095), [Holmes-VAD](https://arxiv.org/abs/2406.12235),
  [Holmes-VAU](https://arxiv.org/abs/2412.06171), [EventVAD](https://arxiv.org/html/2504.13092),
  [VADTree](https://arxiv.org/abs/2510.22693), [AnyAnomaly](https://arxiv.org/abs/2503.04504).
- [Flashback](https://arxiv.org/html/2505.15205), [PANDA](https://arxiv.org/html/2509.26386v1),
  [Vad-R1](https://arxiv.org/abs/2505.19877), [HiProbe-VAD](https://arxiv.org/html/2507.17394v1),
  [CoReVAD](https://arxiv.org/abs/2605.23116), [Probe-VAD](https://arxiv.org/html/2609.17211),
  [rank compression](https://arxiv.org/html/2608.21244), [Your VLM Already Knows When](https://arxiv.org/html/2608.08315).
- [ESOM](https://arxiv.org/html/2604.07772), [QVAD](https://arxiv.org/html/2604.03040),
  [GtS](https://arxiv.org/html/2608.11260), [definition blindness](https://arxiv.org/abs/2607.20780),
  [NOVA](https://arxiv.org/abs/2609.06360), [CEAVAD](https://arxiv.org/html/2608.09908),
  [T3AL](https://arxiv.org/html/2404.05426), [FreeZAD](https://arxiv.org/html/2501.13795).
- [Liberatori 2026](https://arxiv.org/html/2605.22201), [TFVTG](https://arxiv.org/html/2408.16219),
  [Moment-GPT](https://arxiv.org/html/2501.07972), [OZ-TAL](https://arxiv.org/html/2605.09976).

**Prefix sharing and isolation**
- [Hydragen](https://arxiv.org/abs/2402.05099), [SGLang](https://arxiv.org/abs/2312.07104),
  [ChunkAttention](https://arxiv.org/abs/2402.15220), [Prompt Cache](https://arxiv.org/abs/2311.04934),
  [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/), [PCW](https://arxiv.org/abs/2212.10947),
  [APE](https://arxiv.org/abs/2502.05431).
- [SingGuard](https://arxiv.org/html/2606.22873), [InvariRank](https://arxiv.org/abs/2604.27599),
  [T3S](https://arxiv.org/html/2511.17945), [Free-MoRef](https://arxiv.org/abs/2508.02134),
  [ReKV](https://arxiv.org/abs/2503.00540), [STTM](https://arxiv.org/html/2507.07990).
- [CoVe](https://arxiv.org/abs/2309.11495) (abstract only), [NExT-GQA](https://arxiv.org/abs/2309.01327).

**Rank transforms and calibration**
- [McCaw et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8643141/),
  [nonparanormal](https://www.jmlr.org/papers/volume10/liu09a/liu09a.pdf),
  [Efron 2004](https://www.stat.cmu.edu/~jiashun/Teaching/F08STAT756/Lectures/Efron.pdf),
  [Sun &amp; Cai 2009](http://www-stat.wharton.upenn.edu/~tcai/paper/FDR-HMM.pdf),
  [Wang &amp; Wang 2024](https://link.springer.com/article/10.1007/s00180-023-01367-z) (abstract only),
  [Chidambaram et al.](https://arxiv.org/pdf/2001.00057), [FUSE](https://arxiv.org/pdf/2604.18547).
- [Gao &amp; Tan](https://cse.buffalo.edu/~jing/doc/ICDM06.pdf), [Brümmer &amp; Garcia-Romero](https://arxiv.org/pdf/1311.0707),
  [Prototypical Calibration](https://arxiv.org/pdf/2205.10183), [Batch Calibration](https://arxiv.org/abs/2309.17249),
  [Contextual Calibration](https://arxiv.org/abs/2102.09690), [Generative Calibration](https://aclanthology.org/2023.findings-emnlp.152/),
  [STPN](https://arxiv.org/pdf/1712.05080), [UntrimmedNets](https://arxiv.org/abs/1703.03329).

**Temporal statistical models**
- [Richard 2017](https://arxiv.org/abs/1703.08132),
  [Richard &amp; Gall 2016](https://alexanderrichard.github.io/publications/pdf/richard_temporal_action_detection.pdf),
  [NN-Viterbi](https://arxiv.org/pdf/1805.06875), [Hayashi 2017](https://www.jonathanleroux.org/pdf/Hayashi2017ICASSP03.pdf),
  [Yu 2010](https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf),
  [Johnson 2005](https://speechlab.eece.mu.edu/johnson/papers/johnson_spl05.pdf).
- [Kang et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12757708/),
  [Ragu &amp; Jonelagadda 2026](https://arxiv.org/abs/2605.12838), [BOCPD](https://arxiv.org/abs/0710.3742),
  [Turner et al.](https://mlg.eng.cam.ac.uk/pub/pdf/TurSaaRas09.pdf), [Wilson et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC2966286/),
  [sticky HDP-HMM](https://ics.uci.edu/~sudderth/papers/icml08.pdf), [HDP-HSMM](https://arxiv.org/abs/1203.1365).
- [RJaCGH](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030122),
  [Jaynes 1968](https://bayes.wustl.edu/etj/articles/prior.pdf),
  [factorial HMM](https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf),
  [Touloupou et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7455056/).

**Not verified** (named but not opened):
- the WWW'26 evidence-attribution paper; Kriegel et al. 2011; Beasley et al. 2009.
- Rabiner 1989, Durbin et al. 1998, Russell &amp; Moore 1985 and Bonafonte 1996, seen only as citations in Yu 2010 and
  Johnson 2005.
- Brand et al. 1997 (coupled HMMs), CDFL, EndoNet's HMM.
