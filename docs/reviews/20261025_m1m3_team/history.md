# History audit for the M1 / M3 novelty team (2026-10-25)

Written by the history auditor. The repo was read-only: nothing in it was edited, and no hashes or checksums were
computed. Sources: `research-wiki/{STATUS,DIRECTIONS,paradigm_framing}.md`, every `experiments/*/README.md` and
`archive/experiments/*/README.md`, the 2026-08 notes under `archive/detection-2026-08/docs/` and
`archive/idea-stage-2026-08/`, `docs/reviews/*`, the memory files, and the 2026-10-25 scratchpad reports
(`novelty_eval/`, `m3_team/`). The working files behind this document are in
`/tmp/claude-135258174/-home-jehc223-Hate-follow-up/abae8735-0dba-4355-b09e-130e58ba97bd/scratchpad/m1m3_novelty/_parts/`.
Every number here is a transcription and development-selected (read on test). The authority is the `runs/` path
given with it.

Sections:
- §0: how to read the numbers, and the traps;
- §1: table of attempts, by era;
- §2: closed families, with scope;
- §3: open neighbourhoods;
- §4: data on disk for CPU pre-checks;
- §5: the current r6_bma computation, with file:line;
- §6: prior work already searched.

## 0. How to read this file

**Module names.** This file uses the charter's names: M1 = window read, M2 = time level, M3 = video key + fusion.
Older documents number them differently:
- `docs/reviews/20260927_final_novelty_review.md`: M2 = normal scores, M3 = time level, M4 = key + composition.
- STATUS "当前方法" block: M3 = time level, M4 = combination.

When a source says "M4", it means the charter's M3.

**Reference numbers** (pooled ROC / pooled PR / within; within on 84 HateMM and 99 HCS videos that hold both classes):
- **r6_bma as in STATUS (`--key calib`):** HateMM .8971 / .6942 / .7508; HCS .7168 / .6711 / .6373; DeHate .7011 / .1582 / .6431.
  Sources: `runs/20260926_twolevel/r6_bma/metrics.json`, `runs/20260927_dehate_external/r6_bma/metrics.json`.
- **r6 with the raw key (the paper version since the 2026-10-07 user ruling):** HateMM .8955 / .6883 / .7508; HCS .7137 / .6666 / .6373; DeHate .6994 / .1568 / .6431.
  Sources: `runs/20260926_twolevel/final_rawkey/metrics.json`, `runs/20260927_dehate_external/r6_rawkey/metrics.json`.
  See §5.0.
- **Raw window level** (window max read, before the decoder): within .680 / .610; raw visual .614 / .546.
- **Window-level within of the r6 speech read** (analysis-script level, clean-labelled windows): .726 / .606.

**Noise and comparability.**
- **Noise floors** (rule 7): pooled .005, within .01.
- **Sampling noise is larger than that.** Per-video bootstrap SE of within is .028 / .021, and paired 95% half-widths are .02–.05
  (memory `diagnosis-2026-09-26.md`).
- **Window-level and frame-level within are different numbers.** Window-level is an AUC over 8 s windows from an
  analysis script, comparable only inside one experiment. Frame-level comes from the shared evaluator.
- **Two ASR families.** "Old ASR" means reads made before 2026-09-26, when the loader dropped untimed last Whisper chunks
  (42 HateMM / 95 HCS records). "Fixed ASR" means reads made after the fix. Compare reads only within one family.
  The 8-MLLM family reads, HVL, TAD, BND, CVA, PWC and SPVL-r2 are all old ASR.

**Traps that have misled earlier rounds** (check every pre-check against them):
1. **Decoder-made HateMM gains.** HateMM within gains often come from the decoder, not from a better read.
   - hate_video_279 and hate_video_329 carried most of several "gains" while their raw window order did not change
     (visual contrast, Integrator, Program).
   - A per-video constant shift of all reads also changes r6's within (verdict_analysis HCS shift-only +.008).
   - So always report raw within and a shift-only control. Source: `oct_m1_part1.md`, "recurring evaluation trap".
2. **HateMM annotation structure.**
   - 37 of the 84 within-eligible videos are ≥ 75% hateful, and 94% of their negatives sit in the first or last 10%.
   - A position-only curve scores .961 on them.
   - The decoder's +.114 on HateMM comes almost entirely from this group.
   - Source: `experiments/20261014_within_diagnosis/README.md`.
3. **HCS protocol GT counts any non-normal dimension.**
   - 51 of 104 positive HCS test videos have no hateful segment.
   - Every change that moves the video level toward the "hate" definition loses on HCS protocol GT and gains on
     DeHate and HCS hateful-only: DVD T, candidate 49, the window-mean-only key.
   - The user kept the protocol GT on 2026-10-25.
4. **Label-chosen selection is not headroom.** Choosing per window, by label, which read to keep "gains" +.05 to +.09.
   This is label injection, not reachable headroom.
5. **Bounds are weak.**
   - Gold-fitted bounds use 84–215 videos and simple models; they do not show the method is at its limit.
   - "Every window read correctly" is never a ceiling (user rule, memory `feedback-noisy-reads-are-the-premise.md`).
   - Winning a relative rubric is not evidence of absolute room (user correction, 2026-10-25).
6. **HateMM frame gains are chain-mediated.** In-window frames lower HateMM visual window AUC (.644 → .591). Their +.018
   within appears only through the time chain. Any frames-based M1 claim must be split per corpus at the raw-read level.

## 1. Attempts that touched M1 or M3, all eras

Status words:
- **adopted**: in the current method;
- **archived**: archived negative result;
- **closed**: tried and dropped without its own archive item;
- **open**: run and kept for a user decision, or never run.

Candidate numbers follow STATUS: Grounder 1 … Lattice 22, Program 23 … Topic removal 46, M3 share reads 47–49.
"Item N" is the archive item number.

### 1.1 2026-08: detection era and localization idea stage

Two sub-eras with different protocols. Detection era (08-05..08-25): one whole-video call per video, 16 frames,
transcript (300-character truncated before channel restoration, full after), video-level labels, corpora
MHClip-EN/ZH, HateMM, HCS, ImpliHateVid, metric video AUC or macro-F1; readers Qwen3-VL-2B (predecessor substrate)
and 8B. Localization era (08-12..08-29): 4 fps GT on four corpora (611/643 videos), most results selected on the same
test videos (exploratory only). Sources: `archive/detection-2026-08/docs/duplex/<NAME>_NOTE.md`,
`archive/detection-2026-08/docs/analysis/prior_falsification_map.md`, `archive/idea-stage-2026-08/`,
`runs/legacy_idea_discovery_2026-08/`.

| id / name | date | module | mechanism in one line | key numbers (source) | stated cause | status |
|---|---|---|---|---|---|---|
| PP-v1 observe-then-judge | 04/08 (predecessor) | M3 (video verdict) | Video observer writes a ≤40-token (group, stance) sentence; text-only judge scores it | EN oracle acc .7391 vs .7764 (falsification map row 1) | Text bottleneck; abstract stance words merge satire and derogation | closed |
| PP-v2 factored verdict | predecessor | M3 | score = P(target present) × P(derogatory stance) | EN .7267 < baseline (row 2) | Product zeroes one-tall positives | closed (operator); phenomenon later re-tested by DVD |
| PP-v3 polarity-calibrated probes | predecessor | M3 | σ(.5(logit p_violates − logit p_consistent)) | ties baseline (row 3) | Saturated second probe; no paraphrase placebo | closed |
| PP-v4 modality-split calls | predecessor | M3 / modality fusion | Frames-only call and text-only call, rank noisy-OR | EN .7640 vs .7764, ZH .7987 vs .8121; modality carriage real (corr .295/.493, rescue 54–77 %) (row 4) | "The MLLM's own cross-modal attention is a stronger fusion operator than post-hoc noisy-OR" | closed (note: the 09-30 modality gate uses modality-only verdicts as weights on the joint-prefix chains, not as replacement scores) |
| PP-v5 per-rule disjunction | predecessor | M3 | Constrained decode of K binary rule flags, max | below baseline; rule-1 monopoly from decode position (row 5) | Decode-position geometry | closed |
| MADP definition conjunction | 04-18 | M3 | target_real_group ∧ hostile_framing ∧ ¬protected_context in one call | 0/4 datasets on Qwen3-VL-8B (row 17) | Hard conjunction blocks more true positives than false | closed (soft calibrated version = DVD, see 1a) |
| MARS reproduction | predecessor | M3 | Describe → assume-hateful → assume-benign → synthesize (4 calls) | .6584/.7450/.6977 vs single call .7640/.8121/.8047 (row 22; 2B, 47–75 % parse failures) | Assume-hateful stage inflates positives | closed |
| Dual-axis kill test | 08-09 | M3 | Second call with a target-agnostic offensiveness policy | Spearman with hate z .935; gain +.041 [−.003, +.088] < .05 (`DUAL_AXIS_KILLTEST_NOTE.md`) | "The judge reads the evidence, not the question" | closed |
| Speech-act second call | 08-10 | M3 | Ask whether the author endorses or only quotes/reports | Spearman with z .92–.98; residualized AUROC .358 / .498 (`SPEECHACT_SECOND_CALL_NOTE.md`) | Answer = hate scalar renamed; affirms endorsement for quoted hate | closed |
| Readout bottleneck | 08-09 | M1/M3 readout | Unsupervised PCA axes of final-token hidden states (layers 18/27/36) of the whole-video judge | supervised probe LOO .837 (information exists); best unsupervised .746 vs z .749 (EN), .617 vs .637 (HCS); layer-27 PC1 Spearman .992 with z (`READOUT_BOTTLENECK_NOTE.md`) | Dominant variance direction = the readout; the rest is noise for this distinction | closed for unsupervised PCA of judge states |
| Answer contrast | 08-10 | M1/M3 readout | Difference of states under forced Yes vs forced No answers | leading direction Spearman with z .983/.979/.949/.946 (`ANSWER_CONTRAST_PILOT_NOTE.md`) | Concentrates more variance on z (64 %) | closed |
| Spec displacement | 08-10 | M1/M3 readout | Hidden-state difference of the same video read under two policy specs | fails 3–4 of 5 clauses; HCS separation survives destroyed pairing (`SPEC_DISPLACEMENT_PILOT_NOTE.md`) | Not a spec interaction; strict arm's own state | closed |
| Illocution transfer | 08-10 | M1/M3 readout | Speech-act direction fitted on 700 authored text sentences applied to video-evidence states | text LOFO AUROC 1.000; video ZH .334 (inverted), HateMM .483 (`ILLOCUTION_TRANSFER_NOTE.md`) | Text-derived geometry does not transfer to video-evidence states | closed for text-derived directions |
| Entropy test-time adaptation | 08-09 | M3 adaptation | TENT on Yes/No answer distribution | macro-F1 .6460 vs .6562; median entropy 4e-5 nats; 82 % saturated (`ENTROPY_TTA_NOTE.md`) | No gradient | closed |
| Self-trained readout head | 08-09 | M3 | Linear head on hidden states trained on the judge's own extreme pseudo labels | macro-F1 .6960 vs bar .75; fixes 8 of 70 false positives (`SELFTRAINED_READOUT_NOTE.md`) | Trim at the edge, not a new reading | closed |
| Interleaved timeline | 08-09 | M1/M3 input order | Frame i interleaved with the words spoken over it; misaligned control | aligned +.033, misaligned +.038 on the binding stratum (`INTERLEAVED_TIMELINE_NOTE.md`) | Gain is not from pairing | closed |
| Text-region pixel budget | 08-08 | M3 input | Quarter of visual tokens to OCR text crops | .785→.789; avoid-text arm .793 (`TEXT_REGION_BUDGET_NOTE.md`) | No ceiling gain | closed |
| Duplex reading (literal vs pragmatic readers) | 08-05 | M3 | Divergence of two instructed readings | effort placebo reproduces D (+.0051/+.0027 vs .05 bar) (`KILLTEST_POSTMORTEM.md`) | "Instructions cannot suspend comprehension" | closed |
| Duplex readout probe | 08-05 | M3 readout | Label-free linear direction at the judgment token | .8520 vs raw z .8751 (`READOUT_POSTMORTEM.md`) | Redundant with z | closed |
| DAPT precondition | 08-10 | M1/M3 adaptation | Next-token familiarity as the cause of congested mid-band | AUC .5028 on HateMM (`DAPT_PRECONDITION_NOTE.md`) | Familiarity lever absent | closed before training |
| Speaker provenance extraction | 08-08 | M1 input | WavLM speaker windows + YuNet faces to attribute transcript to speakers | extractor failed the 20-video spot check (`PROVENANCE_AND_SUCCESSOR_PROBES_NOTE.md` §1) | Extractor failure, "mechanism remains untested" | superseded by candidate 44 (pyannote, October; §1.5) |
| R1 provenance certificate | 08-25 | M1 | Views: full / payload-only / no-payload / cyclic frame shift / nuisance, certificate of author commitment | residual certificate AUROC .540 / .600 (bar .70) (`R1_PROVENANCE_CERTIFICATE_NOTE.md`) | No stable signed author-commitment variable | closed |
| Temporal attribution of the whole-video verdict | 08-12 | M1 (localization) | grad×input and late-layer attention of the single verdict call mapped to segments | HCS segment macro AUC .5252 / .4819 vs token-density control .5182 (159 videos) (`TEMPORAL_ATTRIBUTION_PILOT_NOTE.md`) | Attribution = token density; attention dominated by sink + question; videos saturated (mean z 11.5) | closed for parameter-free attribution of a whole-video call |
| Sentinel logit-lens localization | 08-12 | M1 | Marker after every segment in one causal forward; read judgment direction at markers | .4994 (sentinel), .5186 (packed questions) on 326 HCS videos (`SENTINEL_LOCALIZATION_NOTE.md`) | Every in-context local probe returns the global verdict smeared; position drift ±.38 | closed (motivated isolation) |
| Isolated chunk diagnostic + masked parallel isolation | 08-18 | M1 | Each transcript chunk judged alone (rules prefix + chunk), packed with a block-diagonal mask | HateMM 212 videos / 2281 chunks: masked within .6197 vs full causal .5622; cross-video causal .850 vs masked .720 (`MASKED_PARALLEL_ISOLATION_NOTE.md`) | Cross-chunk attention contaminates; global verdict helps video level, hurts moment level | adopted → SPVL isolation |
| A01–A12 temporal-grounding transplants | 08-23 | M1 | Generic "hateful" query to existing grounding models and prompt transplants | 611 videos: A08 .521/.300/.539, A10 .520/.318/.615, A12 .573/.329/.580; A01v4 binwise timed multimodal windows retained (`refine-logs/EXPERIMENT_RESULTS_20260823_000000.md`) | Empty or constant curves | A10 kept as OMSL visual stream; rest closed |
| P1–P7 paradigm adaptations (canvas, coarse-to-fine, boundary posterior, set output, role experts, memory) | 08-26 | M1 | e.g. 16-cell visual canvas, controller zoom, set-valued output | 8 videos/dataset; canvas improves HateMM, hurts HCS (`refine-logs/EXPERIMENT_RESULTS_20260826_PARADIGM_ADAPT.md`) | Tiny cohorts; canvas often constant | closed |
| Poset T3AL successor | 08-26 | M1/M3-ish fusion | Transcript pairs with confidence gap ≥ .4 become order constraints on a CoCa visual curve | held-256 within .5766, vs direct transcript +.010 [−.011, +.032] (`idea-stage/IDEA_REPORT.md`) | Not significant over the transcript alone | closed (anchor only) |
| MELT → RAVEL relation parse | 08-27/28 | M1 | Model parses (speaker, hostile act, target, stance), then labels phases per cell | F1@.5 .205, event-merged .165 vs A10 .170 (DIRECTIONS C) | Slot hallucination chains; text-dominated | closed |
| TIDE / WITNESS / RCWL / HYPER | 08-28 | M1 | Local-only views, keep-if-positive with and without context; interval tribunals | no numbers / HYPER collapses to rank 1 (DIRECTIONS C) | — | closed |
| PACT / CCA / CASA / VASTA / CWA / NCL | 08-28 | M1/M3 fusion | Visual proposals give intervals, transcript only vetoes or authorizes | interval F1@.5 ≤ ~.29; PACT +.001 over CCA, 96.7 % fallback (DIRECTIONS C) | Edits too sparse; same-cohort selection | closed |
| LESS / PoS-LESS | 08-28 | M3 (+fusion) | Cross-view Dawid–Skene label model; video intercept + centred within residual | macro .6886/.4643/.6326; novelty 3.5–4.5/10 (LESS novelty response) | "Hierarchy algebraically vacuous"; projection changed 15/611 curves | closed |
| Scale-orthogonal transport PoS | 08-28 | M3 | Hard transcript-support projection owns the within residual; agreeing hard/soft correction owns a constant video-level propensity shift | idea novelty 6/10, evidenced 5/10, dev cohort only (`TRANSPORT_POS_CANDIDATE_V1.md`) | Retrospective development | not carried forward |
| TRACE per-video smoothing strength | 08-28 | M2 | Video chooses its own smoothing | within .6499; uniform smoothing equal (DIRECTIONS C) | Content-blind control equal | closed |
| AMBI / MOSAIC / COHERE | 08-28 | M1 | Model judges whether endpoints belong to the same event; prototype similarity | COHERE .6451 vs .6330, tie-corrected .6331; time-only ordering equal (DIRECTIONS C) | Content-blind control equal | closed |
| Coalition game / permutation family → OMSL v1–v6 | 08-29 | M3 + fusion | Whole-video logit intercept + Möbius decomposition of streams | cyclic maxT .677 vs .645 within; OMSL-v6 see 1a | Dropped for novelty score < 6, not for performance | superseded |

### 1.2 2026-09: 4 fps protocol, SPVL to r6

Unless a row says otherwise, scope = reader Qwen3-VL-8B-Instruct, bf16, greedy Yes/No log-odds; prefix = rules + 20
uniform timestamped frames + full timestamped Whisper transcript; 8 s fixed windows; protocol GT
(`data/gt_4fps`, HCS = any non-normal dimension).

| id / name | date | module | mechanism in one line | key numbers (source) | stated cause of failure or success | status |
|---|---|---|---|---|---|---|
| OMSL-v6 | 08-29, ablated 09-09 | M3 + window streams | Video intercept = whole-video Qwen3-VL-8B z pooled with a visual-occupancy logit; within order = Möbius decomposition of three rank streams (Vid-Group visual, per-ASR-chunk text log-odds with only that chunk's text, ImageBind audio) | .8507/.5781/.6494 and .6692/.6622/.5473; occupancy-only intercept pooled ROC .536/.497; z-only intercept = full (`experiments/20260829_omsl_v6/README.md` Ablations) | Pooled carried entirely by the whole-video z; occupancy, Möbius interactions, permutation count inert | superseded 09-10 (kept as comparator) |
| SPVL round 1: shared prefix + isolated window branches | 09-10 | M1 | One prefix (rules + 20 frames + full transcript); per 8 s window a branch that sees prefix + own question only (block mask / later KV-cache copy); joint branch, rules question | .8853/.6502/.6783 and .6758/.6204/.5806 (`experiments/20260910_spvl/README.md` §8) | Fixed windows +.089/+.052 vs ASR segments; frames +.028/+.011; transcript context +.053 HateMM, −.010 HCS | adopted (core of current M1) |
| SPVL M3 "verdict + extent" key | 09-10 | M3 | Video intercept = z_video + mean window z, from the same forward; four extent estimators tried (mean, median, logit of mean probability, top-half mean) | pooled PR +.036/+.030 over z_video alone; estimator spread ≤ .013 (§8 Reading); r2 HCS PR .622→.666 (§9) | Verdict says whether, not how much: top-10 HCS videos by verdict had 69 % positive frames vs 77 % for the old judge (§7 test read) | adopted (calibrated 09-26; raw key again since 2026-10-07) |
| Branch visibility (causal mask instead of isolation) | 09-10; DeHate 09-29 | M1 | Window branches see earlier branches' tokens | round-1: within −.014 HateMM, +.011 HCS (§8 row "causal mask"); DeHate sequential: within −.031 [−.060, −.001], pooled ROC −.015, 21 % of windows change sign (`experiments/20260928_infer/README.md` §11.1) | Reads drift; helps on no corpus | closed; isolation kept |
| Window unit and size | 09-10 | M1 | ASR segments instead of fixed 8 s; S = 4 s, 16 s; K = 8 frames | ASR within .5890/.5284; S4 .6573/.5714, S16 .6362/.5730; K8 .6608/.5759 (§8) | 8 s and 20 frames best on both corpora; ASR segments up to 146 s and leave silence unscored | closed |
| Old 2026-08 judge as intercept; Vid-Group as primary order; legacy chunk replica | 09-10 | M3 / M1 | Intercept from the 2026-08 judge (no transcript on HCS); Vid-Group visual order first; 2026-08 per-chunk text scorer | old-judge intercept pooled .8541/.5883, .6724/.6611; Vid-Group +.006/−.013; legacy chunk within .5588/.5246 (§8) | New verdict with transcript ranks HCS videos better (video ROC .804 vs .685) | closed |
| SPVL-r2: stance turn + evidence question + dual branches | 09-10 | M1 | Model's own whole-video Yes/No appended as an assistant turn; window question = "is this one of the segments where the violating content occurs"; separate visual and speech branch per window, max | .8919/.6831/.6976 and .7119/.6664/.6001; stance +.008/+.012; dual vs joint HCS +.028, HateMM −.002 (§9) | Combination passes; parts individually only on one corpus (stance HCS, dual HCS) | adopted (current M1 read) |
| Triple branch (joint + visual + speech) | 09-10 | M1 | Add a joint branch to the dual pair | .6931/.6003 within (§9) | No gain | closed |
| One frame per window (`data/frames_w8`) | 09-10; rechecked 09-11 (HVL fill), 09-27 (K7), 09-28 | M1 input | Window-centre frame added so every visual branch has its own frame | 5× cost; SPVL-r2: HateMM within −.009, HCS PR +.013 (§10); HVL fill .6871/.5981 vs .6926/.6007; K7 under r3: HateMM +.006 within, PR −.006; HCS +.016 / PR +.013 (`experiments/20260926_twolevel/README.md` §17.1); fitted on gold, visual-only within falls .718→.684 (`experiments/20260928_headroom/README.md` C) | Frameless windows not judged differently; HCS-only gain; silent HCS positives are offensive-not-hate content (label mismatch) | closed for both-corpora gate; HCS gain recorded as possible input change (input ≠ novelty, rule 5) |
| Seven/eight-MLLM family study | 09-10/11 | M1 (+M3) | Same pipeline on Qwen3-VL 2B/4B/32B, Qwen2.5-VL-7B, InternVL3.5-8B, LLaVA-OV-7B, Gemma-3-12B; arms full / winonly / noctx / noframes / asr / nostance / joint | Per-window-alone within .62–.64 / .49–.58 on every model; pipeline .646–.701 / .565–.604 (§11 table; `runs/20260910_spvl/mllm_table.md`) | Ceiling not Qwen-specific; transcript context used by Qwen3-VL and Gemma only | reference evidence |
| HVL E0: model cites where the hate is | 09-11 | M1 | Greedy generation TARGET / FORM / EVIDENCE(start–end: note) after the verdict; cited spans painted as scores | cited-span within .571/.506 vs per-window alone .628/.576 (`experiments/20260911_hvl/README.md` §7) | Model says who and how, not where | closed |
| HVL hypothesis-conditioned window reads | 09-11 | M1 | Hypothesis in context, then independent Yes/No per window; variants TARGET/FORM only, another video's hypothesis | own hyp .6314/.5597; TARGET/FORM .6504/.5676 (83 % windows "present" vs 62 %); permuted hyp .6952/.5460 vs base .6926/.6007 (§8) | Confirmation (88 % vs 44 % present inside vs outside own cited spans); hypothesis content not used | closed |
| HVL temporal state chain | 09-11 | M1 | Sequential start/continue/stop/none per window on one growing cache; binary sequential variant; shuffled-order control | .6590/.5356; binary .6613/.5573; shuffled .5640/.5115 (§8) | Persistence: 77–82 % "continue" | closed (same persistence works when imposed outside the model: TIL prior) |
| HVL revision (verdict revised after verification) | 09-11 | M3 | Summary of present windows fed back, whole-video question asked again → z_rev as intercept | Spearman(z_rev, z_video) .76–.84; Spearman(z_rev, #present) −.17 to −.32; intercept within noise HateMM, worse HCS (§8) | Restates the verdict and reads a long list of present windows as evidence against it | closed |
| HVL neighbour transcripts | 09-11 | M1 | Window branch also gets neighbouring windows' transcripts | .6753/.5946, HateMM −.017 (§8) | Local context worse than full transcript in prefix | closed |
| PWC pairwise window comparison (E0 kill test) | 09-12 | M1 | Two windows in one query ("which contains more endorsed attack; context-supplying does not count"), A/B/C, swap-symmetrised | sign accuracy .654/.570 vs .664/.565 for sign(z_i − z_j); generic wording .661/.550; joint-context absolute .685/.575; timestamps-only control .546/.511 (`experiments/20260912_pwc/README.md` §7b; `runs/20260912_pwc/e0b/`) | All read forms in the same band; comparison recovers the same information (H2: no discriminative signal, not H1: no common scale) | closed (re-closed by candidate 42 in October) |
| Context contrast | 09-12 | M1 | z(full context) − z(window alone) | .467/.494 vs .699/.602 (§7b) | Subtracts the useful signal | closed |
| Topic cross-tabulation (diagnostic) | 09-12 | M1 diagnostic | Window log-odds by GT × "window transcript mentions the hypothesis target" | mention +8.1/+7.0 log-odds, GT +4.3/+4.3 (unweighted); count-weighted ratio 1.15 / 1.47 (§7c; TAD §1) | Per-window read is largely topic detection; target string came from the model's own hypothesis (caveat) | finding; repeated on DeHate 09-27 |
| TAD round 1: subtract a topic read | 09-12 | M1 | Extra branch "does this window refer to a protected group"; r = a − β t, β per-video OLS clipped [0, 2] or fixed | OLS .6159/.5793; β = 1 .5749/.5764; topic alone .6567/.5467 vs act .6926/.6007 (`experiments/20260912_tad/README.md` §5b) | Topic is positively predictive of GT; removing it removes signal; monotone in β | closed |
| TAD round 2: five-way speech act | 09-12 | M1 | attacks / reports / quotes / condemns / unrelated, act_margin = log P(attacks) − logsumexp(others) | .6800/.5998; rank sum with a .6945/.6076 (§5c) | argmax attacks 79.6 %/56.0 %, carve-outs 1.9 %/9.2 %: collapses to the topic axis | closed |
| TAD §5d gold-fitted ceilings (diagnostic) | 09-12; audited 09-13 | M1 diagnostic | Logistic / boosted classifiers fitted on test GT over scalar reads, window shape, ImageBind audio, neighbours; over branch final hidden state; over prefix image-token states | window-level: unfitted act .7581/.6212; logreg on reads .7422/.6250; GBT + audio .7928/.5525; hidden 64 PCA .7470/.6078; prefix probe .6568/.5042 vs .7303/.6143 (§5d) | Stated: per-window feature family at ceiling. **09-13 audit** (`research-wiki/STATUS.md` 2026-09-22 section, last bullet): hidden-state probe stored the speech vector when present but was described as visual; speech-branch probe reaches HateMM +.010 (noise line); prefix probe limited by the causal mask, so "ordering not in the representation" is not supported | diagnostic; claims weakened by the audit |
| BND onset / offset reads | 09-12 | M1 | Per window "does the violating content BEGIN / END here"; cumulative presence from the two | window-level begin .6466/.5605; cumulative .6146/.5265; rank sum with act .6956/.5949 vs act .7581/.6212 (`experiments/20260912_bnd/README.md` §3) | Not a renamed act read (Spearman .644/.731) but weaker; argmax = first GT window 22.7 %/15.6 % | closed |
| SDL: self-distilled MIL LoRA | 09-12 | M1 adaptation (verdict as pseudo label) | LoRA rank 8 on language attention; pseudo label 1[z_video > 0]; max-pool MIL BCE, then hinge | r1 within .5026/.5091; r3 .5092/.5337 (`experiments/20260912_sdl/README.md` §5b–§5e) | r1 stance turn leaks the bag label; r2 no gradient (saturated); r3 "one window up, rest down" is false (median positive share .24/.47), curve flattened | closed |
| NGA: negative-only adaptation | 09-12 | M1 adaptation | Hinge only on pseudo-negative videos; anchor on pseudo-positives in r2/r3 | r2 .6812/.5996 vs frozen .6968/.6020 (`experiments/20260912_nga/README.md` §5b) | Global shift (r1) or pinned model (r2); only 79 pseudo-negatives (10 HCS) | closed (with SDL: whole adaptation family) |
| CVA counterfactual exclusion | 09-12 | M1 | s_i = z_video − z(video question while told to disregard interval i) | .8834/.6572/.6747 and .6724/.6215/.5779; window-level .7183/.5908 vs direct .7563/.6190; shuffled .5079/.5143 (`archive/experiments/20260912_cva/README.md` §9–§10) | Model honours the exclusion, but the clause costs ~4 log-odds regardless of interval (6× the interval's effect); video-constant cancellation is void under a within-video rank | archived |
| TIL shifted second grid + interval observation model | 09-22 | M1 measurement geometry (+M2) | Second 8 s grid offset 4 s; interval-support likelihood on 4 s cells | A4 .7321/.6203 vs A3 (average + prior) .7366/.6232 vs A1 (one grid + prior) .7486/.6247 (`experiments/20260922_til/README.md` §8) | Half-overlapping reads agree only .57/.63 (Spearman): a read is not a stable function of window content; no boundary information | closed |
| TIL §6 item 5: branch choice by modality-only verdicts; mean of branches | 09-22 | M1/M3 fusion | Select/weight branch by transcript-only / frames-only whole-video verdicts; mean instead of max | +.009 / −.010; mean −.003/−.005 (§6) | Not both corpora | closed then; reopened as the 09-30 modality gate |
| GLR generative likelihood ratio | 09-26 | M1 | Teacher-forced log p(window words | violating speaker) vs compliant-speaker mixture, per token; words not in context | window-level .49–.57 vs z_speech .6935/.6035; frame-level within −.12 to −.22 / −.05 to −.08 (`experiments/20260926_glr/README.md` §8) | Speech unpredictable (−5.6/−6.5 nats/token); condition moves ±.01 nats/token; signal only between videos (video AUC .61–.77) | closed |
| Quote matching of HVL cited evidence (proxy, not run as method) | 09-26 | M1 | Match model's quoted evidence words to window words | window-level .598/.552 vs z_speech .694/.604 (GLR §7) | Model cites the wrong evidence | closed |
| twolevel round 1: Bayes video posterior and product composition | 09-26 | M3 | P(V = 1 | verdict, all reads) by EM; frame = P(V) × P(h_t | V = 1); intervals | full pooled .8470/.5969 vs .8956/.6888 HateMM; Bayes V as intercept .8230/.5576, .7041/.6311 (`experiments/20260926_twolevel/README.md` §9) | EM treats each read as strong independent evidence; summed evidence overconfident, ranks videos worse than z_video + mean | closed |
| twolevel §10.2 item 6: other principled video levels | 09-26 | M3 | Reads counted as ICC-effective reads; two-component mixture over verdict + mean read per modality; each as intercept or product | best .8961/.6725, .7109/.6490 vs current .8956/.6888, .7136/.6671; every variant ≥ .013 pooled PR below on at least one corpus (§10.2 item 6) | Calibrated posteriors have small spread; frames of different videos interleave more | closed |
| twolevel §11: key scale and calibrated key | 09-26 | M3 | logit P(V = 1 | K) = aK + b from a label-free two-Gaussian mixture on K = z_video + mean window z; scale scan s | c_m2 .8970/.6959, .7170/.6706 vs raw .8955/.6888, .7137/.6664; test-best s = .125 .8993/.7010, .7217/.6749 (not used); expected-hate-fraction key .8632/.6180, .7272/.6722; posterior key .8960/.6735, .7114/.6490 (§11.1, §11.5) | Raw key overweighted against the rank term; calibration gives s ≈ .35 | adopted 09-26; calibration removed by the user 2026-10-07 (paper uses the raw key) |
| Product composition P(V | K) × P(h_t | V = 1) | 09-26, final 09-27 | M3 | Frame score = log P(V) + log P(h_t), intervals at ≥ .5 | r6 `final_full`: .8904/.7133 vs .8971/.6942 (ROC −.007, PR +.019); HCS .7391/.6779 vs .7168/.6711 (+.022/+.007) (§20.1) | HateMM ROC falls beyond the .005 floor | kept as interval output only; near miss on frame metrics |
| Viterbi interval decoding | 09-26 | M3/M2 output | MAP segmentation of each chain, gated by P(V | K) ≥ .5 | F1@.3 .324/.245 vs product .318/.316 (§13) | Merges HCS's short segments | closed |
| Reading components under the new time level | 09-26 | M1 ablation | stance / transcript / frames / dual removed under r2 and r3 | under r3: stance −.014/−.011, transcript −.039/+.001, frames −.060/−.109, dual +.009/−.074 (§14.4) | Under r3 transcript fails HCS and dual fails HateMM | reference |
| Decoder probability as frame score | 10-07 | M3 ablation | No key, no rank: OR-fused P(h_t | V = 1) directly | .8563/.6234 vs .8971/.6942; HCS .7339/.6662 vs .7168/.6711; DeHate .6721/.1361 vs .7011/.1582; among hateful videos only it ranks better (HateMM .694 vs .677, HCS .703 vs .666) (§23) | Key adds pooled ROC on HateMM (+.041) and DeHate; nothing on HCS | reference ablation |
| Raw vs calibrated key on eight MLLMs | 10-07 | M3 | Declared rule over 8 backbones | means of Δ < .001 both corpora; USEFUL only by condition (a) on HateMM PR, 3/8 models (§24) | Calibration hurts when fitted slope a > 1 (LLaVA, Qwen2.5 on HateMM, 2B on HCS) | calibration removed by user ruling 2026-10-07 |
| DVD: definition-conjunctive video verdict (concern K3) | 09-27 | M3 | Extra isolated video-level branches T (protected-group target) and E (endorsement); noisy-AND P(viol | K) P(T) P(E), each calibrated label-free | HCS ROC −.022, PR −.033; DeHate +.012/+.028; verdict + T: DeHate +.035/+.038, HCS strict-hate GT +.0195/+.011 (`experiments/20260927_dvd/README.md` §9) | HCS protocol GT counts non-hate offensive content (51/104 positive videos have no hateful segment); E never pulls weight | closed under protocol GT (user kept HCS protocol GT 2026-10-25, per charter) |
| Error analysis and closure of K4 (topic confound), K5 (short hate) | 09-27 | M1 diagnostic | Cross-corpus error table; K4 closed by pointing at the eight archived read-out directions + TAD ceiling | mention effect +.56/+.61/+.50 vs gold +.48/+.27/+.29 (standardized, incl. DeHate); 55–81 % of top false alarms in hateful videos are > 8 s from gold (`experiments/20260927_error_analysis/README.md`) | Content confusion, not boundary spill | closed as "documented limit of the per-window output" (scope: Yes/No outputs) |
| Headroom A: perfect / gold-fitted video level | 09-28 | M3 diagnostic | A1 perfect hateful/non-hateful split; A2 + true share order; A3 video score fitted on gold from all cached reads incl. DVD | A1 .9261/.7353, .7510/.6865; A2 .9791/.9343, .8923/.8790; A3 .8691/.6331 (video AUC .883 vs method .927), .7120/.6566 (`experiments/20260928_headroom/README.md` A) | Headroom exists (mostly in ordering hateful videos by share), but the current reads cannot reach it; DeHate fitted PR .217 vs .158 | diagnostic; "needs new information" for M3 |
| Headroom B/C: gold-fitted within combination | 09-28 | M1 diagnostic | Boosted / ridge on within-video read features + method outputs | +.015 [−.015, +.049] HateMM, −.016 HCS; visual-only .7100/.5410 vs speech-only .7571/.5791 (B, C) | No better within combination found (lower bound, small n) | diagnostic |
| Reader ensemble (8 MLLMs) and verdict-only ensemble | 09-28 | M1 + M3 | Normal-score average of all 8 models' reads (or only their verdicts) under r6 | all 8: within +.029 [+.001, +.061] HateMM, HCS PR +.023; ~10× cost; verdict-only 8: HCS −.100/−.070 (headroom "Is it worth trying") | Several readers carry information one lacks; verdict-only breaks the calibrated key; excluded by rule 3 / 14(e) | closed by rule (user rule change needed) |
| Codex proposal 2: layer-dependent context access in the window branch | 09-28 | M1 | Early layers see full prefix, last layers see only policy + own window | Premise check only, on input-removal reads: no mixture of reduced-context reads gains .01 (context_mix table) | Premise "late access to verdict swamps window evidence" not supported by input removal; the README states only a GPU pilot tests the actual computation | closed: Grounder (candidate 1, 2026-10-02) ran this computation with an `early` control (§1.4, §2 M1-f) |
| Codex proposal: dense student | 09-28 | M1–M3 | Small student distilled from method outputs | not run | Not training-free; nothing corrects the teacher's mention/attack confusion | withdrawn |
| Joint vs dual branch, three corpora | 09-29/30 | M1 | One joint window branch vs visual + speech | HateMM +.002, HCS −.028, DeHate +.027 (infer §11.1); joint read positive in 95–99 % of windows where speech is positive (§13) | The joint read is the speech read; dual pays only where the picture alone carries hate (HCS 19 % of hateful windows) | dual kept; not claimable (14g) |
| Per-video / per-corpus modality choice ceilings | 09-30 | M1/M3 fusion | Oracle choice of max vs joint per video | per-video oracle +.034/+.036/+.065 (DeHate); per-corpus gains only on DeHate (§14.3) | Label-free per-corpus signals (picture-only share, visual autocorrelation) order corpora correctly on 2 of 3 signals, unvalidated | assessment only |
| Modality gate by modality-only verdicts | 09-30 | M1 + M3 fusion | Each modality chain weighted by the calibrated verdict from a frames-only prefix / transcript-only prefix; P = 1 − Π(1 − γ_m P_m) | within .7513/.6325/.6638 vs best fixed .7607/.6373/.6700 (−.009/−.005/−.006); gate − OR +.0005/−.0048/+.0207 (infer §15.1) | Meets the user's "reach each corpus's best fixed design" rule; soft weighting only (gate predicts per-video best choice at ROC .40/.53/.51); fails 14g; ~2.5× reading cost | open (adoption = user decision) |
| DeHate leave-one-out of reading components | 09-29 | M1 ablation | noframes / noctx / nostance / asrwin reads on DeHate | transcript −.049 within, ROC −.078; ASR windows −.065; isolation −.031; frames −.003 (infer §12.1) | DeHate hate is in the speech | reference |
| Label-free baselines on our evaluator | 10-08 | comparator | ZS-CLIP, ImageBind audio, LAVAD, AV²A, SeViLA Localizer (per-frame yes-logit, QVHighlights-trained), UniTime, MULDE, CLAP, LaGoVAD, Qwen2.5-VL grounding | every within .47–.55 on both corpora; best pooled SeViLA .6271/.5762 (`experiments/20261008_baselines/README.md` §6) | — | reference |

### 1.3 Rows that touched only M2 (listed so they are not mistaken for M1 / M3 attempts)

Duration prior (TIL A1, 80 s), Gaussian smoothing control, twolevel rounds 1–6 (EM evidence, negative-binomial
durations, minimum two windows, per-video length averaging), slip phase (§15), carrier fusion (§10.2 item 7,
−.016/−.045), shared chain (−.007/−.043), stance-leak term (dropped), normal scores (+.011 when removed under r6),
reading-condition emitters (`experiments/20260928_infer` §1, inert, shuffled control equal), three nested levels
(§2, HateMM −.063), joint segmentation with shared boundaries (§3, +.008/−.0075/+.006), Codex coupling by speech
continuity (premise failed: continuity adds +.002/+.008 AUC to change prediction; coarse Whisper segments only).

### 1.4 2026-10-02 to 10-05: candidates 1–22 plus two analyses

Common baseline and how to read the numbers (this section):

- Unless stated, every candidate here is paired against "base_gridA" = `runs/20260926_glr/base_gridA`
  (Qwen3-VL-8B, repaired Whisper ASR loader, 20 cached frames, full timestamped transcript, policy, global
  Yes/No verdict with the model's own hard answer appended as the stance turn, isolated 8 s visual and speech
  window branches) decoded by the unchanged r6_bma (M2 + M3). Base numbers (pooled ROC / pooled PR / frame
  within; within on 84 HateMM / 99 HCS mixed videos):
  HateMM .897119 / .694235 / .750782 (raw max-window within .680011, raw visual .613908);
  HCS .716825 / .671072 / .637349 (raw max .610130, raw visual .546450).
  Every candidate below re-read base and reproduced all 333 globals and 13,939 branches exactly.
- "raw within" = within-video ROC of the raw window max(visual, speech) curve before r6; this is the number
  that says whether the window read itself got better.
- Gate used throughout: same main metric +.01 on BOTH corpora, no pooled loss > .005, no within loss > .01.
- GT for all of these is the protocol GT (`data/gt_4fps/{HateMM,HateClipSeg}.npz`), HCS = any non-normal
  category. DeHate was not run for any candidate in this section except verdict_analysis.
| id / name | date | module | mechanism (one line) | key numbers (HateMM ; HCS) | stated cause of failure or success | status |
|---|---|---|---|---|---|---|
| verdict_analysis (paired analysis, no new calls) | 10-02 | M1 (stance turn) + M3 interaction | Same reader with vs without the appended global question/answer turn, both through current r6; plus a no-GT control that adds one per-video constant (the mean with-minus-without shift) to the no-turn reads | with − without within: HMM −.0016 [−.026,+.017], HCS +.0059 [−.011,+.023], DeHate +.0100 [−.008,+.029]; shift-only control HCS within .6395 (≥ real turn .6371). Raw max per-video Spearman between conditions .93/.92/.91; false-Yes normal videos lifted +3 logits. `runs/20261002_verdict_analysis/{main_full,main_nostance,main_shift_only}/metrics.json`, `analysis/comparisons.json` | "at least on HCS, the positive result does not require the reader to produce better window rankings"; the turn changes the score distribution seen by r6's corpus fits; wrong-No can collapse a video (hate_video_279 .765 → .125); correct Yes can also hurt | analysis only (no method change) |
|   scope | | | old-ASR SPVL reader pair `runs/20260910_spvl/mllm/q3vl-8b/{full,nostance}` for HMM/HCS (NOT the base_gridA family); DeHate pair from 20260927/20260928 | | | |
| revisable_prior (shared random-effect EM) | 10-02 | M3 + M2 (video regime and global/window coupling) | No-stance reader; one per-video random effect b added to both the global read and all window reads, latent video regime V, joint EM with Gaussian-quadrature integration; output = regime log odds + centred rank of P(h=1 given V=1) | full model HMM .569 / .360 / .713, HCS .514 / .490 / .632 vs append+r6 .894/.691/.764, .716/.672/.637. Regime odds video ROC .637/.625 vs raw global read .933/.806; offset absorbs 80%/68% of global variance. `runs/20261002_revisable_prior/{r1_full_px,r1_independent,r1_no_global_px}/metrics.json`, `analysis/failure_diagnostics.json` | "the nuisance term absorbs discriminative video-level signal as well as possible context bias"; an all-off path with a high common offset can explain sustained high reads, so free revision destroys a correct global ranking | archived |
|   scope | | | old-ASR SPVL reader pair (333 videos); CPU only | | | |
| 1 Grounder (late-layer local access) | 10-02 | M1 | Last 9 of 36 layers of each window branch may attend only to its own window's frames/words + scaffold (no other media, no verdict turn); controls verdict_only, all_local (all layers), shifted (wrong window, same token count), early (first 9 layers) | late within .750171 (−.0006) ; .637249 (−.0001). all_local −.046 / −.049; late vs shifted +.0008 / +.0001 although support overlap < 1% and mean branch-logit change only .06. `runs/20261002_m1_grounder/r1_full_decoded/<arm>/metrics.json` | "Correct support in the last quarter does not have the predicted substantial effect"; removing broad context everywhere hurts | archived |
| 2 Selector (window-selective heads) | 10-02 | M1 | Per head/layer, if the last query token's mean per-token affinity to own-window media exceeds other-window media, block other-window media for that head | HMM .897708 / .691099 / .709021 (−.042, CI excludes 0); HCS .713008 / .667350 / .622332 (−.015). Raw within also falls .680 → .653, .610 → .580. `runs/20261002_m1_selector/r1_main_decoded/{base,select}` | "Attention affinity did not identify a restriction that improved localization" | archived |
| 3 Attributor (integrated gradients of the verdict) | 10-02/03 | M1 (window score from the verdict) | Window score = N × summed signed integrated-gradient attribution of z_video to scalar gates on cached media value vectors of that window (FP32 queries, 32–256 node quadrature) | HMM .874816 / .613965 / .621149 (−.130); HCS .657016 / .573880 / .567891 (−.069); raw within .560 / .538. Cost 58 / 32 GPU min vs 5 / 4. `runs/20261002_m1_attributor/r1_main_fp32_mem_decoded/` | Degradation "already visible before r6"; "does not rule out all attribution methods". (Aug-2026 grad×input and attention attribution had also tracked token density.) | archived |
| 4 Marginalizer (average over both verdicts) | 10-03 | M1 (stance turn) | Read every window under a forced "Yes" and a forced "No" verdict continuation; window score = logit of the mean of the two probabilities | not run | Rule-4 STOP: MARS (arXiv 2601.15115) and RAMF (arXiv 2512.02743) already reason under opposite hate / non-hate assumptions on the same media and combine them | archived at proposal (untested) |
| 5 Eraser (occlusion of a window) | 10-03 | M1 (window score from the verdict) | z = F(full video) − F(video with that window's frames, or that window's words, physically removed), full prefix re-encoded per removal | HMM .878392 / .618500 / .619622 (−.131); HCS .672767 / .615419 / .506248 (−.131); raw .556 / .510. 48 / 41 GPU min. `runs/20261003_m1_eraser/r1_main_decoded/` | "Necessity of a window to the global response has not yielded a better localization signal"; does not isolate redundancy vs positional disruption | archived |
| 6 Factorizer (block-causal prefix) | 10-03 | M1 | Re-encode the prefix with media tokens allowed to see only their own window's media + scaffold; then forced native answer and ordinary window questions with full access | HMM .894802 / .675287 / .749240 (−.002); HCS .704883 / .671429 / .567329 (−.070); raw −.028 / −.065. Explicit-causal plumbing control ≈ base. `runs/20261003_m1_factorizer/r1_main_decoded/{base,causal,factor}` | HCS ordering degraded before and after r6; removing cross-window mixing during encoding is not a fix | archived |
| 7 Contraster (DoLa depth contrast) | 10-03 | M1 (readout) | Window score = Yes/No margin of log p(final layer) − log p(premature layer), premature layer chosen per branch by max Jensen-Shannon among even layers 2–34 | within +.0001 / +.0002. All 13,939 branches selected layer 2. `runs/20261003_m1_contraster/r1_main_decoded/` | "dynamic layer selection has no realized per-window variation"; does not rule out other layer pools | archived |
| 8 Visual contrast (VCD) | 10-03 | M1 (visual branch) | R1: re-encode the whole prefix with noised images (step 500), forced native answer, visual margin 2a − b. R2: noise only the queried window's frames | R1 HMM .892394 / .680132 / .770392 (+.020, PR −.014); HCS .724669 / .672273 / .648014 (+.011). Cached controls: within-video shuffled noisy reads HMM .775 (better than matched), HCS .634; matched subtraction supported only on HCS. R2 HMM .7537, HCS .6307; raw visual −.070 / −.016; 4.8× native cost. `runs/20261003_m1_visual_contrast/{r1_main,r1_controls,r2_main}_decoded/` | HMM gain is not window-matched (hate_video_279 and 329 with unchanged raw order give most of it; decoder interaction); local noise lost the gain | archived |
| 9 Allocator (Kernel SHAP over windows) | 10-03 | M1 (window score from the verdict) | Each window's frames + words form one player; coalitions physically re-encoded; window score = Shapley share of the global margin | not run | Rule-4 STOP: SHAP already used in hateful video detection | archived at proposal (untested) |
| 10 Integrator (FutureMask on the prefix) | 10-03 | M1 | Image-token rows of the prefix may attend to later tokens (R1 all later; R2 later text only; R3 = R2 visual + native speech, CPU composition; R4 later ASR of the frame's own window only) | R1 HMM .891985 / .681604 / .766819 (PR −.013), HCS .709646 / .670246 / .636054. R2 HMM .7744 (+.024), HCS .6401, HCS ROC −.008 PR −.007. R3 HMM .8951 / .6935 / .7743, HCS .7081 / .6642 / .6421. R4 HMM .7700, HCS .6359. `runs/20261003_m1_integrator/r{1..4}_main_decoded/` | HMM gains come mostly from two videos with unchanged raw order (hate_video_279, 329: +.0148 of the mean) — decoder interaction, not new timing evidence; HCS pooled loss persists | archived (initial + 3 revisions) |
| 11 Amplifier (PAI) | 10-03 | M1 (visual branch) | Last query token's attention logits to all image keys a += .5·abs(a) in layers 2–35, plus per-token contrast 1.1·amplified − .1·(same question on a no-frame prefix) | HMM .893032 / .685114 / .756861 (+.006, PR −.009); HCS .712161 / .669913 / .628277 (−.009); raw visual −.015 / +.005. `runs/20261003_m1_amplifier/r1_main_decoded/` | "More image attention by construction is not evidence that it attended to useful evidence" | archived |
| 12 Recycler (VAR sink redistribution) | 10-03 | M1 (visual branch) | Detect attention-sink tokens by activation; in heads with enough image mass, move .6 of sink attention onto non-sink image tokens | within +.0036 / −.0009; raw visual −.0025 / −.0028. `runs/20261003_m1_recycler/r1_main_decoded/` | activation established, no useful localization | archived |
| 13 Reinforcer (VISTA steering + SLA) | 10-03 | M1 (visual branch) | Add .17 × (native − no-image) last-token residual direction at every block, norm-preserving; final logits .7 final + .3 mean of layers 30–34 | HMM .891214 / .661409 / .750129 (PR −.033); HCS .722658 / .670072 / .636515; raw max −.035 / −.018. `runs/20261003_m1_reinforcer/r1_main_decoded/` | image-dependent update verified, no localization gain | archived |
| 14 Projector (ACG) | 10-03 | M1 (visual branch) | Per head, last row: O + 1.4 × (component of O − U orthogonal to U), U = same attention with image keys masked | within +.0001 / +.0005. `runs/20261003_m1_projector/r1_main_decoded/` | changes reads, no useful localization | archived |
| 15 Highlighter (VGA) | 10-03 | M1 (visual branch) | In layers 4–17, add .2·γ·(salience-weighted value of in-window image tokens) to the last-row attention output; salience = truncated vocabulary entropy of each image token | HMM .897078 / .694292 / .754667 (+.004); HCS .716890 / .671082 / .637897 (+.0005); raw max slightly down. `runs/20261003_m1_highlighter/r1_main_decoded/` | no localization mechanism | archived |
| 16 Stabilizer (PAS RoPE phases) | 10-03 | M1 (prefix encoding) | Extra temporal rotary phase (0 / .5 by head parity) on image query rows during prefix encoding; global and branches re-read on that prefix | within +.0006 / +.0012; 1 HMM verdict flip. `runs/20261003_m1_stabilizer/r1_main_decoded/` | no effect | archived |
| 17 Preserver (MAD-RAG) | 10-03 | M1 (visual branch) | Read the visual question on a frames + policy-only reference prefix (no ASR, no verdict) and mix .5·O + .5·O_ref attention outputs at all 36 layers into the full-context read | HMM .895610 / .689651 / .752658 (+.002); HCS .715189 / .672282 / .629401 (−.008); raw visual .614 → .547, .546 → .513; raw max CIs exclude 0 (negative) | "preserving this reference attention output did not preserve useful visual localization" | archived |
| 18 Explorer (feedback frame acquisition) | 10-03/04 | M1 (visual input) | For each visual window, decode up to 4 new in-window frames (chosen by attention prior × temporal distance, entropy gate .1/.3), append them after the verdict-conditioned cache with real timestamps, re-ask the visual question. R2 entropy-only; R3 always 2 frames; R4 expanded reads on the observational prefix without the verdict dialogue | R1 HMM .8976 / .6895 / .7548 (+.004), HCS .7319 / .6840 / .6584 (all three +.01). R3 HMM within +.0069, PR −.0053; HCS .7351 / .6857 / .6632 (+.026). R4 HMM .8978 / .6942 / .7599 (+.009), HCS .7225 / .6765 / .6526 (+.015). HMM raw visual within falls every round (.614 → .587–.601); HCS rises (.546 → .571–.589). 2.7–3.7× native. `runs/20261003_m1_explorer/{r1,r3,r4}_main_decoded/explore/metrics.json` | HMM final gain "is not new within-window evidence" (one single-window video carries it); new frames repeat persistent branding / static hateful imagery in GT-negative tails (HMM 45, 151, 304) | archived (initial + 3 revisions); user accepted R1 as positive progress |
| 19 Latents (visual latent optimization) | 10-04 | M1 (visual branch) | Insert 4 continuous vectors after the visual question; 5 Adam steps of contrastive warm-up toward high-attention patches + 15 NES steps maximizing top-20 entropy progression; read Yes/No from the selected state | HMM .894437 / .691121 / .727686 (−.023); HCS .685951 / .652592 / .566666 (−.071); raw visual −.134 / −.040. 6.3× native. `runs/20261004_m1_latents/r1_full_main_decoded/optimized/` | "optimization active does not mean effective localization"; paper-equation adaptation, official contextual-backbone variant untested | archived |
| 20 Acoustic (Whisper character alignment as attention prior) | 10-04 | M1 (speech branch, input timing) | Whisper-large-v3 character teacher forcing + head filter → forward-backward monotone path posterior → per-word window support; speech body = words whose most likely window is this one; every layer adds log(support) to attention on transcript words | HMM .893585 / .685160 / .758773 (+.008, PR −.009); HCS .716933 / .669970 / .645755 (+.008). Raw shared-speech within +.030 / +.013. 2.8× native. `runs/20261004_m1_acoustic/r1_soft_main_decoded/optimized/` | "positive dual-corpus within trend"; no metric +.01; hard / proportional / unweighted / shifted controls never run | archived |
| 21 Tree (VideoTree adaptation) | 10-04/05 | M1 (+ global read in R1) | 1 fps Qwen image features → KMeans tree with Qwen captions and Qwen relevance digits; R1 puts all node captions into a new global prefix and adds up to 2 in-window leaf frames + ancestor captions to each visual read; R2 native anchor/speech, tree only in the visual branch; R3 tree text as a shared factual turn after the verdict; R4 local packet to both branches | R1 HMM .895411 / .679329 / .779693 (+.029, PR −.015), HCS .729941 / .689819 / .650289 (+.013, ROC +.013, PR +.019). R2 HMM .897530 / .690654 / .772521 (+.022), HCS .729631 / .682939 / .642520 (ROC +.013, PR +.012). R3 HMM .7612, HCS .6338. R4 HMM .765801 (PR −.019), HCS .635046. HMM raw max within negative in every round (R4 CI wholly negative). 15× native. `runs/20261004_m1_tree/r{1..4}_full_main_decoded/optimized/` | HMM final gains without raw improvement; shared caption context changes speech reads and raises negatives; "hierarchy context can come from distant visually similar frames" | archived (initial + 3 revisions) |
| 22 Lattice (5-best ASR of the 8 s audio crop) | 10-04/05 | M1 (speech branch, new input) | Re-recognize each window's own 8 s audio with Whisper beam 5; R1 word-confusion slots with structured attention (mass-weighted keys, mutually exclusive alternatives, longest-path positions); R2 complete hypotheses as exclusive paths; R3 read only one end-of-path token per hypothesis | R1 HMM .896571 / .677466 / .799002 (+.048, CI [.018,.084]; PR −.017), HCS .714415 / .667120 / .640013 (+.003). R2 HMM .896267 / .677443 / .802410 (+.052, CI [.020,.088]; PR −.017), HCS .712939 / .667846 / .639245. R3 HMM .7501, HCS .6015 (raw max CIs negative). 7.7× native (60 min ASR). `runs/20261004_m1_lattice/r{1,2,3}_full_main_decoded/optimized/` | new local transcripts "recover content missing in native repetition/music crops", but recovered profanity and quoted threats in GT-negative windows get high speech scores (e.g. non_hate_video_188, hate_video_114, HCS lyric profanity); onebest / flat controls never run, so the gain cannot be split between "new transcript" and "lattice structure" | archived (initial + 2 revisions) |

Scope sub-lines (apply to candidates 1–22 unless noted):
- Reader: frozen Qwen3-VL-8B-Instruct BF16 (Attributor FP32 queries). Only Lattice/Acoustic also use
  Whisper-large-v3 (input side). No other reader was tested on any of these mechanisms.
- Inputs: base_gridA family (repaired ASR loader, 20 frames at nominal seek times with 1,030 / 1,233 frameless
  HMM / HCS windows, full transcript, policy, own hard stance), 8 s windows, 333 test videos (215 / 118).
- Visual-branch-only interventions (Amplifier, Recycler, Reinforcer, Projector, Highlighter, Preserver, Latents,
  visual contrast, Explorer): speech and global stayed exactly native, so these test only the visual read.
- All mechanism constants were transferred from the source papers without a scan (rule 7); "closed" means
  closed at those constants, not over every setting.
- No component controls were ever run for any candidate except visual contrast R1 (cheap cached controls) and
  Grounder (all arms in one run), because the main gate failed first.

### 1.5 2026-10-05 to 10-17: candidates 23–46 and the within diagnosis

Reference (all rows compare against this unless stated): r6_bma, `runs/20260926_twolevel/r6_bma/metrics.json`,
HateMM pooled ROC / pooled PR / within .8971 / .6942 / .7508 (84 videos), HateClipSeg .7168 / .6711 / .6373 (99 videos).
"Window-level within" below = per-video AUC over 8 s windows (positive = GT share >= .5, clean negative = 0), analysis-script
level, only comparable within the same window set. r6 speech read at that level: .726 / .606 (candidates 45, 46).
Deltas are "HateMM ROC/PR/within ; HateClipSeg ROC/PR/within" relative to r6_bma.

Common to every row unless the scope line says otherwise: reader frozen Qwen3-VL-8B-Instruct; shared prefix = rules + 20
native frames (`data/frames_k20`, 18-20 actual) + full Whisper-large-v3 transcript with segment timestamps + video verdict
+ own hard Yes/No stance; per-8 s-window isolated visual and speech Yes/No branches; window text = proportional slice of
Whisper segments (`src/video_inputs.window_text`); fusion max(V, S); fixed r6 time level; HCS protocol GT (any non-normal
dimension positive). DeHate was not run for any row in this section.

| id / name | date | module | mechanism in one line | key numbers (r6-relative unless stated; source) | stated cause of failure or success | status |
|---|---|---|---|---|---|---|
| C23 Program (ViperGPT-style executable temporal evidence program), R1 interface B + R2-R4 | 10-05 | M1 | Same Qwen plans a typed program per window (span/frame/local/context/scope/action/join/emit), an interpreter executes it on real frames and ASR spans, fresh factual sub-reads (speaker role, quote mode, actor/action/target), the executed record is appended before the unchanged V/S question; R4 instead masks the visual query's attention to only program-selected prefix image keys in all 36 layers | R1: +.0001/-.0030/+.0190 ; +.0026/+.0013/+.0022. R2 within +.0154/+.0036; R3 +.0131/-.0023; R4 -.0062/-.0044 (raw visual within -.041/-.011). `runs/20261005_m1_program/r{1..4}_handles_full_main_decoded/optimized/metrics.json`. Cost about 229 min / 333 videos, 25x native | Interface A: model outputs violated the schema, 0 module calls. B: scope speaker/mode resolved in only 1 HMM / 3 HCS windows, context and join never used; injected everyday action descriptions pushed true positive windows down (R3 error analysis); 7/84 HMM videos with unchanged raw order carried .0126 of the R1 within gain (decoder effect). R4 key-access restriction lost visual order | archived (item 26, 3/3 revisions used) |
| C24 QuoteGraph (quotation / reference graph + asymmetric attention) | 10-05 | M1 (speech) | Same Qwen extracts exact-span discourse relations (mention, quote, attribution, rejection ...) per 8-window chunk; graph traversal (<= 2 hops) picks up to 8 remote context spans; one speech forward where context tokens can reach the local body but not the question (additive -inf mask in all 36 layers) | R1: HMM within +.0110, PR -.0116; HCS -.0078/-.0143/-.0077. R2 (native question when no context): +.0009/+.0029/-.0091 ; -.0005/-.0001/+.0009. `runs/20261005_m1_quote_graph/r{1,2}_full_main_decoded/optimized/metrics.json`. 83 min, 9x | Graph context existed in only 178/3768 and 198/3591 windows; the R1 HMM gain sat in videos with NO graph context (+.0157) and came from the new scaffold wrapper text shifting S up by about 2 logits on negatives; graph videos lost (-.017) | archived (item 23) |
| C25 Provenance (persistent entity and discourse graph, Semantic-Event-Graph style) | 10-05/06 | M1 | Per window, a constrained neutral ledger (entities, actions, quotes, speaker) from 2 real in-window frames + local ASR; per 8-window block, same_entity / quotes / responds_to / retracts / speaker_change links; graph traversal retrieves up to 4 remote windows' frames + ASR; fresh full multimodal V and S with the packet | B: -.0017/-.0056/+.0040 ; +.0033/-.0008/+.0187. Raw S within HMM -.034 (CI negative), raw max -.070. `runs/20261005_m1_provenance/r1_handles_full_main_decoded/optimized/metrics.json`. 23.2 GPU hours, 151x native | Interface A: all 158 ledgers schema-rejected. B: remote-speech context lowered direct positive S (examples hate_video_347/337) and raised benign speech (HCS bit_vON...) | stopped 2026-10-07 at user request; not archived; 0/3 revisions used |
| C26 Interval witness (binary temporal tree, witness composition, disagreement-triggered re-read) | 10-05/06 | M1 | Per window a fresh leaf observation (present/absent/UNKNOWN + witnesses) from 2 local frames; W-1 text-only parent calls; deterministic existential composition; one repair re-read per disagreeing leaf; R1 re-asks G and all V/S on the reconciled tree; R2 only V reads the leaf-to-root path | R1: -.0198/-.0483/+.0164 ; -.0253/-.0153/-.0086 (591 min). R2: -.0048/-.0048/+.0029 ; +.0084/+.0093/+.0097 (341 min, 34x). `runs/20261005_m1_interval_witness/r{1,2}_full_main_decoded/optimized/metrics.json` | Repair hardly exercised (4/7 repairs in 333 videos); R1 global re-read flipped HCS video signs (25 wrong vs 6 right); no +.01 in R2 | archived (item 30) |
| C27 Verification (factored Chain-of-Verification before measurement) | 10-05 | M1 | Draft four neutral fields (act/actor/target/quotation_owner) per window, plan 1-2 questions, independent fresh re-observation per question, typed replacement, then ordinary V/S with compiled facts | Fixed 5 only: 96/62 verification calls, literal changes 83/60. No full run (Slurm 193 cancelled before start 2026-10-07). `runs/20261005_m1_verification/r1_full_smoke_analysis/plumbing_summary.json` | Never measured | not run (prototype + reviews PASS) |
| C28 TTF (anchor-identity temporal token fusion) | 10-05 | M1 (visual) | Compress prefix image tokens: tokens with cosine > .70 to the anchor frame's 3x3 neighbourhood replaced by the anchor token; anchor-first reordering with original mRoPE triples; native G/S kept, new V | +within .0082/.0063, others in noise; raw V within -.022/+.007. `runs/20261005_m1_ttf/r1_full_main_decoded/optimized/metrics.json`. 10.5 min, 1.16x | No +.01; final micro-gain not a raw localisation gain | archived (item 24) |
| C29 RoTE (duration-normalised interval rotary binding of ASR keys) | 10-05 | M1 (speech) | In every one of 36 layers, replace the temporal rotary channels of ASR-token keys by an interval (sinc-weighted) encoding of their Whisper segment [start,end]; each speech query at the window centre; implemented as additive attention bias | -.0025/-.0279/-.0127 ; -.0232/-.0120/-.0467; raw max within .633/.557 vs .680/.610. `runs/20261005_m1_rote/r1_full_main_decoded/optimized/metrics.json`. 1.09x | Losses on every metric, raw temporal order worse | archived (item 25) |
| C30 OTT (importance-mass temporal optimal transport of visual tokens) | 10-05 | M1 (visual) | Vision last-layer attention saliency, coverage-greedy token keep, Sinkhorn transport between adjacent frames, chain merge of tokens to .50 retention; new V on compressed prefix | within +.0058/-.0027, rest in noise. `runs/20261005_m1_ott/r1_full_main_decoded/optimized/metrics.json`. 10.6 min, 1.17x | No +.01 | archived (item 27) |
| C31 Spatial search (V*-style target-conditioned region search + crop memory) | 10-05/06 | M1 (visual) | 2 real in-window frames; Qwen proposes one target, searches up to 2 nodes of a 4-way region split, real crops; V reads original frames + crops next to the question | +.0018/+.0017/+.0070 ; +.0031/+.0092/+.0055 (all six positive). `runs/20261005_m1_spatial_search/r1_full_main_decoded/optimized/metrics.json`. 143.6 min, 15.7x | No single +.01 | archived (item 29) |
| C32 Text tracking (pixel-tracked on-screen text occurrences) | 10-06 | M1 (visual) | OCR by Qwen on 2 local frames, PyrLK forward-backward tracking at 4 fps, mismatch-triggered re-observation, window lookup of supported text crops appended to V | Fixed 5 only: 13/14 raw TEXT outputs hit caps; guard FAIL. `runs/20261005_m1_text_tracking/r1_full_smoke_analysis/`, `docs/reviews/20261006_m1_text_tracking_gpu_interface_diagnosis.md` | Generation interface (word/token caps), not an idea verdict | not run (interface failure) |
| C33 MERIT (four-key multi-key retrieval + neighbour filter) | 10-06 | M1 | Per window, caption + event/dialogue/object/summary keys from 2 local frames + ASR; same-Qwen last-layer mean-pooled text embeddings; MaxSim top-2 anchors +-1 neighbours, ID-only filter, up to 4 remote windows' frames/ASR into V and S; R2 keeps native S | R1: +.0016/-.0096/+.0178 ; +.0029/+.0022/+.0076. R2: +.0015/-.0027/+.0064 ; +.0042/+.0072/+.0158. raw V within R1 -.021/+.019. `runs/20261006_m1_merit/r{1,2}_full_main_decoded/optimized/metrics.json`. 304 min, 30x | Remote ASR in S lowers HMM positive S (hate_video_348/327); LOCAL-only and REMOTE changes both contribute errors | stopped 2026-10-07; not archived; 1/3 revisions |
| C34 Ordered slots (Q2E-style prequel/current/sequel event slots) | 10-06 | M1 | Captions of 2 local frames, generated prequel/current/sequel slot texts per 8-window block, embedding match picks earlier/later real windows; V reads LOCAL + prequel + sequel frames (+ASR in R1/R3); R2/R3 keep native S | R1 B: -.0027/-.0195/+.0072 ; +.0091/+.0069/+.0166. R2: +.0009/+.0025/+.0174 ; +.0055/+.0064/+.0048 (all six positive). R3: +.0005/+.0005/+.0110 ; +.0046/+.0057/-.0012. `runs/20261006_m1_ordered_slots/r{1,2,3}_full_main_B_decoded/optimized/metrics.json` | Interface A captions all capped. R3 diagnostic: for all 3071 REMOTE windows V equals R1; all V changes came from LOCAL-only fallback windows, i.e. the in-window frames, not the slots | stopped 2026-10-07; not archived; 2/3 revisions |
| C35 VTimeCoT (visual timeline tools: progress bar, highlight, cut) | 10-06/07 | M1 (visual) | 1 fps real frames; Qwen query + clip relevance retrieval; up to 3 model-chosen tool actions drawing a progress bar, highlighting and cutting real clips; V reads up to 22 tool-state images | Interface C: +.0002/-.0018/+.0036 ; +.0030/+.0028/-.0021. `runs/20261006_m1_vtimecot/r1_full_main_C_decoded/optimized/metrics.json`. 251 min | All inside noise; A/B interface failures (field caps) | archived (item 31) |
| C36 VideoEvent (query-relevant events + representative background) | 10-06 | M1 | Captions of 2 local frames; one relevance call scores +-4 neighbouring captions; smoothed above-mean contiguous event; V gets current + up to 4 event windows' frames + background frame; S gets remote event ASR | -.0013/-.0148/-.0009 ; +.0053/-.0005/-.0038. `runs/20261006_m1_videoevent/r1_full_main_decoded/optimized/metrics.json`. 121 min, 13x | HMM PR loss beyond noise, no +.01 | archived (item 28) |
| C37 ReKV (layer-wise retrieval of source-bound visual KV) + controls + HL0 | 10-06/07 | M1 (visual) | 0.5 fps real frames encoded chronologically into KV blocks (4-frame direct history); each layer retrieves top-4 remote blocks by pooled question Q vs block key mean; window's own 0.5 fps frames always included (LOCAL) | R1: -.0006/-.0013/+.0137 ; +.0167/+.0136/+.0159 (performance gate PASS). Controls (`controls_main_decoded/<arm>`): no remote (L0), fixed layers, nearest, wrong time, no history: none removes >= .01 in both corpora. HL0 (LOCAL only, no history, no remote): +.0022/-.0034/+.0117 ; +.0146/+.0140/+.0206. `runs/20261006_m1_rekv/{r1_full_main,history_local_main}_decoded/` | Independent claim review: retrieval, per-layer selection, history and binding claims removed; what gains is the dense in-window frames input | stopped (performance PASS, mechanism not supported); not archived; KV proofs (103 GB) deleted |
| C38 TextTiling (lexical discourse partition + source-scoped speech) | 10-06 | M1 (speech) | New greedy Whisper with DTW word times; TextTiling partition; LOCAL = words whose midpoint lies in the window; same-Qwen scope parser picks LOCAL act spans + needed CONTEXT spans; compiled packet replaces S | R1: +.0007/-.0127/+.0169 ; +.0057/+.0043/+.0103 (within >= .01 both; HMM PR loss beyond noise). R2 (packet only when compiled): -.0002/-.0143/+.0116 ; +.0045/+.0015/+.0020. raw S within R1 -.007/+.008. `runs/20261006_m1_texttiling/r{1,2}_full_main_decoded/optimized/metrics.json` | Typed packets and fallback both weaken positive S on HMM; within gain not visible in raw S | stopped 2026-10-07; not archived; 1/3 revisions |
| C39 StreamingTOM (change-preserving quantized media memory) + controls + decomposition + placement controls | 10-07/10 | M1 (visual) | 0.5 fps frames; static/dynamic token split (adjacent cosine > .9), clustered static + saliency dynamic tokens to 50/frame, uint4 KV memory, per-layer question retrieval of 4 remote frames; LOCAL in-window frames uncompressed next to the question | R1: +.0016/+.0007/+.0128 ; +.0189/+.0162/+.0224 (gate PASS). no_remote: +.0032/+.0010/+.0148 ; +.0216/+.0201/+.0199 (= whole gain). E local_clean (r6 + LOCAL frames + time labels): +.0007/-.0037/+.0136 ; +.0205/+.0195/+.0296. Placement: adjacent_native (prefix in-window frames re-shown next to question, no new image) -.0001/-.0004/+.0224 ; +.0084/+.0061/+.0128; prefix_local (0.5 fps frames inserted into prefix) +.0013/+.0020/+.0132 ; +.0099/+.0101/+.0145; adjacent_local +.0006/-.0056/+.0182 ; +.0215/+.0182/+.0264. `runs/20261007_m1_streamingtom/{r1_full_main,controls_*,local_controls_main,placement_controls_main}_decoded/<arm>/metrics.json` | Remote memory, retrieval and compression each within noise of removal; gain = window's own frames. Two separate causes, each sufficient: placement next to the question and more frames per window; they stack on HCS, not on HMM | stopped; components not supported; input change, adoption is the user's decision |
| StreamingTOM selection and combination simulations (CPU) | 10-10/11 | M1 | Take the extra read only in windows picked by label-free rules (confidence, in-video rank, speech disagreement, coverage), or combine the five stored reads (mean, median, agreement of two/three reads, neighbour direction) | Best rules inside six size-matched random bands (random halves span .008-.018 within). Label oracle +.053/+.048 (a), +.067/+.090 (E). Mean of five reads +.0215/+.0212, not above best single read. `runs/20261007_m1_streamingtom/{selection_analysis,combination_analysis}/summary.json` | Pre-read signals carry no information about whether the extra read helps; the reads share the model's belief, so averaging removes nothing; the oracle injects label information and is not reachable headroom | closed (analysis) |
| C40 MuKV (cross-granularity coherent KV retrieval) | 10-07 | M1 (visual) | Segment/frame/quadrant memories, attention + FFT compression, cross-grain coherence reranking | Not run; source frames extracted (`data/temporal_multigrain_kv/`) and CPU prototype only. `experiments/20261007_m1_mukv/README.md` | Paused by user 2026-10-07 | not run |
| C41 Grounded adjacent acceptance; revision 1 probe-before-frames | 10-10/11 | M1 (visual) | Main: after the adjacent_native read, a text probe asks which shown frame violates; accept a rise only if a frame is cited, a fall only if none. Rev 1: after the native read, probe A "which prefix frames are inside this window", probe B "which frame did you base that on"; flagged windows take the stored extra read (a, l or E) | Main: grounded vs accept_all within -.0015/-.0014; vs random acceptance within +.0065/-.0025 and +.0061/+.0062. Rev 1: six selections, none above all six random controls (primary -.0058/-.0084). Probe A correct in only 36 %/44 % of windows that contain a prefix frame. `runs/20261010_m1_grounded_adjacent/{analysis,probe_analysis}/summary.json` | The model's answers about frame positions and about its own basis carry no window-level information about whether the extra read helps; rule B just tracks the sign of the native read | archived (item 32) |
| C42 Pairwise evidence comparison with both windows' frames shown | 10-13 | M1 (visual) | Ring graph (distances 1, 2) per video, both orders asked "which window more clearly shows visual violating content? A/B/C", antisymmetric log-odds, Laplacian least squares u; cmp_order = mean(l)+u, cmp_sum = l+u | cmp_order - full_l within -.0099/-.0033; cmp_sum - full_l +.0017/-.0068; edge accuracy .630/.554 vs pointwise .586/.575; gap-1 pairs at chance (accuracy .48/.42, swap agreement .42/.51). `runs/20261013_m1_pairwise_evidence/analysis/summary.json`, `error_analysis/summary.json`. About 100 min, 4-5 reads per window | Relative judgements carry the same information as absolute ones with the same frames; neighbouring windows cannot be told apart, only distant pairs | archived (item 33; pairwise family closed) |
| C43 Audio-bound speech evidence (Qwen3-Omni-30B-A3B hears the window clip) + vl_aligned follow-up | 10-14 | M1 (speech) | Omni Thinker reads per window: visual_local, the window's 16 kHz audio clip next to the question, proportional text, CTC-aligned word slice, deranged clip, neighbour clip. Follow-up: aligned word slice (MMS_FA) under the r6 Qwen3-VL reader | audio arm +.0072/+.0183/+.0599 ; -.0249/-.0089/+.0320 (HCS pooled loss -> gate FAIL). audio - audio_swap within +.152/+.088 (swap falls far below r6: .658/.582). audio - text_aligned +.022/+.003. vl_r6_aligned (r6 visual + aligned slice): +.002/+.001/+.017 ; -.002/+.008/+.029. vl_text_aligned (adjacent_local + aligned): +.003/+.003/+.029 ; +.018/+.023/+.049. Window AUC HMM speech_audio .787 vs aligned .687 vs prop .656. `runs/20261014_m1_audio_bound/{main_decoded,vl_aligned_decoded}/<arm>/metrics.json`, `main_analysis/error_analysis.txt` | Binding passes; hearing beyond aligned text only on HateMM (ASR-silent positives: audio AUC .848 vs visual fallback .633); Omni reader ranks HCS videos worse (-.049 ROC with identical inputs). Aligned slice is a reader-independent input gain | not promoted; aligned slice = input change for user decision; hearing on silent windows left open |
| C44 Speaker-bound reading (pre-check only) | 10-14 | M1 (speech) | pyannote 3.1 diarization; would bind stance to speaker turns | P1 multi-speaker share HateMM .286 (gate .30); P2 speaker-identity GT ceiling - r6 +.036/+.077 (gate .05); false highs not more from other speakers (.26 vs .30); pooling reads by speaker HCS -.063. `runs/20261014_m1_speaker_bound/precheck/summary.json` | HateMM mostly single-speaker; per-window differences within a speaker are useful information | archived at pre-check (item 34) |
| C45 Generation-side reading (verbatim-copy resistance; minimal compliant rewrite) | 10-17 | M1 (speech) | On the stance cache, teacher-forced per-token log-loss of the window text repeated "as the speaker"; greedy minimal rewrite, edit rate by LCS | Window-level within copy_mean .541/.504, edit_rate .585/.518 vs z_speech .726/.606; profanity lexicon .618/.541; word count .626/.504. Decoded within -.040/-.052 and -.045/-.063. `runs/20261017_m1_generation_side/main_precheck/summary.json` | Model copies slurs with zero loss; rewrite softens profanity and political names, leaves explicit hate unchanged | archived at pre-check (item 35) |
| C46 Topic-direction removal at read time | 10-17 | M1 (speech) | Mean-difference direction (protected vs non-protected group fillers in 40 carrier sentences) at embedding + 36 layers; projected out of the speech branch residual stream at every position | Direction valid (held-out AUC .938-.998). Window-level within z_abl .712/.607 vs .726/.606; shuffled-label orthogonal controls .729/.605; Gaussian .723/.607; Spearman with r6 .98-.99. Decoded -.0001/-.0003/-.0043 ; -.0003/-.0001/-.0046. `runs/20261017_m1_topic_removal/main_precheck/summary.json` | Reads drop 1-1.7 logit, 5-8 % flip sign, but positives and negatives drop together; within-video order unchanged; the judgement does not depend on that linear feature in the branch (prefix still carries the target) | archived at pre-check (item 36) |
| Within diagnosis of adjacent_local (analysis only) | 10-14 | M1 + M2 | Error analysis of r6, adjacent_local, E | see section 3 below; `runs/20261014_within_diagnosis/{table.txt,extra.txt,summary.json}` | - | analysis |
| Unrun 2026-10-06 pool items: TFVTG transition grounding (rank 9), BaGLM prerequisite-progress memory (8), actor-object transition witness paths (5), segmental acoustic recurrence (6), LocalAgreement acquisition horizon (7) | 10-06 | M1 | All select remote real sources (by dynamic/static transition scores, dependency-progress belief, actor/object identity paths, same-video MFCC segmental DTW, streaming-ASR confirmation horizon), then a fresh V/S read | Jury PASS only; no runs. `experiments/20261006_m1_ideation/CANDIDATES.json`, `docs/reviews/20261006_m1_ideation_jury.md` | Not reached before the 2026-10-07 stop | proposed only |

Scope sub-lines (reader / inputs / GT):
- C23, C24, C26, C27, C33, C34, C35, C36, C25: Qwen3-VL-8B; r6 prefix; C25, C26, C27, C31, C33, C34, C36 add 2 real in-window frames at 1/3 and 2/3 (PTS-decoded); C35 1 fps; ASR = proportional slices of the original Whisper segments (not word times) except C38/C43.
- C28, C30: only the 20 native prefix frames, compressed; no new frames.
- C29: proportional ASR segment intervals bound into rotary channels; no word times.
- C37, C39, C40: 0.5 fps real frames (`data/temporal_retrieved_kv/`, `data/temporal_multigrain_kv/`).
- C38: second Whisper-large-v3 greedy transcript in 30 s blocks with DTW word times (`data/temporal_discourse_speech/`); the prefix keeps the original transcript.
- C41, C42: Qwen3-VL-8B; stored placement reads; C42 shows 0.5 fps frames of both windows.
- C43: reader Qwen3-Omni-30B-A3B Thinker (bf16, MoE numerics floor .1-.4 log-odds per read); follow-up uses Qwen3-VL-8B; aligned slice from MMS_FA (`data/asr_whisper_aligned/`).
- C44: pyannote 3.1 turns (`data/speaker_turns/`); pre-check reads GT.
- C45, C46: speech branch only, proportional slice, stance cache; window-level within on clean-labelled text windows (HMM 57 videos / 1359 windows, HCS 90 / 2420).
- GT: HCS protocol GT everywhere in this section.

### 1.6 2026-10-25: module evaluation and the M3 round (candidates 47–49)

Default scope for this section:
- reader Qwen3-VL-8B on the r6 prefix, fixed ASR;
- protocol GT;
- "ρ(share)" is the Spearman correlation with the GT hateful-frame share, computed among hateful videos only (HateMM 85, HCS 103). The window mean scores .416 / .533 on it.
- pooled deltas are against r6_bma (calib key), HateMM; HCS.

| id / name | date | module | mechanism in one line | key numbers (source) | stated cause of failure or success | status |
|---|---|---|---|---|---|---|
| Module novelty evaluation (rubric C1–C5) | 10-25 | M1, M2, M3 | Score each module's remaining novelty room on a fixed rubric | After the adversarial pass: M3 10, M1 9, M2 8. HCS key from the window mean alone: pooled +.022 / +.024 (CI excludes 0). The same key on HateMM: ROC −.011. Under HCS hateful-only GT: −.024 / −.026 (`scratchpad/novelty_eval/M3_report.md`, `M1_report.md`) | The HCS room is a GT-definition effect. HateMM has no room on the same inputs. M1: "reader-side novelty exhausted; only input organisation gains" | analysis (memory `module-novelty-eval-2026-10-25.md`) |
| Prevalence estimators as the key's extent term (SLD, PCC) | 10-25 | M3 | Per-video class prior from the window reads: Saerens EM and probabilistic classify-and-count | ρ(share) .389 / .437 and .387 / .455; as key −.003 to −.03 (`scratchpad/novelty_eval/M3_report.md` C1; `novelty_eval/m3_check/m3_pcc_pilot.py`) | No more extent information than the plain mean | closed |
| Extent estimators from existing caches (M3 scout §0 and task 3) | 10-25 | M3 | DVD T/A/E reads; HVL cited-span share; Storey π0; CC/PCC × P(V\|K); M2 implied share dh/(dg+dh); M2 posterior mean; window mean of the no-stance reads (old ASR); window mean of the context-free single-window reads (old ASR) | ρ(share) in the same order: .14–.21 / .26–.32; .201 / −.055; .205 / .398; .345 / .532 and .349 / .538; .207 / .418; .136 / .542; .384 / .464; .362 / .224. All below .416 / .533 (`scratchpad/m3_team/scout_report.md` §0, `scout_report3.md` §0) | The window mean correlates with z_video because both see the full input, not because of the stance turn (no-stance reads correlate more, .928 / .885 vs .888 / .819). Without context the extent information is lost | closed |
| Window means of other M1 read sets as the extent term | 10-25 | M3 (through M1 inputs) | ρ(share) of the window mean under other fixed-ASR read sets | adjacent_local .424 / .615; vl_text_aligned .438 / .603; vl_r6_aligned .446 / .502; joint .423 / .360; frames-only prefix .293 / .557; transcript-only prefix .374 / .202 (`scratchpad/m3_team/strategist/check_reads.txt`) | In-window frames raise the HCS extent quantity. This matches adjacent_local's decoded HCS pooled .7383 / .6893 vs .7168 / .6711, with HateMM flat | finding; an input change, so adoption is the user's decision |
| Share as a likelihood on M2's length grid (scout B) | 10-25 | M3 → M2 | Read share enters the (gap, hate) grid prior | With the GT share as the observation: within .7460 / .6356 (sd 1) and .7506 / .6366 (sd .5), vs r6 .7508 / .6373 (`scout_report.md` §1) | Even a perfect share does not move within | refuted on CPU |
| Per-video shift of M2's posterior to match the share (scout C) | 10-25 | M3 | λ_v chosen so that mean σ(logit p_t + λ_v) = share | With the GT share: HCS .8683 / .8436, vs key + rank with the same share .8923 / .8790 (`scout_report.md` §1) | Worse than key + rank even with the gold share | refuted on CPU |
| Synthetic share reads (threshold for any new extent read) | 10-25 | M3 diagnostic | Synthetic read of a given ρ(share) plugged into the key | Product key log P(V) + log share: HateMM −.08 to −.11 ROC even at ρ = .90. Summed key needs ρ ≥ ~.55 (HateMM) or ~.45 (HCS) to gain ≥ .01 ROC (`scout_report.md` §2, `extent_read_threshold.py`) | Product form unsafe on HateMM; sets the bar for a new extent read | diagnostic |
| C47 whole-video share read | 10-25 | M3 | Isolated branch on the full prefix: "How much of this video's running time contains content that violates…" with six options A–F; read as the expectation E[c] or as a proportional-odds latent η; replaces the window mean (arms A1, A2, A1η, A2η) | P3 fails: ρ(E[c], z_video) .965 / .904, ρ(η, z_video) .973 / .958. Videos with z_video > 0 answer F: 142 / 146 and 108 / 108. P1 .254 / .340. A1 pooled HateMM −.164 / −.295 (`experiments/20261025_m3_coverage/README.md` §8, §10; `runs/20261025_m3_coverage/main/reads.jsonl`) | "The share question still reads 'whether'"; the option distribution is near one-hot (max p > .99 in .85 / .93 of videos) | archived (item 37) |
| C48 share read after the stance turn | 10-25 | M3 | Same question after the verdict turn; K = z_video + s·η_st (s matches the sd of the window mean), calibrated | P1 .195 / .252; partial ρ given the window mean +.037 / −.203. B1 pooled HateMM −.011 / −.038, HCS −.035 / −.048, DeHate −.033 / −.006 (README §10) | Conditioning lowers the correlation with the verdict (.886 / .705) but the read carries no share information | archived (item 38) |
| C49 mutually invisible segment presence reads | 10-25 | M3 | Video cut into 4 equal parts; each part read alone (its 5 frames + its transcript) with the verdict question; mean segment log-odds q_v (C1) or a group-testing share (C2) replaces the window mean | P1 .278 / .419 (DeHate −.045); ρ(q, z_video) .899 / .895. C1 pooled HateMM −.010 / −.048, HCS −.011 / −.017; HCS hateful-only +.018 / +.019; DeHate −.018 / +.015 (README §12; `runs/20261025_m3_coverage/seg_main/segments.jsonl`) | Without the full context the share information is lost; following the "hate" definition helps hateful-only GT and DeHate PR but loses on protocol GT and HateMM | archived (item 39) |
| Round-3 CPU composition variants (39) | 10-25 | M3 | 27 two-part frame scores log P(V) + log q_t; 8 linear "share × rank" forms; 6 rank-width rules (s = .5, .25, .125, d′ rule, per-video 4P(1−P)); per-video proportion MLE; segment-model Bayes factor | Best passing G1: d′ width rule +.003 / +.003, +.005 / +.005 (fails G2). r6 key × M2 posterior (= c_full) −.007 / +.019, +.022 / +.007 (fails G1). MLE ρ(share) .295 / .516 (README §13; `runs/20261025_m3_coverage/explore_{hurdle,shares}/summary.txt`) | "No form passes G1 and G2 without relying on the HCS GT definition"; composition room is about +.003–.005 | closed |
| DVD verdict + protected-group target, re-decoded on r6 | 10-25 | M3 | Noisy-AND of the calibrated key and the calibrated z_target | HateMM −.0008 / −.0015; HCS −.031 / −.037; DeHate +.035 / +.038; HCS hateful-only +.020 / +.012 (README §14; `runs/20261025_m3_coverage/dvd_r6/summary.txt`) | Same pattern as C49 | awaiting a user decision (HCS protocol GT kept, so it fails the gate) |
| Threshold-ladder share (scout fallback 1) | 10-25 | M3 | Four isolated Yes/No branches: "does violating content take up more than t of the running time?", t ∈ {1/10, 1/4, 1/2, 3/4}; key = mean cumulative logit | not run. Question text in `scratchpad/m3_team/engineer/ladder_q.txt`; builder in `src/prefix_questions.py` | Designed GT-free pre-check (20 videos): ℓ_t monotone in ≥ 80% of Yes videos, ℓ_{1/10} − ℓ_{3/4} sd > 1 logit, ρ(ℓ_{3/4}, z_video) < .9 | open (never run; see §3 P1) |
| "Is most of this quarter violating?" on the full prefix (scout fallback 2) | 10-25 | M3 | Four isolated branches, one per video quarter, on the full context; share = mean σ | not run. CPU proxy (window reads pooled per quarter) ρ(share) .204 / .479, or .284 / .510 with a sigmoid (`scout_report2.md`) | Weak support: needs information the window reads lack | open (weak) |

## 2. Closed families (for the reader Qwen3-VL-8B), with the scope of the evidence

**Default scope** unless a family says otherwise:
- reader frozen Qwen3-VL-8B-Instruct, bf16, greedy Yes/No log-odds at one position;
- the r6 or SPVL-r2 prefix (rules + 20 uniform frames + full timestamped Whisper transcript + verdict + one-word stance turn);
- isolated 8 s branches;
- HateMM and HCS test with protocol GT;
- decoded by the fixed r6 time level.

"Closed" means the evidence rules the family out *in that scope*. Each entry names what it does **not** cover, so
designers do not over-close.

### 2.1 M1 (how a window is read)

**M1-a. Asking the per-window question in another verbal form.**
- What was tried:
  - pairwise "which window is more violating": PWC with timestamps (09-12), C42 with both windows' frames shown (10-13);
  - onset / offset (BND);
  - five-way speech act (TAD r2);
  - subtracting a topic read (TAD r1);
  - context contrast z(full) − z(alone);
  - verbal exclusion of an interval (CVA);
  - joint or triple branch;
  - wrapper wording (QuoteGraph R1).
- Result: every form lands at or below the absolute Yes/No read (window-level about .76 / .62). Neighbouring windows are
  compared at chance (C42 gap-1 accuracy .48 / .42). The reads move by wording (QuoteGraph scaffold: +2 logits on
  negatives) without becoming more informative.
- Scope:
  - Sept items are old ASR and window level; C42 is fixed ASR with frames.
  - The question is always one next-token decision.
  - Speech-side pairwise with aligned words was never run, but PWC (text) already covers it in spirit.

**M1-b. Feeding the model's own generated text back before the read.**
- What was tried:
  - HVL hypotheses with cited spans, TARGET / FORM-only, revision, sequential state chain;
  - MARS reproduction (2B, detection era);
  - RAVEL relation parse;
  - Tree captions in the shared prefix;
  - Program, Provenance, Ordered slots, VTimeCoT, Text tracking records;
  - Interval-witness leaf statuses.
- Measured failure modes:
  - confirmation: 83% of windows "present"; 88% inside vs 44% outside the model's own cited spans;
  - persistence: 77–82% "continue";
  - restatement: z_rev vs z_video .76–.84;
  - injected neutral descriptions pull true positive windows down (Program R3).
- A permuted (other video's) hypothesis works as well as the model's own, so the content is not used.
- Scope: Qwen3-VL-8B greedy generation, plus 2B for MARS.
- Caveat: five structured-generation candidates failed at the schema interface (Program A, Provenance A with 158/158
  ledgers invalid, Ordered slots A, VTimeCoT A/B, Text tracking). That family is closed *in practice* for this reader,
  not falsified as an idea. Factored verification (CoVe, candidate 27) was never measured end to end.

**M1-c. The model's own statements about where the evidence is.**
- What was tried:
  - HVL E0 cited spans: frame within .571 / .506;
  - GLR quote-to-window matching: window .598 / .552 vs speech read .694 / .604;
  - C41 probe A ("which prefix frames lie in this window"): correct in 36% / 44%;
  - C41 probe B ("which frame did you base it on"): tracks the sign of the read;
  - AMBI / COHERE same-event judgements: matched by time-only controls.
- Result: near chance within video.
- Scope: text probes on the same cache, frames referred to by `[t=..s]` labels.

**M1-d. Label-free choice, per window, of whether to use an extra read.**
- What was tried:
  - pre-read margin, rank, speech disagreement, coverage;
  - model citations (C41);
  - agreement between reads, neighbour direction;
  - averaging five stored reads;
  - Explorer's entropy trigger.
- Result: all inside six size-matched random bands (the random halves span .008–.018 within). The +.05 to +.09
  label-chosen selection is label injection, not headroom.
- Scope: the five stored placement reads plus the C41 probes
  (`runs/20261007_m1_streamingtom/{selection,combination}_analysis/summary.json`).

**M1-e. Window score from the whole-video verdict's dependence on the window.**
- Gradient forms:
  - 08-12 grad×input and attention on one whole-video call: equal to token density (.5252 vs .5182);
  - Attributor, integrated gradients through cached values: −.130 / −.069; raw window within .560 / .538.
- Physical removal: Eraser occlusion with re-encoding, −.131 / −.131.
- Verbal exclusion: CVA, .6747 / .5779 vs .6976 / .6001.
- Sentinel markers in one causal pass (08-12): .4994.
- Scope: Qwen3-VL-8B, both the saturated whole-video call and the r6 prefix.
- Not run: Shapley over windows (Allocator, stopped on novelty because SHAP is already used in hate-video detection).
  Shapley differs from leave-one-out exactly where windows are redundant.

**M1-f. Changing which tokens a window branch (or the prefix encoding) may attend to.**
- What was tried:
  - Grounder: last 9 of 36 layers see only own-window media + scaffold, no verdict;
  - Selector: per-head restriction by affinity;
  - Program R4: visual query restricted to program-selected image keys;
  - QuoteGraph: asymmetric context mask;
  - Factorizer: block-causal prefix;
  - Integrator: future-aware visual rows;
  - Stabilizer: RoPE phase;
  - RoTE: interval rotary on ASR keys;
  - branch visibility: causal mask, sequential branches, neighbour transcripts.
- Key fact (Grounder): showing the right vs a wrong window's media in the last 9 layers moved branch logits by only .06.
  Restricting every layer costs −.046 / −.049.
- **This also closes the Codex round-2 proposal** ("late layers see only policy + own window"). `sep_aug.md` lists it as
  never run, but Grounder (2026-10-02) ran exactly that computation, with an `early` control.
- Scope: all 36 language layers; mask constants as declared.
- Not covered: interventions that *re-present* the window's own evidence at the question (see §3 O1).

**M1-g. Hallucination-mitigation transfers on the visual branch, and input contrast.**
- What was tried: PAI, VAR sink redistribution, ACG, VGA (salience-weighted in-window image values added in layers
  4–17), MAD-RAG, VISTA + SLA, DoLa (the dynamic layer choice collapsed to layer 2 for all 13,939 branches), VCD with
  whole-prefix or own-window noise.
- Result: within changes within ±.009, raw visual order never better on both corpora, HateMM PR −.009 to −.033 in several.
- Scope: visual branch only; source-paper constants, no scan (closed at those constants).

**M1-h. Hidden-state directions.**
- What was tried:
  - 2026-08, on the whole-video judge (final prompt token, layers 18 / 27 / 36):
    - unsupervised PCA (PC1 Spearman .992 with z);
    - forced-answer contrast (.983);
    - spec displacement;
    - text-derived illocution direction (text 1.0, video .334 / .483).
  - 2026-10-17, candidate 46: a "protected group mentioned" direction projected out of the speech branch at 37 points.
    - Order unchanged: Spearman .98–.99 with r6; window-level .712 / .607 vs .726 / .606.
- Not covered:
  - per-window hidden-state readouts under fixed ASR (never cached; the 09-12 gold probe was mis-built per the 09-13 audit);
  - directions learned from video-evidence states, e.g. an SAE;
  - other directions (attack, endorsement);
  - ablation in the prefix rather than the branch.

**M1-i. Generation-side behaviour, and the likelihood of the words as the score.**
- Candidate 45: copy loss .541 / .504 and rewrite edit rate .585 / .518, both below a profanity word list (.618 / .541).
  The model copies slurs with zero loss.
- GLR: window-level .49–.57. Words are unpredictable (−5.6 / −6.5 nats/token) and the speaker condition moves them by about .01.
- Scope: speech branch, proportional slice, Qwen3-VL-8B instruct.

**M1-j. More of the same video added to the branch.**
- What was tried: ReKV and StreamingTOM retrieval / memory, MERIT, Provenance, Ordered slots, VideoEvent, VTimeCoT,
  Interval witness, QuoteGraph context, Tree hierarchy context, HVL neighbour transcripts.
- Result:
  - removing the remote part never costs .01;
  - remote speech lowers positive speech reads on HateMM;
  - every gain traced to the window's own frames (StreamingTOM `no_remote` = R1; Ordered slots R3: all REMOTE windows unchanged).
- Scope: with the transcript and 20 frames already in the prefix.

**M1-k. Compressing or merging visual tokens** (TTF, OTT, StreamingTOM uniform). Within changes ≤ .008.

**M1-l. Adapting the reader with its own verdict as supervision.**
- What was tried: SDL MIL LoRA (3 rounds), NGA negative-only (3 rounds), entropy TTA, self-trained head, DAPT precondition.
- Scope: LoRA rank 8, transductive on test. Pseudo label 1[z_video > 0] gives only 79 pseudo-negatives, 10 of them on HCS.

**M1-m. Window geometry.**
- What was tried: ASR-segment windows, 4 s and 16 s windows, 8 frames, the shifted second grid (TIL: half-overlapping reads agree only .57 / .63).
- One frame per window: an HCS-only gain at 5× cost.

**M1-n. Speaker binding** (candidate 44 pre-check).
- 71% of HateMM within videos have 0–1 speakers; the identity ceiling is +.036.
- The HCS-only signal (+.077 ceiling) is single-corpus.
- The "does this speaker endorse" question had already failed three times (DVD E, TAD, GLR).

**M1-o. Swapping or pooling readers.**
- Seven other MLLMs reach the same ceiling (within .65–.70 / .55–.60).
- Qwen3-Omni loses HCS video ranking with identical inputs (−.049 ROC).
- An 8-reader average gains, but it is an ensemble, forbidden by rules 3 and 14e.

**M1-p. Per-window latent optimisation** (paper-equation form): −.023 / −.071. Off-the-shelf temporal grounding / VAD
readers: within .47–.62.

### 2.2 M3 (video key and fusion)

**M3-a. A second verbal whole-video question on the same full prefix collapses onto the verdict.**
- What was tried:
  - 08-09 dual-axis offensiveness (.935);
  - 08-10 speech-act / endorsement (.92–.98);
  - HVL revision (.76–.84);
  - DVD T / A / E (.89–.99 with z_video);
  - C47 six-option share (.965 / .904, near one-hot);
  - same-prefix Yes/No in the M3 scout (.89–.99).
- Exception: DVD T moves the ranking, but only toward the hate definition (loses on HCS protocol GT).
- Scope: Qwen3-VL-8B full prefix; option and Yes/No forms.
- Not covered: the threshold-ladder Yes/No form (§3 P1).

**M3-b. Share read conditioned on the model's own verdict** (C48). The correlation with the verdict drops, but no share
information remains (partial ρ given the window mean ≤ +.037). Scope: options form after the stance turn.

**M3-c. Context-free sub-video reads as the extent term.**
- C49 quarter segments: ρ(share) .278 / .419.
- Window mean of context-free 8 s reads (old ASR): .362 / .224.
- Scope: segment = its own frames and transcript only. Intermediate contexts were not tried (§3 P3).

**M3-d. Definition conjunctions / products at video level.**
- PP-v2 product, MADP hard conjunction (0/4 corpora on 8B), DVD noisy-AND.
- All lose on HCS protocol GT.
- Not novel either: Hypothesis Engineering (TRAC 2022), xList-Hate, CLUE.

**M3-e. Recombining the existing reads into a better key.**
- What was tried:
  - Bayes posterior over all reads (overconfident: EM evidence is 5–14× the GT slope);
  - ICC-effective reads;
  - verdict + mean-read mixture;
  - expected hate fraction key (.8632 / .6180);
  - posterior key;
  - decoder probability as the frame score;
  - revisable shared random offset (pooled collapses to .57 / .51; regime video ROC .637 / .625 vs .933 / .806 for z_video);
  - product P(V|K) × P(h|V) (HateMM ROC −.007);
  - the 39 round-3 forms;
  - prevalence estimators SLD / PCC;
  - Storey π0;
  - median / top-half / probability mean of the window reads (spread ≤ .013).
- Gold-fitted bound: a video score fitted on test GT from all cached reads, DVD included, is *below* the label-free key
  (video AUC .883 vs .927 HateMM, .766 vs .803 HCS; `experiments/20260928_headroom/README.md` A3).
- Scope: Qwen3-VL-8B reads on the r6 / SPVL prefix. The room left is about +.003–.005.

**M3-f. Key calibration form.**
- The two-Gaussian calibration: pooled +.002 / +.006, mean across 8 MLLMs < .001.
- The user removed it on 2026-10-07; do not compare against it.
- The calibration is identical to Gao & Tan 2006 and Brümmer & Garcia-Romero 2014, so nothing to claim there.

**M3-g. Modality-only verdicts as replacement scores** (PP-v4 noisy-OR, detection era: below the joint call). Used as
chain gates (2026-09-30) they meet the user's "best fixed configuration" rule but fail 14g (−.0005 / +.005). Adoption
is the user's decision.

### 2.3 What is NOT closed (common over-closure mistakes)

- **Hidden states and attention of the window branches.** August closed only the whole-video judge. October candidate
  46 closed one linear direction. Nothing was read from fixed-ASR branch states.
- **Evidence binding.** Re-showing the window's own evidence next to its question is the one reproducible M1 gain. It
  was only ever run as an input change. It is not a closed family.
- **The stance turn.** It was never isolated under fixed ASR on HateMM / HCS: no fixed-ASR no-stance read, no fixed-Yes /
  fixed-No, no question-without-answer.
- **Verdict marginalisation** (both forced verdicts, then average). Stopped on novelty grounds only (MARS, RAMF); its
  performance is unknown.
- **Silent-window evidence on HateMM.** It was tried only by swapping the whole reader to Qwen3-Omni.
- **The threshold-ladder share question.** Designed, never run.

## 3. Open neighbourhoods (near past attempts, never actually tested)

Each entry gives:
- **why open**;
- **nearest closed result**;
- **cheapest check**: CPU on caches where possible (§4 lists the paths);
- **risk**: the rule-4 / rule-5 objection or the known failure it must avoid.

The user's criteria still apply. The change must be a mechanism-level redesign of the module, not an add-on, not chosen
because of an error type, and not an input change presented as novelty (rule 5).

### 3.1 M1

**O1. Binding the window's own evidence to its question, as a mechanism.** (Strongest open item.)

Evidence that binding is what the read uses:
- Re-showing the window's own prefix frames right before the question, with no new image (`adjacent_native`), gives
  within +.0224 / +.0128 and no loss. Metrics .8970 / .6938 / .7732 and .7252 / .6771 / .6501 in
  `runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_native/metrics.json`. This is the largest
  HateMM within gain of any arm.
- The model finds the window's frames by timestamp label in only 36% / 44% of covered windows (C41 probe A).
- The window's own audio clip is read for its own window: a wrong clip drops within to .658 / .582 (C43 `audio_swap`).
- The aligned word slice gives +.017 / +.029 under the r6 reader (C43 follow-up).
- Frames + aligned slice give +.029 / +.049 (`vl_text_aligned`).
- Placement and frame count are separately sufficient (`prefix_local` +.013 / +.015). They stack on HCS and not on HateMM.

Why it is still open:
- Every run was framed as an input change.
- No run made binding itself the claimed mechanism: one uniform rule that binds each branch to "its" evidence by
  position, with a wrong-binding control and a no-binding control, as one component under 14g.

Nearest closed results — interventions that *restrict* or *re-weight* access lost:
- Grounder late-local masks: logits move .06.
- Program R4: key restriction.
- VGA / Highlighter: salience-weighted in-window values added in layers 4–17, +.004.
- RoTE: interval rotary on ASR keys, all metrics down.

Re-presentation won. So the open form is re-presentation at the question, not masking.

Cheapest check:
- Already measured: the placement reads in `runs/20261007_m1_streamingtom/placement_controls_main/records/`.
- CPU: per-corpus raw window AUC and raw within for `adjacent_native` vs native, with a per-video shift-only control
  (trap 1).
- GPU, if a KV-level form (copy the in-window frame KV next to the question instead of re-encoding) is proposed:
  minutes. `adjacent_native` cost +4.7 min for both corpora. `src/window_token_regions.py` and
  `src/pre_rotary_memory.py` already give exact token spans and pre-rotary handling.

Risks:
- Rule 4 case 4 ("only prompt layout") and rule 5 (input): a reviewer can call re-showing frames engineering. The
  claim has to rest on the measured lookup failure and on a binding ablation.
- On HateMM the frames gain is chain-mediated (trap 6). Report raw reads per corpus.

**O2. Depth trajectory of the window read.**

Why open:
- Per-layer Yes/No logits are cached for every visual and speech branch of the r6 reader. That is 17 premature layers
  (2, 4, …, 34) plus the final layer, with full-vocabulary log normalisers (`runs/20261003_m1_contraster/r1_main/checks.jsonl`,
  333 rows, 13,939 branches).
- Only DoLa's dynamic choice was ever scored, and it collapsed to layer 2.
- Fixed-layer contrasts, a slope over depth, or the layer of first commitment were never evaluated.

Nearest closed:
- DoLa: within ±.0002.
- 2026-08: the dominant hidden direction of the judge equals the readout (.992).

Cheapest check: CPU, window-level within of each per-layer margin vs the final margin, both corpora.

Risk: depth readouts of a one-token answer may just track the final margin. Rule 4 STOP if it is "DoLa with another
layer" (DoLa is the source).

**O3. Hidden states or attention of the window branches as the read.**

Why open:
- Nothing was ever saved from fixed-ASR branches.
- The only branch states on disk are last-layer, old ASR, 183 videos (`runs/20260912_tad/e0_hidden`).
- The 09-12 gold probe built on them was mis-described (it stored the speech vector when present). Its speech version
  reached only +.010 on HateMM, the noise line.
- August closed only the whole-video judge.
- A direction learned from video-evidence states (an SAE, or a video-side consistency search) was named as "the only
  surviving route" in `archive/detection-2026-08/docs/duplex/ILLOCUTION_TRANSFER_NOTE.md` §4 and never run.
- Branch attention: the Explorer cache holds, for every visual window of the r6 reader, the last query token's
  attention prior over the 20 prefix frames (all 36 layers; `runs/20261003_m1_explorer/r1_main/details/<ds>/<vid>.json`,
  field `traces[].initial_prior`). A CPU check of "attention mass on the window's own frames" as a binding-quality
  signal is possible.

Nearest closed: M1-h (label-free directions on judge states), M1-f (Selector used branch attention affinity as a mask
and lost).

Cost:
- GPU to save, for example, last-layer branch states: about 13,939 × 4096 × 2 bytes ≈ 114 MB per layer for both corpora.
- The memory rule says give the size and ask first (`feedback-no-persistent-audit-tensors.md`).

Risk: label-free readouts from states are a family where August found only "z renamed".

**O4. The stance turn, isolated under fixed ASR; and verdict marginalisation.**

Why open:
- The stance turn shifts every window by +2.39 / +2.08 logits in Yes videos with order Spearman .93 / .92. It is
  partly a per-video shift effect (verdict_analysis, old ASR).
- No fixed-ASR no-stance read exists for HateMM / HCS. Forced-Yes, forced-No and question-without-answer branches were
  never run.
- Marginalising the window read over both forced verdicts (candidate 4) was stopped on novelty only, against MARS /
  RAMF. Its performance is unknown.

Nearest closed:
- HVL: a permuted hypothesis works as well as the model's own, so conditioning content is not used.
- 08-10 answer contrast on judge states: the contrast direction equals z, Spearman .983.

Cheapest check: GPU, one extra window pass with forced "No" (about 4–9 min for HateMM + HCS, timing from
`runs/20260928_infer/main/reads_*`). GT-free first test: is the within-video sd of z(w | Yes) − z(w | No) > .5 logit?
If the contrast is a per-video constant, it is dead.

Risk: rule 4 STOP precedent (MARS, RAMF) unless the mechanism differs from "reason under both assumptions and average".

**O5. Fresh local ASR of the window's own audio, single best.**

Why open:
- Lattice R1 / R2 gave HateMM within +.048 / +.052 (CIs above 0) with HateMM PR −.017.
- The `onebest` control (new 8 s transcript, ordinary read) that would split "better local transcript" from "lattice
  structure" was coded and reviewed but never run.

Data: 5-best hypotheses are in `runs/20261004_m1_lattice/r2_full_main/records/*.json` as token-id paths; the 1-best can
be decoded on CPU. A Qwen read needs GPU (minutes).

Risk: this is an input change (rule 5). It also overlaps with the aligned-slice gain (O1).

**O6. Silent-window evidence on HateMM without swapping the reader.**

Why open:
- The Omni audio read on ASR-silent windows scored .848 vs .633 for the visual fallback, and +.170 within on 12 videos
  with a silent positive.
- It was tested only by replacing the whole reader with Qwen3-Omni, which loses HCS video ranking (−.049 ROC).
- Rule 3 allows a separate frozen audio model for the audio modality.

Data: `data/audio16k/` (HateMM, HCS), ImageBind audio embeddings (§4).

Risks: HateMM-only by construction (HCS silent positives are visual). Adding a third modality chain is an add-on unless
framed as the speech channel's own read.

**O7. Per-video modality selection beyond the verdict gate.**

Why open:
- Per-video oracle between max and joint is +.034 / +.036 (+.065 DeHate) (`experiments/20260928_infer/README.md` §14.3).
- Gate v1 reaches each corpus's best fixed configuration but fails 14g.
- Gate v2 (inside EM), selectors from the video's own reads (picture-only-positive share, branch autocorrelation) and
  the per-modality chain-vs-independent Bayes factor were never run. All are CPU on cached reads
  (`runs/20260928_infer/main/reads_{joint,noctx,noframes}`).

Risk: rule 3 / 13 (must be one rule for all corpora); earlier selection attempts landed inside random bands.

**O8. Cached component reads never evaluated.** (Low value.)
- Amplifier's text-only reference: the visual question on a prefix without frames.
- Preserver's reference-only visual read: frames + policy, no ASR, no verdict.
- Reinforcer's SLA-only logits.

All are a CPU re-score (`runs/20261003_m1_{amplifier,preserver,reinforcer}/r1_main/tokens/`).

**O9. Remaining literal gaps** (weighed against by M1-j or M1-b):
- the unrun 10-06 pool items: TFVTG transition sources, BaGLM progress memory, actor-object paths, segmental-DTW
  acoustic recurrence, LocalAgreement horizon;
- MuKV;
- factored verification (CoVe, candidate 27);
- latent optimisation with the official contextual-backbone variant;
- Shapley over windows (novelty STOP);
- other residual directions: attack or endorsement; ablating in the prefix; reading `proj_topic` (37 floats per speech
  window, cached) as a covariate.

### 3.2 M3

**P1. Threshold-ladder share questions.**

Why open: designed by the M3 scout as fallback 1 and never run. The design is four isolated Yes/No branches "does
violating content take up more than t", read as cumulative logits. Each threshold has its own Yes/No margin instead of
six options competing in one softmax (C47's near-one-hot failure).

Data: question text and builder already exist (`scratchpad/m3_team/engineer/ladder_q.txt`, `src/prefix_questions.py`).

Cheapest check: GPU, 20 videos, GT-free:
- monotone in ≥ 80% of Yes videos;
- ℓ_{1/10} − ℓ_{3/4} sd > 1 logit;
- ρ(ℓ_{3/4}, z_video) < .9.

Then P1 against .416 / .533, and partial ρ given the window mean > 0. Cost: 333 videos about 11 min.

Risk: every full-prefix question so far collapsed onto the verdict (.89–.99). The synthetic-read threshold says ρ(share)
must reach about .55 / .45 to gain .01.

**P2. Verdict attribution or occlusion mass as the extent term.**

Why open: the Attributor per-token integrated gradients of z_video and the Eraser leave-one-window-out verdicts were only
ever used as window scores, never as "how many windows carry the verdict".

Data:
- `runs/20261002_m1_attributor/r1_main_fp32_mem/tokens/<ds>/<vid>.npz` (`attribution`, `local` = windows × tokens);
- `runs/20261003_m1_eraser/r1_main/checks.jsonl` (`windows[].minus_visual`, `minus_speech`).

Cheapest check: CPU. ρ(share | hateful) and partial ρ given the window mean.

Risk: both are full-context quantities, and the M3 round found every full-context quantity collinear with the window
mean. As window scores they were much worse than the reads (−.13).

**P3. Intermediate context for sub-video reads.**

Why open: C47 / C48 (full context) collapse onto the verdict, and C49 (segment alone) loses the share information.
Nothing between them was tried, for example:
- segment frames with the full transcript;
- segment reads after the stance turn;
- the scout's fallback 2: "is most of this quarter violating" on the full prefix.

Cheapest check: GPU, minutes per variant. Fallback 2's CPU proxy is weak (.204 / .479).

Risk: an extra read added to the key is an add-on unless it replaces the window mean.

**P4. Extent through better window reads (cross-module fact, not a closed or open M3 family).**
- Window means of in-window-frame reads have ρ(share) .424 / .615 vs .416 / .533. adjacent_local's decoded HCS pooled
  is +.022 / +.018 through its own key.
- So M1's binding gain partly reaches pooled through M3. A design that improves M1 binding should report the M3 effect
  separately.

### 3.3 History check of the M3 literature scout's top 5 (`lit_m3.md` §H)

1. **Per-video random effect inside the segment model.**
   - Already tried: the 2026-10-02 *revisable prior* (`archive/experiments/20261002_revisable_prior/README.md`) is this
     model. One shared b ~ τ·N(0, 1) enters the global read and every window read; V shifts the global mean and gates
     the explicit-duration chains; EM is joint with Gauss–Hermite quadrature; Kim & Smyth JMLR 2006 is named as the
     ancestor.
   - Result: b absorbed 80% / 68% of the global variance. Regime video ROC .637 / .625 vs .933 / .806 for z_video.
     Pooled .569 / .360 and .514 / .490.
   - `lit_m3.md` does not list it among the closest attempts.
   - Differences still possible: a V-dependent prior on b, or the stance read instead of the no-stance read (the 10-02
     run used old-ASR no-stance reads). A designer must say why the absorption would not recur.
2. **Joint decoding under forced Yes / forced No.**
   - Candidate 4 (Marginalizer) was the averaging version, STOPped on novelty.
   - The contrast idea meets the 08-10 answer-contrast result (z renamed) and the stance-shift fact (order .93 / .92).
   - Still untested; see O4 for the GT-free first test.
3. **Shared null from pseudo-benign videos, share = 1 − π̂0.** The M3 scout computed Storey π0 with the null taken from
   windows of verdict-No videos: ρ(share) .205 / .398 (`scratchpad/m3_team/scout_report.md` §0). The conformal p-value
   form is the same estimator family. Effectively closed unless the null is built differently.
4. **Share as an unbalanced-OT variable.** Proposed in 2026-08 (transport PoS, built on LESS / CoCa curves, dev cohort
   only) and never run on SPVL or r6 reads. Open. Nearest closed shares: M2 posterior mean .136 / .542, M2 implied share
   .207 / .418, per-video MLE .295 / .516.
5. **Video sets gating local models.** Nearest closed:
   - reading-condition emitter groups (09-28: within noise; a shuffled control matched them);
   - LESS / two-view mixtures;
   - the three-layer reads.

   Open in the literal sense, weighed against by those.

(`lit_m1.md` was not yet written when this file was finished, so it is not cross-checked here.)

## 4. Data on disk usable for CPU pre-checks

Scope: `/home/jehc223/Hate-follow-up/data/` and `/home/jehc223/Hate-follow-up/runs/` on uoa-lab1. Every entry below was
checked by counting files and parsing one sample (json / numpy); no hashes or checksums were computed.
Corpora: HMM = HateMM test (215 videos), HCS = HateClipSeg test (118 videos in the evaluated cohort), DH = DeHate test
(1341 read, 1151 in the GT cohort). "r6 prefix" = Qwen3-VL-8B-Instruct, rules + 20 uniform frames (`data/frames_k20`) +
full timestamped Whisper transcript with the fixed ASR loader (2026-09-26, `src/video_inputs.py:23 load_asr(fill_untimed=True)`),
verdict question, stance turn. "Old ASR" = reads made before 2026-09-26, when Whisper chunks with a missing start/end were
dropped (HMM 42 / 2308 chunks, HCS 95 / 5454, DH 171 / 10310 chunks have a None time in the raw file).

Free disk on the data filesystem (`/`, nvme0n1p2): 1.8T total, 1.5T used, **253G free (86% used)**. The memory rule
`feedback-no-persistent-audit-tensors.md` applies: do not save per-frame / per-token tensors by default; give the size and
ask first.

### 4.1 The current method's reads and outputs (start here)

| path | holds | corpora | coverage (checked) | producer | notes |
|---|---|---|---|---|---|
| `runs/20260926_glr/base_gridA/predictions.jsonl` | **the r6_bma window reads.** Per video: `extra.z_video` (verdict Yes/No log-odds), `extra.stance` ("Yes"/"No" = sign of z_video; the stance turn is only this word, no generated text), `extra.windows[i] = {i, start, end, z_visual, z_speech, z}` (8 s grid, offset 0; z = max), `prefix_tokens`, `n_windows`, `n_frames`, `n_branches`; `score_curve` (4 fps, raw max read) | HMM, HCS | 333 rows (215 + 118), 0 errors | `experiments/20260922_til/til_measure.py`, Qwen3-VL-8B, lab-server sc448960, 2026-09-26 | fixed ASR; r6 prefix. Windows without speech have no `z_speech`. Config holds the exact three question strings. |
| `runs/20260927_dehate_external/reads_gridA/predictions.jsonl` | same fields as above | DH | 1341 rows, 0 errors | same script, sc448960, 2026-09-27 | the DeHate r6 reads |
| `runs/20260926_twolevel/r6_bma/` (= `final_m2/`) | `predictions.jsonl`: `score_curve` (final 4 fps frame score), `extra.z_video`, `extra.p_video_logodds` (calibrated video key, logit P(V=1\|K)), `extra.cell_prob` (M2 posterior per 4 s cell after the OR); `params.json`: per-corpus EM / calibration parameters (pi, m0, m1, t2, per-modality emission mu00/mu10/mu11/s2, iota, key_ab); `run.log` | HMM, HCS | 333 rows | `experiments/20260926_twolevel/twolevel_r2.py`, uoa-lab1, 2026-09-27, about 2 min CPU | per-modality posteriors are NOT stored separately; re-running `twolevel_r2.py` (CPU, ~2 min) gives them. Ablation runs next to it: `final_{nocoupling,nokey,norank,k1,sharedchain,rawkey,rawscale,gauss2/4/8/16,postdirect}`. |
| `runs/20260927_dehate_external/r6_bma/`, `runs/20260926_twolevel/final_dehate/` | r6_bma decoded outputs for DH | DH | 1341 rows | same | |
| `runs/20261025_m3_coverage/keys/r6_key.jsonl`, `r6_key_dehate.jsonl` | per video `z_video`, `window_mean`, `z_target` (= K = z_video + window mean) | HMM+HCS 333; DH 1341 | parsed | candidate 47–49 analysis | the raw M3 key, ready to use |

### 4.2 Other window-read sets under the r6 prefix (fixed ASR), Qwen3-VL-8B

All have the same per-window `{z_visual, z_speech, z}` layout as `base_gridA` unless noted. 333 rows = HMM + HCS.

| path | what changes vs r6 | corpora / rows | producer / date | notes |
|---|---|---|---|---|
| `runs/20260928_infer/main/reads_noctx/` | no full transcript in the prefix (frames-only context); `z_video` here = **frames-only verdict** | HMM+HCS 333 | `til_measure.py --no-transcript-context`, uoa-lab2, 2026-09-30 | used as the visual gate of the modality-gate study |
| `runs/20260928_infer/main/reads_noframes/` | no frames in the prefix; only `z_speech`; `z_video` = **transcript-only verdict** | HMM+HCS 333 | `--frames 0` | speech gate |
| `runs/20260928_infer/main/reads_joint/` | one joint branch per window (`z_joint`) | HMM+HCS 333 | `--branches joint` | |
| `runs/20260928_infer/dehate/reads_{noctx,noframes,joint,nostance,asrwin,seq}/` | as above, plus no stance turn, ASR-segment windows, sequentially visible branches | DH 1341 each | uoa-lab2, 2026-09-29/30 | `reads_nostance` exists for DH only under fixed ASR; for HMM/HCS a no-stance read exists only with old ASR (§4) |
| `runs/20261007_m1_streamingtom/local_controls_main/local_clean/` | **visual branch sees the window's own 0.5 fps frames** (arm "E", the in-window-frames gain); `z_speech` = r6 | HMM+HCS 333 | StreamingTOM controls, uoa-lab1/3, 2026-10-08 | per-video records in `local_controls_main/records/<ds>/` (215 + 118) |
| `runs/20261007_m1_streamingtom/placement_controls_main/{adjacent_local,adjacent_native,prefix_local,local_clean_replay}/` | placement controls of the in-window frames (next to the question vs in the prefix; native frames moved next to the question) | HMM+HCS 333 each | 2026-10-09/10 | `controls_dualpath_main/{base,nearest,no_remote}` = the original StreamingTOM arms |
| `runs/20261014_m1_audio_bound/vl_aligned/{vl_r6_aligned,vl_text_aligned}/` | speech branch shows the **forced-aligned word slice** (word times from `data/asr_whisper_aligned`) instead of the proportional cut; Qwen3-VL-8B | HMM+HCS 333 | uoa-lab1 Slurm 411, 2026-10-14 | the reader-independent aligned-slice gain; windows with no aligned words have no `z_speech` |
| `runs/20261014_m1_audio_bound/main/{audio,text_aligned,text_prop,audio_swap,audio_nbr,r6_inputs}/` | **Qwen3-Omni-30B-A3B reader**: speech branch hears the window's audio clip / aligned text / proportional text / another window's clip / the neighbour's clip; visual = in-window frames (except `r6_inputs`) | HMM+HCS 333 each (also split `main_HateMM/`, `main_HateClipSeg/`) | 2026-10-14 | different reader (not Qwen3-VL); diag `main_decoded_diag/{audio_r6verdict,r6_omniverdict}` swaps the verdict |
| `runs/20261010_m1_grounded_adjacent/main/records/<ds>/*.json` | per window: native reads, in-window frame read, generated "which frame violates" probe (≤16 tokens), grounded flag; arms base/grounded/accept_all/inverted/random | HMM+HCS 333 records | 2026-10-10 | candidate 41 |
| `runs/20261010_m1_grounded_adjacent/probe_main/records/` | generated answers to "which frames are inside this window" and "which frame did you base that on" (text, matched times) | HMM+HCS 333 | 2026-10-11 | candidate 41 probe-before-frames |
| `runs/20261013_m1_pairwise_evidence/main/records/<ds>/*.json` | pairwise window comparisons (A/B/C first-token softmax, both orders, ring graph distance 1 and 2), with both windows' in-window frames; native V and S | HMM+HCS 333 | 2026-10-13 | candidate 42; `selection/` holds decoded aggregations |
| `runs/20261017_m1_generation_side/main/records/<ds>/*.json` | per speech window: **per-token teacher-forced loss of repeating the window's own text** (`A.token_losses`, `A.tokens`, copy_mean/max, first_loss, refuse_mass) and a **generated minimal compliant rewrite** (`B.rewrite`, edit_rate) | HMM+HCS 333; 6580 speech windows, 168,801 tokens | 2026-10-17 | the copy is with the text visible, so the losses are ~0 (not a transcript likelihood) |
| `runs/20261017_m1_topic_removal/main/records/<ds>/*.json` | per speech window: `z_abl` (read with topic direction projected out), `proj_topic` (**projection of the speech-branch residual on the topic direction at each of 37 residual points**), `z_shuf` (5 control directions), `z_rand` | HMM+HCS 333 | 2026-10-17 | scalar projections only, not the residual vectors |
| `runs/20261025_m3_coverage/main/reads.jsonl`, `dehate/reads.jsonl` | candidates 47/48: the **six-option share question** ("How much of this video's running time contains violating content", A–F), read in two modes (`direct` after the prefix; `stance` after the verdict turn); per mode the log-prob of each option letter, `option_mass_full_vocab_log`, top token | HMM+HCS 333; DH 1341 | `read_extra.py`, uoa-lab1, 2026-10-25 | option distributions, not full vocabulary |
| `runs/20261025_m3_coverage/seg_main/segments.jsonl`, `seg_dehate/segments.jsonl` | candidate 49: video cut into J = 4 equal segments; each segment read alone (its 5 frames + its transcript) with the verdict question; `z_seg[4]` and per-segment metadata (n_words, frame times) | HMM+HCS 333; DH 1341 | 2026-10-25 | context-free segment reads |
| `runs/20260927_dvd/reads_main/reads.jsonl`, `reads_dehate/reads.jsonl` | three extra **video-level** Yes/No reads on the r6 prefix: `z_target` (aimed at a protected group), `z_endorse` (speaker endorses), `z_attack` (attacks / demeans people), plus `z_video` | HMM+HCS 333; DH 1341 | `dvd_measure.py`, sc448960, 2026-09-27 | question strings in `reads_main/config.json` |
| `runs/20261003_m1_*`, `runs/20261004_m1_*`, `runs/20261005_m1_*`, `runs/20261006_m1_*` `<arm>/predictions.jsonl` | the October M1 candidates (attributor, grounder, selector, amplifier, contraster, eraser, explorer, factorizer, highlighter, integrator, preserver, projector, recycler, reinforcer, stabilizer, visual_contrast, acoustic, latents, lattice, tree, interval_witness, ott, program, provenance, quote_graph, rote, spatial_search, ttf, merit, ordered_slots, rekv, texttiling, videoevent, vtimecot): one `base` replica of r6 plus the candidate arm(s), same window layout | HMM+HCS 333 each | 2026-10-02 to 10-07 | all archived; useful only as alternative window reads |

### 4.3 Token-level and representation-level caches (Qwen3-VL-8B, r6 prefix unless noted)

| path | holds | corpora / coverage | producer | notes |
|---|---|---|---|---|
| `runs/20261003_m1_factorizer/r1_main/tokens/<ds>/<vid>.npz` (same in `20261003_m1_integrator/r4_main/tokens`) | `input_ids` (the exact r6 prefix token ids, e.g. 2343), `visual`, `speech` masks, `local` (n_windows × prefix tokens: which prefix tokens belong to each window), `groups` | HMM 215 + HCS 118 | Oct-3 M1 runs | **a ready map from prefix tokens to windows**; no hidden states |
| `runs/20261002_m1_attributor/r1_main_fp32_mem/tokens/<ds>/<vid>.npz` | `attribution` (signed integrated-gradient attribution of the **verdict margin** to each prefix token, through cached media values), `endpoint` (plain gradient), `media`, `visual`, `speech` masks, `local` (windows × tokens) | HMM 215 + HCS 118, 12M total | candidate (attributor), 2026-10-02 | token-level verdict attribution; archived because it did not localise |
| `runs/20261003_m1_highlighter/r1_main/tokens/<ds>/<vid>.npz` | `salience` (per image token, from the vocabulary projection of visual-token hidden states), `guidance` (windows × prefix tokens), `highlight` (windows × windows), `frame_indices`, `image_positions`, `frame_times` | HMM 215 + HCS 118 | highlighter, 2026-10-03 | |
| `runs/20261003_m1_{amplifier,projector,reinforcer,preserver}/r1_main/tokens/` | per-window Yes/No margin matrices for intervention arms (`eager`, `amplified`, `project`, `steered_*`, `preserve`, `reference`), shape windows × windows, plus token ids | HMM 215 + HCS 118 | Oct-3 | |
| `runs/20260912_tad/e0_hidden/hidden/<ds>__<vid>.npz` | **last-layer hidden state at the answer position of each window branch** (`keys` like `3_visual`, `3_speech`; `H` (n_branches × 4096) fp16) | HMM 84 + HCS 99 (only videos with both classes), 53M | `tad.py --save-hidden 1`, sc448960, 2026-09-12 | **old ASR**, SPVL-r2 prefix (same questions as r6 except the loader); the 09-13 audit noted the probe built on it had implementation limits |
| `runs/20260912_tad/prefix_probe/prefix_hidden/<ds>__<vid>.npz` | last-layer hidden of the frame image tokens in one prefix forward, pooled per 8 s window that contains a frame (`keys` = window indices, `H` (n × 4096) fp16) | HMM 84 + HCS 99, 21M | `prefix_probe.py`, 2026-09-12 | old ASR; only windows that hold one of the 20 frames |
| `data/topic_direction/qwen3vl8b/*.pt` | unit "protected group mentioned" direction per residual point (37 × 4096), body / last-position variants, 5 shuffled controls | synthetic sentence bank, no corpus data | candidate 46, 2026-10-17 | |
| `data/temporal_multikey_memory/<ds>/<vid>/metadata.json` (+ `frames/`, 2 PNG per window) | per window: **generated captions** (caption, event, dialogue, object, summary) and **4096-d last-layer mean-pooled embeddings** of each caption key and of query texts | HMM 215 + HCS 118, 4.6G | MERIT, sc448960, 2026-10-06 | embeddings of generated text, not of the window branch read |
| `data/temporal_ordered_slots_B/<ds>/<vid>/metadata.json` | per window: generated captions, three generated "search slots" (prequel / …), caption and slot embeddings | HMM 215 + HCS 118, 4.4G | ordered_slots, 2026-10-06 | |
| `data/temporal_entity_discourse_graph_handles/<ds>/<vid>/metadata.json` | per window: generated entity / action / quote ledgers, a link graph across windows, 2 PNG frames per window | HMM 215 + HCS 118, 4.0G | provenance candidate, sc474399, 2026-10-05 | |
| `data/temporal_quotation_graph/<ds>/<vid>.json` | generated quotation graph (nodes, edges, source-bound character coordinates) per video | HMM 215 + HCS 118 | quote_graph, 2026-10-05 | |
| `runs/20260926_glr/glr_pilot/predictions.jsonl` | per window with speech: **summed log-probability of the window's spoken words** under several speaker conditions (`lp` keys like `full\|assistant\|H1`, `H0_discuss`, `H0_report`, `H0_crude`, and a `none` context), token counts | HMM+HCS 333 | GLR, 2026-09-26 | sums per window, not per token; the words are not in the context |

### 4.4 Older read sets (old ASR, SPVL-r2 prefix) — usable for relative checks only

| path | holds | corpora / rows | notes |
|---|---|---|---|
| `runs/20260910_spvl/mllm/<model>/<arm>/predictions.jsonl` | the **8-MLLM family**: models q3vl-2b, q3vl-4b, q3vl-8b, q3vl-32b, q25vl-7b, internvl35-8b, llava-ov-7b, gemma3-12b; arms `full`, `joint`, `noctx`, `noframes`, `nostance`, `winonly` (text-only window read, no frames, no context), `asr` (ASR-segment windows) | HMM+HCS, 333 rows in every one of the 56 files | old ASR (pre-fix); this is what the 8-MLLM robustness checks of M2 used; no DeHate |
| `runs/20261002_verdict_analysis/analysis/per_window.csv`, `per_video.csv` | per window (21,734 rows incl. DH): z_full vs z_no-stance per branch, window text, **`gt_fraction` (from test GT)**; per video: verdict class, within AUCs, shifts | HMM, HCS (old ASR family reads q3vl-8b full / nostance), DH | contains GT columns: analysis only, never a method input |
| `runs/20260911_hvl/e0_hypothesis/predictions.jsonl` | **generated hypotheses** per video: target, form codes, evidence list (start, end, note), raw text; window reads under the hypothesis | HMM 84 + HCS 99 (173 non-empty hypotheses) | old ASR |
| `runs/20260912_tad/e0`, `e0_acts` | per window: topic read `t` ("refers to a protected group"), five-way speech-act distribution (`acts`: attacks / reports / quotes / condemns / unrelated) | HMM 84 + HCS 99 | old ASR |
| `runs/20260912_bnd/e0` | per window: `b_begin`, `b_end` ("violating content starts / ends in this window") | HMM 84 + HCS 99 | old ASR |
| `runs/20260912_cva/e0` | per window: `z_excl` (verdict with the window excluded) | HMM+HCS 333 | old ASR |
| `runs/20260912_pwc/e0b/pairs.jsonl` | 704 window pairs (pairs chosen with GT labels: `label_i`, `label_j`), carrier / generic comparison log-probs | HMM 295 + HCS 409 pairs | pair selection used GT, diagnostic only |
| `runs/20260922_til/gridA`, `gridB` | reads on two 8 s grids offset by 4 s | HMM+HCS 333 | old ASR |
| `runs/20260928_headroom/reads_{verdict8,ens8,ensq3,ensq3small,single8}/` | reads combined across the 8 family models (ensemble diagnostics) | HMM+HCS 333 | ensembles are forbidden as a method (rule 3) |
| `runs/20260912_{sdl,nga}/` | reads after LoRA self-training rounds | HMM+HCS 333 | |
| `data/omsl_v6_inputs/holistic_consistent/<ds>/scores.jsonl` | 2026-08 whole-video judge z (`hidden/` subfolders are empty) | HMM 215, HCS | dataset transcripts, 2026-08 protocol |
| `data/omsl_v6_inputs/text_unified_qwen3vl8b_chunk_scores_b1_fullcoverage.jsonl` | 2026-08 PACT per-ASR-chunk log-odds | 4 corpora (MHC retired) | |

### 4.5 Transcripts and word timestamps

| path | holds | corpora / coverage | notes |
|---|---|---|---|
| `data/asr_whisper_large_v3/<ds>/timestamped_chunks.jsonl` | **the r6 transcript**: Whisper large-v3 segment chunks `{start, end, text}` (segments can exceed 100 s), `wav_duration` | HMM 215, HCS 393 (all; test 118 among them), DH 1341 | no word times. Some chunks have a None start/end; the fixed loader fills them (previous end / audio duration). `all_splits_chunks.jsonl`: train/val/test rows for the baselines (HMM 1068) |
| `data/asr_whisper_aligned/<ds>/words.jsonl` | **word times for the r6 transcript** by CTC forced alignment (torchaudio MMS_FA, 20 ms): `words: [{w, s, e, star}]`, `status` | HMM 215 (208 aligned, 5 no_text, 2 failed), HCS 118 (all aligned); **no DH** | uoa-lab1, 2026-10-14 |
| `data/temporal_discourse_speech/<ds>/<vid>/metadata.json` (+ `audio.npy`, `observed.npy`, `alignment/block_*.npy`) | a **second Whisper large-v3 decode** (greedy, 30 s crops) with DTW word times from the official alignment heads: `words: [{text, tokens, start, end, block}]`, plus a TextTiling lexical partition and per-window LOCAL / CONTEXT word ids | HMM 215 + HCS 118 (333 metadata files, 107,136 words), 5.5G | the PROVENANCE text still says only five videos were generated; the 333 files exist (texttiling `r2_full_main` ran on them). Its text is not byte-identical to the r6 transcript. |
| `data/speaker_turns/<ds>/turns.jsonl` | pyannote 3.1 speaker turns `{spk, s, e}`, `n_speakers` | HMM 215, HCS 118, all ok; no DH | candidate 44 |
| `data/audio16k/<ds>/*.wav` | 16 kHz mono wav of every test video | HMM 215, HCS 118; no DH (DH audio only via ImageBind features) | |
| `data/wav16k_mono/<ds>/` | 16 kHz wavs for the weak-supervision baselines | HMM 1066, HCS 394 | no PROVENANCE.md in the folder |

### 4.6 Frames and frame / audio features

| path | holds | corpora / coverage | notes |
|---|---|---|---|
| `data/frames_k20/<ds>/<vid>/f{k}_t{t}.jpg` | **the r6 prefix frames**: 20 uniform frames, timestamp in the name | HMM 4300 (215 × 20), HCS 2358 (118 × 20 − 2), DH 26,818 | |
| `data/frames_w8/<ds>/<vid>/` | one frame at the centre of every 8 s window | HMM 3763, HCS 3588; no DH | "one frame per window" arm |
| `data/temporal_retrieved_kv/<ds>/<vid>/frames/*.png` + `metadata.json` | **the in-window 0.5 fps frames** (decoded PTS, dedup, missing marked) used by the in-window-frames reads | HMM 215 + HCS 118 (~29k PNG, 6.3G); no DH | |
| `data/temporal_multigrain_kv/<ds>/<vid>/frames/` | four in-window frames per window at 1/8, 3/8, 5/8, 7/8 of the window | HMM 215 + HCS 118 (6.4G) | MuKV |
| `data/retrieval_hate_repro/imagebind_image/<ds>/*.npy` | ImageBind-Huge vision embeddings of every 4 fps frame, (T, 1024) fp16 | HMM + HCS 333 | copied from Retrieval-hate |
| `data/imagebind_image_4fps/DeHate/*.npy` | same pipeline | DH 1341 | campus2, 2026-10-10 |
| `data/retrieval_hate_repro/imagebind_audio/<ds>/*.npy`, `data/imagebind_audio_2s/DeHate/`, `data/omsl_v6_inputs/audio_embeddings/` | ImageBind-Huge audio embeddings of 2 s clips, (n, 1024) fp16; text anchors "normal" / "hateful" in `data/assets/imagebind/` | HMM + HCS 333; DH 1341 | |
| `data/dense4fps_clipL336/DeHate/*.npy` | CLIP ViT-L/14-336 pooler output per 4 fps frame, (T, 1024) | DH 1341 | the HMM / HCS versions are in the Retrieval-hate repository (outside this repo) |
| `data/weaksup_1fps/{clip_b16_1fps,vit_b16_imagenet_1fps,bert_sentence_1fps,vggish_1s}/`, `data/wav2clip_1s/`, `data/wav2vec2_base_1s/` | 1 fps / 1 s features (CLIP-B16 512, ViT-B16 768, BERT sentence 768, VGGish 128, Wav2CLIP 512, wav2vec2 768) for train + val + test | HMM 1068, HCS 394, DH 6499 | weak-supervision baseline inputs; `labels/` holds train/val labels only |
| `data/clara_rationale/<ds>/Qwen/*_rationale.json` | **generated whole-video descriptions** by a two-call Qwen3-VL-8B read (20 frames, title, description, full transcript; 8 fields: objective summary, visual description, …) | HMM 1068, HCS 394 (all splits) | CLARA baseline (`experiments/20261008_baselines/clara/README.md`); no PROVENANCE.md in the folder |
| `data/clara_emb/`, `data/clara_raw/`, `data/sage_frames16*/` | CLARA window embeddings (.pt) and raw frames; SAGE 16-frame stacks | HMM, HCS | baseline caches, no PROVENANCE.md |
| `data/lexicons/{ldnoobw,davidson2017}/` | profanity list (403 terms), Davidson 2017 hate n-grams (178) | — | label-free "no model" controls |

### 4.7 Ground truth (evaluation and error analysis only, never a method input)

| path | holds | coverage |
|---|---|---|
| `data/gt_4fps/{HateMM,HateClipSeg,DeHate}.npz` | `video_ids`, `split`, `duration`, `y4` (object array of int8 4 fps label arrays), `spans`, `n_spans` | HMM 215, HCS 119 (cohort 118; `yt_NzvfkIYS5Yg` is outside), DH 1151; all test |
| `data/gt_4fps_hate_only/HateClipSeg.npz` | HCS with only the "Hateful" dimension positive (base rate .198 vs .471) | 119, frame-aligned with the main HCS array; secondary evaluation only |

### 4.8 Not cached (a GPU read would be needed)

Checked by grepping the experiment code for saves of hidden states, attention and log-probs and then looking for the outputs:

- **Hidden states of the r6 window branches under the fixed ASR.** The only branch hidden states on disk are
  `runs/20260912_tad/e0_hidden` (last layer, answer position, old ASR, 183 videos with both classes, no DeHate). No
  intermediate-layer branch states anywhere; candidate 46 kept only scalar projections (`proj_topic`, 37 per window).
- **Hidden states of the verdict / stance position** (the video-level representation). Not saved in any run.
- **Attention maps** (any layer, any head) for verdict or window branches. Never persisted. The selector / grounder /
  recycler runs saved head indices or token masks only; MuKV / ReKV / StreamingTOM `proof/` tensors exist only for CPU
  fixtures (synthetic or one video), the per-video proofs were deleted on 2026-10-09 (memory rule).
- **KV caches** of the prefix. Not saved (only fixture proofs).
- **Full-vocabulary answer distributions.** Every read stores only the Yes/No log-odds (logsumexp over Yes and No forms).
  Exceptions: the share question (six option letters + total option mass + top token) and the PWC / pairwise comparisons
  (three letters).
- **Per-token log-probabilities of the transcript under the r6 prefix** (likelihood of the words when they are not shown).
  GLR has only per-window sums under speaker conditions; generation-side has per-token losses only for copying the visible
  text (all near 0).
- **Multiple-question reads per window** on the fixed-ASR r6 prefix (topic, speech act, begin / end, endorse per window):
  the topic / act / begin-end reads exist only with old ASR on 183 videos (§4). Per-window endorsement was never read.
- **Free-text stance or rationale under the r6 prefix.** The stance turn is the single word "Yes"/"No". Generated
  descriptions exist only from the CLARA baseline (separate two-call read) and HVL (old ASR, 183 videos).
- **DeHate** has no in-window-frame reads, no aligned word times, no speaker turns, no 16 kHz wav, no per-token caches, no
  Omni reads, no family (8-MLLM) reads; it does have r6 reads, the modality-only / joint / no-stance / sequential reads,
  DVD video-level reads, share-question reads and segment reads.
- **Per-modality M2 posteriors and per-video EM states** are not stored but are a ~2 min CPU rerun of `twolevel_r2.py`.
- **A no-stance read under the fixed ASR for HateMM / HateClipSeg** does not exist (only old-ASR `mllm/q3vl-8b/nostance`
  and DeHate `reads_nostance`).
- **Context-free single-window reads under the fixed ASR** do not exist; the only one is old-ASR `mllm/q3vl-8b/winonly`
  (text only, no frames).

Cost reference for a new full read pass (from run records): a full window-read set for HMM+HCS took 3.5–9 min on one
GPU (`runs/20260928_infer/main/reads_*`, 333 videos; DeHate 0.8–1.2 s per video); one extra video-level question on the
r6 prefix took about 2 s per video including the prefix forward (`runs/20261025_m3_coverage/main/reads.jsonl`,
`seconds`); the in-window-frames visual arm about 18 min for HMM+HCS (StreamingTOM README).

Deleted on 2026-10-09 (user-approved disk cleanup; records, predictions and metrics remain):
`temporal_interval_witness`, `temporal_optimal_transport` (vision
saliency + projector/DeepStack), `temporal_evidence_program_handles`, `temporal_token_fusion`, `temporal_spatial_search`,
`temporal_query_events`, `temporal_time_tools*`; KV/feature proof tensors of ReKV (103 GB), StreamingTOM (143 GB), TextTiling
(28 GB). Records, predictions, metrics remain.

### 4.9 Additional per-branch caches from the October M1 candidates (r6 reader, fixed ASR, HateMM 215 + HCS 118)

The inventory sweep above lists only some of these. All were checked by listing files and parsing one record.

| path | holds | notes |
|---|---|---|
| `runs/20261003_m1_contraster/r1_main/checks.jsonl` (75 MB, 333 rows) | Per window branch (visual and speech; 13,939 branches): `label_logits` (18 rows = layers 2, 4, …, 34 plus the final layer, × 12 Yes/No token ids), `log_normalizers` (full-vocabulary log-sum-exp per row), `js` per layer, `native_margin`. `native_global` equals the base_gridA z_video | **A logit-lens depth trajectory of every r6 window read**; only the dynamic DoLa choice was ever scored (§3 O2) |
| `runs/20261003_m1_eraser/r1_main/checks.jsonl` (333 rows) | Per window: verdict margin with that window's frames removed (`minus_visual`) and with its words removed (`minus_speech`), removed frame and word ids, remaining prefix tokens | Leave-one-window-out verdicts (§3 P2) |
| `runs/20261002_m1_attributor/r1_main_fp32_mem/tokens/<ds>/<vid>.npz` | Per prefix token: signed integrated-gradient attribution of z_video, endpoint gradient, media / visual / speech masks, `local` (windows × tokens) | Verdict attribution (§3 P2) |
| `runs/20261003_m1_explorer/{r1,r3,r4}_main/details/<ds>/<vid>.json` | Per visual window: `initial_z`, binary entropy, **`initial_prior` = attention of the visual question's last token over the 20 prefix frames** (all 36 layers, pre-RoPE), acquisition rounds with new-frame PTS and margins; decoded source PTS lists | Branch attention over frames (§3 O3) |
| `runs/20261003_m1_visual_contrast/{r1_main,r2_main}/checks.jsonl` | Per visual window: clean margin, noised margin (R1 whole prefix, R2 own window), frame count | |
| `runs/20261004_m1_lattice/r2_full_main/records/*.json` | Per speech window: 5-best Whisper hypotheses of the window's own 8 s audio (token-id paths, beam weights) | The source of a "onebest" control (§3 O5). `data/temporal_speech_lattice` was deleted; the text can be recovered by decoding the ids |
| `runs/20261004_m1_acoustic/r1_soft_main/details/*.json` | Per word: Whisper character-alignment support for each window; per window: selected words, word mass | `data/acoustic_path_support` deleted |
| `runs/20261004_m1_tree/r1_full_main/details/*.json` | Qwen captions of clustered 1 fps frames with times and parent links; per-window packets | |
| `runs/20261002_m1_grounder/r1_full/<arm>/predictions.jsonl` | Window reads for the late / early / all_local / shifted / verdict_only arms | |
| `runs/20261007_m1_streamingtom/placement_controls_main/records/<ds>/<vid>.json` | Per window: native_visual, native_speech, stored_local_clean, local_clean_replay, adjacent_local, prefix_local, adjacent_native, LOCAL frame times, native_inside | **The five-read table for any binding or selection check** (§3 O1) |

### 4.10 Shared code that designers can reuse (in `src/`, tracked unless noted)

| file | what it does |
|---|---|
| `src/mllm_judge.py` | The frozen judge: prompt material, prefix KV cache, isolated / sequential branches, Yes/No margins, option log-probs (`cached_choices`), greedy generation, teacher-forced log-probs |
| `src/stance_cache.py` | Builds the native r6 prefix + stance cache with independent local branches |
| `src/prefix_questions.py` (2026-10-25) | Several whole-video questions on one r6 prefix, each in an isolated branch; also builds per-segment prefixes (candidate 49) |
| `src/window_token_regions.py` | Exact frame / transcript token spans of each window inside the prefix (from Grounder) |
| `src/pre_rotary_memory.py`, `src/source_key_mask.py`, `src/sparse_visual_prefix.py`, `src/source_image_branch.py`, `src/cached_image_features.py` | Pre-rotary key / query handling, key masks, and new-image branches on the cached prefix (from ReKV / StreamingTOM) |
| `src/window_level.py` | Window-level pre-check helpers. **It reads test GT**, so analysis only |
| `src/temporal.py` | Explicit-duration chain kernels shared with `twolevel_r2.py` |
| `src/video_inputs.py` | Manifest, ASR loader (fixed), 20-frame paths, proportional window text |

## 5. The current r6_bma computation as implemented

Paths are relative to `/home/jehc223/Hate-follow-up`. Line numbers are from the working tree at commit 92a3b7f
(2026-10-25).

### 5.0 Which run is "the method"

- Reads (M1): `runs/20260926_glr/base_gridA/predictions.jsonl` (333 lines: HateMM 215 + HateClipSeg 118 test videos).
  Produced 2026-09-26 on lab-server (host sc448960) by `experiments/20260922_til/til_measure.py` with
  `--window-seconds 8 --window-offset 0` (config `runs/20260926_glr/base_gridA/config.json`; transformers 5.15.1,
  torch 2.11.0, conda env HateVLM). DeHate reads: `runs/20260927_dehate_external/reads_gridA` (same script).
- Decode (M2 + M3): `experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib
  --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2` (launcher
  `experiments/20260926_twolevel/launch/run_r6.sh`, line 7 and 9). Output `runs/20260926_twolevel/r6_bma/` =
  `final_m2/`. Metrics: HateMM .8971 / .6942 / .7508, HateClipSeg .7168 / .6711 / .6373; DeHate
  `runs/20260927_dehate_external/r6_bma/metrics.json` .7011 / .1582 / .6431.
- **Discrepancy to know about (key calibration).** STATUS.md and the twolevel README describe `--key calib`. The user
  ruled on 2026-10-07 that the calibration is useless and the paper uses `--key raw` (memory file
  `feedback-calibration-removed.md`; Overleaf commit e3d885d). The raw-key run is
  `runs/20260926_twolevel/final_rawkey/metrics.json`: HateMM .8955 / .6883 / .7508, HateClipSeg .7137 / .6666 /
  .6373; DeHate `runs/20260927_dehate_external/r6_rawkey/metrics.json` .6994 / .1568 / .6431. Within is identical
  (the key is constant within a video). The M3 team round of 2026-10-25 used the `calib` numbers as its G1
  reference. Designers should state which reference they compare to; the user does not want a "calibrated vs raw"
  column in reports.

### 5.1 M1: the per-window read (frozen Qwen3-VL-8B-Instruct, bf16, greedy, no generation)

Code: `experiments/20260922_til/til_measure.py` (`score_video`, lines 51-128), `src/mllm_judge.py` (prompt
material and the cache path), `src/video_inputs.py` (ASR, frames, window text).

Inputs per video:
- 20 frames `data/frames_k20/<ds>/<vid>/f{k:02d}_t{t}.jpg`, sampled at t_k = (k + 0.5) D / 20
  (`data/frames_k20/PROVENANCE.md`; selection in `src/video_inputs.py:52-61`). Each frame resized under the
  pixel cap 65536-100352 (`src/mllm_judge.py:19-23`), 91 visual tokens per frame on Qwen3-VL (base_gridA
  `verify.json`).
- Whisper large-v3 chunk transcript `data/asr_whisper_large_v3/<ds>/timestamped_chunks.jsonl`, loaded by
  `load_asr` (`src/video_inputs.py:23-49`) with the 2026-09-26 fix (an untimed last chunk takes the previous end /
  the audio duration). Chunks average 11-16 s; there are no word timestamps in this cache.
- Window text: `window_text` (`src/video_inputs.py:69-87`) slices each overlapping Whisper chunk **proportionally by
  word count** (not by word timestamps). This is the source of the "boundary text spill" error class
  (`experiments/20261014_within_diagnosis/README.md`).

Prefix (one KV cache per video; `prefix_messages`, `src/mllm_judge.py:153-179`):
1. System: `You are a content moderation analyst. Answer based strictly on observable evidence.` (line 30)
2. User content, in order:
   - `Observe this video. Below are 20 frames sampled from the video, each preceded by its timestamp. The full
     transcript with timestamps follows the frames.`
   - for each frame: `[t=2.4s]` then the image;
   - `Transcript:` then one line per chunk, e.g. `[0.0s-94.9s] text` (`transcript_block`, `src/video_inputs.py:90-93`;
     `(no speech detected)` when empty);
   - `Based on this platform's hate speech policy, the following types of content are prohibited:` + the nine
     YouTube rules (lines 31-39, verbatim):
     1 Encourage violence against groups based on protected status; 2 Incite hatred against groups based on
     protected status; 3 Dehumanize groups by comparing to non-human entities; 4 Praise or glorify violence against
     protected groups; 5 Use slurs and stereotypes that incite hatred (including through humor, irony, or coded
     language); 6 Claim groups are inferior based on protected status; 7 Promote hateful supremacism; 8 Make
     conspiratorial claims about groups based on protected status; 9 Deny or minimize major violent events;
   - the reader instruction `READER_PRAG` (lines 40-49, verbatim): "Judgment mode: full reading. Judge what the video
     communicates as a whole: both what is explicitly said or shown and what is conveyed beyond the surface. Consider
     coded or euphemistic references, wordplay and homophones, ironic or mocking framing, how the title reframes the
     visuals, the joint effect of images, audio, and text, and how the referenced group or an informed viewer would
     receive the message. A rule is violated if either the explicit content or the conveyed meaning violates it."
   The prompt material is the frozen 2026-08 text (header of `src/mllm_judge.py`).

Video verdict (`til_measure.py:68, 76`): a new user turn with `VIDEO_QUESTION` (`src/mllm_judge.py:50-51`, verbatim):
`Does this video contain content that violates any of the above rules?` blank line `Answer "Yes" or "No".`
z_video = Yes/No log-odds at the first assistant position: logsumexp of the fp32 lm_head logits over the first
tokens of {Yes, " Yes", yes, " yes", YES, " YES"} minus the same over the No variants (`label_ids` lines 114-120,
`margins_fp32` lines 129-134). The verdict tokens are written into the shared cache (`in_place=True`).

Stance turn (`til_measure.py:77-81`): **not generated text**. The assistant turn is the single word `Yes` if
z_video > 0 else `No`, plus the template's end-of-turn tokens (`answer_ids`, `src/mllm_judge.py:261-275`), appended
to the cache (`extend_cache`). So every window branch sees: prefix + verdict question + the model's own one-word
answer.

Window branches (`til_measure.py:84-101`): fixed 8 s grid [8k, 8k+8) clipped to the duration (`grid_windows`,
lines 39-48). For each window, on a **deep copy** of the cache (isolated: no branch sees another branch,
`cached_margin(in_place=False)`, `src/mllm_judge.py:308-315`), a new user turn from `yesno_question`
(`src/mllm_judge.py:54-68`), verbatim:
- visual branch: `Consider only window {i+1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video. Look only at the
  frames whose timestamps fall inside this window and judge the visual content alone (imagery, gestures, symbols,
  on-screen text), ignoring the speech.` blank line `Is THIS window one of the segments where visual content that
  violates the above rules occurs?` blank line `Answer "Yes" or "No".`
- speech branch (only if the window text is non-empty, line 92): `Consider only window {i+1} of {n}, from
  {t1:.1f}s to {t2:.1f}s of this video. Judge only what is spoken in this window: {window text}` blank line `Is THIS
  window one of the segments where speech that violates the above rules occurs?` blank line `Answer "Yes" or "No".`
- Read: the same Yes/No log-odds as the verdict. Stored per window as `z_visual`, `z_speech` and
  `z = max(z_visual, z_speech)` (line 102) in `extra.windows`.
- Note: the visual branch shows no new image. It asks the model to find "the frames whose timestamps fall inside
  this window" among the 20 prefix frames; 27 % / 34 % of windows (HateMM / HCS) contain no prefix frame
  (`experiments/20261007_m1_streamingtom/README.md`, placement section). This lookup is what the
  `adjacent_native` re-show removes.
- Cost: one prefix prefill (about 2,300 tokens for a 95 s HateMM video, 1,820 of them visual) plus one short
  branch forward per (window, modality); 2 "calls" recorded per video.

### 5.2 M2: label-free segment inference (for reference)

`experiments/20260926_twolevel/twolevel_r2.py` and `twolevel.py`:
- window to 4 s cells, each 8 s window observes a pair of cells (`twolevel.py:45-64`);
- each modality's reads -> normal score of their rank within the corpus (`twolevel_r2.py:716-726`); the key
  keeps the raw reads (comment line 718);
- per corpus EM (`em`, line 470 ff.): verdict z_video ~ N(m_V, tau^2); per-modality Gaussian emissions with a
  shared variance, mu00 (non-violating video), mu10 = mu00 under `--noleak`, mu11 (hateful);
- per modality one chain; segment shape k = 2 windows / 4 s cell = 4 sub-states (`--min-windows 2`); mean gap and
  hate lengths averaged per video over a 6 x 6 log grid on [two windows, the video's length], prior weight
  proportional to gap x hate (`video_terms`, lines 342-382, prior lines 352-355);
- OR fusion: p_t = 1 - prod_m (1 - P(chain m hateful at t | V = 1, reads)) (lines 795-806).

### 5.3 M3: video key and fusion

- Key: K_v = z_video + mean over windows of z_w, where z_w = max(z_visual, z_speech) **raw** log-odds
  (`intercept`, `twolevel.py:360-361`; computed at `twolevel_r2.py:809`).
- `--key calib` (STATUS description): logit P(V = 1 | K) = a K + b from a two-component one-dimensional Gaussian
  mixture with shared variance fitted by EM on the corpus's K values (initialised at the 10th / 90th percentiles,
  2000 iterations; `key_calibration`, `twolevel_r2.py:423-438`; applied line 820). Fitted on the main reads: a = .357, b = .399
  (HateMM); a = .341, b = 3.508 (HCS) (`runs/20260926_twolevel/r6_bma/params.json` `key_ab`). It is a monotone map per corpus,
  so it changes pooled only by changing the spread of keys relative to the rank term. `--key raw` (paper version)
  uses K itself (line 826).
- Frame score (`twolevel_r2.py:828`): score_t = key_v + centred_rank(logit p_t mapped to 4 fps frames), where
  `centered_rank` (`twolevel.py:346-351`) = (average rank - .5) / n - .5, minus its mean, so it lies in
  (-.5, .5). The key is constant within a video; within-video order comes only from M2's p_t.
- Consequences designers should keep in mind:
  - the key's unit is raw logits (sd of z_video 12.2 / 7.5 across videos; `experiments/20261025_m3_coverage/
    README.md` §2.2), while the rank term spans width 1, so pooled metrics are almost a lexicographic order:
    videos by key first, then frames by rank (the twolevel README §11 notes "the key's unit vs rank width 1 is a
    hidden constant"; scaling K down to 1/8 raised pooled on both corpora, §11.5, not adopted because no
    label-free rule sets the scale);
  - `--arm full` instead outputs log P(V | K) + log p_t and intervals (F1@.3 .325 / .246).
- Ablations (README §20.1): video term pooled ROC -.325 / -.147; within term within -.251 / -.137; calibration
  -.002 / -.006 pooled; only z_video in the key (no window mean) -.0135 / -.046 and -.037 / -.048 pooled
  (`experiments/20261025_m3_coverage/README.md` §12, "只用 z_video" row).

## 6. Prior work already searched by the rule-4 reviews

The full per-review table (review file, verdict, works searched, closest work, advisories that predicted the failure) is in `_parts/reviews.md` §1. The deduplicated list and the claim limits follow.

### 6.1 Prior work named across the reviews, by theme

Format: work (venue or arXiv id) - which reviews named it. Review names are the candidate names above; "final" = the
2026-09-27 final novelty review; "TIL", "jury 10-04/10-05/10-06" as above.

#### 6.1.1 MLLM read-out, probing and internal intervention
- Yes/No token read-out per clip: Probe-VAD 2609.17211 (final, pairwise 42, segment presence 49); "Your VLM Already Knows When" 2608.08315 (final, generation side 45); "A VLM Answer Is Not an Anomaly Score" 2608.21244, rank compression (final, TIL); CEAVAD 2608.09908, Qwen3-VL-8B two-token posteriors (final); definition blindness 2607.20780 (final).
- Option-distribution read-outs: G-Eval; RJScore in Video-SafetyBench 2505.11842; Q-Align; CORDIAL 2609.29807; proportional odds (McCullagh 1980); CLUE (CVPR 2025) (coverage 47).
- Attention steering / masking by layer or head: PASTA 2311.02262, AutoPASTA 2409.10790, TRACE 2607.11436, GuardAlign 2602.24027 (Grounder); Gaze Heads 2606.14703, Localization Heads 2503.06287, IGAR 2603.06001, Retrieval Heads Meet Vision 2608.27417 (Selector); PAI 2407.21771 (Amplifier); visual attention sink 2503.03321 (Recycler); VGA 2511.20032 (Highlighter); FutureMask 2505.18605 (Integrator); ACG 2601.13707 (Projector); MAD-RAG 2602.00344 (Preserver); PAS 2511.10979 (Stabilizer).
- Contrastive decoding and depth read-outs: DoLa 2309.03883, LayerCD 2509.25177, Risk Awareness Injection 2602.03402 (Contraster); VCD 2311.16922, Visual Evidence Prompting (ACL 2025), MMSafeAware (ACL 2025) (visual contrast, Amplifier).
- Residual steering and concept removal: VISTA 2502.03628 (Reinforcer); FBHM 2605.31349 (Reinforcer, topic removal 46); Arditi 2406.11717, RepE 2310.01405, ROBOSHOT 2309.04344, INLP, LEACE 2306.03819, Representation Surgery 2402.09631, 2604.18901, SARSteer 2510.17633, refusal-direction transfer 2606.31876, V-DEAL 2607.21151, CASteer, UniSteer 2605.30076, FairSteer 2504.14492, SAE toxicity steering 2505.14536 (topic removal 46).
- Latent optimization at test time: Unsilencing Visual Latents 2605.02735 (jury 10-04, Latents); SCANNER 2602.00132 TTA in hateful video (jury 10-04, Latents, final).
- Attribution: Integrated Gradients (ICML 2017), Chefer 2012.09838, Astorino CLiC-it 2023, hateful-memes causal analysis 2308.11585 (Attributor); SHAP 1705.07874, Play Fair 2011.12372, Explaining by Removing (JMLR 2021), Text or Image? (EACL Findings 2024), Sci Reports 2025 hate SHAP (Allocator); occlusion 1311.2901, AttriBoT 2411.15102, video occlusion 2207.12859, VisualBERT memes 2410.13488 (Eraser).
- Generation-side and refusal signals: MULI 2405.18822, Prefix Probing 2512.16650, Emulated Disalignment 2402.12343, shallow alignment 2406.05946, self-diagnosis 2103.00453, OR-Bench 2405.20947, CISPA 2603.11914, ParaDeHate 2506.01484, MaRCo 2212.10543, CF-Detox 2405.09948, AraDetox 2608.22894, ARCADE 2603.21298, GPT-4o-mini memes 2509.13608 (generation side 45).

#### 6.1.2 Context-conditioned reading, own verdict in context, verification
- Decide at video level, then localize under it: T3AL 2404.05426, FreeZAD 2501.13795, Liberatori 2605.22201, STPN 1712.05080, UntrimmedNets 1703.03329 (final, stance share 48); hurdle / two-part models Cragg 1971, Mullahy 1986 (stance share 48).
- Global context conditions clips: PANDA 2509.26386 (final); ESOM 2604.07772 (final).
- Chain-of-Verification 2309.11495, cited as the opposite choice to the stance turn (final, Verification 27, stance share 48).
- Opposing hypotheses: MARS 2601.15115, RAMF 2512.02743 (Marginalizer STOP and nearly every review).
- Retrieval and verification in hateful video: MATCH (TCSVT 2026, jianlang.org/papers/MATCH.pdf) and MAESTRO (DSTA report; full text never accessible) (Explorer, Eraser, juries, Interval witness, Program, Provenance, Quote graph, Verification, grounded 41).
- Citation verification: GroundedVQA 2608.15574 (grounded 41).
- Severity + label jointly: BingoGuard 2503.06550; Filter-And-Refine 2507.17204 (stance share 48).
- Self-consistency 2203.11171 (Marginalizer).

#### 6.1.3 KV, prefix, memory and streaming video LLMs
- Shared prefix and isolation: Hydragen 2402.05099, SGLang 2312.07104, ChunkAttention 2402.15220, Prompt Cache 2311.04934, vLLM prefix caching, PCW 2212.10947, APE 2502.05431, SingGuard 2606.22873, InvariRank 2604.27599, T3S 2511.17945, 2605.20194, Free-MoRef 2508.02134 (final); Block-Attention (ICLR 2025) (Factorizer); SafeWatch 2412.06878 policy-block encoding (final, Factorizer, generation side 45).
- Video KV reuse and retrieval: ReKV 2503.00540 (ICLR 2025), STTM 2507.07990 (final, jury 10-06); MuKV 2605.22269, StreamingTOM 2510.18269 (jury 10-06).
- Token compression: TTF 2605.07355, OTT-Vid 2605.11803 (jury 10-05).
- Frame acquisition / trees: EcoFrame 2608.03918 (Explorer); VideoTree 2405.19209 (jury 10-04, Tree, Interval witness).

#### 6.1.4 Training-free temporal grounding and localization
- TFVTG 2408.16219 (final, jury 10-06); Moment-GPT 2501.07972, OZ-TAL 2605.09976 (final, TIL, coverage 47); VTimeCoT 2510.14672, V* 2312.14135, VideoAgent 2403.11481, MERIT 2608.07663, Q2E 2506.10202, VideoEvent (LREC 2026), TIE/RoTE 2605.10543 (jury 10-05); BaGLM 2510.16989, VideoGEM 2503.20348 (jury 10-06); Semantic Event Graphs 2601.06097 (Provenance); ViperGPT 2303.08128, VPD 2312.03052 (Program).
- Pairwise / relative judgments: PRP (Qin et al. 2023), Oak et al. 2501.13977, X-CoT 2509.21559, Compare2Score 2405.19298, GenArena, HodgeRank, VideoJudge 2509.21451, AoT-PsyPhyBENCH 2510.26241, TimeCatch 2608.23474, Bridge-STG 2604.08014, VeRVE 2601.12193, 2607.19027, 2605.21954 (pairwise 42).
- Speech / audio side: Whisper internal aligner 2509.09987 (Acoustic, jury 10-04); Lattice Transformer SLU (ASRU 2019), WCN 2401.02921 (Lattice); TextTiling (ACL 1994), Whisper-Streaming (2023), Park & Glass (ASRU 2005), Jacobs et al. (Interspeech 2023) (jury 10-06); LiveCC (CVPR 2025), HowToCaption 2310.04900, In-Sync 2604.22817, TiTok 2610.09408, Auto-AEG 2607.04383, AVTrace 2609.19991, AVID 2604.13593, DAVE, AV-SyncBench 2607.00726, Dolphin (ICLR 2025) (audio 43).
- Quotation / coreference: Michel 2406.11380, Vishnubhotla (ACL 2023 short), TikTalkCoref 2504.14321 (Quote graph).

#### 6.1.5 Training-free video anomaly detection
- LAVAD 2404.01014, AnomalyRuler 2407.10299, VERA 2412.01095, Holmes-VAD 2406.12235, Holmes-VAU 2412.06171, EventVAD 2504.13092, VADTree 2510.22693, AnyAnomaly 2503.04504, Flashback 2505.15205, Vad-R1 2505.19877, HiProbe-VAD 2507.17394, CoReVAD 2605.23116, QVAD 2604.03040, GtS 2608.11260, NOVA 2609.06360 (final); MM-VAD 2603.13374 (audio 43, pairwise 42); SphereVAD, ARMS / AR-MIL (coverage 47).
- Within-video metric: 2608.21854 (Song & Lee), 2608.11985, 2211.15597 (final, pairwise 42).

#### 6.1.6 Hateful video and hateful memes
- Localization: LELA 2602.09637 (only training-free hate localizer; per-frame captions to LLM, max over modalities); MultiHateLoc 2512.10408; TANDEM 2601.11178 (trained, plus zero-shot Qwen3-Omni baseline); SafeLens (AAAI-26 demo); HateClipSeg 2508.01712 (ActionFormer baseline).
- Detection: MARS, RAMF, CLARA 2608.15905, HVGuard (EMNLP 2025), MM-HSD 2508.20546, ImpliHateVid (ACL 2025), IARE 2606.11953, SCANNER 2602.00132, MoRE, LEAF (ACL Findings 2026), SAGE (ACL 2026), MATCH, MAESTRO, WWW'26 Yadav & Singh evidence attribution (doi 10.1145/3774905.3796488; Zenodo code read in the Allocator review), Few-Shot Detection of Hate Videos (doi 10.1145/3733813.3764370), MultiHateGNN (BMVC 2025), Robust Framework 2502.07138, Hate-LLaMA, Buffalo capstone 2025-01, GuardReasoner-Omni 2602.03328, Sci Reports 2025 SHAP framework.
- Problem statements: temporal label noise 2508.04900, distributed implicit harm 2609.00206, harmful-content omission in video LLMs 2508.10974 (final).
- Speech toxicity without ASR: PolySpeechTox / SoftPrompt-TSD, MuTox, ToxiAlert 2605.15984, An et al. 2408.06065 (audio 43).
- Datasets: HateMM 2305.03915, MultiHateClip 2408.03468, DeHate, HarmVideoBench, DIH, VHD11K (coverage 47), EgoSafetyBench (segment presence 49), TikTok audit 2608.17583.
- Memes: FBHM 2605.31349, MMSafeAware, Text or Image?, VisualBERT memes, UCA (LREC 2024), GPT-4o-mini memes, ARCADE.
- Text-hate debiasing: Kennedy 2005.02439, Xia 2005.12246, Sen (NAACL 2022), Schafer (WASSA 2024), NLPGuard 2407.01697, 2109.10441, Wiegand (NAACL 2019), Yoder 2210.10839 (topic removal 46).

#### 6.1.7 Weak-supervision label models, quantification, group testing
- Only indirectly: quantification with classify-and-count (2604.21549 and Indeed work; coverage 47); Neural Group Testing 2011.10704, Dorfman / Thompson estimators (segment presence 49); random-effects segmental HMM, Kim & Smyth (JMLR 2006) (revisable prior). No review searched Dawid-Skene or Snorkel-type label models; DIRECTIONS records the 2026-08 LESS attempt (Dawid-Skene across views) without a review file.

#### 6.1.8 Calibration and score fusion
- Gao & Tan (ICDM 2006), Brummer & Garcia-Romero 1311.0707, Prototypical Calibration 2205.10183, Batch Calibration 2309.17249, Contextual Calibration 2102.09690, Generative Calibration (EMNLP Findings 2023), FUSE 2604.18547 (label-free fusion of judge scores on different scales) (final).

#### 6.1.9 Temporal statistical models (M2, for completeness)
- Sun & Cai 2009, Wang & Wang 2024, nonparanormal, McCaw 2020, Efron 2004, Chidambaram 2001.00057, Hayashi (ICASSP 2017), Richard 2016/2017, NN-Viterbi 1805.06875, Yu 2010, Johnson 2005, BOCPD 0710.3742, Turner 2009, Wilson 2010, sticky HDP-HMM, HDP-HSMM 1203.1365, RJaCGH, Jaynes 1968, factorial HMM, Touloupou 2019, Kang et al. (JMIR AI 2025), Ragu & Jonelagadda 2605.12838 (final); TASL, MLLM4WTAL 2411.08466 (TIL).

### 6.2 What the final review and the Codex consultations said about M1 and M3

#### 6.2.1 M1 (window reads)
- **Can claim** (final §6.1): "each fixed window is read against one cached context that holds the whole video (timestamped
  frames, timestamped transcript, policy) and the model's own verdict, in isolated per-modality Yes/No branches", first for
  hateful video and training-free video localization as far as searched. Evidence: reading windows alone costs within
  .021/.071 and pooled ROC .035/.080 (`robust/winonly_r3` vs `robust/full_r3`, pre-ASR-fix cache).
- **Cannot claim**: isolation mask, prefix caching, position restart (systems prior art: Hydragen, Prompt Cache, PCW, APE,
  SingGuard, InvariRank, T3S, ReKV, STTM); dual branches as a gain (HateMM within +.009 when removed); transcript context as a
  gain (HCS about 0); frames (an input); fixed 8 s windows (standard); Yes/No read-out (Probe-VAD, 2608.08315, 2608.21244);
  "the stance turn makes windows more discriminative" (it shifts all windows by +2.39/+2.08 logits in Yes videos, order Spearman
  .93/.92); "first training-free hate localization" (LELA).
- Closest structural precedent: ESOM 2604.07772 (Qwen3-VL-8B, KV-cached policy prefix reused for every window; no video in the
  prefix, no verdict). Closest for the stance idea: T3AL / FreeZAD / STPN.
- paradigm_framing §2 adds: do not claim "video level first, then local" (T3AL) or training-free MLLM localization (LELA, LAVAD).

#### 6.2.2 M3 (video key + fusion; "M4" in the final review)
- **Cannot claim anything as new**: the calibration logit P(V=1|K) = aK + b from a shared-variance two-Gaussian mixture is
  identical to Gao & Tan 2006 and Brummer & Garcia-Romero 2014 (and Prototypical Calibration); classify-then-localize is STPN /
  UntrimmedNets; video decision then within-video centring is T3AL. Calibration fails 14g as a gain (PR +.007/+.004). The "no
  key" and "no within term" ablations hold by construction only.
- Allowed framing: "the scoring rule matched to the two metric families" (pooled = order between videos, within = order inside
  a video; cite 2608.21854), with the product as the interval rule.
- Candidate 47 review: readout of a share by option-distribution expectation is not new (G-Eval, RJScore, Q-Align, CORDIAL); only
  "the estimated quantity (share) and its link to pooled ranking" could be claimed, subject to 14g. Candidate 48 review: if the
  conditioning is claimed, its removal must cost >= .01 pooled on both corpora; cite hurdle models, T3AL, BingoGuard, MARS.
  Candidate 49 review: segment presence reads are classify-and-count too; cite SafeLens and multi-clip averaging (TSN /
  SlowFast, not searched).
- Codex round 1: "further unsupervised duration fitting or probabilistic video-level recombination: repeated failures, with no
  new information source" - reject. The M3 round of 2026-10-25 confirmed this (composition room about +.003-.005).

#### 6.2.3 Reviewer suggestions for mechanism-level redesign that were never acted on (as far as the review record shows)
1. **Codex round 1, rank 2: dense temporal reader distilled from one frozen teacher with known temporal correspondences**
   (cropped / resampled views with coordinate maps, consistency loss), replacing window questions and the duration model at
   deployment. Withdrawn by Codex in round 2 ("no credible source of better supervision") without any pilot. Note it involves
   training a student, which the current "training-free" framing would have to accommodate.
2. **Codex round 2, the one constraint change: a small amount of segment-level supervision** to teach mention vs endorsed
   attack. Not acted on; it breaks the label-free claim.
3. **Final review §8 gaps, still open per STATUS line 339 and 543**: (1) a same-backbone training-free comparator, either LELA
   re-implemented with Qwen3-VL-8B or Qwen3-VL-8B per-window Yes/No reads alone plus Gaussian smoothing (the `winonly` reads
   exist); the user deferred LELA rerun to before submission (STATUS 543); (2) re-run the M1 component ablations on the final
   reads (fixed ASR) with paired bootstrap; isolation has only a round-1 number; (6) describe the single-new-video procedure
   (stored reference set for ranks, EM, key calibration), since all three are fitted transductively on the test corpus.
4. **Codex round 1 diagnostic**: "the verdict is placed before the window questions, so global/local agreement may partly come
   from the conditioning." Partly addressed (nostance ablation; Explorer R4 dropped the verdict dialogue for added reads), never
   tested as a mechanism.
5. **10-06 jury: five PASSed M1 candidates never run**: TFVTG-style dynamic/static transition source grounding (rank 9),
   BaGLM prerequisite-progress source memory (rank 8), actor-object transition witness paths with two-frame identity (rank 5),
   LocalAgreement acquisition-horizon speech reader (rank 7), segmental-DTW acoustic recurrence with two-ended context
   disambiguation (rank 6). MuKV 40 was prepared and paused by the user (2026-10-07). All of these are evidence-acquisition
   designs (they change which sources a fresh read sees), which rule 5 counts as inputs unless the selection mechanism itself
   passes 14g; the ReKV / StreamingTOM experience is that the gain went to in-window frames.
6. **Grounder review**: required an `early` control (same mask size at the start of the stack) to support any "timing of
   access" claim. Auditor check: it ran (Grounder arms late / early / all_local / shifted / verdict_only, §1.4).
7. **Topic removal 46 review**: suggested a matched shuffled-label direction control (it was added) and a direction fitted only
   on non-attack carriers (the full-bank direction contains "this is now a violation" as well as "a group is referred to").
   Auditor check: adopted before the run; the direction was fitted on the even-numbered non-attack carriers only (`archive/experiments/20261017_m1_topic_removal/README.md` §2).
