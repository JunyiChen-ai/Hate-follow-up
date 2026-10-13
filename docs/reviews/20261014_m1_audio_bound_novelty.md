# Proposal review (rule 4): candidate 43, audio-bound speech evidence, 2026-10-14

Reviewer: one independent agent (same model as the main session), before any GPU run. Commit: `93311fc`
(branch `follow-up`). Read: `RESEARCH_ITERATION_RULES.md` rule 4 and rule 5 (and rule 3 for the ensemble check,
rule 8 and rule 14(f) for the comparator advisories); the proposal `experiments/20261014_m1_audio_bound/README.md`
(all eight sections); `experiments/20261014_within_diagnosis/README.md` (the diagnosis that motivated it);
`archive/experiments/20261004_m1_acoustic/README.md` lines 1–60 (candidate 20); the previous record
`docs/reviews/20261013_m1_pairwise_evidence_novelty.md`. Repository check: a case-insensitive grep for "omni",
"raw audio", "audio clip", "audio-language", "SALMONN", "Qwen2-Audio", "Audio Flamingo" over `experiments/*/README.md`,
`archive/experiments/*/README.md`, `research-wiki/*.md`, `docs/reviews/*.md` hits only the proposal itself and
`experiments/20261008_baselines/README.md` (the ZS-ImageBind audio baseline row). No earlier experiment in this
repository gave raw audio to an MLLM reader. This file is the only file written by this review. No hashes were
computed. No GPU job was run.

## What was searched (2026-10-14, WebSearch, 16 queries; 11 abstracts or full texts opened with WebFetch)

| query | closest hit (one line each) |
|---|---|
| Qwen2.5-Omni OR Qwen3-Omni hate speech detection raw audio toxicity moderation | GuardReasoner-Omni 2602.03328 (opened): trained SFT + RL guardrail for text / image / video / audio, per-item harmfulness; Qwen3-Omni-30B-A3B-Instruct appears as a baseline (83.6 % on audio prompt harmfulness); no localization inside an item. ToxiAlert 2605.15984: speech toxicity dataset with Qwen2-Audio and Gemini-2.5-Flash as general baselines |
| audio language model toxicity detection speech without ASR transcription 2026 polySpeechTox | PolySpeechTox / SoftPrompt-TSD (Findings of ACL 2026): a frozen audio language model prompted end to end for utterance-level toxicity in 53 languages, no ASR step; MuTox (Meta): zero-shot audio toxicity, utterance level. Neither is video, neither localizes inside an item |
| speech hate speech localization timestamp span detection audio temporal | An et al. 2408.06065 (opened): "explainable audio hate speech detection", an end-to-end audio model that outputs the hateful time frames, trained on synthetic speech, audio only, no video, no LLM; TANDEM and MultiHateLoc (below); LENS-DF 2507.16220 localizes deepfake speech, not hate |
| hateful video detection audio-language model raw audio LLM multimodal 2025 2026 | HVGuard (EMNLP 2025, landing page opened; PDF text not extractable): MLLM chain-of-thought rationales fused by a trained mixture-of-experts, video-level, audio path not stated on the page; MM-HSD 2508.20546: audio embeddings, video-level; Buffalo capstone report 2025-01: VideoLLaMA2-AV on HateMM, video-level |
| MACIL-SD audio-visual hateful video localization weakly supervised | MACIL-SD (ACM MM 2022, 2207.05500): weakly supervised audio-visual violence detection on XD-Violence; no hateful-video use found |
| omni-modal LLM audio-visual temporal grounding zero-shot audio event localization Qwen2.5-Omni evaluation | TiTok 2610.09408 (zero-shot Qwen2.5-Omni 12.6 mIoU on multi-segment grounding); Auto-AEG 2607.04383 (fine-tunes Qwen3-Omni-30B for audio event grounding); AVTrace 2609.19991 (Qwen3-Omni-30B Acc@0.5 s .253 on span grounding). All ask the model to emit timestamps; none is a per-window read |
| video LLM ASR transcript timestamps alignment word-level speech text interleaved temporal binding | LiveCC (CVPR 2025): interleaves ASR words with frames by timestamp as training data, interpolating within 2–3 s caption chunks; HowToCaption 2310.04900: ASR drift handled by searching an offset with a video-language model; In-Sync 2604.22817: speech LLM emitting word timestamps. The timing problem is known; none binds speech evidence by cutting audio per window for a frozen reader |
| Gemini audio input hate speech detection spoken content moderation evaluation audio LLM | 2411.17123 (Gemini on text and image moderation, no audio); 2410.23861 (audio jailbreaks of Gemini-1.5-Pro); AHELM 2508.21376 (no toxicity task). No Gemini raw-audio hate speech evaluation found |
| HateMM audio-visual LLM VideoLLaMA2 OR Qwen2.5-Omni OR video-SALMONN hateful video classification zero-shot audio input | Hate-LLaMA (GitHub): Video-LLaMA fine-tuned on HateMM, video-level, 71 % accuracy; TANDEM zero-shot baselines (below) |
| training-free video anomaly detection audio language model raw audio LLM scoring segments zero-shot audio-visual | LAVAD (captions + LLM, no audio); NOVA 2609.06360 and AnyAnomaly (visual only); a related-work sentence says a follow-up "fuses audio via audio-language models" (unnamed in the snippet) — anomaly detection, not hate |
| omni model video content moderation segment-level harmful speech audio timestamps zero-shot Qwen-Omni / Gemini | AVID 2604.13593 (zero-shot omni models weak at fine temporal grounding, fine-tuned to fix it); 2604.03995 (spoken cues lower Qwen2.5-Omni-7B harm detection on Meta-Harm); Gemini docs on timestamp prompting. No segment-level harmful-speech localization by an omni model found |
| SafeLens hateful video segment policy LLM AAAI 2026 OR ImpliHateVid OR DeHate audio baseline audio-language model | SafeLens (AAAI 2026, Social-AI-Studio): segment-level hate detection, Qwen2.5-VL + a fine-tuned Llama-3-8B, speech enters as text; nothing on DeHate / ImpliHateVid audio-LLM baselines |
| audio-visual LLM evaluation "shuffled audio" OR "mismatched audio" OR "swapped audio" control temporal alignment probe | DAVE (OpenReview): diagnostic benchmark for unimodal shortcuts in audio-visual LLMs; AV-SyncBench 2607.00726: offset / jitter / speed perturbations, retrieval style; Dolphin (ICLR 2025): audio-visual LLMs tend to ignore audio. Mismatched-audio controls exist as a diagnostic idea; none applied to hateful video |
| LAVAD extension audio-language model fusion training-free video anomaly detection XD-Violence | MM-VAD 2603.13374 ("geometry-aware semantic reasoning", multimodal training-free VAD, "remains effective without audio input"); EventVAD 2504.13092 (visual). Anomaly detection, not hate |
| HateMM MultiHateClip baselines audio modality MFCC VGGish wav2vec Whisper audio encoder | HateMM (ICWSM 2023) and MultiHateClip 2408.03468: audio baselines are MFCC + fully connected layers; MultiHateClip also Wav2Vec2-BERT; MultiHateGNN (BMVC 2025). Audio enters as features into trained classifiers; no LLM read of audio |
| SALMONN OR Audio Flamingo OR Qwen2-Audio zero-shot toxic speech hate speech classification raw audio benchmark 2025 | ToxiAlert 2605.15984 (Qwen2-Audio as a general baseline on utterance toxicity); LALM-as-a-Judge 2602.04796 (safety of spoken dialogues); AudioBench (no toxicity task) |

Named papers opened in full or by abstract: TANDEM 2601.11178 (abstract and HTML v3, two passes), LELA 2602.09637
(abstract and HTML), MultiHateLoc 2512.10408 (abstract), An et al. 2408.06065 (abstract), HateClipSeg 2508.01712
(abstract), CLARA 2608.15905 (abstract), MARS 2601.15115 (abstract; training-free video-level hateful video
detection by adversarial reasoning, audio handling not stated), GuardReasoner-Omni 2602.03328 (abstract), HVGuard
(ACL Anthology landing page).

## Closest prior work and the exact difference

1. **TANDEM (2601.11178, AAAI-ICWSM 2027).** The nearest work on both counts. Its trained method gives each
   30-second chunk's audio track, saved as a 16 kHz mono WAV, to Qwen2-Audio-7B-Instruct, and the chunk's frames to
   Qwen2.5-VL-7B; the two are fine-tuned in alternation (SFT on 100 videos whose silver labels were produced by
   Qwen3-Omni-30B-A3B-Thinking and "strictly filtered against ground truth", then GRPO / GSPO with a reward on the
   dataset labels, timestamp IoU and target F1). The model writes label, time spans and targets as XML text per
   chunk; localization is scored by Avg IoU and Acc@0.5 on HateMM and MHC-en. It is not label-free. Its Table 2 also
   reports zero-shot baselines, among them **Qwen3-Omni-30B-A3B-Thinking with audio + video, Avg IoU .53 / Acc@0.5
   .55 on HateMM**, which is above TANDEM's own best row (.43 / .31) on those two metrics (as read from the HTML; the
   paper gives the baselines only a modality tag and does not say whether they saw the same 30-second chunks or the
   same prompt). The paper does not mention Whisper, transcript timestamp errors, or any binding or mismatched-audio
   control, and says it did not fine-tune an omni model because RL support for omni models is immature.

   Exact difference from the proposal: (i) TANDEM trains with labels and timestamp rewards; the proposal is
   label-free. (ii) TANDEM's unit is a 30-second chunk inside which the model itself must generate the hateful span's
   start and end; the proposal's unit is an 8-second window whose own clip is the only speech evidence shown, read as
   a Yes / No log-odds at the answer position; no timestamps are generated. (iii) The proposal keeps the whole-video
   prefix (20 frames, full transcript, verdict) and attaches the clip next to the question, the same placement that
   carried the frames result of 2026-10-10; TANDEM has no such structure. (iv) TANDEM declares no mechanism and has no
   swap or aligned-text control. (v) TANDEM's zero-shot Qwen3-Omni row is a comparison baseline, not a method with a
   claim; it is nonetheless a published, label-free, raw-audio localization number on HateMM.

2. **LELA (2602.09637).** Training-free, per-frame hate scores on HateMM and MultiHateClip with GPT-4o Mini; speech
   enters as Whisper text ("speech captions") and music as LP-Music-Caps captions; no model hears audio. Same task and
   label regime as the proposal; differs in exactly the component the proposal changes (the speech evidence is
   ASR text placed by ASR timestamps). The strongest published training-free comparator.

3. **An et al. 2408.06065.** Hate time frames located directly from raw speech by an end-to-end audio model, trained
   on synthetic data; audio only, no video, no LLM. Establishes "span-level hate from raw audio" as an existing task
   formulation; the proposal does not localize inside the clip, the window is the unit.

4. **PolySpeechTox / SoftPrompt-TSD (Findings of ACL 2026), MuTox, ToxiAlert.** A frozen audio language model judges
   toxicity of an utterance without ASR. The proposal's "hearing" effect (b) is this known capability; nothing in it
   is new on its own.

5. **MultiHateLoc (2512.10408), SafeLens (AAAI 2026), HVGuard (EMNLP 2025), CLARA (2608.15905), MM-HSD, Hate-LLaMA.**
   Trained or weakly supervised; speech as Whisper text split by sentence timestamps (MultiHateLoc, SafeLens) or as
   audio features / embeddings; Hate-LLaMA and the VideoLLaMA2 capstone feed audio to a video LLM but at video level
   and after fine-tuning. None is a label-free per-window read.

6. **Repository.** `experiments/20261008_baselines/README.md`: ZS-ImageBind (audio) scores 2-second raw audio clips at
   0.5 fps by text similarity — per-window raw audio evidence in a label-free localization comparator already exists
   here, though not as an MLLM read and without a transcript prefix or binding claim. Candidate 20
   (`archive/experiments/20261004_m1_acoustic/`) kept the proportional text slice and used a Whisper cross-attention
   alignment as a word-support prior inside the Qwen3-VL read (within +.008 / +.008, archived); README §4 of the
   proposal states this difference correctly, and the `speech_aligned` control is the clean version of the timing fix
   candidate 20 approximated.

Result of the search: raw audio of hateful videos has been given to an audio-language model for localization
(TANDEM, trained) and a Qwen3-Omni model has been run zero-shot on HateMM to emit hateful spans (TANDEM's baseline
table). No paper was found that, in a label-free setting, replaces the ASR-timestamped transcript slice of a window
by the window's own audio clip as the speech evidence of a per-window read, nor one that separates the binding effect
(swapped clip), the alignment effect (word-aligned text) and the reader swap (same reader, old inputs) from each other.

## STOP checks

1. **Source method already used for hateful video detection / localization: not met, but it is the closest call of
   the recent reviews and the claim must be narrow.** The generic ingredient — an audio or omni language model hears
   the audio of a hateful video and judges or localizes hate — has been used (TANDEM; its zero-shot Qwen3-Omni row;
   Hate-LLaMA at video level). The mechanism the proposal declares has not: label-free, per-window, the window's own
   clip in place of the ASR-cut text, presented next to the question on a cached whole-video prefix, read as a
   log-odds into a fixed time-level model, with controls that attribute the gain to binding, alignment or hearing.
   Rule 4 admits a mechanism designed from test error analysis; the diagnosis (ASR timestamp binding is poor: 60 % /
   68 % of windows draw text from a segment longer than 30 s, silent-by-ASR positives sink, boundary spill) is the
   documented origin, and TANDEM neither tests nor claims this. The transfer-from-another-task reading does not apply
   because the candidate is not presented as a transfer.
2. **Pure ensemble: no.** One frozen reader (Qwen3-Omni-30B-A3B-Instruct, Thinker only) for every branch; the window
   read `max(visual, speech)` is two reads of the same model, as in r6 (rule 3: whole-video and per-segment
   judgments by one MLLM are not an ensemble). Whisper produces the prefix transcript (speech-text modality) and the
   reader hears the window audio (audio modality); rule 3 allows one frozen model per modality and no predictions are
   averaged. The forced aligner (MMS_FA) produces an input for a control arm, not a prediction.
3. **Pure calibration / post-processing / smoothing: no.** The time-level model and evaluator are unchanged; the
   change is in the measurement taken per window. No labels enter any step.
4. **Pure engineering trick ("only swapping an encoder"): no, with the condition stated below.** The reader does
   change (Qwen3-VL-8B to Qwen3-Omni-30B-A3B), and that alone would be a model swap. The proposal is more than that
   because (a) the evidence given to the speech branch changes in kind and in how it is placed in time (an audio
   clip cut at the window bounds instead of a text slice placed by ASR timestamps), which is a change in what is
   measured, not only in what measures it; (b) the reader swap is isolated by `r6_inputs` (new reader, old inputs)
   and `text_prop` (new reader, local frames, old text slice), so the design can attribute any gain to the reader,
   the frames, the alignment or the audio separately; (c) a falsifiable mechanism statement with two gates (M1 swap,
   M2 aligned text) is declared before the run. Prompt wording changes only by replacing `{text}` with the clip; no
   constant is tuned; no resolution changes. A pure encoder swap would have none of (a)–(c). The condition: if
   `audio − text_prop` is within noise and the gain sits in `r6_inputs − r6_bma`, the outcome *is* a reader swap and
   can only be reported as such; the README already commits to this reading (R reported, M1 / M2 as gates).

Not grounds for STOP under rule 4 and therefore not used: the literature's finding that zero-shot omni models
localize poorly when asked for timestamps (TiTok, AVTrace, AVID) — the proposal does not ask the model to localize;
the finding that audio-visual LLMs tend to ignore audio (Dolphin) — the swap control tests exactly this; whether an
8-second clip carries enough speech; whether the clip read degenerates to the prefix verdict.

## Verdict

**PASS** — none of the four rule-4 conditions is met. The mechanism (window-bound raw audio as the speech evidence of
a label-free per-window read, with binding, alignment and reader-swap controls) is not used for hateful video
detection or localization in the literature searched; the candidate is not an ensemble, not post-processing, and more
than a model swap because the evidence and its time binding change and the swap is isolated by declared arms.

## Advisories (do not block)

- **Claims that must not be made.** Not "first to feed raw audio to an audio / omni LLM for hateful video detection
  or localization" (TANDEM's trained method; Hate-LLaMA; the VideoLLaMA2 capstone). Not "first label-free use of
  Qwen3-Omni for hate localization on HateMM" (TANDEM Table 2, zero-shot Qwen3-Omni-30B-A3B-Thinking, Avg IoU .53 /
  Acc@0.5 .55). Not "audio-LM toxicity judgment without ASR" (PolySpeechTox, MuTox, ToxiAlert). Not "window-level raw
  audio evidence in label-free hate localization" (the repository's own ZS-ImageBind audio baseline). Not
  "mismatched-audio controls" as a method (DAVE, AV-SyncBench, Dolphin). What could be claimed if M1 and M2 pass: that
  in a label-free per-window read the speech evidence should be bound to the window by the input cut rather than by
  ASR timestamps, and that the within-video gain comes from this binding (swap control) beyond what word-aligned text
  recovers (alignment control) and beyond the reader change (`r6_inputs`).
- **Comparators (rules 8 and 14(f)).** TANDEM is a trained comparator that uses labels and timestamps; it belongs in
  the reference rows with that marked. Its zero-shot Qwen3-Omni baseline is label-free; under rule 8 a new label-free
  comparator enters the table only after a rerun under this evaluator, and its generation-of-spans read-out would
  have to be turned into a frame score by a declared rule. Rule 14(f) asks, when a stronger MLLM than the comparators
  is used, for "strongest label-free comparator + same model" or a reason why not; `r6_inputs` covers the current
  method with the new reader, not T3AL or LELA. The README should state before the full run how 14(f) will be met.
- **Variant note.** TANDEM ran the Thinking variant with generation; the proposal runs the Instruct Thinker with a
  log-odds read-out. Say so when positioning against it.
- **Extra records worth keeping (not required).** (i) The audio read on windows the ASR calls silent (`(no speech)`
  in `speech_prop`), split by window gold: this is the direct test of the "silent positives sink" motivation and
  costs nothing extra. (ii) A second swap partner — the neighbouring window (i±1) besides the random derangement —
  would separate "any audio of this video" from "audio near this window", since boundary spill is one of the two
  named error sources. (iii) The fraction of windows whose aligned slice differs from the proportional slice, and the
  alignment failure count, so M2 can be read. (iv) The README §4 follow-up already notes that the > 30 s-segment split
  is confounded by video type; keep that caveat when the audio arm's gain is attributed to long-segment windows.
- **Wording of effect (b).** The alignment control licenses only "beyond aligned text"; it does not isolate prosody or
  delivery from word content heard more accurately than the ASR transcribed it. Do not write "detects tone-based hate"
  unless a control isolates it.
- **Cost (rule 5).** Declared (6 branch forwards per window, 12.5 audio tokens per second, about 4 h H200 per corpus
  pair, forced alignment under 15 min). Not a review matter; noted as present.
