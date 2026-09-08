# Pre-registration — Speaker-provenance restoration probe

**Frozen:** 2026-08-08 before speaker embeddings, clusters, visual-role features,
or provenance-conditioned judge scores are computed.  
**Purpose:** gate a candidate information-adding method, not tune prompts.  
**Primary arm:** the existing 90-video ImpliHateVid temporal-attribution cohort
(30 corrected-8B FP, 30 cue-matched TP, 30 cue-carrying TN).

## Phenomenon and candidate mechanism

Flat ASR preserves *what* was said but erases utterance ownership. News,
interviews, inserted clips, quotation, and response videos can therefore expose
the judge to a hateful proposition without identifying whether the video's
primary voice advances or answers it. Prompt-side stance elicitation and
frame--speech temporal alignment have failed; neither supplied speaker
ownership. The proposed input intervention restores speaker turns and coarse
audiovisual roles before one unchanged holistic moderation call.

This probe tests whether ownership is new usable evidence. It does not assume
that anonymous `Speaker A/B` labels are semantic provenance.

## Frozen extraction

1. Reuse the existing Whisper large-v3 timestamped chunks and 16-kHz WAVs.
2. Split valid speech intervals into 1.5--3.0 s windows. Embed each window with
   the general-purpose `microsoft/wavlm-base-plus-sv` speaker-verification
   encoder.
3. Per video, choose one or two speaker clusters using a frozen one-vs-two
   silhouette rule: use two only with at least four windows, both clusters at
   least two windows, and cosine silhouette at least 0.20; otherwise use one.
4. Assign each ASR chunk by the majority speaker-window overlap. The speaker
   with the greatest total assigned speech duration is `PRIMARY`; the other is
   `SECONDARY`.
5. Sample native video frames inside each valid ASR interval. A frozen OpenCV
   frontal-face detector records whether a stable face track is visible during
   the interval. Role tags are `PRIMARY_ONSCREEN`, `PRIMARY_VOICEOVER`,
   `SECONDARY_ONSCREEN`, or `SECONDARY_VOICEOVER`. These are evidence tags, not
   inferred endorsement labels.
6. Invalid or reversed timestamps use neighboring valid boundaries; chunks
   that cannot be placed retain `[SPEAKER_UNKNOWN]` and are counted.

No hate lexicon, group list, dataset label, or moderation score enters this
stage.

## Frozen quality gate

Do not call the judge unless all hold:

- at least 75/90 videos retain nonempty attributed speech;
- at least 25 videos meet the frozen two-speaker rule;
- at least 80% of nonempty ASR chunks receive a non-unknown speaker;
- at least 50 videos have a valid onscreen/voiceover observation for one chunk;
- a deterministic 20-video spot check finds no more than 4 gross ownership
  failures (one speaker's consecutive speech systematically split or clearly
  different alternating speakers systematically merged).

Failure moves directly to the predeclared successor in **Next-step map**; it
does not license threshold or clustering changes.

## Four evidence conditions

- **F — flat:** existing restored transcript, no turn boundaries. Existing
  corrected-8B joint score; no new call.
- **P — provenance:** same words in the same order with frozen speaker/role
  tags.
- **S — segmentation control:** same turn boundaries, every tag replaced by
  `[SPEAKER]`.
- **R — permutation control:** same words and boundaries; PRIMARY/SECONDARY and
  ONSCREEN/VOICEOVER tags deterministically swapped per video (seed 20260808).

P/S/R use the same frames, title, rules, binary question, raw-z readout, and
frozen Qwen3-VL-8B judge. These are diagnostic counterfactual arms; a deployed
method would use P only and still make one moderation call per video.

## Frozen signal rule

Let deltas be condition minus F. Labels name diagnostic cohorts only.
The provenance mechanism **PASSES** only if all hold:

1. FP median `delta_P <= -1.0`.
2. TP median `delta_P >= -0.25`.
3. AUC(TP vs FP) by P exceeds F by at least 0.05.
4. P exceeds S by at least 0.03 AUC(TP vs FP).
5. P exceeds R by at least 0.03 AUC(TP vs FP).
6. At the frozen F valley, at least 30% of FPs flip down under P while no more
   than 10% of TPs flip down.
7. On TN vs FP, P moves toward indifference relative to F by at least 0.10 AUC;
   otherwise it is merely reproducing the original ranking.

All clauses and 2,000-draw paired-bootstrap intervals are reported regardless
of outcome. No threshold is selected from P/S/R.

## Next-step map tied to the method goal

- **Pass:** implement the full single-call provenance-restoration method and
  run ImpliHateVid, HateMM, and MHClip held-out evaluation.
- **Ownership quality passes but P equals S:** turn segmentation, not
  provenance, is the active signal; develop a turn-structured evidence method
  only if it reaches the same performance bars under a new preregistration.
- **P equals R or harms TP:** speaker tags are unused/noisy; close provenance
  and immediately run the next ranked information-adding probe: generic-harm
  nuisance measurement with a frozen general-purpose video encoder, targeting
  HateMM's G/O false-positive mass.
- **Extraction quality fails:** close this implementation without judge calls
  and move to that same nuisance-measurement successor.

