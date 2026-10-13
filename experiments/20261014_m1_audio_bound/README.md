# M1 candidate 43: audio-bound speech evidence (declared 2026-10-14, before any run)

Status: proposal. Nothing has been run. All constants below are frozen before the pilot (rule 7). Numbers cited from
the diagnosis are development-selected (rule 10).

## 1. Evidence and hypothesis

The within-video diagnosis of the current best configuration (`experiments/20261014_within_diagnosis/README.md`)
found that the speech channel's evidence is bound to time by the ASR's timestamps, and that this binding is poor:

- The window's transcript slice comes from Whisper segments cut in proportion to time (`src/video_inputs.py`
  `window_text`). Whisper's long-form pipeline often returns one segment per 30 s chunk or longer: 60 % of the
  HateMM windows and 68 % of the HateClipSeg windows draw their text from a segment longer than 30 s (extra.txt H
  and the follow-up check in §4). The words shown for a window are then only approximately the words spoken in it.
- Whole regions are mistimed or dropped on songs and long monologues: hate_video_283 (a rap whose lyrics start at
  0 s) has its first timestamp at 145.6 s; hate_video_139 at 48.6 s; hate_video_132 has one 146 s segment and a
  100 s gap. 19 of the 46 silent positive windows of HateMM lie before the first timestamp of a video with more
  than 30 words of text; 5 / 84 and 8 / 99 within-eligible videos have timed coverage below 50 % despite more than
  30 words.
- Silent positive windows (no timed speech, gold hateful) are 7 % of the HateMM windows but take part in 22 % of
  the mis-ordered pairs; they sink to rank percentile .72 (positives with speech: .42).
- Clean negatives next to a positive window with a Whisper segment spanning both mis-order at .346 / .402 against
  .188 / .290 for non-adjacent negatives; they hold 25 % / 31 % of the mis-ordered pairs.
- The same diagnosis showed what the frame side already established: when the window's own frames are shown next
  to the question, the visual read improves per window on HateClipSeg (+.039 window AUC), and the model no longer
  has to find the window's evidence by timestamp (probe A of candidate 41: it finds the in-window frames only
  36 % / 44 % of the time).

**Hypothesis.** The speech channel should be bound to the window the same way the frames are: the reader hears
the window's own audio, presented next to the question, instead of reading a text slice whose placement in time
is decided by the ASR. Two effects are expected and are separated by the controls: (a) *binding*: the evidence is
the window's own sound, so mistimed or proportionally cut transcripts no longer move speech evidence into the
wrong window and silent-by-ASR windows are no longer silent; (b) *hearing*: sung lyrics, delivery and words the
ASR dropped or softened are judged from the audio itself.

**What would falsify it.** If the audio arm does not beat the same reader given a correctly word-aligned text
slice, (b) is not supported. If the audio arm does not beat the same reader given another window's audio, (a) is
not supported and the audio acts as context only.

## 2. Mechanism (one reader for every read; no ensemble)

Reader: `Qwen/Qwen3-Omni-30B-A3B-Instruct`, Thinker only (`Qwen3OmniMoeThinkerForConditionalGeneration`,
transformers 5.15.1 as in `.cache/envs/HateVLM`), bf16, no generation: every read is the Yes/No log-odds at the
answer position, as in r6 (`src/mllm_judge.py` `cached_margin`). Seed 0. The vision encoder of Qwen3-Omni is the
Qwen3-VL encoder (technical report §2); the pixel cap stays `MAX_PIXELS = 100352`.

Per video, one prefix exactly as r6 / `adjacent_local`: system message, 20 uniform frames with `[t=..s]` labels,
the full Whisper transcript with timestamps, the policy rules, the verdict question and the model's own stance
answer. No audio in the prefix (R1).

Per 8 s window (grid offset 0, as r6), branches on the prefix cache:

| branch | content next to the question | role |
|---|---|---|
| `visual_local` | the window's 0.5 fps frames with time labels (as `adjacent_local`) | visual read of every arm except `r6_inputs` |
| `visual_native` | nothing (the 20 shared frames only, as r6) | visual read of `r6_inputs` |
| `speech_audio` | the window's audio clip `[start, end)` | **candidate** |
| `speech_prop` | the transcript slice cut in proportion to time (r6 text, byte-identical) | control: reader swap |
| `speech_aligned` | the transcript slice cut by forced-aligned word times | control: binding by alignment without hearing |
| `speech_swap` | the audio clip of another window of the same video | control: binding by presentation |

Window read `z = max(visual, speech)` and the fixed `r6_bma` time level (`twolevel_r2.py` with
`--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`),
the sole evaluator, 4 fps test GT. Arms = combinations:

| arm | visual | speech |
|---|---|---|
| `audio` (main) | `visual_local` | `speech_audio` |
| `text_aligned` | `visual_local` | `speech_aligned` |
| `text_prop` | `visual_local` | `speech_prop` |
| `audio_swap` | `visual_local` | `speech_swap` |
| `r6_inputs` | `visual_native` | `speech_prop` |

Speech question (frozen; the r6 wording with the text slice replaced by the clip):

```
Consider only window {i+1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video. Judge only what is spoken in this
window; its audio follows.

<audio clip>

Is THIS window one of the segments where speech that violates the above rules occurs?

Answer "Yes" or "No".
```

`speech_prop` and `speech_aligned` use the r6 wording `Judge only what is spoken in this window: {text}` with
`(no speech)` when the slice is empty. `visual_local` / `visual_native` use the r6 visual wording (and the
`adjacent_local` label format for the local frames).

Audio clip: the video's audio track decoded once per video with ffmpeg to 16 kHz mono PCM, cut to `[start, end)`,
no padding, given as a waveform to the processor (the Omni audio encoder emits 12.5 tokens / s: at most 100 tokens
per clip). Windows whose track is silent still get their clip.

`speech_swap`: window `i` gets the clip of window `π(i)`, `π` a derangement from `numpy.random.default_rng([1,
video_index])` (`video_index` = position in the manifest); a one-window video keeps its own clip.

`speech_aligned`: word times from a forced alignment of the full Whisper text to the full audio
(`torchaudio.pipelines.MMS_FA`, CTC forced alignment; text normalised to lower-case letters and apostrophes for the
aligner, the original words kept for display). A window's slice = the words whose midpoint lies in `[start, end)`,
in order. If the alignment fails for a video (no letters, exception), the slice falls back to the proportional cut
and the record says so; the count is reported. Asset: `data/asr_whisper_aligned/<corpus>/words.jsonl` with
`PROVENANCE.md` (rule 5: word-level ASR timestamps are an input, not a method change).

Constants: window 8 s, offset 0, 20 shared frames, 0.5 fps local frames, pixel cap 100352, clip = window bounds,
derangement seed `[1, video_index]`, aligner MMS_FA, fallback as stated. Identical for both corpora. No scans.

## 3. Gates (declared)

Noise floor: pooled .005, within .01 (rule 7). Within-video macro ROC is the target metric.

- **Performance (rule 8)**: `audio` against `r6_bma` (`runs/20260926_twolevel/r6_bma/metrics.json`): within
  ≥ +.01 on both corpora, no metric below r6 by more than the floor on either corpus.
- **Mechanism, binding (M1)**: `audio` − `audio_swap` ≥ +.01 within on both corpora.
- **Mechanism, hearing beyond alignment (M2)**: `audio` − `text_aligned` ≥ +.01 within on both corpora.
- **Reader swap (R)**: `r6_inputs` − `r6_bma` is reported; it is not a gate. If it is below −.02 within on a
  corpus, the performance gate is read with that offset stated, and the mechanism gates stand on their own.
- Readings: M1 and M2 pass → the mechanism is supported as stated. M1 passes, M2 fails, and
  `text_aligned` − `text_prop` ≥ +.01 → the gain is time binding by alignment; the audio reader is not needed for
  it; reported as such and the aligned-text variant becomes the candidate for rule 8. M1 fails → audio acts as
  context, candidate archived under rule 9 unless rule 9's revision path applies.

## 4. Difference from earlier attempts

- Candidate 20 (`archive/experiments/20261004_m1_acoustic/`): a Whisper cross-attention aligner turned into a
  word-support attention prior inside the Qwen3-VL read; the text shown stayed the proportional slice; within
  +.008 / +.008, archived. Here nothing is re-weighted: the reader hears the clip, and the aligned-text control
  tests the pure timing fix that candidate 20 approximated.
- Candidates 39–42 and the placement controls changed what the visual branch is shown; the speech branch was
  never changed. This is the speech-side counterpart of the placement finding (2026-10-10).
- TextTiling (38), Interval (26), VideoEvent (36): re-segmentation by text or events; the units here stay the
  8 s grid.
- Follow-up check (2026-10-14, test reads): speech window AUC of windows whose text touches a > 30 s segment vs the
  rest is .751 / .672 on HateMM (34 / 48 videos) and .603 / .686 on HateClipSeg (88 / 50 videos). The two subsets
  are different videos (monologues vs mixed content), so this is not a clean measure of the cut's damage; it is
  recorded because it does not support the binding effect on HateMM by itself.

## 5. Prior art the reviewer must check (rule 4)

TANDEM (arXiv 2601.11178: audio-language + vision-language models trained with reinforcement learning, timestamps
and targets on HateMM); LELA (2602.09637: training-free, speech via Whisper text and music via LP-MusicCaps
captions, no audio model); MultiHateLoc (2512.10408, weakly supervised with audio features); An et al. 2024
(2408.06065, end-to-end audio hate-span detection trained on synthetic data); audio-LM toxicity classification
without ASR (ACL Findings 2026, polySpeechTox); MACIL-SD; Qwen-Omni models used for moderation; our own
candidate 20 and the placement controls. The claim is narrow: label-free localization where the per-window speech
evidence is the window's own audio heard by the reader, with the binding (swap) and alignment (text) controls.

## 6. Pilot and plumbing checks (before the full run)

Fixed smoke subset: `experiments/20261007_m1_streamingtom/inputs.py` `selected_rows(smoke=True)` (five videos).
Checks, all recorded under `runs/20261014_m1_audio_bound/smoke/`:

1. For the first video of each corpus and every branch type, the cached-branch margin equals a standalone forward
   of the full conversation (|Δ| ≤ 1e-3) and the token ids agree (the seam check of r6).
2. `speech_prop` text equals r6's window text byte for byte; the `r6_inputs` prefix text equals the r6 prefix text.
3. Audio clips: sample count = (end − start) × 16000 ± 1, clip order = window order, `speech_swap` permutation is a
   derangement.
4. `GT_read is False` in every record; no label file is opened by the scorer.
5. Peak memory and seconds per branch, to confirm the cost estimate below before the full run.

Full run: one Slurm job per corpus on uoa-campus3 (H200 143 GB; the model needs about 70 GB in bf16), inside the
campus budget of two jobs; records rsynced back before any STATUS update.

## 7. Cost

Per video: one prefix (about 2.4 k text tokens + 20 × 91 image tokens, as r6) and 6 branch forwards per window
(two visual, four speech), each a few hundred tokens on the cache; audio encoding of one 8 s clip per window twice
(own and swapped). With 3 B active parameters on an H200, the estimate is 0.15–0.3 s per branch forward: 7359
windows × 6 × 0.25 s ≈ 3 h for both corpora plus prefixes and model load; declared budget 4 h of H200 time per
corpus pair (confirmed on the smoke run). One-off: model download (about 70 GB), audio decoding (CPU, minutes),
forced alignment (uoa-lab1 Slurm, under 15 min of GPU). On a new video the method needs: ASR as now, the audio
track, one prefix and two branch forwards per window (visual local + audio) — the same count as `adjacent_local`.

## 8. Test-read log (rule 10)

2026-10-14: the diagnosis files (`runs/20261014_within_diagnosis/`), the Whisper chunk files of hate_video_283,
_139, _132 and bit_6aHW6jocI9kJ, five window-centre frames of silent positive windows, and the follow-up speech AUC
split in §4. These informed the hypothesis and the choice of controls. No constant was chosen from them.

## 9. Rule-4 review outcome and additions (2026-10-14)

`docs/reviews/20261014_m1_audio_bound_novelty.md`: PASS. Closest prior work: TANDEM (2601.11178; trained with
RL, Qwen2-Audio-7B hears 30 s chunks, Qwen2.5-VL-7B sees frames; its Table 2 also lists a zero-shot
Qwen3-Omni-30B-A3B-Thinking span baseline on HateMM), LELA (training-free, speech as Whisper text), An et al.
2408.06065 (audio-only span model trained on synthetic data). Advisories adopted:

- Claims never to be made: first raw-audio or omni-LLM use for hateful video localization; first label-free
  Qwen3-Omni on HateMM; audio-LM toxicity without ASR; "tone-based hate". Effect (b) is worded only as "beyond the
  aligned text". TANDEM is cited as a trained reference.
- Rule 14(f) (strongest label-free comparator with the same model): the strongest label-free comparator in the
  table (T3AL) has no MLLM, so "T3AL with Qwen3-Omni" does not exist. If the candidate passes the performance gate,
  the same-model comparator will be the zero-shot whole-video span output of Qwen3-Omni (the TANDEM Table 2
  baseline) rerun under this repository's evaluator with a declared span-to-frame rule (frames inside a predicted
  span = 1, outside = 0, ties broken by the video-level Yes/No log-odds); otherwise it is not run. `r6_inputs` is
  the current method with the same model.
- Extra records in the full run (diagnostic, not gates): the audio read on ASR-silent windows split by gold; a
  second swap partner = the neighbouring window (`i+1`, last window takes `i−1`) in addition to the random
  derangement; the fraction of windows whose aligned slice differs from the proportional slice.
