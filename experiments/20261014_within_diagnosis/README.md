# Within-video error diagnosis of the current best configuration (2026-10-14)

Analysis only; reads test labels (rule 10); nothing here feeds a method. Asked by the user after candidate 42
(pairwise comparison, archived item 33): before proposing another read-out, find out where the configuration that
currently scores best within video (r6 reads with each window's own 0.5 fps frames shown next to the question,
`adjacent_local` = the `full_l` control of candidate 42) is wrong inside videos. Questions: which channel decides the
mis-ordered windows, how many windows have no speech, whether the errors sit in a few videos, and how much the decoder
contributes. The last diagnosis of this kind (`experiments/20260927_error_analysis/`) was on r3_m2 without frames.

Machine: sc474397 (CPU only). Code: `diagnose.py` (A–E), `diagnose_extra.py` (F–L). Output
`runs/20261014_within_diagnosis/{table.txt,summary.json,per_video.json,extra.txt,extra.json}` and six fixed-decoder
decodes under `runs/20261014_within_diagnosis/decoded/<tag>/metrics.json`.

## 1. Inputs and definitions

| configuration | raw reads | decoded curve (r6_bma time level) |
|---|---|---|
| r6 | `runs/20260926_glr/base_gridA` | `runs/20260926_twolevel/r6_bma` |
| adjacent (frames next to the question) | `runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local` | `.../placement_controls_main_decoded/adjacent_local` |
| E (frames with time labels) | `runs/20261007_m1_streamingtom/local_controls_main/local_clean` | `.../local_controls_main_decoded/local_clean` |

Gold `data/gt_4fps/<corpus>.npz` (test split); transcripts `data/asr_whisper_large_v3` through `src.video_inputs`;
frames-only-prefix reads `runs/20260928_infer/main/reads_noctx` for check K. Windows are the 8 s reading windows of
the raw record; the window gold is the share of positive 4 fps frames inside it, a positive window has share ≥ .5, a
clean negative has share 0. "Within" is the evaluator's per-video frame ROC over the videos with both classes
(84 HateMM, 99 HateClipSeg); window-level AUCs are the same per-video average over windows. The decoded window score is
the mean of the decoded curve over the window's frames. The speech reads are identical in the three configurations
(asserted), so every difference between r6 and the frames configurations is in the visual read.

Fixed-decoder variants (same flags as r6_bma: `--noleak --transform nscore --key calib --duration bma --bma-prior
length --min-windows 2 --bma-grid 6 --arm m2`, plus `--chains z_speech`, `--chains z_visual` or `--nocoupling`) were
decoded from the r6 and adjacent reads into `runs/20261014_within_diagnosis/decoded/` (commands in
`diagnose_extra.py` docstring; about one minute each).

## 2. Findings (HateMM / HateClipSeg; numbers from `table.txt` and `extra.txt`)

### 2.1 Levels: reads, combination, decoder (adjacent unless stated)

| | HateMM | HateClipSeg |
|---|---|---|
| within, decoded (the metric): r6 → adjacent | .751 → .769 | .637 → .664 |
| within of the raw max read, piecewise constant: r6 → adjacent | .680 → **.655** | .610 → .632 |
| window AUC visual read: r6 → adjacent | .644 → **.591** | .569 → .607 |
| window AUC speech read (identical) | .710 | .577 |
| window AUC max of the two: r6 → adjacent | .745 → .719 | .633 → .654 |
| window AUC of the decoded window mean: r6 → adjacent | .760 → .777 | .650 → .671 |
| decoder effect per video (decoded − raw within), r6 / adjacent | +.071 / +.114 | +.027 / +.032 |
| videos the decoder helps / hurts by more than .02 (adjacent) | .61 / .29 | .46 / .37 |
| decoded within, speech chain only (both configurations) | .738 | .578 |
| decoded within, visual chain only: r6 → adjacent | .677 → **.661** | .580 → .605 |
| decoded within, independent cells (no chain): r6 → adjacent | .633 → **.622** | .580 → .607 |

- **On HateMM the frames gain is not a better per-window visual read.** The visual window ordering gets worse
  (.644 → .591), the raw max read gets worse (.680 → .655), the visual chain alone gets worse (.677 → .661), and with
  independent cells the frames lose .010. The +.018 appears only with the chain. Split by how well the speech read
  orders the video (terciles of the r6 speech window AUC, extra.txt J): on the 25 videos where speech orders well
  (speech AUC .96, within already .92) the frames make the visual ordering much worse (−.150) and the decoded within
  drops (−.022); on the 25 videos where speech orders badly (speech AUC .39, within .52) the visual ordering does not
  improve (−.015), the raw within barely moves (+.009), yet the decoded within rises +.054. The frames make the visual
  reads more extreme (SD 6.3 → 9.2 logits, share positive .42 → .49) without ordering windows better; what the chain
  does with these reads was not isolated further.
- **On HateClipSeg the frames gain is a genuine read improvement**: visual window AUC +.039, raw within +.022,
  independent cells +.027, visual chain alone +.025, and it is present in all three speech terciles.
- **Within-video ordering is carried by speech on HateMM** (window AUC .71 against .59–.64 visual; the decoded score
  tracks the speech read more, Spearman .56 against .34). On HateClipSeg the channels are even (.58 speech, .61 visual
  after frames). The two reads are nearly uncorrelated inside a video (Spearman .10–.13) in every configuration.
- **A visual read without the transcript in the prefix orders worse, not better** (check K, `reads_noctx`): window AUC
  .543 / .522 against .644 / .569 for r6. The transcript context is where most of the visual read's within-video
  information comes from; an "independent visual channel" is not a direction.

### 2.2 Mis-ordered pairs: which channel, and would the other channel have been right (adjacent)

Pairs = (positive window, clean negative window) of the same video; mis-ordered when the negative's decoded score is
at least the positive's. Shares among mis-ordered pairs, with the base rate among all clean negatives in brackets.

| | HateMM | HateClipSeg |
|---|---|---|
| mis-ordered pair rate, decoded / raw max | .214 / .262 | .320 / .345 |
| negative window's max read came from speech | .71 (base .51) | .54 (base .43) |
| from the visual read | .21 (base .40) | .33 (base .44) |
| negative window has no speech | .08 | .13 |
| positive window has no speech | .22 (no-speech windows: .07 of all) | .15 (.12 of all) |
| visual-only order would be correct | .39 | .39 |
| speech-only order would be correct (both windows with speech) | .29 | .24 |
| neither channel orders the pair correctly | .43 | .47 |
| negative adjacent to a positive window | .30 | .38 |
| negative mentions a protected group / positive mentions | .18 / .15 | .14 / .11 |

- The negatives that rise above positives are **speech-driven well above the base rate** (.71 against .51, .54 against
  .43): the transcript of a non-hateful window inside a hateful video reads as hateful. In **43–47 % of the
  mis-ordered pairs neither channel orders the pair correctly**, so no label-free choice between the channels could
  fix them.
- **No-speech positive windows sink** (HateMM): they are 7 % of the windows but appear in 22 % of the mis-ordered
  pairs. Their decoded rank percentile inside the video is .72 (0 = top) against .42 for positives with speech, and the
  same .74 for no-speech negatives: with only a weak visual read, a window without speech lands near the bottom
  whatever its label. On HateClipSeg the effect is small (.51 against .44).
- **Boundary text spill** (extra.txt H): Whisper segments have median 2.0 / 2.4 s but a mean of 11.4 / 15.6 s and
  18–20 % are longer than 8 s; `window_text` splits such a segment between windows in proportion to time. Clean
  negatives next to a positive window with a segment spanning both: 74 / 189 windows, mis-order rate **.346 / .402**
  against .188 / .290 for non-adjacent negatives and .202 / .318 for adjacent negatives without a crossing segment;
  they hold 25 % / 31 % of all mis-ordered pairs. If they had the non-adjacent rate, mis-ordered pairs would fall by
  about 11 % / 9 % (window AUC about +.02; less at frame level because the decoder already smooths).

### 2.3 Where the errors are: videos, hate share, fragmentation, position

| adjacent | HateMM | HateClipSeg |
|---|---|---|
| per-video within quantiles 10 / 25 / 50 / 75 / 90 | .45 / .65 / .86 / .95 / .99 | .40 / .54 / .67 / .81 / .93 |
| videos below .5 / above .9 | 15 % / 39 % | 21 % / 15 % |
| share of the total deficit Σ(1 − AUC) in the worst 10 % / 20 % of videos | 30 % / 54 % | 21 % / 38 % |
| within by gold runs: 1 / 2 / 3–4 / ≥ 5 (n) | .82 (57) / .69 (12) / .67 (6) / .62 (9) | .80 (16) / .64 (17) / .66 (38) / .61 (28) |
| within by hate share < .25 / .25–.5 / .5–.75 / ≥ .75 (n) | .72 (19) / .69 (13) / .74 (15) / .83 (37) | .70 (19) / .66 (23) / .61 (31) / .71 (26) |
| inverted videos r6 / adjacent / both | 15 / 13 / 12 | 22 / 21 / 16 |

- HateMM errors are **concentrated**: a third of the videos are at .95 or better and the worst eight hold 30 % of
  the deficit. HateClipSeg errors are **diffuse**: median .67, 39 % of videos below .6, 66 of 99 videos have three or
  more gold runs (within .64; one or two runs .72).
- **Annotation structure on HateMM** (extra.txt I): 37 of the 84 videos (44 %) are at least 75 % hateful, and in
  them 94 % of the negative frames lie in the first or last 10 % of the video. A position-only curve (higher towards
  the middle) scores .961 within on these 37 videos against the method's .834, and .713 against .769 over all 84.
  The decoder's +.114 on HateMM is almost entirely this group: raw .577 → decoded .834 (+.257) on the 37 videos,
  within ±.02 on the other 47 (+.009, +.010, −.020 by hate-share bin). On HateClipSeg the position curve reaches .575
  (method .664); the decoder adds +.044 on sparse videos and +.088 on the ≥ 75 % group.
- The hard cases are therefore **sparse and fragmented hate** (HateMM share < .5: 32 videos at .70; HateClipSeg
  three or more runs: 66 videos at .64; hate share < .5: 42 videos at .68), where both reads order windows at .55–.62 and neither channel is right in
  almost half the errors. This is the same finding as 2026-09-27 (K5) and the 2026-09-28 headroom analysis, now on
  the frames configuration.

### 2.4 Decoder failures that are concrete

- The three HateMM within-eligible videos the model judges not hateful (stance No, z_video < 0): raw ordering .880,
  decoded .524. Example hate_video_36 (24 s): every read at the floor (−12 to −16 logits), p_video log-odds −11,
  the cell posterior follows the chain's shape (.40 → .62 rising to the end) while the gold is in the middle. When
  all reads sit at the floor the time level has no evidence and its prior decides the order. On HateClipSeg the same
  group (5 videos) is helped by the decoder (+.10), so this is not a general fix.
- Worst HateMM videos: hate_video_228 and hate_video_349 (90–92 % hateful, within .08 / .18): the unmarked edge
  ranks top. hate_video_408 (6 % hateful): the speech read is +3.8 to +6.6 on windows 1–4 (gold 0) and −7.5 on the
  gold window; the visual read is right (1.0) but far below the speech read in scale. hate_video_177 (10 % hateful):
  raw .41, decoded .11.

## 3. What this says about the next step

1. **The frames trick is two different things on the two corpora.** On HateClipSeg it improves the visual read
   per window; on HateMM it does not, and the +.018 within comes from the chain on videos where speech is poor. A
   mechanism built on "the model localizes better when it sees the window's own frames" is supported on HateClipSeg
   only.
2. **The remaining errors are mostly not a combination or read-out problem.** Neither channel is right in 43–47 %
   of the mis-ordered pairs; the speech read raises non-hateful windows of hateful videos; comparisons between windows
   carry the same information (candidate 42). This is the limit of the per-window reads noted since 2026-09-27.
3. **Two bounded, concrete error sources** came out, with upper bounds on what fixing them is worth:
   - boundary text spill from long Whisper segments: about 10 % of the mis-ordered pairs, window AUC about +.02 at
     most (a word-level transcript cut would address only the window slice; the whole transcript stays in the prefix);
   - no-speech positive windows sinking on HateMM: 22 % of the mis-ordered pairs there, nothing comparable on
     HateClipSeg; it needs a stronger visual read on silent windows, which the frames did not provide on HateMM.
4. **HateMM within is to a large part annotation structure** (intros and outros of mostly-hateful videos). Gains on
   HateMM within from decoder changes would mostly be fitting that structure; they should not be read as better
   localization.
5. The "setwise worst-of-k" idea listed after candidate 42 is not supported by this diagnosis: it compares the same
   reads, and the errors it would have to fix (speech-driven false highs, silent positives, fragmented gold) are not
   ordering errors between distant windows.

## 4. Test-read log (rule 10)

2026-10-14: read the files in §1 (predictions and decoded curves of r6, adjacent_local, local_clean; `reads_noctx`;
gold arrays; Whisper transcripts), produced the outputs in `runs/20261014_within_diagnosis/`, and ran six label-free
decodes of existing reads with the fixed r6 decoder (`decoded/`). No method, constant or design changed. Any later
candidate that cites these numbers is development-selected.
