# Cross-corpus error analysis of r3_m2 (2026-09-27)

Analysis only (reads test labels, rule 10); nothing here feeds a method. Code `error_analysis.py`; output
`runs/20260927_error_analysis/{table.txt,report.json}`. Inputs: `runs/20260926_twolevel/r3_m2` (HateMM, HateClipSeg),
`runs/20260927_dehate_external/r3_m2` (DeHate), the grid-A reads, `data/gt_4fps`, the Whisper transcripts, the DeHate
label file (modality flags).

## Findings

| | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| hateful videos | 40 % | 87 % | 20 % |
| pooled PR: actual / oracle video / oracle within | .695 / .675 / .778 | .670 / .604 / .753 | .158 / **.351** / .231 |
| video AUC / AP of the key | .927 / .901 | .803 / .967 | .727 / .361 |
| verdict > 0 on non-hateful videos | 49 % | 67 % | 57 % |
| group mention in transcript: false-positive vs true-negative videos | .41 vs .17 | .70 vs .00 | .34 vs .13 |
| window read ~ gold label + group mention (standardized within video) | +.48 / +.56 | +.27 / +.61 | +.29 / +.50 |
| within (share inverted) | .753 (20 %) | .640 (26 %) | .654 (31 %) |
| within when hate covers < 25 % of the video | .667 | .687 | .590 |
| window reads alone: visual / speech / max | .614 / .651 / .680 | .546 / .569 / .610 | .546 / .614 / .641 |
| top false alarms inside hateful videos within 8 s of hate (all negatives) | 45 % (34 %) | 42 % (36 %) | 19 % (14 %) |

1. The dominant error level depends on how many videos are hateful. On DeHate (20 % hateful, the realistic case) a
   perfect video level would more than double pooled PR; on HateMM and HateClipSeg the within level limits it.
2. The verdict is positive for half or more of the non-hateful videos on all three corpora. Reading the top-ranked
   non-hateful videos (their transcripts, and DeHate titles):
   - abuse that targets individuals or non-protected groups: officials called "child-murdering government whore",
     "shut up, retard", gaming trash talk;
   - use rather than endorsement of slurs: songs, satire, historical clips, in-group use in rap;
   - reports and discussion of hate: a congressional hearing on antisemitism, local news on hateful symbols.
3. At window level the protected-group mention moves the read as much as or more than the gold label on every corpus
   (HateClipSeg and DeHate about twice as much). This repeats the 2026-09-12 finding on a third corpus.
4. Most top false alarms inside hateful videos are far from the gold span (55-81 % beyond 8 s), so they are content
   confusions, not boundary spill; smoothing cannot remove them.
5. Short and sparse hate is the weak within case on all corpora; speech reads beat visual reads on all three.

## Closure of concerns K4 and K5 (2026-09-27, autonomous iteration; development-selected)

### K4: the window read responds to the topic (a protected-group mention) about as much as to hate (finding 3)

Closed without a new run. Eight earlier directions attacked this confound and were archived after their declared
rounds. `experiments/20260912_sdl/README.md` §1 collects the numbers:
- **Read-out changes:**
  - pairwise comparison of windows (PWC);
  - subtracting a topic read (TAD, within .6159 / .5793 against .6926 / .6007);
  - a five-way speech-act read with report, quote and counter-speech as separate answers (TAD, .6945 / .6076).
- **Adapting the model without labels:** SDL and NGA, three rounds each. Both flatten the curves.
- **Other reads:**
  - onset reads (BND);
  - counterfactual removal of intervals (CVA, `archive/experiments/20260912_cva/`);
  - hypothesis-conditioned windows (HVL, two rounds).

**Upper bound, TAD §5d.** A classifier fitted on the test labels, over every cached per-window output including the
branch hidden state, does not beat the unfitted read: .742 / .626 against .758 / .621. So no label-free read-out of
these reads can separate topic from act better than the read itself. The seven-MLLM study shows the same within
ceiling across models (`experiments/20260910_spvl/README.md` §11).

DVD (`experiments/20260927_dvd/README.md` §9) was the planned source of conditions to ask per window, and it adds two
facts:
- The protected-group condition T is itself a topic question. It helps at the video level only where the labels
  follow the hate definition. Asking it per window would add the feature the confound consists of.
- The endorsement condition E is the act side of the confound. It did not carry weight even at the video level:
  dropping it never costs .01. TAD's five-way act read already asked the act question per window.

**K4 closes as a documented limit of the frozen reader's per-window output.** The input side is tested separately:
per-window frames, K7 (`experiments/20260926_twolevel/README.md` §17).

### K5: short and sparse hate is the weak within case (finding 5)

The check declared with K2 is done (`experiments/20260926_twolevel/README.md` §16.3). Source:
`runs/20260926_twolevel/analysis_r4/table.txt`. It covers videos where hate covers less than 25 % of the frames, 19
per corpus. Within, with the 95 % interval of the paired difference:

| comparison | HateMM | HateClipSeg |
|---|---|---|
| `r4_bma` − `r3_m2` (per-video segment lengths) | .6994 vs .6674, +.032 [−.009, +.079] | .6672 vs .6866, −.019 [−.069, +.027] |
| `r4_k1` − `r4_bma` (no minimum length) | +.010 [−.045, +.064] | −.028 [−.069, +.011] |
| `r4_nocoupling` − `r4_bma` (independent cells) | +.004 [−.050, +.058] | −.084 [−.147, −.018] |

**No time-level variant is better on this subset on both corpora.** On HateMM the subset hardly depends on the time
level: independent cells do as well as the chain. On HateClipSeg the chain helps. Letting each video have its own
time scale (K2) moves the two corpora in opposite directions, and both intervals include 0.

**K5 closes: the time level does not solve it.** On HateMM the subset is limited by the window reads (K4).

## Test-read log

2026-09-27: the files above, plus the transcripts and DeHate titles of the 12 top-ranked non-hateful videos per
corpus. No method changed.
2026-09-27 (closure of K5): `runs/20260926_twolevel/analysis_r4/` (reads the gold for the coverage subset). No method
changed.
