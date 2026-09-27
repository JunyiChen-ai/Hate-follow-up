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

## Test-read log

2026-09-27: the files above, plus the transcripts and DeHate titles of the 12 top-ranked non-hateful videos per
corpus. No method changed.
