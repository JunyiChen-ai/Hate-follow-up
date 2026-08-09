# Ranking-error autopsy: MHClip-ZH test

**Date:** 2026-08-09. **Type:** diagnostic autopsy. No preregistration and no
method proposal. All statistics are on the frozen single-call Qwen3-VL-8B judge
scores already committed for this corpus. Four one-call diagnostic arms were run
on the free RTX 5090 (596 judge calls, about three GPU-minutes); each ablates one
field of the existing prompt and none is a candidate component.

## Headline

MHClip-ZH's labelled-oracle ceiling of 0.7932 macro-F1 is not an anomaly needing
an explanation. It is 3.2 points **above** what an equal-variance binormal
ranking of AUC 0.855 at prevalence 0.302 delivers, and it sits mid-pack against
the three sister corpora measured under the same judge. Everything to explain is
in the ranking, and the ranking's residual is on the negative side: at the oracle
threshold there are 21 false positives against 7 false negatives, so three
quarters of the residual error is normals scoring high.

Those normals are there by design. Forty-two percent of the shipped titles carry
the dataset harvester's highlighted query term, and those terms are offensive
words. Thirty-one of the 104 normals were collected with such a query, their
median raw z is +2.25 against -6.5 for the rest of the negative class, and they
supply 11 of the 21 oracle false positives. MHClip-ZH is a keyword-matched
hard-negative corpus, and the judge is doing what a lexical prior does on one.

The channel result is the sharp one. Blanking the transcript field moves corpus
AUC from **0.855 to 0.856**, a paired-bootstrap interval of -0.033 to +0.041.
Blanking the title moves it to **0.776**. Blanking both leaves the sixteen frames
at **0.747**. The speech channel is worth about three AUC points on its own and
every one of them is already carried by the title, which is why channel
restoration measured neutral on this corpus. This is not a delivery failure that
a better recogniser would fix; it is redundancy.

## 1. Stratified decomposition

MHClip-ZH test_clean, 149 videos, 8B raw z. The primary collapse maps Hateful and
Offensive to positive.

| Stratum | Positives | Negatives | ROC-AUC | 95% bootstrap CI |
|---|---:|---:|---:|---|
| Hateful vs Normal | 17 | 104 | 0.877 | 0.772–0.963 |
| Offensive vs Normal | 28 | 104 | 0.841 | 0.765–0.908 |
| Hateful vs Offensive | 17 | 28 | 0.665 | 0.478–0.843 |
| Union vs Normal (primary) | 45 | 104 | 0.855 | 0.789–0.913 |

The union figure is the exact rank-weighted average of the first two rows, to
four decimals. Unlike MHClip-EN, the Hateful class is ranked above the Offensive
class rather than below it, but only by 3.6 points on 17 videos, so no class-pair
stratum explains the corpus number.

Score distribution by annotated class, raw z:

| Class | n | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Hateful | 17 | -8.25 | 6.50 | 9.25 | 12.00 | 13.50 |
| Offensive | 28 | -4.00 | 6.00 | 7.88 | 9.25 | 12.50 |
| Normal | 104 | -17.75 | -10.25 | -5.00 | 2.75 | 11.00 |

The shape matters more than the location. The positive class occupies a narrow
band, interquartile range 3.5 and standard deviation 5.19; the negative class is
wide, interquartile range 13.0 and standard deviation 8.34. Eighteen percent of
normals sit above the positive class's lower quartile. The ranking is not two
separated modes with a few strays: it is a tight positive band with a long
negative right tail pushed up into it.

Class composition by score band:

| Band | n | Hateful | Offensive | Normal | Positive rate |
|---|---:|---:|---:|---:|---:|
| z < -5 | 54 | 2 | 0 | 52 | 0.037 |
| -5 ≤ z < 0 | 22 | 0 | 4 | 18 | 0.182 |
| 0 ≤ z < 5 | 14 | 0 | 1 | 13 | 0.071 |
| 5 ≤ z < 9 | 33 | 4 | 15 | 14 | 0.576 |
| z ≥ 9 | 26 | 11 | 8 | 7 | 0.731 |

## 2. Why the oracle stops at 0.7932

The labelled-oracle operating point is z ≥ 5.0: 38 true positives, 21 false
positives, 7 false negatives, 83 true negatives. Positive-class F1 is
2·38/(2·38+21+7) = 0.7308 and normal-class F1 is 2·83/(2·83+7+21) = 0.8557, so
the macro figure is 0.7932. Recall is already 0.844; precision is 38/59 = 0.644.
The binding quantity is positive-class precision, and precision is bounded by the
21 normals at or above the threshold.

The error budget, relabelling errors hypothetically to see which side owns the
gap:

| Counterfactual | Macro-F1 |
|---|---:|
| As shipped | 0.7932 |
| All 21 false positives corrected | 0.9416 |
| All 7 false negatives corrected | 0.8493 |
| Half the false positives corrected | 0.8671 |

Removing the k highest-scoring negatives, against removing the k lowest-scoring
positives:

| k | Negatives removed: AUC / oracle | Positives removed: AUC / oracle |
|---:|---|---|
| 0 | 0.855 / 0.793 | 0.855 / 0.793 |
| 3 | 0.874 / 0.811 | 0.885 / 0.811 |
| 7 | 0.896 / 0.836 | 0.914 / 0.836 |
| 10 | 0.911 / 0.855 | 0.923 / 0.843 |
| 21 | 0.949 / 0.938 | — |

**The ceiling is not low for the ranking it has.** Setting an equal-variance
binormal ranking to each corpus's measured AUC and prevalence and maximising
macro-F1 over the threshold gives a reference value. Observed minus reference,
same judge, four corpora:

| Corpus | AUC | Prevalence | Oracle macro-F1 | Binormal reference | Observed − reference | FP share of errors |
|---|---:|---:|---:|---:|---:|---:|
| HateMM | 0.923 | 0.400 | 0.8879 | 0.8415 | +0.046 | 0.43 |
| ImpliHateVid | 0.947 | 0.498 | 0.8925 | 0.8741 | +0.018 | 0.60 |
| MHClip-EN | 0.785 | 0.304 | 0.6888 | 0.6998 | -0.011 | 0.81 |
| **MHClip-ZH** | **0.855** | **0.302** | **0.7932** | **0.7609** | **+0.032** | **0.75** |

MHClip-ZH converts its ranking into a threshold decision slightly better than the
reference and better than two of the three sister corpora. There is no
threshold-side pathology to explain. The 0.7932 is the AUC and the prevalence,
arithmetically.

## 3. Where the ranking loses: the negative class was harvested with slurs

The shipped titles carry the collection query term wrapped in the harvester's
HTML highlighting. Sixty-three of the 149 test titles carry it, and the terms are
offensive words: anti-effeminacy slurs, gendered insults, common profanities and,
in nine cases, an anatomical term for male genitalia. The markup is metadata, but
the words are part of the title the judge reads.

| Population | n | Median z |
|---|---:|---:|
| Normal, title carries a harvest term | 31 | +2.25 |
| Normal, plain title | 73 | -6.50 |
| Positive, title carries a harvest term | 32 | +8.63 |
| Positive, plain title | 13 | +6.50 |

Eleven of the 21 oracle false positives are keyword-harvested normals. The corpus
AUC decomposes exactly over the four pair populations:

| Pair cell | Pairs | Weight | AUC | Contribution |
|---|---:|---:|---:|---:|
| harvested positive × harvested negative | 992 | 0.212 | 0.780 | 0.165 |
| harvested positive × plain negative | 2336 | 0.499 | 0.891 | 0.445 |
| plain positive × harvested negative | 403 | 0.086 | 0.723 | 0.062 |
| plain positive × plain negative | 949 | 0.203 | 0.899 | 0.182 |
| **total** | **4680** | **1.000** | | **0.8547** |

Against the 73 plain negatives alone the judge reaches AUC 0.894 and oracle
macro-F1 0.859. Against the 31 harvested negatives alone it reaches 0.763 and
0.749. The nine videos harvested with the anatomical term are all annotated
Normal, are all traditional-medicine sexual-health advertorials, have median z of
+4.25 against the negative class's -5.00, and four of them sit above the oracle
threshold. Dropping just those nine moves corpus AUC to 0.870 and the oracle to
0.822.

This is a real property of the corpus, not a judge defect, and it is the single
largest identified effect: roughly 4 AUC points and 6 macro-F1 points relative to
a negative class drawn the same way as the majority.

## 4. Which channel carries the ranking

Four one-call arms, each blanking one field of the frozen prompt and leaving the
sixteen frames, the prompt, the model and the raw-z readout untouched. The
annotation loader is patched at import; `extract_duplex_readout.py` itself is
imported and driven, not reimplemented.

| Arm | Title | Transcript | AUC | Δ vs baseline | Paired 95% CI | Oracle macro-F1 |
|---|:-:|:-:|---:|---:|---|---:|
| BASELINE | yes | yes | 0.8547 | — | — | 0.7932 |
| NOTRANSCRIPT | yes | blank | 0.8563 | +0.002 | -0.033 – +0.041 | 0.7706 |
| NOMARKUP | words only | yes | 0.8581 | +0.003 | -0.022 – +0.029 | 0.8135 |
| NOTITLE | blank | yes | 0.7756 | **-0.079** | -0.164 – -0.001 | 0.7478 |
| FRAMESONLY | blank | blank | 0.7472 | **-0.108** | -0.191 – -0.026 | 0.7141 |

Read as marginal contributions over the frames alone: the title is worth +0.109
AUC, the transcript is worth +0.028, and the two together are worth +0.108. The
transcript's contribution is entirely subsumed by the title's. Given the title,
deleting the transcript costs nothing measurable.

Three facts corroborate this from the shipped data. First, the three transcript
variants that already exist on disk are indistinguishable: fresh Whisper 0.8547,
the dataset's own transcript 0.8665, forced-Chinese Whisper 0.8577. Second,
seven videos received a byte-identical 23-character judge transcript, because the
recogniser emitted the same channel-plug boilerplate for all of them, and their
scores still span 27.25 z units from -17.25 to +10.00. Third, the 33 videos whose
transcript is recogniser boilerplate rank at AUC 0.849, against 0.854 for the 116
whose transcript is real speech.

Stripping only the harvester's HTML tags while keeping every title word changes
nothing: median absolute z shift is 0.00 and the AUC interval covers zero. The
judge is reading the title's words, not the collection artefact.

## 5. False-negative taxonomy

The twenty lowest-scoring positives. Codes were written from the transcript, the
title and four frames of each video.

| Code | Meaning | n | Rate | Wilson 95% | Median z |
|---|---|---:|---:|---|---:|
| G | gender-stereotype commentary or skit, no slur | 6 | 30% | 15–52% | 1.00 |
| F | fiction or filmed conflict; the offence is a plot or scene event | 6 | 30% | 15–52% | 6.88 |
| T | title-borne: the offence is only in the uploader's title | 4 | 20% | 8–42% | 1.88 |
| S | sexual content or covert footage as the whole basis of the label | 2 | 10% | 3–30% | 6.38 |
| X | the video is not in Chinese | 1 | 5% | 1–24% | 5.75 |
| I | implicit protected-group hostility carried by a meme juxtaposition | 1 | 5% | 1–24% | 8.00 |

Twelve of the twenty involve no protected group in any form: all six F, both S,
the one X video and three of the four T. The G cohort is the interesting one: six videos whose whole subject is a
gendered stereotype about partners or family roles, delivered as talk-show
commentary or comedy, with a median z of 1.00 against a positive-class median of
8.50. The T cohort is the sharper anomaly. In all four the judge received the
slur, in the title, and still scored the video near or below zero, because the
footage is a dance performance, a personal clip, a comment-thread screen
recording and a street scene respectively. The judge is weighing the visual
evidence against the title and siding with the visuals.

## 6. False-positive taxonomy

The twenty highest-scoring normals.

| Code | Meaning | n | Rate | Wilson 95% | Median z |
|---|---|---:|---:|---|---:|
| A | profanity or insult with no group target, often only in the title | 6 | 30% | 15–52% | 9.00 |
| M | clinical sexual-health advertorial, anatomical vocabulary | 4 | 20% | 8–42% | 8.00 |
| O | other benign, mild suggestion or a folkloric nickname | 3 | 15% | 5–36% | 7.50 |
| Q | counter-speech: reproduces prejudice in order to criticise it | 2 | 10% | 3–30% | 9.75 |
| P | protected group present as topic, not attacked | 2 | 10% | 3–30% | 8.00 |
| F | fiction or filmed conflict | 1 | 5% | 1–24% | 9.00 |
| R | news or report about harm | 1 | 5% | 1–24% | 8.75 |
| D | content the judge is arguably right about, annotated Normal | 1 | 5% | 1–24% | 7.50 |

Ten of the twenty, codes A and M, are lexical: an offensive word or a clinical
anatomical word appears in the title or the narration and nothing in the video
targets anyone. Four more, Q and P, are the mirror-image error that every corpus
in this project has produced: a video whose subject is prejudice, quoted in order
to criticise it, or a protected group discussed sympathetically. One negative,
code D, is covert hotel-room footage annotated Normal, which is the only case
here where the judge looks right and the label looks wrong. That is one video out
of twenty, against seventeen out of fifty on HateClipSeg. **MHClip-ZH's negative
class is not full of mislabelled hate.**

## 7. Construct measurement: protected target versus target-free offensiveness

All 45 union positives were coded for whether the video expresses hostility
toward a group defined by a protected attribute. This coding was done by the same
reader who could see the scores, so it is score-aware and descriptive; it is not
a blind audit like the one run on MHClip-EN and HateClipSeg.

| Positive population | n | Median z | AUC vs Normal | 95% CI | Fine labels |
|---|---:|---:|---:|---|---|
| Explicit protected-group hostility | 13 | 10.00 | 0.917 | 0.804–0.988 | 10 H, 3 O |
| Implicit protected-group hostility | 8 | 5.63 | 0.731 | 0.581–0.872 | 2 H, 6 O |
| No protected group involved | 24 | 7.88 | 0.862 | 0.789–0.923 | 5 H, 19 O |
| All positives | 45 | 8.50 | 0.855 | 0.789–0.913 | |

Twenty-four of 45 positives, 53%, carry no protected-group target: covert
footage, infidelity confrontations, filmed bullying, gaming-fandom feuds,
celebrity slut-shaming, insults aimed at one named individual. MHClip-EN's blind
audit put the same figure at 69%.

**The construct mismatch is present on ZH but it does not bind.** On MHClip-EN
the target-free positives ranked at 0.760 against explicit hostility's 0.983, a
gap of 22 points, and they were the largest population, which is what dragged
that corpus down. On MHClip-ZH the target-free positives rank at 0.862, *above*
the corpus figure and only 5.5 points below explicit hostility. The judge handles
Chinese target-free offensiveness about as well as it handles Chinese
protected-group hostility. What separates on ZH is implicitness, not
target-hood: 0.917 explicit against 0.731 implicit, the same axis MHClip-EN's
autopsy ended on.

The Hateful-only stratum does not separate cleanly the way MHClip-EN's explicit
stratum did. Hateful versus Normal is 0.877; restricting to the ten Hateful
videos that are also explicit protected-group hostility gives 0.909, and the
explicit stratum across both positive labels gives 0.917. The lift from the fine
label is small because five of the seventeen Hateful videos express no
protected-group hostility at all and two more express it only implicitly.

## 8. Covariate screen

For every code with at least four videos, against the corpus medians of
voice-activity fraction 0.637, judged transcript length 88 characters and
duration 30.8 seconds:

| Side | Code | n | Median z | Median VAD | Median judged chars | Median duration | Boilerplate | Harvest keyword |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FN | G | 6 | 1.00 | 0.555 | 89.0 | 26.7 s | 0 | 4 |
| FN | F | 6 | 6.88 | 0.580 | 90.5 | 36.4 s | 1 | 2 |
| FN | T | 4 | 1.88 | 0.021 | 22.5 | 15.3 s | 2 | 4 |
| FP | A | 6 | 9.00 | 0.341 | 28.5 | 13.7 s | 3 | 4 |
| FP | M | 4 | 8.00 | 0.913 | 249.0 | 52.6 s | 2 | 4 |

Three covariates were measured corpus-wide.

**Recogniser boilerplate: common, and irrelevant.** In 33 of 149 videos, 22.1%,
the text the judge read is a recogniser artefact rather than the video's speech:
an English stock closing line, a subtitling-group credit, a foreign-language
sign-off or one channel's donation plug. Every one of the 33 passed the
degeneracy gate, because the gate rejects only on the conjunction of near-silence
and extreme repetition. Their median voice-activity fraction is 0.291 against
0.677 for the rest. And it makes no difference to the ranking: AUC 0.849 on the
boilerplate subset against 0.854 on the clean subset, consistent with the
NOTRANSCRIPT arm showing the whole channel is worth nothing.

**Non-Chinese recogniser output: also common, also irrelevant.** In 24 of 149
videos, 16.1%, the recogniser's dominant script is not Han. Their median
voice-activity fraction is 0.000; ten of the 24 overlap the boilerplate set.
Whisper is inventing text for silent videos. The two failures are the same
failure, and neither moves the number.

**Harvest keyword: the one covariate that separates, and it is label-free.**
Presence of the harvester's highlighted term in the title is a standalone
detector of the binary label at AUC 0.707 with no model involved, it marks 52% of
the top-scoring normals, and it splits the corpus into a hard half at 0.780 and
an easy half at 0.899. This is the first covariate in this project that
identifies the misranked population. It is also an artefact of how MHClip was
collected and should not be treated as a general routing signal.

**Duplicates.** Two exact judge-transcript duplicate groups exist and one
disagrees about the binary collapse, but both groups are boilerplate collisions
rather than duplicated videos, so this says nothing about the annotation.

## 9. What this licenses

1. **MHClip-ZH's oracle ceiling needs no separate explanation.** It is 3.2 points
   above a binormal reference at the same AUC and prevalence and better placed
   than two of the three sister corpora. Anyone quoting 0.7932 as evidence of a
   thresholding problem is quoting the AUC and the 30% prevalence.
2. **The corpus's difficulty is a keyword-matched negative class.** Thirty
   percent of the normals were harvested with an offensive query term; against
   the plain negatives the same judge reaches 0.894 AUC and 0.859 oracle
   macro-F1. Any cross-corpus comparison involving MHClip-ZH should say this,
   because it is a property of the sampling frame, not of the judge.
3. **The open problem on MHClip-ZH is implicitness, at 0.731.** Explicit
   protected-group hostility ranks at 0.917 and implicit hostility at 0.731, on
   eight videos with a wide interval. This is the same open problem MHClip-EN's
   autopsy ended on, now found in a second language, which is the first evidence
   that it is not an English-corpus artefact.
4. **The title is a first-class evidence channel on this corpus and the speech is
   not.** Any future ZH measurement that manipulates inputs must report what it
   does to the title, because that field carries eight AUC points and the
   transcript carries none.

## 10. What this kills

1. **"The judge cannot read Chinese" is dead.** Explicit protected-group
   hostility in Chinese ranks at 0.917, the Hateful class at 0.877, and the plain
   negatives at 0.894. Language is not the constraint.
2. **The restoration-class hypothesis is dead on MHClip-ZH, and for a reason
   worth recording.** Channel restoration measured neutral here, and the natural
   reading was that the speech-poor corpus starved the channel. That reading is
   wrong. The channel is not starved; it is redundant. Blanking the transcript
   entirely costs 0.002 AUC with an interval of -0.033 to +0.041, because the
   title already carries what the speech would say. A better recogniser, a better
   gate or a longer transcript cap cannot recover points that are already
   counted.
3. **The degeneracy gate's blind spot is real and does not matter.** It admits
   all 33 recogniser-boilerplate transcripts, and those 33 videos rank at 0.849
   against 0.854 for the rest. Fixing the gate on this corpus would change
   nothing measurable.
4. **The construct-mismatch story does not transfer from MHClip-EN.** Both
   corpora put a majority of non-protected-target content in the positive class,
   53% here against 69% there, but on ZH that population ranks at 0.862, above
   the corpus figure. Target-free offensiveness is not what makes MHClip-ZH hard.
5. **The annotation-error story does not transfer from HateClipSeg either.** One
   of the twenty highest-scoring normals is content the judge is arguably right
   about, against seventeen of fifty on HateClipSeg. The ZH negative class is
   clean; it is merely adversarially sampled.

## Reproduction

```bash
python3 scripts/duplex/zh_ranking_autopsy_decompose.py
python3 scripts/duplex/zh_ranking_autopsy_ceiling.py
python3 scripts/duplex/zh_ranking_autopsy_packet.py
python3 scripts/duplex/zh_ranking_autopsy_analyze.py
python3 scripts/duplex/zh_channel_ablation.py {notitle,notranscript,nomarkup,framesonly}
python3 scripts/duplex/zh_channel_ablation_analyze.py
```

Item-level codings, the coded packet, the frame montages, the four ablation arms
and the aggregate summaries are under `results/ranking_autopsy/zh/`, which is
gitignored because those files carry video identifiers and transcript text. Only
aggregate counts and rates appear above.
