# Implicitness and evidence volume: is the judge's implicit-hate deficit real?

**Date:** 2026-08-09. **Type:** diagnostic cross-corpus analysis. CPU only, no
model call, no GPU. All statistics use the frozen single-call Qwen3-VL-8B raw-z
scores already committed for MHClip-EN, MHClip-ZH and ImpliHateVid. Label use is
evaluative only.

## Headline

The hypothesis under test was that the judge's implicit-hate deficit is bound to
evidence volume: implicit hostility needs context, so the judge should succeed on
implicit hate where the transcript is long and fail where it is short. The
hypothesis fails on every measurement taken here, and it fails in the wrong
direction twice.

Three findings, in order of how much they change the picture.

First, the MHClip-EN implicit and explicit strata are not an independent
partition of the positive class. The seven videos the autopsy called explicit are
exactly the seven highest-scoring of the seventeen protected-target positives,
with the lowest explicit case at z = +8.50 and the highest implicit case at
z = +6.00. If a coder assigned seven of seventeen labels without reference to the
score, the chance of landing on exactly the top seven is one in 19,448. The
MHClip-EN contrast of 0.983 against 0.725 is therefore a partition by score
reported as a partition by content, and it cannot be quoted as a measurement of
implicitness. MHClip-ZH's split does not have this defect: its lowest-scoring
explicit case sits at z = -8.25 and all eight implicit cases outrank it.

Second, ImpliHateVid carries gold implicitness labels in its own video
identifiers, and those labels are score-blind. Under them the judge separates
explicit hate from normal at 0.9731 and implicit hate from normal at 0.9255. The
implicitness cost on the one corpus where implicitness is labelled independently
of the score is **4.8 AUC points**, not the 25.7 points MHClip-EN reports or the
18.6 points MHClip-ZH reports.

Third, evidence volume does not track the deficit. MHClip-ZH's explicit stratum
has a median fresh transcript of 28 characters and its implicit stratum has 89,
so the stratum with more speech is the one that ranks 18.6 points lower. Within
the pooled MHClip implicit set the Spearman correlation between score and fresh
transcript length is -0.199 on eighteen videos. Binning implicit positives by
transcript length and scoring each bin against normals from the same bin leaves
the MHClip implicit AUC flat at 0.713, 0.712 and 0.753 across the three bins.
Matched head to head in the 150-to-600-character bin, MHClip implicit positives
rank at 0.712 and ImpliHateVid implicit positives rank at 0.911. Matching the
evidence does not close the gap.

An evidence-volume effect does exist, but it lives inside ImpliHateVid, which is
the corpus where the judge is already strong. There the score correlates with
transcript length at +0.240 among implicit positives and +0.346 among explicit
positives, and the twenty lowest-scoring positives are evidence-thin at a median
of 685 fresh characters against 1,480 for the rest, with a Mann-Whitney p of
0.014. Converted into ranking, that effect is worth about two AUC points between
the middle and the top length bin. It is real, it is small, and it is not what
separates MHClip's implicit stratum from ImpliHateVid's.

## 1. The three corpora side by side

All strata are scored against their own corpus's shipped normals. MHClip strata
come from score-aware coding and carry the provenance warning in the headline.
ImpliHateVid strata come from the corpus's own gold identifier prefixes.

| Corpus | Stratum | n | Median z | Median fresh chars | Median VAD fraction | Median duration | ROC-AUC | 95% bootstrap CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MHClip-EN | Explicit | 7 | 13.50 | 748 | 0.889 | 49.5 s | 0.9828 | 0.957–1.000 |
| MHClip-EN | Implicit | 10 | 0.25 | 523 | 0.673 | 44.4 s | 0.7254 | 0.618–0.828 |
| MHClip-EN | No protected target | 32 | 3.00 | 499 | 0.749 | 35.9 s | 0.7599 | 0.673–0.842 |
| MHClip-EN | Normal | 112 | -9.50 | 432 | 0.751 | 32.4 s | — | — |
| MHClip-ZH | Explicit | 13 | 10.00 | 28 | 0.440 | 15.1 s | 0.9172 | 0.807–0.987 |
| MHClip-ZH | Implicit | 8 | 5.63 | 89 | 0.798 | 28.1 s | 0.7314 | 0.577–0.867 |
| MHClip-ZH | No protected target | 24 | 7.88 | 63 | 0.582 | 31.3 s | 0.8620 | 0.792–0.920 |
| MHClip-ZH | Normal | 104 | -5.00 | 111 | 0.674 | 32.3 s | — | — |
| ImpliHateVid | Explicit, gold | 91 | 12.25 | 1471 | 0.715 | 100.3 s | 0.9731 | 0.957–0.986 |
| ImpliHateVid | Implicit, gold | 108 | 7.88 | 1242 | 0.772 | 102.2 s | 0.9255 | 0.895–0.951 |
| ImpliHateVid | Normal, gold | 201 | -16.50 | 1026 | 0.848 | 62.3 s | — | — |

The within-corpus explicit-minus-implicit gap is the quantity that controls for
how hard each corpus's negative class is. It is 0.257 on MHClip-EN, 0.186 on
MHClip-ZH and 0.048 on ImpliHateVid. The first of those three is inflated by the
partition defect described in the headline. The third is the only one measured
against labels that never saw a score.

Two of the three corpora also fail to separate the two strata statistically. On
MHClip-ZH the explicit interval runs 0.807 to 0.987 and the implicit interval
runs 0.577 to 0.867, and the two overlap. On ImpliHateVid the intervals do not
overlap, so a 4.8-point implicitness effect is established there on 199
positives.

## 2. Score against evidence volume, within strata

Spearman correlation of raw z with each covariate, computed inside each stratum
so that the between-stratum score difference cannot drive the result.

| Population | n | Fresh chars | Judge chars | VAD fraction | Duration |
|---|---:|---:|---:|---:|---:|
| MHClip-EN implicit | 10 | -0.127 | -0.127 | -0.188 | -0.067 |
| MHClip-EN explicit | 7 | +0.179 | +0.143 | -0.214 | +0.321 |
| MHClip-ZH implicit | 8 | +0.476 | +0.476 | +0.048 | +0.286 |
| MHClip-ZH explicit | 13 | +0.101 | +0.101 | +0.253 | +0.289 |
| MHClip pooled implicit | 18 | **-0.199** | -0.199 | -0.126 | -0.121 |
| MHClip pooled explicit | 20 | +0.253 | +0.243 | +0.249 | +0.395 |
| ImpliHateVid implicit | 108 | **+0.240** | +0.299 | +0.227 | +0.230 |
| ImpliHateVid explicit | 91 | **+0.346** | +0.420 | +0.326 | +0.264 |
| ImpliHateVid normal | 201 | +0.137 | +0.095 | -0.100 | +0.185 |

No MHClip correlation reaches significance at any conventional level, and the
pooled implicit correlation carries the wrong sign for the hypothesis. Every
ImpliHateVid positive correlation is significant at p below 0.02. The
evidence-volume relationship is a property of ImpliHateVid, not a property of
implicit hate.

## 3. Evidence-matched comparison

Implicit positives were binned by fresh transcript length and scored against
normals from the same corpus in the same bin. Cells with fewer than five
positives or fewer than five negatives are marked unpowered.

| Population | Bin | Positives | Normals | Median z | AUC in bin | Note |
|---|---|---:|---:|---:|---:|---|
| MHClip-EN implicit | <150 | 1 | 24 | 2.50 | 0.792 | unpowered |
| MHClip-EN implicit | 150–600 | 5 | 44 | 2.25 | 0.680 | |
| MHClip-EN implicit | >600 | 4 | 44 | -0.88 | 0.753 | unpowered |
| MHClip-ZH implicit | <150 | 5 | 69 | 5.25 | 0.707 | |
| MHClip-ZH implicit | 150–600 | 3 | 32 | 7.25 | 0.786 | unpowered |
| MHClip-ZH implicit | >600 | 0 | 3 | — | — | empty |
| MHClip pooled implicit | <150 | 6 | 93 | — | **0.713** | pair-weighted |
| MHClip pooled implicit | 150–600 | 8 | 76 | — | **0.712** | pair-weighted |
| MHClip pooled implicit | >600 | 4 | 47 | — | 0.753 | unpowered |
| ImpliHateVid implicit, sample of 40 | 150–600 | 8 | 46 | 6.25 | 0.910 | seed 20260808 |
| ImpliHateVid implicit, sample of 40 | >600 | 30 | 155 | 9.13 | 0.940 | |
| ImpliHateVid implicit, all 108 | 150–600 | 19 | 46 | 5.25 | **0.911** | |
| ImpliHateVid implicit, all 108 | >600 | 85 | 155 | 8.50 | **0.931** | |

The preregistered comparison is the 150-to-600-character row. There MHClip
implicit positives rank at 0.712 and ImpliHateVid implicit positives rank at
0.911, a gap of 19.9 points at matched evidence volume. The MHClip implicit AUC
does not rise with the bin, and the ImpliHateVid implicit AUC rises by only 2.0
points from the middle bin to the top bin.

Two cells cannot be filled. MHClip-ZH has three normals above 600 characters and
no implicit positives there. ImpliHateVid has four implicit positives below 150
characters and no normals at all below 150, so the thinnest bin has no
within-corpus comparison. Both absences are reported rather than papered over.

The explicit contrast is the sharpest single number in this section. MHClip-ZH's
thirteen explicit positives all sit in the shortest bin, below 150 characters,
and against the 69 normals in that same bin they reach 0.918. The judge separates
Chinese explicit hostility at 0.918 with a median of 28 transcript characters.
Evidence volume is plainly not what it needs.

## 4. Negative-class control

The MHClip-EN negative class was cleaned by removing the eight shipped Normals
that blind coders in the annotation-validity audit placed in the Hateful or
Offensive categories. The MHClip-ZH negative class was cleaned by removing the
31 normals whose shipped title carries the harvester's highlighted query term.

| Corpus | Stratum | AUC, shipped normals | AUC, clean normals | Delta |
|---|---|---:|---:|---:|
| MHClip-EN | Explicit | 0.9828 | 0.9842 | +0.001 |
| MHClip-EN | Implicit | 0.7254 | 0.7543 | **+0.029** |
| MHClip-EN | No protected target | 0.7599 | 0.7814 | +0.021 |
| MHClip-ZH | Explicit | 0.9172 | 0.9315 | +0.014 |
| MHClip-ZH | Implicit | 0.7314 | 0.7937 | **+0.062** |
| MHClip-ZH | No protected target | 0.8620 | 0.9061 | +0.044 |

The removed normals are the high-scoring ones, as expected: median z of +3.13 on
the eight EN removals against -9.88 for the normals kept, and +2.25 on the 31 ZH
removals against -6.50 for the normals kept. Cleaning helps the implicit stratum
more than the explicit stratum, because the explicit stratum is already ranked
above almost every normal and has little left to gain.

Cleaning the negative class closes 2.7 of the 25.7-point MHClip-EN gap and 4.8 of
the 18.6-point MHClip-ZH gap. In cross-corpus terms it closes 2.9 of the 20.0
points separating MHClip-EN implicit from ImpliHateVid implicit, and 6.2 of the
19.4 points on MHClip-ZH.

Applying both controls together, clean negatives and same-bin length matching,
gives a pair-weighted MHClip-EN implicit AUC of 0.746 and a MHClip-ZH implicit
AUC of 0.790. Length matching subtracts a fraction of a point on both corpora
after the negative class has been cleaned, which is to say it contributes
nothing.

## 5. ImpliHateVid cross-check

The ImpliHateVid test split holds 91 explicit positives, 108 implicit positives
and 201 normals. Among all 199 positives the score correlates with fresh
transcript length at +0.282 and with judge-visible transcript length at +0.347,
both significant.

The twenty lowest-scoring positives are seventeen implicit and three explicit,
with a median z of -7.75 against +11.00 for the positive class as a whole. They
are evidence-thin on two covariates and not on the other two.

| Covariate | Median, bottom twenty | Median, remaining positives | Mann-Whitney p |
|---|---:|---:|---:|
| Fresh transcript chars | 685 | 1480 | 0.014 |
| Judge-visible chars | 685 | 1017 | 0.147 |
| VAD speech fraction | 0.766 | 0.757 | 0.787 |
| Audio duration | 56.1 s | 105.0 s | 0.072 |

The bottom cohort is shorter and quieter than the rest, but its median of 685
fresh characters is still above the median of every MHClip stratum in this study,
and it is 7.7 times MHClip-ZH's implicit median of 89. ImpliHateVid's
evidence-poor tail is richer than MHClip's evidence-rich head. No transcript was
read for this section, and no video identifier appears anywhere in it.

## 6. Implicitness type

The two MHClip implicit strata are not made of the same material as
ImpliHateVid's. Counts below are a keyword tally over the score-aware coder
notes, and a video can carry more than one attribute.

| Stratum | n | Gender or sexuality | Race, nationality or language | Religion or custom |
|---|---:|---:|---:|---:|
| MHClip-EN implicit | 10 | 9 | 0 | 1 |
| MHClip-EN explicit | 7 | 6 | 3 | 2 |
| MHClip-ZH implicit | 8 | 6 | 1 | 1 |
| MHClip-ZH explicit | 13 | 6 | 3 | 0 |
| MHClip pooled implicit | 18 | **15** | **1** | 2 |
| MHClip pooled explicit | 20 | 12 | 6 | 2 |

Fifteen of the eighteen pooled implicit videos are gender-stereotype or
sexuality-stereotype content, and exactly one touches race, nationality or
language. Both autopsies independently identified that same population as a
low-scoring cohort under their own taxonomy codes: MHClip-EN's G code covers four
of the twenty-five worst false negatives and MHClip-ZH's G code covers six of
twenty with a median z of +1.00. ImpliHateVid is built from United States social
media hate speech, and its implicit cases are predominantly race, ethnicity and
immigration directed.

This confound cannot be resolved with the data on disk. Within MHClip there is
one implicit positive that is not gender-directed, so an implicit non-gender
comparison group does not exist. The type hypothesis is therefore live and
unmeasured rather than confirmed.

## 7. Hypothesis ranking, with the arithmetic

The quantity being explained is stated two ways. The within-corpus gap is
explicit AUC minus implicit AUC inside one corpus, which controls for negative
class difficulty. The cross-corpus deficit is ImpliHateVid's implicit AUC of
0.9255 minus each MHClip corpus's implicit AUC.

| Quantity | MHClip-EN | MHClip-ZH |
|---|---:|---:|
| Within-corpus gap, shipped negatives | 0.257 | 0.186 |
| Within-corpus gap, clean negatives | 0.230 | 0.138 |
| Cross-corpus deficit against ImpliHateVid implicit | 0.200 | 0.194 |
| Closed by cleaning the negative class | 0.029 | 0.062 |
| Closed by length matching, on top of clean negatives | -0.008 | -0.004 |
| Residual after both controls | **0.179** | **0.136** |

**Rank 1: stratum construction, worth all 25.7 points on MHClip-EN and zero
elsewhere.** The MHClip-EN explicit set is the score-ordered top seven of the
seventeen protected positives. That partition cannot produce anything except a
large gap, so MHClip-EN contributes no independent evidence that implicitness is
hard. Removing MHClip-EN from the evidence base leaves one score-aware
replication on MHClip-ZH, at 0.186 with overlapping confidence intervals, and one
score-blind measurement on ImpliHateVid, at 0.048.

**Rank 2: negative class, worth 2.9 points on MHClip-EN and 6.2 points on
MHClip-ZH of the cross-corpus deficit.** That is 14 percent of the MHClip-EN
deficit and 32 percent of the MHClip-ZH deficit. This is the same effect the two
autopsies already documented from the other direction, and it is larger on ZH
because ZH's harvest-keyword normals are 30 percent of its negative class.

**Rank 3: implicitness type, unquantified but the largest live candidate for the
residual.** Fifteen of eighteen pooled MHClip implicit videos are gender or
sexuality stereotype content delivered as comedy or commentary, against one that
is race or nationality directed. ImpliHateVid's implicit stratum is the opposite
mixture. The residual after the first two controls is 17.9 points on MHClip-EN
and 13.6 points on MHClip-ZH, and this hypothesis is the only one remaining that
could carry it. It cannot be tested on the data now on disk.

**Rank 4: small-sample noise, sufficient to explain the MHClip-ZH gap and not the
cross-corpus one.** On MHClip-ZH the explicit interval of 0.807 to 0.987 overlaps
the implicit interval of 0.577 to 0.867, so the within-corpus gap there is not
established. Neither MHClip implicit interval contains ImpliHateVid's 0.9255, so
the cross-corpus deficit is not noise. Ten and eight videos put the MHClip point
estimates at roughly plus or minus ten AUC points either way.

**Rank 5: evidence volume, worth zero and signed the wrong way.** The pooled
MHClip implicit correlation with transcript length is -0.199. Within-bin AUCs are
flat at 0.713, 0.712 and 0.753. Length matching after cleaning the negatives
costs 0.8 points on MHClip-EN and 0.4 points on MHClip-ZH. MHClip-ZH's explicit
stratum reaches 0.918 on a median of 28 transcript characters.

## What this licenses

1. **ImpliHateVid's gold prefixes are the only usable implicitness measurement in
   this project.** They are score-blind, they cover 199 positives rather than ten
   or eight, and they put the implicitness cost at 4.8 AUC points with
   non-overlapping intervals. Any future statement about implicit hate should be
   sourced from that number.
2. **The MHClip-ZH implicit figure of 0.731 may be quoted with two caveats
   attached.** It rises to 0.794 against a negative class drawn without the
   harvester's offensive query terms, and its confidence interval overlaps the
   explicit stratum's.
3. **The type question is the successor probe, if one is wanted.** Whether the
   judge is weak on implicit hostility in general or specifically on
   gender-stereotype humour is answerable, but not with the eighteen coded
   MHClip videos now on disk. It would need a stratum of implicit
   race-or-nationality hostility inside a corpus where implicitness is labelled
   without reference to the score.
4. **The evidence-volume relationship inside ImpliHateVid is a real observation
   about that corpus.** The score tracks transcript length at rho +0.24 to +0.35
   among its positives, and its lowest-scoring positives are its shortest ones.
   The relationship is worth about two AUC points across the length bins, so it
   is a description of the corpus rather than a lever.

## What this kills

1. **The evidence-volume account of the implicitness deficit is dead.** It
   predicted a positive correlation between score and transcript length inside
   the MHClip implicit strata, and the pooled correlation is -0.199. It predicted
   that matching evidence would close the gap, and matching evidence leaves a
   19.9-point gap in the one bin where both corpora have data. It predicted that
   short-transcript videos would be where the judge fails, and MHClip-ZH's
   explicit stratum reaches 0.918 on a 28-character median.
2. **The claim that the implicitness deficit is twice replicated is dead as
   stated.** One of the two replications, MHClip-EN, partitions the positive
   class by score and reports the partition as content coding. The remaining
   replication has overlapping intervals. The honest statement is that
   implicitness costs 4.8 AUC points where it is measured score-blind, and that
   MHClip's implicit strata are additionally weak for reasons this analysis
   attributes to the negative class and to what those strata contain.
3. **A restoration-class mechanism is not licensed.** A mechanism that gets more
   context to the judge on short videos would have to move the population that
   the deficit lives in. That population shows no relationship to context volume,
   is already separated at 0.918 by the judge when the hostility is explicit and
   the transcript is 28 characters, and sits at a flat 0.71 across every length
   bin. There is nothing for extra evidence to restore. This is the same verdict
   the MHClip-ZH autopsy reached on the transcript channel through a different
   route, where blanking the transcript entirely cost 0.002 AUC.
4. **The paradox that opened this analysis is dissolved and it was not a
   paradox.** ImpliHateVid does not rank at 0.947 in spite of being an implicit
   corpus. It is not an entirely implicit corpus: 91 of its 199 positives are
   gold-labelled explicit and rank at 0.9731, and its 108 implicit positives rank
   at 0.9255. The corpus figure is the weighted average of those two, and both
   components are high.
5. **A general claim that the judge cannot detect implicit protected-group
   hostility is dead.** It detects 108 gold implicit cases at 0.9255. What
   remains open is narrower and is stated in the licensing section above.

## Reproduction

```bash
/home/jehc223/venvs/SafetyContradiction/bin/python \
    scripts/duplex/implicitness_evidence_analyze.py
```

The machine-readable output is `results/implicitness_analysis/results.json`. The
input codings live under `results/ranking_autopsy/` and
`results/annotation_validity/`, both gitignored because they carry video
identifiers and transcript text. Only aggregate counts, medians and rates appear
above. No model call was made at any point in this analysis.
