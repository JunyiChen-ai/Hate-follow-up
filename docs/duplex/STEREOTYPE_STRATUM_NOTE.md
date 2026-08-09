# Stereotype stratum on HateClipSeg: does the gender deficit replicate?

**Date:** 2026-08-09. **Type:** diagnostic content coding plus analysis. CPU
only, no model call, no GPU. All statistics use the frozen single-call
Qwen3-VL-8B raw-z scores already committed for HateClipSeg. Label use is
evaluative only.

## Headline

The implicitness note left one model-side weakness open. On the pooled MHClip
implicit stratum the judge ranks gender and sexuality stereotype content at
about 0.71, flat across every evidence-length bin, while ranking ImpliHateVid's
race and nationality dog-whistles at 0.9255. Fifteen of those eighteen MHClip
videos are gender or sexuality content, so the stratum and the attribute are
confounded and cannot be separated inside MHClip. HateClipSeg was the candidate
third corpus because its insulting-only stratum is 59 videos ranked at 0.637
against the same negative class the corpus ships.

One hundred forty-nine HateClipSeg videos were coded from their fresh
transcripts and four sampled frames each: all 59 insulting-only videos, all 50
videos carrying no offensive category, and a 40-video random sample of the 180
hateful-labelled videos drawn at seed 20260808. Each video received one code
for what its offence rests on: gender or sexuality stereotype and demeaning
gender framing (GS), other-group stereotype or conspiracy framing (OS), direct
and overt hostility on a non-gender axis (EX), or none of these (NO).

Three findings, in order of how much they change the picture.

First, the replication fails where it matters and succeeds only where the
comparison is unsound. Inside the insulting-only stratum, which holds the
corpus's own severity assignment fixed, the sixteen gender-stereotype videos
rank at 0.640 and the seventeen other group-directed videos rank at 0.768. That
is a 12.8-point gap whose confidence intervals overlap across most of their
range and whose rank test returns p = 0.17. Pooling the insulting-only stratum
with the hateful sample produces 0.692 for gender against 0.879 for overt
hostility, which looks exactly like the MHClip pattern, but that pooled contrast
mixes two annotation strata and is therefore a measurement of the corpus's
severity labelling as much as of the judge.

Second, the gender cohort explains none of the 0.637. Dropping all sixteen
gender-stereotype videos from the insulting-only stratum moves its AUC from
0.6373 to 0.6363, a change of one thousandth of a point in the wrong direction.
The stratum is weak across every code it contains, including the 26 videos that
carry no group content at all, which rank at 0.550. What does move the number is
the negative class: scored against the 34 shipped normals this coding finds
genuinely benign, the same 59 videos rank at 0.788.

Third, the corpus itself does not call this content hateful. Of the 23
gender-stereotype videos found across all three cohorts, sixteen are annotated
insulting only, four are annotated normal, and three carry the hateful label.
Those three rank at 0.970, above the eighteen overtly hostile hateful-labelled
videos at 0.932. When HateClipSeg's own annotation says a gender-directed video
is hate, the judge puts it at the top of the corpus.

## 1. What the coded cohorts contain

Codes are exclusive and describe what the video's offence principally rests on.
The register column counts videos whose hostility is overt rather than delivered
as mockery, stereotype or conspiracy claim. AUC is against the 50 videos that
carry no offensive category, which is the negative class the ranking autopsy
used to produce 0.637.

### Insulting-only stratum, 59 videos

| Code | n | Median z | Median fresh chars | Median VAD | Overt register | AUC vs clean normal | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---|
| GS | 16 | 10.13 | 2404 | 0.51 | 4 | 0.640 | 0.499–0.775 |
| OS | 9 | 12.75 | 2999 | 0.81 | 0 | 0.776 | 0.618–0.910 |
| EX | 8 | 13.38 | 1971 | 0.55 | 8 | 0.760 | 0.580–0.910 |
| NO | 26 | 6.50 | 2419 | 0.51 | 0 | 0.550 | 0.422–0.679 |

### Hateful-labelled random sample, 40 videos

| Code | n | Median z | Median fresh chars | Median VAD | Overt register | AUC vs clean normal | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---|
| GS | 3 | 16.25 | 2759 | 0.73 | 2 | 0.970 | 0.917–1.000 |
| OS | 14 | 14.38 | 3082 | 0.76 | 0 | 0.848 | 0.736–0.939 |
| EX | 18 | 15.25 | 2540 | 0.50 | 18 | 0.932 | 0.865–0.984 |
| NO | 5 | 13.25 | 3107 | 0.66 | 0 | 0.798 | 0.592–0.952 |

### Videos carrying no offensive category, 50 videos

| Code | n | Median z | Median fresh chars | Median VAD | Overt register |
|---|---:|---:|---:|---:|---:|
| GS | 4 | 6.13 | 648 | 0.00 | 2 |
| OS | 2 | 12.63 | 2717 | 0.69 | 0 |
| EX | 10 | 13.00 | 222 | 0.02 | 10 |
| NO | 34 | -1.13 | 1948 | 0.06 | 0 |

Sixteen of the fifty shipped normals carry group-hostile or group-stereotyping
content under this codebook, and their median score is far above the benign
remainder. This reproduces, from a different codebook, the finding the ranking
autopsy already reported: the earlier autopsy counted seventeen such videos
under codes for explicit hostility and extremist glorification. The two
codebooks disagree on one video and agree on the substance.

## 2. The replication question

The question is whether gender-stereotype content ranks near 0.71 while overt
hostility ranks near 0.90 inside one corpus against one negative class. Two
answers exist depending on whether the corpus's own severity labelling is held
fixed.

| Comparison | GS n | GS AUC | GS CI | Comparator n | Comparator AUC | Comparator CI | Gap | Rank-test p |
|---|---:|---:|---|---:|---:|---|---:|---:|
| Insulting-only stratum, GS vs EX | 16 | 0.640 | 0.499–0.775 | 8 | 0.760 | 0.580–0.910 | 0.120 | 0.374 |
| Insulting-only stratum, GS vs EX and OS together | 16 | 0.640 | 0.499–0.775 | 17 | 0.768 | — | 0.128 | 0.171 |
| Hateful sample, GS vs EX and OS together | 3 | 0.970 | 0.917–1.000 | 32 | 0.895 | — | **-0.075** | 0.329 |
| Both positive cohorts pooled, GS vs EX | 19 | 0.692 | 0.553–0.816 | 26 | 0.879 | 0.794–0.951 | **0.187** | 0.0068 |

The pooled row is the one that matches MHClip. Its 0.692 sits inside the
interval around MHClip's 0.71 and its 0.879 sits near ImpliHateVid's 0.9255. It
is also the only row in the table that is not a controlled comparison, because
nineteen gender-stereotype videos supply three cases to the hateful cohort and
sixteen to the insulting cohort while twenty-six overtly hostile videos supply
eighteen and eight. Holding the annotation stratum fixed shrinks the gap to
0.128 in the stratum where the gender cohort is large, and reverses it in the
stratum where the gender cohort is small.

Splitting the gender cohort by register does not locate a mockery-specific
deficit either. Within the insulting-only stratum the twelve videos delivered as
mockery, comedy or stereotype rank at 0.653 and the four delivered as overt
hostility rank at 0.603. Dropping the six codes marked low-confidence raises the
gender cohort to 0.686 and the overt-hostility cohort to 0.733, a gap of 4.7
points on eleven and seven videos.

**Verdict: partial replication at best, and the part that replicates is the part
that is confounded.** The direction is right in three of the four rows above.
The magnitude only reaches the MHClip figure when the comparison pools across
the corpus's own hateful-versus-insulting boundary. No controlled row separates
the two populations statistically.

## 3. How much of the 0.637 is gender-attributable

The insulting-only stratum reproduces the autopsy figure exactly at 0.6373 on 59
videos. Each row below removes one code from the stratum and rescores the
remainder against the same 50 negatives.

| Stratum after removal | n | AUC vs clean normal | Change |
|---|---:|---:|---:|
| All insulting-only videos | 59 | 0.6373 | — |
| Excluding GS | 43 | 0.6363 | **-0.0010** |
| Excluding OS | 50 | 0.6124 | -0.0249 |
| Excluding EX | 51 | 0.6180 | -0.0193 |
| Excluding NO | 33 | 0.7061 | +0.0688 |

**The gender-attributable share of the 0.637 is zero.** Removing the entire
gender cohort leaves the stratum one thousandth of a point lower, because the
cohort ranks at 0.640 and the stratum ranks at 0.637. The stratum is not weak
because of a hard sub-population; it is uniformly weak. The only removal that
raises it materially is removing the 26 videos with no group content at all,
which are the population the corpus calls insulting for profanity, personal
feuds, political abuse and crude comedy.

The negative class matters more than any code. Rescored against the 34 shipped
normals this coding finds benign, the whole stratum moves from 0.637 to 0.788, a
gain of 15.1 points. The gender cohort moves from 0.640 to 0.803 and the overt
cohort from 0.760 to 0.879. This is the same effect the ranking autopsy measured
at corpus level and the implicitness note measured on MHClip, and on this corpus
it is roughly ten times the size of the gender-versus-overt contrast.

## 4. Pooled across corpora, descriptively

The fifteen MHClip gender videos come from the implicitness note's keyword tally
over the score-aware coder notes, reconstructed here with the same rule so the
count reproduces. The percentile is each video's rank position inside its own
corpus's full score distribution.

| Corpus | Gender-stereotype videos | Median z | Median percentile in corpus | AUC vs that corpus's normals |
|---|---:|---:|---:|---:|
| MHClip-EN | 9 of 10 implicit | 0.50 | 65.2 | 0.744 |
| MHClip-ZH | 6 of 8 implicit | 1.00 | 51.3 | 0.676 |
| HateClipSeg | 23 of 149 coded | 10.75 | 39.5 | 0.669 |
| Combined | **38** | — | 49.5 | — |

The three AUC figures agree with each other more closely than any of them agrees
with ImpliHateVid's 0.9255. That is the strongest statement this analysis
supports, and it is weaker than it looks for three reasons. The MHClip figures
come from strata whose coding was score-aware, a defect the implicitness note
established and did not repair. The HateClipSeg figure is against a negative
class that is itself a third group-hostile content. And the percentile column is
not comparable across corpora, because HateClipSeg is 87 percent positive under
its own offensive union while MHClip-EN is 30 percent, so the same content sits
at a lower percentile in the corpus with more positives.

The combined count is 38, not the 50 or more that would make this a settled
phenomenon. The honest summary is that three corpora now place
gender-stereotype content between 0.67 and 0.74 against their own negatives,
that no one of the three measurements is clean, and that all three lack a
within-corpus comparison group of implicit non-gender hostility large enough to
separate the attribute from the register.

## 5. Confound checks

### Evidence covariates

The nineteen gender-stereotype videos in the two positive cohorts are compared
against the twenty-six overtly hostile ones. No covariate separates them.

| Covariate | Median, GS | Median, EX | Mann-Whitney p |
|---|---:|---:|---:|
| Fresh transcript characters | 2538 | 2443 | 0.991 |
| Judge-visible characters | 2538 | 2276 | 0.899 |
| VAD speech fraction | 0.620 | 0.540 | 0.565 |
| Audio duration | 253.4 s | 239.0 s | 0.145 |

All nineteen gender videos and all twenty-six overtly hostile videos received a
transcript that passed the degeneracy gate, so the judge read speech in every
case. The evidence-volume account, already dead on MHClip, has nothing to work
with here either.

### Annotation side

This is the check that changes the reading. It asks where HateClipSeg's own
annotators placed each content code.

| Code | Total coded | Annotated hateful | Annotated insulting only | Annotated with no offensive category |
|---|---:|---:|---:|---:|
| GS | 23 | 3 | 16 | 4 |
| OS | 25 | 14 | 9 | 2 |
| EX | 36 | 18 | 8 | 10 |
| NO | 65 | 5 | 26 | 34 |

Gender-stereotype content is the only code that HateClipSeg predominantly files
under insulting rather than hateful. Sixteen of the nineteen that carry any
offensive label carry the lower one. The judge's low ranking of that population
is therefore not straightforwardly a model deficit against the benchmark's own
boundary; on this corpus the judge and the annotators broadly agree that the
content is offensive without being hate. The three cases where the annotators
disagree and mark the content hateful are the three the judge ranks at 0.970.

The comparison population has the mirror property. Ten of the thirty-six
overtly hostile videos sit in the class annotated with no offensive category at
all, which is the annotation error the ranking autopsy documented. Those ten are
in the negative class of every AUC in this note and they depress all of them.

### Severity, not attribute

A rival reading of the pooled 0.692-against-0.879 gap is that gender-stereotype
content in this corpus is simply milder, and that the judge is tracking severity
rather than failing on an attribute. The matched-stratum rows in section 2 are
consistent with that reading. Within the insulting-only stratum the gap shrinks
to 0.128 and is not significant; within the hateful stratum it reverses. This
analysis cannot distinguish "the judge under-ranks gender-stereotype content"
from "gender-stereotype content is milder and the judge ranks by severity",
because on this corpus the two hypotheses make the same prediction everywhere
except in the three-video cell where they disagree and the attribute hypothesis
loses.

## What this licenses

1. **A three-corpus descriptive statement, with caveats attached.** Gender and
   sexuality stereotype content ranks between 0.669 and 0.744 against its own
   corpus's negatives on MHClip-EN, MHClip-ZH and HateClipSeg, on a combined 38
   videos. None of the three measurements is confound-free and the count is not
   large.
2. **The negative-class finding, again and larger.** On HateClipSeg the shipped
   negative class costs the insulting-only stratum 15.1 AUC points, which is
   larger than every content-code contrast in this note put together. Any future
   number quoted from this corpus should say which negative class it uses.
3. **The insulting-only stratum is a severity stratum, not a content stratum.**
   Twenty-six of its 59 videos carry no group content of any kind and rank at
   0.550. The stratum's 0.637 is a statement about what HateClipSeg calls
   insulting, not about a capability of the judge.
4. **The three-video cell is worth naming.** Where HateClipSeg's annotators call
   gender-directed content hateful, the judge ranks it at 0.970, above the
   overtly hostile hateful-labelled videos. Three videos cannot carry a claim,
   but they are the only direct test in this corpus of whether the judge can rank
   gender-directed hate when the benchmark agrees it is hate, and it passes.

## What this kills

1. **The claim that the gender deficit is now a three-corpus phenomenon is dead
   as stated.** HateClipSeg reproduces the direction and, when pooled across
   annotation strata, the magnitude. It does not reproduce it in any comparison
   that holds the corpus's own severity assignment fixed. The controlled gap is
   0.128 with p = 0.171 on sixteen videos against seventeen, and it reverses to
   -0.075 in the other stratum.
2. **The hypothesis that gender-stereotype content is what makes HateClipSeg's
   insulting-only stratum weak is dead.** Removing that content changes the
   stratum's AUC by -0.001. The successor probe the implicitness note proposed
   has been run on the population it named, and the answer is no.
3. **The evidence-volume account stays dead on a third corpus.** Transcript
   length, judge-visible length, speech fraction and duration all fail to
   separate the gender cohort from the overtly hostile one, at p between 0.15
   and 0.99, and every video in both cohorts reached the judge with speech.
4. **A pure model-deficit framing of the gender finding is no longer available.**
   HateClipSeg files sixteen of nineteen offensive-labelled gender-stereotype
   videos as insulting rather than hateful. The judge scoring that population
   below overt hate is, on this corpus, agreement with the annotation rather than
   failure against it. Whether MHClip's annotators would have drawn the same line
   is a question about MHClip's construct, and it is now the live question rather
   than the model's capability.
5. **The severity-versus-attribute ambiguity cannot be resolved with the data on
   disk.** Every controlled comparison available here has either sixteen videos
   against seventeen or three against thirty-two. Resolving it needs a corpus
   with implicit non-gender hostility and implicit gender hostility inside the
   same severity stratum, which is the same gap the implicitness note identified
   and which this corpus does not fill.

## Reproduction

```bash
/home/jehc223/venvs/SafetyContradiction/bin/python \
    scripts/duplex/stereotype_stratum_packet.py
/home/jehc223/venvs/SafetyContradiction/bin/python \
    scripts/duplex/stereotype_stratum_analyze.py
```

The machine-readable output is `results/stereotype_stratum/results.json`. The
per-video codes live in `results/stereotype_stratum/coding.tsv` and the coding
packet with transcripts and frames lives beside it, both gitignored because they
carry video identifiers and transcript text. Only aggregate counts, medians and
AUCs appear above. No model call was made at any point in this analysis, and no
lexicon was consulted.
