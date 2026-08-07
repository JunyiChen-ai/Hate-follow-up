# Ranking-error autopsy: MHClip-EN test and HateClipSeg

**Date:** 2026-08-08. **Type:** diagnostic autopsy. No preregistration, no method
proposal, no new model call. All measurement is on the frozen single-call
Qwen3-VL-8B judge scores already committed for these two corpora.

## Headline

The two weak corpora are weak for two different reasons, and neither reason is a
failure to deliver evidence to the model.

On **HateClipSeg**, one third of the negative class is hate content annotated as
normal. Ten of the fifty videos that the offensive-union collapse treats as
negatives are explicit protected-group hostility, and seven more glorify a
historical extremist movement. Removing those seventeen raises the corpus
ROC-AUC from **0.754 to 0.900**, which is fourteen and a half points and more
than the entire gap between HateClipSeg and HateMM. The judge was ranking those
videos near the top of the corpus, which under the shipped labels is scored as
error. One of the seventeen is provably a labelling mistake rather than a policy
choice: it shares a byte-identical transcript, duration and score with another
video in the same corpus that is annotated hateful, insulting and sexual.

On **MHClip-EN**, the weakness is in the positive class, not the negative class,
and it is a target-axis mismatch. Two thirds of the positive class carries no
hostility toward any protected group at all; it is profanity, sexual
suggestiveness, interpersonal cruelty and narrated harm. The judge separates
explicit protected-group hate from normal at **0.983** on this corpus. It
separates implicit protected-group hate at **0.725** and non-protected
offensiveness at **0.760**. The corpus figure of 0.785 is the weighted average of
those three populations, and it is low because the third population is the
largest one.

Neither corpus supports the reading that the judge cannot see hate. On both, the
judge ranks unambiguous protected-group hostility at the top of the score range.

## 1. Stratified decomposition

### MHClip-EN test_clean, 161 videos, 8B raw z

The primary collapse maps Hateful and Offensive to positive.

| Stratum | Positives | Negatives | ROC-AUC | 95% bootstrap CI |
|---|---:|---:|---:|---|
| Hateful vs Normal | 13 | 112 | 0.776 | 0.681–0.864 |
| Offensive vs Normal | 36 | 112 | 0.788 | 0.700–0.866 |
| Hateful vs Offensive | 13 | 36 | 0.417 | 0.252–0.592 |
| Union vs Normal (primary) | 49 | 112 | 0.785 | 0.715–0.854 |

The two positive classes are separated from Normal at the same level, within a
single point of each other, and the union figure is their exact rank-weighted
average. The Hateful class is if anything ranked slightly below the Offensive
class, which is what the third row says. So the weak overall number is not
caused by the Offensive class dragging the union down, and it is not caused by
the judge missing the Hateful class specifically. No class-pair stratum explains
it.

Score distribution by annotated class, raw z:

| Class | n | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Hateful | 13 | -8.75 | -2.50 | 0.00 | 6.00 | 13.50 |
| Offensive | 36 | -14.50 | -3.81 | 4.25 | 8.88 | 16.75 |
| Normal | 112 | -22.25 | -15.06 | -9.50 | -0.63 | 15.00 |

### HateClipSeg, 394 videos, 8B raw z

The shipped annotation carries six video-level categories. Counts over the 394
scored videos, with videos able to carry several categories at once:

| Category | Videos |
|---|---:|
| insulting | 251 |
| hateful | 180 |
| violence | 170 |
| sexual | 59 |
| normal | 50 |
| harm | 13 |

Each category against the fifty videos that carry no offensive category at all:

| Category | Positives | ROC-AUC | 95% CI |
|---|---:|---:|---|
| hateful | 180 | 0.843 | 0.778–0.903 |
| violence | 170 | 0.771 | 0.693–0.839 |
| insulting | 251 | 0.757 | 0.684–0.826 |
| harm | 13 | 0.722 | 0.555–0.868 |
| sexual | 59 | 0.712 | 0.618–0.804 |

Restricting to videos whose only category is the one named, which removes the
overlap between categories:

| Sole category | Videos | Median z | ROC-AUC vs clean normal |
|---|---:|---:|---:|
| hateful | 31 | 14.75 | 0.858 |
| insulting | 59 | 9.75 | 0.637 |
| violence | 24 | 7.63 | 0.606 |
| sexual | 5 | 6.25 | 0.644 |

The judge places hateful-only content almost a full standard deviation above the
other three exclusive categories. Both corpus-level figures decompose exactly:

- Offensive union, 0.754 = (180/344) × 0.843 + (164/344) × 0.655, where 0.655 is
  the AUC of the 164 offensive-but-not-hateful videos against clean normals.
- Hateful strict, 0.771 = (164/214) × 0.749 + (50/214) × 0.843, where 0.749 is
  the AUC of hateful videos against the offensive-but-not-hateful videos that the
  strict collapse pushes into the negative set.

Both weak numbers are therefore produced by categories other than hateful, in
one case on the positive side and in the other on the negative side.

## 2. False-negative taxonomy

### MHClip-EN, the 25 lowest-scoring positives

Codes were written from what the transcripts and frames actually show.

| Code | Meaning | n | Rate | Wilson 95% |
|---|---|---:|---:|---|
| T | target-free offensiveness: profanity, insult, interpersonal cruelty, no group | 7 | 28% | 14–48% |
| G | gender-generalising commentary, framed as relationship or dating talk | 4 | 16% | 6–35% |
| I | implicit protected-group hate: irony, euphemism, mocking framing, overlay | 4 | 16% | 6–35% |
| R | depiction or report of harm: narrated abuse, vigilante confrontation | 4 | 16% | 6–35% |
| S | sexual content or suggestiveness as the whole basis of the label | 3 | 12% | 4–30% |
| E | evidence-thin: no speech and nothing objectionable in the sampled frames | 3 | 12% | 4–30% |

Seventeen of the twenty-five worst false negatives, codes T, R, S and E, involve
no protected group in any form. Four are genuine model failures on protected-group
hostility that the video expresses indirectly.

### HateClipSeg, the 25 lowest-scoring union positives

| Code | Meaning | n | Rate | Wilson 95% |
|---|---|---:|---:|---|
| M | song whose objectionable content is entirely in sung lyrics | 7 | 28% | 14–48% |
| E | evidence-starved: degeneracy gate rejected the transcript, judge saw none | 6 | 24% | 11–43% |
| P | political speech, rally, protest or attack advertisement | 3 | 12% | 4–30% |
| I | carries the hateful label and the judge missed it | 3 | 12% | 4–30% |
| N | news or true-crime report about a crime | 2 | 8% | 2–25% |
| B | broadcast-form monologue, profane but not group-targeted | 2 | 8% | 2–25% |
| V | vigilante confrontation | 1 | 4% | 1–20% |
| S | sexual comedy sketch | 1 | 4% | 1–20% |

Twenty-two of the twenty-five carry no hateful label at all, and the median
hate-annotated duration fraction across the cohort is zero. Of the three that do
carry the hateful label, two received a transcript that the automatic speech
recogniser had filled with hallucinated repetition, so the hate cue was not in
the restored channel at all.

## 3. False-positive taxonomy

### MHClip-EN, the 25 highest-scoring negatives

| Code | Meaning | n | Rate | Wilson 95% |
|---|---|---:|---:|---|
| C | crude or sexual content, no target | 8 | 32% | 17–52% |
| A | aggression or insult without a group target | 7 | 28% | 14–48% |
| P | protected identity present but not attacked: satire, documentary, news | 3 | 12% | 4–30% |
| V | generic violence or crime | 3 | 12% | 4–30% |
| N | benign, nothing objectionable | 3 | 12% | 4–30% |
| D | appears to target a protected group despite the Normal label | 1 | 4% | 1–20% |

The false-positive side of MHClip-EN is small in absolute terms: only eleven
negatives sit above z = +7, and the twenty-fifth ranked negative is already at
z = +0.75 against a negative-class median of -9.5. The dominant codes, C and A,
are the exact mirror of the false-negative codes T and S. The judge treats
crudeness and aggression as weak evidence in both directions, and MHClip-EN's
annotation treats them as decisive in one direction only.

### HateClipSeg, the complete negative class

The offensive-union collapse leaves only fifty negatives, so all fifty were
coded rather than a top-scoring slice.

| Code | Meaning | n | Rate | Wilson 95% | Median z |
|---|---|---:|---:|---|---:|
| O | other: political commentary, conspiracy, news, music, benign | 29 | 58% | 44–71% | -5.50 |
| H | explicit protected-group hostility, annotated normal | 10 | 20% | 11–33% | 13.00 |
| X | extremist-movement glorification without explicit slurs, annotated normal | 7 | 14% | 7–26% | 12.75 |
| R | content about hate: news report, hearing, documentary | 4 | 8% | 3–19% | 9.50 |

The class is bimodal. The seventeen videos coded H or X have median z = 13.0 and
a minimum of +8.75, placing every one of them inside the score range of the
hateful-labelled positives, whose median is 14.75. The remaining thirty-three
negatives have median z = -1.75. The judge is not confusing the two groups; the
annotation is.

Two independent checks support reading H and X as annotation error rather than a
defensible policy line. First, the highest-scoring negative in the corpus shares
an exact transcript, an exact duration and an identical score with a second video
that is annotated hateful, insulting and sexual, so two copies of one video
received opposite labels. Second, four of the seven videos coded X are two
duplicate pairs plus a third near-duplicate, meaning the extremist material
entered the normal class more than once through the same route. Across the whole
corpus, eight exact-transcript duplicate groups exist and two of them disagree
about the binary collapse.

## 4. What the corrections are worth

| Corpus and collapse | As shipped | After the correction | Delta |
|---|---:|---:|---:|
| HateClipSeg, offensive union, H and X negatives removed | 0.754 | **0.900** | +0.147 |
| HateClipSeg, hateful strict, H and X negatives removed | 0.771 | 0.783 | +0.012 |
| HateClipSeg, hateful strict, H and X moved to the positive side | 0.771 | 0.780 | +0.010 |
| MHClip-EN, primary union, restricted to protected-target positives | 0.785 | 0.831 | +0.046 |
| MHClip-EN, restricted to explicit protected-target positives | 0.785 | **0.983** | +0.198 |

The asymmetry between the two HateClipSeg rows is arithmetic, not judgement. The
union collapse has only fifty negatives, so seventeen mislabelled ones are a
third of the denominator. The strict collapse has 214 negatives because it pushes
164 offensive-but-not-hateful videos onto the negative side, and the same
seventeen are then a small minority. HateClipSeg's strict number stays near 0.78
under every correction tried, and that residual is owned by the
hateful-versus-offensive-only boundary, not by the normal class.

MHClip-EN broken out by what the positive video actually does:

| Positive population | n | Median z | ROC-AUC vs Normal | 95% CI |
|---|---:|---:|---:|---|
| Explicit protected-group hostility | 7 | 13.50 | 0.983 | 0.957–1.000 |
| Implicit protected-group hostility | 10 | 0.25 | 0.725 | 0.621–0.824 |
| No protected group involved | 32 | 3.00 | 0.760 | 0.671–0.845 |
| All positives | 49 | 3.25 | 0.785 | 0.714–0.850 |

All seven explicit cases score above +8.5 against a Normal median of -9.5. Six of
the thirteen videos annotated Hateful in MHClip-EN express no protected-group
hostility of any kind, which is why the Hateful-versus-Normal stratum is no
better than the Offensive one.

## 5. Segment-level dilution on HateClipSeg

The segment-level annotation file has not been used by any previous analysis.
For each of the 180 hateful-labelled videos it gives the fraction of running time
that is annotated as hate.

| Statistic | Value |
|---|---:|
| Median hate-annotated duration fraction | 0.418 |
| Spearman correlation of raw z with hate fraction | +0.195 |
| Spearman correlation of raw z with total duration | +0.021 |
| Spearman correlation of raw z with hate-annotated seconds | +0.202 |
| Median z, lowest hate-fraction quartile | 13.00 |
| Median z, highest hate-fraction quartile | 15.00 |

Splitting the hateful videos at the median hate fraction:

| Half | n | ROC-AUC vs clean normal | vs offensive-only |
|---|---:|---:|---:|
| Sparse hate | 90 | 0.815 | 0.711 |
| Dense hate | 90 | 0.872 | 0.787 |

Dilution is real and it is small. Videos whose hate is confined to a short stretch
score about two z units lower and lose about six AUC points relative to videos
that are hateful throughout. Both halves nevertheless sit above the corpus-level
0.771, so sparse hate is not what makes HateClipSeg hard. Total video duration
has no relationship with the score at all, which rules out a simple
long-video-dilutes-everything account.

## 6. Covariate screen

For every taxonomy code with at least five videos, the label-free covariates were
compared against the cohort median. The question is whether any measurable
quantity, available without labels, identifies the misranked population.

**Language identification is unavailable.** The stored language field is null for
all 161 MHClip-EN and all 394 HateClipSeg transcripts, and the per-chunk language
dictionary is empty in every record. Language identification therefore cannot be
screened, and the several non-English videos found by reading were identified by
reading, not by a recorded covariate.

**MHClip-EN: nothing separates the misranked positives.** Median duration is 47
seconds for the protected-target positives against 36 seconds for the
non-protected ones and 32 seconds for the Normal class. Median voice-activity
fraction is 0.84, 0.75 and 0.75 respectively. Median transcript length delivered
to the judge is 535, 444 and 365 characters. These are ordering differences of a
few percent on populations whose score distributions differ by more than twenty
z units. No cheap covariate flags the population the judge gets wrong.

**HateClipSeg: the covariates separate the negative-class codes, but they track
the wrong thing.** The H negatives have median voice-activity 0.579 and a median
of 2282 transcript characters reaching the judge. The X negatives have median
voice-activity 0.000 and 105 characters, because they are music and archival
footage with no speech, and they still score at median z = 12.75. The O negatives
have median voice-activity 0.020 and 28% of them reached the judge with no
transcript at all. Across the fifty negatives, the Spearman correlation between
raw z and voice-activity fraction is +0.320; across all 394 videos it is +0.232,
and the correlation with transcript length delivered is +0.254. What this says is
that speech presence predicts a higher score. It does not isolate mislabelled
hate, because the X group scores just as high with no speech.

**The degeneracy gate is not the binding constraint but it is not free either.**
Under the union collapse, 4.1% of positives and 16.0% of negatives reached the
judge with no transcript. Restricting the corpus to the 372 videos that did get a
transcript moves the union AUC from 0.754 to 0.724 and the strict AUC from 0.771
to 0.765. Both move down, so the gate-rejected videos were, on balance, being
ranked correctly. Within the twenty-five worst union false negatives, however,
the evidence-starved code E covers six videos and five of those six received no
transcript at all, so the gate does concentrate in the worst cohort even though it
does not drive the corpus number.

## 7. What this licenses

1. **HateClipSeg's offensive-union number should not be used as a measure of the
   judge.** A third of its negative class is content the judge is right to score
   high. Any future comparison on this corpus should report the hateful-strict
   collapse, and should say that even that collapse leaves 164 offensive-only
   videos on the negative side.
2. **The open problem on HateClipSeg is the hateful-versus-offensive-only
   boundary, at 0.749.** That is the only number on this corpus that survives
   every correction attempted here. It is a question about what separates
   protected-group hostility from insult, sexual content and violence, and it is
   the same question the HateMM false-positive audit ended on.
3. **The open problem on MHClip-EN is implicit protected-group hostility, at
   0.725.** The judge is at 0.983 when the hostility is stated and at 0.725 when
   it is carried by irony, euphemism, political framing or a mocking overlay. Ten
   videos is a small cohort and the interval is wide, so this needs a larger
   cohort before it carries weight.
4. **The two corpora agree on one axis.** On both, the score tracks whether a
   protected group is being attacked. HateClipSeg's labels disagree with that
   axis in the negative class and MHClip-EN's labels disagree with it in the
   positive class, which is why the same judge looks weak on both for opposite
   reasons.

## 8. What this kills

1. **"The judge cannot see hate on the weak corpora" is dead.** On HateClipSeg
   the hateful category reaches 0.843 against clean normals and the exclusively
   hateful subset reaches 0.858. On MHClip-EN explicit protected-group hate
   reaches 0.983. The evidence is reaching the model and the model is using it.
2. **Sparse or short hate segments are not the explanation for HateClipSeg.**
   The segment file, used here for the first time, gives a dilution effect of
   about six AUC points between the sparse and dense halves, and both halves beat
   the corpus number. Total duration is uncorrelated with the score.
3. **No label-free covariate identifies the misranked population.** Transcript
   length, voice-activity fraction, video duration and segment density were all
   screened. On MHClip-EN none of them separates the misranked positives from the
   rest. On HateClipSeg the covariates that do separate the negative-class codes
   separate them by speech presence, which does not distinguish mislabelled hate
   from ordinary content. Any proposal to route or gate videos by these
   quantities has no support here.
4. **The remaining evidence-delivery hypotheses are closed on these two
   corpora.** Temporal coverage, on-screen-text legibility, speaker provenance,
   prosody and visual generic-harm fusion had already failed under frozen rules.
   This autopsy adds that the transcript-degeneracy gate, the one remaining
   delivery mechanism that was never tested, moves the corpus AUC in the wrong
   direction when removed. There is no delivery hypothesis left to test.
5. **A single scalar threshold on these corpora cannot be evaluated honestly
   until the labels are fixed.** The closeout note already showed that the
   winning threshold rule flips when the HateClipSeg collapse changes. This
   autopsy shows why: the collapse changes which mislabelled videos sit on which
   side.

## Reproduction

Decomposition, packet construction and aggregate analysis:

```bash
python3 scripts/duplex/ranking_autopsy_decompose.py
python3 scripts/duplex/ranking_autopsy_packet.py
python3 scripts/duplex/ranking_autopsy_montage.py en
python3 scripts/duplex/ranking_autopsy_analyze.py
```

Item-level codings, the coded packets, the frame montages and the aggregate
summary are under `results/ranking_autopsy/`, which is gitignored because those
files carry video identifiers and transcript text. `lexicons.json` was never
opened. No model call was made at any point in this autopsy.
