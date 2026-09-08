# Pre-registration — Spec-displacement pilot (layer-27 displacement between two committed policy specs)

**Frozen:** 2026-08-09, before any new judge call. Every prompt string, every
constant, every threshold and every decision rule below was written into this
file and committed before the first forward pass of the union arm or the
off-construct arm.
**Compute:** one GPU pass over four corpus × arm cells (1,110 forward passes,
roughly eight GPU-minutes at the measured 0.34 s/video), then CPU analysis.
**Status:** mechanism pilot. It does not itself claim a deployment result; it
gates whether spec-conditioned readout geometry becomes a method family.

## Phenomenon

The ill-posedness result says the decision boundary is annotation-owned: the
same corpus and the same scores flip which label-free rule is optimal when the
label collapse changes (HateClipSeg valley 0.652 under the union collapse
against 0.448 under the strict collapse; anchored 0.547 against 0.677). Task
specification is therefore a missing input, not a nuisance. Separately, a
supervised probe on layer-27 hidden states separates target-free offence from
protected-target hate at leave-one-out AUC 0.837 where the scalar readout
scores 0.321 in the same direction. The distinction is present in the
representation and absent from every unsupervised summary of a single call's
states.

The two facts point at the same object from opposite sides. A video whose class
membership depends on which policy is in force is exactly a video the two
specifications disagree about. If the judge represents the policy it was given
rather than only the evidence it saw, then reading the same video under two
committed policies and subtracting should move that video differently from a
video both policies agree on.

## Mechanism

Judge each video twice with the frozen single-call judge, changing nothing but
the committed policy text: once under the strict protected-group rule list
(`YOUTUBE_RULES`, the frozen hate-speech spec of the base project) and once
under the union offensiveness rule list (the target-agnostic spec frozen in the
dual-axis pre-registration's appendix and never reworded since). Discard both
scalars. Read the layer-27 hidden state at the final prompt position in each
arm and take the displacement

    Δh_i = h_i^union − h_i^strict.

Decompose it into a corpus-level carrier and a per-video residual,

    m = mean_i Δh_i,   d = m / ‖m‖,   r_i = Δh_i − m.

The carrier is what the policy swap does to every video alike — the model
re-reading a different rule list. The residual is what the swap does to *this*
video, which is the only place a per-video spec disagreement can live.

The prediction is that the displacement is large for videos whose membership is
spec-dependent (union-positive, strict-negative) and small both for
protected-target hate, which is positive under both specs, and for benign
content, which is negative under both. A score built from Δh is then a
label-free flip predictor, and a flip predictor is precisely the missing input
the ill-posedness proof named: it says which videos the boundary choice is
about.

This is not the falsified dual-axis design. That design compared the two
scalars and asked whether the second question measured a second construct; it
did not, at +0.041 with a bootstrap interval spanning zero. Here both scalars
are discarded and the object of study is the difference of the internal states
under the two specs, which the dual-axis test never looked at. It is also not
the falsified unsupervised-readout design, which searched for structure inside
one call's states; the direction here is supplied by the contrast between two
committed specs rather than by the corpus's own variance.

## Falsification

The story dies if any of the following holds. The displacement carries no
per-video information, so pairing video *i*'s union state with video *j*'s
strict state predicts flips as well as the true pairing (control i). The
displacement is a generic prompt-change signature rather than a spec signature,
so swapping in an off-construct policy about spam and copyright — same
structure, no hostility vocabulary — yields the same predictor (control ii).
The score does not separate the flip stratum from the both-positive stratum in
either corpus (clause 1). Or the score adds nothing operational: an operating
point built from it does not attain the two per-collapse bests that no single
scalar rule attains (clause 4), or the point it builds transfers to the wrong
collapse as well as to its own (control iii).

## Frozen material

### Corpora and arms

| Corpus | n | Strict arm | Union arm | Off-construct arm |
|---|---:|---|---|---|
| MHClip-EN test_clean | 161 | on disk, `results/testruns/mhclip_en/judge_8b/hidden` | to score | to score |
| HateClipSeg test_clean | 394 | on disk, `results/hateclipseg/judge_8b/hidden` | to score | to score |

The strict-arm hidden states already exist and are not recomputed. The two new
arms reuse the frozen judge exactly: `src/duplex/extract_duplex_readout.py`
supplies the prompt skeleton, the `prag` judgment-mode block, the system
message, the 16-frame uniform grid, the `MAX_PIXELS = 100352` /
`MIN_PIXELS = 65536` budget, the Yes/No token-id sets, the fp16 37 × 4096
per-layer dump and the raw readout z. The transcript settings are the frozen
ones of each corpus's committed run: uncapped transcripts and the corpus's own
`c2_overrides.json`. The only substitution between arms is the policy sentence
and the rule list, plus the scope sentence that the union spec already carries.

The dual-axis run scored the union arm's z on MHClip-EN without storing hidden
states. Reproducing those 161 z values is a free integrity check: Spearman
between the new union z and `results/dual_axis/z_off_scores.jsonl` must be at
least 0.999. This is reported, not gating, because a difference would indicate
a pipeline change rather than a result.

### The three policy texts

The strict spec is `YOUTUBE_RULES` (nine rules, 527 characters) inside the
unmodified `DUPLEX_PROMPT`, with no scope sentence. The union spec is the
dual-axis appendix verbatim — seven rules, 423 characters, plus its scope
sentence — and is never reworded. The off-construct spec is written here for
the first time and is frozen on commit:

```
1. Duplicated re-uploads of one recording posted repeatedly
2. Reposting footage owned by another uploader without authorisation
3. Thumbnails or titles that misdescribe the recording
4. Redirects to off-platform storefronts, affiliate links, or downloads
5. Artificial inflation of view counts, likes, or subscriber numbers
6. Undisclosed paid promotion, sponsorship, or placement
7. Auto-generated or bulk-scraped material republished as original
```

Its policy sentence reads "Based on this platform's spam and copyright policy,
the following types of content are prohibited:" and its scope sentence, which
occupies the same slot as the union spec's, reads:

```
These rules are about distribution and ownership alone and apply no matter what the video is about. Content violates them even when the subject matter is entirely ordinary (such as music, cooking, travel, sport, news, gaming, or classroom teaching), and even when it was uploaded by one person, an organisation, or nobody at all.
```

The off-construct list is matched to the union list on rule count (7 against
7), on word count (59 against 59) and on characters (448 against 423, within
6 %); it costs 94 Qwen3-VL tokens against the union list's 105. Its scope
sentence is matched the same way (329 characters and 66 tokens against 320 and
64). It contains no word denoting hostility, insult, offence, obscenity or any
protected attribute. Everything else in the prompt is byte-identical across the
three arms.

### Readouts

Layer 27 of the 37 stored rows is primary and carries the verdict. Layers 18
and 36 are secondary and are reported at every clause. The full 37-row sweep is
descriptive only. Each corpus is analysed with its own carrier and its own
components; the two corpora are never pooled. All arithmetic is in float32 in
the raw hidden-state space, with no per-dimension standardisation.

Two readouts are pre-registered and no others will be computed:

- **c1, carrier-aligned displacement.** `c1_i = <Δh_i, d>`. Its sign is fixed by
  construction, since `d` is the corpus mean direction; the hypothesis
  direction is "large `c1` means flip", and every clause is evaluated
  one-sided in that direction.
- **c2, residual interaction axis.** The first principal component of the
  residual set `{r_i}`, computed by singular value decomposition of the
  mean-centred residual matrix, projected per video. Its reported orientation
  flips the axis so that its Spearman correlation with `z_strict` is
  non-negative.

Declared refinement, made before scoring. The instruction to orient c2 by
`z_strict` fixes the sign deterministically but does not fix which direction
the hypothesis predicts: flip videos sit in the *middle* of the `z_strict`
range, below protected-target hate and above benign content, so a `z_strict`
anchor is not informative about the flip direction. c2 therefore carries a
one-bit orientation freedom that c1 does not. It is paid for rather than
hidden: every c2 clause is evaluated two-sided as `max(AUC, 1 − AUC)` against a
floor raised by 0.02 over c1's, the winning direction is chosen once on
MHClip-EN and then applied unchanged to HateClipSeg and to every control and
clause, and the `z_strict`-oriented sign is reported alongside.

Verdicts are per readout and are never mixed. A readout SURVIVES only if it
passes clauses 1 to 4 on its own; the pilot SURVIVES if either pre-registered
readout does. Two candidates is the entire family and no third will be tried.

## Frozen clauses

| Clause | Rule | Floor (c1 / c2) |
|---|---|---|
| C1a | MHClip-EN, layer 27: the score separates the 34 blind-coded no-protected-target positives from the 15 blind-coded protected-target positives | AUC ≥ 0.72 / 0.74 |
| C1b | HateClipSeg, layer 27: the score separates the 164 union-positive strict-negative videos from the 180 strict-positive videos | AUC ≥ 0.70 / 0.72 |
| C2 | The shuffled-pairing placebo's AUC is at least 0.10 below the real AUC on both corpora | real − shuffled ≥ 0.10 |
| C3 | The off-construct arm's flip-prediction AUC at layer 27 is at most 0.60 on both corpora | ≤ 0.60 |
| C4 | The frozen composition rule reaches macro-F1 ≥ 0.6522 under the HateClipSeg union collapse and ≥ 0.6767 under the strict collapse | both |

C1's references: the scalar readout scores 0.321 on the MHClip-EN pair in the
hypothesis direction, which is 0.679 read favourably, and the supervised probe
reaches 0.837 leave-one-out. The floor of 0.72 sits above the favourable
reading of the scalar and well below the supervised ceiling. C4's references
are the two per-collapse bests in
`results/hateclipseg/gated_anchor_results.json`: the KDE valley reaches 0.6522
under the union collapse and 0.4479 under the strict one, while the anchored
rule reaches 0.6767 under strict and 0.5474 under union. No single scalar rule
attains both.

### Controls

**(i) Shuffled-video pairing.** A single fixed derangement π of each corpus's
video order, drawn once from `numpy.random.default_rng(20260808)` by rejection
until no fixed point remains, defines `Δh_i^shuf = h_{π(i)}^union − h_i^strict`.
Both readouts are recomputed end to end on the shuffled displacements. The
carrier is invariant under any permutation, so this placebo destroys the
per-video pairing and leaves everything else intact. The shuffled c2 is scored
two-sided, which makes the control harder to beat.

**(ii) Off-construct policy arm.** `Δh_i^spam = h_i^spam − h_i^strict`, both
readouts recomputed end to end, evaluated on the same two strata. For the
readout under verdict its AUC must sit at or below 0.60 on both corpora; the
full two-readout, two-corpus table is reported either way.

**(iii) Crossover.** The composition rule of C4 produces two decisions, one per
collapse. Each must be better on its own collapse than the other one is:
macro-F1(strict decision | strict collapse) > macro-F1(union decision | strict
collapse), and macro-F1(union decision | union collapse) > macro-F1(strict
decision | union collapse). Both inequalities are required. An operating point
that scores as well on the collapse it was not built for is not spec-specific.

### The composition rule of C4

Frozen here in full, with its thresholds, before any score exists.

Let `t_z` be the label-free KDE-valley threshold of the frozen recipe
(`crossbench_analyze.kde_valley`: Gaussian KDE, Scott bandwidth, 4001-point
grid over [min − 2, max + 2], minimum density strictly between the two highest
local maxima) applied to HateClipSeg's strict-arm z. Let `t_c` be the same
recipe applied to that corpus's own c distribution; if the recipe reports fewer
than two modes, `t_c` falls back to the free two-component Gaussian mixture of
`anchored_operating_point._fit_gmm` at seed 20260808 with 50 restarts and a
sigma floor of 0.1, taking the posterior-0.5 boundary that lies between the two
component means. Valley first, mixture only on failure; no other recipe is
tried. Define

    B = { i : z_strict,i ≥ t_z }        the offensive base set
    F = { i : c_i ≥ t_c }               the spec-dependent flip set

    union-collapse decision:   i ∈ B
    strict-collapse decision:  i ∈ B and i ∉ F

Declared refinement, made before scoring. The instruction described the strict
decision as the strict-spec z valley itself. That valley reaches 0.4479 under
the strict collapse, so a rule of that exact form could not meet the 0.6767
floor whatever the pilot found, which would make C4 unfalsifiable in the wrong
direction. The rule above is the same composition read the other way round —
since `B = (B \ F) ∪ (B ∩ F)`, the union decision is exactly "strict decision
OR c above its cut", as instructed — with the strict decision being the one
that the flip set has been removed from. The union half is then inherited from
the valley and met with equality at 0.6522; the strict half is what c has to
earn. Both halves are still required.

## What is reported regardless of verdict

The energy split of the displacement, `‖m‖²` against `mean_i ‖r_i‖²`, per
corpus and per layer, which says how much of the policy swap is corpus-level
carrier and how much is per-video interaction. Spearman correlations of both
readouts with `z_strict` and with `z_union` on both corpora. The cosine between
the residual PC1 and the supervised layer-27 probe direction of the
readout-bottleneck test, refitted on the same 49 MHClip-EN videos and mapped
back into raw hidden-state space; the probe stays a measurement instrument and
never becomes a method component. The full 37-layer sweep of C1a and C1b for
both readouts. The reproduction check against the dual-axis z.

## Interpretation boundaries

All four clauses hold for one readout: spec-conditioned readout geometry
becomes a method family and earns a follow-on pre-registration on the remaining
corpora. This pilot is still not a deployment claim, because C4's thresholds
are read on the corpus they are fitted to.

C1 fails: the displacement carries no flip information at the layer where the
distinction is known to be linearly present, and the family dies. Combined with
the dual-axis and readout-bottleneck results, that would close the last route
by which a second call could recover the construct distinction from this
judge's states.

C1 holds but C2 fails: the signal is prompt identity rather than a per-video
spec disagreement, and the score is measuring which prompt was used, not which
video was read.

C1 and C2 hold but C3 fails: the displacement is a generic prompt-change
signature. The score would then be a measure of how much any policy rewrite
moves a video, which is a property of the video's length or ambiguity rather
than of the hate-versus-offence boundary.

C1 to C3 hold but C4 fails: the flip set is identifiable but not usable at an
operating point, which is the same wall the extreme-pseudo-label route hit at
+0.04. Recorded and stopped rather than rescued.

No video id, transcript text or title reaches any committed artifact.
