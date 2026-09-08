# PACT: Proposal-Lattice Authority through Cross-modal Temporal Warranting

**Direction:** A publishable, label-free extension of VASTA/CCA that preserves
their working visual-support and selective-negative-evidence mechanisms while
making multimodal interaction affect interval selection itself.  Until the
mechanism gates pass, `warrant` is the defensible term; `proof` or
`certificate` would overstate what native-evidence consistency establishes.

**Status:** Revised after two-agent adversarial discussion.  This is a
falsifiable candidate, not yet a result.  It should replace CCA as the main
research hypothesis only if the cache-only pilot below passes.

## One-sentence thesis

A hateful event should be localized as a minimally supported
**evidence-complete interval** on a frozen visual support lattice: vision defines which temporal
claims are legal, timestamp-aligned speech decides which of those claims are
semantically complete, and a prediction is changed only when the same discrete
claim survives label-free perturbations.

This changes the scientific object from a fused frame score to a
**evidence-carrying interval**.  The warrant is operational: it records the visual
proposals that make the endpoints legal, the native speech evidence that makes
the interval preferable to adjacent time, and the perturbations under which
the selected endpoint pair remains invariant.

## What we actually build

1. Keep the eight frozen A10/Vid-Group proposals as a discrete interval lattice
   instead of immediately collapsing them into one envelope.
2. Score the real timestamped transcript inside every legal interval and its
   adjacent native flanks with one frozen language model and an explicit-empty
   reference.  No video label, learned router, or dataset identity is used.
3. Replace the working top-8 closure only when there is a unique,
   support-minimal interval supported by both modalities.  Visual endpoint
   cells must first be invariant to leave-one-proposal-out perturbations, and
   the semantic decision must then be invariant to timestamp perturbations.
   Otherwise
   output the current closure exactly.

Frame decoding, frame sampling, ASR, timestamp alignment, frozen feature
extraction, and curve rasterization are preprocessing and are not modules.

## The three method modules

### Module 1: Robust Visual Claim Lattice

Let the frozen top-eight proposals be

\[
P_k=[l_k,r_k],\qquad k=1,\ldots,8,
\]

where `P1` is the rank-1 anchor.  Construct

\[
\mathcal L_v=
\left\{
\operatorname{hull}\!\left(\bigcup_{k\in A}P_k\right):
1\in A\subseteq\{1,\ldots,8\}
\right\}.
\]

After deduplication, this contains at most 128 candidates per video.  Every
candidate contains `P1`, and every endpoint comes from a frozen visual
proposal.  Thus transcript evidence can select a boundary but cannot invent
one.  The current working interval

\[
I_0=\operatorname{hull}\left(\bigcup_{k=1}^8P_k\right)
\]

is explicitly included as the safe fallback.

The initial lattice is filtered *before language is consulted*.  For every
non-anchor proposal `Pj`, rebuild the lattice without `Pj` and map endpoints to
the frozen 4-FPS cells.  The visually admissible set is

\[
\mathcal A_v=\bigcap_{j=2}^{8}\mathcal L_v^{-j},
\]

where intersection means the same endpoint-cell pair is present after every
leave-one-proposal-out rebuild.  Equivalently, both boundaries must have
redundant proposal support rather than be furnished by one fragile proposal.
This turns the visual obligation into a nontrivial candidate filter; language
never sees visually unstable claims.

This is a method module, not proposal preprocessing: retaining the partial
order between proposal subsets and rejecting fragile endpoint claims is what
makes warranted multimodal boundary selection possible.

### Module 2: Native Cross-modal Evidence Obligations

For each candidate `I`, the visual temporal obligation is discrete:

\[
T(I)=\mathbf 1[I\in\mathcal A_v].
\]

There is no visual confidence threshold or learned calibration.  Membership
establishes that the interval retains the rank-1 support component, uses only
frozen visual endpoints, and has leave-one-proposal-out stable endpoints.

Let `z(I)` be the frozen language model's Yes-minus-No log odds for a complete
hateful relation in the timestamped speech overlapping `I`.  Let `z_empty` be
the score for the exact same prompt with an explicit empty transcript.  Let
`F_L(I)` and `F_R(I)` be equal-duration, adjacent, native transcript flanks.
The semantic obligation is

\[
S(I)=
\mathbf 1[z(I)>z_{empty}]
\land
\mathbf 1[z(I)>z(F_L(I))]
\land
\mathbf 1[z(I)>z(F_R(I))].
\]

An unavailable equal-duration flank is an abstention, not a pass.  The final
implementation may compare interval likelihoods in a single batched call, but
the pilot uses the already cached timestamped generic chunk logits.  Scores
are compared only within the language modality; they are never added to visual
logits.

This obligation deliberately uses one complete-relation judgment rather than
asking an MLLM to separately generate source, target, hostility, and stance.
That avoids a cascade in which the weakest hallucinated slot controls the
whole interval.

### Module 3: Non-compensatory Stable Warrant Decoder

Compute the semantic obligation under the original timestamps and deterministic
one-chunk shifts in both directions.  Define the feasible set

\[
\mathcal C=\{I\in\mathcal A_v:T(I)\land S_{-1}(I)\land S_0(I)\land S_{+1}(I)\}.
\]

Every interval retains its generating proposal subset `A`.  A candidate is
**support-minimal** if no feasible candidate generated by a strict proposal
subset `B ⊂ A` remains.  This is set inclusion over proposal support, not
shortest duration and not a length penalty.  PACT changes the boundary only
when:

1. `C` has a unique support-minimal endpoint pair `I*`; and
2. that pair already passed visual leave-one-out stability in Module 1 and
   semantic timestamp-jitter stability in the construction of `C`.

Call the conjunction of these two stability obligations `R(I*)`.  The decoder
is therefore

\[
\widehat I=
\begin{cases}
I^*, & T(I^*)\land S(I^*)\land R(I^*),\\
I_0, & \text{otherwise}.
\end{cases}
\]

There is no weighted sum, epsilon, shortest-length penalty, dataset-level
switch, or label-derived threshold.  If multiple minimal intervals remain,
the warrant is ambiguous and the system falls back rather than tie-breaking with
another heuristic.

Existence uses the same claim-specific authority principle.  Visual consensus
remains authoritative.  Under visual disagreement, `EMPTY` is allowed only
when one frozen visual expert is negative **and** both the full transcript and
the `I0` transcript fall below their explicit-empty references.  It is thus a
  joint negative warrant, not a text-only deletion.  Text cannot create an
event from two visual-negative experts.

## Why this is a real upgrade over CCA

CCA estimates text-veto agreement on the visual-consensus subset and transfers
that dataset-level estimate to visual-disagreement videos.  Its largest flaw is
the unverified cross-jurisdiction assumption; agreement with visual pseudo
labels is not correctness.

PACT eliminates this transfer.  Jurisdiction is discovered per video and per
claim:

- vision has jurisdiction over legal temporal support and endpoint creation;
- timestamped speech has jurisdiction over semantic completeness and native
  temporal contrast;
- neither may compensate for a failed obligation from the other modality;
- modality-specific stability, not cohort identity, grants permission to
  change the prediction.

CCA remains a useful historical ablation, not a module in PACT.

## Why this is not MELT or HYPER

MELT generates an event tuple, a multimodal phase field, and new lifecycle
boundaries.  Its counterfactual citation certificate was an audit performed
after generative hypothesis induction, and the repository observed zero
certified refinements.  PACT instead:

- induces no source/target/stance tuple and no ENTER/SUPPORT/EXIT sequence;
- generates neither boundaries nor free-form rationales;
- uses only real native intervals and flanks, with no masked-video OOD edit;
- makes native evidence obligations the interval decoder itself;
- returns the known working `I0` exactly when warrant coverage is zero.

HYPER asks an MLLM to conduct pairwise proposal tournaments and collapsed to
rank 1 in the existing pilot.  PACT has no generative tournament or arbitrary
pairwise winner: it performs deterministic inclusion-minimal search from
cached likelihood obligations.

## Why both modalities genuinely affect the boundary

Vision alone determines the legal endpoint lattice but cannot choose among its
members.  Speech alone determines semantic feasibility but cannot create an
endpoint.  A boundary change exists only at their intersection.  The following
controls make this empirical rather than rhetorical:

- shuffling transcript timestamps must alter the selected interval and remove
  the gain;
- a transcript-only decoder must be unable to emit endpoints outside the
  visual lattice;
- a visual-only minimal-support rule and the current `I0` closure must both be
  weaker than the full decoder;
- report the fraction of warranted edits whose left endpoint, right endpoint,
  and existence decision actually change.

## Minimum-cost pilot

### Pilot 0: cache-only mechanism gate

- **Cohort:** first use the existing label-blind 32/51-video mechanism cohort;
  if it passes, run the deterministic rule on all common cached videos across
  HateMM, HateClipSeg, MHC, and MHC-zh.
- **Inputs:** existing NMS=.5 top-eight A10 proposal banks, existing timestamped
  generic per-chunk log odds, existing A08/A12 visual states, and the frozen
  explicit-empty anchor.
- **Compute:** CPU enumeration of at most 128 candidates per video; expected
  minutes and no new GPU inference.
- **Comparisons:** `I0`, visual-only inclusion-minimal interval, text-only
  transcript span, timestamp-shuffled PACT, PACT without stability, VASTA, and
  PACT full.
- **Report:** F1@.3/.5/.7, edit coverage, correct/wrong edit counts, endpoint
  change rate, fallback rate, per-dataset direction, and aligned-minus-shuffle
  paired bootstrap intervals.

### Promotion gates

The pilot promotes only if all hold:

1. warranted boundary-edit coverage is between 5% and 30%;
2. full PACT improves both macro F1@.5 and F1@.7 over `I0`/VASTA, while F1@.3
   decreases by no more than .002;
3. aligned PACT beats timestamp-shuffled PACT by at least .003 F1@.5;
4. full PACT beats both visual-only and text-only controls by at least .003
  F1@.5;
5. correct boundary edits exceed wrong boundary edits, with at least ten
   warranted edits; and
6. neither modality can account for more than 80% of changed decisions when
   the other modality's obligation is held fixed.

If cache-only scalar semantics passes, a second pilot may replace `z(I)` with
one frozen Qwen3-VL interval-level complete-relation likelihood.  Prompt wording
is frozen before labels are opened; no slot generation is added.

### Hard kill criteria

Kill PACT as the main method if any of these occurs:

- zero/near-zero warrant coverage, reproducing MELT's failure;
- the decoder selects a constant interval or only changes EMPTY predictions;
- aligned timestamps do not beat the timestamp shuffle;
- boundary improvements disappear against either unimodal control;
- F1@.5 gain comes with an F1@.7 loss; or
- performance requires dataset identity, a learned router, target labels, or
  tuning the logical obligations.

## Four-dataset implementation map

The same code path applies to all four datasets.  A10 provides the visual
lattice; existing timestamped ASR provides speech evidence; A08 and A12 provide
existence states.  Chinese transcripts are passed unchanged to the same frozen
multilingual scorer.  Missing speech causes abstention and exact visual
fallback rather than a dataset-specific policy.  No dataset name is visible to
the decoder.

## Expected contribution and honest prior

- **Contribution type:** new task formulation plus a training-free method.
- **Task-specific novelty prior:** 5.8/10 before the pilot; 6.4--6.7 only if the
  aligned, multimodal boundary mechanism passes; below 6/10 if only existence
  filtering improves.
- **General-method novelty prior:** 4.3/10 until a literature novelty audit is
  completed.
- **Workability prior:** 25--35%.  Exact fallback makes non-degradation
  plausible, but obtaining enough stable boundary edits without systematic
  over-contraction is the main risk.
- **Estimated implementation:** one day for the cache-only decoder and audit;
  one additional GPU-hour or less only if the interval-level frozen scorer is
  promoted.

## Paper story

> Existing label-free video localizers ask which frames have the largest
> multimodal score.  We ask what evidence makes an interval legally assertable.
> Vision proposes the only admissible temporal claims; timestamp-aligned speech
> proves semantic completeness against native neighboring time; and the system
> changes a prediction only when this cross-modal warrant is stable.  The result
> is not another fusion score but a prediction that carries its own multimodal
> evidence.
