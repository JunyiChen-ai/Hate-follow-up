# NCL: Null-Competitive Localization over Temporal and Modality-Coalition Lattices

## Status and verdict

NCL is the strongest current **idea-level** response to the observed evidence:
top-8 closure is a strong extent readout, repeated boundary regeneration and
contraction are fragile, and the remaining bottleneck is deciding whether a
localized event exists at all.

The method is not yet empirically validated.  Its task-specific novelty is
about **5.6/10 before coverage and performance evidence**, and can reach
**6.3--6.7/10** only if the falsification pilot below passes.

## Paradigm

Existing methods treat `EMPTY` as what happens when a video-level score fails a
threshold.  NCL makes `EMPTY` a first-class temporal hypothesis:

> A nonempty event is an existential claim: one localized proposal jointly
> supported by vision and speech is sufficient.  Empty is a universal claim:
> every localized proposal must fail to improve on the same model's no-media
> null under every modality coalition.

This is **null-competitive localization**, not post-hoc classification.  The
method reasons on the product of two discrete structures:

- a temporal hypothesis set `{P1,...,P8, EMPTY}`;
- a modality-coalition lattice `{no-media, visual, transcript, joint}`.

The asymmetry between `exists` and `for all` is deliberate and is the core
method claim.

Frame decoding, ASR, timestamp alignment, frame sampling, frozen feature
extraction, proposal extraction, and curve rasterization are preprocessing and
are not modules.

## Module 1: Existential Temporal Witness Bank

The frozen A10 top-eight proposals

\[
\mathcal P=\{P_1,\ldots,P_8\}
\]

are retained as candidate-specific **existence witnesses** rather than
immediately being collapsed into a single video-level feature.  The established
connected top-eight closure

\[
I_{\mathrm{closure}}=
\operatorname{CC}_{P_1}\left(\bigcup_{k=1}^8P_k\right)
\]

remains the nonempty extent readout.

This factorizes existence from extent:

- a proposal `Pk` may witness that an event exists;
- winning existence returns the high-recall closure;
- NCL does **not** claim that every frame or every discarded proposal inside
  the closure has been semantically verified.

The supported proposal, its duration, and the closure-over-witness expansion
ratio must be reported for every nonempty certificate.  This prevents the
proposal ledger and the final boundary from being rhetorically conflated.

## Module 2: Native Modality-Coalition Evidence Lattice

For each temporal hypothesis `Pk`, query one frozen MLLM with the identical
hate-existence question under four native channel configurations:

\[
\mathcal C=\{0,V,T,VT\}.
\]

- `0`: no image and no transcript;
- `V`: native frames from `Pk`, no transcript;
- `T`: native timestamped transcript from `Pk`, no image;
- `VT`: both native frames and the aligned transcript from `Pk`.

No black frames, masks, deleted regions, synthetic silence, video labels, or
dataset identity are supplied.  `0` is the actual existence null.  A temporal
permutation is retained only as an alignment control; it is **not** an empty
hypothesis because it preserves hateful content.

Using the same prompt, batch size 1, left padding, and Yes-minus-No next-token
log odds, define

\[
f_C(k)=\log p_\theta(\mathrm{Yes}\mid E_C(P_k))
-\log p_\theta(\mathrm{No}\mid E_C(P_k)).
\]

The four native views permit the diagnostic coalition contrasts

\[
m_V=f_{VT}-f_T,\qquad
m_T=f_{VT}-f_V,\qquad
\delta=f_{VT}-f_V-f_T+f_0.
\]

These are **channel-ablation contrasts**, not causal main effects.  NCL does not
claim intervention on the underlying event.

For two fixed native frame phases, compute the unique coalition winner

\[
w_k=\arg\max_{C\in\mathcal C} f_C(k).
\]

`w_k` is valid only if the same unique winner occurs under both phases;
otherwise the proposal abstains.  Its semantics are deliberately narrow:

- `w_k=VT`: irredundant joint support; the joint view beats the null and both
  unimodal subcoalitions, so both channels are load-bearing;
- `w_k=0`: null-dominated / eliminated proposal; this is lack of support under
  the tested coalitions, **not affirmative benign counterevidence**;
- `w_k=V` or `T`: real unimodal evidence; do not delete it and do not claim
  multimodal support;
- unstable/tied: abstain.

This allows genuine unimodal hate without allowing one modality to cause an
NCL amendment: a unimodal winner blocks the universal empty claim and defers to
the frozen base localizer.

## Module 3: Quantifier-Asymmetric Null Competition

Define the nonempty and empty warrants

\[
C_+=\mathbf1\left[\exists k\in\{1,\ldots,8\}:w_k=VT\right],
\]

\[
C_0=\mathbf1\left[\forall k\in\{1,\ldots,8\}:w_k=0\right].
\]

They are logically mutually exclusive.  If an implementation reports both,
that is an error, not an ambiguity case.

The decoder is

\[
\widehat I=
\begin{cases}
I_{\mathrm{closure}}, & C_+=1,\\
\varnothing, & C_0=1,\\
I_{\mathrm{base}}, & C_+=C_0=0.
\end{cases}
\]

`I_base` is the frozen predeclared CCA/visual-state decision.  The fallback is
an explicit defer state; NCL is promoted only if it does not dominate more than
90% of cases.  Every NCL-induced rescue requires a `VT` winner, and every
NCL-induced deletion requires the no-media null to beat `V`, `T`, and `VT` for
every temporal hypothesis.  Neither visual nor transcript evidence can cause a
method-induced state change alone.

## Why the four arms are functional rather than decorative

A simple joint gate tests only `fVT > f0`.  It cannot determine whether that
decision is caused entirely by a spurious visual or transcript channel.  NCL
uses every arm for a distinct proof obligation:

- `fVT > fT` makes visual evidence necessary for joint support;
- `fVT > fV` makes transcript evidence necessary;
- `fVT > f0` beats the no-media existence null;
- `f0 > fV,fT,fVT` eliminates a proposal without treating silence in one
  channel as sufficient evidence of emptiness.

The contrasts and the full Boolean-lattice winner are therefore the mechanism,
not an explanatory ablation added after a joint classifier.

## Why this is not just a video-level gate

NCL does not score the full video once.  The null must compete separately with
all eight localized hypotheses.  One `VT`-supported local witness defeats the
universal empty hypothesis, while `EMPTY` wins only after every temporal
witness is eliminated.  The following controls are mandatory:

- a single full-video `fVT > f0` classifier;
- `max_k fVT(k) > f0` without subcoalitions;
- `K=1` and `K=4/6/8` proposal banks;
- shuffled proposal times and frozen proposal-order perturbations;
- the same transcript content circularly reassigned across proposal windows.

If NCL behaves identically to a full-video or max-score classifier, the
localization paradigm claim fails.

## Minimum falsification pilot

### Stage 0: label-blind coverage sanity

- **Cohort:** deterministic hash-first 32 videos, eight per dataset.  Selection
  may not use video labels or ground truth.
- **Inference:** eight proposals, four native channel configurations, two fixed
  visual phases.  Batch examples for efficiency but keep per-example batch-1
  logits reproducible.
- **Before GT:** freeze and hash all `f_C(k)`, winners, warrants, and outputs.

Hard coverage gates:

1. `C+` occurs on at least two videos;
2. `C0` occurs on at least two videos;
3. defer/fallback is below 90%;
4. `C+` and `C0` are never simultaneously true;
5. at least one `V` or `T` unimodal winner is preserved as defer rather than
   incorrectly deleted;
6. every output is invariant to batching and proposal presentation order.

Failure of any gate kills the full experiment.

### Stage 1: frozen full-cohort audit

Run the unchanged rule on the common four-dataset cohort, freeze predictions,
then open GT.  Report:

- correct false-positive removals;
- wrongful deletion of positive events;
- correct rescues of baseline-empty events;
- wrongful nonempty rescues;
- positive-only boundary quality, which must remain bit-identical because the
  closure is unchanged;
- C+/C0/defer rates by dataset and visual-consensus state;
- supported-proposal versus closure expansion.

Required comparisons:

- CCA, VASTA, visual OR, AND, majority, and the best current existence gate;
- full-video classifier and simple joint `fVT > f0`;
- visual-only, transcript-only, and every three-arm coalition ablation;
- matched-count random deletion and matched-count random rescue;
- transcript length, transcript empty/nonempty, and ASR-coverage heuristics;
- temporally permuted transcript and shuffled proposal intervals;
- `K=1/4/6/8` and proposal-order controls.

Promotion requires:

- at least ten NCL-induced state changes, with neither rescue nor deletion
  accounting for more than 80%;
- correct changes exceeding wrong changes in both directions;
- macro interval F1@.5 improvement over CCA and visual OR, with no more than
  .002 loss at F1@.3 or F1@.7;
- full NCL beating simple joint, all unimodal/three-arm controls, and
  matched-count random changes;
- native alignment beating temporal permutation;
- positive direction on at least three datasets;
- confirmatory evaluation on a prospective temporally annotated cohort.

## Main risks and honest claim boundary

The largest technical risk is that log-odds across no-media, image-only,
text-only, and joint chat templates may reflect input-format calibration rather
than evidence contribution.  Batch-1/left-padding reproducibility, two frame
phases, prompt-paraphrase stability, and irrelevant-native-evidence controls
are therefore required.

The largest research risk is excessive defer: if C+/C0 cover fewer than 10% of
videos, the coalition lattice is a decorative wrapper around CCA.  It must then
be killed rather than sold as a paradigm.

Safe claims if the pilot succeeds:

- `EMPTY` is treated as a first-class temporal hypothesis;
- nonempty and empty decisions follow existential and universal evidence
  obligations over localized proposals;
- method-induced changes require both modality channels to be load-bearing;
- extent is deliberately inherited from the validated closure.

Unsafe claims:

- `w_k=0` is affirmative proof that content is benign;
- coalition contrasts are causal effects;
- every frame inside the returned closure is jointly verified;
- SOTA or novelty above 6 before prospective validation.

## Paper story

> Existing label-free localizers threshold a relevance score and call the
> failure case empty.  We instead let EMPTY compete with localized event
> hypotheses over a native modality-coalition lattice.  One irreducibly joint
> local witness is sufficient to establish existence; emptiness must eliminate
> every temporal witness under every coalition.  We call this
> null-competitive localization.

