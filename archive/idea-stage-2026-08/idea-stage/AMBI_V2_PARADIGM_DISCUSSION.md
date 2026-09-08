# AMBI-V2: From Frame Scoring to Active Boundary Interrogation

## Decision after multi-agent discussion

The paper should not be told as another fusion model or as "T3AL plus an
MLLM."  Its central change of viewpoint is:

> A label-free localizer should first discover a defensible event core and
> then actively ask where membership in that multimodal event ceases, instead
> of independently assigning a hate score to every frame.

This is both closer to the mechanism that currently works (high-IoU endpoint
repair) and more distinctive than adding another dense score branch.

Decoding, sampling, ASR, feature extraction, timestamp alignment, and canvas
rendering are preprocessing and are not counted as modules.

## Module 1 -- Scale-Orthogonal Event-Core Discovery

The existing Scale-Orthogonal Evidence Transport field is retained.  It
separates within-video temporal evidence from video-level propensity and emits:

1. a dense multimodal evidence field;
2. an incumbent interval;
3. a conservative high-confidence event core.

The core is an anchor, not the final prediction.  This module answers **what
event should be localized?**, but deliberately delegates its exact extent.

## Module 2 -- Counterfactual Active Boundary Interrogation

Each boundary is treated as an active same-event membership search.  Starting
from a broad local bracket, the method repeatedly magnifies the most uncertain
transition between the event core and its temporal context.  At each round the
MLLM sees the entire ordered local canvas rather than isolated frames:

- core reference frames establish event identity;
- ordered boundary frames show visual continuity or change;
- timestamped transcript exposes whether the hostile assertion continues;
- acoustic residuals expose speaker/scene continuity even when words alone
  look similar;
- modality-specific dense residual bars expose the model-independent temporal
  evidence.

The MLLM answers a comparative question -- whether each cell belongs to the
**same event as the core** -- and never emits a generic hate score or a final
timestamp.  The next query bisects or zooms around the largest uncertain
membership transition.  This makes inference an evidence-acquisition policy,
not dense MLLM rescoring.

The counterfactual warrant is computed inside the same module.  A proposed
direction is valid only if it survives the appropriate evidential tests:

- language removal must alter a language-driven assertion decision;
- visual temporal shuffle must alter visual event-continuity evidence;
- audio temporal shift must alter acoustic continuity when audio is available;
- no single modality may reproduce the full authorization pattern.

These interventions are not an extra post-hoc explanation module.  They define
whether an endpoint authorization exists.  Missing evidence abstains; observed
contradiction vetoes.  This prevents transcript or the MLLM prior from silently
dominating the multimodal decision.

## Module 3 -- Separation-of-Powers Boundary Arbitration

Endpoint authority is deliberately split:

1. Module 2 authorizes only the **direction** of a boundary edit;
2. Module 1's dense field supplies the numerical coordinate within the active
   bracket;
3. the event core vetoes an edit that deletes the anchor evidence.

Left and right endpoints are adjudicated independently.  Any failed check
falls back only on that side, so an uncertain endpoint cannot damage the other
one.  The output is an interval plus an auditable chain of acquired evidence,
counterfactual warrants, and accepted/rejected actions.

## Why this is more than a fancy wrapper

The method changes the computational object:

- T3AL and ordinary zero-shot baselines estimate a temporal score and then
  threshold/group it;
- full-video temporal canvases ask an MLLM to rescore a fixed grid;
- AMBI-V2 starts from an event core, actively acquires evidence only at disputed
  boundaries, and treats multimodal same-event membership as the primitive.

The claim is therefore **active multimodal boundary interrogation**, not a new
fusion formula.  The three modules answer three non-overlapping questions:

1. Which event is the anchor?
2. Where does membership in that event cease under modality interventions?
3. Which proposed edit is safe to execute?

## Evidence already obtained

The one-shot V1 approximation raised development F1@0.7 from `0.217274` to
`0.254239` on 278 videos (`+0.036965`) while F1@0.3 and F1@0.5 each changed by
`-0.001497`.  This supports the endpoint-repair mechanism, but its bootstrap
confidence interval crosses zero and the gain is not positive on three of four
datasets.  It does not yet establish SOTA or balanced multimodal reasoning.

## Decisive experiment sequence

### A. Matched-action semantic kill test

Keep Module 1, Module 3, and exactly the same number of endpoint edits.  Compare:

- dense-transition ranking with no MLLM semantics;
- shuffled-transcript/placebo interrogation;
- factual AMBI interrogation.

If factual AMBI does not beat both controls, the semantic-interrogation story
is killed and the result is only conservative boundary gating.

### B. Active-search ablation

At equal MLLM token budget, compare the fixed eight-cell query against two-round
coarse-to-fine acquisition.  Active search must reduce endpoint error or improve
F1@0.7; otherwise adaptivity is not a contribution.

### C. Sealed multimodal falsification

On a genuinely untouched cohort, replay the frozen system with visual frame
shuffle, ASR timestamp shift, and audio temporal shift.  Each intervention must
selectively change authorization or correct-action rate, and no one modality
may reproduce at least 80% of the factual actions.

Primary confirmation gate:

- delta F1@0.7 at least `+0.03`, paired CI lower bound above zero;
- F1@0.3 and F1@0.5 no worse than `-0.005`;
- positive F1@0.7 direction on at least three of four datasets;
- factual AMBI beats the matched-action dense-only control.

## Honest novelty estimate

- Idea novelty: **6.5--7/10** if active acquisition and counterfactual warrants
  are part of inference and pass their ablations.
- Current implemented/evidenced novelty: **4.5/10**.
- If only the existing one-shot canvas survives: approximately **5/10**.
- A credible SOTA claim requires the sealed gate plus comparison under one
  protocol against T3AL and the strongest label-free localization baselines.

