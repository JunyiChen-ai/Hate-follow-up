# PACT-v3: Persistent Proposal Filtration with Native Semantic Shell Warrants

**Status:** Third-round revision after two measured failures invalidated PACT-v2.
This is the only PACT variant that should be piloted further.

## Why v2 is killed

Two label-free coverage checks are decisive:

1. Exact leave-one-proposal-out lattice intersection leaves only `P1` on
   603/611 videos; only eight retain more than one candidate.  It is a 98.7%
   rank-1 collapse, so it cannot serve as visual stability.
2. None of the 611 top-k hull candidates has two valid equal-length adjacent
   flanks.  The old semantic obligation therefore has zero coverage.

Neither failure may be repaired with a relaxed threshold or a one-sided flank
chosen after seeing results.

## Thesis

Frozen proposal rank defines a natural temporal filtration.  A boundary scale
is editable only if it persists across proposal arrivals and the retained core
has greater null-referenced speech-evidence density than **every** visual shell
that would be discarded.  Text supplies a binary semantic warrant; vision
retains authority over which warranted scale is selected.

Frame decoding, ASR, timestamp alignment, proposal extraction, and curve
rasterization are preprocessing, not modules.

## Module 1: Rank-Induced Visual Persistence Filtration

Let `P1,...,P8` be the frozen proposals in their native model rank order.  At
rank `k`, form the connected component containing the center of `P1`:

\[
I_k=\operatorname{CC}_{P_1}\left(\bigcup_{i=1}^{k}P_i\right),
\qquad k=1,\ldots,8.
\]

The path is nested, `I1 ⊆ ... ⊆ I8`.  Map endpoints to the frozen 4-FPS
cells and collapse consecutive duplicate states.  Write the resulting states
as

\[
H_1\subsetneq H_2\subsetneq\cdots\subsetneq H_M=I_8.
\]

For a state `H`, define its visual persistence

\[
\pi_v(H)=\#\{k:I_k=H\},
\]

the number of consecutive frozen proposal-rank arrivals for which its endpoint
cells remain unchanged.  This is an ordinal persistence statistic, not a
tuned temporal tolerance.

The 611-video cache gives state-count distribution

| states/video | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|
| videos | 1 | 89 | 260 | 232 | 29 |

Thus 610/611 videos retain multiple visual scales.  A non-fallback state is
visually admissible only if

\[
\pi_v(H_j)>\pi_v(H_M).
\]

This strict comparison means an edit must be more rank-persistent than the
working top-8 closure.  It contains no numerical threshold.  In the current
cache, 251/611 videos have at least one such state, providing a 41.1% visual
upper bound before semantic filtering.

## Module 2: Native Semantic Shell Warranting

Every visual expansion creates a real, native temporal annulus

\[
A_j=H_{j+1}\setminus H_j,\qquad j=1,ldots,M-1.
\]

Let `z_c` be the frozen transcript scorer's log odds for timestamped ASR chunk
`c`, and let `z_empty` be the same prompt's explicit-empty score.  Construct a
null-surplus field

\[
q(t)=
\begin{cases}
\operatorname{mean}_{c\ni t}(z_c-z_{empty}), & t\text{ is ASR-covered},\\
0, & \text{otherwise}.
\end{cases}
\]

ASR gaps are neutral: they contribute zero evidence, not fabricated negative
evidence.  For any native region `U`, define its duration-normalized density

\[
\mu_t(U)=\frac{1}{|U|}\int_Uq(t)\,dt.
\]

For timestamps shifted deterministically by `delta` in `{-1,0,+1}` ASR chunks,
the binary text warrant is

\[
W_t(H_j)=
\bigwedge_{\delta\in\{-1,0,+1\}}
\left[
\mu_t^\delta(H_j)>0
\;\land\;
\mu_t^\delta(H_j)>
\max_{r=j}^{M-1}\mu_t^\delta(A_r)
\right].
\]

This replaces the impossible two-flank test.  It asks whether the retained
core is above the model's explicit null and denser than **each** discarded
visual growth shell.  The maximum prevents a strong discarded shell from being
hidden by averaging it with a large weak complement.  All regions are native
video time; there is no masking, black-frame edit, or OOD counterfactual.

The existing whole-discarded-shell proxy produced 62 stable warrants over all
611 records (10.1%), or 22.3% among the 278 videos with matching ASR-cache keys.
That is only a coverage diagnostic, not performance evidence.  The stricter
max-annulus formula above must be measured before any ground truth is consulted.
It cannot be silently replaced by the whole-shell average if coverage falls.

## Module 3: Persistence-First Non-compensatory Decoder

Define

\[
\mathcal C=\{H_j:j<M,\;\pi_v(H_j)>\pi_v(H_M),\;W_t(H_j)=1\}.
\]

Text only determines membership in `C`; its raw margin never ranks candidates.
Vision selects

\[
H^*=\arg\max_{H\in\mathcal C}\pi_v(H).
\]

The deterministic output rule is:

1. if `C` is empty, output `H_M=I8`;
2. if one state has maximum persistence, output it;
3. if maximum-persistence states tie, output the one with the largest extent
   (the conservative edit);
4. if a 4-FPS endpoint-cell tie still remains, output `I8`.

There are no learned weights, raw cross-modal score sums, dataset switches,
numeric stability thresholds, or minimum-`k` preference.  The last point is
important: selecting the first text-feasible `k` would reproduce the known
top-k over-contraction failure.  In PACT-v3, speech can reject a visual scale
but cannot force the smallest interval; frozen visual persistence determines
the final scale.

The VASTA joint existence rule remains a conservative fallback inside this
decoder and is not claimed as new: under visual disagreement, an empty output
still requires one negative visual expert and stable negative transcript
evidence.  PACT-v3's new claim concerns boundary-scale selection.

## Why both modalities are necessary

- Removing vision removes proposal rank, nested states, persistence, shells,
  and every legal endpoint.  Text cannot emit a boundary.
- Removing text leaves no shell warrant and therefore returns `I8` exactly.
- Text is binary and cannot compensate for poor visual persistence with a large
  logit.
- Vision cannot authorize an edit whose core fails null or contains weaker
  semantics than a discarded shell.

This is not generic rank fusion: only vision produces and orders candidates;
speech verifies a claim defined by the visual filtration.  The cross-modal
object is the **core-versus-visual-shell warrant**, not a weighted combination
of two rankings.

## Direct implementation checklist

1. Reuse the frozen NMS=.5 top-eight proposal JSONL.
2. For each video, rasterize each prefix union at 4 FPS and extract the
   rank-1-center connected component.
3. Collapse duplicate endpoint-cell states and store each state's rank lifetime
   `pi_v` plus every incremental annulus.
4. Join timestamped per-chunk logits by sanitized `(dataset, video_id)` only;
   report missing keys, never infer them from filename labels.
5. Build `q(t)` with explicit-null surplus and neutral ASR gaps.
6. Repeat the warrant for original, minus-one-chunk, and plus-one-chunk timestamp
   assignments.
7. Apply the exact persistence-first decoder; record the reason for every edit
   or fallback.
8. Before loading GT, hash predictions and report candidate/warrant/edit
   coverage by dataset.
9. Only then evaluate interval F1 and boundary edit correctness.

## Cache-first promotion and kill gates

### Label-free coverage gate

- at least 10 and between 5% and 30% of all common videos must receive a final
  non-fallback boundary edit;
- at least three datasets must contain edits;
- the max-annulus rule, timestamp jitter, and visual persistence condition are
  frozen before GT;
- every fallback must be bit-identical to `I8`.

If max-annulus coverage is below ten, the mechanism is killed.  The code may
report whole-shell averaging as an explicitly different ablation, but it cannot
be promoted under the same hypothesis.

### Performance/mechanism gate after prediction freeze

- correct boundary edits must exceed wrong edits, with at least ten edits;
- macro F1@.5 and F1@.7 must both improve over VASTA/`I8`;
- F1@.3 may fall by at most .002;
- aligned timestamps must beat timestamp shuffle by at least .003 F1@.5;
- full PACT-v3 must beat `rank1`, `minimum-k semantic`, visual-persistence-only,
  text-shell-only, and whole-shell-average controls;
- gains must not come only from EMPTY decisions or one dataset.

Failure of any boundary or alignment gate kills PACT-v3 as the main method.

## Relationship to earlier methods

- **CCA:** cohort-level pseudo-agreement estimates a text veto switch and
  transfers it across visual jurisdictions.  PACT-v3 uses no cohort statistic
  or dataset identity; authority is resolved per video on the exact boundary
  claim.
- **MELT:** generates event tuples, phase labels, and lifecycle boundaries.
  PACT-v3 generates none of these; endpoints and shells come only from the
  frozen visual filtration, and text returns a binary native-evidence warrant.
- **HYPER:** asks an MLLM to conduct proposal tournaments.  PACT-v3 uses no
  generated pairwise choice and no language rank over candidates.
- **VASTA:** remains the safe interval/existence fallback.  PACT-v3 preserves
  its working top-8 closure whenever multimodal boundary evidence is absent or
  ambiguous.

## Honest assessment

- Task-specific novelty before GT: **6.0--6.2/10**.
- If four-dataset or prospective evaluation shows positive boundary edits,
  aligned-over-shuffled gain, and load-bearing visual/text controls:
  **6.4--6.7/10**.
- Main risk: the stricter maximum-annulus warrant may have insufficient
  coverage, or persistent early states may still over-contract.
- Current status: conditional mainline; run the label-free coverage diagnostic
  before any further MLLM inference.

## Paper story

> Hateful-event boundaries are not selected by fusing visual and language
> scores.  Frozen visual proposals induce a sequence of persistent temporal
> scales.  Timestamped speech warrants removing a scale only when the retained
> core is semantically above null and every discarded visual shell is weaker.
> Language decides whether a boundary edit is defensible; visual persistence
> decides which defensible boundary is written.

