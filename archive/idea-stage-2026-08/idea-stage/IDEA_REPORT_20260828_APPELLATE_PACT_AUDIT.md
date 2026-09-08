# Appellate-PACT Audit: Visual Ambiguity Invokes Language Jurisdiction

## Verdict

**Kill the current `invert visual_warrant` implementation as a main method.**
It has a promising task-specific idea, but the implemented gate is not an
ambiguity gate and the clean mechanism has too little coverage to support a
paper claim.

The concept may remain a conditional pilot under a corrected name and rule:
**Appellate-PACT**, or “claim-specific cross-modal jurisdiction.”  It reaches
task-specific novelty above 6 only after the clean tri-state rule and a full
timestamp-orbit control produce prospective boundary gains.

## What the current six edits actually contain

The visual witness scores two deterministic frame samplings.  The implemented
positive rule is `both scores > 0`; `--invert` accepts everything else.  On the
development cohort:

- 16 candidates are positive visual consensus;
- 3 are true sampling disagreement/ambiguity;
- 3 are negative visual consensus.

Thus half of the reported six “ambiguity” edits are not ambiguous: both visual
views say that the shell is not weaker than the core.  Letting text contract
those intervals overrides stable visual counterevidence.

On HateClipSeg train+validation, the corresponding counts are six positive,
one true ambiguity, and two negative.  The reported three edits again mix one
ambiguity with two visual rejections.

## Read-only diagnostic decomposition

Dataset-macro interval F1 on the development cohort:

| Gate over the same text candidate | Edits | F1@.3 | F1@.5 | F1@.7 |
|---|---:|---:|---:|---:|
| CCA fallback | 0 | .334516 | .282342 | .173008 |
| Current inverted gate | 6 | .334516 | .283519 | .174488 |
| True ambiguity only | 3 | .334516 | .283519 | .174488 |
| Negative visual consensus only | 3 | .334516 | .282342 | .173008 |
| Positive visual consensus only | 16 | .334516 | .282342 | .173311 |
| Positive consensus plus true ambiguity | 19 | .334516 | .283519 | .174791 |

The clean three ambiguity edits happen to preserve the current threshold
metrics, while the three logically invalid negative edits do not cross an F1
threshold.  This rescues the point estimate but not the evidence: three edits
out of 611 are only 0.49% coverage.

On HateClipSeg train+validation:

| Gate | Edits | F1@.3 | F1@.5 | F1@.7 |
|---|---:|---:|---:|---:|
| CCA | 0 | .200750 | .097561 | .058161 |
| Current inverted gate | 3 | .200750 | .097561 | .058161 |
| True ambiguity only | 1 | .200750 | .097561 | .058161 |
| Positive consensus plus ambiguity | 7 | .198874 | .097561 | .056285 |

The single clean ambiguity edit changes no interval-F1 threshold.  Accepting
the visually positive branch, while logically natural, reduces F1@.3 and
F1@.7 on this cohort.  Therefore a full tri-state appellate system is not yet
empirically coherent across cohorts.

## The clean unified story

> A modality does not receive a global fusion weight.  It receives
> claim-specific jurisdiction.  Vision has original jurisdiction over temporal
> boundaries because it creates the legal endpoint set.  It may support,
> reject, or abstain on a proposed core-versus-shell amendment.  Only an
> abstention opens an appeal to timestamped speech, and the appeal succeeds
> only when its decision is unique to the native temporal alignment.

This is a better story than symmetric multimodal fusion, prompt voting, or
cohort-level CCA transfer.  Its paper-level framing would be:

**“When Vision Abstains: Claim-Specific Cross-Modal Jurisdiction for
Label-Free Hateful Video Localization.”**

CCA may remain the inherited existence substrate, but it is not a new module
or the novelty claim of this boundary method.

## Three modules

Data decoding, frame sampling, ASR, timestamp alignment, proposal extraction,
and 4-FPS rasterization are preprocessing and are not modules.

### Module 1: Persistent Boundary Motions

Frozen proposals `P1,...,P8` induce the nested rank filtration

\[
I_k=\operatorname{CC}_{P_1}\left(\bigcup_{i\leq k}P_i\right).
\]

After collapsing repeated 4-FPS endpoint cells, each non-closure state is a
legal motion `H -> I8` with visual rank persistence.  Language cannot generate
an endpoint.

### Module 2: Tri-State Visual Jurisdiction

For two fixed native frame samplings `o in {0.25,0.75}`, the visual MLLM
computes

\[
v_o(H)=\log p(\mathrm{Yes})-\log p(\mathrm{No})
\]

for whether visible hateful evidence is more concentrated in the retained core
than the discarded shell.  The jurisdiction state is

\[
J_v(H)=
\begin{cases}
\mathrm{SUPPORT}, & v_{.25}>0\land v_{.75}>0,\\
\mathrm{REJECT}, & v_{.25}\leq0\land v_{.75}\leq0,\\
\mathrm{ABSTAIN}, & \text{otherwise}.
\end{cases}
\]

`REJECT` can never be inverted into language authority.  `ABSTAIN`, rather
than generic non-support, is the only state that opens an appeal.

### Module 3: Alignment-Exclusive Language Appeal

Let `Sel(q)` be the frozen semantic-shell selector under the native timestamped
chunk field `q`.  Let `rho_r q` circularly rotate chunk evidence through all
non-identity timestamp assignments.  A language appeal for candidate `H`
succeeds only if

\[
T(H)=
\mathbf1[\mathrm{Sel}(q)=H]
\land
\mathbf1[\forall r\ne0:\mathrm{Sel}(\rho_rq)\ne H]
\land
\mathbf1[\mathrm{Sel}(q^{-1})=\mathrm{Sel}(q^{+1})=H].
\]

The last term retains deterministic plus/minus-one-chunk stability.  The
all-rotation term replaces the current single hash rotation; one negative
sample cannot establish alignment specificity.

For the narrow “language only on ambiguity” experiment, the decoder is

\[
\widehat I=
\begin{cases}
H,&J_v(H)=\mathrm{ABSTAIN}\land T(H)=1,\\
I_8,&\text{otherwise}.
\end{cases}
\]

This rule is clean but currently has only three development edits and one
HateClipSeg train+validation edit before the stricter full-orbit control.

A broader appellate decoder could let visual `SUPPORT` directly authorize `H`,
use `REJECT` to return `I8`, and invoke text only for `ABSTAIN`.  That is more
logically complete, but current HateClipSeg results show that its positive
visual branch can hurt.  It must be preregistered and retested rather than
selectively enabled on the development cohort.

## Is novelty at least 6?

- Current inverted implementation: **4.7--5.1/10**.  It is a post-hoc polarity
  flip that conflates rejection and ambiguity.
- Clean ambiguity-only mechanism before new evidence: **5.7--6.0/10**.  The
  formulation is interesting, but three/one edits make it a sparse diagnostic,
  not a demonstrated localization paradigm.
- Clean tri-state jurisdiction plus full alignment-orbit warrant:
  **5.9--6.2/10 conceptually**.
- With prospective multi-dataset positive boundary edits and load-bearing
  visual/text controls: **6.3--6.6/10 task-specific**.

General-method novelty remains below this because selective cascades and
abstention routing are established ideas.  The task-specific novelty must come
from different modalities owning different temporal claims, not from calling
an inverted binary gate “jurisdiction.”

## Maximum risk

The largest risk is **post-hoc sparse routing masquerading as multimodal
reasoning**.  Six development edits can easily be a lucky subset selected after
many variants, and the clean rule has only three.  The same Qwen3-VL family is
also used for visual and transcript judgments, so apparent modality
complementarity may be correlated prompt instability rather than independent
evidence.

Secondary risks:

- the visual MLLM samples only 16 frames and may miss short shell evidence;
- a single circular timestamp rotation is not a sufficient alignment control;
- HateClipSeg equality means the edits did not cross thresholds, not that they
  were correct;
- development reuse and non-significant small gains do not support SOTA;
- enabling the visually positive branch is logically clean but currently harms
  the sealed HateClipSeg cohort.

## Promotion or kill test

Before ground truth:

1. split the witness into exact `SUPPORT/REJECT/ABSTAIN` states;
2. run the full non-identity circular timestamp orbit for `ABSTAIN` only;
3. freeze all predictions and report effective edits by dataset;
4. require at least ten clean appellate edits across at least three datasets;
5. require every `REJECT` case to fall back bit-identically to CCA/`I8`.

After freezing:

- correct boundary edits must exceed wrong edits;
- F1@.5 and F1@.7 must improve without F1@.3 decreasing more than .002;
- native alignment must beat full-orbit and matched-random ambiguity gates;
- ambiguity routing must beat text-only, visual-only, positive-only, and
  edit-count-matched random routing;
- the result must survive a prospective temporally annotated cohort.

If the full-orbit clean rule yields fewer than ten edits, or if its improvement
is confined to one dataset, kill Appellate-PACT as the mainline.  It may remain
an analysis of rare complementary-modality cases, but not the paper's central
method.

