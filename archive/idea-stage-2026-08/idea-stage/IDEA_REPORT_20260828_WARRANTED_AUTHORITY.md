# Research Idea Report: Warranted Multimodal Authority

**Direction:** Upgrade the working VASTA mechanism into a stronger, unified method for label-free hateful-video localization without allowing one modality to dominate.

## Recommended Idea: CWA — Counterfactually Warranted Authority

### Method

1. Build matched evidence views for each modality. For text, compare the real timestamped transcript with an explicit empty input and an equal-length within-video temporal shift. For vision, compare the proposed interval with equal-length real neighboring flanks and an outside-only view; do not use black frames or mask seams.
2. Each modality independently returns one of `support`, `oppose`, or `abstain`. It obtains a warrant only if real evidence beats both its own null and its matched perturbation in the same direction, with stable sign across two deterministic views.
3. Resolve the prediction with a non-compensatory authority lattice. Visual consensus remains authoritative; only visual disagreements are adjudicated. Text may veto existence but may never create or move boundaries. Conflicting or absent warrants fall back exactly to frozen VASTA.

### Core hypothesis

Hateful-video localization should allocate claim-specific decision rights to modalities only after their evidence survives matched counterfactual views, rather than fuse incomparable multimodal scores.

### Why this is more than VASTA

VASTA assigns rights from expert identity and a single empty-transcript reference. CWA makes the right itself falsifiable: evidence must remain informative relative to both a content-free null and a content-matched perturbation. The same principle applies independently to visual support and transcript opposition.

### Three paper modules

1. **Matched Multimodal Evidence Views**
2. **Discrete Evidence Warranting**
3. **Non-compensatory Authority Lattice**

Frame sampling, ASR alignment, proposal generation, view rendering, and curve rasterization are preprocessing, not modules.

### Minimum pilot

- Stage A: reuse the existing 279 visual-disagreement videos; test transcript real-vs-empty and real-vs-time-shift sign consistency, net correct flips, warrant coverage, and matched-count controls.
- Stage B: on 32–64 videos, evaluate visual proposed interval vs equal-length native flanks/outside-only views. Compare CWA with frozen VASTA without tuning on the sealed cohort.
- Stage C: only if A/B pass, freeze CWA and evaluate once on a new sealed cohort.

### Hard kill criteria

- Warrant sign repeatability below 90%, or matched perturbation/seam controls reproduce at least 80% of the main effect.
- Full CWA differs from text-only or visual-only by less than .003 F1@.5, or one modality owns more than 80% of accepted decisions.
- The new rule changes fewer than 10% of disagreement cases, correct flips do not exceed wrong flips, or it fails to beat VASTA and a matched-shuffle policy.
- Final sealed F1@.5 does not improve with a positive paired interval, or fewer than three of four datasets improve.

### Assessment

- Novelty prior: **6.8/10**; potentially **7–7.5/10** only after matched-intervention validity, independent multimodal contribution, and sealed improvement are demonstrated.
- Workability prior: **40–50%**.
- Main reviewer risk: intervention views may measure editing/OOD artifacts rather than evidence relevance.
- Safe claim: **counterfactual-view consistency warrant**, not causal effect or statistical certificate.

### Stage-A result update

A cache-only version using two same-dataset, nearest-length transcript controls
was executed on the 611-video development cohort.  It retained 27 opposed
warrants instead of VASTA's 48 null-referenced vetoes and reduced macro
F1@0.5 from 0.28138 to 0.27728.  This variant is **killed**: cross-video
length matching removes too many correct vetoes and is not the mechanism to
carry forward.

The next falsification test is visual rescue on only the 48 frozen VASTA veto
cases.  Two offset samplings compare the A10 proposal core with an equal-count
outside-only native-frame view.  A stable proposal-over-outside sign may grant
visual support the right to reject a text veto; otherwise the prediction stays
identical to VASTA.

## Ranked alternatives

### 2. SAFE — Self-null-calibrated Authority State Estimation

Extend VASTA's explicit text null to modality-specific nulls and combine only discrete `support/oppose/abstain` states. It is the safest performance upgrade, but likely only 6.2–6.6 novelty unless matched interventions change real decisions.

### 3. SAGE — Sufficient-and-Necessary Evidence Localization

Search for the shortest proposal that remains sufficient when kept and necessary when removed. This is the cleanest conceptual localization objective and has high novelty potential, but prior repository pilots show low/zero certificate coverage and editing-distribution artifacts. Keep only as a Stage-B probe, not the main method yet.

## Eliminated directions

| Direction | Reason |
|---|---|
| Temporal argument graph | Same MLLM self-generates speaker, target, hostility, and stance; cascading hallucination and text dominance. |
| Canvas reranking / weighted fusion | No new scientific object; easy to classify as generic late fusion. |
| Prompt jury / debate / self-consistency | Correlated prompt votes do not establish independent evidence. |
| Learned or soft router | Requires labels/pseudo-labels and collapses the authority story into ordinary mixture-of-experts. |
| Hard causal certificate from masked videos | Mask seams and removed context are OOD; causal language is not defensible. |

## Recommended story

> Existing multimodal localization asks which modality has the larger score. We ask which modality has the right to decide which claim. A modality earns that right only when its real evidence survives both its own null and a matched perturbation; decision rights remain claim-specific, so vision owns temporal support while language can only oppose event existence under visual disagreement.

This yields a single contribution—**warranted allocation of modality-specific decision rights**—rather than a stack of extent, routing, counterfactual, and calibration modules.
