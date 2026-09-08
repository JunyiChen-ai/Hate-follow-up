# Provisional novelty check: dual-channel proof-carrying localization (run09)

Cutoff 2026-08-29; performance-blind. This is a mandatory-audit-backed self-review, not a fresh-agent verdict or method acceptance.

## Verdict

| Scope | Score /10 | >=6? |
|---|---:|:---:|
| M1 dual-channel joint-orbit localization | **6.0** | Yes, narrowly |
| M2 hierarchical free energy | 2.8 | No |
| M3 Pareto + selection-safe component | **5.8** | No |
| Task | 2.5 | No |
| Mechanism | **6.0** | Yes, narrowly |
| Integration | **6.2** | Yes |
| **Overall** | **5.9** | **No** |

The dual-channel design legitimately removes most of run07's dead-calibration penalty. Dense prediction remains the empirically selected rank transport, while exact p/e certificates are forwarded unchanged, explicitly not presented as posterior probabilities, and consumed by M3. M1 therefore falls only `0.2` points from the `6.2` standalone witness rather than `1.8` points. Calibration still does not characterize the dense prediction channel.

## Selection-safe e audit

Let the non-full interval weight be

`w(I)=1/[(n-1)(n-|I|+1)]`.

For every length, the weights over placements total `1/(n-1)`; summing lengths `1,...,n-1` gives one. Every fixed-interval mean of valid frame e-values is an e-variable. Consequently the complete weighted interval mixture is an e-variable. For any adaptively selected interval `I*`, its nonnegative component

`w(I*) × mean_{t in I*} E_t`

is bounded by that mixture and therefore also has null expectation at most one. The adaptive-choice argument is correct under the stated global `C_n × C_n` invariance null.

The claim needs two restrictions:

1. It is evidence against the **global relative-phase null allocated to the selected interval**, not proof that the selected interval contains hate or has correct boundaries.
2. It is numerically vacuous in this cohort: every selected component is below one, ranging from `9.59e-8` to `0.00481`. No selected component supplies evidence against the null in the usual e-value sense.

Also, `duration_prior=None` is inaccurate: equal mass over lengths is a fixed, parameter-free structural length prior.

## Implementation gate

Status: **PASS WITH WARNINGS**.

- All rank-prediction, certificate, dual-channel, free-energy, boundary, final and witness artifacts contain 611 unique aligned keys.
- Dual-channel prediction curves equal the rank-transport curves exactly; p/e arrays equal the certificate source exactly.
- Every M3 stored interval weight, interval mean e-value and selected component exactly matches recomputation.
- M3 applies the fixed-4fps contract before candidate construction, preserves dense scores, outputs 611 nonempty intervals, and leaves zero terminal samples for the final contract to remove.
- Independent M3 reconstruction on the 30 shortest videos exactly matches criteria, bottlenecks, candidate counts, lexsort and endpoints.
- Recomputed metrics and strict-common comparison agree within `2.22e-16`.

## Strongest rejection

“Proof-carrying localization” overstates what the certificate proves. The architecture cleanly separates prediction from statistical evidence, but the carried selected component is a tiny globally valid e-value, never exceeding one, and says nothing interval-specific about hate semantics or boundary correctness. The Pareto interval itself remains an uncalibrated composition of direct/certificate contrasts, ERASER-style sufficiency/necessity and lexicographic maximin. LELA already occupies the exact task. Thus the integration is novel enough to exceed six, but the overall localization paradigm remains just below six.

## Cleaner refinement

The original local p-values are already adjusted against the maximum over all pairs and frames. Therefore, after arbitrary interval selection, the minimum globally adjusted p-value inside the selected interval remains no smaller than the global minimum and is selection-safe without the tiny complete-mixture component weight. Forward that selected-interval adjusted p and its valid p-to-e calibration, with precisely global-null—not boundary-correctness—semantics.

Closest foundations: Westfall–Young maxT; Hemerik–Goeman group randomization; Vovk–Wang p/e calibration; Xu–Wang–Ramdas post-selection e-inference; ERASER rationale faithfulness; Walther–Perry scans; LELA and MultiHateLoc.

`review_independence=self-audit`; `acceptance_status=provisional_pending_fresh_reviewer`.
