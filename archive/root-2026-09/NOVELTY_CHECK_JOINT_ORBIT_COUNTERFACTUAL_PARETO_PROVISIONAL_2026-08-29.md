# Provisional novelty check: joint-orbit e-transport + counterfactual Pareto interval (run08)

Cutoff 2026-08-29; performance-blind. This is an implementation-gated self-review, not a fresh-agent verdict or method acceptance.

## Verdict

| Scope | Score /10 | >=6? |
|---|---:|:---:|
| M1 joint witness before readout | 6.2 | Yes, provisionally |
| M1 actual witness + e-transport | **5.8** | No |
| M2 hierarchical free energy | 2.8 | No |
| M3 counterfactual maximin interval | **5.5** | No |
| Task | 2.5 | No |
| Mechanism | 5.8 | No |
| Integration | **6.0** | Yes, narrowly |
| **Overall** | **5.7** | **No** |

The new chain fixes run07's dead-calibration problem: exact p/e magnitudes are stored and used, and M3 consumes the same certificate. This is a meaningful integration upgrade. It still does not establish a new end-to-end statistical localization principle.

## p/e audit

- Each local witness p-value is the inclusive orbit tail of its statistic against the single maximum over every pair and time point. Under the explicitly assumed complete `C_n × C_n` group invariance, this is a conservative single-step adjusted randomization p-value.
- Taking the minimum across active pairs at one time is valid here because all pair statistics use the same maximum reference; it equals the global-max tail evaluated at the largest active local statistic.
- `e(p)=1/(2√p)` is a valid p-to-e calibrator: it is decreasing and its integral on `[0,1]` equals one. See Vovk & Wang, Annals of Statistics 2021: https://doi.org/10.1214/20-AOS2020
- The output logit update is `z + log(e) - mean(log(e))`. The stored `e` remains an e-value, but `e/geometric_mean(e)` is generally **not** an e-variable. The updated posterior therefore has no e-value, p-value, FWER or anytime-valid interpretation.
- M3's adaptive interval search and rank/maximin selection also do not transfer the local adjusted-p guarantee to the selected interval.

## Implementation gate

Status: **PASS WITH WARNINGS**.

- All witness, nominal, transport, free-energy, boundary and final stages contain the same 611 unique keys.
- Every stored `joint_adjusted_p` and `joint_evalue` exactly matches independent recomputation; output logits match within `2.67e-15`, and video logit means within `6.67e-16`.
- Forty-eight videos have all pairs null. Across all other videos, only six framewise e-values exceed one; the observed e range is `[0.5, 1.033804]`.
- M3 lexsort is correct: maximize bottleneck, then maximize rank sum, then minimize length. Exact remaining ties implicitly use candidate enumeration order, hence earlier start.
- Independent reconstruction on the 30 shortest videos matches every criterion rank, bottleneck, candidate count and selected endpoint. Two apparent endpoint differences are exactly explained by duration clipping.
- Audit totals match: 814,724 candidates, median 405, maximum 33,669, no pre-contract fallback.
- Recomputed metrics/common comparison match within `1.11e-16`.
- **Timeline defect:** six intervals select only the extra terminal sample and become empty when the fixed-4fps contract removes that sample. Candidate generation must use the contracted timeline.

## M3 novelty judgment

M3's exact four-criterion combination is uncommon in hateful-video localization and now establishes a coherent certificate/propensity interface. Nevertheless:

- direct and certificate inside/outside contrasts are scan statistics;
- retention sufficiency and deletion necessity closely parallel rationale faithfulness/comprehensiveness metrics such as ERASER: https://aclanthology.org/2020.acl-main.408/
- criterion midranks plus a minimum and lexicographic tie-break are classical maximin multi-objective selection;
- it does not compute a Pareto front, and “counterfactual” means deterministic masking, not an identified causal counterfactual;
- candidate selection is not included in an orbit/null calibration.

Thus M3 is medium novelty (`5.5`), not a new interval-inference paradigm.

## Strongest rejection

LELA already occupies the exact training-free multimodal hate-localization task. M1 is a careful application of established group randomization, maxT and p-to-e calibration. M3 combines established inside/outside scans, rationale sufficiency/comprehensiveness and lexicographic maximin selection. The shared certificate makes the integration genuinely coherent, but no theorem or calibrated construction survives centering and adaptive interval selection. The six terminal-only selections further show that the claimed intrinsic endpoint mechanism is not yet timeline-safe. Therefore integration reaches a borderline 6, but the overall contribution remains below 6.

Closest foundations include Westfall–Young maxT, Hemerik–Goeman group randomization, Harris/Yuan–Shou shift tests, Vovk–Wang p/e calibration, ERASER rationale fidelity, Walther–Perry scan calibration, LELA, MultiHateLoc, T3AL and Memory Matters.

`review_independence=self-audit`; `acceptance_status=provisional_pending_fresh_reviewer`.
