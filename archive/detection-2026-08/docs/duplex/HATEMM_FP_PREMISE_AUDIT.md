# HateMM corrected-8B false-positive premise audit

**Date:** 2026-08-08  
**Cohort:** all 70 `Non Hate` examples above the corrected-8B run's frozen KDE
threshold (`-2.3455`) on `test_clean`  
**Decision:** **REFUTED under the preregistered rule**

## Result

| Primary category | Count | Rate | Wilson 95% CI |
|---|---:|---:|---:|
| G — generic hostility/violence, no protected target | 15 | 21.4% | 13.4–32.4% |
| Q — quoted/reported/countered protected hate | 11 | 15.7% | 9.0–26.0% |
| O — offensive, but not protected-target hate | 15 | 21.4% | 13.4–32.4% |
| P — protected identity without hostility | 5 | 7.1% | 3.1–15.7% |
| D — likely dataset/policy disagreement | 17 | 24.3% | 15.8–35.5% |
| I — insufficient evidence | 1 | 1.4% | 0.3–7.7% |
| X — other | 6 | 8.6% | 4.0–17.5% |

The preregistration called the premise `REFUTED` when the point estimate for G
was below 30%. The observed estimate is **15/70 = 21.4%**, so it fails before
considering the lower confidence bound. Even assigning the only insufficient
example to G gives 16/70 = 22.9%.

The broader, deliberately secondary `G+O` bucket is 30/70 = 42.9%. This says
that non-policy violence/offensiveness is a real source of false positives, but
it does not establish the proposed protectedness mechanism: O also contains
personal profanity, sexual/shock content, and other cases where protected-target
substitution is not the operative distinction.

## Interpretation

The false-positive population is heterogeneous rather than dominated by one
missing protected-target gate:

1. **Protected identity can already be present in a genuinely non-hateful
   example.** Q alone is 11/70. News reports, testimony, confrontation of a
   racist speaker, and counter-speech mention or quote protected-target hate but
   differ in stance and endorsement. A mechanism that only estimates whether a
   protected target is causally relevant will not resolve these cases.
2. **Dataset/policy disagreement is large.** In 17/70 examples the available
   evidence appears policy-violating despite the `Non Hate` label, including
   overt supremacist songs and explicit racial or antisemitic generalizations.
   These are not clean false positives that a method should learn to suppress.
3. **Generic violence is real but not dominant.** The 15 G cases include riots,
   shootings, arrests, fights, and threat rhetoric with no protected target.
   This can support a targeted component or diagnostic slice, not the main
   explanatory claim for the full error budget.

The evidence therefore rejects **protectedness alone** as the central new
mechanism. If this direction is continued, the empirically motivated object is
at least two-dimensional: **target type × communicative stance/endorsement**,
with dataset-disagreement handling kept separate from model errors. That is a
different hypothesis and needs its own preregistered pilot rather than being
claimed as support for PCE after the fact.

## Reproduction

- Frozen taxonomy and decision rule:
  `docs/duplex/PREREG_hatemm_fp_taxonomy.md`
- Audit packet builder: `scripts/duplex/hatemm_fp_audit_packet.py`
- Aggregate analyzer: `scripts/duplex/hatemm_fp_audit_analyze.py`
- Item-level blinded coding and generated summary are under
  `results/hatemm_fp_audit/` (gitignored because they contain sensitive sample
  descriptions and alias mappings).

Run:

```bash
python3 scripts/duplex/hatemm_fp_audit_analyze.py \
  --coding results/hatemm_fp_audit/coding.tsv \
  --output results/hatemm_fp_audit/summary.json
```
