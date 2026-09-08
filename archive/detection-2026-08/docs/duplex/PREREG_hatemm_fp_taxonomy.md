# Pre-registration — HateMM false-positive premise audit

**Frozen:** 2026-08-08 before reading the 70 false-positive contents.
**Purpose:** test the premise that a substantial part of HateMM's corrected-8B
false-positive budget is generic hostility/violence without a protected target.
This is error analysis, not a method experiment.

## Cohort

All `Non Hate` videos in HateMM `test_clean` whose corrected 8B raw score is at
or above that run's own frozen KDE valley. Recompute the cohort from source
scores and annotations; expected `n=70`, threshold `-2.3455`. Assign seed-20260808
random aliases before content review.

## Evidence order

Review the fresh/gated transcript and title first. Inspect the frozen 16-frame
contact sheet whenever text is empty, too short, internally ambiguous, or
insufficient to determine target/context. The reviewer may use both modalities
but may not inspect scores, exact IDs, or neighboring gold examples while
assigning categories.

## Exclusive primary taxonomy

- **G — generic hostility/violence:** hostile, violent, threatening, degrading,
  or graphic content, but no target defined by a policy-protected attribute.
- **Q — quoted/reported/countered protected hate:** protected-target hostility
  appears, but the video reports, quotes, documents, mocks, or opposes it rather
  than endorsing it.
- **O — offensive but not protected-target hate:** profanity, personal insult,
  sexual material, shock content, or abuse toward a non-protected individual.
- **P — protected identity mention without hostility:** a protected group is
  salient, but no policy-level hostile predicate is asserted toward it.
- **D — likely dataset disagreement/noise:** the available evidence appears to
  satisfy the supplied YouTube hate policy despite the dataset's `Non Hate`
  label. Use conservatively and state the apparent rule.
- **I — insufficient evidence:** title/transcript/16 frames do not support a
  defensible content classification.
- **X — other:** none of the above; provide a short reason.

Each item also receives confidence `high/medium/low` and a one-sentence evidence
note. Do not infer speaker endorsement merely from hostile words in a transcript.

## Premise decision

The PCE premise is **SUPPORTED** only if the Wilson 95% lower bound for category
G is at least 0.30 of all 70 false positives. It is **WEAK** if point estimate G
is at least 0.30 but its lower bound is below 0.30. It is **REFUTED** if the G
point estimate is below 0.30.

Secondary reading: `G+O` measures generic non-policy hostility broadly, but
cannot rescue a failed G premise. `Q` is counterevidence to PCE because protected
identity is present in a non-hateful item. `D+I` bounds annotation/evidence
uncertainty.

