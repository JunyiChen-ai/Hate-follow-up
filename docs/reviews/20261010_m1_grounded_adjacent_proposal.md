# Proposal review (rule 4): candidate 41, grounded acceptance of adjacent-frame reads, 2026-10-10

Reviewer: one independent agent (same model as the main session), before any GPU run. Read: rule 4, the proposal
`experiments/20261010_m1_grounded_adjacent/README.md`, the placement-controls and selection-analysis sections of
`experiments/20261007_m1_streamingtom/README.md`.

## Literature searched (2026-10-10)

Eleven web / arXiv queries covering hateful, harmful and toxic video detection and localization with MLLM
self-verification or evidence citation; video anomaly detection (LAVAD, VERA, VADTree, Probe-VAD); video temporal
grounding with self-verification (TimeSearch-R, TimeRefine, TFVTG, TIMEPROVE and others); grounded video QA
(GroundedVQA 2608.15574: per-claim timestamp citations checked by a verifier; Xiao et al. CVPR 2024). In hateful
video work the reviewer found evidence-grounded verification only at video level with retrieved or fused evidence
and trained predictors: MATCH (TCSVT 2026, CLIP-retrieved evidence units judged by a verifier LMM, trained MLP),
CLARA (2608.15905, trained video-level verifier with no frame or timestamp field), Yadav & Singh WWW Companion '26
(video-level evidence attribution; abstract only). LELA (2602.09637) and MARS (2601.15115) have no citation gate.
The proposed mechanism (post-read probe asking the model to cite an in-window shown frame, per-window accept or
fall back conditioned on the citation and the direction of change, no labels) was not found in hateful or harmful
video work; the nearest idea is GroundedVQA's citation verification in video QA, a transfer allowed by rule 4.

## Repository matches

The proposal's summaries of Explorer 18, HVL, Attributor, Grounder, Verification 27 and Provenance 25 check out.
Also related, not listed: Interval Witness 26 (`archive/experiments/20261005_m1_interval_witness`): frame witnesses
cite local frame ids and a field without a witness stays unknown, applied to factual fields, not to a window score;
GLR section 7 scored the model's cited timestamps directly (within .580 / .526); Program 23 generated source
programs. None uses a post-read citation probe to choose between two reads of the same window.

## STOP cases

1. Already used for hateful video detection / localization: no. 2. Pure ensemble: no (one read per window is
chosen; user ruling 2026-10-10). 3. Pure calibration / post-processing / smoothing: no. 4. Engineering trick only:
no; a stated hypothesis from the selection error analysis, a second probe on the extended cache, a parse-and-match
of the cited time, a direction-conditioned decision rule and declared controls.

Advisory, not a STOP: GroundedVQA reports that asking a VLM directly whether a frame supports a claim caught none of
the fabricated claims; HVL E0 and GLR show this model's own citations localize near chance. The random-acceptance
control is the right test.

VERDICT: PASS
