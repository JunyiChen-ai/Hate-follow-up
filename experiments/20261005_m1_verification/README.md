# Candidate27: factored factual reobservation before temporal measurement

Rank9, last entry of the unchanged original9 pool:
`experiments/20261004_m1_ideation/CANDIDATES.json`, jury
`docs/reviews/20261004_m1_ideation_jury.md`. Declared2026-10-05 onsc474397.
Independent once-only proposal review PASS: `docs/reviews/20261005_m1_verification_proposal.md`; prototype/independent code review PASS; no scientificGPU yet.
Native formal reference r6_bma, full215HateMM+118HateClipSeg, canonical4fps,
fixedr6, all results development-selected. Candidate25/26 outcomes pending;
this independent proposal does not prejudge them or reset a previous family.

## Complete mechanism

An earlier description can contaminate its own verification. Transfer the
complete factored Chain-of-Verification (draft, plan, independent answers,
revised final response), with a typed actual source/time record between factual
reobservation and final temporal scoring. This is not a first-verification claim.
The hypothesis is that independently remeasuring local actor/target/quotation
facts corrects misplaced hateful meaning before ordinary independent readings.
Do not compare opposite hate hypotheses or average independently scored answers.

Reuse native actual20 requested overview frames (18-20 actual), fullASR and
shared actual PTS thirds-based2localframes/8s window. Source paths/video IDs
never enter model-visible text; actual coordinates/owning window do. RawASR
crops remain proportional words, not verified word timing; two frames do not
prove absence. Use one frozen Qwen3-VL-8B, seed0, FP32greedy constrained known
source handles, whole incomplete outputUNKNOWN, every token/forward charged.
No labels, new model, goldentities or native global verdict in factual stages.

For every original window execute:

1. One neutral draft call sees native overview/fullASR and actual local2frames/
   body, but no hate stance or old answers. It returns exactly four named fields
   `act`, `actor`, `target`, `quotation_owner`; each has `value` up to24words/
   64content tokens, zero-to-two actual LOCAL source handles and `known` boolean.
   Known requires one actual owned witness. UNKNOWN is valid. Do not generate
   hate probabilities, protected identity guesses or names absent from sources.
   A speech/quotation identity not established by the sources stays UNKNOWN.
2. One text-only planner sees this actual draft, the local source availability,
   and the neutral field task. It emits one or two distinct field names and an
   open factual question per field, up to24words/64content tokens. If no actual
   local source exists emit zero questions. The questions may identify a topic
   from the draft but should not assert its proposed answer. No yes/no hate
   question, alternatives, numeric confidence or retry. Questions are recorded
   verbatim; answer leakage cannot be certified absent just by hiding a field.
3. Each planned question gets a fresh independent multimodal Qwen call, seeing
   that one question, original native overview/fullASR and actual current local
   media/body, but no draft values/answers, other verification answers, planner
   reasoning or native global stance. Return the same field record with real
   current source handles or UNKNOWN. At most2 calls; no voting/comparison of
   numeric prediction margins. Exact source ID/speech span/frame coordinates
   are validated against current actual pixels/ASR, never an external answer.
4. Execute an explicit source/time compiler: preserve unqueried valid draft
   fields; replace each queried field with its valid fresh observation, including
   UNKNOWN when unsupported. Preserve old/new values and a literal-changed flag.
   Changed wording is not proof of semantic contradiction or corrected identity.
   Out-of-window evidence cannot fill a local field. The final revised response
   is the ordinary fresh modality Yes/No reading of original media plus compiled
   facts. It sees verified values and their provenance, not obsolete replaced
   draft values; the raw draft remains audit data. Native currentG and its own
   hardstance remain unchanged and freshly paired. Visual and available-speech
   final branches are fresh independent full multimodal calls with native20/
   fullASR/currentstance, compiled same-window source record, actual local2frames,
   local body forS, and the unchanged original modality yesno_question. Final
   max(V,S), nativeG and fixedr6 only; no generated categorical score conversion.

Constants before generation:8s/4fps;2localframes;0questions only if zero actual
local sources, otherwise1-2 distinct fields; neutral4fields;1024draft tokens,
256planner tokens,256tokens per verification;24words/64content tokens pervalue
orquestion;≤2witnesses perknownfield; all unique overlapping-safe contiguous
1-16word exactspan handles; no temporal/context retrieval, one fixed pass, no
confidence trigger/parameter sweep. Same literal systems/instructions/JSON
schema will be frozen in spec.json before implementation or GPU; proposal review
may identify a concrete missing definition for same once-review clarification.

Inputs `data/temporal_factual_verification/`, all outputs
`runs/20261005_m1_verification/r1_*`. Stable shared src imports only, no imports
from another experiment. Previous categorical model observations are never
reused as current factual input. Existing decoded pixels may be reused only
with actual rawPTS/PNG/source checks and their original decode cost retained.

## Source and target prior scope

[CoVe](https://aclanthology.org/2024.findings-acl.212.pdf)3.1-3.4 was actually
reopened/read2026-10-05, including factored execution and final revision. It
uses one LLM, excludes the draft and other answers during independent checking.
Its original3.4 final revision sees the baseline and verification question/answer
pairs. This target adaptation uses typed replacement, retains unqueried valid
fields, and excludes obsolete replaced draft values before ordinary modality
reading: four functional phases, not literal3.4 reproduction or Factor+Revise
additional inconsistency cross-check.
The source does not execute video witnesses. This is a source/time-constrained
multimodal adaptation, not an exact reproduction of its fewshot tasks.
[LEAF](https://aclanthology.org/2026.findings-acl.604.pdf)3.2/3.3 actually read
from the local primary PDF/text: Reason/Explain then label-guided correction
when prediction disagrees withGT, grounding and supervised distillation. Its
verification is already target-domain prior; we do not transfer gold-label
correction or claim first grounding. Actual downloaded primary scope evidence
`runs/20261005_m1_verification/source_reads/{cove,leaf}.{pdf,txt}`.
[MATCH](https://jianlang.org/papers/MATCH.pdf)III-B-D reopened/read: opposing
clues are paired with actual retrieved units and explicitly shown to its verifier,
then supplied to a trained predictor. Independent factored factual remeasurement
must be distinguished from merely renamed clue verification. [IARE](https://arxiv.org/html/2606.11953v1)
4.2-4.3 and5.3 read in prior jury/currentprimary scope; augmented reasoning and
supervised preference training are not this frozen factual pipeline. Full target
CoVe application search is mandatory once by an independent proposal reviewer;
no assertion of search absence or generic verification novelty.

## Cost and evidence required

Estimated150-420GPUmin, unmeasured and uncertain given actual source-generation
cost of other candidates. Per new video: Wdraft+Wplanner+betweenW and2W fresh
verification calls when sourcesavailable, then≤2W final readings and nativeG/
stance setup. Each draft/verification/read repeats original media encoding;
these calls are explicitly charged, not free because weights are frozen or
observations stored. Existing overview/ASR/source pixels reusable; newly generated
facts and all reasoning tokens charged. Generation=3-4added calls/window,
no new retrieval/frame/encoder. Record actual source acquisition/model forwards/
image forwards/tokens/peak and standalone new-video time per corpus; wrong-source
or same-budget control costs reported as measured, never assumed token-matched.

First run source/native fixed5 (samefirst2/corpus+HMM114) withoutGT. Require all
actualsource/structured token/field replacement/independent call binding and
fresh repeat checks, count verification execution and valid/UNKNOWN fields
honestly; no semantic correctness claimed from parser success. Only then full333
main. Native allraw/all6 exact, shared primary metric+.01 BOTH and no pooled
loss>.005/withinloss>.01 against currentr6. Any qualifyinggain but mainFAIL ->
logged actual postscore GT analysis then≤3revisions; noanygain ->archive.

Only after mainPASS complete full controls: (a) draft-only fields with samefinal
media/reader, (b) samequestions/calls but verification sees originaldraft values,
(c) equalverification count fixedfield order instead of generated questions,
(d) fixed half-window rotation of actual verification donor media/facts, preserving
truthful donor provenance and explicitly declared wrong-local-binding intervention.
Full-vs-draft and full-vs-answer-visible must show same mainmetric+.01 loss BOTH
for factored correction claim; wrong source must falsify proper witness use.
Question planning is an implementation detail unless equalcount fixedfield
control also satisfies dual.01. Structural source validation is necessary
integrity infrastructure, not independently claimed novelty. Audit a fixed
source sample's actual changed facts and quotation attribution (model prose is
not certified semantic truth), raw V/S/max ordering and pairedwithin CIs. No
real corrections, gains only fromwrappers, or controls belowthreshold defeat the
claimed mechanism even if a mainmetric rises. Independent final review/actual
local raw outputs/STATUS required before any successful goal report.

## Literal preimplementation definition

`spec.json` freezes all draft/planner/verification systems and instructions,
fourfield/plan schema, source availability, cap/UNKNOWN behavior, complete
source/compiler replacement/exclusion rules and final native conversation
serialization. Empty/incomplete generated planner retains cost and no questions,
not an invented successful verification. Final reader uses newly compiled facts,
not obsolete replaced draft values. Filepaths remain internal audit information
and are absent from model-visible source descriptions. Prototype implemented, no
scientificGPU yet. Independent once-only proposal review PASS; literal source-adaptation and complete updated four-field serialization clarified within that review. Independent code review PASS.


## Prototype and pre-GPU validation2026-10-05

Implemented self-contained scientific prototype `interface.py`, `inputs.py`,
`extract.py`, `reader.py`, `measure.py`, `analyze.py`. No inter-experiment imports;
shared actual-source acquisition, structured FP32 generation and native Judge/
stance cache only. `analyze.py` delegates four primary evaluations/fixedr6 runs
to unchanged canonical evaluator/existing twolevel CLI; it never reimplements
ROC/PR/within calculation. All score computations excludeGT.

Author `selfcheck.py` completed33checks, including retained-unqueried fields,
queriedUNKNOWN replacement, obsolete values absent final record, independently
recreated original verification context, overlapping unique spans, invalid
ownership/type/duplicate/cap rejection and actual fixed5 source/rawPTS/PNG/input
binding. All7359currentwindows examined,956862 legal speech handles retained.
Actual10 draft/verification source templates:22images and3106–6572expanded
inputtokens; source path/class-bearing IDs absent from visible text. Evidence
`runs/20261005_m1_verification/cpu_checks/summary.json`. Synthetic grammar choices
are explicit author fixtures, not observed model semantics. An initial author
fixture incorrectly encoded description+closingquote as one token sequence;
fixed only fixture to match production's separately emitted closingquote and
reran PASS. Actual scientific pipeline has not run.

Before fixed5 source-interface acceptance require actual independent verification
calls>0 in EACH corpus, plus complete native allraw/currentG parity, allsource/
token/field-replacement/context bindings and fresh repeat equality. UNKNOWN
answers and invalid/capped planners remain explicitly counted, never forced
correct. This minimal execution guard verifies the method executes; it does not
assert correctness, require literal changes, select windows fromGT, or certify
mechanism. Wholecap source costs are retained. Native G/ownstance is measured
once; paired total forwards=base.calls+new.calls−3+repeatdiagnostics, image
forwards=1+newV/Sbranches+diagnostics. Standalone new-video cost includes source
acquisition+nativeG/ownstance setup+allfreshnewV/S reads, excluding pairednative
branches/diagnostics. No second global verdict or score ensemble.

Run in committed Slurm launch only:
`sbatch experiments/20261005_m1_verification/launch/lab1.sbatch` (fixed5 default);
`sbatch --export=ALL,SCOPE=main experiments/20261005_m1_verification/launch/lab1.sbatch`
(full333 only after actual fixed5 PASS). NoGT fixed5 prepare:
`python experiments/20261005_m1_verification/analyze.py --stage prepare --smoke`.
Full CPU detached analysis entry `launch/run_analysis.sh`. Actual machine
availability/clean sync and disk checked immediately before dispatch; no newjob
submitted yet. Single independent rule6 code review PASS; source scientific
capacity and correctness still unmeasured.


Independent initial code review PASS (same-family provisional), frozen
`docs/reviews/20261005_m1_verification_code.md`. Actual production acquisition/
source replay plus real36layerBF16 CPU native/freshreader generation, six speech
availability/repeat layouts,16current source/token/stance/cost/field corruption
rejections and actual5templates passed. Complete/capped draft/plan/verification
production generators exercised, final compiler retains unqueried valid fields
and never falls back to queried obsolete draft. This is executable input/code
validation, not observed8B semantics or performance; noGT/scientificGPU yet.
