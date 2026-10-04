# Candidate26: interval witness composition and targeted observation reconciliation

Rank8 from the unchanged nine-candidate pool:
`experiments/20261004_m1_ideation/CANDIDATES.json`; jury
`docs/reviews/20261004_m1_ideation_jury.md`. Proposed2026-10-05 onsc474397;
initial scientific implementation and CPU checks complete; independent code review PASS, scientific GPU not run. Current formal reference remains r6_bma.
Whole215HateMM+118HateClipSeg, canonical4fps, development-selected.

## Complete mechanism and hypothesis

An interval-level interpretation must preserve the actual leaf that owns its
witness. A whole-video claim cannot establish an act in every window. Construct
a complete binary temporal partition of the existing8second windows, compose
source-bound observations with explicit existential/coverage rules, and use
cross-level disagreements to request one fresh local observation. The resulting
source tree, not its categorical statuses as numeric scores, enters new global
and independent local readings. Hypothesis: correcting temporal ownership and
missing observation before scoring improves within-video ordering while
preserving pooled ranking. Generic extra reasoning or prompt phrasing is not
the proposed scientific contribution.

Input: exact current20 requested native overview frames (actual18-20), full
current ASR, and two actual local frames/window from the existing shared
`src/actual_video_frames.py` PTS-based thirds selection. Their decode and
per-new-video costs are charged; original decoded witnesses may be reused only
after actual source/pixel checks and retaining their original acquisition cost.
No new model, label, detector, gold entity list or word timestamp. Proportional
ASR crops and nominal native overview times remain explicitly limited.

One frozen Qwen3-VL-8B-Instruct for all generation/readings; greedyFP32 head,
seed0, identical flow/constants both corpora. Native pairedG/own hard YesNo/V/S
is freshly measured and must reproduce current allraw/all6 exactly. Use target
HF5.15.1 cache semantics or explicit correct source-generation positions;
never modify the frozen native Judge or evaluator to make a candidate pass.

1. Leaf observations: one fresh call per window, seeing its actual two frames,
   literal local ASR and native overview, with no native global verdict or prior
   answers. Generate `status=present/absent/UNKNOWN`, a max24word description,
   and zero to two exact LOCAL frame/text witnesses. `present` requires at least
   one owned witness. `absent` means no act in the inspected sources, not absence
   over all unseen video frames. An unavailable modality/invalid record remains
   explicitly UNKNOWN. No confidence or numeric hate margin is generated.
2. Temporal tree: leaves0..W-1, recursively split contiguous half-open ranges at
   floor midpoint. One text-only Qwen parent call for each of W-1 nodes sees its
   two child records, literal source bodies/coordinates of descendant witnesses
   and observed coverage. It proposes a status, at most two actual descendant
   witness IDs and a max24word contextual description. Parent source IDs are
   restricted to this interval; unsupported IDs never pass validation.
3. Executable composition: deterministic presence if any child is present;
   inspected-source absence only if both children are absent with their declared
   available-source coverage; otherwise UNKNOWN. Parent proposals are compared
   to this composition. A present parent witness whose owning leaf is absent or
   UNKNOWN, or an absent parent with present descendants, identifies a concrete
   disagreement. These are model observation disagreements, not certified facts.
4. One repair sweep: at most one fresh reread per original leaf. For a present
   parent disagreement, target the earliest proposed witness leaf whose current
   status disagrees; for absent disagreement, earliest present witness leaf;
   otherwise earliest UNKNOWN descendant. Coalesce repeated targets. Choose
   the deepest triggering parent, then earliest interval/leaf for ties. The
   repair sees the original actual local sources and that parent's literal
   interpretation context, but excludes the prior leaf answer/status. No new
   frames, iterative retries, score threshold or confidence trigger. Preserve
   original and repaired records. Recompose all parents deterministically from
   final leaves without further model calls; retain original parent proposals
   as interpretation context with resolved/unresolved flags.
5. New measurement: native20/fullASR plus the chronological reconciled tree
   supplies a fresh globalG and its own hard YesNo stance. Each original window
   gets fresh independent visual and available-speech branches on the same
   current conversation, plus its actual local media and exact leaf-to-root
   source path. Remote witness sources have interpretation-context roles and
   explicit owning intervals; only current sources establish local occurrence.
   Preserve original modality question and `max(V,S)` then unchanged r6 M2-4.
   No averaging certificates, predictions, alternative hypotheses or model
   branches. All final numeric margins come from this one fresh reader.

## Constants and source interfaces, before any generation

8s windows/4fps fixed;2actual local frames at thirds; binary midpoint partition;
seed0;512generated tokens each leaf/parent/repair; descriptions24whitespace
words/64content tokens;≤2witnesses; one coalesced repair/leaf/one sweep. No scan.
Source text witness candidates are all unique contiguous1-16word substrings,
represented by current word character boundaries/coordinate handles, with the
exact body once. Frame witnesses cite only actual current local frame IDs.
Parent witness handles include actual owning leaf and are restricted to its
descendants. Use the reviewed shared measured constrained source generator for
literal JSON/known handles; optional witnesses/UNKNOWN remain legal. Every
forced/chosen token is forwarded, recorded and replayed. Whole incomplete/cap
outputs become UNKNOWN; no salvage or invented source. Grammar checks are
structural and never certify semantic correctness.

New input cache `data/temporal_interval_witness/` with PROVENANCE.md; all output
`runs/20261005_m1_interval_witness/r1_*`. No experiment-to-experiment imports:
reuse stable shared `src/` infrastructure only. Inputs/readings/actual costs are
recorded per video. Initial fixed5 is the same first2per corpus plus HMM114,
chosen without annotations. Require actual source/token/native/repeat/coverage
binding and count parent/repair execution; report zero repairs honestly. Do not
require, force or invent a fixed5 semantic disagreement. Full333 determines
whether repair is actually exercised and whether any mechanism claim survives.

## Source scope and target neighbors

Self-designed interval witness composition/repair, not a reproduction of
VideoTree. [VideoTree](https://arxiv.org/html/2405.19209v3)3.1-3.3/Algorithm1 uses
adaptive semantic clustering/relevance expansion and selected captions; it
motivates structured video representation. Its clustering and confidence rules
are not used here. Existing actual source read is retained in
`runs/20261004_m1_tree/proposal_review/videotree_primary_text.txt`; reopened the
primary page2026-10-05 while preparing this proposal.
[MATCH](https://jianlang.org/papers/MATCH.pdf)III-A-D already retrieves and
cross-verifies opposing hate/nonhate clues then trains a rationale-enhanced
predictor. Reopened the full primary PDF2026-10-05. It prevents any first
verification/first grounding claim. This candidate does not transfer its
opposing hypotheses, CLIP pairing or trained predictor.

MAESTRO already dynamically chooses tools/chunks and iterates local/global
reasoning. Official full DSTA PDF access previously403; primary indexed
appendix scope only, documented in
`runs/20261004_m1_tree/r2_source_scope/scope.json`. Its complete mechanism is
not declared excluded. Independent proposal reviewer must actually search
target-domain temporal ownership/composition/reconciliation and apply the four
STOP reasons in rule4, once. Narrow candidate claim is executed complete
temporal witness ownership/coverage composition and its disagreement-triggered
observation repair, not generic tree prompting, verification or tool use.

## Cost and decisive evidence

Unmeasured estimate180-420GPUmin complete333; potential substantial source
generation cost is explicit. New-video calls Wleaf+(W-1)parent+Rrepair, R≤W,
plus one new global/stance setup and up to2W fresh readings. Source image
encoding repeated in each leaf/repair and all full-reader native/local images
is charged; parent text sizes/caption tokens/actual generation forwards and
peak memory reported. Cached observations cannot hide new-video acquisition.
Existing overview/ASR/actual decoded pixels can be reused as sources, never old
categorical model outputs from another method. No rented compute or new dataset.

Performance: native complete allraw/all6 exact; SAME main metric +.01 in BOTH,
no pooled loss>.005 or within loss>.01 against
`runs/20260926_twolevel/r6_bma/metrics.json`. If no qualifying gain archive;
qualifying but mainFAIL allows logged postscore GT error analysis/≤3revisions.

Only after mainPASS run complete controls: (a) unreconciled same tree/leaf facts,
(b) equal actual repair count at uniformly spaced distinct leaf indices with
same repair source budget, (c) flattened chronological same facts/media with
parent composition removed, (d) fixed half-leaf rotation of child ownership,
retaining literal donor provenance and honestly declared wrong-binding
intervention. Report actual tokens/media/costs; do not assert matched tokens
where text serialization differs. Optional nativeG control separates new
global ranking from local source ordering. Require full-versus-unreconciled and
full-versus-equal-count common-metric .01 losses BOTH for repair novelty;
ownership composition separately needs the same dual-corpus ablation plus
wrong-ownership falsification. Unsupported components are downgraded/removed,
not claimed novel. Audit actual semantic witness/repair correctness on fixed
sources, raw ordering and paired within CIs; absence of real repairs defeats
a repair claim even if final metrics rise. Independent code/final review and
local result/STATUS records precede any successful goal report.

## Frozen literal/input clarification before implementation

`spec.json` now declares actual leaf/parent/repair literal system/instructions,
native nine-rule policy, JSON schema, coverage, full descendant handle/table
range, repair conflict sorting and final reader serialization. Present means
a local act violating the original immutable src/mllm_judge YOUTUBE_RULES.
No labels/examples from any dataset. Absent normalizes UNKNOWN unless both
visual and speech observed-source modalities are available; missing ASR is
not silence. Parent ALL must cover both modalities of every descendant.
Parent may propose ALL real descendant source handles, not just previously
chosen witnesses; it reads all literal descendant ASR bodies/word boundaries,
but only frame coordinates/leaf descriptions, never descendant pixels. Such
frame references are inspection suggestions, not visual verification.
Repair context excludes old local records/child table/parent status/native
verdict; only parent range/description and proposed literal witness sources
remain. One deepest-parent trigger per leaf; exact tie order in spec. NewG
appends complete canonical final tree to native prefix; own hard stance;
independent local branches add source path/local media and original questions.
This is pre-run design material, not an implemented or successful method.

## Initial implementation and CPU verification

Independent proposal PASS: `docs/reviews/20261005_m1_interval_witness_proposal.md`;
unchanged review supplement `runs/20261005_m1_interval_witness/proposal_review/spec_confirmation.md`.
Actual code `interface.py`, `inputs.py`, `extract.py`, `reader.py`, `measure.py`,
`analyze.py`; only stable src imports, shared canonical evaluator/native Judge unchanged.
Author CPU checks `runs/20261005_m1_interval_witness/cpu_checks/summary.json`:
29 executable composition/coverage/owned-source/repair-order and actual model-visible path checks; all7359
source windows current-ASR span preflight; actual fixed5 rawPTS/PNG source validation
and leaf input encoding3202/3157/3457/3483/3180 expanded tokens22images.
Existing candidate25A pixels used only as CPU input-check sources; scientific
26 extraction decodes its own sources and charges actual acquisition. Synthetic
grammar/composition fixture choices are not Qwen observations or semantic evidence.
No GT or performance measured. Root native runtime Slurm143 was cancelled before execution after the reviewer
found the project HF_HOME model directory incomplete. Existing complete local
model cache is now linked into project HF_HOME, old partial directory preserved;
actual tokenizer/config and all750 indexed safetensors parameters parse, same
Slurm native fixed5 will be resubmitted.

Independent once-only code review PASS: `docs/reviews/20261005_m1_interval_witness_code.md`.
Fixed actual overlapping-substring uniqueness and removed model-visible internal
frame paths with class-bearing video names before any scientific GPU run. Full
source/audit paths and actual image content retained. Real36layer BF16 tinyCPU
multimodal production newG/ownstance/V/S/strict binding and structured generation
proved executable; these random-model CPU fixtures are not8B observations.
