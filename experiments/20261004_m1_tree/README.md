# M1 candidate21: complete semantic cluster tree and local evidence reading

Declared 2026-10-04 while candidate20 was running. Current branch: R1 complete333
retained for common within gains but excessive HateMM PR loss; R2 native G/S with
tree local visual is now in its complete run. Chronological GPU/analysis records
are below. Proposal review
PASS: `docs/reviews/20261004_m1_tree_proposal.md`; implementation and CPU checks
prepared after PASS, independent rule6 review PASS (below). Current formal method
remains r6_bma. Development-selected. No Explorer fifth revision:
the video-wide clustering/captioning/breadth/depth representation and new global
read are different; no Explorer score or acquisition policy is imported.

## Hypothesis and actual source scope

Distinct visual events can be omitted by uniform overview frames or interpreted
without their event context. A complete query-adaptive semantic hierarchy can
select representative observations and associate local observations with their
event context before the frozen model makes its single local decision.

Primary source: [VideoTree](https://arxiv.org/html/2405.19209v3), sections3.1–3.3,
4 implementation,9 tree-structure discussion,10 algorithm and11 implementation.
Actual read response: `runs/20261004_m1_ideation/backup_source_reads/videotree_source.json`.
The source clusters visual features, captions centroid-nearest real frames,
doubles root breadth until sufficient high-relevance clusters or a cap, expands
medium roots one level and high roots two levels, and reasons over all node
captions in chronological order. Tree construction and presenting explicit links
to the answer model are different operations; the paper does not require an
explicit tree string for reasoning. Original EVA-CLIP/captioner/LLM are replaced
by the SAME frozen Qwen3-VL-8B. This is a declared adaptation, not an exact source
reproduction. The source has benchmark-specific settings; here one fixed setting
is used for both corpora. The numeric settings below belong to this adaptation.

Target neighbors and source-read limitations are in the independent ideation
jury `docs/reviews/20261004_m1_ideation_jury.md`: MATCH already retrieves clues;
MAESTRO already performs global/local rereading; RAMF has local/global reasoning.
No claim of first retrieval, first rereading or first caption-based hate analysis.
MAESTRO official full PDF could not be accessed there; its hierarchy overlap
remains a stated uncertainty for the proposal reviewer, not presumed absence.

## Exact common R1 specification, frozen before scoring

1. Reuse original media, native20-frame overview, complete native ASR, fixed
   policy/questions and Qwen weights. Independently collect exact native global,
   stance, visual/speech window reads and native r6 for the paired reference.
   Scoring and feature extraction read no labels, GT, old prediction or metrics.
2. Decode actual video presentation timestamps with PyAV, normalized to container
   presentation origin (or first decoded frame when unavailable). Build a pool
   at targets .5,1.5,... seconds below duration, selecting the first actual frame
   at or after each target and below manifest duration. Deduplicate source frame
   indices. Every fixed8s window lacking a pool frame adds the actual decoded
   frame inside it nearest its center; earlier PTS breaks ties. No out-of-window
   substitution. A video/window without any actual decoded frame is explicitly
   recorded; it receives no invented local image. Store source indices/PTS/time
   base/input path. Native JPEGs and their legacy annotated times stay unchanged.
3. Encode pool frames with Qwen's existing image processor bounds65536–100352
   pixels, batch8, the same unique vision encoder. For each image take the FP32
   spatial mean of its final merged visual embeddings supplied to the language
   model, then L2 normalize with norm floor1e-12. No second visual model, temporal
   averaging, fit or score. Feature work and raw decoding are charged per video;
   cache reuse does not make this free for a new video.
4. Root clustering uses sklearn KMeans, random_state0,n_init10,max_iter300,
   tol1e-4, Euclidean distance on normalized features. Try k4,8,16, limited by
   the number of members. Discard empty clusters; do not duplicate identical
   clusters. Order nonempty clusters by their representative time then member
   source index. Representative is the member nearest the FP32 center, earliest
   actual time/index breaks ties. The representative is a real observed frame.
5. Caption each unique representative with the same Qwen, no ASR/global verdict
   or moderation label in this pass. Frozen system: `Describe observable visual
   evidence accurately.` User: `Describe the visible people, actions, gestures,
   symbols, and any legible on-screen text in this frame. Do not infer information
   outside the image. Use at most two sentences.` Greedy generation, maximum96
   new tokens, normal EOS. Reuse only the caption of the EXACT same source frame,
   never a caption of another frame. Preserve generated text, IDs and truncation.
6. In each breadth round, give all current representative captions, actual times,
   frozen platform policy and whole-video question to a fresh same-Qwen textual
   prefix. For each node independently ask: `Rate how informative observation
   {node_id} is for answering the question, including evidence for either answer.
   1 = not informative; 2 = somewhat informative; 3 = highly informative.
   Answer with one digit: 1, 2, or 3.` Use its last hidden row and FP32 restricted
   logits for exact single tokens1/2/3; argmax, ties prefer1 then2 then3. Crop the
   cache and restore rotary state between node queries. This conditional digit
   read is a declared interface adaptation of the paper's joint relevance prompt.
   Stop on >=2 roots with relevance3, k16, or inability to increase actual cluster
   count. Retain only LAST-round roots for the tree; rejected-round calls remain
   charged. Captions may be reused by exact frame identity across rounds.
7. A root scored1 stays a leaf. A root scored2 is split once into at most2
   subclusters. A root scored3 is split into at most2 subclusters and each child
   into at most2 grandchildren. Same KMeans/representative rules. A singleton or
   unsplittable cluster stops at its actual depth, without duplicate parent nodes.
   Root relevance determines depth; no recursive new relevance scores. Caption
   all new representative frames by step5. Save complete membership, centers,
   parent links, root scores, depth and actual frames. No node combines independent
   model predictions. Root cap16, maximum2 levels below root, branch width2.
8. Build the new global prefix from the original overview frames/full ASR/policy
   and ALL retained tree node captions in chronological representative-time order.
   Each record states node ID, parent ID (ROOT for roots), actual frame time and
   caption. Literal section header is `Tree observations (frame descriptions,
   not independent moderation decisions):` followed by a newline. Literal record
   is `[node={id}; parent={parent_or_ROOT}; depth={depth}; t={time:.3f}s]` then a
   newline, the exact caption and a newline. IDs are `r000`, `r000.c000`,
   `r000.c000.c000` in deterministic ordered clusters. Insert this section after
   the native transcript and before the unchanged native policy. Node order and
   full exact text are recorded. Keep each node record,
   including shared representative-caption reuse; no selected-summary generation.
   Ask the exact original VIDEO_QUESTION, use its native FP32 Yes/No margin and
   append its own hard answer as stance. This replaces the native global/stance
   for the new method, not a blend or an alternate-score selection.
9. For a local8s visual window, find terminal clusters with pool members inside
   that window. For each leaf choose the in-window member nearest its own center,
   actual earlier-time/index ties. Rank leaves by descending root relevance,
   then candidate distance, then candidate time/index and leaf ID. Take up to2
   distinct actual frames; if none exist use no added image. The pool coverage
   rule in step2 supplies a real member when decoded coverage exists. The local
   packet contains these images with actual times and each selected leaf's
   root-to-parent records, marked as interpretation context and with their actual
   representative times. Duplicate ancestor records are listed once. The local
   evidence instruction refers to the selected images in this window; ancestors
   outside it are context, not occurrence evidence. This is instruction-level
   scope, not hard information isolation: the full native overview, complete ASR
   and all global tree captions remain visible in the shared prefix. Literal
   packet header: `Interpretation context for the selected in-window images.
   Context descriptions may show frames outside this window; their times remain
   explicit. They do not establish an occurrence inside the window:` then newline.
   Use the same literal node record format; deduplicate ancestors by node ID in
   selected-image chronological order, root to parent. Then literal header
   `Local observed frames:` and newline; each image is preceded by
   `[t={actual_time:.3f}s]` and newline, in chronological time/index order. If no
   ancestor or local frame, the corresponding section contains `(none)`.
   Append packet + the EXACT original
   visual yesno_question after the new stance-conditioned global prefix. Read
   once, crop/restore independent cache and multimodal rotary state. Every speech
   window is read independently with the EXACT native question/body on the same
   new global context, only when the native body is nonempty. No speech-image
   joint branch, previous local answer or history-dependent window processing.
10. Raw window score is max(new visual,new speech), missing speech -> visual.
    Global/new visual/new speech replace the three native values directly. Use
    the unchanged r6 script through subprocess and canonical4fps evaluator.
    Dataset flow and every constant are identical. Seed0, greedy captions,
    no label fitting/threshold and no score/feature ensemble. No scan declared.

Native overview and original ASR preprocessing are common deployment costs and
reported separately. Added pool preprocessing, every rejected breadth round,
caption token/vision/LM forward, relevance prefix/query, new global prefix and
local image reencoding are included in new-method cost. FP32 features/metadata
and generated observations belong in `data/semantic_cluster_tree/` with
PROVENANCE; scoring treats them read-only. Scores/logs/details/config/PID belong
in `runs/20261004_m1_tree/`. No content hashes. Long tasks use lab Slurm.

## Cost, validation and controls before any performance claim

Manifest-only planning: HateMM29268.55s/215videos, HCS28264.56s/118videos;
approximately57698 1fps feature images, plus sparse coverage repairs. Each video
has <=112 final tree nodes (16 roots +32children +64grandchildren), up to28
root candidates across breadth rounds; unique caption frames may overlap.
Every generated caption costs 1 vision/prefix + up to97 decoder forwards; every
relevance round costs1 prefix + its actual root count queries. New moderation
read costs3+W+available_speech branches, plus up to2W vision images. Paired native
collection is an additional reference cost, not deployment cost. Estimate60–180
GPUmin on5090 for333; unmeasured. Fixed5-video noGT smoke (first2 per corpus plus
HateMM hate_video_114) verifies costs/memory/native parity, never selects constants.
Longest actual prefix is checked; memory changes cannot silently change inputs.

Required implementation checks: actual PyAV PTS/index and in-window pixels;
processor bounds/image embedding splits; complete membership/centroid/nearest
representatives including identical/singleton features; fixed deterministic
breadth/depth and reuse by source index; generated captions from true witnesses;
single-digit FP32 relevance and independent queries; caption/parent/local frame
alignment; new global and local values actually enter final scores; no GT path;
all native333 values exact; cache/input versions and complete333 identity/shape;
appended-image positions and every branch cache/rope restored; all actual costs;
canonical evaluator and unchanged4fps protocol. Independent rule6 review required.

Main gate: same main metric improves>=.01 BOTH corpora vs current/native r6;
pooled losses<=.005, within losses<=.01. Report all6 numbers,84/99 within videos,
raw visual/max/speech ordering and paired video bootstrap. No anymain+.01 ->
archive; a qualifying single-corpus gain allows at most3 declared revisions.

If main passes, run full333 controls, with exact matched native references:

- `flat`: exact main nodes/captions/local pixels/budgets, remove parent-link and
  ancestor association from the reader but preserve the same observation texts.
  This tests the NEW reading association, not the already-completed acquisition.
  No source claim that explicit tree strings are necessary. Ancestor association
  is only claimed if dual-corpus same-main loss>=.01; otherwise delete/demote it.
- `temporal`: match main actual root breadth and each root's child/grandchild
  counts, partition pool in contiguous equal-count temporal groups, choose real
  middle members and caption them with the same model/budget. Relevance scores
  from the main determine matched branching ONLY for this diagnostic. This is
  not a deployable algorithm; compare complete acquisition with equal node/call
  opportunity. Preserve actual costs; do not equate equal calls with equal seconds.
- `no_depth`: same final root nodes/relevance/captions and new global/read protocol,
  omit descendants and select local frames by their root center. Tests hierarchical
  acquisition separately from a relevance-selected root caption inventory.
- `wrong_links`: same acquired nodes/captions/local pixels, rotate root-to-leaf
  ancestry packets by floor(number_of_terminal_leaves/2), leaving actual local
  source times and pixel selection fixed. One leaf -> no-op, reported explicitly.
  Reused wrong contexts retain their OWN true caption/time, not fabricated source
  times. This falsifies claimed ancestor interpretation, not extra pixel benefit.
- `no_added_pixels`: identical tree/captions/context but visual window uses only
  original overview observations, testing actual leaf evidence contribution.

Whole acquisition must beat matched temporal acquisition by>=.01 on the SAME
main metric for BOTH corpora before calling the semantic hierarchy effective.
Each separately claimed breadth/depth/reading component must independently meet
rule14g; run additional no_breadth (k4, same subsequent depth rules) if breadth
is separately claimed. Failure causes deletion/demotion or a scoped negative
conclusion, never an untested mechanism assertion. Maintain paired raw ordering
analysis and falsifiable context binding; final independent source/novelty and
mechanism review required. No Overleaf or user notification from this experiment.

## Implementation preparation (not a result)

Implemented `tree.py`/`extract.py` acquisition and `measure.py` paired read;
`analyze.py` only orchestrates the canonical evaluator/fixed r6 and postscore
analysis. Three meaningful CPU checks PASS, source
`runs/20261004_m1_tree/cpu_checks/selfcheck.log`: independent centroid/partition
and leaf/ancestor oracle, complete breadth/depth plus identical/singleton cases,
and actual PyAV encoded-video PTS sampling/seek plus tiny-window coverage repair.
Python compilation and launch shell syntax pass. No runtime estimates have been
replaced by measured GPU cost, no actual model parity or semantic mechanism is
claimed. Independent rule6 code review PASS:
`docs/reviews/20261004_m1_tree_code.md`. Fixed the single observed missing `os`
import; reviewer independently confirmed target HF5.15 increment positions and
image feature API, plus offline real tokenizer digit/FP32 read and CPU cases.
Evidence `runs/20261004_m1_tree/code_review/`. Actual native/append-image parity
remains for GPU smoke. Run only after candidate20 branching.

Initial sc474398/Slurm102 exited before media processing because the first raw
HateMM video was unavailable there. This operational input failure is not a method verdict or a
revision. Logs returned locally; use sc474399 (uoa-lab2), which has both complete
raw corpora. Full333 noGT media preflight recorded before resubmission:
`runs/20261004_m1_tree/cpu_checks/media_preflight_lab2.json`.
Independent host-interface confirmation PASS:
`docs/reviews/20261004_m1_tree_host_interface.md`; actual lab2/lab3 HF5.15.1
image output and incremental position interfaces agree by direct source reading.
`sbatch experiments/20261004_m1_tree/launch/lab2.sbatch smoke` then `main`.
After each remote job, return both this run and `data/semantic_cluster_tree/`
with ordinary rsync before local noGT prepare; full scores then run
`setsid nohup bash experiments/20261004_m1_tree/launch/run_analysis.sh` locally.

sc474399/Slurm103 passed media access and captioning, then failed in the first
pure-text relevance round: target HF5.15 leaves `rope_deltas=None` for text-only
prefixes. Fixed snapshot/restore to preserve None (ordinary text positions follow
KV length), rather than unconditionally cloning it. No video tree/score was
completed; no input version, constant or scientific design changed. Partial
witnesses and all failure logs returned locally. This is an implementation repair,
not an idea verdict or a counted revision. Independent narrow confirmation PASS:
`docs/reviews/20261004_m1_tree_text_rope_fix.md`; actual target source and CPU
None/tensor normal/exception cache checks agree. Retry the same fixed five.

## Fixed five-video smoke (no performance selection)

sc474399/Slurm104 completed both acquisition and paired reads; run and all new
inputs returned to this machine before CPU prepare. NoGT prepare PASS:
`runs/20261004_m1_tree/r1_full_smoke_analysis/plumbing_summary.json`.
All five native global/window/frame curves match the original reader exactly;
five fresh whole-render token/image-grid checks and cached prefix-position checks
pass. New global changes in5/5, visual in158 windows, speech in134. This establishes
that the intervention is executed, not that it improves performance or mechanism.
Peak18.88/18.84GiB (HateMM/HCS), including actual extraction; five-video new
processing166.97s versus native12.52s. Caption and rejected breadth costs are
included. First-two-per-corpus extrapolation is about79.20/57.52min, unvalidated
for complete333. No constant was selected from this check. Continue the fixed
complete run with identical inputs/settings.

## R1 complete333 result and branch (2026-10-05)

sc474399/Slurm105 DONE; all run artifacts and acquisition inputs returned to
sc474397 before canonical CPU analysis. Full native global/windows/curves and
all six final metrics reproduce exactly. Source:
`runs/20261004_m1_tree/r1_full_main_analysis/{alignment,summary}.json`.

| Final4fps/fixedr6 | ROC | PR | within (eligible) |
|---|---:|---:|---:|
| HateMM | .8954106478490577 | .6793293896829375 | .7796925196653864 (84) |
| HateClipSeg | .7299414573719176 | .6898192706910828 | .650288654397814 (99) |

Authoritative final numbers:
`runs/20261004_m1_tree/r1_full_main_decoded/optimized/metrics.json`;
native `.../base/metrics.json` matches current r6. All development-selected.
Final within gains+.02891048/+.01293954; HateMM PR−.01490521 exceeds noise.
Therefore performance gate FAIL, any_qualifying_gain TRUE: retain this family
for at most3 result-guided revisions. No promotion, full controls not run.
Raw max within−.02173608/−.00180368, raw visual−.01909018/+.02071501,
shared speech−.00111867/+.00404552; final within is not evidence of a dual-corpus
raw ordering improvement. mechanism_supported remains false. Paired final within
CI95 [.000724,.058337]/[−.014568,.040819], not blind confirmation.

New-video processing: HateMM76.244min, HCS62.255min, total138.499min/native9.195min
(15.06x); acquisition65.102/51.973min included. Peaks19.119/19.110GiB;
4210/3091 caption generations and actual forward/image accounting are in alignment.
Reusing these inputs for revisions does not remove their new-video acquisition cost.

Test access for R1 analysis: the two canonical `data/gt_4fps/*.npz`, R1 native/new
raw and decoded predictions/metrics, plus per_video/summary/alignment in the R1
analysis directory. Findings above motivate separating the enriched global reading
from the native whole-video anchor; no test labels enter later scoring/fitting.

## R2 declaration: native anchor/speech, tree local visual (revision1/3)

Declared2026-10-05 after R1 complete results, before R2 code/GPU. Postscore files
read: R1 raw/decoded native/new predictions, R1 per_video.json, and the two test
GT arrays; descriptive artifact `runs/20261004_m1_tree/r1_error_analysis/{summary,per_video}.json`.
HateMM positive-frame-weighted global delta−1.1695, mean raw-max delta+.4348,
K delta−.7347. Global decreased for26315/28319 positive frames' parent videos.
Positive/negative group shifts differ; these are not causal attribution or proof
that native G alone repairs PR. Raw max ordering also fell. This motivates isolating
the visual acquisition/local read instead of modifying whole-video/speech context.

One identical pipeline on both corpora: retain exact native20-frame/fullASR prefix,
one native whole-video Yes/No margin and own hard stance; retain native independent
speech margin. Replace only the visual branch with actual R1 selected in-window
leaf pixels plus exact ancestor descriptions. No full-tree observations in the
global/speech prefix, no second whole-video decision, no scalar replacement after
scoring, no averaging or blending. Tree acquisition/packets, selection, all literal
local headers/questions/pixel settings and numerical constants unchanged from R1.
Global/tree-prefix and speech-context changes are removed from the novelty claim.
The retained claim is semantic breadth/depth acquisition with local leaf regrounding;
raw ordering, matched temporal/flat/incorrect-link controls still required.

The same native stance cache is cropped/restored after every reference/new visual
and shared speech read. New S and G must equal paired native exactly. Fixed five
noGT smoke first, fresh full-render input/grid/positions plus cloned visual cache
replay; then full333 and canonical/fixedr6 evaluation. No GT in either reader.
Production Qwen moderation calls3+W+available_speech, including the single shared
prefix/global/answer. Paired native reference adds W visual calls; smoke adds one
cloned visual diagnostic/video, separately counted/timed. New-video cost includes
native prefix+speech+new visual AND all original tree acquisition117.075min;
no new feature/caption calls when reusing this input. Preliminary total125-145min,
unmeasured until smoke/full. Acquisition reuse is an iteration saving, not free
new-video processing. Outputs `r2_full_{smoke,main}` and their decoded/analysis dirs.
This is the first of the three allowed revisions, not a new candidate/queue reset.

R2 implementation/CPU checks: `measure.py --revision r2` builds one native stance
cache, interleaves paired native visual/shared speech/new visual, restores each
branch, and independently clones the first new visual branch in smoke. Production
and diagnostic forwards/times are recorded separately; whole input acquisition
remains in new-video cost. CPU orchestration fixture and validator corruption
checks PASS in `runs/20261004_m1_tree/cpu_checks/r2_selfcheck.log`; these are not
actual GPU model parity. Default R1 CLI/config/record interpretations remain intact.
R2 input acquisition is reused, not rerun; exact native input/metadata coverage is
still parsed before every measurement. Launch after narrow independent code
confirmation and current machine/code synchronization:

```bash
sbatch experiments/20261004_m1_tree/launch/lab2.sbatch smoke r2
# Return runs and data to local, then noGT prepare first:
/home/jehc223/miniconda3/envs/HateVideo/bin/python experiments/20261004_m1_tree/analyze.py --stage prepare --smoke --revision r2
# Only after five-video parity/clone/fresh checks PASS:
sbatch experiments/20261004_m1_tree/launch/lab2.sbatch main r2
# Return runs/data; detach canonical CPU analysis on sc474397:
setsid nohup bash experiments/20261004_m1_tree/launch/run_analysis.sh r2 > runs/20261004_m1_tree/r2_analysis_launcher.log 2>&1 < /dev/null &
```

Independent R2 narrow code confirmation PASS:
`docs/reviews/20261005_m1_tree_r2_code.md`; CPU evidence
`runs/20261004_m1_tree/r2_code_review/{independent_r2.py,independent_r2.log}`.
No observation-blocking bug; shared native context/production standard branches,
G/S equality, actual call/cost separation and corruption rejection checked.
The image/context forward is a CPU stub here; actual fixed-five GPU parity remains
required. Full333 R1 noGT prepare regression also PASS in
`runs/20261004_m1_tree/cpu_checks/r1_prepare_regression.log`.

R2 fixed5 GPU smoke sc474399/Slurm115 DONE and run/input returned locally before
noGT prepare: native full G/windows/curves exact, G/S unchanged,158 visual windows
changed,5 fresh full-input/grid/prefix-position and5 cloned-margin checks PASS.
Source `runs/20261004_m1_tree/r2_full_smoke_analysis/plumbing_summary.json`.
Peaks18.877/18.842GiB. Whole new-video sample costs104.387s HMM/56.796s HCS,
including88.474s/48.860s acquisition; reference native8.422s/4.071s.
Rough extrapolation78.148/55.850min (~134.00min total), not full measured cost.
Only prefix/local read is rerun using existing inputs. Proceed with fixed complete333;
no smoke GT or performance/constant selection.

## R2 controls scope, declared while main is running

These controls run only after a complete R2 performance pass. They all use the
same native G/own stance/shared S and unchanged r6, never R1 tree-global context.
One full333 control run also rereads R2 main and native for exact raw parity.
GT is read only after every arm has finished and been returned locally.

`flat` retains the exact current-window ancestor caption/time inventory and local
pixels, lists observations chronologically without node/parent/depth labels. It
tests explicit reading-link representation only: it does NOT remove the semantic
acquisition or which ancestor observations were selected. No broader claim that
this control removes all selection/association. `wrong_links` independently tests
association: keep exact main local pixels; rotate terminal-leaf ancestor chains
by floor(leaf_count/2), preserve all actual source times/text, deduplicate as main.
Singleton/no-change cases are reported. `no_added_pixels` retains the exact main
ancestor packet but removes new local images. `no_depth` retains final roots,
their actual captions/relevance/features, removes descendants, and applies the
same center-nearest local rule. Any extra witness is decoded into a separate
control input cache, never by modifying the scored main input.

R2 `temporal` matches the full final topology AND each node's actual member count:
assign all pool indices in time order to roots with the main root member counts,
recursively to children with each main child's member count. This replaces the
earlier R1 equal-count temporal-group plan, which can fail to reproduce a small
group's declared child counts. Actual counts/topology and all partition coverage
must match exactly, with no duplicate or invented membership. Choose each temporal
node's middle real member (index len(group)//2), compute its actual feature mean,
retain corresponding main root relevance for diagnostic depth matching AND the
unchanged local selection priority (descending root relevance before distance), and
caption with the identical frozen model/prompt/96-token cap. Local selection uses
the same in-window center-nearest rule. This diagnostic is not deployable because
it consumes main topology; it isolates semantic membership from chronological
membership with the same node opportunities and frozen main per-root priorities,
not a claim of equal measured time
or necessarily equal unique-caption calls. Reuse only exact same source-frame
captions; all actual unique/reused caption forwards/tokens and added witnesses are
reported. Whole new-video costs remain charged. All literal headers/constants
otherwise remain as declared; no scan or result-driven control selection.

Scope limitation: inherited root priorities describe the original semantic roots,
not freshly judged temporal roots. Their effect on local ranking is intentionally
held fixed along with topology/counts; this control cannot establish superiority
over a deployable temporal algorithm that recalculates its own priorities/depth.
The R2 specification explicitly replaces R1's "main relevance for branching only"
wording; input construction alone cannot verify a stronger end-to-end competitor.

For each claimed component, require a same-main decline>=.01 in BOTH corpora;
otherwise delete/demote that component and evaluate any changed final code fully.
Complete acquisition must beat the temporal control on the same main metric by
>=.01 BOTH; all raw ordering/incorrect-link effects and eligible-video bootstrap
are reported. A flat/no-pixels control cannot establish acquisition effectiveness
alone. These declarations do not imply mechanism support or a main result.

Pure control geometry CPU implementation and complete333 input preflight PASS:
`runs/20261004_m1_tree/cpu_checks/r2_control_geometry.log` and
`runs/20261004_m1_tree/cpu_control_preflight/{summary,per_video}.json`.
No control captions, predictions, GT access or GPU. Placeholder strings were
transient geometry fixtures only, discarded; temporal changed-window counts refer
to selected indices/links, not ungenerated observations. Additional actual caption
and witness opportunities are reported, not inferred actual processing seconds.
Independent narrow scope PASS: `docs/reviews/20261005_m1_tree_r2_controls_geometry.md`.
The not-yet-implemented control acquisition/reader still requires its own narrow
code confirmation before running.

Waiting-time source scope check: DSTA official MAESTRO indexed appendix confirms
tool/chunk selection and iterative local/global reasoning; full report/poster
still403, no claim of excluding all clustering overlap. Evidence and exact query
scope `runs/20261004_m1_tree/r2_source_scope/`. No scoring/constants change and
no reopened proposal review. No first generic retrieval/tool/rereading claim.

## R2 complete333 and branch (2026-10-05)

sc474399/Slurm116 DONE333; outputs/inputs local before canonical CPU analysis.
All native raw reads/curves and six final metrics exact; G/S unchanged,7359 visual
windows changed. `runs/20261004_m1_tree/r2_full_main_analysis/{alignment,summary}.json`.
Authoritative final `runs/20261004_m1_tree/r2_full_main_decoded/optimized/metrics.json`:
HateMM ROC/PR/within .8975298327452879/.6906542914243219/.7725214943334161 (84),
HCS .7296312611274366/.6829391036197541/.6425199749331614 (99).
Delta+.000411/−.003580/+.021739;+.012806/+.011867/+.005171 respectively.
All losses inside noise, but no same metric+.01 BOTH; performance FAIL,
any_qualifying_gain TRUE. Retain for second revision,2/3 revisions remain.
Raw max within−.027095/+.009278, raw visual−.049536/+.022907,
shared speech exactly0. No dual-corpus raw improvement or mechanism claim.
Paired final CI95[−.000185,.046749]/[−.017174,.027705]. No full controls run.
New-video processing74.216min HMM/60.187min HCS=134.404min/native9.190min (~14.62x),
including117.075min original acquisition. Peak19.119/19.110GiB.

R2 descriptive test access: current full raw/decoded predictions, per_video.json,
both canonical test GT arrays; `runs/20261004_m1_tree/r2_error_analysis/{summary,per_video}.json`.
G and S stayed exactly native; HCS visual raw ordering improved but raw max/final
within improvement is smaller. HMM visual ordering fell despite final within
improvement; no claim of new evidence localization established. R1's full tree
context gave different raw visual/speech behavior; comparison also changes stance
and G, so it is not proof that one context factor alone caused the difference.

## R3 declaration: native anchor, factual tree context before local reads

Declared2026-10-05 after full R2 analysis, before R3 code/GPU; second revision2/3,
one remaining revision after it. Source/error files above and R1 analysis are the
only result inputs to design; scoring reads none of them or GT. No constants scan.
Hypothesis: complete chronological tree observations can aid both local branches
without changing the original whole-video anchor/own hard stance. This keeps the
complete acquisition and observation inventory, not an alternate-score mixture.

Build exact native20/fullASR prefix, ask original VIDEO_QUESTION once and append
its native hard Yes/No. Paired native V/S are read independently on this cache.
Then append one factual user turn containing the EXACT existing tree_text (same
chronological nodes/captions/actual source times/links), followed by the literal
non-decision assistant acknowledgement `Context recorded.`. Cache this extension
in one text-only forward, without a moderation question/global reread. Its global
margin/stance remain the original measured values; no overwriting a different
judgment or combining two scores. Reconstruct actual IDs/grids/3D positions and
verify the original native prefix positions remain exact. This shared factual
extension is visible to all later independent local branches.

New visual uses the exact existing local leaf pixels/ancestor packet and original
visual question; new speech uses the exact original speech question/body on the
extended cache. Empty native speech stays absent. New S and V both replace their
native reads directly; max and fixed r6 remain untouched. G is native; S is now
new, rather than copied native. Full overview/ASR/all tree observations remain
visible: no hard evidence isolation claim. All acquisition/caption/local
constants, policy, seeds and prompts unchanged; the exact acknowledgement above
is cache/conversation bookkeeping and is not a novelty component.

Production moderation calls4+W+available_S (native3 plus1 factual extension);
paired reference adds W+available_S calls. Fixed5 smoke adds one cloned new V
and one cloned new S when speech exists, separately counted/timed. Fresh render
and native prefix geometry verified both at factual extension and appended local
pixels, clone margins exact, then the same full333. All input acquisition117.075min
remains charged per new video; no new captions/features for these existing inputs.
Preliminary full135–150GPUmin, unmeasured until smoke/main. Outputs
`r3_full_{smoke,main}` with matching analysis/decoded directories.

If R3 passes, full controls must be adapted to this declared shared factual
extension/new S, with exact native/R3 raw parity and same component/falsifiable
gates; the R2 native-S control reader is not a substitute. Pure geometry remains
reusable. No mechanism/promotion conclusion from current positive R1/R2 alone.

R3 independent narrow code confirmation PASS:
`docs/reviews/20261005_m1_tree_r3_code.md`. CPU evidence
`runs/20261004_m1_tree/r3_code_review/{extension_oracle,binding_oracle}.{py,log}`.
Detected saved extension token/role binding gap was fixed before any GPU run:
resume and prepare rebuild the current native frames/ASR/stance conversation and
literal factual extension, distinguish native prefix P from stance-cache N, and
compare suffix text/IDs and full lengths. Actual Qwen CPU processor validated eight
corruption rejections; stub cache oracles verify calls/restoration/cost/empty speech.
These are not GPU model parity. Existing CPU selfcheck and R2 full noGT prepare
regression PASS in `runs/20261004_m1_tree/cpu_checks/r3_existing_regression.log`
and `r2_prepare_after_r3.log`. Shared CPU renderer is in `src/mllm_renderer.py`;
the prepared Program helper now imports it with equivalent rendering semantics.
No Program scoring or GPU run was performed for this extraction.

Conditional fixed5 then main commands: `sbatch .../launch/lab2.sbatch smoke r3`,
return runs/data, `analyze.py --stage prepare --smoke --revision r3`; after PASS
`sbatch .../launch/lab2.sbatch main r3`, return runs/data and detach
`launch/run_analysis.sh r3`. All commands run from project root on correct host.

R3 fixed5 sc474399/Slurm117 DONE, all outputs/inputs returned locally before
noGT prepare PASS. Native full reads/curves exact, G unchanged;158 V/134 S windows
changed. Extension fresh and local fresh checks5 each, V and S cloned margins5
each exact; current native conversation/suffix token binding passes for all five.
Source `runs/20261004_m1_tree/r3_full_smoke_analysis/plumbing_summary.json`.
Peaks19.314/18.842GiB, prefix max9667/6393tokens; optimized whole sample109.149s
HMM/58.757s HCS including88.474/48.860s acquisition; native8.405/4.068s.
Rough extrapolation80.109/57.778min=137.886min, not full measured cost.
Continue identical full333. No smoke GT, metric-based selection or constant change.

## R3 full controls specification, before control code/GPU

Only run after R3 complete performance pass. Independent-arm inventory is main,
flat,wrong_links,no_depth,no_added_pixels,temporal,temporal_fresh_priority, plus
native. Main/native must exactly match complete R3/current r6 before any control
GT analysis. Shared native prefix/G/own stance and reference V/S are measured once;
each arm clones this restored native cache, appends its factual observation turn
and the SAME acknowledgement, independently reads V/S, then deletes its cache.
Production4+W+S per arm; joint actual3+W+S+A*(1+W+S), plus separately recorded smoke
clones. Native-cache copying is experimental duplication overhead, not hidden
new-video acquisition. All input acquisition and actual arm times are reported.

flat factual inventory uses TREE_HEADER then each chronological node's
`[t={time:.3f}s]` newline exact caption newline, without node/parent/depth labels;
local packet uses the declared flat transform. wrong_links changes LOCAL ancestor
association only; shared full inventory remains correct, so this control cannot
test necessity of all global association or prove the model cannot recover it.
no_depth factual inventory contains final original roots only and local pixels
are root-selected. no_added_pixels factual inventory/local ancestor context remain
exact main, only added local images removed. Native overview pixels remain present.
temporal uses the declared exact-count/topology time-membership construction and
inherited main priorities for local selection, with accurate scope limitation.

temporal_fresh_priority strengthens that diagnostic: same temporal nodes/captions/
exact topology/member counts, but independently apply the original same-Qwen
relevance reader to the TEMPORAL root observations once, use those root relevance
values for local ranking and propagate to children. Actual branch counts remain
matched to main, rather than expand again. No shared/averaged moderation decision
or new label/model. Both variants remain topology-matched diagnostics; no claim
of reproducing a fully independently adaptive deployable temporal method. This
additional control addresses the inherited-priority confound, and complete
acquisition must beat BOTH temporal variants by>=.01 on a common main metric BOTH
corpora. No result-guided selection between the two.

Reuse main features and exact same source-frame captions. Newly required temporal
captions and local witnesses live in `data/semantic_tree_controls/r3/` with readable
provenance, never alter `data/semantic_cluster_tree/`. Existing exact witness PNGs
may be linked relatively; otherwise decode the actual indexed PTS into this new
cache. No CPU placeholder becomes an observation. Additional actual caption
calls/tokens/time and fresh-priority prefix/query calls are counted. Full333 CPU
geometry opportunity counts predict6018 new caption frames and6926 additional
local witnesses, not measured GPU output. Approximate extra acquisition80–120min,
seven-arm local reads130–180min plus diagnostics; total210–300GPUmin, unmeasured.
Use fixed5 noGT checks first, then full333 only after independent narrow code
confirmation. Same constants/prompts/seed/window/grid/frozen models across corpora.
Every claimed part and incorrect-link/raw ordering/paired bootstrap still obeys
the existing mechanism gates; format details without contribution are demoted.

Prepared R3 controls code is in `control_inputs.py`, `control_extract.py`,
`control_measure.py`, `control_analyze.py`, `launch/lab2_controls.sbatch` and
`launch/run_control_analysis.sh`. Independent narrow code confirmation PASS:
`docs/reviews/20261005_m1_tree_r3_controls_code.md`, actual independent CPU evidence
`runs/20261004_m1_tree/r3_controls_code_review/`. No GPU or control observations
generated. Fixed missing actual-forward counter initialization, added per-arm
actual new source-witness decode cost (including no_depth), bound each new caption
text to decoded generation tokens and opened its representative image, and report
actual incorrect-link changes separately from prediction changes. Joint physical
acquisition/reader/copy/diagnostic cost is distinct from per-arm reused prefixes.
The Slurm launcher requires the completed R3 main `performance_pass=True` summary;
this guard forbids the current failed R3 from running controls. Current native CPU
image-prefix encoding is reused only for identical messages/files within a video;
all GPU branches remain independent. This is prepared code, not mechanism evidence.

## R3 complete result and recorded error analysis

sc474399/Slurm118 completed all333 at2026-10-05 04:01NZDT; run and source inputs
returned locally before canonical prepare/evaluation. Native G/windows/curves and
all six final metrics exactly reproduce the current method. G unchanged;7359 V
and6580 S windows changed. Authority:
`runs/20261004_m1_tree/r3_full_main_decoded/optimized/metrics.json`.

| Dataset | pooled ROC | pooled PR | within (eligible) |
|---|---:|---:|---:|
| HateMM | .8934449148927398 | .6755683644708480 | .7612246847768677 (84) |
| HateClipSeg | .7166345686592962 | .6703579776980031 | .6337975399819128 (99) |

Deltas vs fixed current method HMM−.003673747/−.018666239/+.010442641;
HCS−.000190295/−.000714107/−.003551574. No common+.01; HMM PR beyond noise:
performance FAIL, any qualifying gain TRUE. This is revision2/3, so one revision
remains; no promotion and no control GPU. Raw max within−.02442806/+.00158927,
visual−.06037818/+.02376880, shared speech−.00195528/+.00406198. Final paired within
CI95[−.019960,.043817]/[−.030969,.023454]. Mechanism unsupported; all development-selected.

Actual new-video processing76.48405min HMM/62.46508min HCS=138.94913min,
native9.18662min (~15.13x), with117.07481min original acquisition included.
Peaks19.11876/19.10972GiB; prefix max9667/10170; actual paired moderation forwards
15274/13936, diagnostics0. Details
`runs/20261004_m1_tree/r3_full_main_analysis/{alignment,summary,per_video}.json`.

Postscore testGT diagnostic actually read R3 raw/decoded base+optimized predictions,
the canonical per_video report and both current4fps GT files, saved in
`runs/20261004_m1_tree/r3_error_analysis/{summary,per_video}.json`. G delta/stance
flips0; positive-frame-weighted mean raw/K delta HMM−1.38752 vs negative-frame
weighted+1.59279, HCS−.14980 vs+1.21101. This is descriptive, not a sole-cause claim.
Additional recorded examples
`runs/20261004_m1_tree/r3_error_analysis/window_examples.json` inspected actual R2/R3
raw reads, current tree nodes/packets, ASR and GT in HMM303/428/189 and HCS
yt_rOPzqNGEvT4/yt_ksQif48f7Jw. R3 complete visual-text extension changes speech
readout even on game/static-background hate-speech clips; e.g HMM303's144–152s
native S16.50062→13.10628 while V4.33819→−3.45207. Current hierarchy context can
come from distant visually similar frames (true times retained). These observations
motivate the last declared R4, not a label-conditioned scoring or constant scan.

## R4 last revision, declared before code/GPU

Third and FINAL result-guided revision in family21. Keep the original complete
VideoTree acquisition, source inputs and every constant. Native20/fullASR G and
own hard stance remain exact; do NOT insert R3's shared full-tree factual turn.
Independently re-read EACH local branch from the original restored native cache:
V with the existing actual selected leaf pixels/ancestor context and exact native
visual question; S, when native ASR body is nonempty, with the SAME local packet
and actual pixels followed by the exact native speech question. V and S never
see each other's answer or cache. All original overview/full ASR remains visible.
This tests whether providing the actual window-associated evidence to both reads
can improve interpretation while avoiding R3's shared full-video caption inventory.
No calibrated or averaged old logits, no per-corpus routing, no label input.

New V must EXACTLY reproduce complete R2 new V in noGT prepare, alongside exact
all333 native reads. New S is measured, not replayed. The performance hypothesis
is that local grounding is less disruptive to speech evidence than R3's shared
inventory while retaining R2 visual gains; it may instead weaken valid speech,
which the same full gate will decide. Production3+W+available_S; paired reference
addsW+available_S; fixed5 adds one cloned V and one cloned available S, both fresh
rendered from actual images and independent caches. Prefix/crop/rope, current
input binding, window/pixel coverage and all physical calls/cost checks required.

Reuse original117.07481min acquisition for current videos, charged for new videos;
new S re-encodes up to2 actual local images per available speech window. No new
model or source-caption call. Preliminary140–160GPUmin full333 including acquisition,
unmeasured; diagnostic clones separately timed. Only fixed5 noGT smoke after narrow
independent code confirmation, then identical full333. If it fails, archive this
family with best retained numbers; no R5. If it passes, controls must use this
actual local-packet-to-BOTH-branches reader, not the prepared R3 extension reader,
and retain the same temporal/fresh-priority/raw/incorrect-link/ablation gates.

R4 CPU implementation preparation: existing five meaningful selfchecks PASS in
`runs/20261004_m1_tree/cpu_checks/r4_existing_regression.log`; actual R2 fixed5
noGT prepare after shared local-content extraction PASS in
`r2_smoke_prepare_after_r4.log`. Independent R4 CPU oracle
`runs/20261004_m1_tree/r4_code_review/{oracle.py,oracle.log}` uses real cached
Qwen processor/native and witness images, explicit synthetic model/positions,
six smoke/main speech-availability cases, independently recomputed R2 V equality,
exception crop/rope recovery and twelve current input/token corruption rejections.
This is CPU evidence only; actual GPU/native/R2 parity remains pending.

Independent R4 narrow code confirmation PASS, same-family provisional:
`docs/reviews/20261005_m1_tree_r4_code.md`. No observation-validity blocker remains;
fixed5 actual GPU native/R2 V/fresh/cloned margins remains required before full333.

R4 fixed5 sc474399/Slurm119 DONE, outputs and source inputs returned locally
before noGT prepare PASS. Native all reads/curves exact; new V exactly equals R2
new V, G unchanged;158 V/134 S windows changed vs native. Actual V/S fresh checks5
each, cloned margins5 each, current native/local image/token binding all pass.
Authority `runs/20261004_m1_tree/r4_full_smoke_analysis/plumbing_summary.json`.
Peak18.87652/18.84197GiB; native prefix max5829/3083; new-video sample total
111.17696s HMM/60.00556s HCS incl88.47373/48.85964s original input acquisition.
Reference8.41256/4.07185s; actual paired forwards375/234 incldiagnostics6/4,
diagnostic seconds.67100/.43647. V images170/99; S images149/78, actual new S
read9.99192/4.61175s. Rough full79.20228/59.00547min=138.20775min, not actual
full cost or performance. Continue identical full333; no GT or parameter change.

## R4 controls adaptation, before control code/GPU or main result

Same seven independent arms and exact temporal/fresh-priority topology controls
as declared above, but no arm appends any shared tree observation/ack turn.
Each arm clones the same restored native G/stance cache and independently reads
its packet into BOTH V and available S with their exact native questions. The
stored arm-wide factual observation field is unused in R4. flat changes only
local ancestor formatting/inventory order; wrong_links changes the local ancestor
association in both branches, retaining actual local pixels and original full
overview/ASR. Correct global tree links are not appended, but the native video
context remains visible; do not claim absence of recoverable global information.
no_depth/no_added_pixels and two matched temporal variants use their declared
actual packets in both branches. All seven moderation reads are freshly measured.

Production3+W+S, actual joint3+W+S+A*(W+S)+separately measured smoke clones.
Each arm has actual first V/first available S full-render and clone checks; no
extension check exists. Complete control main/native must exactly reproduce R4,
and corresponding current V/S suffix tokens/source files must bind to that arm's
packet. Acquisition machinery/prompts are unchanged, isolated inputs now in
`data/semantic_tree_controls/r4/`, version of the unchanged input transformation
retained with explicit control_run_revision=r4. Costs include actual extra temporal
captions/fresh priorities and per-arm required new witnesses including no_depth;
joint physical source work/copies/diagnostics separate. Preliminary additional
acquisition80–120min and seven-arm reads150–220min, total230–340GPUmin, unmeasured.
Existing common dual-corpus performance/acquisition-vs-BOTH-temporal/part necessity,
actual incorrect-binding coverage, raw ordering and paired CI gates remain fixed.
If R4 main passes, narrow independent review then fixed5 noGT controls/full333;
launcher requires R4 complete performance_pass=True. Otherwise no control GPU.

R4 controls narrow independent code confirmation PASS, same-family provisional:
`docs/reviews/20261005_m1_tree_r4_controls_code.md`, actual CPU evidence in
`runs/20261004_m1_tree/r4_controls_code_review/`. Seven-arm six availability/smoke
cases with actual cached processor/images and explicit model/position stubs,
nine corruption rejections, exception crop/rope restoration, per-arm V/S image
and physical calls/cost, revision paths/guard and default R3 compatibility PASS.
No GT, saved prediction or real summary read by reviewer; no control GPU/input
generation. This does not establish actual control model parity or mechanism.
