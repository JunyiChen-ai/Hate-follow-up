# M1 candidate21: complete semantic cluster tree and local evidence reading

Declared 2026-10-04 while candidate20 is running. This is a prepared independent
backup; no performance result, no GT analysis or GPU run yet. Proposal review
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
