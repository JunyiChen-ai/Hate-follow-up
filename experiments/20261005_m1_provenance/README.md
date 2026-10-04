# M1 candidate25: provenance-bound temporal entity and discourse graph

Declared2026-10-05 while candidate23/24 complete333 are running. This is rank7
of the preserved nine-candidate jury, not a result-driven reopening of Explorer,
Tree, Program or QuoteGraph. Proposal review required before implementation/GPU.
Native Qwen3-VL-8B global/own hard stance and fixed-r6 are the reference. Both
HateMM215/HateClipSeg118, unchanged4fps, same inputs/constants/code per corpus.
Development-selected. No GT/predictions/metrics in acquisition or scoring.

## Hypothesis, sources and narrow claim

An entity shown or mentioned repeatedly can be the speaker, quoted author or
target of different local acts. Persistent source-bound entity occurrences and
directed discourse links can retrieve the actual evidence needed to interpret a
local act. The proposed intervention is graph traversal that controls actual
endpoint media retrieval, followed by independent local measurement. A generated
ledger, extra frames, neutral description or reasoning alone is not the claim.

[Semantic Event Graphs](https://arxiv.org/html/2601.06097v1), sections2.1–2.4,
Algorithm1/limitations5.2–5.3, uses YOLOv11 persistent tracks and proximity START/
END events, a temporal MultiDiGraph and lexical graph queries for video QA.
Its five-video/120-question auto-generated and auto-judged proof of concept
does not establish discourse ownership. This is a SELF-DESIGNED discourse/media
adaptation, not a reproduction of its detector/tracker or lexical retrieval.
Do not use its content-hash event deduplication; only readable coordinate tuples.

[MATCH](https://jianlang.org/papers/MATCH.pdf), III-A–D, already retrieves
spatiotemporal evidence for opposing clues and uses a VLM verifier and trained
predictor. [RAMF](https://arxiv.org/html/2512.02743v1),3.1–3.2/3.4–3.5, already
uses objective and competing explanations with learned fusion. The related
[WWW2026 evidence-structure source](https://doi.org/10.1145/3774905.3796488)
was located but its full ACM page was inaccessible; do not claim its full method
has been excluded. Source read scope: `runs/20261005_m1_provenance/source_reads/`.
Independent rule4 review must actually search target-domain use of the complete
method. Generic graph/evidence/reasoning novelty is not claimed. Unlike24's
transcript-span graph and asymmetric text attention, this method discovers
multimodal entity occurrences and retrieves actual cited endpoint pixels. No
trained detector, gold identity inventory, extra language model or prediction vote.

## Frozen complete initial method, before data acquisition

1. Same native20 JPEGs/full Whisper segments/native8s windows as current r6.
   Raw video is decoded on one host, once, to index actual frame PTS relative to
   container start (first decoded time only if no container start). In each
   [a,b) window select the first actual in-window frame at/after a+(b-a)/3 and
   a+2(b-a)/3; if none lies after a target, choose the last actual in-window
   frame. Dedup actual frame index, keep earlier time/index ordering. No-frame
   windows remain explicitly uncovered. Save lossless PNGs, actual PTS/timebase,
   dimensions, source path and selection rule in the new cache. Existing native
   JPEG times remain nominal, not certified PTS; ASR crops remain proportional
   word crops from shared `window_text`, not verified word timestamps.
2. For each window make one fresh SAME-Qwen neutral ledger generation with
   native20 overview images, the selected actual local images and exact local
   ASR body. No global verdict, moderation rules or hate hypotheses in this
   generation. Overview images are context and cannot be cited as local
   witnesses. Source IDs `p0,p1` refer only to selected actual local frames;
   source `u` is the exact full local body. System: `Describe observable people,
   actions and discourse sources. Return only the requested JSON. Do not judge
   policy violations or invent identities.` User lists all available sources,
   times, source roles and the following schema/instructions literally.
3. Ledger schema:
   `{"entities":[{"id":"e0","description":"person at podium","frame":"p0","quote":""}],"actions":[{"id":"a0","description":"speaking at podium","actor":"e0","target":"UNKNOWN","frame":"p0"}],"utterance":{"speaker":"UNKNOWN"},"quotations":[{"id":"q0","quote":"exact local substring","owner":"UNKNOWN"}]}`.
   Entity IDs e0–e3 (max4); action IDs a0–a1 (max2); quote IDs q0–q1 (max2).
   Entity source is either an available local frame with empty quote, or null
   frame with a nonempty exact, uniquely occurring local substring. Visual
   descriptions and action descriptions each<=24 whitespace words. Actions cite
   an available local frame and an existing local entity or UNKNOWN as actor/
   target. Utterance speaker and quotation owner are existing local entities or
   UNKNOWN. Quotations must be exact uniquely occurring body substrings; derive
   their character bounds from that unique literal match, never approximate or
   correct the model string. Empty arrays/UNKNOWN are permitted. User instruction:
   `Record only observable local entities and actions. Use frame IDs only from
   the local image list. Copy textual mentions or quotations exactly from u.
   Infer an utterance speaker or quote owner only when supported; otherwise use
   UNKNOWN. Do not add confidence, hate labels, decisions or summaries.`
   Greedy seed0, max512 generated tokens, no retry/beam/salvage/coercion. Duplicate
   JSON keys, schema/types/IDs/bounds/word-cap/source errors or whole-call
   truncation make the entire ledger UNKNOWN, with actual raw tokens retained.
4. Always create the actual utterance node `w0000/u` etc when local speech is
   nonempty. Valid ledger entities/actions/quotes receive source IDs
   `w0000/e0`, `w0000/a0`, `w0000/q0`. Their times and media references are supplied
   by source coordinates, not generated timestamps. Valid local actor/target/
   speaker/quote-owner fields compile into directed typed links. Structural
   validation certifies source existence and types, not factual identity or
   ownership. UNKNOWN entities are never fabricated to satisfy an edge.
   Literal local-link types/directions: actor is action->actor entity; target is
   action->target entity; speaker is utterance->speaker entity; quote_owner is
   quote->owner entity; quotes is local utterance->its copied quote node.
5. Link all windows into one persistent video graph. For each disjoint block
   of8 anchor windows, one fresh text-only SAME-Qwen generation sees the complete
   source-bound ledger/node table for this video (including complete literal
   utterance bodies), IDs and all available accepted local links. It emits only
   `{"edges":[{"from":"w0000/e0","to":"w0008/e1","type":"same_entity"}]}`.
   Types: same_entity between entity occurrences; quotes/responds_to/retracts/
   speaker_change between utterance/quote nodes; quote_owner from utterance/
   quote to entity. Require distinct endpoints, at least one endpoint in the
   anchor block, both exact existing IDs, and the declared endpoint types.
   Literal system: `Link observable entities and discourse sources across video
   time. Return only the requested JSON. Do not judge policy violations or
   invent identities.` Direction: quotes is quoting utterance/quote->the cited
   utterance/quote; responds_to is response->the utterance/quote being addressed;
   retracts is retraction->the earlier utterance/quote it retracts (require earlier
   target window); speaker_change is earlier utterance->later utterance (both
   utterance nodes, distinct windows). quote_owner is utterance/quote->entity
   attributed as its author. same_entity is symmetric; canonicalize endpoints
   by their existing source-ID order. A node's kind/time constraints are enforced;
   semantic role correctness remains an empirical question.
   Instruction: `Link source observations across time. same_entity means the
   same observed or named referent, not merely similar descriptions. quotes,
   responds_to, retracts and quote_owner preserve their actual directed roles.
   speaker_change identifies a change of speaker between source utterances.
   Use only supplied node IDs; omit unsupported relations. Return only edges,
   without descriptions, confidence, labels or decisions.` Max64 edges/call,
   max2048 generated tokens, greedy seed0/no retry. Entire invalid/truncated
   block becomes UNKNOWN; keep no valid subset. Dedup identical endpoint/type
   tuples; no hashes or score-dependent graph selection.
6. Compile same_entity connected components with union-find; canonical component
   ID is its earliest occurrence by window/local ID. Keep each original occurrence
   and witness. A window seeds its own utterance/entities/actions/quotes. Local
   membership traversal is window->nodes only, never reverse expansion of an
   unrelated remote window. same_entity traversal costs zero hops; all other
   accepted links cost one hop and are undirected for retrieval while their
   original direction/type is retained literally. Traverse at most2 semantic
   hops. Retrieve unique reached nonlocal source windows: nearest2 strictly
   earlier and nearest2 strictly later, ordered absolute time distance/window
   index. The result is executed from graph structure, without hate score,
   embeddings, lexical query filtering, query-dependent confidence or GT.
7. For each selected source window, retrieve its actual up-to2 acquired PNGs and
   exact ASR body, plus accepted path metadata and validated source-bound factual
   records. Context is explicitly interpretation-only. Local occurrences can
   be supported only by current-window frames/body. Global native20/fullASR
   remain visible; no hard global-isolation claim. Each final visual/speech
   branch gets native prefix and native own hard stance, literal packet metadata,
   selected remote actual pixels/body, current actual local pixels/body and the
   ORIGINAL corresponding local question. Speech remains absent where native
   body is empty; visual remains available with explicit missing local-frame
   coverage. Never reuse a previous window's answer or branch cache.
   Literal packet header: `Graph-linked sources below are interpretation context.
   A source from another window does not establish an occurrence inside the
   current window. Source-bound descriptions and links are model observations,
   not independently certified facts. Use current-window pixels and speech to
   identify the local act.\n`.
   Each source record begins `[source_window={i}; role={local|context};
   t={a:.6f}-{b:.6f}s]\n`; each actual image begins `[frame={id};
   actual_t={time:.6f}s; index={index}]\n` followed by its image placeholder.
   Factual text is deterministic JSON serialization of the accepted field
   subset; typed paths are literal existing from/to/type triples and source IDs.
   Visual branch field subset: frame-supported entities/actions and their actor/
   target/same_entity relations only, with all cited retrieved/local images;
   no added ASR bodies, textual mention/quote/speaker observations or discourse
   prose. Speech branch subset: full literal local/retrieved ASR bodies, accepted
   text/frame-supported entities, utterance speaker, quotations/owners, relevant
   source IDs and all accepted selected-path relations, plus actual retrieved/
   local images for visible speaker/referent context; action descriptions omitted.
   Missing/invalid fields are absent, not negative judgments. Original native
   fullASR remains in the shared prefix of both branches. Append the entire
   unchanged `yesno_question(i,n,a,b,body,kind)` for the corresponding branch,
   including its header/body and ORIGINAL question/answer instruction. No new
   stance question or generated decision enters either packet.
8. Actual final branches are independent full multimodal forwards (fresh cache,
   same native messages/history, appended multimodal source records, use_cache
   false and native FP32 Yes/No head). Their native20 images/fullASR/own stance
   are identical to the paired native path; the old global margin is retained.
   This deliberately pays full-prefix/vision re-encoding cost to make retrieval
   unambiguous. Save literal messages, all image paths/grids/expanded token IDs,
   selected nodes/paths/media/current source records and actual forward counts.
   max(new V,new S), canonical4fps curve and unchanged r6; graph has no numeric
   score/posterior/label fitting. No native/new score averaging or corpus routing.

## Costs, execution checks and decision rules

Reuse native20/fullASR/Qwen/raw videos. New videos pay full actual PTS indexing,
up-to2W local images, W<=512token ledger generations, ceil(W/8)<=2048token link
generations, and W+nonempty-speech-W full multimodal local forwards. Re-encoding
native20 and retrieved context in each forward is a real cost; it is not made
free by caching or a frozen model. Estimate120–300GPUmin/full333 is unmeasured
and may be exceeded; report actual generation tokens/forwards/encoder calls/
decode time/GPU seconds/peak and native standalone/new standalone separately.
Source-only size preflight audits native/local inputs before GPU; actual complete
generated video/link tables are size-checked before their link GPU prefill and
recorded, because their literal observations do not exist before acquisition.
no scientific truncation/cap relaxation to fit a machine without declaration.
Cache `data/temporal_entity_discourse_graph/` with PROVENANCE; outputs
`runs/20261005_m1_provenance/r1_*`; all source acquisition and measurement for
the whole candidate stay on one host. Long GPU runs only committed Slurm.

Before GPU: independent rule4 proposal PASS, deterministic CPU graph/path/source/
typed-link/UNKNOWN/PTS fixtures and independent rule6 observation-code review.
Actual fixed5 is the same source-selected first2 per corpus plus HMM114; no GT
or parameter choice. Require native allraw exact and all six metrics in full333;
actual nonempty local ledger AND nonlocal retrieved source windows in EACH
corpus for the intended mechanism to have been exercised. Replay literal
sources/geometry/ledger/link parsing/graph traversal/media/token rendering;
repeat one actual full multimodal branch per available kind/video exactly.
Empty ledgers/edges remain UNKNOWN, never fabricated factual evidence. Actual
counts/invalid outputs/nonlocal retrieval/no-op coverage must be reported. No
main if implementation/native/source or declared exercise checks fail.

Identical full333 first. Performance PASS requires the SAME main metric+.01
BOTH vs current r6, all pooled losses<=.005 and within losses<=.01, reporting all
six and84/99 defined within videos. No qualifying gain->archive; any gain but
noPASS->logged actual GT error analysis and at most3 result-guided revisions.
No test label enters method/fitting/threshold. If mainPASS, freeze full333:

- no_cross_edges: same acquired pixels/ledgers/local fields, remove all inferred
  crosswindow edges before traversal, no remote retrieval. This tests the whole
  persistent retrieval component; does not isolate more pixels from structure.
- nearest_time: same per-window earlier/later remote-window counts as full;
  retrieve temporally nearest real windows, independent of graph. All images
  and utterance bodies remain actual. Pixel counts match (handle missing-frame
  count patterns by deterministic nearest assignment before scoring). Text-token
  lengths may differ and must be reported; this comparison does not alone prove
  equal-language-budget superiority. Predeclare assignment and coverage before
  any control score. Full must beat this alternative by common+.01 BOTH to
  support the persistent graph over temporal acquisition.
- wrong_ownership: degree- and kind-preserving endpoint permutation within each
  directed edge-type table; retain entity occurrence/actual media/ledger strings,
  source identities and times. Use fixed half-list rotation with no seed/hash;
  report self-edge/unchanged/no-op coverage rather than invent replacements.
- wrong_cited_media: identical graph/packet/ledger/local actual sources, rotate
  only nonlocal retrieved witness images and ASR bodies among the acquired
  eligible nonlocal windows in that video, never move local witnesses. Record
  actual mismatches/no-op coverage, keep real donor coordinates explicit.
- no_entity_components: remove only inferred same_entity links before traversal;
  preserve discourse links/ledgers/media. A standalone entity-persistence claim
  is allowed only if its removal causes common-main .01 loss BOTH.

Every claimed novelty component needs rule14(g) common .01 dual-corpus ablation;
wrong binding/source interventions and raw V/S/max ordering with paired-video
intervals must falsifiably support the explanation. No-op controls/structural
validity/longer descriptions cannot establish mechanism. Postscore actual
source-ownership audit distinguishes correct source acquisition from changed
interpretation; all GT-read files/findings/design decisions go in this README.
Independent final claim/code confirmation and local returned metrics/STATUS
must precede any goal-complete report. No Overleaf or external notification.

Source-only input availability audit found the native k20 request legitimately
has18–20 cached frames in some videos. All references to native20 above mean
the EXACT current `frame_paths(dataset,video,20)` result, including actual count;
never fill/drop frames to force20 or alter the paired native reference. Actual
counts/source paths are in `source_reads/input_availability.json`.

Independent once-only proposal PASS: `docs/reviews/20261005_m1_provenance_proposal.md`.
Independent once-only observation code review PASS with narrow repair confirmations:
`docs/reviews/20261005_m1_provenance_code.md`. Four actual bugs fixed before GPU:
legal duplicate tuple dedup; actual full raw PTS/index/source-pixel revalidation
and generating/current-host path distinction; exact cap-flag binding; cost/time/
peak/encoder counters replay-bound to actual traces. No scientific method or
exercise guard weakened. Source verification inside acquisition is charged in
decode_seconds; additional CPU code/preflight checks are experiment costs.
Author300 all-pairs path oracles, typed-source/whole-rejection fixtures and actual
fixed5 PTS/PNG/CPU token inputs PASS after source fix, authority
`runs/20261005_m1_provenance/cpu_checks/summary.json`. New acquisition shared
helpers are src/{actual_video_frames,source_generation}.py; no cross-experiment
import or copied canonical evaluator. Independent real36layerBF16 CPU generation
and actual fixed5 input/source/40corruption tests PASS, not actual8B measurements.
Full333 actual native/text CPU source preflight PASS, authority
`runs/20261005_m1_provenance/source_preflight/summary.json`; all raw/native sources
available. Actual generated complete link tables are counted before their GPU
prefills, never silently truncated. Actual SAME fixed5 GPU next only after code
synchronization/fresh idle-node check; main remains blocked on actual native/
fresh-repeat and each-corpus nonempty-ledger/nonlocal-source exercise PASS.
