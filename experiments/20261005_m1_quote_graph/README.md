# M1 candidate24 backup: source-bound quotation and reference graph

Declared2026-10-05 while candidate22 full333 is collecting. Independent proposal
PASS; CPU prototype and independent code review PASS. Actual fixed-five GPU
validation is now complete; no GT or performance score yet. Candidate22 is now archived after R3 no qualifying gain; candidate23's
actualfixed5 failed its unchanged mechanism-exercise guard and needs a new
source/decoder interface. Result branch now permits this independent candidate's
fixed5 GPU verification. No scientific code, input or constant changed in switching.
Native Qwen3-VL-8B global/own stance/visual and fixed r6 remain formal reference.
Development-selected. No gold speaker/entity/quote inventory, no trained target
extractor or additional language encoder. All graph judgements use the SAME Qwen.

Actualfixed5 dispatch2026-10-05 onsc474399/uoa-lab2 Slurm131, committed
`launch/lab2.sbatch smoke`. Current machine/code/disk check and unchanged foreign
home scope note saved in `runs/20261005_m1_quote_graph/machines_before_smoke{,_note}.txt`;
all laboratory code synchronized/clean, selected5090 idle/569Gfree. Entire graph
acquisition and paired native/new scoring stay onthishost. Actual8B/native/clone/
contextless checks passed as documented below; no performance metrics or GT selection yet.

## Hypothesis and source limits

A locally uttered hostile phrase may quote another person, deny a statement, or
depend on a referent named elsewhere. A document graph can bind these sources to
the local utterance before one semantic speech read. The distinct mechanism is
executable span/edge packet construction plus asymmetric context-to-local reading,
not a new stance prompt or an automatically correct ownership annotation.

[Michel et al.](https://arxiv.org/html/2406.11380v2), section4/AppendixA, use
overlapping quote-attribution chunks and incremental predictions with a supplied
gold character/alias inventory; their limitations explicitly exclude full entity
discovery. [Vishnubhotla et al.](https://aclanthology.org/2023.acl-short.64.pdf),
sections4/5/6/limitations, separate entity discovery, coreference, quotation
identification and speaker attribution; experiments use trained components and
corrected/annotated upstream information. These are literary pipelines, not
frozen video graph encoders. Prior read log:
`runs/20261004_m1_ideation/backup_source_reads/quotation/source_scope.json`;
primary methods and literal prompts reopened2026-10-05. The graph compiler and
attention operator below are SELF-DESIGNED, inspired by the task decomposition,
not a faithful source-method reproduction or evidence that ASR speaker IDs exist.
ASR may lack quote marks, reliable identities and word timestamps. Nested/indirect
speech prediction here is a declared extension, not supported source performance.

HVL/Factorizer/native hard stance already test interpretation or isolation;
Program23 executes typed temporal/media operations and adds their factual results.
This proposal instead discovers exact transcript-span relations, traverses their
document graph, and changes which newly encoded context tokens can reach a local
decision. MATCH/RAMF/MAESTRO/LEAF already ground or retrieve context; no first
grounding/reasoning/quote-analysis claim. Independent rule4 review must actually
search for this complete method in hateful-video work before implementation.

## Common complete R1 specification, before extraction/scoring

1. Full215HateMM/118HCS; actual native20 JPEGs/full ASR, frozen policy/questions,
   Qwen weights, own native hard Yes/No stance and8s grid. Recollect native all
   global/window/curve reads exactly and all six fixed-r6 metrics. Graph extraction
   and scoring open no GT, labels, previous predictions or metrics. Missing native
   speech stays absent. No per-corpus choice, multiple-model vote or fitting.
2. The source utterances are the EXACT shared `window_text` strings for each fixed
   window; identity `u0000` etc in chronological order, with nominal window interval
   and literal character offsets. These are proportional ASR crops, not verified
   word timestamps or speaker turns. Empty bodies have no extracted span nodes.
   Process chunks of8 consecutive windows with stride6, stop at video end. Each
   chunk sees its literal source bodies, IDs and offsets, plus previously accepted
   graph nodes/edges wholly contained in the two overlapping windows. No native
   global verdict, policy, image, hateful score or external entity list in extraction.
3. One fresh text-only SAME-Qwen greedy generation/chunk, max2048 new tokens,
   temperature0, no retry/resampling. System: `Extract discourse relations from
   supplied transcript spans. Return only the requested JSON. Do not judge policy
   violations or invent speaker identities.` User gives chunk data and this schema:
   `{"nodes":[{"id":"x0","window":0,"start":0,"end":3,"kind":"mention"}],
   "edges":[{"from":"x0","to":"u0000","type":"refers_to"}]}`. Instruction:
   `Find exact spans for mentions, quoted or reported speech, attribution cues,
   rejection, endorsement, negation and correction. Mention links indicate the
   same referent; attributed_to links a quoted span to its speaker mention.
   Quotes/rejects/endorses/negates/corrects link their actual cue or utterance to
   the affected span. Use UNKNOWN by omitting unsupported edges. Every span must
   use exact source-body offsets and every endpoint must be supplied or defined
   here. Never add free text, labels, confidence, summaries or conclusions.`
   Node kinds are mention, quote, attribution, rejection, endorsement, negation,
   correction. Edge types are same_referent, attributed_to, refers_to, quotes,
   rejects, endorses, negates, corrects. Max32 nodes/48 edges per chunk; exceeding
   limits, duplicate JSON keys, parse/schema error or truncated output invalidates
   the ENTIRE chunk, recorded UNKNOWN. No silent truncation or corrective call.
   The same whole-chunk rejection applies to any invalid node bounds/type/window,
   unresolved endpoint or typed-edge constraint in step4: do not retain a valid
   subset of an invalid chunk. JSON windows/utterance IDs are0-based; printed
   moderation headers and context window numbers are1-based.
4. Validate integer character bounds0<=start<end<=len(body), current chunk window,
   allowed kinds/types and endpoint presence. same_referent requires two mentions,
   attributed_to requires quote->mention. A model edge's semantic truth is NOT
   certified by these checks; only actual source existence/types are certified.
   Add deterministic local membership edges utterance->its spans. Canonical node
   key(window,start,end,kind), ordered by that tuple; duplicate exact nodes merge.
   Valid edges map to canonical source IDs, dedup exact endpoint/type. First valid
   overlapping prediction retained for an existing identical node/edge; no score
   or label-based resolution. same_referent components are derived with union-find
   and assigned earliest member IDs, not gold characters or string-match heuristics.
5. Per native nonempty speech window, seed its utterance and all local span nodes.
   Treat accepted semantic links as undirected for retrieval while retaining their
   direction/type in the packet. Collapse same_referent components as one virtual
   retrieval vertex, retaining exact member spans. Traverse at most2 semantic
   edges; membership does not consume a semantic hop.
   Membership traversal is directed utterance->its spans only; no span->utterance
   return, so a reached remote span cannot fetch unrelated spans in that window
   via zero-hop parent traversal. same_referent component traversal is bidirectional
   and consumes zero semantic hops; other accepted relation types consume one.
   Retrieval is over these exact edges, not over a generated character name list.
   Collect nonlocal REAL span nodes, not entire remote utterances, reached through
   those paths. Sort shortest
   semantic hop count, absolute source-window distance, window,start,end,kind/ID;
   keep at most8 unique context spans. No score/query embedding/rating or selected
   hate words. Record all traversed edges, omissions and no-context windows.
6. The speech suffix has current window header, literal relation metadata for
   the selected packet, up to8 separately source-tagged exact context spans, the
   complete unchanged native local body, then the ORIGINAL speech question and
   Yes/No answer header. Metadata reports directed edge type and IDs, nominal
   source-window interval and body character offsets; no model-generated prose.
   Literal scaffold: `Consider only window {i} of {n}, from {a:.1f}s to {b:.1f}s
   of this video. Source-linked spans below are interpretation context. A span
   from another window is not an occurrence in the current window.\nRelations:\n`.
   Each context record is `[source={id}; window={j}; t={c:.1f}-{d:.1f}s;
   chars={lo}:{hi}; kind={kind}]\n{exact_text}\n`; local record is
   `Local speech:\n{exact_native_body}\n`. Keep unknowns implicit via missing
   relations, no fabricated no-hate or no-speech verdict.
   The final question is only this literal tail (not the entire yesno_question
   helper, which would repeat header/body): `\n\nIs THIS window one of the segments
   where speech that violates the above rules occurs?\n\nAnswer "Yes" or "No".`
7. Append the full compiled suffix to an independent immutable native stance
   cache in ONE Qwen forward with BF16 additive bias in all36 language layers.
   Scaffold/metadata tokens see native prefix and their earlier scaffold tokens.
   Each context record sees native prefix/scaffold and its own earlier tokens,
   excludes other context records/local/query tokens. Local-body tokens see native
   prefix/scaffold, ALL selected context records and earlier local tokens. Final
   question/answer header sees native prefix/scaffold/local/earlier query tokens,
   but EXCLUDES direct access to added context records. Allowed entries0, others
   negative infinity; serial3-axis positions, physical KV identity unchanged.
   All delimiters belong to their record; preceding newline belongs to scaffold
   or preceding record explicitly in saved ranges. Literal segmented tokenization
   and actual ranges/IDs are saved; no claim of ordinary BPE equality. Only final
   native FP32 Yes/No margin replaces S. Original native overview/full ASR remain
   visible in shared prefix: hard global source isolation is NOT claimed. The
   tested constraint applies to newly encoded context tokens, which influence
   the query through the local representation rather than direct new context KV.
8. Crop/restore all KV layers and rotary delta after every window, no previous
   local answers. Same native G/V; max(V,new S), original shared evaluator4fps
   and unchanged fixed r6. No graph numeric scores, calibration, threshold or
   per-corpus route. Seed0; missing/invalid graph still has unchanged local body
   and empty context packet, not an invented local score. This is the complete
   proposed method, not source speaker-attribution accuracy transplanted to video.

## Costs and required decision evidence

Reuse raw media/native JPEG/full ASR/Qwen; no new encoder/audio/frames. New video
needs ceil-like stride6 chunk generations (up toceil(W/6), actual count recorded)
and one extra speech forward per native nonempty window. Each generation<=2048
tokens; count every prefix/generated token, truncation/invalid result and actual
encoder/language forward, GPU time/peak. Actual physical joint native+new calls
and standalone attribution separately recorded. Preliminary30–90GPUmin/full333,
unmeasured; this replaces the earlier loose ideation25–60 estimate. Extraction
cache belongs in `data/temporal_quotation_graph` with provenance; scores/logs in
`runs/20261005_m1_quote_graph`. CPU prototype only after independent proposal PASS.

Fixed5 noGT validation: real JSON/exact span provenance, overlaps/type/merge and
an independently enumerated toy graph path oracle; actual selected source spans;
actual every-layer mask/range/serial position; normal/exception cache recovery;
native exact; first new S cloned-cache exact; contextless suffix matches identical
serial token forward within previously declared operational BF16 tolerance .01.
Then identical full333; no pilot tuning or partial-GT choice. Independent code
review before GPU. Structural JSON validity is not a factual source-ownership audit.

Main gate: same main metric+.01 BOTH vs current r6, all pooled losses<=.005/within
<=.01, all six report84/99. No any qualifying main gain->archive; any gain but no
PASS->postscore logged error analysis/max3 revisions. If PASS, full333 controls:

- packet_serial: exact same selected spans/relations/local tokens, ordinary causal
  mask and positions. Tests asymmetric encoding beyond graph-text/context prompting.
- nearest_context: same per-window context-span count and token budget, select
  nearest source spans instead of semantic paths, same structural encoding. Fill
  only with actually available spans; actual length/count mismatch explicitly
  reported, no invented padding/text. Tests relational packet acquisition.
- wrong_edges: rotate source endpoint assignments within each semantic edge type
  by floor(K/2), keep node source identity/truthful times/types; recompute paths.
  Singleton/unchanged components separately report actual coverage.
- no_coreference: remove same_referent connectivity, preserve other actual nodes
  and relations, recompute packet. A component claim only if its removal gives
  common-metric+.01 dual-corpus drop; otherwise demote to implementation detail.

Structural novelty requires full beats packet_serial AND nearest_context on the
same common main metric by .01 BOTH, plus rule14g for claimed parts, wrong-binding
coverage, postscore exact-source/ownership audit, raw S/max ordering, paired within
CI and final independent novelty/mechanism review. nearest_context imperfect
matching limits interpretation; alternative perfectly matched packet pool must
be decided BEFORE control scores if needed. No claim that extra context spans
prove scope correctness; negative controls and human-readable sources decide.
No GPU beyond candidate22/23 result branching, no Overleaf or notifications.

## Implementation and checks (2026-10-05)

Independent proposal review PASS:
`docs/reviews/20261005_m1_quote_graph_proposal.md`. The primary methods/prompts
and target-neighbor evidence are retained under
`runs/20261005_m1_quote_graph/proposal_review/`. Novelty remains provisional:
the full WWW2026 evidence-structure neighbor text was inaccessible, not evidence
of absence. No complete source-pipeline reproduction or gold speaker accuracy
claim is made.

`graph.py` executes exact-span whole-chunk validation, overlap merge, components,
two-hop traversal and source packet compilation. `extract.py` calls shared
`src/mllm_generate.py` for fresh text-only greedy generation; it stores actual
rendered prompts, prompt/generated IDs and deterministic parsed results for
current-source replay. `reader.py` performs the single structural suffix forward;
`measure.py` records paired native/new scores only after source validation and
complete atomic per-video records. CPU processor replay includes image-expanded
native prefix IDs; scoring does not read earlier predictions. `analyze.py`
validates those bindings without GT, then invokes the shared evaluator and
unchanged fixed-r6 executable as separate postscore stages. No copied evaluator.

Author CPU checks ran on sc474397 in HateVideo, with no GPU or GT:
`runs/20261005_m1_quote_graph/cpu_checks/selfcheck.log`. A manually constructed
referent/quote/correction chain checks zero-hop referent membership and the
two-hop bound, including an unrelated remote-window span that must not be
retrieved. An independently enumerated matrix oracle checks every suffix row's
context/local/question visibility and the no-context causal degeneration. Python
compilation and shell syntax checks passed. These are implementation checks,
not dataset scores or ownership evidence. Independent code review PASS:
`docs/reviews/20261005_m1_quote_graph_code.md`. The distinct reviewer additionally
tested 14 graph starting vertices against a path oracle, real 36-layer BF16
attention/KV execution and exception recovery, actual five-video CPU processor
input expansion, 17 corruption cases, 11 whole-chunk rejection cases and explicit
generation/evaluation stubs. Evidence is in
`runs/20261005_m1_quote_graph/code_review/`. Actual 8B fixed-five GPU validation
completed in Slurm131; results are documented below.

When this candidate's result branch is reached, launch on sc474399 only after
code review and synchronization:
`sbatch experiments/20261005_m1_quote_graph/launch/lab2.sbatch smoke`, return both
run and input cache to sc474397, then run
`python experiments/20261005_m1_quote_graph/analyze.py --stage prepare --smoke`.
Only after these noGT checks pass, launch the identical `main` command. Complete
results are returned locally before the detached CPU
`launch/run_analysis.sh` evaluation. All predicted performance and cost remain
unmeasured until these actual runs.

## Actual fixed-five validation and main transition (2026-10-05)

Host sc474399/uoa-lab2, Slurm131 finished08:19:06; both run outputs and
`data/temporal_quotation_graph/` returned immediately to sc474397 without delete
or checksum options. Local CPU `analyze.py --stage prepare --smoke` PASS.
Authority `runs/20261005_m1_quote_graph/r1_full_smoke_analysis/plumbing_summary.json`:
5videos, native allraw exact, global/visual unchanged,134new speech windows,
5clone/contextless checks passed; no GT read. This is plumbing evidence only.

HateMM:16chunks/10rejected/0truncated,3628generated tokens,0graph nodes/edges,
0of96windows have new context. HCS:11chunks/8rejected/0truncated,3102generated
tokens,13nodes/12edges,13of62windows with38added records. Empty/rejected chunks
remain UNKNOWN under the original declaration; no coercion or new coverage guard.
The graph mechanism was not exercised on these HateMM examples; changed S alone
cannot establish it. Full333 must report actual coverage and all six metrics.
Measured new-video processing59.603089s HMM/47.591444s HCS versus native
8.437781s/4.095749s, including extraction50.901563s/42.972942s. Peaks19.000096/
18.843460GiB. Full runtime remains unmeasured; no smoke-GT selection.

Proceed with identical frozen R1 `launch/lab2.sbatch main` after committing the
actual checks and current machine synchronization. Whole333 stays onsc474399.
Return BOTH outputs and derived input cache before canonical local evaluation;
controls remain conditional on the declared complete main gate.

Main Slurm132 dispatched2026-10-05 onsc474399 after fixed5 PASS and current
`machines_before_main{,_note}.txt`: all four laboratory revisions clean/synced,
lab2 selected5090 idle/569Gfree, exact foreign-home STRAY names unchanged from
smoke. Whole215+118 acquisition and paired scoring use identical frozen R1.
Actual complete results pending; no GT/performance verdict.

Complete333 Slurm132 finished09:51:11 onsc474399. Both graph inputs and paired
run returned immediately tosc474397; local noGT input/native alignment and
canonical/fixed-r6 analysis launched. No performance verdict before report.

R1 complete333 canonical/fixed-r6 report finished onsc474397. Authority
`runs/20261005_m1_quote_graph/r1_full_main_decoded/optimized/metrics.json`:
HMM ROC/PR/within .8952316993171214/.6826257279789387/.7617883418675452 (84);
HCS .709027535899747/.6567629524600227/.6296614019643806 (99). Native allraw/all6
exact; G/V unchanged,6580S changed. HMMwithin+.011006298 (paired95CI
[-.012058777,.036432422]) is a qualifying development signal, not robust mechanism
evidence; HMM PR-.011608875, HCSROC/PR/within-.007797328/-.014309132/-.007687712.
Performance FAIL/any qualifying gain TRUE; keep R1 and begin actual logged GT
error analysis for at most3 revisions. Rawmax within-.021992913/-.008209162;
rawspeech shared-.007728745/-.008996150, intervals contain0. No claimed mechanism
or controlGPU after failed main. Actual new-video processing83.13min/native9.04x
(source73.37min); full source coverage178/3768HMM and198/3591HCS windows have
nonempty remote spans. Exact cost/coverage authority
`runs/20261005_m1_quote_graph/r1_full_main_analysis/alignment.json`.
