# Candidate28: anchor identity temporal token fusion

Proposed2026-10-05 onsc474397. Selectedrank1 from the complete9 pool in
`experiments/20261005_m1_ideation/CANDIDATES.json`; single independent proposal
review `docs/reviews/20261005_m1_ideation_jury.md` PASS, same-family provisional.
Current formal reference r6_bma. Bothcomplete215HateMM/118HateClipSeg,
canonical4fps, development-selected. No scientific GPU or GT yet.

## Source and falsifiable hypothesis

Primary TTF: https://arxiv.org/html/2605.07355v1, actualsection3.1 Eq1–12 read.
Official https://github.com/Cominder/ttf checked2026-10-05 via its contentsAPI:
onlyLICENSE, no implementation. Actualread evidence in
`runs/20261005_m1_ttf/source_reading/official_repository_contents.json`.
This is a paper-based implementation with explicit multi-image/DeepStack
adaptations, not an author-code reproduction. No first-token-pruning claim.

Hypothesis: identity compression of repeated background and retention of
temporally changing tokens improves within-video ordering. Efficiency alone,
compression rate or any pooled gain is not evidence of this mechanism.

## Frozen initial mechanism/constants

One frozenQwen3-VL-8B-Instruct, BF16/SDPA/fp32YesNo head, seed0, sameflowboth.
Originalnative20requested frames (actual18–20), originalfullASR, originalpixel
caps65536/100352, 8s windows, originalG/ownhardYesNo andoriginalspeech margins.
No extra frames/captions/models. GT never enters measurement or constants.

1. Capture actual final-projector and everyDeepStack feature from the native
   vision forward. Native ordinary prefix/G/stance/V/S is measured unchanged;
   capture never alters its output. Reuse these same features for the new
   language prefix; account the original vision acquisition cost once.
2. Allnativeframes must share one actual mergedHxW grid, verified from actual
   processor output. Select anchor by largest cosine of spatialmean with
   allframe/tokenmean; fp32, normalization epsilon1e-12, earliest tie.
3. For every source token, evaluate cosine to the anchor's clipped3x3 grid
   neighbors, offsets lexicographic(-1,-1)through(1,1). Duplicateborder indices
   are masked after the first occurrence. Earliest maximum wins. Retain source
   iff bestsimilarity<=.70; keep everyanchor token. Exact identity replacement,
   no weighted/convex feature mean. Preserve both retained sourceindices and
   removed-source-to-anchor indices. Constant.70 is sourcepaper's8B setting,
   not chosen with current labels/metrics; no scan.
4. Paper's anchor-first ordering is retained. Re-render the unchanged native
   prompt with whole timestamp/image chunks ordered anchorfirst, then all
   nonanchors in originalchronologicalorder. ASR/policy/system/question stay
   literalunchanged. Match embeddings/DeepStack to this same sourceorder.
   Keep all text tokens and everyimage boundary, including zero-retained
   sourceimage blocks. Gather surviving visual rows within each block.
5. Position adaptation: compute uncompressed reorderedprompt nativeQwen3
   positions. Replace each visual row's triple by its actual originalnative
   row's triple (native independent-image offset encodes sourceidentity; it is
   not literalvideo(k,y,x)). Gather surviving positions alongside embeddings,
   visual masks and everyDeepStack row. Text positions remain those of the
   uncompressed reorderedprompt. Do not regenerate an image_grid pretending
   sparse rows form a dense image. Call the actual language model directly
   with explicit3-plane mRoPE, compressedmask and gatheredDeepStack features.
6. Cache length is actualcompressedL. Qwen's native logical suffix offset is
   max(prefillposition)+1; set rope_delta=logicaloffset-L, then native cache
   append consumes exactly the originalglobalquestion suffix and its native
   hardanswer. This uses the native stance; there is no secondglobaldecision.
   This explicitly adapts paperEq12 to Qwen's nontrivialmRoPE delta: actual
   cache indices and logicalrotary coordinates are separate. EachlocalV is an
   independent unchangedquestion suffix on this cache, restored after reading.
   OriginalnativeS/G plus newV -> originalmax -> unchangedr6 M2–4.

Two languageprefixes are part of this complete method to preserve G/S while
changing only visualrepresentation. Newvideo calls: native3+W+Ws for pairing,
and new3+W; deployedmethod native3+Ws andnew3+W. Onevisionencoderforward/video
shared by both. Extra3 languageforwards/video, no extra generation. Source
memory held only withinvideo. Actualmatching/re-render/prefill/query wall/GPU
seconds and peakmemory will be recorded; estimated333 GPU12–25min is unmeasured.
Longvideo capacity must be checked with actualinputs, not token-size conjecture.

## Integrity and controls

Fixed5: firsttwo percorpus plusHMMhate_video_114. Require complete current
nativeallraw exact, G/S exact, actualtokencompression inbothcorpora, clonedcache
repeats, originalfull/no-compression native-order manualprefill equivalence,
allretained positions/DeepStack/source mapping replay. Unknown18frame exception
is retained. NoGT/fiveperformancegate. Full333 follows independentcode review.

Only afterperformancegate, whole333 samecanonical/r6 controls:
- No compression with same anchororder/position interface; isolates retaining
  temporalresiduals from mere ordering/interface effects.
- Same totalretainedvisualcount with deterministicuniform selection andsame
  anchororder. Anchortokens alwaysretained; sourcebudget evenly spread over
  chronologicalsource slots, round-robinremainders; evenly spaced rowindices.
- Firstframe anchor with originalmatching/.70; onlyclaimsemanticanchorselection
  ifremoving it causes same mainmetric>=.01 drop inbothcorpora.
- Wrongtime: same retainedembeddings/text/count/order/G/S, cyclic halfcircle
  sourceframe reassignment of visual mRoPE triples; unchangedspatialcoords.
- Wrongcorrespondence: anchorneighbor sourcegrid map cyclichalfwidthshifted,
  then recompute matching/gating; genuinelydifferent sourcecorrespondence,
  not a mere permutation of equivalent frame-independent matching calls.

Same-retained-set deletion is algebraicallyidentical to identityreplacement;
verify equivalence, never label it a scientific ablation. If ordering or
component ablations fail dualmetric.01, downgrade contribution orremove it;
do not stop on main numbers without mechanism/sourcefalsification evidence.

## Execution and current state

Prototype/CPU/sourcealignment checks, independentcode review andGPU pending.
No GT or predictions were read to propose this candidate. All numerical results
will cite canonical evaluator output paths, not markdown authority.

## Actual CPU checks2026-10-05

Author checks `runs/20261005_m1_ttf/cpu_checks/summary.json`: realQwen3 36language/3visionlayers,2actualDeepStack layers, FP32/BF16 and18/20frames. Native manualprefill everyKV andquerymargin exact; actualrepeatedpixelinputs compress8/72 or8/80 tokens, sparseposition/DeepStack gathers exact andclone/cache exact. Initialfixture lacked an LMhead, fixedonlyfixture; earlier randomimages retainedallrows and was insufficient for sparsecoverage, so actualrepeatedpixels were added. Both earlierattempt logs retained. These are numericalinterface checks, not semantic/performance evidence. Full333 actualinput preflightstartedCPU; independentcode reviewrequested from existingjury instance separatefromauthor.

## Independent code/fullinput checks2026-10-05

Singleindependentrule6review `docs/reviews/20261005_m1_ttf_code.md` PASS, noobservationalbug; reviewer reusedjuryinstance separatefromimplementationauthor. Actualindependentnonrectangularscalaroracle and36layernonfirstanchor sparse/fullforward/cache-restore oraclePASS; FP32difference<=5.3644e-7,BF16<=.00610 underdeclared.02tolerance, notbitwisefull-vs-cached. Realfull333native JPEG/ASR/processor-gridpreflight `runs/20261005_m1_ttf/full_input_preflight/summary.json` PASS, 332x20+1x18, allpervideogridscompatible, maxactualprefix5829H114. Actual8B/CUDA/sourcecompression/coststillunmeasured. Standalonephasecost excludes auditcachewrites/serialization andpairing diagnostics; actualSlurmwallreportedseparately, notcalledfullend-to-end cost.
