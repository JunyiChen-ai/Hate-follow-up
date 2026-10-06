# Candidate37: layer-wise source-bound video KV retrieval

Selected first from the unchanged20261006 nine-candidate pool after one fresh
independent rule4 review, PASS/same-family provisional:
`docs/reviews/20261006_m1_ideation_jury.md`. No code review, scientific GPU,
target GT/error analysis or performance yet. Current formal method r6_bma,
complete215HateMM+118HateClipSeg/canonical4fps/fixedr6 only. All future results
development-selected. Other active candidates and their budgets are unchanged.

Primary actual read: [ReKV](https://arxiv.org/html/2503.00540v1) section3/4.2,
ICLR2025 firstthree pages and method, and [official repository](https://github.com/Becomebright/ReKV)
README/model directory. Core transfer: chronological source encoding with a
bounded direct attention window, offloaded contextual KV, per-layer internal
query/key retrieval and answering from retrieved media. Original LLaVA/1D
RoPE/64-frame retrieval differ from this singleQwen/mRoPE/local-support design.
Official attention implementation has now been read as recorded below; don't
claim full source-code reproduction or source-paper efficiency on Qwen.

## Complete R1 definition before implementation

- Same frozenQwen3-VL-8B/seed0/native20(actual18–20)/fullASR/nativeG/ownhardstance,
  independent original8s V/S/max and fixedr6. Only V reads new actual media
  memory; S remains freshly paired originalnative. No caption generation,
  semantic ledger, classifier ensemble, final margin subtraction or smoothing.
- Acquire actual0.5fps source frames: targets0,2,4,... strictly before manifest
  duration; choose first decodedframe at/after target, deduplicate actualindices.
  No invented tail frame when target is uncovered. Origin containerstart_time
  else firstPTS, same PyAV18 runtime for generation and strict replay. Record
  all actualPTS/index/shape/coverage; evaluation stays4fps. Actualtime determines
  half-open8s ownership, never the nominal target. Raw media/native/ASR may be
  reused with provenance, but new decoding/encoding/storage/reads are paid.
- Preserve original image processor bounds65536–100352pixels and all native
  DeepStack processing. For each chronological source frame form one closed
  user observation containing a neutral source label, actualtimestamp and full
  original image, no assistant answer or task decision. It follows the native
  conversation plus at most4 immediately preceding source observation blocks.
  Direct attention is bounded; contextual ancestors may contain older frames
  and the whole native context. Record direct and inherited ancestry honestly.
- Save each block's per-layer post-normalization/pre-RoPE K and V in modeldtype,
  original3axis positions, exacttokens/imagegrid and actualimage-token indices.
  Per-layer representative is FP32 mean of only realimage-token K rows,
  concatenating8KVheads. Metadata/text rows remain in the sourceblock for
  answering but do not enter representative pooling. Sourceblocks are computed
  by actual new vision+LM forwards, not synthetic embeddings or native-frame
  deletion. Pre-RoPE storage permits exact same-position reconstruction with
  the actual Qwen rotation operator, avoiding approximate inverse rotations.
- Before each V question, compulsory LOCAL blocks are every available0.5fps
  frame whose actualPTS is in this window. No sourceframe from another window
  can be compulsoryLOCAL. Each language layer independently selects at most4
  remote blocks from this same video. Pool the current question-token Q rows
  after Qwenq_norm and beforeRoPE; group32queryheads into8KVhead groups by
  contiguous mean of4heads, then concatenate and average questionrows. Cosine
  against that layer's image-key representatives uses CPU FP32 for both actual
  selection and strict proof replay (4threads); per-layer host transfer and
  synchronization are charged. This avoids mixing GPU selection arithmetic and
  CPU replay arithmetic. zero vector gives
  zero similarity, deterministic ties by actualsource index. No label, margin,
  generated confidence, corpus route, probe score or fitted threshold enters.
- Questionrows mean the unchanged originalvisualquestion characters in the
  tokenized suffix, excluding role/source instructions and assistant header;
  intersecting tokenizer offsets determine rows and are stored. Layer0 query
  is not already conditioned on LOCAL; later layers are, through preceding
  attention. Retrieval at a layer uses its existing incoming Q, so it has no
  circular dependency on that layer's not-yet-selected memory and no extra
  query-probe model call. No guarantee is made that retrieval is semantically
  correct. Selectedremote blocks are context only, with honest owned times.
- For each layer chronologically pack the union ofLOCAL and selectedremote
  blocks after unchangednative prefix. Translate each block's three axes by
  the same integer, preserving internal spatial offsets. The next block starts
  after previous maximumlogicalposition; this layer's questiontext positions
  follow its selected union. Rotate originalpre-RoPE K with those positions,
  leaving V unchanged. Per-layer questionQ/K receive the matching actualQwen
  mRoPE; do not silently force one layer's physicalcachelength on another.
  Keep prefix K/V/positions unchanged. All question suffix tokens see prefix+
  complete selectedsource context plus only earlier suffix tokens; build the
  explicit rectangular causal mask, not an upper-left triangle over pastKV.
- The suffix preserves the native conversation and originalvisualYes/No
  question, adding only a fixed role declaration: frames inside the stated
  window are occurrence evidence, other owned timestamps only interpretation
  context. No paths/video IDs or retrieval scores are model-visible. Read one
  final FP32 Yes/No margin; no source-stage scored answer is spliced in. Native
  independent S/G/ownstance, originalmax and r6 remain unchanged.
- Source memory is per-video, reusable across that video's independentqueries,
  not across videos. With noLOCAL use fresh originalnative V (missing support,
  not absence). With LOCAL but noREMOTE execute LOCAL-only newV; no forced
  donor or droppedLOCAL. Restore nativeKV/rope and all attention methods after
  every source/query/error path. No method-state leaks to S or the next video.

Actual inspected model config:4096hidden/36language layers/32Qheads/8KVheads/
128head_dim, defaultrope_theta5000000, interleavedmrope_section[24,20,20],
visionpatch16/merge2/DeepStackvisionindices[8,16,24]. These are model configuration
facts, not new fitted parameters. Tiny CPU checks must exercise actual36layers,
GQA mapping, sourcevision+DeepStack, all3positions andnative restoration.

## Resource and provenance contract

Source F=ceil(duration/2) targets before actual dedup; sourceF vision/LMprefills,
native3+W+Ws calls, sourceF and WnewV calls in paired runs, clonecalls separate.
Deployment omits paired originalV extraW; one native prefix/G/stance, Fsource
prefills, WnewV and Ws originalS. Record exact executedcalls/tokens, CPU decode,
encoding/transfers/indexing/packing, GPU seconds/peak and I/O wallclock; source
cost and actualmetadata binding are not omitted because they are cached.
Planning range20–100additionalGPUseconds/minute plus .2–1s/window newV is
unmeasured. Fixed5 measures before full333. No claimed duration-free retrieval.

Dense all333 persistent KV is NOT required. Allper-layer KV costs144KiB/token
inBF16 before temporary buffers; liveonevideo can be large. Use owned per-video
RAM or temporary `.npy` blocks under its run, with position/index/bits parsed;
metadata, actualPNGframes, representativekeys, call/cost and selectiontraces
persist under `data/temporal_retrieved_kv/` and runoutputs. Finish video, then
release only its rebuildable temporary denseKV; preserve allresults/inputs.
Audit/controls that rebuild KV charge fresh source work, no freecached claim.
Exact sourcegrid/token counts and longestrealvideo preflight determine peak
RAM/disk before Slurm. No reduction of sourcefps/attentionrange or policy/
sysctl changes to meet capacity. No hashes/checksums/GitID resultprovenance.

Allscientific code self-contained in this experiment, stable sharedsrc imports
only. Whole333 source+reader run ononehost, bothresults/cache immediately return
ROOT before STATUS resultclaims. Config/path/date/model/runtime/actualcommand
provenance and source-generation hostname kept. No newdataset/rental/notification.

## Required checks, gates and mechanism controls

Author actual333raw/native/ASR/processor/capacity preflight, actualsourcepix/PTS/
tokens and numerical operator checks, onceindependent Rule6 code review, then
samefirst2/corpus+HMM114 real8B fixed5. Require actual sourceprefills/LOCAL/
remote layerselection inbothcorpora, nativeallraw/G/S exact, independentclones,
no label-access and no cache/method contamination. Empty/purelocal inputs have
defined honest behavior; don't change cohort or executionguard to obtain PASS.
CPU checks must compare source prefix+first<=5frames to an independently packed
full multimodal causal reference where histories actually coincide, samephase
identityrotation/native restoration and changedactualpixels/wrongblock effects.
Cached contextual blocks after slidinghistory changes are not claimed equal to
fresh source re-encoding under a different history.

Maingate: canonicalall6 against runs/20260926_twolevel/r6_bma/metrics.json,
sameprimary+.01 BOTH; pooledloss<=.005/withinloss<=.01; eligible84/99. Rule9
noanystandard+.01 archives; qualifyinggain permits loggedactualGTerroranalysis
and<=3revisions, not rawbranch diagnostics or roundedboundary values. MainPASS
is necessary but not sufficient for goal completion.

OnlyaftermainPASS run all333 controls: sameaddedLOCAL-only (delete remote);
equal-count chronology-only remote; one fixedlayer0 selectionsharedacrosslayers;
wrongactualremote blocks atmatched count/token buckets withtrue donor times and
LOCAL kept; time/role error intervention as explicitly declaredwrong-binding,
never disguised main data. Report source/rebuild and controlcosts. Claimed
retrieval/layerdependence contributions each require deletionloss>=.01 onthe
same metric BOTH, plus rawV/S/max ordering and falsifiable source-role evidence.
DenseLOCAL coverage/ordinary bookkeeping cannot be relabeled as retrieval gains.
Unsupported contributions are removed or implementationdetails. Necessaryfinal
independent review/localrawauthority/STATUS precede goal success; no claim from
proposalPASS, sourceexecution or finalr6 alone.

## Actual official implementation reading and initial software checks

2026-10-06 actualofficial mainbranch raw files downloaded (notexecuted) to runs/20261006_m1_rekv/source_reading/: model/abstract_rekv.py andmodel/attention/{rekv_attention,kv_cache_manager,rope}.py. Read sourceprefill vsretrieval switching, noquery-cacheupdate retrieval path, unrotatedQKV storage and positional transformation, meanquery/blocktopk and vector scoring. ActualVectorTensor topk uses FP32 unnormalizeddot product, while sourcepaper specifies cosine; thisR1 keeps predeclared paper-cosine andQwenpost-q/k-norm GQA adaptation. This is not an officialcode numeric reproduction. Originalinitialtokens/distanceceiling and1D rotary differ from preservedfullnative prefix/per-layer threeaxis source packing. No officialsysctl/environmentcommands wereexecuted, noexternalpredictor/model installed.

Author initialsoftware arithmetic PASS at runs/20261006_m1_rekv/operator_cpu_checks/summary.json: independentNumPymeans cover GQA4×1/2/8KVheads, actualimage-row-only pooling, stableties/zeronorm andfullLOCAL exclusion; actualQweninterleaved rotary identity/translation equalsits primitive exactly FP32/BF16; nine rectangularpast+suffix masks match independentexplicitcausality. These are arithmeticchecks only, not36layer sourceimage/fullreference/GPU/pretrained/nativeparity/GT/performance evidence. Collector/source memory/productionreader andfullpreflight still toimplement, thenonceindependentcode review.

Actualowned syntheticvideo sampling/pixel CPU checks PASS runs/20261006_m1_rekv/source_frame_cpu_checks/summary.json: first-at/after2second targets yieldoriginalindices0/16/32/48/64, halfopen8s ownership assignslastindex64 tosecondwindow, uncovered10secondtarget remainsNone. Decoderorigin/PTS/PNGpixels reread exactly. No model/GT/performance or newdataset; this checks sourceacquisition arithmetic only. Complete per-layer sourcecollection/offloadedKV andquestion reader still toimplement.

R1 pre-GPU numericbackend clarified while implementingproofreplay: FP32 cosine/topk is executedonCPU inbothgenerator andstrictreplay, withactualper-layer host-transfer timecharged. No scores/GT/hyperparameter scan or scientificGPUhas occurred; this is a sameR1 implementation definition, not a revisedperformanceresult.

## Complete prototype and one independent code review

Full actualsource frames/collector/pervideooffload/36layer retrieval/newV/strictproof/currentnativebinding/canonicalCLI/fixedr6 prototype andlablaunchers implemented. Actual333raw/native/realprocessor/YesNo source-token andquestionoffset preflightPASS runs/20261006_m1_rekv/full_input_preflight/summary.json; first-grid estimates largest8.7204GiB(non_hate_video_356) andallpersistent499.41GiB are estimates, notactualpeaks. Mandatorypervideo denseKV release avoidssilent499GiBglobalcache. No sourcefps/grid/algorithm changed.

Actual36language-layer/32Q8KV128head/3DeepStack random-weight CPU FP32/BF16×native18/20 sourceprefill+layerretrieval testsPASS: firstfive coincident causalhistories agreewithindependentfullmultimodal reference max4.77e-7 FP32/.006744 BF16 atprior1e-4/.025; sourceclones/nativeallKV/S exact, sevenframe slidingdirect/inherited ancestry explicit, pixelchange entersmargin, LOCAL-only exercised. Productionread_video/strictvalidator realdecodedownedfixture FP32/BF16 PASS andwrongtime/selectionrejected; independentreader additionallyexercised nonemptyS andtrueuncoveredtail. Explicitfixturetokenization/randomweights do notestablishproductiontokenizer/8B semantics; realprocessor/input preflight isseparate.

OnceindependentRule6 review PASS afteronlythreeconcretefixes, docs/reviews/20261006_m1_rekv_code.md andruns/independent_code_review/. Interruptedowned denseKV ispreserved infailed_temporary andsamevideo rebuilt; completedrecord reusefailsclosedonchangedmodel/runtime/implementation/nativeencodedpixels withactualarrays/nohashes. Actualper-attempt extraction/reader/preparewallcost capturesmodelsetup, validation/proof/recordI/O andfailedattempts separately; processingestimate includesoriginaldecode/source/nativebinding/V/S, pairednative/clones/audits nothidden. Numericmechanism/guards unchanged; no new scientific revision/GT/score tuning. Reviewconfirmsthethreefixes only, notanothergeneralreview. Actual8B fixed5 isnext, noGPU/performance/goalcompletionyet.

Actual8B unchangedfixed5 source+reader submittedsc474398/Slurm222 afterall4codeb0f6516 match/unchangedforeignwork/STRAY/1.3Tfree andoncecodePASS. NormalQOS queuesafterROOT216/short217; nofull333 beforeactualsamecohortstrictsource/current/native/allraw/clone/remote-execution guardPASS. No pipelinecost/modelperformance resultyet, sameR1/guard/budget0/3. Machineevidence runs/20261006_m1_rekv/machines_before_smoke{,_note}.txt.
