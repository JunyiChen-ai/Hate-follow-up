# Candidate37: layer-wise source-bound video KV retrieval

Selected first from the unchanged20261006 nine-candidate pool after one fresh
independent rule4 review, PASS/same-family provisional:
`docs/reviews/20261006_m1_ideation_jury.md`. Current status: complete333 R1 main performance gate PASS; mechanism controls implemented and independent control code review PASS, not yet scientifically executed. Main authority is `runs/20261006_m1_rekv/r1_full_main_decoded/optimized/metrics.json`; fresh interpretation partial/integrity WARN. Current formal method r6_bma,
complete215HateMM+118HateClipSeg/canonical4fps/fixedr6 only. All results
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

Actual unchanged fixed5/sc474398/Slurm222 DONE2026-10-06 15:33:49; BOTH inputs/runs immediately returnedROOT. Same-core-runtime noGT prepare PASS at runs/20261006_m1_rekv/r1_full_smoke_analysis/plumbing_summary.json: native allraw/G/S exact, sourceactualPTS/pixels/retrieval proof,624 sourceframes/prefills,5688 remote-layer reads,158 changedV,5 exactclones. Processing-stage estimate109.2292s vs pairednative12.4685s; source83.9130s, per-attempt setup/audit/I/O costs separately retained. PeakGPU18.1754GiB; largest fixed5 temporary dense5.443 GiB, not complete333 capacity claim. No GT/mainperformance/mechanism result. SameR1 complete333 will use onehost sc474398 with pervideo dense release.

SameR1 complete333 source+reader submittedsc474398/Slurm225 afteractualfixed5/noGT strictPASS andsynchronizedfourlab code/disk checks. NormalQOS waitsbehindshortROOT224+original186; no dependency/jobchain/GT/sciencechange. Samehost firstfive sourceframes mayreused with original decodecost, all333 sourceKV/readers freshinthisrun, pervideo denseKVrelease. No mainperformance yet.

Costreport clarification: new-reader/source processing estimates are notcomplete end-to-end newvideo deployment costs. Native20/fullASR caches areexisting inputs andfuturevideos stillpayfor their acquisition. Historicaloriginal ASR rows test215/118 containbatch-average times1545.58s/2085.63s (25.7597/34.7605min), inventory runs/native_input_cost_inventory.json withactualinput/script paths; notcurrent-runtime benchmark orsamehost latency promise. Oldwaveform extraction/modelsetup/native20 extraction components were notmeasured andcannotbezeroed/hidden. Original extraM1source costs remaincharged; addnative acquisition separately andmeasure/materiallydisclose beforea finaldeploymentcostclaim. This clarification changesnocode/margins/performance.

ReKV37 sameR1 whole333/sc474398/Slurm225 DONE2026-10-06 17:26:55,1:46:20/0:0. BOTHinputcache andallruns(proofs/nativepixel/key/query arrays,20.37GB transferredrunpayload) returnedROOT beforeanymainclaim. Matchingruntime strictprepare/canonical+fixedr6 nowrunning, nomainmetrics/performance/mechanismyet. Source-only initialreturn andfinalBOTH returnlogs retained, sourcePROVENANCEactualhost/date/stage updated.

## Complete R1 main performance and independent interpretation

Authority runs/20261006_m1_rekv/r1_full_main_decoded/optimized/metrics.json; all333 ROOTreturned/currentinput/nativeallraw/canonicalall6/fixedr6 complete. HMM ROC/PR/within .8965650607897894/.692932199898954/.7644661840611953 (84/215); HCS .733478025286554/.68466730263327/.653241581067832 (99/118). Relativefixedr6 deltasHMM -.0005536006/-.0013024034/+.0136841401; HCS +.0166531614/+.0135952182/+.0158924673. Samewithin BOTH>=+.01 andallpooledlosseswithin.005: numericalperformancePASS. No component controls completed, no mechanismclaim orpromotion; formalr6 unchanged, usergoal stillactive.

Onefresh independent result-to-claim judgmentpartial: C1supportedfixedcohort/C2missing, same-family provisional, runs/result_claim_review/{CLAIMS_FROM_RESULTS.md,verdict.json,own_evidence.json}. Independent integrityaudit WARN, notfabrication/leak: owncanonical recomputation/all333 pairedrecords/28895 sourceblocks/264672 layerchoices andfirstvideoactualpixels percorpusexact, raw+decodedmetrics exact. Report runs/integrity_audit/{EXPERIMENT_AUDIT.md,EXPERIMENT_AUDIT.json,own_evidence.json}. Need preciseexistingprotocolcoverage/grid/developmentselection/cost/transductive qualifiers.

Fixedcohort is215HMM/118HCS; actualHCSGTfilehas119testkeys, exclusionyt_NzvfkIYS5Yg duehistoricallytruncatedmedia predatesmethod. GTfloor/predceil/canonicalmin handling alreadysharedwithbaseline, notchanged. Correcteddata/gt_4fps/PROVENANCE.md wordingonly; noGT payload or evaluator changes. Fixedr6 meansfixedalgorithm/hyperparameters, stillunlabelledcorpus ranktransform/calibration/fitting, notpureinductive independentnewvideo adaptation. NewM1 source/readers label-free; allresultsdevelopment-selected. Nativeinputacquisition costdebt retained.

Mainprocessing-stage82.740783min (HMM41.9392/HCS40.8016), source64.151116min (incldecode25.34074min), pairednative9.148497min; actualSlurm1:46:20 andROOTprepare531.666sec separate, overlappingstagesnotsummedtwice. GPUpeak18.1754GiB, largestpervideo temporarydense9,502,801,920bytes. Actualsource28895prefills/vision, changedV7352, missingLOCAL7 honestfallbacks, S/G nativeexact. These execute/evidence counts are notcausalmechanism results.

Independentreviewer-led requiredcontrols at ABLATION_PLAN.md andruns/ablation_planner/plan.json, preparedonly. Allpriority1 component/matched-token/wrongbinding/ancestry controlsbeforegoalcomplete. Parentimplementation/feasibilitynext; sourceKV isreleasedaftermain, mustfreshrebuildandcharge. Keepallnegativecontrols andall6 metrics.

Complete control implementation prepared: controls.py runs fresh full-source KV once/video for R0/L0/C0/L1/C1/L2/D0 with seven independent queries (R0 original identity replay), separate T0 wrong-presented timestamps with frozen original LOCAL/remote IDs/packing, and H0 direct source history0 while retaining native global context. Actual own model forwards/vision and shared vs standalone costs stay separate; complete video records atomically commit before dense release, interrupted dense saved then rebuild. Default original source/reader/validator behavior remains unchanged. No main controls GPU/results yet. Once independent control code review in progress.

Author actual36-layer 32Q/8KV/3DeepStack FP32/BF16 production CPU control orchestration/clone/native identity/query proofs/budget corruption/ancestor rejection PASS at runs/controls_cpu_checks/summary.json; fixture source encoder does not represent timestamp semantics, explicitly limited software evidence. Separately actual native tokenizer/processor fixed5 full624 source inputs all change under T0, while actual pixels/image grid/image rows/relative positions/token lengths are exact (runs/control_expanded_preflight/summary.json,10.865s); full333 real-tokenizer permutation/maxcross-window preflight also PASS. These establish actual intervention construction only, not8B causality/metrics. Slurm lab3 fixed5/full333 samehost launchers areprepared; fullcontrol budget reviewer forecast5–8GPUhours includingfreshsources/queries/I/O, not measured.

Actual post-whole error_analysis.py read only runs/r1_full_main/{base,optimized}/predictions.jsonl, r1_full_main_decoded/{base,optimized}/predictions.jsonl, anddata/gt_4fps/HateMM.npz+HateClipSeg.npz after wholecanonicalresults. Canonicalhelpers only; neverimported bysource/scoring/fitting/thresholdpaths. Authority runs/r1_full_main_error_analysis/{summary,per_video}.json. Pairedraw within V/S/max HMM -.034701833/0/-.014337744; HCS +.021885806/0/+.015501943. Seligible82/97 finite speechframe subset, notstandard84/99. Finalwithin +.013684140/+.015892467; allnonzero pairedbootstrap95 intervalscontain0 (mainfinalHMM[-.007422558,.037551632],HCS[-.009518665,.041269455]), no statisticalsignificance claim. HMM rawV/max losewhilefixedr6finalimproves: cannotclaimdirectvisualordering gain; completecomponent/matcheddonor/time controls remainnecessary toexplain decoder interaction, no mainrevisionpredeclared fromthisdiagnostic.

Independent controls review identified concrete cost defects; fixedbeforeGPU: config/ASR failure nowinside attemptaudit try/finally, allfailedactualcalls recorded; batch_peak_GiB capturedafter allsevenarms/clones andaggregated separately fromoriginalR0 processingchecks. No source/selection/scientificchange. Independentconfirmationpending.

Selection batch deliberately executes fresh original main/source/query proof followed by seven declared arms including an additional R0 identity callback: eight real query passes. The extra verification adds about .217h to minimal stage forecast, still covered by original5–8GPUh allowance. All extra forwards/diagnostics/actualtiming are charged; no silent seven-pass cost claim. T0/H0 each load/rebuild separately, never hold incompatible denseKV together. Deletion/binding gates follow frozen Rule14(g)/ABLATION_PLAN on any common primary across BOTH; report main-gain concordance separately as diagnostic, not new gate.

One independent controls code review PASS (same-family provisional): docs/reviews/20261006_m1_rekv_controls_code.md; full actual request/decisions/commands/source snapshots under runs/controls_code_review/traces/controls-code-review/. Reviewer independently checked160D0/56T0 exhaustive cases, actual36-layer FP32/BF16 timestamp-aware fixture withnonemptyS/noLOCAL/exception/native restore, actualT0 sourcekeys andmargins change withIDs/packing fixed, realnativeprocessor all624inputs change/609crosswindows/pixels/grid/tokenbudget/positions exact. Atomic/resume/config+ASR failure audit andpostcallbackbatchpeak fixes confirmed. NoGT/GPU/pretrained quality claim; unchangedfixed5 actual8B required next.

Originalfixed5 actual8B controls submittedsc474398/Slurm230 after onceindependentcontrolsPASS andall4operativecodeagreement/currentidlelab3/1.3Tfree/unchangedforeignwork+STRAY noted in machines_before_controls_smoke{,_note}.txt. Source/native/originalmain proofs alreadysamehost; no slicing/splice/GPUbypass. Selection/time/history sequentialsamejob withfreshKV, normaluserbudget ROOT226otherallocatedGPU/193held. Full333controls notsubmitted beforeactual guard; noGT/metrics/mechanismyet.
