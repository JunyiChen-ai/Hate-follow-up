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

Actual8B originalfixed5 fullcontrols/sc474398/Slurm230 DONE2026-10-06 18:49:02,7:23/0:0. ImmediatelyBOTH returnedROOT(1.236GB newrunpayload; originalsourcecacheunchanged/resynced); strictROOT noGT allthree modes PASS runs/controls_smoke_analysis/plumbing_summary.json. FreshR0sourcekeys/query/selection/margins/nativeallraw exact; eachnonidentityarm changesall158V margins, nativeG/S exact; all50clones exact. ActualT0/H0 sourcekeys change inall5, T0609sourceblocks cross8sownership andactualselectedrolechanges22532/45216blockreads; matchedD0fullreplacement5045/5688 remote-layerchoices. PeakGPU18.175344GiB; selectionbatch212.014s/time83.431s/history67.089sprocessing (source decoding/newvideo nativeacquisition separatelycharged; actual7:23 wall andROOTstrictprepare separate). Full333 controls ready, noGT/controlmetrics or mechanismestablished. Additional actualinput-sensitivity summary runs/controls_smoke_analysis/actual_intervention_summary.json.

Full333 allrequiredcontrols submittedsamehostsc474398/Slurm231 after actualfixed5/noGT strictPASS andcurrentall4codeagreement/unchangedforeign+STRAY/1.3Tfree/idlelab3. Selection/time/history sequentialsamejob, full215/118 source/native/currentR0proof availableonsamehost; freshpervideoKV/releasedafteratomicallarms. No sharding/dependency/jobchain/GT; originalconstants/method/gates retained. CurrentROOT226othernormalbudgetGPU/193held; no bypass. All9arm/all6canonical/raw metrics andexposure/costs must be kept. Forecast5–8GPUhours, actualunknown; no mechanismclaim/goalcomplete.

control_exposure_report.py addedas standalone noGT descriptive report (not imported by source/scoring). Itreconstructs exact complete-token/image budgets, unchanged/partial/fullactualremote changes, truePTS/LOCAL roles, reference-vs-donor ancestor unions/overlaps fromcurrentfullrecords; writesstreamedJSONL toboundhostRAM. Actualsmoke39,816layer-arm records reconcile withR0 exact/budgetmatchedC1L2D0; imagebudgetsame here, ancestoroverlapsubstantial andreportedhonestly. Scope runs/controls_smoke_analysis/{exposure_summary.json,selection_exposure.jsonl}, not newefficacygate/novelty/confirmed333effect. Fullrun231modelcode unchanged.

Full333 selectionbatch R0/L0/C0/L1/C1/L2/D0 samehostsc474398/job231 stageDONE2026-10-06 21:55:12 afterwhole333source/native/actualarmproof+atomicrecords; mainT0fullsource rebuildstarted21:55:15samejob, H0pendingsequential. SelectionoutputbeingreturnedROOT; partialstagecompletion notwholejob/metrics/mechanismclaim. Allsource/newvideo/shared-vs-standalone/extrareplay/costattempts retained. NoGT/controlmetrics evaluateduntilallrequiredmodebinding completes.

Selection-only completedstage whole333 outputs/proofs/nativepixel/key/query arrays29,682,297,743bytes returnedROOT22:03, no hashes/delete. SourcecachealreadyROOTsameoriginalhost; T0full333continuesremote/H0pending. ROOTnoGT strictselectionbatch currentsource/native/R0/actualarmqueries/budgets binding started runs/controls_selection_binding/. This iscompleted-stage validationonly; no selectionmetrics/mechanismclaim orwholejobDone. Originalwholeall3mode strictprepare remainsrequired onceT0/H0returned.

Completedselectionbatch ROOTstrictnoGT binding PASS2026-10-06 22:23: runs/controls_selection_binding/summary.json coverage333/R0 exact/nativeallraw exact/currentactualsourcepixels/actualarmquery+tokenbudgets allreplayed,1058.225746s stagewall. Originalemptyheredoclaunch didnotexecute; savedprepare.py+run_corrected.log reproducibleactualentryretained. ThisPASSscope selectiononly, noGT/efficacy/mechanismclaim, T0/H0 wholejobstillpending. Originalfullmodeprepare remainsnecessary beforeallarm canonicalmetrics.

T0 wrong-presentedtimestamp complete333samehostsc474398/job231 stageDONE2026-10-06 23:07:35, actualsourceKVre-encoding/frozenR0selection/alloriginalpixels+PTS preserved; stageoutputreturnROOTstarted. H0 full333zero-directhistory withnativeglobalcontext kept began23:07:38samejob, finalwholejobpending. NoGT/controlmetrics/causalclaim; allsource/currentcosts retained, wholeall3mode ROOTstrictprepareandcanonicalmetrics stillrequired.

T0completedstage full333 outputs/proofs/nativepixel/key/query arrays20,373,793,620bytes returnedROOT23:12, no hashes/delete. ROOT noGT currentactualsource/native/presentedtimestamp/wholeinputbudget/frozenR0selection+packing strictstagebindingstarted runs/controls_time_binding/. H0full333continuesremote; noGT/controlmetrics/wholejobDone ormechanismclaim. Fullall3modeprepare remainsrequired afterfinalH0return.

T0 whole333 ROOTstrictnoGT source/native/wrong-presentedtime inputs/frozenR0selection+packing binding PASS2026-10-06 23:21–22, authorityruns/controls_time_binding/summary.json,499.344993s wall. This iscompleted input/binding evidenceonly; noGT/controlmetrics/causalclaim, H0stillwhole333remote. Allstageproofs/costs retained; finalall3modeprepare thenall9canonicalmetrics stillrequired.

Allrequiredwhole333 controlmodes selection/T0/H0 samehostsc474398/Slurm231 DONE2026-10-07 00:07:34,5:13:18/0:0. ImmediatelyfinalBOTH fullruns+originalinputcache returnedROOT; finalnewH0payload20,358,948,450bytes transfercomplete00:13(earlierselection29.68GB/T020.37GBalreadyROOT), no hashes/delete. OriginalsourcePNG/actualPTS untouched, allmodel/source/query/nativepixel proofs andatomicrecords retained; denseKVreleasedpervideo. ROOTfull3mode matchingHateVLM strictinput/native/source/intervention/cost binding thensolecanonicalall9metrics/fixedr6 started. No controlmetrics/mechanismclaimuntilcomplete; originalallnegativecosts andnativeacquisitiondebt retained.

Completeall9controls canonicalraw+fixedr6 full333 finished2026-10-07 00:50 afterwholejob231+BOTHROOT+all3strictcurrent/native/source/intervention/costPASS. Authority perarm `runs/20261006_m1_rekv/controls_main_decoded/<arm>/metrics.json`; completeall6 table/gates runs/controls_main_analysis/summary.json. R0all6 exact. Allretainedcandidatecomponent gates FAIL: L0/L1/L2/H0 have no primary deletionloss>=.01BOTH; D0/T0 no commonprimaryabove-noise correctbindingbenefit. ThisdoesnoteraseR1performancePASS, but C2notestablished, nopromotion/usergoalcompletion.

Within main-minusL0 HMM+.000651632/HCS−.006229796; main-minusL1−.000213594/−.002800852; main-minusL2+.001802072/−.000276642. Explicitremote/perlayerselection cannotbeclaimednecessary. H0withinloss+.005142572/+.014927408, effectonlyHCSabovefloor, notsharednovelty; H0pooledHCS−.012658ROC/−.009738PR, HMMpooledimprove. D0withinloss+.002668482/+.004564697 withinnoise; T0+.002343673/+.003620501 withinnoise. T0 HCSpooledROC/PRdrops.013196/.009304 butHMMROConly.000544/PRimproves.002818, so nocommonbindinggate. No roundedboundary/subset/margin-mixture salvage.

Allactualinterventions/sourcebytes/prompt/tokenbudget andnativechecks passed; weakcausaleffect cannotbeexplained away as failuretodeliver, althoughD0 inheritedancestoroverlap remainsa limitation. RawV/S/max/final/errorordering diagnosticsnowpostwhole canonicalhelpers with--controls; allGT/filepaths logged, never source/scoring/fitting/thresholds. Necessaryindependentresult-to-claim mechanism interpretationnext once diagnosticscomplete. Rule14(g): unsupportedcomponentsmustremove/downgrade, notretainednoveltyclaims. AnyR2scientificrevision requiresconcreteactualerroranalysis andsameBOTH protocol; budget0/3 untilreviseddesignrun, noR2yetdeclared.

Actualpostwholeall9rawV/S/max/final/order/ties/branch-winner canonicalhelperdiagnostics complete runs/controls_main_error_analysis/{summary,per_video}.json; GT/readpaths explicitlylisted, never methods/source/fitting/thresholds. Main-minusL0 rawV HMM−.006627757/HCS−.017134528, rawmax+.008086782/−.014162833, whilefinal+.000651632/−.006229796. Main-minusH0 rawV−.030831379/−.032497006, rawmax+.020413376/−.007018961, final+.005142572/+.014927408. Thus neitherdirectrawVimprovementnorcommonhistorycontributionisestablished; decoderinteractions/contrarymovements reported. NativeS allarrays/ranks exact acrossarms, R0allraw/finalexact. Requiredsecondary result-to-claim interpretation ofcompletedcontrols nowrequested, not anotherproposal/process/code-review orautomaticstop. Needactualsourceerroranalysis beforeconcreteR2; no R2predeclared/goalcomplete.

2026-10-07 completed-controls independent judgment is **partial: C1 yes, C2 no**, same-family provisional. Report and full independent recomputation/traces: `runs/20261006_m1_rekv/mechanism_claim_review/`. It independently reproduced all 66 final metrics and exact R0/native G/S/stance. Unsupported explicit retrieval, per-layer selection, history necessity and correct-binding claims are removed from the current supported claim; R1's descriptive development-selected final performance improvement remains. This is not a promotion or goal completion. The sole recommended follow-up is the original priority-2 H0+L0 factorial cell, not k/layer sweeps or a dense-only novelty claim.

Post-whole actual source examples read main/L0/H0 records and both GT arrays; exact paths and observations are in `runs/20261006_m1_rekv/controls_main_error_analysis/source_examples.json`. HateMM `hate_video_151` last window (GT-negative) native V −2.186878 becomes main +14.654293 / L0 +17.406372 / H0 +11.892506. HateClipSeg `bit_Ydtmd4TNPmKl` [184,192) (GT-negative) native V −2.962082 becomes main +4.324009 / L0 +3.237038 / H0 +5.282963. Explicit remote and/or source ancestry deletion does not remove these particular high late scores. These are observations, not proof of a single causal explanation; native whole-video context, appended LOCAL content and the reader suffix remain. Labels remain outside source/inference/fitting/threshold paths.

**H0+L0 (HL0) frozen before execution, 2026-10-07:** use zero source-to-source history AND zero explicit remote reads. Preserve every true 0.5fps LOCAL block, real pixels/PTS, source role text, native full context/G/own stance/full ASR/native S, original model/constants/max/fixed r6 and original 215/118 cohort. No new method revision is spent: this is the already planned conditional interaction diagnostic, not revised R2. Contrasts: L0−HL0 tests history without explicit remote; H0−HL0 tests explicit remote without history; (main−L0)−(H0−HL0) is descriptive interaction. It cannot retroactively satisfy failed original component gates or certify dense-only novelty.

Implementation `history_local.py` freshly rebuilds H0 source KV once/video, performs unchanged H0 query replay with exact keys/queries/selections/raw margins versus the completed same-host H0 records, then performs independent L0 queries before atomic complete-record persistence and dense release. The helper's optional arm subset defaults to the old full seven-arm behavior. Full fresh native checks, H0 replay, extra HL0 queries, smoke clones, source decoding, model forwards, validation/I/O/failed attempts and standalone costs are recorded separately; original native20/ASR acquisition is additional. Previously released dense KV is not reusable. Forecast **1–2 allocated GPU hours** for whole333 including fresh rebuild, H0 replay, validation/I/O; not measured. Same sc474398/lab3, original fixed5 first, one independent narrow observation-code confirmation, then whole333 only after actual strict noGT guards. Canonical evaluator and fixed r6 only. Launch `launch/lab3_history_local.sbatch` with SCOPE=smoke/main; ROOT strict/canonical `launch/run_history_local_analysis.sh`. No HL0 GPU/result yet.

HL0 one independent narrow delta confirmation PASS, `docs/reviews/20261007_m1_rekv_HL0_delta.md`, full traces and own fixtures in `runs/20261006_m1_rekv/history_local_delta_review/` (same-family provisional). Four actual36-layer FP32/BF16×native18/20 cases each include nonempty S/noLOCAL, clone/KV/rope/method restore, exact fresh H0, L0subset=old default L0, ancestry/remote/call/H0margin corruption refusal. Independent all333 current actual rawPTS/PNG/native/runtime/H0 keys/queries/selection proof replay PASS,28895 source/7missingLOCAL,522.824s. Atomic/resume/failure costs/postcallbackpeak/outputnames/launch and canonical-argv guards PASS. No GT/metrics/GPU were read/run by reviewer; author fixture path-harness failure retained and fixed, no production bug. Original fixed5 actual8B next; conditional diagnosis does not establish efficacy or restore failed novelty.

HL0 originalfixed5 same sc474398/lab3 Slurm239 submitted2026-10-07 01:16–17 after independent narrowPASS/authorCPU/current333H0source bindingPASS and all4 code agreement/ROOT foreignwork untouched/lab3idle1.2Tfree. Current STRAY entries are unrelated work; lab-server listing changed by others, no task files outside repo. Source+H0 replay+L0 reader complete5 samehost, originalcache/proofs readonly, no GT/performance. Normal two-active userGPU budget with ROOT238; held193 unchanged. Standard wait_HL0_smoke.log matches remotejob disappearance/Traceback/FAILED; full333 remains unsubmitted until realfixed5 and ROOT strictprepare pass.

Actual HL0 originalfixed5 samehost/sc474398/Slurm239 DONE2026-10-07 01:18:31,1:47/0:0. ImmediatelyBOTH run/input returnedROOT (newrunpayload372,405,977bytes); unchangedactual0.5fps cache/PTS/RGB retained. ROOT matchingruntime strictprepare PASS, runs/history_local_smoke_analysis/{plumbing_summary,actual_intervention_summary}.json. Allfresh H0 source keys/query/selection/native/raw margins exactly reproduce prior samehost full H0; trueLOCAL-only/no direct/inherited history verified, nativeG/S/stance/allraw exact, all158newV changed and all158HL0 V differ from H0,10clones exact. Source624 actualframeprefills, batch1257LM/629vision forwards,81.407173s processing (including freshH0/nativechecks+extraHL0), peak18.117209GiB; standalone88.512194s includes originalsource decoding, actualallocated1:47 and validation/setup/I/O recorded separately, originalnativeacquisition additional. Only execution evidence, no GT/HL0metrics or mechanism. Whole333 samehost now ready;1–2GPUh still forecast; no R2 revisionbudgetspent.

HL0 whole333 submitted same sc474398/lab3 Slurm2402026-10-07 01:22–23, after originalfixed5 real8B/BOTH ROOTreturn/strict noGT PASS and current all4 code agreement/foreignwork preserved/lab3idle1.2Tfree. ROOT238 othernormaluserGPU, held193 untouched. Fresh H0 source rebuild+exact H0 replay+HL0 independentquery for all215/118, onehost/no sharding/probe/GT/source edits; all source/standalone/batch/native/extra replay/failure/validation/I/O costs retained, native acquisition additional. Original diagnostics/negativearms and thresholds unchanged, no R2science revision. Standard wait_HL0_main.log matches job disappearance/Traceback/FAILED and parent verifies real DONE; full results must returnROOT before STATUS numerical updates/canonicalmetrics. Forecast1–2GPUh not measured.

HL0 whole333samehost/sc474398/Slurm240 DONE2026-10-07 02:34:57.639,1:12:12/0:0. ImmediatelyBOTH returnROOTstarted02:35:29, original0.5fpscache inputrsyncdone andnew~20GBoutputs transferinprogress. FreshH0 source/replay/HL0 pervideo completed, fulljobcost/proofs retained; noGT/HL0metrics orconditionalmechanism verdict beforefullROOTreturn/strictprepare/canonicalfixedr6. Original9arms/mainknownfailure remains; thiscellcannotretroactivelypromoteoldretrievalclaim.
