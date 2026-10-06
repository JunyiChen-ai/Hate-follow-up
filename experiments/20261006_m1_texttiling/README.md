# Candidate38: lexical discourse partition and source-scoped speech

Selected from the unchanged 20261006 nine-candidate pool, rank2. Its one
independent Rule4 PASS is `docs/reviews/20261006_m1_ideation_jury.md`;
same-family provisional. This is not a new review or a reset of Acoustic20,
Lattice22, QuoteGraph24 or Provenance25. No main predictions/GT or new ASR
were read to choose these constants. All eventual comparisons are
development-selected. The complete source/reader/strict replay/canonical CLI
and Slurm prototype is implemented; actual8B/Whisper measurements and its one
Rule6 independent review are still required. No GPU experiment is authorized
by a CPU PASS alone.

The hypothesis is that a literal act should be attributed to its actual8s
window, while a lexically connected utterance elsewhere can explain it.
Unrelated-topic attacks should not become direct evidence for this window.
This replaces S only: native20(actual18–20), full original ASR, original G,
its own hard Yes/No stance, independent native V, max(V,S), fixed r6 and
canonical4fps are unchanged. Both fixed datasets use the same pipeline.

## Sources and declared adaptation

Actual primary reading on2026-10-06: [Hearst ACL1994 algorithm section](https://people.ischool.berkeley.edu/~hearst/papers/tiling-acl94/acl94.html),
tokenization/block cosine/one smoothing/depth/spacing; [official Whisper timing
implementation](https://raw.githubusercontent.com/openai/whisper/main/whisper/timing.py),
alignment heads, teacher-forced cross attention, normalization/median/DTW/word
boundaries and later heuristics. Source scope saved under this run's
source_reading/. The implemented Unicode/no-stop-list/mean-depth/no paragraph
snapping is a declared adaptation, not a reproduction of the paper's hidden
image threshold. The source parser and local speech measurement are our
composition. DTW estimates time; it does not verify an act's semantics.

## Frozen R1 specification before GPU

`spec.json` is the machine-readable authority. New ASR uses the same frozen
Whisper-large-v3, greedy beam1, max448 total decoder tokens, transcribe/no
timestamp output and one language identification on first actual audio.
Actual audio is placed at resampled16000Hz PTS relative to the video's origin;
overlap keeps the first observed sample, gaps are zero at their original times,
not compressed. Non-overlapping nominal30s blocks cover actual video time.
Empty audio blocks do not create words. Each block generates a separate single
transcript and teacher-forces those actual tokens for official-head DTW at20ms.
Save raw DTW and used clipped times separately; no subsequent duration/median
word heuristics, no proportional segment word times. A capped ASR block remains
explicitly incomplete and its observed token prefix remains attributable; no
retry or continuation invents missing words. Distinct words on different block
sides retain distinct IDs even if their literal text repeats; no transcript
overlap is introduced. Word assembly follows actual HF original-token groups:
whitespace/punctuation for spaced languages, Unicode for zh/ja/th/lo/my/yue.
No punctuation merge changes their times. HF eager attention already contains
softmax probabilities, so normalize it without a second softmax. Use the
Whisper generation-config official10 alignment heads, FP32 per-head weights
normalized across all teacher input rows, width7 reflection median, then mean
heads and rows from no-timestamps through the last content token (exclude final
teacher EOS). Zero standard deviation becomes zero normalized weights. Crop
to ceil(actual samples/320) encoder frames at20ms, bounded by encoder length.
DTW costs areFP32, official CPU strict diagonal/strict up/otherwise left ties;
save raw jumps before group lookup. Cumulative word token boundaries index
those jumps; clip only to actual crop endpoints and retain raw endpoints too.
Teacher EOS closes alignment when it fits the448 decoder capacity. A capped
prefix4+content444 preserves every actual token and teacher-forces those448
rows without an extraEOS; its terminal content query supplies the final DTW
boundary. This explicit capacity adaptation normalizes the actual448 rows
and does not silently drop the last word. Teacher EOS is never a generated
word or evidence. These details are fixed before source extraction.

Lexical sequence: Unicode letter/number runs with internal apostrophes,
casefold, no stemming or stop list. A source word with multiple lexical runs
contributes each run but preserves the same original word ID and midpoint.
Pseudo sentences contain20 lexical runs, including a final short sentence.
Only gaps with6 complete pseudo sentences on each side have a cosine;
shorter documents are one segment. Cosine uses raw frequency vectors, zero
norm returns0. Smooth once with the available adjacent gap scores in width3
(no invented padding). At each gap follow nondecreasing scores left/right
until the first decrease to find peaks. Depth is both peak differences.
Eligible valley plateaus choose their earliest gap; flat zero-depth plateaus
do not split. Retain depth strictly above the mean of all valid gap depths.
Visit depth descending, tie early; accepted boundaries must be at least3
pseudo sentences apart. Do not cut an original source word even if a multi-run
word crosses a pseudo boundary: advance to the next original word ID, dedup
resulting identical boundaries. Segments partition original words once.

Each8s half-open window owns source words whose used time midpoint falls
inside; exact8s is owned by the next window, a midpoint at/after manifest
duration is explicitly out of scope. Every lexical segment touched by a LOCAL
word contributes its other original words as CONTEXT. Keep IDs/times/order;
no fixed neighborhood is substituted. More than2048 source words or6144
encoded parser input tokens rejects the whole parser packet with a recorded
reason; nothing is silently truncated. Grammar-selected span endpoints refer
to actual word IDs, up to2 spans/role, each at most32 original words, same
role and contiguous supplied original IDs. UNKNOWN/invalid/incomplete parser
uses a fresh direct-aligned-LOCAL S; UNKNOWN never means no hate. No LOCAL
words uses a fresh original S if the original window has speech, otherwise
keeps native absent-S. Source model selections can be wrong even when IDs are
valid. Execution guard requires at least one compiled LOCAL act in each
dataset, exact native allraw/G/V/stance, real input/times, and untouched cloned
native cache before a full run; no GT or score is used in this guard.

The final new S is a fresh independent branch of its own native stance cache.
The original speech question/window coordinates remain verbatim after the
declared source instruction and a literal compiled record. Native prefix/full
ASR is not edited. This record lists accepted LOCAL acts and explaining
CONTEXT with actual times, never relocates the latter to the current window.
Fresh fallback also remains independent of V. One window still emits one S.

## Cost and falsifiable mechanism

For B=ceil(T/30), Ns aligned-speech windows: B greedy ASR generations and B
teacher-forced alignments, one language detection, up to Ns grammar parses and
Ns new S measurements; V/G stay native. Original ASR/native20 acquisition and
all new audio decode/transfer/Whisper/parse/branch work are charged for a new
video. Paired native S, repeats and controls are experimental overhead and
reported separately. Initial unmeasured5090 estimate for120s is60–240s added
work, plus native costs; long-topic parser prefill may be more expensive.
No claim that frozen/cached extraction removes this deployment cost.

Claim one complete part: lexical partition scoped LOCAL-act/CONTEXT compiler.
Deletion uses the exact same new words/times and fresh direct LOCAL S, so
better ASR alone cannot prove the claim. Additional matched-nearby-context,
uncompiled-role, circular wrong CONTEXT and swapped LOCAL/CONTEXT interventions
are mandatory before mechanism success. No new corpus, ensembling or hate
score postprocessing. Main gates compare all6 final metrics to
`runs/20260926_twolevel/r6_bma/metrics.json`: one same primary metric >=+.01
on BOTH, pooled losses<=.005, within losses<=.01. Claimed part deletion must
lose one common primary >=.01 on BOTH, with real raw V/S/max ordering and
wrong binding controls. Whole333 and all sources must run on one host and be
returned to ROOT before any STATUS claim. No actual performance exists yet.

## Software evidence before GPU

`runs/20261006_m1_texttiling/operator_cpu_checks/summary.json` validates the
frequency cosine against independent dense vectors, flat/short/empty streams,
an800-word two-topic boundary, Unicode and original-word coverage, exact8s
midpoint ownership. `alignment_cpu_checks/summary.json` matches15 official
DTW path examples, exact official-head probability normalization/median
matrix, zero-std/short-tail and actual Whisper tokenizer literal coverage.
Initial test-only short-token endpoint assumption was corrected; failed log
preserved, no scientific timing operator changed.

`compiler_cpu_checks/summary.json` validates actual Qwen tokenizer prefix-free
span options, shared Stream replay, wrong role rejection and frozen question
suffix; initial fixture incorrectly expected context beyond its own topic
boundary, corrected to one-topic fixture, failure kept. No compiler algorithm
or execution guard changed. `whisper_cpu_checks/summary.json` exercises actual
HF32 encoder+32 decoder layers/20 heads with official10 heads and random
narrow weights on1s synthetic features; not pretrained recognition or ASR
truth. All are CPU implementation evidence, not actual8B or performance.

One fresh independent Rule6 instance is reviewing the final prototype and may
write its own meaningful CPU evidence under `independent_code_review/`.
Actual333 raw/native/real processor preflight is running; future source word
capacity and actual source semantics are not thereby validated. No new source
cache has been generated, no GT read, no main result and no budget reset.

Independent Rule6 confirmed only B1 capped teacher-capacity and B2 incomplete lexical block fixes. No content tokens were dropped and no execution guard changed. Actual narrow random-weight HF production Recognizer.block greedy reached448 tokens/retained444 content/445 DTW rows; actual36-layer random Qwen production reader/parser/validator exercised fresh S, ten complete native KV restorations and wrong role/source text/native pixel rejection. Evidence runs/20261006_m1_texttiling/independent_code_review/{summary,fix_summary,cost_summary}.json. This is software evidence only, not actual pretrained8B/ASR semantics/performance. Actual333 raw/native/real-processor preflight PASS runs/full_input_preflight/summary.json. The one Rule6 review is PASS at docs/reviews/20261006_m1_texttiling_code.md (same-family provisional). No GPU started; actual fixed5 is next.

ROOT deployment activation defect found before execution: originalowned223 remainedPENDING and was canceled; lab1 now activates actualrepo .cache/envs/HateVLM venv, matching torch2.11/HF5.15/av18.1. Same independent reviewer narrowly confirmed launcher only, originalreview append and launcher_fix_* evidence. No scientific choice/spec/source/reader changed, no GPU/GT/performance, budget0/3 remains. Fixed5 will be resubmitted from synchronized correctedcode.

Corrected unchanged fixed5 submittedROOT/sc474397/Slurm224 afterfourlab code agreement, idleROOT/437Gfree, priorforeignSTRAY unchanged andactualenv narrowPASS. 223 remainednever-executed/canceled; no new method version. Full333 stays forbidden untilsamefixed5 actualsource/native/currentinput/clone/compiledboth guardPASS.

Actualoriginalfixed5/sc474397/Slurm224 DONE15:40:36 andROOT noGT preparePASS at runs/20261006_m1_texttiling/r1_full_smoke_analysis/plumbing_summary.json. Allsource audio/DTW/word/time/grammar/nativepixel/runtime/native allraw/G/V/stance/clone andcompiledboth guards passed. Realpretrained ASR and8B executed, but literal ASR/act selections are not proven semantictruth. Complete333samehost isnext; no GT/mainperformance/mechanism. SourcePROVENANCE updated toactualhost/stage evidence.

SameR1 actualfixed5 strictPASS之后完整333 source+reader提交ROOT/sc474397/Slurm226，NORMAL QOS排队等待原186和ReKV225；同host已有5source复用但原成本保留，其余fresh提取/全333新读取。机器一致性/容量/foreign未改证据 runs/20261006_m1_texttiling/machines_before_main{,_note}.txt。无主指标/机制。

Whole333 originalR1/source+pairedreader ROOT/sc474397/Slurm226 DONE2026-10-06 19:53:19,1:27:08/0:0. FullsourceDTW/greedyASR completed18:48, native+Sreader thenall333; mainmethod/constants unchanged. Allinputs/runs alreadyROOT; matchingHateVLM18 strictcurrentaudio/source/compiler/native/time/processor prepare followedsolecanonicaleval+fixedr6 started. No mainmetrics/claimuntilcomplete; no sourcecosthidden.

CompleteR1 whole333 ROOT226 strictcurrentaudio/DTW/source/actualcompiler/time/nativeallraw/G/V/ownstance proof andcanonicalall6 baseexact PASS. Authority runs/20261006_m1_texttiling/r1_full_main_decoded/optimized/metrics.json: HMM ROC/PR/within .897858055413/.681488129264/.767683886808 (84), HCS .722572216403/.675367316904/.647693960425 (99). DeltasHMM+.000739394/−.012746474/+.016901843; HCS+.005747352/+.004295233/+.010344847. Samewithin>=+.01bothpositivepreserved; HMMPRbeyondnoise→overallperformanceFAIL. Rule9qualifiesactualGTerroranalysis/up to3revisions budget0/3; no controlGPU/mechanism/promotion.

Postwhole error_analysis.py canonicalhelperrawV/S/max/final ordering diagnosticsnowreadall333 raw+decoded native/optimized anddata/gt_4fps/HateMM.npz+HateClipSeg.npz onlyaftercomplete; source/scoring/fitting/thresholds neverimportit. Exactreadpaths/statistics runs/r1_full_main_error_analysis/summary.json, novelty notinferredfromsource execution. R2notyetdeclared; actualerroranalysisbeforeanynextdesign,37fullmechanism remainspriority.

Actualpostwhole rawpairedwithin V/S/max HMM0/−.007357597/−.001762262, HCS0/+.008246798/+.006939113; final+.016901843/+.010344847. Allnonzero pairedbootstrap95 contain0, no statisticalsignificance/generalization/causalclaim. Descriptive parser-status analysis readall333 R1 records/packets/traces plusrespective4fpsGT onlyafterwholecompletion; sourcefiles/GTneverenterscoring/fitting/thresholds. Authority runs/r1_full_main_error_analysis/{parser_status_groups,parser_status_windows}.json. HMMcompiledGTpositive marginmean−5.1545 vsGTnegative−1.2759; uncompiledtoken-cap GTpositive−4.5644 vsGTnegative−.2398. Thus bothtypedrecords anduncompiledfallbackcanattenuatepositiveevidence; no assumptiononefixmustwork.

Actuallyinspected HMM hate_video_330 window112–120s andhate_video_247 window192–200s R1 records/packets/nativeASR/newquestions; bothGTpositive, parser unavailable duefullinput8823/6524tokens exceeding6144 cap, yetR1 injectsalignedwordlist intoS. NativeS11.823/11.422 falls−4.630/−4.747. Exactreadpaths andGTfractions arealreadyin parser_status_windows.json; originalpacket/newquestion remainsinmainrecords. Theseareexamples, notwholemechanismproof.

R2 predeclared revision1/3: applysourcecompiledS onlywhenpacket.reason=='compiled' (validliteralLOCAL act); otherwise freshnativeS exactly, includingcap/UNKNOWN/no-local/incomplete cases. Keepallsourcewords/greedyASR/DTW/lexicalpartitions/parserprompts/caps/choices, compiledrecord/newquestion andnativeG/V/fullASR/ownstance/max/fixedr6 unchanged. Availabilitydependsunlabelledsourceparserstatusonly, neverhate margin/GT/corpus; UNKNOWN isno newevidence, not no-hate. SameflowBOTH, freshfullR2 sourceparser andreader, nohistoricalpredictionmixture. Expect removingunavailablewordlistaugmentation mayrecoverHMMPR whilevalidcompiledspans preservewithin; unknownuntilwhole333. Originalsource21.064951min charged; parser HMM59314/HCS52689=112003forwards/40.503809min reruns andisnotzeroed; no newaudio/ASR/DTWgeneration. CPUdefaultR1identity/sourcecaps/nativefallback/state/cost andoneindependentnarrowdelta confirmation thenoriginalfixed5 beforefull333.37fullmechanismpriority; noR2implementation/GPUyet.

R2 implementedstatus-only source availability policy. Authorall333/7359 realpacket questions PASS runs/r2_reader_cpu_checks/question_summary.json: defaultR1 readablebefore_R2 snapshotexact, compiledquestions unchanged, alluncompilednativequestions exact. Actualnarrow36layer randomFP32 Qwen/8Q2KVx8 +actualStream forwards compiled/UNKNOWN/noLOCAL/absentnative speech PASS model_summary.json: oldR1 numerical/source behavior exact (walltimesexcluded), compiledS R1exact, unavailableS nativeexact, packets/parsercounts/nativeG/V/state/clones unchanged, revisionmutation rejected. Initialtestimport-path andrealgenerationtiming equality failures retainedwithcorrectedlogs; no sciencechange. Fixturedoesnotprovepretrainedsemantics/GPU; onceindependentnarrowdelta inprogress, noR2GPU/GT/metrics. Runtime/nativebindingimplementation unchangedintentionally; separateR2 reader markers/names failclosed. OriginalASR/DTW/source/parsercosts preserved.

CompleteR1 processing-stage estimate78.1895917min includingoriginalsource21.0649513min andouterparserwall40.5038088min; pairednative9.2500587min (ratio8.452875). This isnotcompleteend-to-endfuturevideo latency: originalnative20/ASR acquisitionstilladditional; actualSlurm1:27:08 andROOTstrictprepare250.874s separate. Parsergeneration-internal seconds sum2405.083792 (40.084730min), acontainedsubstage ofouterparserwall, notaddedtwice. Exactcalls ASRencoder4509/decoder141805, sourceparser112003; allR2originalsourcecalls/cost+rerunparsercount maintained.

R2一次独立窄delta确认PASS docs/reviews/20261006_m1_texttiling_R2_delta.md（same-family provisional），own完整request/response/command trace runs/r2_delta_review/。独立8actual36layer FP32/BF16×nativeS present/absent×compiled/UNKNOWN inclnoLOCAL与全部KV/rope/clone/成本/host/revision mutations；all333/7359真实tokenizer+grammar/partition/scope与R1snapshot/R2question检查PASS。2129compiled/320UNKNOWN/1300noLOCAL/3610inputcap，源1263.897080s、112003freshparserforwards保留，内层generation与外层parserwall不重叠相加；启动/命名/guard/prepare失败阻断通过。只未标注sourcepacket投影、不读GT/预测/metrics、无GPU/生产code改动。真实原fixed5 ready，完整333仍须actualguard后提交，revision1/3；37全机制优先。

R2 actualoriginalfixed5 submittedsameROOTsourcehost/sc474397/Slurm236 afteronceindependentnarrowPASS/currentall4codeagreement/unchangedforeign+STRAY/382Gfree. OriginalwholeASR/DTW sourceonlyread, freshparser+S, no sourcegeneration/rebudget. Normalqueue behind235/231, no override/dependency/jobchain; originalsource/parsercost obligations unchanged, revision1/3. Full333R2notsubmittedbeforeactualguard; noR2GT/metric/mechanismyet.

R2 actualoriginalfixed5 sameROOT/sc474397/Slurm236 DONE2026-10-06 20:33:31,1:29/0:0; input/runsonROOT, sourceunchanged. MatchingHateVLM strictnoGT preparePASS runs/r2_full_smoke_analysis/plumbing_summary.json; directactualcomparewithfullR1 sameparserstatus/selection, compiledS/question pervalueexact, noncompiledS=native/None, nativeG/allraw/V exact. Actualsource/audio/compiler/3axis/nativebinding/cost/5clones andcompiled+changedS BOTH guardPASS, directidentity R1_R2_actual_identity.json. Full333 sameROOThostnext; originalASR/DTW/source21.06495min andfreshparserallcosts retained, no R2GT/mainmetrics/mechanismyet.
