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

## Native runtime and unchanged source grammar capacity

Root Slurm145 completed2026-10-05 11:24:58: actual fixed5 current native G/own
stance/allV/S windows exact against frozen base_gridA, noGT. Authority
`runs/_setup_local_hatevlm/native_smoke/summary.json`; isolated runtime now usable.
Source-only all333 menu audit found largest root26310 unique speech candidates
(old prefix guard would enumerate692216100 ordered pairs). Guard-only lexicographic
adjacent check retains exact accepted/rejected options and token/forward/event
semantics; independent narrow equivalence PASS
`docs/reviews/20261005_structured_choice_prefix_guard.md`, actual26310tokenizer
menu and54241ordered cases plus realBF16 CPU generation. No scientific input,
constant or output schema changed; not a result-guided version. This arithmetic
bound is not an old-loop GPU/CPU timing measurement.

## Current dispatch2026-10-05

Actual root native runtime145 PASS precedes scientific fixed5 submission147 onsc474397. Slurm147 is PENDING(QOSMaxGRESPerUser) behind running137/146; no scientific execution or semantic result inferred. Machine evidence `runs/20261005_m1_interval_witness/machines_before_smoke.txt` and `_note.txt`.

Scientific fixed5 Slurm147 completed13:37:39 onsc474397. Root noGT prepare PASS,
`runs/20261005_m1_interval_witness/r1_full_smoke_analysis/plumbing_summary.json`:
all5/native allraw exact, all158V/134S changed,10actual repeated reads exact.
Actual leaves96HMM/62HCS, parents93/60, repairs0/0, token caps0. Present leaves
14/2, UNKNOWN79/52; no unresolved parents. These actual5 did not exercise repair;
no forced disagreements or lowered guard, and no semantic truth/repair benefit
claim. Global read changes in the declared new-tree method; native remains exact.
Standalone739.085634sHMM+278.959479sHCS;25.24/21.07GiB peak. Actual long H114
alone681.00s includes178.21s source. Smoke rough full378.38min excludes the long
H114 extrapolation and is only a rough input-size estimate; actual full333 cost
must be reported. No GT/main metrics or mechanism controls yet. Full tree
capacity beyond these5 is unverified; retain all actual sources/UNKNOWN/costs.

## Full333 launch2026-10-05

Runhostsc474397; `sbatch --export=ALL,SCOPE=main experiments/20261005_m1_interval_witness/launch/lab1.sbatch` submittedSlurm151 afteractualall4clean/samecode andexactunchangedforeignSTRAY comparison. Localauthority `runs/20261005_m1_interval_witness/machines_before_main_clean.txt` andmatchingnote. InitiallyPENDING QOSMaxGRESPerUser with146/150running; no schedulerbypass. SameR1/sourceguard/constants asactualfive147, all333 andbothcorpora ononehost. Scientificmainmetrics/GT notread; largestfulltreecapacityunmeasured andactualfailurewillberetained.

## Execution failure2026-10-05, not idea result

Slurm151 completed116 whole atomic video inputs, then OOM in source parent full-text prefill at15:36 (first incomplete next row). No mainGT/pairedmetrics. Initial stack retained `runs/20261005_m1_interval_witness/prefill_mlp_fix/initial_151_oom.log`. Predeclared memory-only fix: for text parent generation wrap original tokenwise MLP in4096-token row batches; complete attention/input/positions/KV/grammar/greedy tokens untouched, one model forward still one. No truncation/source filtering/new scientific revision. Existing metadata retained/currentvalidated. Author actual36layer FP32/BF16 oracle and separate narrow independent equivalence review required before GPU resume; actualGPU fixedsmall-parent source comparison required.

内存修复作者检查PASS：`runs/20261005_m1_interval_witness/prefill_mlp_fix/author/summary.json`，真实36层FP32/BF16完整hidden/KV逐值相同，实际4097-row BF16逐值同、FP32最大7.45e-9/容差1e-6。原始FP32严格逐值断言失败日志保留，未改来源/数值分数门。唯一独立窄确认`docs/reviews/20261005_m1_interval_mlp_memory_fix.md` PASS，独立full-model FP32舍入差异已明报，BF16 exact。实际8B parent<=16000跨4096源生成与原保存token逐值重放脚本已准备；GPU结果与原失败大父节点容量未证明。

实际8B来源检查sc474397/Slurm166 PASS：HMMH114父节点13395token与HCSbit_0EH父节点14445token，原始完整未分块生成逐值复现已保存token，4096-row版本再逐值复现相同50/48生成token、events与compiledrecord。来源`runs/20261005_m1_interval_witness/prefill_mlp_fix/gpu/summary.json`。没有新GT/标签，sourcecache未改。原失败更大父节点的实际容量待相同full333续跑确认。

Samefull333R1 resumedsc474397/Slurm168 afterallfourlabs a9245b8clean/actualfixedparentGPU PASS/rootidle476Gfree/167soleactiveGPU, exactforeignSTRAY namesunchanged. Evidence `runs/20261005_m1_interval_witness/prefill_mlp_fix/machines_before_resume{,_note}.txt`; original116atomic whole sources replayed, no partial source result accepted, mainmetrics pending.

Slurm168已在18:31:34越过原失败第117视频HMMnon_hate_video_356，完整125叶/124父节点：最大实际父input68908tokens，sourcepeak29.7964GiB，处理300.725s；完整新metadata已严格通过。原始catalog不截断，常数/限制不变。此前CPU/8B小父等价检查支持内存修复，本次只证明此原失败大节点已成功，不是完整性能结果。


2026-10-06完整333来源的独立成本汇总已从本机每video metadata直接解析，不读GT/预测/metrics：`runs/20261005_m1_interval_witness/source_cost_audit/summary.json`。HMM215 source137.9096020220294min、3768leaf/3553parent/4repair；HCS118 source131.28529595768583min、3591leaf/3473parent/7repair，总269.1948979797152min。实际source forwards373788/355371、vision3772/3598、peak29.7964/22.2036GiB。此处只是完整来源获取，原缓存获取成本全部保留；配对reader尚未完成，不是完整部署成本/性能/机制证据。完整Slurm墙钟另包含输入重放/审计/配对及IO，不以冻结/缓存掩盖新视频成本。


完整333 source+paired在本机sc474397/Slurm168于2026-10-06 04:20:55 DONE，Slurm COMPLETED9:58:58/0:0。旧HateVideo(av17.0.0) CPU prepare在6个HCS原始container origin=-.007（saved0）处严格失败，尚未GT/eval，原log保留`runs/20261005_m1_interval_witness/audit_runtime_fix/`。生成/推断HateVLM av18.1.0全333 header origin精确匹配；仅launcher prepare切换同生成runtime，evaluate/fixedr6/report仍原HateVideo，不改方法/缓存/预测/GT/评测器/对齐tolerance/成本/修订。独立窄确认PASS `docs/reviews/20261006_m1_interval_audit_runtime_fix.md`：6真实视频40,083帧metadata/383源pixels全部原严格validate PASS、旧runtime复现FAIL、prepare失败阻断GT评测保留。整轮prepare已按正确runtime重启；尚无main指标。


## R1完整六项与实际error analysis：development-selected

权威 `runs/20261005_m1_interval_witness/r1_full_main_decoded/optimized/metrics.json`：HMM ROC/PR/within .877271805178758/.6458885274621576/.7671560289673688（84），HCS .6914757417786734/.655757374846682/.6287608107712183（99）。HMM within+.016374保留（pairedCI跨0），pooledROC/PR−.019847/−.048346；HCS −.025349/−.015315/−.008588，主门FAIL。原生全部raw/六项精确，原4fps评测/r6未改。实际新处理18766.61/16722.18秒合计591.48min，包含原source269.19min及全部newG/V/S图像处理；不是只报cached推断。源repair4/7、状态改变3/7真实保留，无机制声明。

原始V within+.016359/+.036839（CI均跨0），rawmax−.032333/+.004466，HMM rawmax CI全负；S共享帧−.008697/+.004967。不是主门的替代指标。

本轮GT error analysis实际读取：`data/gt_4fps/{HateMM,HateClipSeg}.npz`、本轮base/optimized raw predictions、decoded optimized metrics和`r1_full_main_analysis/per_video.json`，全部文件清单/发现写在`runs/20261005_m1_interval_witness/r1_error_analysis/summary.json`。HCS全局符号从正确到错误25、反向6；HMM反向37、变差3。不能仅从联合pooled下降断言G单独致因；视觉/语音/全局作用需实际分开跑。观察用于下一设计，不进入任何方法计算/拟合/阈值/常数，不称blind confirmation。

## R2事前设计：修订1/3

保留原native20/fullASR G、其自身hard YesNo stance及原独立S；新V只读取同一R1完整source cache的reconciled leaf-to-root路径及真实LOCAL两帧，原visual问句/policy/8s/4fps/max/r6不变。不把整树重新注入global，不改S问句或源路径事实、词句/字段/新数据获取规则。每窗V按实际nativeG/stance conversation重新前向，禁止拿R1 tree-conditioned margins与旧G/S拼接当新结果。

原Source全部路径/token/pixel/版本先严格重放，原获取完整成本照样计入；CPU/token/缓存隔离与独立窄code确认后原fixed5，再同机器完整333。R1初版+当前R2修订1/3，其余2次预算不变。不跑controlsGPU直到共同主门通过。R2仅定位阅读路径隔离，不把G/S保留本身当novelty；最终路径/ownership/repair仍须原双语料消融及错误绑定门。

R2 reader/analyzer作者实际fixed5当前source/token/pixel/native-prefix binding PASS，独立窄确认`docs/reviews/20261006_m1_interval_native_context_visual_code.md`：8组actual36layer FP32/BF16×18/20×full/smoke production reader/nativeG/V/S/新V freshfull-forward/原S计费复用/原source成本/重复与篡改拒绝通过；CLI R2委托与R1默认验证保持。Launcher补在`launch/lab1_r2.sbatch`及`run_analysis_r2.sh`，GPU只测R2（缓存源不重新获取、原成本全算），prepare同HateVLM18/5.15再HateVideocanonical，launcher已获同一独立窄确认PASS：bash-n及stub实际成功/prepare失败阻断检查通过，smoke沿用CLI prepare --smoke，main shell先严格prepare。无实际R2GPU/指标。

R2同原fixed5/sc474397提交Slurm208，PENDING(QOSMaxGRESPerUser)，等待186/192当前两GPU预算；原完整source成本保留。派发前四实验室bc26a0b clean一致、既有foreign STRAY逐行未变，证据`runs/20261005_m1_interval_witness/r2_machines_before_smoke{,_note}.txt`。尚无R2实际GPU/指标，不启动机制控制。

## R2 scheduling preparation on verified idle lab-server

sc448960 actualnative5 already exact with identical runtime/model/native inputs (ROOT authority`runs/_setup_lab_server_hatevlm/native_smoke/summary.json`, Slurm210). Prepare same R2 fixed5/whole333 reader there to prioritize the retained R1 within signal while ROOT208 remains pending. Replicate the entire original333 source cache, preserving all originalsc474397 acquisition metadata/cost; no new sources/margins/revision. Require actual currentsource/token/pixel replay onserver before dispatch, narrow newlauncher confirmation, then samefixed5 source-path V fresh readings and complete333 entirely onthat host. All new outputs return ROOT beforeresults. Host selection is scheduling only, budget remains R2=revision1/3; originalROOT208 will be held/cancelled only if the verified server path is used, never duplicate or combine readers. No serverR2 execution yet.

NewserverR2 launcher唯一窄确认PASS：`docs/reviews/20261006_m1_interval_lab_server_launch.md`，只host/env/partition改变，smoke/main只measure_r2、无extract/失败停止。同方法/sourcecost/修订预算保持。只读CPU `source_binding_check.py`调用原source validate与native Yes/No source-path encoding，记录当前像素/token/文件只读与原sourcecost，不计算margin/GT，结果须实际PASS才迁移调度。

Server全333 cache已复制，实际fixed5当前raw/source/token/pixel及native Yes/No path binding PASS并回传ROOT `r2_lab_server_input_check/summary.json`；原获取成本/metadata不变。ROOT208仍零执行，own lab调度hold用于相同R2迁移，非goal pause，避免重复读取。GPU前具体计费遗漏修复：原R2已记录的native输入binding CPU秒数现在计入optimized standalone，validator同步；不改变任何输入/读数/调用/预测/预算，须独立窄确认后commit/pull再server GPU。

两处计费修复唯一窄确认PASS：`docs/reviews/20261006_m1_interval_r2_cost_fix.md`，AST仅两行、native成本不变、正确加binding秒及旧遗漏/其他遗漏拒绝，软件成本检查不是额外8B读数。其余source/数值/调用不变，准备当前相同R2 fixed5在已核对空闲server运行。

Same R2 fixed5/sc448960 Slurm211 completed08:52:10/3:32/0:0, newreads/runoutputs returned ROOT. Root matching-runtime noGT prepare PASS: `runs/20261005_m1_interval_witness/r2_full_smoke_analysis/plumbing_summary.json` nativeallraw/global exact, originalS exact,158fresh changedV,5actual Vrepeats exact, original source/currentpixels/tokens/ownership retained. Actualnew-video processing HMM294.8254s/HCS167.1904s includes originalsource218.0830/131.9568s plus nativeprefix/newV/originalS/binding; no additional source generation or tree-conditioned G/S calls; nativeG/S freshly measured. F5 did not exercise repair, same original zero retained, no mechanismclaim. ROOT208 scheduling-held thencancelled0allocated afterserverPASS, no duplicate samples or revision. Whole333 sameR2 ready onserver; no mainGT/performance.

Complete333 sameR2/sc448960 submittedSlurm213 with SCOPE=main afteractualfixed5ROOTPASS/fourcommittedcode e0c3dda match/unchanged foreign userwork/STRAY/sc448960 idle707Gfree. Evidence`r2_machines_before_server_main{,_note}.txt`. No source re-extraction, alloriginal269.19min sourceacquisition cost preserved and entirecurrent inputsvalidatedpervideo; originalROOT208 cancelled0allocated. New source-path V/nativeG/S independentlyfresh, samecode/constant/revision1/3. Job213 RUNNING, no mainGT/metrics yet.
