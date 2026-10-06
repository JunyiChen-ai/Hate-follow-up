**淘汰原因（2026-10-07，累计第31项）**：可靠C完整333统一评测的六项差异均在噪声内、无任一提升≥.01，按规则9归档；输入/工具执行与A/B失败证据保留，用户要求本轮结束后停止，故不开新实验。

# M1候选35：完整视觉时间工具交互

2026-10-06，R1在本候选实际8B/GT前冻结，主机sc474397实现；运行主机待派发。
原九池C1/rank8一次方案PASS：`docs/reviews/20261005_m1_ideation_jury.md`。
独立代码审查及实际GPU未进行，不存在本候选性能数字。

## 来源与定位假设

实际读[VTimeCoT v1](https://arxiv.org/html/2510.14672v1) §3.1–3.3、Algorithm1及§4实现细节。
[官方项目页](https://vtimecot.github.io/)当前仍标Code Coming soon，未声称核验官方工具代码。
原文把进度条、高亮、真实切片和更新后的视觉记忆用于多步视频推理。
本方法保留这条功能链，单Qwen片段相关性替代VideoCLIP-XL cosine，typed工具替代自由Python，
native20替代原32初始帧，最后为逐窗V margin而非时间区间/QA答案；不是原模型数值复现。
主张限于真实媒体驱动的时间工具交互，不能只做overlay或生成动作文字就称完整迁移。
原源阅读记录`runs/20261006_m1_vtimecot/source_reading/`。

假设：当前窗和跨片段事件在同一真实时间坐标上可见，检索→高亮→真实切片→再次观察
减少整视频事件的错误本窗归属，改善原始V/max及最终within排序。保持native G/ownstance/S、max/r6。
MATCH/MAESTRO等目标任务已有检索/工具推理，不能声称首次检索、首次agent或首次grounding。

## 冻结输入、检索和工具

完整数值常数及literal prompts在`spec.json`，两语料同一流程。
唯一冻结Qwen/Qwen3-VL-8B-Instruct，greedy FP32 head、seed0；训练/适配/常数/阈值不使用标签。
最终评测仍为4fps原GT/原split，1fps只作为新增真实来源采样，绝非旧1fps协议。
对k秒目标取首个不早于目标且小于真实duration的原始PTS帧，去重；按原8s半开窗口分组，
每窗最多8帧，若实际PTS空缺导致某窗>8则按时间均匀取8真实序号；空窗明确缺失，不插值。
保存全解码PTS/shape和原RGB PNG，返回后只读重解码逐像素验证；原时间原点为container.start_time
或第一帧实际时间，与共享actual_video_frames规则一致。工具切片只取已逐像素验证的原1fps来源，
不重新编码视频、不虚构新帧；该缓存等价于对指定真实区间取原采样帧，成本完整保留。

查询：native20 overview/full ASR生成固定两字段，每字段8words/16tokens/整条96tokens，
UNKNOWN/空/字段cap不可用、重复条件去重、整条cap全拒绝，实际Q≤2；无重试/强迫非空。
每个真实8帧片段与本窗ASR构造固定source prefix，加固定assistant `Source supplied.`作为
明确协议字串（不是模型观察/答案）。视觉prefill每片段一次，两个实际查询独立suffix和
16token有界整数0–10/UNKNOWN生成，复用前缀并完整恢复KV/rope。所有条件只参与检索。
按有效整数降序、时间ID升序取最多8窗，不设正分阈值，0也仅代表最近候选、不能证明事件存在。
连续窗合成真实半开区间，UNKNOWN片段不补分数。整数从不进入最终仇恨分数。

初始memory为新增1fps真实来源中最多20帧按时间均匀选择，无重复；原生G输入不改变。
最多3个有条件动作：progress_bar（可TERMINATE/UNKNOWN）→highlight（选实际query ID）→cut
（选已高亮query的实际连续区间ID）。每步均由当前实际memory+之前实际工具结果及reason驱动，
不是后台自动执行完整三个动作；拒绝/UNKNOWN/整条或字段cap停止，不补动作。动作中性reason
16words/32tokens，整条128tokens；已执行工具和停止记录保留。
progress_bar把64px条垂直拼接到原帧下方，原RGB全部保留。条margin=min(12,width//8)，
时间映射按floor(clamp(t/duration,0,1)*(right-left))，原全视频秒坐标保持不变，cut后不重置原点。
remaining灰、已播放蓝，光标绿，highlights橙，最终当前窗红，RGB值见spec；时间文字t/duration均一位小数。
highlight真正改变当前memory条的真实区间；cut真正取被选区间的原始帧、时间均匀至最多8帧，
保留progress/highlight状态重新绘条并更新memory。空源/越界必须失败，不伪造实际执行。
动作后再生成一次96token中性反馈，看到实际最终memory；反馈、reason、queries和相关性分数
均不进入最终证据。每个实际步骤的model-visible像素、input tokens、工具状态和来源序号均记录。

最终每窗保留原全局prefix/ownstance和原视觉问题，接收本窗最多8真实LOCAL帧及最多8memory帧，
真实PTS/来源区间和图形当前窗标记；当前/最终memory重复原帧序号只保留一次、LOCAL优先。
为保留原Algorithm1的实际交互历史，另按执行顺序附每个动作前memory最多两张均匀真实帧、
当时实际条状态和执行过的typed action/真实参数。不同状态的同一原帧必须保留，因为条像素不同；
最多16最终帧+3×2历史帧=22（统一保守cap24）。原始query只作明确未核验检索请求，
reason/feedback/relevance数值不进读取。每一步的不同实际媒体及作用域进入最终V，
不是把交互结果压成一个最终静态图。该完整化在原型完成/任何GPU与GT前冻结，
属于原已审完整视觉交互功能的实现定义，没有新方法族或性能版本。
无本窗真实帧则fresh native V回退，原S逐值保持，max及固定r6不变。
图形描绘的是工具条件/真实时间，不是仇恨概率或证据认证；远程来源保留自身时间。
时间文字为固定3×5 bitmap字形、整数放大2倍，不依赖主机字体或系统font版本。
共享src/source_image_branch负责实际新图/DeepStack/位置及KV/rope恢复，生成的文字不作为事实。

## 新视频成本与完整运行

1次query生成+W个clip视觉prefill+QW个独立相关性suffix/最多16tokens+≤3个planner生成+
1次最终实际memory反馈，再W个增强V替换原V（配对实验额外保留native V）。
初始及更新memory编码/图形渲染/全视频decode/PNG IO/检索与原缓存源成本都计费；
片段视觉只在同一视频的Q条件间共享，不称离线免费处理。
原候选粗预算333个典型视频65–195 GPUmin未测量，长视频/多图模型的真实墙钟可能更长；
先fixed5测量，主实验前以实际每窗/query/tool耗时估算完整7359窗，禁止把猜测当实测。
本机原生JPEG/ASR可复用；新的来源缓存`data/temporal_time_tools/`，出处含host/date/输入路径/生成命令。
所有source及paired reader的完整333同一机器，无切片、依赖链、普通SSH GPU。

fixed5守门：各语料实际progress/highlight/cut和媒体更新至少一次，真实retrieval/newV>0，
native allraw/G/S及clone精确，完整input/pixel/token/PTS/检索/工具/成本只读重放。
不降低guard、不强迫模型执行、不把cap/UNKNOWN改成有效来源。
主门统一评测器与固定r6，215/118与within84/99完整六项，相对正式r6同指标双+.01、
其余pooled损失≤.005/within≤.01。Rule9无任一+.01归档；有信号记录实际GT error analysis后≤3修订。
所有数字development-selected，性能与机制均达标才结束当前目标。

主门通过才跑全333控制：同实际媒体集合及模型调用数的静态一次性memory（删除执行步骤角色、
把各步真实视图按原PTS合并呈现，保留每个原像素视图，不省帧；实际text token差异单列，不伪称天然精确匹配）；
去图形但保留实际切片/时间文字；
同帧数/预算均匀来源替代语义检索；循环错配高亮与实际切片来源；同长度错误真实cut区间。
每项作为novelty的部件删除都须同一主指标双语料至少下降.01，否则删除或降级。
原始V/max排序和错误绑定干预必须支持时间工具机制，不以最终r6数字或执行次数作证明。


完整prototype及作者科学CPU检查PASS：full333 raw/JPEG/ASR/native三轴预检；8组actual36layer FP32/BF16×18/20×2/24新图cached/fullreference/allKV/clone；actual10窗合成video真实1fps选帧/查询两路/三工具真正执行与更新视图/完整只读pixel/token/状态重放；实际native tokenizer/processor+36layers共享clip prefix与两个query/每个生成hidden的独立完整前向检查，FP32最大2.21e-6、BF16hidden最大.03125，在事前1e-4/.05软件容差内，所有源prefix KV逐值不变；12组actual36layer生产reader14language/5或1vision，S原调用复用计费、nativeG/allraw/clone/KV/rope及24图新V检查。都是randomweights/软件输入证据，CPU processor在sourcecache测试特意缩小到1024/2048pixel，不是真实8B/GPU/性能。初小于top8的fixture期待错误/扩展fixture宽度笔误日志保留，仅修测试，生产检索未改。来源见manifest列出的本机runs，唯一独立Rule6审查进行中。


唯一独立Rule6代码审查PASS，来源`docs/reviews/20261006_m1_vtimecot_code.md`及`runs/20261006_m1_vtimecot/independent_code_review/`：独立合成10窗实际1fps/video collector与TERMINATE/UNKNOWN/三工具真正执行、CUT24–48秒实际来源和只读pixel/token/工具重放通过；actual36layer两query clip prefix共享/allprefixKV每步逐值不变/两次均从n开始/1vision15language/完整三轴positions等式与独立全uncached每个生成state通过（FP32最大1.55e-6，BF16hidden .03125）；12组productionreader14language/5或1vision、24图历史/原S复用/全部KV/rope/nativeG allraw和计费通过。same-family provisional，无生产改动，未读真实GT/预测/metrics/真实权重/CUDA；actual8Bfixed5 ready，不是科学性能。


实际8B固定5在sc474398提交Slurm206，source/paired reader整个固定5同机运行，正常等待168/186释放activeQOS2。派发前四机8db32c6一致/clean，原foreign STRAY逐行未变，lab3 idle/1.4T/固定5 raw header CPU实际解析PASS；完整333 raw此前在lab3已核验。来源`runs/20261006_m1_vtimecot/machines_before_smoke{,_note}.txt`及已回传本机的`lab3_smoke_raw_coverage.json`。尚未GPU/GT/性能。


2026-10-06实际Slurm206固定5完成，source及runs均回传本机后才核对。Actualfixed5 BOTH returned; source/native allraw/G/S replay reached original tools guard FAIL; query fields capped, no actual queries/tools; narrow diagnosis/interface redesign needed, no GT/performance verdict。来源`runs/20261006_m1_vtimecot/r1_full_smoke_analysis/`，原失败/UNKNOWN/cap未改。此轮未GT/指标，不能裁定idea优劣或算性能修订。


## 来源接口B：事前声明，未读GT/指标

A固定5严格source/native验证后原工具执行门失败，全部查询触8word cap，planner reason触16word cap；原始action为4次PROGRESS_BAR/1次TERMINATE，因无有效reason编译为UNKNOWN。独立窄诊断未发现实现偏离，不能用没有保存的logits推断closing token原因。A失败/原caps/guard/数据保留。

B只向query/plan/feedback system增加显式短字段结束指令：每个query最多6词，每个reason/description最多12词，立即闭合字段或用UNKNOWN。原8/16words、16/32content tokens及96/128whole generation caps完全保留，relevance接口/真实工具/图像/检索/最终reader/G/S/max/r6不变。不是性能修订、贡献或新候选，预算不重置；不强迫生成有效查询/动作。独立spec_B、data/temporal_time_tools_B、source_smoke_B/r1_full_smoke_B等缓存/输出，默认A不变。先实际A只读重放及B软件接口/工具/路径隔离检查、独立窄确认，再同固定5 GPU；原执行guard不得降低。新增调用上界与原A相同，实际有效query/工具可能增加并如实计费。

B作者A全5实际source/currenttokens/pixels只读重放PASS（`interface_B_cpu_checks/A_summary.json`），B真实synthetic80s视频collector/token/工具状态/输入pixels/篡改拒绝PASS（`collector_cpu_checks_B/summary.json`）；唯一独立窄确认`docs/reviews/20261006_m1_vtimecot_interface_B_code.md` PASS，独立A重放/B三工具与TERMINATE/UNKNOWN、A/B全CLI路径隔离、shell成功/失败阻断及环境资源均检查。随机/脚本化CPU提供者不是实际8B观察或性能；同原fixed5 B GPU ready，原预算0/3和caps/guard保留。

同原fixed5接口B在sc474397提交Slurm209，PENDING(QOSMaxGRESPerUser)，等待186/192及207/208；来源获取/配对同ROOT，代码9d78838四实验室clean一致、foreign STRAY精确未变，证据`runs/20261006_m1_vtimecot/machines_before_B_smoke{,_note}.txt`。仍未B实际GPU/GT/指标。

B同fixed5/sc474397/209 DONE09:15:03；ROOT匹配runtime currentsource/nativeallraw/G/S/clone核对到原toolguard FAIL。真实10query均model_quote可用、158actual clip prefixes/316relevance calls；5planner reason仍wordcap，rawaction4PROGRESS_BAR/1TERMINATE编译为UNKNOWN，actualtools0。源记录`runs/20261006_m1_vtimecot/source_interface_B_diagnosis/summary.json`。不降guard/不salvage/不判idea/不进mainGT；需要具体接口诊断，B数据及budget0/3保持。

B独立窄诊断：`docs/reviews/20261006_m1_vtimecot_interface_B_diagnosis.md`，真实5当前source/token/pixel/原grammar只读重放PASS，12wordreason指令实际已传入，仍五个16wordcap，whole30–41tokens未触128，无实现偏离。不从没有的logits猜原因。准备只读`closing_diagnostic.py`/ROOTSlurmlaunch，重算同fixed5首planner原B生成，现成logits观察rawtop/quoteleadingterminal/barequote rank/实际选择，不加forward或改choice/cap/guard；原token/events/input/grid/positions/selection必须exact，源bytes/mtime保持。只诊断，没有新的interface或revision，不salvage旧B。5planner调用预计<5GPUmin(未测)，cost另记；独立窄确认后再普通Slurm提交，不进入主实验/GT。

只读closing observer独立窄确认PASS：`docs/reviews/20261006_m1_vtimecot_closing_diagnostic_code.md`。实际全部5原B首planner CPU输入/语法重放、原head/选择/forward/tokens/events不变、九种记录差异拒绝、源bytes/mtime只读/缓存释放/模块恢复/ROOTlaunch通过。只软件/绑定确认，未GPU/GT/实际截断原因证据。原失败/caps/完整工具执行guard/预算0/3保持，diagnostic可正常调度。

只读原B首planner同5GPU诊断提交ROOT217，PENDINGnormalQOS，all4 committed24de625match/unchangeduserforeignwork/STRAY/451Gfree，证据`machines_before_closing_diagnostic{,_note}.txt`。不改变source或choice/guard/预算，无GT/截断原因结论；所有diagnostic成本与原source分开记录。

## Actual closing diagnosis and declared C interface (no performance revision)

Read-only ROOT217 first originalplanner fixed5 DONE2026-10-06 15:31:22; originalB tokens/events/input/grid/positions/selection exact, sourceunchanged, noGT. Actual quote-leading compoundterminators wereobserved: planner-compatible closings beat actualcontinuing token on3/27 HMM1,6/29 HMMnonhate4,6/23 HCS0EH,4/29 HCS0nX and3/18 HMM114 recordedsteps. Counts compare only observedtop8 terminals andchosenlogit actuallyinrawtop8; no unseenlogits inferred. Evidence runs/20261006_m1_vtimecot/closing_diagnostic_B/{summary,observed_closing_summary,Hate*}.json; allfirstplanner reasons stillwordcap inoriginalB. This diagnoses an actualsyntax-mask restriction, notsource semantictruth ormethod performance.

C predeclared interfacebeforeGPU: sameB prompts/query/relevance/tools/16word+32token plannerfield caps/maxsteps/mandatorynaturallyclosedreason/alloriginalguards, no forcedreason salvage orlargerbudget. Acceptmodel-chosen quote-leading compoundtoken onlyif the textafterclosingquote is a prefix ofthe next literalJSON syntax thatwriteralreadyforces; whitespace permittedonlyoutsideJSON strings. Appendthe whole actualtoken once tocache andrecordit. The nextforce consumes only thealready-emittedmatchingprefix, thenencodesremainingliteralcharacters; neverretokenize orremove generatedcachetokens. Unsupportedterminators remainineligible; fieldword/token cap stillUNKNOWN, greedymax-token incomplete stillUNKNOWN. Query nextliteral iscomma+openingquote orclosingbracket, planner nextliteralcomma+actionkey+openingvaluequote, feedbackclosingbrace. No otherfieldsemantic/sourceinput/function/score changes. Own scopedStream classonlywhileCgeneration/replay, restoredfinally; defaultsA/B unchanged.

spec_C.json preservesB scientificconstants withjustinterface/version/closuredefinition. C independentcache data/temporal_time_tools_C andoutputs *_C; oldA/B includingallfailedsources remainreadonly. Sameoriginalfixed5 mustpassoriginalthree-real-tools/relevance/nativeallraw/G/S/clones/currentpixel/coordinate guardsbeforeanywhole333/mainGT. This is sameR1 implementationrepair, budget0/3, notnewfamily orscience-performance revision. C prototype implemented; actualGPU notrun, independent narrowclosure confirmation pending.

AuthorC realQwen tokenizer/scopedStream physicaltokens/logicalJSON/pendingconsumption/unchangedwordcap/exceptionrestoration PASS runs/compound_cpu_checks/summary.json. Synthetic80s actualvideo/toolrenderer/collector with explicitlyscripted provider andcompoundtokens PASS runs/collector_cpu_checks_C/summary.json: all3 tools genuinelyexecute, actualPTS/currentpixels/finalhistory/metadatareplay/mutationreject. Notpretrained/GPU/performance evidence. A/Bactualfixed5 sources unchanged afternewwrapper CPUreplay PASS logscompound_AB_replay/{A,B}.log. No sourcebudget/guard changed. Oneindependent narrowclosure confirmation inprogress, no newgeneral/proposalreview.

C具体closure接口窄独立确认PASS docs/reviews/20261006_m1_vtimecot_compound_C_delta.md：49实际tokenizer闭合/whitespace/grammar组合、51随机smallmodel物理生成/位置/cache/callcount，A/B actual5源不改重放、C三tool/controller/inputpixel重放与篡改拒绝、原caps/UNKNOWN/guard、恢复与目录隔离PASS。科学模型/提示/常数/tool功能/GT路径不改。下一步原fixed5实际8B；无GPU/performance，sameR1 budget0/3。

C原fixed5 ROOT/sc474397/Slurm228已提交，正常QOS在227短修订后等待225/186预算；四lab一致/foreignSTRAY未改/437Gfree、CPU+唯一窄confirmationPASS后启动。Csource+reader同host、新独立cache/output，旧A/B不改，原守门/预算0/3保留。无实际C GPU/GT/主性能。机器证据machines_before_C_smoke{,_note}.txt。

Coriginalfixed5 ROOT228 complete17:36:55,9:10/0:0; source+reader/rootnoGT preparePASS runs/r1_full_smoke_C_analysis/plumbing_summary.json. Actualall3tools2times ineachdataset,158clipprefix/316relevance,158newV, nativeallraw/G/S and10clones/source/currentpixels/PTS/actualtokens/positions/cost exact. Five-source literalinterpretations unverified; HMMhate_video1 legitimatelyTERMINATE no tools, otherfour full3tools, notforcedsalvage. Processing-stage346.4496s inclsource238.2905 vsnative12.7641; wall/setup/paired/audit separate. Moreexpensive thanR2/newASRspeech pilot, sofull333 nextafterlowerincrementalcompute comparisons; originalCbudget0/3/sourcehost/rootcache retained. NoC mainGT/metrics/mechanism.

2026-10-07 lower incremental comparisons have completed; C full333 is now next while ReKV's missing factorial control is reviewed. Current C full333 raw/native/ASR/processor/mRoPE preflight was rerun in the actual ROOT HateVLM runtime and passed (`full_input_preflight/run_C_current.log`, `summary.json`); no GT was read. Same ROOT/sc474397 source+reader, original C input/closure/tools/model/constants and existing guards unchanged, no slicing or pilot-host splicing. Existing five completed same-host C source records may be reused with their original cost, each source and reader is validated under current C. Fixed5 source/newV scaling over 7359 windows is ~4.4 GPU processing hours; forecast **4–8 allocated GPU hours** including paired native/setup/validation/I/O, not a measurement, detailed `runs/20261006_m1_vtimecot/full_C_forecast.json`. Original native20/fullASR input acquisition is additional. Original R1 budget0/3; all six canonical metrics/raw ordering/negative results and costs must be recorded before any performance or mechanism decision. Full main not yet submitted.

Actual full333 C ROOT/sc474397/Slurm238 submitted and RUNNING2026-10-07 01:05–06 after current C full333 preflight and original fixed5 strictPASS. All4 operative code agreement and unchanged foreign/STRAY documented machines_before_C_main{,_note}.txt; ROOT300Gfree/idle, normal local partition/no budget bypass, held193 unchanged. Standard detached wait_C_main.log observes job disappearance/Traceback/FAILED and actual reader DONE. Same source/reader host, scientific C constants, guard and budget0/3 unchanged. No main metrics/GT/mechanism claim yet.

## Complete C result; current round finished and stopped

Whole333 source+reader on sc474397 / Slurm238 completed2026-10-07 07:32:35 NZDT, allocation6:26:51, exit0:0; source EXTRACTION_DONE333 at05:04:13. All outputs and caches are already on ROOT; no host splicing or remote return was needed. Strict current RAW/PTS/RGB, literal C query/tool/current-image/coordinate/token input and native allraw/G/S/stance replay passed333 before main GT evaluation (`runs/20261006_m1_vtimecot/r1_full_main_C_analysis/alignment.json`). Actual changed visual windows7352 (HMM3764/HCS3588), source clip prefixes7352, relevance14415; three real tools executed in both corpora: progress/highlight/cut142/142/133 HMM,93/93/92 HCS. No forced salvage, source generation or current-input guard was weakened. Main has no clones; the original fixed5 had10.

| Dataset | pooled ROC-AUC | pooled PR-AUC | within macro ROC-AUC | defined videos |
|---|---:|---:|---:|---:|
| HateMM | .8973527478747025 | .6924774224517328 | .7544228928629766 | 84/215 |
| HateClipSeg | .7198610766666074 | .6738456466050611 | .6352575236637795 | 99/118 |

Authority: `runs/20261006_m1_vtimecot/r1_full_main_C_decoded/optimized/metrics.json`; sole evaluator4fps/test/fixedr6, development-selected. Against `runs/20260926_twolevel/r6_bma/metrics.json`, HMM ROC/PR/within deltas+.0002340864605928/−.0017571808944782/+.0036408488970097; HCS+.0030362127519478/+.0027735622048741/−.0020915901403192. Paired base reproduces allsix r6 values exactly; frame counts, predicted/overlap/eligible cohorts and skipped IDs all match. Every difference lies inside the predeclared noise floors, no primary gains>=.01, common performance FAIL (`r1_full_main_C_analysis/summary.json`). Actual tools entering inputs do not establish useful localization or mechanism. Main GT `data/gt_4fps/{HateMM,HateClipSeg}.npz` was used only after complete inference/strict replay by the sole evaluator; no wrap-up main-GT error analysis or design revision was performed.

Actual source10531.649624s (174.53min, includes source decode), paired model/source processing15258.931473s (254.32min), standalone15040.766780s (250.68min); source146893LM+8821vision and reader22297LM+7685vision forwards, peak19.437665GiB. Source-generation and per-window reading are new-video work, not free preprocessing; original native20/fullASR acquisition is additional. Allocation/setup/current-input audit/I/O/paired baseline/failure costs remain separate. Final strict+canonical CPU command took50:01.82 and exited0 (`analysis_C_launcher.log`, /usr/bin/time); no efficiency claim. A/B interface failures and actual C fixed5 evidence remain preserved and are not scientific revisions.

Rule9: no>=.01 gain, so archive as candidate31 with this negative result; no C scientific revision or mechanism controls. The user's latest instruction is to finish the already-running round and stop, so do not move to another candidate. Formal method remains r6_bma, original research goal not complete.
