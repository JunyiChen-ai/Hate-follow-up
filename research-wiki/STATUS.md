# 当前研究状态

截至 **2026-10-06**。当前入口如下；下方按日期保留历史更新。

- 当前正式方法仍为 **r6_bma**；M1自主迭代的共同性能门和机制门均未完成，继续运行，未更新Overleaf。
- 累计归档 **27** 项；最近Program最终R4和OTT完整主门均FAIL，权威结果/归档入口见下方对应更新。
- 区间见证26完整333/sc474397/168已DONE，本机prepare原av17与生成av18在6文件origin差异严格失败；仅prepare runtime窄修复独立PASS，匹配环境完整核对重跑中，尚无主指标。
- **运行中**：来源图25，sc474399/186，HMM全215 source已齐、HCS来源继续获取。
- MERIT33实际固定5/sc474398/202完成并BOTH回传，本机源/原生全读数/G/双branch clone/真实remote进入V/S守门PASS；完整333 ready，尚无GT/性能。
- 文字发生32/201、有序事件槽34/205、视觉时间工具35/206实际固定5均完成并BOTH回传；均源输入/原生读数核对到原执行guard FAIL，未GT/性能，不降guard或裁定idea。各实验README记录实际原因及来源。
- **运行中**：空间搜索31原完整333/sc474398/192已释放own lab hold；按smoke实测较低处理成本先恢复，source及reader同机完整运行。
- **主agent调度持有**：事实核验27完整333/sc474397/193仍未启动，优先较低成本完整候选；方法版本/修订预算不变。
- 全部主结果继续用HateMM/HateClipSeg、统一4fps、固定r6及唯一评测器；development-selected，不挑语料/指标，完整新视频来源成本保留。

**2026-10-04 M1 自主迭代继续**：用户确认 Explorer R1 为正向进展，并要求达到性能与机制目标后再停止。
当前正式方法仍为r6_bma。Explorer R4完整333在sc474399/Slurm87完成并回传本机，
统一评测HateMM ROC/PR/within .897753/.694180/.759909（84），
HCS .722478/.676508/.652582（99）；within+.009127/+.015233，HMM PR仅−.000054。
来源`runs/20261003_m1_explorer/r4_main_decoded/explore/metrics.json`。
仍未过声明的双语料within各+.01门，原始visual/max排序HMM下降，不能声称机制成立。
实际新增像素处理合计35.43min（native3.71倍）；详情与最佳正向版本保留在
`archive/experiments/20261003_m1_explorer/README.md`。初版+三修订已用完，累计归档18项。
继续独立候选选题、审查与完整实验；性能与机制目标均未完成。全部development-selected，不更新Overleaf。

候选19连续视觉状态优化完整333已完成并回传：sc474399/Slurm93。
HateMM ROC/PR/within .894437/.691121/.727686（84），
HCS .685951/.652592/.566666（99），全部development-selected。
权威来源`runs/20261004_m1_latents/r1_full_main_decoded/optimized/metrics.json`。
配对native精确复现当前全部六项，7359视觉窗干预进入读数，但原始visual/max排序下降，
没有任一主指标+.01，按规则9直接归档为第19项；不跑完整控制/不再修订。
实际成本64.34min、native约6.28倍。明细唯一入口
`archive/experiments/20261004_m1_latents/README.md`，官方contextual版本尚未测试。
候选20声学路径分布条件化读取完整333已在sc474399/Slurm100完成回传，配对native全部六项精确。
HateMM ROC/PR/within .893585/.685160/.758773（84）；HCS .716933/.669970/.645755（99）。
within+.007991/+.008406，HMM PR−.009075，无任一主指标+.01，按规则9归档为第20项。
权威来源`runs/20261004_m1_acoustic/r1_soft_main_decoded/optimized/metrics.json`。
原始speech共享帧子集有正向趋势，但不是标准主指标、不作续跑门；机制控制未运行。
实际新处理合计25.26min/native2.77倍，全部development-selected；明细
`archive/experiments/20261004_m1_acoustic/README.md`。
候选21完整语义聚类树已完成初版和三次修订，按规则9归档；不跑R5或控制GPU。
完整原生读数及六项指标各轮精确复现，所有输入/输出已回传本机。
Tree R1 HateMM ROC/PR/within .895411/.679329/.779693（84），HCS .729941/.689819/.650289（99）；
within+.028910/+.012940，但HMM PR−.014905。保留此正向结果；未过门。
来源`runs/20261004_m1_tree/r1_full_main_decoded/optimized/metrics.json`。
R2 HateMM .897530/.690654/.772521，HCS .729631/.682939/.642520；
HMM within+.021739、HCS ROC/PR+.012806/+.011867，所有损失在噪声内，但无同指标双语料+.01。
来源`runs/20261004_m1_tree/r2_full_main_decoded/optimized/metrics.json`，此正向版本保留。
R3 HateMM .893445/.675568/.761225，HCS .716635/.670358/.633798，HMM PR退化/无共同提升。
来源`runs/20261004_m1_tree/r3_full_main_decoded/optimized/metrics.json`。
R4完整333在sc474399/Slurm120完成，HateMM .894415/.674850/.765801，
HCS .719421/.675408/.635046；within+.015019/−.002303，HMM PR−.019384，performance FAIL。
来源`runs/20261004_m1_tree/r4_full_main_decoded/optimized/metrics.json`。
R4原始max within−.044756/+.002938，HMM区间全负；G不变、新V与R2逐值精确，6580 S变化。
实际新视频处理141.71min/native15.43倍，包含获取117.07min。全部development-selected。
控制代码及独立CPU/代码审查PASS仅为prepared，不是机制证据，完整主门失败所以未跑控制GPU。
细节唯一入口`archive/experiments/20261004_m1_tree/README.md`；累计归档21项。
候选22声学词格结构读取的方案、独立代码审查和CPU检查已PASS；现已按规则9归档。
固定五视频实际音频CPU预检通过；AAC重叠PTS问题已修并获独立窄确认。
入口`archive/experiments/20261004_m1_lattice/README.md`；Slurm121完成5视频真实beam输入并回传，首个结构读取因CUDA bias类型失败。
已修为BF16 query相同类型并获独立窄CPU确认PASS；Slurm122相同5完成并回传本机，noGT prepare PASS。
原生全读数精确、G/V不变，134新S变化，clone和单路径结构/普通顺序margin各5精确一致。
来源`runs/20261004_m1_lattice/r1_full_smoke_analysis/plumbing_summary.json`。
R1完整333在sc474399/Slurm123完成并回传本机，native原始读数/六项精确。
HateMM ROC/PR/within .896571/.677466/.799002（84），HCS .714415/.667120/.640013（99）；
HMM within+.048220、配对区间[.017740,.084486]为正，保留这项正向进展。
但HMM PR−.016768/HCS within仅+.002664，无双语料共同提升，performance FAIL、机制未建立。
权威来源`runs/20261004_m1_lattice/r1_full_main_decoded/optimized/metrics.json`。
实际新处理70.51min/native7.67倍；来源/完整误差分析在实验README。
按规则9继续R2：保留完整beam的词语关联、替换独立槽重组；修订1/3，CPU/独立窄检查PASS。
R2固定5在sc474399/Slurm124完成并回传；noGT prepare PASS、native全读数精确、134新S变化、5clone/unit均精确。
来源`runs/20261004_m1_lattice/r2_full_smoke_analysis/plumbing_summary.json`。
R2完整333在sc474399/Slurm125于07:26:23完成并回传，native原始读数/六项精确。
HateMM ROC/PR/within .896267/.677443/.802410（84），HCS .712939/.667846/.639245（99）。
HMM within+.051628、配对区间[.019888,.088288]为正，保留这项进展；HMM PR−.016791/HCS within+.001896，performance FAIL。
权威来源`runs/20261004_m1_lattice/r2_full_main_decoded/optimized/metrics.json`；实测新视频72.45min/native7.88倍。
R2六个结构/转写/质量/错来源对照已准备，独立CPU/代码审查PASS；仅prepared，完整主门通过才启动控制GPU。
审查来源`docs/reviews/20261005_m1_lattice_path_controls_code.md`；尚无实际控制读数。
完整误差分析已记录，R3改为每条完整转写一个末端表示、只由末端进入查询；修订2/3。
CPU与独立窄代码检查PASS，来源`docs/reviews/20261005_m1_lattice_terminal_code.md`。
实际8B固定5在sc474399/Slurm128完成并回传，noGT/native allraw exact、134newS、5clone/unit差值0。
完整333在sc474399/Slurm129于08:04:15完成并回传，native原始读数/六项精确。
R3 HateMM ROC/PR/within .892420/.673876/.750112（84），HCS .705600/.644986/.601483（99）。
无任一主指标+.01，HCS within−.035866、HMM/HCS PR−.020359/−.026086，performance FAIL；原始max两语料下降且区间全负。
来源`runs/20261004_m1_lattice/r3_full_main_decoded/optimized/metrics.json`；新视频72.55min/native7.88倍。
按规则9归档第22项，不跑R4/控制GPU；R1/R2正向within与六项完整数字保留。
R3七臂matched控制独立CPU/代码检查PASS，来源`docs/reviews/20261005_m1_lattice_terminal_controls_code.md`；仅prepared，主门通过才运行。
算法/成本/全部读过的GT与设计关联唯一明细仍为实验README；全部development-selected，机制尚未建立。
备用候选23可执行时间/来源/话语程序的独立方案/代码审查与CPU检查已PASS。
原先声明的固定5在空闲sc474398/Slurm130完成并回传，但noGT prepare因实际感知模块调用0未过机制执行检查。
保留真实非法参数/超预算/缺窗/UNKNOWN记录，独立接口窄诊断确认未观察到实现bug、原始模型输出不符合接口；
来源`docs/reviews/20261005_m1_program_gpu_interface_diagnosis.md`。需要新接口设计；不降低guard、不启动主实验、不作性能/idea裁定。
入口`experiments/20261005_m1_program/README.md`，错误来源`runs/20261005_m1_program/r1_full_smoke_analysis/run.log`；未读GT。
新source-handle接口B已事前声明、CPU来源/解码检查和独立窄code/interface审查PASS；
来源`docs/reviews/20261005_m1_program_handle_code.md`。A失败记录保留，B用独立缓存/输出，
守门只增加真实有效感知检查、不放松。Slurm133获取4/5后第五长planner prefill OOM，
partial runs/inputs已立即回传；allocator-only相同5重试Slurm134于09:00:18完成并回传本机。
B noGT prepare PASS，native allraw/G及10clone精确，实际126/91个结构有效感知调用；来源
`runs/20261005_m1_program/r1_handles_full_smoke_analysis/plumbing_summary.json`。未读GT、没有性能结论。
固定5实际远程context/join均0，speaker/mode多为UNKNOWN；结构有效不等于语义或机制成立，明细只记实验README。
纯输入完整333大小审计发现plain最大的三个chunk展开约53k，超过fixed5；prefill内存生命周期修复及独立等价窄确认PASS，
来源`docs/reviews/20261005_m1_program_handle_prefill_memory_fix.md`。Slurm136最长完整视频于09:17:59成功并回传，53,137实际tokens/29.03GiB；本机source/token重放PASS。
来源`runs/20261005_m1_program/handle_capacity_validation/summary.json`；相同完整333 B已在sc474398/Slurm137启动；输入获取与配对native/new读取同机，无性能结论。
候选23 R1 B完整333在sc474398/Slurm137于13:18:34完成，inputs/runs均已回传本机，native allraw/六项精确。
HateMM ROC/PR/within .897219/.691274/.769788（84）、HCS .719377/.672421/.639552（99）；
within+.019006/+.002202，其他损失在既定容差内，保留HMM正向进展；无同指标双语料+.01，performance FAIL。
权威来源`runs/20261005_m1_program/r1_handles_full_main_decoded/optimized/metrics.json`；实际新处理233.83min/native25.51倍。
程序/来源完整重放PASS但context/join均0，话语角色仅1/3窗解析；原始max within两语料下降，尚无机制结论。
实际GT误差分析已记录，R2事前声明只有可用程序事实才加入证据，缺事实则重新调用原生分支；修订1/3，全333来源/token CPU和独立窄代码确认PASS，R2固定五视频在sc474398/Slurm149于13:57:01完成并回传，本机noGT/native allraw/G/10clone/实际回退精确PASS，70V/2S变化；相同完整333 R2已在sc474398/Slurm150运行。来源`runs/20261005_m1_program/r2_handles_full_smoke_analysis/plumbing_summary.json`。
R2完整333在sc474398/Slurm150于14:20:56完成并回传，native allraw/六项精确。HMM ROC/PR/within .897479/.696264/.766186（84），HCS .716819/.673020/.640984（99）；within+.015404/+.003635，其他损失在容差内，但无同指标双语料+.01，performance FAIL，原始max两语料下降。来源`runs/20261005_m1_program/r2_handles_full_main_decoded/optimized/metrics.json`。实际新处理229.45min/native25.07倍，保留HMM正向进展；完整实际GT/来源误差分析已记录，R3已事前声明UNKNOWN标记归一/视觉要求可用action（修订2/3）；实际完整333原R1/R2 query/token逐值不变、源码缓存不变与R3CPU检查PASS，独立窄代码/sourceoracle/36层CPU确认PASS，来源`docs/reviews/20261005_m1_program_resolved_action_code.md`，R3固定5在sc474398/Slurm154于15:02:27完成并回传，本机noGT/native allraw/G/10clone/source/freshfallback PASS，相同完整333 R3已在sc474398提交Slurm155；无R3性能/机制结论。
全部development-selected；事实/算法/成本/来源唯一明细为实验README。
候选24引语/指代图的独立方案/代码审查PASS，CPU原型与独立路径/真实36层可见性检查完成；
22已归档、23接口未可靠执行；24固定5在sc474399/Slurm131完成并回传本机，noGT prepare PASS。
原生allraw精确、G/V不变、134新S变化、5clone/contextless检查通过；来源
`runs/20261005_m1_quote_graph/r1_full_smoke_analysis/plumbing_summary.json`。
HMM三个样本图节点0/非空上下文0of96；HCS13节点/12边、13of62窗获得上下文。
仅实现验证，不是机制证据；原声明允许空图/拒绝块，不改接口/guard。相同R1完整333在sc474399/Slurm132于09:51:11完成，输入/配对结果已全部回传本机；native allraw/六项精确，完整主门FAIL。
R1 HMM ROC/PR/within .895232/.682626/.761788（84），HCS .709028/.656763/.629661（99）；
HMMwithin+.011006（配对区间含0）保留为开发期信号，但HMM PR−.011609/HCS各项下降，尚无机制证据。
来源`runs/20261005_m1_quote_graph/r1_full_main_decoded/optimized/metrics.json`；按规则9开始实际误差分析/最多三修订，尚未跑控制GPU。
R1实际GT误差分析已记录：图上下文只覆盖约5%窗口，HMM正向信号集中于无图上下文视频，不能归因于图机制。
R2事前声明空packet精确原生S、有真实context保留原结构读数；修订1/3，实际fixed5 source/token CPU及独立窄代码修复确认PASS，相同fixed5 Slurm139于10:17:49完成并回传，noGT/native allraw/G/V、5clone/contextless PASS，13HCS S变化/0HMM，完整333在sc474399/Slurm141于10:31:46完成并回传，native allraw/六项精确。
R2 HMM .897998/.697106/.741690（84）、HCS .716345/.671003/.638255（99）；无任一主指标+.01，按规则9归档，不跑R3/控制GPU。
来源`runs/20261005_m1_quote_graph/r2_full_main_decoded/optimized/metrics.json`；实际82.85min/native9.01倍，R1正向信号保留。
来源`runs/20261005_m1_quote_graph/r2_cpu_checks/summary.json`；入口`archive/experiments/20261005_m1_quote_graph/README.md`；当时累计归档23项。
备用候选25持久实体/话语图驱动实际端点媒体检索：独立方案/代码审查、真实fixed5来源CPU检查与完整333输入预检PASS；
入口`experiments/20261005_m1_provenance/README.md`。固定5在sc474399/Slurm138于10:12:37完成，BOTH inputs/runs已回传本机。
全部158源ledger实际schema fields拒绝，五视频graph edges/remote contexts均0，声明的实际执行guard未满足；不降低guard、不跑主实验/GT性能。
来源`runs/20261005_m1_provenance/source_interface_diagnosis/summary.json`；本机完整source/native验证及10repeat精确，随后声明guard失败；独立窄接口诊断确认未发现实际GPU实现bug；A源接口未可靠执行。事前声明独立B结构约束来源接口；实际fixed5 CPU来源/grammar replay与完整333 span输入预检PASS，独立窄代码确认PASS，最大实际source22image7701tokens/6550全部span保留；相同fixed5在sc474399/Slurm142于11:16:41完成，BOTH已回传本机；本机noGT native allraw/10repeat精确，原source guard PASS：HMM96有效ledger/51remote窗、HCS60/57（2wholecap拒绝保留），相同完整333 B的sc474399/Slurm146于14:32:51因调度优先较少新增调用的候选取消并保留检查点，运行2:56:43，已完成的原子来源缓存及全部runs均回传本机；将以原版本续跑，未读GT/无性能结论。来源`runs/20261005_m1_provenance/r1_handles_full_smoke_analysis/plumbing_summary.json`。
来源`docs/reviews/20261005_m1_provenance_gpu_interface_diagnosis.md`；无机制结论。
备用候选26区间来源见证组合/分歧触发重读，独立方案审查PASS；入口`experiments/20261005_m1_interval_witness/README.md`，科学代码及29项CPU检查/独立代码审查PASS，实际GPU进展见下；来源`docs/reviews/20261005_m1_interval_witness_code.md`。
本机新隔离`.cache/envs/HateVLM`环境已完成安装，与实际lab2核心Torch2.11cu128/HF5.15.1一致；独立基础设施窄检查PASS，项目离线缓存缺口已通过现有完整本地模型symlink修复；Slurm145于11:24:58完成，原生fixed5全部G/stance/V/S逐值精确，来源`runs/_setup_local_hatevlm/native_smoke/summary.json`；科学fixed5在sc474397/Slurm147于13:37:39完成，本机noGT source/native allraw/10repeat PASS，158V/134S变化，修复次数0保留；相同R1完整333已在sc474397提交Slurm151，当前已实际运行获取完整来源；无性能结论。来源`runs/20261005_m1_interval_witness/r1_full_smoke_analysis/plumbing_summary.json`。
备用候选27独立事实重观察/typed修订：原9池rank9最后一项，一次独立方案审查PASS；来源`docs/reviews/20261005_m1_verification_proposal.md`。完整factored CoVe四阶段功能迁移的target差异、来源归属与成本已事前声明；`experiments/20261005_m1_verification/README.md`及`spec.json`为入口，原型/33项CPU检查/唯一独立代码审查PASS，科学fixed5在sc474397/Slurm148于13:56:29完成，本机noGT/native allraw/G/10repeat/来源执行PASS，158V/134S变化，96/62次实际独立核验；字面修改不代表正确纠错。完整333待启动/未读GT。来源`runs/20261005_m1_verification/r1_full_smoke_analysis/plumbing_summary.json`。来源`docs/reviews/20261005_m1_verification_code.md`。
下一批完整九项候选的一次fresh独立方案裁定全部PASS（same-family provisional），选择TTF作低新增调用的备用具体化；TTF候选28具体原型/实际full333 JPEG/ASR/grid输入预检/独立code与真实36层缩小CPU接口检查PASS，入口`experiments/20261005_m1_ttf/README.md`，code审查`docs/reviews/20261005_m1_ttf_code.md`；实际8B fixed5在sc474398/Slurm152于14:34:04完成，BOTH回传后本机samebackend noGT/native allraw/G/S/来源/5clone PASS，158V变化，相同R1完整333在sc474398/Slurm153于15:01:56完成，全部runs/projector输入已回传，本机严格重放和统一评测正在运行，无主实验性能结论。来源`runs/20261005_m1_ttf/r1_full_smoke_analysis/plumbing_summary.json`。来源`docs/reviews/20261005_m1_ideation_jury.md`，完整原始方案与成本见`experiments/20261005_m1_ideation/CANDIDATES.json`。
候选28 TTF完整333本机重放/统一评测完成，原生allraw/六项精确。HMM ROC/PR/within .897926/.697255/.758985（84），HCS .712970/.671399/.643623（99）；within+.008203/+.006274，无任一主指标+.01，按规则9归档第24项，不跑R2/控制GPU。来源`runs/20261005_m1_ttf/r1_full_main_decoded/optimized/metrics.json`；阶段成本10.51min/native1.16倍，实际Slurm墙钟14:49，原始max两语料下降。全部development-selected，无机制结论；唯一明细`archive/experiments/20261005_m1_ttf/README.md`。
候选29 RoTE沿用九项jury的一次方案PASS，原论文/已发布官方sinc代码实际读取；原型与事前常数在`experiments/20261005_m1_rote/README.md`。完整333纯输入offset/复制body/归一数值域预检PASS，最小C_r .610283、6580 speech窗、最长原生prefix5829/最长speech suffix756；来源`runs/20261005_m1_rote/full_input_preflight/summary.json`。科学真实36层FP32/BF16/18–20frame CPU检查PASS，独立code审查/36层masked-softmax/来源oracle PASS，成本漏计已窄修确认；来源`docs/reviews/20261005_m1_rote_code.md`，实际8Bfixed5已在sc474399提交Slurm156，等待151/155的QOS预算，尚无GPU结果/GT。
候选29 Slurm156于15:28:27完成并回传：noGT/native allraw/G/V/source/5clone PASS，134S改变；完整333待提交，无性能结论。来源`runs/20261005_m1_rote/r1_full_smoke_analysis/plumbing_summary.json`。
候选30 OTT沿用九池rank3的一次proposal PASS，实际论文/官方源码已读取，常数与Qwen3多图/DeepStack完整适配已事前声明；入口`experiments/20261005_m1_ott/README.md`，实现待完成、未读GT。
候选23 R3完整333已回传并统一评测，native allraw/六项精确；HMM .897462/.695865/.763927（84），HCS .716868/.672507/.635003（99），within+.013145/−.002346，共同门FAIL/损失在噪声内。来源`runs/20261005_m1_program/r3_handles_full_main_decoded/optimized/metrics.json`；保留正向进展，实际误差分析后还剩一次修订，无机制结论。
候选29完整333相同R1已在sc474399/Slurm157运行；候选26 Slurm151因大文本父节点prefill OOM退出，116完整检查点保留，未读主GT/无性能裁定；4096-row tokenwise MLP内存修复待科学等价和独立窄确认后续跑。
候选29 Slurm157完整333于15:52:37 DONE，全部runs已回传，本机严格prepare/统一评测进行中。候选26内存修复作者CPU与独立窄确认PASS（FP32微小舍入差异明报），实际8B来源重放待跑；候选30原型/完整输入/36层作者CPU PASS，独立审查服务usage-limit中断尚未PASS，partial成本漏项已修并待独立继续确认。
候选29 RoTE本机完整统一评测完成/native all6exact：HMM .894607/.666285/.738129（84），HCS .693622/.659037/.590675（99），within−.012653/−.046674，无任一+.01、损失超噪声；按规则9归档第25项，不跑R2/控制GPU。权威来源`runs/20261005_m1_rote/r1_full_main_decoded/optimized/metrics.json`，明细`archive/experiments/20261005_m1_rote/README.md`；实际新阶段9.93min/native仪器配对1.09倍，全部development-selected，无机制证据。
来源图25以原B/sc474399/Slurm167续跑完整333（64atomic source缓存保留）；区间26实际8B父节点内存等价检查在sc474397/Slurm166运行，来源时间/参数未改。

**累计二十五个归档小结（规则11，不超过10行）**：
- 最近五项为Tree、Lattice、QuoteGraph、TTF、RoTE；全部完整主结果与原生六指标已核对，归档不等于目标完成。
- Tree R1 within+.028910/+.012940，但HMM PR−.014905；Lattice R2 HMM within+.051628，HCS仅+.001896，两项正向进展保留。
- QuoteGraph R2/TTF R1无任一主指标+.01，RoTE完整双语料退化；不为负结果继续调参或跑控制GPU。
- Program R3保留HMM within+.013145，还剩一次修订；Provenance/Interval继续完整测量，OTT完成一次独立代码审查后运行。
- 当前固定r6与评测器未改；性能与机制目标都未完成，继续自主迭代。

候选26实际8B内存修复Slurm166 PASS，13395/14445token实际父节点原生与分块的生成tokens/events/compiledrecord逐值相同，来源`runs/20261005_m1_interval_witness/prefill_mlp_fix/gpu/summary.json`；相同完整333待续跑确认原失败更大节点容量。
候选26已以原R1/sc474397/Slurm168续跑完整333，116atomic来源严格重放；候选30完整独立代码续审PASS，来源`docs/reviews/20261005_m1_ott_code.md`，实际8Bfixed5待提交。Program R4为最后修订3/3，程序直接绑定原图key的统一方案已事前声明/完整源41817前版本token兼容/36层CPU PASS，独立窄确认进行中，无实际GPU/R4结果。
区间26 Slurm168已成功越过原失败第117视频，68908token完整父节点/29.80GiB；仍在获取后续完整来源。Program R4唯一独立窄确认PASS，来源`docs/reviews/20261005_m1_program_source_bound_code.md`，与OTT一同ready for actualfixed5，等待当前167/168预算，无性能/机制结论。
OTT固定5/Program最终R4固定5已在sc474398分别提交Slurm170/171，等待167/168实际GPU预算；两项唯一必要代码审查和科学CPU检查均PASS，未实际GPU/主结果。
候选31空间目标搜索沿原九池rank4一次proposal PASS，实际V*论文/官方搜索实现已读，冻结单Qwen的两节点/实际ROI/时间和位置记忆功能适配已事前声明；入口`experiments/20261005_m1_spatial_search/README.md`，原型未实现/未GPU/未GT。
候选31完整空间搜索/实际ROI/source-image reader原型与统一评测接口已实现；完整333实际原视频/JPEG/ASR输入预检、实际processor/36层FP32BF16科学CPU及唯一独立代码审查PASS。来源`docs/reviews/20261005_m1_spatial_search_code.md`，实际8Bfixed5待派发、未GPU/GT；目标性能与机制未完成。
OTT Slurm170固定5/Program R4 Slurm171固定5均DONE并立即回传：严格noGT/source/currentinput/native allraw/clone PASS，完整333待提交；无性能结论。来源`runs/20261005_m1_ott/r1_full_smoke_analysis/plumbing_summary.json`与`runs/20261005_m1_program/r4_handles_full_smoke_analysis/plumbing_summary.json`。Provenance167长LINK prefill OOM，116完整HMM/原partialcache与runs已回传；4096 tokenwiseMLP/allocator内存修复和独立窄集成确认PASS（`docs/reviews/20261005_m1_provenance_link_memory_fix.md`），实际LINK8B重放待执行，无主GT/性能裁定。
OTT完整333同R1在sc474398/Slurm180于21:50:39 DONE，runs和新增源特征正在立即回传，统一评测尚未完成；Program最终R4完整333在同机Slurm181运行。两项均从实际fixed5严格PASS后启动，原生输入和固定r6不变，没有主性能/机制裁定。
空间搜索31固定5已在完整原视频均齐全的sc474399提交Slurm184；来源图25实际LINK内存修复重放在同机Slurm183排队，重放通过后才恢复原B完整333。派发前四实验室代码14477a8一致/clean，既有foreign STRAY逐行未变，lab2空闲/568Gfree；正常等待用户GPU QOS预算。区间26 Slurm168来源获取已完成，配对读取进行中。当前零标签性能与机制目标仍未完成。

独立事实核验27已在原fixed5运行主机sc474397排入完整333 Slurm185；派发前四实验室64b65f9一致/clean和既有foreign STRAY逐行未变，完整输入/spec/读取路径不变，正常等待根机168结束与用户GPU预算，来源`runs/20261005_m1_verification/machines_before_main{,_note}.txt`。当前零标签性能与机制目标仍未完成。

来源图25实际8B LINK重放Slurm183 PASS且完整runs已回传本机：15630/15270token输入，原生与4096分块生成tokens/events/compiledlinks均与原保存来源相同。权威`runs/20261005_m1_provenance/link_mlp_memory_fix/gpu/summary.json`；相同原B完整333以sc474399/Slurm186续跑，正常排队，116HMM和原HCS检查点严格复用、原获取成本保留。派发前四实验室代码a070c00一致/clean、旧foreign STRAY逐行未变，来源`link_mlp_memory_fix/machines_before_resume{,_note}.txt`。无主GT或性能裁定。

Program23最终R4完整333/sc474398/Slurm181已完成回传与统一评测，原生allraw/六项精确。HMM ROC/PR/within .897327/.696904/.744547（84），HCS .717330/.672824/.632911（99）；within−.006236/−.004438，无任一+.01，按规则9归档第26项，不跑R5/控制GPU；R1/R2/R3正向结果保留。来源`runs/20261005_m1_program/r4_handles_full_main_decoded/optimized/metrics.json`；实际新处理228.48min/native24.95倍，全部development-selected、无机制证据，明细`archive/experiments/20261005_m1_program/README.md`。

空间搜索31固定5/sc474399/Slurm184于22:15:48 DONE，runs及sourcecache均已立即回传，本机samebackend noGT prepare PASS：native allraw/G/S/source pixels/coords及5clone精确，HMM/HCS真实FOUND和变化V为78/54；来源`runs/20261005_m1_spatial_search/r1_full_smoke_analysis/plumbing_summary.json`。计划在空闲sc474398完整重算333全部来源与读取，不复用lab2五视频source以保证本轮整体同机；172缺失原视频正由本机补齐，原帧/spec不变。原核验185仍PENDING/0秒，因较低调用空间搜索优先而取消待重新排入，不是idea裁定或目标暂停。

OTT180完整333的源重放线程问题已完成作者和独立窄确认：同原production4线程下全部333份实际保存特征重算的完整plan逐值相同，不放宽任何guard。独立来源`docs/reviews/20261005_m1_ott_replay_thread_fix.md`；全部inputs/runs已回传，noGT fullprepare/currentinputs/native allraw PASS，canonical全六项/固定r6评测恢复运行。无性能或机制结论。

空间搜索31完整333已在sc474398提交Slurm192：172缺失HMM原视频已从本机补齐，全部333实际resolve/PyAV首帧PTS与shape验证PASS，不复用lab2固定5来源、完整来源与读取本轮同lab3。来源`runs/20261005_m1_spatial_search/{lab3_raw_coverage.json,machines_before_main.txt,machines_before_main_note.txt}`，四实验室0b6d7c5 clean/synced、旧foreign STRAY逐行未变。原独立核验27已在sc474397重新提交完整333 Slurm193，192优先入队；之前185取消时实际0秒，未改数据/spec或科研版本。两项正常等待168/186预算。

OTT30完整333/sc474398/Slurm180全部inputs/runs回传、线程等价重放与统一评测完成，原生allraw/六项精确。HMM ROC/PR/within .898340/.698636/.756590（84），HCS .713885/.669485/.634652（99）；within+.005808/−.002697，无任一+.01，按规则9归档第27项，不跑R2/控制GPU。来源`runs/20261005_m1_ott/r1_full_main_decoded/optimized/metrics.json`；实际新阶段10.64min/native1.17倍，全部development-selected、无机制证据，明细`archive/experiments/20261005_m1_ott/README.md`。

候选32像素跟踪的文字发生记忆沿九池C3/rank5一次proposal PASS，已实际读VideoAgent/LELA/官方OpenCV；PyrLK/前后向/外观/失配重观察/发生查询及全部常数已事前冻结，入口`experiments/20261005_m1_text_tracking/README.md`。原型待实现，未CPU/GPU/GT，无性能结论；目标仍未完成。

2026-10-06文字发生记忆32完整prototype/actual333input/36层source-image CPU与pixel/controller/token grammar检查PASS；一次必要独立代码审查正在进行，未GPU/GT。来源图25原失败长视频non_hate_video_356的完整source已成功并回传本机，最大LINK62303tokens/1987输出，容量证据`runs/20261005_m1_provenance/link_mlp_memory_fix/actual_long_source_capacity.json`；原B完整333 Slurm186继续获取，尚无主性能裁定。

文字跟踪32一次独立Rule6代码审查PASS，来源`docs/reviews/20261006_m1_text_tracking_code.md`及`runs/20261005_m1_text_tracking/independent_code_review/`；12组actual36layer生产read_video/validate_bundle含空/首窗/次窗lookup、4occurrence/5实际图像、所有KV/nativeG/allraw/S、freshfallback/modelcalls/完整source计费通过。真实合成video PTS→跟踪/cut/gap→两次freshrepair→只读pixel/metadata replay、损坏crop拒绝、真实token转义/cap/半截escape UNKNOWN通过。same-family provisional，与作者不同，未CUDA/预训练权重/真实GT/分数。实际8Bfixed5 ready，尚无性能结论。

文字跟踪32实际8B固定5已在完整原视频及fixedopencv均齐全的sc474398提交Slurm201；派发前四实验室8ad6a25一致/clean、既有foreign STRAY逐行未变，来源`runs/20261005_m1_text_tracking/machines_before_smoke{,_note}.txt`。唯一独立代码审查及作者科学CPU/输入均PASS，正常等待168/186用户GPU预算及更早空间搜索192；尚无该候选GPU/GT/性能结果。

候选33 MERIT沿原九池C7/rank6一次proposal PASS，已实际读论文/官方MaxSim、邻域过滤、agent、embedding源码；单Qwen功能迁移的wrapper/content pooling/empty/两轮/真实ID filter/媒体预算已事前冻结，入口`experiments/20261006_m1_merit/README.md`。保守完整预算24.5–73.6GPU小时未测，低调用已排队候选优先；原型未实现、未GT/GPU，无性能结论。

MERIT33完整source collector/两轮真实ID过滤/V+S/统一评测prototype已实现，authorfull333input/actual36layersourceimage/真实token-contentpool/生产reader CPU PASS；一次独立Rule6代码审查进行中，未GPU/GT。权威软件证据在`runs/20261006_m1_merit/{full_input_preflight,model_cpu_checks,embedding_cpu_checks,reader_cpu_checks}/summary.json`，不作性能结论。

MERIT33一次独立Rule6代码审查PASS，来源`docs/reviews/20261006_m1_merit_code.md`及`runs/20261006_m1_merit/independent_code_review/`。实际合成video PTS→128tokencaption/key→MaxSim/neighbors/insufficient第二轮→sourceID union→当前inputtoken/pixel只读重放通过；真实36层CPU生产reader/current+remote/V+S/clone/KV/模型计数和source成本通过，source不伪造filter文字为事实。same-family provisional，与作者不同，未真实GT/预测分数/指标；实际8Bfixed5 ready，未GPU/性能。

MERIT33 actual8B固定5已在sc474398提交Slurm202，正常排在较低调用192/201之后等待168/186预算；派发前四实验室e611276一致/clean、既有foreign STRAY逐行未变，来源`runs/20261006_m1_merit/machines_before_smoke{,_note}.txt`。唯一独立code与作者输入/科学CPU均PASS；无该候选实际GPU或GT/性能结果。

区间26完整333来源成本已按本机metadata汇总，来源`runs/20261005_m1_interval_witness/source_cost_audit/summary.json`；原获取成本保留、完整部署/墙钟待配对结束后核对。Slurm168仍在读取，未主GT/性能裁定。

**2026-10-03 camera-ready 排版修复完成**：TRIAGE（5081）和 HateLens（5097）已修复通知中的页边界越界，
两篇完整 PDF 均通过 aclpubcheck，并已推送至各自 Overleaf 项目。原稿位置见 `paper/README.md`，
验证记录及可提交 PDF 见 `runs/20261003_camera_ready/validation.json`。会议系统尚需上传修正版。

**2026-10-02 M1 自主迭代进行中（2026-10-03 更新）**：用户要求修改第一个模块，
并建立涨点机制。入口 `experiments/20261002_m1_iteration/README.md`。
累计归档27个候选：Grounder、Selector、Attributor、Eraser、Factorizer、Contraster、视觉对比、Amplifier、Recycler、Integrator、Reinforcer、Projector、Highlighter、Stabilizer、Preserver、Explorer、Latents、Acoustic、Tree、Lattice、QuoteGraph、TTF、RoTE、Program、OTT性能或机制未过门；Marginalizer、Allocator方案STOP。
最近完成Eraser实际删除：HateMM ROC/PR/within .878392/.618500/.619622，
HCS .672767/.615419/.506248，within各下降约.131，原始排序也下降；完整333基线精确复现。
来源 `runs/20261003_m1_eraser/r1_main_decoded/erase/metrics.json`，详情归档README。
Factorizer完整三臂也未过门：HateMM .894802/.675287/.749240，HCS .704883/.671429/.567329；
HCS within−.0700，普通掩码对照接近基线，来源 `runs/20261003_m1_factorizer/r1_main_decoded/factor/metrics.json`。
Contraster完整结果也无有效提升：within+.0001/+.0002，13939分支全选layer2；
来源 `runs/20261003_m1_contraster/r1_main_decoded/contrast/metrics.json`，已归档。
第八候选视觉对比完整333完成：HateMM .892394/.680132/.770392，HCS .724669/.672273/.648014；
within+.01961/+.01066，但HMM PR−.01410，未晋级。
来源 `runs/20261003_m1_visual_contrast/r1_main_decoded/contrast/metrics.json`；
四项缓存对照完成：HCS支持窗口匹配，HMM打乱不降，尚不能作双语料定位机制。
R2局部扰动完整333完成：HateMM .895698/.689917/.753746，HCS .719246/.671060/.630659；
无任一主指标+.01，visual原始排序两语料下降，耗时44.34min，约原读取4.82倍。
来源 `runs/20261003_m1_visual_contrast/r2_main_decoded/contrast/metrics.json`，此族已归档；
入口 `archive/experiments/20261003_m1_visual_contrast/README.md`，保留R1最佳数字及失败对照。
第九候选Allocator因目标视频任务已有SHAP使用在方案审查STOP，未实施/未跑GPU。
第十候选Integrator R1完整333三臂完成：HateMM .891985/.681604/.766819，HCS .709646/.670246/.636054；
HMM within+.01604但PR-.01263，HCS ROC-.00718，未晋级。来源 `runs/20261003_m1_integrator/r1_main_decoded/future/metrics.json`。
原始visual排序两语料下降；HMM两个raw排序未变的case贡献.01476，不能解释为新定位证据。
R2仅开放未来text keys，完整333完成：HateMM .892960/.686597/.774351，HCS .708410/.664196/.640054；
HMM within+.02357，但pooled仍退化；HCS visual原始排序+.018未传递到双分支结果。
来源 `runs/20261003_m1_integrator/r2_main_decoded/future/metrics.json`。
R3只用新视觉+原语音，完整缓存分支实验完成：HateMM .895125/.693457/.774319，HCS .708136/.664223/.642109；
来源 `runs/20261003_m1_integrator/r3_main_decoded/future/metrics.json`。HMM pooled恢复但HCS仍未过门。
R4最后修订完整333：HateMM .896931/.696579/.769990，HCS .708595/.664496/.635892；
within+.019208/-.001457，HCS pooled仍退化，初版+3修订已用完，此族归档为第12项。
来源 `runs/20261003_m1_integrator/r4_main_decoded/future/metrics.json`，入口 `archive/experiments/20261003_m1_integrator/README.md`。
第十一候选Amplifier完整333完成：HateMM .893032/.685114/.756861，HCS .712161/.669913/.628277；
无任一主指标+.01，HMM PR下降，归档为第10项。来源 `runs/20261003_m1_amplifier/r1_main_decoded/pai/metrics.json`。
第十二候选Recycler完整333完成：HateMM .896713/.692858/.754383，HCS .718114/.671503/.636422；
within+.003601/-.000927，无主指标+.01，归档为第11项，视觉原始排序也未提升。
来源 `runs/20261003_m1_recycler/r1_main_decoded/recycle/metrics.json`；机制激活不等于有效定位。
第十三候选Reinforcer完整333完成：HateMM .891214/.661409/.750129，HCS .722658/.670072/.636515；
无主指标+.01，HMM PR−.032826，归档为第13项；来源 `runs/20261003_m1_reinforcer/r1_main_decoded/full/metrics.json`。
第十四候选Projector完整333完成：HateMM .896986/.694111/.750851，HCS .717685/.671516/.637883；
within仅+.000069/+.000534，归档为第14项；来源 `runs/20261003_m1_projector/r1_main_decoded/project/metrics.json`。
第十五候选Highlighter完整333完成：HateMM .897078/.694292/.754667，HCS .716890/.671082/.637897；
within仅+.003885/+.000548，无主指标+.01，归档为第15项。来源 `runs/20261003_m1_highlighter/r1_main_decoded/highlight/metrics.json`。
第十六候选Stabilizer完整333完成：HateMM .897011/.694135/.751389，HCS .715720/.670317/.638519；
within仅+.000607/+.001170，无主指标+.01，归档为第16项。来源 `runs/20261003_m1_stabilizer/r1_main_decoded/stable/metrics.json`。
第十七候选Preserver完整333完成：HateMM .895610/.689651/.752658，HCS .715189/.672282/.629401；
无主指标+.01，原始visual/max排序两语料下降，归档为第17项。来源 `runs/20261003_m1_preserver/r1_main_decoded/preserve/metrics.json`。
第十八候选Explorer已用完三次修订并归档；R1为用户认可的正向进展，R4 pooled恢复但HMM within仍差.000873未过事前门。权威结果与去向见页首及`archive/experiments/20261003_m1_explorer/README.md`。
运行环境更新：实验室四机启用了Slurm-only GPU访问策略，普通SSH会话访问nvidiactl被EPERM拒绝；使用qian_pilot的local-主机名分区。当前任务不更新Overleaf。

**累计二十个归档小结（规则11，不超过10行）**：
- 最近五项为Stabilizer、Preserver、Explorer、Latents、Acoustic；尚无完整双语料性能与机制过门版本。
- Explorer R4 within+.009127/+.015233，HMM仍低于事前双语料.01门，初版+三修订已用完。
- Acoustic两语料within均有小于.01的正向趋势，但HMM PR下降超过噪声；原始speech子集不能替代主门。
- 两者权威指标与成本入口见本页页首及归档README；不宣称归档等于目标解决。
- 接下来运行完整语义聚类树获取，检验层级/语义选择对照及真实叶观测，不恢复Explorer第五版。
- 当前仍为r6_bma，全部development-selected，零标签/评测协议/Overleaf不变。

**累计十五个归档小结（规则11，不超过10行）**：
- 最近五项为Recycler、Integrator、Reinforcer、Projector、Highlighter；未找到通过双语料性能和机制门的版本。
- Integrator多轮只在HateMM有定位提升，HCS pooled退化；其允许的3次修订已用完。
- 这五项的最后一项Highlighter within仅+.003885/+.000548，原始max排序两语料均下降。
- 来源 `runs/20261003_m1_highlighter/r1_main_decoded/highlight/metrics.json`；无新增机制结论。
- 累计双语料within较强的视觉对比R1仍因HMM PR下降与机制对照失败而未采用。
- 继续Stabilizer及Preserver；当前r6不变，全部development-selected，不更新Overleaf。

**累计十个归档小结（规则11，不超过10行）**：
- 后五项覆盖内部层间对比、像素扰动对比、图像注意力增强、前缀隔离；SHAP迁移因已有目标任务方法STOP。
- 双语料within同时有幅度的最佳已归档版本仍是视觉对比R1：+.01961/+.01066，但HateMM PR−.01410，未过门。
- 来源 `runs/20261003_m1_visual_contrast/r1_main_decoded/contrast/metrics.json`；它的HMM匹配对照未支持定位解释。
- Integrator R2保留作分支修订；它的HCS视觉排序收益尚未传到双分支输出。
- 接下来验证保留原语音的分支修订与VAR注意力重分配；当前方法仍未替换，全部development-selected。

**五个候选小结（规则11，不超过10行）**：
- 尝试了查询注意力限制、选择注意力头、缓存值归因、实际删除；双假设因已有方法STOP。
- 完整实验都复现原生基线；没有一个达到任一最终指标+.01的保留门。
- 其中最接近原方法的主方案是Grounder late：HateMM .897570/.695834/.750171；
  HCS .716291/.670781/.637249，来源 `runs/20261002_m1_grounder/r1_full_decoded/late/metrics.json`。
- 梯度归因和实际删除的原始窗口排序已下降，不能把下游拟合当作唯一原因。
- 继续编码阶段隔离与模型层间读出；当前方法、论文和Overleaf不变，全部development-selected。

**2026-10-02 可修正全局先验实验完成，负结果归档**：用户授权的共享随机偏差机制已实现、
通过独立方案/代码审查、解析积分与数值精度检查，并跑完 full / independent / no-global 三臂。
匹配旧 Reader 上，full 的 ROC / PR / within 为 HateMM **.5693 / .3603 / .7134**，
HCS **.5139 / .4895 / .6324**（`runs/20261002_revisable_prior/r1_full_px/metrics.json`）；
相对 Append+r6 没有任一指标提高 .01，within −.0506 / −.0047，按规则 9 归档。
共享偏差本身保留了强视频判别信息，分离后反而破坏视频排序；允许修正不等于可靠纠错。
完整对照、错误案例、成本与去向见 `archive/experiments/20261002_revisable_prior/README.md`。
新增 MLLM/GPU 开销为零；当前 r6、论文和 Overleaf 不变。development-selected，旧缓存试验。

**2026-10-02 全局裁定的案例分析完成**：见 `experiments/20261002_verdict_analysis/README.md`。
对匹配 Reader 缓存重跑当前 r6、分析改善/退化案例并做保持原始窗口排序的平移对照后，
不支持把加入模型自己的 Yes/No 描述为稳定的定位收益；部分变化来自分数水平与下游拟合的交互。
主语料仍是旧 Reader 家族，最新 Reader 的该消融缺口保留。development-selected；方法与论文未改。

## 当前方法：r6_bma（2026-09-27，K2 第 6 轮通过后替换 r3_m2；development-selected）

r6_bma = r3_m2（用户 2026-09-27 定的默认方法）去掉时序层里以秒设定的常数，其余不变。
- M1 读取，与 SPVL-r2 相同。Qwen3-VL-8B 先读共享前缀（规则 + 20 帧 + 整段转录），给出整段裁定，再过立场轮，然后每个 8 秒窗读画面、语音两个隔离分支。
- M2 读数变证据。每个模态的读数先换成语料内排名的正态分数，再由 EM 估"违规 / 非违规"两类读数分布。
- M3 时序。每个模态一条链，任一条链处在违规状态即算违规。最短段 = 两个读数窗（形状 4 由读取网格推出，不是选的）。平均段长不再设定，在每个视频内对 [两窗, 视频长度] 按长度均匀先验积分掉。
- M4 组合。整段分 = 两类混合无标签校准后的对数几率；帧分 = 整段分 + 视频内居中秩。`--arm full` 输出区间。
- 代码：`experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`，读数 `runs/20260926_glr/base_gridA`。最终运行与消融见该 README §20。

结果（test，4 fps；pooled ROC / pooled PR / within）：

| 方法 | HateMM | HateClipSeg | DeHate（external，1151 视频） |
|---|---|---|---|
| **r6_bma**（`runs/20260926_twolevel/r6_bma/metrics.json`；DeHate `runs/20260927_dehate_external/r6_bma/metrics.json`） | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 | .7011 / .1582 / .6431 |
| r3_m2（用户定的默认，`runs/20260926_twolevel/r3_m2/metrics.json`；DeHate `runs/20260927_dehate_external/r3_m2/metrics.json`） | .8971 / .6953 / .7525 | .7168 / .6705 / .6397 | .7009 / .1578 / .6539 |
| SPVL-r2 + 时长先验（`runs/20260926_glr/infer/base_gridA_d80/metrics.json`） | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 | .6996 / .1570 / .6364 |
| SPVL-r2（`runs/20260926_glr/infer/base_gridA_spvlr2/metrics.json`） | .8952 / .6867 / .6800 | .7134 / .6667 / .6101 | .6993 / .1570 / .6406 |
| T3AL 重跑，零标签 | .6091 / .3096 / .5068 | .6246 / .5645 / .5003 | 未跑 |
| ZS-ImageBind，零标签（DeHate `runs/20260927_dehate_external/zs_imagebind/metrics.json`） | .5928 / .3108 / .5343 | .5917 / .5495 / .5241 | .5538 / .0962 / .5131 |
| MultiHateLoc，视频级标签 | .7618 / .5188 / .6108 | .5056 / .4885 / .4996 | .6102 / .1289 / .5420 |
| Fed-WSVAD 3 clients，视频级标签（DeHate `runs/20260927_dehate_external/weaksup/metrics.json`，3 seed 均值） | — | — | .7007 / .1752 / .5055 |

- r6_bma 相对 r3_m2：六个数都在噪声内；8 个 MLLM 上 within 两语料都 8/8 不低于 r3（`experiments/20260926_twolevel/README.md` §19.3）。DeHate within 低 .011（external，不作门）。
- DeHate 是 external validation（`experiments/20260927_dehate_external/README.md`）：within 比所有 baseline 高 .10 以上；pooled ROC 与 Fed-WSVAD 持平；pooled PR 低 .017。
- 最终运行与消融（`experiments/20260926_twolevel/README.md` §20.1，区间为按视频的配对 bootstrap）。
  - 去掉时间耦合 within −.114 / −.057，两语料都过 .01；视频项、within 项也过。最短长度（−.030 / −.009）、分模态 OR（−.007 / −.043）、校准 key（pooled −.002 到 −.006）都只在一个语料或都不到 .01。
  - 正态分数：去掉反而 +.011 / +.011（区间含 0）。是否在 r6 下仍需要，未查；要查需另起一轮声明（8 个 MLLM，CPU 约 20 分钟）。
  - 对最佳高斯平滑（σ 8 秒）：HateMM +.063（区间不含 0），HCS +.006，DeHate −.004。
- 三语料 error analysis：`experiments/20260927_error_analysis/README.md`。
- 最终 novelty 复查（K6）：`docs/reviews/20260927_final_novelty_review.md`。整体对仇恨视频是新的，但每个部件都有近邻工作。
  - 可写的贡献：M1 整体；模型外、无标签、以读数窗为单位的显式时长时序层（去耦合 within −.11 / −.06）；8 个 MLLM 上都有效；DeHate within 超过弱监督 baseline。
  - 不能写成贡献：M2 正态分数、M4 校准与组合、隔离 / 前缀缓存、双分支、转录语境；也不能写"比平滑好"（HCS 只高于最佳高斯 .006–.009）。
  - 论文前的缺口（该文档 §8）：同 backbone 的 training-free 对照（规则 14f）；M1 各部件在最终读数上重测；逐条核对引文。

## 2026-09-27 自主迭代：concern 清单（用户要求全部解决后停止）

**截至 2026-09-28：解决 K1、K2；部分解决 K6；K3、K4、K5、K7 未解决。**09-27 我用"关闭 = 过门，或按声明做完并写明证据"这个自定标准报了"全部关闭"并停了循环，这不等于用户要的"解决"，是错误的汇报。只有通过声明的门才算解决。每项的门在各自实验 README 里先声明后运行。

**待用户裁定（评测协议）**：HCS 的 GT 用 6 维片段标签里任一非正常维度（仇恨 + 4 类其它冒犯内容），测的是"冒犯内容"定位，不是仇恨定位；测试集 51 / 104 个正例视频只有非仇恨的冒犯内容（`experiments/20260927_dvd/README.md` §9.3，`hcs_strict.py`）。是否改成只算仇恨维度由用户决定；改了则全表 HCS 数字重算，DVD（裁定 + T）可按声明的门重跑。

**2026-09-28 条件不变时的上限分析**（`experiments/20260928_headroom/README.md`，读 GT，只做分析）：
- 整段层（K3）：若整段判断完全分开仇恨 / 非仇恨视频，HateMM +.029 / +.041、HCS +.034 / +.015，够过门。更正 09-27 的说法：并非任何整段层改动都过不了门。
  - 但用真实标签在现有读数上（含 DVD 的 T / E / A）训练整段分，HateMM、HCS 都不如现方法（视频 AUC .883 对 .927，.766 对 .803）。
  - 缺的信息不在现有读数里。
- 视频内（K4 / K5）：用真实标签在现有读数加方法输出上训练，最多 HateMM +.015（区间含 0），HCS 不涨；短仇恨子集也不涨。这是弱估计（84–215 个视频、简单模型、没有按时间序列建模），不能说明 within 已到现有读数的上限。
- 画面（K7）：只用画面读数比只用语音低 .045 / .038；每窗补一帧不增加可学到的信息。
- 结论：这些拟合没有在现有读数上找到更好的组合，但样本小、模型简单。它不能证明后续方法已到顶，也不能说只有换读数才行。（2026-09-28 更正：原写法说这几项必须靠新的信息来源。）
- 8 秒窗的粒度本身不限制定位：把每窗的真实仇恨比例当作读数，within 为 .954 / .986，短仇恨子集也一样，所以短仇恨定位差不是因为窗太粗。这是把真实标签当输入，不是方法的上限。（2026-09-28 更正：原写法据此说"差距来自每窗读得准不准"，把读数有误差当成了要去掉的问题。读数有误差正是后续方法要处理的前提。）
- 多个 MLLM 一起读能加信息（HateMM within +.03，HCS PR +.02），但读取成本 6–10 倍，而且规则 14(e) 不允许方法里有 ensemble，要走须用户改规则。
- Codex（gpt-6-astra）咨询，记录在 `docs/reviews/20260928_codex_mechanism_review.md`。它的第一个提议是按字幕是否连贯决定时间耦合，数据检查不支持：字幕连贯几乎不能预测相邻两窗标签是否相同，在读数之外只加 +.002 / +.008 AUC，区间含 0（`experiments/20260928_headroom/README.md` "Codex consultation" 节）。第二个提议是从当前方法蒸馏一个稠密小模型，未测。
- Codex 第二轮（`docs/reviews/20260928_codex_mechanism_review_round2.md`）：
  - 撤回蒸馏提议，只留一个小试验：每个窗分支的后几层只看本窗内容，不再看整片裁定和别的窗。
  - 缓存检查不支持这个试验的前提：把少上下文的读数混进全上下文读数，窗级 within 最多 +.003；去掉整片裁定 HCS −.020（`experiments/20260928_headroom/README.md` "Codex consultation, round 2" 节）。
  - 它的结论：现有约束下没有高把握的路。最值得放开的约束是允许少量片段级标注去训练读取模型（区分"提到群体"和"攻击群体"），但这就不再是 label-free。

**2026-09-28 推理部分的三个改写（用户方向：不掉分即可，目的是让 M2–M4 不那么简陋；`experiments/20260928_infer/README.md`，CPU，缓存读数）：三个都没过"消融可见"的门，`r6_bma` 不变。**
- 按读取条件分组估证据（画面读数按窗内有没有自己的帧，语音读数按词数）：六个数都在噪声内（within +.0003 / −.0024），打乱分组的对照一样。分组后的两类分布只差一点，改不了视频内排序。不采用。
- 三层读数（无话题 / 有话题无攻击 / 攻击，攻击嵌在话题里）：前置检查不成立，高话题段（中位 2 窗）比仇恨段（4 / 3 窗）短，34–43% 的仇恨段起点不在话题段里。仍跑了一次：HateMM within −.063（区间不含 0）。EM 把中间层放到 0 附近，证据斜率 3.8 → 9.9，一半的格被判成"有话题无攻击"。关闭。
- 两模态共享段边界、各自状态：HateMM +.008、HCS −.0075，区间都含 0，pooled 不变。介于独立链和共用链之间，没有可显示的作用。不采用。
- 代码：`twolevel_r2.py` 的链改成按相位描述（`KINDS`），默认参数下精确复现 `final_m2`。
- DeHate（external，2026-09-29，uoa-lab3 跑，结果已回传 `runs/20260928_infer/dehate/`，README §10.1）：相对 r6_bma 的 .7011 / .1582 / .6431，分组证据 within +.0013（打乱对照 +.0009）、共享边界 +.0058 [−.007, +.019]、三层 −.0163；pooled 都在 ±.0003 内。三个语料同一结论。
- M1 的两个消融在 DeHate（2026-09-29，uoa-lab2 跑，已回传 `runs/20260928_infer/dehate/{reads_joint,reads_seq,m1_joint,m1_seq,analysis_m1}`，README §11.1）：相对 r6_bma 的 .7011 / .1582 / .6431，**每窗一个联合分支** .7031 / .1565 / .6700（within +.027 [−.001, +.056]），**分支按顺序互相可见** .6859 / .1554 / .6123（within −.031 [−.060, −.001]，pooled ROC −.015）。联合 vs 双分支的 within 差在三个语料上是 HateMM +.002、HCS −.028、DeHate +.027：双分支不是稳定收益，只有 HCS 支持，rule 14g 下本来就不是可写的部件；分支可见在 DeHate 掉分，主语料的旧测量是 −.014 / +.011（更正：此前报的 −.034 / −.009 把 SPVL 第 1 轮的因果掩码臂错对到 r2 基线）。方法不变，双分支去留待用户裁定。
- 全部 leave-one-out 消融在 DeHate（2026-09-29，uoa-lab2，已回传 `runs/20260928_infer/dehate/`，README §12.1；相对 r6_bma 的 .7011 / .1582 / .6431，within）：区间不含 0 的有视频项（pooled ROC −.150）、视频内项（−.143）、整段转录（−.049，pooled ROC −.078）、固定 8 秒窗对 ASR 段（−.065）、分支隔离（−.031）；时序层各部件方向与主语料相同但区间含 0：耦合 −.021、最短两窗 −.016、EM 证据 −.014、分模态 OR −.013、立场轮 −.010；反向的两项：正态分数在 DeHate 有用（去掉 −.010，主语料 +.011）、固定 80 秒比按视频积分高 .011；帧在 DeHate 几乎无用（−.003，主语料 −.060 / −.109），联合分支比双分支高 .027。主语料读取消融用的是 9 月 10 日 spvl.py 的旧读数（旧 ASR 加载器）套 r3 时序层，DeHate 用当前脚本重读，去掉的东西相同、运行不同。方法不变。
- 结论与上限分析一致：在现有读数上改组合方式，within 的变化小于噪声下限；把证据估得比读数该有的更强，就掉分。更复杂的推理写得出来，但消融里看不见，就不能作为贡献写。

**2026-09-30 按视频的模态门控（用户任务：画面和语音合成一个机制，无标签、无超参地适应每个语料，达到各语料最优配置即可；`experiments/20260928_infer/README.md` §15，uoa-lab2）：判定过了，是否替换 r6_bma 待用户裁定。**
- 机制：每个模态的链按该模态单独的整片裁定做门（画面链 ← 只有帧的前缀的裁定，语音链 ← 只有转录的前缀的裁定），门 = 该裁定经一维两高斯混合校准的"违规"概率，融合时 P(hate_t) = 1 − Π_m (1 − γ_m · P_m)。EM 不变。每个视频多两次前缀编码（读取时间约 2–2.5 倍），不加窗分支。
- within（当前读数家族）：双分支 OR .7508 / .6373 / .6431；联合分支 .7607 / .5824 / .6700；只语音 .7378 / .5781 / .6525；只画面 .6766 / .5795 / .5673；**门控 .7513 / .6325 / .6638**，离各语料最优 −.009 / −.005 / −.006，pooled 变化 ≤ .001。单侧门控各偏一边（只门画面：.7397 / .6375 / .6601；只门语音：.7599 / .6303 / .6361），对称的两侧门控是唯一不需按语料选的版本。
- 限制：门不能按视频判断 OR 和只语音谁更好（ROC .40 / .53 / .51），起作用的是软加权；按视频的上限 .778 / .676 / .717 远高于此。去掉门在主语料的代价 −.0005 / +.005，rule 14g 下不能作为部件写；主语料的三套新读数在 `runs/20260928_infer/main/reads_{joint,noctx,noframes}`。

| # | concern | 做法 | 状态 |
|---|---|---|---|
| K1 | 默认方法换成 r3_m2 | 更新 STATUS、`CLAUDE.md` 项目条 | 已解决（2026-09-27） |
| K2 | 时序模块里人为设秒（平均 80 秒、形状 4） | 段长不再以秒设定：最短 = 两个读数窗（由读取网格决定），平均段长在 [两窗, 视频长度] 上积分掉 | **已解决，第 6 轮通过**（2026-09-27）。第 4 轮（对数均匀先验）8 个 MLLM 里 HateMM 只有 4/8；第 5 轮（全语料共用）HCS within −.0103，且画面链停在网格上限；第 6 轮（长度均匀先验）四条全过：不掉分、去耦合 −.114 / −.057、MLLM 8/8 与 8/8、网格一致。r6_bma 成为当前方法。代价：最短长度在此先验下只值 .030 / .009；DeHate within −.011。`experiments/20260926_twolevel/README.md` §16–§19 |
| K3 | 整段判断误报：非仇恨视频 49–67% 判违规；DeHate PR 低于 Fed-WSVAD；区间精度 .115 | 按仇恨定义拆开整段判断（针对受保护群体 T / 说话人认同 E），在视频层做合取（DVD） | **未解决**（DVD 一轮未过门，2026-09-27）。HCS pooled ROC −.022、PR −.033。原因：HCS 的 GT 把非仇恨的冒犯内容也算正例，测试集 104 个正例视频里 51 个没有仇恨片段，T 对其中 26% 答否。DeHate +.012 / +.028；只用裁定 + T 为 +.035 / +.038，超过 Fed-WSVAD。HCS 只算仇恨类（诊断，不是协议）+.018 / +.008。E 不起作用。`experiments/20260927_dvd/README.md` §9 |
| K4 | 窗级话题混淆：提到群体的效应 ≥ 真实标签；视频内远处误报 | 用 K3 得到的定义条件去问窗 | **未解决**（这轮没有新尝试；09-28 上限分析：现有读数加标签也不超过现方法）。之前 8 个方向都已按轮数归档；用 test 标签在全部窗级输出（含隐藏状态）上拟合分类器也不超过原始读数（TAD §5d）。T 本身是话题问题；E 在视频层也不起作用；逐窗问动作已由 TAD 五类动作做过。`experiments/20260927_error_analysis/README.md` 末节 |
| K5 | 短而稀疏的仇恨 within 弱 | 随 K2 检查（覆盖率 < 25% 子集） | **未解决**（只随 K2 检查；时序层解决不了）。每语料 19 个视频：按视频段长 HateMM +.032、HCS −.019，区间都含 0；去掉最短长度或去掉耦合也没有两语料一致更好。同上 |
| K6 | 故事：r3_m2 = 默认机制 + 补丁 | K2 去掉秒常数；K3 提供新机制；最终 novelty 复查 | **部分解决**。秒常数已去掉（K2）。K3 没有带来新机制（DVD 未过门）。最终复查定了可写与不可写的贡献，并按"任务挑战 → 模块 → 消融"给出故事骨架：上下文依赖 → M1；仇恨持续、无标签也不调宽度 → M3；pooled 指标主要反映视频排序 → M4 + 报 within；多次调用成本 → M1 的共享前缀。`docs/reviews/20260927_final_novelty_review.md` |
| K7 | 画面分支弱于语音 | 在已有每窗一帧读数上用 r3 组合复查（CPU） | **未解决**（每窗补一帧未过门）。每窗补一帧：HCS within +.016、PR +.013；HateMM within +.006、PR −.006。门要求两语料 within +.01 且 pooled 不掉。`experiments/20260926_twolevel/README.md` §17.1 |

## 前一方法：SPVL-r2（2026-09-10 晋级，development-selected）

`experiments/20260910_spvl/README.md`。范式：stance-conditioned evidence localization。Qwen3-VL-8B 两次前向 / 视频：(1) 公共前缀 = 规则 + 20 帧带时间戳 + 整段 Whisper 转录带时间戳，读整视频裁定 log-odds；(2) 模型自己的裁定接进前缀，每个 8 秒窗一个画面分支和一个语音分支（"这一窗是否是违规内容所在片段"），分支之间用 block-diagonal mask 隔离，窗口分 = 两分支最大值；帧分 = (裁定 + 逐窗均值) + 视频内中心化秩残差。零训练、零标签；预处理只有 Whisper 和抽帧；去掉了 v6 的 CLIP、Vid-Group、ImageBind 和十几次文本调用。

## 最新权威结果（test，4 fps；pooled ROC / pooled PR / within-video macro ROC，三项并列主指标）

| 方法 | HateMM | HateClipSeg |
|---|---|---|
| **SPVL-r2**（`runs/20260910_spvl/full2_dual_evid_stance/metrics_izv_plus_mean_rrank.json`） | **.8919 / .6831 / .6976** | **.7119 / .6664 / .6001** |
| SPVL-r2 + 每窗一帧（`full3_dual_evid_stance_w8/metrics_izv_plus_mean_rrank.json`，5 倍代价） | .8936 / .6786 / .6885 | .7199 / .6798 / .6004 |
| OMSL-v6（前一方法，`runs/20260829_omsl_v6/v6_migrated_seed0_20260909/metrics.json`） | .8507 / .5781 / .6494 | .6692 / .6622 / .5473 |
| MultiHateLoc-DMS 重跑，弱监督（`runs/20260829_omsl_v6/multihateloc_frozen_current4fps_v1_metrics.json`） | .7618 / .5188 / .6108 | .5056 / .4885 / .4996 |
| T3AL 重跑，611 视频（`runs/20260829_omsl_v6/t3al_anchor_s20250819_metrics.json`） | .6091 / .3096 / .5068 | .6246 / .5645 / .5003 |
| ZS-ImageBind 重跑，2026-09-12（`runs/20260912_baselines/zs_imagebind/metrics.json`；HCS 3 个视频无可解码视频流） | .5928 / .3108 / .5343 | .5917 / .5495 / .5241 |

PR 的随机水平 = 帧正例率：HateMM .242、HateClipSeg .473。within 只在正负帧都有的视频上算：HateMM 84 / 215，HCS 99 / 118。晋级门（相对 v6，噪声下限 pooled .005 / within .01）：HateMM +.041 / +.105 / +.048，HCS +.043 / +.004 / +.053，全过。消融表 `runs/20260910_spvl/ablation_table.md`；MHC 历史数字见 git 历史。

## 输入与缓存

`data/omsl_v6_inputs/`（约 82M，`PROVENANCE.md`）：manifest、A10 视觉曲线、Qwen3-VL-8B 分块文本 log-odds、ImageBind 音频 embedding、整视频 logit。`data/gt_4fps/`：GT 数组。`data/assets/imagebind/`：ImageBind 权重与缓存的文本锚点。1 fps 时代输出在 `runs/legacy_1fps/`（本机 17G + lab2 回传 51G），2026-08 idea discovery 中间产物在 `runs/legacy_idea_discovery_2026-08/`。lab2 上另有 `results/steward_private/thvl_bench`（23G，THVL-Bench 数据集材料，含加密标签），未回传，留在 lab2。

## 运行任务与监控

HVL（假设–验证闭环，`experiments/20260911_hvl/`）2026-09-11 试运行两轮结束：假设条件化、顺序状态链、相邻窗语境、判定修订全部低于 SPVL-r2（within 子集基线 .693 / .601），机制诊断见其 README §7–§8；按规则 9 归档为负结果，SPVL-r2 仍是当前方法。MLLM family / 尺寸鲁棒性研究（spvl README §11）2026-09-11 04:24 全部完成：7 个模型 × 7 个臂，结果表 `runs/20260910_spvl/mllm_table.md`，解读在 README §11 Results。结论：方法在 7 个 MLLM 上都有效（within 从模型自身的 .62–.64 / .49–.58 提到 .65–.70 / .55–.60），固定窗、M3、帧三个部件在 ≥5/7 模型上方向一致；立场条件化只在 HateMM 上过数；整段转录语境只对 Qwen3-VL 系列和 Gemma 有用。派发记录：2026-09-10 21:00 起并行跑在 uoa-lab3（Qwen3-VL-8B 一致性检查、4B、2B）、uoa-lab2（InternVL3.5-8B、LLaVA-OneVision-7B）、lab-server（Gemma-3-12B）、uoa-campus1（Qwen3-VL-32B，job 16689）、uoa-campus2（Qwen2.5-VL-7B，job 19985）。输出 `runs/20260910_spvl/mllm/<tag>/<arm>/`，汇总 `runs/20260910_spvl/mllm_table.md`。round-3 消融已完成（`runs/20260910_spvl/abl3_*/`）。

## 2026-09-12 这一轮：PWC 负结果 + 机制诊断

**PWC（窗间成对比较，`experiments/20260912_pwc/`）在 E0 被证伪，已归档。** 声明的门是比较的准确率比
`sign(z_i − z_j)` 在两语料都高 ≥5 点；实际 −1.0（HateMM）/ +0.5（HCS）。704 对，
`runs/20260912_pwc/e0b/summary.json`。读出本身没问题：交换一致率 .82–.86、A/B/C 边缘概率无偏好、撤掉转录
掉 .119/.054。所有读出方式（绝对、相对、联合上下文）都落在 .55–.69 同一带内，彼此符号一致率 .71–.83。
结论：视频内排序的上限不是读出方式，是模型能分辨多少（H2 而非 H1）。

顺带零成本否掉的机制：**语境对比**（`z(全局语境) − z(窗单独)`）within .467/.494，对照 `z(全局语境)`
的 .699/.602（`runs/20260910_spvl/mllm/q3vl-8b/{full,winonly}`）。

**机制诊断（PWC README §7c）**：把窗按 GT 标签 × 该窗是否提到目标群体交叉列表，"提到目标群体"值
**+8.1（HateMM）/ +7.0（HCS）** log-odds，"真的是仇恨窗"只值 **+4.3**。冻结 MLLM 的窗级判断主要是
话题检测而不是行为检测。这解释了 PWC 的失败、七个模型同一个 within 天花板、以及视频级强而窗级弱。

**TAD（`experiments/20260912_tad/`）两轮都没过门，已归档。**
- round 1 减掉话题维度：`a − β·t` 在两语料都掉，且随 β 单调下降（HateMM .6926 → .6419 → .6159 → .5749）。
  原因是 topic 单独就能排到 .6567/.5467——提到受保护群体在这两个语料里是正向预测的，减掉等于减信号。
  加法对照（秩和）也掉（.6858/.5811）。用别的视频的 t 作对照，β 中位数塌到 0。
- round 2 五选一言语行为：模型 79.6%/56.0% 的窗答 `attacks`，`unrelated` 18.6%/34.9%，三个豁免选项
  （reports/quotes/condemns）合计只有 1.9%/9.2%。五选一塌回二元，就是话题轴（与 act 读数 Spearman .73/.70）。
  within：actmargin .6800/.5998，与 act 的秩和 .6945/.6076，都在噪声内。
- **oracle 天花板**（真标签监督、按视频分折、out-of-fold，仅诊断）：在 MLLM 各路读数 + 窗形状 +
  ImageBind 音频 + 相邻窗读数上拟合，都打不过不拟合的原始窗分（HateMM .758 vs .742/.731/.744；
  HCS .621 vs .625/.626/.614）。唯一例外 GBT+音频在 HateMM +.035，HCS −.069 不迁移。
  结论：这一族标量特征的组合空间已到顶，伪标签自训练头也被它框住。
- 规则 4 review 的第一个对照（act 臂必须复现 SPVL-r2）抓到 compose 里一个重采样 bug，修好后 act =
  .6926/.6007，与 HVL 同代码路径基线一致。诊断的效应量按计数加权后是 1.15 倍（HateMM）/ 1.47 倍（HCS），
  不是之前写的 1.6–1.9 倍。

**表示层天花板也到顶**（`runs/20260912_tad/e0_hidden/`）：把窗分支读出位置的隐状态（Yes/No 投影丢掉的
那部分）拿去拟合，64/128/256 维分别是 .747/.608、.746/.594、.698/.552，都不如不拟合的 .758/.621。
局限已记：向量读在决策之后，前缀里的预决策表示没探。

**改模型这一族（`experiments/20260912_sdl/` 已归档，`experiments/20260912_nga/` 进行中）**

冻结模型的读出、特征组合、表示三条路都封死之后，唯一没试的杠杆是改模型本身。用模型自己的整视频裁定当
伪标签（零人工标注，与 T3AL 同属 test-time adaptation），rank-8 LoRA 只训语言层注意力投影，前缀 no-grad
且关适配器，推理与训练一致。

| 轮次 | 目标 | HateMM within | HCS within | 失败机制 |
|---|---|---|---|---|
| 冻结基线（同代码路径） | — | .6968 | .6020 | — |
| SDL r1 | MIL + 保留立场轮 | .5026 | .5091 | 立场轮把袋标签泄露给每个窗分支，学成按伪标签整体平移（伪负 −17.67 / 伪正 +1.09） |
| SDL r2 | 去立场轮，BCE | 中止 | 中止 | 冻结模型已满足袋约束且 log-odds 饱和到 ±15，损失 0.000–0.03，无梯度 |
| SDL r3 | hinge on raw log-odds | .5092 | .5337 | "只有一个窗为正"是假约束（仇恨占比中位 .24/.47），判正率 45.8%→12.3%，曲线被压平 |
| NGA r1 | 只约束伪负视频 | .6787 | .5875 | 可用整体压低满足，均值 −22.6，标准差 4.46→1.33 |
| NGA r2 | + 正例锚定（0.1） | .6812 | .5996 | 锚定修好压缩（标准差 4.40/5.50、判正率 48.2%/50.8%、Spearman .95），但模型几乎不动 |
| NGA r3 | 锚定 0.03（声明的扫描） | .6723 | .6009 | 三个锚定值都在基线下 / 噪声内，无趋势 |

**NGA 三轮用完未过门，已归档；整个"改模型"方向结束。**合并结论：模型自己产生的视频级伪标签
不足以教会它"视频里哪一段是仇恨的"。三种用法各自因不同且已诊断的原因失败——标签泄露进被监督者的
上下文、正例侧袋约束在数据上为假、只用负例侧可被整体压低满足；三处同时修好之后，剩下的监督量是
333 个视频里的 79 个（HCS 只有 10 个），模型不动。

**最后一个诊断：预决策的前缀表示也没有信息**（`runs/20260912_tad/prefix_probe/`）。用真标签拟合模型在
整视频前缀里对每个窗画面的上下文编码：.6568/.5042（64 维逻辑回归）、.6067/.5178（128 维 GBT），而同一批
窗上不拟合的冻结读数是 .7303/.6143。**比决策本身更差**，在画面承载的 HCS 上也是。所以视频内排序不是被
Yes/No 投影丢掉的，它一开始就不在表示里。局限：只定位了图像 token，转录行的映射没建。

规则 4 的三次 proposal review 全部放行，并抓到多个实现问题（compose 重采样 bug、适配器作用域不一致、
mean-pool 对照训反、稀疏系数被隐式放大、`--only-within-defined` 用 GT 选训练集违反规则 10），均已修复。

**新增 secondary 评测**（用户裁定 2026-09-12）：`data/gt_4fps_hate_only/HateClipSeg.npz`，只取 Hateful 这
一类（base rate .198 vs 主表 .471，含正负帧的视频 51 vs 99）。主表不变，只作并列诊断。

SPVL-r2 在两个口径下（`runs/20260910_spvl/full2_dual_evid_stance/metrics_izv_plus_mean_rrank{,__hate_only}.json`）：

| GT 口径 | pooled ROC | pooled PR | 随机水平 | within | n |
|---|---|---|---|---|---|
| 主表（offensive 并集） | .7119 | .6664 | .473 | .6001 | 99 |
| 只取 Hateful | **.7774** | .4323 | .200 | **.6236** | 51 |

PR 对随机水平的倍数：1.41 → **2.16**。口径对齐后三项全部变好，说明主表上 HCS 的数字被标签定义压低，
这一点现在有量化依据。

## 2026-09-12 收口：范式边界已测定，等用户裁定范围

一天内五个候选（PWC、TAD×2、SDL×3、NGA×3）全部归档为负结果，但它们合起来测定了一条边界：

**冻结 Qwen3-VL-8B 在 8 秒窗粒度上的视频内定位能力是窗级 .758 / .621（帧级 within .6968 / .6020），
下游任何东西都突破不了它。** 依据：
- 换读出（相对 / 绝对 / 联合上下文）全落在同一个 .55–.69 准确率带；
- **用真标签**拟合所有缓存特征（读数、窗形状、ImageBind 音频、相邻窗）打不过不拟合的读数；
- **用真标签**拟合分支隐状态（决策之后）打不过；
- **用真标签**拟合前缀视觉表示（决策之前）更差；
- 零标注的视频级自监督适配（六轮）全部低于基线。

**提问形式也全部试完**（`experiments/20260912_bnd/`，已归档）：问"违规内容是不是从这一窗开始/结束"，
begin .6466/.5605、积分成的状态 .6146/.5265、与 act 的秩和 .6956/.5949，都低于 act 的 .7581/.6212。
它是独立测量（与 act 读数 Spearman .644/.731，未退化），模型也有一点起点感（begin 的最大值落在第一个
GT 正窗上 22.7%/15.6%，随机约 4–5%），但问变化没能把话题混淆差分掉。

至此，冻结模型上六种提问形式（绝对、相对、减话题、五选一行为、问变化、反事实排除）+ 四道真标签天花板
（特征组合、决策后隐状态、决策前前缀表示）+ 六轮零标注适配，全部测完。

**2026-09-12 傍晚补充：上面那句"天花板"是在三个未被动过的常量下测的**——冻结模型、固定 8 秒网格、
全局共享 prefix。当天晚些时候的诊断（`archive/experiments/20260912_cva/` §1(b)，输出
`runs/20260912_cva/diag/frame_coverage.txt`）说明第三个常量本身有问题：20 帧共享 prefix 下，
**HateMM 24.2%、HCS 34.1% 的窗口一帧都没有**，而视觉分支在同一批视频内配对比较时，在**没有帧**的窗口上
排序反而更好（.5663 → .7251，.5417 → .5622）。它和语音分支的视频内 Spearman 只有 +.111 / +.176，
不是在抄语音分支。所以 .758 / .621 是这个决策网格的上界，不是模型的上界。这条诊断至今没有被任何候选解决。

### CVA（反事实归因，第 6 个归档候选）

`archive/experiments/20260912_cva/`。用整片判断的差分给区间打分：`s_i = z_video − z_excl(i)`，
`z_excl(i)` 是同一个整片问题、但要求模型忽略区间 i。规则 4 放行（检索确认 LOO/occlusion 归因在文献里只有
LLM 文本上下文归因、对已训练分类器的事后解释、弱监督 VAD 的训练期正则三种用法），规则 6 code review 干净。

E0（`runs/20260912_cva/e0/metrics_izv_plus_mean_rrank.json`，一次运行，333 视频 0 错误）：
HateMM `.8834 / .6572 / .6747`，HCS `.6724 / .6215 / .5779`。六项全部低于 SPVL-r2，规则 9 归档。
代价预测兑现：HateMM 186 s、HCS 173 s，读数约为 SPVL-r2 的一半。

三条有信息量的对照：
- 打乱对照（视频内置换 s）掉到 .5079 / .5143 —— 模型**确实**按被排除的区间响应，机制不是虚构的，
  只是比直接问窗口弱（.7183 / .5908 对 .7563 / .6190）。
- `rank(s) == rank(−z_excl)` 在 **100%** 的视频上成立。视频内 `z_video` 是常数，而残差本来就是视频内秩，
  所以减掉一个视频内常数**对 within 指标是空操作**。提案 §2 承诺要抵消的那一项，在 CVA 写出来之前就已经
  被秩残差抵消掉了。这是提案里的实质错误。抵消论证要有用，被减掉的那一项必须在视频内变化（TAD 减的
  确实在视频内变化，所以它能改变秩——只是改错了方向）。
- `s` 的均值是 +3.57 / +3.98（排除子句本身值约 4 log-odds，与排除哪个区间无关），而视频内标准差只有
  0.70 / 0.51，对比 SPVL-r2 直接读数的 6.20 / 5.69。子句自身效应约为区间效应的六倍。

**需要用户裁定的范围选项**（当前方法仍是 SPVL-r2，SOTA 门是过的）：
1. 放宽标注预算，允许真视频级标签做弱监督（失去 label-free 定位，但 backbone 远强于 MultiHateLoc）；
2. 把 HCS hate-only 口径扶正为主表（今天已建好 GT，口径对齐后三项全部变好）；
3. 加数据集（规则 1 需用户同意）；
4. 换任务粒度，报区间级指标（F1@IoU，从未报过）；
5. 接受经验研究框架（用户 2026-09-12 已明确否决）。

其它待办：
- LELA（GPT-4o-mini）对照在本评测器下重跑（规则 14f）。用户 2026-09-12 裁定：先不做，投稿前再补。
- OMSL-v6 目录保留为对照，下次整理时移入 `archive/experiments/`。

## 2026-09-22：TIL（区间测量的时序推断）—— 错位网格证伪，显式时长先验过门

`experiments/20260922_til/`（规则 4 放行、规则 6 无阻塞，审稿记录 `docs/reviews/20260922_til_proposal_review.md`）。
测量：与 SPVL-r2 相同的逐窗读数，跑在两套 8 秒网格上（A 从 0 秒起，B 从 4 秒起；`runs/20260922_til/gridA`、`gridB`，
lab3 / lab2，各 333 视频 0 错误）。推断：4 秒格上的显式时序模型（区间支撑观测模型 + 以秒计的停留先验，参数零标签）。

结果（`runs/20260922_til/infer/<arm>/metrics.json`，pooled ROC / PR / within）：

| 配置 | HateMM | HateClipSeg |
|---|---|---|
| SPVL-r2（当前方法） | .8919 / .6831 / .6976 | .7119 / .6664 / .6001 |
| **单网格 A + 时长先验 80 s（A1）** | **.8921 / .6840 / .7486** | **.7131 / .6677 / .6247** |
| 双网格 + 区间观测模型 + 先验（A4，候选） | .8920 / .6835 / .7321 | .7129 / .6677 / .6203 |
| 双网格平均 + 先验（A3，零假设） | .8920 / .6836 / .7366 | .7130 / .6677 / .6232 |

- **错位网格被证伪**：A4 − A3 = −.005 / −.003，加第二套网格反而比单网格差（A3 − A1 = −.012 / −.002）。原因：
  半重叠窗口的读数只有 .57 / .63 的 Spearman 一致性，逐窗读数不是窗内内容的稳定函数，没有可反演的边界信息。
  "测量几何 + 观测模型"的范式主张按声明撤回。
- **显式时长先验过晋级门**：within +.051 / +.025，pooled 三项都在噪声内；停留 40–160 s 扫描最差 .7459 / .6210；
  按语料或合并尺度一样；缓存的 8 个 MLLM 上 7 个方向一致（README §6）；去掉先验 −.074 / −.023（规则 14g 过）。
  但按规则 4 情形 (3) 它是纯平滑，不能作为新方法，只能作为 SPVL-r2 的显式部件（规则 3）。是否把
  "SPVL-r2 + 时长先验" 定为当前方法的报数，等用户裁定。
- 对比 HVL：同一个持续性，让模型在上下文里自己维护（状态链）掉 .03 / .04–.07；作为显式先验加在独立读数上涨 .05 / .02。
  这是"模型只做测量、推理放在模型外"这条设计原则目前最强的证据。
- 2026-09-13 的代码审计（四个独立 agent，覆盖 HVL / PWC / TAD / SDL / NGA / BND / CVA 与共享代码）：没有推翻归档结论的
  bug；四条表述待改（帧覆盖线索只在 HateMM 成立；前缀表示探针受因果掩码限制，"排序不在表示里"无证据；隐状态探针误用
  visual 分支，speech 分支 HateMM +.010 到噪声线；SDL/NGA 基线应为无立场路径 .6899 / .5881）。尚未写入上面各节。

## 2026-09-26：ASR 修复、基线重测、平滑对照、GLR 负结果

**ASR bug 修复**（`src/video_inputs.py` `load_asr`）：Whisper 最后一段缺结束时间戳（偶尔也缺开始）时，这一段此前被整段丢掉。
测试集 HateMM 42 条、HCS 95 条转录受影响（例：hate_video_321 在 14 秒之后没有转录）。现在补上：缺开始取上一段的结束，缺结束取
音频时长；`fill_untimed=False` 复现旧行为。SPVL 以来的所有 run 都受影响；`experiments/20260910_spvl/spvl.py` 里的旧副本没改。

**修复后重测**（lab-server，`runs/20260926_glr/base_gridA`，`til_measure.py` 网格 A，代码不变），pooled ROC / pooled PR / within：

| 组合 | HateMM | HateClipSeg | 来源 |
|---|---|---|---|
| SPVL-r2，修复前 | .8919 / .6820 / .6926 | .7129 / .6673 / .6007 | `runs/20260922_til/infer/A0_spvl_replicate/metrics.json` |
| SPVL-r2，修复后 | .8952 / .6867 / .6800 | .7134 / .6667 / .6101 | `runs/20260926_glr/infer/base_gridA_spvlr2/metrics.json` |
| + 时长先验 80 s，修复前 | .8921 / .6840 / .7486 | .7131 / .6677 / .6247 | `runs/20260922_til/infer/A1_gridA_prior80/metrics.json` |
| + 时长先验 80 s，修复后 | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 | `runs/20260926_glr/infer/base_gridA_d80/metrics.json` |

变化都在按视频抽样的噪声内（within 均值的标准误 .028 / .021；规则里 .01 的噪声下限只考虑了换种子，没考虑换视频）。
之后的比较以修复后的数字为基线。

**平滑对照**（`experiments/20260922_til/README.md` §10）：同一批读数上用高斯平滑代替时长先验。σ = 8 s 最好
（within .6877 / .6312），σ ≥ 32 s（与先验时长同量级）比不平滑（.6642 / .6153）还差。先验的涨幅不是线性平滑能给的：
它在证据变化大的地方保留边界，只填弱的空隙。HCS 上最好的高斯只差 .005，HateMM 上差 .067。

**GLR（生成式似然比，`experiments/20260926_glr/`）负结果，已归档。** 思路：不再问模型"这一窗是不是仇恨"，而是比较这一窗
实际说的话在"违规说话人"和"合规说话人"（讨论 / 转述批评 / 粗口但不针对群体）两种条件下的概率，希望话题在两边抵消、只剩行为。
lab-server，333 视频 0 错误，约 12 s / 视频。预先声明的门（窗级 within 相对语音分支 `z_speech`，按视频配对 bootstrap，
两语料 95% 区间都 > 0）主变体和三个扫描变体全部不过：窗级 within .49–.57，`z_speech` 是 .6935 / .6035，差值区间全在 0 以下
（`runs/20260926_glr/analysis/table.txt`）。帧级把语音分支换成它，within 掉 .12–.22 / .05–.08
（`runs/20260926_glr/infer/glr_*/metrics.json`）。不是代码问题（校验通过、窗对齐无误、去掉长度影响后不变）。原因：口语的词
本身很难预测（每 token −5.6 / −6.5 nats），两种说话人条件平均只改 ±.01 nats / token；判断最明确的窗上方向是对的，但差值只有
窗间波动的四分之一左右。它剩下的那点信号在视频之间（视频级 AUC .61–.77），视频内部接近随机。

当前方法不变：SPVL-r2（+ 时长先验，是否定为报数方法仍等用户裁定）。

## 2026-09-26（续）：部件的概率化改写，第 1 轮不通过

用户方向：方法已经 work，但部件太简陋。保留每个部件的机制，改得更有原则；不掉分、消融还在起作用即可。
`experiments/20260926_twolevel/`（只用 CPU，缓存读数 `runs/20260926_glr/base_gridA`）：
- 时长先验和双分支改成每个模态一条链，参数由 EM 从无标签读数里估；
- 截距和秩改成 P(整片违规) × P(这一帧在违规段 | 整片违规)，并能输出区间。

按预先定的规则不通过（`runs/20260926_twolevel/analysis/table.txt`）。只换时序层，within 就掉 .042 / .018；
再换组合方式，HateMM pooled 再掉 .05 / .09。机制本身还在：去掉时间耦合，within 再掉 .075 / .028。
新增的区间输出 F1@.3 约 .30（现在的方法没有区间）。

诊断出的原因：EM 估出的读数模型把每个窗的读数当成很强、彼此独立的证据，持续先验几乎不起作用；
同样原因让贝叶斯的整片概率排视频不如 `z_video + 窗均值`。现在的方法恰恰靠"读数是弱证据、先验说了算"。
同一段里的读数高度相关（共享整段语境），有原则的改法要把这一点建模进去。
次要：静音处强制语音为 0 会掉分；两条链取"任一"在 HCS 上比先取最大差 .013。

## 2026-09-26（续）：部件改写第 2 轮通过（时序层）

`experiments/20260926_twolevel/README.md` §10（CPU，同一份缓存读数）。组合方式不变，时序层改了三处：
- **证据**：每个模态的"非违规均值、违规均值、方差"由 EM 从无标签读数估出，代替"除以语料标准差"；
- **持续时间**：违规段和间隔的长度服从负二项分布（形状 4，平均 80 秒），一个窗不能单独成段，代替几何分布；
- **模态**：每个模态一条链，任一条处在违规状态就算违规，代替取两分支最大值。

结果（`runs/20260926_twolevel/analysis_r2nl/table.txt`；k = 4、按模态分链、去掉泄漏项都是看 test 选的，
development-selected）：

| | HateMM | HateClipSeg |
|---|---|---|
| 现方法（SPVL-r2 + 时长先验） | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 |
| 改写后（`r2_noleak`） | .8955 / .6888 / .7579 | .7137 / .6664 / .6428 |

- 六个数都在噪声内：不掉，也不算涨。
- 去掉时间耦合，within 掉 .147 / .043（区间都不含 0），持续先验仍在起作用。
- 换回几何分布只在 HateMM 掉 .045，所以"段长分布"不能单独作为 novelty 主张。
- 两个模态共用一条链的融合，在两个语料上都比按模态分链差。
- 新增区间输出，F1@.3 .321 / .283。

找到的机制：EM 估出的证据很强（是真实强度的 5–14 倍）。在几何分布下，一个窗的强读数就能自成一段，平滑失效；
规定一段至少跨两个窗以后，平滑恢复。现方法靠"除以标准差把证据压弱"达到同样效果。

试过、没用的：
- 每个视频内先减均值再跑 EM；
- 按"留一窗预测"选证据强度（选出的正是第 1 轮的设置）；
- 模型里保留立场泄漏项（去掉反而更好）；
- 五种原则化的视频层写法，都让 HateMM pooled PR 掉 ≥ .013。

视频层的原则化写法都不如现截距。原因：现截距用 MLLM 原始 logit，数值跨度大，不同视频的帧几乎不交错；换成校准过的后验，
跨度变小，交错增多，PR 下降。

**组合方式（同一 README §11）。** 截距 K = z_video + 窗均值保持不变，改成它的对数几率：logit P(V=1 | K) = aK + b。
a、b 由 K 在本语料上的两类高斯混合无标签估出（a = .357 / .341）。视频内项仍是居中秩。

| | HateMM | HCS |
|---|---|---|
| 时序层改写 + 校准截距（`c_m2`，`runs/20260926_twolevel/analysis_c/table.txt`） | .8970 / .6959 / .7579 | .7170 / .6706 / .6428 |

- 相对现方法六个数都不掉；只有 HateMM pooled PR +.007 超过噪声下限。
- 去掉视频项，pooled ROC 掉 .33 / .15；去掉视频内项，within 掉到 .5。
- 扫描发现：截距缩得越小（视频内排序的权重相对越大），两个语料的 pooled 都越高（缩到 1/8 时
  .8993 / .7010、.7217 / .6749）。但目前没有无标签的办法定这个宽度，所以没用，只作开发期证据。
- 区间输出来自乘积 P(V=1 | K) × P(帧违规 | V=1)：F1@.3 .318 / .316。


## 2026-09-26（续）：部件改写第 3 轮、跨模型检查

`experiments/20260926_twolevel/README.md` §12–§14（CPU，缓存读数）。

**跨模型检查。** 第 2 轮的时序层在 8 个 MLLM 的缓存读数上，within 不掉的只有 5/8（HateMM）和 7/8（HCS）。
掉分的是 EM 把证据估得极强的模型：LLaVA、Gemma 的画面分支达到现方法证据的 13–20 倍。

**第 3 轮（最后一轮）。** EM 之前，把每个模态的读数换成它在语料内排名的正态分数：只用读数的先后顺序，不用各模型不同的
logit 尺度。

| | HateMM | HCS |
|---|---|---|
| 现方法（SPVL-r2 + 时长先验） | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 |
| 第 3 轮（`r3_m2`，`runs/20260926_twolevel/analysis_r3/table.txt`） | .8971 / .6953 / .7525 | .7168 / .6705 / .6397 |

- 六个数都在噪声内，不掉。
- 去掉时间耦合，within 掉 .116 / .060（区间都不含 0）；换回几何分布掉 .014 / .010。
- 8 个 MLLM 上 within 不掉 7/8、8/8，平均 +.021 / +.025（`runs/20260926_twolevel/robust/table_r3.txt`）。
  仍掉的：Qwen3-VL-2B 在 HateMM 上 −.027，LLaVA 在 HateMM PR 上 −.014。
- 区间输出 F1@.3 .322 / .273。
- 读数部件在新方法下：去掉立场掉 .014 / .011，去掉帧掉 .060 / .109。整段转录在 HCS 上不再有作用（+.001），
  双分支在 HateMM 上不再有作用（+.009）。
- 所有选择都是看 test 定的（development-selected）：正态分数、形状 4、按模态分链、去泄漏项、校准截距。

改写后的方法和现方法相比，改了五处：
1. 读数变证据：排名正态分数 + EM；
2. 段长：负二项分布；
3. 模态：分链取"任一"；
4. 截距：校准成对数几率；
5. 新增区间输出。

读数模块（SPVL-r2 的 MLLM 调用）不变，没有新增模型调用。是否用它替换现方法，等用户裁定。

## 2026-09-27：让模型自己学"失手"和持续时间，不通过

`experiments/20260926_twolevel/README.md` §15（CPU，缓存读数）。

**先用标签做的分析（只作分析）。**
- 违规视频里，高分却不在违规段的连续窗，55% / 74% 只有一个窗长；真违规段中位数 4 / 3 个窗。
- 孤立误报窗在视频内的平均排名，直接用读数时是 .63 / .78，第 3 轮压到 .34 / .49，低于真违规窗的 .55。
- 加上立场后，判为违规的视频里所有窗整体抬高约 2.4 / 2.1 logit，窗之间的先后几乎不变（Spearman .93 / .92）。

**试的改法：** 在时间层加一类"失手"时段，它的读数和违规一样高，但可以出现在任何视频里。所有时段的时长由 EM 学，
代替手设的"形状 4、平均 80 秒"。

**结果（`runs/20260926_twolevel/analysis_s/table.txt`）：不通过。**
- HCS within 掉 .019；去掉"失手"类反而更好（+.002 / +.011）；8 个 MLLM 上 within 不掉的只有 6/8、5/8。
- 学出的"失手"平均持续 48–104 秒，并不短。
- 原因：不用标签时，短暂的误报和短暂的真违规分不开（真违规段有 22–25% 只有一个窗长），多出来的那一类被拿去解释别的结构。

**附带试了"两类时段、时长由 EM 学"：** 主结果不掉（.7586 / .6284），但跨模型只有 6/8、7/8 不掉，LLaVA 在 HateMM 上掉 .102，
也不通过。

**结论：** "违规会持续"只能作为事先声明的假设放进方法（第 3 轮的最短时长），无法从读数里学出来。候选仍是第 3 轮（`r3_m2`）。

## 资料与历史

[试过的做法（方向索引）](DIRECTIONS.md)、[CLAUDE.md](../CLAUDE.md)、[1 fps 协议与 baseline 表（历史）](../docs/protocol_1fps_legacy/)、[基础论文 TRIAGE 与检测时代记录](../archive/README.md)、[2026-08 idea discovery 报告](../archive/idea-stage-2026-08/idea-stage/IDEA_REPORT.md)。


候选34有序条件事件槽来源选择：原九池C8/rank7一次方案PASS，实际Q2E v1 §3.1–3.4/A.3/A.4及官方生成入口/模板核读；不是原多模型熵融合复现，三槽目标可分离，不主张DP复杂性。入口`experiments/20261006_m1_ordered_slots/README.md`。完整prototype/全333真实输入/8组实际36层新图cached/fullreference/精确正文pooling/有序可行解枚举及20组生产reader CPU PASS；仅软件输入检查，唯一独立Rule6 code review进行中，无本候选GPU/GT/性能结论。来源`runs/20261006_m1_ordered_slots/`各CPU summary及manifest。

候选34唯一独立Rule6审查PASS，来源`docs/reviews/20261006_m1_ordered_slots_code.md`：实际合成video acquire/validate与20组真实36层CPU生产reader/独立token pooling/原始输入归属和计费验证通过；只代表软件输入检查，实际8B固定5待提交。

候选34 actual8B固定5已在sc474398提交Slurm205，派发检查通过、来源头部CPU记录已回传本机；等待QOS预算，无GT/性能。来源仍为实验README/manifest。


候选35完整视觉时间工具交互：原九池C1/rank8一次方案PASS，VTimeCoT v1方法/Algorithm1及实现细节实际核读，官方code未发布不声称复现。入口`experiments/20261006_m1_vtimecot/README.md`。冻结1fps真实来源/同Qwen clip相关性prefix共享/模型驱动progress/highlight/cut/真实更新与history views进入V、G/S固定；不是旧1fps评测。完整prototype、全333输入、实际36层CPU source cache/cachedvsfull/allKV/最多24新图、实际10窗video全部工具和只读重放、12组productionreader作者CPU PASS，仅软件输入证据，唯一独立Rule6审查进行中，无实际GPU/GT/性能。来源`runs/20261006_m1_vtimecot/`各CPU summary及manifest。

候选35一次独立Rule6审查PASS，来源`docs/reviews/20261006_m1_vtimecot_code.md`：实际collector/缓存共享全部KV与三轴/fullreference/12组生产reader/nativeG/S/allraw/24图history及计费核验通过。same-family provisional，仅软件输入证据；actual8B固定5待提交，未GT/性能。

候选35实际8B固定5已在sc474398提交Slurm206，派发检查通过/source与reader同机，等待activeQOS2；未本候选GPU/GT/性能。来源实验README/manifest及本机机器检查。

2026-10-06调度顺序：尚未分配GPU的完整192/193由主agent hold，先取得201/202/205/206实际8B固定5结果；168/186完整运行继续。没有方法/修订/成本重置，原192/193作业保留待主agent释放。本机记录`runs/20261005_m1_{spatial_search,verification}/scheduling/hold_before_start.json`；属于实验室own hold，非校区初始自动held任务。

2026-10-06 actual反馈：26完整333 DONE、来源全部本机；时间原点差异为av17/18生成与审计runtime差异，仅prepare切换后原精确帧/像素条件保留、独立PASS，全轮核对重跑中。33固定5本机严格PASS；32/34/35固定5原执行guard FAIL，真实cap/UNKNOWN保留，未GT/指标/idea裁定。原192/193 own lab hold将按较低成本完整候选恢复；细节与权威路径在各实验README。

空间搜索31原完整333作业192已释放并RUNNING/sc474398，短fixed5反馈已经BOTH回传；按smoke实测新处理成本选为下一完整候选，不评价尚未生成的主指标。33完整ready，27/193保留ownlabhold。来源实验README与本机释放机器检查/调度记录。

有序事件槽34接口A实际caption全部不可用，独立窄诊断无实现偏离（153wordcap/4tokencap/1UNKNOWN）。显式B只增加模型可见12/6words结束指令，原caps/guard/科学计算不变，独立缓存/输出，默认A及原失败保留；作者CPU/独立窄确认PASS，来源`docs/reviews/20261006_m1_ordered_slots_interface_B_code.md`。同固定5 B actual8B ready，未GT/性能/预算重置。35窄diagnosis也无实现bug、全部query/reason触wordcap；来源对应review，尚未改guard/数据。
