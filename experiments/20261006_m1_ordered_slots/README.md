# M1候选34：条件事件槽的有序实际来源选择

2026-10-06，R1在任何实际8B或本候选GT读取前冻结。原九池C8/rank7的一次独立方案裁定PASS：
`docs/reviews/20261005_m1_ideation_jury.md`。本机sc474397实现，运行主机待派发记录。

## 科学假设与来源

孤立窗口可能缺少前置对象或后续回应，导致本窗可见/可听证据解释错误。
为当前事件构造中性前置/当前/后续检索条件，按真实时间绑定前后媒体，再读取当前证据，
有望改善原始V/S/max及最终within排序。生成的条件不是已发生事实，不能进入最终证据。

实际核读[Q2E v1](https://arxiv.org/html/2506.10202v1) §3.1–3.4、A.3/A.4及
[官方代码](https://github.com/dipta007/Q2E)的query_decomp、frame_caption、frame2video_caption与
contextualized_frame_caption+ASR、prequel/during/sequel/refine_event_query提示。
本地读取记录在`runs/20261006_m1_ordered_slots/source_reading/`。
Q2E以事件分解、顺序描述和多通道融合做视频检索；本方法自研有时间可行域的实际来源绑定，
只借鉴前后事件条件和顺序描述。原多编码器、五事件扩展、翻译链及熵融合未迁移，
不称完整Q2E复现，不称首次事件分解/检索。

三槽目标可分离：current固定为当前窗，prequel严格更早，sequel严格更晚。
实现直接取两侧最优值，与三状态单路径最优化等价；无跨窗口资源耦合，
不将DP或全局优化复杂性当作贡献，不添加未审的新耦合。

## 冻结算法、常数和缺失规则

所有prompt及数值常数在`spec.json`；两语料相同。Qwen/Qwen3-VL-8B-Instruct冻结，
贪心、seed0、FP32 Yes/No margin。原native20请求/实际18–20帧、完整ASR、G及其自身hard stance、
独立8s V/S、max及固定r6全部保持原协议。来源获取为同一视频的真实输入；不读取标签。

每8s窗在1/3、2/3时间点取首个不早于目标的实际帧，去重，尾部目标无帧时用本窗最后真实帧；
保存原始解码序号、PTS及RGB PNG，半开区间归属，不插值。每窗一次两帧+本窗ASR的caption生成，
前一窗有效caption作为明确未核验上下文（缺失用UNKNOWN），16words/32tokens/整条96tokens上限。
按最多8窗的连续批生成三槽，固定窗口ID/顺序，槽8words/16tokens、整批768tokens。
空、UNKNOWN/NONE开头或字段上限均不可用；整条/整批截断或恰好总token上限均全拒绝。
不重试，不补猜测，不为执行guard强迫可用条件。

caption及可用三槽用同一个Qwen做短文本prefill embedding，不另加编码器。
固定embedding_system，按实际native fast tokenizer offset精确选user正文重叠且非special的token，
last-layer FP32 mean、L2归一化、epsilon1e-12。系统/角色/尾部token不入池，不截断。
无有效token或零/非有限向量为不可用。cosine用CPU FP32、无跨视频/语料归一化。
current始终绑定本窗；其可用匹配分数是目标常数，不影响前后选择。
prequel候选j<i、sequel候选j>i，且caption embedding可用。
每侧最大cosine严格大于NONE分数0才绑定一窗，否则NONE；平局最早真实窗ID。
同时间窗按原半开区间序号处理，自己不会成为remote。
匹配分数只选择输入，不进入最终仇恨分数。

最终V附加当前LOCAL两帧及最多前后各两帧，最多6张新图，以真实window ID/边界/PTS及
prequel/current/sequel角色标记呈现，只有原始ASR，不传caption、槽条件、embedding或匹配分数。
不存在的槽明确NONE；缺少当前真实帧则新鲜原生V回退。
S只在本窗ASR非空时读取，追加真实前后ASR及来源角色/时间；无前后speech时新鲜原生S逐值相同。
所有remote事件保留自身时间，不作为当前事件发生事实。
共享`src/source_image_branch.py`使用实际新图编码、DeepStack与3-axis位置，恢复原KV/rope。

## 成本与运行边界

W次caption、ceil(W/8)次三槽生成、最多4W次text embedding prefill，再新V/S独立读取。
7359窗的生成token上界为96W+768Σceil(W_video/8)，不把批边界合并跨视频。
原始帧/ASR可复用，来源描述、槽、向量在视频内复用；新视频仍支付所有来源获取成本。
按原候选粗预算每100窗10–40 GPUmin，完整7359窗约12.3–49.1 GPUhours加普通reader，
未测量、非实际时间承诺；固定5先测，报告各语料实际decode/caption/slots/embedding/reader秒数、
语言/视觉forward数及显存。完整原来源成本在缓存续跑时照样计入，不只报CPU选择时间。
额外配对native、clone及审计开销单列；Slurm墙钟另报。一个实验全部333在同一机器获取并读取。

## 验证、门与可证伪控制

固定5只验证实际源条件/两侧绑定、新V/S进入分数、native所有原始读数、G/stance/KV/rope及clone；
各语料须真实remote>0、新V>0、新S>0，失败保留UNKNOWN并诊断，不降低guard。
严格当前raw/pixel/token/ASR/embedding pooling/槽批/时间选择重放通过，才调用唯一评测器
`src/eval/evaluate.py`与`evaluate_four_datasets.py`和既有固定r6脚本。
完整215/118视频、within84/99、六项一起报告，以`runs/20260926_twolevel/r6_bma/metrics.json`为正式参照。
性能门为同一主指标两语料均+.01且其余pooled不下降>.005、within不下降>.01。
Rule9：无任一+.01直接归档；有单项信号先记录真实test error analysis，再至多三次设计修订。
全部development-selected，无未揭盲确认声明。

只有完整主门通过才跑全333控制：同源数量但取消时间侧约束；一个current条件替代前后三槽；
来源相同但删去槽角色标记；前后媒体及ASR交换真实来源/角色；同时间侧、相同帧数及ASR长度匹配错误来源；
当前LOCAL帧增强但无remote。顺序caption作为来源实现细节，只有删除它也通过双语料至少一主指标下降.01
才可升级为novelty主张。任何拟主张部件都必须过同样消融门，否则删除或降级。
原始V/S/max排序与源绑定错误控制必须支持定位机制，不能仅以最终r6涨点归因。

## 进度

方案已放行；R1固定spec/原型实现中。尚无本候选实际GPU、GT或性能数字。


完整prototype已实现：真实media顺序caption、按八窗三槽、精确user-token hidden pooling、可分离时间侧源分配、纯真实V/S证据与唯一评测器/固定r6命令。全333 raw/JPEG/ASR/native三轴输入PASS，8组真实36层FP32/BF16×18/20×2/6新图cached/fullreference/KV/clone PASS，embedding pooling2组、已知向量独立可行解枚举/NONE/tie/native-token JSON/cap拒绝PASS，20组生产reader前置/后续/双侧/无remote/缺帧科学CPU PASS。均randomweights/软件输入检查，不是预训练模型性能。来源在manifest列出的runs。唯一独立Rule6审查进行中；无实际GPU/GT/性能数字。


唯一独立Rule6审查PASS，来源`docs/reviews/20261006_m1_ordered_slots_code.md`及`runs/20261006_m1_ordered_slots/independent_code_review/`。独立合成10窗video实际PTS→source acquisition→只读验证、两批槽/32embedding/433source forwards计费、NONE/稳定平局/严格前后源归属通过；独立20组真实36层production reader/不同remote speech实际token与margin变化/KV/rope/native所有原始读数/无生成假设进入证据通过，真实tokenizer正文pooling独立完整前向精确。same-family provisional，不是预训练8B/GPU/性能结论。


实际8B固定5已在sc474398提交Slurm205，等待现有168/186释放实验室active QOS2预算。派发前四机da6668b一致/clean，既有foreign STRAY逐行未变；lab3实际固定5 raw header解析PASS且完整333原始视频已核验。来源`runs/20261006_m1_ordered_slots/machines_before_smoke{,_note}.txt`与`lab3_smoke_raw_coverage.json`（已回传本机）。source/paired reader固定5同机完整运行，无依赖链。尚未实际GPU/GT/性能。


2026-10-06实际Slurm205固定5完成，source及runs均回传本机后才核对。Actualfixed5 BOTH returned; source/native allraw replay reached original remote-source guard FAIL; all caption embeddings unavailable, no remote; narrow diagnosis/interface redesign needed, no GT/performance verdict。来源`runs/20261006_m1_ordered_slots/r1_full_smoke_analysis/`，原失败/UNKNOWN/cap未改。此轮未GT/指标，不能裁定idea优劣或算性能修订。


## R1来源接口B：事前声明，不是性能修订

A完整固定5严格重放到原remote执行门失败：158 captions中153达到16word cap、4达到32token cap、1明确UNKNOWN，0可用caption索引；有效slot127但没有真实可检索caption源。独立窄诊断无实现偏离，不能宣称combined closing token是logits首选，原结果/失败日志/UNKNOWN不变。来源`docs/reviews/20261006_m1_ordered_slots_gpu_interface_diagnosis.md`。

B只把原模型可见生成指令明确限制为caption最多12词、每个条件最多6词，要求立即结束JSON字段；原16/8word caps、32/16token caps、96/768whole caps、cap-to-UNKNOWN、所有source/当前input/remote执行守门全部不变。不salvage旧A，不换固定5，不强迫可用字段，不扩大模型调用/输入或改变有序选择/embedding/最终V/S/G/r6。明确字数指令是执行接口修复，不作科研贡献，也不重置方法预算；没有读GT/性能。

B `spec_B.json`，通过`SOURCE_INTERFACE=B`选择；默认A仍精确读取原spec/cache。B独立`data/temporal_ordered_slots_B`及`source_smoke_B`/`r1_full_smoke_B`，杜绝替换A缓存。先作者源码/原A重放隔离检查和独立窄确认，再同一固定5 GPU。若仍不能真实执行，继续保留失败，不评价idea/不降低guard。

B独立窄代码确认PASS，来源`docs/reviews/20261006_m1_ordered_slots_interface_B_code.md`：A全5真实source/currenttokens/pixels只读重放保持，B actualsynthetic80s video acquire/validate与真实tokenizer/processor/来源memory归属、A/B×smoke/main×extract/measure/analyze cache/raw/decoded/analysis路径隔离通过。不是新proposal或性能版本，不是B真实8B执行成功；同固定5 B GPU ready。

同原固定5接口B在sc474397提交Slurm207，当前PENDING(QOSMaxGRESPerUser)，等待186/192占用的实际2GPU用户预算；不绕过调度。派发前四实验室50ea174 clean一致、既有foreign STRAY逐行未变，证据`runs/20261006_m1_ordered_slots/machines_before_B_smoke{,_note}.txt`。B来源获取及配对读取均同一ROOT机器，无实际B执行/GT/性能结论。

## Conditional read-only closing diagnostic (prepared, not run)

If B fixed5 still fails its original execution guard, inspect the first caption of each of the same5 videos with `closing_diagnostic.py`/`launch/lab1_closing_diagnostic.sbatch`. The observer records raw top tokens, quote-leading terminal punctuation candidates, bare-quote rank and actually selected token from existing description logits. It adds no forward, changes no choice/cap/grammar/reader, writes separate `runs/.../closing_diagnostic_B` only, and requires original saved B tokens/events/inputs/positions exactly replayed plus source metadata unchanged. No GT or performance. This is an execution diagnosis, not a new interface/method or evidence that a masked closing token caused actual failures. Author actualtokenizer/synthetic-head observer test preserves original tokens/events/selection and forward count (`closing_diagnostic_cpu_checks/summary.json`); initial import-order issue corrected before GPU, per-generation diagnostic cache references cleared. Unique independent narrow confirmation and actual B guard failure precede any diagnostic GPU submission. Only5 caption calls, rough unmeasured <5GPUmin including model load; actual diagnostic time/calls recorded separately from all original scientific source costs.

Conditional closing observer独立窄确认PASS：`docs/reviews/20261006_m1_ordered_slots_closing_diagnostic_code.md`，真实tokenizer/合成head的bare/compound/cap/softcap记录不干预、九类原重放差异拒绝、首fixed5绑定/源只读/缓存释放/模块恢复及资源启动通过。只准备，不是8B原因证据；原B guard尚待真实GPU，条件未满足，diagnostic未提交。
