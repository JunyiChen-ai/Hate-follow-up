# 2026-10-06 M1 九候选一次独立 Rule 4 裁定

裁定：**9 PASS、0 STOP**。下一可执行候选推荐 **ReKV 层级来源 KV 检索**，其次为词汇衔接分区的 speech 读取。PASS 仅表示这次实际检索和完整机制比对未证实 Rule 4 的四种 STOP 情形；不是已证明 novel、有效、可靠实现、晋级或可以发表。排序是实验优先级，不是性能预测。

审稿实例：fresh `gpt-6-astra`，与主作者及三个候选生成实例不同；路由为 **same-family provisional**，不是 cross-family 或 accepted assurance。一次审稿覆盖 `experiments/20261006_m1_ideation/CANDIDATES.json` 原样九项；没有质量筛除、改写候选或重开第二轮广泛审查。审稿日期 2026-10-06；没有原始 GT、预测数组或 metrics 文件读取，没有 GPU/作业，没有科学代码或共享 STATUS 修改。STATUS/README 内已有历史数字仅随状态文字读取，没有据此重算结果。后续具体规格澄清是这次审稿的延续。

## 依据与实际阅读范围

已读 `CLAUDE.md`、`RESEARCH_ITERATION_RULES.md` 第 4 条及相邻运行/晋级/消融规则、`research-wiki/STATUS.md` 当前状态，九项完整候选。家庭边界实际查阅以下 README 的机制/来源/算法部分：

- `archive/experiments/20261003_m1_integrator/README.md`。
- `archive/experiments/20261004_m1_{acoustic,lattice,tree}/README.md`。
- `archive/experiments/20261005_m1_{program,quote_graph,ttf,ott,interval_witness}/README.md`。
- `archive/experiments/20261006_m1_videoevent/README.md`。
- `experiments/20261005_m1_{provenance,text_tracking}/README.md`。
- `experiments/20261006_m1_{merit,ordered_slots}/README.md`。

花括号只是列举路径的简写。Program 已耗尽、Interval26 已归档第30项；Provenance25、MERIT33、OrderedSlots34 的原预算不变，32/35 的接口失败不作为 idea 失败。本报告不授权任何这些家族改名重启。

独立执行的原始搜索响应和原文阅读响应保存在 `runs/20261006_m1_ideation/jury_evidence/`。`search0/1/2.txt` 是初次逐候选搜索；`target_followup.txt`、`target_exact_followup.txt` 为实际目标任务跟进；`source0..8.txt`、`method0/1/2/4/5.txt` 为来源论文；`target_primary.txt`、`target_methods.txt`、`additional_primary.txt` 为相邻目标方法及 Whisper 代码。TFVTG 浏览器超时，实际 HTTP 下载成功，`tfvtg.pdf` / `tfvtg.txt` 已读 §§3.1–3.3、4.1。已有 `source_reading/primary_read_3_access_gaps.txt` 中 CVF403 是原收集者的访问失败；本次以 arXiv 原文独立核读，不把失败当未有先例。

本次实际来源阅读：ReKV §3/4.2；MuKV §3.1–3.5、公式1–9；StreamingTOM §3.3–3.4；BaGLM §3.2/4.1–4.3；VideoGEM §3的 self-self attention、层权重及3.4分解；TextTiling 作者 HTML 的 tokenization/similarity/boundary algorithm；Whisper-Streaming §2–3；Park–Glass §2–3；Whisper `timing.py` 的 DTW、`find_alignment` 及词时间后续启发式。不是全部论文/代码审计。MuKV/StreamingTOM 官方实现未独立细读，MuKV FFT 的轴和频率索引到 token 的对应仍需落实；TextTiling 原阈值公式为图片，未验证该图公式；Park–Glass 所引高效最低平均子序列算法未另读。

目标任务已有普通检索/证据结构，不能据本报告声称首次：实际读 [MATCH](https://jianlang.org/papers/MATCH.pdf) 的引言机制说明（双立场线索、检索局部证据、验证、融合；没有在此宣称逐行读完其全算法），以及 [RAMF](https://arxiv.org/html/2512.02743v1) §§3.1–3.3。搜索还命中 MoRE、LEAF、IARE 等；这些搜索命中不能替代全方法阅读。已有 MAESTRO/WWW2026 全文访问缺口仍保留。本次检索没有找到九项完整来源机制已用于 hateful video detection/localization 的可验证主文证据。有限检索中的未发现不等于证明不存在。

对声学检索，专门核读 [Jacobs 等，Interspeech2023](https://www.isca-archive.org/interspeech_2023/jacobs23_interspeech.pdf) 引言及 §§2–3：这是 radio 的 ASR/AWE keyword spotting，DTW在相关工作中出现；它不足以证明本候选的同视频无词表局部声学重复检索、双源语境区分链已经用于 hateful **video**。禁止据此泛化为“从未用声学方法检测 hate”。

## 共通约束判定

九项都规定单一冻结 Qwen3-VL-8B、原 native20/fullASR/native G/自己的 hard stance、每个原8秒窗独立新 V/S、max、固定 r6、唯一4fps评测器、两主语料同流程且零目标标签。三项 speech 的 Whisper 是现有唯一 ASR 来源抽取器；重新调用会增加成本，不是第二 hateful predictor。九项均有选择/构建来源之后的新鲜最终读取，未把检索相似度、状态 posterior、进度、DTW代价或其他窗的 hate margin 当最终分数。因此这不是纯 ensemble 或最终 hate-score smoothing/calibration。

这些是**规格级符合**，不等于实现已经验证。共同实施义务：冻结真实源时标、缺源/UNKNOWN/截断/尾窗规则、完整prompt及上限；逐窗恢复 KV/rope；最终 margin 不复用采集阶段判断；G/stance和不改动分支原生复现；保留原执行guard，失败修实现不裁定idea。来源选择可以看跨窗解释上下文，但必须保留本窗LOCAL及remote实际所属时间，不能将remote行为当成本窗事实。混合媒体的源解释记录必须说明进入V/S的字段，不能让一个已评分分支影响另一个。

所有新视频都支付 native输入获取和原native计算、新增解码/视觉/ASR、source生成/prefill/传输/检索及新reader。下文 GPU 时间全部来自原候选的**未测预算**，不是测量，也不是速度承诺。替代部署native局部读取与实验中额外配对native/clone/消融须分别计数，避免双计或漏计。完整333之前先真实固定5测调用数、token、GPU/CPU秒、内存峰值；只需按规则做一次独立Rule6实现审查。

## 逐项裁定（保留原顺序与 exact dedup_key）

### 1. `rekv_layerwise_source_memory_local_conditioned_reader` — PASS，实验顺位1

实际搜索：`"ReKV" ("hateful" OR "hate") video`；补查 `"hateful video" "retrieval" evidence`。实际原文 [ReKV](https://arxiv.org/html/2503.00540v1) 支持滑窗编码、缓存存储、逐层内部Q/K检索和KV回答；原默认检索64帧，本候选四块与Qwen三轴适配不能称原配置复现。

完整科学干预是局部问题从额外实际媒体记忆选择不同上下文再产生一个V，不是单纯offload提速。与Integrator的native前缀可见边修改、Tree的caption树、MERIT的生成四键/邻域过滤功能不同。**不继承这些旧家族预算**；将来若移除来源检索而只改native前缀mask，则返回相应旧家族，不能沿本PASS开新预算。

实施前补齐：GQA查询头与8个KV头的映射；检索究竟使用旋转前还是旋转后Q/K；问题token在各层何时已条件化于LOCAL、怎样避免检索依赖自身尚未取得的KV；源块祖先信息记录；三轴位置平移及不同层不同选中集合的真实时间标记。缓存重定位不等于重新编码，数值检查应验证声明的算子而非假定二者等价。无remote、无local、短视频的fresh fallback须确定。

每视频新增约F=ceil(T/2)视觉/LM源prefill；无caption生成和额外probe（依赖实现确实能层内检索）。原预算每分钟20–100 GPU秒，另每窗0.2–1秒。优势是source绑定可直接通过真实KV块交换检验、没有语言源生成可用率瓶颈；主要风险为缓存/位置实现。先做同新增LOCAL帧的local-only和等数chronological control，不能把更密本窗输入收益归给检索。

### 2. `mukv_three_grain_coherence_retrieval_local_reader` — PASS，实验顺位4

实际搜索：`"MuKV" ("hateful" OR "hate") video`，同上目标retrieval跟进。实际原文 [MuKV](https://arxiv.org/html/2605.22269v1) 确实包含三次粒度prefill、attention/FFT压缩、并行候选再跨粒度rerank及KV读取。不同粒度是同模型同来源表征，不是多个独立预测器ensemble。

它与Tree的语义caption层级、TTF/OTT的原native token合并功能不同；也不是ReKV只换top-k常数，三种独立上下文化记忆与一致性rerank构成新增完整机制。**独立机制边界成立，不重启旧预算**；若最终只实现frame-only查询检索，则属于ReKV方案的简化，不再单开MuKV修订额度。

缺失规格：FFT精确轴、复数幅度/均值顺序、零范围min-max；原频率bin索引如何保留同索引token必须忠实说明，不叫已证实时间显著性。各粒度历史attention范围、patch块如何串行/批量prefill、查询pool行、Q/K维度、LOCAL与重复segment/frame/patch来源的role/位置需冻结。4帧/3prefill只是逻辑近似，若四quadrant分别调用要如实增加。

原预算约3W记忆prefill+W中立query，4W源图，3–12 GPU秒/窗；尚未计准三历史流随长度的增长。三粒度架构容易引入大存储与prefill，不保证比ReKV便宜。等active-token frame-only、去rerank、同LOCAL-only及错误coarse-to-fine绑定必须区分密度、粒度和一致性贡献。

### 3. `streamingtom_causal_dualpath_quantized_media_memory` — PASS，实验顺位3

实际搜索：`"StreamingTOM" ("hateful" OR "hate") video`，同上目标retrieval跟进。实际原文 [StreamingTOM](https://arxiv.org/html/2510.18269v1) 支持相邻帧静动划分、静态合并/动态显著性选择及分组量化检索。候选完整链另有强制LOCAL及独立hate读取；不是单独四位量化的工程技巧。

与TTF的anchor替换、OTT的相邻最优传输不同；相同目标“保留变化”不能当成同一算法。与ReKV同属宽泛媒体记忆方向，但固定双路保留机制是可独立删除的科学差异。**原样可另立机制，不继承TTF/OTT预算**。若最终仅量化ReKV而取消双路机制，则不得以新科学family重启。

需要冻结首帧/网格变化规则、静/动集合空集与50-token不足、密度峰值聚类参数/合并权重、真实Qwen vision attention显著性、DeepStack相同成员映射、scale=0量化、group代表层/头与mRoPE。源prefill的历史范围未给出，不能暗改为任意短上下文来兑现线性预算。causal仅指源构建；native整段前缀与跨整视频检索使总体仍离线。

F个视觉/压缩prefill，若需W个probe要明计；原预算每分钟15–80 GPU秒另每窗0.2–1秒。量化不消除完整vision成本。先同budget静动vs均匀与去remote；四位vs全精度是数值/效率控制，不能默认有定位贡献。

### 4. `dynamic_static_transition_source_grounding` — PASS，实验顺位9

实际搜索：`"TFVTG" ("hateful" OR "hate")`、`"Training-free Video Temporal Grounding" "hate"`、`"hateful video" ("transition" OR "prerequisite" OR "tracking")`。实际读 [TFVTG ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10687.pdf) 的动态/静态proposal、关系约束及原GPT-4/BLIP-2配置。候选Qwen支持概率、差分0、不平滑、有限邻域是已声明适配。

动态/静态分数用于选真实源，随后fresh V/S，故非纯hate后处理。OrderedSlots只取前/当前/后语义最优源，VideoEvent按相关性成组，这里枚举变化段和后状态联合proposal；**完整机制不同，不继承34/36预算**。

须明确单点转变段怎样定义差分（防止空差分条件自动成立）、边界共享点、静态外部点集合、重复proposal/不足top3、混合speech/visual事件的各自网格以及未知关系/无合法组合。原顺序条件允许重叠，不应在论文称严格“前事件结束早于后事件开始”。当前LOCAL语音保留和remote选ASR的坐标要完整落地。

成本是最低优先级主因：W生成、最高36W视觉支持+18W语音支持，再最终最多2W reader；每100窗20–90 GPU分钟且reader另计，7359窗仅该预算约24.5–110.4 GPU小时。不是STOP。只有来源和动态项的等数/错绑定控制成功，才谈机制；稀疏观测不足交给实际数据。

### 5. `prerequisite_progress_source_memory` — PASS，实验顺位8

实际搜索：`"BaGLM" ("hateful" OR "hate")`，同上transition/prerequisite目标检索。实际读 [BaGLM](https://arxiv.org/html/2510.16989v1) 的依赖矩阵、进度、readiness/validity和Bayesian更新。原方法输出step后验；本适配以该状态选原来源再独立读取，**不是把第二时序hate后处理叠到r6**。

不同于Program的生成工具程序、OrderedSlots的独立槽检索、Interval的跨层证书。**当前完整依赖/进度选择机制独立**；不能退化成旧program或旧图遍历后另开预算。

缺失规格：依赖边的方向与矩阵indices；“完成程度”是前置均值、最小值还是原文加权均值；零前置/后继、循环图、初始空历史、NONE的更新；选择“使完成程度最大”的具体前置源。原文进度公式和1-progress存在尺度需要说明，本候选除9是适配。选项token合法性、归一化范围与zero likelihood必须先冻结。

最多8W+ceil(W/8)调用，包含最多2W最终reads；候选同时写“10–45 GPU分钟/100窗加reader”，应消除reader计费歧义并实测。最高约6W+ceil(W/8)为采集阶段。反向依赖/错绑定、去进度、去依赖需分别验证；普通视频可能缺程序结构是风险，不是拒绝条件。

### 6. `actor_object_transition_witness_paths` — PASS，实验顺位5

实际搜索：`"hateful video" "object" "tracking"`、`"VideoGEM" ("hateful" OR "hate")`及transition跟进。实际 [VideoGEM](https://arxiv.org/html/2503.20348v1) 为动作/动词/对象空间grounding与层加权，没有实现候选的双实例连续性和关系变化路径。候选明确是**自设计**，不冒充完整VideoGEM迁移，因此来源只提供局部分解线索不构成工程STOP。

与Provenance25最近：25由文本ledger建立same_entity/话语边，检索两跳附近端点；本候选在真实相邻双帧判断主体与对象实例连续性，再对关系序列做变化点源选择。双视觉身份约束加关系转变是完整不同的可删机制；与32的像素文字可见性跟踪也不同。**原样可独立，不授予25/32重启**；若省去双帧身份和关系序列，只剩实体图端点检索，就应继承25。

必须解决具体规格含糊：method写“根据初始两帧观察”生成最多三状态，同时要求每状态绑定已有帧；是否仅允许两帧见过的状态必须固定，不得运行后扩为看全路径。另需same概率平局、box坐标/空对象、log(0)、无变化与变化平局、四窗块边界和当前窗落在变化一侧的source选择。生成框不是正确实例的证明。

最多6W−1调用含最终V；每100窗10–40 GPU分钟未测。中立结构化生成的实际可用率可能是主要限制。对象错配/身份删除与同路径静态最强source控制可直接检验源绑定及变化贡献；仅展示跟踪例子不够。

### 7. `lexical_discourse_partition_scoped_speech_measurement` — PASS，实验顺位2

实际搜索：`"TextTiling" "hate" video`、`"TextTiling" "hateful video"`。实际读 [Hearst作者TextTiling原文](https://people.ischool.berkeley.edu/~hearst/papers/tiling-acl94/acl94.html) 算法部分及 [Whisper timing.py](https://raw.githubusercontent.com/openai/whisper/main/whisper/timing.py)。原论文词形/停用词/paragraph修正与本候选Unicode+无停用词+均值阈值不同，须如实称适配。

不是只修时间戳或换prompt：确定词汇边界限制解释上下文、逐字LOCAL act/CONTEXT编译后新speech读取构成完整方法。QuoteGraph24的语义span关系/图遍历/非对称attention和Acoustic20的时间分布attention都不在这里；**当前机制独立**，不能把旧quote graph改名搬回。

补齐末尾不足20词、边缘不足6伪句、零词频范数、平滑padding、同深度冲突边界、词中点恰在8秒端点、全视频短于12伪句规则，以及source parser的完整词数/token/截断上限。30秒ASR块接缝不可通过复制词制造两个事件；DTW是估计时间，保存未经后续heuristic改写与实际使用时间的区别。UNKNOWN/无act仍须明确fresh读取与缺源行为。

120秒预算约B=4次ASR生成及对齐、Ns中立解析+替代speech，新增60–240 GPU秒，另native原ASR/native成本。文本分区CPU便宜且控制干净，故列第二；长话语context成本可能上升。必须保留完全相同新转录/时间的direct-local对照，matched邻近context和错绑定，使时间修复不冒充分区机制。

### 8. `localagreement_acquisition_horizon_source_dependency_reader` — PASS，实验顺位7

实际搜索：`"LocalAgreement" "hate" video`、`"LocalAgreement" "hateful"`、`"Whisper-Streaming" "hateful video"`。实际读 [Whisper-Streaming](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) §§2–3及上述Whisper对齐代码。来源稳定确认不等于语义因果依赖，本候选正确把确认时间与发生时间分开；4秒媒体步进、large-v3、无VAD是适配。

不是多转录hate投票、不是Lattice22竞争词DAG，也不是Acoustic20的单转录时间支持attention；完整“增量获取历史→horizon→局部解释编译”可独立。**不继承已归档speech家族预算**。若只保留更稳定ASR文本而取消horizon条件读取，就只剩输入修复，不能主张本机制。

补齐最长公共词前缀的大小写/标点规则、跨buffer已确认词去重、DTW时间修订、同词重复出现ID、尾部UNCONFIRMED如何进入LOCAL、horizon究竟是哪两次实际音频覆盖的差集。超过30秒但没有确认segment时原候选缺明确规则，须在实现前声明，不能静默丢词。无需把horizon称causal解释。

120秒约30次Whisper生成/对齐而不是只4个块；新增105–420 GPU秒加native。部署仍要native fullASR；全部重复音频秒计费。与相同已确认词/同词时间的无history及等词数静态context比较，再错绑horizon；稳定性好本身不支持hate定位贡献。

### 9. `segmental_acoustic_recurrence_local_context_disambiguation` — PASS，实验顺位6

实际搜索：`"segmental DTW" "hate" video`、`"segmental dynamic time warping" "hateful video"`，并追读Jacobs2023目标相邻音频文章。实际 [Park–Glass](https://people.csail.mit.edu/jrg/2005/Park_ASRU.pdf) §§2–3为局部声学路径检索；原phone分段/PCA/讲座设置与候选8秒窗/视频内归一不同。完整声明是segmental检索加真实两端媒体语境区分，不是原聚类pipeline完整复现。

与Acoustic20的文字–声学对齐不同；与Provenance25的生成语义关系检索不同，检索边来自局部声学匹配，不赋身份；不是凭重复声段传播hate score。**原样独立机制**。已有MFCC或一般音频检索不构成这个完整方法已用于目标的证据。

缺失规格较多：MFCC采样率、分析窗、filterbank/预加重/c0、方差0处理、band起点/步长/DP边界/路径平局、100步与实际持续时间区别、最低平均subpath的精确算法。5ms下8秒窗约1600步；每对距离矩阵约256万cell，不能只报O(W²)而漏掉窗内二次工作。逐pair处理可控制内存；精确子序列枚举的额外时间先实测。

至多2W端点ASR、W对照生成、最终V/S；120秒原预算90–390 GPU秒，DTW CPU成本另计。中点单帧可能不含目标行为，保持UNKNOWN。最终V/S接收共同双源record的哪些字段必须冻结，独立cache不自动证明模态字段隔离。保留当前端新图/转录的local-only、whole-window-DTW和邻近窗控制；正确donor媒体绑定须优于错配，而不能只是额外语音重识别改善。

## KV 三项资源落实义务

主agent补充的当前机器快照为root约60GiB RAM、常用Slurm32GiB内存、磁盘442GiB空闲，lab-server707GiB空闲；这是审稿输入，未独立重新探测。它不能代替提交前实时检查。按声明Qwen36层、8 KV heads、128 head dim、K/V BF16，**每个完整语言token的全层KV为147456 bytes=144KiB**。这是解析配置的算术估算，不含张量复制、allocator、prefix/文本、DeepStack/vision、模型权重或attention工作区。

令p为每新增帧实际进入LM的视觉token数、F=ceil(T/2)、L为native prefix token数。以下为容量公式而非已测峰值：

| 候选 | 单视频源KV保存量下界/近似 | 活跃GPU附加KV与限制 |
| --- | --- | --- |
| ReKV | `144KiB × Fp`，另源时间文本/位置/代表keys | 每层最多本窗约4帧+4 remote，加native `144KiB×(L+8p)`；源prefill另需本帧和4历史帧；重叠实现若clone需额外峰值 |
| MuKV | W块、四帧segment+一middle frame+全四quadrants合约p：保留后约 `144KiB×3.4Wp`，另文本/代表keys | 单次四LOCAL约4p加2 segment约6.4p+2 frame约.2p+2 quadrant约.05p，共约10.65p+L；建流阶段可能更大 |
| StreamingTOM | payload约 `36KiB×50F`，另每组scale/offset及代表keys；若这三者各层FP32，约720KiB/F组（K/V各两组scale/offset，加一组key），总约2.46MiB/F组 | 4 uncompressed LOCAL约4p +4 remote×50，再native L；源prefill的完整历史策略尚待声明 |

举例120秒、F60、W15、p256：ReKV源KV约2.11GiB；MuKV约1.79GiB；StreamingTOM约148MiB（含上述简化scale/key估算）。不含各项公共和临时开销。p512时前两者翻倍；临近视频结束的短窗取实际帧数。60分钟视频同p下ReKV源KV约63.3GiB，不能假定32GiB作业内存足够。四个remote的固定活跃预算也不保证完整source prefill或全历史检索CPU时间有界。

若仅作full333磁盘量级示例，按历史7359窗口每窗约四源帧、p256，ReKV密集持久化约1.01TiB、MuKV约0.85TiB；这是条件推算，不是当前媒体精确盘点，已大于442/707GiB单机余量。StreamingTOM按相同比例约71GiB且仍要另加输入和trace。**不要求跨run永久保存全333的dense KV**；原候选所需的是每视频真实来源、算法轨迹、选择index、source ancestry与运行中的来源记忆，可逐视频构建/读取、CPU或磁盘offload、完成该视频后释放可重建KV。若审计/消融重建，新增GPU秒完整计入；若选择持久化，必须给出峰值和完整磁盘预算。不要删除原始结果或已有缓存腾空间，也不能为了省内存悄悄改变source attention范围/帧率/每层检索。MuKV及StreamingTOM源prefill历史规则必须先补充，否则其GPU峰值上界未确定。

## 排序与下一步

| 顺位 | 原候选 | 主要排序依据 |
| --- | --- | --- |
| 1 | ReKV | 无中立文本生成，来源块绑定清晰，local-only/错块控制直接；先解决Qwen缓存位置与每视频内存 |
| 2 | TextTiling作用域 | CPU分区轻，单次speech编译；同转录/时间消融易隔离输入改善 |
| 3 | StreamingTOM | 无caption、量化存储较小，但双路/DeepStack/位置需更多实现验证 |
| 4 | MuKV | 跨粒度错绑定可测，三prefill和历史/FFT规格提高实现与内存成本 |
| 5 | actor-object路径 | 身份与状态控制明确，额外结构化生成较多 |
| 6 | segmental recurrence | 源对比较清楚，CPU二次匹配及端点ASR成本仍未知 |
| 7 | LocalAgreement horizon | 单speech读数链清楚，重复音频重解码成本较高 |
| 8 | BaGLM来源记忆 | 图/进度可删，但多次每窗进度读取且非程序视频适用性未证实 |
| 9 | TFVTG来源选择 | 完整机制可执行，上界54W支持读取使采集成本最高 |

这是对原样池的一次排序，没有删除低顺位候选，也没有让排名覆盖既有正在运行的方法。推荐把ReKV具体化为可审核规格与prototype，先同native科学CPU和真实固定5执行/成本核对；如出现容量或接口问题，只据具体观测修实现，不以想象退化STOP，也不降低现有guard。

所有PASS完成门相同：完整HateMM215/HateClipSeg118调用唯一4fps评测器；**同一主指标两语料均提升≥.01**，其它pooled损失≤.005、within损失≤.01；六项并列，within正负混合视频84/99。每项拟主张novel部件删除后，至少同一主指标在两语料均下降≥.01；不足就删掉或降为实现细节。还须raw V/S/max视频内排序、等获取量与wrong-binding证据，区分来源覆盖、检索、角色绑定和固定r6产生的变化。全部未来选择后结果标development-selected。通过proposal绝不是目标已经完成。
