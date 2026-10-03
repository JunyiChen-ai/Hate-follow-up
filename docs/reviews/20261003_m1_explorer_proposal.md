# Explorer 独立 proposal review

日期：2026-10-03。结论：**PASS**。本轮实际查新未找到 `RESEARCH_ITERATION_RULES.md` 第 4 条四类 STOP 的成立证据。可以进入实现和独立代码检查；此结论不代表效果、效率或机制解释已经成立。

审查实例：`/root/m1_grounder_proposal_review`，沿用与主会话相同模型。审查 candidate18 的 `experiments/20261003_m1_explorer/README.md`，并复核审查过程中补入的旧帧时间戳、候选 PTS 和错窗配对说明。只读取规则、提案、输入接口/旧抽帧代码及 primary 文献；未读取 Preserver 性能或任何 GT，未实施方法、启动实验或计算内容哈希。本文是本候选的一次 proposal review，不重开以前候选的审查。

## 四项 STOP 判定

| STOP 条件 | 本轮判定与理由 |
| --- | --- |
| 来源已用于 hateful video detection / localization | 未找到 EcoFrame 或同一熵反馈预算与 attention × distance 新帧获取机制在目标任务使用的 primary 证据。存在局部证据检索近邻 MATCH，须缩小贡献表述，详见下文。 |
| 纯 ensemble | 否。全部读取来自同一个冻结 MLLM，最后一次局部 margin 是输出；没有多个独立模型的预测组合。 |
| 纯 calibration / 后处理 / 平滑 | 否。熵决定是否获取实际像素并再次运行模型，注意力决定获取位置；不是将旧分数换一个标量映射。 |
| 纯工程技巧 | 否。候选定义了触发、位置选择、反馈刷新、终止和读取的完整闭环。仅增加帧数或缓存复用本身没有 novelty，提案已要求与均匀补帧比较。 |

此前候选失败、二元熵可能不能识别错误、注意力可能没有语义作用等，都不是规则允许的实现前 STOP 理由。

## 来源核对与迁移范围

实际读取 [EcoFrame 原文](https://arxiv.org/html/2608.03918v1) 的 §2、§4、§5.1、附录 A.3/B/C。其任务是长视频问答，主实验为 Video-MME、LongVideoBench、MLVU；[作者仓库](https://github.com/AK-DREAM/EcoFrame) 本轮可见内容只有 README，并称代码待发布，故不能称作者代码复现。

来源 Eq.1 使用回答序列的全词表熵；Eq.2 对最终问题位置的 pre-RoPE Q/K 在视觉 keys 内 softmax，平均 heads/layers，再求帧内均值。§4 使用逐轮放宽的熵门、几何预算、最近帧时间单元上的注意力先验与距离乘积扩展候选池，再经独立 CLIP 相关性和覆盖度重选输入。附录 B 默认参考层为 19–21。§5.1 的 Video-MME 设置包括初始熵阈值 .1、增量 .2、注意力指数 .5；另外两个数据集使用不同熵设置。

Explorer 的二类熵、原生支持强制触发、全 36 层、目标窗内每轮两帧、最多四帧、没有 CLIP 重选、在 verdict-conditioned KV 后追加图片，均属于当前适配。尤其应明确 all36 不继承来源的深层选择，.1/.3 在二类分布上也不继承原文的数值有效性。当前声明不做扫描、两语料同一规则，可以实验检验。

这里的 pre-RoPE attention 是提取的选帧信号，不是模型实际使用的完整 post-RoPE attention，也不能直接称为模型对仇恨证据的概率。帧均值再统一归一化不改变单轮候选排序，因为只引入共同正比例因子；全层平均则确实改变信号，不能略去。

## 实际目标任务检索

本轮查询包括 `EcoFrame`、`2608.03918` 与 hate / hateful / HateMM / HateClipSeg 的组合，以及 hateful video 与 adaptive frame、active sampling、uncertainty、entropy、frame selection、keyframe、iterative、resample、acquisition 的组合。没有将搜索摘要或综述当作已采用机制的证据。下列 primary 方法/输入段落用于排除仅凭题名查新：

| primary 来源 | 与本候选有关的实际内容及界限 |
| --- | --- |
| [MATCH，作者原文](https://jianlang.org/papers/MATCH.pdf)；[作者条目](https://jianlang.org/papers/MATCH.html) | §III-C 将丰富的均匀帧及邻近 OCR/ASR 构成时空单元，按线索与图像/文本的 CLIP 相似度检索给 verifier；§IV-A 声明初始 16 帧、64 个证据单元。已经在 HateMM/MultiHateClip 进行局部证据检索和验证。未见熵决定新帧预算或 attention × distance 的反馈获取。 |
| [MoRE，作者原文](https://jianlang.org/papers/MoRE.pdf)；[官方代码仓库](https://github.com/Jian-Lang/MoRE) | 检索其他视频实例的上下文，结合模态专家。不是从当前视频按窗口读数补取像素。 |
| [MARS](https://arxiv.org/html/2601.15115v1) | 预先均匀选帧，再做多阶段推理；帧数敏感性不等于反馈式帧获取。 |
| [CLARA](https://arxiv.org/html/2608.15905v1) | clip 表示、局部/整体对比、rationale 融合；rationale 输入为 20 个均匀帧，后续验证复用输入。 |
| [SAGE](https://aclanthology.org/2026.acl-long.817.pdf)、[LEAF](https://aclanthology.org/2026.findings-acl.604.pdf) | 分别为专家语义交流/裁决与 self-grounding CoT/蒸馏；未见本次来源获取规则。 |
| [LELA](https://arxiv.org/html/2602.09637v1)、[MultiHateLoc](https://arxiv.org/html/2512.10408v1)、[HateClipSeg](https://arxiv.org/html/2508.01712v2) | 检查定位方法和输入描述，未发现 EcoFrame 或同一反馈获取机制。定位、切窗或选择展示关键帧本身不构成来源采用。 |

MATCH 是必须讨论的实质近邻。不能声称第一次为仇恨视频补充局部证据、第一次让推理指向视频证据，或将原始视频证据检索整个概念当作新贡献。当前可检验的窄命题是：在既有整体上下文和局部读取上，支持/不确定性控制的有界获取以及内部 attention × distance 选择，是否以明确成本改善定位。未发现目标任务采用是本轮检索结论，不是穷尽文献的证明。

## 已修正的确定性接口问题

实际检查 `src/video_inputs.py::frame_paths` 和 `experiments/20260910_spvl/prep_frames.py`：旧 JPEG 文件名存两位小数请求时间，ffmpeg 接受三位小数 seek，失败可提前 .5 秒重试而不改文件名，没有保存实际源帧 PTS/index。故旧支持统计只能表示原生标注时间上的覆盖。

当前 README 已保留原生 JPEG/时间 token，并将原生支持定义为 nominal timestamp；从原 k/index 与 duration 重建初次和 fallback seek，保守排除可能的旧帧。新候选使用 `(i+.5)/4`，映射到首个不早于目标的实际 PTS，拒绝落窗外/覆盖外的映射，并对新增集合按源帧 index 去重。这些修正合理，但保守排除不等于已证明每个旧 JPEG 的真实身份。因此最终只能报告“新增集合严格去重，并保守避开旧缓存可能对应帧”，不能声称完全证明与旧缓存零重复。

错窗对照原来在所有窗口直接循环移位，会与 0/2/4 等不同获取数冲突。现按每视频实际最终帧数 c>0 分组，组内循环移位、保留接收窗口时间 slots 与 main 轮数，singleton/零帧标 unmatched，已消除该数量矛盾。还须保持每轮实际数，包括候选不足两帧的例外，不能只凭最终数推断每轮数据。完整语料比较与有效错配子集须同时报告。

以下是评分前需落实的窄定义，不是另设 STOP：

- 获取公式中的候选 t 应明确是映射后的实际 PTS，而非尚未解码的目标中心点；原帧位置继续使用原生 nominal time。nearest-cell 的等距和相同时间标注用确定性规则处理。
- `uniform` 对照需唯一、可执行的位置选择规则，并沿用相同候选合法性、去重和不足候选规则。不得通过读取主结果选择均匀策略。
- baseline 原生精确恢复不能因追加图像而改变 global/answer/speech 或跨窗状态。需要实际检查追加多图的 `image_grid_thw`、image-token 对齐、完整 attention mask、绝对/mRoPE 位置和 rotary 状态，不能只靠字符串拼接正确。

## 成本核对

令 V 为视觉窗口数、B 为视觉加可用语音问题数、R_w 为实际获取读取次数。`3+B+sum(R_w)` 与声明一致；单窗最多新增两次语言 forward。第二轮从原生 cache 重新追加完整四帧，因此上界是每窗 `2+4=6` 次额外图像编码，不是四次。原生 visual pass 可共享给 paired baseline，但提取 attention 的时间和存储归 Explorer。

除新图外，保存全部层 pre-RoPE 图像 K、计算各层/头 QK 和 softmax 也有额外成本。可以保留未重复的 8 KV heads 并按需广播，不能把长期保存 32 头展开张量当作免费。原生 KV、额外图像 K、追加窗口激活和解码成本均应纳入实际峰值和耗时。当前 30–70 GPU 分钟只是估计，本审查未验证。

相同调用次数和帧数不保证严格等耗时：processor 的视觉 token 数、尺寸、有效候选数及选择器开销可能不同。matched-uniform 至少匹配每窗每轮实际图片数、相同尺寸政策和处理后 token 数，并报告剩余实际耗时差。只有完成测量才可称 compute-matched；此前应称 matched calls/frame budget。

## 消融能证伪什么

1. **零获取**：验证恢复原生读数和最终管线。若不能恢复，先修实现，不能评价机制。
2. **重放 main 轮数的均匀获取**：能证伪“相同窗口预算下 attention × distance 选帧优于均匀选择”。它是条件于主 trace 的诊断，不能部署，也不能单独证明熵或无支持触发分配预算更好。
3. **固定四帧均匀补充**：是必要的简单输入增强基线。若它解释主增益，不能把多输入包装成新机制。不同预算/轮数下的比较主要说明准确率与成本的整体取舍，不能单独归因于熵。
4. **同数错窗帧**：检验新内容与目标窗口时间对应是否必要；同时改变内容、图像顺序和时间对应，不等于只干预时间语义。报告有效错配覆盖，unchanged 样本不能作为肯定证据。

若单独声称 attention 是一个有效贡献，需要同 main 轮数/候选规则的 distance-only（A=1）控制，否则相对 uniform 的优势也可能仅来自最大间隔覆盖。若单独声称熵预算是贡献，需要固定选择器、预声明的预算调度消融或等预算分配控制；当前设计可以先把整个获取闭环作为唯一候选机制，不拆成未经支持的多个贡献。以上只规定结论边界，不要求主结果前预先跑所有控制。

只有完整两语料结果、对应移除门和成本比较成立，才能讨论“为什么涨点”。原生缺帧统计、熵下降、attention 改变或个别案例均不能代替定位效果及因果对照。当前主门、三指标、raw visual/speech/max、支持分组、global 正误分组和完整成本记录已能承载后续判断。

审查结束前再次只读核对最新 README：已补 MATCH 近邻、all36 相对来源 19–21 的适配、actual-PTS 选帧和等距/重合时间处理、确定性 quantile uniform、相同轮数的 A=1 distance-only 控制、图像 grid/token 数匹配要求，以及 matched calls/frame counts 的措辞。熵门已明确降为预算实现细节，不单独主张 novelty。评分期新图见证写 runs，后续复用缓存单独 promote 到 data 并记 PROVENANCE。以上已回应本轮要求，不另开审查或追加主结果前实验。

**最终：PASS。** 保留上述来源、metadata 和对照边界，进入实现与独立代码检查；不据此更新论文或宣称机制有效。
