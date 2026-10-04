# 候选24 source-bound quotation and reference graph：独立 proposal review

2026-10-05；独立实例 /root/latents_proposal，gpt-6-astra，与主 agent 相同模型、不同实例。**PASS，same-family provisional。** 这是一次规则4提案审查及窄定义确认，不是效果或最终 novelty 认证。候选24仍为备选，不越过22/23的结果分流。

## 审查范围与四类判定

实际阅读 experiments/20261005_m1_quote_graph/README.md 全文及本次澄清；CLAUDE.md沿用同会话已完整读取版本，重新读取 RESEARCH_ITERATION_RULES.md 规则4；只读文档、公开论文及证据。没有实现、GPU、GT、预测读取、内容哈希或外部通知。

| 唯一允许的 STOP 类别 | 结论 |
|---|---|
| 完整来源方法已用于 hateful video detection/localization | 本轮实际检索与可访问一手材料未确认此事实，不触发 STOP。具体检索范围和不可访问部分如下；不是全球不存在的证明。 |
| 纯 ensemble | 不成立。图抽取与新 speech reader 使用同一冻结 Qwen，无候选图投票、无额外分类器/encoder分数混合；native G/stance/V不变，只有新S进入既定max/r6。 |
| 纯 calibration/后处理/平滑 | 不成立。精确原文跨度建图、路径选择和每层attention可达关系先改变语义读取，再测量一个margin；没有事后图分数校准。 |
| 纯工程输入/prompt技巧 | 不成立。完整方法包括无gold库存的跨度发现、typed关系、共指合并、两跳来源包构造及context→local→question的非对称编码。其算法操作超过JSON格式、改措辞或拼接上下文；是否有效由全量实验及已声明对照判断。 |

本轮没有以ASR噪声、可能shortcut、共享prefix可见、预计成本或预测收益作为STOP理由。文学来源只是启发，不将自研图/attention算子错误归给来源。即使不属于忠实迁移，规则4允许自设计的完整机制。

## 实际检索与一手阅读

检索请求/原始返回保存于 runs/20261005_m1_quote_graph/proposal_review/web_request_01.json至05及对应response。查询包含：
- `"hateful video" "coreference"`
- `"hateful video" "quotation" graph`
- `"hate video" "discourse graph"`
- `"HateMM" "speaker attribution"`
- `"hateful video" "quotation attribution"`
- `"hateful video" "graph" "coreference"`

另搜显式证据归属和MAESTRO身份/共指工具近邻。无关检索命中不作为学术证据；未命中不作为absence证明。意外返回的外部checksum及未采用opaque gist URL已剔除，retention_note.json记录；没有计算或依赖它们。

**文学来源。** [Michel等](https://arxiv.org/html/2406.11380v2) 实际读§4、Appendix A、Fig7/8 literal prompts、§7。4096-token/1024-stride quote attribution使用给定quote标识和gold character-to-alias清单；重叠预测可回传并修订。作者明确非完整角色发现。候选的8窗/stride6、独立跨度发现、首次接受合并、无gold角色库是不同实现，不能引用来源准确率来保证ASR归属。

[Vishnubhotla等](https://aclanthology.org/2023.acl-short.64.pdf) 实际读§4/5/6及limitations，直接下载PDF/text保存在证据目录。其模块化区分角色发现、共指、引语检测及归属，阶段评测以注释纠正上游，归属模型含训练和标注候选；仅研究引号标记的引语，不涵盖自由间接话语/引语嵌套。候选的嵌套/间接话语与拒绝/否定图为自行扩展，不是假借文学pipeline效果。

**目标及邻近任务。**

| 来源与实际范围 | 对候选的约束 |
|---|---|
| [MATCH](https://jianlang.org/papers/MATCH.pdf)，本轮重读III-B/C/D | 已有相反立场线索、CLIP局部证据配对、verifier与训练predictor。不能主张首次时空grounding；所读方法未定义本候选的跨度共指图/非对称reader。 |
| [RAMF](https://arxiv.org/html/2512.02743v1)，本轮重读3.2及相邻fusion概述，非全文逐式审计 | 客观描述、对立假设和local/global融合已用；一般上下文解释不是本候选的新意。 |
| [Yadav/Singh，WWW2026](https://doi.org/10.1145/3774905.3796488)，出版方摘要实际访问；PDF/full抓取失败 | 已有多源共享证据结构、实体、确定性优先排序、跨模态冲突及可选约束LLM reasoner。摘要未显示精确跨度quotation/coreference图与此attention算子的完整方法，但不能宣称已排除全文重叠。 |
| [TikTalkCoref](https://arxiv.org/abs/2504.14321)，原始摘要 | 短视频+评论的多模态共指已有，不等于hateful-video检测/定位；不能将共指本身称为首次视频应用。本轮未读其完整模型。 |
| MAESTRO、LEAF、IARE | 复用本独立实例在候选23的实际来源阅读，不伪称本轮全文重读。MAESTRO已有真实tool/chunk调用与迭代；官方全文403，仅索引附录可访问。LEAF3.2已有标签纠正grounding，IARE4.2–4.3已有注释上下文/SFT/DPO。这些重叠否定“首次推理/归属/工具”的宽泛主张，不确立完整本方案已使用。 |

上述复用证据及URL在 docs/reviews/20261005_m1_program_proposal.md 与 runs/20261005_m1_program/proposal_review/；MAESTRO [官方报告](https://www.dsta.gov.sg/staticfile/ydsp/projects/files/reports/Report%20-%20Hear%20Me%20Out%20%28%26%20Think%29_%20MAESTRO%2C%20A%20Multimodal%20Agentic%20Model%20with%20Efficient%2C%20Synergistic%20Text-Reasoning%20Optimisation%20Framework.pdf) 的访问局限仍保留。可接受的研究单位只能是**精确来源图发现/绑定→确定性关系路径取包→非对称语义读取的完整方法**。

## 已确认的窄定义修订

本次仅请求并实际确认三项会改变实验观察的定义：
1. membership只允许utterance→span，不允许span经remote parent零跳获取同窗其他跨度；same_referent虚拟组件双向零semantic-hop，其余语义关系一跳。
2. 最终问题明确为原speech terminal question加Yes/No尾部，不再调用整个yesno_question重复header/body；JSON/utterance为0-based，印出的窗口编号为1-based。
3. 任一越界node、错type/window、悬空endpoint或typed-edge失败均整chunk UNKNOWN，与schema失败一致，不默默保留“有效子集”。

未遗留本轮定义阻塞。后续规则6只需核对实现是否忠实进入分数：窗口原文/offset一致、合并与可达路径正确、实际36层mask与token范围、cache/rotary恢复、native G/V固定及无GT路径。结构合法不证明语义归属正确；共享native ASR仍可见，算子只限制新增context KV的直接路径，不能写硬全局隔离。

Program23的typed时序/事实感知执行与本方案的文档图/结构读取具有已声明差别；本审查不预先判断二者实验效果，不授权重置任何既有家族的结果分流或修改预算。

## 成本、对照和声明边界

新视频仍支付stride6/chunk8的2048-token图生成以及每非空窗一次speech重读，30–90GPUmin/full333未测量。缓存复用不是免费推理；固定无GT检查记录实际调用、无效chunk、tokens、显存和时间。没有因成本否决。

若主结果过门，packet_serial必须使用完全相同来源包和token，检验非对称编码；nearest_context检验关系取包。当前nearest的实际长度/count可能无法完美匹配，README已要求在读取对照分数前解决matched-pool定义或明确降低归因强度，不能把更多上下文当关系机制证据。wrong_edges保留真实source坐标并报有效扰动覆盖；no_coreference对应独立共指贡献。两部件和完整机制按既定同一主指标双语料.01标准及规则14g检验，所有三指标和development-selected标识保留。

**正式结论：PASS，same-family provisional。** 本轮规则4审查结束；不预判效果，不替代独立code review和最终机制/novelty审查。
