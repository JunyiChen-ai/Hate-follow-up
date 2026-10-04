# Candidate21 semantic cluster tree：独立 proposal review

日期：2026-10-04。**PASS（same-family provisional，独立 agent）**。

依据 `RESEARCH_ITERATION_RULES.md` 第4条，当前方案没有触发四类 STOP。放行范围是 README 定义的完整语义聚类树获取与新全局/局部读取，不是仅增加局部帧。候选21仍是备选；本审查不评价正在运行的候选20，不批准任何性能或机制结论。没有运行 GPU、读 GT、改实现、计算哈希或发送外部通知。

## 本次阅读与检索

读取 `experiments/20261004_m1_tree/README.md`、权威 `CLAUDE.md`、规则4及相关规则、`docs/reviews/20261004_m1_ideation_jury.md` 第三项与近邻记录、`experiments/20261004_m1_ideation/CANDIDATES.json` 中 `complete_semantic_cluster_tree_with_leaf_regrounding`，以及归档 Explorer 的机制/常数/审查/归档范围。既有原始模型接口阅读沿用本 agent 前次对 `src/mllm_judge.py` 的只读核对；没有声称审阅尚未存在的 candidate21 实现。

在线重新打开并实际核对 [VideoTree §3.1–3.3、§9–11](https://arxiv.org/html/2405.19209v3)，另用原始 HTML 文本确认网页工具未充分展示的完整方法段。读取 [MATCH 原论文](https://jianlang.org/papers/MATCH.pdf) §III 总览、预处理、证据匹配与实现段；检查 MAESTRO 官方报告实际可访问的索引内容，直接全文打开失败。

检索请求及原始响应存于 `runs/20261004_m1_tree/proposal_review/web_request_01.json` 至 `03.json` 及对应 `web_response_*.json`；实际主文文本存于同目录 `videotree_primary_text.txt`。不把此前 jury 的阅读范围冒充本次新阅读全文。

| 实际查询/一手来源 | 发现与限定 |
|---|---|
| `"hateful video" "VideoTree"`；`"VideoTree" "hate" detection` | 未找到完整 VideoTree 已用于目标 hateful video detection/localization 的明确证据；返回一般视频理解和引用交集不等于使用证据。 |
| `"hateful video" hierarchical semantic cluster tree` | 找到 clustering/视频解释等近邻，未定位同一完整获取机制。 |
| `MAESTRO hateful video clustering hierarchy`；官方域名限定的 `Global Local`、`state tool`、`hierarchical`、`cluster` 查询 | 实际取得官方报告摘要与 tool/chunk 提示段；“hierarchical”命中 HTS-AT 声音模型说明，不足以证明 VideoTree 式视频聚类树。部分 cluster 查询返回无关官方资料，不计作支持。 |
| [MATCH](https://jianlang.org/papers/MATCH.pdf) | 正反线索提议、CLIP 与时空单元的匹配、验证及解释增强预测已经用于目标任务；其可访问方法不等于按 caption relevance 扩宽/扩深的语义聚类树。不能声称首次证据检索。 |
| [MAESTRO 官方报告](https://www.dsta.gov.sg/staticfile/ydsp/projects/files/reports/Report%20-%20Hear%20Me%20Out%20%28%26%20Think%29_%20MAESTRO%2C%20A%20Multimodal%20Agentic%20Model%20with%20Efficient%2C%20Synergistic%20Text-Reasoning%20Optimisation%20Framework.pdf) | 可访问索引确立已有 global/local 循环及选 tool/chunk 继续获取信息。直接全文失败，不能声称排除其所有内部层级机制；当前未取得“相同完整来源方法已用”的肯定证据。 |

因此结论是“在本次限定检索中未发现”，不是绝对全球首次。MAESTRO 的全文访问限制保留为透明的不确定性，而非擅自视为已排除重叠或直接 STOP。

## 四类 STOP 的具体判定

1. **来源已用于目标任务：未成立。** 已确认的目标近邻覆盖检索、caption、global/local 重读，不构成来源完整 adaptive breadth/depth clustering acquisition 已用于目标的证据。
2. **纯 ensemble：未成立。** 唯一 Qwen 视觉 tower 产生聚类特征，同一冻结 Qwen caption/评分/读数；没有第二视觉模型或独立模型预测融合。一个确定树产生一组新 global/visual/speech 值直接替换原生值，不和 Explorer/native 分数择优、平均。
3. **纯 calibration / 后处理：未成立。** 树控制真实帧与生成观测的获取，并改变语义测量的输入；固定 r6/4fps evaluator 未改。无标签阈值拟合、语料特定路由或新增分数平滑。
4. **纯工程：未成立。** 包含特征聚类、代表观测、相关性判定、宽度迭代、相关性控制深度、确定最终树及逐窗叶观测读取，具有完整的科研机制与反事实对照。不是仅换 prompt、分辨率或添加两张图。

## 源方法范围与 Explorer 边界

源 VideoTree 通过聚类代表帧 captions 判定相关性，扩宽至足够高相关根或上限，再按根相关性扩深，最终按时间排序节点描述。§9 明确区分树对获取的作用与显式树顺序对 reasoner 的作用；不能把本方案 ancestor packet 的必要性归为源论文已经验证。[VideoTree](https://arxiv.org/html/2405.19209v3)

本方案保留完整获取流程；用唯一 Qwen 替代来源的不同视觉/描述/推理模型、固定4/8/16宽度与2分支、独立 digit relevance、增加窗口真实像素重读及新 global/stance，均已作为适配声明。`max_breadth` 时保留最后根集、不可再分时停止，也已有明确定义；没有源算法关键步骤被无声明删除。

Explorer 的归档 README 明确是保留 native global/speech 的单窗 entropy/support 反馈 acquisition，原 attention-distance 候选策略和三次修订预算已经结束。candidate21 的视频级特征聚类树、caption 预算分配及新 global/全部窗口读取改变了完整方法单位，可以作为独立候选。**此 PASS 不重置 Explorer 预算**；若落地删除树构建仅恢复单窗口反馈加帧，不能沿用本审查结论。

## 窄定义问题与后续核验

目前没有阻止实现的方法定义缺口。实施前需把 README 目前语义描述的 global node records、local packet、祖先“interpretation context”标记落成逐字固定模板并记录；这属于落实现有定义，不能评分后按结果选择措辞。数字1/2/3须验证确为单 token，caption generation 的完整 system/user 模板及 EOS/截断都应保存。

“局部证据 ONLY selected images”是输入指令规定的证据范围，不是硬 attention 隔离：原 overview、本窗口原生帧及全部节点 captions 仍在全局缓存。实现与结论不能声称其它内容不可见，或 caption 本身已证明局部发生。原问题必须保持原文，并用实际 PTS 明确区分本地像素和外窗语境。

帧池、KMeans、叶/窗口 membership、源索引去重、singleton/相同特征不可分、caption精确帧复用、新 global hard stance、每窗独立 cache/rotary 状态都应由后续规则6代码审查核对。所有新值实际进入固定 r6，而 native 配对仍精确复现；不得用误差容忍修改评测器。现在尚无实现，不把这些检查写成已通过。

## 获取机制与读取机制分开证明

- `temporal` 的匹配树形/调用机会对照检查语义获取是否优于等机会时间分组；它是依赖主方法树形的诊断，不是可部署 baseline。须记录真正发生的节点/调用数，空组、重复代表 caption 复用会让上限相同不等于实际成本相同。
- `no_depth` 检查获取的层次；若 breadth 被单独主张，则做已声明 `no_breadth`。来源 tree acquisition 成立不能由 `flat` 单独判断。
- `flat` 和 `wrong_links` 保持已获取节点/像素/文字，检验新增的读取关联。若全局 records 仍泄露正确 parent IDs，`wrong_links` 只证明局部冲突包影响，不能说整棵树关联已替换；按实际干预范围解释。`flat` 必须去除所有预定父子标记，并保持观测文本，不顺带删除观测信息。
- `no_added_pixels` 区分真实局部像素的额外贡献。只有更多像素带来提升、层级与语义对照未通过时，不得称层级机制成立。

各独立主张沿用 README/规则14g 的双语料同主指标≥.01消融门，全部六指标与 raw ordering 同报。不得要求运行前证明提升，也不因潜在 shortcut、caption 错误或新 global 可能退化而 STOP。

成本已完整声明，60–180 GPU分钟尚未测量。最坏112个最终节点之外还要计此前 breadth round 的 discarded caption/relevance、新 global 长前缀及逐窗像素重编码；不是只统计最终树。固定无GT smoke用于成本/容量核验，不用于挑树深或词句。内存不足须显式记录实现处理，不能静默截断树/转录或缩小方法预算。

归档前窄范围确认：实际重读更新后的 README 步骤8/9，global node/parent/depth/time 与 local context/image packet 的 literal 格式及确定节点 ID 已固定；也已明确 instruction-level scope、全 overview/ASR/global captions 仍可见。上述两项意见已解决，没有待审批条件；未重新展开泛化审查。

最终裁定：**PASS，限当前完整候选21的备选方案。** 后续是否启用由主 agent 按候选20的实际完整结果与既有流程决定。
