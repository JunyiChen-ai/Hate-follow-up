# M1 下一批候选准备（2026-10-05）

用户自主目标仍在运行；Program23 R2、Provenance25、Interval26、Verification27
保持原实验及修订预算。使用 idea-creator 继续准备来源明确、低新增成本的方案。
项目规则优先：同模型独立实例、完整333统一评测才作性能证据，固定五视频只作
实现检查，不新增数据集，不因技能阶段预算终止用户目标。所有产物在本项目内。

已归档或用完修订的家族不得通过改名复活；精确清单读取 STATUS 和归档 README。
当前累计归档23项，程序23有正向进展保留。
原九项候选池已全部进入独立方案流程；其中25/26/27尚未完整性能裁定。

新景观检索实际运行七种表述，原始结果及官方HTML阅读记录位于
`runs/20261005_m1_ideation/landscape_search.json`，方法继续阅读另存。
本项目没有 papers/ 或 literature/ 文献库；已有原始阅读在各 runs/source_reads
及 third_party/vtimecot_source_read。query_pack 2026-10-05 strict扫描通过后
立即读取；只有索引框架，没有新的研究证据。正式状态与归档是失败经验来源。

新读 primary MERIT2608.07663v1 方法3.1–3.4和消融/成本4.4：多键片段索引及
按需邻域过滤，不依赖预先的全局复杂图。原模型/编码器异于单一Qwen约束，
任何迁移须保持完整功能并声明替换与新视频成本，不宣称本任务效果或新颖性。
重新核读 VTimeCoT2510.14672v1 方法3.1–3.3：同步进度条、视频文本检索高亮、
实际工具交互/切片和动态视频记忆；单纯加时间条不是完整迁移。
VideoZeroBench2604.01569v1 摘要/引言区分答案正确与实际来源定位；只作诊断背景，
不加入它作为新数据集。TFVTG v2 HTML访问失败，保留原轮实际v1读取范围。

生成镜头为视觉时间工具、源内部时序表达、低成本检索/语义输入绑定。
每镜头3项，readonly返回带dedup_key的结构体，不排名、不删项、不写共享文件。
主agent仅按完全相同dedup_key机械去重，全部可行项交一个fresh独立jury；
same-family provisional，rule4只有四项 STOP，最终机制取决于真实完整消融。
尚未选择/实施新候选，尚未GPU/GT/性能检查。

派发记录：visual_time_2和temporal_rep_2为fresh同模型readonly shard；第三次spawn因agent thread limit失败，复用闲置latents_proposal作cheap_binding_2生成镜头，明确不让它排名或裁定。最终jury已成功创建fresh独立实例，完整读全部产物，不伪称第三镜头fresh。

九项全部生成完毕，原始JSON从各只读镜头的实际assistant最终消息逐字解析保存，未让镜头写共享文件；全部写入CANDIDATES.json。按完全相同dedup_key机械去重没有重复，未预筛删除任何项。尚未独立jury裁定或选择新实验。
