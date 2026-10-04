# M1 候选扩展（2026-10-04，进行中）

目标、正式r6与评测器固定，Explorer初版+三修订已归档；此次扩展不重置该族预算。
使用idea-creator与novelty-check。仓库规则优先于技能默认：同模型gpt-6-astra独立实例；
输出位于本实验目录，完整215+118作性能证据，固定五视频只检查实现；
不因技能的阶段预算终止用户自主目标，不新增语料，不把普通输入/提示改动当novelty。

三个readonly生成镜头：声学/话语时间绑定、完整global/local证据重读、正交M1机制。
每项给dedup_key；主agent按完全相同字符串机械去重，不作新颖性预筛。
全部可行项连同风险交给一个fresh独立jury；其same-family判断标provisional，
最终是否可写机制仍由完整结果、部件消融和错误绑定对照决定。
无papers/或literature/本地文献库。research_wiki.py已定位，缺失query_pack重建后
strict扫描通过，空pack没有研究证据；失败经验来自已读的项目状态与归档。

实际检索的六个景观查询：2026 hateful video temporal localization label free SAGE MAESTRO grounding；
training free video temporal localization active evidence hierarchical 2025 2026；
speech word alignment uncertainty multimodal video reasoning zero shot 2025 2026；
training free multimodal video reasoning scene graph evidence retrieval 2026；
zero shot video question answering temporal ordering speech lattice inference；
MAESTRO hateful video adaptive global local reasoning loop paper。

主agent阅读范围与初步景观（检索记录，不是方法新颖性裁定）：
- Graph-to-Frame RAG，https://arxiv.org/html/2604.04372v1 ，摘要/引言与3.2–3.4：
  图构建、检索、渲染推理帧；原实现GPT4o+4omini，迁移必须改成同一个Qwen并计新视频成本。
- CASTLE层级知识图，https://arxiv.org/html/2606.01933v1 ，摘要/引言：多跳视频检索。
- MAGIC-Video，https://arxiv.org/html/2605.08271v1 ，摘要与引言跨模态碎片、粗粒度信息损失。
- TFVTG，https://arxiv.org/html/2408.16219v1 ，摘要/引言：子事件语义、顺序与动态转场。
- Video-R2，https://arxiv.org/html/2511.23478v1 ，摘要/引言：SFT+GRPO的时间/思维一致性，
  不是可直接无训练搬来的性能证据。
- TimeLogic，https://arxiv.org/html/2606.01631v1 ，摘要/引言：主动多粒度采帧；
  原方案按语料/题型预算，不能照搬本项目的按语料流程。
- SAGE，https://aclanthology.org/2026.acl-long.817/ ，权威摘要与官方实现README；
  PDF工具打开失败，监督模态专家/仲裁已用于hateful video，不能作为迁移novelty。
- CLARA，https://arxiv.org/html/2608.15905v1 ，摘要/引言、3.4–3.5及B.3：
  clip融合、视频rationale、监督gatedTransformer已用于hateful video。
- NeuS-QA，https://ojs.aaai.org/index.php/AAAI/article/view/37834 ，权威摘要；
  PDF错误URL未成功，尚不声称完整复现其逻辑自动机。
- WCN spoken understanding，https://arxiv.org/html/2401.02921v1 ，摘要/引言与2.1：
  ASR备选词上下文表示；RNNT/Kaldi并非现Whisper，完整迁移需声明。
- Whisper内部对齐，https://arxiv.org/html/2509.09987v1 ，摘要与方法/限制：
  character教师强制、headfilter/hardDTW；posterior支持是拟议扩展。
- Unsilencing Visual Latents，https://arxiv.org/html/2605.02735v1 ，摘要/引言、
  方法3.2/3.3/Algorithm1/实现：4latent、2positive4negative、5warmup15search，
  原论文未充分给学习率/初始化，须明确适配选择，不声称精确复现。
- Quotation attribution，https://arxiv.org/html/2406.11380v2 ，摘要与相关工作；
  其gold人物表不适用zero-label视频。

九个候选完整保存在CANDIDATES.json，按dedup_key精确去重后无重复，没有预筛删除。
独立jury排序：视觉潜变量优化、声学对齐后验、完整VideoTree；其余六项保留。
审查docs/reviews/20261004_m1_ideation_jury.md，原始响应与检索记录位于
runs/20261004_m1_ideation/jury_evidence/。本轮选择候选19连续视觉状态优化进入方案审查与实施，
其余候选没有被判STOP或失败。官方潜变量代码已公开，初读“未公开”由实际访问纠正，
论文与代码的差异在候选19 README明确记录。尚无新GT分析或性能结果，所有结果均development-selected。

维护记录：检索既有ASR输入脚本时遇到legacy baseline bootstrap中的内容摘要校验与第三方Git标识固定版本；按仓库禁用规则删除该校验/固定标识，模型可用性由消费者实际解析检查。默认模型输出路径归入仓库data/assets，未运行bootstrap、未重算任何历史baseline。Shell语法检查通过。
