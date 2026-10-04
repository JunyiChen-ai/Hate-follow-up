# 候选23 executable temporal evidence program：一次独立 proposal review

日期：2026-10-05。审稿实例：/root/latents_proposal，gpt-6-astra，与主 agent 不同实例、相同模型；**same-family provisional**。结论：**PASS（规则4）**。本结论允许按既定队列准备实现，不启动或取代 Tree21/Lattice22；GPU 启动仍服从前序家族的结果分流。不是效果、晋级或最终 novelty 认证。

## 范围与结论依据

审查对象为 experiments/20261005_m1_program/README.md 的完整方案及本次窄澄清后的版本。已读 CLAUDE.md、RESEARCH_ITERATION_RULES.md、原始 CANDIDATES.json 对应项与 ideation jury 第5名；只读 src/mllm_judge.py 的冻结输入/问题定义。未改生产、未读 GT 或预测、未运行 GPU、未计算哈希，未修改其他候选。

本轮独立实际检索包括：
- `"ViperGPT" "hateful" video`
- `"visual program" "hate" video detection`
- `"hateful video" "program" reasoning`
- `"ViperGPT" "HateMM"`
- `"hateful video" "programmatic"`
- `"hate video" "code generation"`
- `"hateful video" "executable"`

另查 VPD meme、Robust Framework 及 MAESTRO tool/program。原始请求与完整工具返回在 runs/20261005_m1_program/proposal_review/web_request_01.json 至 web_request_07.json 及对应 response；不是以父 agent 的检索结论代替独立核查。检索结果含无关网页，它们不作为学术证据。有限检索未发现完整来源方法已用于目标任务，**不等于证明全球不存在**。

| 规则4的实际 STOP 类别 | 判定与依据 |
|---|---|
| 来源完整方法已用于 hateful video detection/localization | 未成立。已核读的目标近邻有客观描述、相反假设、检索、验证、真实工具调用，但没有确认 ViperGPT 式生成依赖程序、解释器执行时间/来源组合与事实模块的完整迁移已用于目标。MAESTRO 全文访问受限，保留明确不确定性。 |
| 纯 ensemble | 未成立。一个冻结 Qwen 同时承担 planner、事实模块及最终 reader；模块无 hate verdict，只有每个 native 分支的一次 Yes/No margin 进入原有 raw max/r6。不是多个候选程序/分类器的分数混合。 |
| 纯 calibration/后处理/平滑 | 未成立。程序先控制实际文本/图像取值、时间绑定及事实查询，再重新进行局部测量；不对既有分数作校准或平滑。 |
| 纯工程输入/prompt 技巧 | 未成立。声明的研究对象包含生成程序→执行 typed 依赖→按源调用事实感知→返回执行记录→局部读取。时间点归属、字符裁剪、引用上下文及 join 被解释器实际执行，非只验证 JSON/ID 或改措辞。完整机制是否有效由实验判断。 |

## 来源迁移及近邻

[ViperGPT](https://arxiv.org/pdf/2303.08128) 已实际阅读本地下载全文中的3.1–3.3、4.4和 Appendix B 的 VideoSegment API。原来源是 LLM 生成 Python、调用预训练感知模块、由解释器执行，视频通过片段与有序帧接口处理。在线核读官方 [run_program](https://raw.githubusercontent.com/cvlab-columbia/viper/main/main_batch.py) 的 compile/执行及错误处理和 [VideoSegment](https://raw.githubusercontent.com/cvlab-columbia/viper/main/video_segment.py) 类/帧接口。不能把原 API 的帧索引当精确秒时间。候选的限制型解释器、ASR 字符区间、scope、统一 Qwen 及无 retry，都是明确适配；不是官方实现复现。

| 一手来源、实际阅读范围 | 已有机制与本方案边界 |
|---|---|
| [MARS](https://arxiv.org/html/2601.15115v1)，2.2.1–2.2.4，直接请求段落保存在 mars_methods.txt | 客观描述、hate/nonhate 假设及合成已有；中性事实描述或分阶段调用本身不是新贡献。 |
| [MATCH](https://jianlang.org/papers/MATCH.pdf)，III-C/D | 局部 frame+transcript 单元由 CLIP 相似度配对线索，verifier 生成解释，再训练 rationale-enhanced predictor。时空证据检索/验证已有；所读定义不是生成 typed 依赖程序。 |
| [RAMF](https://arxiv.org/html/2512.02743v1)，3.2及相邻总体方法；未逐式审核3.4–3.5 | 三阶段客观/相反假设文本进入融合；不把“增加事实描述”作为程序贡献。 |
| [LEAF](https://aclanthology.org/2026.findings-acl.604.pdf)，3.2含 Ground/Eq3–4及3.3开头，PDF/text保存在本审稿证据目录 | 标签纠正后的解释与视频 grounding、逐阶段蒸馏已有。零标签提案不迁移其 golden-label 路径。 |
| [IARE](https://arxiv.org/html/2606.11953v1)，4.2–4.3，直接段落保存在 iare_methods.txt | 注释上下文进入 SFT，正确/错误理由进入 DPO；不等于实际执行来源/时间操作。 |
| [MAESTRO 官方报告](https://www.dsta.gov.sg/staticfile/ydsp/projects/files/reports/Report%20-%20Hear%20Me%20Out%20%28%26%20Think%29_%20MAESTRO%2C%20A%20Multimodal%20Agentic%20Model%20with%20Efficient%2C%20Synergistic%20Text-Reasoning%20Optimisation%20Framework.pdf)，官方索引摘要、CONTINUE/chunk/tool 指令和附录1.6–1.8摘录 | **已在 hateful video 实际调用 DeepFace/YOLO 等工具并迭代更新内容**。工具编排或 fresh perception 首次应用的主张不成立。正文 open 返回403，不能宣称排除全部重叠；已访问材料未确立完整 typed span/scope/join 程序机制。 |

[Robust Framework 作者 slides](https://dipteshkanojia.co.uk/files/ppt-mm4sg-www-2025-hate.pdf) 的零起始页4/9/12（物理页5/10/13）已读：PALI-X-VPD 在 Meme-based Methods/HMC 表，HateMM 视频表是 embedding/order-aware fusion。[同篇论文](https://arxiv.org/html/2502.07138v1) 的概述亦明确两个不同任务。[VPD 原始论文摘要](https://arxiv.org/abs/2312.03052) 与一手 PDF 检索段落确认 Hateful Memes；本轮未阅读全文逐项排除其全部应用。不能把 meme 等同 video 而触发 STOP，也不能把程序方法称为从未用于仇恨内容。

因此可保留的迁移单位是**完整可执行时间/来源/话语 scope 依赖程序**，不是新增 caption、工具调用、JSON 格式、来源标记或“证据推理”泛称。

## 本次窄澄清已确认

主 agent 修订后已重新实际阅读 README，以下均完成，不遗留本轮 proposal 定义阻塞：
1. 帧时间改为 native_filename_nominal，明确 seek/fallback 与未验证 decoded PTS。ASR 字符时间只是比例估计；不能声称精确媒体时间或词对齐。
2. frame 是时间点；join 明确 span.start <= frame.time < span.end 且同目标窗口。不会把零长度区间误判为全部不相交。
3. 已冻结 planner literal system、API/schema、source table、结尾指令；8窗/2048生成、每窗12操作/2感知调用、96 factual tokens 保持声明。缺失为 UNKNOWN、同窗重复全部拒绝、越 chunk 拒绝记录、无 fewshot/retry。字符按 Unicode、完整包含 ceil/floor 及1e-9算术容差明确。
4. MAESTRO 已知真实工具编排重叠已写入范围，不再以 fresh perception 首次应用作为区别。

这只是定义澄清，不是理论失败判决或新增一轮泛化审稿。实现后规则6仍需核对：操作实际控制模块输入/最终记录，源坐标未重写，UNKNOWN 不作为不存在，缓存独立，无标签进入计算。来源合法仅证明引用有效，不保证语义判断正确；最终 reader 保留 native 全局信息，不能称硬信息隔离。

## 成本与后续声明边界

新增成本已明确为每视频 ceil(W/8) 次 planner、每窗0–2次 fresh factual generation及正常局部重读；原20帧/ASR复用不消除生成成本。120–360 GPUmin 是未测估计，不是承诺。固定无 GT 五视频测成本并检查实际执行；没有以成本或预测收益否决方法。

若完整两语料通过主门，已预声明未执行程序、匹配 planner/module 预算、同事实输出 flat、错误窗口/参数绑定、scope/join 去除等控制。新增事实调用的收益不能代替程序依赖的贡献。完整程序与单个 operator 的贡献按 README/规则14g分别验证；最终三项指标、计算代价及 development-selected 标识不能省略。

**正式 verdict：PASS；same-family provisional。** 本次规则4审查结束，不预判效果，不授权越过 Tree/Lattice 的结果队列，也不替代后续独立 code review 或最终机制/novelty 审查。

