# Projector 独立 proposal review

日期：2026-10-03。结论：**PASS**，仅表示按 `RESEARCH_ITERATION_RULES.md` 第 4 条可以实现和实验，不表示性能、机制解释或最终 novelty 已获验证。

独立审稿实例：`/root/m1_grounder_proposal_review`，与主会话同模型。审查声明为 `experiments/20261003_m1_projector/README.md`；未修改声明或实现，未运行 GPU，未读取 GT、待完成候选预测或性能。未使用内容哈希。

## 四项 STOP 判定

| 条件 | 判断与证据 |
| --- | --- |
| 来源已用于 hateful video detection / localization | 本次实际检索及原文核读未发现 ACG 或同一 masked-reference attention-output orthogonal guidance 已用于目标任务。此结论受检索范围限制，不宣称穷尽证明。 |
| 纯 ensemble | 否。同一冻结 MLLM、同层同一组 Q/K/V 的两个 attention reduction；没有组合独立模型。 |
| 纯 calibration / 后处理 / 平滑 | 否。在各层 attention 输出进入 `o_proj`、残差及后续层之前改变内部计算，原始窗口读数因此可能改变；不是只变换现有分数。 |
| 纯工程技巧 | 否。完整迁移 masked reference、对比方向和正交投影组成的干预，而非只调常数、措辞或输入分辨率。Qwen3 适配细节本身不单独构成贡献。 |

## 主来源与声明核对

实际阅读 [ACG arXiv v2](https://arxiv.org/html/2601.13707v2) 的 Algorithm 1、§3.2–3.3、§4.1、§4.3、§5.3–5.4 和附录 C.1。作者、题目、CVPR Findings 2026 归属可由 [CVF 官方条目](https://openaccess.thecvf.com/content/CVPR2026F/html/Jo_Attention-Space_Contrastive_Guidance_for_Efficient_Hallucination_Mitigation_in_LVLMs_CVPRF_2026_paper.html) 核对；直接页面访问曾返回 403，检索索引可读条目，方法判断以可读原文为依据。检索未核验到官方实现，不能称作者代码复现。

声明中的 `O + 1.4 * perpendicular(O-U)`、从最后文本位置遮蔽图像键、同层重用 Q/K/V、在输出投影前修改 attention 输出，均符合源算法。全层对应 ACG-Full；原文另有仅早期层的 ACG-Fast。1.4 是源 Qwen-VL 常数，原论文在 CHAIR 上选取 guidance scale；不能写成源方法无调参。§4.3 新增实验是 LLaVA-NeXT 7B/13B，没有 Qwen3 证据。源论文 wall-clock 比例不能直接当作本候选成本。

以下是已正确披露的迁移边界：Qwen3 的 GQA/QK normalization/mRoPE、20 帧不连续图像键、每头分别投影、固定 epsilon、BF16/FP32 混合计算以及二元答案读出。公开算法不足以验证这些细节与作者代码一致。

几何解释须保持准确：删除的是**对比修正方向**沿 masked output 的分量，并没有删除普通输出 O 中全部相应分量。epsilon 和 dtype 转换意味着数值正交近似成立；每头输出空间中的正交也不保证经过 `o_proj` 后仍正交。U 仍包含可能携带视觉信息的文本状态，不是真正重新编码的无图像分支；这些限制不构成 STOP。

## 实际目标任务检索

本轮实际提交的检索覆盖：

- 精确标题 / `2601.13707` 与 `hate`、`hateful video`、`github`；
- `ACG` 与 `HateMM`，`HateClipSeg` 与 `contrastive guidance`；
- `hateful video` 与 `orthogonal projection`、`textual orthogonalization`、`attention output`、`unconditional`、`masked guidance`。

没有检出目标任务采用证据。ACG 缩写和泛称 attention/contrastive 的命中不能直接作为来源采用证据；例如 MultiHateClip 检索中的 ACG 指动画、漫画、游戏内容类别。

为避免只按标题或缩写判断，另读取以下 primary 方法内容，并检查来源题名、编号、orthogonal / unconditional 等线索：

- [RAMF](https://arxiv.org/html/2512.02743v1)：§3 的 adversarial reasoning 与多模态融合采用生成的双立场理由及融合层，未见同层 masked attention-output 正交指导。
- [CLARA](https://arxiv.org/html/2608.15905v1)：§3 采用片段划分、模态特征、MoE 和 rationale 引导的时序表示，未见 ACG 机制。
- [IARE](https://arxiv.org/html/2606.11953v1)：面向可解释仇恨视频判别和证据理由，原文未见 ACG / 该正交指导。
- [SAGE](https://aclanthology.org/2026.acl-long.817.pdf)：§3 的模态专家、全局交互与实例裁定并非冻结模型内部的同层 masked-reference 投影。

这些邻近工作已讨论模态信息被稀释或噪声遮蔽；不能把这一宽泛问题本身写成首次发现。允许检验的贡献边界是 ACG 内部输出干预对本任务的有效迁移及其可证伪证据。

## 与旧候选、成本及解释边界

Projector 的对象是 attention **输出向量**：Grounder/Selector 改直接可访问的键或头；Amplifier 改图像键 logits；Recycler 改 attention probability 的预算；Reinforcer 用另一次前缀编码获得残差方向。它们不等于当前计算图。VCD 的最终 logit 对比也不能替代这次试验。不因此前负结果推断此候选必然无效。

调用声明在当前接口下自洽：部署 `3+B`，native/eager/guided 配对 `3+B+2V`，恢复检查另加 `3+B`。无额外完整前缀或第二 KV context，但每层引入最后一行的普通和 masked attention 计算；可能还存在实现上的冗余 native reduction。因此相同外层调用数不是零额外开销，GPU 时间和峰值显存必须由 smoke 实测替代预估。

native、matched eager 和 guided 三臂能区分 native kernel 与干预结果；晋级同时比较 native/current 和 eager 有必要。所谓 native prefix、global、answer、speech 不变是缓存和对应读取不变，不能据此声称 guided 最后 query token 在后续层的隐藏状态仍相同。全部实际图像键均被遮蔽，没有目标窗口专属视觉键干预，因此仅凭投影统计不能证明时间选择性。

同一 1.4 下移除正交投影能检验该部件是否必要；norm-matched 随机方向能检验收益是否仅为扰动幅度。这些控制可证伪，但 alone 不足以确认改善来自正确时间位置的语义证据。声明已承认还需内容敏感控制。若取得 qualifying gain，应先固定随机方向生成的 shape、归一化、零范数处理和遍历顺序，再运行控制；不以投影接近零、单个案例或外层调用不变替代定位和成本证据。

本轮无须修改核心机制。后续仅需忠实实现已披露边界，执行声明的独立代码检查、完整评测及有条件的机制消融；不增加第 4 条以外的提案否决条件。
