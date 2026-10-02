# Highlighter 独立 proposal review

日期：2026-10-03。结论：**PASS**。按 `RESEARCH_ITERATION_RULES.md` 第 4 条，四种 STOP 情形均未成立；有效性和机制主张仍须正式实验验证。

审查实例：`/root/m1_grounder_proposal_review`，沿用与主会话相同模型；这是本候选的一次独立提案审查。读取 `experiments/20261003_m1_highlighter/README.md`、主来源及作者代码，没有修改提案或实现，没有运行 GPU，没有读取 GT、Reinforcer / Projector 的预测或性能，没有使用内容哈希。

## 四项 STOP 判定

| 条件 | 本轮判断 |
| --- | --- |
| 来源已用于 hateful video detection / localization | 实际检索及相关 primary 方法核查未发现 VGA 或相同视觉 token 词表显著性加权 value guidance 已用于目标任务。结论有检索范围边界，不是穷尽证明。 |
| 纯 ensemble | 否。词表投影、原始 attention 输出和附加 value 向量都来自同一冻结 MLLM，没有独立模型预测组合。 |
| 纯 calibration / 后处理 / 平滑 | 否。干预在指定层的 attention 输出进入 `o_proj` 与后续层之前执行，改变原始窗口读取；不是变换已得分数。 |
| 纯工程技巧 | 否。迁移了显著性分布、value guidance 与 head balancing 的方法机制；窗口限定是适配。若最终仅常数或实现调整有效，则不得单独包装成贡献，但不能在实验前据此 STOP。 |

## 来源和作者代码

实际核读 [VGA arXiv v1](https://arxiv.org/html/2511.20032v1) 的 §3–4、§5.3 及附录 A.3 / E。论文后来发表于 [CVPR 2026 官方条目](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Tell_Model_Where_to_Look_Mitigating_Hallucinations_in_MLLMs_by_CVPR_2026_paper.html)。本轮以提案明确指定的 v1 和下列公开代码为定义依据，不把不同版本混作相同实现。

实际从 [作者仓库](https://github.com/beta-nlp/VGA) 的 raw 文件读取以下内容，并查看了仓库内 `third_party/vga_source_read/` 对应快照；未作内容哈希校验：

- [Qwen 评测入口](https://github.com/beta-nlp/VGA/blob/main/eval/object_hallucination_vqa_qwen25-vl.py)，第 95–123 行：先读取前缀输出 logits；object-agnostic 分支对完整词表 softmax 后的 top-10 概率求 `-p*log(p)/log(10)`，随后跨图像位置 sum-normalize。**没有对 top-10 概率重新归一化**。这与论文 Eq.4 缺少概率乘数的写法不同，提案已经正确选择并披露 code-defined 公式。
- [Qwen attention 实现](https://github.com/beta-nlp/VGA/blob/main/qwen2_5_vl/modeling_qwen2_5_vl.py)，第 913–938 行：在 SDPA 输出上增加 guidance 加权的 visual values；系数包含 `G>1e-8` 的比例以及根据 cosine、跨头归一化、`relu(2-headw)` 计算的 balancing。非归一化分支是向量相加，符合声明。
- [作者运行配置](https://github.com/beta-nlp/VGA/blob/main/scripts/all.sh)，第 9–12、23–26 行：`attn_coef=0.2`、`head_balancing=simg`，Qwen2.5 的层为 4–15（包含两端）。评测入口 `attn_norm` 默认 false，支持提案采用不重归一化的输出相加。

原论文参数经过幻觉 benchmark 选择，不能写 parameter-free。论文与代码的层数和参数命名差异已经披露。Highlighter 的 4–17 层、Qwen3、多个不连续帧块、窗口内 G、FP32 运算和最后 query row 是适配，不能归为已验证的作者配置。没有连续 caption 生成，省去后续生成词触发的 guidance 抑制，与本次一位置二分类读出相容。

论文 §5.3 区分 object-directed confidence 与 object-agnostic salience；后者并非对任意二元问题都等价于对象 grounding。提案没有对象标签，使用后者合理构成待检验的迁移，但不能据源论文单图实验宣称它能确认仇恨语义。熵统计也不是“越大越确定”的通用证据。

## 实际目标任务查新

本轮检索包含以下组合：精确论文标题 / `2511.20032` 与 hate / hateful video；`Vision-Guided Attention` 与 hate / hateful video；`VGA` 与 HateMM / hateful video；`HateClipSeg` 与 vision-guided / 来源编号；`hateful video` 与 visual semantic salience、value saliency attention。没有发现目标任务使用来源方法的 primary 证据。一般 visual attention、语义融合或 saliency 字样不足以判为 VGA 采用。

另外对如下 primary 原文检查 VGA、来源编号、Vision-Guided Attention、Visual Semantic Salience / Confidence，并核对邻近机制：

- [HVGuard](https://aclanthology.org/2025.emnlp-main.456.pdf)：§3 为多模态提取、MLLM CoT rationale 与 MoE 分类。
- [LEAF](https://aclanthology.org/2026.findings-acl.604.pdf)：§3 为 Self-Grounding CoT 与阶段蒸馏。
- [SAGE](https://aclanthology.org/2026.acl-long.817.pdf)：模态专家、全局交互和实例裁定。
- [CLARA](https://arxiv.org/html/2608.15905v1)：clip 级 MoE、对比学习和 rationale 融合。
- [RAMF](https://arxiv.org/html/2512.02743v1)：双立场 reasoning 与融合。
- [MultiHateLoc](https://arxiv.org/html/2512.10408v1)：时序编码、跨模态对比、动态融合和 MIL。

这些原文未见该来源或同一内部 value-guidance 机制。对“增强视觉证据”“关注局部”这类宽泛动机不作首次性主张。按本仓库规则，尚无目标采用证据的完整来源迁移可以进入实验。

## 需要保持的窄范围边界

1. **数值定义**：显著性求和中的 `p=0` 应按 `0 log 0 = 0` 实现，不能让 underflow 产生 NaN 后误入“全零”分支。cosine 应明确按两个范数分别 clamp，与作者 `F.cosine_similarity` 语义一致；跨头和为零时设 gamma=1 是声明中的适配，不是作者分支。无需改主机制或扫描常数。
2. **局部支持不等于局部信息隔离**：G 和 D 的直接 value 支持被限制到本窗帧，但这些 native cached values 已经包含因果前缀上下文。全局判断、ASR 和其他帧仍通过普通 attention 可用。可以说附加读取定位到本窗 token，不能说附加向量只含本窗独立语义。
3. **对照能证明什么**：uniform-local-G 会改变 D、head gamma，并可能改变 active fraction f。因此它能消融整套显著性加权规则，不能单独分离“空间语义排序”与幅度效应。需记录 f、更新范数及 head 系数，并据实际差异限制解释；如要进一步主张语义内容而非幅度，须在观察结果前声明相应匹配控制。
4. **错位支持的可执行范围**：qualifying gain 后再固定 derangement 的具体对象、遍历顺序、可用供体及不同 token 数处理。只有实际改变 frame/value 支持的配对才算错位；无帧窗、单支持和无法错配视频应单列，不能把不变样本算作反事实干预证据。当前声明承认了这项边界。

## 成本与实验判定

外层调用数声明自洽：部署 `3+B`，native/guided 配对 `3+B+V`，原生前缀与 global / answer / speech 复用。成本增量包括图像位置 hidden states 的保存、全词表 FP32 输出投影和各层附加 value reduction；128 行分块限制临时 logits，不能消除投影矩阵或数据转换开销。需用五视频 smoke 实测替代 15–25 GPU 分钟估算，不能借用源论文的单图 TTFT 比例。

native SDPA 保留使原生/零干预精确对照成为合适检查；仍须验证 branch KV 裁剪、早期 query row 和 global/speech 的不变性，避免额外 guidance 泄漏到其他读取。原生无帧窗口回退以及末窗时间边界均已明确，无需在提案阶段新增处理规则。

均匀权重与错位支持控制具备可证伪性；是否通过两语料晋级及组件消融门由完整结果决定。显著性图、非零更新或少数案例本身不能解释涨点。**本轮 PASS 不要求预先相信该机制有效，也不因此前候选负结果而否决它。**
