# Stabilizer 独立 proposal review

日期：2026-10-03。结论：**PASS**。按 `RESEARCH_ITERATION_RULES.md` 第 4 条，未发现四类 STOP 的成立证据；放行实现不代表涨点、时序稳定性或机制贡献已经成立。

审查实例：`/root/m1_grounder_proposal_review`，与主会话相同模型。读取候选 `experiments/20261003_m1_stabilizer/README.md`、仓库规则、PAS 主来源/作者 demo、Qwen3 源码及目标任务文献。包含本次审查期间主 agent 增补的 all-.5、交换分组控制和目标安装版本说明。未修改方法或实现，未运行模型、训练或 GPU，未读取 GT、Highlighter 预测或性能，未使用内容哈希。

## 四项 STOP 判定

| 条件 | 本次结论 |
| --- | --- |
| 来源已用于 hateful video detection / localization | 实际检索及 primary 核查未发现 PAS 或同一按头分组的 temporal RoPE phase intervention 在目标任务中的使用。检索有范围限制，不宣称穷尽证明。 |
| 纯 ensemble | 否。同一 MLLM 一次前缀编码中改变不同 attention heads 的 Q，没有多个独立模型或多次主方法预测平均。 |
| 纯 calibration / 后处理 / 平滑 | 否。虽然来源名称含 Smoothing，这里改变的是注意力之前的 Q 相位和后续前缀表示，并非对最终窗口/帧分数做平滑。 |
| 纯工程技巧 | 否。完整迁移源码及论文定义的按头相位干预；不是只换位置编码配置或调一个性能常数。具体配对适配本身不是额外创新。 |

不以理论保证是否适用、效果可能很小或先前候选失败为 STOP 理由。

## 来源及公开代码核对

实际阅读 [PAS arXiv v1](https://arxiv.org/html/2511.10979v1) 的 §3、Algorithm 1、§4.1–4.3；[CVPR 2026 官方条目](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_PAS_A_Training-Free_Stabilizer_for_Temporal_Encoding_in_Video_LLMs_CVPR_2026_paper.html) 可核对作者和发表信息。

原文 Algorithm 1 在每层原生位置编码之后，只对 visual/video token 的 Q 施加 temporal phase operator，再进入普通 attention 和多头输出。§4.2 主设置为两个组、偏移 `[0, 0.5]`，因此提案采用非零中心分组有实际来源；不能把它写成围绕零点的正负对称偏移。原文默认时间单位是合并视频 token 的 bin，本候选没有继承这一物理解释。

实际读取 [作者 README](https://github.com/Bowen-Sun-0728/PAS/blob/main/README.md) 和 [inference_eval.py](https://github.com/Bowen-Sun-0728/PAS/blob/main/inference_eval.py)，并查看 `third_party/pas_source_read/` 的对应快照：

- README 明确公开版本是简化 demo，参数含义、运行行为和精确 benchmark 表现可能与论文内部实现不同；示例使用 `0,10`，并明确其尺度不等于论文的理论偏移。
- `inference_eval.py` 的 parser 默认 `0,0.5`、query-only、video scope；`_expand_to_heads` 按 head index 循环分配偏移。
- `_make_inv_full` 从 temporal section 长度重新构造频率，`_build_temporal_half_mask` 用连续 section，`_rotate_half` 用相邻偶奇通道。它们不能直接复制为 Qwen3 的实现。

提案已经把“遵循论文的相位算子”与“逐位复现 demo”分开。源数学结论依赖内容与频率条件；学习得到的不同 heads 经 softmax 和 `o_proj` 后不是同一标量 kernel 的凸平均。任意有限窗 DFT 对分数位移的幅值不变性也不能直接继承。保留这些限制，不以源定理替代本任务证据。

## 实际 Qwen3 布局和位置单位

独立读取本机 transformers 的 `modeling_qwen3_vl.py`，并实际读取 [HF 官方 v5.15.1 Qwen3 源码](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)。主 agent 另在声明中记录已读取运行机安装版本；本审稿没有远程运行模型。

`apply_interleaved_mrope` 先以 T 为基础，再把 H 的 `1:60:3`、W 的 `2:60:3` 覆盖进去。对 section `[24,20,20]` 和 half-dim 64，T 索引是 `{0,3,...,57,60,61,62,63}`，共 24 个。`rotate_half` 配对 `(j,j+64)`，不是 `(2j,2j+1)`。因此声明的 mask、native `inv_freq[j]` 和 split-half 旋转方向在静态上自洽。

`.5` 只能称原生 temporal **position** 单位。当前输入是多个独立 image block；单图内部 temporal 坐标为常数，跨图坐标还受前置文本和已分配 multimodal 坐标范围影响，并不等于采样帧号、半秒或半个物理帧。保持 timestamp 文本不变也不把这些位置变成秒。故可检验的机制是 position-phase intervention 对当前表示和读取的影响，不能预先称修复 ASR 对齐、真实时间 jitter 或满足 Nyquist。

实现时仍需实际张量 oracle：在同一输入 Q 上比较额外旋转与仅 temporal position 加偏移的原生 rotary 输出，核对真实 `inv_freq`、axis assignment、attention scaling 和正负方向；要在实际部署版本上做，不能只验证自己复制的 mask。非目标通道/文本行必须直接保留；零偏移路径须满足声明的原生精确恢复。FP32 旋转后的范数近似保持是代数检查，BF16 cast 和后续网络不能要求严格数学等式。

## 全局与两个窗口分支的一致性

本候选重新编码整个共享前缀，随后停止相位干预。每个 arm 都读取自己的 global margin，按既有规则选择并附加自己的 Yes/No，再用同一前缀读取 visual 和 speech windows，最后进入原有 r6。这个定义是一条完整、一致的方法，不是拼接两个候选中较好的分支。

“不直接修改 K/V 或文本 Q”不等于这些量相对 native 永远不变：更早层改变的 attention 会传播到更晚层的 K/V、文本状态及全局裁定。README 已正确披露。需防止实现中意外复用 native verdict 或 native speech cache，也需在窗口 suffix 时真正关闭干预。

global、verdict、两模态都可能改变，因此 pooled 增益可能来自整视频排序，不能独立支持定位机制。声明要求报告 within、原始 visual/speech/max 排序、global answer 翻转及单窗效应，符合这一解释边界。无需为了本轮 proposal review 再新增固定 global 的主方法。

## 对照是否足够

原先 all-head `.25` 仅匹配偏移均值：主方法 half-0/half-.5 的平均平方偏移是 `.125`，all-.25 是 `.0625`。实际 Q 扰动还依赖 head 的内容与频率。两者比较不能单独排除幅度差异或特定 odd heads 更适合受干预。

当前声明已新增 all-.5 和交换 0/.5 分组。all-.5 检查同一个非零偏移作用到全部 heads 的情况，但均值也不同；交换分组检查是否只由选中的 head 身份获益。它们是解释性控制，不是从中事后挑选主方法。这组控制与均值对照共同形成可证伪的证据；仍不能自动升级为普遍的“频谱平滑定理”或“时序鲁棒性”。

在运行 phase-jitter 控制前还须固定以下小范围细节：对每个待比较 arm 的每个 head **同加** `eta ∈ {-.25,0,+.25}`，仅作用于同样的前缀图像 Q；每个情形重新读取其 global、自己的 verdict 及双分支；预先定义 score-sensitivity 统计、按视频汇总方式及 answer-flip 统计。与各 arm 自己的 eta=0 比较，避免把不同中心偏移误作相同扰动。该诊断测量位置算子敏感性，不是帧重采样或现实 timestamp 噪声。

组件主张仍须通过仓库消融门。稳定性主张还须在相同扰动和输入下直接看到更低敏感性，且不能只靠常数化分数、饱和或丢失有效排序解释；这些由实验判定，不构成实施前否决。

## 成本

部署 `3+B`，native/phase 完整配对 `6+2B`，声明自洽。主方法没有新增媒体抽取或多个 phase forward。额外旋转为随图像 token 数、head 数、temporal dims、层数线性增加的操作，但实际成本还包括索引、复制、FP32 临时张量及 kernel 调度；不能把原论文 throughput 比例直接移用。

五视频 smoke 需实测 wall time / peak memory 并替代 20–25 GPU 分钟估算。每个新的完整 phase-control arm 是另一次 `3+B` 流程；非零 jitter 也各需重新前缀编码。不能把条件触发的这些研究成本混作主部署成本，也不能遗漏它们。

## 实际目标任务检索记录

本轮实际查询包含：来源完整标题和 `2511.10979` 与 hate / hateful；`Phase Aggregated Smoothing` 与 hate / hateful video；`HateMM` 与 RoPE；`HateClipSeg` 与 PAS / 来源编号；`hateful video detection/localization` 与 rotary / phase / positional encoding。未找到目标采用的 primary 证据；普通时序 Transformer 或文章中的一般 phase 字样不等于 PAS。

同时核查以下 primary 原文是否存在来源编号、完整方法名、rotary / RoPE，并比较目标方法类型：

- [LELA](https://arxiv.org/html/2602.09637v1)：训练无关的多模态描述、逐阶段 prompting 和组合匹配，不是 phase 干预。
- [MultiHateLoc](https://arxiv.org/html/2512.10408v1)：§3.2 的 modality-specific temporal Transformer 及后续跨模态学习，并未采用 PAS。
- [HVGuard](https://aclanthology.org/2025.emnlp-main.456.pdf)、[LEAF](https://aclanthology.org/2026.findings-acl.604.pdf)：rationale / self-grounding / distillation 路线。
- [SAGE](https://aclanthology.org/2026.acl-long.817.pdf)、[CLARA](https://arxiv.org/html/2608.15905v1)、[RAMF](https://arxiv.org/html/2512.02743v1)、[IARE](https://arxiv.org/html/2606.11953v1)：专家、clip 表示或 reasoning 融合，未见 PAS 或同一按头 temporal-phase 算子。

本轮 **PASS** 的边界是：完整来源机制可以在目标任务中实施检验。不能先宣称解决了真实时间对齐，也不能依据 norm preservation 或借来的理论承诺定位涨点。
