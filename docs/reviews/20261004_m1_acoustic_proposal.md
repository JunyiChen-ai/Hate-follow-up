# Candidate20 acoustic：独立方案审查

2026-10-04。**PASS（same-family provisional，独立 agent）**。

按 `RESEARCH_ITERATION_RULES.md` 第4条，本次实际检索未发现完整来源方法或所提组合已经用于 hateful video detection / localization；其它三类 STOP 也不成立。可进入实现与既定独立代码审查。这个结论不表示时间估计准确、方法有效或机制成立；这些由完整333与事前消融裁定。本审查未运行 GPU、未读 GT/预测，未实施算法、未改父 agent README、常数或评测器。

## 阅读与检索范围

完整读取当前 `CLAUDE.md`、`RESEARCH_ITERATION_RULES.md`、`experiments/20261004_m1_acoustic/README.md`，以及 `runs/20261004_m1_ideation/backup_source_reads/timing.py`、`retokenize.py`；读取 STATUS 当前候选19归档与候选20入口。未重审历史结果或其 metrics。

实际在线读 [Whisper Has an Internal Word Aligner §II-A/B/C、IV-C](https://arxiv.org/html/2509.09987v1) 和 [官方仓库使用代码](https://github.com/30stomercury/whisper-char-alignment)，并核对下表近邻。原始检索请求/响应保存在 `runs/20261004_m1_acoustic/proposal_review/web_request_01.json` 至 `05.json` 和对应 `web_response_*.json`；完整查询清单亦在这些请求中。

| 实际查询或近邻 | 检索/核验结论 |
|---|---|
| `"Whisper Has an Internal Word Aligner" hate`；`"hate" "whisper-char-alignment"` | 找到原论文、作者/官方资料，未发现目标仇恨视频上的该方法应用。 |
| `"hateful video" "alignment" uncertainty`；`"hateful video" "alignment uncertainty"` | 返回多模态对齐、时间错位和 uncertainty 相关工作，未找到本方案路径分布条件化读数。 |
| `"hateful video" "Gibbs" OR "DTW" OR "forward-backward"`；`"hateful video" "dynamic time warping"` | 未发现同类路径边缘分布进入冻结语义 reader 的方法；DTW 命中另见下方 MultiHateGNN 核验。 |
| `"HateClipSeg" "word" alignment`；`"hateful video" "word alignment" Whisper` | 返回目标任务/Whisper 近邻，没有找到本轮完整机制。 |
| [MultiHateLoc §3.1](https://arxiv.org/html/2512.10408v1) | 使用 Whisper 语句起止时间、BERT 语句特征及区间内重复填充，并以视频级标签监督。已有时间对齐，不等于无监督字符路径分布或 Qwen attention 支持条件化。 |
| [CLARA §3.1–3.2](https://arxiv.org/html/2608.15905v1) | 使用 Whisper-large-v3 语句时间边界切 clips，后续模态特征融合/监督分类；没有本轮声学路径边缘量干预 reader。 |
| [UCA，ACL Anthology 原始摘要](https://aclanthology.org/2024.lrec-main.1475/) | image/text memes 的跨模态分布不确定性与融合，不是视频词时间路径不确定性；这里只核对摘要所定义的问题与方法，不声称全文审计。 |
| [MultiHateGNN 原论文 §3 与参考文献18](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_408/paper.pdf) | “dynamic time warping” 命中的是 MFCC 引用文献标题；方法正文该处用 MFCC 音频特征。不能把搜索词命中当成本轮完整方法已用。 |

这些检索支持“本次未发现既有使用”，不支持绝对的全球首次声明。不能泛称目标任务从未使用 Whisper、时间对齐或不确定性。

## 四种 STOP 逐项裁定

1. **来源已用于目标任务：未成立。** 目前检索到的目标方法均不等于字符教师强制/动态筛头、Gibbs 单调路径分布、逐词支持进入冻结 reader 的完整链路。
2. **纯 ensemble：未成立。** Whisper 产生声学支持而非第二份仇恨 prediction，Qwen 为唯一语义 reader；多头来自同一个既有模型。对齐路径边缘化没有对多个模型或路径的仇恨预测取平均。
3. **纯 calibration / 后处理：未成立。** 干预进入 speech margin 生成前的全部 branch attention；最终 max 与 r6 原样保留。概率归一化属于明确的中间声学模型，不是拟合仇恨分数或按标签调阈值。
4. **纯工程：未成立。** 本方案不仅重切文本/换时间戳，还定义单调路径能量、前向后向边缘量，以及由这些量驱动的单次语义读数。若实施只留下 MAP body 与普通 Qwen、没有真实 soft prior 进入 attention，就不属于本次获 PASS 的方案；单独时间戳替换不构成其 novelty。

## 来源忠实性与须纠正的窄问题

论文方法包含字符教师强制、每 utterance 基于行列 L2 范数筛头及 DTW；Gibbs 路径模型和 Qwen 条件化是本轮扩展，不是该论文已有结果。README 已明确论文先平均再列归一化与官方先逐头归一化再平均的差异。[论文](https://arxiv.org/html/2509.09987v1)

**一项具体来源措辞须纠正：** 官方 `timing.py::force_align` 在完整 token rows 上筛头/归一化，随后 `matrix[len(tokenizer.sot_sequence):-1]`，因此保留 no_timestamps 行及字符输入行。当前方案只在字符行上评分/列归一化，并仅用字符自身输入行形成路径；这是一项明确适配，不能称与官方行范围完全相同。第一字符前驱行与自身行的区分也不能简单写成前者是“错误”——论文自回归公式以预测 token 为索引，而发布代码的边界行处理另有约定。保持当前已声明自身输入字符行即可，在 README 解释其实际范围；不要求因此改常数或重开方案选择。[官方代码入口](https://github.com/30stomercury/whisper-char-alignment/blob/main/timing.py)

其余边界已声明充分：HF large-v3、首块语言识别、长片分块、保留原词 ID、不展开数字、ASCII 标点处理、valid audio 帧范围、median3、top10 和列 norm 下限均是适配约定。不能把原论文在其它模型/语料的 word-boundary 准确率转移为本轮的保证。不存在要求引入第二个 aligner 或改用监督头选择的缺口。

## 实现时必须保持的算法语义

以下是现有定义的核验要点，不是新增性能门或理论 STOP。

- 路径权重按全部访问 cell 的 C 之和及每次转移 `-log(3)` 定义。若 alpha 含当前 cell，则 beta 应排除当前 cell；否则 `alpha+beta-logZ` 会重复计分。起终点处理、K=1/T=1 和 tiny lattice 枚举可直接验证 DP 与显式路径求和一致。
- 按行归一化 occupancy 是本方案定义的 token-time 支持，不自动等于开始/结束时间后验或“先逐路径归一化再平均”。Viterbi 对照必须使用同一 C 和相同转移势；不能无意退回原始无转移惩罚 DTW。
- 原始词 ID、每字符可能多个 tokenizer tokens、被删除标点、切片合回权重必须可追踪；特别是一个字符的多 token 跨切片时不能重复按整字符计权。每词最终窗质量应有限、非负、总和1；不能因尾帧/重复词而丢失或重复质量。
- Whisper QK 抽取不得把 SDPA 没返回 attention 误当零图；仅 valid 帧做 softmax。HF 分词不能自动重复添加语言/任务控制 tokens，语言识别只在语言 token 集合内贪心。score/norm 使用的字符行范围要与记录一致。
- Qwen prior 必须加在真实可见 key 的 logits、softmax 之前，保持 causal mask，且作用于全部 branch query rows。完整前缀 ASR 与 body 重复词分别通过 occurrence/word ID 映射，不能靠第一次文本匹配；展开图像 token 后偏移必须实际核对。
- 缓存的 contextual 表征仍可能携带全局信息，已在 README 承认；这不使方案无效，但只能声称支持条件化读取，不能声称信息完全隔离。原 global/visual 不重算或污染；branch cache/rope delta 与 attention 实现恢复必须验证。
- 原生精确配对使用原生 body 和原生 attention；`unweighted` 保持新的 MAP body，所以不能把它的全1 prior 误当完整 native parity。无 MAP 但非零声学支持窗仍读一次的规则必须真正实现。

## 机制与代价审查

声明的 `hard`、`proportional`、`unweighted`、`shifted` 足以分别质疑软传播、真实声学支持、attention 条件化及时间匹配解释。full 对 native/proportional 的同主指标双语料提升，以及 soft 对 hard、weighted 对 unweighted 的必要性门，必须按 README 的方向解释。若只因 body 更准而涨点而 prior 无必要贡献，就不能以整个路径分布条件化主张 novelty；删除/降级后按既有规则重新评价完整方法。错误时间绑定控制会同时改变 body 和 prior，不能声称独立隔离 attention 的纯因果作用。

代价来源明确：每声学块一次 encoder/teacher-forced decoder，加一次短语言 decoder/video；每有支持窗单次 Qwen speech forward，无路径数量倍增的语义查询。复用原音频、ASR/模型权重不等于新视频免费。30–90分钟只是尚未测量估计，应在 smoke 记录字符块数、重叠音频秒数、QK/DP 时间和峰值内存后更新；不能隐去重叠块和真实 attention 获取的成本。顺序释放 Whisper 再载 Qwen符合预算思路。性能及必要性仍待完整两语料实测，审查不以潜在 shortcut、注意力不够尖锐或可能无涨点为 STOP。

归档前确认：父 agent 已在未实施前修正 README，明确官方特殊行评分/no_timestamps 保留与当前字符行适配的区别；没有改变本轮常数。因此该来源表述问题已解决，不是待批准条件。

本次正式结论为 PASS；可以实施并进入一次独立 code review。
