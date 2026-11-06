# 范式定位与顶会写作先例（2026-10-24）

活文档。目的：(1) 把当前方法 r6_bma 的 novelty 落到一个核心挑战和它的子挑战上，每个模块对应一个子挑战；(2) 收集顶会里"整体 claim novel、单个模块并不 novel"的工作，看它们怎么写、怎么处理简单模块。数字是转录，权威数字以各行来源里 `runs/` 的原始输出为准；所有数字为 development-selected。

## 1. 核心挑战

**一句话**：仇恨标注的单位是行为，模型能读的单位是时刻。
Hate is annotated as an act; a model can only read a moment.

**标签定义句**：帧标签回答的是"这一刻仇恨行为是否在进行"，不是"这一刻说的这句话是否仇恨"。逐句打分回答的是后一个问题，任务问的是前一个。
Frames are labelled by whether the hateful act is ongoing at that moment, not by whether the sentence spoken at that moment is hateful on its own.

**支撑**（详见本文件 §4 文献）：HateMM 让标注者画"他们觉得是仇恨的 frame span"，rationale 均长 1.71 分钟（视频均长 2.56 分钟）；MultiHateClip 让标"仇恨陈述出现的那一段的起止"（英文均 33 s / 37 s）；HateClipSeg 原文 "Offensive content is frequently intertwined with benign segments" 与 "short clips often lack broader context needed to disambiguate subtle or implicit hate speech"。三套 test GT 上约 90% 的仇恨时长落在 ≥16 s 的连续段内（HateMM .91 / HCS .91 / DeHate .89，本机 4 fps 数组计算，2026-10-18）。

**Case（Figure 1 候选）**：HateClipSeg `yt_yuzFCQoKuPQ`，308 s 恶作剧电话，gold [8,153]、[163,308]。37 个仇恨窗只有 12 个点名群体；单窗读 10/37 为负，带整段读 1/37。16 s "You got Arabian dust in your ears?" 单读 +5.5 / 带整段 +17.0；56 s 受害者回话单读 −6.2 / 带整段 −3.8（第二类，读不出，靠段结构）。来源 `runs/20260910_spvl/mllm/q3vl-8b/{winonly,full}`、`runs/20261002_verdict_analysis/analysis/per_window.csv`（旧 ASR 缓存）。

段内有两类"看起来不像仇恨"的窗，写作时必须分开：
- 第一类：单看不像，放在行为里就是攻击（对象在前面说过）。上下文能救。对应 M1。
- 第二类：放在行为里也不是攻击（受害者回话、沉默、离题句），但行为没停。上下文救不了，靠段结构。对应 M2。gold 正、带上下文仍读负的窗占比：HateMM 14%、HCS 32%、DeHate 43%（混有第一类未救回的读错）。

## 2. 子挑战 → 模块 → 消融

| 行为的性质 | 子挑战 | 模块 | 消融（HateMM / HCS，within 除非注明） | 来源 |
|---|---|---|---|---|
| 一个时刻是行为的一步 | 读任一时刻时要知道它属于哪个行为（谁攻击谁、立场） | M1：整段转录 + 20 帧 + 模型裁定作共享前缀，每窗隔离分支 | 单窗读 −.021 / −.071；pooled ROC −.035 / −.080 | `docs/reviews/20260927_final_novelty_review.md` §4 |
| 行为按时间持续，不按句子 | 读数单位应是时间，不是 ASR 句 | 固定 8 s 网格 | +.089 / +.052 | `experiments/20260910_spvl/README.md` |
| "在进行"的分数水平每个视频不同 | 无标签地定每个视频的两类水平 | M2：语料秩归一 + 每视频两类 EM | 去掉时间耦合整体 −.114 / −.057 | `experiments/20260926_twolevel/README.md` §20.1 |
| 行为有起止和最短长度 | 无标签推出段的范围 | M2：显式时长链，最短两窗，长度先验来自该视频 | 对最优宽度高斯平滑 +.063 / +.006 / −.004（DeHate）；该宽度用 test 标签挑 | `runs/20260926_twolevel/final_gauss{2,4,8,16}` |
| 行为可以换通道 | 通道切换不应被当成证据下降 | M2：每通道一条链，任一在违规态即在进行 | 并成一条链 −.007 / −.043 | 同上 |
| 行为占多长决定视频排序 | 视频级裁定只说有无，不说多少 | M3：视频键 = 裁定 + 窗均值 | 去掉视频键 pooled ROC −.325 / −.147；占比与裁定 Spearman .19 / .32，与窗均值 .42 / .53 | `runs/20260910_spvl/ablation_table.md` |

**结构建议**：一个核心挑战、两个方向。M1 把行为带进时刻的读数（act-conditioned read），M2 从时刻读数里恢复行为的范围（act extent inference）。不写"M1 提供 preliminary score"。Intro 只列前两个子挑战（直接从 case 看得见），通道切换与排序放方法节。

**可以 claim 的**：问题形式的改变。现有无标签方法给时刻打分再后处理（阈值、固定宽度平滑）；我们把定位当成对一个持续行为的推断。对立词：act inference 对 moment scoring。

**不 claim 的**：Yes/No log-odds 读出（Probe-VAD、CEAVAD 已有）；前缀缓存与分支隔离（系统先例 Hydragen、Prompt Cache）；"先视频级再局部"（T3AL 已有，但 T3AL 的视频级只选类别，之后仍是时刻打分 + 视频均值阈值 + 移动平均）；"training-free MLLM 定位"（LELA、LAVAD 已有）。

**最薄的点**：时长链对最优宽度平滑在 HCS、DeHate 只持平；通道链只在 HCS 成立。两者合起来两语料都有支撑，分开各缺一边。这是 M2 唯一值得小幅 tailored 修改的位置。

## 3. 顶会先例：整体 novel、模块不 novel 的写法

选取标准：顶会、近三年、任务相近（training-free / zero-shot 的视频定位或异常检测、冻结 VLM 的仇恨检测）、方法由多个简单模块组成。

| 工作 | 整体 claim 的落点（原话） | 简单模块与它被赋予的"问题" | 简单模块在消融里 | 自认简单的话 |
|---|---|---|---|---|
| LAVAD，CVPR 2024 | "the first training-free method for VAD"；"a novel, training-free paradigm" | 字幕清洗 = 余弦相似 argmax，问题 "raw captions can be noisy"；时间摘要 = 给 LLM 全局上下文，问题 "frame-level caption lacks details about the global context"；分数精炼 = K=10 近邻 softmax 加权，问题 "without taking into account the whole set of scores" | 三个模块各一行，−3.8 / −7.58 / −7.49 AUC | 无；自述 "all five elements involve only off-the-shelf frozen models" |
| T3AL，CVPR 2024 | "the first method that tackles ZS-TAL without training data"；"from a radically novel perspective" | 视频级伪标签 = 平均特征 argmax，问题 "noise coming from non-informative frames"；阈值 = 视频均值，"Instead of using a fixed threshold"；移动平均 "to further enhance temporal consistency" | 伪标签、均值阈值、平滑都没有单独消融 | "a simple use of the video-level pseudo-labelling" |
| AnomalyRuler，ECCV 2024 | "the first reasoning approach for the one-class VAD task"；"a novel rule-based reasoning framework" | 感知平滑 = EMA + 窗内多数投票，命名为 "a novel mechanism named Exponential Majority Smoothing"，问题 "movements are continuous and should exhibit consistent patterns over time" | 平滑作为一个整体消融一行 | "align with the spirit of CoT yet are further refined for the VAD task" |
| TFVTG，ECCV 2024 | 不用 novel；写法是 "A naive baseline is to enumerate proposals"，再给两个失败："distinguish the temporal boundaries of multiple events"、"sensitive to the dynamic transition of events"，每个失败对应一个模块 | 静态打分 "Inspired by SPL"；装配 = 枚举 k^m 组合过滤 | 分表消融每个模块，只在 Charades-STA | 对最近工作："Luo et al. and VTG-GPT made the first attempt ... overlooked the order and relationship" |
| VERA，CVPR 2025 | "the first approach to adapt frozen VLMs as an integrated system for VAD"；"a new learning paradigm for VAD" | 场景上下文集成 = ImageBind top-K softmax 加权，问题 "only examine a short moment in a long video without considering any context"，是全文最大增益 +8.43；高斯平滑引教科书，+0.95；位置权重 +1.07 | 累加式四行 | 分段方式 "Following [52]"（LAVAD） |
| VADTree，NeurIPS 2025 | 从不说 first；"Unlike existing temporal window-based approaches, VADTree adaptively organizes video content"；称 LAVAD 为 "pioneering" | 树 = 在最高置信边界递归切，问题 "inherent limitations of uniform sampling for arbitrary-length anomalies"；双簇 "the classic two cluster ... strategy"；打分 "Following LAVAD" | 累加式 71.57 → 75.67 → 83.05 → 84.74 | "classic"、"Following"、"Inspired by" |
| Zero-shot STVG，NeurIPS 2025 | 章节标题 "A straightforward solution for zero-shot STVG"；"Though simple, this solution has achieved comparable or even superior performance" | 子查询分解 = LLM 抽属性/动作 + 固定模板，问题 "grounding tokens often neglect some important cues" | 累加式 | 见左 |
| SeViLA，NeurIPS 2023 | "a novel video-language framework"：同一个冻结 BLIP-2 用两次（定位器、回答器），不同提示 | 定位器的问题 "uniform frame sampling can often lead to missing important visual cues"；理由是省标注和显存 "adding Localizer has a very small additional memory footprint" | 对均匀采样一行 | 承认 "has no temporal modeling/training" |
| Pro-Cap，ACM MM 2023（仇恨表情包） | "the first work to leverage PVLMs in a zero-shot manner" | 探针问题 = 手写问题问冻结 BLIP-2，问题 "generic descriptions of images may omit crucial details"；拼接 + BERT | "No Centric" 一行 | "we heuristically use answers to all probing questions" |

### 规律

1. 整体 novelty 钉在**设定或问题形式**上，不钉在技术上："first training-free"、"first reasoning approach"、"first to adapt frozen VLMs as an integrated system"、"unlike window-based approaches"。技术层面的 first 一个都没有。
2. 每个简单模块都被写成**对一个有名字的失败的回答**，前面一句话点出失败，后面一句话给模块。TFVTG 最干净：naive baseline → 两个失败 → 两个模块。
3. 简单模块**起名字**，名字描述它解决的问题，不描述技术："Image-Text Caption Cleaning"、"Perception Smoothing"、"Scene-context ensembling"、"Intra-cluster refinement"。EMA 加多数投票可以叫 "Exponential Majority Smoothing" 并称 novel mechanism。
4. 借来的部分**点名致谢**："Following LAVAD"、"Inspired by SPL"、"the classic two-cluster strategy"、"align with the spirit of CoT"。承认借用没有影响接收。
5. 消融里简单模块**各占一行**，增益小也照报（VERA 平滑 +0.95，无方差）。审稿人要的是每个模块有一行，不是每行都大。
6. 与最近工作的区分写在**设定、成本或范围**上（"requires further training"、"training set inaccessible"、"video-level annotation data"、"spatial not temporal"），不写在机制上。对更早的 training-free 尝试用 "made the first attempt ... overlooked X" 句式。
7. 同一个冻结模型用两次、靠提示分工（SeViLA）可以作为 framework 的核心卖点，理由是省标注、省显存、对提示不敏感。

## 4. 对我们论文的映射

- **整体 claim**：形式改变 + 设定。候选句："Existing label-free localizers score moments and post-process the scores; we cast localization as inference of a persistent act." 对 LELA 用 TFVTG 句式："LELA made the first attempt at training-free hate localisation by scoring each frame independently from its captions; it overlooks that the hateful act which a frame belongs to is defined elsewhere in the video and runs for minutes."
- **M1 的写法**：按 LAVAD 时间摘要、VERA 场景上下文的先例，作为对"时刻不知道自己属于哪个行为"这个失败的回答，起名 act-conditioned read（或 act-conditioned window read）。前缀缓存与分支隔离在实现段点名系统先例。消融一行：单窗读 −.021 / −.071。SeViLA 先例支持"同一模型先裁定再读窗"作为框架的一部分来写。
- **M2 的写法**：按 VADTree 对 LAVAD 固定窗、TFVTG 对 naive baseline 的先例，对立面是"固定宽度平滑假设持续长度已知"，我们的段模型在每个视频内无标签推出起止和最短长度。消融两行：对各窗独立、对最优宽度平滑（说明宽度用 test 标签挑）。
- **M3 的写法**：对 T3AL 的区分："T3AL's video-level pseudo-label only selects the class; ours enters every window's read and, together with the mean window score, orders videos by how much of them the act covers."
- **自认简单的话可以写**："The components are deliberately simple; each answers one property of the act, and each is ablated."
- **contribution 三条草案**：(1) 把无标签仇恨视频定位重新表述为对持续行为的推断，并说明为什么逐时刻打分从定义上答错了问题；(2) 一个两级方法：行为条件下的 MLLM 读数 + 每视频无标签的段推断（时长链、按通道、OR）；(3) 三套语料上超过 training-free 对照，每个部件有消融，并给出 Figure 1 的 case。

## 5. 文献（本节只列出处，供 related work 与 Intro 引用）

- 行为 / 语境：Langton 1993、2012；McGowan 2009、2019（conversational exercitives）；Rabat Plan of Action（OHCHR 2012，六要素第一条 context）。
- 对象只说一次：HateCheck F12/F13 "Hate expressed through reference in subsequent clauses / sentences"（Röttger et al. ACL 2021）；CAD（Vidgen et al. NAACL 2021）"for understanding who a generic pronoun referred to"，25–32% 需上文；Latent Hatred（ElSherief et al. EMNLP 2021）隐式 4,909 对显式 933。
- 单句误判：Yu et al. NAACL 2022（38.3% 判断随上文改变）；Menini et al. 2021（abusive 18% → 10%）；Pavlopoulos et al. ACL 2020（约 5%，拼接上下文没帮到分类器）。
- 视频标注单位：HateMM（ICWSM 2023）；MultiHateClip（ACM MM 2024）；HateClipSeg（arXiv 2508.01712）；DeHate（ACM MM 2025）。
- 先例论文：LAVAD arXiv 2404.01014；T3AL 2404.05426；AnomalyRuler 2407.10299；TFVTG 2408.16219；VERA 2412.01095；VADTree 2510.22693；zero-shot STVG 2509.15178；SeViLA 2305.06988；Pro-Cap 2308.08088；LELA 2602.09637（arXiv，未见 venue）。
