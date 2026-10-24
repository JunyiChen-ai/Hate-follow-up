# 规则 4 proposal review：候选 49（互不可见的逐段存在判断估份额）

日期 2026-10-25。**VERDICT: PASS**

审查由一名独立 agent 做，只审一次。读过：规则、`CLAUDE.md`、README §1–§11、三份以往审查、`DIRECTIONS.md`、CVA README、`src/prefix_questions.py` 的分段函数。没有改仓库，没有用 GPU，没有算哈希。

## 检索（18 次）

检索词：
- 仇恨视频逐段投票、逐段 MLLM 判断占比；
- 有害视频逐 clip 判断再聚合；长视频审核分块；
- HateMM 上 clip 平均或 MIL；MultiHateLoc 的聚合；零样本仇恨视频分段提问；
- 无训练 VAD 里异常片段的比例；
- group testing 用于审核、视频、LLM；Thompson 估计量用于分类器输出；
- 长视频 QA 分块投票；视频安全的逐 clip 不安全比例。

**打开过全文**：SafeLens（AAAI-26 demo）、MultiHateLoc 2512.10408、CLARA 2608.15905、TikTok 审计 2608.17583、ACL 2026 视频假新闻基准（相关段落）。

**只看了摘要**：Neural Group Testing 2011.10704、Probe-VAD、EventVAD、CoReVAD、LAVAD、2510.05864、LifelongMemory、2503.20472、EgoSafetyBench。

**结果**：
- **SafeLens 最接近**。它按句子或场景切段，每段由在 HateClipSeg 上 LoRA 微调的 Llama-3-8B 判断，视频摘要报告仇恨事件数。但它是有监督的；段判断就是定位输出；事件数不当份额用，不参与视频排序，也不和整段裁定组合。
- MultiHateLoc 是用视频级标签训练的 MIL。CLARA 有监督。TikTok 审计每个视频只调用一次。
- 无训练 VAD 的段分数直接作帧分数。
- Neural Group Testing 用在图像审核上，目的是节省计算，不估计单个条目内的份额。

**结论**：没有找到仇恨视频工作用互不可见的逐段存在判断来估计份额并组成视频键。

## 四种 STOP

1. **来源方法已用于仇恨视频**：不成立，见检索。
2. **纯 ensemble**：不成立。
   - 只有一个冻结的 MLLM，问同一个问题。
   - z_video 读整段，z_j 读互不重叠的四段，输入不同，估计的量也不同（"有没有"对"占多少"）。
   - 这不是"多个独立模型对同一信号的判断"，规则 3 明文允许。
   - q_v 的结构接近多 clip 测试取平均。它是否只是把裁定再读一遍，交给 P3 判断。
3. **纯校准 / 后处理**：不成立。
   - q_v 和 p̂ 来自新读数。
   - group testing 的反演作用在新读数上，属于估计量本身。
   - s 和两高斯校准沿用 r6，不作主张。
4. **纯工程技巧（换窗长）**：不成立。
   - 如果保留整段前缀、只把 8 秒窗换成 D/4，才属于这一条。
   - 本候选改的是输入：每段单独建前缀，不接立场轮，读数只进视频键。
   - 离它最近的旧读数是 winonly（8 秒窗、无上下文、无帧）。区别是段长按视频比例、每段带自己的 5 帧、问的是存在、读数只进键。

## 运行前建议（只是建议）

1. C1 不用 group testing。C1 当选时：
   - 主张写成"互不可见的存在读数作为键的量项"；
   - Dorfman / Thompson 只作 C2 的来源；
   - 引用 SafeLens 和多 clip 测试平均（TSN / SlowFast，本次未检索）作为先例。
2. 加一个 CPU 诊断臂：K = z_video + s·mean(winonly z_w)，用来分开"互不可见"和"段长"两个因素。winonly 用的是旧 ASR 缓存，报数时注明。
3. 另报 ρ(q_v, 窗均值)。
4. C2 跨视频的差别主要来自 m̄，而 m̄ 随时长变化。另报 ρ(logit p̂, 时长) 和 ρ(q_v, 时长)。
5. 候选 47 把窗均值称为 classify-and-count。C1 / C2 也是 classify-and-count，区别只在于各段互不可见、单元更粗。论文措辞要和这一点一致。
