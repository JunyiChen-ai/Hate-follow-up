# Verdict-conditioned context: paired case analysis (2026-10-02)

Status: complete. Host: uoa-lab1 / sc474397 (recorded in run.log).
User request: determine which cases benefit from appending the model's own global
Yes/No judgement to local-query context, and why. No method selection or paper edit.

## Declared scope, inputs and controls

- Cached paired Reader outputs for HateMM/HateClipSeg:
  `runs/20260910_spvl/mllm/q3vl-8b/{full,nostance}/predictions.jsonl`.
  Both have 333 valid videos, identical video/window coverage and exactly identical
  global log odds. These use the older ASR loader; they are NOT the latest main-table
  Reader family. Both are passed through the current r6_bma inference code with
  identical arguments, independent label-free corpus fits.
- DeHate Reader pair:
  `runs/20260927_dehate_external/reads_gridA/predictions.jsonl` and
  `runs/20260928_infer/dehate/reads_nostance/predictions.jsonl`.
  Both have 1341 valid videos, identical grids and global log odds. Reuse matched
  r6 outputs in `runs/20260926_twolevel/final_dehate/final_m2/` and
  `runs/20260928_infer/dehate/abl_nostance/` (1151 GT-overlapping videos).
- Treatment is the full global question/answer dialogue turn, not just one answer
  token. Both conditions already receive the same full video frames/transcript.
- Read `data/gt_4fps/{HateMM,HateClipSeg,DeHate}.npz`, test only, for evaluation and
  case analysis AFTER inference. No labels enter fits or score construction.
  All findings are exploratory, development-selected evidence.
- HCS primary GT includes several offensive categories, not only hateful content.
- No new MLLM calls. Main paired CPU inference expected a few minutes. DeHate
  inference already exists. No parameter tuning, training or threshold selection.

## Planned diagnostics (declared before detailed outcome inspection)

1. Paired per-video within AUC and bootstrap of its mean difference; global
   Yes/No correctness and positive-frame coverage strata. Report gains and losses.
2. Compare raw visual/speech/max window read ordering with final ordering; quantify
   common score shifts versus content-dependent changes. A constant per-video
   offset alone cannot improve raw within-video AUC, but can affect corpus fitting
   and temporal inference. No semantic explanation inferred from an offset alone.
3. Inspect deterministic extreme gain/loss cases plus global-No failures using
   actual transcripts (old loader for main pair) and cached sampled frames. Case
   selection is explicitly illustrative, not a prevalence estimate.
4. Optional diagnostic only if needed: preserve each no-verdict modality's window
   ordering but add that video's observed mean treatment shift; this uses paired
   predictions, not labels, and is a reconstruction rather than deployable method.

Use only shared `src/eval/evaluate.py` metric functions and the official evaluator
CLI. Save per-video/per-window tables, alignment checks, summary and case evidence
in `runs/20261002_verdict_analysis/`. Updates to STATUS will link this document.

## Reproduction

`bash experiments/20261002_verdict_analysis/launch/run_main_pair.sh`

Then, using the HateVideo Python executable:

```
bash experiments/20261002_verdict_analysis/launch/run_shift_control.sh
python experiments/20261002_verdict_analysis/analyze.py
python experiments/20261002_verdict_analysis/make_cases.py
python experiments/20261002_verdict_analysis/verify_and_compare.py
```

Code provenance: current local sources, 2026-10-02; r6 runner numerical logic
unchanged, forbidden Git identifier logging replaced with readable provenance.

## 结论：能找到获益案例，但当前证据不支持“稳定有效”的前提

两组主语料缓存都重新经过当前 r6_bma；DeHate 复用已有同版本配对结果。
主语料有裁定组精确复现之前同缓存的 r6 输出（333 条曲线最大绝对差为 0）。
共检查 1484 个有 GT 的视频，405 个有正负帧的视频可计算 within AUC。
HCS 有一个 GT 视频 `yt_NzvfkIYS5Yg` 在两组中共同缺失；其余覆盖完整。
此次约 6 分钟 CPU 推理，新增 MLLM 调用为 0。

### 1. 全量结果，而不是只挑改善案例

以下三元组均为 pooled ROC / pooled PR / within AUC，统一 4 fps test。
差值方向都是 **有全局问答轮 − 无全局问答轮**。

| 数据集 | 无全局问答轮 | 有全局问答轮 | within 差值及 95% 配对 bootstrap CI（百分点） | 改善 / 退化 / 不变视频数 |
|---|---|---|---|---|
| HateMM | .896761 / .696719 / .765482 | .894242 / .691086 / .763920 | −0.16 [−2.61, +1.70] | 39 / 25 / 20 |
| HateClipSeg | .705523 / .658747 / .631225 | .715843 / .671566 / .637100 | +0.59 [−1.09, +2.32] | 47 / 50 / 2 |
| DeHate（external） | .699898 / .161872 / .633036 | .701149 / .158160 / .643068 | +1.00 [−0.83, +2.87] | 86 / 82 / 54 |

权威指标文件：

- HateMM/HCS：[有裁定](../../runs/20261002_verdict_analysis/main_full/metrics.json)、[无裁定](../../runs/20261002_verdict_analysis/main_nostance/metrics.json)。
- DeHate：[有裁定](../../runs/20260926_twolevel/final_dehate/final_m2/metrics.json)、[无裁定](../../runs/20260928_infer/dehate/abl_nostance/metrics.json)。
- 配对统计：[comparisons.json](../../runs/20261002_verdict_analysis/analysis/comparisons.json)。CI 为 10000 次按视频配对抽样；单次确定性读数，没有独立 MLLM 重复运行。

**更正此前口头结论**：旧 r3 Decoder 的正向均值不能直接当作当前 r6 的收益。
当前这组匹配缓存上，HateMM 的方向已变为轻微负向；三个语料 CI 都含 0。
本轮主语料仍是旧 ASR loader 的匹配 Reader 对，不能替换最新 Reader 的论文主表。

### 2. 哪些实际案例受益，以及变化在哪里

案例按最终差值极端值选取，再补充最大原始读数收益案例。
检查了转录、20 张缓存帧、逐窗双分支读数和最终曲线。
下面是具体机制现象的例子，**不是这些语义类别普遍受益的频率估计**。

**A. 开场闲聊与后续明确辱骂的区分：HateMM `hate_video_299`。**
约 40 秒视频聊天，GT 正区间 21–39 秒。前面是寒暄，后面出现针对黑人的直接种族辱骂。
全局裁定 Yes。加入这轮后，8–16 秒闲聊的 speech log odds 从 −1.3 降至 −10.3；
16–24 秒包含辱骂的窗口从 16.0 升至 20.3。
最终 within AUC 从 **.755 升至 .935**。
但 32–40 秒也被压低，原始 max 读数 AUC 反而 .885 → .785。
所以准确说法是：部分明确证据与闲聊的分数间隔扩大，经时序处理后改善最终定位；
不能把它概括为所有局部排序都变好。
[曲线](../../runs/20261002_verdict_analysis/analysis/figures/HateMM_hate_video_299_scores.png)
／[采样帧](../../runs/20261002_verdict_analysis/analysis/figures/HateMM_hate_video_299_frames.jpg)。

**B. 仇恨主要由语音承载时，双分支的相对响应改变：HCS `bit_O1kfCUve96Vz`。**
视频在播客谈话与网页截图之间切换，转录包含针对宗教和族群的贬损。
全局 Yes 后，开头 0–8 秒 speech 11.4 → 15.3，8–16 秒 4.6 → 9.6；
多个 visual 窗口反而降低，例如 24–32 秒 −0.9 → −5.1。
最终 AUC **.595 → .843**，原始 max AUC 仅 .727 → .756。
观察支持“改变两路证据强弱及其下游解释”；并未证明模型新增了跨模态推理能力。
例如 72–88 秒非 GT 正区间的语音分也升高，局部误报仍在。
[曲线](../../runs/20261002_verdict_analysis/analysis/figures/HateClipSeg_bit_O1kfCUve96Vz_scores.png)
／[采样帧](../../runs/20261002_verdict_analysis/analysis/figures/HateClipSeg_bit_O1kfCUve96Vz_frames.jpg)。

**C. ASR 几乎无有效内容，画面文字读数获益，但不能包装成理解仇恨意图：HCS `bit_Jl81Ars8alKV`。**
这是关于南非土地政策、白人农民及美国政策的新闻剪辑，主要信息出现在画面字幕；
旧 ASR 只有零散“Thank you”等。全局 Yes 后，24–32 秒 visual −2.8 → 2.4，
64–72 秒 −6.7 → 1.1，GT 标注 23–122.75 秒。
最终 AUC **.508 → .910**，是 HCS 最大改善。
但从采样帧可见大量新闻转述，HCS 主 GT 又含非仇恨冒犯类别，
这个数值改善不能证明模型更会区分“攻击”与“报道攻击”。
[曲线](../../runs/20261002_verdict_analysis/analysis/figures/HateClipSeg_bit_Jl81Ars8alKV_scores.png)
／[采样帧](../../runs/20261002_verdict_analysis/analysis/figures/HateClipSeg_bit_Jl81Ars8alKV_frames.jpg)。

**D. Reader 改善未必转化为最终收益：HateMM `hate_video_318`。**
连续辱骂、几乎全片为正，只有三个窗口。原始 max AUC **.469 → .802**，
但经过 r6 后两组都是 **.865**。短视频中，少数窗口相对名次的变化可带来很大的
原始 AUC 差异；Decoder 已恢复的部分不会再次变成最终收益。

### 3. 失败案例：不能说全局错误会被自动纠正

- **错误 No：HateMM `hate_video_279`**。14.44 秒，只有两窗，ASR 仅“Outro Music”，
  GT 为 6–14 秒；缓存帧是人物动作剪辑，现有转录不支持解释具体仇恨语义。
  全局 z = −11.70，加入 No 后最终 AUC **.765 → .125**。
  两窗原始 max 排序完全不变；第一窗 speech −13.9 → −11.9，第二窗约 −15.1 → −14.7，
  下游拟合和时序推断却改变了最终顺序。
  这是系统在错误 No 条件下可明显退化的反例，不能据此单独分离 No token 与下游重拟合的因果贡献。
  [曲线](../../runs/20261002_verdict_analysis/analysis/figures/HateMM_hate_video_279_scores.png)。
- **正确 Yes 也会退化：HCS `bit_EH4buPGuFok7`**。转录含针对犹太人的辱骂与威胁，
  夹有质问、旁观者说话等片段。Yes 后一些直接辱骂窗口的 speech 升高，但 visual 普遍降低、
  中间一些窗口 speech 也降低；GT 标注的是较长连续事件区间。
  最终 AUC **.933 → .702**。全局判断正确不等于各窗口或完整事件范围一定判断更好。
  [曲线](../../runs/20261002_verdict_analysis/analysis/figures/HateClipSeg_bit_EH4buPGuFok7_scores.png)。

按 GT 正视频里的裁定分组，最终 within 的平均变化为：

| 组 | HateMM | HCS | DeHate |
|---|---|---|---|
| 正确 Yes | +0.57 pp（n=81） | +0.78 pp（n=94） | +0.35 pp（n=185） |
| 错误 No | −19.79 pp（n=3） | −3.00 pp（n=5） | +4.27 pp（n=37） |

主语料错误 No 的样本太少，不能估计一般失败率；DeHate 方向不同且 CI 含 0，
也说明 No 不是硬性关掉局部正例的门。按正帧覆盖率分组同样没有跨三个语料一致的受益区间。
所有子组都是探索性分析，没有多重比较校正；不能用事后显著子组构造普遍机制结论。

### 4. 为什么会有收益：分数变化比“新增语义理解”更有证据

原始 max 读数的有/无轮次逐视频 Spearman 中位数为 **.931 / .918 / .914**。
其原始 within AUC 平均上升 **1.16 / 1.42 / 1.21 pp**，但三个 CI 也都含 0。
相对次序大体保留，显著变化之一是读数水平：误报为 Yes 的正常视频，
max 读数逐视频平均抬升 **3.16 / 2.90 / 2.92 logits**。
这说明它也会强化误报，不能说其功能是抑制正常视频的错误激活。

为区分局部重排序和分数水平的作用，追加了一个**不使用 GT 的诊断重建**：
给每个无裁定视频的全部窗口、全部模态加上同一个常数，常数是该视频
有/无轮次 max 读数的平均差。所有原始模态内及 max 窗口排名、并列关系保持不变，
再经过同样的 r6 corpus fit 和推断。

| 条件 | HateMM ROC / PR / within | HCS ROC / PR / within |
|---|---|---|
| 无全局轮 | .896761 / .696719 / .765482 | .705523 / .658747 / .631225 |
| 仅作视频内常数平移 | .894056 / .691133 / .754031 | .715576 / .671068 / .639505 |
| 真实全局轮 | .894242 / .691086 / .763920 | .715843 / .671566 / .637100 |

来源：[平移对照 metrics.json](../../runs/20261002_verdict_analysis/main_shift_only/metrics.json)。
该常数依赖两组真实缓存，**不是可部署的新方法，也不是因果中介效应估计**。

HCS 仅平移就获得 within **+0.83 pp**，真实全局轮为 **+0.59 pp**；pooled 两项也很接近。
所以至少在 HCS，整体正向结果并不要求 Reader 产生更好的窗口排名。
视频之间不同的偏移会改变语料级正态分数和拟合证据分布，继而改变时序后验、融合与最终排序。
HateMM 则是平移有损失，真实全局轮相对平移恢复部分性能，但没有超过无轮版本。
两个数据集“真实轮 − 平移”的 within CI 都含 0。

因此较谨慎的解释是：**先作全局回答，改变了模型随后给局部证据打分的分布；
某些视频中，这使明确证据与背景更容易被下游模型区分，但也会强化偏见或改变事件边界。**
不能由当前数据推出“更懂反讽/指代/隐式仇恨”，也不能将对话轮收益全部归因于 Yes/No 本身。

### 5. 对论文的含义与尚缺的验证

这轮分析不足以把 self-verdict 写成经验证的核心贡献。可以如实描述实现与动机，
将其视为一个条件化选择，并披露收益有限、依赖 Decoder 和数据集。
本轮没有更改方法、论文、主表，也没有用 GT 调参数。

若要在论文强调这一设计，下一步应先在**最新主语料 Reader 家族**重跑无全局轮，
再考虑固定 Yes、固定 No 或保留问题但不提供裁定等控制，分离模型自裁定和一般对话形式的作用。
这些控制需要重新调用 MLLM，**本轮未启动**。其必要性取决于是否仍要把该设计作为论文主张。

## 输出与检查

- [逐视频结果](../../runs/20261002_verdict_analysis/analysis/per_video.csv)、[逐窗口转录与读数](../../runs/20261002_verdict_analysis/analysis/per_window.csv)。
- [子组统计](../../runs/20261002_verdict_analysis/analysis/summary.json)、[对齐检查](../../runs/20261002_verdict_analysis/analysis/alignment.json)。
- [8 个案例的完整证据](../../runs/20261002_verdict_analysis/analysis/selected_cases.json)，各案例都有 PNG/PDF 曲线和带时间戳的帧拼图。
- [数值验证](../../runs/20261002_verdict_analysis/analysis/verification.json)：逐视频均值与官方 metrics 完全一致，现有全组 r6 输出精确复现；平移对照读取前后排名一致。
- 读取了上述 test GT、缓存预测、ASR 和采样帧；发现全局轮的收益不稳且与下游分布拟合交互；**未改变任何方法设计**。全部为 development-selected 分析。
