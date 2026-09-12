# 当前研究状态

截至 **2026-09-09**。依据：`runs/20260829_omsl_v6/v6_migrated_seed0_20260909/metrics.json`（当前代码，本机 uoa-lab1，conda HateVideo）和 `runs/20260829_omsl_v6/orthogonal_mobius_semantic_localizer_full643_v6_metrics.json`（2026-08-29 原始）。

## 当前目标与结论

项目定义为 label-free hateful video temporal localization（零仇恨标注），主数据集 HateMM + HateClipSeg，4 fps 协议，主指标 pooled ROC / PR（2026-09-09 裁定，见 `CLAUDE.md`）。当前方法 OMSL-v6 在这两个数据集上三项指标都高于同协议的 MultiHateLoc（视频级标签训练）和 T3AL（零标签）重跑值。这是在同一批数据上反复选出来的开发期结果，不是未揭盲的确认结果（2026-08-29 审计定性 exploratory，见 `archive/root-2026-09/EXPERIMENT_AUDIT.md`）。

## 当前方法：SPVL-r2（2026-09-10 晋级，development-selected）

`experiments/20260910_spvl/README.md`。范式：stance-conditioned evidence localization。Qwen3-VL-8B 两次前向 / 视频：(1) 公共前缀 = 规则 + 20 帧带时间戳 + 整段 Whisper 转录带时间戳，读整视频裁定 log-odds；(2) 模型自己的裁定接进前缀，每个 8 秒窗一个画面分支和一个语音分支（"这一窗是否是违规内容所在片段"），分支之间用 block-diagonal mask 隔离，窗口分 = 两分支最大值；帧分 = (裁定 + 逐窗均值) + 视频内中心化秩残差。零训练、零标签；预处理只有 Whisper 和抽帧；去掉了 v6 的 CLIP、Vid-Group、ImageBind 和十几次文本调用。

## 最新权威结果（test，4 fps；pooled ROC / pooled PR / within-video macro ROC，三项并列主指标）

| 方法 | HateMM | HateClipSeg |
|---|---|---|
| **SPVL-r2**（`runs/20260910_spvl/full2_dual_evid_stance/metrics_izv_plus_mean_rrank.json`） | **.8919 / .6831 / .6976** | **.7119 / .6664 / .6001** |
| SPVL-r2 + 每窗一帧（`full3_dual_evid_stance_w8/metrics_izv_plus_mean_rrank.json`，5 倍代价） | .8936 / .6786 / .6885 | .7199 / .6798 / .6004 |
| OMSL-v6（前一方法，`runs/20260829_omsl_v6/v6_migrated_seed0_20260909/metrics.json`） | .8507 / .5781 / .6494 | .6692 / .6622 / .5473 |
| MultiHateLoc-DMS 重跑，弱监督（`runs/20260829_omsl_v6/multihateloc_frozen_current4fps_v1_metrics.json`） | .7618 / .5188 / .6108 | .5056 / .4885 / .4996 |
| T3AL 重跑，611 视频（`runs/20260829_omsl_v6/t3al_anchor_s20250819_metrics.json`） | .6091 / .3096 / .5068 | .6246 / .5645 / .5003 |
| ZS-ImageBind 重跑，2026-09-12（`runs/20260912_baselines/zs_imagebind/metrics.json`；HCS 3 个视频无可解码视频流） | .5928 / .3108 / .5343 | .5917 / .5495 / .5241 |

PR 的随机水平 = 帧正例率：HateMM .242、HateClipSeg .473。within 只在正负帧都有的视频上算：HateMM 84 / 215，HCS 99 / 118。晋级门（相对 v6，噪声下限 pooled .005 / within .01）：HateMM +.041 / +.105 / +.048，HCS +.043 / +.004 / +.053，全过。消融表 `runs/20260910_spvl/ablation_table.md`；MHC 历史数字见 git 历史。

## 输入与缓存

`data/omsl_v6_inputs/`（约 82M，`PROVENANCE.md`）：manifest、A10 视觉曲线、Qwen3-VL-8B 分块文本 log-odds、ImageBind 音频 embedding、整视频 logit。`data/gt_4fps/`：GT 数组。`data/assets/imagebind/`：ImageBind 权重与缓存的文本锚点。1 fps 时代输出在 `runs/legacy_1fps/`（本机 17G + lab2 回传 51G），2026-08 idea discovery 中间产物在 `runs/legacy_idea_discovery_2026-08/`。lab2 上另有 `results/steward_private/thvl_bench`（23G，THVL-Bench 数据集材料，含加密标签），未回传，留在 lab2。

## 运行任务与监控

HVL（假设–验证闭环，`experiments/20260911_hvl/`）2026-09-11 试运行两轮结束：假设条件化、顺序状态链、相邻窗语境、判定修订全部低于 SPVL-r2（within 子集基线 .693 / .601），机制诊断见其 README §7–§8；按规则 9 归档为负结果，SPVL-r2 仍是当前方法。MLLM family / 尺寸鲁棒性研究（spvl README §11）2026-09-11 04:24 全部完成：7 个模型 × 7 个臂，结果表 `runs/20260910_spvl/mllm_table.md`，解读在 README §11 Results。结论：方法在 7 个 MLLM 上都有效（within 从模型自身的 .62–.64 / .49–.58 提到 .65–.70 / .55–.60），固定窗、M3、帧三个部件在 ≥5/7 模型上方向一致；立场条件化只在 HateMM 上过数；整段转录语境只对 Qwen3-VL 系列和 Gemma 有用。派发记录：2026-09-10 21:00 起并行跑在 uoa-lab3（Qwen3-VL-8B 一致性检查、4B、2B）、uoa-lab2（InternVL3.5-8B、LLaVA-OneVision-7B）、lab-server（Gemma-3-12B）、uoa-campus1（Qwen3-VL-32B，job 16689）、uoa-campus2（Qwen2.5-VL-7B，job 19985）。输出 `runs/20260910_spvl/mllm/<tag>/<arm>/`，汇总 `runs/20260910_spvl/mllm_table.md`。round-3 消融已完成（`runs/20260910_spvl/abl3_*/`）。

## 2026-09-12 这一轮：PWC 负结果 + 机制诊断

**PWC（窗间成对比较，`experiments/20260912_pwc/`）在 E0 被证伪，已归档。** 声明的门是比较的准确率比
`sign(z_i − z_j)` 在两语料都高 ≥5 点；实际 −1.0（HateMM）/ +0.5（HCS）。704 对，
`runs/20260912_pwc/e0b/summary.json`。读出本身没问题：交换一致率 .82–.86、A/B/C 边缘概率无偏好、撤掉转录
掉 .119/.054。所有读出方式（绝对、相对、联合上下文）都落在 .55–.69 同一带内，彼此符号一致率 .71–.83。
结论：视频内排序的上限不是读出方式，是模型能分辨多少（H2 而非 H1）。

顺带零成本否掉的机制：**语境对比**（`z(全局语境) − z(窗单独)`）within .467/.494，对照 `z(全局语境)`
的 .699/.602（`runs/20260910_spvl/mllm/q3vl-8b/{full,winonly}`）。

**机制诊断（PWC README §7c）**：把窗按 GT 标签 × 该窗是否提到目标群体交叉列表，"提到目标群体"值
**+8.1（HateMM）/ +7.0（HCS）** log-odds，"真的是仇恨窗"只值 **+4.3**。冻结 MLLM 的窗级判断主要是
话题检测而不是行为检测。这解释了 PWC 的失败、七个模型同一个 within 天花板、以及视频级强而窗级弱。

**TAD（`experiments/20260912_tad/`）两轮都没过门，已归档。**
- round 1 减掉话题维度：`a − β·t` 在两语料都掉，且随 β 单调下降（HateMM .6926 → .6419 → .6159 → .5749）。
  原因是 topic 单独就能排到 .6567/.5467——提到受保护群体在这两个语料里是正向预测的，减掉等于减信号。
  加法对照（秩和）也掉（.6858/.5811）。用别的视频的 t 作对照，β 中位数塌到 0。
- round 2 五选一言语行为：模型 79.6%/56.0% 的窗答 `attacks`，`unrelated` 18.6%/34.9%，三个豁免选项
  （reports/quotes/condemns）合计只有 1.9%/9.2%。五选一塌回二元，就是话题轴（与 act 读数 Spearman .73/.70）。
  within：actmargin .6800/.5998，与 act 的秩和 .6945/.6076，都在噪声内。
- **oracle 天花板**（真标签监督、按视频分折、out-of-fold，仅诊断）：在 MLLM 各路读数 + 窗形状 +
  ImageBind 音频 + 相邻窗读数上拟合，都打不过不拟合的原始窗分（HateMM .758 vs .742/.731/.744；
  HCS .621 vs .625/.626/.614）。唯一例外 GBT+音频在 HateMM +.035，HCS −.069 不迁移。
  结论：这一族标量特征的组合空间已到顶，伪标签自训练头也被它框住。
- 规则 4 review 的第一个对照（act 臂必须复现 SPVL-r2）抓到 compose 里一个重采样 bug，修好后 act =
  .6926/.6007，与 HVL 同代码路径基线一致。诊断的效应量按计数加权后是 1.15 倍（HateMM）/ 1.47 倍（HCS），
  不是之前写的 1.6–1.9 倍。

**表示层天花板也到顶**（`runs/20260912_tad/e0_hidden/`）：把窗分支读出位置的隐状态（Yes/No 投影丢掉的
那部分）拿去拟合，64/128/256 维分别是 .747/.608、.746/.594、.698/.552，都不如不拟合的 .758/.621。
局限已记：向量读在决策之后，前缀里的预决策表示没探。

**改模型这一族（`experiments/20260912_sdl/` 已归档，`experiments/20260912_nga/` 进行中）**

冻结模型的读出、特征组合、表示三条路都封死之后，唯一没试的杠杆是改模型本身。用模型自己的整视频裁定当
伪标签（零人工标注，与 T3AL 同属 test-time adaptation），rank-8 LoRA 只训语言层注意力投影，前缀 no-grad
且关适配器，推理与训练一致。

| 轮次 | 目标 | HateMM within | HCS within | 失败机制 |
|---|---|---|---|---|
| 冻结基线（同代码路径） | — | .6968 | .6020 | — |
| SDL r1 | MIL + 保留立场轮 | .5026 | .5091 | 立场轮把袋标签泄露给每个窗分支，学成按伪标签整体平移（伪负 −17.67 / 伪正 +1.09） |
| SDL r2 | 去立场轮，BCE | 中止 | 中止 | 冻结模型已满足袋约束且 log-odds 饱和到 ±15，损失 0.000–0.03，无梯度 |
| SDL r3 | hinge on raw log-odds | .5092 | .5337 | "只有一个窗为正"是假约束（仇恨占比中位 .24/.47），判正率 45.8%→12.3%，曲线被压平 |
| NGA r1 | 只约束伪负视频 | .6787 | .5875 | 可用整体压低满足，均值 −22.6，标准差 4.46→1.33 |
| NGA r2 | + 正例锚定（0.1） | .6812 | .5996 | 锚定修好压缩（标准差 4.40/5.50、判正率 48.2%/50.8%、Spearman .95），但模型几乎不动 |
| NGA r3 | 锚定 0.03（声明的扫描） | .6723 | .6009 | 三个锚定值都在基线下 / 噪声内，无趋势 |

**NGA 三轮用完未过门，已归档；整个"改模型"方向结束。**合并结论：模型自己产生的视频级伪标签
不足以教会它"视频里哪一段是仇恨的"。三种用法各自因不同且已诊断的原因失败——标签泄露进被监督者的
上下文、正例侧袋约束在数据上为假、只用负例侧可被整体压低满足；三处同时修好之后，剩下的监督量是
333 个视频里的 79 个（HCS 只有 10 个），模型不动。

**最后一个诊断：预决策的前缀表示也没有信息**（`runs/20260912_tad/prefix_probe/`）。用真标签拟合模型在
整视频前缀里对每个窗画面的上下文编码：.6568/.5042（64 维逻辑回归）、.6067/.5178（128 维 GBT），而同一批
窗上不拟合的冻结读数是 .7303/.6143。**比决策本身更差**，在画面承载的 HCS 上也是。所以视频内排序不是被
Yes/No 投影丢掉的，它一开始就不在表示里。局限：只定位了图像 token，转录行的映射没建。

规则 4 的三次 proposal review 全部放行，并抓到多个实现问题（compose 重采样 bug、适配器作用域不一致、
mean-pool 对照训反、稀疏系数被隐式放大、`--only-within-defined` 用 GT 选训练集违反规则 10），均已修复。

**新增 secondary 评测**（用户裁定 2026-09-12）：`data/gt_4fps_hate_only/HateClipSeg.npz`，只取 Hateful 这
一类（base rate .198 vs 主表 .471，含正负帧的视频 51 vs 99）。主表不变，只作并列诊断。

SPVL-r2 在两个口径下（`runs/20260910_spvl/full2_dual_evid_stance/metrics_izv_plus_mean_rrank{,__hate_only}.json`）：

| GT 口径 | pooled ROC | pooled PR | 随机水平 | within | n |
|---|---|---|---|---|---|
| 主表（offensive 并集） | .7119 | .6664 | .473 | .6001 | 99 |
| 只取 Hateful | **.7774** | .4323 | .200 | **.6236** | 51 |

PR 对随机水平的倍数：1.41 → **2.16**。口径对齐后三项全部变好，说明主表上 HCS 的数字被标签定义压低，
这一点现在有量化依据。

## 2026-09-12 收口：范式边界已测定，等用户裁定范围

一天内五个候选（PWC、TAD×2、SDL×3、NGA×3）全部归档为负结果，但它们合起来测定了一条边界：

**冻结 Qwen3-VL-8B 在 8 秒窗粒度上的视频内定位能力是窗级 .758 / .621（帧级 within .6968 / .6020），
下游任何东西都突破不了它。** 依据：
- 换读出（相对 / 绝对 / 联合上下文）全落在同一个 .55–.69 准确率带；
- **用真标签**拟合所有缓存特征（读数、窗形状、ImageBind 音频、相邻窗）打不过不拟合的读数；
- **用真标签**拟合分支隐状态（决策之后）打不过；
- **用真标签**拟合前缀视觉表示（决策之前）更差；
- 零标注的视频级自监督适配（六轮）全部低于基线。

**提问形式也全部试完**（`experiments/20260912_bnd/`，已归档）：问"违规内容是不是从这一窗开始/结束"，
begin .6466/.5605、积分成的状态 .6146/.5265、与 act 的秩和 .6956/.5949，都低于 act 的 .7581/.6212。
它是独立测量（与 act 读数 Spearman .644/.731，未退化），模型也有一点起点感（begin 的最大值落在第一个
GT 正窗上 22.7%/15.6%，随机约 4–5%），但问变化没能把话题混淆差分掉。

至此，冻结模型上六种提问形式（绝对、相对、减话题、五选一行为、问变化、反事实排除）+ 四道真标签天花板
（特征组合、决策后隐状态、决策前前缀表示）+ 六轮零标注适配，全部测完。

**2026-09-12 傍晚补充：上面那句"天花板"是在三个未被动过的常量下测的**——冻结模型、固定 8 秒网格、
全局共享 prefix。当天晚些时候的诊断（`archive/experiments/20260912_cva/` §1(b)，输出
`runs/20260912_cva/diag/frame_coverage.txt`）说明第三个常量本身有问题：20 帧共享 prefix 下，
**HateMM 24.2%、HCS 34.1% 的窗口一帧都没有**，而视觉分支在同一批视频内配对比较时，在**没有帧**的窗口上
排序反而更好（.5663 → .7251，.5417 → .5622）。它和语音分支的视频内 Spearman 只有 +.111 / +.176，
不是在抄语音分支。所以 .758 / .621 是这个决策网格的上界，不是模型的上界。这条诊断至今没有被任何候选解决。

### CVA（反事实归因，第 6 个归档候选）

`archive/experiments/20260912_cva/`。用整片判断的差分给区间打分：`s_i = z_video − z_excl(i)`，
`z_excl(i)` 是同一个整片问题、但要求模型忽略区间 i。规则 4 放行（检索确认 LOO/occlusion 归因在文献里只有
LLM 文本上下文归因、对已训练分类器的事后解释、弱监督 VAD 的训练期正则三种用法），规则 6 code review 干净。

E0（`runs/20260912_cva/e0/metrics_izv_plus_mean_rrank.json`，一次运行，333 视频 0 错误）：
HateMM `.8834 / .6572 / .6747`，HCS `.6724 / .6215 / .5779`。六项全部低于 SPVL-r2，规则 9 归档。
代价预测兑现：HateMM 186 s、HCS 173 s，读数约为 SPVL-r2 的一半。

三条有信息量的对照：
- 打乱对照（视频内置换 s）掉到 .5079 / .5143 —— 模型**确实**按被排除的区间响应，机制不是虚构的，
  只是比直接问窗口弱（.7183 / .5908 对 .7563 / .6190）。
- `rank(s) == rank(−z_excl)` 在 **100%** 的视频上成立。视频内 `z_video` 是常数，而残差本来就是视频内秩，
  所以减掉一个视频内常数**对 within 指标是空操作**。提案 §2 承诺要抵消的那一项，在 CVA 写出来之前就已经
  被秩残差抵消掉了。这是提案里的实质错误。抵消论证要有用，被减掉的那一项必须在视频内变化（TAD 减的
  确实在视频内变化，所以它能改变秩——只是改错了方向）。
- `s` 的均值是 +3.57 / +3.98（排除子句本身值约 4 log-odds，与排除哪个区间无关），而视频内标准差只有
  0.70 / 0.51，对比 SPVL-r2 直接读数的 6.20 / 5.69。子句自身效应约为区间效应的六倍。

**需要用户裁定的范围选项**（当前方法仍是 SPVL-r2，SOTA 门是过的）：
1. 放宽标注预算，允许真视频级标签做弱监督（失去 label-free 定位，但 backbone 远强于 MultiHateLoc）；
2. 把 HCS hate-only 口径扶正为主表（今天已建好 GT，口径对齐后三项全部变好）；
3. 加数据集（规则 1 需用户同意）；
4. 换任务粒度，报区间级指标（F1@IoU，从未报过）；
5. 接受经验研究框架（用户 2026-09-12 已明确否决）。

其它待办：
- LELA（GPT-4o-mini）对照在本评测器下重跑（规则 14f）。用户 2026-09-12 裁定：先不做，投稿前再补。
- OMSL-v6 目录保留为对照，下次整理时移入 `archive/experiments/`。

## 资料与历史

[CLAUDE.md](../CLAUDE.md)、[1 fps 协议与 baseline 表（历史）](../docs/protocol_1fps_legacy/)、[基础论文 TRIAGE 与检测时代记录](../archive/README.md)、[2026-08 idea discovery 报告](../archive/idea-stage-2026-08/idea-stage/IDEA_REPORT.md)。
