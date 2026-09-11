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

**当前在跑**：表示层天花板测试——保存窗分支读出位置的隐状态，测 Yes/No 投影丢掉的排序信息还有多少
（`tad.py --save-hidden 1`，`runs/20260912_tad/e0_hidden/`）。这个数不受上面 oracle 的约束，决定下一步是
在表示上训练轻量 MIL 头（用模型自己的视频级裁定当伪标签），还是必须动模型本身。

**新增 secondary 评测**（用户裁定 2026-09-12）：`data/gt_4fps_hate_only/HateClipSeg.npz`，只取 Hateful 这
一类（base rate .198 vs 主表 .471，含正负帧的视频 51 vs 99）。主表不变，只作并列诊断。

SPVL-r2 在两个口径下（`runs/20260910_spvl/full2_dual_evid_stance/metrics_izv_plus_mean_rrank{,__hate_only}.json`）：

| GT 口径 | pooled ROC | pooled PR | 随机水平 | within | n |
|---|---|---|---|---|---|
| 主表（offensive 并集） | .7119 | .6664 | .473 | .6001 | 99 |
| 只取 Hateful | **.7774** | .4323 | .200 | **.6236** | 51 |

PR 对随机水平的倍数：1.41 → **2.16**。口径对齐后三项全部变好，说明主表上 HCS 的数字被标签定义压低，
这一点现在有量化依据。

## 下一步

1. TAD E0 判定：若 `Spearman(act, topic) ≥ .9` 则该方向当场停；否则校正后的残差要比 act 残差在两语料
   within 都高 ≥.01 才进 E1 全量。
2. LELA（GPT-4o-mini）对照在本评测器下重跑（规则 14f）。用户 2026-09-12 裁定：先不做，专心做方法，投稿前再补。
3. 方法改进从 SPVL-r2 出发；OMSL-v6 目录保留为对照，下次整理时移入 `archive/experiments/`。

## 资料与历史

[CLAUDE.md](../CLAUDE.md)、[1 fps 协议与 baseline 表（历史）](../docs/protocol_1fps_legacy/)、[基础论文 TRIAGE 与检测时代记录](../archive/README.md)、[2026-08 idea discovery 报告](../archive/idea-stage-2026-08/idea-stage/IDEA_REPORT.md)。
