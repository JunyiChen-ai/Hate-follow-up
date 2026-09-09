# 当前研究状态

截至 **2026-09-09**。依据：`runs/20260829_omsl_v6/v6_migrated_seed0_20260909/metrics.json`（当前代码，本机 uoa-lab1，conda HateVideo）和 `runs/20260829_omsl_v6/orthogonal_mobius_semantic_localizer_full643_v6_metrics.json`（2026-08-29 原始）。

## 当前目标与结论

项目定义为 label-free hateful video temporal localization（零仇恨标注），主数据集 HateMM + HateClipSeg，4 fps 协议，主指标 pooled ROC / PR（2026-09-09 裁定，见 `CLAUDE.md`）。当前方法 OMSL-v6 在这两个数据集上三项指标都高于同协议的 MultiHateLoc（视频级标签训练）和 T3AL（零标签）重跑值。这是在同一批数据上反复选出来的开发期结果，不是未揭盲的确认结果（2026-08-29 审计定性 exploratory，见 `archive/root-2026-09/EXPERIMENT_AUDIT.md`）。

## 当前方法：OMSL-v6

`experiments/20260829_omsl_v6/README.md`。三模块：冻结 Qwen3-VL-8B 整视频 logit 作视频截距；视觉 / 时间戳文本 / ImageBind 音频三路流的 Möbius 联盟分解，交互项按 31 次块置换空分布校准；视觉主导的字典序打破平局，残差去均值后加截距。无学习参数，推理零 MLLM 调用，输出帧分数，不输出区间。2026-09-09 迁入并去哈希（置换种子改为常数 0），pooled 指标不变，within 第四位小数变化。

## 最新权威结果（test，4 fps；pooled ROC / pooled PR / within-video macro ROC，三项并列主指标，2026-09-10 裁定）

| 方法 | HateMM | HateClipSeg | MHC（历史） | MHC_zh（历史） |
|---|---|---|---|---|
| **OMSL-v6**（`v6_migrated_seed0_20260909/metrics.json`） | **.8507 / .5781 / .6494** | **.6692 / .6622 / .5473** | .7458 / .4970 / .7011 | .7522 / .5354 / .6837 |
| MultiHateLoc-DMS 重跑（`multihateloc_frozen_current4fps_v1_metrics.json`） | .7618 / .5188 / .6108 | .5056 / .4885 / .4996 | .7814 / .4790 / .5013 | .8778 / .6565 / .4962 |
| T3AL 重跑，611 视频，seed 20250819（`t3al_anchor_s20250819_metrics.json`，同目录） | .6091 / .3096 / .5068 | .6246 / .5645 / .5003 | .5975 / .2834 / .5605 | .6966 / .3584 / .4996 |

PR 的随机水平 = 帧正例率：HateMM .242、HateClipSeg .473。T3AL 覆盖 611 / 643 视频，base rate 不同，只看趋势。T3AL 的超参 preset 是按 val 集 pooled PR-AUC 选的（`Retrieval-hate/scripts/repro_campaign/t3al_select.py`），方法本身不读标签。within 只在正负帧都有的视频上计算：HateMM 84 / 215，HateClipSeg 99 / 118。

2026-09-10 修正：此前 T3AL 行的数字（.6886 / .4315 / .6636 等）误抄自 `endpoint_equilibrium_t3al_lcurve_v1_metrics.json`，那是 2026-08 idea discovery 里基于 T3AL 曲线的区间端点重解码变体，不是 T3AL 本身。

## 输入与缓存

`data/omsl_v6_inputs/`（约 82M，`PROVENANCE.md`）：manifest、A10 视觉曲线、Qwen3-VL-8B 分块文本 log-odds、ImageBind 音频 embedding、整视频 logit。`data/gt_4fps/`：GT 数组。`data/assets/imagebind/`：ImageBind 权重与缓存的文本锚点。1 fps 时代输出在 `runs/legacy_1fps/`（本机 17G + lab2 回传 51G），2026-08 idea discovery 中间产物在 `runs/legacy_idea_discovery_2026-08/`。lab2 上另有 `results/steward_private/thvl_bench`（23G，THVL-Bench 数据集材料，含加密标签），未回传，留在 lab2。

## 运行任务与监控

无运行中的实验。lab2 结果回传 rsync 在本机后台（`runs/legacy_1fps/lab2/rsync_reproduction.log`）。

## 下一步

1. 校区服务器首次使用：clone 仓库到 `/data/jehc223/Hate-follow-up`，建 `HateVideo` 环境，同步 HateClipSeg / HateMM 原始视频（campus2 缺 HCS，campus3 全缺）。
2. 方法改进从 OMSL-v6 出发；任何新实验建 `experiments/<日期>_<slug>/`，输出到 `runs/`，结束更新本文件。

## 资料与历史

[CLAUDE.md](../CLAUDE.md)、[1 fps 协议与 baseline 表（历史）](../docs/protocol_1fps_legacy/)、[基础论文 TRIAGE 与检测时代记录](../archive/README.md)、[2026-08 idea discovery 报告](../archive/idea-stage-2026-08/idea-stage/IDEA_REPORT.md)。
