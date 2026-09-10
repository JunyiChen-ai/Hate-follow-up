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

无。MLLM family / 尺寸鲁棒性研究（实验 README §11）2026-09-11 04:24 全部完成：7 个模型 × 7 个臂，结果表 `runs/20260910_spvl/mllm_table.md`，解读在 README §11 Results。结论：方法在 7 个 MLLM 上都有效（within 从模型自身的 .62–.64 / .49–.58 提到 .65–.70 / .55–.60），固定窗、M3、帧三个部件在 ≥5/7 模型上方向一致；立场条件化只在 HateMM 上过数；整段转录语境只对 Qwen3-VL 系列和 Gemma 有用。派发记录：2026-09-10 21:00 起并行跑在 uoa-lab3（Qwen3-VL-8B 一致性检查、4B、2B）、uoa-lab2（InternVL3.5-8B、LLaVA-OneVision-7B）、lab-server（Gemma-3-12B）、uoa-campus1（Qwen3-VL-32B，job 16689）、uoa-campus2（Qwen2.5-VL-7B，job 19985）。输出 `runs/20260910_spvl/mllm/<tag>/<arm>/`，汇总 `runs/20260910_spvl/mllm_table.md`。round-3 消融已完成（`runs/20260910_spvl/abl3_*/`）。

## 下一步

1. LELA（GPT-4o-mini）对照在本评测器下重跑或说明不可行（规则 14f，需要 API 费用，待用户裁定）。
2. HCS 无语音仇恨子集低于随机：GT 是 offensive 并集而 prompt 是仇恨规则，属标签定义问题，写进论文的 limitation。
3. 方法改进从 SPVL-r2 出发；OMSL-v6 目录保留为对照，下次整理时移入 `archive/experiments/`。

## 资料与历史

[CLAUDE.md](../CLAUDE.md)、[1 fps 协议与 baseline 表（历史）](../docs/protocol_1fps_legacy/)、[基础论文 TRIAGE 与检测时代记录](../archive/README.md)、[2026-08 idea discovery 报告](../archive/idea-stage-2026-08/idea-stage/IDEA_REPORT.md)。
