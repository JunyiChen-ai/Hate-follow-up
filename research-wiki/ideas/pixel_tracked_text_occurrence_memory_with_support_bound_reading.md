---
type: idea
node_id: idea:pixel_tracked_text_occurrence_memory_with_support_bound_reading
title: "像素跟踪的屏幕文字发生记忆与支撑区间读取"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 像素跟踪的屏幕文字发生记忆与支撑区间读取

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：每窗用同一 Qwen 对两张实际局部帧提出屏幕文字区域及逐字内容，不生成仇恨结论。对这些区域在真实 4 fps 像素上执行 CPU 光流/模板跟踪，使用前后向一致性与外观残差识别失配；镜头切换终止轨迹，不以语言推断跨镜头身份。
来源：https://arxiv.org/html/2403.11481v2
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

