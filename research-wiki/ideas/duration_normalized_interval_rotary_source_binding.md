---
type: idea
node_id: idea:duration_normalized_interval_rotary_source_binding
title: "时长归一化的区间旋转来源绑定"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 时长归一化的区间旋转来源绑定

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：沿用现有ASR原segment区间和token到segment映射，不新增对齐器、不生成词时间、不读取标签；保留native前缀、G、ownstance和visual分支。
来源：https://arxiv.org/html/2605.10543v1
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

