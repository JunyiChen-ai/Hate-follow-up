---
type: idea
node_id: idea:target_conditioned_spatial_search_with_temporal_crop_memory
title: "目标条件化空间搜索与时间绑定裁剪记忆"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 目标条件化空间搜索与时间绑定裁剪记忆

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：对每个 8 秒窗的原材料及最多两张真实局部帧，同一 Qwen 识别影响判断但看不清的可观察目标，例如牌子上的文字、手势或被指向对象；最多一个搜索目标，无缺失目标则执行原生 V。
来源：https://arxiv.org/html/2312.14135v2
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

