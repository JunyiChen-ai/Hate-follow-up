---
type: idea
node_id: idea:importance_mass_temporal_transport_representation
title: "重要性质量约束的跨帧传输表示"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 重要性质量约束的跨帧传输表示

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：在原native帧的同Qwen视觉编码中执行源方法的saliency-weighted coverage选择；由去掉某保留token后的覆盖损失计算leave-one-out重要性，再以负softmax赋传输质量，重要token对应较小可压缩质量。
来源：https://arxiv.org/html/2605.11803v1
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

