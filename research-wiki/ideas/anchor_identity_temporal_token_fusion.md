---
type: idea
node_id: idea:anchor_identity_temporal_token_fusion
title: "锚帧身份替代与变化残留表示"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 锚帧身份替代与变化残留表示

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：复用原native20帧、ASR、单Qwen3-VL-8B；独立保留native G、ownstance和speech读数。对同一批图像的最终projector输出执行完整TTF：用帧均值与全局均值的余弦关系选锚帧，在3×3邻域匹配源token。
来源：https://arxiv.org/html/2605.07355v1
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

