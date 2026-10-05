---
type: idea
node_id: idea:multikey_episodic_retrieval_with_neighbor_filtering
title: "多键片段检索与邻域过滤"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 多键片段检索与邻域过滤

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：按原 8 秒窗口建立单视频记忆。每窗用同一冻结 Qwen3-VL-8B 读取两张实际 PTS 帧和本窗 ASR，一次生成短描述及 event/action、dialogue/mention、object/state、summary 四个键；记录原始窗口和帧坐标。键只供检索，不包含仇恨判断。
来源：https://arxiv.org/html/2608.07663v1
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

