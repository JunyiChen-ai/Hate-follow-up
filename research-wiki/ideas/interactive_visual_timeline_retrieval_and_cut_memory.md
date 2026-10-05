---
type: idea
node_id: idea:interactive_visual_timeline_retrieval_and_cut_memory
title: "完整视觉时间工具交互与动态视频记忆"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 完整视觉时间工具交互与动态视频记忆

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：复用 native20、完整 ASR、原 G/ownstance。另将真实视频按既定 8 秒窗形成 1 fps 片段。Qwen 从原材料生成最多两个具体可观察事件查询；同一 Qwen 对每段执行查询相关性读取，相关性只用于检索实际媒体，不进入最终仇恨分数。
来源：https://arxiv.org/html/2510.14672v1
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

