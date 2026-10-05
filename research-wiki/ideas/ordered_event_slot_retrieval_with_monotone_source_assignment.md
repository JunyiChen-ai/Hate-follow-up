---
type: idea
node_id: idea:ordered_event_slot_retrieval_with_monotone_source_assignment
title: "有序事件槽的实际来源选择"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 有序事件槽的实际来源选择

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：同一冻结 Qwen 为每个原生窗口生成短的上下文描述，输入为两张实际 PTS 帧、本窗 ASR 和前一窗短描述。随后按每批八窗生成各窗的 prequel/current/sequel 三个中性检索槽；槽是待查条件，不是事实，也不包含候选仇恨答案。
来源：https://arxiv.org/html/2506.10202v1
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

