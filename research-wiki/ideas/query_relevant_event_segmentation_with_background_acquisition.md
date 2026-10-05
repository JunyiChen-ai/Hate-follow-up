---
type: idea
node_id: idea:query_relevant_event_segmentation_with_background_acquisition
title: "相关事件分段与背景获取"
stage: proposed
outcome: pending
added: 2026-10-05T01:04:17Z
based_on: []
target_gaps: []
tags: ["M1", "label-free", "proposed"]
---

# 相关事件分段与背景获取

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
方法：同一冻结 Qwen 从每窗两张实际 PTS 帧及本窗 ASR 生成中性短描述。对当前窗，从本窗描述构造关于动作、对象和场景的事实检索请求；在固定前后四窗的候选区间内，一次生成各描述的相关度 1–10，不能输出仇恨概率或最终答案。
来源：https://aclanthology.org/2026.lrec-1.395.pdf
完整方法/控制/成本见 experiments/20261005_m1_ideation/CANDIDATES.json。

## Key risks
一次独立rule4审查PASS，same-family provisional；性能/机制未测，具体实施缺口和目标任务近邻见 docs/reviews/20261005_m1_ideation_jury.md。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

