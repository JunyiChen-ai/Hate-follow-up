---
type: idea
node_id: idea:executable_temporal_evidence_program
title: "executable temporal evidence program"
stage: proposed
outcome: pending
added: 2026-10-04T12:38:48Z
based_on: []
target_gaps: []
tags: ["m1", "zero-label", "development-selected"]
---

# executable temporal evidence program

**stage:** `proposed`  ·  **outcome:** `pending`

Canonical details: `experiments/20261005_m1_program/README.md`. Interface A actual fixed-five failed its noGT execution guard; independent diagnosis found noncompliant generated programs, no observed implementation bug. Explicit source-handle interface B was declared and independently reviewed, and its same fixed-five GPU validation is running. No GT/performance verdict. Source timing and complete main/mechanism gates remain unchanged; all review conclusions are same-family provisional.

## Thesis
按候选实验README验证完整时间/来源/scope程序依赖；工具调用已见MAESTRO，不作为首次贡献。仍待完整双语料主指标与机制控制。

## Key risks
CPU审查不等于真实Qwen数值parity或语义正确性；输入frame是nominal时间，字符是比例时间。必须过匹配感知预算/flat/错误绑定及rule14g。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

