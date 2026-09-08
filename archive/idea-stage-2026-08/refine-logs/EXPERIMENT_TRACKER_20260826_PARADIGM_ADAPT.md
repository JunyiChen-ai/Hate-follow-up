# Experiment Tracker — Paradigm Adaptation

| Run | Artifact | Status |
|---|---|---|
| P1 Stage-A | `results/idea_discovery/paradigm_adapt/p1_stage_a_8_clean16.jsonl` | killed |
| P2 Stage-A | `p2_stage_a_8.jsonl` | promoted |
| P2 Stage-B decode-safe | `p2_stage_b_32_decodefix.jsonl` | complete; survives |
| P3 Stage-A | `p3_stage_a_8.jsonl` | killed versus P2 parent |
| P4 Stage-A | `p4_stage_a_8.jsonl` | killed |
| P5 Stage-A | `p5_stage_a_8.jsonl` | killed |
| P6 TimeLens provenance rerun | `p6_timelens_stage_a_8.jsonl` | complete |
| P6 Stage-A | `p6_stage_a_8.jsonl` | provisional survivor; exploratory only |
| P7 orders 0/1/2 | `p7_stage_a_order{0,1,2}.jsonl` | killed for order fragility |

Deployment interpreter: `/home/jehc223/venvs/SafetyContradiction/bin/python`.

Next action: run a fresh-split P2/P6 comparison; do not enlarge P1, P3, P4, P5, or P7 without a new mechanism.
