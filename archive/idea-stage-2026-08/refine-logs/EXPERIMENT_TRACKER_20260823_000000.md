# Experiment Tracker

| Run ID | Milestone | System | Scope | Priority | Status | Notes |
|---|---|---|---|---|---|---|
| R000 | M0 | protocol/schema/evaluator | unit tests | MUST | DONE | schema, label isolation, pooled/within-video/interval metrics pass |
| R001 | M0 | NumPro smoke | 2 videos | MUST | DONE | parser and inference pass; empty prediction discovered |
| R002 | M0 | temporal-consistency smoke | synthetic curves | MUST | DONE | deterministic unit checks pass |
| R003 | M1 | asset audit | 12 works | MUST | PARTIAL | 9 ready, 2 partial, A02 URL unresolved; A08/A09/A10/A12 cloned |
| R101 | M2 | A01 NumPro | 32 videos | MUST | FAIL_GATE | 100% coverage, 0/32 nonempty |
| R102 | M2 | A02 VTimeCoT | 32 videos | MUST | FAIL_GATE | constant empty curves |
| R103 | M2 | A03 Temporal Consistency | 32 videos | MUST | FAIL_GATE | within-video ROC 0.5; sparse parse failures |
| R104 | M2 | A04 OmniVTG | 32 videos | MUST | FAIL_GATE | constant empty curves |
| R105 | M2 | A05 MUSEG | 32 videos | MUST | FAIL_GATE | constant empty curves; sparse parse failures |
| R106 | M2 | A06 DisTime | 32 videos | MUST | FAIL_GATE | only 2 nonempty; pooled artifact |
| R107 | M2 | A07 ED-VTG | 32 videos | MUST | FAIL_GATE | only 1 nonempty; pooled artifact |
| R107b | M2 | binwise timed multimodal | 32 videos | MUST | PROMISING | 21/32 varying; pooled strong, within-video mixed |
| R108 | M2 | A08 TGB | 32 videos | MUST | TODO | one frozen config |
| R109 | M2 | A09 GroundingGPT | 32 videos | MUST | TODO | one frozen config |
| R110 | M2 | A10 Vid-Group | 32 videos | MUST | TODO | one frozen config |
| R111 | M2 | A11 Seq2Time | 32 videos | MUST | TODO | one frozen config |
| R112 | M2 | A12 TimeLens | 32 videos | MUST | DONE | official 8B checkpoint; avg within=.565, F1@.5=.130 |
| R200 | M3 | frozen evaluation | 4 datasets | MUST | PARTIAL | A01-A07 and binwise pilot evaluated; waits for official checkpoints |
| R300 | M4 | top candidates | full, 3 seeds | MUST | BLOCKED | waits for M3 |
