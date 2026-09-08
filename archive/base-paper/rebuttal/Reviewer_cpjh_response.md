We thank the reviewer for the thorough summary and for highlighting the efficiency–effectiveness routing design, the modularity of the framework, and the breadth of the validation. The three concerns are exactly the right questions to ask of a multi-scale system, and we ran the corresponding experiments.

**W1: Were the label-free / few-shot baselines evaluated at the same maximum model size (72B)?**

Two clarifications. First, the published-method baselines (MARS, Mod-HATE, LoReHM, ALARM) were reproduced with the backbones their papers specify (e.g., MARS with Qwen2.5-VL-32B); we did not downscale any of them. Second, the zero-shot rows in Table 1 already include models up to 32B, and to close the remaining gap we have now evaluated **Qwen2.5-VL-72B-AWQ — the largest model in our verifier pool — as a uniform single-pass classifier on all four test sets** (every evaluable video under the same evaluation protocol as Table 1), with the same per-dataset policy prompt our verifiers use (ACC / M-F1):

| Model on every video | HateMM | MHClip-EN | MHClip-ZH | ImpliHateVid | Avg ACC |
|---|---|---|---|---|---|
| Qwen2.5-VL-72B-AWQ | 79.9 / 78.8 | 74.5 / 60.5 | 80.5 / 73.2 | 69.1 / 66.4 | 76.0 |
| TRIAGE (one 2B pass per video + 0.69 verifier calls per video on average) | 86.0* / 85.3 | 79.5 / 73.2 | 83.9 / 81.9 | 82.8 / 82.7 | 83.1 |

Giving the maximum scale to *every* video is worse than TRIAGE on all four datasets (−7.1 avg ACC) while requiring a 72B forward pass per video. The same holds for each of our other five verifiers run uniformly. This is the core of our claim: the advantage comes from *where* compute is allocated, not from access to a large model. (*The HateMM TRIAGE cell reads 86.0 here versus 86.5 in Table 1: a post-submission reproducibility audit found that one boundary-region verifier verdict differs in the current verifier outputs, so we re-pinned all reported numbers to the exactly reproducible artifacts and will correct Table 1 in the revision. No other cell is affected, and no conclusion changes.)

**W2: More recent supervised baselines.**

We agree the supervised group should be as current as possible and have taken two steps. (a) We verified the most recent supervised systems with HateMM results: HCC1 (Koushik et al., WWW Companion 2025), which uses a fixed train/test protocol comparable to ours, reports ACC 85.4 / M-F1 84.8 — below TRIAGE's 86.0 without any labels; MM-HSD (ACM MM 2025) reports 87.8 / 87.4 under 5-fold cross-validation, which is not directly comparable to our fixed 215-video split (we will cite it with that protocol note). (b) We reproduced SAGE (Huang et al., ACL 2026), the strongest recent supervised system, on our exact fixed splits using the authors' released code with their hyperparameters (ACC / M-F1):

| Method (labels used) | HateMM | MHClip-EN | MHClip-ZH | ImpliHateVid |
|---|---|---|---|---|
| SAGE, authors' code on our fixed splits (supervised) | 79.5 / 78.3 | 72.7 / 68.4 | 67.8 / 66.5 | **85.8** / **85.8** |
| SAGE, published under its own protocol† (supervised) | 87.1 / 86.3 | 83.8 / 79.6 | 79.0 / 74.8 | — |
| TRIAGE (label-free) | **86.0** / **85.3** | **79.5** / **73.2** | **83.9** / **81.9** | 82.8 / 82.7 |

†SAGE's paper uses a random 7:1:2 HateMM split and the original MultiHateClip partitions with multi-seed averaging — not directly comparable to fixed-split numbers; the magnitude of this protocol effect is documented in SAGE's own paper (their HCC1 re-run: 78.3 under their split vs HCC1's reported 85.4 on its fixed split). SAGE does not evaluate ImpliHateVid; the cell above is our extension of the authors' code to that dataset. Reproduction notes: our fixed validation splits for model selection; our common ASR transcripts instead of Whisper-turbo; single seed; two minimal code fixes required to run, neither touching the method. Under the common fixed-split protocol, label-free TRIAGE exceeds the reproduced SAGE on three of the four benchmarks; SAGE stays ahead on ImpliHateVid, the benchmark with the largest and class-balanced labeled training set (1,283 videos) — consistent with our position that supervised systems lead exactly where ample in-domain labels exist, while the label-free framework closes or reverses the gap where labels are scarce or culturally hard. The revision will include SAGE (published and reproduced, with protocol notes) in the supervised group.

**W3: Deployment cost of hosting 72B-scale models; performance with a 7B/8B-scale verifier pool.**

We ran this ablation within the model-space robustness study of the paper's Generalizability section (§4.6), whose verifier pool already contains the small models below. Restricting the verifier pool to small models only — InternVL3.5-8B, Qwen3-VL-8B, and Gemma-3-12B, so the largest hosted model is 12B and the whole system fits on a single 48GB GPU — yields, for the best of the six orderings (ACC / M-F1):

| Verifier pool | HateMM | MHClip-EN | MHClip-ZH | ImpliHateVid | Avg ACC | Verifier calls/video |
|---|---|---|---|---|---|---|
| Small only (≤12B) | 84.7 / 83.4 | 77.0 / 67.2 | 81.2 / 78.2 | 82.5 / 82.5 | 81.4 | 0.68 |
| Default (27B/32B/72B) | 86.0 / 85.3 | 79.5 / 73.2 | 83.9 / 81.9 | 82.8 / 82.7 | 83.1 | 0.69 |

The small-pool variant loses 1.7 avg ACC yet still **outperforms the best label-free / few-shot baseline on all four datasets** (Table 1: HateMM 79.5, MHClip-EN 76.4, MHClip-ZH 75.2, ImpliHateVid 80.3), and the result is insensitive to verifier order (all six orderings within 0.3 avg ACC). So the framework does not depend on hosting a 72B model; the large pool buys the last ~1.7 points. We will add this table and a deployment-footprint discussion (weights memory per configuration) to the revision.

We hope these experiments resolve the reviewer's concerns and would be glad to provide further detail during the discussion period.
