# Triplet-judge round — experiment results (2026-04-20)

Raw-data dump for the boundary-rescue triplet-judge experiment round.
No analysis, no story.

## Setup

- **Stage-1 slugs (6)**: `2b`, `qwen2.5-vl-7b`, `gemma-3-12b-it`, `gemma-3-12b-it-16f`, `pixtral-12b-2409`, `minicpm-v-26`
  - Dropped: `internvl3-14b` — vLLM 0.11.0 + InternVL3-14B multi-image chat template bug (`TypeError: can only concatenate str (not "list") to str` on every call, all 3338 inferences returned `{"score":null,"skipped":true}`)
- **Judge pool (8)**: `qwen3-vl-8b`, `gemma-3-12b-it`, `gemma-3-27b-it`, `qwen2.5-vl-32b-awq`, `qwen2.5-vl-72b-awq`, `internvl35-8b`, `llava-onevision-qwen2-7b-ov-hf`, `minicpm-v-26`
- **Triplets**: C(8,3) = 56
- **Band**: `candidates_entropy_band_<slug>.jsonl` — videos where Shannon entropy of stage-1 score exceeds mean entropy (label-free, Phase D)
- **Rescue rule**: per video in band, collect ≥2 valid judge preds ∈ {0,1}; apply majority vote; flip if majority differs from stage-1
- **Stage-1 baseline criterion per ds (protocol)**: EN=TR-Otsu, ZH=TR-GMM, HM=TF-li_lee, IH=TR-GMM
- **V1 pinned targets**: EN 0.7826/0.6958, ZH 0.8255/0.8023, HM 0.8465/0.8362, IH 0.8204/0.8199
- **Test sizes**: EN 161, ZH 149, HM 215, IH 401
- **Total cells**: 6 slugs × 56 triplets × 4 datasets = 1344 cells
- **Raw data**: `results/boundary_rescue/grid_eval/grid_raw.jsonl` (1344 lines)
- **Summary markdown**: `results/boundary_rescue/grid_eval/grid_summary_top.md`

## Per-slug stage-1 baselines

| slug | EN acc/mF1 | ZH acc/mF1 | HM acc/mF1 | IH acc/mF1 |
|---|---|---|---|---|
| 2b | 0.7640/0.6532 | 0.7919/0.7577 | 0.8047/0.7930 | 0.8180/0.8180 |
| qwen2.5-vl-7b | 0.7391/0.6315 | 0.7383/0.7158 | 0.8233/0.8089 | 0.8080/0.8075 |
| gemma-3-12b-it | 0.7267/0.6809 | 0.6913/0.6855 | 0.7907/0.7881 | 0.7880/0.7864 |
| gemma-3-12b-it-16f | 0.7578/0.7271 | 0.7114/0.6984 | 0.8047/0.7980 | 0.8030/0.8034 |
| pixtral-12b-2409 | 0.7019/0.6247 | 0.6376/0.6425 | 0.7628/0.7559 | 0.8304/0.8323 |
| minicpm-v-26 | 0.7143/0.6851 | 0.7047/0.6995 | 0.6837/0.6834 | 0.8304/0.8325 |

## Entropy band sizes

| slug | EN / 161 | ZH / 149 | HM / 215 | IH / 401 |
|---|---|---|---|---|
| 2b | 91 | 88 | 100 | 153 |
| qwen2.5-vl-7b | 94 | 79 | 67 | 203 |
| gemma-3-12b-it | 31 | 18 | 18 | 127 |
| gemma-3-12b-it-16f | 20 | 16 | 25 | 59 |
| pixtral-12b-2409 | 35 | 55 | 44 | 115 |
| minicpm-v-26 | 81 | 59 | 40 | 253 |

## Per-slug grid summary

| slug | #triplets | 4/4 V1 | ≥3/4 V1 | 4/4 S1 | ≥3/4 S1 | avg Δacc across triplets | max avg Δacc | best triplet (max #V1) |
|---|---|---|---|---|---|---|---|---|
| 2b | 56 | 0 | 4 | 29 | 49 | +0.0155 | +0.0265 | gemma-3-27b-it,qwen2.5-vl-32b-awq,llava-onevision-qwen2-7b-ov-hf |
| qwen2.5-vl-7b | 56 | 0 | 0 | 50 | 54 | +0.0122 | +0.0219 | gemma-3-12b-it,gemma-3-27b-it,qwen2.5-vl-32b-awq |
| gemma-3-12b-it | 56 | 0 | 0 | 4 | 42 | +0.0094 | +0.0131 | qwen3-vl-8b,gemma-3-27b-it,qwen2.5-vl-72b-awq |
| gemma-3-12b-it-16f | 56 | 0 | 0 | 0 | 56 | +0.0076 | +0.0140 | gemma-3-27b-it,internvl35-8b,llava-onevision-qwen2-7b-ov-hf |
| pixtral-12b-2409 | 56 | 0 | 0 | 0 | 56 | +0.0260 | +0.0373 | gemma-3-27b-it,qwen2.5-vl-32b-awq,llava-onevision-qwen2-7b-ov-hf |
| minicpm-v-26 | 56 | 0 | 0 | 0 | 7 | +0.0323 | +0.0419 | gemma-3-27b-it,qwen2.5-vl-32b-awq,qwen2.5-vl-72b-awq |

## Global top-10 strict-beat-V1 cells

| rank | slug | triplet | #V1 | Σ vs-V1 (acc+mF1) | EN acc/mF1 | ZH acc/mF1 | HM acc/mF1 | IH acc/mF1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2b | gemma-3-27b-it,qwen2.5-vl-32b-awq,llava-onevision-qwen2-7b-ov-hf | 3 | +0.0360 | 0.783/0.711✓ | 0.826/0.806✓ | 0.842/0.832 | 0.833/0.832✓ |
| 2 | 2b | gemma-3-12b-it,gemma-3-27b-it,qwen2.5-vl-32b-awq | 3 | +0.0167 | 0.789/0.722✓ | 0.805/0.781 | 0.847/0.838✓ | 0.833/0.832✓ |
| 3 | 2b | gemma-3-12b-it,gemma-3-27b-it,llava-onevision-qwen2-7b-ov-hf | 3 | +0.0075 | 0.789/0.722✓ | 0.799/0.774 | 0.851/0.842✓ | 0.830/0.829✓ |
| 4 | 2b | qwen3-vl-8b,gemma-3-12b-it,gemma-3-27b-it | 3 | -0.0246 | 0.783/0.711✓ | 0.799/0.769 | 0.851/0.842✓ | 0.825/0.824✓ |
| 5 | 2b | gemma-3-27b-it,qwen2.5-vl-32b-awq,internvl35-8b | 2 | +0.0241 | 0.770/0.690 | 0.826/0.800 | 0.856/0.847✓ | 0.833/0.832✓ |
| 6 | 2b | qwen3-vl-8b,gemma-3-27b-it,qwen2.5-vl-32b-awq | 2 | +0.0208 | 0.770/0.690 | 0.819/0.794 | 0.860/0.852✓ | 0.833/0.832✓ |
| 7 | 2b | gemma-3-27b-it,qwen2.5-vl-32b-awq,qwen2.5-vl-72b-awq | 2 | +0.0132 | 0.770/0.690 | 0.819/0.796 | 0.860/0.853✓ | 0.828/0.827✓ |
| 8 | 2b | gemma-3-27b-it,qwen2.5-vl-32b-awq,minicpm-v-26 | 2 | -0.0192 | 0.770/0.695 | 0.812/0.785 | 0.856/0.848✓ | 0.823/0.822✓ |
| 9 | 2b | gemma-3-12b-it,gemma-3-27b-it,internvl35-8b | 2 | -0.0312 | 0.789/0.717✓ | 0.799/0.769 | 0.847/0.838✓ | 0.820/0.819 |
| 10 | 2b | gemma-3-12b-it,gemma-3-27b-it,minicpm-v-26 | 2 | -0.0423 | 0.789/0.717✓ | 0.792/0.760 | 0.851/0.843✓ | 0.818/0.816 |

## Per-dataset best cell (max acc, then max mF1)

| dataset | V1 acc/mF1 | best slug | best triplet | best acc/mF1 | Δ vs V1 acc/mF1 |
|---|---|---|---|---|---|
| MHClip_EN | 0.7826/0.6958 | 2b | g12b+g27b+q32 | 0.7888/0.7218 | +0.0062/+0.0260 |
| MHClip_ZH | 0.8255/0.8023 | 2b | g27b+q32+llava-ov | 0.8255/0.8062 | 0.0000/+0.0039 |
| HateMM | 0.8465/0.8362 | 2b | q3-8+g27b+mcpm | 0.8605/0.8528 | +0.0140/+0.0166 |
| ImpliHateVid | 0.8204/0.8199 | gemma-3-12b-it-16f | q3-8+g27b+llava-ov | 0.8329/0.8346 | +0.0125/+0.0147 |

## Judge frequency in global top-10 strict-beat-V1 cells

| judge | count / 10 |
|---|---|
| gemma-3-27b-it | 10 |
| qwen2.5-vl-32b-awq | 6 |
| gemma-3-12b-it | 5 |
| llava-onevision-qwen2-7b-ov-hf | 2 |
| qwen3-vl-8b | 2 |
| internvl35-8b | 2 |
| minicpm-v-26 | 2 |
| qwen2.5-vl-72b-awq | 1 |

## Per-slug Δacc vs stage-1 distribution across 56 triplets

### Min per-ds Δacc (worst dataset's gain)
| slug | p10 | p25 | p50 | p75 | p90 | max | % ≥2pp | % ≥3pp |
|---|---|---|---|---|---|---|---|---|
| 2b | -0.0100 | -0.0031 | +0.0000 | +0.0031 | +0.0071 | +0.0150 | 0.0% | 0.0% |
| qwen2.5-vl-7b | -0.0023 | +0.0000 | +0.0000 | +0.0062 | +0.0124 | +0.0175 | 0.0% | 0.0% |
| gemma-3-12b-it | -0.0155 | -0.0062 | -0.0025 | -0.0025 | -0.0025 | +0.0000 | 0.0% | 0.0% |
| gemma-3-12b-it-16f | -0.0248 | -0.0248 | -0.0186 | -0.0124 | -0.0124 | -0.0062 | 0.0% | 0.0% |
| pixtral-12b-2409 | -0.0661 | -0.0623 | -0.0561 | -0.0474 | -0.0449 | -0.0399 | 0.0% | 0.0% |
| minicpm-v-26 | -0.0324 | -0.0274 | -0.0224 | -0.0175 | -0.0137 | -0.0075 | 0.0% | 0.0% |

### Avg per-ds Δacc (mean across 4 datasets)
| slug | p10 | p25 | p50 | p75 | p90 | max | % ≥2pp | % ≥3pp |
|---|---|---|---|---|---|---|---|---|
| 2b | +0.0079 | +0.0131 | +0.0161 | +0.0189 | +0.0217 | +0.0265 | 17.9% | 0.0% |
| qwen2.5-vl-7b | +0.0075 | +0.0091 | +0.0125 | +0.0149 | +0.0176 | +0.0219 | 5.4% | 0.0% |
| gemma-3-12b-it | +0.0062 | +0.0080 | +0.0099 | +0.0109 | +0.0125 | +0.0131 | 0.0% | 0.0% |
| gemma-3-12b-it-16f | +0.0046 | +0.0057 | +0.0079 | +0.0091 | +0.0111 | +0.0140 | 0.0% | 0.0% |
| pixtral-12b-2409 | +0.0207 | +0.0229 | +0.0251 | +0.0293 | +0.0316 | +0.0373 | 96.4% | 19.6% |
| minicpm-v-26 | +0.0253 | +0.0293 | +0.0319 | +0.0366 | +0.0389 | +0.0419 | 98.2% | 71.4% |

## Per-dataset Δacc distribution — across all 1344 cells, split by ds

| dataset | p10 | p25 | p50 | p75 | p90 | % ≥2pp | % ≥3pp |
|---|---|---|---|---|---|---|---|
| MHClip_EN | -0.0186 | -0.0062 | +0.0062 | +0.0373 | +0.0497 | 33.6% | 31.0% |
| MHClip_ZH | +0.0067 | +0.0134 | +0.0201 | +0.0537 | +0.0671 | 66.1% | 34.8% |
| HateMM | +0.0140 | +0.0221 | +0.0326 | +0.0465 | +0.0605 | 75.0% | 50.3% |
| ImpliHateVid | -0.0524 | -0.0224 | -0.0025 | +0.0125 | +0.0187 | 5.7% | 0.0% |

## Entropy-band tightness sweep (2b slug only)

Tightening band by dropping samples with entropy below the q-quantile of the band's entropy distribution. q=0.0 = current full band.

| q | EN band | ZH band | HM band | IH band | best #V1 | best triplet | best per-ds acc |
|---|---|---|---|---|---|---|---|
| 0.00 | 91 | 88 | 100 | 153 | 3 | g27b+q32+llava-ov | EN 0.783 ZH 0.826 HM 0.842 IH 0.833 |
| 0.30 | 64 | 61 | 70 | 107 | 2 | g12b+g27b+llava-ov | EN 0.764 ZH 0.799 HM 0.847 IH 0.835 |
| 0.50 | 46 | 44 | 50 | 77 | 2 | g12b+g27b+q32 | EN 0.783 ZH 0.785 HM 0.828 IH 0.838 |
| 0.70 | 28 | 27 | 30 | 46 | 2 | g12b+g27b+q32 | EN 0.789 ZH 0.799 HM 0.814 IH 0.828 |
| 0.85 | 14 | 14 | 15 | 23 | 1 | g27b+q32+iv35-8 | EN 0.776 ZH 0.799 HM 0.809 IH 0.825 |
| 0.95 | 5 | 5 | 5 | 8 | 1 | q3-8+g12b+g27b | EN 0.770 ZH 0.792 HM 0.809 IH 0.820 |

## Side tests (label-free gates on top of triplet rescue) — all negative

### Quote-count gate (forward direction: `accept flip iff quote_count ≥ τ`)
Per-judge AUROC of rationale quote count predicting judge correctness, on all 8 judges × 4 datasets:
- All 32 cells have AUROC ∈ [0.31, 0.56]
- Majority below 0.5 (negative predictor)
- gemma-3-27b-it specifically: AUROC 0.474/0.457/0.411/0.517 on EN/ZH/HM/IH

### Quote-count gate (reverse direction: `accept flip iff quote_count ≤ τ`)
Tested on single-judge and triplet variants, 2b + entropy band, τ ∈ {q0,q10,q25,q50,q75,q90,inf}:
- Single-judge best: `gemma-3-27b-it τ=q90` → 4/4 V1 (but τ=inf gives 4/4 too, gate contributes marginal IH gain)
- Triplet best: 3/4 V1 with `gemma+gemma+qwen-32B τ=q90`
- No improvement over no-gate majority on any triplet

### TokenSAR gate (from 2026-04-18 round, already documented in `label_free_universal_rule_attempts_2026_04_18.md`)
- AUROC 0.50-0.66 across judges × datasets
- Best 1/4 V1

### InternVL3-14B stage-1 (dropped)
- vLLM 0.11.0 + InternVL3 multi-image chat template bug
- `Batch failed`: 1672, `single failed`: 3338 (every call)
- All outputs `{"score":null,"skipped":true}`
- Job 8821 COMPLETED 4h30m wall-clock, 0 usable scores

## Artifacts

- Main grid: `results/boundary_rescue/grid_eval/grid_raw.jsonl` (1344 cells), `grid_summary.json`, `grid_summary_top.md`
- Stage-1 scores: `results/holistic_<slug>/<ds>/<split>_binary.jsonl`
- Threshold summary: `results/boundary_rescue/threshold_search_summary.{json,md}`
- Per-slug pinned baseline: `results/boundary_rescue/v2_baseline_<slug>_protocol.json` (for slug≠2b)
- Per-slug entropy band: `results/boundary_rescue/<ds>/candidates_entropy_band_<slug>.jsonl`
- Per-slug baseline preds: `results/boundary_rescue/<ds>/baseline_preds_v2_<slug>_<crit>.jsonl`
- Judge outputs: `results/boundary_rescue/<ds>/offline_test_<judge>.jsonl` (standard prompt), `offline_test_ih_<judge>.jsonl` (IH-prompt for ImpliHateVid, 8/8 judges ≥383/401 coverage)
- Code: `src/boundary_rescue/{threshold_search,baseline_preds_v2,select_entropy_band,grid_eval_all}.py`
- Sbatch: `scripts/run_holistic_{qwen7b,gemma12b,gemma12b_16f,pixtral12b,minicpm26,internvl14b}.sh`
