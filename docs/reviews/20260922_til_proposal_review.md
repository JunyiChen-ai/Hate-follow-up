# TIL proposal review (rule 4) and code review (rule 6), 2026-09-22

Verbatim summaries of the two independent-agent reviews of `experiments/20260922_til/`.

## Proposal review: 放行

Not pure smoothing: grid B is a new set of MLLM reads on new intervals; the interval-support observation model
uses the measurement geometry; the arm table makes "averaging + smoothing" (A3) the null the method must beat.
A1 alone (one grid + prior) would fall under rule-4 case (3) and can only be an explicit component of SPVL-r2.

Literature searched (queries → closest hits): hateful video temporal localization + HMM/duration prior →
HateClipSeg (2508.01712; ActionFormer baseline), MultiHateLoc (2512.10408; weakly supervised MIL, training-time
smoothing, no shifted grids, no HMM). LELA (2602.09637): per-frame independent scores, max over modalities, no
cross-segment inference. MM-HSD (2508.20546), TANDEM (2601.11178): trained LSTM/TCN. Training-free TAL with
MLLMs + shifted/overlapping windows → FreeZAD (2501.13795), OZ-TAL (2605.09976); closest window + state machine
is audio deepfake localization (2609.10051). LAVAD (2404.01014): LLM temporal summary + similar-frame averaging;
T3AL (2404.05426): moving average + mean threshold + grouping; VADTree (2510.22693): non-overlapping hierarchy;
CoReVAD (2605.23116). Weakly supervised TAL with HMM/Viterbi → NN-Viterbi, TASL (ICCV21), MLLM4WTAL
(2411.08466): trained action segmentation. HSMM + VLM scores: none. Test-time shifted grids over VLM reads /
noisy-sensor interval posterior: none. Conclusion: shifted-grid interval reads + interval-support explicit
inference has not been used for hateful video localization.

Required fixes before running (all applied in the README): (1) "pooled expected unchanged" is wrong — the
residual moves pooled; report and gate all three; (2) A1 − A0 mixes scaling and prior → arm A1b added; (3) grid B
recomputes z_video for the stance turn → call count corrected; (4) D = 80 s is development-selected; (5) the
fallback (A4 ≈ A3) is a component of SPVL-r2, not a new method; gate against the STATUS row; (6) s_m per corpus
is label-free and rule-3 compatible but transductive beyond rule 13's examples → arm A4p (pooled std) added.
Non-blocking: HVL loss is .03 on HateMM and .04–.07 on HCS; "noisy-OR" is a deterministic OR support; first and
last cells have single-grid coverage; rank compression of VLM answers (2608.21244) relates to the scaling.

## Code review: no blocking finding

A0 reproduces SPVL-r2 exactly (333 frame curves identical). Pair-state forward–backward verified by brute-force
enumeration for n = 1..10 (max posterior error 8.5e-10, from the 1e-9 clip). chain2 equals chain_pair with
single observations (8e-12). --dwell 0 returns the cell values exactly. No GT read in the scoring path; the
evaluator is the shared one. Findings: arms use the intersection of successful videos of the runs they read, so
all runs must have the same successful set (check n = 333); B1's intercept comes from grid B (within only);
s_m differs slightly between one-run and two-run arms (third decimal); run_measure.sh could exit before writing
RUN_FAILED when no log line matches the grep (fixed with `|| true`); the verify gate does not rerun on resume;
posterior clip at 1e-9 caps |log-odds| at 20.7 (never reached on cached data).
