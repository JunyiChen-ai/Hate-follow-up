# Strategy Plan — Submission 9795 (TRIAGE) rebuttal

## Global themes (shared across reviewers)

- **T1 — "Method, not backbone."** (7rqV-C3, cpjh-C1, ARuf-C2) Every backbone TRIAGE uses is already a zero-shot row in Table 1 and every one is far below TRIAGE; the gain is the routing + arbitration structure. Strengthen with E3 (strongest single backbone, 72B, applied to every video, still below TRIAGE at higher cost) and existing ablations (w/o Verification, w/ Self-Con).
- **T2 — "The framework is scale- and prevalence-robust."** (cpjh-C3, ARuf-C1) E2 shows an all-small verifier panel retains most of the gain (deployable on a single 24–48GB GPU); E1 shows the label-free calibration survives severe class imbalance (or characterizes where it degrades — report honestly either way).
- **T3 — "Baseline set is fair and current."** (cpjh-C1/C2, 7rqV-C2) Clarify max scale used by each label-free baseline; verify SAGE and recent supervised methods; add what is real and reproducible, cite what is not comparable, and say why.
- **T4 — "Novelty is the label-free routing signal, not the pipeline shape."** (7rqV-C1, ARuf-C2) Coarse-to-fine is common; deriving a calibrated routing signal *without labels* from the pool score distribution, plus Bayesian sequential arbitration with early stop, is the contribution; the ablation table is the falsification test (w/o Calibration −6.3, w/ Random −2.4, w/ Self-Con −2.9).

## Pivotal reviewers
- **cpjh** (confidence 5, overall 3.5, all three concerns addressable with experiments) — highest leverage.
- **7rqV** (excitement 2.5 — risk of drag-down; C2/C3 addressable with evidence).

## Experiments (Phase 3.5 evidence sprint)

### E1 — Prevalence-shift robustness (ARuf-C1) [CPU expected]
Subsample each test pool to hateful prevalence π ∈ {5%, 10%, 20%, natural}, keeping all normal videos and subsampling hateful ones (repeat over ≥10 seeds). Re-run the ENTIRE label-free pipeline on each subsampled pool: stage-1 scores (frozen per-video) → calibration fit on the subsampled pool → entropy band → sequential resolver using frozen per-video verifier outputs. Report ACC/M-F1 (and mapper-only vs full TRIAGE) per π.
- **Prediction**: band mechanism still concentrates errors; absolute M-F1 drops for everyone at low π (fewer positives), but TRIAGE's delta over mapper-only persists. **Disconfirmation**: if calibration collapses at π=5–10% (band swallows everything or nothing), report the failure mode + the natural mitigation (prior-anchored fit), honestly.
- **Blocker check**: requires verifier outputs on ALL test videos (else newly-banded videos lack votes) — inventory agent verifying.

### E2 — Small-verifier-only panel (cpjh-C3) [CPU expected]
Extract from the existing 720-config order sweep the panels composed only of {Qwen3-VL-8B, InternVL3.5-8B, Gemma-3-12B} (all ≤12B; largest fits a single consumer-class GPU). Report best/mean per-dataset ACC/M-F1 + calls/video, alongside default panel and best label-free baseline. Add peak-VRAM based deployment discussion.

### E3 — Strongest-backbone control: Qwen2.5-VL-72B-AWQ zero-shot on all test videos (cpjh-C1, 7rqV-C3) [GPU if missing]
Single-pass zero-shot row for the largest model in our pool on all 4 datasets, same protocol as other zero-shot rows. If TRIAGE (2.x calls/video, mostly 2B) beats 72B-everywhere, the backbone-strength explanation is refuted directly.
- Coverage check first (verify-all artifacts may already contain it).

### E4 — SAGE + recent supervised baselines (7rqV-C2, cpjh-C2) [TBD]
Gate on lit-check: (a) SAGE exists + video task + code → reproduce under our protocol; (b) exists but meme/text-only or no code → cite + explain non-comparability + offer camera-ready comparison; (c) not verifiable → say so politely, ask reviewer for the exact citation.

## Blocked claims
- None yet. All numbers must trace to results/rebuttal/ or existing frozen artifacts.

## Response budget (per_reviewer_thread)
- ARuf ≈ 3000-4000 chars; cpjh ≈ 4500-5000 chars (pivotal, 3 experiments); 7rqV ≈ 4500 chars.
