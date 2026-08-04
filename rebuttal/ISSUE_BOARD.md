# Issue Board — Submission 9795 (TRIAGE), ACL ARR 2026 May

Venue mode: `per_reviewer_thread` (OpenReview ARR author response, one comment per reviewer).
Round: initial rebuttal. Status legend: open / answered / deferred / needs_user_input.

---

## Reviewer ARuf (Overall 3, Soundness 4, Confidence 3) — stance: positive-leaning

### ARuf-C1 — Low-prevalence (severe minority) setting
- **raw_anchor**: "Can the proposed method be extended to settings where hate video constitutes severe minority ... (e.g., <10% of the sample set is hateful)?"
- **issue_type**: empirical_support | **severity**: major | **reviewer_priority**: pivotal (addressable, concrete ask)
- **response_mode**: grounded_evidence (`evidence_source: needs_experiment` → E1 prevalence-shift experiment)
- **why it bites**: the Boundary Mapper's label-free calibration fits a 2-component model to the pool score distribution; under 5–10% prevalence the positive mode shrinks and the fit could collapse. This is a real scientific question about the method, not a nitpick.
- **status**: open → E1

### ARuf-C2 — Novelty: borrowed concepts / engineering extension; broaden related work on multi-(M)LLM applications
- **issue_type**: novelty | **severity**: major
- **response_mode**: structural_distinction + commitment to expand §2 (related work on multi-LLM systems)
- **status**: open (text-only; no experiment)

### ARuf-C3 — Venue fit (*ACL vs CV venues)
- **issue_type**: other (venue fit) | **severity**: minor (explicitly delegated to ACs)
- **response_mode**: direct_clarification (task is language-centric: hate speech is an NLP problem; signal carried by speech/transcript/on-screen text; ARR keywords include hate-speech detection; ACL has hosted hateful memes/HateMM-line work)
- **status**: open (text-only)

---

## Reviewer cpjh (Overall 3.5, Confidence 5) — stance: swing, PIVOTAL reviewer

### cpjh-C1 — Were label-free/few-shot baselines evaluated at the same 72B max scale?
- **issue_type**: baseline_comparison | **severity**: major | **reviewer_priority**: pivotal
- **response_mode**: grounded_evidence (`evidence_source: partially needs_experiment` → E3)
- **facts**: Table 1 zero-shot rows already include Gemma-3-27B and Qwen2.5-VL-32B-AWQ (both below TRIAGE). Missing: a Qwen2.5-VL-72B-AWQ single-model zero-shot row on all 4 datasets = strongest single verifier applied to every video. Also: MARS/LoReHM/ALARM baselines run with their published backbones.
- **status**: open → E3 (72B zero-shot full-test) + clarification text

### cpjh-C2 — More recent supervised baselines
- **issue_type**: baseline_comparison | **severity**: major
- **response_mode**: grounded_evidence + nearest_work_delta (already include CMFusion 2025, MoRE, ImpliHateVid — check years; add any verifiable 2025-2026 supervised HVD method with public code → E4/lit-check)
- **status**: open → lit-check

### cpjh-C3 — TRIAGE with verifier pool restricted to small (7B/8B-scale) MLLMs; deployment hardware cost
- **issue_type**: empirical_support / practical_significance | **severity**: major | **reviewer_priority**: pivotal
- **response_mode**: grounded_evidence (`evidence_source: likely frozen` → E2: all-small verifier panels from the 720-config order sweep, e.g. {Qwen3-VL-8B, InternVL3.5-8B, Gemma-3-12B}; report per-dataset ACC/M-F1 + peak-VRAM/deployability discussion)
- **status**: open → E2

---

## Reviewer 7rqV (Overall 3, Excitement 2.5, Confidence 4) — stance: swing-negative

### 7rqV-C1 — Limited methodological novelty (coarse-to-fine = common paradigm)
- **issue_type**: novelty | **severity**: major
- **response_mode**: structural_distinction (what generic coarse-to-fine does NOT give you: label-free calibrated routing signal from the pool distribution; Bayesian sequential updating with verifier-effect scaling and early stop; band-error-concentration phenomenon specific to MLLM hate scoring — tie to ablation: w/o Calibration −6.3, w/ Random −2.4)
- **status**: open (text-only)

### 7rqV-C2 — Missing baselines: post-training LLM methods; SAGE (ACL 2026)
- **issue_type**: baseline_comparison | **severity**: major | **reviewer_priority**: pivotal
- **response_mode**: nearest_work_delta + narrow_concession (see rebuttal/LIT_SAGE_REPORT.md — earlier "SAGE not found" finding was WRONG, corrected by user)
- **finding 2026-07-09 (corrected)**: SAGE (ACL 2026 long.817) is a SUPERVISED MoE (RoBERTa/MFCC/VideoMAE + tribunal gating), HateMM + MultiHateClip only, no ImpliHateVid, no public code, HateMM under random 7:1:2 split. SAGE ACC: HateMM 87.10 / MHClip-YT 83.75 / MHClip-Bili 79.01 vs TRIAGE 86.5 / 79.5 / 83.9. Positioning: SAGE goes in the supervised group; TRIAGE is label-free, wins ZH, covers ImpliHateVid; soften "beats best supervised on 3 datasets" claim w.r.t. SAGE with split caveat. Post-training LLM moderators (MuPHI, ExPO-HM, RA-HMD) are meme/image-domain + label-dependent; MARS (already in Table 1) is the video representative.
- **status**: REVISED 2026-07-09 — SAGE code IS public (paper footnote, user-supplied: github.com/XinLiao04/SAGE; our PDF extraction missed footnotes). E4b: reproducing SAGE on OUR fixed splits (sage-repro engineer) → protocol-matched supervised comparison instead of cite-with-footnote. Cited numbers + split caveat remain the fallback if reproduction hits a wall before the response deadline.

### 7rqV-C3 — Why does label-free beat supervised? Backbone strength vs method?
- **issue_type**: empirical_support | **severity**: major | **reviewer_priority**: pivotal
- **response_mode**: grounded_evidence — decompose: (a) every individual backbone zero-shot row in Table 1 is far below TRIAGE (e.g., best single zero-shot HateMM 77.9 vs TRIAGE 86.5); (b) ablation w/o Verification (mapper only) and w/ Self-Con (same backbone, repeated) both well below; (c) E3 adds strongest-backbone-72B zero-shot on everything; (d) supervised methods' ceiling analysis: small train sets + train/test distribution. Experiment E3 completes this argument.
- **status**: open → E3 + analysis text

---

## Cross-reviewer experiment map

| Exp | Answers | What | Compute |
|---|---|---|---|
| E1 | ARuf-C1 | Prevalence-shift: subsample pools to 5%/10%/20%/natural hateful rate, re-run full label-free pipeline offline, multi-seed | CPU (if verifier outputs cover full test sets) |
| E2 | cpjh-C3 | TRIAGE with all-small verifier panel (≤12B / ≤8B), per-dataset table + VRAM discussion | CPU (frozen order-sweep) |
| E3 | cpjh-C1, 7rqV-C3 | Qwen2.5-VL-72B-AWQ zero-shot on all test videos of 4 datasets (strongest-backbone control) | GPU if not already covered |
| E4 | cpjh-C2, 7rqV-C2 | SAGE + recent supervised baselines: verify, cite, reproduce if feasible | TBD after lit-check |
