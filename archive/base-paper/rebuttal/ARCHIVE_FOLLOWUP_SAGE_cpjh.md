# Follow-up post for Reviewer cpjh thread — SAGE reproduction results

*(post as an official comment in the cpjh thread, referencing our W2 promise)*

---

As promised in our response to W2, we have completed the protocol-matched reproduction of SAGE (Huang et al., ACL 2026) using the authors' released implementation, trained and evaluated on the exact fixed splits used throughout our paper (test sets: HateMM 215, MHClip-EN 161, MHClip-ZH 149). Results (ACC / macro-F1):

| Method (labels used) | HateMM | MHClip-EN | MHClip-ZH |
|---|---|---|---|
| SAGE, authors' code on our fixed splits (supervised) | 79.5 / 78.3 | 72.7 / 68.4 | 67.8 / 66.5 |
| SAGE, published under its own protocol* (supervised) | 87.1 / 86.3 | 83.8 / 79.6 | 79.0 / 74.8 |
| TRIAGE (label-free) | **86.0** / **85.3** | **79.5** / **73.2** | **83.9** / **81.9** |

*SAGE's paper uses a random 7:1:2 HateMM split and the original MultiHateClip partitions, with multi-seed averaging — not directly comparable to fixed-split numbers. The magnitude of the protocol effect is documented in SAGE's own paper: their re-run of HCC1 yields 78.3 on their split versus HCC1's reported 85.4 on its fixed split.

Under the common fixed-split protocol, TRIAGE exceeds the reproduced SAGE on all three benchmarks without using any labels. On MHClip-ZH we additionally verified the gap mechanistically: all three of SAGE's supervised experts saturate at ~0.68 accuracy (text 0.678, audio 0.671, vision 0.678 — none dead or dominant), indicating that the limiting factor is the hate- and culture-specific knowledge learnable from a few hundred labeled Chinese videos — knowledge TRIAGE's verifiers inherit from large-scale multimodal pretraining rather than from target-domain labels, which is precisely the label-efficiency gap our paper targets.

Reproduction protocol (full details will be in the revised appendix; per-video predictions and diagnostics are archived): authors' code and hyperparameters verbatim (AdamW lr 1e-4, wd 5e-5, batch 16, 30 epochs, warmup+cosine, best-validation selection; two minimal fixes required for the code to run, neither touching the method); our fixed validation splits for model selection; transcripts from our common ASR pipeline rather than Whisper-turbo (the only substantive deviation, affecting the text expert's input); single seed (seed 0). We will add SAGE with both its published and reproduced numbers, plus these protocol notes, to the supervised group in the revision.
