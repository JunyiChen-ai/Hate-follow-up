# Follow-up post for Reviewer 7rqV thread — SAGE reproduction results

*(post as an official comment in the 7rqV thread, referencing our W2/W3 promises; self-contained)*

---

As promised in our response to W2, we have completed the protocol-matched comparison with SAGE (Huang et al., ACL 2026), the supervised baseline the reviewer named, using the authors' released implementation trained and evaluated on the exact fixed splits used throughout our paper. Results (ACC / macro-F1):

| Method (labels used) | HateMM | MHClip-EN | MHClip-ZH |
|---|---|---|---|
| SAGE, authors' code on our fixed splits (supervised) | 79.5 / 78.3 | 72.7 / 68.4 | 67.8 / 66.5 |
| SAGE, published under its own protocol* (supervised) | 87.1 / 86.3 | 83.8 / 79.6 | 79.0 / 74.8 |
| TRIAGE (label-free) | **86.0** / **85.3** | **79.5** / **73.2** | **83.9** / **81.9** |

*SAGE's published numbers use a random 7:1:2 HateMM split and the original MultiHateClip partitions with multi-seed averaging, so they are not directly comparable to fixed-split results; SAGE's own paper documents this protocol sensitivity (their HCC1 re-run: 78.3 under their split versus HCC1's reported 85.4 on its fixed split). Our reproduction: authors' code and hyperparameters verbatim, our fixed validation splits for model selection, our common ASR transcripts instead of Whisper-turbo, single seed.

This result also sharpens the answer to W3 (backbone attribution and why label-free can compete with supervision). The reproduced SAGE is a fair head-to-head: the same test videos, a recent supervised architecture, and in-domain labeled training — yet TRIAGE leads on all three benchmarks with zero labels. The MHClip-ZH cell shows the mechanism: all three of SAGE's supervised experts saturate at ~0.68 accuracy (text 0.678, audio 0.671, vision 0.678 — none dead or dominant), so the binding constraint is the hate- and culture-specific knowledge learnable from a few hundred labeled Chinese videos. TRIAGE's verifiers inherit that knowledge from large-scale multimodal pretraining, and the framework's contribution — the part removing it costs 2.9–6.3 points in Table 2 — is converting it into calibrated, selectively-spent decisions without consulting labels. We will include SAGE (published and reproduced, with protocol notes) in the revised supervised group.
