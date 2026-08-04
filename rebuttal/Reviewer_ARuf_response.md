We thank the reviewer for the careful reading and for recognizing the practical relevance of label-free hateful video detection, the per-component analyses, and the effectiveness–efficiency validation. We address the three concerns below.

**W1: Does the method extend to settings where hateful videos are a severe minority (<10%)?**

This is an excellent question, and we ran the experiment. The only component that could break under prevalence shift is the Boundary Mapper's unsupervised calibration, which fits a two-component model to the pooled score distribution. We therefore resampled every dataset to hateful rates of 5%, 10%, and 20% (keeping all normal videos, subsampling hateful ones; 30 random subsamples per setting) and re-ran the entire label-free pipeline from scratch on each subsampled pool — threshold fitting, boundary-region localization, and adaptive resolution. Mean macro-F1 over the four datasets:

| Hateful rate | Mapper only | TRIAGE | Qwen2.5-VL-72B on every video | Gemma-3-27B on every video |
|---|---|---|---|---|
| 5% | 57.6 | 62.3 | 63.6 | 62.1 |
| 10% | 65.7 | **69.8** | 68.4 | 68.7 |
| 20% | 72.3 | **77.0** | 70.8 | 73.6 |
| natural | 75.6 | **80.8** | 69.6 | 74.8 |

Three findings. (1) The label-free calibration does not collapse: at 5–10% prevalence the fitted threshold and boundary region remain usable, and performance degrades smoothly rather than catastrophically. (2) TRIAGE's mechanism survives imbalance: the gain over its own mapper persists at every prevalence (+4.7 macro-F1 at 5%, +4.1 at 10%, +4.7 at 20%), i.e., the boundary region still concentrates mapper errors and arbitration still repairs them. (3) TRIAGE remains the strongest method at 10% and 20%. At the most extreme 5% setting, running the largest verifier (Qwen2.5-VL-72B) on *every* video edges ahead by 1.3 macro-F1 — a 72B forward pass per video, versus TRIAGE's single 2B pass plus at most 0.8 verifier calls per video on average — and that uniform-72B approach is the weakest of the four methods at 20% and at natural prevalence. We will add this study, with per-dataset tables and standard deviations, to the revision.

**W2: Technical concepts borrowed from existing literature; broaden related work on multi-(M)LLM systems.**

We agree that the individual ingredients (next-token probability readout, mixture calibration, sequential testing, LLM judging) each exist in prior literature, and we do not claim otherwise. The contribution is the structure that connects them *without labels*: (a) the routing signal is derived from the unlabeled pool's own score distribution, so the framework decides *where* to spend stronger reasoning with no supervision; and (b) resolution is a sequential Bayesian update over heterogeneous verifiers with label-free stopping, not repeated sampling. These choices are load-bearing rather than decorative: as reported in Table 2 of the paper, removing pool-based calibration costs 6.3 accuracy points on average, replacing entropy-based selection with a random subset of the same size costs 2.4, and replacing heterogeneous verifiers with self-consistency over one strong model costs 2.9. A generic coarse-to-fine cascade specifies none of these mechanisms. We will expand the related-work section with a discussion of multi-(M)LLM application systems (LLM-as-judge pipelines, routing/cascade systems, multi-agent verification) and position our design against them explicitly.

**W3: Venue fit.**

We respectfully note that hate-speech detection is an ARR/ACL research-area keyword (our submission is under Computational Social Science with hate-speech detection as a keyword), and that the hateful signal in these benchmarks is predominantly carried by language — speech transcripts, on-screen text, and their interaction with visuals; MHClip-ZH additionally tests cross-lingual transfer of moderation policy. The *ACL community has an active line of work on multimodal hate (Hateful Memes, HateMM, and ImpliHateVid, the latter published at ACL 2025). We agree the final judgment belongs to the ACs, and we appreciate the reviewer flagging it transparently.

We hope these results address the reviewer's concerns, and we welcome further questions during the discussion period.
