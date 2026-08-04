# Literature verification report — SAGE + recent hateful-video baselines

Produced by lit-check subagent, 2026-07-09. All claims URL-backed per subagent; items marked "verify" have confirmed titles/venues but unextracted metric tables.

## ⚠ CORRECTION (2026-07-09, user-supplied): SAGE DOES exist

**SAGE: Synergistic Adaptive Gating of Experts for Hateful Video Detection** — Huang, Liao, Wang, Li, Wang, Jiang, Li, Wang. ACL 2026 Long Papers, https://aclanthology.org/2026.acl-long.817/. **Supervised** multimodal framework (disentangled experts + instance-level decision arbitration for the "feature dilution" problem). Evaluated on **HateMM and MultiHateClip**. Abstract claims "accuracy gains of 6.37% to 21.23% and macro-F1 gains of 6.77% to 28.01%". Part 1 below is superseded: the initial accepted-papers sweep missed this paper. Deep extraction (exact tables, split protocol, code availability) in progress → see LIT_SAGE_REPORT.md when it lands.

Impact on strategy: SAGE is supervised, so it belongs in the supervised group (cpjh-C2 + 7rqV-C2 both partially answered by one comparison). The label-free-SOTA claim is not directly challenged; the "rivals supervised" claim must be re-checked against SAGE's numbers under protocol comparability.

## Part 1 — [SUPERSEDED — see correction above] "SAGE (ACL 2026)": CANNOT be verified as a hateful-content baseline

Checked ACL 2026 accepted-papers list (main + Findings) directly plus targeted searches. **No ACL 2026 paper named "SAGE" concerns hateful/harmful video, meme, or hate-speech detection.** Only two SAGE papers in the ACL 2026 program, neither relevant:
- SAGE: Search-AuGmented Evaluation of LLMs on Free-Form QA (ACL 2026 main) — LLM QA eval
- SAGE: Sparse Adaptive Guidance for Dependency-Aware Tabular Data Generation (ACL 2026 main) — tabular synthesis

Other "SAGE" hits, all unrelated: SAGE LLM Safety Evaluation (EMNLP 2025 Industry, aclanthology.org/2025.emnlp-industry.2), SAGE Self-Aware Guard Enhancement (arXiv 2505.12060, jailbreak defense), SAGE long-video reasoning agent (arXiv 2512.13874), SAGE tool-augmented multi-agent (arXiv 2601.09750).

Sources: https://2026.aclweb.org/program/accepted_papers/ , https://2026.aclweb.org/program/findings/

Recommended line: state we searched the ACL 2026 program and found no SAGE hateful-video/meme baseline, politely ask for a citation, and add concrete recent baselines to address the underlying concern.

## Part 2 — Recent hateful-VIDEO methods (2025–2026), verified

| Method | Venue/date | Type | Datasets + numbers | Code |
|---|---|---|---|---|
| **MARS** arXiv 2601.15115 | Jan 2026 | Training-free MLLM reasoning (Qwen2.5-VL-32B + LLaMA4/GPT5/Gemini) | HateMM ACC 75.8 / M-F1 75.8; MultiHateClip-ZH ACC 75.9 / M-F1 71.3. No ImpliHateVid | github.com/Multimodal-Intelligence-Lab-MIL/MARS |
| **MM-HSD** arXiv 2508.20546 (EPFL/Idiap) | Aug 2025 | Supervised, 4-modality cross-modal attention + late fusion | HateMM ACC 0.878 / M-F1 0.874 / F1(hate) 0.853. No MultiHateClip | github.com/idiap/mm-hsd |
| **HCC1** (Koushik et al.) | 2025 | Supervised late fusion (HateXplain+CLIP+CLAP) | HateMM ACC 0.854 / M-F1 0.848 | verify |
| **MultiHateLoc** arXiv 2512.10408 | Dec 2025 | Supervised temporal MIL, localization | SOTA on HateMM + MultiHateClip localization | verify |
| **ImpliHateVid two-stage contrastive** aclanthology 2025.acl-long.842 | ACL 2025 | Supervised two-stage contrastive | Origin of ImpliHateVid benchmark | verify |
| **Decoding Multimodal Cues** arXiv 2606.11953 | Jun 2026 | Explainable; introduces Ex-HateMM, Ex-ImpliHateVid | explainable variants | verify |
| TANDEM (2601.11178), Reasoning-Aware Multimodal Fusion (2512.02743), Cross-Modal Transfer memes→videos (WWW 2025, 2501.15438) | 2025-26 | Supervised | HateMM / MultiHateClip | mixed |

Note: repo already reproduces MARS (src/mars_repro), and MARS is already a Table 1 row. MM-HSD (~0.878 ACC) and HCC1 (~0.854 ACC) are the strongest supervised HateMM comparators for cpjh-C2.

## Part 3 — Post-training LLM moderation methods (7rqV-C2)

- **MuPHI / MuPHIRM** arXiv 2605.29951 — reasoning-augmented training via multi-perspective reward optimization (RL-style post-training) for multimodal harm; image-text VLM, not video.
- **ExPO-HM** arXiv 2510.08630 (ICLR 2026) — explain-then-detect via policy optimization for hateful memes; also arXiv 2606.15307 (RL with CoT supervision).
- **RGCL / RA-HMD** (ACL 2024 / EMNLP 2025 Oral, github.com/JingbiaoMei/RGCL) — retrieval-guided contrastive + robust LMM adaptation for hateful memes; code public.

Honest framing: these post-training moderators are meme/image-text and require labeled hate data for the post-training step — the supervision our label-free method avoids. MARS is the only video member of this family and is already compared.

## Bottom line
1. No verifiable "SAGE (ACL 2026)" hateful-content baseline — say so, ask for the cite, pivot to additions.
2. Add MM-HSD / HCC1 (supervised, HateMM) to the supervised group discussion; MARS already covers recent label-free.
3. Cite MuPHI / ExPO-HM / RA-HMD as post-training family, noting meme-domain + label-dependence.

---

## Precision pass (2026-07-09) — exact numbers + split protocols

Key headline: **only HCC1 shares our fixed-split protocol; MM-HSD and MARS both use 5-fold CV → NOT directly comparable to our Table 1.**

### MM-HSD — arXiv 2508.20546; published ACM MM 2025 (dl.acm.org/doi/10.1145/3746027.3754558)
- HateMM (their Table 2): ACC 0.878±0.009; macro-F1 0.874±0.009; Hate-F1 0.853±0.009; Hate-P 0.849±0.017; Hate-R 0.857±0.000.
- Protocol: **5-fold CV** on 85% of data, 15% held out; folds of 698 train / 175 val (§5.1). NOT our fixed 215-video split — must be footnoted if cited.
- 4 modalities incl. audio: Detoxify/RoBERTa (~109M) transcript, wav2vec2-large-xlsr-53 (~315M) audio, ViT (~86M) frames, PaddleOCR+Detoxify on-screen text; 4.6M-param classifier. HateMM only.

### HCC1 — "Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Content", Koushik, Kanojia, Treharne; WWW Companion 2025 (arXiv 2502.07138)
- HCC1 = HateXplain + CLAP + CLIP, concatenation fusion (their Table 2a).
- HateMM: **ACC 0.854 / macro-F1 0.848** (Table 2a), **fixed split** 779 train / 87 val / **217 test** → protocol-comparable to our Table 1 (cleanest supervised comparator; below TRIAGE 86.5/85.8).

### ImpliHateVid framework — Rehman et al., ACL 2025 long (2025.acl-long.842; arXiv 2508.06570)
- Binary (their Table 2): ACC 87.53% / F1 87.73%; split 1283/325/401. Our Table 1 "ImpliHateVid 87.5" row is **verified faithful**. (Three-class Table 3 M-F1 69.18% is a separate task.)

### MARS — arXiv 2601.15115
- MultiHateClip = **Chinese subset only** (959 videos), **5-fold CV** 70/10/20 (~192 test/fold, §3.1) → not our 149-video fixed ZH test. ZH: ACC 75.9±0.032 / M-F1 71.3±0.036; HateMM (1083 videos, 5-fold): 75.8±0.021 / 75.8±0.021. No ImpliHateVid.
- Note: our Table 1 MARS row is our own reproduction under OUR fixed splits — different from their published protocol; keep the distinction explicit in the response.

### Usage guidance
- Directly quotable on HateMM (same protocol family): HCC1 0.854/0.848 < TRIAGE 86.5/85.8.
- Quote-with-footnote (5-fold CV): MM-HSD 0.878/0.874; MARS published numbers.
