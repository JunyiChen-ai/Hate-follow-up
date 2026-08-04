# SAGE (ACL 2026 long.817) — deep extraction

Produced by lit-check subagent, 2026-07-09, from https://aclanthology.org/2026.acl-long.817/ (pp. 17953–17963).

## Identity
"SAGE: Synergistic Adaptive Gating of Experts for Hateful Video Detection" — Jie Huang, Xin Liao, Junjie Wang, Mingyang Li, Wenshuo Wang, Ziyou Jiang, Shoubin Li, Qing Wang. ACL 2026 Long.

## Method: SUPERVISED mixture-of-experts
- Three disentangled unimodal experts — RoBERTa (text), MFCC (audio), VideoMAE (vision) — global expert deliberation, instance-level "tribunal" gating. Trained end-to-end: cross-entropy + expert-supervision + load-balancing losses (Appendix B, Eqs. 11–14). AdamW lr 1e-4, batch 16, 4× RTX A6000 48GB (Appendix E). Multi-seed + paired t-test.
- **Requires labeled training data on each target dataset** — exactly the supervision TRIAGE's setting excludes.

## Datasets & protocol
- HateMM + MultiHateClip only. **No ImpliHateVid.** MultiHateClip split into MHClip-YouTube (EN, 830 videos: 67H/190O/573N) and MHClip-BiliBili (ZH, 839: 100H/168O/571N); binary = Hateful+Offensive vs Normal (same collapse as ours).
- **HateMM: random 7:1:2 split (~216 test), multi-seed — NOT our fixed 215-video split.** MHClip: "follows the original data splitting strategy (Wang et al., 2024b)" — same family as our splits; exact partition identity UNCONFIRMED.

## SAGE numbers (their Table 1, ACC / M-F1 / M-P / M-R; Inf. 567.98 ms)
- HateMM: **0.8710 / 0.8628** / 0.8711 / 0.8572
- MHClip-YouTube (EN): **0.8375 / 0.7962** / 0.8055 / 0.7887
- MHClip-BiliBili (ZH): **0.7901 / 0.7484** / 0.7551 / 0.7430

Vs TRIAGE (our fixed splits): HateMM 86.5/85.8 (SAGE +0.6 ACC, different split); MHClip-EN 79.5/73.2 (SAGE-YT +4.3 ACC, split caveat); MHClip-ZH 83.9/81.9 (**TRIAGE +4.9 ACC**); ImpliHateVid — SAGE absent.

Their strongest baselines (ACC, HateMM / YT / Bili): MM-HSD 0.8203/0.7438/0.6914; MoRE 0.8110/0.7400/0.7500; HCC1 re-run **0.7834** (vs HCC1's own 0.854 on its fixed split → HateMM numbers are highly split-sensitive — important caveat for all cross-paper comparisons). MLLM zero-shot rows all far below (best Keye-VL-8B 0.7872 HateMM).

"6.37%–21.23% gains" in abstract = absolute pp averaged over ALL baselines per category (includes weak unimodal); margin over strongest single baseline is smaller (HateMM +5.07 over MM-HSD).

## Code
**No GitHub/checkpoint URL anywhere in the paper.** Reproduction on our fixed splits = full reimplementation + MoE training (4×A6000-scale) — not feasible in the rebuttal window.

## Rebuttal positioning (honest)
- SAGE is supervised, in-domain-trained; belongs in Table 1's supervised group, not the label-free group. TRIAGE's claims: (1) label-free SOTA — untouched by SAGE; (2) "outperforms best supervised on 3 datasets" — must be softened w.r.t. SAGE: under SAGE's (different-split) numbers, TRIAGE is −0.6 HateMM, −4.3 EN, +4.9 ZH, and SAGE has no ImpliHateVid result.
- Do NOT claim we beat SAGE. Frame: TRIAGE approaches/matches a fully-supervised ACL 2026 SOTA with zero labels, wins on ZH, and covers ImpliHateVid where SAGE reports nothing; annotation cost of SAGE's pipeline recurs per deployment surface.
- Commit in revision: add SAGE (cited numbers + split-protocol footnote) to the supervised group; discuss split sensitivity (SAGE's own HCC1 re-run 0.7834 vs 0.854 original).

UNCONFIRMED: expert param counts; exact MHClip test sizes; code release; split identity with ours.

---

## ⚠ CORRECTION 2 (2026-07-09, user-supplied): SAGE code IS public

The paper's footnote (missed by our PDF extraction — footnotes were not parsed) states: "The SAGE implementation is available at https://github.com/XinLiao04/SAGE".

Repo verified 2026-07-09: matches the paper; complete training pipeline (model/Main.py, Trainer.py, Runner.py), preprocessing (16-frame extraction, audio extraction, audio→transcript), config-driven (config/config.yaml), supports HateMM (7:1:2) and MultiHateClip (Bilibili & YouTube). Python 3.11 + PyTorch. No pretrained checkpoints (best-val model saved during training). No license file.

**Revised E4 decision: REPRODUCE SAGE on our fixed splits** (HateMM 215-test, MHClip-EN 161, MHClip-ZH 149; ImpliHateVid as stretch). Expert backbones are ~100-300M and the datasets are ~1k videos each, so training fits comfortably on 1 GPU. This upgrades the 7rqV/cpjh response from cite-with-footnote to a protocol-matched supervised comparison.

**Process lesson (recorded):** two "not found" claims about SAGE (existence; code URL) were both wrong — absence-of-evidence from a single retrieval pass must never be reported as evidence-of-absence in reviewer-facing text.

## Second source (2026-07-09, user-supplied): hjandlm/SAGE

There are TWO GitHub repos: `XinLiao04/SAGE` (implementation, cloned to external_repos/SAGE — code verified git-tracked on main) and `hjandlm/SAGE` (first-author repo: README + paper PDF only, NO code). The hjandlm README corroborates our PDF extraction: SAGE 87.10/86.28 HateMM, 83.75/79.62 MHClip-YT, 79.01/74.84 MHClip-BB; MoRE 81.10/80.04 strongest multimodal baseline; ablation table (w/o Linguistic Expert 78.34, w/o Tribunal 84.33, etc.); datasets HateMM 428H/652N=1080, MHClip-YT 830, MHClip-BB 839; Whisper-Turbo ASR; RoBERTa=cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual; VideoMAE-base; L=2, K=2; GPT-4V 7654ms/sample vs SAGE ~0.57ms overhead. Note: README says gains "6.64% to 21.23%" (paper abstract said 6.37%) — do not quote the range; quote absolute numbers only.

---

## E4b reproduction — HateMM seed0 verdict (2026-07-10)

**SAGE on OUR fixed HateMM split (215 test, seed 0, authors' code): ACC 79.5 / M-F1 78.3 / M-P 79.0 / M-R 77.9** (results/rebuttal/E4_sage/HateMM/seed0/metrics.json; best_val_f1 0.763 @ ep10).

Sanity check (sage-repro, three-part): (1) healthy single-peak convergence (val arc 0.39→0.76 by ep7-10, mild overfit after; not undertrained); (2) predictions non-degenerate (78/137 pred split vs 86/129 true; prob_pos spans 0.0004-0.9995, not saturated; failure mode = Hate recall 0.70, the honest error); (3) the number lands exactly in the band SAGE's own paper documents for supervised HateMM under a matched protocol (their HCC1 re-run: 85.4→78.34). Gap attribution ranked: split protocol ~4-6pp > transcript source ~1-3pp > single seed ±1-2pp; possible train/test near-duplicate leakage in random splits is plausible but UNCONFIRMED — do not assert.

Reviewer-facing phrasing (agreed): "under the common fixed split, protocol-matched SAGE reaches 79.5 ACC, consistent with the split-sensitivity SAGE's own paper documents; TRIAGE reaches 86.0 on the identical 215-video test set with zero labels." Footnote transcript deviation; Whisper-turbo re-run available (~1 GPU-h) if pressed. Do NOT phrase as "SAGE is weak". Single-seed per user directive 2026-07-10.

Pending: MHClip-YT (16363) — decisive cell (SAGE paper led EN 83.75 vs our 79.5); MHClip-BL (16362) — expected TRIAGE win.

## E4b FINAL (2026-07-10): all three cells table-ready

SAGE (authors' code, our fixed splits, seed 0): HateMM 79.5/78.3, MHClip-EN 72.7/68.4, MHClip-ZH 67.8/66.5 — TRIAGE (86.0/79.5/83.9) leads on all three. Verdicts: HateMM ready; EN ready w/ single-seed footnote (80-video val selection noise); ZH ready — diagnostic falsified the sentiment-noise hypothesis: all three experts uniform at ~0.68 (text 0.678/audio 0.671/vision 0.678), genuine annotation-scale ceiling. Mechanistic extras (per-video evidenced, keep as backup, do not foreground): GED deliberation homogenizes experts (argmax agreement 91-97%); gate rides one modality (fusion==text on 100% of HateMM/EN videos; on HateMM gate puts 0.928 on text while audio/vision test-generalize better, ~2.4pp cost). Artifacts: results/rebuttal/E4_sage/<ds>/seed0/{metrics,predictions,diag}.json. Follow-up drafts: rebuttal/FOLLOWUP_SAGE_{cpjh,7rqV}.md.

## E4c ImpliHateVid + four-dataset diag (2026-07-11, FINAL)

IH seed0: ACC 85.8 / M-F1 85.8 (> TRIAGE 82.8) — cleanest cell (val 0.8585 ≈ test 0.8579; symmetric confident predictions; 1.7pp below IH's bespoke framework 87.5). Final: TRIAGE 3-1 vs protocol-matched SAGE.

Four-dataset expert/gate diag (backup evidence, per-video files in results/rebuttal/E4_sage/*/seed0/diag.json):
- ZH: experts uniform ~0.68 (text .678/audio .671/vision .678), gate text .459/vision .518 — capped by scarce, culturally-hard labels.
- EN: experts .70-.73, gate rides text (.79).
- HateMM: experts .795-.819, gate rides text (.928) though audio/vision test better.
- IH: experts ALL strong (text .855/audio .850/vision .838), fusion .858, gate genuinely distributed (.233/.467/.300).
Mechanism contrast (do not foreground; backup): same architecture — with ample balanced in-domain labels (IH 1283) every expert learns and the tribunal truly fuses; with scarce/culturally-hard labels (ZH 579) all experts cap at ~0.68. This is the annotation-scale thesis rendered in SAGE's own components.
