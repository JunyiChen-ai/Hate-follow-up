# Baseline reproduction list — weakly-supervised frame-level localization (2026-08-18)

Scope: methods trained with VIDEO-LEVEL labels only (no frame/timestamp
supervision), outputting per-frame/per-snippet scores; venue CCF-A or
CORE A*; open source with training code. Every repo URL below was
fetched and verified (HTTP 200 + README + metadata) on 2026-08-18.
Target use: retrain on HateMM (1083 videos, video labels for training,
span gold for evaluation) on one RTX 5090.

## Cost structure (applies to the whole table)

Model bodies are all tiny (0.3M–20M params; minutes to train on a
5090). The real cost is feature extraction: I3D 10-crop ≈ 4–8 h for
HateMM's 43 h of video; CLIP ViT-B/16 per-frame ≈ tens of minutes;
I3D+TV-L1 two-stream (WTAL line) adds tens of hours of optical flow —
the WTAL family's main practical blocker; VGGish audio ≈ minutes.
Environment trap: most repos pin torch 1.3–1.8 / cuda 10–11; RTX 5090
is Blackwell (sm_120) so everything must be ported to torch≥2.7+cu128
(model bodies are simple; apex-dependent repos — MIST, UMIL — need
apex→torch.amp surgery).

## Ranked table (reproduction priority = code health × task closeness × influence)

| # | Method | Venue/tier | Features | Repo (verified) | Health | Feasibility note |
|---|---|---|---|---|---|---|
| 1 | UR-DMU | AAAI'23 CCF-A | I3D 10-crop | github.com/henrryzh1/UR-DMU | 92★, 2025-01 | Only modern MIL repo WITH feature-extraction code included; shortest path to a HateMM branch |
| 2 | RTFM | ICCV'21 CCF-A | I3D 10-crop | github.com/tianyu0207/RTFM | 346★, 2025-10 | De-facto standard; 11 files; basis of UR-DMU/S3R/MACIL-SD |
| 3 | HL-Net/XDVioDet | ECCV'20 A* | I3D RGB + VGGish | github.com/Roc-Ng/XDVioDet | 137★, 2024-05 | Only A* audio-visual weakly-supervised per-segment scorer; matches HateMM's audio-carries-hate phenomenon; sparse README |
| 4 | VadCLIP | AAAI'24 CCF-A | CLIP ViT-B/16 | github.com/nwpu-zxr/VadCLIP | 235★, 2024-03 | Cheapest features (minutes); strongest modern CLIP-based baseline |
| 5 | PEL4VAD | TIP'24 CCF-A | I3D + CLIP text prompts | github.com/yujiangpu20/PEL4VAD | 113★, 2024-08 | Best engineering (ckpts + training logs); use as correctness anchor for the pipeline |
| 6 | MACIL-SD | MM'22 CCF-A | I3D + VGGish | github.com/JustinYuu/MACIL_SD | 42★, 2022-07 | 0.68M-param audio-visual; XD-only dataloader must be rewritten |
| 7 | MGFN | AAAI'23 CCF-A | I3D 10-crop | github.com/carolchenyx/MGFN. (trailing dot) | 96★, 2023-04 | Train/test complete; scrappy README |
| 8 | DeepMIL | CVPR'18 CCF-A | C3D/I3D/R3D | ekosman/AnomalyDetectionCVPR2018-Pytorch (194★, pushed 2026-08-17!) or Roc-Ng/DeepMIL | Most actively maintained repo in the table | ekosman fork = only full video→features→train chain; reports AUC 0.70 vs paper 0.75 |
| 9 | S3R | ECCV'22 A* | I3D (RTFM features) | github.com/louisYen/S3R | 81★, 2022-09 | Best README; dictionary-learning extra step; old env |
| 10 | DELU | ECCV'22 A* | THUMOS I3D two-stream | github.com/MengyuanChen21/ECCV2022-DELU | 49★, 2024-04 | Healthiest WTAL repo; blocker = optical flow |
| 11 | P-MIL | CVPR'23 CCF-A | two-stream + proposals | github.com/RenHuan1999/CVPR2023_P-MIL | 44★ | Needs upstream CO2-Net proposals first — double porting |
| 12 | CO2-Net | MM'21 CCF-A | two-stream | github.com/harlanhong/MM2021-CO2-Net | 42★, torch 1.3 | WTAL family's base; oldest env |
| 13 | UMIL | CVPR'23 CCF-A | NONE (end-to-end X-CLIP from frames) | github.com/ktr-hubrt/UMIL | 66★, apex | Skips feature extraction entirely; apex surgery + 2-GPU script to fix; fits 32 GB |
| 14 | VERA | CVPR'25 CCF-A | raw frames + InternVL2-8B | github.com/vera-framework/VERA | 86★, 2026-03 | Closest to our MLLM narrative; per-segment VLM calls = expensive, tension with our ≤2-call cap if used as anything but a baseline |
| 15 | MIST | CVPR'21 CCF-A | h5py features | github.com/fjchange/MIST_VAD | 137★, author warns "may have unknown bugs" | High friction; only if the self-training branch is needed |
| 16 | W-TALC | ECCV'18 A* | I3D/UNT | github.com/sujoyp/wtalc-pytorch | 132★, torch 0.4.1 | Historical anchor only; do not spend porting time |

Interchangeable WTAL extras (all verified, pick one at most): CoLA
(CVPR'21, zhang-can/CoLA), ASM-Loc (CVPR'22, boheumd/ASM-Loc), DDG-Net
(ICCV'23, XiaojunTang22/ICCV2023-DDGNet).

## Top 5 to reproduce first

1. **UR-DMU** — features+training+eval in one repo; shortest HateMM path.
2. **RTFM** — the standard; porting it unlocks half the table.
3. **HL-Net/XDVioDet** — the only audio-visual A* entry; without it the audio modality has no baseline.
4. **VadCLIP** — best cost/benefit; CLIP features in minutes.
5. **PEL4VAD** — correctness anchor (ckpts + logs to validate our pipeline before trusting HateMM numbers).

Alternative: to skip feature extraction entirely, promote UMIL
(end-to-end) into slot 3.

## Checked and excluded

- **Empty-shell repos (README-only, "code coming soon"):** PE-MIL
  (CVPR'24), MSL (AAAI'22), TDSD (MM'24), CU-Net (CVPR'23, stage-1
  training code missing).
- **No official repo found:** STPrompt (MM'24), TPWNG (CVPR'24),
  LEC-VAD (ICML'25).
- **Stronger-than-video-level supervision:** HolmesVAU (CVPR'25,
  segment/event instructions), Hawk (NeurIPS'24), GlanceVAD (point).
- **Training-free (nothing to retrain):** LAVAD (CVPR'24), LELA
  (arXiv 2602.09637, no code found).
- **Venue below bar:** BN-WVAD (arXiv; good torch 2.0 code though),
  MSBT, CLIP-TSA, TEVAD, AR-Net.
- **Framework-dead:** UntrimmedNet (Caffe+Matlab), GCN label-noise
  cleaner (CVPR'19, Caffe/TSN era) — cite, do not reproduce.
- **MultiHateLoc (WWW'26, 2512.10408): NO code release found.** Our
  closest direct competitor (weakly-supervised hate localization,
  evaluates on HateMM + MHC). Options: email authors (Zeyu Fu, Univ.
  of Exeter) for code, or reimplement from the paper. Reviewers will
  ask for this one.
