# Protocol survey — every hateful-video temporal localization work (2026-08-18)

Full extraction with quotes in session log; this file records the
verified comparison table and the headline finding. The field is six
papers deep; three appeared in the last nine months.

| Work | Task form | Unit + grid | Span→unit rule | Metric | Inheritable GT |
|---|---|---|---|---|---|
| LELA (2602.09637) | frame prediction | NOT STATED | NOT STATED | frame ROC-AUC, AP | no (inherits "LAVAD practice" by one citation) |
| MultiHateLoc (2512.10408) | frame prediction | T="number of frames", fps NOT STATED | NOT STATED | frame mAP, AUC | no (repo LICENSE-only) |
| HateClipSeg Task 2 (2508.01712) | interval proposals | STATED: 4 FPS moment rate (training side) | NOT STATED (ActionFormer default; eval matching rule unstated) | Acc/P/R/F1(O) at tIoU {0.3,0.5,0.7}; NO mAP | partial: segment CSV with timestamps; no frame array, no eval code |
| HateClipSeg Task 3 | per-timestamp online classification | LSTR window 32 s stride 0.25 s; eval grid NOT STATED | NOT STATED | Acc, Macro-F1 | same CSV |
| TANDEM (2601.11178) | LLM-generated intervals | 30 s chunks, no quantization | PARTIAL: max-IoU over all GT (over-prediction unpenalized); chunk gold split NOT STATED | Avg IoU, Acc@0.5, positives only | no ("upon acceptance") |
| SafeLens (AAAI-26 demo) | segment classification on SELF-produced segments | Whisper sentences + ViT scene cut (τ unstated) | NOT STATED | duration-weighted + segment-level F1 over 19 videos, GitHub-only | no |
| Label-noise diag (2508.04900) | gold-trimmed clip classification | variable clips, no fps | STATED (clearest in the field): in-span=hate; pre/post non-overlap=non-hate | Macro-F1, 5-fold CV | YES: code + segment manifests |
| ViToSA (2506.00636) | ASR text spans (no time axis evaluated) | NOT STATED | NOT STATED | Acc/MF1/WF1 by citation | HF dataset (text spans) |
| RAMF (2512.02743) | video-level only — not localization | — | — | Macro-F1 | — |

HateClipSeg Task 2 baseline (ActionFormer, official): at tIoU 0.5,
Visual-only F1(O) 52.65 BEATS trimodal 50.92 — fusion is net negative
in the official baseline. TANDEM's own localization numbers are weak
(HateMM Avg IoU 0.18–0.32) vs its 0.73 target-ID headline.

## Headline finding

Across all six localization works: NOT ONE publishes a frame-level
ground-truth array (no analogue of VAD's gt-ucf.npy), and NOT ONE
states the span→frame conversion rule for a frame-level metric. The
two frame-prediction papers (LELA, MultiHateLoc) omit both the grid
and the conversion — their numbers are neither reproducible nor
mutually comparable. Consequence for us: any frame-level number we
produce must ship its own frozen protocol (1 fps grid, half-open
containment, floor rule for uncovered frames — already frozen in
PREREG_frame_level_evaluation_hatemm.md) AND the released GT arrays;
publishing the standardized protocol + arrays for HateMM/MHC/
HateClipSeg is itself a citable contribution the field visibly lacks.
