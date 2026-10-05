# Candidate32 actual fixed-five interface diagnosis

2026-10-06 — **No observed implementation bug; actual interface execution remains insufficient.** Same-family provisional, reused independent reviewer distinct from the author. This is a narrow Rule6 continuation, not an idea/performance review.

Evidence: `runs/20261005_m1_text_tracking/source_interface_diagnosis/independent/{report.md,diagnose.py,summary.json,run.log}`. Independent production source validation passes all five real records, including raw PTS/pixels, current token/grammar replay and controller history.

Fourteen raw TEXT fields produce one accepted TEXT. Thirteen correctly become UNKNOWN: four hit 32 words (37 content tokens), nine hit 64 tokens with continuation drift. Thus the failures are not all newline/token-cap cases. All fourteen saved literal sequences replay exactly; no whole-generation cap occurred.

The tokenizer has combined closings `"}`/`"},`, but the literal interface accepts only a standalone closing quote. This restriction is verified. No unmasked logits were saved, so preferred combined closure or its ability to fix the failures is unproven.

The only accepted `$600` is paired with a wrong actual box: its mapped crop `[10,230,114,240]` contains ground texture, not the visible HUD text. Mapping matches the model's normalized coordinates exactly. Next-frame tracking reproduces 30 initial/zero valid points and the saved rejection; fresh repair returns UNKNOWN.

Keep caps, UNKNOWN and the failed execution guard. A separately declared interface B may test token-complete closing syntax; it cannot assume recovery or ignore the independent box failure. No production changes, GT/prediction-score/metric access, GPU, pretrained weights, hashes or Git-ID provenance. No scientific performance conclusion is drawn.
