# Highlighter actual-frame metadata repair — independent confirmation

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`. Narrow follow-up to the completed rule-6 review; this does not reopen the method review or alter its frozen report.

**PASS.** The change in `experiments/20261003_m1_highlighter/analyze.py` replaces an incorrect requirement for exactly 20 cached frames with validation against `src.video_inputs.frame_paths(dataset, video_id, 20, 'k20')`: a nonempty set of at most 20 frames, matching timestamps, image-count length and expanded frame indices. `data/frames_k20/PROVENANCE.md` already allows fewer frames when ffmpeg cannot seek near the end. No frames are synthesized or changed.

Independently executed the repaired `prepare` on the completed `runs/20261003_m1_highlighter/r1_main` collection, writing only `runs/20261003_m1_highlighter/independent_review/frame_metadata_fix/alignment.json`. It returned `PREPARED 333`. Direct metadata inspection confirms 18 frames for HateClipSeg `bit_AxrVklzh9Cyf`. The complete validation still checks native parity, paired coverage, local support, scores, geometry and the 4-fps grid.

This confirmation parsed saved predictions solely for integrity checks; it did not load GT, read performance metrics, execute evaluation or use GPU. The repair changes validation only. Reader computation, saved predictions, canonical evaluation and r6 remain unchanged, so the complete CPU evaluation may proceed. No production code was edited by the reviewer and no hashes were used.
