# 1 fps protocol (legacy, frozen 2026-08)

These documents and the toolchain in `scripts/duplex/` (feature extraction, GT arrays,
`frame_eval_common.py`) and `scripts/reproduction_baselines/` (VadCLIP, DSANet, MACIL-SD,
MultiHateLoc, VERA, LAVAD adapters and the official-validation sweep) belong to the
1 fps frame-level protocol used from 2026-08-18 to 2026-08-26. Outputs live in
`runs/legacy_1fps/lab1/` (this machine) and `runs/legacy_1fps/lab2/` (copied from uoa-lab2
on 2026-09-09). Paths inside the scripts were rewritten on 2026-09-09 from `results/…`
and `docs/duplex/…` to these locations; nothing else changed.

The project's current protocol is 4 fps (CLAUDE.md). 1 fps numbers are historical and
are not compared with 4 fps numbers (different video cohorts, GT arrays and upsampling).
