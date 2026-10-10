# Independent code review: candidate 41 revision 1, probe before frames (2026-10-11, commit ed075f8)

Read-only review by a same-model agent of `experiments/20261010_m1_grounded_adjacent/probe_native.py`,
`probe_native_analyze.py`, `launch/lab{1,2,3}_probe.sbatch`, `launch/run_probe_analysis.sh`, using the real
Qwen3-VL-8B processor on CPU, transformers 5.15.1 / torch 2.11 (same in lab2 / lab3 envs), the finished smoke
(Slurm 372, 5 videos / 158 windows) and a read-only dry run of `selections()` on the smoke records.

## Verdict

No blocker in the code. The 333 run can proceed on lab1 as declared. One launch precondition elsewhere: lab3
lacks `runs/20261007_m1_streamingtom/placement_controls_main/records` and neither lab2 nor lab3 has the output
directory or the current commit, so `SCOPE=both` would fail there before the main run.

## Findings

1. Launch precondition (blocker only if lab2 / lab3 is used): `git pull`, `mkdir -p
   runs/20261010_m1_grounded_adjacent`, and on lab3 rsync the placement records first. Nothing missing on lab1.
2. Minor: the smoke prepare returns before `selections()`, so the hybrid / oracle / random / folder code is not
   exercised in the job. The reviewer ran it read-only on the smoke records: 48 selections built; `full_a`,
   `full_l` equal the stored `adjacent_native` / `adjacent_local` predictions on 5 / 5; `full_E` equals
   `local_controls_main/local_clean` and `combination_analysis/full_E` on 5 / 5; `sel_A_a` takes the arm exactly
   where flagged and present; `oracle_a` changes windows only toward the label; random counts per seed bracket the
   flagged count. No bug.
3. Minor, declared constant kept: number matching within .05 s of a prefix frame. A prefix frame lies within
   tolerance of a window bound in 82 / 3768 HateMM and 80 / 3591 HateClipSeg windows; in the smoke 11 / 158 replies
   contained a bound-valued number, all fabricated labels (e.g. `[t=64.0s]` for a frame at 71.15 s, `192.0s,
   193.0s, …`), none an echo of the probe text. Expected overlap about 5 to 6 windows per corpus, a property of the
   frame grid, not of content, so it cannot correlate with the arm's helpfulness. A stricter matcher would turn
   17 / 158 legitimate bare citations (`2.4s, 7.1s`) into spurious flags, so it is not safer. The decisive
   statistic, computable offline from the stored replies, is the number of windows whose `flag_a` changes when
   bound-valued numbers are dropped.

## Checked and found correct

Token sequences of both probes after `ctx['head']` (consecutive user turns allowed; probe B carries the hard
answer and the same end-of-turn tokens as the stance seam in `build()`); positions take the `rope_deltas` branch
of `compute_3d_position_ids` for the prefill and each generated token, the branch `margin()` uses, which does not
write `rope_deltas`; `crop` with a positive value is the absolute-length path in 5.15.1; cache length, `rope_deltas`,
`img_tokens`, `_prefix_text` unchanged after `probes()`; the smoke prepare asserted base predictions equal to r6
on all 158 windows with the probes interleaved. Forward counts: 3 for `build()`, V and S per window, 1 + generated
per probe (EOS breaks before a forward; at the cap the last forward is executed), vision 1; held on all five
videos including four 48-token replies. `parse_a` / `parse_b` as declared in README steps 2 and 3; the analysis
re-derives both flags. Pilot rule not triggered (correct_a .635 / .387, inside_b .135 / .258). Observation:
flag_b .865 / .742, so `sel_B_*` is close to `full_*`. Selections, random controls (per-video generator, rate over
eligible windows), oracle (visual direction), hybrid (identical to `combination_analysis.hybrid`), evaluator and
decoder flags identical to `analyze.py`; gate as declared. Launch: prepare does not load the model; imports resolve
in both the sbatch and the shell script; `--mem=32G`. Full-run failure modes: every video has a covered window;
the 18-frame HateClipSeg video handled; empty bodies give S = None; 48-token replies without EOS handled; JSON
escapes non-ASCII; peak 19.49 GiB in the smoke vs 19.50 GiB over the candidate 41 main run; probes .25 to .29 s per
window, about 31 to 36 min for 7359 windows. Resuming a partial `probe_main` on another host fails the config host
assertion, as in `grounded.py`.
