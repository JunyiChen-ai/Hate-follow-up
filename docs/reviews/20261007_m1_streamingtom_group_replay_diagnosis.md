# Candidate39 — narrow group replay diagnosis and repair confirmation

2026-10-07, sc474397. Continuation of the same Rule6 scientific review by the independent gpt-6-astra instance; **PASS — repair confirmed, same-family provisional**. No broad review, proposal, scientific gate or tolerance was added. No production code was edited by the reviewer; no GPU, GT, evaluator or quality-metric analysis was run.

The first post-return strict preparation of actual8B original fixed5 failed at complete grouping-plan equality. Independent replay of all **624 source frames across both corpora** isolates the defect to CPU thread configuration, not selected source content:

| Replay | Complete plan mismatches | Scientific group differences |
|---|---:|---:|
| Four CPU threads, matching generation | 0 / 624 | 0 |
| One CPU thread, original analysis setting | 52 / 624 | 0 |

At one thread the only unequal fields are DPC diagnostic density, separation and score arrays. Maximum absolute differences are respectively **2.384185791015625e−7, 7.152557373046875e−7 and 4.917383193969727e−7**. Complete remaining dictionaries are exact: similarities, static/dynamic token IDs, budgets, centers, retained roots, members, weights and algorithm parameters. Four-thread replay reproduces even all diagnostic values exactly. Independently recomputed compressed projector and all three DeepStack outputs also equal saved values exactly.

Generation explicitly calls `torch.set_num_threads(4)`. Analysis relied on environment defaults, where `MKL_NUM_THREADS=1` took precedence over `OMP_NUM_THREADS=4`. Thus the original validator wrongly required bitwise equality across differing CPU reduction configurations. The observed inputs require no floating-point tolerance, changed grouping arithmetic or rerun/salvage of margins.

The author's targeted repair adds only `torch.set_num_threads(4)` at the start of `validate_bundle`, plus generation/replay thread counts in the preparation report. The reviewer inspected the exact readable diff: `plan == record['plan']`, all pixel/native/source/layout/dual-path/execution checks, spec, source grouping and score computation remain unchanged. Original failed preparation evidence and cost records remain retained.

Independent repair confirmation deliberately starts with one thread and calls the complete production `analyze.prepare(..., smoke=True)` on the original actual8B fixed5 records, writing its output only under the reviewer's evidence directory. **PREPARE_PASS, exit 0**, all five videos, initial threads 1 → replay threads 4, elapsed 25.413 s. The completed result is in `confirmation.json` and `traces/confirmation.log`. This executes strict data/native/source/query replay and original plumbing guards only, not evaluation or performance reporting.

Evidence: `runs/20261007_m1_streamingtom/group_replay_independent_review/`: own `check.py` and `confirm.py`, both-corpus per-video summaries, original task/decision/commands, raw responses, pre/post source snapshots and exact author diffs. This diagnosis concerns the original fixed5 inputs; it makes no claim that arbitrary cross-thread floating-point decisions can always be substituted. Exact scientific grouping validation stays mandatory.
