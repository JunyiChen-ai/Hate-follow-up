# TIDE U32 experiment code review

Reviewer: fresh secondary Codex agent, same-family provisional.

The first review blocked deployment on mismatched arm carriers and nulls,
unmatched transcript support, unsafe partial writes, OOM handling, normalized
rather than timestamp-based 4 FPS mapping, and cumulative call accounting.

Implemented fixes:

- all arms now use the same 896x448 32-cell carrier and focus marker;
- local joint/text use the same neighbor-bin transcript support;
- every arm/bin has a structure- and prompt-matched evidence-free null;
- one common 12k-character ASR budget is frozen before arm derivation;
- unmasked batches are capped at two and exceptions terminate the run;
- all four predictions are buffered before writing;
- canonical times are `(j+0.5)/4`;
- calls are counted per arm and semantic query count is explicit;
- `--bins` is asserted to equal the fixed U32 carrier.

Second review verdict: no remaining blocking issue after the U32 assertion.
