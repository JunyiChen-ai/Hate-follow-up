# Structured choice prefix guard: independent equivalence review

Date: 2026-10-05. Reviewer: independent gpt-6-astra instance, same-family provisional. Verdict: **PASS** for this narrow CPU guard optimization. Candidate 26's completed initial review is not reopened.

Scope: `src/structured_source_generation.py`, `Stream.choose`, replacing the strict-token-prefix all-pairs assertion with lexicographic sorting and adjacent-pair checking. The preceding nonempty, unique-option and unique-token-sequence assertions remain. Sorting creates a separate local list; generation still uses the original sequences and option ordering. Tokenization, allowed-token FP32 argmax, forced forwards, cache positions, events and cap behavior are unchanged.

Equivalence: if a sequence has any strict descendant, its immediate lexicographic successor is a descendant. Every sequence between that prefix and any descendant must itself begin with the prefix. Thus adjacent checking rejects exactly when the old all-pairs check rejects; any adjacent rejection is also an old all-pairs rejection. Empty sequences and duplicates retain their earlier explicit rejection.

Independent executable evidence:

- Exhaustively ran old and current production `choose` implementations on 54,241 ordered menus: binary token alphabet, sequence lengths 0–3, menu sizes 0–4. This includes empty menus/sequences, duplicate token sequences, strict prefixes and every ordering. Acceptance/rejection, replay tokens and events were identical. Duplicate literal options were separately rejected by both versions. The old method was reconstructed only in test memory by replacing the current guard with its prior all-pairs expression.
- Rebuilt the current HateMM `non_hate_video_376` root speech menu directly from the current manifest and ASR: 26,310 choices. Actual Qwen tokenizer sequences passed an independent prefix-set oracle, without relying on either pairwise or sorted checking. Production replay of first, middle and last choices preserved the exact selected literal, tokens and complete event. Tokenization, independent checking and three production replays together took approximately 1.45 seconds on this CPU. The old loop's 692,216,100 pair count is arithmetic, not an executed old-loop timing benchmark.
- Ran actual random two-layer BF16 Qwen3-VL text-model computation on CPU under Transformers 5.15.1, using the actual Qwen tokenizer and identical frozen weights for old/new generation. Every returned generation field except measured seconds was exactly equal, including tokens, text, choice events, selections, input IDs, positions, truncation and forward counts. Complete generation used 20 tokens/21 forwards; caps of 1 and 8 used 2 and 9 forwards respectively and yielded identical whole-cap incomplete results. Both outputs also passed production generation replay validation.

Reproducible script and raw output:
`runs/20261005_m1_interval_witness/choice_prefix_guard_review/oracle.py`
`runs/20261005_m1_interval_witness/choice_prefix_guard_review/oracle.log`

No production files were modified by this reviewer. No GPU, GT, predictions, main summaries, hashes or model downloads were used. This establishes guard equivalence and actual tiny-model CPU behavior, not new 8B GPU parity or scientific performance. No observation-changing defect remains in this narrowly reviewed change.
