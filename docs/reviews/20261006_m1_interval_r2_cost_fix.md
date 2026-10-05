# Interval-witness R2 input-binding cost fix — narrow confirmation PASS

2026-10-06. Independent GPT-6-astra reviewer; same-family provisional. Scope is only the two accounting edits in `experiments/20261005_m1_interval_witness/measure_r2.py`. Scientific reader review remains closed; no production edits were needed during confirmation.

Evidence: `runs/20261005_m1_interval_witness/r2_cost_confirmation/{check.py,summary.json,run.log}`. Executed `.cache/envs/HateVLM/bin/python runs/20261005_m1_interval_witness/r2_cost_confirmation/check.py`; result `R2_COST_CONFIRMATION_PASS`.

Comparing parsed current source with the prior committed source confirms that reverting exactly the added `+input_binding_seconds` in optimized cost and corresponding subtraction in validation restores identical AST. Source acquisition cost, native/base arithmetic, model margins, calls, repeats, configuration and all other computation are unchanged.

Executed the actual two cost expressions and actual final cost assertions extracted from the AST, using synthetic positive component times. Input-binding times 0, 0.125 and 8.75 seconds give optimized totals 62, 62.125 and 70.75 seconds; native cost remains 19 seconds in every case. Correct records pass. Omitting source, prefix, fresh visual or native speech time is rejected. Omitting a positive input-binding time is now rejected; a zero binding time naturally makes no numerical difference. The existing validation tolerance is unchanged at 1e-6 seconds.

The corrected optimized total includes source preprocessing + native prefix + current-input CPU binding + fresh visual branches + reused native speech computation. Native base continues to include its original prefix + native visual + native speech costs. Repeated diagnostic work is still excluded from standalone inference cost as before.

This is an accounting bug fix, not a performance revision. No GPU, GT, real prediction values, metrics or model inference was used. This confirmation does not repeat the independent 36-layer reader checks or claim actual GPU execution success.
