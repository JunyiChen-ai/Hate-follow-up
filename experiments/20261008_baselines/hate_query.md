# Hate query shared by the label-free baselines (fixed 2026-10-08)

Every new label-free baseline uses the text below, so that methods differ only in how they use it.

**Exception.** ZS-CLIP, ZS-ImageBind (audio) and LAVAD already have HateMM / HateClipSeg results re-evaluated from the
Retrieval-hate runs (`README.md` in this directory). On DeHate they must reproduce those runs' setup exactly,
including the prompts, so that each method has one setup across the three corpora.

## 1. Rules (for LLM / MLLM prompts)

These are the rules our own Reader uses, copied verbatim from `experiments/20260910_spvl/spvl.py` (`YOUTUBE_RULES`):

```
1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events
```

Question when a method asks yes/no about a clip, window or segment:
`Does this segment contain content that violates any of the above rules? Answer "Yes" or "No".`

When a method has its own prompt template with an anomaly description slot, put the rules into that slot and keep
the rest of the template unchanged.

## 2. Short query (for text–video similarity and grounding models)

- Positive (anomaly / target event): `hateful content that attacks or demeans people based on protected attributes`
- Negative (normal), for methods that need a pair: `normal content`

Methods that take a list of anomaly descriptions (for example PRISM's description pools) use the positive query plus
the nine rules above as the hateful pool, and `normal content` as the normal pool. They must not use any label to
write or select these descriptions.

## 3. Test cohorts (exact; no subsets)

- HateMM: the 215 video IDs of `runs/20260926_twolevel/final_rawkey/predictions.jsonl` (dataset HateMM)
- HateClipSeg: the 118 video IDs of the same file (dataset HateClipSeg)
- DeHate: the 1,151 test video IDs in `data/gt_4fps/DeHate.npz`

A run counts only if it has a finite score on every GT frame of every cohort video. Fallbacks for missing input are
in `run_plan.md` §1.3.
