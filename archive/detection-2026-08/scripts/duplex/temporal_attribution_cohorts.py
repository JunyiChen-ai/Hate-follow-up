"""Freeze the 90-video temporal-attribution pilot cohort."""

import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEED = 20260808
PER_STRATUM = 15
ATTR_RE = re.compile(
    r"\b(?:said|says|saying|according|report|reported|reports|reporting|"
    r"quote|quoted|quoting|claim|claimed|claims|condemn|condemned|condemns|"
    r"criticize|criticized|criticises|oppose|opposed|opposes|satire|satirical|"
    r"joke|joking|news|interview|respond|responded|response|rebut|rebuttal|"
    r"deny|denied|denies|accuse|accused|allege|alleged)\b", re.I)


def main():
    source = json.load(open(os.path.join(ROOT, "results/stance_gate/cohorts.json")))
    arm = source["arms"]["ihv_train"]
    overrides = json.load(open(os.path.join(ROOT, "results/c2_fullcorpus/c2_overrides.json")))
    rng = random.Random(SEED)
    selected = {}
    strata = {}
    for cell in ("fp", "tp", "tn"):
        ids = sorted(arm["cohorts"][cell])
        marked = [v for v in ids if ATTR_RE.search(overrides.get(v, ""))]
        plain = [v for v in ids if v not in set(marked)]
        m = sorted(rng.sample(marked, min(PER_STRATUM, len(marked))))
        p = sorted(rng.sample(plain, min(PER_STRATUM, len(plain))))
        need = 2 * PER_STRATUM - len(m) - len(p)
        rest = sorted(set(ids) - set(m) - set(p))
        fill = sorted(rng.sample(rest, min(need, len(rest)))) if need else []
        chosen = sorted(m + p + fill)
        selected[cell] = chosen
        strata[cell] = {
            "pool_marker": len(marked), "pool_nonmarker": len(plain),
            "selected_marker": sum(v in set(marked) for v in chosen),
            "selected_nonmarker": sum(v not in set(marked) for v in chosen),
        }
    out = {
        "seed": SEED, "dataset": "ImpliHateVid", "split": "train",
        "source": "results/stance_gate/cohorts.json:ihv_train",
        "attribution_regex": ATTR_RE.pattern,
        "cohorts": selected, "strata": strata,
        "joint_z": {v: arm["joint_z"][v] for ids in selected.values() for v in ids},
        "marker_positive": {
            v: bool(ATTR_RE.search(overrides.get(v, "")))
            for ids in selected.values() for v in ids
        },
    }
    path = os.path.join(ROOT, "results/temporal_attribution/cohorts.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(strata, indent=2))
    print(f"Wrote {path}: {sum(map(len, selected.values()))} videos")


if __name__ == "__main__":
    main()
