#!/usr/bin/env python3
"""Record the declared final runs after numerical checks, without reading metrics."""
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
OUT=ROOT/"runs/20261002_revisable_prior"
selected={}
for arm,tag in (("full","r1_full_px"),("no_global","r1_no_global_px"),("independent","r1_independent")):
    run=OUT/tag
    assert "RUN_DONE" in (run/"run.log").read_text(), tag
    ps=json.load((run/"params.json").open());qs=json.load((run/"quadrature.json").open())
    assert set(ps)==set(qs)=={"HateMM","HateClipSeg"}, tag
    acceptable=all(p["ordered_means"] and p["converged"] for p in ps.values()) and all(not c["needs_refit"] for c in qs.values())
    selected[arm]={"tag":tag,"path":str(run.relative_to(ROOT)),"numerically_acceptable":acceptable,
        "selection_basis":"predeclared final runs; convergence, semantic ordering and integration precision only; no metrics read"}
(OUT/"numerical_selection.json").write_text(json.dumps(selected,indent=2)+"\n")
print(json.dumps(selected,indent=2))
