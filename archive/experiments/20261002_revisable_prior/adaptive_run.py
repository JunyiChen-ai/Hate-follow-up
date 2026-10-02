#!/usr/bin/env python3
"""Numerical continuation with path-envelope quadrature; no metrics read."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
OUT=ROOT/"runs/20261002_revisable_prior"
print("host",socket.gethostname(),flush=True)
(OUT/"adaptive.pid").write_text(str(os.getpid()))
selected={}
for arm, source in (("full","r1_full_q15"),("no_global","r1_no_global"),("independent",None)):
    tag="r1_"+arm+"_px" if source else "r1_independent"
    run=OUT/tag
    if source:
        if run.exists():raise RuntimeError(f"Refuse overwriting {run}")
        subprocess.run([sys.executable,str(ROOT/"archive/experiments/20261002_revisable_prior/model.py"),
            "--arm",arm,"--tag",tag,"--grid-step","1","--px","--max-it","300",
            "--init-params",str(OUT/source/"params.json")],cwd=ROOT,check=True)
    ps=json.load((run/"params.json").open());qs=json.load((run/"quadrature.json").open())
    acceptable=all(p["ordered_means"] and p["converged"] for p in ps.values()) and all(not c["needs_refit"] for c in qs.values())
    selected[arm]={"tag":tag,"path":str(run.relative_to(ROOT)),"numerically_acceptable":acceptable,
        "selection_basis":"convergence, semantic ordering and integration precision only; no metrics read"}
    (OUT/"numerical_selection.json").write_text(json.dumps(selected,indent=2)+"\n")
    print("SELECTED",arm,acceptable,flush=True)
print("ADAPTIVE_DONE",flush=True)
