#!/usr/bin/env python3
"""Sequential numerical-resolution continuation; never reads metrics or labels."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
OUT=ROOT/"runs/20261002_revisable_prior"


def main():
    print("host",socket.gethostname(),flush=True)
    (OUT/"refinements.pid").write_text(str(os.getpid()))
    selected={}
    for arm in ("full","independent","no_global"):
        tag="r1_"+arm
        current=OUT/tag
        params=json.load((current/"params.json").open())
        checks=json.load((current/"quadrature.json").open())
        if arm=="independent":
            resolutions=[7] if not all(p["converged"] for p in params.values()) else []
        else:
            resolutions=[15,31,63,127]
        for nodes in resolutions:
            accurate=all(not c["needs_refit"] and "p95_abs_change" in c for c in checks.values())
            converged=all(p["converged"] for p in params.values())
            if accurate and converged:break
            newtag=f"r1_{arm}_q{nodes}"
            new=OUT/newtag
            if new.exists():raise RuntimeError(f"Refuse overwriting {new}")
            print("CONTINUE",arm,tag,"->",newtag,flush=True)
            subprocess.run([sys.executable,str(ROOT/"archive/experiments/20261002_revisable_prior/model.py"),
                            "--arm",arm,"--tag",newtag,"--nodes",str(nodes),"--max-it","300",
                            "--init-params",str(current/"params.json")],check=True,cwd=ROOT)
            current=new;tag=newtag
            params=json.load((current/"params.json").open());checks=json.load((current/"quadrature.json").open())
        acceptable=all(p["converged"] and p["ordered_means"] for p in params.values()) and all(not c["needs_refit"] for c in checks.values())
        selected[arm]={"tag":tag,"path":str(current.relative_to(ROOT)),"numerically_acceptable":acceptable,
                       "selection_basis":"convergence, ordered means and quadrature only; no metric files read"}
        (OUT/"numerical_selection.json").write_text(json.dumps(selected,indent=2)+"\n")
        print("SELECTED",arm,tag,acceptable,flush=True)
    print("REFINEMENTS_DONE",flush=True)


if __name__=="__main__":main()
