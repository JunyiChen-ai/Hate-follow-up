#!/usr/bin/env python3
"""Export clean T3AL and Poset base curves in the shared prediction schema."""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import (ASR, poset_curve,
    rows, transcript_rows)
from scripts.label_free_adapt.schema import Prediction, append_jsonl, curve_to_intervals


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True)
    ap.add_argument('--t3al-curves',type=Path,required=True);ap.add_argument('--topology-curves',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();chunks=transcript_rows()
    for row in rows(a.cohort):
        d,v,dur=row['dataset'],row['video_id'],float(row['duration'])
        t=np.load(a.t3al_curves/d/f'{v}.npy');q=np.load(a.topology_curves/d/f'{v}.npy')
        p=poset_curve(q,chunks.get((d,v),[]),dur)
        for method,curve in [('t3al_base',t),('poset_base',p)]:
            curve=np.asarray(curve,float);curve=(np.argsort(np.argsort(curve,kind='stable'),kind='stable')+.5)/len(curve)
            pred=Prediction(method,d,v,dur,score_curve=curve.tolist(),
                intervals=curve_to_intervals(curve,dur,.75),calls=0,
                raw={'source':'clean inference-only curve'})
            append_jsonl(a.out,pred)
    print(json.dumps({'out':str(a.out),'records':len(rows(a.cohort))*2}))

if __name__=='__main__':main()
