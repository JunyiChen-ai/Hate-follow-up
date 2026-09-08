#!/usr/bin/env python3
"""Minimal connected hull of geometric and cross-fitted modality witnesses.

The visual-only Vid-Group anchor is tested by language and audio, which did not
participate in its construction.  Three additional spans are selected by each
pair of modalities and tested by the held-out third modality.  The prediction
is the unique shortest connected interval containing the anchor and all three
cross-fitted witnesses.  Certificates remain scoped to their generating spans;
no claim is made that the outer hull itself has a p-value.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_crossfitted_interval_witness import heldout_test
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_shapley_boundary import midrank

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prediction',type=Path,required=True);ap.add_argument('--prediction-method',required=True);ap.add_argument('--crossfit',type=Path,required=True);ap.add_argument('--crossfit-method',required=True);ap.add_argument('--anchor',type=Path,required=True);ap.add_argument('--anchor-method',required=True);ap.add_argument('--fields',type=Path,required=True);ap.add_argument('--fields-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 pred=load(a.prediction,a.prediction_method);cross=load(a.crossfit,a.crossfit_method);anchor=load(a.anchor,a.anchor_method);fields=load(a.fields,a.fields_method);keys=sorted(set(pred)&set(cross)&set(anchor)&set(fields));audit={'n':len(keys),'anchor_constant_tests':0,'hull_equals_anchor':0}
 with a.out.open('w') as h:
  for key in keys:
   row=pred[key];scores=np.asarray(row['score_curve'],float);n=min(len(scores),max(1,int(np.floor(float(row['duration'])*4))));scores=scores[:n];rate=float(row.get('native_rate',4) or 4);ai=anchor[key]['intervals'][0];alo=max(0,min(n-1,int(np.floor(float(ai[0])*rate))));ahi=max(alo+1,min(n,int(np.ceil(float(ai[1])*rate))));ev=fields[key]['modality_evidence']['main_effects'];anchor_tests=[]
   for heldout in ('language','audio'):
    values=midrank(np.asarray(ev[heldout],float)[:n]);stat,p,e=heldout_test(values,alo,ahi);audit['anchor_constant_tests']+=int(np.ptp(values)<=1e-12);anchor_tests.append({'heldout_modality':heldout,'lo':alo,'hi':ahi,'heldout_contrast':stat,'conditional_orbit_pvalue':p,'conditional_evalue':e})
   witnesses=cross[key]['modality_evidence']['crossfitted_interval_witnesses'];spans=[(alo,ahi)]+[(int(w['lo']),int(w['hi'])) for w in witnesses];lo=min(s[0] for s in spans);hi=max(s[1] for s in spans);audit['hull_equals_anchor']+=int((lo,hi)==(alo,ahi));fold_e=[float(w['conditional_evalue']) for w in witnesses]+[float(w['conditional_evalue']) for w in anchor_tests];display_e=float(np.mean(fold_e));out=dict(row);out['method']='witness_complete_hull_v1';out['score_curve']=scores.tolist();out['intervals']=[[lo/rate,min(float(row['duration']),hi/rate),display_e]];out['modality_evidence']={**out.get('modality_evidence',{}),'visual_anchor_interval':[alo,ahi],'visual_anchor_heldout_tests':anchor_tests,'crossfitted_interval_witnesses':witnesses,'witness_display_mean_evalue':display_e};out['raw']={**out.get('raw',{}),'module':'witness_complete_hull','anchor_source_method':a.anchor_method,'crossfit_source_method':a.crossfit_method,'readout':'unique_shortest_connected_interval_containing_visual_anchor_and_all_crossfitted_witness_spans','certificate_scope':'component_spans_only_not_outer_hull','hull_score_is_display_only':True,'score_threshold':None,'duration_tuning_parameter':None,'numeric_weights':0,'dataset_parameters':0,'label_selected_parameters':0,'gt_access':False};h.write(json.dumps(out,separators=(',',':'))+'\n')
 a.out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
