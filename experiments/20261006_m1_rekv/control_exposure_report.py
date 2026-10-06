"""Descriptive actual-budget/ancestor/ownership exposure; never reads GT."""
import argparse
import json
from collections import Counter
from pathlib import Path
from inputs import ROOT,DATASETS,selected_rows
from controls import ARMS,record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();scope='smoke' if a.smoke else 'main'
    root=ROOT/'runs/20261006_m1_rekv';out=root/('controls_'+scope+'_analysis');out.mkdir(parents=True,exist_ok=True)
    totals={ds:{arm:Counter() for arm in ARMS} for ds in DATASETS};row_count=0
    manifest=out/'selection_exposure.jsonl';stream=manifest.with_suffix('.partial').open('w')
    for row in selected_rows(a.smoke):
        bundle=record(root/('controls_'+scope+'_selection'),row);blocks=bundle['source_blocks']
        def ancestors(ids):return set(x for i in ids for x in blocks[i]['ancestors'])
        for arm,data in bundle['controls'].items():
            for t in data['traces']:
                if not t['branch']:continue
                for layer,rr in enumerate(t['branch']['layers']):
                    original=bundle['traces'][t['i']]['branch']['layers'][layer]['remote_ids'];selected=rr['remote_ids']
                    local=t['local_ids'];reference_ancestors=ancestors(original);selected_ancestors=ancestors(selected);local_ancestors=ancestors(local)
                    changed=len(set(selected)-set(original));kind='unchanged' if set(selected)==set(original) else 'fully_changed' if original and not set(original)&set(selected) else 'partially_changed'
                    budgets=dict(reference_tokens=sum(blocks[i]['shape'][3] for i in original),selected_tokens=sum(blocks[i]['shape'][3] for i in selected),
                        reference_image_tokens=sum(len(blocks[i]['image_rows']) for i in original),selected_image_tokens=sum(len(blocks[i]['image_rows']) for i in selected))
                    matched=Counter(blocks[i]['shape'][3] for i in original)==Counter(blocks[i]['shape'][3] for i in selected)
                    if arm in ('R0','C1','L2','D0'):assert matched
                    entry=dict(dataset=row['dataset'],video_id=row['video_id'],window=t['i'],layer=layer,arm=arm,
                        true_window_bounds=t['bounds'],true_local_ids=local,reference_remote_ids=original,selected_remote_ids=selected,change_kind=kind,
                        selected_true_times=[blocks[i]['actual_time'] for i in selected],reference_true_times=[blocks[i]['actual_time'] for i in original],
                        selected_remote_ancestor_union=len(selected_ancestors),reference_remote_ancestor_union=len(reference_ancestors),
                        remote_ancestor_overlap=len(selected_ancestors&reference_ancestors),remote_ancestor_union=len(selected_ancestors|reference_ancestors),
                        selected_remote_LOCAL_ancestor_overlap=len(selected_ancestors&local_ancestors),matched_complete_token_multiset=matched,**budgets)
                    stream.write(json.dumps(entry)+'\n');row_count+=1
                    count=totals[row['dataset']][arm];count['layers']+=1;count[kind+'_layers']+=1;count['changed_remote_ids']+=changed
                    count['exact_token_multiset_layers']+=matched;count['image_token_budget_changed_layers']+=budgets['reference_image_tokens']!=budgets['selected_image_tokens']
                    count['selected_remote_ancestors']+=len(selected_ancestors);count['reference_remote_ancestors']+=len(reference_ancestors)
                    count['remote_ancestor_overlap']+=len(selected_ancestors&reference_ancestors);count['remote_ancestor_union']+=len(selected_ancestors|reference_ancestors)
    (out/'exposure_summary.json').write_text(json.dumps(dict(GT_read=False,coverage=len(selected_rows(a.smoke)),scope='descriptive actual complete-token and image budgets/ancestor overlaps; no efficacy or exposed-subset gate',
        datasets={ds:{arm:dict(v) for arm,v in arms.items()} for ds,arms in totals.items()}),indent=2)+'\n')
    stream.close();manifest.with_suffix('.partial').replace(manifest)
    # Compact JSONL avoids expanding full333 multi-arm per-layer manifests into
    # hundreds of MB of repeated indentation; original records remain authority.
    print('EXPOSURE_REPORT_DONE',row_count,flush=True)


if __name__=='__main__':main()
