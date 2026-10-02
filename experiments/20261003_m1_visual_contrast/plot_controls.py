#!/usr/bin/env python3
"""Plot completed R1 controls from canonical metrics and paired diagnostics."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
P=ROOT/'runs/20261003_m1_visual_contrast'
ARMS=('contrast','scale','video_shift','shuffle','common_shift')
LABELS=('Matched contrast','Visual scaling','Video-mean subtraction','Shuffled subtraction','Common shift')


def metric(arm,raw=False):
    group='r1_main' if arm in ('base','contrast') else 'r1_controls'
    if not raw:group+='_decoded'
    path=P/group/arm/'metrics.json'
    return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']},str(path.relative_to(ROOT))


def main():
    report=json.loads((P/'r1_controls_analysis/summary.json').read_text())['datasets']
    out=P/'r1_controls_analysis';rows=[]
    fig,axes=plt.subplots(2,3,figsize=(12,5.5),sharey=True,layout='constrained')
    columns=((True,'within_video_macro_ROC_AUC','raw_within_vs_base','Raw within-video AUC'),
             (False,'within_video_macro_ROC_AUC','within_vs_base','Final within-video AUC'),
             (False,'frame_PR_AUC',None,'Final pooled PR-AUC'))
    for di,ds in enumerate(('HateMM','HateClipSeg')):
        for col,(raw,name,ci_key,title) in enumerate(columns):
            ax=axes[di,col];base,bsource=metric('base',raw)
            threshold=1. if ci_key else .5
            ax.axvspan(-threshold,threshold,color='#ececec',zorder=0)
            ax.axvline(0,color='#444444',lw=.8)
            for i,arm in enumerate(ARMS):
                current,source=metric(arm,raw);delta=(current[ds][name]-base[ds][name])*100
                color='#0072B2' if arm=='contrast' else '#666666'
                lo=hi=None
                if ci_key:
                    stat=report[ds][arm][ci_key]
                    assert np.isclose(stat['mean']*100,delta,atol=1e-10)
                    lo,hi=np.array(stat['ci95'])*100
                    ax.errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color=color,capsize=3,ms=5)
                else:ax.plot(delta,i,'o',color=color,ms=5)
                rows.append({'dataset':ds,'arm':arm,'stage':'raw' if raw else 'final','metric':name,
                    'delta_percentage_points':delta,'ci95_low':lo,'ci95_high':hi,
                    'native_metrics':bsource,'candidate_metrics':source})
            ax.set_yticks(range(len(ARMS)),LABELS);ax.set_ylim(len(ARMS)-.5,-.5)
            ax.grid(axis='x',color='#dddddd',lw=.5);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            if di==0:ax.set_title(title,fontsize=11)
            if col==0:ax.set_ylabel(ds,fontsize=11)
            if di==1:ax.set_xlabel('Change from native (percentage points)')
    for col in range(3):
        low=min(ax.get_xlim()[0] for ax in axes[:,col]);high=max(ax.get_xlim()[1] for ax in axes[:,col])
        for ax in axes[:,col]:ax.set_xlim(low,high)
    fig.suptitle('R1 visual contrast: numerical gain does not establish local grounding\n'
        'Development-selected; within CIs use paired video bootstrap (84 / 99 videos); PR points have no CI.\n'
        'Gray bands mark declared noise tolerances, not confidence intervals.',fontsize=11)
    fig.savefig(out/'diagnostics.pdf');fig.savefig(out/'diagnostics.png',dpi=180)
    with (out/'diagnostics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.close(fig);print('PLOT_DONE',out/'diagnostics.png')


if __name__=='__main__':main()
