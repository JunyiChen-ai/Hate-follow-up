"""R2 delegates canonical evaluation/report; only source reading validation differs."""
import argparse
import analyze as original
from measure_r2 import ROOT,selected_rows,validate_bundle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    stem='r2_full_'+('smoke' if a.smoke else 'main');root=ROOT/'runs/20261005_m1_interval_witness'/stem;out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    original.validate_bundle=validate_bundle
    if a.stage=='prepare':
        original.prepare(root,out,a.smoke,expected_smoke_repeats=len(selected_rows(True)) if a.smoke else None,global_exact=True)
    elif a.stage=='evaluate':assert not a.smoke and a.name;original.evaluate(root,decoded,a.name)
    else:assert not a.smoke;original.report(root,decoded,out)


if __name__=='__main__':main()
