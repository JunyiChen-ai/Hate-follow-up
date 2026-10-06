"""Actual completed source packets and native questions; no labels/score use."""
import json
import os
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from inputs import ROOT,SPEC,CACHE,selected_rows
import reader


def main():
    out=ROOT/'runs/20261006_m1_texttiling/r2_reader_cpu_checks';prior={'__name__':'before_R2','__file__':str(ROOT/'experiments/20261006_m1_texttiling/reader.py')}
    exec(compile((out/'before_R2_reader.py').read_text(),'before_R2_reader.py','exec'),prior)
    counts=Counter();rows=[]
    for row in selected_rows():
        # Read only the unlabelled source compiler output and question strings.
        # Predictions and hate margins are never used to select a revision case.
        bundle=json.loads((ROOT/'runs/20261006_m1_texttiling/r1_full_main/records'/row['dataset']/(row['video_id']+'.json')).read_text())
        words=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text())['words']
        for trace,packet in zip(bundle['traces'],bundle['packets']):
            scoped=packet['scope'];original=trace['original_question']
            reader.REVISION=1;before=reader.speech_question(original,words,scoped,packet)
            assert before==prior['speech_question'](original,words,scoped,packet)
            reader.REVISION=2;after=reader.speech_question(original,words,scoped,packet)
            if packet['reason']=='compiled':assert after==before
            else:assert after==original
            counts[packet['reason']]+=1
        rows.append(dict(dataset=row['dataset'],video_id=row['video_id'],windows=len(bundle['packets'])))
    assert sum(counts.values())==7359 and len(rows)==333
    (out/'question_summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=333,windows=7359,reason_counts=dict(counts),
        R1_snapshot_questions_exact=True,compiled_questions_exact=True,noncompiled_questions_native_exact=True,
        scope='actual complete source packet/question construction; no model forward/quality or predicted-margin use'),indent=2)+'\n')
    print('REAL_QUESTION_CHECK_PASS',dict(counts),flush=True)


if __name__=='__main__':main()
