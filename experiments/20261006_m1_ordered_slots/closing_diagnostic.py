"""Read-only generation diagnostic; observes existing logits, changes no choices."""
import json
import logging
import re
import socket
import time
import torch
from inputs import ROOT,SPEC,CACHE,selected_rows,caption_content,DATASETS
import src.structured_source_generation as generation
from interface import caption_writer
from retrieval import INTERFACE
from extract import validate
from src.video_inputs import load_asr
from src.mllm_judge import Judge,MODEL
OriginalStream=generation.Stream


class AuditStream(OriginalStream):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.audit=[];self.in_description=False
    def logits(self):
        z=super().logits()
        if self.in_description and not self.replay:
            terminal=self.j.diagnostic_terminal_ids;q=self.j.tok.encode('"',add_special_tokens=False)[0]
            ids=torch.tensor(terminal,device=z.device);_,indices=z[ids].topk(min(8,len(terminal)));_,raw=z.topk(8)
            def record(k):return dict(id=int(k),decoded=self.j.tok.decode([int(k)]),logit=float(z[k]))
            self.audit.append(dict(offset=len(self.tokens),bare_quote=record(q),bare_quote_global_rank=int((z>z[q]).sum())+1,raw_top=[record(k) for k in raw.tolist()],terminal_top=[record(terminal[k]) for k in indices.tolist()]))
        return z
    def append(self,token):
        if self.in_description and self.audit and self.audit[-1]['offset']==len(self.tokens):self.audit[-1]['actual_chosen']=int(token)
        return super().append(token)
    def description(self):
        old=self.in_description;self.in_description=True
        try:return super().description()
        finally:self.in_description=old


def terminal_ids(tok):
    return [i for i in range(len(tok)) if i not in tok.all_special_ids and (s:=tok.decode([i])).startswith('"') and re.fullmatch(r'["},\]\s]+',s)]


@torch.no_grad()
def main():
    out=ROOT/'runs/20261006_m1_ordered_slots'/('closing_diagnostic_'+INTERFACE);out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()]);logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,source_interface=INTERFACE,spec=SPEC,GT_read=False,scope='first original caption of unchanged fixed5; read-only token-choice diagnostics, no performance or source mutation',code='experiments/20261006_m1_ordered_slots/closing_diagnostic.py;2026-10-06',torch=torch.__version__,transformers=transformers.__version__)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    j.diagnostic_terminal_ids=terminal_ids(j.tok);assert j.tok.encode('"',add_special_tokens=False)[0] in j.diagnostic_terminal_ids
    instances=[]
    class Capture(AuditStream):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);instances.append(self)
    generation.Stream=Capture;asr={ds:load_asr(ds) for ds in DATASETS};rows=[]
    try:
        for row in selected_rows(True):
            path=CACHE/row['dataset']/row['video_id']/'metadata.json';text=path.read_text();stamp=path.stat().st_mtime_ns;m=json.loads(text);segments=asr[row['dataset']].get(row['video_id'],[]);validate(j,m,row,segments,path.parent)
            items,paths=caption_content(m['windows'][0],'UNKNOWN');old=m['generations'][0]
            g=generation.generate(j,ROOT,SPEC['caption_system'],items,paths,SPEC['caption_generation_tokens'],caption_writer)
            for key in ('tokens','events','selection','prompt','input_tokens','image_grid','positions','rope_delta','truncated'):assert g[key]==old[key],('replayed generation differs',key)
            assert path.read_text()==text and path.stat().st_mtime_ns==stamp
            stream=instances[-1];audit=stream.audit;stream.cache=None;stream.hidden=None;instances.clear();assert all(a.get('actual_chosen') is not None for a in audit)
            record=dict(dataset=row['dataset'],video_id=row['video_id'],original_path=str(path.relative_to(ROOT)),generation=g,audit=audit,source_readonly=True,original_tokens_exact=True)
            (out/(row['dataset']+'_'+row['video_id']+'.json')).write_text(json.dumps(record,indent=2)+'\n');rows.append(dict(dataset=row['dataset'],video_id=row['video_id'],original_tokens_exact=True,description_steps=len(audit),cost_seconds=g['seconds'],forwards=g['actual_forwards'],vision=g['actual_vision_forwards']))
            logging.info('DIAGNOSTIC_ROW %s/%s %d steps',row['dataset'],row['video_id'],len(audit))
    finally:generation.Stream=OriginalStream
    for hook in hooks:hook.remove()
    assert j.forward_calls==sum(r['forwards'] for r in rows) and j.vision_calls==sum(r['vision'] for r in rows)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope=cfg['scope'],actual_forwards=j.forward_calls,actual_vision=j.vision_calls,rows=rows),indent=2)+'\n');logging.info('DIAGNOSTIC_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
