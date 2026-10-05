"""Actual saved LINK token replay before same-interface long-input resume."""
import json
import logging
import os
import socket
import torch
from inputs_handles import ROOT,CACHE,DATASETS,link_content
from graph import LINK_SYSTEM,compile_graph
from handle_interface import write_links,compiled_links
from src.mllm_judge import Judge,MODEL
from src.structured_source_generation import generate
from src.qwen3_mlp_memory import chunked_mlp


def main():
    out=ROOT/'runs/20261005_m1_provenance/link_mlp_memory_fix/gpu';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()]);logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    choices={'HateMM':('non_hate_video_199',2),'HateClipSeg':('bit_0nXuyV2rypaf',3)}
    torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))];checks=[]
    for ds in DATASETS:
        vid,index=choices[ds];m=json.loads((CACHE/ds/vid/'metadata.json').read_text());saved=m['links'][index]
        local=compile_graph([l['parsed'] for l in m['ledgers']],[]);anchors=saved['anchors'];content=link_content(local,anchors)
        assert 4096<len(saved['generation']['input_tokens'])<=16000
        writer=lambda stream:write_links(stream,local,anchors)
        ordinary=generate(j,ROOT,LINK_SYSTEM,content,[],2048,writer)
        assert ordinary['tokens']==saved['generation']['tokens'] and ordinary['events']==saved['generation']['events']
        with chunked_mlp(j.model.model,4096):small=generate(j,ROOT,LINK_SYSTEM,content,[],2048,writer)
        assert small['tokens']==ordinary['tokens'] and small['events']==ordinary['events']
        assert compiled_links(small,local['nodes'],anchors)==saved['parsed']
        checks.append(dict(dataset=ds,video_id=vid,index=index,original_source_tokens_exact=True,chunk_source_tokens_exact=True,ordinary=ordinary,chunk=small))
        logging.info('%s/%s input%d generated%d exact',ds,vid,len(small['input_tokens']),len(small['tokens']))
    for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,PASS=True,checks=checks),indent=2)+'\n');logging.info('LINK_MEMORY_GPU_PASS')


if __name__=='__main__':main()
