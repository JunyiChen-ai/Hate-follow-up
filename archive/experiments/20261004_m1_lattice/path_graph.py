"""R2 keeps each complete decoded hypothesis intact in one correlated path."""
import numpy as np

VERSION='R2 complete-hypothesis speech lattice, sources2026-10-05'
SCAFFOLD=('Consider only window {i} of {n}, from {a:.1f}s to {b:.1f}s of this video.\n'
    'The following speech lattice contains mutually exclusive complete transcriptions of the same audio. '
    'Words within one alternative belong together; alternatives must not be combined. '
    'A null path means no words. The complete video transcript is interpretation context.\nSpeech lattice:\n')


def tokenize_paths(beams,tokenizer):
    assert len(beams)==5 and all(np.isfinite(b['score']) for b in beams)
    scores=np.asarray([b['score'] for b in beams],dtype=np.float32)
    weights=np.exp(scores-scores.max(),dtype=np.float32);weights/=weights.sum(dtype=np.float32)
    merged={}
    for i,b in enumerate(beams):
        text=b['text']
        if text not in merged:merged[text]=dict(text=text,mass=0.,beams=[])
        merged[text]['mass']+=float(weights[i]);merged[text]['beams'].append(i)
    paths=sorted(merged.values(),key=lambda p:(-p['mass'],min(p['beams']),p['text']))
    ids=[];nodes=[];width=0
    for pi,p in enumerate(paths):
        tokens=tokenizer.encode(' '+p['text'],add_special_tokens=False) if p['text'] and p['mass']>0 else []
        p['tokens']=tokens;width=max(width,len(tokens))
        for k,t in enumerate(tokens):
            nodes.append(dict(path=pi,subword=k,logical=k,mass=p['mass'],physical=len(ids)));ids.append(t)
    return dict(ids=ids,nodes=nodes,paths=paths,path_length=width,beam_weights=weights.tolist())


def bias(graph,h,t,n):
    g=len(graph['ids']);length=h+g+t;out=np.full((length,n+length),-np.inf,dtype=np.float32)
    out[:,:n]=0.
    for q in range(h):out[q,n:n+q+1]=0.
    if g:
        paths=np.asarray([v['path'] for v in graph['nodes']]);own=(paths[:,None]==paths[None,:])&np.tri(g,dtype=bool)
        out[h:h+g,n:n+h]=0.
        out[h:h+g,n+h:n+h+g]=np.where(own,0.,-np.inf)
        masses=np.log(np.asarray([v['mass'] for v in graph['nodes']],dtype=np.float32))
    for q in range(h+g,length):
        out[q,n:n+h]=0.
        if g:out[q,n+h:n+h+g]=masses
        out[q,n+h+g:n+q+1]=0.
    return out


def positions(graph,h,t,start):
    return list(range(start,start+h))+[start+h+v['logical'] for v in graph['nodes']]+list(range(start+h+graph['path_length'],start+h+graph['path_length']+t))
