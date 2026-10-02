"""Keep future non-ASR text; match newly available speech to frame read windows."""
import numpy as np


def aligned_edges(visual,speech,local):
    visual=np.asarray(visual,bool);speech=np.asarray(speech,bool);local=np.asarray(local,bool)
    P=len(visual)
    assert speech.shape==(P,) and local.ndim==2 and local.shape[1]==P
    assert not (visual&speech).any()
    assert (local[:,visual].sum(0)==1).all(),'image tokens must have exactly one frame window'
    allowed=np.arange(P)[None,:]<=np.arange(P)[:,None]
    non_asr_text=~visual&~speech
    for mask in local:
        rows=mask&visual;keys=non_asr_text|(mask&speech)
        allowed[np.ix_(rows,keys)]=True
    return allowed
