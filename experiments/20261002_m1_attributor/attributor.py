"""Signed integrated gradients on frozen prefix media-value paths; no labels."""
import copy
import time
from contextlib import contextmanager
import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def cache_branch(cache):
    """DynamicLayer.update replaces K/V with cat outputs; share only immutable inputs."""
    out=copy.copy(cache)
    out.layers=[copy.copy(layer) for layer in cache.layers]
    assert all(type(layer).__name__=="DynamicLayer" and layer.is_initialized for layer in out.layers)
    assert all(not layer.keys.requires_grad and not layer.values.requires_grad for layer in out.layers)
    return out


def window_contributions(attribution,regions,texts):
    """Conserve supported attribution, including shared ASR timestamp tokens."""
    masks=np.stack(regions["local"]).astype(float)
    count=masks.sum(axis=0)
    weights=masks/np.maximum(count,1.)
    n=len(masks);values=[];assigned=0.;density=[]
    for i,txt in enumerate(texts):
        w={};d={}
        for kind,region in (("visual",regions["visual"]),("speech",regions["speech"])):
            if kind=="speech" and not txt.strip():continue
            membership=weights[i]*region
            contribution=float(membership@attribution)
            w["z_"+kind]=n*contribution;d["z_"+kind]=n*float(membership.sum())
            assigned+=contribution
        values.append(w);density.append(d)
    total=float(np.sum(attribution))
    return values,density,{"token_sum":total,"assigned_sum":assigned,
        "unassigned_sum":total-assigned,"unassigned_abs":float(np.abs(attribution[count==0]).sum()),
        "shared_tokens":int((count>1).sum()),"supported_tokens":int((count>0).sum())}


class Attributor:
    def __init__(self,judge):
        assert judge.family=="qwen3_vl"
        self.judge=judge
        self.attentions=[x.self_attn for x in judge.model.model.language_model.layers]
        self.original=ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.active=None
        self.query_embeds=None
        self.readout=None
        ALL_ATTENTION_FUNCTIONS.register("sdpa",self.forward)

    def close(self):
        assert self.active is None
        ALL_ATTENTION_FUNCTIONS.register("sdpa",self.original)

    @contextmanager
    def query_precision(self,ids,fp32=False):
        """Numerical diagnostic: same frozen BF16 weights, promoted for query math.

        Prefix encoding/cache remain native. Temporarily offload unused vision,
        embedding and full LM-head weights so FP32 language blocks fit a 5090.
        """
        if not fp32:
            yield
            return
        j=self.judge;lm=j.model.model.language_model
        embedding=j.model.get_input_embeddings();head=j.model.get_output_embeddings()
        modules=list(dict.fromkeys((j.model.model.visual,embedding,head)))
        with torch.no_grad():
            self.query_embeds=embedding(torch.tensor([ids],device=j.device)).detach().float()
            token_ids=torch.tensor(j.yes_ids+j.no_ids,device=j.device)
            self.readout=head.weight[token_ids].detach().float()
            for module in modules:module.to("cpu")
            torch.cuda.empty_cache()
            lm.layers.to(dtype=torch.float32);lm.norm.to(dtype=torch.float32)
        try:yield
        finally:
            with torch.no_grad():
                lm.layers.to(dtype=torch.bfloat16);lm.norm.to(dtype=torch.bfloat16)
                self.query_embeds=None;self.readout=None
                torch.cuda.empty_cache()
                for module in modules:module.to(j.device)

    def query_kwargs(self,ids):
        if self.query_embeds is not None:
            assert self.query_embeds.shape[1]==len(ids)
            return {"inputs_embeds":self.query_embeds}
        return {"input_ids":torch.tensor([ids],device=self.judge.device)}

    def tensor_margin(self,h):
        j=self.judge
        if self.readout is not None:
            logits=h.float()@self.readout.T
            if j.softcap:logits=torch.tanh(logits/j.softcap)*j.softcap
        else:
            token_ids=torch.tensor(j.yes_ids+j.no_ids,device=j.device)
            logits=j._logits_fp32(h,token_ids)
        logits=logits[0];ny=len(j.yes_ids)
        return torch.logsumexp(logits[:ny],0)-torch.logsumexp(logits[ny:],0)

    def native_margin(self,cache,ids):
        assert self.active is None
        with torch.no_grad():
            out=self.judge.model.model(**self.query_kwargs(ids),past_key_values=cache_branch(cache),use_cache=True)
            return float(self.tensor_margin(out.last_hidden_state[0,-1:]))

    def forward(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        if self.active is not None and module in self.attentions:
            a=self.active;P=a["prefix_len"]
            assert value.shape[-2]==P+a["query_len"] and dropout==0
            self.visited.append(module.layer_idx)
            # fp32 gate product, BF16 attention like the native reader. No in-place edits.
            gated=(value[:,:,:P,:].float()*a["gate"][None,None,:,None]).to(value.dtype)
            value=torch.cat((gated,value[:,:,P:,:]),dim=-2)
        return self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)

    @contextmanager
    def gates(self,gate,query_len):
        assert self.active is None
        self.active={"gate":gate,"prefix_len":len(gate),"query_len":query_len}
        self.visited=[]
        try:yield
        finally:self.active=None
        assert self.visited==list(range(len(self.attentions))),self.visited

    def point(self,cache,ids,media,alpha,gradient=True):
        """One global read with independent cache and one gate shared across layers."""
        j=self.judge;P=cache.get_seq_length();assert len(media)==P
        c=cache_branch(cache)
        with torch.set_grad_enabled(gradient):
            leaf=torch.full((P,),float(alpha),device=j.device,dtype=torch.float32,requires_grad=gradient)
            gate=torch.where(torch.as_tensor(media,device=j.device),leaf,torch.ones_like(leaf))
            with self.gates(gate,len(ids)):
                out=j.model.model(**self.query_kwargs(ids),past_key_values=c,use_cache=True)
                h=out.last_hidden_state[0,-1:]
                margin=self.tensor_margin(h)
            grad=torch.autograd.grad(margin,leaf)[0] if gradient else None
        z=float(margin.detach())
        vector=grad.detach().float().cpu().numpy().astype(float) if gradient else None
        assert np.isfinite(z) and (vector is None or np.isfinite(vector).all())
        assert cache.get_seq_length()==P and c.get_seq_length()==P+len(ids)
        return z,vector

    def integrate(self,cache,ids,media,smoke=False):
        torch.cuda.synchronize();tick=time.perf_counter()
        f0,_=self.point(cache,ids,media,0.,gradient=False)
        f1,endpoint=self.point(cache,ids,media,1.,gradient=True)
        torch.cuda.synchronize();endpoint_seconds=time.perf_counter()-tick
        expected=f1-f0;tolerance=max(.25,.05*abs(expected))
        if getattr(self,"adaptive",False):
            from scipy.integrate import quad_vec
            calls=[]
            def integrand(alpha):
                z,g=self.point(cache,ids,media,float(alpha),gradient=True)
                calls.append({"alpha":float(alpha),"margin":z,"gradient_sum":float(g.sum()),
                    "gradient_l1":float(np.abs(g).sum())})
                return g
            tick=time.perf_counter()
            attr,err,info=quad_vec(integrand,0.,1.,epsabs=.05,epsrel=.025,norm=lambda v:float(np.abs(v).sum()),
                limit=32,quadrature="gk21",full_output=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            residual=abs(float(attr.sum())-expected);passed=bool(info.success and residual<=tolerance)
            diagnostics={"f0":f0,"f1":f1,"expected_sum":expected,"adaptive":True,"quadrature_error_l1":float(err),
                "quad_success":bool(info.success),"quad_status":int(info.status),"quad_message":info.message,
                "completeness_residual":residual,"tolerance":tolerance,"calls":calls,
                "intervals":info.intervals.tolist(),"interval_errors":info.errors.tolist(),
                "trials":[],"accepted_nodes":len(calls) if passed else None,"numerical_pass":passed,
                "actual_forwards":2+len(calls),"actual_backwards":1+len(calls),"endpoint_seconds":endpoint_seconds,
                "deployed_seconds":endpoint_seconds+seconds,"deployed_forwards":2+len(calls),"deployed_backwards":1+len(calls)}
            return attr,endpoint,diagnostics,{256:attr}
        trials=[];solutions={};forwards=2;backwards=1;accepted=None;previous=None
        for n in (16,32,64,128,256):
            torch.cuda.synchronize();tick=time.perf_counter()
            nodes,weights=np.polynomial.legendre.leggauss(n)
            attribution=np.zeros(len(media),float)
            for x,w in zip((nodes+1.)/2.,weights/2.):
                _,g=self.point(cache,ids,media,float(x),gradient=True)
                attribution+=w*g
            forwards+=n;backwards+=n
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            residual=abs(float(attribution.sum())-expected)
            relative_l1=None if previous is None else float(np.abs(attribution-previous).sum()/max(np.abs(attribution).sum(),1e-12))
            stable=relative_l1 is not None and relative_l1<=.05
            trials.append({"nodes":n,"seconds":seconds,"sum":float(attribution.sum()),"completeness_residual":residual,
                "relative_l1_vs_previous":relative_l1,"tolerance":tolerance,"pass":residual<=tolerance and stable})
            solutions[n]=attribution
            previous=attribution
            if accepted is None and residual<=tolerance and stable:accepted=n
            if accepted is not None and not smoke:break
        # Keep all smoke integrations for numerical stability, but primary is the
        # first passing grid, exactly as in deployment. Never normalize residual away.
        n=accepted or 256
        return solutions[n],endpoint,{"f0":f0,"f1":f1,"expected_sum":expected,
            "trials":trials,"accepted_nodes":accepted,"numerical_pass":accepted is not None,
            "actual_forwards":forwards,"actual_backwards":backwards,
            "endpoint_seconds":endpoint_seconds,
            "deployed_seconds":endpoint_seconds+sum(t["seconds"] for t in trials if t["nodes"]<=n),
            "deployed_forwards":2+sum(k for k in (16,32,64,128,256) if k<=n),
            "deployed_backwards":1+sum(k for k in (16,32,64,128,256) if k<=n)},solutions
