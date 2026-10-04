"""Fresh text-only greedy generation using the existing frozen Judge model."""
import time
import torch


@torch.no_grad()
def generate_text(j, system, user_text, max_tokens):
    torch.cuda.synchronize(); start = time.perf_counter(); first = j.forward_calls
    messages = [j.turn('system',system), j.turn('user',user_text)]
    rendered = j.render(messages,True); encoded = j.encode(rendered,[])
    j.model.model.rope_deltas = None
    output = j.model.model(**j.model_inputs(encoded),use_cache=True)
    cache = output.past_key_values; hidden = output.last_hidden_state[0,-1]; del output
    if not hasattr(j,'generation_W32'):
        j.generation_W32 = j.model.get_output_embeddings().weight.float()
    tokens = []; stopped = False
    for _ in range(max_tokens):
        logits = hidden.float() @ j.generation_W32.T
        if j.softcap:
            logits = torch.tanh(logits/j.softcap)*j.softcap
        token = int(logits.argmax())
        if token in j.eos_ids:
            stopped = True; break
        tokens.append(token); hidden = j._step(cache,[token])
    del cache
    torch.cuda.synchronize()
    return dict(text=j.tok.decode(tokens,skip_special_tokens=True).strip(),tokens=tokens,
        truncated=not stopped,prompt=rendered,input_tokens=encoded['input_ids'][0].tolist(),
        seconds=time.perf_counter()-start,actual_forwards=j.forward_calls-first,max_tokens=max_tokens)
