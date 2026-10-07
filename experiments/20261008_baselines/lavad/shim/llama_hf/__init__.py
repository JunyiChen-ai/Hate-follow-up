"""Drop-in replacement for Meta's `llama` package, backed by HF transformers.

Why this exists (repro campaign, Phase A):

  * LAVAD (CVPR 2024) calls `llama.Llama.build(...)` on the *original* Meta
    `llama-2-13b-chat` checkpoint.  That checkpoint is gated on HF, and its
    original format is sharded model-parallel MP=2, i.e. it needs two GPUs
    under Meta's own loader.  This workstation has one RTX 5090.
  * URF-HVAA (NeurIPS 2025) does the same with `llama3.1-8b`.
  * Meta's `llama` package also pins fairscale + `torch.distributed` init,
    which is dead weight for single-GPU inference.

This shim exposes exactly the two names both repos import -- `Dialog` and
`Llama` -- and reproduces `chat_completion`'s contract:

    Llama.build(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, seed=...)
    generator.chat_completion(dialogs, max_gen_len=..., temperature=..., top_p=...)
        -> [{"generation": {"role": "assistant", "content": str}}, ...]

`ckpt_dir` is reinterpreted as an HF model id or local HF-format directory;
`tokenizer_path` is ignored (the HF tokenizer ships with the model).  Sampling
semantics follow Meta's `generation.py`: temperature == 0 means greedy argmax,
otherwise top-p sampling on logits/temperature.  Chat formatting uses the
model's own chat template, which for Llama-2-chat and Llama-3.1-Instruct is the
same [INST]/<|start_header_id|> layout Meta's ChatFormat emits.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, TypedDict

import torch


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


Dialog = List[Message]

# Ungated mirrors of the gated meta-llama originals, resolved from the
# directory name the upstream scripts pass in.
_ALIASES = {
    "llama-2-13b-chat": "NousResearch/Llama-2-13b-chat-hf",
    "llama-2-7b-chat": "NousResearch/Llama-2-7b-chat-hf",
    "llama3.1-8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}


LLAMA2_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
)


def _resolve(ckpt_dir: str) -> str:
    # An explicit override always wins -- it is how a smoke test substitutes a
    # small stand-in model for the 13B.
    override = os.environ.get("LLAMA_HF_MODEL")
    if override:
        return override
    if os.path.isdir(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "config.json")):
        return ckpt_dir
    key = os.path.basename(os.path.normpath(ckpt_dir)).lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"llama_hf: cannot map ckpt_dir={ckpt_dir!r} to an HF model. "
        f"Known: {sorted(_ALIASES)}; or set LLAMA_HF_MODEL."
    )


class Llama:
    def __init__(self, model, tokenizer, max_seq_len: int, max_batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str = "",
        max_seq_len: int = 4096,
        max_batch_size: int = 8,
        model_parallel_size: int | None = None,
        seed: int = 1,
        **kwargs: Any,
    ) -> "Llama":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        model_id = _resolve(ckpt_dir)

        load_kwargs: Dict[str, Any] = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        # 13B bf16 = 26 GB; on a 32 GB card shared with other jobs that is not
        # safe, so LLAMA_HF_4BIT=1 switches to NF4 (~7.5 GB).
        if os.environ.get("LLAMA_HF_4BIT", "0") == "1":
            from transformers import BitsAndBytesConfig

            load_kwargs.pop("torch_dtype")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.chat_template is None:
            # The ungated Llama-2 mirrors ship no chat_template. Fall back to the
            # canonical Llama-2-chat layout, which is byte-for-byte what Meta's
            # own ChatFormat.encode_dialog_prompt emits for a [system, user] pair.
            tok.chat_template = LLAMA2_CHAT_TEMPLATE
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # required for batched decoder-only generation
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
        return Llama(model, tok, max_seq_len, max_batch_size)

    @torch.inference_mode()
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int | None = None,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if max_gen_len is None:
            max_gen_len = self.max_seq_len - 1

        prompts = [
            self.tokenizer.apply_chat_template(
                d, tokenize=False, add_generation_prompt=True
            )
            for d in dialogs
        ]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len - max_gen_len,
            add_special_tokens=False,  # the chat template already adds BOS
        ).to(self.model.device)

        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=max_gen_len)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)  # Meta: temperature == 0 -> argmax

        out = self.model.generate(
            **enc, pad_token_id=self.tokenizer.pad_token_id, **gen_kwargs
        )
        texts = self.tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return [
            {"generation": {"role": "assistant", "content": t.strip()}} for t in texts
        ]

    # Meta's package also exposes text_completion; LAVAD/URF do not use it, but
    # keep a faithful stub so an accidental call fails loudly instead of silently.
    def text_completion(self, *a: Any, **k: Any):
        raise NotImplementedError("llama_hf shim implements chat_completion only")
