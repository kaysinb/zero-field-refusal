"""Models and tokenizers at pinned revisions, with the numerics of the protocol.

The teacher must be θ₀ exactly as the student sees it at step 0 (invariant 1): the same revision,
dtype, attention implementation and chat template. Generation must not inherit sampling defaults
from the checkpoint: Qwen2.5-Instruct ships temperature 0.7, top-p 0.8, top-k 20 and a repetition
penalty, any of which would turn the zero field into a real field (docs/architecture.md,
"Generation settings").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DTYPES = {"float32": torch.float32}


def set_numerics(model_cfg: DictConfig) -> None:
    """Disable TF32 unless the config allows it; the protocol keeps it off."""
    allow = bool(model_cfg.tf32)
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    torch.set_float32_matmul_precision("high" if allow else "highest")


def load_tokenizer(model_cfg: DictConfig) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_cfg.name, revision=model_cfg.revision)


def load_model(
    model_cfg: DictConfig,
    device: str | torch.device,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:
    """The pinned checkpoint in eval mode, with a neutral generation config."""
    set_numerics(model_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        revision=model_cfg.revision,
        dtype=DTYPES[model_cfg.dtype],
        attn_implementation=model_cfg.attn_implementation,
    )
    model.to(device).eval()
    install_neutral_generation_config(
        model, end_token_ids(tokenizer, model_cfg), tokenizer.pad_token_id
    )
    return model


def install_neutral_generation_config(
    model: PreTrainedModel, end_ids: Sequence[int], pad_id: int
) -> None:
    """Replace the checkpoint's generation config by one that sets only the end and pad ids.

    Callers pass every sampling field explicitly; the teacher additionally verifies that no
    logits processor changed the distribution it sampled from.
    """
    model.generation_config = GenerationConfig(eos_token_id=list(end_ids), pad_token_id=pad_id)


def end_token_ids(tokenizer: PreTrainedTokenizerBase, model_cfg: DictConfig) -> list[int]:
    """Token ids of ``model.end_tokens``; generation stops at the first of any of them."""
    ids = []
    for token in model_cfg.end_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"End token {token!r} is not in the tokenizer's vocabulary")
        ids.append(int(token_id))
    return ids


def encode_chat(tokenizer: PreTrainedTokenizerBase, user_text: str) -> list[int]:
    """Token ids of a one-turn chat ending with the assistant generation prompt.

    The chat template adds the model's default system prompt, as in deployment.
    """
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=True
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return [int(i) for i in ids]


def freeze(model: PreTrainedModel, names: Sequence[str]) -> list[tuple[str, torch.nn.Parameter]]:
    """Freeze the named modules and return every other parameter, made trainable.

    Qwen2.5-0.5B and 1.5B tie the output head to the input embeddings; 7B does not, so both modules
    are named in the protocol and checked separately.
    """
    modules = {
        "input_embeddings": model.get_input_embeddings(),
        "output_head": model.get_output_embeddings(),
    }
    unknown = sorted(set(names) - set(modules))
    if unknown:
        raise ValueError(f"Unknown modules to freeze: {unknown}; known: {sorted(modules)}")
    frozen = {id(p) for name in names for p in modules[name].parameters()}
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad_(id(param) not in frozen)
        if param.requires_grad:
            trainable.append((name, param))
    return trainable
