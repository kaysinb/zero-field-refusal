import pytest
import torch
from conftest import END_IDS, PAD_ID, tiny_model
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GenerationConfig, PreTrainedTokenizerFast

from zero_field_refusal.models import (
    encode_chat,
    end_token_ids,
    freeze,
    install_neutral_generation_config,
    set_numerics,
)

# The sampling defaults Qwen2.5-Instruct ships in generation_config.json.
QWEN_DEFAULTS = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1)
ZERO_FIELD = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = ["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "system", "user"]
    words += ["assistant", "hello", "world"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="<|endoftext|>"
    )
    tokenizer.chat_template = (
        "<|im_start|> system default <|im_end|> "
        "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
        "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
    )
    return tokenizer


def generate(model, **sampling):
    torch.manual_seed(0)
    ids = torch.randint(7, 64, (2, 5))
    return model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        do_sample=True,
        max_new_tokens=6,
        output_logits=True,
        output_scores=True,
        return_dict_in_generate=True,
        **sampling,
    )


def test_qwen_defaults_would_reshape_sampling_without_the_neutral_config() -> None:
    model = tiny_model()
    model.generation_config = GenerationConfig(**QWEN_DEFAULTS, eos_token_id=END_IDS)
    out = generate(model, temperature=1.0)  # only the temperature passed: Qwen's top-k leaks
    assert not all(torch.equal(a, b) for a, b in zip(out.logits, out.scores, strict=True))


def test_neutral_config_keeps_only_end_and_pad_ids() -> None:
    model = tiny_model()
    model.generation_config = GenerationConfig(**QWEN_DEFAULTS, eos_token_id=END_IDS)
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    diff = model.generation_config.to_diff_dict()
    diff.pop("transformers_version", None)
    assert diff == {"eos_token_id": END_IDS, "pad_token_id": PAD_ID}
    out = generate(model, **ZERO_FIELD)
    assert all(torch.equal(a, b) for a, b in zip(out.logits, out.scores, strict=True))


def test_numerics_disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    set_numerics(OmegaConf.create({"tf32": False}))
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"


def test_end_token_ids_resolve_and_reject_unknown_tokens() -> None:
    tokenizer = tiny_tokenizer()
    cfg = OmegaConf.create({"end_tokens": ["<|im_end|>", "<|endoftext|>"]})
    assert end_token_ids(tokenizer, cfg) == [3, 1]
    with pytest.raises(ValueError, match="<\\|eot\\|>"):
        end_token_ids(tokenizer, OmegaConf.create({"end_tokens": ["<|eot|>"]}))


def test_encode_chat_returns_token_ids_with_the_default_system_prompt() -> None:
    tokenizer = tiny_tokenizer()
    ids = encode_chat(tokenizer, "hello world")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids).split() == [
        "<|im_start|>", "system", "[UNK]", "<|im_end|>",
        "<|im_start|>", "user", "hello", "world", "<|im_end|>",
        "<|im_start|>", "assistant",
    ]  # fmt: skip


@pytest.mark.parametrize("tied", [True, False])
def test_freeze_leaves_neither_input_embeddings_nor_head_trainable(tied: bool) -> None:
    model = tiny_model(tied=tied)
    trainable = freeze(model, ["input_embeddings", "output_head"])
    frozen = {
        id(p)
        for module in (model.get_input_embeddings(), model.get_output_embeddings())
        for p in module.parameters()
    }
    assert trainable and all(id(p) not in frozen for _, p in trainable)
    assert all(not p.requires_grad for p in model.parameters() if id(p) in frozen)
    assert all(p.requires_grad for _, p in trainable)
    expected = sum(p.numel() for p in model.parameters() if id(p) not in frozen)
    assert sum(p.numel() for _, p in trainable) == expected


def test_freeze_rejects_unknown_module_names() -> None:
    with pytest.raises(ValueError, match="lm_heads"):
        freeze(tiny_model(), ["lm_heads"])


def test_load_model_neutralizes_the_saved_generation_config(tmp_path) -> None:
    from conftest import tiny_tokenizer

    from zero_field_refusal.models import load_model

    tiny_model().save_pretrained(tmp_path)
    GenerationConfig(**QWEN_DEFAULTS, eos_token_id=[3, 1]).save_pretrained(tmp_path)
    tokenizer = tiny_tokenizer()
    cfg = OmegaConf.create({"name": str(tmp_path), "revision": None, "dtype": "float32",
                            "attn_implementation": "sdpa", "tf32": False,
                            "end_tokens": ["<|im_end|>", "<|endoftext|>"]})  # fmt: skip
    model = load_model(cfg, "cpu", tokenizer)
    assert not model.training
    assert next(model.parameters()).dtype == torch.float32
    diff = model.generation_config.to_diff_dict()
    diff.pop("transformers_version", None)
    assert diff == {"eos_token_id": [3, 1], "pad_token_id": tokenizer.pad_token_id}
    assert model.config._attn_implementation == "sdpa"
