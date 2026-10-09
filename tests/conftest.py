"""Tiny random Qwen2 models for CPU tests of the zero-field pipeline; nothing is downloaded."""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

VOCAB = 64
END_IDS = [5, 6]  # like Qwen2.5: <|im_end|> and <|endoftext|>
PAD_ID = 6  # the pad id equals one of the end ids, as in Qwen2.5


def tiny_model(*, tied: bool = True, seed: int = 0, head_std: float = 0.5) -> Qwen2ForCausalLM:
    """A random fp32 model; an untied head with a small ``head_std`` samples nearly uniformly."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=tied,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(torch.float32).eval()
    torch.nn.init.normal_(model.model.embed_tokens.weight, std=0.5)
    if not tied:
        torch.nn.init.normal_(model.lm_head.weight, std=head_std)
    return model


@pytest.fixture
def model() -> Qwen2ForCausalLM:
    return tiny_model()


def prompts(n: int, *, seed: int = 0, min_len: int = 3, max_len: int = 9) -> list[list[int]]:
    """Prompts of different lengths without end tokens."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        length = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        out.append(torch.randint(7, VOCAB, (length,), generator=g).tolist())
    return out


SPECIAL = ["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
TEMPLATE_WORDS = ["system", "user", "assistant", "default", "A", "B", "C", "D"]


def tiny_tokenizer(vocab_size: int = VOCAB) -> PreTrainedTokenizerFast:
    """A whitespace word-level tokenizer with a Qwen-like chat template and a default system turn.

    Words ``w0``, ``w1``, ... fill the vocabulary; ``<|endoftext|>`` is the pad token.
    """
    words = SPECIAL + TEMPLATE_WORDS
    words += [f"w{i}" for i in range(vocab_size - len(words))]
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
