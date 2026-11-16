"""The Llama tokenizer's pins (models.pin_tokenizer): a pad token from the model config for a
tokenizer that ships none, and the chat template's date fixed so that a prompt never depends on
the day it is encoded. A config without the keys (Qwen2.5) leaves the tokenizer as shipped."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from zero_field_refusal import models
from zero_field_refusal.config import load_config

# Llama 3.x's own date logic, cut down: the day of the call unless date_string is defined
TEMPLATE = (
    '{%- if not date_string is defined %}{%- if strftime_now is defined %}'
    '{%- set date_string = strftime_now("%d %b %Y") %}{%- else %}'
    '{%- set date_string = "26 Jul 2024" %}{%- endif %}{%- endif %}'
    "Today Date: {{ date_string }}|{% for m in messages %}{{ m['content'] }}{% endfor %}"
)


class FakeTokenizer(SimpleNamespace):
    vocab = {"<|finetune_right_pad_id|>": 128004, "<|eot_id|>": 128009}
    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def render(self, text, **context):
        from jinja2 import Environment

        env = Environment()
        env.globals["strftime_now"] = lambda fmt: "16 Nov 2026"  # the day of the call
        return env.from_string(self.chat_template).render(
            messages=[{"role": "user", "content": text}], **context)


def cfg(**keys):
    return OmegaConf.create(keys)


def test_the_date_is_pinned_whatever_the_day():
    tok = FakeTokenizer(chat_template=TEMPLATE, pad_token=None)
    assert tok.render("hi").startswith("Today Date: 16 Nov 2026")  # unpinned: the call's day
    models.pin_tokenizer(tok, cfg(chat_date="26 Jul 2024"))
    assert tok.render("hi") == "Today Date: 26 Jul 2024|hi"


def test_the_pad_token_is_set_from_the_vocabulary():
    tok = FakeTokenizer(chat_template=TEMPLATE, pad_token=None)
    models.pin_tokenizer(tok, cfg(pad_token="<|finetune_right_pad_id|>"))
    assert tok.pad_token == "<|finetune_right_pad_id|>"
    with pytest.raises(ValueError, match="vocabulary"):
        models.pin_tokenizer(FakeTokenizer(chat_template=TEMPLATE), cfg(pad_token="<|nope|>"))


def test_no_keys_change_nothing():
    tok = FakeTokenizer(chat_template=TEMPLATE, pad_token="<|endoftext|>")
    models.pin_tokenizer(tok, cfg())
    assert tok.chat_template == TEMPLATE and tok.pad_token == "<|endoftext|>"


@pytest.mark.parametrize("date", ['26 Jul" %}{{x', "{{1}}", 7])
def test_a_date_that_is_not_plain_text_is_refused(date):
    with pytest.raises(ValueError):
        models.pin_tokenizer(FakeTokenizer(chat_template=TEMPLATE), cfg(chat_date=date))


def test_a_template_without_the_date_refuses_a_pinned_date():
    with pytest.raises(ValueError, match="date_string"):
        models.pin_tokenizer(FakeTokenizer(chat_template="{{ messages }}"), cfg(chat_date="1 Jan"))


def test_the_llama_config_pins_both_and_qwens_neither():
    llama = load_config("model/llama-3.2-1b-instruct").model
    assert llama.pad_token == "<|finetune_right_pad_id|>" and llama.chat_date == "26 Jul 2024"
    assert llama.revision == "9213176726f574b556790deb65791e0c5aa438b6"
    qwen = load_config("model/qwen2.5-1.5b-instruct").model
    assert qwen.get("pad_token") is None and qwen.get("chat_date") is None
