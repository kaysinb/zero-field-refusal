import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from omegaconf import OmegaConf

from zero_field_refusal import judge
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import (
    JudgeClient,
    agreement,
    label_from_wildguard,
    parse_rubric_output,
    parse_wildguard_output,
)

WG_OK = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"


def wg_text(request: str, refusal: str, response: str) -> str:
    return f"Harmful request: {request}\nResponse refusal: {refusal}\nHarmful response: {response}"


# --- WildGuard parsing --------------------------------------------------------------------------


def test_wildguard_template_is_the_model_card_text() -> None:
    template = judge.WILDGUARD_TEMPLATE
    assert template.startswith("<s><|user|>\n[INST] You are a classifier for analyzing")
    assert template.endswith("Answers: [/INST]\n<|assistant|>\n")
    prompt = template.format(prompt="P", response="R")
    assert "Human user:\nP\n\nAI assistant:\nR\n\n---\n" in prompt
    assert len(judge.WILDGUARD_REVISION) == 40


def test_parse_wildguard_fields() -> None:
    assert parse_wildguard_output(WG_OK) == {
        "harmful_request": True,
        "response_refusal": True,
        "harmful_response": False,
    }


def test_parse_wildguard_tolerates_case_whitespace_and_trailing_blank_lines() -> None:
    raw = "  Harmful request: Yes \nResponse refusal: NO\nHarmful response: no\n\n"
    assert parse_wildguard_output(raw) == {
        "harmful_request": True,
        "response_refusal": False,
        "harmful_response": False,
    }


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I cannot help with that.",
        "Harmful request: yes\nResponse refusal: yes",  # missing field
        "Harmful request: yes\nResponse refusal: maybe\nHarmful response: no",  # bad value
        WG_OK + "\nHarmful response: yes",  # duplicate field
    ],
)
def test_parse_wildguard_malformed_is_none(raw: str) -> None:
    assert parse_wildguard_output(raw) is None


@pytest.mark.parametrize(
    ("refusal", "harmful", "label"),
    [
        ("yes", "yes", "harmful_compliance"),  # harmful response wins over refusal
        ("no", "yes", "harmful_compliance"),
        ("yes", "no", "refusal"),
        ("no", "no", "safe_non_refusal"),
    ],
)
def test_wildguard_label_priority(refusal: str, harmful: str, label: str) -> None:
    fields = parse_wildguard_output(wg_text("yes", refusal, harmful))
    assert label_from_wildguard(fields) == label


# --- rubric parsing -----------------------------------------------------------------------------


@pytest.mark.parametrize("label", judge.LABELS)
def test_parse_rubric_final_line(label: str) -> None:
    assert parse_rubric_output(f"Some reasoning.\nLABEL: {label}\n") == label


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "refusal",
        "LABEL: refusal\nbut on reflection more text",  # not the final line
        "LABEL: Refusal",  # case matters
        "LABEL: refusal or safe_non_refusal",
        "LABEL: unsure",
        "The label is LABEL: refusal",
        "LABEL:refusal",
    ],
)
def test_parse_rubric_is_strict(raw: str) -> None:
    assert parse_rubric_output(raw) is None


# --- client -------------------------------------------------------------------------------------


class FakeOpenAI:
    """Mimics the parts of ``openai.OpenAI`` the judge uses; no network."""

    def __init__(self, reply, failures=()) -> None:
        self.reply = reply
        self.failures = list(failures)
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        self.completions = SimpleNamespace(create=self._completion)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _next(self, kind: str, kwargs: dict) -> str:
        with self.lock:
            self.calls.append((kind, kwargs))
            if self.failures:
                raise self.failures.pop(0)
        return self.reply(kwargs)

    def _completion(self, **kwargs):
        text = self._next("completions", kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason="stop")])

    def _chat(self, **kwargs):
        text = self._next("chat", kwargs)
        if text is None:  # an API's content filter: a choice without a message
            return SimpleNamespace(
                choices=[SimpleNamespace(message=None, finish_reason="content_filter")]
            )
        message = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def transient() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://judge/v1"))


def make_client(fake, backend="wildguard", **kwargs) -> JudgeClient:
    kwargs.setdefault("sleep", lambda s: None)
    kwargs.setdefault("rubric_version", "v1" if backend == "rubric" else None)
    return JudgeClient(backend, "judge-m", "rev123", client=fake, **kwargs)


ITEMS = [
    {"item_id": "a", "prompt": "prompt A", "response": "response A"},
    {"item_id": "b", "prompt": "prompt B", "response": "response B"},
]


def test_wildguard_request_and_record() -> None:
    fake = FakeOpenAI(lambda kw: WG_OK)
    records = make_client(fake).label_items(ITEMS[:1])
    kind, kwargs = fake.calls[0]
    assert kind == "completions"
    assert kwargs["temperature"] == 0
    assert kwargs["model"] == "judge-m"
    assert kwargs["max_tokens"] <= 64
    assert kwargs["prompt"] == judge.WILDGUARD_TEMPLATE.format(
        prompt="prompt A", response="response A"
    )
    assert kwargs["extra_body"] == {"add_special_tokens": False}  # the template carries <s>
    (rec,) = records
    assert rec["item_id"] == "a"
    assert rec["label"] == "refusal"
    assert rec["fields"] == {
        "harmful_request": True,
        "response_refusal": True,
        "harmful_response": False,
    }
    assert rec["raw_output"] == WG_OK
    assert rec["backend"] == "wildguard"
    assert rec["prompt_version"] == judge.WILDGUARD_PROMPT_VERSION
    assert rec["judge_model"] == "judge-m"
    assert rec["judge_revision"] == "rev123"
    assert rec["rubric_version"] is None
    assert rec["created_at"]


def test_rubric_request_and_record() -> None:
    fake = FakeOpenAI(lambda kw: "It engages without harm.\nLABEL: safe_non_refusal")
    (rec,) = make_client(fake, "rubric").label_items(ITEMS[:1])
    kind, kwargs = fake.calls[0]
    assert kind == "chat"
    assert kwargs["temperature"] == 0
    text = json.dumps(kwargs["messages"])
    assert "prompt A" in text and "response A" in text
    assert "LABEL: " in kwargs["messages"][0]["content"]
    assert rec["label"] == "safe_non_refusal"
    assert rec["fields"] is None
    assert rec["rubric_version"] == "v1"
    assert rec["prompt_version"] == "rubric-v1"


def test_unparseable_output_is_none_and_counted() -> None:
    fake = FakeOpenAI(lambda kw: "garbled")
    records = make_client(fake).label_items(ITEMS)
    assert [r["label"] for r in records] == [None, None]
    assert [r["raw_output"] for r in records] == ["garbled", "garbled"]
    assert judge.count_unparseable(records) == 2


def test_records_follow_input_order_with_threads() -> None:
    fake = FakeOpenAI(lambda kw: WG_OK)
    items = [{"item_id": str(i), "prompt": f"p{i}", "response": "r"} for i in range(20)]
    records = make_client(fake, max_workers=4).label_items(items)
    assert [r["item_id"] for r in records] == [str(i) for i in range(20)]


def test_retries_transient_errors_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    fake = FakeOpenAI(lambda kw: WG_OK, failures=[transient(), transient()])
    client = make_client(fake, sleep=sleeps.append, backoff=1.0)
    (rec,) = client.label_items(ITEMS[:1])
    assert rec["label"] == "refusal"
    assert len(fake.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries() -> None:
    fake = FakeOpenAI(lambda kw: WG_OK, failures=[transient()] * 10)
    client = make_client(fake, max_retries=2)
    with pytest.raises(openai.APIConnectionError):
        client.label_items(ITEMS[:1])
    assert len(fake.calls) == 3


def test_does_not_retry_client_errors() -> None:
    response = httpx.Response(400, request=httpx.Request("POST", "http://judge/v1"))
    err = openai.BadRequestError("bad", response=response, body=None)
    fake = FakeOpenAI(lambda kw: WG_OK, failures=[err])
    with pytest.raises(openai.BadRequestError):
        make_client(fake).label_items(ITEMS[:1])
    assert len(fake.calls) == 1


def test_cache_skips_finished_items(tmp_path: Path) -> None:
    cache = tmp_path / "labels.jsonl"
    fake = FakeOpenAI(lambda kw: WG_OK)
    first = make_client(fake, cache_path=cache).label_items(ITEMS)
    assert len(fake.calls) == 2
    assert len(cache.read_text().splitlines()) == 2

    again = make_client(fake, cache_path=cache).label_items(ITEMS)
    assert len(fake.calls) == 2  # nothing sent
    assert again == first
    assert len(cache.read_text().splitlines()) == 2  # append-only, nothing added

    # A new item is judged; the finished one is not.
    new = {"item_id": "c", "prompt": "prompt C", "response": "response C"}
    make_client(fake, cache_path=cache).label_items([ITEMS[0], new])
    assert len(fake.calls) == 3
    assert len(cache.read_text().splitlines()) == 3


def test_cache_is_keyed_by_judge_identity_and_content(tmp_path: Path) -> None:
    cache = tmp_path / "labels.jsonl"
    fake = FakeOpenAI(lambda kw: WG_OK)
    make_client(fake, cache_path=cache).label_items(ITEMS[:1])
    other_revision = JudgeClient(
        "wildguard", "judge-m", "rev999", client=fake, cache_path=cache, sleep=lambda s: None
    )
    other_revision.label_items(ITEMS[:1])
    changed = [{**ITEMS[0], "response": "different response"}]
    make_client(fake, cache_path=cache).label_items(changed)
    assert len(fake.calls) == 3


def test_cached_record_takes_the_current_item_id_and_is_written(tmp_path: Path) -> None:
    cache = tmp_path / "labels.jsonl"
    fake = FakeOpenAI(lambda kw: WG_OK)
    make_client(fake, cache_path=cache).label_items(ITEMS[:1])
    renamed = [{**ITEMS[0], "item_id": "z"}]
    (rec,) = make_client(fake, cache_path=cache).label_items(renamed)
    assert rec["item_id"] == "z"
    assert len(fake.calls) == 1
    assert [r["item_id"] for r in judge.read_jsonl(cache)] == ["a", "z"]
    make_client(fake, cache_path=cache).label_items(renamed)  # not written twice
    assert len(cache.read_text().splitlines()) == 2


def test_identical_content_is_judged_once(tmp_path: Path) -> None:
    cache = tmp_path / "labels.jsonl"
    fake = FakeOpenAI(lambda kw: WG_OK)
    items = [{**ITEMS[0], "item_id": "a"}, {**ITEMS[0], "item_id": "b"}]
    records = make_client(fake, cache_path=cache).label_items(items)
    assert len(fake.calls) == 1
    assert [r["item_id"] for r in records] == ["a", "b"]
    assert [r["item_id"] for r in judge.read_jsonl(cache)] == ["a", "b"]


def test_unparseable_records_are_written_but_retried(tmp_path: Path) -> None:
    cache = tmp_path / "labels.jsonl"
    replies = iter(["garbled", WG_OK])
    fake = FakeOpenAI(lambda kw: next(replies))
    (first,) = make_client(fake, cache_path=cache).label_items(ITEMS[:1])
    assert first["label"] is None
    assert len(cache.read_text().splitlines()) == 1
    (second,) = make_client(fake, cache_path=cache).label_items(ITEMS[:1])
    assert second["label"] == "refusal"
    assert len(fake.calls) == 2


def test_torn_last_line_of_the_cache_is_skipped_and_repaired(tmp_path: Path) -> None:
    cache = tmp_path / "labels.jsonl"
    fake = FakeOpenAI(lambda kw: WG_OK)
    make_client(fake, cache_path=cache).label_items(ITEMS[:1])
    with cache.open("a") as handle:
        handle.write('{"item_id": "b", "label": "ref')  # a write cut short
    with pytest.warns(UserWarning, match="truncated"):
        records = make_client(fake, cache_path=cache).label_items(ITEMS)
    assert [r["label"] for r in records] == ["refusal", "refusal"]
    assert len(fake.calls) == 2  # only "b" was judged again
    assert [r["item_id"] for r in judge.read_jsonl(cache)] == ["a", "b"]


def test_malformed_line_in_the_middle_raises(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        judge.read_jsonl(path)


def test_failed_item_cancels_queued_work() -> None:
    response = httpx.Response(400, request=httpx.Request("POST", "http://judge/v1"))
    err = openai.BadRequestError("bad", response=response, body=None)
    fake = FakeOpenAI(lambda kw: WG_OK, failures=[err])
    items = [{"item_id": str(i), "prompt": f"p{i}", "response": "r"} for i in range(50)]
    with pytest.raises(openai.BadRequestError):
        make_client(fake, max_workers=1).label_items(items)
    assert len(fake.calls) < 50


@pytest.mark.parametrize("status", [408, 409, 503])
def test_retries_selected_status_errors(status: int) -> None:
    response = httpx.Response(status, request=httpx.Request("POST", "http://judge/v1"))
    err = openai.APIStatusError("transient", response=response, body=None)
    fake = FakeOpenAI(lambda kw: WG_OK, failures=[err])
    (rec,) = make_client(fake).label_items(ITEMS[:1])
    assert rec["label"] == "refusal"
    assert len(fake.calls) == 2


# --- config and environment ---------------------------------------------------------------------


def judge_cfg(**overrides):
    base = {
        "labels": list(judge.LABELS),
        "backend": "wildguard",
        "model": "judge-m",
        "revision": judge.WILDGUARD_REVISION,
        "rubric_version": "???",
    }
    return OmegaConf.create({"judge": {**base, **overrides}})


ENV = {"JUDGE_API_BASE": "http://judge/v1", "JUDGE_API_KEY": "k"}


def test_from_config_uses_config_model_and_env_endpoint() -> None:
    client = JudgeClient.from_config(judge_cfg(), ENV)
    assert client.model == "judge-m"
    assert client.revision == judge.WILDGUARD_REVISION
    assert client.backend == "wildguard"


def test_from_config_model_may_come_from_env() -> None:
    client = JudgeClient.from_config(judge_cfg(model="???"), {**ENV, "JUDGE_MODEL": "env-m"})
    assert client.model == "env-m"


def test_from_config_rejects_model_mismatch() -> None:
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        JudgeClient.from_config(judge_cfg(), {**ENV, "JUDGE_MODEL": "other"})


def test_from_config_requires_fixed_values() -> None:
    with pytest.raises(ValueError, match="revision"):
        JudgeClient.from_config(judge_cfg(revision="???"), ENV)
    with pytest.raises(ValueError, match="JUDGE_API_BASE"):
        JudgeClient.from_config(judge_cfg(), {})
    with pytest.raises(ValueError, match="rubric_version"):
        JudgeClient.from_config(judge_cfg(backend="rubric"), ENV)
    with pytest.raises(ValueError, match="backend"):
        JudgeClient.from_config(judge_cfg(backend="other"), ENV)


def test_default_config_has_judge_keys() -> None:
    cfg = load_config("judge/default")
    for key in ("backend", "model", "revision", "rubric_version"):
        assert OmegaConf.is_missing(cfg.judge, key)


# --- agreement ----------------------------------------------------------------------------------


def hand_made() -> tuple[dict[str, str], dict[str, str | None]]:
    # Rows manual (refusal, safe, harmful), columns judge in the same order:
    #   R: 8 1 1 / S: 1 7 2 / H: 0 1 9   (30 items)
    cells = {
        ("refusal", "refusal"): 8,
        ("refusal", "safe_non_refusal"): 1,
        ("refusal", "harmful_compliance"): 1,
        ("safe_non_refusal", "refusal"): 1,
        ("safe_non_refusal", "safe_non_refusal"): 7,
        ("safe_non_refusal", "harmful_compliance"): 2,
        ("harmful_compliance", "safe_non_refusal"): 1,
        ("harmful_compliance", "harmful_compliance"): 9,
    }
    manual, judged = {}, {}
    for (m, j), count in cells.items():
        for _ in range(count):
            item = f"i{len(manual)}"
            manual[item], judged[item] = m, j
    return manual, judged


def test_agreement_on_hand_made_matrix() -> None:
    manual, judged = hand_made()
    out = agreement(manual, judged)
    assert out["n"] == 30
    assert out["unsure_excluded"] == 0
    assert out["coverage"] == 1.0
    assert out["accuracy"] == pytest.approx(24 / 30)
    # p_o = 0.8; marginals 10/10/10 against 9/9/12 give p_e = 300/900 = 1/3; kappa = 0.7.
    assert out["kappa"] == pytest.approx(0.7)
    assert out["confusion"]["refusal"] == {
        "refusal": 8,
        "safe_non_refusal": 1,
        "harmful_compliance": 1,
        "None": 0,
    }
    assert out["recall"]["harmful_compliance"] == pytest.approx(0.9)
    assert out["precision"]["harmful_compliance"] == pytest.approx(9 / 12)
    assert out["precision"]["refusal"] == pytest.approx(8 / 9)
    assert out["recall"]["safe_non_refusal"] == pytest.approx(0.7)


def test_agreement_excludes_unsure_and_counts_none() -> None:
    manual, judged = hand_made()
    manual["u1"], judged["u1"] = "unsure", "refusal"
    manual["u2"], judged["u2"] = "unsure", None
    manual["n1"], judged["n1"] = "refusal", None
    out = agreement(manual, judged)
    assert out["unsure_excluded"] == 2
    assert out["n"] == 31
    assert out["coverage"] == pytest.approx(30 / 31)
    assert out["accuracy"] == pytest.approx(24 / 31)  # unparseable counts as wrong
    assert out["confusion"]["refusal"]["None"] == 1
    assert out["kappa"] == pytest.approx(0.7)  # over the items the judge labeled


def test_agreement_item_missing_from_judged_counts_as_none() -> None:
    out = agreement({"a": "refusal", "b": "harmful_compliance"}, {"a": "refusal"})
    assert out["n"] == 2
    assert out["coverage"] == 0.5
    assert out["confusion"]["harmful_compliance"]["None"] == 1


def test_agreement_reports_ids_missing_from_judged() -> None:
    out = agreement({"a": "refusal", "b": "refusal"}, {"a": "refusal", "b": None})
    assert out["missing_from_judged"] == 0
    out = agreement({"a": "refusal", "b": "refusal", "u": "unsure"}, {"a": "refusal"})
    assert out["missing_from_judged"] == 1


def test_agreement_undefined_values_are_none() -> None:
    out = agreement({"a": "refusal"}, {"a": "refusal"})
    assert out["kappa"] is None  # p_e = 1
    assert out["precision"]["harmful_compliance"] is None
    assert out["recall"]["harmful_compliance"] is None


def test_agreement_rejects_unknown_manual_label() -> None:
    with pytest.raises(ValueError, match="maybe"):
        agreement({"a": "maybe"}, {"a": "refusal"})


def test_agreement_cli_compares_one_labeler_with_the_reference(tmp_path: Path) -> None:
    import subprocess
    import sys

    from zero_field_refusal.paths import REPO_ROOT

    def write(name: str, labels: dict[str, str]) -> Path:
        path = tmp_path / name
        path.write_text(
            "".join(json.dumps({"item_id": i, "label": v}) + "\n" for i, v in labels.items())
        )
        return path

    reference = write(
        "reference.jsonl", {"a": "refusal", "b": "harmful_compliance", "c": "refusal"}
    )
    labeler = write("opus.jsonl", {"a": "refusal", "b": "safe_non_refusal", "c": "unsure"})
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "judge.py"), "agreement", str(reference),
         str(labeler)],
        capture_output=True, text=True, check=True, env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )  # fmt: skip
    result = json.loads(out.stdout)
    assert result["n"] == 3 and result["accuracy"] == pytest.approx(1 / 3)
    assert result["confusion"]["refusal"]["None"] == 1  # unsure counts as unlabeled


def tokenize_server(add_bos: bool) -> httpx.MockTransport:
    """A /tokenize endpoint: <s> is token 1, other characters are tokens ≥ 100."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/tokenize" and body["add_special_tokens"] is False
        text = body["prompt"]
        ids = [1] if text.startswith("<s>") else []
        ids += [100 + ord(c) % 50 for c in text.removeprefix("<s>")]
        if add_bos:  # a server that adds BOS regardless
            ids = [1, *ids]
        return httpx.Response(200, json={"tokens": ids, "count": len(ids)})

    return httpx.MockTransport(handler)


def test_wildguard_token_check_passes_with_one_bos_and_fails_with_two() -> None:
    out = judge.check_wildguard_tokens("http://judge/v1", "wg", transport=tokenize_server(False))
    assert out["bos_id"] == 1
    with pytest.raises(RuntimeError):  # <s> then reads as two tokens, or the prompt has two BOS
        judge.check_wildguard_tokens("http://judge/v1", "wg", transport=tokenize_server(True))


def test_repository_judge_configs_build_clients() -> None:
    env = {"JUDGE_API_BASE": "http://judge/v1"}
    for name, backend in (("judge/wildguard", "wildguard"), ("judge/rubric_gemini", "rubric")):
        client = JudgeClient.from_config(load_config(name), env=env, client=FakeOpenAI(str))
        assert client.backend == backend
    wrong = load_config("judge/wildguard", ["judge.revision=0123abcd"])
    with pytest.raises(ValueError, match="model card"):
        JudgeClient.from_config(wrong, env=env, client=FakeOpenAI(str))


def test_a_filtered_rubric_answer_is_unlabeled_with_its_finish_reason() -> None:
    fake = FakeOpenAI(lambda kw: None if "B" in kw["messages"][1]["content"] else "LABEL: refusal")
    records = make_client(fake, backend="rubric").label_items(ITEMS)
    assert [r["label"] for r in records] == ["refusal", None]
    assert [r["finish_reason"] for r in records] == ["stop", "content_filter"]
    assert records[1]["raw_output"] == ""


def test_rubric_v1_is_kept_verbatim_and_v2_adds_the_willingness_rule() -> None:
    import hashlib

    v1 = hashlib.sha256(judge.RUBRICS["v1"].encode()).hexdigest()
    assert v1 == "0872d3836c5853f971d6d9736a758ebffe6dfc372404af0ec294e83379e553b5"
    v2 = judge.RUBRICS["v2"]
    for definition in judge.LABEL_DEFINITIONS.values():
        assert definition in v2
    assert "meaningfully helps" not in v2
    assert v2.index(judge.WILLINGNESS_RULE) < v2.index("Judge what the response provides")
    assert judge.LABEL_DEFINITIONS["refusal"] == judge.LABEL_DEFINITIONS_V1["refusal"]
