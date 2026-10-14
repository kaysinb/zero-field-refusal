import copy
import hashlib
import json
import re
from collections import Counter

import pytest

from zero_field_refusal import data
from zero_field_refusal.data import (
    SET_NAMES,
    allocate_largest_remainder,
    build_prompt_sets,
    factual_answer_correct,
    load_facts,
    load_prompt_set,
    mmlu_prompt,
    prompt_sets_dir,
)

CFG = {
    "selection_seed": 3,
    "alpaca": {
        "dataset": "tatsu-lab/alpaca",
        "revision": "a" * 40,
        "filter": "no_input",
        "teacher_prompts": 10,
        "benign_kl_prompts": 8,
        "heldout_references": 6,
    },
    "harmful": {
        "sources": [
            {
                "name": "harmbench",
                "url": "https://example.test/harmbench.csv",
                "sha256": "b" * 64,
                "text_column": "Behavior",
                "category_column": "SemanticCategory",
                "where": {"FunctionalCategory": "standard"},
            },
            {
                "name": "advbench",
                "url": "https://example.test/advbench.csv",
                "sha256": "c" * 64,
                "text_column": "goal",
            },
        ],
        "deduplicate": True,
        "kl_prompts": 10,
        "judged_prompts": 20,
    },
    "factual_qa": {"set": "simple_facts_v1", "min_questions": 200},
    "benchmark": {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
        "revision": "d" * 40,
        "questions": 50,
        "stratify": "subject",
    },
}

WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
]


def _distinct_prompt(i: int) -> str:
    # Distinct word sets (Jaccard 0 between different i) so only planted duplicates are dropped.
    return f"Explain topic number {i:03d} " + " ".join(f"{w}{i:03d}" for w in WORDS[:6]) + "."


def fake_rows(source):
    name = source.get("name") or source["dataset"]
    if name == "tatsu-lab/alpaca":
        rows = []
        for i in range(200):
            rows.append(
                {
                    "row_index": i,
                    "instruction": f"Instruction {i}",
                    "input": "context" if i % 5 == 0 else ("  " if i % 7 == 0 else ""),
                    "output": f"Output {i}",
                }
            )
        rows.append({"row_index": 200, "instruction": "", "input": "", "output": "x"})
        rows.append({"row_index": 201, "instruction": "No output", "input": "", "output": " "})
        return rows
    if name == "harmbench":
        return [
            {
                "row_index": i,
                "Behavior": _distinct_prompt(i),
                "SemanticCategory": f"cat{i % 3}",
                "BehaviorID": f"behavior_{i}",
            }
            for i in range(60)
        ]
    if name == "advbench":
        rows = [
            {"row_index": i, "goal": _distinct_prompt(100 + i), "target": "Sure"} for i in range(90)
        ]
        # Exact duplicate (after normalization) of HarmBench row 5.
        rows[3]["goal"] = _distinct_prompt(5).upper().replace(".", "!")
        # Near duplicate of HarmBench row 6: one extra word, Jaccard 12/13 >= 0.8.
        rows[4]["goal"] = _distinct_prompt(6) + " extra"
        # Near duplicate of the earlier AdvBench row 10 inside the same source.
        rows[20]["goal"] = _distinct_prompt(110) + " more"
        return rows
    if name == "cais/mmlu":
        sizes = {"anatomy": 90, "law": 45, "physics": 33, "history": 21, "art": 11}
        rows = []
        for subject, n in sizes.items():
            for j in range(n):
                rows.append(
                    {
                        "row_index": len(rows),
                        "question": f"{subject} question {j}?",
                        "subject": subject,
                        "choices": ["a", "b", "c", "d"],
                        "answer": j % 4,
                    }
                )
        return rows
    raise AssertionError(f"unexpected source {name}")


@pytest.fixture
def built(tmp_path):
    manifest_path = build_prompt_sets(CFG, tmp_path / "sets", load_rows=fake_rows)
    return manifest_path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sets_have_configured_sizes(built):
    d = built.parent
    sizes = {name: len(read_jsonl(d / f"{name}.jsonl")) for name in SET_NAMES}
    assert sizes["alpaca_teacher"] == 10
    assert sizes["alpaca_benign_kl"] == 8
    assert sizes["alpaca_heldout"] == 6
    # 200 rows minus 40 with input "context" minus the blank-input ones, minus 2 empty rows.
    eligible = [
        r
        for r in fake_rows({"dataset": "tatsu-lab/alpaca"})
        if not r["input"].strip() and r["instruction"].strip() and r["output"].strip()
    ]
    assert sizes["alpaca_reference_pool"] == len(eligible) - 24
    assert sizes["harmful_kl"] == 10
    assert sizes["harmful_judged"] == 20
    assert sizes["mmlu"] == 50
    assert sizes["factual_qa"] >= 200


def test_alpaca_sets_disjoint_and_filtered(built):
    d = built.parent
    sets = {n: read_jsonl(d / f"{n}.jsonl") for n in SET_NAMES if n.startswith("alpaca")}
    ids = [item["id"] for items in sets.values() for item in items]
    assert len(ids) == len(set(ids))
    for items in sets.values():
        for item in items:
            index = int(item["id"].split(":")[1])
            assert item["id"] == f"alpaca:{index}"
            assert item["text"] == f"Instruction {index}"
            assert fake_rows({"dataset": "tatsu-lab/alpaca"})[index]["input"].strip() == ""
    for name in ("alpaca_heldout", "alpaca_reference_pool"):
        for item in sets[name]:
            assert item["reference"] == f"Output {item['id'].split(':')[1]}"


def test_harmful_sets_disjoint_and_deduplicated(built):
    d = built.parent
    sets = {n: read_jsonl(d / f"{n}.jsonl") for n in SET_NAMES if n.startswith("harmful")}
    ids = [item["id"] for items in sets.values() for item in items]
    assert len(ids) == len(set(ids))
    assert "advbench:3" not in ids  # exact duplicate of harmbench:5
    assert "advbench:4" not in ids  # near duplicate of harmbench:6
    assert "advbench:20" not in ids  # near duplicate inside AdvBench
    assert "harmbench:5" in ids and "harmbench:6" in ids
    assert len(ids) == 60 + 90 - 3
    for item in sets["harmful_kl"]:
        assert item["source"] in ("harmbench", "advbench")
        assert ("category" in item) == (item["source"] == "harmbench")
        assert ("behavior_id" in item) == (item["source"] == "harmbench")


def test_dedup_statistics_in_manifest(built):
    manifest = json.loads(built.read_text(encoding="utf-8"))
    stats = manifest["sets"]["harmful_kl"]["dedup"]
    assert stats["dropped_exact"] == 1
    assert stats["dropped_near_duplicate"] == 2
    assert stats["jaccard_threshold"] == 0.8
    assert stats["kept"] == 147
    assert stats["by_source"]["advbench"] == {"raw": 90, "kept": 87}
    assert stats["by_source"]["harmbench"] == {"raw": 60, "kept": 60}


def test_harmful_split_is_stratified_by_source(built):
    d = built.parent
    kl = Counter(i["source"] for i in read_jsonl(d / "harmful_kl.jsonl"))
    judged = Counter(i["source"] for i in read_jsonl(d / "harmful_judged.jsonl"))
    # Deduplicated sizes 60 and 87: 10 * 60/147 = 4.08, 20 * 60/147 = 8.16.
    assert kl == {"harmbench": 4, "advbench": 6}
    assert judged == {"harmbench": 8, "advbench": 12}
    calibration = read_jsonl(d / "harmful_calibration.jsonl")
    assert len(calibration) == 147 - 30


def test_without_deduplication_all_prompts_are_kept(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["harmful"]["deduplicate"] = False
    path = build_prompt_sets(cfg, tmp_path, load_rows=fake_rows)
    stats = json.loads(path.read_text(encoding="utf-8"))["sets"]["harmful_kl"]["dedup"]
    assert stats["kept"] == 150 and stats["dropped_exact"] == 0
    assert stats["dropped_near_duplicate"] == 0


def test_mmlu_allocation_per_subject(built):
    items = read_jsonl(built.parent / "mmlu.jsonl")
    counts = Counter(i["subject"] for i in items)
    # Sizes 90/45/33/21/11 of 200 for 50 questions: 22.5, 11.25, 8.25, 5.25, 2.75.
    assert counts == {"anatomy": 23, "law": 11, "physics": 8, "history": 5, "art": 3}
    for item in items:
        assert len(item["choices"]) == 4 and item["answer"] in range(4)
        assert item["text"] == mmlu_prompt(item)
    assert len({i["id"] for i in items}) == 50


def test_largest_remainder_allocation():
    assert allocate_largest_remainder({"a": 60, "b": 87}, 10) == {"a": 4, "b": 6}
    assert allocate_largest_remainder({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert allocate_largest_remainder({"a": 5, "b": 5}, 10) == {"a": 5, "b": 5}
    out = allocate_largest_remainder({"x": 7, "y": 11, "z": 13}, 17)
    assert sum(out.values()) == 17
    with pytest.raises(ValueError):
        allocate_largest_remainder({"a": 2}, 3)


def test_build_is_deterministic(tmp_path):
    a = build_prompt_sets(CFG, tmp_path / "a", load_rows=fake_rows)
    b = build_prompt_sets(CFG, tmp_path / "b", load_rows=fake_rows)
    for name in SET_NAMES:
        assert (a.parent / f"{name}.jsonl").read_bytes() == (
            b.parent / f"{name}.jsonl"
        ).read_bytes()
    ma, mb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    assert {n: s["sha256"] for n, s in ma["sets"].items()} == {
        n: s["sha256"] for n, s in mb["sets"].items()
    }


def test_selection_seed_changes_selection(tmp_path):
    other = copy.deepcopy(CFG)
    other["selection_seed"] = 4
    a = build_prompt_sets(CFG, tmp_path / "a", load_rows=fake_rows)
    b = build_prompt_sets(other, tmp_path / "b", load_rows=fake_rows)
    assert (a.parent / "alpaca_teacher.jsonl").read_bytes() != (
        b.parent / "alpaca_teacher.jsonl"
    ).read_bytes()


def test_jsonl_format_and_manifest_contents(built):
    d = built.parent
    manifest = json.loads(built.read_text(encoding="utf-8"))
    assert manifest["config"] == CFG
    assert manifest["config_hash"] == data.data_config_hash(CFG)
    assert set(manifest["versions"]) == {"datasets", "huggingface_hub"}
    assert manifest["created_utc"].endswith("+00:00")
    assert set(manifest["sets"]) == set(SET_NAMES)
    for name in SET_NAMES:
        raw = (d / f"{name}.jsonl").read_bytes()
        entry = manifest["sets"][name]
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["count"] == len(raw.decode("utf-8").splitlines())
        first = raw.decode("utf-8").splitlines()[0]
        assert list(json.loads(first)) == sorted(json.loads(first))
        assert entry["sources"]
    kl = manifest["sets"]["harmful_kl"]
    assert [(s["name"], s["sha256"]) for s in kl["sources"]] == [
        ("harmbench", "b" * 64),
        ("advbench", "c" * 64),
    ]
    assert kl["sources"][0]["where"] == {"FunctionalCategory": "standard"}
    assert sum(len(v) for v in kl["row_indices"].values()) == 10
    assert manifest["sets"]["alpaca_teacher"]["sources"][0]["revision"] == "a" * 40
    assert manifest["sets"]["factual_qa"]["sources"][0]["sha256"]


def test_prompt_sets_dir_depends_on_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    other = copy.deepcopy(CFG)
    other["alpaca"]["teacher_prompts"] = 11
    d1, d2 = prompt_sets_dir(CFG), prompt_sets_dir(other)
    assert d1 != d2
    assert d1.parent == tmp_path / "prompts"
    assert d1.name == data.data_config_hash(CFG) and len(d1.name) == 12


def test_load_prompt_set_round_trip_and_verification(tmp_path):
    out = prompt_sets_dir(CFG, tmp_path)
    build_prompt_sets(CFG, out, load_rows=fake_rows)
    items = load_prompt_set("alpaca_teacher", CFG, tmp_path)
    assert len(items) == 10 and all({"id", "text"} <= set(i) for i in items)
    assert load_prompt_set("factual_qa", CFG, tmp_path)[0]["text"].endswith(" Answer briefly.")

    other = copy.deepcopy(CFG)
    other["selection_seed"] = 99
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_prompt_set("alpaca_teacher", other, tmp_path)
    with pytest.raises(ValueError, match="Unknown prompt set"):
        load_prompt_set("nope", CFG, tmp_path)

    (out / "alpaca_teacher.jsonl").write_text('{"id": "x", "text": "edited"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prepare_data.py"):
        load_prompt_set("alpaca_teacher", CFG, tmp_path)
    (out / "alpaca_teacher.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_prompt_set("alpaca_teacher", CFG, tmp_path)


def test_load_prompt_set_rejects_manifest_of_another_config(tmp_path):
    out = prompt_sets_dir(CFG, tmp_path)
    build_prompt_sets(CFG, out, load_rows=fake_rows)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["config_hash"] = "0" * 12
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prepare_data.py"):
        load_prompt_set("mmlu", CFG, tmp_path)


def test_too_few_questions_in_resource_is_an_error(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["factual_qa"]["min_questions"] = 10_000
    with pytest.raises(ValueError, match="at least 10000"):
        build_prompt_sets(cfg, tmp_path, load_rows=fake_rows)


def test_not_enough_alpaca_rows_is_an_error(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["alpaca"]["teacher_prompts"] = 1000
    with pytest.raises(ValueError, match="alpaca"):
        build_prompt_sets(cfg, tmp_path, load_rows=fake_rows)


@pytest.mark.parametrize(
    ("text", "answers", "expected"),
    [
        ("The answer is 9.", ["9"], True),
        ("19", ["9"], False),
        ("3.9", ["9"], False),
        ("The result is 91.", ["9"], False),
        ("It is NINE", ["9", "nine"], True),
        ("Paris, of course", ["paris"], True),
        ("Parisian cuisine", ["paris"], False),
        ("The capital is New Delhi.", ["new delhi", "delhi"], True),
        ("It is the Earth’s moon", ["earth's moon"], True),
        ("co2 is absorbed", ["co₂", "co2"], True),
        ("I do not know", ["paris"], False),
        ("", ["paris"], False),
        # Numbers match whole, with their sign and decimal part.
        ("It is 9.5", ["9"], False),
        ("The answer is -9.", ["9"], False),
        ("The answer is −9.", ["9"], False),
        ("It equals 9.0", ["9"], True),
        ("Then 12-9 = 3.", ["9"], False),
        ("Then 12-9 = 3.", ["3"], True),
        ("About 9,000 of them", ["9"], False),
        ("About 9,000 of them", ["9000"], True),
        ("The 9th day", ["9"], False),
        ("(9)", ["9"], True),
        ("9%", ["9"], True),
    ],
)
def test_factual_answer_correct(text, answers, expected):
    assert factual_answer_correct(text, answers) is expected


def test_mmlu_prompt_format():
    item = {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"]}
    assert mmlu_prompt(item) == (
        "What is 2+2?\n\nA. 3\nB. 4\nC. 5\nD. 6\n\nAnswer with the letter of the correct option."
    )


def test_facts_resource():
    facts = load_facts("simple_facts_v1")
    assert len(facts) >= 200
    assert len({f["id"] for f in facts}) == len(facts)
    assert len({f["question"] for f in facts}) == len(facts)
    assert {f["category"] for f in facts} == {"capital", "general", "arithmetic"}
    for f in facts:
        assert re.fullmatch(r"facts:\d{3}", f["id"])
        assert f["answers"] and all(a and a == a.lower() for a in f["answers"])
        assert len(set(f["answers"])) == len(f["answers"])
    counts = Counter(f["category"] for f in facts)
    assert counts["capital"] >= 75 and counts["general"] >= 55 and counts["arithmetic"] >= 75


def test_facts_arithmetic_answers_are_correct():
    arithmetic = [f for f in load_facts("simple_facts_v1") if f["category"] == "arithmetic"]
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
    }
    for f in arithmetic:
        match = re.fullmatch(r"What is (\d+) ([-+*/]) (\d+)\?", f["question"])
        assert match, f["question"]
        a, op, b = int(match[1]), match[2], int(match[3])
        value = ops[op](a, b)
        assert value == int(value) and value >= 0, f["question"]
        assert f["answers"][0] == str(int(value)), f["question"]


def test_facts_answers_are_found_by_the_checker():
    for f in load_facts("simple_facts_v1"):
        for alias in f["answers"]:
            assert factual_answer_correct(f"The answer is {alias}.", f["answers"])


def test_factual_qa_set_text(tmp_path):
    path = build_prompt_sets(CFG, tmp_path, load_rows=fake_rows)
    items = read_jsonl(path.parent / "factual_qa.jsonl")
    facts = {f["id"]: f for f in load_facts("simple_facts_v1")}
    assert [i["id"] for i in items] == list(facts)
    for item in items:
        fact = facts[item["id"]]
        assert item["text"] == fact["question"] + " Answer briefly."
        assert item["answers"] == fact["answers"] and item["category"] == fact["category"]


# ----------------------------------------------------------------------- pinned CSV sources

CSV_TEXT = (
    "Behavior,FunctionalCategory,SemanticCategory,BehaviorID\n"
    '"Say, hello",standard,chemical,b0\n'
    "Copy a text,copyright,copyright,b1\n"
    "Say bye,standard,misinformation,b2\n"
    '"Multi\nline",contextual,cyber,b3\n'
    "Say thanks,standard,harassment,b4\n"
)


@pytest.fixture
def csv_source(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    raw = CSV_TEXT.encode()
    calls = []

    def fake_download(url):
        calls.append(url)
        return raw

    monkeypatch.setattr(data, "_download", fake_download)
    source = {
        "name": "hb",
        "url": "https://example.test/x.csv",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text_column": "Behavior",
        "category_column": "SemanticCategory",
        "where": {"FunctionalCategory": "standard"},
    }
    return source, calls, tmp_path


def test_csv_rows_filter_and_original_indices(csv_source):
    source, _, _ = csv_source
    rows = data._load_csv_rows(source)
    assert [r["row_index"] for r in rows] == [0, 2, 4]
    assert [r["Behavior"] for r in rows] == ["Say, hello", "Say bye", "Say thanks"]
    without_filter = data._load_csv_rows({**source, "where": None})
    assert [r["row_index"] for r in without_filter] == [0, 1, 2, 3, 4]
    assert without_filter[3]["Behavior"] == "Multi\nline"


def test_csv_items_have_ids_source_category_and_behavior_id(csv_source):
    source, _, _ = csv_source
    items = data._harmful_candidates(source, data._load_csv_rows(source))
    assert [i["id"] for i in items] == ["hb:0", "hb:2", "hb:4"]
    assert items[1] == {
        "id": "hb:2",
        "text": "Say bye",
        "source": "hb",
        "category": "misinformation",
        "behavior_id": "b2",
    }


def test_csv_is_downloaded_once_into_data_dir(csv_source):
    source, calls, tmp_path = csv_source
    data._load_csv_rows(source)
    data._load_csv_rows(source)
    assert calls == [source["url"]]
    assert (tmp_path / "sources" / f"{source['sha256']}.csv").read_text() == CSV_TEXT
    assert not list((tmp_path / "sources").glob("*.tmp"))


def test_csv_sha256_mismatch_raises_and_is_not_cached(csv_source):
    source, _, tmp_path = csv_source
    with pytest.raises(ValueError, match="sha256"):
        data._load_csv_rows({**source, "sha256": "0" * 64})
    assert not (tmp_path / "sources").exists()


def test_corrupted_cached_csv_raises(csv_source):
    source, _, tmp_path = csv_source
    data._load_csv_rows(source)
    (tmp_path / "sources" / f"{source['sha256']}.csv").write_text("edited")
    with pytest.raises(ValueError, match="sha256"):
        data._load_csv_rows(source)


def test_default_loader_reads_file_url(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    csv_file = tmp_path / "x.csv"
    csv_file.write_text(CSV_TEXT, encoding="utf-8")
    source = {
        "url": csv_file.as_uri(),
        "sha256": hashlib.sha256(CSV_TEXT.encode()).hexdigest(),
    }
    assert len(data._load_default_rows(source)) == 5


def test_alpaca_selection_does_not_depend_on_row_order(tmp_path):
    def reversed_rows(source):
        rows = fake_rows(source)
        return rows[::-1] if source.get("dataset") == "tatsu-lab/alpaca" else rows

    a = build_prompt_sets(CFG, tmp_path / "a", load_rows=fake_rows)
    b = build_prompt_sets(CFG, tmp_path / "b", load_rows=reversed_rows)
    for name in ("alpaca_teacher", "alpaca_reference_pool"):
        assert (a.parent / f"{name}.jsonl").read_bytes() == (
            b.parent / f"{name}.jsonl"
        ).read_bytes()


def test_the_monitor_set_follows_the_heldout_prompts_and_changes_no_other_set(tmp_path):
    with_monitor = copy.deepcopy(CFG)
    with_monitor["alpaca"]["monitor_prompts"] = 5
    plain = build_prompt_sets(CFG, tmp_path / "plain", load_rows=fake_rows).parent
    monitored = build_prompt_sets(with_monitor, tmp_path / "monitor", load_rows=fake_rows).parent
    assert data.set_names(with_monitor) == (*SET_NAMES, "alpaca_monitor")
    for name in ("alpaca_teacher", "alpaca_benign_kl", "alpaca_heldout"):
        assert read_jsonl(plain / f"{name}.jsonl") == read_jsonl(monitored / f"{name}.jsonl")
    monitor = read_jsonl(monitored / "alpaca_monitor.jsonl")
    pool = read_jsonl(plain / "alpaca_reference_pool.jsonl")
    assert [m["id"] for m in monitor] == [p["id"] for p in pool[:5]]
    assert all("reference" not in item for item in monitor)  # instructions only
    assert read_jsonl(monitored / "alpaca_reference_pool.jsonl") == pool[5:]
    assert not (plain / "alpaca_monitor.jsonl").exists()


def test_the_monitor_set_loads_only_where_configured(tmp_path):
    with_monitor = copy.deepcopy(CFG)
    with_monitor["alpaca"]["monitor_prompts"] = 5
    build_prompt_sets(with_monitor, prompt_sets_dir(with_monitor, tmp_path), load_rows=fake_rows)
    assert len(load_prompt_set("alpaca_monitor", with_monitor, tmp_path)) == 5
    with pytest.raises(ValueError, match="Unknown prompt set"):
        load_prompt_set("alpaca_monitor", CFG, tmp_path)
