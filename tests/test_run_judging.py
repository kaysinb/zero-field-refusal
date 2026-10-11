"""Labeling run directories with a judge: items, label files the analysis accepts, the cache."""

import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from zero_field_refusal.analysis.evidence import collect_runs, label_path
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import JudgeClient, judge_identity
from zero_field_refusal.run_judging import judge_items, label_runs

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "evidence" / "runs"
EXPERIMENT = "evidence_fixture"
JUDGE = "wildguard"
RUN = "sgd_lr0.01_s0"  # completed, 10 responses of the trained model
FAILED = "sgd_lr0.03_s0"
PROMPTS = {f"judged:{i}": f"Placeholder prompt {i}." for i in range(10)}
WG_REFUSAL = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"


class FakeWildGuard:
    """The parts of ``openai.OpenAI`` the WildGuard backend uses; counts the requests."""

    def __init__(self, reply: str = WG_REFUSAL) -> None:
        self.reply, self.calls, self.lock = reply, 0, threading.Lock()
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        with self.lock:
            self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(text=self.reply, finish_reason="stop")])


def copy_runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    shutil.copytree(FIXTURE, root)
    return root


def run_dir(root: Path, run: str = RUN) -> Path:
    return root / EXPERIMENT / run


def factory(fake: FakeWildGuard):
    cfg = load_config(f"judge/{JUDGE}")
    env = {"JUDGE_API_BASE": "http://judge/v1"}
    return lambda path: JudgeClient.from_config(cfg, env, client=fake, cache_path=path,
                                                sleep=lambda s: None)  # fmt: skip


def test_items_cover_every_response_keyed_by_condition_and_prompt() -> None:
    items = judge_items(run_dir(FIXTURE), PROMPTS)
    assert [item["item_id"] for item in items] == [f"trained/judged:{i}" for i in range(10)]
    assert items[3]["prompt"] == "Placeholder prompt 3."
    assert items[3]["response"] == "Placeholder response 3."


def test_a_response_to_a_prompt_outside_the_set_stops_the_run() -> None:
    with pytest.raises(ValueError, match="1 unexpected .first: trained/judged:9"):
        judge_items(run_dir(FIXTURE), {k: v for k, v in PROMPTS.items() if k != "judged:9"})


def test_a_prompt_answered_twice_under_one_condition_stops_the_run(tmp_path) -> None:
    root = copy_runs(tmp_path)
    responses = run_dir(root) / "responses" / "harmful_judged.jsonl"
    lines = responses.read_text().splitlines()
    responses.write_text("\n".join([*lines, lines[0]]) + "\n")
    with pytest.raises(ValueError, match="answers trained/judged:0 twice"):
        judge_items(run_dir(root), PROMPTS)


def test_a_truncated_responses_file_stops_that_run_and_the_others_go_on(tmp_path) -> None:
    root = copy_runs(tmp_path)
    responses = run_dir(root) / "responses" / "harmful_judged.jsonl"
    responses.write_text("\n".join(responses.read_text().splitlines()[:7]) + "\n")
    shutil.rmtree(run_dir(root) / "judge")
    other = run_dir(root, "adamw_lr3e-05_s0")
    shutil.rmtree(other / "judge")
    fake = FakeWildGuard()
    summaries = label_runs([run_dir(root), other], JUDGE, lambda cfg: PROMPTS, factory(fake))
    assert "3 missing (first: trained/judged:7)" in summaries[0]["error"]
    assert not label_path(run_dir(root), JUDGE).exists()
    assert summaries[1] == {"run_id": "adamw_lr3e-05_s0", "items": 10, "unparseable": 0}


def test_the_expected_conditions_follow_the_run_config() -> None:
    from omegaconf import OmegaConf

    from zero_field_refusal.run_judging import response_conditions

    cfg = OmegaConf.load(run_dir(FIXTURE) / "config.yaml")
    assert response_conditions(cfg) == ["trained"]
    cfg.eval.controls.line_scan.enabled = True
    cfg.eval.controls.random_direction.enabled = True
    assert response_conditions(cfg) == [
        "trained", "line_scan_+0.25", "line_scan_+0.5", "line_scan_+1", "line_scan_-0.25",
        "line_scan_-0.5", "line_scan_-1", "random_global", "random_per_tensor"]  # fmt: skip
    cfg.run_type = "theta0"
    assert response_conditions(cfg) == ["theta0"]


def test_labeled_runs_enter_the_analysis_and_a_rerun_sends_nothing(tmp_path) -> None:
    root = copy_runs(tmp_path)
    shutil.rmtree(run_dir(root) / "judge")
    fake = FakeWildGuard()
    summaries = label_runs([run_dir(root), run_dir(root, FAILED)], JUDGE, lambda cfg: PROMPTS,
                           factory(fake))  # fmt: skip
    assert summaries == [{"run_id": RUN, "items": 10, "unparseable": 0},
                         {"run_id": FAILED, "skipped": "status 'failed'"}]  # fmt: skip
    assert fake.calls == 10
    labels = label_path(run_dir(root), JUDGE).read_text()
    records = [json.loads(line) for line in labels.splitlines()]
    identity = judge_identity(load_config(f"judge/{JUDGE}"))
    assert all(record[key] == identity[key] for record in records for key in identity)

    row = collect_runs(root, EXPERIMENT, JUDGE, identity).considered.set_index("run_id").loc[RUN]
    assert (row["included"], row["harmful"], row["judged"]) == (True, 0, 10)

    label_runs([run_dir(root)], JUDGE, lambda cfg: PROMPTS, factory(fake))
    assert fake.calls == 10  # every item came from the label file


def test_unparseable_outputs_are_counted_and_retried(tmp_path) -> None:
    root = copy_runs(tmp_path)
    shutil.rmtree(run_dir(root) / "judge")
    broken = FakeWildGuard(reply="I cannot classify this.")
    summaries = label_runs([run_dir(root)], JUDGE, lambda cfg: PROMPTS, factory(broken))
    assert summaries[0]["unparseable"] == 10
    fixed = FakeWildGuard()
    summaries = label_runs([run_dir(root)], JUDGE, lambda cfg: PROMPTS, factory(fixed))
    assert (summaries[0]["unparseable"], fixed.calls) == (0, 10)
