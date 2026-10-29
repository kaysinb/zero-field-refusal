"""Documentation contract tests use disposable files, never models, networks or git commits."""

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_docs_check", ROOT / "scripts/check_docs.py")
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


def write_map(root, data):
    OmegaConf.save(OmegaConf.create(data), root / "docs/map.yaml")


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "# Overview\n\n[Rule][rule]\n\n[rule]: docs/spec.md#rule\n"
        "\n`[not a link](absent.md)`\n\n```markdown\n[example](missing.md)\n```\n"
    )
    (tmp_path / "docs/spec.md").write_text("# Specification\n\n## Rule\n\nNormative text.\n")
    data = dict(
        version=1,
        package="example_package",
        documents=[
            dict(path=p, purpose=p) for p in ["README.md", "docs/README.md", "docs/spec.md"]
        ],
        historical_docs=["docs/reports/*.md"],
        config_examples=[],
        topics=[
            dict(
                id="rule",
                title="Rule",
                source="docs/spec.md#rule",
                review=["README.md"],
                watch=["src/future/*.py"],
            )
        ],
    )
    write_map(tmp_path, data)
    (tmp_path / "docs/README.md").write_text(
        "# Documentation\n\n" + docs.index_content(data) + "\n\nManual workflow stays here.\n"
    )
    return tmp_path, data


def test_actual_repository_and_config_examples():
    checker = docs.Check(ROOT)
    data = docs.load_map(ROOT)
    checker.validate(data)
    assert checker.errors == []
    assert checker.configs == len(data["config_examples"])


def test_default_check_is_read_only_and_reports_dependents(repo, capsys):
    root, _ = repo
    before = hashes(root)
    assert docs.main(["--root", str(root), "--changed", "docs/spec.md"]) == 0
    output = capsys.readouterr().out
    assert "REVIEW [rule]" in output and "review: README.md" in output
    assert "advisory" in output
    assert hashes(root) == before


@pytest.mark.parametrize("path", ["docs/spec.md", "./docs/spec.md", "docs//spec.md"])
def test_changed_path_aliases_report_dependents(repo, capsys, path):
    root, _ = repo
    assert docs.main(["--root", str(root), "--changed", path]) == 0
    assert "REVIEW [rule]" in capsys.readouterr().out


def test_map_path_aliases_report_dependents(repo):
    _, data = repo
    data["topics"][0]["source"] = "./docs//spec.md#rule"
    data["topics"][0]["watch"] = ["./src/future/*.py"]
    assert docs.impacted_topics(data, ["docs/spec.md"])
    assert docs.impacted_topics(data, ["src/future/experiment.py"])


def test_config_import_does_not_write_bytecode(repo):
    root, data = repo
    package = root / "src/doc_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text(
        "def load_config(*args, **kwargs):\n    return {'seed': 0}\n\n"
        "def config_hash(cfg):\n    return 'hash'\n"
    )
    (root / "configs").mkdir()
    (root / "configs/smoke.yaml").write_text("seed: 0\n")
    data["package"] = "doc_fixture"
    data["config_examples"] = [{"config": "smoke"}]
    write_map(root, data)
    before = hashes(root)
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_docs.py"), "--root", str(root), "--changed"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert hashes(root) == before
    assert not list(root.rglob("__pycache__"))


def test_index_symlink_cannot_write_outside_repository(repo):
    root, data = repo
    index = root / "docs/README.md"
    outside = root.parent / (root.name + "-outside.md")
    outside.write_text(index.read_text())
    before = outside.read_bytes()
    index.unlink()
    index.symlink_to(outside)
    data["documents"] = [d for d in data["documents"] if d["path"] != "docs/README.md"]
    write_map(root, data)
    assert docs.main(["--root", str(root), "--write-index", "--changed"]) == 1
    assert outside.read_bytes() == before


@pytest.mark.parametrize(
    "target,diagnostic",
    [
        ("docs/missing.md", "missing target"),
        ("docs/spec.md#absent", "missing anchor"),
        ("../outside.md", "escapes repository"),
    ],
)
def test_broken_links_fail(repo, target, diagnostic):
    root, data = repo
    with (root / "README.md").open("a") as f:
        f.write(f"\n[bad]({target})\n")
    check = docs.Check(root)
    check.validate(data)
    assert any(diagnostic in e for e in check.errors)


def test_heading_and_html_anchors_and_fences():
    anchors, links = docs.markdown_info(
        '# A & B\n\n## A & B\n\n<a id="custom"></a>\n'
        '<a href="local.md#target">target</a>\n\n'
        "```markdown\n# Fake\n[not real](absent.md)\n```\n"
    )
    assert {"a--b", "a--b-1", "custom"} <= anchors
    assert "fake" not in anchors
    assert [r for r, _ in links] == ["local.md#target"]


def test_stale_index_is_detected_and_only_generated_region_is_written(repo):
    root, data = repo
    data["topics"][0]["title"] = "Updated topic"
    write_map(root, data)
    before = hashes(root)
    assert docs.main(["--root", str(root), "--changed"]) == 1
    assert hashes(root) == before
    assert docs.main(["--root", str(root), "--write-index", "--changed"]) == 0
    after = hashes(root)
    assert {p for p in before if before[p] != after[p]} == {"docs/README.md"}
    assert (root / "docs/README.md").read_text().endswith("Manual workflow stays here.\n")


def test_bad_source_does_not_get_generated_into_index(repo):
    root, data = repo
    data["topics"][0]["source"] = "docs/spec.md#typo"
    write_map(root, data)
    before = (root / "docs/README.md").read_bytes()
    assert docs.main(["--root", str(root), "--write-index", "--changed"]) == 1
    assert (root / "docs/README.md").read_bytes() == before


@pytest.mark.parametrize("problem", ["duplicate_id", "duplicate_doc", "unknown_field", "escape"])
def test_invalid_map_rejected(repo, problem):
    root, data = repo
    if problem == "duplicate_id":
        data["topics"].append(data["topics"][0].copy())
    elif problem == "duplicate_doc":
        data["documents"].append(data["documents"][0].copy())
    elif problem == "unknown_field":
        data["topics"][0]["reviews"] = data["topics"][0].pop("review")
    else:
        data["topics"][0]["source"] = "../outside.md"
    write_map(root, data)
    with pytest.raises(ValueError):
        docs.load_map(root)


def test_new_document_must_be_registered(repo):
    root, data = repo
    (root / "docs/new-rule.md").write_text("# New rule\n")
    checker = docs.Check(root)
    checker.validate(data)
    assert any("Unregistered document: docs/new-rule.md" in e for e in checker.errors)


def test_ignored_local_document_is_not_scanned(repo):
    root, data = repo
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("*.local.md\n")
    (root / "docs/private.local.md").write_text("[Local-only link](missing.md)\n")
    checker = docs.Check(root)
    assert checker.validate(data) == 3
    assert checker.errors == []


@pytest.mark.parametrize("exists", [True, False])
def test_public_link_cannot_depend_on_ignored_target(repo, exists):
    root, data = repo
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("*.local.md\n")
    if exists:
        (root / "private.local.md").write_text("# Local instructions\n")
    with (root / "README.md").open("a") as handle:
        handle.write("\n[Instructions](private.local.md)\n")
    checker = docs.Check(root)
    checker.validate(data)
    assert any("Git-ignored target" in error for error in checker.errors)


def test_ignore_exception_allows_public_example(repo):
    root, data = repo
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(".env*\n!.env.example\n")
    (root / ".env.example").write_text("EXAMPLE=unused\n")
    with (root / "README.md").open("a") as handle:
        handle.write("\n[Example](.env.example)\n")
    checker = docs.Check(root)
    checker.validate(data)
    assert checker.errors == []


def test_map_cannot_publish_ignored_document(repo):
    root, data = repo
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("*.local.md\n")
    (root / "docs/private.local.md").write_text("# Local instructions\n")
    data["documents"].append(dict(path="docs/private.local.md", purpose="Local instructions"))
    before = (root / "docs/README.md").read_bytes()
    checker = docs.Check(root)
    checker.validate(data, write_index=True)
    assert any("Git-ignored target" in error for error in checker.errors)
    assert (root / "docs/README.md").read_bytes() == before


def test_historical_report_never_rewritten(repo):
    root, data = repo
    (root / "docs/reports").mkdir()
    report = root / "docs/reports/old.md"
    report.write_text("# Old result\n\nA result under an older protocol.\n")
    data["documents"][0]["purpose"] = "New overview label"
    write_map(root, data)
    original = report.read_bytes()
    assert docs.main(["--root", str(root), "--write-index", "--changed"]) == 0
    assert report.read_bytes() == original


def test_review_and_future_code_changes_both_trigger_impact(repo):
    _, data = repo
    for path in ["README.md", "src/future/losses.py", "docs/map.yaml"]:
        assert docs.impacted_topics(data, [path])[0][0]["id"] == "rule"
    assert docs.impacted_topics(data, ["unrelated.txt"]) == []


def test_no_commit_git_checkout_includes_untracked_files(repo):
    root, _ = repo
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    changed, note = docs.changes_from_git(root)
    assert "README.md" in changed and "docs/spec.md" in changed
    assert "No initial commit" in note
    with pytest.raises(ValueError, match="Unknown Git revision"):
        docs.changes_from_git(root, "missing-revision")


def test_invalid_explicit_base_is_an_error(repo):
    root, _ = repo
    assert docs.main(["--root", str(root), "--base", "unknown"]) == 1


def test_config_templates_allow_missing_but_never_unknown_overrides(repo):
    root, data = repo
    (root / "configs").mkdir()
    (root / "configs/smoke.yaml").write_text("required_value: ???\nseed: 0\n")
    data["package"] = docs.load_map(ROOT)["package"]
    data["config_examples"] = [dict(config="smoke", allow_missing=True)]
    check = docs.Check(root)
    check.validate(data)
    assert not any("Config example" in e for e in check.errors)
    data["config_examples"][0]["overrides"] = ["seed_typo=3"]
    check = docs.Check(root)
    check.validate(data)
    assert any("Unknown config key" in e for e in check.errors)


def test_required_values_in_runnable_example_fail(repo):
    root, data = repo
    (root / "configs").mkdir()
    (root / "configs/smoke.yaml").write_text("required_value: ???\n")
    data["package"] = docs.load_map(ROOT)["package"]
    data["config_examples"] = [dict(config="smoke")]
    check = docs.Check(root)
    check.validate(data)
    assert any("Set these config values" in e for e in check.errors)


def test_the_launchers_conflict_note_names_an_operations_section_that_exists():
    """The title the launcher's note quotes leads a bullet (or is a heading) of
    docs/operations.md, and the note says where it is."""
    from zero_field_refusal.launch import LOST_CHECKPOINT_NOTE

    (title,) = re.findall(r'\(docs/operations\.md, "([^"]+)"\)', LOST_CHECKPOINT_NOTE)
    lines = (ROOT / "docs/operations.md").read_text().splitlines()
    assert any(
        line.startswith(f"- **{title}.**") or re.fullmatch(rf"#+ {re.escape(title)}", line)
        for line in lines
    )
