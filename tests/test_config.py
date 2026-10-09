from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.paths import CONFIG_ROOT


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write(tmp_path, "base.yaml", "seed: 0\nmodel:\n  name: base\n  dtype: float32\n")
    write(tmp_path, "model/qwen2.5-1.5b.yaml", "model:\n  name: qwen\n")
    write(
        tmp_path,
        "optimizer/adamw.yaml",
        "optimizer:\n  name: adamw\n  lr: 1.0e-5\n  betas: [0.9, 0.999]\n",
    )
    write(tmp_path, "optimizer/sgd.yaml", "optimizer:\n  name: sgd\n  lr: ???\n")
    write(
        tmp_path,
        "experiments/exp.yaml",
        "defaults: [base, model/qwen2.5-1.5b, optimizer/adamw]\nexperiment: exp\nlr: 1.0e-5\n",
    )
    return tmp_path


def test_defaults_merge_in_order_then_own_keys(root: Path) -> None:
    cfg = load_config("experiments/exp", root=root)
    assert cfg.model.name == "qwen"  # later default wins
    assert cfg.model.dtype == "float32"  # untouched keys survive
    assert cfg.experiment == "exp"
    assert "defaults" not in cfg


def test_dotted_names_resolve(root: Path) -> None:
    assert load_config("model/qwen2.5-1.5b", root=root).model.name == "qwen"


def test_override_existing_key(root: Path) -> None:
    cfg = load_config("experiments/exp", ["lr=3e-5", "model.dtype=bfloat16"], root=root)
    assert cfg.lr == pytest.approx(3e-5)
    assert cfg.model.dtype == "bfloat16"


def test_override_unknown_key_fails(root: Path) -> None:
    with pytest.raises(KeyError, match="model.dtpye"):
        load_config("experiments/exp", ["model.dtpye=bfloat16"], root=root)


def test_plus_adds_new_key(root: Path) -> None:
    cfg = load_config("experiments/exp", ["+model.revision=abc"], root=root)
    assert cfg.model.revision == "abc"


def test_double_plus_is_rejected(root: Path) -> None:
    with pytest.raises(ValueError, match=r"\+\+model.revision"):
        load_config("experiments/exp", ["++model.revision=abc"], root=root)


def test_missing_value_loads_and_can_be_overridden(root: Path) -> None:
    write(root, "m.yaml", "defaults: [optimizer/sgd]\n")
    assert OmegaConf.is_missing(load_config("m", root=root).optimizer, "lr")
    assert load_config("m", ["optimizer.lr=0.1"], root=root).optimizer.lr == 0.1


def test_group_override_replaces_the_section(root: Path) -> None:
    cfg = load_config("experiments/exp", ["optimizer=sgd", "optimizer.lr=0.1"], root=root)
    assert OmegaConf.to_container(cfg.optimizer) == {"name": "sgd", "lr": 0.1}  # no adamw keys
    assert cfg.model.name == "qwen"


def test_group_override_with_unknown_file_fails(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="optimizer/sdg"):
        load_config("experiments/exp", ["optimizer=sdg"], root=root)


def test_diamond_defaults_merge_shared_file_once(tmp_path: Path) -> None:
    write(tmp_path, "base.yaml", "x: 0\ny: 0\n")
    write(tmp_path, "b.yaml", "defaults: [base]\nx: 1\n")
    write(tmp_path, "c.yaml", "defaults: [base]\ny: 2\n")
    write(tmp_path, "a.yaml", "defaults: [b, c]\n")
    assert OmegaConf.to_container(load_config("a", root=tmp_path)) == {"x": 1, "y": 2}


@pytest.mark.parametrize("defaults", ["base", "", "{base: 1}"])
def test_defaults_must_be_a_list(tmp_path: Path, defaults: str) -> None:
    write(tmp_path, "base.yaml", "x: 0\n")
    write(tmp_path, "s.yaml", f"defaults: {defaults}\nz: 1\n")
    with pytest.raises(TypeError, match="defaults must be a list"):
        load_config("s", root=tmp_path)


def test_cycle_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", "defaults: [b]\n")
    write(tmp_path, "b.yaml", "defaults: [a]\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config("a", root=tmp_path)


def test_mul_resolver(tmp_path: Path) -> None:
    write(tmp_path, "m.yaml", "a: 20\nb: 96\nc: 0.5\nn: ${mul:${a},${b}}\nf: ${mul:${a},${c},2}\n")
    cfg = load_config("m", root=tmp_path)
    assert cfg.n == 1920 and isinstance(cfg.n, int)
    assert cfg.f == 20.0 and isinstance(cfg.f, float)
    with pytest.raises(InterpolationResolutionError, match="mul expects numbers"):
        load_config("m", ["a=x"], root=tmp_path)


def test_hash_ignores_labels_but_not_content(root: Path) -> None:
    cfg = load_config("experiments/exp", root=root)
    relabeled = load_config(
        "experiments/exp", ["experiment=other", "+name=other", "+tracking.x=1"], root=root
    )
    changed = load_config("experiments/exp", ["seed=1"], root=root)
    assert config_hash(cfg) == config_hash(relabeled)
    assert config_hash(cfg) != config_hash(changed)


def test_hash_survives_save_and_reload(root: Path, tmp_path: Path) -> None:
    cfg = load_config("experiments/exp", ["lr=5e-5", "+tag=1e5"], root=root)
    OmegaConf.save(cfg, tmp_path / "saved.yaml", resolve=True)
    assert config_hash(load_config(tmp_path / "saved.yaml", root=root)) == config_hash(cfg)


def test_hash_requires_missing_values(root: Path) -> None:
    write(root, "m.yaml", "defaults: [optimizer/sgd]\nsteps: ???\ntotal: ${mul:${steps},2}\n")
    cfg = load_config("m", root=root)
    with pytest.raises(ValueError, match=r"optimizer\.lr, steps, total"):
        config_hash(cfg)


@pytest.mark.parametrize("path", sorted((CONFIG_ROOT / "experiments").glob("*.yaml")))
def test_repository_experiments_load(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.experiment
