"""Study 3's campaign writer (scripts/study3_campaigns.py) and the prep driver it shares with
study 2 (analysis/campaign_prep.py)."""

import importlib.util
import re
import subprocess

from zero_field_refusal.analysis import campaign_prep
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config

COMMIT = "c" * 40
STUDY2 = REPO_ROOT / "scripts" / "study2_short_campaigns.py"


def script_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joined(text: str) -> list[str]:
    """A driver's commands, continued lines joined and indentation dropped."""
    return [line.strip() for line in re.sub(r" *\\\n *", " ", text).splitlines()]


# ------------------------------------------------------------------ the shared prep


def test_study2s_prep_is_the_shared_one() -> None:
    """Study 2's script runs the moved prep: its driver, its arguments and the names its tests
    and its other commands read."""
    module = script_module(STUDY2, "study2_campaigns")
    assert module.prep_driver is campaign_prep.prep_driver
    assert module.prep is campaign_prep.prep
    assert module.comment is campaign_prep.comment and module.revision is campaign_prep.revision
    for name in ("INVENTORIES", "V3_RECORD", "CURVATURE_PANEL", "CLEAN", "EXISTS", "STAGE_A",
                 "TEACHER", "PASSED", "STAGE0_CONFIG", "STAGE0_OVERRIDES"):  # fmt: skip
        assert getattr(module, name) == getattr(campaign_prep, name), name


def test_the_streams_driver_generates_each_seeds_streams_under_its_own_config(tmp_path) -> None:
    configs = {10100: "experiments/study3_k_only study3.role=search",
               10300: "experiments/study3 study3.role=validation"}  # fmt: skip
    text = campaign_prep.streams_driver([10100, 10300], COMMIT, configs=configs, batch=20,
                                        steps=25, writer="scripts/study3_campaigns.py prep",
                                        title="Study 3", written_at=COMMIT)  # fmt: skip
    out = tmp_path / "prep.sh"
    out.write_text(text)
    subprocess.run(["bash", "-n", str(out)], check=True)
    assert all(len(line) <= 100 for line in text.splitlines() if line.startswith("#"))
    header = " ".join(line[2:] for line in text.splitlines() if line.startswith("# "))
    assert f"Written by scripts/study3_campaigns.py prep at {COMMIT}; do not edit." in header
    assert "Study 3, from commit" in header and "seeds 10100 10300 at batch 20" in header
    commands = joined(text)
    run = 'project-run "$SRC" python scripts/'
    assert f'test "$(git rev-parse HEAD)" = {COMMIT}' in commands
    assert f'project-run "$SRC" python -c "{campaign_prep.CLEAN}"' in commands
    for seed, config in configs.items():
        assert (f"step {run}generate_teacher.py {config} --stream training --seeds {seed} "
                "--responses 500") in commands  # fmt: skip
        assert (f"kept {run}generate_teacher.py {config} optimizer=adam_frozen train.batch_size=20 "
                f"--stream preconditioner --seeds {seed}") in commands  # fmt: skip
    assert "stage0" not in text and "check-streams" not in text and "curvature" not in text
    # the commands compose: the configs as the generator loads them (its overrides after it)
    for config in configs.values():
        name, *overrides = config.split()
        assert load_run_config(name, overrides).train.batch_size == 20
        pre = load_run_config(name, [*overrides, "optimizer=adam_frozen", "train.batch_size=20"])
        assert pre.optimizer.preconditioner.responses == 1000
