"""Study 5's trained displacements (scripts/study5_vectors.py; docs/study5-preregistration.md,
"Extension 2", M1): the Gram matrix, ‖μ‖² free of the noise, μ̂ and μ̂_{−i}, the balanced-sign
mixtures and the removal vectors, on synthetic finished F runs of the study-5 config whose
endpoints are θ₀ + Δθ of a tiny model, with one trained tiny F run among them; and the refusals."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_train import APPENDABLE, SMALL, read_jsonl
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import vectors
from zero_field_refusal.config import config_hash
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.schedule import run_scheduled, save_endpoint
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.tensor_hash import file_sha256, tensors_sha256

STUDY5 = ["experiments/study5", "study5.role=fresh", "optimizer.lr=1.3e-05"]
FREEZE = ["input_embeddings", "output_head"]
CHUNK = 300  # smaller than most tensors of the tiny model: a tensor spans several slices


# ---------------------------------------------------------------------------- synthetic runs


def theta0_of(env) -> tuple[dict[str, torch.Tensor], str]:
    trainable = freeze(env.new_model(), FREEZE)
    return {n: p.detach().clone() for n, p in trainable}, tensors_sha256(trainable)


def fake_run(
    root: Path,
    checkpoints: Path,
    config: list[str],
    seed: int,
    endpoint: dict,
    theta0: dict,
    theta0_sha: str,
) -> Path:
    """A finished F run of ``config`` at ``seed`` as ``read_f_run`` and ``checkpoint_of`` read
    it: its endpoint saved as the run saves it (``save_endpoint``), its step-N ‖θ_N − θ₀‖ as the
    monitor reads it (the fp32 differences reduced in float64)."""
    cfg = load_run_config(config[0], [*config[1:], f"seed={seed}"])
    steps = int(cfg.train.steps)
    run_id = f"fake-s{seed}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    recorded = save_endpoint(list(endpoint.items()), checkpoints / f"{run_id}.pt", steps)
    norm = math.sqrt(
        math.fsum((endpoint[n] - theta0[n]).double().square().sum().item() for n in endpoint)
    )
    meta = {
        "run_id": run_id,
        "config_hash": config_hash(cfg),
        "inputs": {
            "theta0_trainable_sha256": theta0_sha,
            "streams": {},
            "endpoint_checkpoint": recorded,
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    (run_dir / "terminal.json").write_text(
        json.dumps({"steps": steps, "stop": {"reason": "steps"}})
    )
    (run_dir / "monitor.jsonl").write_text(json.dumps({"step": steps, "displacement_norm": norm}))
    (run_dir / "evaluations.jsonl").write_text(json.dumps({"step": steps, "condition": "eval"}))
    return run_dir


def runs_of(tmp_path: Path, env, deltas: dict[int, dict], *, name="runs"):
    """Fake runs of the study-5 config with endpoints θ₀ + δ (fp32), one per seed; returns
    (run dirs, checkpoint directory)."""
    theta0, sha = theta0_of(env)
    root, checkpoints = tmp_path / name, tmp_path / f"{name}-ckpt"
    dirs = [
        fake_run(
            root,
            checkpoints,
            STUDY5,
            seed,
            {n: theta0[n] + d for n, d in delta.items()},
            theta0,
            sha,
        )
        for seed, delta in deltas.items()
    ]
    return dirs, checkpoints


def draws(theta0: dict, seeds, scale: float, common: dict | None = None):
    """δ_s = common + scale · z_s, z_s standard normal per seed."""
    out = {}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        out[seed] = {
            n: scale * torch.randn(t.shape, generator=g) + (0 if common is None else common[n])
            for n, t in theta0.items()
        }
    return out


def flat(vector: dict, names) -> torch.Tensor:
    return torch.cat([vector[n].reshape(-1).double() for n in names])


def actual(checkpoints: Path, run_dirs, theta0: dict) -> torch.Tensor:
    """The n × count float64 matrix of the fp32 differences θ₂₅ − θ₀, read directly."""
    rows = []
    for run_dir in run_dirs:
        state = torch.load(checkpoints / f"{run_dir.name}.pt", weights_only=True)
        rows.append(torch.cat([(state[n] - theta0[n]).reshape(-1).double() for n in theta0]))
    return torch.stack(rows)


@pytest.fixture
def script(env, monkeypatch):  # noqa: F811
    torch.set_num_threads(1)
    module = script_module(REPO_ROOT / "scripts" / "study5_vectors.py", "s5_vectors")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    monkeypatch.setattr(module, "CHUNK", CHUNK)
    return module


def call(module, run_dirs, checkpoints, out, *extra, config=STUDY5) -> int:
    return module.main(
        [
            *config,
            "seed=50000",
            "--runs",
            *map(str, run_dirs),
            "--checkpoints",
            str(checkpoints),
            "--out-dir",
            str(out),
            *extra,
        ]
    )


def load(out: Path, name: str, names) -> torch.Tensor:
    tensors, _ = vectors.load_vector(out / name, names)
    return flat(tensors, names)


# ------------------------------------------------------------------------------- the quantities


def test_gram_means_and_grad_r_equal_direct_computations(
    env, script, tmp_path, monkeypatch, capsys
) -> None:
    """G, the norms, μ̂, every μ̂_{−i}, p_i and cos_i, the split halves and ⟨Δθ_i, ∇r⟩ equal
    their direct computations over the fp32 differences; each file's record is its own; the
    checkpoints are read memory-mapped, at most CHUNK elements of a tensor at a time."""
    theta0, sha = theta0_of(env)
    names = list(theta0)
    seeds = list(range(50000, 50006))
    common = {n: 0.02 * torch.ones_like(t) for n, t in theta0.items()}
    run_dirs, ckpt = runs_of(tmp_path, env, draws(theta0, seeds, 0.03, common))
    grad = {
        n: torch.randn(t.shape, generator=torch.Generator().manual_seed(7))
        for n, t in theta0.items()
    }
    grad_path = tmp_path / "grad" / "grad_r.pt"
    vectors.save_vector(grad_path, grad, names, kind="grad_r", theta0_sha256=sha)
    loads, widths = [], []
    real_load, real_differences = torch.load, script.differences

    def spy_load(path, *args, **kwargs):
        loads.append((Path(path).parent, kwargs.get("mmap")))
        return real_load(path, *args, **kwargs)

    def spy_differences(states, theta0, name, start, stop):
        widths.append(stop - start)
        return real_differences(states, theta0, name, start, stop)

    monkeypatch.setattr(torch, "load", spy_load)
    monkeypatch.setattr(script, "differences", spy_differences)
    out = tmp_path / "out"
    assert call(script, reversed(run_dirs), ckpt, out, "--grad-r", str(grad_path)) == 0
    record = json.loads((out / "record.json").read_text())
    assert all(mmap for parent, mmap in loads if parent == ckpt) and any(
        p == ckpt for p, _ in loads
    )
    assert max(widths) == CHUNK and len(widths) == 2 * sum(
        math.ceil(t.numel() / CHUNK) for t in theta0.values()
    )

    X = actual(ckpt, run_dirs, theta0)  # seed order
    n, count = X.shape
    assert record["seeds"] == seeds and record["n"] == n == 6 and record["count"] == count
    assert record["theta0_trainable_sha256"] == sha
    G = X @ X.T
    np.testing.assert_allclose(record["gram"], G.numpy(), rtol=1e-12)
    norms = G.diagonal().sqrt()
    np.testing.assert_allclose(record["norms"], norms.numpy(), rtol=1e-12)
    for row, run_dir, norm in zip(record["runs"], run_dirs, norms, strict=True):
        assert (
            row["run_id"] == run_dir.name
            and row["config_hash"] == json.loads((run_dir / "meta.json").read_text())["config_hash"]
        )
        assert row["checkpoint"]["sha256"] == file_sha256(ckpt / f"{run_dir.name}.pt")
        assert row["displacement_norm"] == pytest.approx(
            row["displacement_norm_recorded"], rel=1e-12
        )
        assert row["displacement_norm"] == pytest.approx(norm.item(), rel=1e-12)

    S = X.sum(0)
    mean = record["mean"]
    off = (G.sum() - G.trace()) / (n * (n - 1))
    assert mean["norm_sq_noise_free"] == pytest.approx(off.item(), rel=1e-12)
    assert mean["norm_sq"] == pytest.approx((S / n).square().sum().item(), rel=1e-12)
    cos = G / torch.outer(norms, norms)
    assert mean["mean_pairwise_cosine"] == pytest.approx(
        ((cos.sum() - cos.trace()) / (n * (n - 1))).item(), rel=1e-12
    )
    torch.testing.assert_close(
        load(out, "mean.pt", names), (S / n).float().double(), rtol=1e-6, atol=1e-9
    )

    loo = record["leave_one_out"]
    assert record["loo_seeds"] == seeds  # every seed by default
    p, c = [], []
    for i, seed in enumerate(seeds):
        m = (S - X[i]) / (n - 1)
        torch.testing.assert_close(
            load(out, f"mean_minus_{seed}.pt", names), m.float().double(), rtol=1e-6, atol=1e-9
        )
        p.append((X[i] @ m / m.norm()).item())
        c.append(p[-1] / norms[i].item())
    np.testing.assert_allclose(loo["p"], p, rtol=1e-10)
    np.testing.assert_allclose(loo["cos"], c, rtol=1e-10)
    assert loo["p_spread"]["mean"] == pytest.approx(np.mean(p), rel=1e-10)
    assert loo["p_spread"]["sd"] == pytest.approx(np.std(p, ddof=1), rel=1e-8)
    assert loo["p_spread"]["cv"] == pytest.approx(np.std(p, ddof=1) / abs(np.mean(p)), rel=1e-8)
    assert loo["cos_spread"]["mean"] == pytest.approx(np.mean(c), rel=1e-10)

    halves = record["halves"]
    a, b = X[0::2].mean(0), X[1::2].mean(0)
    assert halves["even"]["seeds"] == [50000, 50002, 50004]
    assert halves["odd"]["seeds"] == [50001, 50003, 50005]
    assert halves["even"]["norm"] == pytest.approx(a.norm().item(), rel=1e-12)
    assert halves["odd"]["norm"] == pytest.approx(b.norm().item(), rel=1e-12)
    assert halves["cosine"] == pytest.approx((a @ b / (a.norm() * b.norm())).item(), rel=1e-10)

    g = flat(grad, names)
    dots = (X @ g).numpy()
    gr = record["grad_r"]
    np.testing.assert_allclose(gr["per_run"], dots, rtol=1e-10)
    assert gr["mean"] == pytest.approx(dots.mean(), rel=1e-10)
    assert gr["se"] == pytest.approx(dots.std(ddof=1) / math.sqrt(n), rel=1e-8)
    assert gr["mean_vector"] == pytest.approx((load(out, "mean.pt", names) @ g).item(), rel=1e-10)
    assert gr["mean_vector"] == pytest.approx(gr["mean"], rel=1e-5)
    assert gr["sha256"] == vectors.read_record(grad_path)["sha256"]

    # every file as planned, with its own record; nothing else left in the directory
    expected = (
        ["mean.pt"]
        + [f"mean_minus_{s}.pt" for s in seeds]
        + [f"mixture_{k}.pt" for k in (1, 2, 3)]
        + [f"removal_{part}_{s}.pt" for s in seeds for part in ("mean", "mixture")]
    )
    assert [v["file"] for v in record["vectors"]] == expected
    assert record["removal_seeds"] == seeds  # 50000–50005, all present
    for item in record["vectors"]:
        file_record = vectors.read_record(out / item["file"])
        assert item["sha256"] == file_record["sha256"] == file_sha256(out / item["file"])
        assert file_record["theta0_sha256"] == sha and file_record["kind"] == item["kind"]
        assert item["norm"] == pytest.approx(item["norm_gram"], rel=1e-6)
        assert gr["vectors"][item["file"]] == pytest.approx(
            (load(out, item["file"], names) @ g).item(), rel=1e-9, abs=1e-12
        )
    assert vectors.read_record(out / "mean.pt")["provenance"]["seeds"] == seeds
    assert vectors.read_record(out / "mean_minus_50003.pt")["provenance"]["left_out"] == 50003
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["record.json", *expected, *(f"{f}.json" for f in expected)]
    )

    printed = capsys.readouterr().out
    assert sha in printed and record["vectors"][0]["sha256"] in printed
    for name in ("alpaca_teacher", "harmful_judged", "harmful_evaluation"):
        for item in env.prompt_set(name):
            assert item["text"] not in printed
    assert call(script, run_dirs, ckpt, out) == 1  # never overwritten
    assert json.loads((out / "record.json").read_text()) == record


def test_the_noise_free_norm_and_the_bias_of_the_mean(env, script, tmp_path) -> None:
    """Δθ_i = d + noise_i over 40 seeds: the mean of G_ij over i ≠ j is ‖d‖² up to the noise's
    cross terms, while ‖μ̂‖² exceeds it by mean ‖Δθ_i‖² less that, over n: about
    mean ‖noise_i‖² / n."""
    theta0, _ = theta0_of(env)
    count = sum(t.numel() for t in theta0.values())
    d = {n: torch.full(t.shape, 1 / math.sqrt(count)) for n, t in theta0.items()}  # ‖d‖ = 1
    sigma = math.sqrt(20 / count)  # E‖noise‖² = 20
    seeds = list(range(50000, 50040))
    deltas = draws(theta0, seeds, sigma, d)
    run_dirs, ckpt = runs_of(tmp_path, env, deltas)
    out = tmp_path / "out"
    assert call(script, run_dirs, ckpt, out, "--loo-seeds", "50000") == 0
    record = json.loads((out / "record.json").read_text())
    mean, n = record["mean"], 40
    noise = (
        math.fsum(
            flat({k: v - d[k] for k, v in deltas[s].items()}, theta0).square().sum().item()
            for s in seeds
        )
        / n
    )
    assert noise == pytest.approx(20, rel=0.05)
    assert mean["norm_sq_noise_free"] == pytest.approx(1.0, abs=0.05)
    assert mean["excess"] == pytest.approx(
        (mean["mean_norm_sq"] - mean["norm_sq_noise_free"]) / n, rel=1e-9
    )
    assert mean["norm_sq"] == pytest.approx(mean["norm_sq_noise_free"] + mean["excess"], rel=1e-12)
    assert mean["excess"] == pytest.approx(noise / n, rel=0.05)
    assert mean["norm_sq"] > 1.4  # ‖μ̂‖² ≈ 1.5: biased up by ≈ 20/40
    assert [v["file"] for v in record["vectors"]][:2] == ["mean.pt", "mean_minus_50000.pt"]


def test_mixtures_are_balanced_reproducible_and_cancel_a_common_component(
    env, script, tmp_path
) -> None:
    """Exactly n/2 plus signs at the first n/2 entries of the named permutation; the same
    vectors again from the same runs; a common d added to every Δθ_j leaves the unscaled mixture
    unchanged (to fp32 rounding of the endpoints); each mixture has the norm of μ̂."""
    theta0, _ = theta0_of(env)
    names = list(theta0)
    seeds = list(range(50000, 50008))
    n = len(seeds)
    d = {
        k: 0.05 * torch.randn(t.shape, generator=torch.Generator().manual_seed(99))
        for k, t in theta0.items()
    }
    noise_dirs, noise_ckpt = runs_of(tmp_path, env, draws(theta0, seeds, 0.03), name="noise")
    shift_dirs, shift_ckpt = runs_of(tmp_path, env, draws(theta0, seeds, 0.03, d), name="shift")
    for out, dirs, ckpt in (
        ("noise-out", noise_dirs, noise_ckpt),
        ("shift-out", shift_dirs, shift_ckpt),
        ("again-out", noise_dirs, noise_ckpt),
    ):
        assert call(script, dirs, ckpt, tmp_path / out, "--loo-seeds", "--removal-seeds") == 0
    records = {
        out: json.loads((tmp_path / out / "record.json").read_text())
        for out in ("noise-out", "shift-out", "again-out")
    }
    for k in (1, 2, 3):
        rng_seed = derive_seed(k, "study5", "mixture")
        plus = set(np.random.default_rng(rng_seed).permutation(n)[: n // 2].tolist())
        expected = [1 if j in plus else -1 for j in range(n)]
        assert script.balanced_signs(n, rng_seed) == expected and sum(expected) == 0
        unscaled = {}
        for out, record in records.items():
            row = record["mixtures"][k - 1]
            assert row["k"] == k and row["rng_seed"] == rng_seed and row["signs"] == expected
            m = load(tmp_path / out, f"mixture_{k}.pt", names)
            mean_norm = vectors.read_record(tmp_path / out / "mean.pt")["norm"]
            assert m.norm().item() == pytest.approx(mean_norm, rel=1e-6)
            assert row["target_norm"] == pytest.approx(math.sqrt(record["mean"]["norm_sq"]))
            assert row["scale"] == pytest.approx(row["target_norm"] / row["unscaled_norm"])
            unscaled[out] = m / row["scale"]
            provenance = vectors.read_record(tmp_path / out / f"mixture_{k}.pt")["provenance"]
            assert provenance["signs"] == expected and provenance["seeds"] == seeds
        assert torch.equal(unscaled["noise-out"], unscaled["again-out"])
        torch.testing.assert_close(
            unscaled["shift-out"],
            unscaled["noise-out"],
            rtol=0,
            atol=1e-5 * unscaled["noise-out"].abs().max().item(),
        )
    with pytest.raises(ValueError, match="even"):
        script.balanced_signs(7, 1)


def test_removals_subtract_the_mean_or_a_mixture_of_equal_norm(env, script, tmp_path) -> None:
    """Δθ_i − μ̂_{−i}; Δθ_i − m_{−i} with m_{−i} the balanced-sign mixture of the other seeds but
    the largest, scaled to ‖μ̂_{−i}‖; the recorded norms and cosines are those of the files."""
    theta0, _ = theta0_of(env)
    names = list(theta0)
    seeds = list(range(50000, 50008))
    n = len(seeds)
    common = {k: 0.02 * torch.ones_like(t) for k, t in theta0.items()}
    run_dirs, ckpt = runs_of(tmp_path, env, draws(theta0, seeds, 0.03, common))
    out = tmp_path / "out"
    options = ["--loo-seeds", "--mixtures", "0", "--removal-seeds", "50007", "50002"]
    assert call(script, run_dirs, ckpt, out, *options) == 0
    record = json.loads((out / "record.json").read_text())
    X = actual(ckpt, run_dirs, theta0)
    S = X.sum(0)
    assert record["removal_seeds"] == [50002, 50007] and record["mixtures"] == []
    for row, seed in zip(record["removals"], (50002, 50007), strict=True):
        i = seeds.index(seed)
        loo = (S - X[i]) / (n - 1)
        by_mean = load(out, f"removal_mean_{seed}.pt", names)
        torch.testing.assert_close(by_mean, (X[i] - loo).float().double(), rtol=1e-6, atol=1e-9)
        others = [s for s in seeds if s != seed]
        left_out = max(others)
        assert left_out == (50007 if seed == 50002 else 50006)
        used = [s for s in others if s != left_out]
        plus = set(
            np.random.default_rng(derive_seed(seed, "study5", "removal"))
            .permutation(n - 2)[: (n - 2) // 2]
            .tolist()
        )
        signs = [1 if j in plus else -1 for j in range(n - 2)]
        mixture = sum(sign * X[seeds.index(s)] for s, sign in zip(used, signs, strict=True))
        mixture = mixture / (n - 2) * (loo.norm() / (mixture / (n - 2)).norm())
        by_mixture = load(out, f"removal_mixture_{seed}.pt", names)
        torch.testing.assert_close(
            by_mixture, (X[i] - mixture).float().double(), rtol=1e-6, atol=1e-9
        )
        removed = X[i] - by_mixture
        assert removed.norm().item() == pytest.approx(loo.norm().item(), rel=1e-5)
        assert row["seed"] == seed and row["delta_norm"] == pytest.approx(X[i].norm().item())
        m, mx = row["mean"], row["mixture"]
        assert mx["left_out"] == left_out and mx["seeds"] == used and mx["signs"] == signs
        assert m["removed_norm"] == pytest.approx(loo.norm().item(), rel=1e-12)
        assert mx["removed_norm"] == pytest.approx(loo.norm().item(), rel=1e-12)
        for part, vec in (("mean", by_mean), ("mixture", by_mixture)):
            assert row[part]["result_norm"] == pytest.approx(vec.norm().item(), rel=1e-6)
            assert row[part]["cos_with_delta"] == pytest.approx(
                (vec @ X[i] / (vec.norm() * X[i].norm())).item(), rel=1e-6
            )
        provenance = vectors.read_record(out / f"removal_mixture_{seed}.pt")["provenance"]
        assert provenance["left_out"] == left_out and provenance["signs"] == signs


# --------------------------------------------------------------------------------- the refusals


def test_a_missing_or_altered_checkpoint_is_refused_and_nothing_written(
    env, script, tmp_path
) -> None:
    """A missing or altered checkpoint, θ₀ or a config that is not the runs', a seed given twice,
    an odd n with a mixture, ∇r of another kind or θ₀, an existing vector file: exit 1 and
    nothing written; with --skip-bad the bad checkpoints' runs are left out and recorded."""
    theta0, sha = theta0_of(env)
    names = list(theta0)
    seeds = list(range(50000, 50006))
    run_dirs, ckpt = runs_of(tmp_path, env, draws(theta0, seeds, 0.03))
    out = tmp_path / "out"
    missing = ckpt / f"{run_dirs[1].name}.pt"
    kept = missing.with_name("kept.pt")
    missing.rename(kept)
    assert call(script, run_dirs, ckpt, out) == 1
    assert not out.exists()
    kept.rename(missing)
    altered = ckpt / f"{run_dirs[4].name}.pt"
    state = torch.load(altered, weights_only=True)
    original = altered.read_bytes()
    state[names[0]][0] += 1e-3
    torch.save(state, altered)
    assert call(script, run_dirs, ckpt, out) == 1
    assert not out.exists()

    # --skip-bad: left out and recorded; one left out makes n odd: refused with a mixture
    assert call(script, run_dirs, ckpt, out, "--skip-bad") == 1
    assert not out.exists()
    missing.rename(kept)
    assert call(script, run_dirs, ckpt, out, "--skip-bad") == 0
    record = json.loads((out / "record.json").read_text())
    assert record["seeds"] == [50000, 50002, 50003, 50005] and record["n"] == 4
    assert [(s["seed"], s["run_id"]) for s in record["skipped"]] == [
        (50001, run_dirs[1].name),
        (50004, run_dirs[4].name),
    ]
    assert "no checkpoint" in record["skipped"][0]["reason"]
    assert "sha256" in record["skipped"][1]["reason"]
    kept.rename(missing)
    altered.write_bytes(original)

    def refused(*extra, dirs=run_dirs, config=STUDY5) -> None:
        target = tmp_path / "refused"
        assert call(script, dirs, ckpt, target, *extra, config=config) == 1
        assert not target.exists()

    refused(config=["experiments/study5", "study5.role=fresh", "optimizer.lr=1.83e-05"])
    refused(dirs=[*run_dirs, run_dirs[0]])  # a seed twice
    refused(dirs=run_dirs[:3])  # too few
    refused(dirs=run_dirs[:5])  # odd n with the mixtures
    refused("--removal-seeds", "50001", "49999")
    refused("--mixtures", "-1")
    bad_grad = tmp_path / "grad" / "mean.pt"
    vectors.save_vector(bad_grad, theta0, names, kind="mean", theta0_sha256=sha)
    refused("--grad-r", str(bad_grad))
    other_theta0 = tmp_path / "grad" / "other.pt"
    vectors.save_vector(other_theta0, theta0, names, kind="grad_r", theta0_sha256="0" * 64)
    refused("--grad-r", str(other_theta0))
    meta_path = run_dirs[2] / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["inputs"]["theta0_trainable_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    refused()
    meta["inputs"]["theta0_trainable_sha256"] = sha
    meta_path.write_text(json.dumps(meta))

    # an existing vector file: refused before any write
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "mixture_2.pt").write_bytes(b"")
    assert call(script, run_dirs, ckpt, busy) == 1
    assert [p.name for p in busy.iterdir()] == ["mixture_2.pt"]


def test_a_failure_while_writing_removes_what_was_written(
    env, script, tmp_path, monkeypatch
) -> None:
    """A vector file that fails to write (the third): the call exits 1 and the files it wrote,
    their records and the scratch files are removed."""
    theta0, _ = theta0_of(env)
    run_dirs, ckpt = runs_of(tmp_path, env, draws(theta0, range(50000, 50004), 0.03))
    real, calls = vectors.save_vector, []

    def failing(path, *args, **kwargs):
        calls.append(Path(path).name)
        if len(calls) == 3:
            raise OSError("no space left on device")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(vectors, "save_vector", failing)
    out = tmp_path / "out"
    assert call(script, run_dirs, ckpt, out) == 1
    assert calls[:2] == ["mean.pt", "mean_minus_50000.pt"] and len(calls) == 3
    assert list(out.iterdir()) == []  # the files written and the scratch files removed


# ------------------------------------------------------------------------------ a trained run


def test_a_trained_run_among_synthetic_ones(env, script, tmp_path) -> None:  # noqa: F811
    """A tiny fresh F run, trained with its endpoint checkpoint, beside three synthetic runs of
    its config at seeds 1–3: its ‖Δθ‖ from the checkpoint is the run's own monitor reading."""
    ckpt = tmp_path / "ckpt"
    overrides = [*F_RUN, "+study5.role=fresh"]
    cfg = ready(
        env, scheduled(tmp_path, *overrides, f"tracking.endpoint_checkpoint={ckpt}", policy="short")
    )
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    config = [
        str(tmp_path / "study2.yaml"),
        *SMALL,
        *APPENDABLE,
        "teacher.chunk_size=12",
        *overrides,
    ]
    theta0, sha = theta0_of(env)
    fakes = [
        fake_run(
            tmp_path / "fakes",
            ckpt,
            config,
            seed,
            {n: t + 0.01 * seed for n, t in theta0.items()},
            theta0,
            sha,
        )
        for seed in (1, 2, 3)
    ]
    out = tmp_path / "out"
    argv = [
        *config,
        "--runs",
        str(f_dir),
        *map(str, fakes),
        "--checkpoints",
        str(ckpt),
        "--out-dir",
        str(out),
    ]
    assert script.main(argv) == 0
    record = json.loads((out / "record.json").read_text())
    assert record["seeds"] == [0, 1, 2, 3] and record["removal_seeds"] == []
    trained = record["runs"][0]
    reading = [r for r in read_jsonl(f_dir / "monitor.jsonl") if r["step"] == cfg.train.steps]
    assert trained["run_id"] == "f-fresh"
    assert trained["displacement_norm"] == pytest.approx(reading[0]["displacement_norm"], rel=1e-12)
    meta = json.loads((f_dir / "meta.json").read_text())
    assert trained["checkpoint"]["sha256"] == meta["inputs"]["endpoint_checkpoint"]["sha256"]
    assert record["theta0_trainable_sha256"] == meta["inputs"]["theta0_trainable_sha256"]
