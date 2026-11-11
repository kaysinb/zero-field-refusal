"""Study 5 extension 2's small-amplitude surrogate and token readouts along a vector (M3) on CPU:
the readouts on a constructed case and against a direct full-forward computation, then the script
on a tiny fresh F run, its vector given as a file or as the run's checkpoint."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch
from conftest import tiny_model
from test_ladder import _NoRun
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_surrogate import SPEC_YAML
from test_train import APPENDABLE, SMALL
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import surrogate, vectors
from zero_field_refusal.evaluation import set_point
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.teacher import Sample
from zero_field_refusal.tensor_hash import tensors_sha256
from zero_field_refusal.train import Evaluator

SCRIPT = REPO_ROOT / "scripts" / "study5_tokens.py"
SCALES = [1.0, -1.0, 0.0, 0.5, -0.5]  # θ₀ is not the last point
# Just below 1/64: the tiny model's near-uniform head puts about half the vocabulary under it
RARE = 0.0156


def module():
    return script_module(SCRIPT, "s5_tokens")


def direct(student, reference, samples, rare: float) -> dict[str, float]:
    """The token means by a full forward pass of each sample alone (``model.forward``, every
    position's logits), position by position in float64."""
    sums = dict.fromkeys(("entropy", "rare_mass", "delta_logprob", "theta0_entropy",
                          "theta0_rare_mass"), 0.0)  # fmt: skip
    count = 0
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor([[*s.prompt_ids, *s.response_ids]])
            lp0 = torch.log_softmax(reference(input_ids=ids).logits[0].double(), dim=-1)
            lp = torch.log_softmax(student(input_ids=ids).logits[0].double(), dim=-1)
            for t, y in enumerate(s.response_ids):
                pos = len(s.prompt_ids) - 1 + t  # the position that predicts response token t
                p0, p = lp0[pos].exp(), lp[pos].exp()
                rare_mask = p0 < rare
                sums["entropy"] += -(p * lp[pos]).sum().item()
                sums["rare_mass"] += p[rare_mask].sum().item()
                sums["delta_logprob"] += (lp[pos, y] - lp0[pos, y]).item()
                sums["theta0_entropy"] += -(p0 * lp0[pos]).sum().item()
                sums["theta0_rare_mass"] += p0[rare_mask].sum().item()
                count += 1
    return {"count": count, **{key: value / count for key, value in sums.items()}}


# ------------------------------------------------------------------------------- the readouts


def test_readouts_on_a_constructed_case() -> None:
    """Two positions with written-out distributions: entropy, rare mass under θ₀'s mask and the
    own token's log-probability change equal the formulas."""
    tokens = module()
    p0 = torch.tensor(
        [[0.6, 0.3, 0.0995, 0.0005], [0.25, 0.25, 0.4999, 0.0001]], dtype=torch.float64
    )
    p = torch.tensor([[0.5, 0.3, 0.19, 0.01], [0.1, 0.2, 0.6, 0.1]], dtype=torch.float64)
    y = torch.tensor([1, 3])
    rare, entropy0, mass0, own0 = tokens.reference_readouts(p0.log(), y, 1e-3)
    assert rare.tolist() == [[False, False, False, True], [False, False, False, True]]
    for row in range(2):
        h0 = -math.fsum(q * math.log(q) for q in p0[row].tolist())
        assert entropy0[row].item() == pytest.approx(h0, rel=1e-13)
    assert mass0.tolist() == pytest.approx([0.0005, 0.0001], rel=1e-12)
    assert own0.tolist() == pytest.approx([math.log(0.3), math.log(0.0001)], rel=1e-13)
    entropy, mass = tokens.entropy_and_mass(p.log(), rare)
    for row in range(2):
        h = -math.fsum(q * math.log(q) for q in p[row].tolist())
        assert entropy[row].item() == pytest.approx(h, rel=1e-13)
    assert mass.tolist() == pytest.approx([0.01, 0.1], rel=1e-12)
    # a looser threshold marks more of the vocabulary
    rare, _, mass0, _ = tokens.reference_readouts(p0.log(), y, 0.26)
    assert rare.tolist() == [[False, False, True, True], [True, True, False, True]]
    assert mass0.tolist() == pytest.approx([0.1, 0.5001], rel=1e-12)


def test_token_statistics_equal_a_direct_computation() -> None:
    """Micro-batched, padded readouts of the training path equal each sample's own full forward
    pass; at θ₀ itself every difference is exactly zero."""
    torch.set_num_threads(1)
    tokens = module()
    reference = tiny_model(tied=False)
    student = tiny_model(tied=False)
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        for _, p in student.named_parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    g = torch.Generator().manual_seed(2)
    samples = []
    for i in range(5):
        prompt = torch.randint(7, 64, (int(torch.randint(3, 8, (1,), generator=g)),), generator=g)
        response = torch.randint(7, 64, (int(torch.randint(2, 9, (1,), generator=g)),), generator=g)
        samples.append(Sample(prompt_id=f"p{i}", prompt_ids=tuple(prompt.tolist()),
                              response_ids=(*response.tolist(), 5), logprobs=(),
                              ended=True))  # fmt: skip
    expected = direct(student, reference, samples, 1e-3)
    assert 0 < expected["theta0_rare_mass"] < 1 and 0 < expected["rare_mass"] < 1  # a real mask
    got = tokens.token_statistics(student, reference, samples, pad_id=6, micro_batch_size=2,
                                  rare=1e-3)  # fmt: skip
    assert got["count"] == expected["count"] == sum(len(s.response_ids) for s in samples)
    for key in ("entropy", "rare_mass", "delta_logprob", "theta0_entropy", "theta0_rare_mass"):
        assert got[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-9), key
    assert got["delta_logprob"] != 0
    same = tokens.token_statistics(reference, reference, samples, pad_id=6, micro_batch_size=2,
                                   rare=1e-3)  # fmt: skip
    assert same["delta_logprob"] == 0.0
    assert (same["entropy"], same["rare_mass"]) == (same["theta0_entropy"],
                                                    same["theta0_rare_mass"])  # fmt: skip
    for key in ("theta0_entropy", "theta0_rare_mass"):  # θ₀'s values do not depend on θ(a)
        assert same[key] == got[key]


# ------------------------------------------------------------------------------- the script


def f_run(env, tmp_path):  # noqa: F811
    """A tiny fresh F run with its endpoint checkpoint; the script's config and spec."""
    torch.set_num_threads(1)
    ckpt = tmp_path / "ckpt"
    overrides = [*F_RUN, "+study5.role=fresh"]
    cfg = ready(env, scheduled(tmp_path, *overrides, f"tracking.endpoint_checkpoint={ckpt}",
                               policy="short"))  # fmt: skip
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12",
              *overrides]  # fmt: skip
    return cfg, f_dir, ckpt, config, spec_path


def theta0_of(env, cfg):  # noqa: F811
    trainable = freeze(env.new_model(), cfg.train.freeze)
    return trainable, {name: p.detach() for name, p in trainable}


def write_displacement(env, cfg, ckpt, path, *, theta0_sha256=None):  # noqa: F811
    """Δθ of the run's checkpoint as a vector file (``vectors.displacement``)."""
    trainable, theta0 = theta0_of(env, cfg)
    names = [name for name, _ in trainable]
    end = torch.load(ckpt / "f-fresh.pt", weights_only=True)
    vectors.save_vector(path, vectors.displacement(end, theta0, names), names,
                        kind="displacement", provenance={"run_id": "f-fresh"},
                        theta0_sha256=theta0_sha256 or tensors_sha256(trainable))  # fmt: skip
    return path


def options(spec_path, out, *, rare=RARE, scales=SCALES):
    return ["--scales", *map(str, scales), "--out", str(out), "--surrogate", str(spec_path),
            "--panel-size", "3", "--rare", str(rare)]  # fmt: skip


def test_tokens_script_end_to_end(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    cfg, f_dir, ckpt, config, spec_path = f_run(env, tmp_path)
    f_inputs = json.loads((f_dir / "meta.json").read_text())["inputs"]
    vector = write_displacement(env, cfg, ckpt, tmp_path / "vectors" / "delta.pt")
    made = []

    def new_model():
        made.append(env.new_model())
        return made[-1]

    script = module()
    monkeypatch.setattr(
        script, "environment", lambda cfg, device: dataclasses.replace(env, new_model=new_model)
    )
    out = tmp_path / "records" / "tokens.json"
    argv = [*config, "--vector", str(vector), *options(spec_path, out)]
    assert script.main(argv) == 0
    record = json.loads(out.read_text())

    trainable, theta0 = theta0_of(env, cfg)
    theta0_sha = tensors_sha256(trainable)
    assert record["theta0_trainable_sha256"] == theta0_sha == f_inputs["theta0_trainable_sha256"]
    assert record["vector"]["source"] == "vector" and record["vector"]["file"] == "delta.pt"
    assert record["vector"]["sha256"] == vectors.read_record(vector)["sha256"]
    assert record["vector"]["provenance"] == {"run_id": "f-fresh"}
    assert record["scales"] == SCALES and [p["a"] for p in record["points"]] == SCALES
    assert record["surrogate_sha256"] == surrogate.load_spec(spec_path).sha256
    panels = record["panels"]
    assert (panels["harmful"]["name"], panels["harmful"]["count"]) == ("harmful_kl", 3)
    assert (panels["benign"]["name"], panels["benign"]["count"]) == ("alpaca_benign_kl", 3)

    # (1) a = 0 is θ₀: every token difference exactly zero, entropy and rare mass θ₀'s
    zero = record["points"][SCALES.index(0.0)]
    assert zero["tokens"]["delta_logprob"] == 0.0
    assert zero["tokens"]["entropy"] == record["theta0"]["tokens"]["entropy"]
    assert zero["tokens"]["rare_mass"] == record["theta0"]["tokens"]["rare_mass"]
    assert zero["r"] == record["theta0"]["r"] and zero["surrogate"] == record["theta0"]["surrogate"]
    assert 0 < record["theta0"]["tokens"]["rare_mass"] < 1  # the mask is not trivial

    # (2) the token means on the K panel the run's evaluator reads, against a direct computation
    reference = env.new_model()
    samples = Evaluator(cfg, env, _NoRun(), reference).kl_streams["kl_benign"]
    assert record["k_panel"]["stream"] == f_inputs["streams"]["kl_benign"]
    assert record["k_panel"]["samples"] == len(samples) == 6
    assert record["k_panel"]["prompts"] == 3 and record["k_panel"]["rare"] == RARE
    assert record["k_panel"]["micro_batch_size"] == cfg.execution.micro_batch_size
    tokens = sum(len(s.response_ids) for s in samples)
    assert record["k_panel"]["tokens"] == tokens
    v, _ = vectors.load_vector(vector, [name for name, _ in trainable])

    def point_model(a):
        model = env.new_model()
        trainable = freeze(model, cfg.train.freeze)
        set_point(trainable, [theta0[name] for name, _ in trainable],
                  [v[name] for name, _ in trainable], a)  # fmt: skip
        return model.eval()

    one = record["points"][0]
    assert one["a"] == 1.0 and one["tokens"]["count"] == tokens
    expected = direct(point_model(1.0), reference, samples, RARE)
    for key in ("entropy", "rare_mass", "delta_logprob"):
        assert one["tokens"][key] == pytest.approx(expected[key], rel=1e-6, abs=1e-9), key
    for key in ("entropy", "rare_mass"):
        assert record["theta0"]["tokens"][key] == pytest.approx(expected[f"theta0_{key}"],
                                                                rel=1e-6, abs=1e-9)  # fmt: skip
    assert one["tokens"]["delta_logprob"] != 0

    # (3) r from the pieces is surrogate_value at the point; the odd and even parts
    spec = surrogate.load_spec(spec_path)
    harmful = [item["text"] for item in env.prompt_set("harmful_kl")]
    benign = [item["text"] for item in env.prompt_set("alpaca_benign_kl")]
    kw = {"device": env.device, "micro_batch_size": int(cfg.execution.eval_batch_size),
          "max_tokens": spec.max_tokens}  # fmt: skip
    assert record["surrogate_micro_batch_size"] == cfg.execution.eval_batch_size
    for point in (record["points"][0], record["points"][3]):
        model = point_model(point["a"])
        assert point["r"] == surrogate.surrogate_value(
            model, env.tokenizer, harmful, spec.refusals, spec.compliances, **kw
        )
        values = surrogate.prompt_values(model, env.tokenizer, benign, spec.refusals,
                                         spec.compliances, **kw)  # fmt: skip
        assert point["surrogate"]["benign"]["difference"] == math.fsum(values) / 3
        for panel in point["surrogate"].values():
            assert panel["R"] - panel["C"] == pytest.approx(panel["difference"], abs=1e-12)
    r0 = record["theta0"]["r"]
    assert r0 == surrogate.surrogate_value(env.new_model().eval(), env.tokenizer, harmful,
                                           spec.refusals, spec.compliances, **kw)  # fmt: skip
    assert record["points"][0]["r"] != r0
    r = {point["a"]: point["r"] for point in record["points"]}
    assert record["pairs"] == [{"a": a, "odd": (r[a] - r[-a]) / 2,
                                "even": (r[a] + r[-a]) / 2 - r0} for a in (1.0, 0.5)]  # fmt: skip

    # (4) θ₀ restored exactly: the record's hash after the last point, and the models themselves
    assert record["theta0_restored_sha256"] == theta0_sha
    assert len(made) == 2
    for model in made:
        assert tensors_sha256(freeze(model, cfg.train.freeze)) == theta0_sha

    printed = capsys.readouterr().out
    assert theta0_sha in printed and f"r {r0:.9e}" in printed
    for text in harmful + benign + list(spec.continuations):
        assert text not in printed
    assert script.main(argv) == 1  # never overwritten
    assert json.loads(out.read_text()) == record


def test_the_vector_file_and_the_run_checkpoint_agree(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """(5) The same v from a displacement file and from the run's checkpoint gives the same
    numbers; --no-tokens leaves r unchanged."""
    cfg, f_dir, ckpt, config, spec_path = f_run(env, tmp_path)
    vector = write_displacement(env, cfg, ckpt, tmp_path / "vectors" / "delta.pt")
    script = module()
    monkeypatch.setattr(script, "environment", lambda cfg, device: env)
    by_file, by_run, bare = (tmp_path / "records" / f"{name}.json"
                             for name in ("file", "run", "bare"))  # fmt: skip
    assert script.main([*config, "--vector", str(vector), *options(spec_path, by_file)]) == 0
    assert script.main([*config, "--f-run", str(f_dir), "--checkpoints", str(ckpt),
                        *options(spec_path, by_run)]) == 0  # fmt: skip
    a, b = json.loads(by_file.read_text()), json.loads(by_run.read_text())
    for key in ("theta0", "points", "pairs", "theta0_trainable_sha256", "theta0_restored_sha256",
                "names_sha256", "k_panel", "panels"):  # fmt: skip
        assert a[key] == b[key], key
    meta = json.loads((f_dir / "meta.json").read_text())
    assert b["vector"]["source"] == "f_run" and b["vector"]["run_id"] == "f-fresh"
    assert b["vector"]["checkpoint"]["sha256"] == meta["inputs"]["endpoint_checkpoint"]["sha256"]
    assert b["vector"]["checkpoint"]["step"] == 3
    end = torch.load(ckpt / "f-fresh.pt", weights_only=True)
    assert b["vector"]["endpoint_trainable_sha256"] == tensors_sha256(
        [(name, end[name]) for name, _ in freeze(env.new_model(), cfg.train.freeze)]
    )
    assert b["vector"]["norm"] == a["vector"]["norm"]
    assert b["vector"]["count"] == a["vector"]["count"]

    assert script.main([*config, "--vector", str(vector), *options(spec_path, bare),
                        "--no-tokens"]) == 0  # fmt: skip
    c = json.loads(bare.read_text())
    assert c["k_panel"] is None and c["theta0"]["tokens"] is None
    assert all(point["tokens"] is None for point in c["points"])
    assert [p["r"] for p in c["points"]] == [p["r"] for p in a["points"]]
    assert c["pairs"] == a["pairs"]


def test_tokens_script_refusals(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """(6) A vector on another θ₀, an existing output, and malformed requests."""
    cfg, f_dir, ckpt, config, spec_path = f_run(env, tmp_path)
    script = module()
    monkeypatch.setattr(script, "environment", lambda cfg, device: env)
    out = tmp_path / "records" / "tokens.json"
    foreign = write_displacement(env, cfg, ckpt, tmp_path / "vectors" / "foreign.pt",
                                 theta0_sha256="0" * 64)  # fmt: skip
    assert script.main([*config, "--vector", str(foreign), *options(spec_path, out)]) == 1
    assert not out.exists()
    assert script.main([*config, "--vector", str(tmp_path / "none.pt"),
                        *options(spec_path, out)]) == 1  # fmt: skip
    assert script.main([*config, "--vector", str(foreign),
                        *options(spec_path, out, scales=[0.1, 0.1])]) == 1  # fmt: skip
    assert script.main([*config, "--vector", str(foreign),
                        *options(spec_path, out, rare=0.0)]) == 1  # fmt: skip
    with pytest.raises(SystemExit):  # --checkpoints only with --f-run
        script.main([*config, "--vector", str(foreign), "--checkpoints", str(ckpt),
                     *options(spec_path, out)])  # fmt: skip
    with pytest.raises(SystemExit):  # one source, not two
        script.main([*config, "--vector", str(foreign), "--f-run", str(f_dir), "--checkpoints",
                     str(ckpt), *options(spec_path, out)])  # fmt: skip
    assert script.main([*config, "--f-run", str(f_dir), "--checkpoints", str(ckpt),
                        *options(spec_path, out)[:-4], "--panel-size", "100"]) == 1  # fmt: skip
    assert not out.exists()

    # a run recording another θ₀; then an existing output is never overwritten
    meta_path = f_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    theta0 = meta["inputs"]["theta0_trainable_sha256"]
    meta["inputs"]["theta0_trainable_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    argv = [*config, "--f-run", str(f_dir), "--checkpoints", str(ckpt), *options(spec_path, out)]
    assert script.main(argv) == 1
    assert not out.exists()
    meta["inputs"]["theta0_trainable_sha256"] = theta0
    meta_path.write_text(json.dumps(meta))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{}\n")
    assert script.main(argv) == 1
    assert out.read_text() == "{}\n"
