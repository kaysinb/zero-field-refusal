"""Study 5's refusal-direction analysis (``analysis.study5_refdir`` and
scripts/study5_refdir_analysis.py) on synthetic numbers only: the leave-one-stream-out baselines
by hand; the vectorized centring and order scores equal to the scalar ones; the sign test's ties;
the agreement shares; the within-stream test; Holm's readings; the intervals; and the script end to
end on synthetic projections and harm records (a planted stream association read, the cohort and
the missing streams listed, M6's identity edit excluded by its permutation, an M6 order not the
record's stopping the analysis, an unvalidated direction without analysis, the record written
once, numbers only on stdout)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from zero_field_refusal import llama_orders
from zero_field_refusal.analysis import study5_refdir as sr
from zero_field_refusal.linear_response import momentum_weights
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.seeding import derive_seed


def script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = script_module(REPO_ROOT / "scripts" / "study5_refdir_analysis.py", "study5_refdir_script")
N = sr.STEPS * sr.BATCH_SIZE
D = 6


# ------------------------------------------------------------------ pure functions


def test_loso_baselines_by_hand():
    prompts = {1: ["a", "a", "b"], 2: ["a", "c"], 3: ["b"]}
    values = {1: [1.0, 3.0, 5.0], 2: [10.0, 7.0], 3: [20.0]}
    b = sr.loso_baselines(prompts, values)
    assert b == {1: [10.0, 10.0, 20.0], 2: [2.0, None], 3: [5.0]}
    assert sr.centred(values[2], b[2]) == [8.0, None]


def stream_data(seeds, rng, d=D):
    prompts = {s: [f"p{int(i)}" for i in rng.integers(0, 100, N)] for s in seeds}
    vectors = {s: rng.standard_normal((N, d)) for s in seeds}
    return prompts, vectors


def test_vectorized_scores_equal_the_scalar_ones():
    rng = np.random.default_rng(1)
    seeds = [1, 2, 3]
    prompts, vectors = stream_data(seeds, rng)
    prompts[3][0] = "only-here"  # no baseline: left out of the scores
    unit = rng.standard_normal(D)
    unit /= np.linalg.norm(unit)
    weights = momentum_weights(0.9, sr.STEPS)
    rows = sr.centred_vectors(prompts, vectors)
    assert np.isnan(rows[3][0]).all() and not np.isnan(rows[3][1:]).any()
    values = {s: list(vectors[s] @ unit) for s in seeds}
    base = sr.loso_baselines(prompts, values)
    order = [int(i) for i in rng.permutation(N)]
    orders = {(s, o): (None if o == "own" else order) for s in seeds for o in ("own", "new")}
    stream = sr.stream_scores(rows, unit[:, None])
    by_order = sr.order_scores(rows, orders, weights, unit[:, None])
    for s in seeds:
        c = sr.centred(values[s], base[s])
        assert stream[s][0] == pytest.approx(sr.stream_score(c), abs=1e-12)
        assert by_order[(s, "own")][0] == pytest.approx(sr.order_score(c, None, weights), abs=1e-12)
        assert by_order[(s, "new")][0] == pytest.approx(
            sr.order_score(c, order, weights), abs=1e-12
        )
    with pytest.raises(ValueError, match="permute"):
        sr.order_score([0.0] * N, list(range(N - 1)) + [0], weights)


def test_the_order_score_weights_batches_by_position():
    weights = momentum_weights(0.9, sr.STEPS)
    values = [0.0] * N
    for i in range(2 * sr.BATCH_SIZE, 3 * sr.BATCH_SIZE):
        values[i] = 1.0
    assert sr.order_score(values, None, weights) == pytest.approx(weights[2])


def test_the_sign_test_excludes_ties_and_the_shares_agree():
    pairs = [(-1.0, 2.0), (1.0, -2.0), (1.0, 2.0), (0.0, 5.0), (-1.0, 1e-12)]
    t = sr.sign_test(pairs)
    assert (t["n"], t["ties"], t["agreements"]) == (3, 2, 2)
    assert t["p"] == pytest.approx(0.5)
    shares = sr.agreement_shares(np.array([[w] for w, _ in pairs]), [h for _, h in pairs])
    assert shares[0] == pytest.approx(2 / 3)


def test_the_within_stream_test_reads_a_perfect_reversal():
    w = {s: [1.0, 2.0, 3.0] for s in range(10)}
    h = {s: [3.0, 2.0, 1.0] for s in range(10)}
    t = sr.within_stream_test(w, h, draws=20_000)
    assert t["statistic"] == pytest.approx(-1.0)
    assert t["p"] < 1e-3
    flat = sr.within_stream_test(w, {s: [1.0, 3.0, 2.0] for s in range(10)}, draws=20_000)
    assert flat["p"] > 0.2


def test_holm_keeps_the_family_of_four():
    out = sr.holm({"T1": 0.01, "T2": 0.04, "T3": 0.03, "T4": None})
    assert out["T1"]["adjusted"] == pytest.approx(0.04)  # four tests, the unavailable one as p = 1
    assert out["T3"]["adjusted"] == pytest.approx(0.09)
    assert out["T2"]["adjusted"] == pytest.approx(0.09)  # monotone
    assert out["T1"]["reading"] == "retrospectively associated with harm"
    assert {out[k]["reading"] for k in ("T2", "T3")} == {"not established"}
    assert out["T4"] == {"p": None, "adjusted": None, "reading": "unavailable"}
    # the unvalidated branch: T1 = 0.02, T3 = 0.8, the Llama tests unavailable
    out = sr.holm({"T1": 0.02, "T2": None, "T3": 0.8, "T4": None})
    assert out["T1"]["adjusted"] == pytest.approx(0.08)
    assert out["T1"]["reading"] == "not established"
    assert sr.holm({"T1": float("nan"), "T2": 0.01, "T3": 1.0, "T4": 1.0})["T1"]["reading"] == (
        "unavailable"
    )
    with pytest.raises(ValueError, match="family"):
        sr.holm({"T1": 0.01, "T3": 0.02})


def test_undefined_statistics_are_unavailable_not_small_p():
    y = [float(i) for i in range(12)]
    const = sr.permutation_p([1.0] * 12, y, draws=200)
    assert const["p"] is None and const["unavailable"] == "rho undefined"
    assert sr.permutation_p(y, [2.0] * 12, draws=200)["p"] is None
    nan = sr.permutation_p([*y[:-1], float("nan")], y, draws=200)
    assert nan["p"] is None and nan["unavailable"] == "a non-finite value"
    assert sr.permutation_p(y[:2], y[:2], draws=200)["p"] is None  # under three points
    assert sr.bootstrap_rho([1.0] * 12, y, draws=200) == (None, None)
    # T4: a constant triple leaves its stream out, listed; none left: unavailable
    w = {s: [1.0, 2.0, 3.0] for s in range(10)}
    h = {s: [3.0, 2.0, 1.0] for s in range(10)}
    h[4] = [2.0, 2.0, 2.0]
    t4 = sr.within_stream_test(w, h, draws=5_000)
    assert t4["left_out"] == [4] and t4["n"] == 9 and t4["statistic"] == pytest.approx(-1.0)
    gone = sr.within_stream_test(w, {s: [1.0, 1.0, 1.0] for s in range(10)}, draws=500)
    assert gone["p"] is None and gone["unavailable"] == "no stream with a defined rho"
    h[5] = [3.0, float("nan"), 1.0]
    assert sr.within_stream_test(w, h, draws=500)["unavailable"] == "a non-finite value"
    # the sign test, the cluster bootstrap, the shares and the percentiles
    assert sr.sign_test([(1.0, float("inf"))])["p"] is None
    assert sr.sign_test([(0.0, 1.0), (1.0, 0.0)])["unavailable"] == "no untied pair"
    assert sr.cluster_agreement([(1, float("nan"), 1.0)], draws=50)["unavailable"]
    dw = np.array([[1.0, float("nan")], [-1.0, 1.0]])
    shares = sr.agreement_shares(dw, [-2.0, 2.0])
    assert shares[0] == pytest.approx(1.0) and math.isnan(shares[1])
    found = sr.percentile(0.5, [0.0, float("nan"), 1.0])
    assert found == {"percentile": 50.0, "finite": 2, "of": 3}
    assert sr.percentile(float("nan"), [0.0, 1.0])["percentile"] is None


def test_empty_answers_enter_no_score_and_no_baseline():
    rng = np.random.default_rng(5)
    seeds = [1, 2, 3]
    prompts, vectors = stream_data(seeds, rng)
    for s, i in ((1, 0), (2, 7), (2, 8)):
        vectors[s][i] = np.nan  # an answer of only its end token
    prompts[2][7] = prompts[1][3]  # an empty answer to a prompt others answered
    rows = sr.centred_vectors(prompts, vectors)
    unit = np.eye(D)[0]
    values = {s: list(vectors[s] @ unit) for s in seeds}
    base = sr.loso_baselines(prompts, values)
    for s in seeds:
        c = sr.centred(values[s], base[s])
        got = rows[s] @ unit
        assert [None if math.isnan(v) else pytest.approx(v) for v in got] == c
    assert np.isnan(rows[1][0]).all() and np.isnan(rows[2][7]).all()
    # a baseline never holds an empty answer: s1's answer 3 is centred on finite values only
    assert math.isfinite(rows[1][3][0])
    counts = np.full(N, 10)
    counts[0] = 0
    tok = sr.token_rows(vectors[1], counts)
    assert np.array_equal(tok[0], np.zeros(D)) and np.allclose(tok[1], vectors[1][1] * 10 / 96)


def test_intervals_and_partial_rank_correlation():
    assert sr.bootstrap_mean([0.5] * 7, draws=200) == (0.5, 0.5)
    rows = [(1, -1.0, 1.0), (1, 1.0, -1.0), (2, 1.0, 1.0), (2, 0.0, 1.0)]
    c = sr.cluster_agreement(rows, draws=500)
    assert (c["n"], c["ties"], c["agreements"]) == (3, 1, 2)
    assert c["lower"] <= c["share"] <= c["upper"]
    rng = np.random.default_rng(2)
    z = rng.standard_normal(400)
    x, y = z + 0.1 * rng.standard_normal(400), z + 0.1 * rng.standard_normal(400)
    assert sr.spearman(x, y) > 0.9
    assert abs(sr.partial_spearman(x, y, z)) < 0.2
    assert sr.percentile(0.0, [-1.0, 0.0, 1.0, 2.0])["percentile"] == pytest.approx(50.0)


# ------------------------------------------------------------------ the script end to end

QWEN, LLAMA = script.QWEN, script.LLAMA


def write_direction(folder: Path, d: int, model: str, validated: bool = True) -> np.ndarray:
    folder.mkdir(parents=True)
    unit = np.zeros(d)
    unit[0] = 1.0
    record: dict = {"model": model, "model_name": f"m-{model}", "model_revision": "r0",
                    "control_layer": 7}  # fmt: skip
    if validated:
        control = np.zeros(d)
        control[1] = 1.0
        torch.save(
            {
                "chosen": torch.tensor(3 * unit),
                "control": torch.tensor(control),
                "layer": 5,
                "position": 2,
                "control_layer": 7,
            },
            folder / "vectors.pt",
        )
        record["vectors_sha256"] = hashlib.sha256((folder / "vectors.pt").read_bytes()).hexdigest()
    record["chosen"] = {"layer": 5, "position": 2} if validated else None
    (folder / "direction.json").write_text(json.dumps(record))
    return unit


EMPTY = (4, 250, 499)  # each seed's answers of only their end token


def write_answers(folder: Path, seeds, shift: dict[int, float], rng, binding: dict) -> None:
    folder.mkdir(parents=True)
    base = rng.standard_normal((100, D))
    for s in seeds:
        idx = rng.integers(0, 100, N)
        means = np.zeros((N, 2, D), np.float32)
        means[:, 0] = base[idx] + 0.3 * rng.standard_normal((N, D))
        means[:, 0, 0] += shift[s]
        means[:, 1] = rng.standard_normal((N, D))
        counts = rng.integers(5, 97, N)
        means[list(EMPTY)] = np.nan
        counts[list(EMPTY)] = 0
        path = folder / f"s{s}.npz"
        np.savez(
            path, means=means, counts=counts, prompt_ids=np.asarray([f"p{int(i)}" for i in idx])
        )
        receipt = {"seed": s, "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "binding": binding, "empty_answers": len(EMPTY)}  # fmt: skip
        (folder / f"s{s}.json").write_text(json.dumps(receipt))


def identity_draw(seed: int) -> int:
    for draw in range(1000):
        rng = np.random.default_rng(derive_seed(seed, "study5", "shuffle", 1, 2, draw))
        if list(rng.permutation(2)) == [0, 1]:
            return draw
    raise AssertionError("no identity draw")


def harm_records(tmp: Path, shift: dict[int, float], rng) -> dict[str, Path]:
    """h* falls with the stream's shift along the direction (the planted association)."""

    def h_of(s: int) -> float:  # a fraction
        return min(0.95, max(0.02, 0.3 - 0.2 * shift[s] + 0.01 * rng.standard_normal()))

    outcomes = [{"seed": s, "h": h_of(s), "support": {"state": "supported"}} for s in QWEN]
    outcomes[3]["support"]["state"] = "unsupported"  # s50003 leaves the stream cohort
    per_seed = []
    for s in QWEN:
        orders = {
            o: {"order_seed": s + off, "support": {"state": "supported"}, "h": h_of(s)}
            for o, off in (("order1", 1000), ("order2", 2000))
        }
        per_seed.append({"seed": s, "orders": orders})
    per_seed[7]["orders"]["order2"] = {
        "order_seed": 50007 + 2000,
        "h": None,
        "support": {"state": "unsupported"},
    }
    cells = [
        {"seed": s, "order": o, "state": "supported", "h_star": {"harmful": 100 * h_of(s)}}
        for s in LLAMA
        for o, _ in llama_orders.ORDER_OFFSETS
    ]
    rows = []
    for s in QWEN[:6]:
        for edit in (
            {"kind": "shuffle", "first": 1, "last": 5, "draw": 0},
            {"kind": "swap_batches", "step": 2},
        ):
            order = script.order_of(s, edit=edit)
            rows.append(
                {
                    "seed": s,
                    "edit": edit,
                    "tag": edit["kind"],
                    "prediction": {"order": {"permutation_sha256": script.permutation_sha(order)}},
                    "outcome_difference": {"harmful": 0.01 * rng.standard_normal()},
                }
            )
    ident = {"kind": "shuffle", "first": 1, "last": 2, "draw": identity_draw(QWEN[0])}
    rows.append(
        {
            "seed": QWEN[0],
            "edit": ident,
            "tag": "identity",
            "prediction": {
                "order": {
                    "permutation_sha256": script.permutation_sha(
                        script.order_of(QWEN[0], edit=ident)
                    )
                }
            },
            "outcome_difference": {"harmful": 0.0},
        }
    )
    paths = {}
    for name, value in (
        ("final", {"outcomes": outcomes}),
        ("e2", {"per_seed": per_seed}),
        ("ext2", {"m6": {"shuffles": {"rows": rows[:-1]}, "swaps": {"rows": rows[-1:]}}}),
        ("llama", {"kstar": {"cells": cells}}),
    ):
        paths[name] = tmp / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    return paths


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("refdir")
    rng = np.random.default_rng(3)
    shift = {s: float(rng.uniform(-1, 1)) for s in QWEN + LLAMA}
    write_direction(tmp / "qdir", D, "qwen")
    write_direction(tmp / "ldir", D, "llama")
    write_answers(tmp / "qwen", QWEN, shift, rng, script.binding_of("qwen", tmp / "qdir"))
    write_answers(tmp / "llama", LLAMA, shift, rng, script.binding_of("llama", tmp / "ldir"))
    paths = harm_records(tmp, shift, rng)
    return tmp, paths


def argv_of(tmp: Path, paths: dict[str, Path], out: Path, llama_dir: str = "ldir") -> list[str]:
    return [
        "--qwen",
        str(tmp / "qwen"),
        "--qwen-direction",
        str(tmp / "qdir"),
        "--llama",
        str(tmp / "llama"),
        "--llama-direction",
        str(tmp / llama_dir),
        "--final",
        str(paths["final"]),
        "--e2",
        str(paths["e2"]),
        "--ext2",
        str(paths["ext2"]),
        "--llama-analysis",
        str(paths["llama"]),
        "--out",
        str(out),
    ]


def test_the_script_end_to_end(built, capsys):
    tmp, paths = built
    out = tmp / "record.json"
    assert script.main(argv_of(tmp, paths, out)) == 0
    printed = capsys.readouterr().out
    assert all(
        line.split()[0] in {"T1", "T2", "T3", "T4", "record"} for line in printed.splitlines()
    )
    rec = json.loads(out.read_text())
    q, lam = rec["models"]["qwen"], rec["models"]["llama"]
    assert q["stream_missing"] == [50003, 50007] and q["order_missing"] == [50007]
    assert len(q["cohort"]) == 38 and len(lam["cohort"]) == 10
    assert q["stream_test"]["rho"] < -0.8 and lam["stream_test"]["rho"] < -0.8
    assert rec["family"]["T1"]["reading"] == "retrospectively associated with harm"
    assert q["order_test"]["n"] + q["order_test"]["ties"] == 39
    lo_, hi_ = q["stream_test"]["interval"]
    assert lo_ <= q["stream_test"]["rho"] <= hi_
    assert q["m6"]["edits"] == 12 and q["m6"]["identities"] == ["identity-s50000"]
    assert q["own_order"]["n"] == 39
    assert set(q["random"]) == {"stream", "order"}
    assert (
        0 <= q["random"]["stream"]["percentile"] <= 100 and q["random"]["stream"]["finite"] == 1000
    )
    assert q["empty_answers"] == {str(s): list(EMPTY) for s in QWEN}
    assert q["missing_baseline"] == {}
    assert {"stream", "order", "partial_stream"} <= set(q["length"])
    assert q["direction"] == {"layer": 5, "position": 2}
    assert rec["inputs"]["final"] == hashlib.sha256(paths["final"].read_bytes()).hexdigest()
    assert len(q["files"]) == 40
    assert script.main(argv_of(tmp, paths, out)) == 1  # written once


def test_the_vectorized_record_matches_a_scalar_recomputation(built):
    tmp, paths = built
    out = tmp / "record-check.json"
    assert script.main(argv_of(tmp, paths, out)) == 0
    rec = json.loads(out.read_text())["models"]["qwen"]
    ans = script.load_answers(tmp / "qwen", QWEN, script.binding_of("qwen", tmp / "qdir"))
    values = {s: list(ans["means"][s][:, 0]) for s in QWEN}  # the unit is e0
    base = sr.loso_baselines(ans["prompts"], values)
    weights = momentum_weights(0.9, sr.STEPS)
    e2 = {p["seed"]: p["orders"] for p in json.loads(paths["e2"].read_text())["per_seed"]}
    pairs = []
    for s in QWEN:
        if e2[s]["order2"]["h"] is None:
            continue
        c = sr.centred(values[s], base[s])
        w1 = sr.order_score(
            c, script.order_of(s, order_seed=e2[s]["order1"]["order_seed"]), weights
        )
        w2 = sr.order_score(
            c, script.order_of(s, order_seed=e2[s]["order2"]["order_seed"]), weights
        )
        pairs.append((w1 - w2, 100 * (e2[s]["order1"]["h"] - e2[s]["order2"]["h"])))
    assert rec["order_test"]["agreements"] == sr.sign_test(pairs)["agreements"]


def test_an_m6_order_not_the_records_stops(built, capsys, tmp_path):
    tmp, paths = built
    ext2 = json.loads(paths["ext2"].read_text())
    ext2["m6"]["shuffles"]["rows"][0]["prediction"]["order"]["permutation_sha256"] = "0" * 64
    bad = dict(paths)
    bad["ext2"] = tmp_path / "ext2.json"
    bad["ext2"].write_text(json.dumps(ext2))
    out = tmp_path / "record.json"
    assert script.main(argv_of(tmp, bad, out)) == 2
    assert capsys.readouterr().out.startswith("STOP M6")
    assert not out.exists()


def test_an_unvalidated_direction_has_no_analysis(built, tmp_path):
    tmp, paths = built
    write_direction(tmp_path / "ldir-unvalidated", D, "llama", validated=False)
    out = tmp_path / "record.json"
    argv = argv_of(tmp, paths, out)
    argv[argv.index("--llama-direction") + 1] = str(tmp_path / "ldir-unvalidated")
    assert script.main(argv) == 0
    rec = json.loads(out.read_text())
    assert rec["models"]["llama"] == {"direction": "unvalidated"}
    assert rec["family"]["T2"] == {"p": None, "adjusted": None, "reading": "unavailable"}
    assert rec["family"]["T4"]["reading"] == "unavailable"
    p1, p3 = (rec["models"]["qwen"][k]["p"] for k in ("stream_test", "order_test"))
    assert (
        rec["family"]["T1"]["adjusted"]
        == sr.holm({"T1": p1, "T2": None, "T3": p3, "T4": None})["T1"]["adjusted"]
    )


def _copy(built, tmp_path) -> tuple[Path, dict[str, Path]]:
    import shutil

    tmp, paths = built
    work = tmp_path / "w"
    for name in ("qwen", "llama", "qdir", "ldir"):
        shutil.copytree(tmp / name, work / name)
    return work, paths


@pytest.mark.parametrize("case", ["no receipt", "other layers", "npz changed", "vectors changed",
                                  "another model's direction", "no m6 hash"])  # fmt: skip
def test_projections_are_bound_to_their_direction(built, tmp_path, capsys, case):
    work, paths = _copy(built, tmp_path)
    if case == "no receipt":
        (work / "qwen" / "s50001.json").unlink()
    elif case == "other layers":
        receipt = json.loads((work / "qwen" / "s50001.json").read_text())
        receipt["binding"]["layers"] = [6, 7]
        (work / "qwen" / "s50001.json").write_text(json.dumps(receipt))
    elif case == "npz changed":
        z = dict(np.load(work / "qwen" / "s50001.npz"))
        z["counts"] = z["counts"] + 1
        np.savez(work / "qwen" / "s50001.npz", **z)
    elif case == "vectors changed":
        v = torch.load(work / "qdir" / "vectors.pt")
        v["chosen"] = 2 * v["chosen"]
        torch.save(v, work / "qdir" / "vectors.pt")
    elif case == "another model's direction":
        import shutil

        shutil.rmtree(work / "qdir")
        shutil.copytree(work / "ldir", work / "qdir")
    else:
        ext2 = json.loads(paths["ext2"].read_text())
        del ext2["m6"]["swaps"]["rows"][0]["prediction"]["order"]["permutation_sha256"]
        paths = dict(paths)
        paths["ext2"] = tmp_path / "ext2.json"
        paths["ext2"].write_text(json.dumps(ext2))
    out = tmp_path / "record.json"
    assert script.main(argv_of(work, paths, out)) == 2
    assert capsys.readouterr().out.startswith("STOP") and not out.exists()
