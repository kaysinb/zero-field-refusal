"""Study 5's training answers along the refusal direction, analysed on a CPU
(docs/study5-refdir-registration.md; ``analysis.study5_refdir``).

    python scripts/study5_refdir_analysis.py --qwen DIR --qwen-direction DIR --llama DIR
        --llama-direction DIR --final FILE --e2 FILE --ext2 FILE --llama-analysis FILE
        --out RECORD.json

--qwen and --llama are the projection folders of ``scripts/study5_refdir.py project`` (one
``s<seed>.npz`` per seed); --qwen-direction and --llama-direction its direction folders
(``direction.json``, ``vectors.pt``). The harm records are the study's, each read for h* in
percentage points:

- ``--final``, the registered analysis: ``outcomes[]`` with ``support.state`` = supported, ``h``
  (the stream's own order at K*);
- ``--e2``, E2's test record: ``per_seed[].orders.order1|order2`` with ``support.state`` =
  supported, ``h`` and the order's ``order_seed``;
- ``--ext2``, extension 2's labeled record: M6's edits (``m6.shuffles.rows``, ``m6.swaps.rows``),
  their ``edit``, the permutation's sha256 and ``outcome_difference.harmful`` at the fixed rate;
- ``--llama-analysis``, the Llama analysis: ``kstar.cells[]`` with ``state`` = supported and
  ``h_star.harmful``; the orders' seeds are ``llama_orders.order_seed``.

Each order is rebuilt by ``study5_options.training_order``; an M6 order whose sha256 differs from
the record's stops the analysis. The position weights are ``linear_response.momentum_weights(0.9,
25)``. The record (written once) holds T1-T4 with their Holm reading, the secondary statistics and
the controls; stdout carries numbers only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from zero_field_refusal import llama_orders
from zero_field_refusal.analysis import study5_refdir as sr
from zero_field_refusal.linear_response import momentum_weights
from zero_field_refusal.study5_options import training_order

QWEN = tuple(range(50000, 50040))
LLAMA = tuple(range(60000, 60010))
E2_ORDERS = ("order1", "order2")
LLAMA_ORDERS = tuple(o for o, _ in llama_orders.ORDER_OFFSETS)
RANDOM = 1000
BETA1 = 0.9


class Stop(RuntimeError):
    """A rule of the registration stops the analysis."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def permutation_sha(order) -> str:
    return hashlib.sha256(json.dumps(list(order)).encode()).hexdigest()


def order_of(seed: int, *, order_seed: int | None = None, edit: dict | None = None):
    train: dict[str, Any] = {"steps": sr.STEPS, "batch_size": sr.BATCH_SIZE}
    if order_seed is not None:
        train["order_seed"] = order_seed
    if edit is not None:
        train["order_edit"] = edit
    return training_order(OmegaConf.create({"seed": seed, "train": train}))


# ------------------------------------------------------------------ inputs


def harm_of(final: dict, e2: dict, lam: dict) -> dict[str, Any]:
    """h* in percentage points: Qwen's own order, E2's two orders (with their order seeds), and
    Llama's three orders; supported cells only."""
    own = {
        o["seed"]: 100 * o["h"]
        for o in final["outcomes"]
        if o["support"]["state"] == "supported" and o.get("h") is not None
    }
    e2_h, e2_seed = {}, {}
    for p in e2["per_seed"]:
        for name in E2_ORDERS:
            cell = p["orders"][name]
            e2_seed[(p["seed"], name)] = cell["order_seed"]
            if cell["support"]["state"] == "supported" and cell.get("h") is not None:
                e2_h[(p["seed"], name)] = 100 * cell["h"]
    lam_h = {
        (c["seed"], c["order"]): c["h_star"]["harmful"]
        for c in lam["kstar"]["cells"]
        if c["state"] == "supported" and c.get("h_star") is not None
    }
    return {"own": own, "e2": e2_h, "e2_order_seed": e2_seed, "llama": lam_h}


def load_answers(folder: Path, seeds) -> dict[str, Any]:
    out: dict[str, Any] = {"prompts": {}, "means": {}, "control": {}, "counts": {}, "files": {}}
    for s in seeds:
        path = folder / f"s{s}.npz"
        z = np.load(path)
        if z["means"].shape[0] != sr.STEPS * sr.BATCH_SIZE:
            raise Stop(f"s{s}: {z['means'].shape[0]} answers, not {sr.STEPS * sr.BATCH_SIZE}")
        out["prompts"][s] = [str(x) for x in z["prompt_ids"]]
        out["means"][s] = z["means"][:, 0, :].astype(np.float64)
        out["control"][s] = z["means"][:, 1, :].astype(np.float64)
        out["counts"][s] = z["counts"].astype(np.float64)
        out["files"][f"s{s}.npz"] = sha256(path)
    return out


def units_of(direction: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    v = torch.load(direction / "vectors.pt")
    chosen, control = (v[k].double().numpy() for k in ("chosen", "control"))
    return chosen / np.linalg.norm(chosen), control / np.linalg.norm(control)


def random_units(d: int, n: int = RANDOM, seed: int = sr.SEED) -> np.ndarray:
    g = np.random.default_rng(seed).standard_normal((d, n))
    return g / np.linalg.norm(g, axis=0, keepdims=True)


# ------------------------------------------------------------------ design per model


def design(name: str, harm: dict[str, Any], seeds) -> dict[str, Any]:
    """The orders whose W enters the tests, the stream outcome (mean h* over three orders), and the
    order comparison: Qwen's E2 pairs (T3) or Llama's within-stream triples (T4)."""
    if name == "qwen":
        orders: dict = {(s, "own"): None for s in seeds}
        orders.update(
            {
                (s, o): order_of(s, order_seed=harm["e2_order_seed"][(s, o)])
                for s in seeds
                for o in E2_ORDERS
                if (s, o) in harm["e2_order_seed"]
            }
        )
        three = {
            s: [harm["own"].get(s)] + [harm["e2"].get((s, o)) for o in E2_ORDERS] for s in seeds
        }
        pairs = [
            (s, harm["e2"][(s, "order1")] - harm["e2"][(s, "order2")])
            for s in seeds
            if (s, "order1") in harm["e2"] and (s, "order2") in harm["e2"]
        ]
        return {
            "orders": orders,
            "three": three,
            "pairs": pairs,
            "groups": {},
            "order_missing": [s for s in seeds if s not in {p for p, _ in pairs}],
        }
    orders = {
        (s, o): order_of(s, order_seed=llama_orders.order_seed(s, o))
        for s in seeds
        for o in LLAMA_ORDERS
    }
    three = {s: [harm["llama"].get((s, o)) for o in LLAMA_ORDERS] for s in seeds}
    groups = {
        s: [harm["llama"][(s, o)] for o in LLAMA_ORDERS]
        for s in seeds
        if all((s, o) in harm["llama"] for o in LLAMA_ORDERS)
    }
    return {
        "orders": orders,
        "three": three,
        "pairs": [],
        "groups": groups,
        "order_missing": [s for s in seeds if s not in groups],
    }


def scores(rows: dict[int, np.ndarray], units: np.ndarray, des: dict) -> tuple[dict, dict]:
    weights = momentum_weights(BETA1, sr.STEPS)
    return sr.stream_scores(rows, units), sr.order_scores(rows, des["orders"], weights, units)


def stats(stream: dict, order: dict, des: dict, cohort: list[int]) -> dict[str, np.ndarray]:
    """Per direction (column): the stream statistic ρ(P, mean h*) over the cohort and the order
    statistic, the agreement share over E2's pairs (Qwen) or the mean within-stream ρ(W, h*)
    (Llama)."""
    y = [float(np.mean(des["three"][s])) for s in cohort]
    stream_mat = np.stack([stream[s] for s in cohort])  # (streams, directions)
    k = stream_mat.shape[1]
    rho = np.array([sr.spearman(stream_mat[:, j], y) for j in range(k)])
    if des["pairs"]:
        dw = np.stack([order[(s, "order1")] - order[(s, "order2")] for s, _ in des["pairs"]])
        o_stat = sr.agreement_shares(dw, [dh for _, dh in des["pairs"]])
    else:
        per = np.array(
            [
                [
                    sr.spearman([order[(s, o)][j] for o in LLAMA_ORDERS], des["groups"][s])
                    for j in range(k)
                ]
                for s in sorted(des["groups"])
            ]
        )
        o_stat = per.mean(axis=0)
    return {"stream": rho, "order": o_stat}


def first(st: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(v[0]) for k, v in st.items()}


# ------------------------------------------------------------------ one model


def model_record(
    name: str,
    ans: dict,
    unit: np.ndarray,
    control: np.ndarray,
    des: dict,
    harm: dict,
    ext2: dict | None,
) -> dict[str, Any]:
    seeds = sorted(ans["prompts"])
    cohort = [s for s in seeds if all(h is not None for h in des["three"][s])]
    y = [float(np.mean(des["three"][s])) for s in cohort]
    rows = sr.centred_vectors(ans["prompts"], ans["means"])
    tok = {s: ans["means"][s] * (ans["counts"][s][:, None] / sr.TOKEN_DENOMINATOR) for s in seeds}
    rows_tok = sr.centred_vectors(ans["prompts"], tok)
    rows_ctrl = sr.centred_vectors(ans["prompts"], ans["control"])
    res: dict[str, Any] = {
        "cohort": cohort,
        "stream_missing": [s for s in seeds if s not in cohort],
        "order_missing": des["order_missing"],
        "missing_baseline": {
            str(s): [ans["prompts"][s][i] for i in np.flatnonzero(np.isnan(rows[s][:, 0]))]
            for s in seeds
            if np.isnan(rows[s][:, 0]).any()
        },
    }
    all_units = np.column_stack([unit, random_units(unit.shape[0])])
    st, ot = scores(rows, all_units, des)
    allstats = stats(st, ot, des, cohort)
    p_c = {s: float(st[s][0]) for s in seeds}
    w = {key: float(v[0]) for key, v in ot.items()}
    # the family's tests
    stream = sr.permutation_p([p_c[s] for s in cohort], y)
    stream["interval"] = sr.bootstrap_rho([p_c[s] for s in cohort], y)
    res["stream_test"] = stream
    if des["pairs"]:
        rows_p = [(s, w[(s, "order1")] - w[(s, "order2")], dh) for s, dh in des["pairs"]]
        order = sr.sign_test([(dw, dh) for _, dw, dh in rows_p])
        share = sr.cluster_agreement(rows_p)
        order["share"], order["interval"] = share["share"], (share["lower"], share["upper"])
        order["tied"] = [s for s, dw, dh in rows_p if dw == 0 or abs(dh) < sr.TIE_PP]
    else:
        ws = {s: [w[(s, o)] for o in LLAMA_ORDERS] for s in des["groups"]}
        order = sr.within_stream_test(ws, des["groups"])
        per = [sr.spearman(ws[s], des["groups"][s]) for s in sorted(ws)]
        order["interval"] = sr.bootstrap_mean(per)
    res["order_test"] = order
    # secondary, descriptive
    raw = {s: ans["means"][s] for s in seeds}
    res["raw"] = first(stats(*scores(raw, unit[:, None], des), des, cohort))
    res["tok"] = first(stats(*scores(rows_tok, unit[:, None], des), des, cohort))
    if name == "qwen":
        sup = sorted(harm["own"])
        res["own_order"] = {
            "n": len(sup),
            "rho": sr.spearman([p_c[s] for s in sup], [harm["own"][s] for s in sup]),
        }
        res["m6"] = m6_record(ext2, rows, unit)
    # controls, descriptive
    res["random"] = {k: sr.percentile(v[0], v[1:]) for k, v in allstats.items()}
    res["control_layer"] = first(stats(*scores(rows_ctrl, control[:, None], des), des, cohort))
    length = {s: ans["counts"][s][:, None] for s in seeds}
    ls, lo = scores(length, np.ones((1, 1)), des)
    res["length"] = first(stats(ls, lo, des, cohort))
    res["length"]["partial_stream"] = sr.partial_spearman(
        [p_c[s] for s in cohort], y, [float(ls[s][0]) for s in cohort]
    )
    res["files"] = ans["files"]
    return res


def m6_record(ext2: dict, rows: dict[int, np.ndarray], unit: np.ndarray) -> dict[str, Any]:
    """M6's edits against their stream order: ΔW and Δh (pp) per edit; identities by their
    permutation excluded and listed; the cluster-bootstrap agreement over streams."""
    weights = momentum_weights(BETA1, sr.STEPS)
    out, identities, unavailable = [], [], []
    for r in ext2["m6"]["shuffles"]["rows"] + ext2["m6"]["swaps"]["rows"]:
        s, edit = r["seed"], r["edit"]
        order = order_of(s, edit=edit)
        recorded = ((r.get("prediction") or {}).get("order") or {}).get("permutation_sha256")
        if recorded is not None and recorded != permutation_sha(order):
            raise Stop(f"M6 {r['tag']} s{s}: the rebuilt order differs from the record's")
        if order == list(range(len(order))):
            identities.append(f"{r['tag']}-s{s}")
            continue
        diff = r.get("outcome_difference") or {}
        if diff.get("harmful") is None:
            unavailable.append(f"{r['tag']}-s{s}")
            continue
        both = sr.order_scores(
            {s: rows[s]}, {(s, "e"): order, (s, "o"): None}, weights, unit[:, None]
        )
        out.append((s, float(both[(s, "e")][0] - both[(s, "o")][0]), 100 * diff["harmful"]))
    agree = sr.cluster_agreement(out)
    agree.update(
        {
            "edits": len(out),
            "identities": identities,
            "unavailable": unavailable,
            "streams": sorted({s for s, _, _ in out}),
        }
    )
    return agree


# ------------------------------------------------------------------ record


def record_of(args) -> dict[str, Any]:
    harm = harm_of(*(json.loads(p.read_text()) for p in (args.final, args.e2, args.llama_analysis)))
    ext2 = json.loads(args.ext2.read_text())
    out: dict[str, Any] = {
        "kind": "study5-refdir-analysis",
        "version": 1,
        "inputs": {k: sha256(getattr(args, k)) for k in ("final", "e2", "ext2", "llama_analysis")},
    }
    models: dict[str, Any] = {}
    for name, folder, direction, seeds in (
        ("qwen", args.qwen, args.qwen_direction, QWEN),
        ("llama", args.llama, args.llama_direction, LLAMA),
    ):
        found = json.loads((direction / "direction.json").read_text())
        if found.get("chosen") is None:
            models[name] = {"direction": "unvalidated"}  # no projection analysis, no fallback
            continue
        unit, control = units_of(direction)
        res = model_record(
            name,
            load_answers(folder, seeds),
            unit,
            control,
            design(name, harm, seeds),
            harm,
            ext2 if name == "qwen" else None,
        )
        res["direction"] = {k: found["chosen"][k] for k in ("layer", "position")}
        res["direction_files"] = {
            "direction.json": sha256(direction / "direction.json"),
            "vectors.pt": sha256(direction / "vectors.pt"),
        }
        models[name] = res
    out["models"] = models

    def p_of(model: str, test: str) -> float | None:
        return models.get(model, {}).get(test, {}).get("p")

    out["family"] = sr.holm(
        {
            "T1": p_of("qwen", "stream_test"),
            "T2": p_of("llama", "stream_test"),
            "T3": p_of("qwen", "order_test"),
            "T4": p_of("llama", "order_test"),
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in (
        "qwen",
        "qwen-direction",
        "llama",
        "llama-direction",
        "final",
        "e2",
        "ext2",
        "llama-analysis",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"{args.out.name} exists: never overwritten")
        return 1
    try:
        record = record_of(args)
    except Stop as e:
        print(f"STOP {e}")
        return 2
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True, default=float) + "\n")
    for k, v in record["family"].items():
        print(f"{k} p {v['p']} adjusted {v['adjusted']} {v['reading'].replace(' ', '_')}")
    print(f"record sha256 {sha256(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
