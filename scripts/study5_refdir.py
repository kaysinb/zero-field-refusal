"""Study 5's refusal direction at θ₀ and the training answers' projections on it
(docs/study5-refdir-registration.md; ``zero_field_refusal.refdir``).

    python scripts/study5_refdir.py prompts --out docs/study5-refdir-prompts.json
    python scripts/study5_refdir.py direction --model qwen|llama --prompts FILE --out DIR [--bench]
    python scripts/study5_refdir.py project --model qwen|llama --direction DIR --out DIR

``prompts`` (CPU) writes the committed record of the prompt ids: the 100 refusal-drift prompts
(``harmful_kl``) and the 100 held-out Alpaca instructions (``alpaca_heldout``), each split 60/20/20
into direction, selection and confirmation prompts by the permutation
``numpy.random.default_rng(derive_seed(0, *SPLIT_KEYS)).permutation(100)`` (``SPLIT_KEYS`` =
"study5", "refusal-direction", "split"), with each text's sha256.

``direction`` loads θ₀ of the model under its runs' config, checks every prompt against the record,
reads the post-instruction tokens (the template's tail after the instruction, which every
encoded prompt must end with), takes the candidate vectors of layers 1 ≤ ℓ < 0.8 L, and applies
the registered selection rule on the selection prompts and the confirmation on the confirmation
prompts. With ``--bench`` it scores the baselines
and the first candidate layer only, times the projection of ``BENCH_ANSWERS`` answers of the
model's first seed, and writes the extrapolated total; nothing else. ``direction.json`` holds
numbers only; ``vectors.pt`` the chosen vector and the control vector (layer ⌊L/4⌋, the chosen
position). No candidate validated: ``chosen`` is null and nothing else is computed.

``admit`` (CPU) applies the time rule to both models' benches: the Pod's elapsed time since its
start (setup and the benches) plus the two models' estimates of the rest (direction, candidates,
confirmation, projection, each run's load) must not exceed 2.5 h; it exits 1 otherwise and the
Pod stops before the rest. The Pod's deadline (3 h after its start) is its hard cut.

``project`` reads each seed's training stream (the runs' ``StreamPrefix``: the archived token ids)
and writes per seed ``s<seed>.npz``: per answer its prompt id, whether it ended, its number of
answer positions and the mean block input at the chosen and the control layer, with the
projections p and p_tok (p × positions / 96) on both directions. stdout carries numbers, ids and
hashes only; no text is printed or stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zero_field_refusal import refdir, surrogate
from zero_field_refusal.models import encode_chat
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import StreamPrefix, configure_numerics, environment
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.tracking import git_state

SETS = ("harmful_kl", "alpaca_heldout")
PARTS = (("direction", 0, 60), ("selection", 60, 80), ("confirmation", 80, 100))
SPLIT_KEYS = ("study5", "refusal-direction", "split")
QWEN_SEEDS = tuple(range(50000, 50040))
LLAMA_SEEDS = tuple(range(60000, 60010))
MODELS = {
    "qwen": (("experiments/study5", "study5.role=fresh", "optimizer.lr=1.3e-05"), QWEN_SEEDS),
    "llama": (("experiments/study5_llama_theta0", "run_type=theta0"), LLAMA_SEEDS),
}
ANSWERS = 500
TOKEN_DENOMINATOR = 96
BENCH_ANSWERS = 40
ADMIT_SECONDS = 2.5 * 3600  # the registered admission limit; the Pod's own cap is 3 h


def run_config(model: str, seed: int):
    args, _ = MODELS[model]
    return load_run_config(args[0], [*args[1:], f"seed={seed}"])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path.name} exists: never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ prompts


def split_record(rows_by_set: dict[str, list[dict]]) -> dict[str, Any]:
    perm = np.random.default_rng(derive_seed(0, *SPLIT_KEYS)).permutation(100).tolist()
    sets = {}
    for name in SETS:
        rows = rows_by_set[name]
        if len(rows) != 100:
            raise ValueError(f"{name}: {len(rows)} prompts, not 100")
        parts = {part: [rows[i]["id"] for i in perm[a:b]] for part, a, b in PARTS}
        sets[name] = {
            "parts": parts,
            "text_sha256": {r["id"]: sha256_text(r["text"]) for r in rows},
        }
    return {
        "what": "The prompts of study 5's refusal direction (docs/study5-refdir-registration.md).",
        "rule": "numpy.random.default_rng(derive_seed(0, 'study5', 'refusal-direction', "
        "'split')).permutation(100); indices 0-59 direction, 60-79 selection, "
        "80-99 confirmation",
        "permutation": perm,
        "sets": sets,
    }


def cmd_prompts(args) -> int:
    from zero_field_refusal import data

    cfg = run_config("qwen", QWEN_SEEDS[0])
    rows = {name: data.load_prompt_set(name, cfg.data) for name in SETS}
    record = split_record(rows)
    record["prompt_dir"] = data.prompt_sets_dir(cfg.data).name
    write_new(args.out, record)
    print(f"prompts {args.out.name} sha256 {sha256_file(args.out)}")
    return 0


def texts_of(env, record: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    """Each set's parts as texts, every text checked against the record's sha256."""
    out = {}
    for name in SETS:
        by_id = {r["id"]: r["text"] for r in env.prompt_set(name)}
        expected = record["sets"][name]["text_sha256"]
        if {k: sha256_text(v) for k, v in by_id.items()} != expected:
            raise ValueError(f"{name}: the prompt set is not the record's")
        for part, ids in record["sets"][name]["parts"].items():
            out[(name, part)] = [by_id[i] for i in ids]
    return out


# ------------------------------------------------------------------ direction


def estimate_parts(
    load: float,
    means: float,
    baseline: float,
    per_candidate: float,
    candidates: int,
    per_answer: float,
    answers: int,
) -> dict[str, float]:
    """The model's whole run after its bench, in seconds: the direction run (a load, the
    difference of means, the baselines, every candidate, the confirmation: four mean-r passes on
    20 prompts each, twice the baselines' two) and the projection run (a load, every answer at the
    two layers). The bench's own time is not in it: the admission counts it as elapsed."""
    return {
        "direction_load": load,
        "direction_means": means,
        "baselines": baseline,
        "candidates": per_candidate * candidates,
        "confirmation": 2 * baseline,
        "projection_load": load,
        "projection": per_answer * answers,
    }


def setup(model_name: str, seed: int):
    cfg = run_config(model_name, seed)
    numerics = configure_numerics(cfg, cfg.execution.device)
    env = environment(cfg, cfg.execution.device)
    t = time.monotonic()
    model = env.new_model()
    return cfg, numerics, env, model, time.monotonic() - t


def cmd_direction(args) -> int:
    _, seeds = MODELS[args.model]
    if args.out.exists():  # before any model is loaded: a folder is written once
        raise FileExistsError(f"{args.out.name} exists: never overwritten")
    cfg, numerics, env, model, load_s = setup(args.model, seeds[0])
    record = json.loads(args.prompts.read_text(encoding="utf-8"))
    texts = texts_of(env, record)
    tok, device = env.tokenizer, env.device
    spec = surrogate.load_spec()
    encoded = {k: [encode_chat(tok, t) for t in v] for k, v in texts.items()}
    suffix = refdir.post_instruction_ids(tok, [ids for v in encoded.values() for ids in v])
    positions = len(suffix)
    layers = refdir.candidate_layers(model)
    control = refdir.num_layers(model) // 4
    t = time.monotonic()
    means_h = refdir.post_instruction_means(
        model, encoded[("harmful_kl", "direction")], layers, positions, device
    )
    means_s = refdir.post_instruction_means(
        model, encoded[("alpaca_heldout", "direction")], layers, positions, device
    )
    means_s_time = time.monotonic() - t
    vectors, excluded = refdir.candidates(means_h, means_s, layers)
    harmful = texts[("harmful_kl", "selection")]
    harmless = texts[("alpaca_heldout", "selection")]
    base = {
        "model": args.model,
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "chat_date": cfg.model.get("chat_date", None),
        "layers": layers,
        "num_layers": refdir.num_layers(model),
        "control_layer": control,
        "post_instruction_ids": suffix,
        "positions": positions,
        "excluded": excluded,
        "candidates": len(vectors),
        "surrogate_sha256": spec.sha256,
        "prompts_sha256": sha256_file(args.prompts),
        "numerics": numerics,
        "git": git_state(),
        "model_load_seconds": load_s,
        "means_seconds": means_s_time,
    }
    if args.bench:
        sel = refdir.select(
            model, tok, spec, vectors, harmful, harmless, device, layers=[layers[0]]
        )
        prefix = StreamPrefix(run_config(args.model, seeds[0]), env, seeds[0])
        samples = prefix.take(0, BENCH_ANSWERS)
        t = time.monotonic()
        refdir.answer_means(model, samples, [layers[0], control], device)
        per_answer = (time.monotonic() - t) / BENCH_ANSWERS
        per_candidate = math.fsum(r["seconds"] for r in sel["rows"]) / len(sel["rows"])
        parts = estimate_parts(
            load_s,
            means_s_time,
            sel["baseline_seconds"],
            per_candidate,
            len(vectors),
            per_answer,
            ANSWERS * len(seeds),
        )
        estimate = math.fsum(parts.values())
        bench = {
            **base,
            "baseline_seconds": sel["baseline_seconds"],
            "per_candidate_seconds": per_candidate,
            "per_answer_seconds": per_answer,
            "estimate_parts": parts,
            "estimate_seconds": estimate,
        }
        write_new(args.out / "bench.json", bench)
        print(
            f"bench {args.model} candidates {len(vectors)} per_candidate {per_candidate:.2f}s "
            f"per_answer {per_answer:.4f}s estimate {estimate / 3600:.3f} h"
        )
        return 0
    sel = refdir.select(model, tok, spec, vectors, harmful, harmless, device)
    out = {
        **base,
        **{
            k: sel[k]
            for k in ("base_harmful", "base_harmless", "baseline_seconds", "rows", "chosen")
        },
    }
    if sel["chosen"] is None:
        write_new(args.out / "direction.json", out)
        print(f"direction {args.model}: no candidate validated: unvalidated")
        return 0
    layer, pos = sel["chosen"]["layer"], sel["chosen"]["position"]
    chosen = vectors[(layer, pos)]
    a = layers.index(control)
    ctrl = means_h[a, pos] - means_s[a, pos]
    out["confirmation"] = refdir.confirm(
        model,
        tok,
        spec,
        chosen,
        layer,
        texts[("harmful_kl", "confirmation")],
        texts[("alpaca_heldout", "confirmation")],
        device,
    )
    out["norms"] = {"chosen": float(chosen.norm()), "control": float(ctrl.norm())}
    vec_path = args.out / "vectors.pt"
    if vec_path.exists():
        raise FileExistsError("vectors.pt exists")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "chosen": chosen.double(),
            "control": ctrl.double(),
            "layer": layer,
            "position": pos,
            "control_layer": control,
        },
        vec_path,
    )
    out["vectors_sha256"] = sha256_file(vec_path)
    write_new(args.out / "direction.json", out)
    bypass = next(r["bypass"] for r in sel["rows"] if (r["layer"], r["position"]) == (layer, pos))
    print(
        f"direction {args.model} chosen layer {layer} position {pos} bypass {bypass:.4f}"
        f" base_harmful {sel['base_harmful']:.4f} vectors sha256 {out['vectors_sha256']}"
    )
    return 0


# ------------------------------------------------------------------ projections


def binding_of(model: str, direction: Path, model_cfg: Any = None) -> dict[str, Any]:
    """What a projection file belongs to: the model, its revision, the direction record and its
    vectors (each checked against the other), the chosen layer and position and the control layer.
    ``ValueError`` when the folder does not hold one validated direction of ``model``, or, with
    ``model_cfg`` (the runs' model config), one of another model name or revision."""
    found = json.loads((direction / "direction.json").read_text(encoding="utf-8"))
    if found.get("model") != model:
        raise ValueError(f"direction.json is {found.get('model')!r}'s, not {model!r}'s")
    if model_cfg is not None and (found.get("model_name"), found.get("model_revision")) != (
        model_cfg.name,
        model_cfg.revision,
    ):
        raise ValueError("direction.json's model name or revision is not the runs' model's")
    if found.get("chosen") is None:
        raise ValueError("the direction is unvalidated")
    vec_path = direction / "vectors.pt"
    if sha256_file(vec_path) != found.get("vectors_sha256"):
        raise ValueError("vectors.pt is not the direction record's")
    vectors = torch.load(vec_path)
    chosen = {"layer": vectors["layer"], "position": vectors["position"]}
    if chosen != found["chosen"] or vectors["control_layer"] != found.get("control_layer"):
        raise ValueError("vectors.pt's layers are not the direction record's")
    return {
        "model": model,
        "model_name": found.get("model_name"),
        "model_revision": found.get("model_revision"),
        "direction_sha256": sha256_file(direction / "direction.json"),
        "vectors_sha256": found["vectors_sha256"],
        "layers": [vectors["layer"], vectors["control_layer"]],
        "position": vectors["position"],
    }


def kept(out: Path, seed: int, binding: dict[str, Any]) -> bool:
    """Whether the seed's projection file stands: absent (False), or present with a receipt of
    this binding and its sha256 (True); anything else raises ``ValueError``, a missing receipt
    after an interruption included."""
    path, receipt = out / f"s{seed}.npz", out / f"s{seed}.json"
    if not path.exists():
        return False
    if not receipt.exists():
        raise ValueError(f"s{seed}.npz has no receipt (move it aside and project again)")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if record.get("binding") != binding or record.get("npz_sha256") != sha256_file(path):
        raise ValueError(f"s{seed}.npz is not this direction's projection")
    return True


def cmd_project(args) -> int:
    _, seeds = MODELS[args.model]
    found = json.loads((args.direction / "direction.json").read_text(encoding="utf-8"))
    if found.get("chosen") is None:
        print(f"project {args.model}: the direction is unvalidated: no projection")
        return 1
    binding = binding_of(args.model, args.direction, run_config(args.model, seeds[0]).model)
    for seed in seeds:  # every existing file checked before the model is loaded
        kept(args.out, seed, binding)
    vectors = torch.load(args.direction / "vectors.pt")
    _, _, env, model, _ = setup(args.model, seeds[0])
    layers = binding["layers"]
    units = [
        vectors["chosen"] / vectors["chosen"].norm(),
        vectors["control"] / vectors["control"].norm(),
    ]
    for seed in seeds:
        path = args.out / f"s{seed}.npz"
        if kept(args.out, seed, binding):
            print(f"s{seed} exists: kept")
            continue
        prefix = StreamPrefix(run_config(args.model, seed), env, seed)
        samples = prefix.take(0, ANSWERS)
        means, counts = refdir.answer_means(model, samples, layers, env.device)
        m = means.double()
        p = (m[:, 0, :] @ units[0]).numpy()
        q = (m[:, 1, :] @ units[1]).numpy()
        n = np.asarray(counts, dtype=float)
        args.out.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial.npz")
        np.savez(
            tmp,
            means=means.numpy(),
            counts=np.asarray(counts),
            prompt_ids=np.asarray([s.prompt_id for s in samples]),
            ended=np.asarray([s.ended for s in samples]),
            p=p,
            p_tok=np.where(n > 0, p * n / TOKEN_DENOMINATOR, 0.0),
            p_control=q,
            p_control_tok=np.where(n > 0, q * n / TOKEN_DENOMINATOR, 0.0),
        )
        tmp.rename(path)
        identity = prefix.identity(ANSWERS)
        empty = int(np.sum(n == 0))
        receipt = {"seed": seed, "stream": identity, "npz_sha256": sha256_file(path),
                   "binding": binding, "empty_answers": empty}  # fmt: skip
        part = args.out / f"s{seed}.json.partial"
        part.write_text(json.dumps(receipt, sort_keys=True, default=str) + "\n", encoding="utf-8")
        part.rename(args.out / f"s{seed}.json")
        print(
            f"s{seed} answers {len(samples)} empty {empty} mean_p {float(np.nanmean(p)):.5f} "
            f"sha256 {sha256_file(path)[:16]}"
        )
    return 0


# ------------------------------------------------------------------ admission


def admission(benches: list[dict[str, Any]], started: float, now: float,
              limit: float = ADMIT_SECONDS) -> dict[str, Any]:  # fmt: skip
    """The time rule: the Pod's elapsed time since ``started`` (its setup and both benches) plus
    each model's estimate of the rest (``estimate_parts``) must not exceed ``limit``; both models'
    benches are needed."""
    models = sorted(b["model"] for b in benches)
    if models != sorted(MODELS):
        raise ValueError(f"the benches of {sorted(MODELS)} are needed, got {models}")
    elapsed = now - started
    if elapsed < 0:
        raise ValueError("the start lies in the future")
    rest = {b["model"]: math.fsum(b["estimate_parts"].values()) for b in benches}
    total = elapsed + math.fsum(rest.values())
    return {"elapsed_seconds": elapsed, "rest_seconds": rest, "total_seconds": total,
            "limit_seconds": limit, "admitted": total <= limit}  # fmt: skip


def cmd_admit(args) -> int:
    benches = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.bench]
    found = admission(benches, args.started, time.time())
    write_new(args.out, found)
    print(f"admit elapsed {found['elapsed_seconds'] / 3600:.3f} h rest "
          + " ".join(f"{k} {v / 3600:.3f} h" for k, v in sorted(found["rest_seconds"].items()))
          + f" total {found['total_seconds'] / 3600:.3f} h limit {ADMIT_SECONDS / 3600:.1f} h "
          f"admitted {found['admitted']}")  # fmt: skip
    return 0 if found["admitted"] else 1


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("direction")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bench", action="store_true")
    p = sub.add_parser("project")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--direction", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("admit")
    p.add_argument("--bench", type=Path, action="append", required=True)
    p.add_argument("--started", type=float, required=True)  # the Pod's start, Unix seconds
    p.add_argument("--out", type=Path, required=True)
    return parser


COMMANDS = {"prompts": cmd_prompts, "direction": cmd_direction, "project": cmd_project,
            "admit": cmd_admit}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        return COMMANDS[args.kind](args)
    except (ValueError, OSError) as exc:
        print(f"study5_refdir.py {args.kind}: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
