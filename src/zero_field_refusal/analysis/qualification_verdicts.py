"""Study 3: the numerical qualification's verdicts, from the part records alone.

The rules are docs/study3-preregistration.md, "Numerical qualification" ("Qualification"),
"Failure routes", "Voiding and supersession" and "Amendment 2" (the same-path control). The
records are those ``qualification.run_part`` writes; nothing here runs a model, and
``qualification`` re-exports the verdict functions. The module belongs to the analysis package,
so it is outside the GPU-code identity (``code_identity``): a change of a verdict rule moves no
part's GPU code, and the records' GPU-code check stays on the identity of the code the parts ran.
It holds functions of the records only: every constant that decides what a part executes or
what its record says (the batch and steps, the shifts, the trajectories, the comparisons, the
kinds and states, ``record_sha256``) is ``qualification_constants``', inside the identity.

**A case** (``case_verdict``): the 13 comparisons at step 25 (``COMPARISONS``), each
``{"D", "within"}`` against 0.01·K*, its state, kind and reasons.

- Scientific: an instability of the launcher run (its declared stop) or of an unshifted harness
  trajectory, the baseline or the repeat (``instability``: the launcher's declared rules applied
  after the run: a non-finite loss, gradient or weight, the ×10 gradient rule, a monitor K above
  its maximum); a non-finite loss, gradient or model in the reference (only that: the ×10 rule
  and the monitor's bound are not its); ``baseline~reference`` beyond the bound.
- **The four shifts are descriptive** (amendment 8): their comparisons are measured and shown, each
  with ``within``, and rejects nothing, neither does a shift's stop or its loss to fp32 rounding. A
  rest record made under the old rule, ``decided`` on a shift, keeps its scientific reason.
- Technical: ``launcher~baseline``, ``launcher~repeat``, ``baseline~repeat`` or
  ``launcher~reference`` beyond the bound (the last when its baseline counterpart is not); a shift
  that did not run; a non-finite D between finite models; a failed bridge check; v₀
  written; any exception or missing part; parts or a launcher run on more than one Pod (their
  recorded pod ids, where present: one Pod per case). A ``rest`` record that is not bound to the
  baseline given (another case, launcher run, settings or GPU code, or not built on that baseline
  record and checkpoint) is technical and none of its measurements is read.
- **Identity first** (``identity_problems``): the baseline record, one Pod, every part's bridge
  checks (every registered one, ``BRIDGE_KEYS``, present and true; none missing from a complete
  part), v₀ unchanged and the rest's binding to the baseline hold before any measurement is read. A
  failure holds back the measurements of the record it belongs to: one of the baseline makes the
  case technical, nothing read and no scientific reason, whatever the records show; one of the rest
  (unbound, its own checks, a second Pod) makes it technical too, unless the bound, verified
  baseline decided the case: that scientific failure stands, the rest's problem is listed and
  nothing of the rest is read. A part that failed before establishing its bridge checks has no
  measurement read either.
- A case with both kinds, on bound and verified records, is scientific (no rerun passes it), its
  technical reasons kept. States: ``failing``, ``not_qualified``, ``qualified`` (the 13 measured,
  a stopped shift's excepted, and all but the shifts' within), else ``incomplete``. ``warning`` when ``baseline~repeat`` exceeds τ = K*·max(0.001, 3·e_eng): a
  prompt for diagnosis only. A case is named by ``record_sha256``, over its part records.

**The same-path control** (``control_verdict``): accepted with finite records and K_scoring at
step 25 at most 0.01·K*; missing or failed (another Pod than its launcher run's included),
``not_qualified`` (technical); its launcher run stopped by instability, ``failing``.

**A rate** (``rate_verdict``): every case of this candidate, arm, rate and numerical seed is shown.
One counts under the expected GPU code and settings, at the protocol's batch and steps, with a
launcher run identity; one that does not is listed with why, and unless ``voided`` (its
``record_sha256`` → the reason of its supersession) it keeps the rate from ``qualified`` and, if
scientific, makes it ``failing``. A case of another rate, arm, candidate or seed blocks
``qualified`` too; so does a superseded control record of the rate (scientific: ``failing``)
unless voided. A control record of another rate or seed is technical here, never this rate's
scientific evidence. ``failing`` > ``not_qualified`` (a technical case, an FA rate's control not
accepted, an unresolved record) > ``qualified`` (exactly the prescribed seeds, one case each, all
qualified, with the checkpoint, every seed's launcher run located when they are checked) >
``incomplete``. The FA control is held to the same GPU code, settings and launcher run as the
first seed's case. ``engineering_summary``: e_eng, the path maxima, the margins.

**Eligibility** (amendment 10). A rate's ``eligibility`` is its state, except for a ``failing``
rate: its state with the scientific reasons set aside (``_eligibility``). A reference beyond the
bound or non-finite and a declared instability stop no longer reject a rate by themselves, and its
technical reasons keep their route: ``qualified`` (the route admits the rate to validation),
``not_qualified`` or ``incomplete``. The ``state`` stays as the rules above give it. A case's
``technical_reasons`` are the technical part of its ``reasons``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .. import fa_control_diagnostic as fcd
from ..qualification_constants import (
    ACCEPTED,
    BRIDGE_KEYS,
    COMPARISONS,
    FAILING,
    FAMILIES,
    INCOMPLETE,
    NOT_QUALIFIED,
    QUALIFIED,
    REST,
    SCIENTIFIC,
    SHIFTS,
    TECHNICAL,
    UNSHIFTED,
    QualificationError,
    record_sha256,
)
from ..qualification_constants import BATCH as BATCH
from ..qualification_constants import STEPS as STEPS

if TYPE_CHECKING:
    from ..qualification import QualificationScope

NOT_THIS_RATE = "the control is not this rate's"
REST_MISSING = "the rest part has no record"


def _within(d: Any, bound: float | None) -> bool | None:
    return d <= bound if fcd._number(d) and bound is not None else None


def _policy(record: Mapping[str, Any]) -> tuple[Any, Any]:
    """A part record's settings and GPU code (its sha256)."""
    provenance = record.get("provenance") or {}
    return provenance.get("settings"), (provenance.get("gpu_code") or {}).get("sha256")


def _identity(record: Mapping[str, Any] | None) -> Any:
    return ((record or {}).get("launcher") or {}).get("identity")


def _pod_problem(*records: Mapping[str, Any] | None) -> str | None:
    """One Pod per case: the launcher run's (``launcher.pod_id``) and each part's
    (``provenance.pod_id``), where present, must be one; why not, or None."""
    pods = {}
    for record in records:
        if record is None:
            continue
        part = record.get("part")
        launcher = (record.get("launcher") or {}).get("pod_id")
        own = (record.get("provenance") or {}).get("pod_id")
        if launcher is not None:
            pods[f"the launcher run (as the {part} read it)"] = launcher
        if own is not None:
            pods[f"the {part}"] = own
    if len(set(pods.values())) > 1:
        return f"the case ran on more than one Pod: {pods}"
    return None


def _bridge_failures(record: Mapping[str, Any]) -> list[str] | None:
    """The registered bridge checks (``BRIDGE_KEYS``) of a part record that are not true, a
    missing one included (``"<key> (missing)"``); None when it holds no bridge at all."""
    bridge = record.get("bridge")
    if not isinstance(bridge, Mapping) or not bridge:
        return None
    return [key if key in bridge else f"{key} (missing)" for key in BRIDGE_KEYS
            if bridge.get(key) is not True]  # fmt: skip


def _verified(record: Mapping[str, Any] | None) -> bool:
    """Whether a part record's measurements may be read: it holds every registered bridge
    check, each true (a part that failed before establishing them, or a bridge missing one, has
    measurements no check bound)."""
    return record is not None and _bridge_failures(record) == []


def _record_problems(record: Mapping[str, Any], label: str) -> list[str]:
    """A part record's own bridge checks (all true, none missing from a complete part) and v₀."""
    problems = []
    failed = _bridge_failures(record)
    if failed:
        problems.append(f"{label}: {failed}")
    elif failed is None and record.get("status") == "complete":
        problems.append(f"{label}: no checks")
    unchanged = record.get("v0_unchanged")
    if unchanged is False or (record.get("status") == "complete" and unchanged is not True):
        problems.append(f"v₀ changed during the {record.get('part')} part")
    return problems


def _gates(
    baseline: Mapping[str, Any] | None, rest: Mapping[str, Any] | None
) -> tuple[list[str], list[str]]:
    """``identity_problems`` split by the record whose measurements they hold back: the
    baseline's (its record present and the baseline's, one Pod for it and its launcher run, its
    bridge checks and v₀), and the rest's (its bridge checks and v₀, one Pod with the baseline
    and the launcher run, its binding to the baseline)."""
    base: list[str] = []
    if baseline is None:
        base.append("the baseline part has no record")
    elif baseline.get("part") != "baseline":
        base.append("the baseline record is another part's")
    else:
        pods = _pod_problem(baseline)
        if pods is not None:
            base.append(pods)
        base += _record_problems(baseline, "the bridge failed")
    other: list[str] = []
    if rest is not None:
        pods = _pod_problem(baseline, rest)
        if pods is not None and not base:
            other.append(pods)
        other += _record_problems(rest, "the bridge failed in the rest part")
        problem = _binding_problem(baseline, rest)
        if problem is not None:
            other.append(problem)
    return base, other


def identity_problems(
    baseline: Mapping[str, Any] | None, rest: Mapping[str, Any] | None
) -> list[str]:
    """The identity and bridge checks the measurements of a case wait on (module docstring):
    the baseline record present and the baseline's; one Pod for the launcher run and the parts;
    each part's bridge checks (the steps, tokens and denominators, the training stream, v₀'s,
    θ₀'s and the names' hashes, the optimizer, the numerics setup), all true, and none missing
    from a complete part; v₀ unchanged; a rest bound to this baseline (the case, the launcher run
    and checkpoint, the settings and GPU code, the baseline record and checkpoint). The launcher
    run's own identity, config agreement, numerics, code and environment are checked before
    the part runs (``qualification.launcher_problems``), its record then holding nothing
    measured. A problem holds back the measurements of the record it belongs to: one of the
    baseline makes the case technical, nothing read; one of the rest makes it technical unless a
    bound baseline decided the case (scientific), and nothing of the rest is read either way."""
    base, other = _gates(baseline, rest)
    return base + other


def _part_reasons(record: Mapping[str, Any], technical: list[str], science: list[str]) -> None:
    """A part that ran nothing or failed: its kind and reasons."""
    reasons = [f"{record.get('part')}: {r}" for r in record.get("reasons") or []]
    (science if record.get("kind") == SCIENTIFIC else technical).extend(reasons)


def _unstable(name: str, run: Mapping[str, Any], science: list[str]) -> bool:
    """An unshifted trajectory's instability, declared by the launcher's rules after the run
    (``qualification.harness_instability``), or its stop. Returns whether one fired."""
    fired = run.get("instability")
    if fired is not None:
        science.append(f"instability: the harness {name}: {fired.get('reason')!r} at step "
                       f"{fired.get('step')}, {json.dumps(fired.get('detail'))}")  # fmt: skip
        return True
    if run.get("stop") is not None:
        science.append(f"instability: the harness {name} stopped: {run['stop']}")
        return True
    return False


def case_verdict(
    baseline: Mapping[str, Any] | None,
    rest: Mapping[str, Any] | None,
    scope: QualificationScope,
    *,
    e_eng: float | None = None,
) -> dict[str, Any]:
    """A case's 13 comparisons at step 25, its state, kind and reasons (module docstring), from
    its baseline and rest records (None: no record). With ``e_eng``, ``warning`` when
    ``baseline~repeat`` exceeds τ. The case's identity, launcher run, GPU code and settings are
    kept for ``rate_verdict``."""
    science: list[str] = []
    technical: list[str] = []
    measured: dict[str, Any] = dict.fromkeys(COMPARISONS)
    head = baseline or rest or {}
    case = head.get("case") or {}
    try:
        bound = scope.bound(case.get("candidate"))
    except QualificationError as exc:
        bound = None
        technical.append(str(exc))
    gate, held = _gates(baseline, rest)  # first: nothing is read unless it holds
    technical += gate
    if not gate:
        ended = False  # the baseline decided the case: the rest is not needed
        if baseline.get("status") != "complete":  # a failed baseline's trajectory still counts
            _part_reasons(baseline, technical, science)  # when its bridge was established
            ended = True
        run = (baseline.get("trajectories") or {}).get("baseline")
        if run is not None and _verified(baseline) and _unstable("baseline", run, science):
            ended = True
        if baseline.get("status") == "complete":
            first = (baseline.get("comparisons") or {}).get("launcher~baseline") or {}
            measured["launcher~baseline"] = first.get("D_twin")
        # A problem of the rest record is listed and holds back its own measurements; it never
        # hides a scientific failure a bound baseline decided
        technical += held
        if held and not science:
            measured = dict.fromkeys(COMPARISONS)  # a technical case: nothing of it is read
        elif not ended and not held:
            if rest is None:
                technical.append(REST_MISSING)
            else:
                _rest_verdict(rest, measured, technical, science)
    comparisons = {name: {"D": measured[name], "within": _within(measured[name], bound)}
                   for name in COMPARISONS}  # fmt: skip
    for name, c in comparisons.items():
        target, trajectory = name.split("~")
        if c["D"] is not None and not fcd._number(c["D"]):
            technical.append(f"{name}: a non-finite D between finite models ({c['D']})")
        elif c["within"] is not False or trajectory in SHIFTS:  # a shift: descriptive, amendment 8
            continue
        elif trajectory in UNSHIFTED:
            technical.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}")
        elif target == "baseline":
            science.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}")
        elif comparisons[f"baseline~{trajectory}"]["within"] is not False:
            technical.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}, its "
                             "baseline counterpart not")  # fmt: skip
    stopped = {name for name, run in ((rest or {}).get("trajectories") or {}).items()
               if name in SHIFTS and run.get("stop") is not None}  # fmt: skip
    measured_all = [c["within"] is not None or name.split("~")[1] in stopped
                    for name, c in comparisons.items()]  # fmt: skip
    complete = all(measured_all)
    kind = SCIENTIFIC if science else TECHNICAL if technical else None
    within = complete and all(c["within"] for name, c in comparisons.items()
                              if name.split("~")[1] not in SHIFTS)  # fmt: skip
    state = (FAILING if science else NOT_QUALIFIED if technical
             else QUALIFIED if within else INCOMPLETE)  # fmt: skip
    tau = warning = None
    repeat = comparisons["baseline~repeat"]["D"]
    if e_eng is not None and bound is not None:
        tau = scope.warning(e_eng, case["candidate"])
        warning = repeat > tau if fcd._number(repeat) else None
    settings, code = _policy(head)
    reference = ((rest or {}).get("trajectories") or {}).get("reference") or {}
    parts = {"baseline": record_sha256(baseline), "rest": record_sha256(rest)}
    return {"case": case, "state": state, "kind": kind, "reasons": science + technical,
            "technical_reasons": list(technical), "complete": complete,
            "comparisons": comparisons, "bound": bound, "tau": tau,
            "warning": warning, "reference_mode": reference.get("mode"),
            "fallback": reference.get("mode") == fcd.FLOAT64_ACCUMULATION,
            "launcher": _identity(head), "gpu_code": code, "settings": settings,
            "records": parts,
            "record_sha256": hashlib.sha256(json.dumps(parts).encode()).hexdigest()}  # fmt: skip


def _binding_problem(baseline: Mapping[str, Any] | None, rest: Mapping[str, Any]) -> str | None:
    """Why a rest record is not bound to the baseline record given, or None."""
    base = baseline or {}
    if rest.get("part") != "rest":
        return "the rest record is another part's"
    if baseline is None:
        return "the rest record has no baseline record to build on"
    if rest.get("case") != base.get("case"):
        return "the rest record is another case's"
    if _identity(rest) != _identity(base):
        return "the rest part ran against another launcher run or checkpoint"
    if _policy(rest) != _policy(base):
        return "the parts were written under other settings or GPU code"
    if rest.get("status") == "not_run":
        return None  # it ran nothing: its own reasons say why
    if ((rest.get("baseline_record") or {}).get("sha256") != record_sha256(base)
            or rest.get("baseline_checkpoint_sha256")
            != (base.get("checkpoint") or {}).get("sha256")):  # fmt: skip
        return "the rest part does not build on this baseline record and checkpoint"
    return None


def _rest_verdict(
    rest: Mapping[str, Any], measured: dict[str, Any], technical: list[str], science: list[str]
) -> None:
    """The rest record's part of ``case_verdict``, once ``identity_problems`` holds: its
    trajectories are read only when its bridge checks were established (all true)."""
    if rest.get("status") != "complete":  # its trajectories before the failure still count
        _part_reasons(rest, technical, science)
    if rest.get("status") == "not_run" or not _verified(rest):
        return
    trajectories = rest.get("trajectories") or {}
    for name in REST:
        run = trajectories.get(name)
        if run is None:
            if rest.get("status") == "complete":
                technical.append(f"the rest part has no {name}")
            continue
        if run.get("not_run"):
            technical.append(f"{name}: {run.get('stop')}")
        elif name in UNSHIFTED:
            _unstable(name, run, science)
        elif name not in SHIFTS and run.get("stop") is not None:  # the reference: non-finite only
            science.append(f"a non-finite loss, gradient or model in the {name}: {run['stop']}")
    for name, value in (rest.get("comparisons") or {}).items():
        if name in measured and name != "launcher~baseline":
            measured[name] = (value or {}).get("D_twin")


TRAJECTORY_BINDING = ("part", "attempt", "case", "launcher", "baseline_record",
                      "baseline_checkpoint_sha256", "launcher_checkpoint_sha256",
                      "numerics_setup", "K_star", "bound")  # fmt: skip


def rest_from_trajectories(results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """An interrupted rest attempt as a record (None without any result), from the trajectory
    results it published before the interruption (``qualification``, "Trajectory results"):
    the trajectories done and their comparisons, the bridge checks and v₀'s state as of the last
    one, bound to the inputs the results name; ``status: "interrupted"``. ``case_verdict``
    reads it as any rest record: what the results show counts, a scientific failure included.
    Results bound to different inputs make a record without bridge checks, technical, of which
    nothing is read."""
    if not results:
        return None
    first = results[0]
    record: dict[str, Any] = {key: first.get(key) for key in TRAJECTORY_BINDING}
    record.update(status="interrupted", kind=None, reasons=[], provenance=first.get("provenance"),
                  trajectories={}, comparisons={},
                  from_trajectories=[r.get("trajectory") for r in results])  # fmt: skip
    if any({k: r.get(k) for k in TRAJECTORY_BINDING} != {k: first.get(k)
                                                         for k in TRAJECTORY_BINDING}
           or _policy(r) != _policy(first) or r.get("trajectory") not in REST
           for r in results):  # fmt: skip
        record.update(
            kind=TECHNICAL, reasons=["the trajectory results are bound to different inputs"]
        )
        return record
    ordered = sorted(results, key=lambda r: REST.index(r["trajectory"]))
    record["from_trajectories"] = [r["trajectory"] for r in ordered]
    for result in ordered:
        record["trajectories"][result["trajectory"]] = result.get("run")
        record["comparisons"].update(result.get("comparisons") or {})
    record.update(bridge=ordered[-1].get("bridge"), v0_unchanged=ordered[-1].get("v0_unchanged"))
    return record


def control_verdict(
    record: Mapping[str, Any] | None, scope: QualificationScope, candidate: int
) -> dict[str, Any]:
    """Amendment 2: the same-path control is ``accepted`` when its records are finite (no
    divergence, every loss, gradient norm and K finite) and its K_scoring at step 25 is at most
    0.01·K*; a missing or failed control (exceeding, non-finite, incomplete, its streams or v₀ not
    the launcher run's, an error) is ``not_qualified`` (technical); one whose launcher run an
    instability rule stopped is ``failing`` (scientific). Its launcher run, GPU code and settings
    are kept for ``rate_verdict``."""
    bound = scope.control_bound(candidate)
    settings, code = _policy(record or {})
    pods = None if record is None else _pod_problem(record)
    out: dict[str, Any] = {"state": NOT_QUALIFIED, "kind": TECHNICAL, "reasons": [],
                           "K_scoring": None, "bound": bound, "candidate": candidate,
                           "case": None if record is None else record.get("case"),
                           "launcher": _identity(record), "gpu_code": code, "settings": settings,
                           "record_sha256": record_sha256(record)}  # fmt: skip
    reasons = out["reasons"]
    case = (record or {}).get("case") or {}
    if record is None:
        reasons.append("no same-path control record")
    elif pods is not None:  # before anything it measured or its kind: technical
        reasons.append(pods)
    elif record.get("part") != "control" or case.get("candidate") != candidate:
        reasons.append("the record is not this candidate's same-path control")
    elif record.get("status") != "complete":
        reasons += record.get("reasons") or [f"the control did not run ({record.get('status')})"]
        if record.get("kind") == SCIENTIFIC:
            out.update(state=FAILING, kind=SCIENTIFIC)
    else:
        control = record.get("control") or {}
        state = fcd.control_state(control, k_at=[scope.control_steps], bound=bound)
        k = {r.get("step"): r.get("K_scoring") for r in control.get("scoring") or []}.get(
            scope.control_steps
        )
        out["K_scoring"] = k
        unequal = [key for key, c in (record.get("shared_state") or {}).items() if not c["equal"]]
        if unequal or not record.get("shared_state"):
            reasons.append(f"the control's {unequal or 'shared state'} differ from the "
                           "launcher run's")  # fmt: skip
        if state["diverged"]:
            reasons.append("a non-finite record in the control")
        elif not state["complete"] or not fcd._number(k):
            reasons.append(f"the control is incomplete: {control.get('stop')}")
        elif k > bound:
            reasons.append(f"K_scoring {k:.6g} at step {scope.control_steps} beyond {bound:.6g}")
        if not reasons:
            out.update(state=ACCEPTED, kind=None)
    return out


def _same_rate(
    verdict: Mapping[str, Any], *, candidate: int, optimizer: str, lr: float, seeds: Sequence[int]
) -> bool:
    """Whether a case verdict is of this candidate, arm and rate, on one of its numerical seeds."""
    case = verdict.get("case") or {}
    rate = case.get("lr")
    return (case.get("candidate") == candidate and case.get("optimizer") == optimizer
            and fcd._number(rate) and math.isclose(rate, lr, rel_tol=1e-9)
            and case.get("engineering") is False and case.get("seed") in seeds)  # fmt: skip


def _not_counted(
    verdict: Mapping[str, Any], *, batch: int, steps: int, code: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None, launchers: Mapping[int, Mapping[str, Any]] | None,
) -> str | None:  # fmt: skip
    """Why a case of this rate is not counted under the expected policy, or None."""
    case = verdict.get("case") or {}
    if case.get("batch") != batch or case.get("steps") != steps:
        return "not the protocol's batch and steps"
    if code is None or verdict.get("gpu_code") != code.get("sha256"):
        return "other GPU code"
    if policy is None or verdict.get("settings") != policy:
        return "other settings"
    launcher = verdict.get("launcher") or {}
    if not launcher.get("run_id"):
        return "no launcher run identity"
    expected = None if launchers is None else launchers.get(case.get("seed"))
    if expected is not None and expected != launcher:  # an unlocated run is checked by none
        return "another launcher run or checkpoint"
    return None


def _control_problem(
    checked: Mapping[str, Any], *, candidate: int, lr: float, first_seed: int,
    code: Mapping[str, Any] | None, policy: Mapping[str, Any] | None, first_case: Any,
) -> str | None:  # fmt: skip
    """Why a same-path control verdict is not this rate's under the expected policy, or None."""
    case = checked.get("case") or {}
    rate = case.get("lr")
    if (checked.get("candidate") != candidate or case.get("seed") != first_seed
            or not (fcd._number(rate) and math.isclose(rate, lr, rel_tol=1e-9))):  # fmt: skip
        return NOT_THIS_RATE
    if code is None or checked.get("gpu_code") != code.get("sha256"):
        return "the control was written under other GPU code"
    if policy is None or checked.get("settings") != policy:
        return "the control was written under other settings"
    if first_case is not None and checked.get("launcher") != first_case.get("launcher"):
        return "the control read another launcher run than the first seed's case"
    return None


def rate_verdict(
    cases: Sequence[Mapping[str, Any]],
    *,
    scope: QualificationScope,
    candidate: int,
    optimizer: str,
    lr: float,
    code: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None,
    control: Mapping[str, Any] | None = None,
    superseded_controls: Sequence[Mapping[str, Any]] = (),
    launchers: Mapping[int, Mapping[str, Any] | None] | None = None,
    voided: Mapping[str, str] | None = None,
    batch: int = BATCH,
    steps: int = STEPS,
) -> dict[str, Any]:
    """An (arm, rate)'s verdict (module docstring) from its case verdicts (``case_verdict``)
    and, for an FA rate, its same-path control's (``control_verdict``; None: missing) and those
    of its superseded control records (``superseded_controls``). ``code`` is the expected GPU
    code (its ``sha256``), ``policy`` the expected settings (``qualification.settings_record``),
    ``launchers`` the launcher run located for each seed when given (None, or a seed left out:
    not located, and the rate is then at most ``incomplete``), and ``voided`` the superseded
    records (``record_sha256`` → reason)."""
    seeds = list(scope.seeds(candidate))
    voided = dict(voided or {})
    unlocated = [] if launchers is None else [s for s in seeds if launchers.get(s) is None]
    counted, not_counted, void, reasons = [], [], [], []
    for verdict in cases:
        name = (verdict.get("case") or {}).get("name")
        key = verdict.get("record_sha256")
        if key in voided:
            void.append({"case": name, "record_sha256": key, "reason": voided[key]})
            continue
        if _same_rate(verdict, candidate=candidate, optimizer=optimizer, lr=lr, seeds=seeds):
            why = _not_counted(verdict, batch=batch, steps=steps, code=code, policy=policy,
                               launchers=launchers)  # fmt: skip
        else:
            why = "another candidate, arm, rate or seed"
        if why is None:
            counted.append(verdict)
            continue
        same = why != "another candidate, arm, rate or seed"
        technical_reasons = verdict.get("technical_reasons")
        if technical_reasons is None and verdict.get("resolution") is not None:
            technical_reasons = []  # a resolved declared stop (``resolved_verdict``): no other
        entry = {"case": name, "record_sha256": key, "why": why,
                 "kind": verdict.get("kind") if same else None}  # fmt: skip
        if same:  # what amendment 10 reads of it (``_eligibility``)
            entry |= {"seed": (verdict.get("case") or {}).get("seed"),
                      "technical_reasons": technical_reasons}  # fmt: skip
        not_counted.append(entry)
        reasons.append(f"{name}: not counted and not voided: {why}")
    for verdict in counted:
        reasons += [f"{verdict['case']['name']}: {r}" for r in verdict.get("reasons") or []]
    found = [verdict["case"]["seed"] for verdict in counted]
    kinds = {verdict.get("kind") for verdict in counted}
    stale_science = any(entry["kind"] == SCIENTIFIC for entry in not_counted)
    checked, stale_control = None, False
    if optimizer == scope.control_arm:
        checked = dict(control or control_verdict(None, scope, candidate))
        if checked.get("record_sha256") is not None and checked["record_sha256"] in voided:
            void.append({"case": "control", "record_sha256": checked["record_sha256"],
                         "reason": voided[checked["record_sha256"]]})  # fmt: skip
            checked = control_verdict(None, scope, candidate)
        elif checked.get("record_sha256") is not None:
            first = next((v for v in counted if v["case"]["seed"] == seeds[0]), None)
            problem = _control_problem(checked, candidate=candidate, lr=lr, first_seed=seeds[0],
                                       code=code, policy=policy, first_case=first)  # fmt: skip
            if problem not in (None, NOT_THIS_RATE) and checked.get("kind") == SCIENTIFIC:
                # this rate's scientific control record, under other code or settings or of
                # another launcher run, holds until it is voided; another rate's never is this
                # rate's scientific evidence
                stale_science = stale_control = True
                checked.setdefault("reasons", []).append(problem)
            elif problem is not None:
                checked.update(state=NOT_QUALIFIED, kind=TECHNICAL,
                               reasons=[*checked.get("reasons", []), problem])  # fmt: skip
        reasons += [f"control: {r}" for r in checked.get("reasons") or []]
    for old in superseded_controls:  # a superseded control counts as a superseded case does
        key = old.get("record_sha256")
        if key in voided:
            void.append({"case": "control", "record_sha256": key, "reason": voided[key]})
            continue
        problem = _control_problem(old, candidate=candidate, lr=lr, first_seed=seeds[0],
                                   code=None, policy=None, first_case=None)  # fmt: skip
        mine = problem != NOT_THIS_RATE
        not_counted.append({"case": "control", "record_sha256": key,
                            "why": "a superseded same-path control" + ("" if mine else
                                                                     f": {NOT_THIS_RATE}"),
                            "kind": old.get("kind") if mine else None})  # fmt: skip
        reasons.append("control: a superseded record, not counted and not voided")
    stale_science = stale_science or any(e["kind"] == SCIENTIFIC for e in not_counted)
    missing = [s for s in seeds if s not in found]
    duplicated = sorted({s for s in found if found.count(s) > 1})
    if (SCIENTIFIC in kinds or stale_science
            or (checked is not None and checked.get("kind") == SCIENTIFIC)):  # fmt: skip
        state = FAILING
    elif (TECHNICAL in kinds or not_counted
          or (checked is not None and checked.get("state") != ACCEPTED)):  # fmt: skip
        state = NOT_QUALIFIED
    elif (not missing and not duplicated and not unlocated and len(found) == len(seeds)
          and all(v.get("state") == QUALIFIED and v.get("complete")
                  and ((v.get("launcher") or {}).get("checkpoint") or {}).get("sha256")
                  for v in counted)):  # fmt: skip
        state = QUALIFIED
    else:
        state = INCOMPLETE
        reasons += [f"no counted case of seed {s}" for s in missing]
        reasons += [f"more than one case of seed {s}" for s in duplicated]
        reasons += [f"the launcher run of seed {s} is not located" for s in unlocated]
    eligibility = state if state != FAILING else _eligibility(
        counted, not_counted, checked, seeds=seeds, unlocated=unlocated,
        stale_control=stale_control)  # fmt: skip
    return {"candidate": candidate, "optimizer": optimizer, "lr": lr, "state": state,
            "eligibility": eligibility,
            "seeds": seeds, "counted": sorted(found), "not_counted": not_counted,
            "voided": void, "reasons": reasons, "control": checked,
            "warnings": [v["case"]["name"] for v in counted if v.get("warning")],
            "fallbacks": [v["case"]["name"] for v in counted if v.get("fallback")]}  # fmt: skip


def _eligibility(
    counted: Sequence[Mapping[str, Any]], not_counted: Sequence[Mapping[str, Any]],
    checked: Mapping[str, Any] | None, *, seeds: Sequence[int], unlocated: Sequence[int],
    stale_control: bool,
) -> str:  # fmt: skip
    """A ``failing`` rate's state under amendment 10, its scientific reasons set aside: a
    scientific case with no technical reason (its reference beyond the bound or non-finite, a
    declared instability stop) and a control whose launcher run an instability rule stopped
    count as passed. So does such a case of this rate not counted only for its other GPU code
    (a case an engineering cycle keeps whole, or a resolved declared stop): it stands for its
    seed. ``not_qualified`` for any technical case or reason, any other record not counted and
    not voided, a control neither accepted nor stopped, or this rate's stopped control under
    other code, settings or launcher run (``stale_control``); ``incomplete`` for a missing,
    doubled or unlocated seed or a case neither passed nor so admitted; else ``qualified``: the
    rate goes to validation, its verdict ``failing`` as recorded."""
    kept = [e for e in not_counted
            if e.get("kind") == SCIENTIFIC and e.get("why") == "other GPU code"
            and e.get("technical_reasons") == [] and e.get("seed") in seeds]  # fmt: skip
    if (len(kept) < len(not_counted) or stale_control
            or any(v.get("kind") == TECHNICAL or (v.get("kind") == SCIENTIFIC
                                                  and v.get("technical_reasons") != [])
                   for v in counted)
            or (checked is not None
                and checked.get("state") not in (ACCEPTED, FAILING))):  # fmt: skip
        return NOT_QUALIFIED
    found = [v["case"]["seed"] for v in counted] + [e["seed"] for e in kept]
    complete = not unlocated and sorted(found) == sorted(seeds)
    if complete and all(v.get("kind") == SCIENTIFIC or (
            v.get("state") == QUALIFIED and v.get("complete")
            and ((v.get("launcher") or {}).get("checkpoint") or {}).get("sha256"))
            for v in counted):  # fmt: skip
        return QUALIFIED
    return INCOMPLETE


def engineering_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The engineering stage's report from part records (baseline and rest): e_eng, the largest
    D₂₅/K* over the unshifted repeats (``baseline~repeat``); the path maxima, the largest
    D_t/K* at the diagnostic steps per trajectory (shifts pooled); and per comparison family the
    margin, the smallest bound/D₂₅ over the records (∞ for a D of 0). Diagnostics: no verdict."""
    repeats: list[float] = []
    paths: dict[str, list[float]] = {}
    ratios: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for record in records:
        k_star, bound = record.get("K_star"), record.get("bound")
        if not (fcd._number(k_star) and fcd._number(bound)):
            continue
        for name, value in (record.get("comparisons") or {}).items():
            d = (value or {}).get("D_twin")
            target, trajectory = name.split("~")
            family = f"{target}~{'shift' if trajectory in SHIFTS else trajectory}"
            if family not in ratios or d is None:
                continue
            ratios[family].append(bound / d if fcd._number(d) and d > 0 else
                                  math.inf if d == 0 else 0.0)  # fmt: skip
            if name == "baseline~repeat":
                repeats.append(d / k_star if fcd._number(d) else math.inf)
        if record.get("part") == "rest":
            for name, run in (record.get("trajectories") or {}).items():
                values = [r.get("D_twin") for r in run.get("D_twin") or []]
                if values:
                    family = "shift" if name in SHIFTS else name
                    paths.setdefault(family, []).append(fcd._largest(values) / k_star)
    return {"e_eng": max(repeats) if repeats else None, "repeats": len(repeats),
            "path_maxima": {family: max(v) for family, v in paths.items()},
            "margins": {family: min(v) if v else None for family, v in ratios.items()}}  # fmt: skip
