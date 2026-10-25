"""Study 3: the numerical qualification's verdicts, from the part records alone.

The rules are docs/study3-preregistration.md, "Numerical qualification" ("Qualification"),
"Failure routes", "Voiding and supersession" and "Amendment 2" (the same-path control). The
records are those ``qualification.run_part`` writes; nothing here runs a model, and
``qualification`` re-exports every public name. The module belongs to the analysis package, so it
is outside the GPU-code identity (``code_identity``): a change of a verdict rule moves no part's
GPU code, and the records' GPU-code check stays on the identity of the code the parts ran.

**A case** (``case_verdict``): the 13 comparisons at step 25 (``COMPARISONS``), each
``{"D", "within"}`` against 0.01·K*, its state, kind and reasons.

- Scientific: an instability of the launcher run (its declared stop) or of an unshifted harness
  trajectory, the baseline or the repeat (``instability``: the launcher's declared rules applied
  after the run: a non-finite loss, gradient or weight, the ×10 gradient rule, a monitor K above
  its maximum); a non-finite loss, gradient or model in the reference or a shift (only that: the
  ×10 rule and the monitor's bound are not theirs); ``baseline~reference`` or ``baseline~sK``
  beyond the bound.
- Technical: ``launcher~baseline``, ``launcher~repeat`` or ``baseline~repeat`` beyond the bound;
  a launcher comparison beyond the bound whose baseline counterpart is not; a shift lost to fp32
  rounding after its retry; a non-finite D between finite models; a failed bridge check; v₀
  written; any exception or missing part; parts or a launcher run on more than one Pod (their
  recorded pod ids, where present: one Pod per case). A ``rest`` record that is not bound to the
  baseline given (another case, launcher run, settings or GPU code, or not built on that baseline
  record and checkpoint) is technical and none of its measurements is read.
- A case with both kinds, on bound records, is scientific (no rerun passes it), its technical
  reasons kept. States: ``failing``, ``not_qualified``, ``qualified`` (all 13 within), else
  ``incomplete``. ``warning`` when ``baseline~repeat`` exceeds τ = K*·max(0.001, 3·e_eng): a
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
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .. import fa_control_diagnostic as fcd

if TYPE_CHECKING:
    from ..qualification import QualificationScope

BATCH, STEPS = 20, 25  # "Protocol": B = 20 and exactly 25 updates
SCIENTIFIC, TECHNICAL = "scientific", "technical"
QUALIFIED, FAILING, INCOMPLETE = "qualified", "failing", "incomplete"  # a case, a rate
NOT_QUALIFIED = "not_qualified"  # technical
ACCEPTED = "accepted"  # a same-path control
SHIFTS = {"s1": "p1", "s2": "p2", "s3": "s3", "s4": "s4"}  # a shift and its direction's key
UNSHIFTED = ("baseline", "repeat")  # the trajectories the declared instability rules apply to
REST = ("repeat", "reference", *SHIFTS)  # the rest part's trajectories, in order
COMPARISONS = ("launcher~baseline",  # the 13 comparisons at step 25
               *(f"{t}~{name}" for name in REST for t in ("launcher", "baseline")))  # fmt: skip
NOT_THIS_RATE = "the control is not this rate's"
FAMILIES = ("launcher~baseline", "launcher~repeat", "baseline~repeat", "launcher~reference",
            "baseline~reference", "launcher~shift", "baseline~shift")  # fmt: skip


class QualificationError(ValueError):
    """The qualification cannot run as registered (a scope or a request outside it)."""


def record_sha256(record: Mapping[str, Any] | None) -> str | None:
    """The sha256 of a record file as ``qualification.publish`` writes it (so of the file read
    back, too); None for no record."""
    if record is None:
        return None
    return hashlib.sha256((json.dumps(record, indent=1) + "\n").encode()).hexdigest()


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
    pods = _pod_problem(baseline, rest)
    if pods is not None:
        technical.append(pods)
    case = head.get("case") or {}
    try:
        bound = scope.bound(case.get("candidate"))
    except QualificationError as exc:
        bound = None
        technical.append(str(exc))
    ended = False  # the baseline decided the case: the rest is not needed
    if baseline is None:
        technical.append("the baseline part has no record")
    elif baseline.get("part") != "baseline":
        technical.append("the baseline record is another part's")
        baseline = None
    else:
        if baseline.get("status") != "complete":  # a failed baseline's trajectory still counts
            _part_reasons(baseline, technical, science)
            ended = True
        run = (baseline.get("trajectories") or {}).get("baseline")
        if run is not None and _unstable("baseline", run, science):
            ended = True
        if baseline.get("status") == "complete":
            failed = [k for k, v in (baseline.get("bridge") or {}).items() if v is not True]
            if failed or not baseline.get("bridge"):
                technical.append(f"the bridge failed: {failed or 'no checks'}")
            if baseline.get("v0_unchanged") is not True:
                technical.append("v₀ changed during the baseline part")
            first = (baseline.get("comparisons") or {}).get("launcher~baseline") or {}
            measured["launcher~baseline"] = first.get("D_twin")
    if not ended:
        if rest is None:
            technical.append("the rest part has no record")
        else:
            _rest_verdict(baseline, rest, measured, technical, science)
    comparisons = {name: {"D": measured[name], "within": _within(measured[name], bound)}
                   for name in COMPARISONS}  # fmt: skip
    for name, c in comparisons.items():
        target, trajectory = name.split("~")
        if c["D"] is not None and not fcd._number(c["D"]):
            technical.append(f"{name}: a non-finite D between finite models ({c['D']})")
        elif c["within"] is not False:
            continue
        elif trajectory in UNSHIFTED:
            technical.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}")
        elif target == "baseline":
            science.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}")
        elif comparisons[f"baseline~{trajectory}"]["within"] is not False:
            technical.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}, its "
                             "baseline counterpart not")  # fmt: skip
    complete = all(c["within"] is not None for c in comparisons.values())
    kind = SCIENTIFIC if science else TECHNICAL if technical else None
    within = all(c["within"] for c in comparisons.values())
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
            "complete": complete, "comparisons": comparisons, "bound": bound, "tau": tau,
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
    baseline: Mapping[str, Any] | None, rest: Mapping[str, Any], measured: dict[str, Any],
    technical: list[str], science: list[str],
) -> None:  # fmt: skip
    """The rest record's part of ``case_verdict``: nothing of it is read unless it is bound."""
    problem = _binding_problem(baseline, rest)
    if problem is not None:
        technical.append(problem)
        return
    if rest.get("status") == "not_run":
        _part_reasons(rest, technical, science)
        return
    if rest.get("status") != "complete":  # its trajectories before the failure still count
        _part_reasons(rest, technical, science)
    else:
        failed = [k for k, v in (rest.get("bridge") or {}).items() if v is not True]
        if failed or not rest.get("bridge"):
            technical.append(f"the bridge failed in the rest part: {failed or 'no checks'}")
        if rest.get("v0_unchanged") is not True:
            technical.append("v₀ changed during the rest part")
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
        elif run.get("stop") is not None:  # the reference and the shifts: non-finite values only
            science.append(f"a non-finite loss, gradient or model in the {name}: {run['stop']}")
        if name in SHIFTS and "survived" in run and run["survived"] is not True:
            technical.append(f"{name} was lost to fp32 rounding after its retry")
    for name, value in (rest.get("comparisons") or {}).items():
        if name in measured and name != "launcher~baseline":
            measured[name] = (value or {}).get("D_twin")


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
    out: dict[str, Any] = {"state": NOT_QUALIFIED, "kind": TECHNICAL, "reasons": [],
                           "K_scoring": None, "bound": bound, "candidate": candidate,
                           "case": None if record is None else record.get("case"),
                           "launcher": _identity(record), "gpu_code": code, "settings": settings,
                           "record_sha256": record_sha256(record)}  # fmt: skip
    reasons = out["reasons"]
    case = (record or {}).get("case") or {}
    if record is None:
        reasons.append("no same-path control record")
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
        pods = _pod_problem(record)
        if pods is not None:
            reasons.append(pods)
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
        not_counted.append({"case": name, "record_sha256": key, "why": why,
                            "kind": verdict.get("kind") if same else None})  # fmt: skip
        reasons.append(f"{name}: not counted and not voided: {why}")
    for verdict in counted:
        reasons += [f"{verdict['case']['name']}: {r}" for r in verdict.get("reasons") or []]
    found = [verdict["case"]["seed"] for verdict in counted]
    kinds = {verdict.get("kind") for verdict in counted}
    stale_science = any(entry["kind"] == SCIENTIFIC for entry in not_counted)
    checked = None
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
                stale_science = True
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
    return {"candidate": candidate, "optimizer": optimizer, "lr": lr, "state": state,
            "seeds": seeds, "counted": sorted(found), "not_counted": not_counted,
            "voided": void, "reasons": reasons, "control": checked,
            "warnings": [v["case"]["name"] for v in counted if v.get("warning")],
            "fallbacks": [v["case"]["name"] for v in counted if v.get("fallback")]}  # fmt: skip


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
