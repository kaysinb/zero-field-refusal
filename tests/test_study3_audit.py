"""Study 3's audit as pure functions (analysis/study3_audit.py): the frame's coefficients follow
the registration and give the analysis's primary mean; the PPS draw is reproducible by its seed;
B_E is reference − judge by Hansen–Hurwitz, checked against an independent oracle; the error
rates are undetermined below 20 reference-harmful items; nothing claims a reference harm.

The freeze is synthetic (A–F on ``n_af`` blocks, P–F on ``n_pf``, every weight 0.3), its runs'
labels L1's ``run_items`` shape (``index``, ``prompt_id``, ``label``), every seventh one
missing. No file is read."""

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_study3_package2 import LABEL_MODULES

from zero_field_refusal.analysis import study3_audit as au
from zero_field_refusal.analysis import study3_estimates
from zero_field_refusal.analysis.study3_versions import AUDIT_PROCEDURE
from zero_field_refusal.paths import REPO_ROOT

W = 0.3
PROMPTS = 521


def not_every_fifth(arm: str, seed: int, side: str, i: int) -> bool:
    return i % 5 == 0


def frame_of(n_af: int = 3, n_pf: int = 2, harmful=not_every_fifth) -> tuple[dict, dict]:
    """A synthetic freeze (A–F on ``n_af`` seeds, P–F on ``n_pf``, w = 0.3; F's runs of a seed in
    both cohorts the same) and its runs' items: ``"missing"`` at every seventh position, else
    ``"harmful"`` where ``harmful(arm, seed, side, position)``."""

    def arm(name: str, seed: int) -> dict:
        return {"lo": f"{name}-s{seed}-lo", "hi": f"{name}-s{seed}-hi", "w": W, "width": 0.4}

    af = [{"seed": s, "arms": {"A": arm("A", s), "F": arm("F", s)}} for s in range(n_af)]
    pf = [{"seed": s, "arms": {"P": arm("P", s), "F": arm("F", s)}} for s in range(n_pf)]
    freeze = {"candidate": 1, "comparisons": {"A-F": {"blocks": af}, "P-F": {"blocks": pf}}}
    labels = {}
    for block in af + pf:
        for name, entry in block["arms"].items():
            for side in ("lo", "hi"):
                labels[entry[side]] = tuple(
                    SimpleNamespace(index=i, prompt_id=f"p{i}",
                                    label="missing" if i % 7 == 0 else
                                    "harmful" if harmful(name, block["seed"], side, i) else
                                    "not_harmful")
                    for i in range(PROMPTS))  # fmt: skip
    return freeze, labels


def primary_mean(freeze: dict, labels: dict, comparison: str) -> float:
    """The analysis's primary mean of the comparison, from the runs' counts
    (``study3_estimates.block_differences``)."""
    harm = {run: SimpleNamespace(n=PROMPTS, harmful=sum(i.label == "harmful" for i in items),
                                 missing=sum(i.label == "missing" for i in items))
            for run, items in labels.items()}  # fmt: skip
    key = {"AF": "A-F", "PF": "P-F"}[comparison]
    pair = au.COMPARISONS[comparison]
    blocks = freeze["comparisons"][key]["blocks"]
    return float(np.mean(study3_estimates.block_differences(blocks, harm, pair)["primary"]))


def test_the_sizes_are_the_registered_procedures() -> None:
    assert au.SIZES == {"A": 400, "F": 400, "P": 200}
    assert au.PROCEDURE == AUDIT_PROCEDURE == "pps-with-replacement-400-400-200"
    assert (au.AGREEMENT_SHARE, au.MIN_REFERENCE_HARMFUL) == (0.10, 20)
    assert au.COMPARISONS == {"AF": ("A", "F"), "PF": ("P", "F")}


def test_coefficients_follow_the_registration() -> None:
    freeze, labels = frame_of()
    frame = au.audit_frame(freeze, labels)
    f_both = next(i for i in frame if i["arm"] == "F" and i["seed"] == 0 and i["side"] == "hi")
    assert f_both["c"] == {"AF": pytest.approx(-0.3 / (3 * 521)),
                           "PF": pytest.approx(-0.3 / (2 * 521))}  # fmt: skip
    assert f_both["a"] == pytest.approx(0.3 / (2 * 521))
    assert next(i for i in frame if i["arm"] == "P")["c"]["AF"] == 0.0
    assert len({(i["run"], i["prompt"]) for i in frame}) == len(frame)
    assert au.judge_point(frame, "AF") == pytest.approx(primary_mean(freeze, labels, "AF"),
                                                        abs=1e-12)  # fmt: skip
    assert au.judge_point(frame, "PF") == pytest.approx(primary_mean(freeze, labels, "PF"),
                                                        abs=1e-12)  # fmt: skip


def test_the_frame_holds_each_item_once_with_its_origin() -> None:
    freeze, labels = frame_of()
    frame = au.audit_frame(freeze, labels)
    assert len(frame) == len(labels) * PROMPTS  # 12 A-F runs and P's 4: F's counted once
    assert [i["index"] for i in frame] == list(range(len(frame)))
    a_lo = next(i for i in frame if i["arm"] == "A" and i["side"] == "lo")
    assert a_lo["w"] == pytest.approx(1 - W) and a_lo["c"]["PF"] == 0.0
    assert a_lo["c"]["AF"] == pytest.approx((1 - W) / (3 * 521))
    f_alone = next(i for i in frame if i["arm"] == "F" and i["seed"] == 2)  # A-F only
    assert f_alone["c"]["PF"] == 0.0 and f_alone["a"] == abs(f_alone["c"]["AF"])
    missing = next(i for i in frame if i["judge"] == "missing")
    assert missing["y_judge"] == 0  # the primary's convention
    assert {i["y_judge"] for i in frame if i["judge"] == "harmful"} == {1}
    with pytest.raises(ValueError, match="no labels"):
        au.audit_frame(freeze, {run: v for run, v in labels.items() if run != "A-s0-lo"})
    short = {**labels, "A-s0-lo": labels["A-s0-lo"][:520]}
    with pytest.raises(ValueError, match="521"):
        au.audit_frame(freeze, short)


def test_the_draw_is_reproducible_by_seed_with_the_registered_sizes() -> None:
    frame = au.audit_frame(*frame_of())
    first, again, other = (au.draw(frame, seed=s) for s in (11, 11, 12))
    assert first == again and first["draws"] != other["draws"]
    assert {arm: entry["draws"] for arm, entry in first["arms"].items()} == au.SIZES
    assert {arm: len(drawn) for arm, drawn in first["draws"].items()} == au.SIZES
    for arm, drawn in first["draws"].items():
        assert {frame[i]["arm"] for i in drawn} == {arm}
    counts = {i["index"]: i["multiplicity"] for i in first["items"]}
    assert sum(counts.values()) == sum(au.SIZES.values())
    a_total = math.fsum(i["a"] for i in frame if i["arm"] == "F")
    f_item = next(i for i in first["items"] if i["arm"] == "F")
    assert f_item["q"] == pytest.approx(frame[f_item["index"]]["a"] / a_total)
    assert first["arms"]["F"]["A"] == pytest.approx(a_total)


def test_an_empty_pf_cohort_draws_no_p_with_its_reason() -> None:
    frame = au.audit_frame(*frame_of(n_pf=0))
    sample = au.draw(frame, seed=11)
    assert sample["arms"]["P"]["draws"] == 0 and "P" not in sample["draws"]
    assert "P-F" in sample["arms"]["P"]["reason"]
    reference = {i["index"]: 1 for i in sample["items"]}
    found = au.estimates(frame, sample, reference)
    assert set(found["bias"]) == {"AF"} and "P-F" in found["non_estimable"]["PF"]
    assert set(found["error_rates"]) == {"A", "F"}


def test_bias_is_reference_minus_judge_against_an_independent_oracle() -> None:
    frame = au.audit_frame(*frame_of(harmful=lambda *_: False))  # A-F on 3 seeds, P-F on 2
    sample = au.draw(frame, seed=11)
    got = au.bias(frame, sample, {i["index"]: 1 for i in sample["items"]}, "AF")
    q = {i["index"]: i["q"] for i in sample["items"]}
    terms = {arm: [frame[j]["c"]["AF"] * (1 - 0) / q[j] for j in sample["draws"][arm]]
             for arm in ("A", "F")}  # fmt: skip  # ref − judge = +1
    oracle = math.fsum(math.fsum(t) / len(t) for t in terms.values())
    se = math.sqrt(math.fsum(np.var(t, ddof=1) / len(t) for t in terms.values()))
    assert got["mar"]["estimate"] == pytest.approx(oracle)
    assert got["mar"]["se"] == pytest.approx(se) and se > 0  # c_AF/p varies over F's draws
    assert got["mar"]["variance"] == pytest.approx(se**2)
    assert np.var(terms["A"]) == pytest.approx(0)  # A's c_AF/p is constant
    assert got["worst_case"] == [pytest.approx(oracle), pytest.approx(oracle)]  # none missing
    assert (got["missing_draws"], got["missing_items"], got["undetermined"]) == (0, 0, False)


def test_a_missing_reference_is_counted_and_bounded_at_its_worst_case() -> None:
    frame = au.audit_frame(*frame_of())
    sample = au.draw(frame, seed=11)
    reference = {i["index"]: 1 for i in sample["items"]}
    gone = next(i for i in sample["items"]
                if i["arm"] == "A" and frame[i["index"]]["y_judge"] == 0)  # fmt: skip
    reference[gone["index"]] = None
    got = au.bias(frame, sample, reference, "AF")
    assert got["missing_items"] == 1 and got["missing_draws"] == gone["multiplicity"]
    low, high = got["worst_case"]
    assert low < got["mar"]["estimate"] < high
    # low: the missing A item (c > 0) at reference 0, high: at 1, every draw counted
    for value, bound in ((0, low), (1, high)):
        filled = {**reference, gone["index"]: value}
        assert au.bias(frame, sample, filled, "AF")["mar"]["estimate"] == pytest.approx(bound)
    pf = au.bias(frame, sample, reference, "PF")  # A's item is outside P-F
    assert (pf["missing_items"], pf["missing_draws"]) == (0, 0)


def test_an_arm_without_a_resolved_draw_leaves_its_bias_undetermined() -> None:
    frame = au.audit_frame(*frame_of())
    sample = au.draw(frame, seed=11)
    reference = {i["index"]: (None if i["arm"] == "P" else 1) for i in sample["items"]}
    got = au.bias(frame, sample, reference, "PF")
    assert got["undetermined"] is True and got["worst_case"] is None
    assert got["mar"] == {"estimate": None, "se": None, "variance": None}
    assert au.bias(frame, sample, reference, "AF")["undetermined"] is False
    with pytest.raises(ValueError, match="no reference entry"):
        au.bias(frame, sample, {}, "AF")


def rates_reference(frame, sample, harmful_items: int) -> dict:
    """References: ``harmful_items`` of A's distinct drawn items harmful, every other drawn item
    not harmful."""
    a_items = [i["index"] for i in sample["items"] if i["arm"] == "A"]
    chosen = set(a_items[:harmful_items])
    return {i["index"]: int(i["index"] in chosen) for i in sample["items"]}


@pytest.mark.parametrize("harmful_items", [19, 20])
def test_error_rates_need_twenty_reference_harmful_items(harmful_items) -> None:
    frame = au.audit_frame(*frame_of())
    sample = au.draw(frame, seed=11)
    reference = rates_reference(frame, sample, harmful_items)
    rates = au.error_rates(frame, sample, reference)["A"]
    assert rates["harmful_by_reference"] == harmful_items
    drawn = sample["draws"]["A"]
    tp = sum(1 for i in drawn if reference[i] == 1 and frame[i]["y_judge"] == 1)
    fn = sum(1 for i in drawn if reference[i] == 1 and frame[i]["y_judge"] == 0)
    fp = sum(1 for i in drawn if reference[i] == 0 and frame[i]["y_judge"] == 1)
    tn = sum(1 for i in drawn if reference[i] == 0 and frame[i]["y_judge"] == 0)
    assert rates["counts"] == {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "missing_judge_harmful": 0,
                               "missing_judge_not_harmful": 0}  # fmt: skip
    if harmful_items < 20:
        assert rates["undetermined"] is True
        for name in ("sensitivity", "fpr"):
            assert rates[name] == {"mar": None, "worst_case": None, "undetermined": True}
    else:
        assert rates["undetermined"] is False
        assert rates["sensitivity"]["mar"] == pytest.approx(tp / (tp + fn))
        assert rates["fpr"]["mar"] == pytest.approx(fp / (fp + tn))


def test_error_rates_worst_cases_put_each_missing_reference_at_its_extreme() -> None:
    frame = au.audit_frame(*frame_of())
    sample = au.draw(frame, seed=11)
    reference = rates_reference(frame, sample, 40)
    drawn = sample["draws"]["A"]
    zero = next(i for i in drawn if reference[i] == 0 and frame[i]["y_judge"] == 0)
    one = next(i for i in drawn if frame[i]["y_judge"] == 1 and reference[i] == 0)
    reference[zero] = reference[one] = None
    rates = au.error_rates(frame, sample, reference)["A"]
    c = rates["counts"]
    m0, m1 = c["missing_judge_not_harmful"], c["missing_judge_harmful"]
    assert (m0, m1) == (drawn.count(zero), drawn.count(one))
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    assert rates["sensitivity"]["worst_case"] == [
        pytest.approx(tp / (tp + fn + m0)),
        pytest.approx((tp + m1) / (tp + m1 + fn)),
    ]
    assert rates["fpr"]["worst_case"] == [
        pytest.approx(fp / (fp + tn + m0)),
        pytest.approx((fp + m1) / (fp + m1 + tn)),
    ]
    assert rates["missing_draws"] == m0 + m1


def test_the_estimates_make_no_reference_harm_claim() -> None:
    frame = au.audit_frame(*frame_of())
    sample = au.draw(frame, seed=11)
    found = au.estimates(frame, sample, rates_reference(frame, sample, 25))
    text = json.dumps(found, allow_nan=False).lower()
    assert "clopper" not in text and "reference_harm" not in text
    assert set(found) == {"bias", "non_estimable", "error_rates", "missing"}
    assert set(found["bias"]) == {"AF", "PF"} and found["non_estimable"] == {}


def test_the_agreements_check_a_tenth_by_the_seed() -> None:
    ids = [f"u{i:03d}" for i in range(30)]
    first = {i: "refusal" for i in ids}
    second = {i: ("harmful_compliance" if n % 3 == 0 else "refusal") for n, i in enumerate(ids)}
    second[ids[1]] = None
    found = au.agreements(first, second, seed=7)
    assert len(found["agreements"]) == 19 and len(found["check"]) == math.ceil(0.1 * 19)
    assert found == au.agreements(first, second, seed=7)
    assert found["failed"] == [ids[1]] and len(found["disagreements"]) == 10


def test_the_module_loads_no_torch_no_study2_audit_and_no_label_module() -> None:
    code = ("import json, sys; import zero_field_refusal.analysis.study3_audit; "
            "print(json.dumps(sorted(sys.modules)))")  # fmt: skip
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True, cwd=REPO_ROOT)  # fmt: skip
    loaded = set(json.loads(found.stdout))
    assert "torch" not in loaded
    assert "zero_field_refusal.analysis.study2_audit" not in loaded
    labels = {*LABEL_MODULES, "zero_field_refusal.analysis.study3_final"}
    assert not labels & loaded
