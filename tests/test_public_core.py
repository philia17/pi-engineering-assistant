"""Fresh public fixtures for calculators and pre-submission API contracts."""

from __future__ import annotations

import json
import math

import pytest

from examples.offline_demo import build_demo
from rag.agent_contract import AgentRequest, AgentResponse, normalize_citations
from rag.quiz_contract import PrivateQuizQuestion, QuizOption, QuizSource
from rag.quiz_properties import PROPERTY_MAPPINGS
from rag.runtime_automation import _load_runtime_modules, automation_runtime_available
from src.tools.eecc_calculators import dispatch_formula
from src.tools.em_calculators import calculate_phasor_add
from src.tools.pe_calculators import (
    calculate_ideal_transformer_turns_ratio,
    calculate_rc_time_constant,
    calculate_single_phase_complex_power,
)


def test_demo_numerical_results() -> None:
    result = build_demo()
    assert result["shared_heatsink"] == {"total_power_w": 18.0, "theta_sa_c_per_w": 4.5}
    assert result["ideal_transformer"]["turns_ratio"] == 4.0
    assert result["rc_circuit"]["time_constant_s"] == pytest.approx(0.1551)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("turns", [(600.0, 150.0), (75.0, 225.0), (13.0, 13.0)])
def test_transformer_voltage_and_current_ratios_are_reciprocal(
    turns: tuple[float, float],
) -> None:
    result = calculate_ideal_transformer_turns_ratio(
        primary_turns=turns[0], secondary_turns=turns[1]
    )
    assert result.values["turns_ratio"] == pytest.approx(turns[0] / turns[1])
    assert result.values["turns_ratio"] * result.values["current_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("invalid", [0.0, -4.0])
def test_transformer_rejects_nonpositive_turns(invalid: float) -> None:
    with pytest.raises(ValueError):
        calculate_ideal_transformer_turns_ratio(primary_turns=11.0, secondary_turns=invalid)


@pytest.mark.parametrize("resistance,capacitance", [(470.0, 22e-6), (6800.0, 1e-9)])
def test_rc_time_constant(resistance: float, capacitance: float) -> None:
    result = calculate_rc_time_constant(resistance_ohm=resistance, capacitance_f=capacitance)
    assert result.values["time_constant_s"] == pytest.approx(resistance * capacitance)
    assert result.values["fraction_remaining_one_tau"] == pytest.approx(math.exp(-1))


@pytest.mark.parametrize("phase", [-42.0, 0.0, 42.0])
def test_complex_power_preserves_apparent_power(phase: float) -> None:
    result = calculate_single_phase_complex_power(voltage_v=120.0, current_a=2.5, phase_deg=phase)
    p = result.values["real_power_w"]
    q = result.values["reactive_power_var"]
    assert math.hypot(p, q) == pytest.approx(300.0)
    assert q == pytest.approx(300.0 * math.sin(math.radians(phase)))


def test_forward_converter_accepts_explicit_parameters() -> None:
    result = dispatch_formula(
        "forward_turns_ratio", input_voltage_v=96.0, output_voltage_v=18.0, duty_cycle=0.375
    )
    assert result.value == pytest.approx(2.0)  # primary-to-secondary turns ratio


def test_phasor_cancellation() -> None:
    result = calculate_phasor_add(magnitudes=[8.0, 8.0], phases_deg=[0.0, 180.0])
    assert result.values["amplitude_v"] == pytest.approx(0.0)


def test_private_quiz_fields_are_not_serialized() -> None:
    question = PrivateQuizQuestion(
        question_id="generic-addition",
        number=1,
        focus_id="generic-arithmetic",
        prompt="What is 7 + 6?",
        options=tuple(
            QuizOption(option_id, text)
            for option_id, text in zip("ABCD", ["11", "12", "13", "14"], strict=True)
        ),
        correct_option_id="C",
        explanation="Adding six to seven gives thirteen.",
        sources=(QuizSource("examples/generic_arithmetic.md", "Addition"),),
    )
    public = question.public_dict()
    assert "correct_option_id" not in public
    assert "explanation" not in public
    assert "sources" not in public
    assert len(public["options"]) == 4


def test_request_and_response_contract() -> None:
    assert AgentRequest("  generic input  ").cleaned_question() == "generic input"
    payload = AgentResponse(
        status="needs_input", mode="v1", answer="Supply a resistance.", subject="PE"
    ).as_dict()
    assert payload["status"] == "needs_input"
    assert "calculator_values" not in payload
    json.dumps(payload)


def test_duplicate_citations_are_normalized() -> None:
    citations = normalize_citations(["examples/generic.md", "examples/generic.md", " "])
    assert citations == [{"note_path": "examples/generic.md", "heading": None}]


def test_excluded_research_automation_is_unavailable() -> None:
    assert not automation_runtime_available()
    assert PROPERTY_MAPPINGS == ()


def test_unrelated_import_failure_is_not_hidden() -> None:
    def missing_dependency(name: str) -> object:
        del name
        raise ModuleNotFoundError("dependency unavailable", name="unrelated_dependency")

    with pytest.raises(ModuleNotFoundError):
        _load_runtime_modules(missing_dependency)
