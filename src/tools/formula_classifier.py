"""Classify engineering questions into registered calculator formulas."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from src.llm_client import generate, is_eee_related
from src.tools.eecc_calculators import FORMULA_REGISTRY

DEFAULT_MODEL = os.getenv("EEE_RAG_MODEL", "qwen3:1.7b")
DEFAULT_TIMEOUT = int(os.getenv("EEE_CLASSIFIER_TIMEOUT", "60"))

_CLASSIFICATION_PROMPT = """You are an EEE formula classifier. Given an engineering question, pick the SINGLE best formula to solve it. Return ONLY a JSON object with two keys: "formula_name" and "confidence".

Question: {question}

Available formulas:
{formula_list}

Rules for selecting:
1. Identify the CONVERTER TOPOLOGY: "forward converter" / "two-transistor forward" → forward_* formulas. "flyback" or "isolated flyback" → flyback_* formulas. "buck-boost" or "non-isolated flyback" → non_isolated_flyback_*. "DC machine" / "DC motor" / "DC generator" / "armature" → dc_machine_induced_emf.
2. Identify the TARGET QUANTITY: turns ratio N1/N2 → *_turns_ratio_n1_n2. turns ratio N2/N1 → *_turns_ratio_n2_n1. inductance or L → forward_inductance. transistor current or peak current → forward_transistor_current. duty cycle or D → non_isolated_flyback_duty_cycle. EMF or induced voltage → dc_machine_induced_emf.
3. If the question asks for a turns ratio but doesn't specify N1/N2 vs N2/N1, you MUST infer the direction from the voltages: Vin > Vout (step-down) means choose N1/N2; Vin < Vout (step-up) means choose N2/N1. For example, a flyback converting 24 V to 240 V MUST use flyback_ccm_turns_ratio_n2_n1.
4. If no formula matches, set formula_name to "unknown" and confidence to 0.

Return format: {{"formula_name": "<exact_name_from_list>", "confidence": <0.0_to_1.0>}}
Return ONLY the JSON. No explanation.
"""


def available_formula_names() -> list[str]:
    """Return all formula names currently in FORMULA_REGISTRY."""
    return list(FORMULA_REGISTRY)


def _build_formula_list() -> str:
    """Build a human-readable formula list from FORMULA_REGISTRY for the prompt."""
    lines: list[str] = []
    for name, spec in FORMULA_REGISTRY.items():
        lines.append(f"  - {name}: {spec.description} (quantity={spec.quantity}, unit={spec.unit})")
    return "\n".join(lines)


def _strip_code_fence(raw: str) -> str:
    """Remove an optional Markdown code fence around a model response."""
    stripped = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.I)
    return match.group(1).strip() if match else stripped


def _extract_json_object(raw: str) -> str:
    """Extract the first complete JSON object from a model response."""
    stripped = _strip_code_fence(raw)
    start = stripped.find("{")
    if start == -1:
        return stripped
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(stripped[start:], start=start):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
        elif not in_string:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start : index + 1]
    return stripped


def _classify_by_keywords(question: str) -> str | None:
    """Classify unambiguous registered formula requests without an LLM call."""
    normalized = question.casefold().replace("₂", "2").replace("₁", "1")
    asks_turns_ratio = "turns ratio" in normalized or "turn ratio" in normalized

    if any(term in normalized for term in ("dc machine", "dc shunt", "armature")) and any(
        term in normalized for term in ("emf", "induced", "generator", "motor")
    ):
        return "dc_machine_induced_emf"

    if (
        "non-isolated flyback" in normalized or "non isolated flyback" in normalized
    ) and "duty cycle" in normalized:
        return "non_isolated_flyback_duty_cycle"

    if "forward converter" in normalized or "same converter" in normalized:
        if asks_turns_ratio:
            return "forward_turns_ratio"
        if "inductance" in normalized or "inductor ripple" in normalized:
            return "forward_inductance"
        if "transistor current" in normalized or "transistor currents" in normalized:
            return "forward_transistor_current"

    if "flyback" in normalized and asks_turns_ratio:
        if "n2/n1" in normalized or "secondary-to-primary" in normalized:
            return "flyback_ccm_turns_ratio_n2_n1"
        if "n1/n2" in normalized or "primary-to-secondary" in normalized:
            return "flyback_ccm_turns_ratio_n1_n2"
        input_match = re.search(r"(?:vin\s*=?|from)\s*(\d+(?:\.\d+)?)\s*v", normalized)
        output_match = re.search(
            r"(?:produces|provides|supplies|to)\s*(\d+(?:\.\d+)?)\s*v",
            normalized,
        )
        if input_match and output_match:
            input_voltage = float(input_match.group(1))
            output_voltage = float(output_match.group(1))
            return (
                "flyback_ccm_turns_ratio_n2_n1"
                if output_voltage > input_voltage
                else "flyback_ccm_turns_ratio_n1_n2"
            )
        return "flyback_ccm_turns_ratio_n1_n2"
    return None


def _parse_classification_response(raw: str) -> tuple[str, float]:
    """Parse and validate a formula-classification JSON response."""
    try:
        payload: Any = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("Formula classifier returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Formula classifier response must be a JSON object")

    formula_name = payload.get("formula_name")
    if formula_name == "unknown":
        raise ValueError("No registered formula matches the question")
    if not isinstance(formula_name, str) or formula_name not in FORMULA_REGISTRY:
        raise ValueError(f"Formula classifier returned unknown formula: {formula_name!r}")

    confidence_value = payload.get("confidence", 0.0)
    if isinstance(confidence_value, bool):
        confidence = 0.0
    else:
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return formula_name, confidence


def classify_formula(
    question: str,
    *,
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """Use qwen3 to select the best formula for an engineering question.

    Returns a formula-name key from ``FORMULA_REGISTRY``.
    Raises ``ValueError`` if classification fails or the formula is not registered.

    The default timeout can be overridden via the ``EEE_CLASSIFIER_TIMEOUT`` env var.
    """
    effective_model = model or DEFAULT_MODEL
    effective_timeout = timeout or DEFAULT_TIMEOUT
    if not is_eee_related(question):
        raise ValueError("Question is not related to a registered EEE formula")
    deterministic_result = _classify_by_keywords(question)
    if deterministic_result is not None:
        return deterministic_result
    prompt = _CLASSIFICATION_PROMPT.format(
        question=question,
        formula_list=_build_formula_list(),
    )
    response = generate(
        prompt,
        model=effective_model,
        timeout=effective_timeout,
        system_prompt=None,
        think=False,
        num_predict=96,
    )
    answer = response.get("answer")
    if not isinstance(answer, str):
        raise ValueError("Formula classifier returned no text answer")
    formula_name, _confidence = _parse_classification_response(answer)
    return formula_name
