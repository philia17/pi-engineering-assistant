"""Extract calculator variables from natural-language engineering questions."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any

from src.llm_client import generate

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EEE_RAG_MODEL", "qwen3:1.7b")

EXTRACTION_PROMPT = """Extract numeric values from this engineering question. Return ONLY a valid JSON object. No explanation.

Question: {question}

Extract these variables:
{variable_list}

Rules:
- Return every requested variable as a JSON key, exactly once and using the exact key name.
- Match numbers to physical quantities by their units and labels; for example, a value followed by A is a current, not a voltage.
- If a variable is stated directly in the question, extract its numeric value.
- If a variable must be inferred from relationships (e.g. turns ratio), calculate it.
- Convert values to the units requested in each variable description (for example, kHz to Hz).
- If a variable is not mentioned at all, set its value to null.
- Do NOT include units in the values — return bare numbers only.
- Return ONLY the JSON object, nothing else.
"""

_PARAM_DESCRIPTIONS: dict[str, str] = {
    "input_voltage_v": "Input/source DC voltage in volts (Vin or Vs)",
    "output_voltage_v": "Output DC voltage in volts (Vout or Vo)",
    "duty_cycle": "Duty cycle D (0 to 1)",
    "switching_frequency_hz": "Switching frequency in Hz (fs)",
    "ripple_current_a": "Inductor current ripple in amps (delta_IL)",
    "secondary_to_primary_turns_ratio": (
        "Turns ratio N2/N1 (secondary to primary); infer it from the ideal "
        "forward-converter relation Vo/Vin = (N2/N1) * D when applicable"
    ),
    "output_current_a": "Output/load current in amps (Io)",
    "terminal_voltage_v": "Terminal voltage in volts (Vt)",
    "armature_current_a": "Armature current in amps (Ia)",
    "armature_resistance_ohm": "Armature resistance in ohms (Ra)",
}


def _param_description(param_name: str) -> str:
    """Convert a snake_case param name to a human-readable description."""
    return _PARAM_DESCRIPTIONS.get(param_name, param_name.replace("_", " "))


def _strip_code_fence(raw: str) -> str:
    """Remove an optional Markdown code fence around a model response."""
    stripped = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.I)
    return match.group(1).strip() if match else stripped


def _parse_extraction_response(raw: str, param_names: tuple[str, ...]) -> dict[str, float | None]:
    """Parse and validate an LLM JSON extraction response."""
    try:
        payload: Any = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("Variable extractor returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Variable extractor response must be a JSON object")

    unexpected = sorted(set(payload) - set(param_names))
    if unexpected:
        LOGGER.warning("Variable extractor returned unexpected keys: %s", unexpected)

    result: dict[str, float | None] = {}
    for name in param_names:
        value = payload.get(name)
        if value is None:
            result[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Variable {name!r} must be a number or null")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Variable {name!r} must be finite")
        result[name] = numeric_value
    return result


def extract_variables(
    question: str,
    param_names: tuple[str, ...],
    *,
    model: str | None = None,
    timeout: int = 30,
) -> dict[str, float | None]:
    """Use qwen3 to extract named variables from an engineering question."""
    effective_model = model or DEFAULT_MODEL
    extraction_names = param_names
    if "secondary_to_primary_turns_ratio" in param_names:
        auxiliary_names = ("input_voltage_v", "output_voltage_v", "duty_cycle")
        extraction_names += tuple(name for name in auxiliary_names if name not in extraction_names)
    variable_list = "\n".join(f"- {name}: {_param_description(name)}" for name in extraction_names)
    prompt = EXTRACTION_PROMPT.format(
        question=question,
        variable_list=variable_list,
    )
    extracted = _request_extraction(
        prompt, extraction_names, model=effective_model, timeout=timeout
    )
    _infer_forward_turns_ratio(extracted)
    missing = tuple(name for name in param_names if extracted[name] is None)
    if not missing:
        return {name: extracted[name] for name in param_names}

    retry_list = "\n".join(f"- {name}: {_param_description(name)}" for name in missing)
    retry_prompt = EXTRACTION_PROMPT.format(
        question=question,
        variable_list=retry_list,
    )
    retry_prompt += (
        "\nA previous pass missed these variables. Re-read all values and "
        "relationships in the question and return every key listed above."
    )
    retried = _request_extraction(retry_prompt, missing, model=effective_model, timeout=timeout)
    for name, value in retried.items():
        if value is not None:
            extracted[name] = value
    _infer_forward_turns_ratio(extracted)
    return {name: extracted[name] for name in param_names}


def _infer_forward_turns_ratio(variables: dict[str, float | None]) -> None:
    """Derive N2/N1 from extracted forward-converter gain variables."""
    ratio_name = "secondary_to_primary_turns_ratio"
    if ratio_name not in variables:
        return
    input_voltage = variables.get("input_voltage_v")
    output_voltage = variables.get("output_voltage_v")
    duty_cycle = variables.get("duty_cycle")
    if input_voltage and output_voltage is not None and duty_cycle:
        variables[ratio_name] = output_voltage / (input_voltage * duty_cycle)


def _request_extraction(
    prompt: str,
    param_names: tuple[str, ...],
    *,
    model: str,
    timeout: int,
) -> dict[str, float | None]:
    """Send one extraction request and parse its answer."""
    response = generate(
        prompt,
        model=model,
        timeout=timeout,
        system_prompt=None,
        think=False,
        num_predict=512,
    )
    answer = response.get("answer")
    if not isinstance(answer, str):
        raise ValueError("Variable extractor returned no text answer")
    return _parse_extraction_response(answer, param_names)


def has_all_required(variables: dict[str, float | None]) -> bool:
    """Return True if all keys have non-None numeric values."""
    return all(value is not None for value in variables.values())


def missing_variables(variables: dict[str, float | None]) -> list[str]:
    """Return parameter names whose values are None."""
    return [name for name, value in variables.items() if value is None]
