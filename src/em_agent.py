"""EM engineering agent: classify → extract → calculate, else RAG fall-through."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.llm_client import generate
from src.tools.em_calculators import (
    FORMULA_REGISTRY,
    EMCalculatorResult,
    dispatch_formula,
    extract_em_variables,
    parse_matrix_literals,
)
from src.tools.trace import ToolTrace

DEFAULT_MODEL = os.getenv("EEE_RAG_MODEL", "qwen3:1.7b")
DEFAULT_TIMEOUT = int(os.getenv("EEE_CLASSIFIER_TIMEOUT", "60"))
RAG_FALLBACK = "rag_fallback"

_CLASSIFICATION_PROMPT = """You are an EM formula classifier. Pick the SINGLE best calculator for the question, or "unknown" if none fit. Return ONLY JSON with keys "formula_name" and "confidence".

Question: {question}

Available formulas:
{formula_list}

Rules:
1. Matrix multiplication / product of matrices → matrix_multiply.
2. Determinant → matrix_determinant. Diagonal inverse → matrix_inverse_diagonal.
   Other 2x2 inverse → matrix_inverse_2x2. Other inverse sizes → unknown.
3. Add sinusoids / phasors and report amplitude+phase → phasor_add.
4. Capacitor impedance formula or Z_C → impedance_capacitor.
5. Inductor impedance Z_L → impedance_inductor. Resistor Z_R → impedance_resistor.
6. Parallel equivalent impedance formula → impedance_parallel.
7. Series combination → impedance_series.
8. Pure explanation / conceptual questions with no numeric or formula computation → unknown.

Return format: {{"formula_name": "<exact_name_or_unknown>", "confidence": <0.0_to_1.0>}}
Return ONLY the JSON.
"""


@dataclass(frozen=True)
class EMAgentResponse:
    """Structured response from the EM calculator route."""

    answer: str
    tool_name: str
    sources: list[str]
    steps: list[str]
    values: dict[str, float]
    tool_trace: list[dict[str, Any]]
    metrics: dict[str, Any]
    context_chars: int = 0


def available_formula_names() -> list[str]:
    """Return registered EM calculator names."""
    return list(FORMULA_REGISTRY)


def _build_formula_list() -> str:
    """Build a human-readable formula list for the LLM classifier prompt."""
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


def _has_matrix_literal(question: str) -> bool:
    """Return True when the question contains a nested matrix bracket pattern."""
    return bool(re.search(r"\[\s*\[", question))


def classify_em_formula_by_keywords(question: str) -> str | None:
    """Classify unambiguous EM calculator routes without an LLM call.

    Returns a registry key, ``None`` when ambiguous (LLM fallback), or the
    sentinel ``rag_fallback`` when the question is clearly non-calculation.
    """
    normalized = question.casefold()
    normalized = normalized.replace("ω", "omega").replace("∠", " at ").replace("⁰", " degrees")

    explanation_markers = (
        "how are",
        "what does",
        "what mathematical",
        "what do the",
        "under what",
        "explain",
        "why does",
        "why is",
        "when converting",
        "describe",
        "geometrically",
        "distributed",
        "meaning of",
        "interpretation",
        "interpreted",
        "justify",
    )
    # Explanation phrasing wins unless the question embeds an explicit
    # matrix literal or sinusoid that a calculator can evaluate.
    has_explicit_calc_payload = _has_matrix_literal(question) or any(
        marker in normalized for marker in ("sin(omega", "cos(omega")
    )
    if (
        any(marker in normalized for marker in explanation_markers)
        and not has_explicit_calc_payload
    ):
        return RAG_FALLBACK

    # Matrix routes — prefer explicit verbs / bracket patterns.
    if _has_matrix_literal(question) or "matrix" in normalized:
        if "multiply" in normalized or "product" in normalized:
            return "matrix_multiply"
        if "determinant" in normalized or "det(" in normalized:
            return "matrix_determinant"
        if "inverse" in normalized and (
            _has_matrix_literal(question)
            or "2x2" in normalized
            or "2 x 2" in normalized
            or re.search(r"(?<![-\w])diagonal\b", normalized)
        ):
            matrices: list[list[list[float]]] = parse_matrix_literals(question)
            if len(matrices) > 1:
                return RAG_FALLBACK
            if re.search(r"(?<![-\w])diagonal\b", normalized):
                return "matrix_inverse_diagonal"
            if matrices:
                matrix: list[list[float]] = matrices[0]
                size: int = len(matrix)
                if all(len(row) == size for row in matrix) and all(
                    value == 0
                    for i, row in enumerate(matrix)
                    for j, value in enumerate(row)
                    if i != j
                ):
                    return "matrix_inverse_diagonal"
                if size != 2 or any(len(row) != 2 for row in matrix):
                    return RAG_FALLBACK
            return "matrix_inverse_2x2"
        if re.search(r"multiply\s+[a-z]\s*=\s*\[\[", normalized):
            return "matrix_multiply"

    # Phasor / sinusoid addition.
    if (
        "phasor" in normalized
        or "sin(omega" in normalized
        or "sin(ω" in question.casefold()
        or "cos(omega" in normalized
    ):
        if any(term in normalized for term in ("add ", "sum ", "resultant", "amplitude and phase")):
            return "phasor_add"
        if "rectangular" in normalized and "polar" in normalized:
            if "to polar" in normalized or "polar form" in normalized:
                return "phasor_rect_to_polar"
            if "to rectangular" in normalized or "rectangular form" in normalized:
                return "phasor_polar_to_rect"
        if "magnitude" in normalized and ("angle" in normalized or "phase" in normalized):
            return "phasor_magnitude_angle"

    # Impedance formula routes. Combination identities need an explicit
    # formula/equivalent cue so "series RLC ... impedance at resonance"
    # stays on the RAG path.
    if "impedance" in normalized:
        if "capacitor" in normalized or "capacitance" in normalized:
            return "impedance_capacitor"
        if "inductor" in normalized or ("inductance" in normalized and "mutual" not in normalized):
            return "impedance_inductor"
        if "resistor" in normalized:
            return "impedance_resistor"
        formula_cue = any(
            cue in normalized for cue in ("formula", "equivalent", "state the", "expression for")
        )
        if formula_cue and "parallel" in normalized:
            return "impedance_parallel"
        if formula_cue and "series" in normalized:
            return "impedance_series"

    if "z_c" in normalized or "zc =" in normalized.replace(" ", ""):
        return "impedance_capacitor"
    if "z_l" in normalized or "zl =" in normalized.replace(" ", ""):
        return "impedance_inductor"
    if "z_r" in normalized or "zr =" in normalized.replace(" ", ""):
        return "impedance_resistor"

    return None


def _parse_classification_response(raw: str) -> tuple[str, float]:
    """Parse and validate an EM formula-classification JSON response."""
    try:
        payload: Any = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("EM formula classifier returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("EM formula classifier response must be a JSON object")

    formula_name = payload.get("formula_name")
    if formula_name == "unknown":
        return RAG_FALLBACK, 0.0
    if not isinstance(formula_name, str) or formula_name not in FORMULA_REGISTRY:
        raise ValueError(f"EM formula classifier returned unknown formula: {formula_name!r}")

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


def _looks_calculation_ambiguous(question: str) -> bool:
    """Return True when the question has weak calc cues worth an LLM classify."""
    normalized = question.casefold()
    weak_signals = (
        "matrix",
        "phasor",
        "impedance",
        "determinant",
        "multiply",
        "sin(",
        "cos(",
    )
    return _has_matrix_literal(question) or any(signal in normalized for signal in weak_signals)


def classify_em_formula(
    question: str,
    *,
    model: str | None = None,
    timeout: int | None = None,
    use_llm_fallback: bool = True,
) -> str:
    """Classify an EM question into a calculator route or ``rag_fallback``."""
    keyword_route = classify_em_formula_by_keywords(question)
    if keyword_route is not None:
        return keyword_route
    if not use_llm_fallback or not _looks_calculation_ambiguous(question):
        return RAG_FALLBACK

    effective_model = model or DEFAULT_MODEL
    effective_timeout = timeout or DEFAULT_TIMEOUT
    prompt = _CLASSIFICATION_PROMPT.format(
        question=question,
        formula_list=_build_formula_list(),
    )
    try:
        response = generate(
            prompt,
            model=effective_model,
            timeout=effective_timeout,
            system_prompt=None,
            think=False,
            num_predict=96,
        )
    except Exception:  # noqa: BLE001 - fall through to RAG on classifier failure
        return RAG_FALLBACK
    answer = response.get("answer")
    if not isinstance(answer, str):
        return RAG_FALLBACK
    try:
        formula_name, _confidence = _parse_classification_response(answer)
    except ValueError:
        return RAG_FALLBACK
    return formula_name


class EMEngineeringAgent:
    """Route EM calculation questions to deterministic tools with provenance."""

    def classify(self, question: str, *, use_llm_fallback: bool = True) -> str:
        """Return the calculator tool name or ``rag_fallback``."""
        return classify_em_formula(question, use_llm_fallback=use_llm_fallback)

    def try_answer(
        self,
        question: str,
        vectorstore: Any | None = None,
        *,
        candidates: int = 10,
        keep: int = 5,
        use_llm_fallback: bool = True,
    ) -> EMAgentResponse | None:
        """Answer via calculator when routed; return ``None`` for RAG fall-through."""
        run_start = perf_counter()
        trace = ToolTrace()

        started = perf_counter()
        tool_name = self.classify(question, use_llm_fallback=use_llm_fallback)
        trace.record("classify_formula", question, started, tool_name)
        if tool_name == RAG_FALLBACK or tool_name not in FORMULA_REGISTRY:
            return None

        started = perf_counter()
        variables = extract_em_variables(tool_name, question)
        if variables is None:
            trace.record(
                "extract_variables",
                tool_name,
                started,
                "missing_variables",
            )
            return None
        trace.record(
            "extract_variables",
            tool_name,
            started,
            ", ".join(f"{key}={value!r}" for key, value in variables.items()) or "symbolic",
        )

        started = perf_counter()
        try:
            result = dispatch_formula(tool_name, **variables)
        except (TypeError, ValueError, KeyError):
            trace.record("deterministic_calculator", tool_name, started, "error")
            return None
        trace.record(
            "deterministic_calculator",
            result.quantity,
            started,
            result.answer,
        )

        started = perf_counter()
        sources, context_chars = self._retrieve_provenance(
            question,
            vectorstore,
            candidates=candidates,
            keep=keep,
        )
        trace.record(
            "retrieve_provenance",
            tool_name,
            started,
            ", ".join(sources) if sources else "none",
        )

        return EMAgentResponse(
            answer=result.answer,
            tool_name=tool_name,
            sources=sources,
            steps=list(result.steps),
            values=dict(result.values),
            tool_trace=trace.as_dicts(),
            metrics={
                "agent": "EMEngineeringAgent.phase2",
                "task_type": tool_name,
                "latency_s": round(perf_counter() - run_start, 4),
            },
            context_chars=context_chars,
        )

    def answer_with_tool(
        self,
        question: str,
        formula_name: str,
        *,
        variables: dict[str, Any] | None = None,
    ) -> EMCalculatorResult:
        """Dispatch a known formula with explicit or extracted variables."""
        if formula_name not in FORMULA_REGISTRY:
            raise ValueError(f"Unsupported EM calculator formula: {formula_name}")
        params = (
            variables if variables is not None else extract_em_variables(formula_name, question)
        )
        if params is None:
            raise ValueError(f"Could not extract variables for {formula_name}")
        return dispatch_formula(formula_name, **params)

    @staticmethod
    def _retrieve_provenance(
        question: str,
        vectorstore: Any | None,
        *,
        candidates: int,
        keep: int,
    ) -> tuple[list[str], int]:
        """Retrieve reranked sources for provenance; empty when store is absent."""
        if vectorstore is None:
            return [], 0
        from rag.retrievers.rerank import rerank_chunks

        docs = vectorstore.similarity_search(question, k=candidates)
        reranked = rerank_chunks(question, docs, keep=keep)
        sources = [doc.metadata.get("source", "") for doc in reranked]
        context_chars = sum(len(doc.page_content) for doc in reranked)
        return sources, context_chars
