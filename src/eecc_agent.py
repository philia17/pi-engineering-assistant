"""Minimal trace-visible EECC engineering agent demo."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.tools.eecc_calculators import (
    FORMULA_REGISTRY,
    CalculatorResult,
    dispatch_formula,
)
from src.tools.format_checker import extract_numbers
from src.tools.formula_classifier import classify_formula
from src.tools.trace import ToolTrace
from src.tools.variable_extractor import (
    extract_variables,
    has_all_required,
    missing_variables,
)


@dataclass(frozen=True)
class EECCAgentResponse:
    """Structured response from the EECC engineering agent demo."""

    item_id: str
    answer: str
    value: float
    unit: str
    sources: list[str]
    tool_trace: list[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class EECCRoute:
    """One supported EECC calculator route."""

    task_type: str
    formula_summary: str
    variables: dict[str, float]
    question: str = ""


class EECCEngineeringAgent:
    """Small controlled agent that wires retrieval, calculator, and trace."""

    def answer_question(
        self,
        question: str,
        formula_name: str | None = None,
        *,
        use_llm_extraction: bool = True,
    ) -> EECCAgentResponse:
        """Answer a new question using an explicit or automatically selected formula."""
        run_start = perf_counter()
        trace = ToolTrace()
        if formula_name is None:
            started = perf_counter()
            formula_name = classify_formula(question)
            trace.record("classify_formula", question, started, formula_name)
        if formula_name not in FORMULA_REGISTRY:
            raise ValueError(f"Unsupported EECC calculator formula: {formula_name}")
        if not use_llm_extraction:
            raise ValueError(
                "answer_question() requires LLM extraction; call dispatch_formula() "
                "with explicit parameters for offline calculations"
            )

        spec = FORMULA_REGISTRY[formula_name]

        started = perf_counter()
        extracted = extract_variables(question, spec.param_names)
        trace.record(
            "extract_variables",
            formula_name,
            started,
            ", ".join(f"{key}={value}" for key, value in extracted.items()),
        )
        if not has_all_required(extracted):
            missing = ", ".join(missing_variables(extracted))
            raise ValueError(f"Could not extract required variables: {missing}")
        variables = {name: value for name, value in extracted.items() if value is not None}

        route = EECCRoute(
            task_type=formula_name,
            formula_summary=spec.description,
            variables=variables,
            question=question,
        )
        started = perf_counter()
        sources = self._retrieve_formula(route)
        trace.record("retrieve_formula", question, started, spec.description)

        started = perf_counter()
        result = dispatch_formula(formula_name, **variables)
        trace.record(
            "deterministic_calculator",
            result.quantity,
            started,
            f"{result.quantity}={result.value:g} {result.unit}",
        )
        answer = self._format_answer(result)
        started = perf_counter()
        format_passed = self._check_format(answer)
        trace.record(
            "check_format",
            "final answer contains numeric value",
            started,
            "passed" if format_passed else "failed",
        )
        return EECCAgentResponse(
            item_id="",
            answer=answer,
            value=result.value,
            unit=result.unit,
            sources=sources,
            tool_trace=trace.as_dicts(),
            metrics={
                "agent": "EECCEngineeringAgent.llm-extraction",
                "task_type": formula_name,
                "latency_s": round(perf_counter() - run_start, 4),
                "format_passed": format_passed,
            },
        )

    @staticmethod
    def _retrieve_formula(route: EECCRoute, top_k: int = 3) -> list[str]:
        """Retrieve relevant chunks from ChromaDB using the item's question text.

        Falls back to the formula description as a search query if question is empty.
        """
        from rag.pipeline import load_vectorstore

        query = route.question or route.formula_summary
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(query, k=top_k)
        return [doc.metadata.get("source", "") for doc in docs]

    @staticmethod
    def _calculate(task_type: str, variables: dict[str, float]) -> CalculatorResult:
        """Dispatch to the deterministic calculator for a classified route."""
        if task_type not in FORMULA_REGISTRY:
            raise ValueError(f"Unsupported EECC calculator route: {task_type}")
        return dispatch_formula(task_type, **variables)

    @staticmethod
    def _format_answer(result: CalculatorResult) -> str:
        """Format a calculator result as a concise answer."""
        if "mean_transistor_current_a" in result.extra_values:
            mean_current = result.extra_values["mean_transistor_current_a"]
            secondary_peak = result.extra_values["secondary_inductor_peak_a"]
            return (
                f"Secondary inductor peak = {secondary_peak:.4g} A; "
                f"peak transistor current = {result.value:.4g} A; "
                f"mean transistor current = {mean_current:.4g} A"
            )
        if "motor_emf_v" in result.extra_values:
            motor_emf = result.extra_values["motor_emf_v"]
            return (
                f"Generator EMF = {result.value:.4g} {result.unit}; "
                f"Motor EMF = {motor_emf:.4g} {result.unit}"
            )
        if result.unit == "ratio":
            return f"{result.quantity} = {result.value:.4g}"
        return f"{result.quantity} = {result.value:.4g} {result.unit}"

    @staticmethod
    def _check_format(answer: str) -> bool:
        """Validate that the answer contains at least one numeric value."""
        return bool(extract_numbers(answer))
