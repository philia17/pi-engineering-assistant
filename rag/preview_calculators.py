"""Subject calculator routes for Integration v0 with grounded RAG fall-through."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from rag.runtime_automation import PROVENANCE_WARNING, try_runtime_automation
from src.eecc_agent import EECCEngineeringAgent
from src.em_agent import RAG_FALLBACK, EMEngineeringAgent
from src.tools.formula_classifier import _classify_by_keywords
from src.tools.mmt_calculators import (
    classify_mmt_formula_by_keywords,
    extract_mmt_variables,
)
from src.tools.mmt_calculators import (
    dispatch_formula as dispatch_mmt_formula,
)
from src.tools.pe_calculators import (
    classify_pe_formula_by_keywords,
    extract_pe_variables,
)
from src.tools.pe_calculators import (
    dispatch_formula as dispatch_pe_formula,
)
from src.tools.sep_calculators import classify_sep_formula_by_keywords, extract_sep_variables
from src.tools.sep_calculators import dispatch_formula as dispatch_sep_formula
from src.tools.trace import ToolTrace


@dataclass(frozen=True)
class CalculatorRouteResult:
    """One successful specialist calculator answer inside the preview path."""

    answer: str
    tool_name: str
    sources: list[str]
    trace: list[dict[str, Any]]
    values: dict[str, Any]
    answer_engine: dict[str, str] | None = None
    provenance_warning: str | None = None
    trusted_uncited_deterministic: bool = False
    steps: list[str] | None = None


def calculator_routing_enabled() -> bool:
    """Allow deployments to disable specialist routes without changing callers."""
    return os.getenv("EEE_PREVIEW_CALCULATORS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _vectorstore_sources(vectorstore: Any, question: str, *, top_k: int = 3) -> list[str]:
    """Collect subject-scoped provenance paths without touching the shared index."""
    docs = vectorstore.similarity_search(question, k=top_k)
    sources: list[str] = []
    for document in docs:
        source = str(document.metadata.get("source", "")).strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def try_eecc_calculator(question: str, vectorstore: Any) -> CalculatorRouteResult | None:
    """Route an EECC calculation when classification and extraction succeed."""
    formula_name = _classify_by_keywords(question)
    if formula_name is None and os.getenv("EEE_PREVIEW_CALC_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            from src.tools.formula_classifier import classify_formula

            formula_name = classify_formula(question)
        except (ValueError, OSError, RuntimeError):
            return None
    if formula_name is None:
        return None

    agent = EECCEngineeringAgent()
    try:
        response = agent.answer_question(
            question,
            formula_name=formula_name,
            use_llm_extraction=True,
        )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError):
        return None

    # Formula labels returned by the specialist agent are implementation details,
    # not course-source citations.  Preview provenance, when available, can only
    # come from the isolated subject index; the response layer still applies the
    # stricter approved-route gate for EECC.
    sources = _vectorstore_sources(vectorstore, question)
    return CalculatorRouteResult(
        answer=response.answer,
        tool_name=formula_name,
        sources=sources,
        trace=list(response.tool_trace),
        values={"value": response.value},
    )


def try_em_calculator(question: str, vectorstore: Any) -> CalculatorRouteResult | None:
    """Route an EM calculation when the agent does not request RAG fall-through."""
    use_llm = os.getenv("EEE_PREVIEW_CALC_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    routed = EMEngineeringAgent().try_answer(
        question,
        vectorstore,
        use_llm_fallback=use_llm,
    )
    if routed is None or routed.tool_name == RAG_FALLBACK:
        return None
    return CalculatorRouteResult(
        answer=routed.answer,
        tool_name=routed.tool_name,
        sources=list(routed.sources),
        trace=list(routed.tool_trace),
        values=dict(routed.values),
        steps=list(routed.steps),
    )


def _try_deterministic_calculator(
    *,
    classify: Callable[[str], str | None],
    extract: Callable[[str, str], dict[str, Any] | None],
    dispatch: Callable[..., Any],
    expected_subject: str,
    question: str,
    vectorstore: Any,
) -> CalculatorRouteResult | None:
    """Run a fully deterministic classify → extract → dispatch route.

    Returns ``None`` on classification misses, missing parameters, invalid
    inputs, conceptual/formula-lookup questions, or provenance that escapes the
    expected subject, so the grounded RAG path owns the answer.
    """
    trace = ToolTrace()

    started = perf_counter()
    tool_name = classify(question)
    if tool_name is None:
        return None
    trace.record("classify_formula", question, started, tool_name)

    started = perf_counter()
    variables = extract(tool_name, question)
    if variables is None:
        trace.record("extract_variables", tool_name, started, "missing_variables")
        return None
    trace.record(
        "extract_variables",
        tool_name,
        started,
        ", ".join(f"{key}={value!r}" for key, value in variables.items()),
    )

    started = perf_counter()
    try:
        result = dispatch(tool_name, **variables)
    except (TypeError, ValueError, KeyError):
        trace.record("deterministic_calculator", tool_name, started, "error")
        return None
    trace.record("deterministic_calculator", result.quantity, started, result.answer)

    sources = _vectorstore_sources(vectorstore, question)
    subject_prefix = f"rag/data/{expected_subject.strip().lower()}/"
    if not sources or any(not source.startswith(subject_prefix) for source in sources):
        trace.record("source_validation", expected_subject, started, "invalid_subject_sources")
        return None
    return CalculatorRouteResult(
        answer=result.answer,
        tool_name=tool_name,
        sources=sources,
        trace=trace.as_dicts(),
        values=dict(result.values),
        steps=list(result.steps),
    )


def try_mmt_calculator(question: str, vectorstore: Any) -> CalculatorRouteResult | None:
    """Route an MMT calculation through deterministic tools only."""
    return _try_deterministic_calculator(
        classify=classify_mmt_formula_by_keywords,
        extract=extract_mmt_variables,
        dispatch=dispatch_mmt_formula,
        expected_subject="MMT",
        question=question,
        vectorstore=vectorstore,
    )


def try_pe_calculator(question: str, vectorstore: Any) -> CalculatorRouteResult | None:
    """Route a PE calculation through deterministic tools only."""
    return _try_deterministic_calculator(
        classify=classify_pe_formula_by_keywords,
        extract=extract_pe_variables,
        dispatch=dispatch_pe_formula,
        expected_subject="PE",
        question=question,
        vectorstore=vectorstore,
    )


def try_subject_calculator(
    subject: str,
    question: str,
    vectorstore: Any,
    *,
    allow_runtime_automation: bool = False,
) -> CalculatorRouteResult | None:
    """Attempt a specialist calculator for EECC/EM/MMT/PE/SEP; else None for RAG."""
    if not calculator_routing_enabled():
        return None
    normalized = subject.strip().upper()
    if allow_runtime_automation:
        automated = try_runtime_automation(normalized, question)
        if automated is not None:
            return CalculatorRouteResult(
                answer=automated.answer,
                tool_name=automated.route_name,
                sources=[],
                trace=[automated.trace_record()],
                values=dict(automated.values),
                answer_engine=automated.answer_engine(),
                provenance_warning=PROVENANCE_WARNING,
                trusted_uncited_deterministic=True,
            )
    if normalized == "EECC":
        return try_eecc_calculator(question, vectorstore)
    if normalized == "EM":
        return try_em_calculator(question, vectorstore)
    if normalized == "MMT":
        return try_mmt_calculator(question, vectorstore)
    if normalized == "PE":
        return try_pe_calculator(question, vectorstore)
    if normalized == "SEP":
        return _try_deterministic_calculator(
            classify=classify_sep_formula_by_keywords,
            extract=extract_sep_variables,
            dispatch=dispatch_sep_formula,
            expected_subject="SEP",
            question=question,
            vectorstore=vectorstore,
        )
    return None
