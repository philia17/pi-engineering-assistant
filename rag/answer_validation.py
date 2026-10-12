"""Post-answer validation for the unified engineering agent (FP-403)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag.agent_contract import normalize_citations
from rag.answer_gates import Citation, citation_is_valid
from rag.subject_preview import REPO_ROOT

_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_UNIT_TOKEN = re.compile(r"\b(?:V|A|W|Hz|Ohm|Ω|rad|s|ms|μs|us|F|H|J|N|m|kg|C|Wb|T)\b")
_PART_LABEL = re.compile(r"\(([a-z])\)", re.IGNORECASE)

_DETERMINISTIC_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "eecc_magnetic_core_reluctance_flux": frozenset(
        {"reluctance_A_turn_per_Wb", "MMF_A_turn", "flux_Wb"}
    ),
    "em_rc_complete_response": frozenset(
        {"tau_s", "complete_response", "sample_time_s", "sample_voltage_V"}
    ),
}

_DETERMINISTIC_REQUIRED_TEXT: dict[str, tuple[re.Pattern[str], ...]] = {
    "dies_raw_pointer_value_semantics": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"alias.*allocation",
            r"double delete",
            r"independent (?:storage|allocation)",
            r"self-assignment",
            r"pointer.*address",
            r"padding.*layout",
        )
    ),
    "sep_stream_scan_and_array_parameter": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"fscanf.*==\s*1",
            r"feof.*after.*(?:read|attempt).*fail",
            r"ferror.*not.*success",
            r"int\s+(?P<array>[A-Za-z_][A-Za-z0-9_]*)\[\s*[1-9]\d*\s*\].*"
            r"int\s*\*\s*(?P=array).*sizeof\(\s*(?P=array)\s*\).*pointer.*"
            r"sizeof\(\s*(?P=array)\s*\[\s*0\s*\]\s*\)",
            r"pass.*(?:length|count).*separate",
        )
    ),
}


def _subject_prefix(subject: str) -> str:
    """Return the allowed note-path prefix for one subject."""
    return f"rag/data/{subject.strip().lower()}/"


def _is_conceptual_rag_request(question: str | None, calculator_route: bool) -> bool:
    """Recognize a narrow conceptual context without relaxing numeric requests."""
    return bool(
        not calculator_route
        and question
        and not _NUMERIC_TOKEN.search(question)
        and re.search(
            r"\b(?:why|explain|describe|define|symbolically|in terms of)\b", question, re.I
        )
        and not re.search(
            r"\b(?:calculate|compute|evaluate|find|determine|numeric|numerical|"
            r"how\s+(?:much|many)|(?:value|magnitude)\s+of)\b",
            question,
            re.I,
        )
    )


def validation_errors(
    *,
    status: str,
    answer: str,
    subject: str | None,
    route: str | None,
    citations: Sequence[Mapping[str, Any]],
    calculator_values: Mapping[str, Any] | None = None,
    question: str | None = None,
    trusted_uncited_deterministic: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Return human-readable validation failures for a candidate answer."""
    errors: list[str] = []
    if status in {"needs_subject", "error"}:
        return errors
    if status == "needs_input":
        if not subject or route != "input_requirements":
            errors.append("input clarification requires a subject and input_requirements route")
        if not answer.strip().endswith("?") or _NUMERIC_TOKEN.search(answer):
            errors.append("input clarification must ask for missing inputs without numeric results")
        if citations or calculator_values is not None or trusted_uncited_deterministic:
            errors.append("input clarification cannot claim citations or deterministic results")
        return errors
    if status == "abstain":
        if answer.strip() and answer.strip() != "I don't know.":
            errors.append("abstained answers must use the canonical I don't know. text")
        return errors
    if status != "ok":
        errors.append(f"unsupported status for validation: {status}")
        return errors
    if not subject:
        errors.append("non-abstained answer is missing subject")
    if not answer.strip() or answer.strip() == "I don't know.":
        errors.append("non-abstained answer is empty or abstention-like")
    if not citations and not trusted_uncited_deterministic:
        errors.append("non-abstained answer is missing citations")
    prefix = _subject_prefix(subject or "")
    for index, item in enumerate(citations):
        note_path = str(item.get("note_path", ""))
        heading = item.get("heading")
        citation = Citation(
            note_path=note_path,
            heading=heading if isinstance(heading, str) else None,
        )
        if subject and not note_path.startswith(prefix):
            errors.append(f"citations[{index}] escapes subject scope: {note_path}")
        if not citation_is_valid(citation, repo_root=repo_root):
            errors.append(f"citations[{index}] is not a valid repository note: {note_path}")
    calculator_route = bool(
        route and route not in {"rag_fallback", "subject_v1_rag", "subject_router"}
    )
    missing_calc_values = calculator_values is None or not calculator_values
    if calculator_route and missing_calc_values and not _NUMERIC_TOKEN.search(answer):
        errors.append("calculator route is missing numeric values")
    # Unit symbols also occur in prose and symbolic explanations.
    if (
        not _is_conceptual_rag_request(question, calculator_route)
        and _UNIT_TOKEN.search(answer)
        and not _NUMERIC_TOKEN.search(answer)
    ):
        errors.append("answer mentions units without a numeric value")
    capability = route.rsplit(":", maxsplit=1)[-1] if route else None
    if question:
        part_count = len(set(_PART_LABEL.findall(question)))
        if part_count >= 2 and re.search(r"\b(?:calculate|find|determine)\b", question, re.I):
            multipart_keys = _DETERMINISTIC_REQUIRED_KEYS.get(capability or "")
            if (
                not trusted_uncited_deterministic
                or multipart_keys is None
                or calculator_values is None
                or not multipart_keys <= set(calculator_values)
            ):
                errors.append("multipart numeric answer has unevaluated requested results")
    if trusted_uncited_deterministic and route:
        required_keys = _DETERMINISTIC_REQUIRED_KEYS.get(capability or "")
        if required_keys and (
            calculator_values is None or not required_keys <= set(calculator_values)
        ):
            errors.append("deterministic numeric answer is missing required result fields")
        required_text = _DETERMINISTIC_REQUIRED_TEXT.get(capability or "", ())
        if required_text and any(pattern.search(answer) is None for pattern in required_text):
            errors.append("deterministic concept answer is missing a required subpoint")
    return errors


def validate_or_abstain(
    payload: Mapping[str, Any],
    *,
    subject: str | None = None,
    question: str | None = None,
    trusted_uncited_deterministic: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return a corrected payload, abstaining when validation fails after one pass."""
    status = str(payload.get("status", "error"))
    answer = str(payload.get("answer", ""))
    route = payload.get("route")
    route_name = str(route) if isinstance(route, str) else None
    selected_subject = subject or (
        str(payload["subject"]) if payload.get("subject") is not None else None
    )
    citations = normalize_citations(
        list(payload.get("sources") or []),
        list(payload.get("citations") or []),
    )
    calc_values = payload.get("calculator_values")
    values = calc_values if isinstance(calc_values, dict) else None

    # One correction pass: normalize citations into both fields before scoring.
    corrected = dict(payload)
    corrected["citations"] = citations
    corrected["sources"] = citations
    errors = validation_errors(
        status=status,
        answer=answer,
        subject=selected_subject,
        route=route_name,
        citations=citations,
        calculator_values=values,  # type: ignore[arg-type]
        question=question,
        trusted_uncited_deterministic=trusted_uncited_deterministic,
        repo_root=repo_root,
    )
    if not errors:
        corrected["validation"] = {
            "status": "clarification_required" if status == "needs_input" else "pass",
            "method": (
                "input_requirements"
                if status == "needs_input"
                else "bounded_completeness_validation"
            ),
            "semantic_correctness_proven": False,
            "errors": [],
            "corrected": True,
        }
        return corrected

    corrected.update(
        {
            "status": "abstain",
            "answer": "I don't know.",
            "abstain_reason": "answer_validation_failed",
            "citations": [],
            "sources": [],
            "validation": {
                "status": "fail",
                "method": "bounded_completeness_validation",
                "semantic_correctness_proven": False,
                "errors": errors,
                "corrected": True,
            },
        }
    )
    return corrected
