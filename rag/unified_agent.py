"""Eight-subject unified engineering agent (FP-402 / FP-403)."""

from __future__ import annotations

import re
import time
from typing import Any

from rag import pipeline
from rag.agent_contract import (
    SUPPORTED_MODES,
    AgentRequest,
    AgentResponse,
    normalize_citations,
    review_badge,
)
from rag.answer_gates import apply_answer_gates
from rag.answer_validation import validate_or_abstain
from rag.input_requirements import check_input_requirements
from rag.preview_calculators import CalculatorRouteResult, try_subject_calculator
from rag.runtime_models import resolve_model_id
from rag.subject_index import SubjectIndexError, SubjectIndexService
from rag.subject_preview import (
    SubjectPreviewError,
    SubjectPreviewService,
    resolve_subject,
)
from rag.subject_router import route_subject

MAX_CONVERSATION_TURNS: int = 6
MAX_CONVERSATION_CONTEXT_CHARS: int = 6_000
_SUBJECT_TAG = re.compile(r"(?<!\w)@(DIES|EECC|EM|EPC|IS|MMT|PE|SEP)(?!\w)", re.IGNORECASE)


class UnifiedAgentError(ValueError):
    """Raised when a unified request cannot be executed safely."""


class UnifiedEngineeringAgent:
    """Route calculators and grounded RAG behind one stable response contract."""

    def __init__(
        self,
        *,
        preview_service: SubjectPreviewService | None = None,
        v1_service: SubjectIndexService | None = None,
    ) -> None:
        self._preview = preview_service or SubjectPreviewService()
        self._v1 = v1_service or SubjectIndexService()

    def index_status(self, subject: str, *, mode: str = "v1") -> Any:
        """Return index readiness for the selected mode."""
        selected = mode.strip().lower()
        if selected == "preview":
            return self._preview.index_status(subject)
        if selected == "v1":
            return self._v1.index_status(subject)
        raise UnifiedAgentError(f"Unsupported mode: {mode!r}")

    def _vectorstore_and_chain(self, subject: str, *, mode: str) -> tuple[Any, Any]:
        """Load the isolated store/chain for one subject and mode."""
        if mode == "preview":
            chain = self._preview.get_chain(subject)
            return self._preview.get_vectorstore(subject), chain
        chain = self._v1.get_chain(subject)
        return self._v1.get_vectorstore(subject), chain

    def _rag_answer(
        self,
        *,
        subject: str,
        question: str,
        retrieval_query: str | None = None,
        mode: str,
        model: str,
        vectorstore: Any,
        chain: Any,
    ) -> dict[str, Any]:
        """Run reranked retrieval + citation gates for one subject store."""
        result = pipeline.query(
            question,
            chain,
            model=model,
            vectorstore=vectorstore,
            rerank=True,
            retrieval_query=retrieval_query,
        )
        gated = apply_answer_gates(
            answer=str(result.get("answer", "")),
            documents=list(result.get("documents") or []),
        )
        record = resolve_subject(subject)
        review: dict[str, Any]
        if mode == "preview":
            review = {
                "review_status_counts": record.review_status_counts,
                "warnings": ["preview_index_not_production_eligible"],
                "badge": "preview",
            }
        else:
            review = {
                "review_status_counts": record.review_status_counts,
                "eligibility_policy": "eligible_route",
                "warnings": [],
                "badge": review_badge(
                    {
                        "review_status_counts": record.review_status_counts,
                        "eligibility_policy": "eligible_route",
                    },
                    mode=mode,
                ),
            }
        return {
            "status": gated["status"],
            "mode": mode,
            "subject": record.subject,
            "module": record.module,
            "model": model,
            "answer": gated["answer"],
            "sources": gated["sources"],
            "citations": gated["citations"],
            "route": "rag_fallback" if mode == "preview" else "subject_v1_rag",
            "abstain_reason": gated["reason"],
            "trace": [
                {"stage": "calculator_route", "status": "miss"},
                {
                    "stage": "grounded_rag",
                    "status": gated["status"],
                    "reranked": bool(result.get("reranked")),
                },
            ],
            "review": review,
        }

    @staticmethod
    def _conversation_question(
        question: str,
        context: tuple[tuple[str, str], ...],
    ) -> str:
        """Add bounded prior turns to the RAG prompt without storing them."""
        if not context:
            return question
        bounded: list[tuple[str, str]] = []
        total = 0
        for prior_question, prior_answer in reversed(context[-MAX_CONVERSATION_TURNS:]):
            pair = (str(prior_question)[:2_000], str(prior_answer)[:2_000])
            pair_size = len(pair[0]) + len(pair[1])
            if bounded and total + pair_size > MAX_CONVERSATION_CONTEXT_CHARS:
                break
            bounded.append(pair)
            total += pair_size
        bounded.reverse()
        lines = ["Prior turns in this conversation:"]
        for index, (prior_question, prior_answer) in enumerate(bounded, start=1):
            lines.append(f"Turn {index} user: {prior_question}")
            lines.append(f"Turn {index} assistant: {prior_answer}")
        lines.append(f"Current user question: {question}")
        return "\n".join(lines)

    def answer(
        self,
        request: AgentRequest,
        *,
        retrieval_query: object | None = None,
    ) -> AgentResponse:
        """Answer one request through subject routing, tools, RAG and validation."""
        started = time.perf_counter()
        question = request.cleaned_question()
        if not question:
            raise UnifiedAgentError("Question must not be empty")
        if len(question) > 4_000:
            raise UnifiedAgentError("Question exceeds 4000 characters")
        safe_retrieval_query = pipeline.normalize_retrieval_query(retrieval_query)
        mode = (request.mode or "v1").strip().lower()
        if mode not in SUPPORTED_MODES:
            raise UnifiedAgentError("mode must be preview or v1")

        subject_tags = {tag.upper() for tag in _SUBJECT_TAG.findall(question)}
        selected_subject = request.subject.strip().upper() if request.subject else None
        if subject_tags and selected_subject and subject_tags != {selected_subject}:
            candidates = sorted({*subject_tags, selected_subject})
            return AgentResponse(
                status="needs_subject",
                mode=mode,
                answer=(
                    "The subject tag conflicts with the selected subject. "
                    "Please choose one subject."
                ),
                subject=None,
                route="subject_router",
                routing={
                    "status": "clarify",
                    "confidence": 0.0,
                    "reason": "explicit_subject_tag_conflict",
                    "candidates": candidates,
                    "scores": {},
                },
                confidence=0.0,
                timing_ms=round((time.perf_counter() - started) * 1_000, 1),
                review={"badge": "n/a"},
            )

        decision = route_subject(question, explicit_subject=request.subject)
        routing = {
            "status": decision.status,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "candidates": list(decision.candidates),
            "scores": decision.scores,
        }
        if decision.status == "reject":
            raise UnifiedAgentError(f"Unknown subject: {request.subject!r}")
        if decision.status in {"clarify", "multi_subject_fallback"}:
            return AgentResponse(
                status="needs_subject",
                mode=mode,
                answer=(
                    "Please choose a subject. Top candidates: " + ", ".join(decision.candidates)
                ),
                subject=None,
                route="subject_router",
                routing=routing,
                confidence=decision.confidence,
                timing_ms=round((time.perf_counter() - started) * 1_000, 1),
                review={"badge": "n/a"},
            )

        assert decision.subject is not None
        record = resolve_subject(decision.subject)
        clarification = check_input_requirements(
            question,
            subject=record.subject,
            has_conversation_context=bool(request.conversation_context),
            image_provided=request.image_provided,
            retrieval_query=safe_retrieval_query,
        )
        if clarification is not None:
            return AgentResponse(
                status="needs_input",
                mode=mode,
                answer=clarification.question,
                subject=record.subject,
                module=record.module,
                model_requested=request.model,
                route="input_requirements",
                routing=routing,
                trace=[
                    {
                        "stage": "input_requirements",
                        "status": "clarification_required",
                        "missing_fields": list(clarification.missing_fields),
                        "semantic_correctness_proven": False,
                    }
                ],
                review={"badge": "n/a"},
                confidence=decision.confidence,
                timing_ms=round((time.perf_counter() - started) * 1_000, 1),
            )
        model = resolve_model_id(request.model)
        try:
            vectorstore, chain = self._vectorstore_and_chain(record.subject, mode=mode)
        except (SubjectPreviewError, SubjectIndexError) as error:
            raise UnifiedAgentError(str(error)) from error

        # A leading, validated UI subject mention is metadata, not calculator syntax.
        calculator_question = re.sub(
            rf"\A@{record.subject}\s+", "", question, count=1, flags=re.IGNORECASE
        )
        routed = try_subject_calculator(
            record.subject,
            calculator_question,
            vectorstore,
            allow_runtime_automation=mode == "preview",
        )
        trusted_uncited_deterministic = bool(
            isinstance(routed, CalculatorRouteResult)
            and routed.trusted_uncited_deterministic
            and routed.answer_engine
            and routed.answer_engine.get("kind") == "deterministic"
        )
        if routed is not None:
            citations = (
                [] if trusted_uncited_deterministic else normalize_citations(list(routed.sources))
            )
            payload: dict[str, Any] = {
                "status": "ok",
                "mode": mode,
                "subject": record.subject,
                "module": record.module,
                "model": model,
                "model_requested": model,
                "answer_engine": routed.answer_engine,
                "answer": routed.answer,
                "sources": citations,
                "citations": citations,
                "route": routed.tool_name,
                "calculator_values": routed.values,
                "calculator_steps": routed.steps,
                "trace": [
                    {
                        "stage": "calculator_route",
                        "status": "ok",
                        "tool": routed.tool_name,
                    },
                    *routed.trace,
                ],
                "review": {
                    "review_status_counts": record.review_status_counts,
                    "warnings": [
                        *(["preview_index_not_production_eligible"] if mode == "preview" else []),
                        *(
                            [routed.provenance_warning]
                            if routed.provenance_warning is not None
                            else []
                        ),
                    ],
                    "badge": review_badge(
                        {
                            "review_status_counts": record.review_status_counts,
                            "eligibility_policy": None if mode == "preview" else "eligible_route",
                        },
                        mode=mode,
                    ),
                },
            }
        else:
            try:
                conversation_question = self._conversation_question(
                    question,
                    request.conversation_context,
                )
                if safe_retrieval_query is None:
                    payload = self._rag_answer(
                        subject=record.subject,
                        question=conversation_question,
                        mode=mode,
                        model=model,
                        vectorstore=vectorstore,
                        chain=chain,
                    )
                else:
                    payload = self._rag_answer(
                        subject=record.subject,
                        question=conversation_question,
                        mode=mode,
                        model=model,
                        vectorstore=vectorstore,
                        chain=chain,
                        retrieval_query=safe_retrieval_query,
                    )
            except (SubjectPreviewError, SubjectIndexError, OSError, RuntimeError) as error:
                # Unsupported tools / retrieval failures fall through to abstention, not a crash.
                payload = {
                    "status": "abstain",
                    "mode": mode,
                    "subject": record.subject,
                    "module": record.module,
                    "model": model,
                    "model_requested": model,
                    "answer_engine": None,
                    "answer": "I don't know.",
                    "sources": [],
                    "citations": [],
                    "route": "rag_fallback",
                    "abstain_reason": "tool_or_retrieval_failure",
                    "trace": [
                        {"stage": "calculator_route", "status": "miss"},
                        {
                            "stage": "grounded_rag",
                            "status": "error",
                            "error": str(error),
                        },
                    ],
                    "review": {
                        "review_status_counts": record.review_status_counts,
                        "warnings": [str(error)],
                        "badge": review_badge(
                            {"review_status_counts": record.review_status_counts},
                            mode=mode,
                        ),
                    },
                }

        validated = validate_or_abstain(
            payload,
            subject=record.subject,
            question=question,
            trusted_uncited_deterministic=trusted_uncited_deterministic,
        )
        if "validation" in validated:
            validated.setdefault("trace", []).append(
                {
                    "stage": "answer_completeness",
                    "status": validated["validation"]["status"],
                    "method": validated["validation"].get("method"),
                    "semantic_correctness_proven": validated["validation"].get(
                        "semantic_correctness_proven", False
                    ),
                    "errors": list(validated["validation"].get("errors") or []),
                }
            )
        timing_ms = round((time.perf_counter() - started) * 1_000, 1)
        return AgentResponse(
            status=validated["status"],  # type: ignore[arg-type]
            mode=mode,
            answer=str(validated.get("answer", "")),
            subject=str(validated.get("subject")) if validated.get("subject") else None,
            module=str(validated["module"]) if validated.get("module") else None,
            model=model,
            model_requested=(
                str(validated["model_requested"]) if validated.get("model_requested") else model
            ),
            answer_engine=(
                dict(validated["answer_engine"])
                if isinstance(validated.get("answer_engine"), dict)
                else None
            ),
            route=str(validated["route"]) if validated.get("route") else None,
            sources=list(validated.get("sources") or []),
            citations=list(validated.get("citations") or []),
            routing=routing,
            trace=list(validated.get("trace") or []),
            review=dict(validated.get("review") or {}),
            confidence=decision.confidence,
            abstain_reason=(
                str(validated["abstain_reason"])
                if validated.get("abstain_reason") is not None
                else None
            ),
            calculator_values=(
                dict(validated["calculator_values"])
                if isinstance(validated.get("calculator_values"), dict)
                else None
            ),
            calculator_steps=(
                list(validated["calculator_steps"])
                if routed is not None
                and validated["status"] == "ok"
                and validated.get("calculator_steps")
                else None
            ),
            timing_ms=timing_ms,
        )
