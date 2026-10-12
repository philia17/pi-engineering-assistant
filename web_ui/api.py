"""FastAPI application for legacy RAG, integration preview clients, and unified v1."""

from __future__ import annotations

import hashlib
import inspect
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from random import SystemRandom
from time import perf_counter
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.agent_contract import AgentRequest
from rag.quiz_contract import QuizGenerationError
from rag.quiz_scopes import (
    QuizFocus,
    QuizScopeSelection,
    QuizTopic,
    build_scope_catalog,
    resolve_scope,
)
from rag.quiz_service import QUIZ_MODEL_ID, QuizV1IndexError
from rag.quiz_service import admitted_note_paths as _quiz_admitted_note_paths
from rag.quiz_service import generate_question as _generate_quiz_question
from rag.quiz_service import plan_quiz_focuses as _plan_quiz_focuses
from rag.runtime_models import default_model_id, model_options, resolve_model_id
from rag.subject_index import V1_INDEX_ROOT, SubjectIndexError, SubjectIndexService
from rag.subject_preview import (
    INDEX_ROOT,
    SubjectPreviewError,
    SubjectPreviewService,
    corpus_root,
    load_subject_catalog,
    resolve_subject,
)
from rag.subject_router import route_subject
from rag.unified_agent import UnifiedAgentError, UnifiedEngineeringAgent
from web_ui.demo_auth import (
    DemoSession,
    auth_router,
    bind_or_validate_conversation,
    conversation_context,
    demo_session,
    record_conversation_turn,
)
from web_ui.live_vision_contract import validate_image_bytes
from web_ui.quiz_sessions import QuizQuestionState, QuizState, QuizStateError
from web_ui.vision_adapter import VisionAnalysis, analyze_image
from web_ui.vision_contract import build_vision_envelope
from web_ui.vision_fusion import (
    VisionFusionError,
    augment_retrieval_query,
    build_temporary_vision_context,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
INTEGRATION_UI_DIR = Path(__file__).resolve().parent / "integration_v0"
UNIFIED_UI_DIR = Path(__file__).resolve().parent / "integration_v1"

_rag_chain: Any = None
_preview_service = SubjectPreviewService()
_v1_service = SubjectIndexService()
_unified_agent = UnifiedEngineeringAgent(
    preview_service=_preview_service,
    v1_service=_v1_service,
)
_quiz_plan_rng = SystemRandom()


def _web_handoff_origins() -> list[str]:
    """Parse optional external browser origins allowed to call the preview API."""
    configured = os.getenv("EEE_WEB_HANDOFF_ORIGINS", "").strip()
    if not configured:
        return []
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Keep legacy and preview chains lazy so one absent store cannot block all subjects."""
    del app
    global _rag_chain
    _rag_chain = None
    yield
    _rag_chain = None


app = FastAPI(
    title="EEE RAG API",
    description="Local EEE RAG service with eight-subject preview and unified v1 agent paths.",
    version="0.3.0",
    lifespan=lifespan,
)

_handoff_origins = _web_handoff_origins()
if _handoff_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_handoff_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
app.include_router(auth_router)


class QueryRequest(BaseModel):
    """Legacy shared-index text query."""

    question: str


class QueryResponse(BaseModel):
    """Legacy shared-index response."""

    answer: str
    sources: list[str]


class QuizStartRequest(BaseModel):
    """One fixed-count Quiz start request."""

    subject: str
    mode: str = "preview"
    model: str | None = None
    topic_id: str
    knowledge_point_id: str | None = None
    question_count: int = 5


class QuizSubmitRequest(BaseModel):
    """One current-question option submission."""

    question_id: str
    option_id: str


def _required_quiz_session(session: DemoSession | None) -> DemoSession:
    """Require a real server session even when legacy demo auth is disabled."""
    if session is None:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "quiz_session_required",
                "message": "A valid demo session is required for Quiz.",
            },
        )
    return session


def _quiz_state(session: DemoSession | None) -> QuizState:
    """Return the authenticated session's current Quiz."""
    record = _required_quiz_session(session)
    if record.quiz_state is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "quiz_not_found", "message": "No Quiz is active."},
        )
    return record.quiz_state


def _raise_quiz_generation_error(error: Exception) -> None:
    """Map generation failures without exposing model output or retrieved context."""
    if isinstance(error, SubjectPreviewError | SubjectIndexError | QuizV1IndexError):
        raise HTTPException(
            status_code=409,
            detail={"code": "index_not_ready", "message": str(error)},
        ) from error
    if isinstance(error, ValueError) and not isinstance(error, QuizGenerationError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    raise HTTPException(
        status_code=422,
        detail={
            "code": "quiz_generation_invalid",
            "message": "A grounded Quiz question could not be generated.",
        },
    ) from error


def _quiz_scope_catalog(subject: str, mode: str) -> tuple[str, tuple[QuizTopic, ...]]:
    """Build the current subject/mode catalog from admitted manifest paths."""
    selected_subject = resolve_subject(subject).subject
    scope_repo_root = (
        corpus_root().parents[1] if mode == "preview" else Path(__file__).resolve().parents[1]
    )
    note_paths = _quiz_admitted_note_paths(
        subject=selected_subject,
        mode=mode,
        preview_service=_preview_service,
        repo_root=scope_repo_root,
    )
    return selected_subject, build_scope_catalog(scope_repo_root, note_paths)


def _quiz_question_counts(
    selected_focuses: tuple[QuizFocus, ...],
    subject_focuses: tuple[QuizFocus, ...],
    *,
    reserve_repeat: bool,
) -> list[int]:
    """Return the supported public count choices for one resolved scope."""
    supported: list[int] = []
    for count in (1, 3, 5, 7, 10):
        try:
            _plan_quiz_focuses(
                selected_focuses,
                subject_focuses,
                count * 2 if reserve_repeat else count,
            )
        except ValueError:
            continue
        supported.append(count)
    return supported


def _quiz_topics_payload(
    topics: tuple[QuizTopic, ...], *, reserve_repeat: bool = False
) -> list[dict[str, object]]:
    """Serialize the complete catalog with capacity but no private facts or focuses."""
    subject_focuses = tuple(focus for topic in topics for focus in topic.focuses)
    payload: list[dict[str, object]] = []
    for topic in topics:
        focus_by_id = {focus.id: focus for focus in topic.focuses}
        topic_counts = _quiz_question_counts(
            topic.focuses, subject_focuses, reserve_repeat=reserve_repeat
        )
        if not topic_counts:
            continue
        points: list[dict[str, object]] = []
        for point in topic.knowledge_points:
            point_counts = _quiz_question_counts(
                tuple(focus_by_id[focus_id] for focus_id in point.focus_ids),
                subject_focuses,
                reserve_repeat=reserve_repeat,
            )
            if point_counts:
                points.append({**point.as_dict(), "question_counts": point_counts})
        payload.append(
            {
                "id": topic.id,
                "label": topic.label,
                "question_counts": topic_counts,
                "knowledge_points": points,
            }
        )
    payload.append(
        {
            "id": "mixed",
            "label": "Mixed Topics",
            "question_counts": _quiz_question_counts(
                subject_focuses, subject_focuses, reserve_repeat=reserve_repeat
            ),
            "knowledge_points": [],
        }
    )
    return payload


def _legacy_chain() -> Any:
    """Load the historical shared RAG chain only when its legacy route is used."""
    global _rag_chain
    if _rag_chain is None:
        from rag.pipeline import build_rag_chain, load_vectorstore

        _rag_chain = build_rag_chain(load_vectorstore())
    return _rag_chain


def _read_upload(image: UploadFile | None) -> tuple[dict[str, object], bytes | None]:
    """Validate an optional temporary image without persisting user content."""
    if image is None or not image.filename:
        return {"status": "not_provided"}, None
    content_type = image.content_type or ""
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG and WEBP uploads are supported in Integration v0.",
        )
    content = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit.")
    if not content:
        raise HTTPException(status_code=422, detail="Image upload is empty.")
    return (
        {
            "status": "accepted_pending_vision",
            "content_type": content_type,
            "byte_count": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        content,
    )


def _vision_failure() -> VisionAnalysis:
    """Return a stable failure result when adapter orchestration itself fails."""
    return VisionAnalysis(
        status="vision_unavailable",
        provider_kind="orchestration",
        served_model_provider=None,
        provider_model=None,
        observation={
            "visible_labels": [],
            "components": [],
            "connections": [],
            "polarity_or_direction": [],
            "plot_readings": [],
            "uncertainties": ["Vision analysis was unavailable for this request."],
        },
        upstream_status="vision_unavailable",
        error_code="vision_orchestration_failed",
        error="vision_orchestration_failed",
    )


def _with_vision_observability(
    vision: dict[str, object],
    *,
    vision_latency_ms: float,
    retrieval_augmented: bool,
) -> dict[str, object]:
    """Attach the FEI-D02 demo latency and retrieval-augmentation fields."""
    return {
        **vision,
        "vision_latency_ms": vision_latency_ms,
        "retrieval_augmented": retrieval_augmented,
    }


def _vision_and_retrieval_query(
    *,
    image_metadata: dict[str, object],
    image_bytes: bytes | None,
    selected_subject: str,
    question: str,
) -> tuple[dict[str, object], str | None]:
    """Run one bounded VLM call and return a valid retrieval-only hint if available.

    Subject selection is intentionally complete before this helper is called.
    Its output never changes the question passed to the calculator or generator,
    and every adapter/contract failure returns the unchanged text-only route.
    """
    if image_bytes is None:
        return (
            _with_vision_observability(
                build_vision_envelope(
                    image=image_metadata,
                    subject=selected_subject,
                    question=question.strip(),
                ),
                vision_latency_ms=0.0,
                retrieval_augmented=False,
            ),
            None,
        )

    started_at = perf_counter()
    try:
        analysis = analyze_image(
            image_bytes=image_bytes,
            content_type=str(image_metadata.get("content_type") or "application/octet-stream"),
            subject=selected_subject,
            question=question.strip(),
        )
        if not isinstance(analysis, VisionAnalysis):
            analysis = _vision_failure()
    except Exception:  # Provider failures must not turn an image hint into a 500.
        analysis = _vision_failure()
    vision_latency_ms = round((perf_counter() - started_at) * 1000.0, 3)

    try:
        vision = build_vision_envelope(
            image=image_metadata,
            subject=selected_subject,
            question=question.strip(),
            image_bytes=image_bytes,
            analysis=analysis,
        )
        context = build_temporary_vision_context(
            selected_subject=selected_subject,
            image_sha256=str(image_metadata["sha256"]),
            analysis=analysis,
        )
        if context is None:
            return (
                _with_vision_observability(
                    vision,
                    vision_latency_ms=vision_latency_ms,
                    retrieval_augmented=False,
                ),
                None,
            )
        return (
            _with_vision_observability(
                vision,
                vision_latency_ms=vision_latency_ms,
                retrieval_augmented=True,
            ),
            augment_retrieval_query(
                question,
                context,
                selected_subject=selected_subject,
            ),
        )
    except (KeyError, TypeError, ValueError, VisionFusionError):
        # An invalid contract/result can never influence retrieval.
        return (
            _with_vision_observability(
                build_vision_envelope(
                    image=image_metadata,
                    subject=selected_subject,
                    question=question.strip(),
                    image_bytes=image_bytes,
                    analysis=_vision_failure(),
                ),
                vision_latency_ms=vision_latency_ms,
                retrieval_augmented=False,
            ),
            None,
        )


def _retrieval_query_kwargs(
    answer_callable: Callable[..., object],
    retrieval_query: str | None,
) -> dict[str, str]:
    """Pass a hint only to a callable that explicitly supports the optional kwarg.

    This preserves legacy service fakes and adapters while real subject services
    receive a bounded hint only after the checked fusion gate has accepted it.
    """
    if retrieval_query is None:
        return {}
    try:
        parameters = inspect.signature(answer_callable).parameters.values()
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return {"retrieval_query": retrieval_query}
    if any(parameter.name == "retrieval_query" for parameter in parameters):
        return {"retrieval_query": retrieval_query}
    return {}


def _abstain_on_cross_subject_citation(
    payload: dict[str, object],
    *,
    selected_subject: str,
) -> dict[str, object]:
    """Fail closed when a service response exposes a citation outside its locked subject."""
    if payload.get("status") != "ok":
        return payload
    raw_citations = payload.get("citations") or payload.get("sources")
    if not isinstance(raw_citations, list):
        return payload
    allowed_prefix = f"rag/data/{selected_subject.lower()}/"
    for citation in raw_citations:
        if isinstance(citation, str):
            note_path = citation
        elif isinstance(citation, dict):
            note_path = str(citation.get("note_path") or "")
        else:
            continue
        if note_path and not note_path.startswith(allowed_prefix):
            raw_trace = payload.get("trace")
            trace = list(raw_trace) if isinstance(raw_trace, list) else []
            trace.append(
                {
                    "stage": "citation_scope",
                    "status": "rejected_cross_subject",
                }
            )
            return {
                **payload,
                "status": "abstain",
                "answer": "I don't know.",
                "sources": [],
                "citations": [],
                "abstain_reason": "cross_subject_citation",
                "trace": trace,
            }
    return payload


# E2E receipt hook: the registered evaluation runner constructs and appends the
# existing metadata-only receipt after its terminal answer.  Interactive API
# requests do not yet carry the schema-required receipt/trace identity, phase
# timings, or a status mapping for calculator and needs-subject branches, so
# they must not append evidence.


@app.get("/health")
def health() -> dict[str, str]:
    """Keep the original simple health-check route for compatibility."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Answer through the original shared-index endpoint when explicitly requested."""
    from rag.pipeline import query as rag_query

    result = rag_query(request.question, _legacy_chain())
    return QueryResponse(answer=result["answer"], sources=result["sources"])


@app.get("/")
def unified_ui() -> FileResponse:
    """Serve the Workstream 4 unified browser client."""
    return FileResponse(UNIFIED_UI_DIR / "index.html")


@app.get("/quiz")
def quiz_ui() -> RedirectResponse:
    """Redirect the legacy Quiz entry to the canonical browser page."""
    return RedirectResponse(url="/", status_code=307)


@app.get("/preview")
def preview_ui() -> FileResponse:
    """Serve the Full Engineering Demo v2 (Integration v1) browser client."""
    return FileResponse(UNIFIED_UI_DIR / "index.html")


@app.get("/integration-v0")
def legacy_preview_ui() -> FileResponse:
    """Serve the archived Integration v0 preview browser client."""
    return FileResponse(INTEGRATION_UI_DIR / "index.html")


@app.get("/v1")
def unified_ui_alias() -> FileResponse:
    """Alias for the unified v1 browser client."""
    return FileResponse(UNIFIED_UI_DIR / "index.html")


@app.get("/api/v0/health")
def preview_health() -> dict[str, object]:
    """Report eight-subject preview readiness without loading a language model."""
    subjects = sorted(load_subject_catalog())
    states = [_preview_service.index_status(subject) for subject in subjects]
    return {
        "status": "ok",
        "mode": "preview",
        "corpus_root": corpus_root().as_posix(),
        "index_root": INDEX_ROOT.as_posix(),
        "ready_subject_count": sum(state.ready for state in states),
        "subject_count": len(states),
        "preview_index_input_sha256": {
            subject: (state.input_sha256 if state.ready else None)
            for subject, state in zip(subjects, states, strict=True)
        },
        "web_handoff_origins": _web_handoff_origins(),
        "vlm_provider": os.getenv("EEE_VLM_PROVIDER", "none"),
    }


@app.get("/api/v0/subjects")
def preview_subjects() -> dict[str, list[dict[str, object]]]:
    """Expose card-safe catalog and current preview-index status for every subject."""
    response: list[dict[str, object]] = []
    for subject, record in sorted(load_subject_catalog().items()):
        index_status = _preview_service.index_status(subject)
        response.append(
            {
                "subject": record.subject,
                "name": record.name,
                "module": record.module,
                "year": record.year,
                "formal_note_count": record.formal_note_count,
                "review_status_counts": record.review_status_counts,
                "preview_ready": index_status.ready,
                "preview_reason": index_status.reason,
            }
        )
    return {"subjects": response}


@app.get("/api/v0/models")
def preview_models() -> dict[str, object]:
    """Expose only the local models explicitly enabled for browser selection."""
    return {"default_model": default_model_id(), "models": model_options()}


@app.post("/api/v0/answer")
def preview_answer(
    question: Annotated[str, Form()],
    subject: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    mode: Annotated[str, Form()] = "preview",
    image: Annotated[UploadFile | None, File()] = None,
) -> dict[str, object]:
    """Answer a subject-scoped request; subject may be routed when omitted."""
    image_metadata, image_bytes = _read_upload(image)
    selected_mode = (mode or "preview").strip().lower()
    if selected_mode not in {"preview", "v1"}:
        raise HTTPException(status_code=422, detail="mode must be preview or v1")
    try:
        decision = route_subject(question, explicit_subject=subject)
        if decision.status == "reject":
            raise HTTPException(
                status_code=404,
                detail=f"Unknown subject: {subject!r}",
            )
        if decision.status in {"clarify", "multi_subject_fallback"}:
            return {
                "status": "needs_subject",
                "mode": selected_mode,
                "subject": None,
                "answer": (
                    "Please choose a subject. Top candidates: " + ", ".join(decision.candidates)
                ),
                "sources": [],
                "citations": [],
                "route": "subject_router",
                "routing": {
                    "status": decision.status,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "candidates": list(decision.candidates),
                    "scores": decision.scores,
                },
                "image": image_metadata,
            }
        assert decision.subject is not None
        selected_subject = resolve_subject(decision.subject).subject
        vision, retrieval_query = _vision_and_retrieval_query(
            image_metadata=image_metadata,
            image_bytes=image_bytes,
            selected_subject=selected_subject,
            question=question,
        )
        selected_model = resolve_model_id(model)
        if selected_mode == "v1":
            if _retrieval_query_kwargs(_v1_service.answer, retrieval_query):
                payload = _v1_service.answer(
                    selected_subject,
                    question,
                    model=selected_model,
                    retrieval_query=retrieval_query,
                )
            else:
                payload = _v1_service.answer(selected_subject, question, model=selected_model)
        else:
            if _retrieval_query_kwargs(_preview_service.answer, retrieval_query):
                payload = _preview_service.answer(
                    selected_subject,
                    question,
                    model=selected_model,
                    retrieval_query=retrieval_query,
                )
            else:
                payload = _preview_service.answer(selected_subject, question, model=selected_model)
        payload = _abstain_on_cross_subject_citation(
            payload,
            selected_subject=selected_subject,
        )
        payload["model"] = selected_model
        payload["routing"] = {
            "status": decision.status,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "candidates": list(decision.candidates),
        }
    except (SubjectPreviewError, SubjectIndexError) as error:
        message = str(error)
        if message.startswith("Unknown subject"):
            raise HTTPException(status_code=404, detail=message) from error
        if "index is unavailable" in message or "no production-eligible notes" in message:
            raise HTTPException(
                status_code=409,
                detail={"code": "index_not_ready", "message": message},
            ) from error
        raise HTTPException(status_code=422, detail=message) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    if image_metadata["status"] != "not_provided":
        image_metadata = {
            **image_metadata,
            "status": "processed" if vision["status"] == "processed" else str(vision["status"]),
        }

    citations = payload.get("citations")
    if isinstance(citations, list) and citations:
        payload["sources"] = citations
    else:
        raw_sources = payload.get("sources")
        sources = raw_sources if isinstance(raw_sources, list) else []
        payload["sources"] = [{"note_path": source, "heading": None} for source in sources]
    payload["image"] = image_metadata
    payload["vision"] = vision
    return payload


@app.get("/api/v1/health")
def unified_health() -> dict[str, object]:
    """Report unified-agent readiness for preview and formal v1 indexes."""
    subjects = sorted(load_subject_catalog())
    preview_states = [_preview_service.index_status(code) for code in subjects]
    v1_states = [_v1_service.index_status(code) for code in subjects]
    return {
        "status": "ok",
        "mode": "unified",
        "schema_version": "1.0",
        "corpus_root": corpus_root().as_posix(),
        "index_root": INDEX_ROOT.as_posix(),
        "v1_index_root": V1_INDEX_ROOT.as_posix(),
        "ready_subject_count": sum(state.ready for state in preview_states),
        "v1_ready_subject_count": sum(state.ready for state in v1_states),
        "subject_count": len(subjects),
        "preview_index_input_sha256": {
            subject: (state.input_sha256 if state.ready else None)
            for subject, state in zip(subjects, preview_states, strict=True)
        },
        "v1_index_input_sha256": {
            subject: (state.input_sha256 if state.ready else None)
            for subject, state in zip(subjects, v1_states, strict=True)
        },
        "web_handoff_origins": _web_handoff_origins(),
        "vlm_provider": os.getenv("EEE_VLM_PROVIDER", "none"),
    }


@app.get("/api/v1/subjects")
def unified_subjects() -> dict[str, list[dict[str, object]]]:
    """Expose catalog plus preview/v1 index readiness for every subject."""
    response: list[dict[str, object]] = []
    for subject, record in sorted(load_subject_catalog().items()):
        preview_status = _preview_service.index_status(subject)
        v1_status = _v1_service.index_status(subject)
        response.append(
            {
                "subject": record.subject,
                "name": record.name,
                "module": record.module,
                "year": record.year,
                "formal_note_count": record.formal_note_count,
                "review_status_counts": record.review_status_counts,
                "preview_ready": preview_status.ready,
                "preview_reason": preview_status.reason,
                "v1_ready": v1_status.ready,
                "v1_reason": v1_status.reason,
            }
        )
    return {"subjects": response}


@app.get("/api/v1/models")
def unified_models() -> dict[str, object]:
    """Expose the same allowlisted local models as the preview path."""
    return {"default_model": default_model_id(), "models": model_options()}


@app.post("/api/v1/answer")
def unified_answer(
    question: Annotated[str, Form()],
    subject: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    mode: Annotated[str, Form()] = "v1",
    image: Annotated[UploadFile | None, File()] = None,
    conversation_id: Annotated[str | None, Form()] = None,
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Answer through the unified engineering agent contract."""
    bound_conversation_id: str | None = None
    if session is not None:
        bound_conversation_id = bind_or_validate_conversation(session, conversation_id)
    prior_context = conversation_context(session)
    image_metadata, image_bytes = _read_upload(image)
    if image_bytes is not None:
        try:
            validate_image_bytes(
                image_bytes,
                content_type=str(image_metadata["content_type"]),
            )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail="Image bytes are corrupt, mismatched, or exceed resource limits.",
            ) from error
    try:
        decision = route_subject(question, explicit_subject=subject)
        if decision.status == "reject":
            raise UnifiedAgentError(f"Unknown subject: {subject!r}")
        request = AgentRequest(
            question=question,
            subject=subject,
            mode=mode,
            model=model,
            image_provided=image_bytes is not None,
            conversation_context=prior_context,
        )
        if decision.status in {"clarify", "multi_subject_fallback"}:
            response = _unified_agent.answer(request)
        else:
            assert decision.subject is not None
            selected_subject = decision.subject
            vision, retrieval_query = _vision_and_retrieval_query(
                image_metadata=image_metadata,
                image_bytes=image_bytes,
                selected_subject=selected_subject,
                question=question,
            )
            response = _unified_agent.answer(
                AgentRequest(
                    question=question,
                    subject=selected_subject,
                    mode=mode,
                    model=model,
                    image_provided=image_bytes is not None,
                    conversation_context=prior_context,
                ),
                **_retrieval_query_kwargs(_unified_agent.answer, retrieval_query),
            )
    except UnifiedAgentError as error:
        message = str(error)
        if message.startswith("Unknown subject"):
            raise HTTPException(status_code=404, detail=message) from error
        if "index is unavailable" in message or "no production-eligible notes" in message:
            raise HTTPException(
                status_code=409,
                detail={"code": "index_not_ready", "message": message},
            ) from error
        raise HTTPException(status_code=422, detail=message) from error

    payload = response.as_dict()
    if response.status == "needs_subject":
        payload["image"] = image_metadata
        if session is not None and bound_conversation_id is not None:
            payload["conversation_id"] = bound_conversation_id
            record_conversation_turn(session, question, str(payload.get("answer", "")))
        return payload

    payload = _abstain_on_cross_subject_citation(
        payload,
        selected_subject=selected_subject,
    )
    payload["routing"] = {
        "status": decision.status,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "candidates": list(decision.candidates),
        "scores": decision.scores,
    }
    if image_metadata["status"] != "not_provided":
        image_metadata = {
            **image_metadata,
            "status": "processed" if vision["status"] == "processed" else str(vision["status"]),
        }
    payload["image"] = image_metadata
    payload["vision"] = vision
    if session is not None and bound_conversation_id is not None:
        payload["conversation_id"] = bound_conversation_id
        record_conversation_turn(session, question, str(payload.get("answer", "")))
    return payload


@app.post("/api/v1/quiz/start", status_code=201)
def quiz_start(
    request: QuizStartRequest,
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Start one session-owned five-question Quiz and generate only question 1."""
    record = _required_quiz_session(session)
    if record.quiz_state is not None and record.quiz_state.status == "in_progress":
        raise HTTPException(
            status_code=409,
            detail={"code": "quiz_active", "message": "An unfinished Quiz is already active."},
        )
    if request.question_count not in {1, 3, 5, 7, 10}:
        raise HTTPException(
            status_code=422, detail="Quiz question_count must be 1, 3, 5, 7, or 10."
        )
    if request.mode not in {"preview", "v1"}:
        raise HTTPException(status_code=422, detail="Quiz mode must be preview or v1.")
    try:
        selected_subject, topics = _quiz_scope_catalog(request.subject, request.mode)
        selected_scope: QuizScopeSelection = resolve_scope(
            topics,
            topic_id=request.topic_id,
            knowledge_point_id=request.knowledge_point_id,
        )
        fact_pool = tuple(focus for topic in topics for focus in topic.focuses)
        selected_model = QUIZ_MODEL_ID
        scope_key = (
            request.mode,
            selected_subject,
            request.topic_id,
            request.knowledge_point_id,
        )
        excluded_fact_ids = (
            record.previous_quiz_fact_ids(scope_key) if request.mode == "preview" else ()
        )
        focus_plan = _plan_quiz_focuses(
            selected_scope.focuses,
            fact_pool,
            request.question_count,
            excluded_fact_ids=excluded_fact_ids,
            rng=_quiz_plan_rng if request.mode == "preview" else None,
        )
    except Exception as error:
        _raise_quiz_generation_error(error)
    reservation = record.reserve_quiz_start()
    if reservation is None:
        if record.quiz_state is not None and record.quiz_state.status == "in_progress":
            raise HTTPException(
                status_code=409,
                detail={"code": "quiz_active", "message": "An unfinished Quiz is already active."},
            )
        raise HTTPException(
            status_code=409,
            detail={
                "code": "quiz_start_in_flight",
                "message": "A Quiz is already being started for this session.",
            },
        )
    try:
        try:
            question = _generate_quiz_question(
                subject=selected_subject,
                mode=request.mode,
                model=selected_model,
                retrieval_query=selected_scope.retrieval_query,
                display_scope=selected_scope.retrieval_query,
                focus=focus_plan[0],
                fact_pool=fact_pool,
                number=1,
                preview_service=_preview_service,
                v1_service=_v1_service,
                prior_fact_ids=excluded_fact_ids,
            )
        except Exception as error:
            _raise_quiz_generation_error(error)
        quiz = QuizState(
            owner_session_id=record.session_id,
            subject=selected_subject,
            mode=request.mode,
            model=selected_model,
            topic_label=selected_scope.topic_label,
            knowledge_point_label=selected_scope.knowledge_point_label,
            retrieval_query=selected_scope.retrieval_query,
            focus_plan=focus_plan,
            fact_pool=fact_pool,
            current=QuizQuestionState(question=question),
            scope_key=scope_key,
            excluded_fact_ids=excluded_fact_ids,
            requested_total=request.question_count,
        )
        if not record.install_quiz_start(reservation, quiz):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "quiz_start_stale",
                    "message": "The session changed while the Quiz was starting.",
                },
            )
        return quiz.public_envelope()
    finally:
        record.release_quiz_start(reservation)


@app.get("/api/v1/quiz/scopes")
def quiz_scopes(
    subject: str,
    mode: str,
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Return one authenticated subject/mode note-derived Quiz scope catalog."""
    _required_quiz_session(session)
    if mode not in {"preview", "v1"}:
        raise HTTPException(status_code=422, detail="Quiz mode must be preview or v1.")
    try:
        selected_subject, topics = _quiz_scope_catalog(subject, mode)
    except Exception as error:
        _raise_quiz_generation_error(error)
    topics_payload = _quiz_topics_payload(topics, reserve_repeat=mode == "preview")
    return {
        "subject": selected_subject,
        "mode": mode,
        "topics": topics_payload,
    }


@app.get("/api/v1/quiz/current")
def quiz_current(
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Recover the current public question, revealed feedback, or retained summary."""
    return _quiz_state(session).public_envelope()


@app.get("/api/v1/quiz/review/{number}")
def quiz_review(
    number: int,
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Return one retained read-only question only after the Quiz is finished."""
    quiz = _quiz_state(session)
    try:
        return quiz.review(number)
    except QuizStateError as error:
        status_code = 409 if quiz.status != "finished" else 404
        raise HTTPException(status_code=status_code, detail=str(error)) from error


@app.post("/api/v1/quiz/current/submit")
def quiz_submit(
    request: QuizSubmitRequest,
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Grade the first valid answer for the current question exactly once."""
    quiz = _quiz_state(session)
    try:
        return quiz.submit(question_id=request.question_id, option_id=request.option_id)
    except QuizStateError as error:
        status_code = 409 if "already" in str(error) else 422
        raise HTTPException(status_code=status_code, detail=str(error)) from error


@app.post("/api/v1/quiz/current/next")
def quiz_next(
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Generate exactly one next question after explicit, valid submission."""
    record = _required_quiz_session(session)
    quiz = _quiz_state(record)
    if quiz.status != "in_progress":
        raise HTTPException(status_code=409, detail="Quiz is already finished.")
    if not quiz.current.submitted:
        raise HTTPException(status_code=409, detail="Submit the current question before Next.")
    if quiz.generated_count >= quiz.requested_total:
        raise HTTPException(status_code=409, detail="Finish after the final requested question.")
    try:
        quiz.begin_next()
    except QuizStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    try:
        try:
            question = _generate_quiz_question(
                subject=quiz.subject,
                mode=quiz.mode,
                model=quiz.model,
                retrieval_query=quiz.retrieval_query,
                display_scope=quiz.retrieval_query,
                focus=quiz.focus_plan[quiz.generated_count],
                fact_pool=quiz.fact_pool,
                number=quiz.generated_count + 1,
                preview_service=_preview_service,
                v1_service=_v1_service,
                prior_prompts=tuple(quiz.generated_prompt_texts),
                prior_fact_ids=quiz.excluded_fact_ids + tuple(quiz.generated_fact_ids),
            )
        except Exception as error:
            _raise_quiz_generation_error(error)
        if record.invalidated or record.expires_at <= time.time():
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "not_authenticated",
                    "message": "A valid demo session is required.",
                },
            )
        if record.quiz_state is not quiz or quiz.status != "in_progress":
            raise HTTPException(status_code=409, detail="The Quiz changed during generation.")
        try:
            quiz.advance(question)
            record.record_quiz_fact(quiz)
        except QuizStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
    finally:
        quiz.end_next()
    return quiz.public_envelope()


@app.post("/api/v1/quiz/current/finish")
def quiz_finish(
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, object]:
    """Finish the current Quiz early or after all five questions."""
    quiz = _quiz_state(session)
    if quiz.status == "finished":
        return quiz.summary()
    return quiz.finish()


@app.post("/api/v1/quiz/current/exit")
def quiz_exit(
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, str]:
    """Discard the current Quiz while keeping the authenticated session."""
    record = _required_quiz_session(session)
    record.invalidate_quiz()
    return {"status": "exited"}


@app.get("/api/v1/demo/circuit-gallery", dependencies=[Depends(demo_session)])
@app.get("/api/v1/demo/local-note-gallery", dependencies=[Depends(demo_session)])
def public_gallery_manifest() -> dict[str, object]:
    """Keep excluded reference-image galleries explicitly empty."""
    return {"assets": [], "note": "Reference-image datasets are not included."}


app.mount(
    "/static/integration-v0",
    StaticFiles(directory=INTEGRATION_UI_DIR, check_dir=True),
    name="integration-v0-static",
)
app.mount(
    "/static/integration-v1",
    StaticFiles(directory=UNIFIED_UI_DIR, check_dir=True),
    name="integration-v1-static",
)
