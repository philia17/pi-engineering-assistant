"""Unified request/response contract for CLI, API, UI and benchmarks (FP-401)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CONTRACT_SCHEMA_VERSION = "1.0"
SUPPORTED_MODES = frozenset({"preview", "v1"})
ResponseStatus = Literal["ok", "abstain", "needs_subject", "needs_input", "error"]


@dataclass(frozen=True)
class AgentRequest:
    """One unified engineering-agent question."""

    question: str
    subject: str | None = None
    mode: str = "v1"
    model: str | None = None
    image_provided: bool = False
    conversation_context: tuple[tuple[str, str], ...] = ()

    def cleaned_question(self) -> str:
        """Return a stripped question for routing and retrieval."""
        return self.question.strip()


@dataclass
class AgentResponse:
    """Stable response shared by CLI, API, UI and benchmark adapters."""

    status: ResponseStatus
    mode: str
    answer: str
    subject: str | None
    module: str | None = None
    model: str | None = None
    model_requested: str | None = None
    answer_engine: dict[str, str] | None = None
    route: str | None = None
    sources: list[dict[str, str | None]] = field(default_factory=list)
    citations: list[dict[str, str | None]] = field(default_factory=list)
    routing: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    abstain_reason: str | None = None
    calculator_values: dict[str, Any] | None = None
    timing_ms: float | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION
    image: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    calculator_steps: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for JSON transport without dropping explicit nulls needed by clients."""
        payload = asdict(self)
        if self.calculator_values is None:
            payload.pop("calculator_values", None)
        if self.calculator_steps is None:
            payload.pop("calculator_steps", None)
        if self.image is None:
            payload.pop("image", None)
        if self.vision is None:
            payload.pop("vision", None)
        return payload


def normalize_citations(
    sources: list[Any] | None,
    citations: list[Any] | None = None,
) -> list[dict[str, str | None]]:
    """Coerce string or object citations into note_path/heading pairs."""
    raw = citations if citations else sources or []
    normalized: list[dict[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for item in raw:
        if isinstance(item, str):
            note_path = item.strip()
            heading = None
        elif isinstance(item, dict):
            note_path = str(item.get("note_path", "")).strip()
            heading_raw = item.get("heading")
            heading = str(heading_raw).strip() if isinstance(heading_raw, str) else None
            if heading == "":
                heading = None
        else:
            continue
        if not note_path:
            continue
        key = (note_path, heading)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"note_path": note_path, "heading": heading})
    return normalized


def review_badge(review: dict[str, Any] | None, *, mode: str) -> str:
    """Return the UI-facing review method badge for one response."""
    payload = review or {}
    if mode == "preview":
        return "preview"
    warnings = payload.get("warnings") or []
    if any("preview" in str(item) for item in warnings):
        return "preview"
    counts = payload.get("review_status_counts") or {}
    if isinstance(counts, dict) and counts.get("reviewed"):
        return "machine-reviewed"
    if payload.get("eligibility_policy"):
        return "machine-reviewed"
    return "technical-review"
