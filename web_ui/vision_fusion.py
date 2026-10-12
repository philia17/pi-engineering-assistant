"""Bounded, non-citable fusion of validated VLM observations for retrieval.

This module deliberately has no dependency on a vector store, model provider, or
FastAPI request.  FI-011 may use its pure output to augment a *single* retrieval
query, but it must keep course-note citations exclusively in the normal RAG
path.  User-image observations are untrusted request-local hints, never course
source evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fcntl import LOCK_EX, LOCK_UN, flock
from functools import lru_cache
from pathlib import Path
from typing import Literal, NoReturn

import jsonschema

from web_ui.live_vision_contract import LIVE_VISION_SCHEMA_VERSION
from web_ui.vision_adapter import VISION_SCHEMA_VERSION, VisionAnalysis

FUSION_SCHEMA_VERSION: str = "integration_v0.vision_fusion.v1"
E2E_RECEIPT_SCHEMA_VERSION: str = "integration_v0.vision_e2e_receipt.v1"
USER_IMAGE_EVIDENCE_KIND: str = "user_image_observation"
SUBJECT_CODES: frozenset[str] = frozenset({"PE", "IS", "EM", "SEP", "EECC", "EPC", "MMT", "DIES"})
NO_PERSISTENCE_BOUNDARY: dict[str, str] = {
    "user_image_bytes": "not_persisted",
    "vlm_observation": "request_memory_only",
    "rag_data": "not_written",
    "subject_indexes": "not_rebuilt",
    "course_citations": "not_created_from_user_image",
    "receipt_payload": "hashes_and_metadata_only",
}
OBSERVATION_FIELDS: tuple[str, ...] = (
    "visible_labels",
    "components",
    "connections",
    "polarity_or_direction",
    "plot_readings",
    "uncertainties",
)
FIELD_LABELS: dict[str, str] = {
    "visible_labels": "Visible labels",
    "components": "Observed components",
    "connections": "Observed connections",
    "polarity_or_direction": "Observed polarity or direction",
    "plot_readings": "Observed plot readings",
    "uncertainties": "VLM uncertainty",
}
MAX_ITEMS_PER_FIELD: int = 8
MAX_ITEM_CHARACTERS: int = 240
MAX_CONTEXT_CHARACTERS: int = 2_000
MAX_QUESTION_CHARACTERS: int = 4_000
MAX_AUGMENTED_QUERY_CHARACTERS: int = MAX_QUESTION_CHARACTERS + MAX_CONTEXT_CHARACTERS + 2
_SHA256_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
_MACHINE_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,299}$")
_PROVIDER_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9][a-z0-9._-]{1,99}$")
_ERROR_CODE_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_RECEIPT_STATUSES: frozenset[str] = frozenset(
    {"ok", "abstain", "needs_subject", "degraded_text_only", "vision_unavailable", "error"}
)
_VISION_STATUSES: frozenset[str] = frozenset(
    {"not_requested", "processed", "vision_unavailable", "invalid_observation", "error"}
)
E2E_RECEIPT_SCHEMA_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "rag"
    / "evals"
    / "integration_v0"
    / "schemas"
    / "vision_e2e_receipt.schema.json"
)
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
E2E_RECEIPT_ROOT: Path = REPO_ROOT / "rag" / "evals" / "integration_v0" / "e2e_receipts"
_FORBIDDEN_RECEIPT_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "rag" / "data",
    REPO_ROOT / "artifacts" / "integration_v0" / "indexes",
    REPO_ROOT / "artifacts" / "integration_v1" / "indexes",
)
_COMPATIBLE_TOP_LEVEL_STATUSES: dict[str, frozenset[str]] = {
    "processed": frozenset({"ok", "abstain", "error"}),
    "not_requested": frozenset({"ok", "abstain", "needs_subject", "error"}),
    "vision_unavailable": frozenset(
        {"degraded_text_only", "vision_unavailable", "abstain", "error"}
    ),
    "invalid_observation": frozenset(
        {"degraded_text_only", "vision_unavailable", "abstain", "error"}
    ),
    "error": frozenset({"degraded_text_only", "vision_unavailable", "abstain", "error"}),
}


class VisionFusionError(ValueError):
    """Raised when an observation or a receipt would cross a contract boundary."""


@dataclass(frozen=True)
class TemporaryVisionContext:
    """A bounded query hint whose evidence is explicitly non-citable."""

    schema_version: str
    subject: str
    image_sha256: str
    observation_sha256: str
    query_context: str
    included_item_counts: dict[str, int]
    truncated_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Serialize the contract without exposing any course-source citation."""
        return {
            "schema_version": self.schema_version,
            "status": "applied",
            "evidence": {
                "kind": USER_IMAGE_EVIDENCE_KIND,
                "trust": "untrusted",
                "course_source_evidence": False,
                "citation_policy": "vision_observation_is_not_citable",
                "persistence": "memory_only_not_added_to_corpus",
                "image_sha256": self.image_sha256,
                "observation_sha256": self.observation_sha256,
            },
            "retrieval": {
                "subject": self.subject,
                "query_context": self.query_context,
                "max_context_characters": MAX_CONTEXT_CHARACTERS,
                "included_item_counts": dict(self.included_item_counts),
                "truncated_fields": list(self.truncated_fields),
            },
            "sources": [],
            "citations": [],
        }


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of canonical JSON without writing the value anywhere."""
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise VisionFusionError("value must be canonical-JSON serializable") from error
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 of UTF-8 text for a receipt-safe identity."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: str | None, *, field: str, allow_none: bool = False) -> str | None:
    """Validate a lowercase SHA-256 value used as a content identity."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise VisionFusionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: str, *, field: str) -> str:
    """Validate a bounded machine-readable identifier."""
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise VisionFusionError(f"{field} must be a 3-128 character identifier")
    return value


def _require_subject(value: str) -> str:
    """Validate one canonical subject code without selecting a storage path."""
    if value not in SUBJECT_CODES:
        raise VisionFusionError(f"subject must be one of {', '.join(sorted(SUBJECT_CODES))}")
    return value


def _require_machine_id(value: str | None, *, field: str, allow_none: bool) -> str | None:
    """Validate bounded provider/model/revision-style machine identifiers."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not _MACHINE_ID_PATTERN.fullmatch(value):
        raise VisionFusionError(f"{field} must be a machine identifier")
    return value


def _require_provider_id(
    value: str | None,
    *,
    field: str,
    allow_none: bool,
) -> str | None:
    """Validate a lowercase served-model provider identifier."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not _PROVIDER_ID_PATTERN.fullmatch(value):
        raise VisionFusionError(f"{field} must be a machine identifier")
    return value


def _require_error_code(value: str | None, *, required: bool) -> str | None:
    """Validate a sanitized stable error code without persisting error text."""
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _ERROR_CODE_PATTERN.fullmatch(value):
        raise VisionFusionError("vision error_code must be a sanitized machine code")
    return value


def _clean_observation_item(value: str) -> str:
    """Bound one VLM string while preserving its text-only observation role."""
    cleaned = " ".join(value.replace("\x00", " ").split())
    return cleaned[:MAX_ITEM_CHARACTERS]


def _validated_observation(
    observation: Mapping[str, object],
) -> tuple[dict[str, list[str]], set[str]]:
    """Validate the provider-neutral observation shape and apply item bounds."""
    if not isinstance(observation, Mapping):
        raise VisionFusionError("observation must be a mapping")
    if set(observation) != set(OBSERVATION_FIELDS):
        raise VisionFusionError("observation must contain only the six contract fields")
    normalized: dict[str, list[str]] = {}
    truncated_fields: set[str] = set()
    for field in OBSERVATION_FIELDS:
        raw_values = observation.get(field)
        if not isinstance(raw_values, list) or not all(
            isinstance(item, str) for item in raw_values
        ):
            raise VisionFusionError(f"observation.{field} must be a list of strings")
        cleaned_values: list[str] = []
        for raw_value in raw_values:
            cleaned_value = _clean_observation_item(raw_value)
            if not cleaned_value:
                truncated_fields.add(field)
                continue
            if cleaned_value != raw_value:
                truncated_fields.add(field)
            cleaned_values.append(cleaned_value)
        if len(cleaned_values) > MAX_ITEMS_PER_FIELD:
            truncated_fields.add(field)
        normalized[field] = cleaned_values[:MAX_ITEMS_PER_FIELD]
    return normalized, truncated_fields


def _bounded_context_lines(
    observation: Mapping[str, list[str]],
    *,
    truncated_fields: set[str],
) -> tuple[str, dict[str, int], tuple[str, ...]]:
    """Render a short, attribution-preserving context under the hard character cap."""
    prefix = (
        "[Temporary untrusted user-image observation; retrieval hint only. "
        "It is not course-source evidence and cannot be cited.]"
    )
    lines: list[str] = [prefix]
    included_item_counts: dict[str, int] = {field: 0 for field in OBSERVATION_FIELDS}
    current_length = len(prefix)
    for field in OBSERVATION_FIELDS:
        values = observation[field]
        if not values:
            continue
        remaining = MAX_CONTEXT_CHARACTERS - current_length - 1
        if remaining <= 0:
            truncated_fields.add(field)
            continue
        selected_values: list[str] = []
        for value in values:
            candidate = f"{FIELD_LABELS[field]}: " + "; ".join([*selected_values, value])
            if len(candidate) > remaining:
                truncated_fields.add(field)
                break
            selected_values.append(value)
        if not selected_values:
            truncated_fields.add(field)
            continue
        line = f"{FIELD_LABELS[field]}: " + "; ".join(selected_values)
        lines.append(line)
        included_item_counts[field] = len(selected_values)
        current_length += len(line) + 1
        if len(selected_values) < len(values):
            truncated_fields.add(field)
    context = "\n".join(lines)
    if len(context) > MAX_CONTEXT_CHARACTERS:
        raise VisionFusionError("bounded fusion context exceeded its maximum length")
    return context, included_item_counts, tuple(sorted(truncated_fields))


def _build_context_from_observation(
    *,
    subject: str,
    image_sha256: str,
    observation: Mapping[str, object],
) -> TemporaryVisionContext:
    """Build a context after the adapter-aware public gate has accepted the result."""
    normalized_subject = _require_subject(subject)
    normalized_image_sha256 = _require_sha256(image_sha256, field="image_sha256")
    assert normalized_image_sha256 is not None
    validated_observation, truncated_fields = _validated_observation(observation)
    context, item_counts, all_truncated_fields = _bounded_context_lines(
        validated_observation,
        truncated_fields=truncated_fields,
    )
    return TemporaryVisionContext(
        schema_version=FUSION_SCHEMA_VERSION,
        subject=normalized_subject,
        image_sha256=normalized_image_sha256,
        observation_sha256=canonical_json_sha256(observation),
        query_context=context,
        included_item_counts=item_counts,
        truncated_fields=all_truncated_fields,
    )


def _analysis_has_checked_success(analysis: VisionAnalysis) -> bool:
    """Return whether provider-specific upstream provenance proves success."""
    if (
        analysis.status != "processed"
        or analysis.observation is None
        or analysis.request_id is None
        or analysis.error_code is not None
        or analysis.provider_kind not in {"http", "mock", "ollama"}
    ):
        return False
    try:
        _require_provider_id(
            analysis.served_model_provider,
            field="served_model_provider",
            allow_none=False,
        )
        _require_machine_id(analysis.provider_model, field="provider_model", allow_none=False)
        _require_machine_id(analysis.model_revision, field="model_revision", allow_none=False)
        _require_identifier(analysis.request_id, field="vision_request_id")
    except VisionFusionError:
        return False
    if analysis.provider_kind == "http":
        return (
            analysis.checked_contract_version == LIVE_VISION_SCHEMA_VERSION
            and analysis.upstream_status == "ok"
        )
    return (
        analysis.checked_contract_version == VISION_SCHEMA_VERSION
        and analysis.upstream_status == "processed"
    )


def build_temporary_vision_context(
    *,
    selected_subject: str,
    image_sha256: str,
    analysis: VisionAnalysis,
) -> TemporaryVisionContext | None:
    """Return a context only for an exactly successful adapter result.

    A remote HTTP result additionally must have passed the strict live contract
    and carry the exact successful upstream status.  Every unavailable,
    malformed, mismatched, or failed result returns ``None`` so the caller can
    continue through the existing text-only path.
    """
    subject = _require_subject(selected_subject)
    if not _analysis_has_checked_success(analysis):
        return None
    observation = analysis.observation
    if observation is None:
        return None
    try:
        return _build_context_from_observation(
            subject=subject,
            image_sha256=image_sha256,
            observation=observation,
        )
    except VisionFusionError:
        return None


def augment_retrieval_query(
    question: str,
    context: TemporaryVisionContext,
    *,
    selected_subject: str,
) -> str:
    """Combine a user question with its bounded, explicitly non-citable hint."""
    subject = _require_subject(selected_subject)
    if subject != context.subject:
        raise VisionFusionError("selected subject does not match fusion context subject")
    cleaned_question = question.strip()
    if not cleaned_question:
        raise VisionFusionError("question must not be empty")
    if len(cleaned_question) > MAX_QUESTION_CHARACTERS:
        raise VisionFusionError(f"question exceeds {MAX_QUESTION_CHARACTERS} characters")
    augmented = f"{cleaned_question}\n\n{context.query_context}"
    if len(augmented) > MAX_AUGMENTED_QUERY_CHARACTERS:
        raise VisionFusionError("augmented retrieval query exceeds its fixed maximum")
    return augmented


def _require_timing_ms(value: Mapping[str, float | int | None]) -> dict[str, float | None]:
    """Validate a fixed phase-timing shape without recording raw request content."""
    expected_fields = {"vision", "retrieval", "answer", "total"}
    if set(value) != expected_fields:
        raise VisionFusionError("timing_ms must contain vision, retrieval, answer, and total")
    normalized: dict[str, float | None] = {}
    for field in sorted(expected_fields):
        raw_value = value[field]
        if raw_value is None and field != "total":
            normalized[field] = None
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise VisionFusionError(f"timing_ms.{field} must be a non-negative number")
        number = float(raw_value)
        if not math.isfinite(number) or number < 0:
            raise VisionFusionError(f"timing_ms.{field} must be a finite non-negative number")
        normalized[field] = number
    if normalized["total"] is None:
        raise VisionFusionError("timing_ms.total must be present")
    return normalized


def _validate_recorded_at(value: str) -> str:
    """Require a timezone-aware ISO-8601 timestamp supplied by the caller."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise VisionFusionError("recorded_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise VisionFusionError("recorded_at must include a timezone offset")
    return value


def build_e2e_receipt(
    *,
    receipt_id: str,
    recorded_at: str,
    request_sha256: str,
    subject: str,
    text_model: str,
    image_sha256: str | None,
    vision_analysis: VisionAnalysis | None,
    fusion_context: TemporaryVisionContext | None,
    retrieval_trace_id: str,
    retrieval_trace: Sequence[Mapping[str, object]],
    answer: str,
    status: Literal[
        "ok", "abstain", "needs_subject", "degraded_text_only", "vision_unavailable", "error"
    ],
    timing_ms: Mapping[str, float | int | None],
) -> dict[str, object]:
    """Build metadata-only evidence for one end-to-end request without persisting it.

    The caller may pass this object to :func:`append_e2e_receipt_jsonl` after the
    request completes.  Raw question text, image bytes, VLM observations, answer
    text, and citations remain outside the receipt; only their safe identities
    and boundary declarations are recorded.
    """
    normalized_subject = _require_subject(subject)
    normalized_image_sha256 = _require_sha256(
        image_sha256,
        field="image_sha256",
        allow_none=True,
    )
    if status not in _RECEIPT_STATUSES:
        raise VisionFusionError(f"unsupported receipt status: {status!r}")
    if vision_analysis is None:
        if normalized_image_sha256 is not None:
            raise VisionFusionError("an image hash requires a VisionAnalysis result")
        vision_status = "not_requested"
    else:
        if normalized_image_sha256 is None:
            raise VisionFusionError("VisionAnalysis requires an image hash")
        vision_status = vision_analysis.status
        if vision_status == "not_requested":
            raise VisionFusionError(
                "vision status not_requested requires no image and no VisionAnalysis"
            )
    if vision_status not in _VISION_STATUSES:
        raise VisionFusionError(f"unsupported vision status: {vision_status!r}")
    compatible_statuses = _COMPATIBLE_TOP_LEVEL_STATUSES[vision_status]
    if status not in compatible_statuses:
        raise VisionFusionError(
            f"top-level status {status!r} is incompatible with vision status {vision_status!r}"
        )
    if vision_status == "processed":
        assert vision_analysis is not None
        if fusion_context is None:
            raise VisionFusionError("processed vision must provide a validated fusion context")
        if vision_analysis.error_code is not None:
            raise VisionFusionError("processed vision must not carry an error_code")
        _require_provider_id(
            vision_analysis.served_model_provider,
            field="served_model_provider",
            allow_none=False,
        )
        _require_machine_id(
            vision_analysis.provider_model,
            field="provider_model",
            allow_none=False,
        )
        _require_machine_id(
            vision_analysis.model_revision,
            field="model_revision",
            allow_none=False,
        )
        _require_machine_id(
            vision_analysis.checked_contract_version,
            field="checked_contract_version",
            allow_none=False,
        )
        if vision_analysis.request_id is None:
            raise VisionFusionError("processed vision requires request_id")
        if not _analysis_has_checked_success(vision_analysis):
            raise VisionFusionError("processed vision lacks checked upstream success")
    elif fusion_context is not None:
        raise VisionFusionError("non-processed vision must not provide a fusion context")
    if vision_analysis is not None and vision_status != "processed":
        _require_error_code(vision_analysis.error_code, required=True)
    if fusion_context is not None:
        if fusion_context.schema_version != FUSION_SCHEMA_VERSION:
            raise VisionFusionError("fusion context schema version is unsupported")
        if fusion_context.subject != normalized_subject:
            raise VisionFusionError("fusion context subject does not match receipt subject")
        if normalized_image_sha256 != fusion_context.image_sha256:
            raise VisionFusionError("fusion context image hash does not match receipt image hash")
        assert vision_analysis is not None
        if fusion_context.observation_sha256 != canonical_json_sha256(vision_analysis.observation):
            raise VisionFusionError("fusion context observation hash does not match analysis")
    trace = list(retrieval_trace)
    if not all(isinstance(stage, Mapping) for stage in trace):
        raise VisionFusionError("retrieval_trace must contain mapping stages")
    receipt: dict[str, object] = {
        "schema_version": E2E_RECEIPT_SCHEMA_VERSION,
        "receipt_id": _require_identifier(receipt_id, field="receipt_id"),
        "recorded_at": _validate_recorded_at(recorded_at),
        "status": status,
        "request": {
            "sha256": _require_sha256(request_sha256, field="request_sha256"),
            "subject": normalized_subject,
            "text_model": _require_machine_id(
                text_model,
                field="text_model",
                allow_none=False,
            ),
        },
        "image": {
            "provided": normalized_image_sha256 is not None,
            "sha256": normalized_image_sha256,
        },
        "vision": {
            "status": vision_status,
            "adapter_kind": (
                _require_provider_id(
                    vision_analysis.provider_kind,
                    field="adapter_kind",
                    allow_none=False,
                )
                if vision_analysis is not None
                else None
            ),
            "provider": (
                _require_provider_id(
                    vision_analysis.served_model_provider,
                    field="served_model_provider",
                    allow_none=True,
                )
                if vision_analysis is not None
                else None
            ),
            "model": (
                _require_machine_id(
                    vision_analysis.provider_model,
                    field="provider_model",
                    allow_none=True,
                )
                if vision_analysis is not None
                else None
            ),
            "revision": (
                _require_machine_id(
                    vision_analysis.model_revision,
                    field="model_revision",
                    allow_none=True,
                )
                if vision_analysis is not None
                else None
            ),
            "checked_contract_version": (
                _require_machine_id(
                    vision_analysis.checked_contract_version,
                    field="checked_contract_version",
                    allow_none=True,
                )
                if vision_analysis is not None
                else None
            ),
            "request_id": (
                _require_identifier(vision_analysis.request_id, field="vision_request_id")
                if vision_analysis is not None and vision_analysis.request_id is not None
                else None
            ),
            "upstream_status": (
                _require_machine_id(
                    vision_analysis.upstream_status,
                    field="upstream_status",
                    allow_none=True,
                )
                if vision_analysis is not None
                else None
            ),
            "error_code": (
                _require_error_code(
                    vision_analysis.error_code,
                    required=vision_status != "processed",
                )
                if vision_analysis is not None
                else None
            ),
            "evidence_kind": USER_IMAGE_EVIDENCE_KIND if fusion_context else None,
            "observation_sha256": (
                fusion_context.observation_sha256 if fusion_context is not None else None
            ),
            "fusion_context_sha256": (
                sha256_text(fusion_context.query_context) if fusion_context is not None else None
            ),
        },
        "retrieval": {
            "trace_id": _require_identifier(retrieval_trace_id, field="retrieval_trace_id"),
            "trace_sha256": canonical_json_sha256(trace),
            "trace_stage_count": len(trace),
            "query_augmented_with_user_image": fusion_context is not None,
        },
        "answer": {"sha256": sha256_text(answer)},
        "timing_ms": _require_timing_ms(timing_ms),
        "no_persistence_boundary": dict(NO_PERSISTENCE_BOUNDARY),
    }
    validate_e2e_receipt(receipt)
    return receipt


@lru_cache(maxsize=1)
def _e2e_receipt_validator() -> jsonschema.Draft202012Validator:
    """Load the checked-in receipt schema only for explicit persistence validation."""
    try:
        schema = json.loads(E2E_RECEIPT_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VisionFusionError("E2E receipt schema is unreadable") from error
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as error:
        raise VisionFusionError("E2E receipt schema is invalid") from error
    return jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )


def validate_e2e_receipt(receipt: Mapping[str, object]) -> None:
    """Reject a receipt that is not the metadata-only versioned contract."""
    try:
        json.dumps(dict(receipt), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise VisionFusionError("E2E receipt must contain finite JSON values") from error
    errors = sorted(
        _e2e_receipt_validator().iter_errors(dict(receipt)),
        key=lambda error: list(error.path),
    )
    if errors:
        schema_error = errors[0]
        location = ".".join(str(part) for part in schema_error.absolute_path) or "root"
        raise VisionFusionError(f"E2E receipt schema failed at {location}: {schema_error.message}")


def _reject_non_finite_json(value: str) -> NoReturn:
    """Reject JavaScript-style NaN/Infinity constants accepted by stdlib defaults."""
    raise VisionFusionError(f"non-finite JSON constant: {value}")


def _receipt_path(path: Path, *, receipts_root: Path) -> Path:
    """Resolve a JSONL path inside a dedicated receipt root and outside indexes/data."""
    root = receipts_root.resolve()
    if root.name != "e2e_receipts":
        raise VisionFusionError("receipts_root must be a dedicated e2e_receipts directory")
    root_parts = root.parts
    in_rag_data = any(
        root_parts[index : index + 2] == ("rag", "data") for index in range(len(root_parts) - 1)
    )
    if in_rag_data or "indexes" in root_parts:
        raise VisionFusionError("receipt writes are forbidden under rag/data or indexes")
    for forbidden_root in _FORBIDDEN_RECEIPT_ROOTS:
        forbidden = forbidden_root.resolve()
        if root == forbidden or root.is_relative_to(forbidden):
            raise VisionFusionError("receipt writes are forbidden under rag/data or indexes")
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise VisionFusionError("E2E receipt path must stay inside the dedicated receipts root")
    if resolved.suffix != ".jsonl":
        raise VisionFusionError("E2E receipt path must end in .jsonl")
    return resolved


def append_e2e_receipt_jsonl(
    path: Path,
    receipt: Mapping[str, object],
    *,
    receipts_root: Path = E2E_RECEIPT_ROOT,
) -> None:
    """Append one metadata-only receipt, rejecting duplicate receipt identities.

    This helper only appends a single canonical JSON line.  It never rewrites an
    existing receipt file and rejects malformed prior lines rather than silently
    replacing or skipping them.
    """
    resolved_path = _receipt_path(path, receipts_root=receipts_root)
    validate_e2e_receipt(receipt)
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str):
        raise VisionFusionError("receipt must contain a receipt_id")
    _require_identifier(receipt_id, field="receipt_id")
    serialized = json.dumps(
        dict(receipt),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a+", encoding="utf-8") as handle:
        flock(handle.fileno(), LOCK_EX)
        try:
            handle.seek(0)
            existing_ids: set[str] = set()
            for line_number, line in enumerate(handle, start=1):
                try:
                    existing = json.loads(
                        line,
                        parse_constant=_reject_non_finite_json,
                    )
                except json.JSONDecodeError as error:
                    raise VisionFusionError(
                        f"existing receipt JSONL is malformed at line {line_number}"
                    ) from error
                if not isinstance(existing, dict):
                    raise VisionFusionError(
                        f"existing receipt JSONL is not an object at line {line_number}"
                    )
                try:
                    validate_e2e_receipt(existing)
                except VisionFusionError as error:
                    raise VisionFusionError(
                        f"existing receipt JSONL failed validation at line {line_number}: {error}"
                    ) from error
                existing_id = existing["receipt_id"]
                assert isinstance(existing_id, str)
                if existing_id in existing_ids:
                    raise VisionFusionError(
                        f"existing receipt JSONL has duplicate receipt_id: {existing_id}"
                    )
                existing_ids.add(existing_id)
            if receipt_id in existing_ids:
                raise VisionFusionError(f"receipt_id already exists: {receipt_id}")
            handle.seek(0, 2)
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            flock(handle.fileno(), LOCK_UN)
