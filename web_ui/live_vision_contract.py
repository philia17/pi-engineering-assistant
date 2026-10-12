"""Shared strict JSON contract for the GPU-independent live vision service.

The contract is deliberately separate from the preview answer envelope.  It is
the only payload shape exchanged by the Pi HTTP adapter and a cloud vision
service, and it never carries course citations or persistence instructions.
"""

from __future__ import annotations

import base64 as base64_module
import hashlib
import io
import json
import warnings
from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal, TypeAlias, TypeVar

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

LIVE_VISION_SCHEMA_VERSION: Final[str] = "eee.live_vision.v1"
LIVE_VISION_SERVICE_NAME: Final[str] = "eee-live-vision"
LIVE_VISION_MAX_IMAGE_BYTES: Final[int] = 10 * 1024 * 1024
LIVE_VISION_MAX_BASE64_CHARS: Final[int] = ((LIVE_VISION_MAX_IMAGE_BYTES + 2) // 3) * 4
LIVE_VISION_MAX_IMAGE_WIDTH: Final[int] = 8192
LIVE_VISION_MAX_IMAGE_HEIGHT: Final[int] = 8192
LIVE_VISION_MAX_IMAGE_PIXELS: Final[int] = 40_000_000
MACHINE_ID_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,299}$"
_IMAGE_FORMAT_TO_CONTENT_TYPE: Final[dict[str, str]] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

SubjectCode: TypeAlias = Literal["PE", "IS", "EM", "SEP", "EECC", "EPC", "MMT", "DIES"]
LiveVisionStatus: TypeAlias = Literal[
    "ok", "vision_unavailable", "invalid_model_output", "runtime_error"
]
_ModelType = TypeVar("_ModelType", bound="StrictJsonModel")


class StrictJsonModel(BaseModel):
    """Base model that rejects coercion and unknown JSON properties."""

    model_config = ConfigDict(extra="forbid", strict=True)


class LiveVisionImage(StrictJsonModel):
    """Decoded-image-bound request data; bytes are only held in request memory."""

    content_type: Literal["image/jpeg", "image/png", "image/webp"]
    byte_count: Annotated[int, Field(ge=1, le=LIVE_VISION_MAX_IMAGE_BYTES)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
    base64: Annotated[str, Field(min_length=4, max_length=LIVE_VISION_MAX_BASE64_CHARS)]

    @field_validator("base64")
    @classmethod
    def _require_base64(cls, value: str) -> str:
        """Reject whitespace, malformed padding, and non-base64 characters."""
        try:
            decoded = base64_module.b64decode(value, validate=True)
        except ValueError as error:
            raise ValueError("image.base64 must be canonical base64") from error
        canonical = base64_module.b64encode(decoded).decode("ascii")
        if value != canonical:
            raise ValueError("image.base64 must use canonical RFC 4648 encoding")
        return value

    @model_validator(mode="after")
    def _bind_image_identity(self) -> LiveVisionImage:
        """Require declared byte identity to match the decoded in-memory image."""
        decoded = base64_module.b64decode(self.base64, validate=True)
        if len(decoded) != self.byte_count:
            raise ValueError("image.byte_count does not match decoded image bytes")
        if hashlib.sha256(decoded).hexdigest() != self.sha256:
            raise ValueError("image.sha256 does not match decoded image bytes")
        validate_image_bytes(decoded, content_type=self.content_type)
        return self


class LiveVisionGeneration(StrictJsonModel):
    """Deterministic generation controls frozen by the live v1 contract."""

    temperature: Literal[0] = 0
    do_sample: Literal[False] = False
    max_new_tokens: Annotated[int, Field(ge=256, le=8192)] = 2048
    seed: Annotated[int, Field(ge=0, le=2_147_483_647)] = 17


class LiveVisionRequest(StrictJsonModel):
    """Exact request accepted by ``POST /v1/vision/infer``."""

    schema_version: Literal["eee.live_vision.v1"]
    request_id: Annotated[
        str, Field(pattern=r"^[a-z0-9][a-z0-9._:-]{2,127}$", min_length=3, max_length=128)
    ]
    subject: SubjectCode
    question: Annotated[str, Field(min_length=1, max_length=4000)]
    requested_model: (
        Annotated[
            str,
            Field(pattern=MACHINE_ID_PATTERN, min_length=1, max_length=300),
        ]
        | None
    ) = None
    image: LiveVisionImage
    generation: LiveVisionGeneration

    @field_validator("question")
    @classmethod
    def _require_non_blank_question(cls, value: str) -> str:
        """Keep an all-whitespace prompt from reaching a cloud service."""
        if not value.strip():
            raise ValueError("question must contain non-whitespace characters")
        return value


class LiveVisionObservation(StrictJsonModel):
    """Untrusted image-only observation with no source or corpus semantics."""

    visible_labels: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=500
    )
    components: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=500
    )
    connections: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=1000
    )
    polarity_or_direction: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=500
    )
    plot_readings: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=500
    )
    uncertainties: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        default_factory=list, max_length=500
    )


class LiveVisionModel(StrictJsonModel):
    """Non-secret identifier of the serving model implementation."""

    provider: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$", min_length=2)]
    model_id: Annotated[
        str,
        Field(pattern=MACHINE_ID_PATTERN, min_length=1, max_length=300),
    ]
    revision: (
        Annotated[
            str,
            Field(pattern=MACHINE_ID_PATTERN, min_length=1, max_length=300),
        ]
        | None
    ) = None


class LiveVisionError(StrictJsonModel):
    """Stable fail-closed error returned without an observation."""

    code: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{2,63}$", min_length=3, max_length=64)]
    message: Annotated[str, Field(min_length=1, max_length=2000)]
    retryable: bool


class LiveVisionBoundary(StrictJsonModel):
    """Invariant that visual output remains transient, unreviewed user evidence."""

    storage: Literal["memory_only"]
    rag_persistence: Literal["forbidden"]
    fact_status: Literal["untrusted_user_image_evidence"]
    citations: Literal["none"]


class LiveVisionResponse(StrictJsonModel):
    """Exact response accepted by the Pi HTTP adapter."""

    schema_version: Literal["eee.live_vision.v1"]
    request_id: Annotated[
        str, Field(pattern=r"^[a-z0-9][a-z0-9._:-]{2,127}$", min_length=3, max_length=128)
    ]
    status: LiveVisionStatus
    model: LiveVisionModel
    observation: LiveVisionObservation | None
    error: LiveVisionError | None
    boundary: LiveVisionBoundary

    @model_validator(mode="after")
    def _enforce_status_shape(self) -> LiveVisionResponse:
        """Make successful and failed response shapes unambiguous to callers."""
        if self.status == "ok" and self.observation is None:
            raise ValueError("status=ok requires observation")
        if self.status == "ok" and self.error is not None:
            raise ValueError("status=ok requires error=null")
        if self.status != "ok" and self.observation is not None:
            raise ValueError("non-ok status requires observation=null")
        if self.status != "ok" and self.error is None:
            raise ValueError("non-ok status requires error")
        return self


class LiveVisionHealth(StrictJsonModel):
    """Liveness/readiness payload returned without image or model inference."""

    schema_version: Literal["eee.live_vision.v1"]
    service: Literal["eee-live-vision"]
    status: Literal["ok"]
    model_ready: bool
    storage: Literal["memory_only"]
    rag_persistence: Literal["forbidden"]
    auth_mode: Literal["not_required", "bearer_test_token", "bearer_token"]


class LiveVisionModelInfo(StrictJsonModel):
    """Non-secret static model capability payload."""

    schema_version: Literal["eee.live_vision.v1"]
    service: Literal["eee-live-vision"]
    model: LiveVisionModel
    supported_content_types: list[Literal["image/jpeg", "image/png", "image/webp"],] = Field(
        min_length=3, max_length=3
    )
    max_image_bytes: Literal[10485760]
    max_image_width: Literal[8192]
    max_image_height: Literal[8192]
    max_image_pixels: Literal[40000000]
    response_format: Literal["strict_json"]
    storage: Literal["memory_only"]
    rag_persistence: Literal["forbidden"]


def parse_live_vision_request(payload: Mapping[str, Any] | str | bytes) -> LiveVisionRequest:
    """Strictly parse one request from JSON bytes/text or a decoded JSON object."""
    return _parse_model(LiveVisionRequest, payload)


def parse_live_vision_response(payload: Mapping[str, Any] | str | bytes) -> LiveVisionResponse:
    """Strictly parse one response from JSON bytes/text or a decoded JSON object."""
    return _parse_model(LiveVisionResponse, payload)


def parse_live_vision_health(payload: Mapping[str, Any] | str | bytes) -> LiveVisionHealth:
    """Strictly parse one health response."""
    return _parse_model(LiveVisionHealth, payload)


def parse_live_vision_model_info(payload: Mapping[str, Any] | str | bytes) -> LiveVisionModelInfo:
    """Strictly parse one model-info response."""
    return _parse_model(LiveVisionModelInfo, payload)


def validate_image_bytes(image_bytes: bytes, *, content_type: str) -> tuple[int, int]:
    """Decode and verify an allowed image entirely in memory.

    Returns width and height after checking MIME agreement and bounded
    dimensions/pixels. No decoded or encoded bytes are persisted.
    """
    if not image_bytes:
        raise ValueError("image bytes must not be empty")
    if len(image_bytes) > LIVE_VISION_MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the live vision byte limit")
    expected_content_type = content_type.strip().lower()
    if expected_content_type not in _IMAGE_FORMAT_TO_CONTENT_TYPE.values():
        raise ValueError("unsupported image content type")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as image:
                actual_content_type = _IMAGE_FORMAT_TO_CONTENT_TYPE.get(
                    str(image.format or "").upper()
                )
                width, height = image.size
                if actual_content_type != expected_content_type:
                    raise ValueError("declared image content type does not match decoded bytes")
                if width <= 0 or height <= 0:
                    raise ValueError("image dimensions must be positive")
                if (
                    width > LIVE_VISION_MAX_IMAGE_WIDTH
                    or height > LIVE_VISION_MAX_IMAGE_HEIGHT
                    or width * height > LIVE_VISION_MAX_IMAGE_PIXELS
                ):
                    raise ValueError("image dimensions exceed the live vision resource limits")
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as decoded_image:
                decoded_image.load()
    except (
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
    ) as error:
        raise ValueError("image bytes are corrupt or undecodable") from error
    return width, height


def request_json_schema() -> dict[str, Any]:
    """Return the published Draft 2020-12-compatible request JSON schema."""
    return LiveVisionRequest.model_json_schema()


def response_json_schema() -> dict[str, Any]:
    """Return the published Draft 2020-12-compatible response JSON schema."""
    schema = LiveVisionResponse.model_json_schema()
    schema["allOf"] = [
        {
            "if": {
                "properties": {"status": {"const": "ok"}},
                "required": ["status"],
            },
            "then": {
                "properties": {
                    "observation": {"not": {"type": "null"}},
                    "error": {"type": "null"},
                }
            },
            "else": {
                "properties": {
                    "observation": {"type": "null"},
                    "error": {"not": {"type": "null"}},
                }
            },
        }
    ]
    return schema


def _parse_model(
    model_type: type[_ModelType], payload: Mapping[str, Any] | str | bytes
) -> _ModelType:
    """Route raw JSON through Pydantic's strict path without relaxed coercion."""
    if isinstance(payload, str | bytes):
        return model_type.model_validate_json(payload)
    return model_type.model_validate(payload)


def canonical_json(model: StrictJsonModel) -> str:
    """Serialize contract payloads deterministically for request/receipt hashes."""
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "LIVE_VISION_MAX_IMAGE_BYTES",
    "LIVE_VISION_MAX_IMAGE_HEIGHT",
    "LIVE_VISION_MAX_IMAGE_PIXELS",
    "LIVE_VISION_MAX_IMAGE_WIDTH",
    "LIVE_VISION_SCHEMA_VERSION",
    "LIVE_VISION_SERVICE_NAME",
    "MACHINE_ID_PATTERN",
    "LiveVisionBoundary",
    "LiveVisionError",
    "LiveVisionGeneration",
    "LiveVisionHealth",
    "LiveVisionImage",
    "LiveVisionModel",
    "LiveVisionModelInfo",
    "LiveVisionObservation",
    "LiveVisionRequest",
    "LiveVisionResponse",
    "LiveVisionStatus",
    "StrictJsonModel",
    "SubjectCode",
    "ValidationError",
    "canonical_json",
    "parse_live_vision_request",
    "parse_live_vision_response",
    "parse_live_vision_health",
    "parse_live_vision_model_info",
    "request_json_schema",
    "response_json_schema",
    "validate_image_bytes",
]
