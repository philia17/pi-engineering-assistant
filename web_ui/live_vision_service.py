"""Schema-valid, GPU-independent mock implementation of the live vision API.

This module is a contract test service, not a model server.  It accepts image
bytes only for the lifetime of one request and derives a deterministic,
explicitly non-inferential observation from request metadata.
"""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Final, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, status

from web_ui.live_vision_contract import (
    LIVE_VISION_MAX_IMAGE_BYTES,
    LIVE_VISION_MAX_IMAGE_HEIGHT,
    LIVE_VISION_MAX_IMAGE_PIXELS,
    LIVE_VISION_MAX_IMAGE_WIDTH,
    LIVE_VISION_SCHEMA_VERSION,
    LIVE_VISION_SERVICE_NAME,
    LiveVisionBoundary,
    LiveVisionHealth,
    LiveVisionModel,
    LiveVisionModelInfo,
    LiveVisionObservation,
    LiveVisionRequest,
    LiveVisionResponse,
)

MOCK_PROVIDER: Final[str] = "mock"
MOCK_MODEL_ID: Final[str] = "mock-vlm-contract"
MOCK_MODEL_REVISION: Final[str] = "v1"
MOCK_TOKEN_ENVIRONMENT_VARIABLE: Final[str] = "EEE_VLM_MOCK_TOKEN"


def mock_model_info() -> LiveVisionModel:
    """Return the fixed non-GPU model identity used by this contract service."""
    return LiveVisionModel(
        provider=MOCK_PROVIDER,
        model_id=MOCK_MODEL_ID,
        revision=MOCK_MODEL_REVISION,
    )


def _mock_token_is_configured() -> bool:
    """Return whether the optional test-only bearer-token gate is enabled."""
    return bool(os.getenv(MOCK_TOKEN_ENVIRONMENT_VARIABLE, "").strip())


def _auth_mode() -> Literal["not_required", "bearer_test_token"]:
    """Describe the mock's optional authentication policy without exposing a token."""
    return "bearer_test_token" if _mock_token_is_configured() else "not_required"


def _require_optional_mock_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require an environment-only token when the mock test gate is enabled."""
    expected_token = os.getenv(MOCK_TOKEN_ENVIRONMENT_VARIABLE, "").strip()
    if not expected_token:
        return
    expected_header = f"Bearer {expected_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected_header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid bearer token is required for this mock service.",
        )


def deterministic_mock_response(request: LiveVisionRequest) -> LiveVisionResponse:
    """Return a stable response while making no claim that image inference occurred."""
    image_prefix = request.image.sha256[:12]
    observation = LiveVisionObservation(
        visible_labels=[f"mock-image-sha256:{image_prefix}"],
        components=[],
        connections=[],
        polarity_or_direction=[],
        plot_readings=[],
        uncertainties=[
            "Deterministic mock response only; no visual inference or course fact was produced."
        ],
    )
    return LiveVisionResponse(
        schema_version=cast(Literal["eee.live_vision.v1"], LIVE_VISION_SCHEMA_VERSION),
        request_id=request.request_id,
        status="ok",
        model=mock_model_info(),
        observation=observation,
        error=None,
        boundary=LiveVisionBoundary(
            storage="memory_only",
            rag_persistence="forbidden",
            fact_status="untrusted_user_image_evidence",
            citations="none",
        ),
    )


def create_mock_live_vision_app() -> FastAPI:
    """Create the no-GPU mock app used by Pi/browser contract tests."""
    application = FastAPI(
        title="EEE Live Vision Mock Service",
        description=(
            "GPU-independent contract service. It never persists uploaded images or returns "
            "course citations."
        ),
        version="1.0.0",
    )

    @application.get("/health", response_model=LiveVisionHealth)
    def health() -> LiveVisionHealth:
        """Report contract-service readiness without loading or probing a GPU."""
        return LiveVisionHealth(
            schema_version=cast(Literal["eee.live_vision.v1"], LIVE_VISION_SCHEMA_VERSION),
            service=cast(Literal["eee-live-vision"], LIVE_VISION_SERVICE_NAME),
            status="ok",
            model_ready=True,
            storage="memory_only",
            rag_persistence="forbidden",
            auth_mode=_auth_mode(),
        )

    @application.get("/model-info", response_model=LiveVisionModelInfo)
    def model_info() -> LiveVisionModelInfo:
        """Expose the fixed mock capability and bounded input contract."""
        return LiveVisionModelInfo(
            schema_version=cast(Literal["eee.live_vision.v1"], LIVE_VISION_SCHEMA_VERSION),
            service=cast(Literal["eee-live-vision"], LIVE_VISION_SERVICE_NAME),
            model=mock_model_info(),
            supported_content_types=["image/jpeg", "image/png", "image/webp"],
            max_image_bytes=cast(Literal[10485760], LIVE_VISION_MAX_IMAGE_BYTES),
            max_image_width=cast(Literal[8192], LIVE_VISION_MAX_IMAGE_WIDTH),
            max_image_height=cast(Literal[8192], LIVE_VISION_MAX_IMAGE_HEIGHT),
            max_image_pixels=cast(Literal[40000000], LIVE_VISION_MAX_IMAGE_PIXELS),
            response_format="strict_json",
            storage="memory_only",
            rag_persistence="forbidden",
        )

    @application.post(
        "/v1/vision/infer",
        response_model=LiveVisionResponse,
        dependencies=[Depends(_require_optional_mock_token)],
    )
    def infer(request: LiveVisionRequest) -> LiveVisionResponse:
        """Validate and process one in-memory image with deterministic mock output."""
        return deterministic_mock_response(request)

    return application


app: Final[FastAPI] = create_mock_live_vision_app()
