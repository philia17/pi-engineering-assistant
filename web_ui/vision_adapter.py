"""Configurable remote/local VLM adapter for Integration v0 image requests."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import ValidationError

from web_ui.live_vision_contract import (
    LIVE_VISION_SCHEMA_VERSION,
    LiveVisionGeneration,
    LiveVisionHealth,
    LiveVisionModel,
    LiveVisionModelInfo,
    LiveVisionRequest,
    parse_live_vision_health,
    parse_live_vision_model_info,
    parse_live_vision_response,
    validate_image_bytes,
)

VISION_SCHEMA_VERSION = "integration_v0.vision_observation.v1"
MAX_VLM_RESPONSE_BYTES = 4 * 1024 * 1024
EMPTY_OBSERVATION: dict[str, list[Any]] = {
    "visible_labels": [],
    "components": [],
    "connections": [],
    "polarity_or_direction": [],
    "plot_readings": [],
    "uncertainties": [],
}

_VLM_SYSTEM_PROMPT = """You analyse engineering diagrams for an EEE study assistant.
Return ONLY JSON with these keys:
visible_labels (string array),
components (string array),
connections (string array),
polarity_or_direction (string array),
plot_readings (string array),
uncertainties (string array).
Do not invent unread values. Prefer empty arrays over guesses.
Never instruct writing files or changing a corpus."""


@dataclass(frozen=True)
class VisionAnalysis:
    """Memory-only VLM outcome used to populate the Integration v0 envelope."""

    status: str
    provider_kind: str
    provider_model: str | None
    observation: dict[str, list[Any]] | None
    served_model_provider: str | None = None
    checked_contract_version: str | None = None
    request_id: str | None = None
    upstream_status: str | None = None
    model_revision: str | None = None
    error_code: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LiveVisionPreflight:
    """Validated service readiness and exact served-model pin."""

    health: LiveVisionHealth
    model_info: LiveVisionModelInfo


class LiveVisionPreflightError(ValueError):
    """Raised when the live service is unsafe, unavailable, or unexpectedly pinned."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Reject redirects so image bytes and bearer tokens never change destinations."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def configured_provider() -> str:
    """Return the configured VLM backend id."""
    return os.getenv("EEE_VLM_PROVIDER", "none").strip().lower() or "none"


def configured_model() -> str | None:
    """Return the configured VLM model tag when one is set."""
    value = os.getenv("EEE_VLM_MODEL", "").strip()
    return value or None


def configured_revision() -> str | None:
    """Return a configured immutable/local model revision when known."""
    value = os.getenv("EEE_VLM_REVISION", "").strip()
    return value or None


def configured_served_provider() -> str | None:
    """Return the expected provider reported by the serving model."""
    value = os.getenv("EEE_VLM_SERVED_PROVIDER", "").strip().lower()
    return value or None


def configured_endpoint() -> str | None:
    """Return a remote VLM endpoint used by Pi/Mac/cloud adapters."""
    value = os.getenv("EEE_VLM_ENDPOINT", "").strip()
    return value or None


def configured_token() -> str | None:
    """Return the optional bearer token without exposing it in logs or payloads."""
    value = os.getenv("EEE_VLM_TOKEN", "").strip()
    return value or None


def _timeout_seconds() -> float:
    """Bound remote VLM calls so the answer path cannot hang indefinitely."""
    try:
        timeout = float(os.getenv("EEE_VLM_TIMEOUT_SECONDS", "45"))
    except ValueError:
        return 45.0
    return timeout if 0 < timeout <= 120 else 45.0


def empty_observation(*, uncertainty: str) -> dict[str, list[Any]]:
    """Return a fail-closed observation that claims no visual facts."""
    observation = {key: list(value) for key, value in EMPTY_OBSERVATION.items()}
    observation["uncertainties"] = [uncertainty]
    return observation


def _as_string_list(value: Any) -> list[str]:
    """Coerce a JSON field into a list of short strings."""
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            cleaned = entry.strip()
            if cleaned:
                items.append(cleaned[:500])
        elif isinstance(entry, dict):
            label = entry.get("label") or entry.get("name") or entry.get("text")
            if isinstance(label, str) and label.strip():
                items.append(label.strip()[:500])
    return items


def normalize_observation(payload: dict[str, Any]) -> dict[str, list[Any]]:
    """Keep only the Integration v0 observation fields from a model payload."""
    return {
        "visible_labels": _as_string_list(payload.get("visible_labels")),
        "components": _as_string_list(payload.get("components")),
        "connections": _as_string_list(payload.get("connections")),
        "polarity_or_direction": _as_string_list(payload.get("polarity_or_direction")),
        "plot_readings": _as_string_list(payload.get("plot_readings")),
        "uncertainties": _as_string_list(payload.get("uncertainties"))
        or ["Model returned no explicit uncertainty notes."],
    }


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object from a VLM text response."""
    stripped = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.I)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("VLM response did not contain a JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("VLM JSON payload must be an object")
    return payload


def _mock_observation(subject: str, question: str) -> dict[str, list[Any]]:
    """Deterministic observation used by tests and offline smoke."""
    return {
        "visible_labels": [f"{subject} diagram", "Vin", "Vout"],
        "components": ["op-amp", "resistor network"],
        "connections": ["Vin to amplifier input", "feedback to inverting input"],
        "polarity_or_direction": ["signal left-to-right"],
        "plot_readings": [],
        "uncertainties": [
            f"Mock VLM observation for subject {subject}; question length={len(question)}."
        ],
    }


def _is_loopback_host(hostname: str) -> bool:
    """Return whether a hostname is an explicit local tunnel destination."""
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validated_endpoint(endpoint: str) -> SplitResult:
    """Reject unsafe endpoint forms before constructing an authenticated request."""
    try:
        parsed = urlsplit(endpoint)
        _ = parsed.port
    except ValueError as error:
        raise ValueError("VLM endpoint has an invalid port or URL authority") from error
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("VLM endpoint must use HTTPS or loopback HTTP")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("VLM endpoint must not contain userinfo")
    if parsed.fragment:
        raise ValueError("VLM endpoint must not contain a fragment")
    if parsed.query:
        raise ValueError("VLM endpoint must not contain a query")
    if parsed.hostname is None:
        raise ValueError("VLM endpoint must contain a hostname")
    if scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("plain HTTP is allowed only for an explicit loopback tunnel")
    decoded_path = unquote(parsed.path)
    if any(part == ".." for part in decoded_path.split("/")):
        raise ValueError("VLM endpoint path must not contain parent traversal")
    return parsed


def _service_url(endpoint: str, route: str) -> str:
    """Resolve a safe service route while preserving an optional proxy path prefix."""
    parsed = _validated_endpoint(endpoint)
    path = parsed.path.rstrip("/")
    infer_route = "/v1/vision/infer"
    if path.endswith(infer_route):
        path = path[: -len(infer_route)]
    elif path.endswith(route):
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    target_path = f"{path}{route}" if path else route
    return urlunsplit((parsed.scheme, parsed.netloc, target_path, "", ""))


def _open_no_redirect(request: Request, *, timeout: float) -> Any:
    """Open directly with redirects and environment proxy forwarding disabled."""
    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _read_bounded_json(response: Any) -> Any:
    """Read at most the live response cap before decoding JSON."""
    raw = response.read(MAX_VLM_RESPONSE_BYTES + 1)
    if len(raw) > MAX_VLM_RESPONSE_BYTES:
        raise ValueError("VLM response exceeds the configured byte limit")
    return json.loads(raw.decode("utf-8"))


def _request_headers() -> dict[str, str]:
    """Build non-secret HTTP headers and add an environment-only bearer token."""
    headers = {"Accept": "application/json"}
    token = configured_token()
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_contract_payload(endpoint: str, route: str) -> Any:
    """GET one bounded no-redirect JSON contract payload."""
    request = Request(
        _service_url(endpoint, route),
        headers=_request_headers(),
        method="GET",
    )
    with _open_no_redirect(request, timeout=_timeout_seconds()) as response:
        return _read_bounded_json(response)


def preflight_http_service(
    *,
    expected_provider: str,
    expected_model: str,
    expected_revision: str,
    endpoint: str | None = None,
) -> LiveVisionPreflight:
    """Verify health and exact model identity before an FI-010 live batch."""
    active_endpoint = endpoint or configured_endpoint()
    if active_endpoint is None:
        raise LiveVisionPreflightError("EEE_VLM_ENDPOINT is required for live preflight")
    expected_values = {
        "provider": expected_provider.strip().lower(),
        "model": expected_model.strip(),
        "revision": expected_revision.strip(),
    }
    if not all(expected_values.values()):
        raise LiveVisionPreflightError("provider, model, and immutable revision pins are required")
    try:
        health = parse_live_vision_health(_get_contract_payload(active_endpoint, "/health"))
        model_info = parse_live_vision_model_info(
            _get_contract_payload(active_endpoint, "/model-info")
        )
    except (
        OSError,
        URLError,
        HTTPError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
    ) as error:
        raise LiveVisionPreflightError(f"live vision preflight failed: {error}") from error
    if health.model_ready is not True:
        raise LiveVisionPreflightError("live vision model is not ready")
    if health.auth_mode in {"bearer_token", "bearer_test_token"} and configured_token() is None:
        raise LiveVisionPreflightError("live vision service requires a bearer token")
    actual = model_info.model
    mismatches: list[str] = []
    if actual.provider != expected_values["provider"]:
        mismatches.append("provider")
    if actual.model_id != expected_values["model"]:
        mismatches.append("model")
    if actual.revision != expected_values["revision"]:
        mismatches.append("revision")
    if mismatches:
        raise LiveVisionPreflightError("live vision model pin mismatch: " + ", ".join(mismatches))
    return LiveVisionPreflight(health=health, model_info=model_info)


def _ollama_chat_url() -> str:
    """Resolve the Ollama chat endpoint, preferring an explicit remote base."""
    endpoint = configured_endpoint()
    if endpoint:
        return _service_url(endpoint, "/api/chat")
    return "http://127.0.0.1:11434/api/chat"


def _analyze_with_ollama(
    *,
    image_bytes: bytes,
    content_type: str,
    subject: str,
    question: str,
) -> VisionAnalysis:
    """Call a local or remote Ollama vision model without persisting the image."""
    model = configured_model() or "llava"
    request_id = _live_request_id(image_bytes=image_bytes, subject=subject, question=question)
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": _VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Subject: {subject}\nQuestion: {question}\n"
                    "Describe only what is visible in the attached diagram."
                ),
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            },
        ],
        "options": {"temperature": 0},
    }
    try:
        headers = _request_headers()
        headers.update(
            {
                "Content-Type": "application/json",
                "X-EEE-Image-Content-Type": content_type,
            }
        )
        request = Request(
            _ollama_chat_url(),
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with _open_no_redirect(request, timeout=_timeout_seconds()) as response:
            body = _read_bounded_json(response)
    except (
        OSError,
        URLError,
        HTTPError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="ollama",
            served_model_provider="ollama",
            provider_model=model,
            observation=empty_observation(uncertainty=f"Ollama VLM request failed: {error}"),
            request_id=request_id,
            model_revision=configured_revision(),
            error_code="ollama_request_failed",
            error=str(error),
        )

    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="ollama",
            served_model_provider="ollama",
            provider_model=model,
            observation=empty_observation(uncertainty="Ollama VLM returned an empty observation."),
            request_id=request_id,
            upstream_status="vision_unavailable",
            model_revision=configured_revision(),
            error_code="empty_response",
            error="empty_response",
        )
    try:
        observation = normalize_observation(_extract_json_object(content))
    except (ValueError, json.JSONDecodeError) as error:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="ollama",
            served_model_provider="ollama",
            provider_model=model,
            observation=empty_observation(
                uncertainty=f"Ollama VLM returned unusable JSON: {error}"
            ),
            request_id=request_id,
            upstream_status="invalid_model_output",
            model_revision=configured_revision(),
            error_code="invalid_model_output",
            error=str(error),
        )
    return VisionAnalysis(
        status="processed",
        provider_kind="ollama",
        served_model_provider="ollama",
        provider_model=model,
        observation=observation,
        checked_contract_version=VISION_SCHEMA_VERSION,
        request_id=request_id,
        upstream_status="processed",
        model_revision=configured_revision(),
    )


def _analyze_with_http(
    *,
    image_bytes: bytes,
    content_type: str,
    subject: str,
    question: str,
) -> VisionAnalysis:
    """Call a strict v1 Mac/cloud service that keeps image bytes request-local."""
    endpoint = configured_endpoint()
    model = configured_model()
    request_id = _live_request_id(image_bytes=image_bytes, subject=subject, question=question)
    if not endpoint:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=None,
            provider_model=model,
            observation=empty_observation(
                uncertainty="EEE_VLM_ENDPOINT is required for the http VLM provider."
            ),
            request_id=request_id,
            model_revision=configured_revision(),
            error_code="missing_endpoint",
            error="missing_endpoint",
        )
    try:
        infer_url = _live_infer_url(endpoint)
    except ValueError:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=None,
            provider_model=model,
            observation=empty_observation(
                uncertainty="Configured live VLM endpoint failed the transport safety policy."
            ),
            request_id=request_id,
            model_revision=configured_revision(),
            error_code="unsafe_endpoint",
            error="unsafe_endpoint",
        )
    try:
        payload = LiveVisionRequest.model_validate(
            {
                "schema_version": LIVE_VISION_SCHEMA_VERSION,
                "request_id": request_id,
                "subject": subject,
                "question": question,
                "requested_model": model,
                "image": {
                    "content_type": content_type,
                    "byte_count": len(image_bytes),
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "base64": base64.b64encode(image_bytes).decode("ascii"),
                },
                "generation": LiveVisionGeneration().model_dump(),
            },
        )
    except ValidationError as error:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=None,
            provider_model=model,
            observation=empty_observation(
                uncertainty="Live VLM request could not satisfy the strict v1 contract."
            ),
            request_id=request_id,
            model_revision=configured_revision(),
            error_code="invalid_live_request",
            error=f"invalid_live_request: {error.errors()[0]['type']}",
        )

    try:
        headers = _request_headers()
        headers["Content-Type"] = "application/json"
        request = Request(
            infer_url,
            data=json.dumps(payload.model_dump(mode="json"), allow_nan=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with _open_no_redirect(request, timeout=_timeout_seconds()) as response:
            body = _read_bounded_json(response)
    except (
        OSError,
        URLError,
        HTTPError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=None,
            provider_model=model,
            observation=empty_observation(uncertainty=f"Remote VLM adapter failed: {error}"),
            request_id=request_id,
            model_revision=configured_revision(),
            error_code="transport_error",
            error=str(error),
        )
    try:
        response = parse_live_vision_response(body)
    except ValidationError as error:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=None,
            provider_model=model,
            observation=empty_observation(
                uncertainty="Remote VLM adapter returned an invalid live v1 response."
            ),
            request_id=request_id,
            model_revision=configured_revision(),
            error_code="invalid_response_contract",
            error=f"invalid_response_contract: {error.errors()[0]['type']}",
        )
    if response.request_id != request_id:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=response.model.provider,
            provider_model=response.model.model_id,
            observation=empty_observation(
                uncertainty="Remote VLM response did not match the submitted request."
            ),
            checked_contract_version=response.schema_version,
            request_id=request_id,
            upstream_status=response.status,
            model_revision=response.model.revision,
            error_code="response_request_id_mismatch",
            error="response_request_id_mismatch",
        )
    pin_error = _response_pin_error(response.model)
    if pin_error is not None:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=response.model.provider,
            provider_model=response.model.model_id,
            observation=empty_observation(
                uncertainty="Remote VLM served an unexpected provider, model, or revision."
            ),
            checked_contract_version=response.schema_version,
            request_id=request_id,
            upstream_status=response.status,
            model_revision=response.model.revision,
            error_code=pin_error,
            error=pin_error,
        )
    if response.status != "ok" or response.observation is None:
        error_code = response.error.code if response.error is not None else response.status
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="http",
            served_model_provider=response.model.provider,
            provider_model=response.model.model_id,
            observation=empty_observation(
                uncertainty=f"Remote VLM reported unavailable vision: {error_code}."
            ),
            checked_contract_version=response.schema_version,
            request_id=request_id,
            upstream_status=response.status,
            model_revision=response.model.revision,
            error_code=error_code,
            error=error_code,
        )
    return VisionAnalysis(
        status="processed",
        provider_kind="http",
        served_model_provider=response.model.provider,
        provider_model=response.model.model_id,
        observation=normalize_observation(response.observation.model_dump(mode="json")),
        checked_contract_version=response.schema_version,
        request_id=request_id,
        upstream_status=response.status,
        model_revision=response.model.revision,
    )


def _live_infer_url(endpoint: str) -> str:
    """Resolve either the service base URL or the exact v1 inference endpoint."""
    return _service_url(endpoint, "/v1/vision/infer")


def _response_pin_error(model_info: LiveVisionModel) -> str | None:
    """Return a stable error when configured served-model pins do not match."""
    expected_provider = configured_served_provider()
    expected_model = configured_model()
    expected_revision = configured_revision()
    if expected_provider is not None and model_info.provider != expected_provider:
        return "unexpected_served_provider"
    if expected_model is not None and model_info.model_id != expected_model:
        return "unexpected_served_model"
    if expected_revision is not None and model_info.revision != expected_revision:
        return "unexpected_model_revision"
    return None


def _live_request_id(*, image_bytes: bytes, subject: str, question: str) -> str:
    """Create a deterministic, non-secret correlation id for one HTTP request."""
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(b"\x00")
    digest.update(subject.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(question.encode("utf-8"))
    return f"req-{digest.hexdigest()[:32]}"


def analyze_image(
    *,
    image_bytes: bytes,
    content_type: str,
    subject: str,
    question: str,
) -> VisionAnalysis:
    """Analyse an in-memory image and never write user bytes into the corpus."""
    provider = configured_provider()
    if provider in {"", "none", "off", "disabled"}:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind="unconfigured",
            served_model_provider=None,
            provider_model=None,
            observation=empty_observation(
                uncertainty=(
                    "No VLM is configured, so this request contains no visual observation."
                )
            ),
            upstream_status="vision_unavailable",
            error_code="unconfigured",
            error="unconfigured",
        )
    try:
        validate_image_bytes(image_bytes, content_type=content_type)
    except ValueError:
        return VisionAnalysis(
            status="vision_unavailable",
            provider_kind=provider,
            served_model_provider=None,
            provider_model=configured_model(),
            observation=empty_observation(
                uncertainty="Image bytes are invalid, mismatched, or exceed resource limits."
            ),
            upstream_status="vision_unavailable",
            model_revision=configured_revision(),
            error_code="invalid_image",
            error="invalid_image",
        )
    if provider == "mock":
        return VisionAnalysis(
            status="processed",
            provider_kind="mock",
            served_model_provider="mock",
            provider_model=configured_model() or "mock-vlm",
            observation=_mock_observation(subject, question),
            checked_contract_version=VISION_SCHEMA_VERSION,
            request_id=_live_request_id(
                image_bytes=image_bytes,
                subject=subject,
                question=question,
            ),
            upstream_status="processed",
            model_revision=configured_revision() or "mock-v1",
        )
    if provider == "ollama":
        return _analyze_with_ollama(
            image_bytes=image_bytes,
            content_type=content_type,
            subject=subject,
            question=question,
        )
    if provider == "http":
        return _analyze_with_http(
            image_bytes=image_bytes,
            content_type=content_type,
            subject=subject,
            question=question,
        )
    return VisionAnalysis(
        status="vision_unavailable",
        provider_kind=provider,
        served_model_provider=None,
        provider_model=configured_model(),
        observation=empty_observation(
            uncertainty=f"Unknown VLM provider {provider!r}; no visual observation made."
        ),
        upstream_status="vision_unavailable",
        model_revision=configured_revision(),
        error_code="unknown_provider",
        error="unknown_provider",
    )
