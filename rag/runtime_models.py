"""Safe, explicit local-model choices for the Integration v0 runtime."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


@dataclass(frozen=True)
class RuntimeModel:
    """One UI-visible, server-allowlisted Ollama model."""

    id: str
    label: str
    description: str
    tier: str


DEFAULT_MODEL_ID = "qwen3:1.7b"
MODEL_CATALOG: tuple[RuntimeModel, ...] = (
    RuntimeModel(
        id="qwen3:1.7b",
        label="Qwen3 1.7B",
        description="Pi default: balanced speed for first-pass answers.",
        tier="recommended",
    ),
    RuntimeModel(
        id="qwen2.5:1.5b-q4_k_m",
        label="Qwen2.5 1.5B",
        description="Smallest local option: quickest, lower answer headroom.",
        tier="fast",
    ),
    RuntimeModel(
        id="qwen2.5:7b",
        label="Qwen2.5 7B",
        description="More capable general model; noticeably slower on the Pi.",
        tier="quality",
    ),
    RuntimeModel(
        id="qwen2.5-7b-eee-targeted404:q4_lora_v2",
        label="EEE tuned 7B v2",
        description="Experimental EEE-tuned model; compare it in benchmarks first.",
        tier="experimental",
    ),
)


def allowed_models() -> tuple[RuntimeModel, ...]:
    """Return the deployment allowlist, optionally narrowed by configuration."""
    configured = os.getenv("EEE_RAG_ALLOWED_MODELS", "").strip()
    if not configured:
        return MODEL_CATALOG
    allowed_ids = {model_id.strip() for model_id in configured.split(",") if model_id.strip()}
    return tuple(model for model in MODEL_CATALOG if model.id in allowed_ids)


def default_model_id() -> str:
    """Return a configured default that always belongs to the allowlist."""
    configured = os.getenv("EEE_RAG_MODEL", DEFAULT_MODEL_ID)
    allowed = allowed_models()
    if any(model.id == configured for model in allowed):
        return configured
    return allowed[0].id


def installed_model_ids(timeout_seconds: float = 1.5) -> set[str] | None:
    """Read local Ollama tags without loading a model; ``None`` means unknown."""
    request = Request("http://127.0.0.1:11434/api/tags", method="GET")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return None
    models = payload.get("models", [])
    if not isinstance(models, list):
        return None
    return {
        item["name"]
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def model_options() -> list[dict[str, object]]:
    """Return catalog records plus an installed-state observation for the UI."""
    installed = installed_model_ids()
    return [
        {
            **asdict(model),
            "available": installed is None or model.id in installed,
            "availability": "unknown" if installed is None else "installed",
        }
        for model in allowed_models()
    ]


def resolve_model_id(model_id: str | None) -> str:
    """Validate a requested model against the fixed deployment allowlist."""
    selected = model_id or default_model_id()
    if selected not in {model.id for model in allowed_models()}:
        raise ValueError("Requested model is not enabled for this EEE RAG deployment.")
    installed = installed_model_ids()
    if installed is not None and selected not in installed:
        raise ValueError("Requested model is not installed on this runtime.")
    return selected
