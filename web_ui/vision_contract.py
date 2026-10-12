"""Provider-neutral, memory-only JSON contract for Integration v0 image requests."""

from __future__ import annotations

from web_ui.vision_adapter import (
    VISION_SCHEMA_VERSION,
    VisionAnalysis,
    analyze_image,
)


def build_vision_envelope(
    *,
    image: dict[str, object],
    subject: str,
    question: str,
    image_bytes: bytes | None = None,
    analysis: VisionAnalysis | None = None,
) -> dict[str, object]:
    """Return a VLM JSON envelope without claiming unobserved visual facts.

    ``analysis`` lets an orchestration layer use one already-completed in-memory
    provider call for both the response envelope and a strictly bounded retrieval
    hint.  It is deliberately optional so existing callers retain the original
    one-call behaviour.
    """
    if image["status"] == "not_provided":
        return {
            "schema_version": VISION_SCHEMA_VERSION,
            "status": "not_requested",
            "provider": None,
            "input": None,
            "analysis_request": {"subject": subject, "question": question},
            "observation": None,
            "review": {"state": "not_applicable", "accepted_claims": []},
            "storage": "no_image_received",
        }

    content_type = str(image.get("content_type") or "application/octet-stream")
    resolved_analysis: VisionAnalysis
    if analysis is not None:
        resolved_analysis = analysis
    elif image_bytes is None:
        resolved_analysis = VisionAnalysis(
            status="vision_unavailable",
            provider_kind="unconfigured",
            provider_model=None,
            observation={
                "visible_labels": [],
                "components": [],
                "connections": [],
                "polarity_or_direction": [],
                "plot_readings": [],
                "uncertainties": ["Image bytes were not retained in memory for VLM analysis."],
            },
            upstream_status="vision_unavailable",
            error_code="missing_image_bytes",
            error="missing_image_bytes",
        )
    else:
        resolved_analysis = analyze_image(
            image_bytes=image_bytes,
            content_type=content_type,
            subject=subject,
            question=question,
        )

    return {
        "schema_version": VISION_SCHEMA_VERSION,
        "status": resolved_analysis.status,
        "provider": {
            "kind": resolved_analysis.provider_kind,
            "served_model_provider": resolved_analysis.served_model_provider,
            "model": resolved_analysis.provider_model,
            "revision": resolved_analysis.model_revision,
        },
        "contract": {
            "checked_version": resolved_analysis.checked_contract_version,
            "request_id": resolved_analysis.request_id,
            "upstream_status": resolved_analysis.upstream_status,
        },
        "input": {
            "content_type": image["content_type"],
            "byte_count": image["byte_count"],
            "sha256": image["sha256"],
        },
        "analysis_request": {"subject": subject, "question": question},
        "observation": resolved_analysis.observation,
        "review": {"state": "not_reviewed", "accepted_claims": []},
        "storage": "memory_only_not_added_to_corpus",
        "error_code": resolved_analysis.error_code,
        "error": resolved_analysis.error,
    }
