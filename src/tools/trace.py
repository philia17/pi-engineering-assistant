"""Structured trace objects for tool-using agent demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One tool call made by an agent."""

    name: str
    input_summary: str
    output_summary: str
    duration_ms: float


@dataclass
class ToolTrace:
    """Mutable trace container for an agent run."""

    calls: list[ToolCall] = field(default_factory=list)

    def record(
        self,
        name: str,
        input_summary: str,
        started_at: float,
        output_summary: str,
    ) -> None:
        """Append one completed tool call."""
        self.calls.append(
            ToolCall(
                name=name,
                input_summary=input_summary,
                output_summary=output_summary,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )
        )

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return JSON-serializable trace entries."""
        return [
            {
                "name": call.name,
                "input_summary": call.input_summary,
                "output_summary": call.output_summary,
                "duration_ms": call.duration_ms,
            }
            for call in self.calls
        ]
