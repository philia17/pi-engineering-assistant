"""Minimal deterministic engineering agent demo.

This is intentionally narrow: it demonstrates the agent direction by solving
a shared-heatsink numeric fill-in-the-blank calculation from supplied inputs.
The full agent should later route across retrieval, calculation, validation,
and optional VLM tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.tools.calculator import calculate_shared_heatsink
from src.tools.format_checker import check_fill_blank_numeric
from src.tools.trace import ToolTrace


@dataclass(frozen=True)
class AgentResponse:
    """Structured response returned by the engineering agent."""

    answer: str
    sources: list[str]
    tool_trace: list[dict[str, Any]]
    metrics: dict[str, Any]


class EngineeringAgent:
    """Small controlled agent for engineering QA demonstrations."""

    def answer(self, question: str) -> AgentResponse:
        """Answer one question with tool use and a visible trace."""
        run_start = perf_counter()
        trace = ToolTrace()

        started = perf_counter()
        task_type = self._classify_task(question)
        trace.record(
            "classify_task",
            "question text",
            started,
            task_type,
        )

        if task_type != "numeric_fb_shared_heatsink":
            raise ValueError(f"Unsupported demo task: {task_type}")

        started = perf_counter()
        values = self._extract_shared_heatsink_values(question)
        trace.record(
            "extract_variables",
            "theta_JC, theta_CS, P_loss, T_A, T_J(max)",
            started,
            ", ".join(f"{key}={value}" for key, value in values.items()),
        )

        started = perf_counter()
        result = calculate_shared_heatsink(
            devices=2,
            p_loss_w=values["p_loss_w"],
            theta_jc_c_per_w=values["theta_jc_c_per_w"],
            theta_cs_c_per_w=values["theta_cs_c_per_w"],
            ambient_c=values["ambient_c"],
            junction_max_c=values["junction_max_c"],
        )
        trace.record(
            "calculate",
            "shared heatsink thermal equation",
            started,
            f"P_total={result.total_power_w}, theta_SA={result.theta_sa_c_per_w}",
        )

        answer = f"{result.total_power_w:.1f} | {result.theta_sa_c_per_w:.2f}"

        started = perf_counter()
        check = check_fill_blank_numeric(answer, expected_parts=2)
        trace.record(
            "check_format",
            "numeric FB with 2 blanks",
            started,
            "passed" if check.passed else check.error,
        )

        return AgentResponse(
            answer=answer,
            sources=["thermal model: T_J = T_A + P_loss(theta_JC + theta_CS) + P_total theta_SA"],
            tool_trace=trace.as_dicts(),
            metrics={
                "agent": "EngineeringAgent.demo",
                "latency_s": round(perf_counter() - run_start, 4),
                "format_passed": check.passed,
            },
        )

    @staticmethod
    def _classify_task(question: str) -> str:
        lowered = question.lower()
        if "mosfet" in lowered and "heatsink" in lowered and "___" in question:
            return "numeric_fb_shared_heatsink"
        return "unsupported"

    @staticmethod
    def _extract_shared_heatsink_values(question: str) -> dict[str, float]:
        patterns = {
            "theta_jc_c_per_w": r"theta[_ ]?JC\s*=\s*([0-9.]+)|θ_JC\s*=\s*([0-9.]+)",
            "theta_cs_c_per_w": r"theta[_ ]?CS\s*=\s*([0-9.]+)|θ_CS\s*=\s*([0-9.]+)",
            "p_loss_w": r"P_loss\s*=\s*([0-9.]+)",
            "ambient_c": r"T_A\s*=\s*([0-9.]+)",
            "junction_max_c": r"T_J\(max\)\s*=\s*([0-9.]+)",
        }
        values: dict[str, float] = {}
        for name, pattern in patterns.items():
            match = re.search(pattern, question)
            if match is None:
                raise ValueError(f"Could not extract {name}")
            raw_value = next(group for group in match.groups() if group is not None)
            values[name] = float(raw_value)
        return values
