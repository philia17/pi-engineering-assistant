"""Answer-format validation helpers for benchmark-style questions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FormatCheckResult:
    """Structured result from an answer-format check."""

    passed: bool
    expected_parts: int
    actual_parts: int
    numbers: list[float]
    error: str = ""


def extract_numbers(text: str) -> list[float]:
    """Extract decimal/scientific-notation numbers from text."""
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return [float(match) for match in matches]


def check_fill_blank_numeric(answer: str, *, expected_parts: int) -> FormatCheckResult:
    """Validate a numeric fill-in-the-blank answer separated by ``|``."""
    parts = [part.strip() for part in answer.split("|")]
    numbers = extract_numbers(answer)
    if len(parts) != expected_parts:
        return FormatCheckResult(
            passed=False,
            expected_parts=expected_parts,
            actual_parts=len(parts),
            numbers=numbers,
            error=f"expected {expected_parts} part(s), got {len(parts)}",
        )
    if len(numbers) != expected_parts:
        return FormatCheckResult(
            passed=False,
            expected_parts=expected_parts,
            actual_parts=len(parts),
            numbers=numbers,
            error=f"expected {expected_parts} numeric value(s), got {len(numbers)}",
        )
    return FormatCheckResult(
        passed=True,
        expected_parts=expected_parts,
        actual_parts=len(parts),
        numbers=numbers,
    )
