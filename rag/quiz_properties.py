"""Small source-bound property catalog for deterministic Quiz questions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuizPropertyMapping:
    """One reviewed finite-domain property tied to an admitted note section."""

    mapping_id: str
    subject: str
    note_path: str
    focus_heading: str
    concept: str
    question_axis: str
    source_value: str
    domain_id: str
    domain_values: tuple[str, str, str, str]
    source_quote: str


# Course-derived property questions are intentionally absent from this release.
PROPERTY_MAPPINGS: tuple[QuizPropertyMapping, ...] = ()
