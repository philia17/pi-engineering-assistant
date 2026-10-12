"""Private and public contracts for one grounded Engineering Preview Quiz question."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

OPTION_IDS: tuple[str, ...] = ("A", "B", "C", "D")


class QuizGenerationError(ValueError):
    """Raised when one model response cannot form a grounded Quiz question."""


@dataclass(frozen=True)
class QuizOption:
    """One browser-visible multiple-choice option."""

    option_id: str
    text: str

    def as_dict(self) -> dict[str, str]:
        """Return the option's public representation."""
        return {"option_id": self.option_id, "text": self.text}


@dataclass(frozen=True)
class QuizSource:
    """A server-derived source reference revealed only after submission."""

    note_path: str
    heading: str | None

    def as_dict(self) -> dict[str, str | None]:
        """Return the source feedback representation."""
        return {"note_path": self.note_path, "heading": self.heading}


@dataclass(frozen=True)
class QuizWrongOptionReason:
    """One server-authored explanation for a finite-domain wrong value."""

    option_id: str
    text: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        """Return the submitted/review feedback representation."""
        return {
            "option_id": self.option_id,
            "text": self.text,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PrivateQuizQuestion:
    """One validated question including the server-only answer and evidence."""

    question_id: str
    number: int
    focus_id: str
    prompt: str
    options: tuple[QuizOption, ...]
    correct_option_id: str
    explanation: str
    sources: tuple[QuizSource, ...]
    fact_id: str = ""
    why_correct: str = ""
    wrong_option_reasons: tuple[QuizWrongOptionReason, ...] = ()
    reference_title: str = ""
    reference_heading: str = ""

    def public_dict(self, *, submitted: bool = False) -> dict[str, object]:
        """Serialize only fields safe before answer submission."""
        return {
            "question_id": self.question_id,
            "number": self.number,
            "type": "single_best_answer_mcq",
            "prompt": self.prompt,
            "options": [option.as_dict() for option in self.options],
            "submitted": submitted,
        }

    def explanation_dict(self) -> dict[str, object]:
        """Return structured feedback safe only after submission."""
        return {
            "why_correct": self.why_correct or self.explanation,
            "wrong_options": [reason.as_dict() for reason in self.wrong_option_reasons],
            "reference": {
                "title": self.reference_title,
                "heading": self.reference_heading,
            },
        }


def parse_generated_question(
    raw_output: str,
    *,
    question_id: str,
    number: int,
    focus_id: str,
    sources: tuple[QuizSource, ...],
) -> PrivateQuizQuestion:
    """Parse one exact JSON object once; no repair, retry, or prose fallback."""
    try:
        payload: Any = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as error:
        raise QuizGenerationError("Quiz model output is not valid JSON.") from error
    expected_fields = {
        "prompt",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "correct_option_id",
        "explanation",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise QuizGenerationError("Quiz model output has invalid fields.")
    prompt = payload["prompt"]
    explanation = payload["explanation"]
    correct_option_id = payload["correct_option_id"]
    raw_options = [payload[f"option_{option_id.lower()}"] for option_id in OPTION_IDS]
    generated_text = [prompt, explanation, *raw_options]
    if any(not isinstance(value, str) or not value.strip() for value in generated_text):
        raise QuizGenerationError("Quiz text fields must be non-empty strings.")
    if any(any(ord(character) < 32 for character in value) for value in generated_text):
        raise QuizGenerationError("Quiz model output contains a damaged control character.")
    if any("\\" in value for value in generated_text):
        raise QuizGenerationError("Quiz model output contains a forbidden backslash.")
    if correct_option_id not in OPTION_IDS:
        raise QuizGenerationError("Quiz correct option is invalid.")
    options = [
        QuizOption(option_id=option_id, text=text.strip())
        for option_id, text in zip(OPTION_IDS, raw_options, strict=True)
    ]
    return PrivateQuizQuestion(
        question_id=question_id,
        number=number,
        focus_id=focus_id,
        prompt=prompt.strip(),
        options=tuple(options),
        correct_option_id=correct_option_id,
        explanation=explanation.strip(),
        sources=sources,
    )
