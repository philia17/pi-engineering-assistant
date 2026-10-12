"""Direct memory-only Quiz state transitions owned by one demo session."""

from __future__ import annotations

import secrets
from _thread import LockType
from dataclasses import dataclass, field
from threading import Lock

from rag.quiz_contract import OPTION_IDS, PrivateQuizQuestion
from rag.quiz_scopes import QuizFocus


class QuizStateError(ValueError):
    """Raised when a requested Quiz transition is not currently valid."""


@dataclass
class QuizQuestionState:
    """One private question plus its first accepted submission."""

    question: PrivateQuizQuestion
    submitted_option_id: str | None = None

    @property
    def submitted(self) -> bool:
        """Return whether this question has already been graded."""
        return self.submitted_option_id is not None


@dataclass
class QuizState:
    """At most one Quiz attached directly to an authenticated DemoSession."""

    owner_session_id: str
    subject: str
    mode: str
    model: str
    topic_label: str
    knowledge_point_label: str | None
    retrieval_query: str
    focus_plan: tuple[QuizFocus, ...]
    fact_pool: tuple[QuizFocus, ...]
    current: QuizQuestionState
    scope_key: tuple[str, str, str, str | None]
    excluded_fact_ids: tuple[str, ...] = ()
    question_records: list[QuizQuestionState] = field(default_factory=list)
    generated_prompt_texts: list[str] = field(default_factory=list)
    generated_focus_ids: list[str] = field(default_factory=list)
    generated_fact_ids: list[str] = field(default_factory=list)
    quiz_id: str = field(default_factory=lambda: secrets.token_urlsafe(18))
    requested_total: int = 5
    generated_count: int = 1
    answered_count: int = 0
    correct_count: int = 0
    status: str = "in_progress"
    next_in_flight: bool = False
    _next_transition_lock: LockType = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        if self.requested_total not in {1, 3, 5, 7, 10}:
            raise QuizStateError("Quiz requested_total must be 1, 3, 5, 7, or 10.")
        if not self.question_records:
            self.question_records.append(self.current)
        if not self.generated_prompt_texts:
            self.generated_prompt_texts.append(self.current.question.prompt)
        if len(self.focus_plan) < self.requested_total:
            raise QuizStateError("Quiz focus plan is shorter than requested_total.")
        if self.current.question.focus_id != self.focus_plan[0].id:
            raise QuizStateError("Quiz question does not match its server focus.")
        if not self.generated_focus_ids:
            self.generated_focus_ids.append(self.current.question.focus_id)
        if not self.generated_fact_ids:
            self.generated_fact_ids.append(self.current.question.fact_id)

    def public_envelope(self) -> dict[str, object]:
        """Return current public state, revealing feedback only after submission."""
        if self.status == "finished":
            return self.summary()
        payload: dict[str, object] = {
            "schema_version": "quiz.public.v1",
            "quiz_id": self.quiz_id,
            "status": self.status,
            "subject": self.subject,
            "mode": self.mode,
            "model": self.model,
            "topic": self.topic_label,
            "knowledge_point": self.knowledge_point_label,
            "current_number": self.current.question.number,
            "requested_total": self.requested_total,
            "answered_count": self.answered_count,
            "correct_count": self.correct_count,
            "question": self.current.question.public_dict(submitted=self.current.submitted),
        }
        if self.current.submitted:
            payload["feedback"] = self.feedback()
        return payload

    def submit(self, *, question_id: str, option_id: str) -> dict[str, object]:
        """Grade the first valid submission by exact option-ID equality."""
        if self.status != "in_progress":
            raise QuizStateError("Quiz is already finished.")
        if question_id != self.current.question.question_id:
            raise QuizStateError("Question is not current.")
        if option_id not in OPTION_IDS:
            raise QuizStateError("Option must be A, B, C, or D.")
        if self.current.submitted:
            raise QuizStateError("Question has already been submitted.")
        self.current.submitted_option_id = option_id
        self.answered_count += 1
        if option_id == self.current.question.correct_option_id:
            self.correct_count += 1
        return self.feedback()

    def feedback(self) -> dict[str, object]:
        """Return feedback for the already-submitted current question."""
        submitted = self.current.submitted_option_id
        if submitted is None:
            raise QuizStateError("Question has not been submitted.")
        question = self.current.question
        return {
            "schema_version": "quiz.feedback.v1",
            "quiz_id": self.quiz_id,
            "status": self.status,
            "question_id": question.question_id,
            "submitted_option_id": submitted,
            "correct": submitted == question.correct_option_id,
            "correct_option_id": question.correct_option_id,
            "explanation": question.explanation,
            **question.explanation_dict(),
            "sources": [source.as_dict() for source in question.sources],
            "answered_count": self.answered_count,
            "correct_count": self.correct_count,
            "has_next": question.number < self.requested_total,
        }

    def advance(self, question: PrivateQuizQuestion) -> None:
        """Install one successfully generated next question."""
        if self.status != "in_progress":
            raise QuizStateError("Quiz is already finished.")
        if not self.current.submitted:
            raise QuizStateError("Submit the current question before Next.")
        if self.generated_count >= self.requested_total:
            raise QuizStateError("All requested questions have been generated.")
        if question.number != self.generated_count + 1:
            raise QuizStateError("Next question number is invalid.")
        normalized_prompt = " ".join(question.prompt.split()).casefold()
        if normalized_prompt in {
            " ".join(prompt.split()).casefold() for prompt in self.generated_prompt_texts
        }:
            raise QuizStateError("Quiz question prompt exactly duplicates an earlier question.")
        expected_focus = self.focus_plan[self.generated_count]
        if question.focus_id != expected_focus.id:
            raise QuizStateError("Quiz question does not match its server focus.")
        if question.fact_id in self.generated_fact_ids:
            raise QuizStateError("Quiz source fact was already used in this attempt.")
        self.current = QuizQuestionState(question=question)
        self.question_records.append(self.current)
        self.generated_prompt_texts.append(question.prompt)
        self.generated_focus_ids.append(question.focus_id)
        self.generated_fact_ids.append(question.fact_id)
        self.generated_count += 1

    def begin_next(self) -> None:
        """Reserve the single lazy-generation frontier for one request."""
        with self._next_transition_lock:
            if self.status != "in_progress":
                raise QuizStateError("Quiz is already finished.")
            if self.next_in_flight:
                raise QuizStateError("The next Quiz question is already being generated.")
            self.next_in_flight = True

    def end_next(self) -> None:
        """Release the lazy-generation frontier after every request outcome."""
        with self._next_transition_lock:
            self.next_in_flight = False

    def finish(self) -> dict[str, object]:
        """Finish immediately, retaining a memory-only summary."""
        self.status = "finished"
        return self.summary()

    def summary(self) -> dict[str, object]:
        """Return the retained finished summary."""
        return {
            "schema_version": "quiz.summary.v1",
            "quiz_id": self.quiz_id,
            "status": "finished",
            "subject": self.subject,
            "topic": self.topic_label,
            "knowledge_point": self.knowledge_point_label,
            "requested_total": self.requested_total,
            "answered_count": self.answered_count,
            "correct_count": self.correct_count,
            "score": self.correct_count,
            "maximum_score": self.requested_total,
            "review_count": len(self.question_records),
        }

    def review(self, number: int) -> dict[str, object]:
        """Return one read-only retained question only after the Quiz is finished."""
        if self.status != "finished":
            raise QuizStateError("Quiz review is available only after Finish.")
        if number < 1 or number > len(self.question_records):
            raise QuizStateError("Quiz review question is unavailable.")
        record = self.question_records[number - 1]
        question = record.question
        submitted = record.submitted_option_id
        return {
            "schema_version": "quiz.review.v1",
            "quiz_id": self.quiz_id,
            "status": self.status,
            "number": number,
            "review_count": len(self.question_records),
            "has_previous": number > 1,
            "has_next": number < len(self.question_records),
            "question": question.public_dict(submitted=True),
            "submitted_option_id": submitted,
            "correct_option_id": question.correct_option_id,
            "correct": submitted is not None and submitted == question.correct_option_id,
            "explanation": question.explanation,
            **question.explanation_dict(),
            "sources": [source.as_dict() for source in question.sources],
        }
