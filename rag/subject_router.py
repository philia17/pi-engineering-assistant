"""Deterministic subject routing when the client omits an explicit subject."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.subject_preview import SUBJECT_CODE_PATTERN, SubjectRecord, load_subject_catalog

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STEM_LENGTH = 6
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)

# Extra lexical anchors that are stable across note titles and catalog names.
SUBJECT_HINTS: dict[str, frozenset[str]] = {
    "PE": frozenset(
        {"power", "transformer", "phasor", "induction", "machine", "magnetic", "circuit"}
    ),
    "IS": frozenset(
        {"signal", "digital", "analog", "sampling", "lti", "filter", "opamp", "op-amp"}
    ),
    "EM": frozenset(
        {
            "matrix",
            "matrices",
            "vector",
            "calculus",
            "complex",
            "transient",
            "integration",
            "transpose",
            "determinant",
            "eigenvalue",
            "gradient",
            "partial",
        }
    ),
    "SEP": frozenset(
        {"c", "pointer", "array", "struct", "stdio", "programming", "malloc", "printf"}
    ),
    "EECC": frozenset(
        {"converter", "renewable", "flyback", "inverter", "battery", "motor", "boost", "buck"}
    ),
    "EPC": frozenset(
        {"comparator", "oscillator", "adc", "dac", "fet", "bjt", "modulation", "amplifier"}
    ),
    "MMT": frozenset(
        {
            "fourier",
            "laplace",
            "probability",
            "ode",
            "pde",
            "ztransform",
            "dft",
            "transform",
            "statistics",
        }
    ),
    "DIES": frozenset(
        {
            "cpp",
            "c++",
            "class",
            "inheritance",
            "polymorphism",
            "exception",
            "pointer",
            "template",
            "oop",
        }
    ),
}


@dataclass(frozen=True)
class SubjectRouteDecision:
    """One routing outcome with an explicit confidence score."""

    status: str
    subject: str | None
    confidence: float
    scores: dict[str, float]
    reason: str
    candidates: tuple[str, ...]


def _stem(token: str) -> str:
    """Collapse light morphological variants for lexical overlap scoring."""
    if token.isdigit():
        return token
    return token[:_STEM_LENGTH]


def _tokenize(text: str) -> set[str]:
    """Return raw and lightly stemmed content tokens without stopwords."""
    tokens: set[str] = set()
    for match in _TOKEN_PATTERN.finditer(text):
        raw = match.group(0).lower()
        if raw in _STOPWORDS or len(raw) <= 1:
            continue
        tokens.add(raw)
        tokens.add(_stem(raw))
    return tokens


def _subject_lexicon(record: SubjectRecord) -> set[str]:
    """Build the lexical bag used to score one subject."""
    tokens = _tokenize(f"{record.subject} {record.name} {record.module}")
    tokens.update(SUBJECT_HINTS.get(record.subject, frozenset()))
    tokens.add(record.subject.lower())
    tokens.add(record.module.lower())
    return tokens


def score_subjects(question: str) -> dict[str, float]:
    """Score every catalog subject against one question."""
    catalog = load_subject_catalog()
    question_tokens = _tokenize(question)
    if not question_tokens:
        return {code: 0.0 for code in sorted(SUBJECT_CODE_PATTERN)}
    scores: dict[str, float] = {}
    for code, record in catalog.items():
        lexicon = _subject_lexicon(record)
        overlap = question_tokens & lexicon
        scores[code] = float(len(overlap))
    return scores


def route_subject(
    question: str,
    *,
    explicit_subject: str | None = None,
    confidence_threshold: float = 0.55,
) -> SubjectRouteDecision:
    """Route to one subject or request clarification / controlled fallback."""
    if explicit_subject and explicit_subject.strip():
        normalized = explicit_subject.strip().upper()
        if normalized not in SUBJECT_CODE_PATTERN:
            return SubjectRouteDecision(
                status="reject",
                subject=None,
                confidence=0.0,
                scores={},
                reason="unknown_explicit_subject",
                candidates=(),
            )
        return SubjectRouteDecision(
            status="explicit",
            subject=normalized,
            confidence=1.0,
            scores={normalized: 1.0},
            reason="client_explicit_subject",
            candidates=(normalized,),
        )

    scores = score_subjects(question)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_subject, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if best_score <= 0:
        return SubjectRouteDecision(
            status="clarify",
            subject=None,
            confidence=0.0,
            scores=scores,
            reason="no_subject_signal",
            candidates=tuple(code for code, _ in ordered[:3]),
        )
    confidence = best_score / (best_score + second_score) if (best_score + second_score) else 0.0
    if confidence < confidence_threshold:
        tied = tuple(code for code, score in ordered if score == best_score)
        if len(tied) > 1:
            return SubjectRouteDecision(
                status="multi_subject_fallback",
                subject=None,
                confidence=confidence,
                scores=scores,
                reason="tied_top_subjects",
                candidates=tied,
            )
        return SubjectRouteDecision(
            status="clarify",
            subject=None,
            confidence=confidence,
            scores=scores,
            reason="low_confidence",
            candidates=tuple(code for code, _ in ordered[:3]),
        )
    return SubjectRouteDecision(
        status="routed",
        subject=best_subject,
        confidence=confidence,
        scores=scores,
        reason="router_confidence",
        candidates=(best_subject,),
    )
