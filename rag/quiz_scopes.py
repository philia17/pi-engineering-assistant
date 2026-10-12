"""Note-derived Topic and Knowledge Point catalog for Quiz Preview."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

STRUCTURAL_H2_DENYLIST: frozenset[str] = frozenset(
    " ".join(label.split()).casefold()
    for label in {
        "Contents",
        "Learning Outcomes",
        "Summary",
        "Sources",
        "Sources and Review Scope",
        "Review Scope",
        "Further Reading",
        "Exercises",
        "Revision Questions",
        "Scope and Conventions",
        "Scope and Evidence Boundary",
        "Scope, Evidence, and Visual Boundary",
        "Consolidated Review",
        "Formula Index",
        "Formula Index and Review",
        "Topic Map",
        "Module Map and Assessment",
        "Source Boundary",
        "Source Record",
        "Source Index and Review Queue",
        "Resources and Safe Use",
        "Safe Use of Assessment Sources",
        "Review Boundaries and Related Notes",
        "Review Holds Before Promotion",
        "Review Requirements Before Promotion",
        "Visual-Review Hold Points",
        "Visual-Review Queue for This Note",
        "Visual-Review Requirements Before Promotion",
        "Summary Checklist",
        "Engineering Review Context",
        "Final Review Sequence",
        "How to Use the Notes for Revision",
    }
)
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_CONTEXT_DEPENDENT = re.compile(
    r"\b(?:this|that|these|above|below|previous|following)\b", re.IGNORECASE
)
_SENTENCE_END = re.compile(r"[.!?](?:[)\]}`'*_]+)?$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+(.+)$")
_LOWERCASE_FRAGMENT = re.compile(r"^[*_`]*[a-z][a-z-]*\b")
_METADATA_FACT_PREFIX = re.compile(
    r"^(?:\*\*)?(?:provenance|source provenance|supporting clarification)\s*:\s*(?:\*\*)?",
    re.IGNORECASE,
)
_FACT_TOKEN = re.compile(r"[a-z0-9]+")
_FACT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
_GENERIC_H4_EXACT: frozenset[str] = frozenset(
    {
        "assessment focus",
        "common misconceptions",
        "cross-reference",
        "cross-reference: source provenance",
        "definition / principle",
        "diagram",
        "equation",
        "example",
        "formula",
        "operating states / mechanism",
        "review",
        "source provenance",
        "sources",
        "visual review",
        "worked example",
    }
)
_GENERIC_H4_PREFIXES: tuple[str, ...] = (
    "comparison card:",
    "cross-reference:",
    "definition / principle:",
    "diagram card:",
    "formula card:",
    "worked example:",
)
_FORMULA_H4_PREFIXES: tuple[str, ...] = ("equation", "formula")
_FORMULA_H4_GENERIC_TOKENS: frozenset[str] = frozenset({"card", "equation", "formula"})


def _opaque_id(prefix: str, identity: str) -> str:
    return f"{prefix}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


class QuizScopeError(ValueError):
    """Raised when a requested note-derived scope is unavailable."""


@dataclass(frozen=True)
class QuizFact:
    """One private source-derived fact that the model may select by opaque ID."""

    id: str
    focus_id: str
    focus_heading: str
    fact_type: str
    display_text: str
    note_path: str
    concept: str = ""
    question_axis: str = ""
    domain_id: str = ""
    domain_values: tuple[str, ...] = ()
    source_quote: str = ""
    reference_title: str = ""


@dataclass(frozen=True)
class QuizFocus:
    """One private, assessment-bearing admitted-note section."""

    id: str
    label: str
    heading: str
    context: str
    note_path: str
    facts: tuple[QuizFact, ...] = ()


@dataclass(frozen=True)
class KnowledgePoint:
    """One selectable H2 or H3 inside an admitted note."""

    id: str
    label: str
    breadcrumb: str
    level: int
    focus_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "breadcrumb": self.breadcrumb,
            "level": self.level,
        }


@dataclass(frozen=True)
class QuizTopic:
    """One admitted note represented by its sole H1."""

    id: str
    label: str
    knowledge_points: tuple[KnowledgePoint, ...]
    focuses: tuple[QuizFocus, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "knowledge_points": [point.as_dict() for point in self.knowledge_points],
        }


@dataclass(frozen=True)
class QuizScopeSelection:
    """Resolved server-owned display labels and exact retrieval query."""

    topic_label: str
    knowledge_point_label: str | None
    retrieval_query: str
    focuses: tuple[QuizFocus, ...] = ()


def _normalise_fact_text(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _is_context_dependent(text: str) -> bool:
    return _CONTEXT_DEPENDENT.search(text) is not None


def _heading_tokens(text: str) -> set[str]:
    return {
        token
        for token in _FACT_TOKEN.findall(text.casefold())
        if token not in _FACT_STOPWORDS and len(token) > 1
    }


def _normalise_heading(text: str) -> str:
    return " ".join(text.split()).casefold()


def _is_generic_h4(label: str) -> bool:
    normalised = _normalise_heading(label)
    return normalised in _GENERIC_H4_EXACT or normalised.startswith(_GENERIC_H4_PREFIXES)


def _formula_h4_matches_focus(label: str | None, focus_tokens: set[str]) -> bool:
    if label is None:
        return False
    normalised = _normalise_heading(label)
    if normalised in {"equation", "formula"}:
        return True
    if not normalised.startswith(_FORMULA_H4_PREFIXES):
        return False
    descriptive_tokens = _heading_tokens(label) - _FORMULA_H4_GENERIC_TOKENS
    return bool(descriptive_tokens & focus_tokens)


def _is_excluded_h4(label: str | None) -> bool:
    if label is None:
        return False
    normalised = _normalise_heading(label)
    return normalised in {
        "cross-reference",
        "source provenance",
        "sources",
        "visual review",
    } or normalised.startswith("cross-reference:")


def _fact(
    *,
    note_path: str,
    focus_id: str,
    focus_heading: str,
    fact_type: str,
    text: str,
) -> QuizFact | None:
    display_text = _normalise_fact_text(text)
    if (
        len(display_text) < 12
        or _is_context_dependent(display_text)
        or _METADATA_FACT_PREFIX.match(display_text)
    ):
        return None
    if fact_type == "statement" and len(display_text.split()) < 5:
        return None
    if fact_type == "statement" and _LOWERCASE_FRAGMENT.match(display_text):
        return None
    identity = f"{note_path}\0{focus_id}\0{fact_type}\0{display_text}"
    return QuizFact(
        id=_opaque_id("fact", identity),
        focus_id=focus_id,
        focus_heading=focus_heading,
        fact_type=fact_type,
        display_text=display_text,
        note_path=note_path,
    )


def _extract_facts(
    *,
    note_path: str,
    focus_id: str,
    focus_heading: str,
    focus_label: str,
    lines: list[str],
) -> tuple[QuizFact, ...]:
    """Extract a small conservative set of complete statements and expressions."""
    facts: list[QuizFact] = []
    prose: list[str] = []
    equation: list[str] | None = None
    fence_marker: str | None = None
    current_h4: str | None = None
    target_tokens = _heading_tokens(focus_label)

    def fact_heading() -> str:
        return f"{focus_heading} — {current_h4}" if current_h4 else focus_heading

    def admit(candidate: QuizFact | None) -> None:
        if candidate is None:
            return
        if candidate.fact_type == "expression":
            if _formula_h4_matches_focus(current_h4, target_tokens):
                facts.append(candidate)
            return
        anchors = set(target_tokens)
        if current_h4 is not None and not _is_generic_h4(current_h4):
            anchors.update(_heading_tokens(current_h4))
        if _heading_tokens(candidate.display_text) & anchors:
            facts.append(candidate)

    def add_prose() -> None:
        if not prose:
            return
        text = _normalise_fact_text(" ".join(prose))
        prose.clear()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z`$\\(])", text):
            if not _SENTENCE_END.search(sentence):
                continue
            admit(
                _fact(
                    note_path=note_path,
                    focus_id=focus_id,
                    focus_heading=fact_heading(),
                    fact_type="statement",
                    text=sentence,
                )
            )

    for raw_line in lines:
        fence = _FENCE.match(raw_line)
        if fence:
            add_prose()
            marker = fence.group(1)[0]
            fence_marker = marker if fence_marker is None else None
            continue
        if fence_marker is not None:
            continue
        stripped = raw_line.strip()
        heading_match = _HEADING.match(raw_line)
        if heading_match is not None:
            add_prose()
            level = len(heading_match.group(1))
            if level == 4:
                current_h4 = heading_match.group(2).strip().rstrip("#").strip()
            continue
        if _is_excluded_h4(current_h4):
            continue
        if equation is not None:
            equation.append(stripped)
            if stripped.endswith("$$") or stripped.endswith(r"\]"):
                admit(
                    _fact(
                        note_path=note_path,
                        focus_id=focus_id,
                        focus_heading=fact_heading(),
                        fact_type="expression",
                        text=" ".join(equation),
                    )
                )
                equation = None
            continue
        if not stripped:
            add_prose()
            continue
        if stripped.startswith(("|", "![", ">")):
            add_prose()
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*", stripped):
            add_prose()
            continue
        if stripped.startswith("$$") or stripped.startswith(r"\["):
            add_prose()
            if (stripped.endswith("$$") and len(stripped) > 4) or (
                stripped.endswith(r"\]") and len(stripped) > 4
            ):
                admit(
                    _fact(
                        note_path=note_path,
                        focus_id=focus_id,
                        focus_heading=fact_heading(),
                        fact_type="expression",
                        text=stripped,
                    )
                )
            else:
                equation = [stripped]
            continue
        item = _LIST_ITEM.match(raw_line)
        if item is not None:
            add_prose()
            item_text = item.group(1).strip()
            if item_text.startswith("`") and "pp." in item_text:
                continue
            if _SENTENCE_END.search(item_text):
                admit(
                    _fact(
                        note_path=note_path,
                        focus_id=focus_id,
                        focus_heading=fact_heading(),
                        fact_type="statement",
                        text=item_text,
                    )
                )
            else:
                prose.append(item_text)
            continue
        prose.append(stripped)
    add_prose()
    unique: dict[tuple[str, str], QuizFact] = {}
    for candidate in facts:
        unique.setdefault((candidate.fact_type, candidate.display_text.casefold()), candidate)
    return tuple(unique.values())


def _remove_cross_focus_duplicates(focuses: tuple[QuizFocus, ...]) -> tuple[QuizFocus, ...]:
    owners: dict[tuple[str, str], set[str]] = {}
    for focus in focuses:
        for fact in focus.facts:
            key = (fact.fact_type, fact.display_text.casefold())
            owners.setdefault(key, set()).add(focus.id)
    return tuple(
        replace(
            focus,
            facts=tuple(
                fact
                for fact in focus.facts
                if len(owners[(fact.fact_type, fact.display_text.casefold())]) == 1
            ),
        )
        for focus in focuses
    )


def parse_admitted_note(note_path: str, markdown: str) -> QuizTopic | None:
    """Parse one admitted note into its exact H1/H2/H3 scope hierarchy."""
    lines = markdown.splitlines()
    headings: list[tuple[int, int, str, int]] = []
    ordinals: dict[int, int] = {1: 0, 2: 0, 3: 0}
    fence_marker: str | None = None
    for line_number, line in enumerate(lines):
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            continue
        if fence_marker is not None:
            continue
        match = _HEADING.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        if level > 3:
            continue
        label = match.group(2).strip().rstrip("#").strip()
        if not label:
            continue
        ordinals[level] += 1
        headings.append((level, ordinals[level], label, line_number))
    h1s = [heading for heading in headings if heading[0] == 1]
    if len(h1s) != 1:
        return None
    topic_label = h1s[0][2]
    points: list[KnowledgePoint] = []
    h3_focuses: list[QuizFocus] = []
    h2_focuses: list[QuizFocus] = []

    def body_between(start: int, end: int) -> str:
        body_lines = [line.strip() for line in lines[start:end] if not _HEADING.match(line)]
        return "\n".join(line for line in body_lines if line).strip()

    h2_indexes = [index for index, heading in enumerate(headings) if heading[0] == 2]
    for h2_position, heading_index in enumerate(h2_indexes):
        _, h2_ordinal, h2_label, h2_line = headings[heading_index]
        if " ".join(h2_label.split()).casefold() in STRUCTURAL_H2_DENYLIST:
            continue
        next_h2_line = (
            headings[h2_indexes[h2_position + 1]][3]
            if h2_position + 1 < len(h2_indexes)
            else len(lines)
        )
        child_h3s = [
            heading
            for heading in headings[heading_index + 1 :]
            if heading[0] == 3 and h2_line < heading[3] < next_h2_line
        ]
        h2_focus_ids: list[str] = []
        intro_end = child_h3s[0][3] if child_h3s else next_h2_line
        intro = body_between(h2_line + 1, intro_end)
        if intro:
            focus_id = _opaque_id("focus", f"{note_path}\0h2\0{h2_ordinal}\0{h2_label}")
            focus_heading = f"{topic_label} — {h2_label}"
            h2_focuses.append(
                QuizFocus(
                    id=focus_id,
                    label=h2_label,
                    heading=focus_heading,
                    context=f"{h2_label}\n{intro}",
                    note_path=note_path,
                    facts=_extract_facts(
                        note_path=note_path,
                        focus_id=focus_id,
                        focus_heading=focus_heading,
                        focus_label=h2_label,
                        lines=lines[h2_line + 1 : intro_end],
                    ),
                )
            )
            h2_focus_ids.append(focus_id)
        h3_points: list[KnowledgePoint] = []
        for child_index, (_, h3_ordinal, h3_label, h3_line) in enumerate(child_h3s):
            if " ".join(h3_label.split()).casefold() in STRUCTURAL_H2_DENYLIST:
                continue
            next_h3_line = (
                child_h3s[child_index + 1][3] if child_index + 1 < len(child_h3s) else next_h2_line
            )
            body = body_between(h3_line + 1, next_h3_line)
            focus_ids: tuple[str, ...] = ()
            if body:
                focus_id = _opaque_id("focus", f"{note_path}\0h3\0{h3_ordinal}\0{h3_label}")
                focus_heading = f"{topic_label} — {h2_label} — {h3_label}"
                h3_focuses.append(
                    QuizFocus(
                        id=focus_id,
                        label=h3_label,
                        heading=focus_heading,
                        context=f"{h3_label}\n{body}",
                        note_path=note_path,
                        facts=_extract_facts(
                            note_path=note_path,
                            focus_id=focus_id,
                            focus_heading=focus_heading,
                            focus_label=h3_label,
                            lines=lines[h3_line + 1 : next_h3_line],
                        ),
                    )
                )
                h2_focus_ids.append(focus_id)
                focus_ids = (focus_id,)
            h3_points.append(
                KnowledgePoint(
                    id=_opaque_id("point", f"{note_path}\0{3}\0{h3_ordinal}\0{h3_label}"),
                    label=h3_label,
                    breadcrumb=f"{h2_label} — {h3_label}",
                    level=3,
                    focus_ids=focus_ids,
                )
            )
        points.append(
            KnowledgePoint(
                id=_opaque_id("point", f"{note_path}\0{2}\0{h2_ordinal}\0{h2_label}"),
                label=h2_label,
                breadcrumb=h2_label,
                level=2,
                focus_ids=tuple(h2_focus_ids),
            )
        )
        points.extend(h3_points)
    focuses = _remove_cross_focus_duplicates(tuple(h3_focuses + h2_focuses))
    return QuizTopic(
        id=_opaque_id("topic", note_path),
        label=topic_label,
        knowledge_points=tuple(points),
        focuses=focuses,
    )


def build_scope_catalog(repo_root: Path, note_paths: tuple[str, ...]) -> tuple[QuizTopic, ...]:
    """Read only admitted note paths and return deterministic valid topics."""
    topics: list[QuizTopic] = []
    note_markdown: dict[str, str] = {}
    resolved_root = repo_root.resolve()
    for note_path in note_paths:
        if not note_path.startswith("rag/data/"):
            continue
        path = (resolved_root / note_path).resolve()
        if not path.is_relative_to(resolved_root) or not path.is_file():
            continue
        markdown = path.read_text(encoding="utf-8")
        note_markdown[note_path] = markdown
        topic = parse_admitted_note(note_path, markdown)
        if topic is not None:
            topics.append(topic)
    all_focuses = _remove_cross_focus_duplicates(
        tuple(focus for topic in topics for focus in topic.focuses)
    )
    by_id = {focus.id: focus for focus in all_focuses}
    parsed_topics = tuple(
        replace(topic, focuses=tuple(by_id[focus.id] for focus in topic.focuses))
        for topic in topics
    )
    return _bind_property_mappings(parsed_topics, note_markdown)


def _bind_property_mappings(
    topics: tuple[QuizTopic, ...], note_markdown: dict[str, str]
) -> tuple[QuizTopic, ...]:
    """Bind reviewed properties and conservative note facts to each focus."""
    from rag.quiz_properties import PROPERTY_MAPPINGS

    mappings = tuple(mapping for mapping in PROPERTY_MAPPINGS if mapping.note_path in note_markdown)
    facts_by_focus: dict[str, list[QuizFact]] = {}
    focus_by_heading = {focus.heading: focus for topic in topics for focus in topic.focuses}
    topic_by_focus = {focus.id: topic for topic in topics for focus in topic.focuses}
    for mapping in mappings:
        focus = focus_by_heading.get(mapping.focus_heading)
        if (
            focus is None
            or focus.note_path != mapping.note_path
            or not mapping.note_path.startswith(f"rag/data/{mapping.subject.lower()}/")
            or mapping.source_quote not in note_markdown[mapping.note_path]
            or mapping.source_quote not in focus.context
            or len(mapping.domain_values) != 4
            or len(set(mapping.domain_values)) != 4
            or mapping.domain_values.count(mapping.source_value) != 1
        ):
            raise QuizScopeError(
                f"Quiz property mapping {mapping.mapping_id} is not bound to its admitted source."
            )
        topic = topic_by_focus[focus.id]
        fact = QuizFact(
            id=mapping.mapping_id,
            focus_id=focus.id,
            focus_heading=focus.heading,
            fact_type="property",
            display_text=mapping.source_value,
            note_path=mapping.note_path,
            concept=mapping.concept,
            question_axis=mapping.question_axis,
            domain_id=mapping.domain_id,
            domain_values=mapping.domain_values,
            source_quote=mapping.source_quote,
            reference_title=topic.label,
        )
        facts_by_focus.setdefault(focus.id, []).append(fact)
    return tuple(
        replace(
            topic,
            focuses=tuple(
                replace(
                    focus,
                    facts=tuple(facts_by_focus.get(focus.id, ()))
                    + tuple(
                        replace(
                            fact,
                            concept=focus.label,
                            question_axis=(
                                "the source-backed expression"
                                if fact.fact_type == "expression"
                                else "the source-backed statement"
                            ),
                            source_quote=fact.display_text,
                            reference_title=topic.label,
                        )
                        for fact in focus.facts
                    ),
                )
                for focus in topic.focuses
            ),
        )
        for topic in topics
    )


def resolve_scope(
    topics: tuple[QuizTopic, ...],
    *,
    topic_id: str,
    knowledge_point_id: str | None,
) -> QuizScopeSelection:
    """Resolve client IDs against the current catalog and derive the exact query."""
    if topic_id == "mixed":
        if knowledge_point_id is not None:
            raise QuizScopeError("Mixed Topics does not accept a Knowledge Point.")
        return QuizScopeSelection(
            topic_label="Mixed Topics",
            knowledge_point_label=None,
            retrieval_query="Mixed Topics",
            focuses=tuple(focus for topic in topics for focus in topic.focuses),
        )
    topic = next((candidate for candidate in topics if candidate.id == topic_id), None)
    if topic is None:
        raise QuizScopeError("Quiz topic is not available in the current subject and mode.")
    if knowledge_point_id is None:
        return QuizScopeSelection(
            topic_label=topic.label,
            knowledge_point_label=None,
            retrieval_query=topic.label,
            focuses=topic.focuses,
        )
    point = next(
        (candidate for candidate in topic.knowledge_points if candidate.id == knowledge_point_id),
        None,
    )
    if point is None:
        raise QuizScopeError("Quiz knowledge point does not belong to the selected topic.")
    return QuizScopeSelection(
        topic_label=topic.label,
        knowledge_point_label=point.breadcrumb,
        retrieval_query=f"{topic.label} — {point.breadcrumb}",
        focuses=tuple(focus for focus in topic.focuses if focus.id in point.focus_ids),
    )
