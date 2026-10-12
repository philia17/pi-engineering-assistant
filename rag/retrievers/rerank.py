"""Deterministic reranking of dense retrieval candidates.

Scores each candidate chunk with stdlib-only lexical signals (no model, no new
dependencies) so results are reproducible across machines:

1. lexical overlap between question tokens and chunk text;
2. heading-path overlap between question tokens and the chunk's
   ``heading_path`` metadata (attached by ``rag.pipeline.split_documents``);
3. adjacent-card dedup: chunks sharing the same ``source`` and
   ``heading_path`` collapse to the single highest-scoring one.

Ties fall back to the original dense rank, so reranking never scrambles
chunks the scorer cannot distinguish.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Crude stemming: alphabetic tokens are truncated to this many characters so
# that morphological variants match (multiply / multiplying / multiplication
# -> "multip"; impedance / impedances -> "impeda"). Numeric tokens are kept
# verbatim because digits carry exact meaning in calculation questions.
_STEM_LENGTH = 6

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "give",
        "how",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "state",
        "that",
        "the",
        "then",
        "this",
        "to",
        "what",
        "when",
        "which",
        "why",
        "with",
    }
)

HEADING_WEIGHT = 0.5

# Note-template payload markers (see rag/NOTE_AUTHORING_GUIDE.md). When a
# section was split into several chunks, the slice carrying the section's
# conclusion ("**Final answer**" in worked examples, "**Formula**" in formula
# cards) is worth more to the generator than a "Given" slice that merely
# restates the question. Lower index = higher preference.
_PAYLOAD_MARKERS: tuple[str, ...] = ("**final answer**", "**formula**")

# Problem-statement slices restate a question (either the user's, which adds
# nothing, or a *neighboring* worked example's, which actively misleads the
# generator into copying that example's answer). The recovery slot never
# spends its budget on them.
_PROBLEM_MARKERS: tuple[str, ...] = ("**given**", "**find**")


def _is_problem_statement(doc: Document) -> bool:
    """Return True when a chunk is a Given/Find problem-statement slice."""
    text = doc.page_content.casefold()
    return any(marker in text for marker in _PROBLEM_MARKERS)


def _payload_priority(doc: Document) -> int:
    """Rank a chunk by which note-template payload marker it contains.

    A marker only counts when actual payload follows it inside the same
    chunk: section splitting can leave a chunk ending right at the marker
    header, with the formula or answer body in the next chunk.
    """
    text = doc.page_content.casefold()
    for priority, marker in enumerate(_PAYLOAD_MARKERS):
        index = text.find(marker)
        if index < 0:
            continue
        remainder = text[index + len(marker) :]
        if any(char.isdigit() or char in "$=" for char in remainder):
            return priority
    return len(_PAYLOAD_MARKERS)


def _normalize_token(token: str) -> str:
    """Map a raw token to its matching form (crude stem for alphabetic tokens)."""
    if token.isalpha() and len(token) > _STEM_LENGTH:
        return token[:_STEM_LENGTH]
    return token


def tokenize(text: str) -> set[str]:
    """Return the set of normalized, non-stopword tokens in ``text``."""
    return {
        _normalize_token(token)
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _STOPWORDS
    }


def score_chunk(question_tokens: set[str], doc: Document) -> float:
    """Score one candidate chunk against the question token set."""
    if not question_tokens:
        return 0.0
    text_tokens = tokenize(doc.page_content)
    heading_tokens = tokenize(str(doc.metadata.get("heading_path", "")))
    lexical = len(question_tokens & text_tokens) / len(question_tokens)
    heading = len(question_tokens & heading_tokens) / len(question_tokens)
    return lexical + HEADING_WEIGHT * heading


def rerank_chunks(
    question: str,
    docs: list[Document],
    *,
    keep: int = 5,
) -> list[Document]:
    """Score top-N dense candidates and return the best ``keep`` chunks.

    Chunks sharing the same ``source`` and ``heading_path`` metadata collapse
    to a single chunk before the ``keep`` bound is applied. The section is
    *ranked* by its highest-scoring member (a chunk restating the question's
    givens is strong evidence the section is relevant), but it is
    *represented* by the member holding the section's payload (final answer
    or formula slice), falling back to the best dense-ranked member: the
    lexically loudest chunk often just parrots the question, while the
    conclusion slice carries the content the answer actually needs. Chunks
    without a ``heading_path`` are never merged.

    The final slot is a *recovery* slot: when a collapsed slice of an
    already-selected section has a better dense rank than the next section's
    best member, that slice is re-admitted instead. Sections whose answer
    content spans two slices (e.g. formula plus sanity-check values) keep
    both, and a marginal fifth section is dropped in exchange.
    """
    if keep <= 0 or not docs:
        return []

    question_tokens = tokenize(question)
    groups: dict[object, list[tuple[int, Document]]] = {}
    scores: dict[object, float] = {}
    for dense_rank, doc in enumerate(docs):
        score = score_chunk(question_tokens, doc)
        heading_path = doc.metadata.get("heading_path")
        key: object = (
            (doc.metadata.get("source", ""), heading_path)
            if heading_path
            else ("__no_heading__", dense_rank)
        )
        groups.setdefault(key, []).append((dense_rank, doc))
        scores[key] = max(score, scores.get(key, 0.0))

    sections: list[tuple[float, int, Document, list[tuple[int, Document]]]] = []
    for key, members in groups.items():
        min_dense_rank = min(dense_rank for dense_rank, _ in members)
        _, representative = min(
            members,
            key=lambda member: (_payload_priority(member[1]), member[0]),
        )
        sections.append((scores[key], min_dense_rank, representative, members))

    ranked = sorted(sections, key=lambda entry: (-entry[0], entry[1]))

    if len(ranked) < keep:
        return [representative for _, _, representative, _ in ranked]

    head = ranked[: keep - 1]
    chosen = [representative for _, _, representative, _ in head]
    chosen_ids = {id(doc) for doc in chosen}
    leftovers = sorted(
        (
            (dense_rank, doc)
            for _, _, _, members in head
            for dense_rank, doc in members
            if id(doc) not in chosen_ids and not _is_problem_statement(doc)
        ),
        key=lambda member: member[0],
    )
    _, tail_min_rank, tail_representative, _ = ranked[keep - 1]
    if leftovers and leftovers[0][0] < tail_min_rank:
        chosen.append(leftovers[0][1])
    else:
        chosen.append(tail_representative)
    return chosen
