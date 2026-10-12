"""Abstention and citation gates for subject-scoped RAG answers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PREFIX = "rag/data/"
CORPUS_ROOT_ENV = "EEE_RAG_CORPUS_ROOT"
ABSTAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*i\s+don'?t\s+know\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*i\s+do\s+not\s+know\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*cannot\s+answer\s+from\s+the\s+(?:notes|context)\.?\s*$", re.IGNORECASE),
)


@dataclass(frozen=True)
class Citation:
    """One grounded citation exposed to API clients."""

    note_path: str
    heading: str | None

    def as_dict(self) -> dict[str, str | None]:
        """Serialize for JSON responses."""
        return {"note_path": self.note_path, "heading": self.heading}


def citations_from_documents(documents: Iterable[Document]) -> list[Citation]:
    """Build ordered unique note+heading citations from retrieved chunks."""
    seen: set[tuple[str, str | None]] = set()
    citations: list[Citation] = []
    for document in documents:
        note_path = str(document.metadata.get("source", "")).strip()
        if not note_path:
            continue
        heading_raw = str(document.metadata.get("heading_path", "")).strip()
        heading = heading_raw or None
        key = (note_path, heading)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(note_path=note_path, heading=heading))
    return citations


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether one resolved path is inside another."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_corpus_root(
    *,
    repo_root: Path = REPO_ROOT,
    corpus_root: Path | None = None,
) -> Path | None:
    """Return the resolved note corpus root, or None when misconfigured.

    An injected ``corpus_root`` wins; otherwise ``EEE_RAG_CORPUS_ROOT`` is
    honored.  With neither, the repository ``rag/data`` directory is used.
    A configured root must be an absolute existing directory.
    """
    if corpus_root is not None:
        configured = corpus_root
    else:
        raw = os.getenv(CORPUS_ROOT_ENV, "").strip()
        if not raw:
            return (repo_root / CORPUS_PREFIX).resolve()
        configured = Path(raw)
    if not configured.is_absolute():
        return None
    try:
        resolved = configured.resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None
    return resolved


def citation_note_path(
    citation: Citation,
    *,
    repo_root: Path = REPO_ROOT,
    corpus_root: Path | None = None,
) -> Path | None:
    """Resolve a citation to the note file it must point at, or None if invalid.

    Only ``rag/data/<subject>/<file>`` citations are accepted.  The suffix
    after the exact ``rag/data/`` prefix is mapped into the configured corpus
    root, and the resolved target must be a regular file inside that root.
    Traversal, absolute components, symlink escapes, and missing or non-file
    targets all fail closed.
    """
    note_path = citation.note_path
    if not isinstance(note_path, str) or not note_path:
        return None
    if not note_path.startswith(CORPUS_PREFIX):
        return None
    suffix = note_path[len(CORPUS_PREFIX) :]
    if not suffix or "\x00" in suffix:
        return None
    try:
        relative = PurePosixPath(suffix)
    except ValueError:
        return None
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    root = _resolve_corpus_root(repo_root=repo_root, corpus_root=corpus_root)
    if root is None:
        return None
    try:
        resolved = (root / relative).resolve()
    except OSError:
        return None
    if not _is_relative_to(resolved, root):
        return None
    if not resolved.is_file():
        return None
    return resolved


def citation_is_valid(
    citation: Citation,
    *,
    repo_root: Path = REPO_ROOT,
    corpus_root: Path | None = None,
) -> bool:
    """Return whether a citation resolves to an existing note in the corpus root."""
    return citation_note_path(citation, repo_root=repo_root, corpus_root=corpus_root) is not None


def looks_like_abstention(answer: str) -> bool:
    """Detect explicit model abstention phrases."""
    cleaned = answer.strip()
    if not cleaned:
        return True
    return any(pattern.fullmatch(cleaned) for pattern in ABSTAIN_PATTERNS)


def apply_answer_gates(
    *,
    answer: str,
    documents: Sequence[Document],
    repo_root: Path = REPO_ROOT,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed: abstain without evidence; otherwise require valid citations."""
    citations = citations_from_documents(documents)
    if not documents or not citations:
        return {
            "status": "abstain",
            "reason": "no_retrieval_evidence",
            "answer": "I don't know.",
            "citations": [],
            "sources": [],
        }
    invalid = [
        item
        for item in citations
        if not citation_is_valid(item, repo_root=repo_root, corpus_root=corpus_root)
    ]
    if invalid:
        return {
            "status": "abstain",
            "reason": "invalid_citation",
            "answer": "I don't know.",
            "citations": [],
            "sources": [],
        }
    if looks_like_abstention(answer):
        return {
            "status": "abstain",
            "reason": "model_abstention",
            "answer": "I don't know.",
            "citations": [item.as_dict() for item in citations],
            "sources": list(dict.fromkeys(item.note_path for item in citations)),
        }
    # Non-abstained answers must expose at least one note+heading pair when heading exists;
    # if every chunk lacks heading_path, expose note_path with heading=None but keep status ok
    # only when note paths are valid (already checked). Prefer requiring a heading when available.
    headed = [item for item in citations if item.heading]
    exposed = headed or citations
    return {
        "status": "ok",
        "reason": None,
        "answer": answer.strip(),
        "citations": [item.as_dict() for item in exposed],
        "sources": list(dict.fromkeys(item.note_path for item in exposed)),
    }


def validate_response_citations(
    payload: Mapping[str, Any],
    *,
    allowed_note_paths: Sequence[str],
    repo_root: Path = REPO_ROOT,
    corpus_root: Path | None = None,
) -> list[str]:
    """Return validation errors for a non-abstained API-shaped payload."""
    errors: list[str] = []
    status = payload.get("status")
    if status == "abstain":
        return errors
    citations = payload.get("citations") or payload.get("sources")
    if not isinstance(citations, list) or not citations:
        return ["non-abstained answer is missing citations"]
    allowed = set(allowed_note_paths)
    for index, item in enumerate(citations):
        if isinstance(item, str):
            note_path = item
            heading = None
        elif isinstance(item, Mapping):
            note_path = str(item.get("note_path", ""))
            heading = item.get("heading")
            if heading is not None and not isinstance(heading, str):
                errors.append(f"citations[{index}].heading must be a string or null")
                continue
        else:
            errors.append(f"citations[{index}] has an unsupported type")
            continue
        citation = Citation(
            note_path=note_path,
            heading=heading if isinstance(heading, str) else None,
        )
        if note_path not in allowed:
            errors.append(f"citations[{index}] escapes retrieved note set: {note_path}")
        if not citation_is_valid(citation, repo_root=repo_root, corpus_root=corpus_root):
            errors.append(f"citations[{index}] is not a valid repository note: {note_path}")
    return errors
