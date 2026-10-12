"""Formal eight-subject v1 indexes with eligibility filtering and hash invalidation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag import pipeline
from rag.answer_gates import apply_answer_gates
from rag.subject_preview import (
    REPO_ROOT,
    SubjectIndexBuild,
    SubjectIndexStatus,
    SubjectPreviewError,
    SubjectRecord,
    corpus_note_path,
    corpus_root,
    resolve_subject,
    sha256_file,
    sha256_json,
    subject_glob,
)

V1_INDEX_ROOT = REPO_ROOT / "artifacts" / "integration_v1" / "indexes"
SOURCE_MANIFEST_ROOT = REPO_ROOT / "rag" / "evals" / "sources"
ELIGIBILITY_POLICY = os.getenv("EEE_V1_NOTE_POLICY", "eligible_route")


class SubjectIndexError(SubjectPreviewError):
    """Raised when a formal v1 subject index cannot be built or loaded."""


def v1_index_directory(record: SubjectRecord, input_sha256: str) -> Path:
    """Return the content-addressed v1 chroma directory for one subject."""
    return V1_INDEX_ROOT / record.subject.lower() / input_sha256 / "chroma"


def load_eligible_note_paths(subject: str) -> list[str]:
    """Return production-releasable note paths for one subject from its manifest."""
    subject_key = subject.strip().lower()
    manifest_path = SOURCE_MANIFEST_ROOT / subject_key / "source_manifest.json"
    if not manifest_path.is_file():
        raise SubjectIndexError(f"Missing source manifest for {subject}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mappings = payload.get("note_mappings")
    if not isinstance(mappings, list):
        raise SubjectIndexError(f"Invalid note_mappings in {manifest_path}")
    eligible: list[str] = []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        note_path = mapping.get("note_path")
        if not isinstance(note_path, str):
            continue
        issue = mapping.get("mapping_issue")
        if isinstance(issue, str) and issue.startswith("Quarantined"):
            continue
        if ELIGIBILITY_POLICY == "eligible_route":
            if mapping.get("eligible_source_route") is True:
                eligible.append(note_path)
            continue
        if ELIGIBILITY_POLICY == "formal_non_draft":
            # Development-only broader set; still excludes explicit quarantine.
            try:
                path = corpus_note_path(note_path)
            except SubjectPreviewError:
                continue
            text = path.read_text(encoding="utf-8")
            if "review_status: draft" in text:
                continue
            eligible.append(note_path)
            continue
        raise SubjectIndexError(f"Unsupported EEE_V1_NOTE_POLICY={ELIGIBILITY_POLICY!r}")
    return sorted(set(eligible))


def _eligible_inventory(
    record: SubjectRecord,
) -> tuple[list[Any], list[dict[str, str]]]:
    """Load only eligible notes for one subject and hash their exact bytes."""
    eligible_paths = set(load_eligible_note_paths(record.subject))
    documents = pipeline.load_documents(corpus_root(), glob=subject_glob(record))
    source_prefix = f"rag/data/{record.subject.lower()}/"
    inventory: list[dict[str, str]] = []
    selected_docs: list[Any] = []
    for document in documents:
        source = str(document.metadata.get("source", ""))
        if not source.startswith(source_prefix):
            raise SubjectIndexError(f"{record.subject} document escaped subject scope: {source!r}")
        if source not in eligible_paths:
            continue
        try:
            path = corpus_note_path(source)
        except SubjectPreviewError as error:
            raise SubjectIndexError(f"Eligible note is missing: {source}") from error
        inventory.append({"path": source, "sha256": sha256_file(path)})
        selected_docs.append(document)
    inventory.sort(key=lambda item: item["path"])
    selected_docs.sort(key=lambda doc: str(doc.metadata.get("source", "")))
    return selected_docs, inventory


def _build_v1_input(record: SubjectRecord, inventory: list[dict[str, str]]) -> dict[str, object]:
    """Build the metadata whose hash defines one reproducible v1 index."""
    return {
        "schema_version": "1.0",
        "mode": "v1",
        "eligibility_policy": ELIGIBILITY_POLICY,
        "subject": record.subject,
        "module": record.module,
        "notes": inventory,
        "chunk_size": int(os.getenv("EEE_RAG_CHUNK_SIZE", "512")),
        "chunk_overlap": int(os.getenv("EEE_RAG_CHUNK_OVERLAP", "64")),
        "embedding_model": pipeline.EMBED_MODEL,
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    """Return an existing v1 manifest object when present."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubjectIndexError(f"Invalid v1 index manifest: {path}") from error
    if not isinstance(value, dict):
        raise SubjectIndexError(f"v1 index manifest must be an object: {path}")
    return value


def subject_index_status(subject: str) -> SubjectIndexStatus:
    """Return readiness for one formal v1 subject index."""
    record = resolve_subject(subject)
    _documents, inventory = _eligible_inventory(record)
    build_input = _build_v1_input(record, inventory)
    input_sha256 = sha256_json(build_input)
    chroma_path = v1_index_directory(record, input_sha256)
    manifest_path = chroma_path.parent / "index_manifest.json"
    if not inventory:
        return SubjectIndexStatus(
            subject=record.subject,
            module=record.module,
            ready=False,
            reason="no_eligible_notes",
            input_sha256=input_sha256,
            manifest_path=manifest_path,
        )
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return SubjectIndexStatus(
            subject=record.subject,
            module=record.module,
            ready=False,
            reason="index_not_built",
            input_sha256=input_sha256,
            manifest_path=manifest_path,
        )
    expected_index_path = chroma_path.relative_to(REPO_ROOT).as_posix()
    if (
        manifest.get("mode") != "v1"
        or manifest.get("subject") != record.subject
        or manifest.get("input_sha256") != input_sha256
        or manifest.get("index_path") != expected_index_path
        or not chroma_path.is_dir()
        or not any(chroma_path.iterdir())
    ):
        return SubjectIndexStatus(
            subject=record.subject,
            module=record.module,
            ready=False,
            reason="index_stale_or_incomplete",
            input_sha256=input_sha256,
            manifest_path=manifest_path,
        )
    return SubjectIndexStatus(
        subject=record.subject,
        module=record.module,
        ready=True,
        reason=None,
        input_sha256=input_sha256,
        manifest_path=manifest_path,
    )


def build_subject_v1_index(subject: str) -> SubjectIndexBuild:
    """Build or reuse one eligibility-filtered, content-addressed v1 index."""
    record = resolve_subject(subject)
    documents, inventory = _eligible_inventory(record)
    build_input = _build_v1_input(record, inventory)
    input_sha256 = sha256_json(build_input)
    chroma_path = v1_index_directory(record, input_sha256)
    manifest_path = chroma_path.parent / "index_manifest.json"
    if not inventory:
        raise SubjectIndexError(
            f"{record.subject} has no production-eligible notes under policy {ELIGIBILITY_POLICY}"
        )
    existing = _read_manifest(manifest_path)
    if existing is not None:
        if (
            existing.get("input_sha256") != input_sha256
            or existing.get("mode") != "v1"
            or not chroma_path.is_dir()
            or not any(chroma_path.iterdir())
        ):
            raise SubjectIndexError(f"Existing v1 build is incomplete: {manifest_path}")
        return SubjectIndexBuild(
            subject=record.subject,
            input_sha256=input_sha256,
            manifest_path=manifest_path,
            index_path=chroma_path,
            note_count=len(inventory),
            reused=True,
        )
    if chroma_path.exists():
        raise SubjectIndexError(
            f"v1 index directory exists without a valid manifest: {chroma_path}"
        )
    chroma_path.mkdir(parents=True, exist_ok=False)
    pipeline.build_vectorstore(documents, persist_directory=chroma_path)
    manifest = {
        "schema_version": "1.0",
        "mode": "v1",
        "eligibility_policy": ELIGIBILITY_POLICY,
        "subject": record.subject,
        "module": record.module,
        "input": build_input,
        "input_sha256": input_sha256,
        "index_path": chroma_path.relative_to(REPO_ROOT).as_posix(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "note_count": len(inventory),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SubjectIndexBuild(
        subject=record.subject,
        input_sha256=input_sha256,
        manifest_path=manifest_path,
        index_path=chroma_path,
        note_count=len(inventory),
        reused=False,
    )


def changed_subjects(subjects: list[str] | None = None) -> list[str]:
    """Return subjects whose current eligible inventory is not ready."""
    from rag.subject_preview import load_subject_catalog

    catalog = load_subject_catalog()
    requested = subjects or sorted(catalog)
    changed: list[str] = []
    for subject in requested:
        status = subject_index_status(subject)
        if not status.ready and status.reason != "no_eligible_notes":
            changed.append(status.subject)
    return changed


class SubjectIndexService:
    """Lazy retrieval service bound to formal v1 subject indexes."""

    def __init__(self) -> None:
        self._chains: dict[str, Any] = {}
        self._stores: dict[str, Any] = {}

    def index_status(self, subject: str) -> SubjectIndexStatus:
        """Return formal v1 readiness for one subject."""
        return subject_index_status(subject)

    def get_vectorstore(self, subject: str) -> Any:
        """Load the isolated Chroma store for one ready v1 subject index."""
        status = self.index_status(subject)
        if not status.ready:
            raise SubjectIndexError(f"{status.subject} v1 index is unavailable: {status.reason}")
        if status.subject not in self._stores:
            index_path = status.manifest_path.parent / "chroma"
            self._stores[status.subject] = pipeline.load_vectorstore(index_path)
        return self._stores[status.subject]

    def get_chain(self, subject: str) -> Any:
        """Load a RetrievalQA chain for one ready v1 subject index."""
        status = self.index_status(subject)
        if not status.ready:
            raise SubjectIndexError(f"{status.subject} v1 index is unavailable: {status.reason}")
        if status.subject not in self._chains:
            vectorstore = self.get_vectorstore(status.subject)
            self._chains[status.subject] = pipeline.build_rag_chain(vectorstore)
        return self._chains[status.subject]

    def answer(
        self,
        subject: str,
        question: str,
        *,
        model: str | None = None,
        retrieval_query: object | None = None,
    ) -> dict[str, object]:
        """Answer from one formal v1 subject index with rerank and answer gates."""
        cleaned = question.strip()
        if not cleaned:
            raise SubjectIndexError("Question must not be empty")
        record = resolve_subject(subject)
        chain = self.get_chain(record.subject)
        vectorstore = self.get_vectorstore(record.subject)
        safe_retrieval_query = pipeline.normalize_retrieval_query(retrieval_query)
        if safe_retrieval_query is None:
            result = pipeline.query(
                cleaned,
                chain,
                model=model,
                vectorstore=vectorstore,
                rerank=True,
            )
        else:
            result = pipeline.query(
                cleaned,
                chain,
                model=model,
                vectorstore=vectorstore,
                rerank=True,
                retrieval_query=safe_retrieval_query,
            )
        gated = apply_answer_gates(
            answer=str(result.get("answer", "")),
            documents=list(result.get("documents") or []),
        )
        return {
            "status": gated["status"],
            "mode": "v1",
            "subject": record.subject,
            "module": record.module,
            "model": model or pipeline.LLM_MODEL,
            "answer": gated["answer"],
            "sources": gated["sources"],
            "citations": gated["citations"],
            "route": "subject_v1_rag",
            "abstain_reason": gated["reason"],
            "review": {
                "eligibility_policy": ELIGIBILITY_POLICY,
                "warnings": [],
            },
        }
