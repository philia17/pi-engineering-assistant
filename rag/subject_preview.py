"""Build and describe isolated, manifest-bound subject preview indexes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag import pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "rag" / "SUBJECT_CATALOG.json"
DATA_ROOT = REPO_ROOT / "rag" / "data"
CORPUS_ROOT_ENV = "EEE_RAG_CORPUS_ROOT"
INDEX_ROOT = REPO_ROOT / "artifacts" / "integration_v0" / "indexes"
CATALOG_KEYS = frozenset(
    {
        "subject",
        "name",
        "module",
        "year",
        "data_dir",
        "formal_note_count",
        "owners",
        "source_owners",
        "source_owner_state",
        "review_status_counts",
        "index_state",
        "evaluation_state",
        "source_manifest_ref",
    }
)
SUBJECT_CODE_PATTERN = frozenset({"PE", "IS", "EM", "SEP", "EECC", "EPC", "MMT", "DIES"})


class SubjectPreviewError(ValueError):
    """Raised when the preview catalog or its isolated index is invalid."""


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether one absolute path is inside another."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _real_existing_directory(path: Path, *, context: str) -> Path:
    """Require an absolute existing directory with no symlink path component."""
    if not path.is_absolute():
        raise SubjectPreviewError(f"{context} must be an absolute path")
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            details = current.lstat()
        except FileNotFoundError as error:
            raise SubjectPreviewError(f"{context} does not exist") from error
        if stat.S_ISLNK(details.st_mode):
            raise SubjectPreviewError(f"{context} must not contain symbolic links")
        if not stat.S_ISDIR(details.st_mode):
            raise SubjectPreviewError(f"{context} must be a directory")
    return absolute


def corpus_root() -> Path:
    """Return the fixed local corpus or a validated external read-only reference.

    The returned path is only used for reads.  A configured override must be a
    real absolute directory outside this runtime release; it is never created,
    copied, indexed into, or otherwise mutated here.
    """
    configured = os.getenv(CORPUS_ROOT_ENV, "").strip()
    if not configured:
        return DATA_ROOT
    root = _real_existing_directory(Path(configured), context=CORPUS_ROOT_ENV)
    if _is_relative_to(root, REPO_ROOT):
        raise SubjectPreviewError(f"{CORPUS_ROOT_ENV} must be external to the runtime release")
    if root.name != "data" or root.parent.name != "rag":
        raise SubjectPreviewError(f"{CORPUS_ROOT_ENV} must identify an existing rag/data root")
    return root


def corpus_note_path(source: str) -> Path:
    """Resolve canonical ``rag/data`` metadata beneath the selected corpus root."""
    prefix = "rag/data/"
    if not source.startswith(prefix):
        raise SubjectPreviewError(f"Invalid corpus note path: {source!r}")
    relative = Path(source.removeprefix(prefix))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SubjectPreviewError(f"Invalid corpus note path: {source!r}")
    root = corpus_root()
    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            details = current.lstat()
        except FileNotFoundError as error:
            raise SubjectPreviewError(f"Catalogued note is missing: {source}") from error
        if stat.S_ISLNK(details.st_mode):
            raise SubjectPreviewError(f"Catalogued note path contains a symlink: {source}")
        is_final = index == len(relative.parts) - 1
        if is_final and not stat.S_ISREG(details.st_mode):
            raise SubjectPreviewError(f"Catalogued note is not a regular file: {source}")
        if not is_final and not stat.S_ISDIR(details.st_mode):
            raise SubjectPreviewError(f"Catalogued note path is invalid: {source}")
    return current


@dataclass(frozen=True)
class SubjectRecord:
    """One validated subject record from the frozen catalog."""

    subject: str
    name: str
    module: str
    year: int
    data_dir: str
    formal_note_count: int
    review_status_counts: dict[str, int]
    index_state: str
    evaluation_state: str


@dataclass(frozen=True)
class SubjectIndexBuild:
    """A reusable description of one preview-index build."""

    subject: str
    input_sha256: str
    manifest_path: Path
    index_path: Path
    note_count: int
    reused: bool


@dataclass(frozen=True)
class SubjectIndexStatus:
    """The current readiness state for one manifest-bound preview index."""

    subject: str
    module: str
    ready: bool
    reason: str | None
    input_sha256: str
    manifest_path: Path


class SubjectPreviewService:
    """Lazy, subject-isolated retrieval service for the Integration v0 preview."""

    def __init__(self) -> None:
        self._chains: dict[str, Any] = {}

    def index_status(self, subject: str) -> SubjectIndexStatus:
        """Return current readiness without falling back to another subject index."""
        record = resolve_subject(subject)
        _documents, inventory = _note_inventory(record)
        build_input = _build_input(record, inventory)
        input_sha256 = sha256_json(build_input)
        chroma_path = index_directory(record, input_sha256)
        manifest_path = chroma_path.parent / "index_manifest.json"
        manifest = _read_existing_manifest(manifest_path)
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
            manifest.get("subject") != record.subject
            or manifest.get("input_sha256") != input_sha256
            or manifest.get("index_path") != expected_index_path
            or not chroma_path.is_dir()
            or not any(chroma_path.iterdir())
            or chroma_document_count(chroma_path) < 1
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

    def get_chain(self, subject: str) -> Any:
        """Load and cache exactly one ready subject chain."""
        status = self.index_status(subject)
        if not status.ready:
            raise SubjectPreviewError(
                f"{status.subject} preview index is unavailable: {status.reason}"
            )
        if status.subject not in self._chains:
            index_path = status.manifest_path.parent / "chroma"
            vectorstore = pipeline.load_vectorstore(index_path)
            self._chains[status.subject] = pipeline.build_rag_chain(vectorstore)
        return self._chains[status.subject]

    def get_vectorstore(self, subject: str) -> Any:
        """Return the isolated Chroma store bound to one ready subject index."""
        chain = self.get_chain(subject)
        vectorstore = getattr(chain.retriever, "vectorstore", None)
        if vectorstore is None:
            raise SubjectPreviewError(f"{subject} preview retriever has no vectorstore")
        return vectorstore

    def answer(
        self,
        subject: str,
        question: str,
        *,
        model: str | None = None,
        retrieval_query: object | None = None,
    ) -> dict[str, object]:
        """Answer one explicit-subject preview question with sources and status."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise SubjectPreviewError("Question must not be empty")
        if len(cleaned_question) > 4_000:
            raise SubjectPreviewError("Question exceeds 4000 characters")
        record = resolve_subject(subject)
        started = time.perf_counter()
        chain = self.get_chain(record.subject)
        vectorstore = self.get_vectorstore(record.subject)

        from rag.preview_calculators import try_subject_calculator

        routed = try_subject_calculator(record.subject, cleaned_question, vectorstore)
        if routed is not None:
            if record.subject == "EECC" and not has_approved_source_route(record.subject):
                # Calculator arithmetic is deterministic, but an EECC answer is not
                # source-grounded until the missing lecture corpus is registered.
                # Do not turn a retrieved preview note (or the calculator's internal
                # formula label) into a fabricated course citation.
                return {
                    "status": "abstain",
                    "mode": "preview",
                    "subject": record.subject,
                    "module": record.module,
                    "model": model or pipeline.LLM_MODEL,
                    "answer": (
                        "I cannot provide a source-grounded EECC calculation because "
                        "this subject has no approved source route yet."
                    ),
                    "sources": [],
                    "citations": [],
                    "route": routed.tool_name,
                    "calculator_values": {},
                    "abstain_reason": "uncited_no_approved_source_route",
                    "trace": [
                        {
                            "stage": "calculator_route",
                            "status": "rejected_no_approved_source_route",
                            "tool": routed.tool_name,
                        },
                        *routed.trace,
                    ],
                    "review": {
                        "review_status_counts": record.review_status_counts,
                        "warnings": [
                            "preview_index_not_production_eligible",
                            "uncited_no_approved_source_route",
                        ],
                    },
                    "timing_ms": round((time.perf_counter() - started) * 1_000, 1),
                }
            return {
                "status": "ok",
                "mode": "preview",
                "subject": record.subject,
                "module": record.module,
                "model": model or pipeline.LLM_MODEL,
                "answer": routed.answer,
                "sources": routed.sources,
                "route": routed.tool_name,
                "calculator_values": routed.values,
                "trace": [
                    {
                        "stage": "calculator_route",
                        "status": "ok",
                        "tool": routed.tool_name,
                    },
                    *routed.trace,
                ],
                "review": {
                    "review_status_counts": record.review_status_counts,
                    "warnings": ["preview_index_not_production_eligible"],
                },
                "timing_ms": round((time.perf_counter() - started) * 1_000, 1),
            }

        safe_retrieval_query = pipeline.normalize_retrieval_query(retrieval_query)
        if safe_retrieval_query is None:
            result = pipeline.query(
                cleaned_question,
                chain,
                model=model,
                vectorstore=vectorstore,
                rerank=True,
            )
        else:
            result = pipeline.query(
                cleaned_question,
                chain,
                model=model,
                vectorstore=vectorstore,
                rerank=True,
                retrieval_query=safe_retrieval_query,
            )
        from rag.answer_gates import apply_answer_gates

        gated = apply_answer_gates(
            answer=str(result.get("answer", "")),
            documents=list(result.get("documents") or []),
        )
        return {
            "status": gated["status"],
            "mode": "preview",
            "subject": record.subject,
            "module": record.module,
            "model": model or pipeline.LLM_MODEL,
            "answer": gated["answer"],
            "sources": gated["sources"],
            "citations": gated["citations"],
            "route": "rag_fallback",
            "abstain_reason": gated["reason"],
            "trace": [
                {"stage": "calculator_route", "status": "miss"},
                {
                    "stage": "preview_rag",
                    "status": gated["status"],
                    "reranked": bool(result.get("reranked")),
                },
            ],
            "review": {
                "review_status_counts": record.review_status_counts,
                "warnings": ["preview_index_not_production_eligible"],
            },
            "timing_ms": round((time.perf_counter() - started) * 1_000, 1),
        }


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    """Return the canonical JSON digest for a metadata-only value."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_catalog(path: Path) -> dict[str, Any]:
    """Read the catalog as one JSON object."""
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubjectPreviewError(f"Cannot read subject catalog: {path}") from error
    if not isinstance(catalog, dict):
        raise SubjectPreviewError("Subject catalog must be a JSON object")
    return catalog


def load_subject_catalog(path: Path | None = None) -> dict[str, SubjectRecord]:
    """Load exactly the eight catalog subjects without accepting path aliases."""
    catalog = _read_catalog(CATALOG_PATH if path is None else path)
    subjects = catalog.get("subjects")
    if not isinstance(subjects, list):
        raise SubjectPreviewError("Subject catalog must contain a subjects array")

    records: dict[str, SubjectRecord] = {}
    for value in subjects:
        if not isinstance(value, dict) or set(value) != CATALOG_KEYS:
            raise SubjectPreviewError("Subject catalog record has an unexpected schema")
        subject = value.get("subject")
        data_dir = value.get("data_dir")
        name = value.get("name")
        module = value.get("module")
        year = value.get("year")
        note_count = value.get("formal_note_count")
        review_counts = value.get("review_status_counts")
        index_state = value.get("index_state")
        evaluation_state = value.get("evaluation_state")
        if (
            not isinstance(subject, str)
            or subject not in SUBJECT_CODE_PATTERN
            or not isinstance(data_dir, str)
            or data_dir != f"rag/data/{subject.lower()}"
            or not isinstance(name, str)
            or not isinstance(module, str)
            or not isinstance(year, int)
            or not isinstance(note_count, int)
            or note_count < 0
            or not isinstance(review_counts, dict)
            or not isinstance(index_state, str)
            or not isinstance(evaluation_state, str)
        ):
            raise SubjectPreviewError(f"Invalid subject catalog record for {subject!r}")
        if subject in records:
            raise SubjectPreviewError(f"Duplicate subject catalog code: {subject}")
        records[subject] = SubjectRecord(
            subject=subject,
            name=name,
            module=module,
            year=year,
            data_dir=data_dir,
            formal_note_count=note_count,
            review_status_counts={str(key): int(count) for key, count in review_counts.items()},
            index_state=index_state,
            evaluation_state=evaluation_state,
        )
    if frozenset(records) != SUBJECT_CODE_PATTERN:
        raise SubjectPreviewError("Subject catalog must contain the canonical eight subjects")
    return records


def resolve_subject(
    subject: str,
    *,
    catalog: dict[str, SubjectRecord] | None = None,
) -> SubjectRecord:
    """Resolve one canonical subject code without accepting client path input."""
    normalized = subject.strip().upper()
    records = load_subject_catalog() if catalog is None else catalog
    try:
        return records[normalized]
    except KeyError as error:
        raise SubjectPreviewError(f"Unknown subject: {subject}") from error


def subject_glob(record: SubjectRecord) -> str:
    """Return the constrained root-relative Markdown glob for one subject."""
    return f"{record.subject.lower()}/**/*.md"


def index_directory(record: SubjectRecord, input_sha256: str) -> Path:
    """Return the content-addressed index directory under the fixed artifact root."""
    return INDEX_ROOT / record.subject.lower() / input_sha256 / "chroma"


def chroma_document_count(chroma_path: Path) -> int:
    """Return a persisted document count without opening a writable Chroma client.

    ``chromadb.PersistentClient`` is not a read-only API: opening an old HNSW
    store can migrate ``max_seq_id`` into SQLite and rewrite HNSW state.  Health
    and candidate-adjacent status checks must not mutate frozen index bytes, so
    this deliberately uses SQLite's immutable read-only URI instead.  A missing
    schema, sidecar journal, unreadable file, or empty collection is unavailable
    rather than a reason to fall back to a writable Chroma open.
    """
    if not chroma_path.is_dir():
        return 0
    database = chroma_path / "chroma.sqlite3"
    if not database.is_file() or any(
        (chroma_path / f"chroma.sqlite3{suffix}").exists() for suffix in ("-shm", "-wal")
    ):
        return 0
    try:
        connection = sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            collection_count = int(
                connection.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
            )
            embedding_count = int(
                connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError, IndexError):
        return 0
    return embedding_count if collection_count > 0 else 0


def _note_inventory(record: SubjectRecord) -> tuple[list[Any], list[dict[str, str]]]:
    """Load one subject through the common parser and hash its exact note bytes."""
    documents = pipeline.load_documents(corpus_root(), glob=subject_glob(record))
    source_prefix = f"rag/data/{record.subject.lower()}/"
    inventory: list[dict[str, str]] = []
    for document in documents:
        source = str(document.metadata.get("source", ""))
        if not source.startswith(source_prefix):
            raise SubjectPreviewError(
                f"{record.subject} document escaped subject scope: {source!r}"
            )
        path = corpus_note_path(source)
        inventory.append({"path": source, "sha256": sha256_file(path)})
    inventory.sort(key=lambda item: item["path"])
    if len(inventory) != record.formal_note_count:
        raise SubjectPreviewError(
            f"{record.subject} expected {record.formal_note_count} notes, found {len(inventory)}"
        )
    return documents, inventory


def _build_input(record: SubjectRecord, inventory: list[dict[str, str]]) -> dict[str, object]:
    """Build the metadata whose hash defines one reproducible preview index."""
    return {
        "schema_version": "1.0",
        "mode": "preview",
        "subject": record.subject,
        "module": record.module,
        "notes": inventory,
        "chunk_size": int(os.getenv("EEE_RAG_CHUNK_SIZE", "512")),
        "chunk_overlap": int(os.getenv("EEE_RAG_CHUNK_OVERLAP", "64")),
        "embedding_model": pipeline.EMBED_MODEL,
    }


def _read_existing_manifest(path: Path) -> dict[str, Any] | None:
    """Return an existing build manifest only when it is a JSON object."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubjectPreviewError(f"Invalid preview index manifest: {path}") from error
    if not isinstance(value, dict):
        raise SubjectPreviewError(f"Preview index manifest must be an object: {path}")
    return value


def has_approved_source_route(subject: str) -> bool:
    """Return whether a subject has at least one registered, approved note route.

    This deliberately reads only the source manifest.  Preview-index content is
    useful for navigation, but cannot itself upgrade an unregistered note into
    a v1-approved citation.
    """
    path = REPO_ROOT / "rag" / "evals" / "sources" / subject.lower() / "source_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    mappings = payload.get("note_mappings") if isinstance(payload, dict) else None
    return isinstance(mappings, list) and any(
        isinstance(mapping, dict) and mapping.get("eligible_source_route") is True
        for mapping in mappings
    )


def build_subject_preview_index(subject: str) -> SubjectIndexBuild:
    """Build or safely reuse one content-addressed, subject-isolated index."""
    record = resolve_subject(subject)
    documents, inventory = _note_inventory(record)
    build_input = _build_input(record, inventory)
    input_sha256 = sha256_json(build_input)
    chroma_path = index_directory(record, input_sha256)
    manifest_path = chroma_path.parent / "index_manifest.json"
    existing = _read_existing_manifest(manifest_path)
    reusable = (
        existing is not None
        and existing.get("input_sha256") == input_sha256
        and chroma_path.is_dir()
        and any(chroma_path.iterdir())
        and chroma_document_count(chroma_path) >= 1
    )
    if reusable:
        return SubjectIndexBuild(
            subject=record.subject,
            input_sha256=input_sha256,
            manifest_path=manifest_path,
            index_path=chroma_path,
            note_count=len(inventory),
            reused=True,
        )

    if chroma_path.exists() or manifest_path.is_file():
        # A content-addressed directory must never be silently deleted and rebuilt:
        # it may be a partial/stale build that needs operator inspection.  Rebuilding
        # under the same hash would make an interrupted artifact look trustworthy.
        raise SubjectPreviewError(
            f"Preview index directory exists without a valid manifest-bound store: {chroma_path}"
        )
    chroma_path.mkdir(parents=True, exist_ok=False)
    pipeline.build_vectorstore(documents, persist_directory=chroma_path)
    manifest = {
        "schema_version": "1.0",
        "mode": "preview",
        "subject": record.subject,
        "module": record.module,
        "input": build_input,
        "input_sha256": input_sha256,
        "index_path": chroma_path.relative_to(REPO_ROOT).as_posix(),
        "built_at": datetime.now(timezone.utc).isoformat(),
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
