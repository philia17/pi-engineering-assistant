"""Create and verify external mutable Chroma copies for materialized v2 releases.

Opening a persistent Chroma store is not byte-neutral on legacy HNSW layouts.
Materialized FI-013 releases therefore never hand their pinned index directory
directly to Chroma.  The first retrieval copies one approved source index into
an external owner-only root and binds the copy to the release's self-hashed
materialization receipt.  Later retrievals may reuse only that receipt-bound
external copy; a partial or unverifiable workspace fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MATERIALIZATION_RECEIPT_NAME = "engineering_candidate_materialization.json"
MUTABLE_INDEX_COPY_RECEIPT_NAME = "mutable_index_copy_receipt.json"
MUTABLE_INDEX_ROOT_ENV = "EEE_RAG_MUTABLE_INDEX_ROOT"
MUTABLE_INDEX_COPY_SCHEMA_VERSION = "eee.mutable_index_copy.v1"
MATERIALIZATION_SCHEMA_VERSION = "eee.engineering_candidate_materialization.v1"
PREVIEW_MUTABLE_COPY_SCHEMA_VERSION = "eee.preview_mutable_index_copy.v1"
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_SESSION_ID = uuid.uuid4().hex


class MutableIndexRuntimeError(ValueError):
    """Raised when a materialized release cannot prove a safe mutable index copy."""


@dataclass(frozen=True)
class _TreeEntry:
    """One immutable tree entry used for source/copy identity checks."""

    path: str
    mode: str
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class MutableIndexCopy:
    """One verified external mutable copy selected for a Chroma open."""

    source_index: Path
    mutable_index: Path
    receipt_path: Path
    source_tree_sha256: str
    materialization_receipt_sha256: str


def _canonical_json(value: Any) -> bytes:
    """Encode finite JSON exactly as FI-013 self-hashed receipts require."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MutableIndexRuntimeError("mutable index receipt must be finite JSON") from error


def _sha256_json(value: Any) -> str:
    """Return the canonical finite-JSON digest for a receipt payload."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _lstat(path: Path, *, context: str) -> os.stat_result:
    """Read a path entry without following a possible symbolic link."""
    try:
        return path.lstat()
    except FileNotFoundError as error:
        raise MutableIndexRuntimeError(f"{context}: missing path") from error


def _real_directory(path: Path, *, context: str) -> Path:
    """Require one existing absolute directory and reject all symlink ancestors."""
    if not path.is_absolute():
        raise MutableIndexRuntimeError(f"{context}: path must be absolute")
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        details = _lstat(current, context=context)
        if stat.S_ISLNK(details.st_mode):
            raise MutableIndexRuntimeError(f"{context}: symbolic links are forbidden")
        if not stat.S_ISDIR(details.st_mode):
            raise MutableIndexRuntimeError(f"{context}: path component is not a directory")
    return absolute


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether a canonical path is below another canonical path."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    """Hash one stable regular non-symlink file."""
    before = _lstat(path, context="index tree")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise MutableIndexRuntimeError("index tree: expected a regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise MutableIndexRuntimeError("index tree: unreadable file") from error
    after = _lstat(path, context="index tree")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise MutableIndexRuntimeError("index tree: file changed while hashing")
    return digest.hexdigest()


def _tree_entries(root: Path) -> tuple[_TreeEntry, ...]:
    """Fingerprint every regular file below an immutable source/copy tree."""
    directory = _real_directory(root, context="index tree")
    entries: list[_TreeEntry] = []
    for current, directories, files in os.walk(directory, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            details = _lstat(current_path / name, context="index tree")
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise MutableIndexRuntimeError("index tree: directory link or special file")
        for name in sorted(files):
            path = current_path / name
            details = _lstat(path, context="index tree")
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise MutableIndexRuntimeError("index tree: file link or special file")
            entries.append(
                _TreeEntry(
                    path=path.relative_to(directory).as_posix(),
                    mode=f"{stat.S_IMODE(details.st_mode):04o}",
                    byte_size=details.st_size,
                    sha256=_sha256_file(path),
                )
            )
    if not entries:
        raise MutableIndexRuntimeError("index tree: no regular files")
    return tuple(entries)


def _tree_sha256(entries: tuple[_TreeEntry, ...]) -> str:
    """Return a deterministic full-tree identity that includes paths and modes."""
    return _sha256_json([asdict(entry) for entry in entries])


def _load_materialization_identity(release_root: Path) -> dict[str, str] | None:
    """Return a self-hashed FI-013 materialization identity, if this is a release."""
    receipt_path = release_root / MATERIALIZATION_RECEIPT_NAME
    try:
        details = receipt_path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise MutableIndexRuntimeError("materialization receipt: expected regular file")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise MutableIndexRuntimeError("materialization receipt: invalid JSON") from error
    if not isinstance(payload, dict):
        raise MutableIndexRuntimeError("materialization receipt: expected object")
    unsigned = dict(payload)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    candidate_id = unsigned.get("candidate_id")
    candidate_sha256 = unsigned.get("candidate_sha256")
    release_id = unsigned.get("release_id")
    if (
        unsigned.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or not isinstance(receipt_sha256, str)
        or not SHA256_PATTERN.fullmatch(receipt_sha256)
        or receipt_sha256 != _sha256_json(unsigned)
        or not isinstance(candidate_id, str)
        or not IDENTIFIER_PATTERN.fullmatch(candidate_id)
        or not isinstance(candidate_sha256, str)
        or not SHA256_PATTERN.fullmatch(candidate_sha256)
        or not isinstance(release_id, str)
        or not IDENTIFIER_PATTERN.fullmatch(release_id)
    ):
        raise MutableIndexRuntimeError("materialization receipt: binding mismatch")
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "materialization_receipt_sha256": receipt_sha256,
        "release_id": release_id,
    }


def _source_identity(source_index: Path, release_root: Path) -> dict[str, str]:
    """Constrain a load to one materialized preview or v1 Chroma directory."""
    artifacts_root = release_root / "artifacts"
    try:
        relative = source_index.relative_to(artifacts_root)
    except ValueError as error:
        raise MutableIndexRuntimeError("index source is outside materialized artifacts") from error
    parts = relative.parts
    if (
        len(parts) != 5
        or parts[0] not in {"integration_v0", "integration_v1"}
        or parts[1] != "indexes"
        or not re.fullmatch(r"[a-z][a-z0-9]{1,7}", parts[2])
        or not SHA256_PATTERN.fullmatch(parts[3])
        or parts[4] != "chroma"
    ):
        raise MutableIndexRuntimeError("index source has an unsupported materialized layout")
    return {
        "artifact_set": parts[0],
        "subject": parts[2],
        "input_sha256": parts[3],
        "source_relative_path": relative.as_posix(),
    }


def _mutable_root(*, root: Path, release_root: Path) -> Path:
    """Require an operator-precreated owner-only root outside the release."""
    configured = os.getenv(MUTABLE_INDEX_ROOT_ENV, "").strip()
    if not configured:
        raise MutableIndexRuntimeError(
            f"{MUTABLE_INDEX_ROOT_ENV}: required for materialized release"
        )
    candidate = _real_directory(Path(configured), context=MUTABLE_INDEX_ROOT_ENV)
    details = _lstat(candidate, context=MUTABLE_INDEX_ROOT_ENV)
    if details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) != 0o700:
        raise MutableIndexRuntimeError(
            f"{MUTABLE_INDEX_ROOT_ENV}: must be current-user-owned exact mode 0700"
        )
    if _is_relative_to(candidate, release_root) or _is_relative_to(release_root, candidate):
        raise MutableIndexRuntimeError(f"{MUTABLE_INDEX_ROOT_ENV}: must be external to release")
    if _is_relative_to(candidate, root) or _is_relative_to(root, candidate):
        raise MutableIndexRuntimeError(
            f"{MUTABLE_INDEX_ROOT_ENV}: must be disjoint from index source"
        )
    return candidate


def _mkdir_private(path: Path) -> None:
    """Create one private cache component without following a pre-existing link."""
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as error:
        details = _lstat(path, context="mutable index cache")
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise MutableIndexRuntimeError("mutable index cache: unsafe existing path") from error
        if details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) != 0o700:
            raise MutableIndexRuntimeError(
                "mutable index cache: unsafe existing directory"
            ) from error


def _fsync_directory(path: Path) -> None:
    """Flush one external cache directory before its receipt is made durable."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        raise MutableIndexRuntimeError("mutable index cache: directory fsync failed") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise MutableIndexRuntimeError("mutable index cache: directory fsync failed") from error
    finally:
        os.close(descriptor)


def _safe_copy_tree(source: Path, destination: Path, entries: tuple[_TreeEntry, ...]) -> None:
    """Copy an immutable source snapshot into an otherwise empty external directory."""
    destination.mkdir(mode=0o700)
    for entry in entries:
        origin = source / entry.path
        copied = destination / entry.path
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, copied, follow_symlinks=False)
        copied.chmod(int(entry.mode, 8))
        try:
            descriptor = os.open(copied, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as error:
            raise MutableIndexRuntimeError("mutable index copy: copied file is unsafe") from error
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise MutableIndexRuntimeError("mutable index copy: file fsync failed") from error
        finally:
            os.close(descriptor)
        _fsync_directory(copied.parent)
    if _tree_entries(source) != entries:
        raise MutableIndexRuntimeError("index source changed while copying mutable cache")
    if _tree_entries(destination) != entries:
        raise MutableIndexRuntimeError("mutable index copy does not match source snapshot")


def _write_copy_receipt(path: Path, payload: dict[str, object]) -> None:
    """Write a create-only mode-0400 receipt beside a new external index copy."""
    encoded = _canonical_json(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except OSError as error:
        raise MutableIndexRuntimeError("mutable index copy receipt write failed") from error
    finally:
        os.close(descriptor)


def _write_preview_copy_receipt(path: Path, payload: dict[str, object]) -> None:
    """Write a create-only receipt for the expedited unbound preview copy."""
    encoded = _canonical_json(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except OSError as error:
        raise MutableIndexRuntimeError("preview mutable index receipt write failed") from error
    finally:
        os.close(descriptor)


def _development_mutable_copy(source: Path, *, release_root: Path) -> MutableIndexCopy:
    """Copy an unbound development index only into an explicit external root.

    This expedited-preview branch remains deliberately outside materialization
    and promotion receipts, but it must not reopen a frozen source index.  The
    same explicit owner-only external root is therefore required as it is for
    materialized releases.
    """
    root = _mutable_root(root=source, release_root=release_root)
    source_entries = _tree_entries(source)
    source_tree_sha256 = _tree_sha256(source_entries)
    workspace = root / "fast-preview" / RUNTIME_SESSION_ID / source_tree_sha256
    parent = root
    for component in workspace.relative_to(root).parts[:-1]:
        parent /= component
        _mkdir_private(parent)
    try:
        workspace.mkdir(mode=0o700)
    except FileExistsError as error:
        raise MutableIndexRuntimeError(
            "preview mutable index workspace already exists; use a fresh process session"
        ) from error
    mutable = workspace / "chroma"
    _safe_copy_tree(source, mutable, source_entries)
    unsigned: dict[str, object] = {
        "schema_version": PREVIEW_MUTABLE_COPY_SCHEMA_VERSION,
        "source_basename": source.name,
        "source_tree_sha256": source_tree_sha256,
        "runtime_session_id": RUNTIME_SESSION_ID,
        "mutable_root": {
            "path_kind": "external_owner_only_directory",
            "mode": "0700",
            "owner": "current_user",
        },
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    receipt_path = workspace / "preview_mutable_index_copy_receipt.json"
    _write_preview_copy_receipt(receipt_path, receipt)
    _fsync_directory(workspace)
    return MutableIndexCopy(
        source_index=source,
        mutable_index=mutable,
        receipt_path=receipt_path,
        source_tree_sha256=source_tree_sha256,
        materialization_receipt_sha256="preview_unbound",
    )


def _verify_copy_receipt(
    *,
    workspace: Path,
    source_identity: dict[str, str],
    source_tree_sha256: str,
    materialization_identity: dict[str, str],
    session_id: str,
) -> Path:
    """Verify a prior copy receipt while intentionally permitting cache mutations."""
    receipt_path = workspace / MUTABLE_INDEX_COPY_RECEIPT_NAME
    details = _lstat(receipt_path, context="mutable index copy receipt")
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) != 0o400
    ):
        raise MutableIndexRuntimeError("mutable index copy receipt: unsafe file")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise MutableIndexRuntimeError("mutable index copy receipt: invalid JSON") from error
    if not isinstance(payload, dict):
        raise MutableIndexRuntimeError("mutable index copy receipt: expected object")
    unsigned = dict(payload)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    expected = {
        "schema_version": MUTABLE_INDEX_COPY_SCHEMA_VERSION,
        **materialization_identity,
        "source_index": source_identity,
        "source_tree_sha256": source_tree_sha256,
        "runtime_session_id": session_id,
        "mutable_root": {
            "path_kind": "external_owner_only_directory",
            "mode": "0700",
            "owner": "current_user",
        },
    }
    if receipt_sha256 != _sha256_json(unsigned) or unsigned != expected:
        raise MutableIndexRuntimeError("mutable index copy receipt: binding mismatch")
    mutable_index = workspace / "chroma"
    _tree_entries(mutable_index)
    return mutable_index


def mutable_index_copy(source_index: Path, *, release_root: Path) -> MutableIndexCopy:
    """Return a receipt-bound external mutable index without a source fallback.

    Both materialized releases and unbound development/preview trees require a
    configured owner-only external mutable root before Chroma can open an index.
    The unbound branch writes a clearly non-promotion preview receipt; it never
    falls back to the frozen source directory.
    """
    release = _real_directory(release_root, context="runtime release")
    source = _real_directory(source_index, context="index source")
    materialization_identity = _load_materialization_identity(release)
    if materialization_identity is None:
        return _development_mutable_copy(source, release_root=release)
    source_identity = _source_identity(source, release)
    source_entries = _tree_entries(source)
    source_tree_sha256 = _tree_sha256(source_entries)
    root = _mutable_root(root=source, release_root=release)
    receipt_hash = materialization_identity["materialization_receipt_sha256"]
    workspace = (
        root
        / receipt_hash
        / RUNTIME_SESSION_ID
        / source_identity["artifact_set"]
        / source_identity["subject"]
        / source_identity["input_sha256"]
    )
    parent = root
    for component in workspace.relative_to(root).parts[:-1]:
        parent /= component
        _mkdir_private(parent)
    try:
        workspace.mkdir(mode=0o700)
    except FileExistsError as error:
        details = _lstat(workspace, context="mutable index cache")
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISDIR(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o700
        ):
            raise MutableIndexRuntimeError(
                "mutable index cache: unsafe existing workspace"
            ) from error
        mutable = _verify_copy_receipt(
            workspace=workspace,
            source_identity=source_identity,
            source_tree_sha256=source_tree_sha256,
            materialization_identity=materialization_identity,
            session_id=RUNTIME_SESSION_ID,
        )
        return MutableIndexCopy(
            source_index=source,
            mutable_index=mutable,
            receipt_path=workspace / MUTABLE_INDEX_COPY_RECEIPT_NAME,
            source_tree_sha256=source_tree_sha256,
            materialization_receipt_sha256=receipt_hash,
        )
    try:
        mutable = workspace / "chroma"
        _safe_copy_tree(source, mutable, source_entries)
        unsigned: dict[str, object] = {
            "schema_version": MUTABLE_INDEX_COPY_SCHEMA_VERSION,
            **materialization_identity,
            "source_index": source_identity,
            "source_tree_sha256": source_tree_sha256,
            "runtime_session_id": RUNTIME_SESSION_ID,
            "mutable_root": {
                "path_kind": "external_owner_only_directory",
                "mode": "0700",
                "owner": "current_user",
            },
        }
        receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        _write_copy_receipt(workspace / MUTABLE_INDEX_COPY_RECEIPT_NAME, receipt)
        _fsync_directory(workspace)
    except Exception:
        # Preserve the create-only partial workspace for inspection; never
        # delete or silently retry a failed cache preparation.
        raise
    return MutableIndexCopy(
        source_index=source,
        mutable_index=mutable,
        receipt_path=workspace / MUTABLE_INDEX_COPY_RECEIPT_NAME,
        source_tree_sha256=source_tree_sha256,
        materialization_receipt_sha256=receipt_hash,
    )
