#!/usr/bin/env python3
"""Build formal eligibility-filtered v1 subject indexes (FP-301 / FP-305)."""

from __future__ import annotations

import argparse
import contextlib
import os
from collections.abc import Sequence

from rag.pipeline import configure_embedding_runtime
from rag.subject_index import (
    SubjectIndexError,
    build_subject_v1_index,
    changed_subjects,
    subject_index_status,
)
from rag.subject_preview import load_subject_catalog


def _apply_pi_safe_runtime() -> int:
    """Force single-subject, single-thread, low-priority embedding behaviour."""
    thread_count = configure_embedding_runtime(threads=int(os.getenv("EEE_EMBED_THREADS", "1")))
    if hasattr(os, "nice"):
        with contextlib.suppress(OSError):
            os.nice(int(os.getenv("EEE_EMBED_NICE", "10")))
    return thread_count


def main(argv: Sequence[str] | None = None) -> int:
    """Build requested v1 subject indexes, optionally only when stale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Canonical subject codes; default is all eight.",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Rebuild only subjects whose eligible inventory hash is not ready (FP-305).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print readiness for each subject and exit without building.",
    )
    args = parser.parse_args(argv)
    catalog = load_subject_catalog()
    requested = [item.upper() for item in (args.subjects or sorted(catalog))]
    if args.status:
        for subject in requested:
            status = subject_index_status(subject)
            state = "READY" if status.ready else f"NOT_READY:{status.reason}"
            print(f"{status.subject}: {state} input={status.input_sha256[:12]}")
        return 0
    targets = changed_subjects(requested) if args.changed_only else requested
    if args.changed_only and not targets:
        print("PASS: all requested subject v1 indexes are unchanged/ready.")
        return 0
    thread_count = _apply_pi_safe_runtime()
    policy = os.getenv("EEE_V1_NOTE_POLICY", "eligible_route")
    print(
        f"Pi-safe embedding runtime: subjects_serial=1 threads={thread_count} "
        f"nice={os.getenv('EEE_EMBED_NICE', '10')} policy={policy}"
    )
    try:
        for subject in targets:
            status = subject_index_status(subject)
            if status.reason == "no_eligible_notes":
                print(f"SKIP: {subject} (no_eligible_notes) input={status.input_sha256[:12]}")
                continue
            build = build_subject_v1_index(subject)
            action = "REUSED" if build.reused else "BUILT"
            print(f"{action}: {build.subject} ({build.note_count} notes) {build.manifest_path}")
    except SubjectIndexError as error:
        print(f"FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
