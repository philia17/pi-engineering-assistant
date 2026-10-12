#!/usr/bin/env python3
"""Build manifest-bound Integration v0 preview indexes for selected subjects."""

from __future__ import annotations

import argparse
import contextlib
import os
from collections.abc import Sequence

from rag.pipeline import configure_embedding_runtime
from rag.subject_preview import (
    SubjectPreviewError,
    build_subject_preview_index,
    load_subject_catalog,
)


def _apply_pi_safe_runtime() -> int:
    """Force single-subject, single-thread, low-priority embedding behaviour."""
    thread_count = configure_embedding_runtime(threads=int(os.getenv("EEE_EMBED_THREADS", "1")))
    if hasattr(os, "nice"):
        with contextlib.suppress(OSError):
            os.nice(int(os.getenv("EEE_EMBED_NICE", "10")))
    return thread_count


def main(argv: Sequence[str] | None = None) -> int:
    """Build every requested canonical subject index one subject at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Canonical subject codes; default is all eight.",
    )
    args = parser.parse_args(argv)
    catalog = load_subject_catalog()
    requested = args.subjects or sorted(catalog)
    thread_count = _apply_pi_safe_runtime()
    print(
        f"Pi-safe embedding runtime: subjects_serial=1 threads={thread_count} "
        f"nice={os.getenv('EEE_EMBED_NICE', '10')}"
    )
    try:
        for subject in requested:
            build = build_subject_preview_index(subject)
            action = "REUSED" if build.reused else "BUILT"
            print(
                f"{action}: {build.subject} ({build.note_count} notes) "
                f"{build.manifest_path.relative_to(build.manifest_path.parents[3])}"
            )
    except SubjectPreviewError as error:
        print(f"FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
