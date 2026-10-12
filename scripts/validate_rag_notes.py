#!/usr/bin/env python3
"""Validate subject Markdown notes against the shared RAG authoring standard."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED_METADATA: tuple[str, ...] = (
    "subject",
    "module",
    "topic",
    "source_type",
    "owner",
    "last_updated",
    "review_status",
    "canonical_for",
    "aliases",
    "tags",
)
LIST_METADATA: frozenset[str] = frozenset({"canonical_for", "aliases", "tags"})
REVIEW_STATUSES: frozenset[str] = frozenset({"draft", "technical_review", "reviewed"})
BASE_CARD_NAMES: frozenset[str] = frozenset(
    {
        "Definition / Principle",
        "Operating States / Mechanism",
        "Common Misconceptions",
        "Assessment Focus",
        "Cross-Reference",
    }
)
PREFIXED_CARD_NAMES: tuple[str, ...] = (
    "Formula Card: ",
    "Worked Example: ",
    "Comparison Card: ",
    "Diagram Card: ",
)
CARD_FIELDS: dict[str, tuple[str, ...]] = {
    "Formula Card: ": (
        "Use when",
        "Assumptions",
        "Formula",
        "Variables and units",
        "Sanity checks",
        "Do not confuse with",
    ),
    "Worked Example: ": (
        "Given",
        "Find",
        "Select the formula",
        "Substitute",
        "Final answer",
        "Sanity check",
    ),
    "Diagram Card: ": (
        "Purpose",
        "Diagram",
        "Components",
        "Key observation",
        "Do not confuse with",
    ),
}
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
BOLD_LABEL_PATTERN = re.compile(r"^\*\*(.+?)\*\*\s*$", re.MULTILINE)
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
# Navigation / planning docs under rag/data — not subject notes.
SKIP_NOTE_NAMES: frozenset[str] = frozenset(
    {
        "readme.md",
        "conversion_playbook.md",
        "note_conversion_plan.md",
    }
)


def _parse_frontmatter(text: str) -> tuple[dict[str, str | list[str]], int]:
    """Return simple YAML frontmatter and the body start offset."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, 0

    closing_index = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        -1,
    )
    if closing_index < 0:
        return {}, 0

    metadata: dict[str, str | list[str]] = {}
    current_list: str | None = None
    for line in lines[1:closing_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and current_list:
            value = stripped[1:].strip().strip("\"'")
            existing = metadata.setdefault(current_list, [])
            if isinstance(existing, list) and value:
                existing.append(value)
            continue
        if ":" not in line:
            current_list = None
            continue
        key, raw_value = line.split(":", maxsplit=1)
        key = key.strip()
        value = raw_value.strip().strip("\"'")
        if not key:
            continue
        if value:
            metadata[key] = value
            current_list = None
        else:
            metadata[key] = []
            current_list = key

    return metadata, sum(len(line) for line in lines[: closing_index + 1])


def _card_blocks(body: str) -> list[tuple[str, str]]:
    """Return each H4 card title and its content up to the next H1-H4 heading."""
    headings = list(HEADING_PATTERN.finditer(body))
    blocks: list[tuple[str, str]] = []
    for index, heading in enumerate(headings):
        if len(heading.group(1)) != 4:
            continue
        end = len(body)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= 4:
                end = following.start()
                break
        blocks.append((heading.group(2), body[heading.end() : end]))
    return blocks


def _check_card_fields(title: str, block: str) -> list[str]:
    """Check ordered bold labels required by a structured card."""
    prefix = next((item for item in CARD_FIELDS if title.startswith(item)), None)
    if prefix is None:
        return []

    labels = [match.group(1) for match in BOLD_LABEL_PATTERN.finditer(block)]
    required = list(CARD_FIELDS[prefix])
    if prefix == "Diagram Card: ":
        state_labels = {"State interpretation", "Construction steps"}
        if not state_labels.intersection(labels):
            return [
                f"card '{title}' is missing bold label "
                "'State interpretation' or 'Construction steps'"
            ]

    positions: list[int] = []
    for field in required:
        if field not in labels:
            return [f"card '{title}' is missing bold label '{field}'"]
        positions.append(labels.index(field))
    if positions != sorted(positions):
        return [f"card '{title}' has required bold labels in the wrong order"]
    return []


def validate_note(path: Path) -> list[str]:
    """Return human-readable validation errors for one Markdown note."""
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if "REPLACE_" in text:
        errors.append("unreplaced REPLACE_ placeholder")
    metadata, body_offset = _parse_frontmatter(text)
    if not metadata:
        errors.append("missing or unclosed YAML frontmatter")
        body = text
    else:
        body = text[body_offset:].lstrip("\r\n")

    for key in REQUIRED_METADATA:
        value = metadata.get(key)
        if not value:
            errors.append(f"frontmatter field '{key}' is missing or empty")
        elif key in LIST_METADATA and not isinstance(value, list):
            errors.append(f"frontmatter field '{key}' must be a YAML list")

    review_status = metadata.get("review_status")
    if isinstance(review_status, str) and review_status not in REVIEW_STATUSES:
        errors.append(
            "frontmatter field 'review_status' must be draft, technical_review, or reviewed"
        )
    last_updated = metadata.get("last_updated")
    if isinstance(last_updated, str) and not DATE_PATTERN.fullmatch(last_updated):
        errors.append("frontmatter field 'last_updated' must use YYYY-MM-DD")

    headings = list(HEADING_PATTERN.finditer(body))
    h1_count = sum(len(match.group(1)) == 1 for match in headings)
    if h1_count != 1:
        errors.append(f"expected exactly one H1 heading, found {h1_count}")
    previous_level = 0
    for heading in headings:
        level = len(heading.group(1))
        if previous_level and level > previous_level + 1:
            errors.append(f"heading '{heading.group(2)}' skips from H{previous_level} to H{level}")
        previous_level = level
        if level == 4:
            title = heading.group(2)
            approved = any(
                title == base or title.startswith(f"{base}: ") for base in BASE_CARD_NAMES
            ) or any(
                title.startswith(prefix) and len(title) > len(prefix)
                for prefix in PREFIXED_CARD_NAMES
            )
            if not approved:
                errors.append(f"H4 card name '{title}' is not approved")

    for title, block in _card_blocks(body):
        errors.extend(_check_card_fields(title, block))

    if text.count("```") % 2:
        errors.append("unbalanced fenced-code delimiters")
    if text.count("$$") % 2:
        errors.append("unbalanced display-math delimiters")
    for index, line in enumerate(text.splitlines(), 1):
        if line.rstrip() != line:
            errors.append(f"line {index} has trailing whitespace")

    for match in re.finditer(r"```mermaid\n(.*?)```", body, re.DOTALL):
        diagram = match.group(1)
        if "%%{init: {'theme': 'neutral'}}%%" not in diagram:
            errors.append("Mermaid block is missing the neutral-theme directive")
        if len(diagram.splitlines()) >= 60:
            errors.append("Mermaid block must contain fewer than 60 lines")
        if CJK_PATTERN.search(diagram):
            errors.append("Mermaid block must use English labels")

    return errors


def find_notes(paths: list[Path]) -> list[Path]:
    """Expand file and directory arguments into unique Markdown note paths."""
    notes: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".md":
            notes.add(path)
        elif path.is_dir():
            notes.update(
                candidate
                for candidate in path.rglob("*.md")
                if candidate.name.lower() not in SKIP_NOTE_NAMES
            )
    return sorted(notes)


def main() -> int:
    """Validate requested notes and return a shell-friendly status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("rag/data")],
        help="Markdown note files or subject directories (default: rag/data)",
    )
    args = parser.parse_args()
    notes = find_notes(args.paths)
    if not notes:
        print("No Markdown notes found.")
        return 1

    error_count = 0
    for note in notes:
        errors = validate_note(note)
        for error in errors:
            print(f"{note}: {error}")
        error_count += len(errors)
    if error_count:
        print(f"FAILED: {error_count} error(s) across {len(notes)} note(s).")
        return 1
    print(f"PASS: {len(notes)} note(s) follow the shared RAG format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
