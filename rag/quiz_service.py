"""Subject-scoped retrieval and single-call generation for Quiz Preview."""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from random import Random, SystemRandom
from typing import Any

from rag import pipeline
from rag.quiz_contract import (
    PrivateQuizQuestion,
    QuizGenerationError,
    QuizOption,
    QuizSource,
    QuizWrongOptionReason,
)
from rag.quiz_scopes import QuizFact, QuizFocus
from rag.subject_preview import resolve_subject, sha256_file, sha256_json
from src.llm_client import generate

QUIZ_V1_INDEX_ROOT: Path = (
    Path(__file__).resolve().parents[1] / "artifacts" / "integration_v1" / "indexes"
)
V1_ELIGIBILITY_POLICY: str = "eligible_route"
QUIZ_MODEL_ID: str = "qwen3:1.7b"
_SHA256_DIRECTORY = re.compile(r"^[0-9a-f]{64}$")
_v1_runtime_cache: dict[tuple[Path, str], tuple[Any, Any]] = {}
_QUIZ_RANDOM = SystemRandom()
_TEMPLATE_IDS: tuple[str, str, str] = (
    "property_direct",
    "property_concept",
    "property_axis",
)
_SOURCE_FACT_PROXIMITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "begin",
        "big",
        "course",
        "definition",
        "end",
        "for",
        "frac",
        "from",
        "in",
        "is",
        "left",
        "mathrm",
        "negative",
        "of",
        "on",
        "one",
        "or",
        "positive",
        "principle",
        "qquad",
        "right",
        "source",
        "text",
        "the",
        "to",
        "with",
    }
)


class QuizV1IndexError(ValueError):
    """Raised when one unambiguous eval-free formal-v1 artifact is unavailable."""


@dataclass(frozen=True)
class _V1Artifact:
    chroma_path: Path
    note_paths: tuple[str, ...]


QUIZ_SYSTEM_PROMPT: str = (
    "Select one allowed source-bound property mapping and one allowed wording template. Return "
    'only the exact JSON object {"mapping_id":"allowed-id","template_id":"allowed-id"}. '
    "Copy both IDs exactly from "
    "the supplied enums. Add no keys, prose, or Markdown."
)


def _selection_schema(
    mapping_ids: tuple[str, ...], template_ids: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mapping_id", "template_id"],
        "properties": {
            "mapping_id": {"type": "string", "enum": list(mapping_ids)},
            "template_id": {"type": "string", "enum": list(template_ids)},
        },
    }


def _parse_selection(
    raw_output: str,
    *,
    facts: tuple[QuizFact, ...],
    template_ids: tuple[str, ...],
) -> tuple[QuizFact, str]:
    try:
        payload: Any = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as error:
        raise QuizGenerationError("Quiz model selection is not valid JSON.") from error
    if not isinstance(payload, dict) or set(payload) != {"mapping_id", "template_id"}:
        raise QuizGenerationError("Quiz model selection has invalid fields.")
    mapping_id = payload["mapping_id"]
    template_id = payload["template_id"]
    selected = next((fact for fact in facts if fact.id == mapping_id), None)
    if selected is None or template_id not in template_ids:
        raise QuizGenerationError("Quiz model selected an unavailable mapping or template.")
    return selected, template_id


def _is_property_mapping(fact: QuizFact) -> bool:
    return (
        fact.fact_type == "property"
        and bool(fact.concept)
        and bool(fact.question_axis)
        and bool(fact.domain_id)
        and len(fact.domain_values) == 4
        and len(set(fact.domain_values)) == 4
        and fact.domain_values.count(fact.display_text) == 1
        and bool(fact.source_quote)
        and bool(fact.reference_title)
    )


def _is_source_fact(fact: QuizFact) -> bool:
    return (
        fact.fact_type in {"statement", "expression"}
        and bool(fact.display_text)
        and bool(fact.concept)
        and bool(fact.source_quote)
        and bool(fact.reference_title)
    )


def _is_quiz_fact(fact: QuizFact) -> bool:
    return _is_property_mapping(fact) or _is_source_fact(fact)


def _axis_label(question_axis: str) -> str:
    axis = question_axis.strip()
    lowered = axis.casefold()
    for prefix in ("the ", "its "):
        if lowered.startswith(prefix):
            return axis[len(prefix) :]
    return axis


def _source_heading_parts(heading: str) -> tuple[str, ...]:
    return tuple(" ".join(part.split()) for part in heading.split("—") if part.strip())


def _source_fact_terms(fact: QuizFact) -> frozenset[str]:
    return frozenset(
        token
        for value in (fact.concept, fact.source_quote)
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 and token not in _SOURCE_FACT_PROXIMITY_STOPWORDS
    )


def _source_fact_rank(selected_fact: QuizFact, candidate: QuizFact) -> tuple[int, Fraction, int]:
    selected_parts = tuple(
        part.casefold() for part in _source_heading_parts(selected_fact.focus_heading)
    )
    candidate_parts = tuple(
        part.casefold() for part in _source_heading_parts(candidate.focus_heading)
    )
    shared_prefix = 0
    for selected_part, candidate_part in zip(selected_parts, candidate_parts, strict=False):
        if selected_part != candidate_part:
            break
        shared_prefix += 1
    selected_terms = _source_fact_terms(selected_fact)
    candidate_terms = _source_fact_terms(candidate)
    intersection = selected_terms & candidate_terms
    union = selected_terms | candidate_terms
    semantic_proximity = (
        Fraction(len(intersection), len(union) or 1) if len(intersection) >= 2 else Fraction(0)
    )
    return (
        max(shared_prefix - 1, 0),
        semantic_proximity,
        int(candidate.note_path == selected_fact.note_path),
    )


def _source_location_labels(facts: tuple[QuizFact, ...]) -> dict[str, str]:
    """Return unique source-hierarchy labels for one four-option question."""
    base_labels: dict[str, str] = {}
    leaf_labels: dict[str, str] = {}
    for fact in facts:
        parts = _source_heading_parts(fact.focus_heading)
        label = " — ".join(parts[1:3]) if len(parts) > 1 else fact.concept
        base_labels[fact.id] = label
        leaf_labels[fact.id] = parts[2] if len(parts) > 2 else parts[-1]

    def label_terms(label: str) -> frozenset[str]:
        return frozenset(
            token
            for token in re.findall(r"[a-z0-9]+", label.casefold())
            if token not in _SOURCE_FACT_PROXIMITY_STOPWORDS
        )

    qualified_ids: set[str] = set()
    for index, left in enumerate(facts):
        left_terms = label_terms(leaf_labels[left.id])
        for right in facts[index + 1 :]:
            right_terms = label_terms(leaf_labels[right.id])
            union = left_terms | right_terms
            if base_labels[left.id].casefold() == base_labels[right.id].casefold() or Fraction(
                len(left_terms & right_terms), len(union) or 1
            ) >= Fraction(1, 2):
                qualified_ids.update((left.id, right.id))

    labels: dict[str, str] = {}
    for fact in facts:
        label = base_labels[fact.id]
        if fact.id in qualified_ids:
            parts = _source_heading_parts(fact.focus_heading)
            qualifier = parts[3] if len(parts) > 3 else fact.source_quote
            qualifier = " ".join(qualifier.replace("*", "").replace("`", "").split())
            available = 140 - len(label) - 3
            if len(qualifier) > available:
                qualifier = f"{qualifier[: available - 1].rstrip()}…"
            label = f"{label} — {qualifier}"
        if len(label) > 140:
            raise QuizGenerationError("Quiz source location label exceeds 140 characters.")
        labels[fact.id] = label
    if len({label.casefold() for label in labels.values()}) != len(facts):
        raise QuizGenerationError("Quiz source location labels are not unique.")
    return labels


def _source_fact_distractors(
    selected_fact: QuizFact,
    focus: QuizFocus,
    fact_pool: tuple[QuizFocus, ...],
    rng: SystemRandom,
) -> tuple[QuizFact, ...]:
    """Choose concepts by subheading depth, symmetric semantics, then note."""

    by_concept: dict[str, list[QuizFact]] = {}
    selected_concept = selected_fact.concept.strip().casefold()
    for pool_focus in fact_pool:
        if pool_focus.id == focus.id:
            continue
        for fact in pool_focus.facts:
            concept_key = fact.concept.strip().casefold()
            if _is_source_fact(fact) and concept_key and concept_key != selected_concept:
                by_concept.setdefault(concept_key, []).append(fact)
    if len(by_concept) < 3:
        raise QuizGenerationError("This Quiz fact has fewer than three concept distractors.")

    representatives: list[QuizFact] = []
    for concept_key in sorted(by_concept):
        facts = by_concept[concept_key]
        best_rank = max(_source_fact_rank(selected_fact, fact) for fact in facts)
        tied = sorted(
            (fact for fact in facts if _source_fact_rank(selected_fact, fact) == best_rank),
            key=lambda fact: fact.id,
        )
        representatives.append(rng.choice(tied))

    ranked_groups: dict[tuple[int, Fraction, int], list[QuizFact]] = {}
    for fact in representatives:
        ranked_groups.setdefault(_source_fact_rank(selected_fact, fact), []).append(fact)
    chosen: list[QuizFact] = []
    for candidate_rank in sorted(ranked_groups, reverse=True):
        group = sorted(ranked_groups[candidate_rank], key=lambda fact: fact.id)
        chosen.extend(rng.sample(group, len(group)))
        if len(chosen) >= 3:
            return tuple(chosen[:3])
    raise QuizGenerationError("This Quiz fact has fewer than three concept distractors.")


def plan_quiz_focuses(
    selected_focuses: tuple[QuizFocus, ...],
    subject_focuses: tuple[QuizFocus, ...],
    requested_total: int,
    *,
    excluded_fact_ids: tuple[str, ...] = (),
    rng: Random | SystemRandom | None = None,
) -> tuple[QuizFocus, ...]:
    """Plan distinct source facts while distributing questions across concepts."""
    subject_focus_ids = {focus.id for focus in subject_focuses}
    excluded = set(excluded_fact_ids)
    eligible = tuple(focus for focus in selected_focuses if focus.id in subject_focus_ids)
    if rng is not None:
        eligible = tuple(rng.sample(eligible, len(eligible)))
    plan: list[QuizFocus] = []
    depth = 0
    while len(plan) < requested_total:
        added = False
        for focus in eligible:
            available = tuple(
                fact for fact in focus.facts if _is_quiz_fact(fact) and fact.id not in excluded
            )
            if len(available) > depth:
                plan.append(focus)
                added = True
                if len(plan) == requested_total:
                    break
        if not added:
            break
        depth += 1
    if len(plan) < requested_total:
        raise ValueError(
            "This selection does not contain enough source-bound questions for "
            f"{requested_total} questions. Choose another Topic or fewer questions."
        )
    return tuple(plan)


def _normalized_sources(documents: list[Any], subject: str) -> tuple[QuizSource, ...]:
    prefix = f"rag/data/{subject.lower()}/"
    sources: list[QuizSource] = []
    seen: set[tuple[str, str | None]] = set()
    for document in documents:
        raw_path = str(document.metadata.get("source") or "").replace("\\", "/")
        if not raw_path.startswith(prefix):
            raise QuizGenerationError("Retrieved Quiz context crossed the selected subject.")
        raw_heading = str(document.metadata.get("heading_path") or "").strip()
        source = QuizSource(note_path=raw_path, heading=raw_heading or None)
        key = (source.note_path, source.heading)
        if key not in seen:
            seen.add(key)
            sources.append(source)
    if not sources:
        raise QuizGenerationError("Quiz retrieval returned no usable sources.")
    return tuple(sources)


def _select_v1_artifact(
    subject: str,
    *,
    module: str,
    index_root: Path,
) -> _V1Artifact:
    """Select one self-consistent v1 artifact and its admitted note paths."""
    subject_root = index_root / subject.lower()
    expected_chunk_size = int(os.getenv("EEE_RAG_CHUNK_SIZE", "512"))
    expected_chunk_overlap = int(os.getenv("EEE_RAG_CHUNK_OVERLAP", "64"))
    valid_artifacts: list[_V1Artifact] = []
    if subject_root.is_dir():
        for build_dir in subject_root.iterdir():
            if not build_dir.is_dir() or _SHA256_DIRECTORY.fullmatch(build_dir.name) is None:
                continue
            manifest_path = build_dir / "index_manifest.json"
            chroma_path = build_dir / "chroma"
            if not manifest_path.is_file() or not chroma_path.is_dir():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            expected_index_path = (
                f"artifacts/integration_v1/indexes/{subject.lower()}/{build_dir.name}/chroma"
            )
            build_input = manifest.get("input") if isinstance(manifest, dict) else None
            notes = build_input.get("notes") if isinstance(build_input, dict) else None
            if (
                not isinstance(manifest, dict)
                or not isinstance(build_input, dict)
                or not isinstance(notes, list)
                or not notes
                or manifest.get("schema_version") != "1.0"
                or manifest.get("mode") != "v1"
                or manifest.get("subject") != subject
                or manifest.get("module") != module
                or manifest.get("eligibility_policy") != V1_ELIGIBILITY_POLICY
                or manifest.get("input_sha256") != build_dir.name
                or sha256_json(build_input) != build_dir.name
                or type(manifest.get("note_count")) is not int
                or manifest.get("note_count") != len(notes)
                or manifest.get("index_path") != expected_index_path
                or build_input.get("mode") != "v1"
                or build_input.get("schema_version") != "1.0"
                or build_input.get("subject") != subject
                or build_input.get("module") != module
                or build_input.get("eligibility_policy") != V1_ELIGIBILITY_POLICY
                or build_input.get("embedding_model") != pipeline.EMBED_MODEL
                or type(build_input.get("chunk_size")) is not int
                or build_input.get("chunk_size") != expected_chunk_size
                or type(build_input.get("chunk_overlap")) is not int
                or build_input.get("chunk_overlap") != expected_chunk_overlap
                or not any(chroma_path.iterdir())
            ):
                continue
            repo_root = index_root.parents[2].resolve()
            allowed_note_root = (repo_root / "rag" / "data" / subject.lower()).resolve()
            note_prefix = f"rag/data/{subject.lower()}/"
            note_paths: set[str] = set()
            notes_valid = True
            for note in notes:
                if not isinstance(note, dict):
                    notes_valid = False
                    break
                note_path = note.get("path")
                note_sha256 = note.get("sha256")
                if (
                    not isinstance(note_path, str)
                    or not note_path.startswith(note_prefix)
                    or note_path in note_paths
                    or not isinstance(note_sha256, str)
                    or _SHA256_DIRECTORY.fullmatch(note_sha256) is None
                ):
                    notes_valid = False
                    break
                note_paths.add(note_path)
                resolved_note = (repo_root / note_path).resolve()
                if (
                    not resolved_note.is_relative_to(allowed_note_root)
                    or not resolved_note.is_file()
                    or sha256_file(resolved_note) != note_sha256
                ):
                    notes_valid = False
                    break
            if not notes_valid:
                continue
            valid_artifacts.append(
                _V1Artifact(
                    chroma_path=chroma_path,
                    note_paths=tuple(note["path"] for note in notes),
                )
            )
    if len(valid_artifacts) != 1:
        raise QuizV1IndexError(
            f"{subject} Quiz v1 artifact selection requires exactly one valid candidate."
        )
    return valid_artifacts[0]


def _v1_runtime(
    subject: str,
    *,
    module: str,
    index_root: Path,
) -> tuple[Any, Any]:
    """Load exactly one already-built formal-v1 artifact without reading eval inputs."""
    cache_key = (index_root.resolve(), subject)
    cached = _v1_runtime_cache.get(cache_key)
    if cached is not None:
        return cached
    artifact = _select_v1_artifact(subject, module=module, index_root=index_root)
    vectorstore = pipeline.load_vectorstore(artifact.chroma_path)
    chain = pipeline.build_rag_chain(vectorstore)
    runtime = (chain, vectorstore)
    _v1_runtime_cache[cache_key] = runtime
    return runtime


def admitted_note_paths(
    *,
    subject: str,
    mode: str,
    preview_service: Any,
    v1_index_root: Path = QUIZ_V1_INDEX_ROOT,
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    """Return note paths admitted by the current selected-mode manifest."""
    record = resolve_subject(subject)
    if mode == "v1":
        return _select_v1_artifact(
            record.subject,
            module=record.module,
            index_root=v1_index_root,
        ).note_paths
    if mode != "preview":
        raise ValueError("Quiz mode must be preview or v1.")
    status = preview_service.index_status(record.subject)
    if not status.ready:
        raise QuizV1IndexError(
            f"{record.subject} Quiz preview index is unavailable: {status.reason}"
        )
    try:
        manifest = json.loads(status.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuizV1IndexError("Quiz preview index manifest is invalid.") from error
    if not isinstance(manifest, dict):
        raise QuizV1IndexError("Quiz preview index manifest is invalid.")
    build_input = manifest.get("input")
    if not isinstance(build_input, dict):
        raise QuizV1IndexError("Quiz preview index manifest is invalid.")
    notes = build_input.get("notes")
    prefix = f"rag/data/{record.subject.lower()}/"
    input_sha256 = sha256_json(build_input) if isinstance(build_input, dict) else None
    if (
        not isinstance(notes, list)
        or not notes
        or input_sha256 != status.input_sha256
        or manifest.get("input_sha256") != input_sha256
        or manifest.get("mode") != "preview"
        or manifest.get("subject") != record.subject
        or build_input.get("mode") != "preview"
        or build_input.get("subject") != record.subject
    ):
        raise QuizV1IndexError("Quiz preview index manifest is invalid.")
    resolved_repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    allowed_note_root = (resolved_repo_root / "rag" / "data" / record.subject.lower()).resolve()
    paths: list[str] = []
    for note in notes:
        if not isinstance(note, dict):
            raise QuizV1IndexError("Quiz preview admitted note paths are invalid.")
        note_path = note.get("path")
        note_sha256 = note.get("sha256")
        if (
            not isinstance(note_path, str)
            or not note_path.startswith(prefix)
            or note_path in paths
            or not isinstance(note_sha256, str)
            or _SHA256_DIRECTORY.fullmatch(note_sha256) is None
        ):
            raise QuizV1IndexError("Quiz preview admitted note paths are invalid.")
        resolved_note = (resolved_repo_root / note_path).resolve()
        if (
            not resolved_note.is_relative_to(allowed_note_root)
            or not resolved_note.is_file()
            or sha256_file(resolved_note) != note_sha256
        ):
            raise QuizV1IndexError("Quiz preview admitted note paths are invalid.")
        paths.append(note_path)
    return tuple(paths)


def generate_question(
    *,
    subject: str,
    mode: str,
    model: str | None,
    retrieval_query: str,
    display_scope: str,
    focus: QuizFocus,
    fact_pool: tuple[QuizFocus, ...],
    number: int,
    preview_service: Any,
    v1_service: Any,
    generate_fn: Callable[..., dict[str, Any]] = generate,
    v1_index_root: Path = QUIZ_V1_INDEX_ROOT,
    prior_prompts: tuple[str, ...] = (),
    prior_fact_ids: tuple[str, ...] = (),
    rng: SystemRandom = _QUIZ_RANDOM,
) -> PrivateQuizQuestion:
    """Use one model call to select IDs, then assemble one server-owned source question."""
    record = resolve_subject(subject)
    del model
    selected_model = QUIZ_MODEL_ID
    if number not in range(1, 11):
        raise ValueError("Quiz question number must be between 1 and 10.")
    if mode not in {"preview", "v1"}:
        raise ValueError("Quiz mode must be preview or v1.")
    del retrieval_query, preview_service, v1_service, v1_index_root, prior_prompts
    if (
        not focus.note_path.startswith(f"rag/data/{record.subject.lower()}/")
        or ".." in Path(focus.note_path).parts
    ):
        raise QuizGenerationError("Quiz focus crossed the selected subject.")
    subject_prefix = f"rag/data/{record.subject.lower()}/"
    if any(
        not pool_focus.note_path.startswith(subject_prefix)
        or ".." in Path(pool_focus.note_path).parts
        or any(
            fact.focus_id != pool_focus.id
            or fact.note_path != pool_focus.note_path
            or not _is_quiz_fact(fact)
            for fact in pool_focus.facts
        )
        for pool_focus in fact_pool
    ):
        raise QuizGenerationError("Quiz fact pool crossed the selected subject.")
    eligible_facts = tuple(
        fact for fact in focus.facts if _is_quiz_fact(fact) and fact.id not in prior_fact_ids
    )
    if not eligible_facts:
        raise QuizGenerationError("This Quiz concept has no unused source-bound question.")
    candidates = tuple(rng.sample(eligible_facts, min(4, len(eligible_facts))))
    response_format = _selection_schema(tuple(fact.id for fact in candidates), _TEMPLATE_IDS)
    candidate_lines = "\n".join(
        (
            f"- {fact.id}: {fact.concept}; source: {fact.source_quote}"
            if _is_source_fact(fact)
            else f"- {fact.id}: {fact.concept}; {fact.question_axis}"
        )
        for fact in candidates
    )
    prompt = (
        f"Selected subject: {record.subject}\n"
        f"Selected scope: {display_scope}\n"
        f"Assigned course concept: {focus.heading}\n"
        f"Allowed template IDs: {', '.join(_TEMPLATE_IDS)}\n"
        f"Allowed property mappings:\n{candidate_lines}\n\n"
        "Select the property mapping that best represents the assigned course concept and one "
        "allowed template ID. "
        "Return IDs only."
    )
    result = generate_fn(
        prompt,
        model=selected_model,
        system_prompt=QUIZ_SYSTEM_PROMPT,
        think=False,
        response_format=response_format,
        temperature=0,
    )
    raw_output = result.get("answer") if isinstance(result, dict) else None
    if not isinstance(raw_output, str):
        raise QuizGenerationError("Quiz model output is missing.")
    selected_fact, template_id = _parse_selection(
        raw_output, facts=candidates, template_ids=_TEMPLATE_IDS
    )
    if _is_property_mapping(selected_fact):
        option_values = list(selected_fact.domain_values)
        correct_value = selected_fact.display_text
        distractor_facts: tuple[QuizFact, ...] = ()
        source_labels: dict[str, str] = {}
    else:
        distractor_facts = _source_fact_distractors(selected_fact, focus, fact_pool, rng)
        source_labels = _source_location_labels((selected_fact, *distractor_facts))
        correct_value = source_labels[selected_fact.id]
        option_values = [
            correct_value,
            *(source_labels[fact.id] for fact in distractor_facts),
        ]
    rng.shuffle(option_values)
    options = tuple(
        QuizOption(option_id=option_id, text=value)
        for option_id, value in zip(("A", "B", "C", "D"), option_values, strict=True)
    )
    correct_index = option_values.index(correct_value)
    correct_option_id = ("A", "B", "C", "D")[correct_index]
    axis_label = _axis_label(selected_fact.question_axis)
    if _is_source_fact(selected_fact):
        question_prompt = (
            "Which course-note section contains this exact excerpt: "
            f'"{selected_fact.source_quote}"?'
        )
    elif template_id == "property_direct":
        question_prompt = f"For {selected_fact.concept}, which option gives the {axis_label}?"
    elif template_id == "property_concept":
        question_prompt = (
            f"Which option correctly gives the {axis_label} for {selected_fact.concept}?"
        )
    else:
        question_prompt = f"Choose the correct {axis_label} for {selected_fact.concept}."
    why_correct = (
        f'The excerpt is recorded under "{source_labels[selected_fact.id]}" at the exact '
        f'heading "{selected_fact.focus_heading}": {selected_fact.source_quote}'
        if _is_source_fact(selected_fact)
        else (
            f"For {selected_fact.concept}, the correct answer for "
            f"{selected_fact.question_axis} is {selected_fact.display_text}."
        )
    )
    distractor_by_text = {source_labels[fact.id]: fact for fact in distractor_facts}
    wrong_option_reasons = tuple(
        QuizWrongOptionReason(
            option_id=option.option_id,
            text=option.text,
            reason=(
                f'The topic "{option.text}" maps to the exact heading '
                f'"{distractor_by_text[option.text].focus_heading}", which records: '
                f"{distractor_by_text[option.text].source_quote} "
                f'The exact excerpt is instead recorded under "{source_labels[selected_fact.id]}" '
                f'under "{selected_fact.focus_heading}".'
                if option.text in distractor_by_text
                else (
                    f"Option {option.option_id} does not match {selected_fact.question_axis}; "
                    f"for {selected_fact.concept}, the correct answer is "
                    f"{selected_fact.display_text}."
                )
            ),
        )
        for option in options
        if option.option_id != correct_option_id
    )
    explanation = f"Correct: {correct_option_id}. {why_correct}"
    return PrivateQuizQuestion(
        question_id=secrets.token_urlsafe(18),
        number=number,
        focus_id=focus.id,
        fact_id=selected_fact.id,
        prompt=question_prompt,
        options=options,
        correct_option_id=correct_option_id,
        explanation=explanation,
        sources=(
            QuizSource(
                note_path=selected_fact.note_path,
                heading=selected_fact.focus_heading,
            ),
        ),
        why_correct=why_correct,
        wrong_option_reasons=wrong_option_reasons,
        reference_title=selected_fact.reference_title,
        reference_heading=selected_fact.focus_heading,
    )
