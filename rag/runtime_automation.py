"""Feature-gated adapter for frozen question-only deterministic automation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Final

AUTOMATION_ENV: Final = "EEE_PREVIEW_AUTOMATION_V1"
PROVENANCE_WARNING: Final = "deterministic_engine_question_only_not_course_citation"

_RouteQuestion = Callable[[str, str], Any]
_ValidateDecision = Callable[[Any, str, str], Any]
_ValidateReceipt = Callable[[dict[str, Any], str, str], Any]
_DecisionSha256 = Callable[[Any], str]
_ImportModule = Callable[[str], ModuleType]
_OPTIONAL_RUNTIME_PACKAGE: Final = "rag.evals.hard_objective_v1"


def _load_runtime_modules(
    importer: _ImportModule = import_module,
) -> tuple[ModuleType, ModuleType, ModuleType] | None:
    """Load the optional runtime while surfacing broken installed packages."""
    try:
        automation_router = importer("rag.evals.hard_objective_v1.automation_replay_v1.router")
        calculator_router = importer("rag.evals.hard_objective_v1.calculator_replay_v2.router")
        deterministic_solver = importer("rag.evals.hard_objective_v1.deterministic_solver_v1")
    except ModuleNotFoundError as error:
        if error.name is None or not (
            error.name == _OPTIONAL_RUNTIME_PACKAGE
            or _OPTIONAL_RUNTIME_PACKAGE.startswith(error.name + ".")
        ):
            raise
        return None
    return automation_router, calculator_router, deterministic_solver


_runtime_modules = _load_runtime_modules()
if _runtime_modules is None:
    AUTOMATION_ROUTER_VERSION = "unavailable"
    ROUTER_VERSION = "unavailable"
    _decision_sha256: _DecisionSha256 | None = None
    _route_question: _RouteQuestion | None = None
    _validate_decision: _ValidateDecision | None = None
    _validate_receipt: _ValidateReceipt | None = None
else:
    _automation_router, _calculator_router, _deterministic_solver = _runtime_modules
    AUTOMATION_ROUTER_VERSION = str(_automation_router.AUTOMATION_ROUTER_VERSION)
    ROUTER_VERSION = str(_calculator_router.ROUTER_VERSION)
    _decision_sha256 = _automation_router.decision_sha256
    _route_question = _automation_router.route_question
    _validate_decision = _automation_router.validate_decision
    _validate_receipt = _deterministic_solver.validate_receipt


@dataclass(frozen=True)
class RuntimeAutomationResult:
    """One validated calculator or solver result safe for preview adaptation."""

    answer: str
    route_name: str
    lane: str
    capability_name: str
    capability_version: str
    receipt_class: str
    receipt_sha256: str
    decision_sha256: str
    values: dict[str, Any]

    def answer_engine(self) -> dict[str, str]:
        """Return UI-safe engine identity separate from the requested LLM."""
        return {
            "kind": "deterministic",
            "label": f"Deterministic {self.lane.title()}",
            "lane": self.lane,
            "capability": self.capability_name,
            "version": self.capability_version,
            "router_version": AUTOMATION_ROUTER_VERSION,
        }

    def trace_record(self) -> dict[str, Any]:
        """Expose engine provenance without benchmark data or receipt contents."""
        return {
            "stage": "deterministic_automation",
            "status": "validated",
            "lane": self.lane,
            "capability": self.capability_name,
            "engine_version": self.capability_version,
            "router_version": AUTOMATION_ROUTER_VERSION,
            "receipt_class": self.receipt_class,
            "receipt_sha256": self.receipt_sha256,
            "decision_sha256": self.decision_sha256,
            "warning": PROVENANCE_WARNING,
        }


def automation_v1_enabled() -> bool:
    """Return whether the isolated preview automation adapter is explicitly enabled."""
    return os.getenv(AUTOMATION_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def automation_runtime_available() -> bool:
    """Return whether the optional deterministic runtime package is installed."""
    return all(
        function is not None
        for function in (
            _decision_sha256,
            _route_question,
            _validate_decision,
            _validate_receipt,
        )
    )


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def try_runtime_automation(subject: str, question: str) -> RuntimeAutomationResult | None:
    """Return a recomputed deterministic result, or ``None`` for every fallback case."""
    if not automation_v1_enabled():
        return None
    route_question = _route_question
    validate_decision = _validate_decision
    validate_receipt = _validate_receipt
    decision_sha256 = _decision_sha256
    if (
        route_question is None
        or validate_decision is None
        or validate_receipt is None
        or decision_sha256 is None
    ):
        return None
    try:
        decision = route_question(question, subject)
        if decision.status != "ROUTED" or decision.lane not in {"CALCULATOR", "SOLVER"}:
            return None
        if validate_decision(decision, question, subject):
            return None
        if not decision.capability_name or not decision.rendered_answer:
            return None
        if not isinstance(decision.engine_receipt, dict) or not isinstance(decision.result, dict):
            return None
        if decision.lane == "SOLVER":
            if validate_receipt(decision.engine_receipt, question, subject):
                return None
            capability_version = str(decision.engine_receipt.get("capability_version", "")).strip()
            receipt_class = "SolverReceipt"
        else:
            capability_version = ROUTER_VERSION
            receipt_class = "CalcDecision"
        if not capability_version:
            return None
        lane = decision.lane
        capability = decision.capability_name
        return RuntimeAutomationResult(
            answer=decision.rendered_answer,
            route_name=f"deterministic_automation:{lane.lower()}:{capability}",
            lane=lane,
            capability_name=capability,
            capability_version=capability_version,
            receipt_class=receipt_class,
            receipt_sha256=_canonical_sha256(decision.engine_receipt),
            decision_sha256=decision_sha256(decision),
            values=dict(decision.result),
        )
    except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError):
        return None
