"""Conservative, question-only checks for known missing or conflicting inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.tools.mmt_calculators import _explicit_variance_convention, parse_number_list

_GENERAL_REQUEST: re.Pattern[str] = re.compile(
    r"\b(?:explain|describe|define|definition|meaning|general|symbolic|formula|"
    r"expression|derive|derivation|in terms of|how to|how do|how does)\b"
    r"|解释|说明|定义|概念|通式|符号|推导",
    re.IGNORECASE,
)
_CALCULATION_REQUEST: re.Pattern[str] = re.compile(
    r"\b(?:calculate|compute|evaluate|find|determine|work out)\b|计算|求", re.IGNORECASE
)
_NUMBER: re.Pattern[str] = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_MATRIX_OPERATION: re.Pattern[str] = re.compile(
    r"\b(?:det|determinant|inverse)\b|行列式|逆矩阵", re.IGNORECASE
)
_MATRIX_CONTEXT: re.Pattern[str] = re.compile(
    r"\b(?:matrix|matrices|det|determinant)\b|行列式|逆矩阵", re.IGNORECASE
)
_SPECIFIC_MATRIX: re.Pattern[str] = re.compile(
    r"\bdet\s*\([A-Z]\)|\b(?:determinant|inverse)\s+of\s+(?:matrix\s+)?[A-Z]\b"
)
_MATRIX_SIZE: re.Pattern[str] = re.compile(r"\b\d+\s*(?:x|×|by)\s*\d+\b", re.IGNORECASE)
_REQUESTED_MATRIX_ENTRIES: re.Pattern[str] = re.compile(
    r"\b(?:calculate|compute|evaluate|find|determine|work out)\s+"
    r"(?:(?:all|the)\s+)?(?:\w+\s+)?(?:entries|elements)\s+of\b",
    re.IGNORECASE,
)
_MATRIX_DATA: re.Pattern[str] = re.compile(
    r"\d|[=;\[\]{}_]|\b(?:identity|diagonal|triangular|singular|orthogonal|zero|"
    r"entries|elements|rows?|columns?)\b|单位矩阵|对角|元素|行向量|列向量",
    re.IGNORECASE,
)
_CLOSED_LOOP_GAIN: re.Pattern[str] = re.compile(
    r"\bclosed[\s-]*loop\s+(?:voltage\s+)?gain\b|闭环(?:电压)?增益", re.IGNORECASE
)
_OPAMP_CONTEXT: re.Pattern[str] = re.compile(r"\bop[\s-]?amp(?:lifier)?\b", re.IGNORECASE)
_OUTPUT_VOLTAGE: re.Pattern[str] = re.compile(r"\boutput\s+voltage\b", re.IGNORECASE)
_CIRCUIT_DESCRIPTION: re.Pattern[str] = re.compile(
    r"\b(?:connect\w*|between|node|schematic|diagram|figure|follower|"
    r"feedback factor|feedback fraction|open[\s-]?loop|beta|ratio|summing|differential)\b"
    r"|β|电路图|连接|跟随器",
    re.IGNORECASE,
)
_TOPOLOGY: re.Pattern[str] = re.compile(r"\b(?:non[\s-]?)?inverting\b|反相|同相", re.IGNORECASE)
_FEEDBACK_ROLE: re.Pattern[str] = re.compile(r"\b(?:feedback|r[_\s]?f)\b|反馈", re.IGNORECASE)
_INPUT_ROLE: re.Pattern[str] = re.compile(
    r"\b(?:input|ground|r[_\s]?(?:in|i|g))\b|输入|接地", re.IGNORECASE
)
_INPUT_RESISTOR_ROLE: re.Pattern[str] = re.compile(
    r"\b(?:input|ground)\s+resist(?:or|ance)\b|\br[_\s]?(?:in|i|g)\b|输入电阻|接地电阻",
    re.IGNORECASE,
)
_SAMPLING_RATE: re.Pattern[str] = re.compile(
    r"(?:\bf[_\s]*\{?s\}?|\b(?:sampling|sample)\s+(?:frequency|rate)|"
    r"\bsampled\s+(?:(?:(?:a|the|this|my)\s+)?(?:input\s+)?signal\s+|it\s+)?at|"
    r"\bsamples\s+at|"
    r"采样频率|采样率)\s*(?:=|is|of|:|为)?\s*"
    r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*(?:[kmg]?hz|[km]?samples?/?s)?|"
    r"\b\d+(?:\.\d+)?\s+samples?\s+(?:per|every)\s+second\b",
    re.IGNORECASE,
)
_SAMPLING_ADEQUACY: re.Pattern[str] = re.compile(
    r"\balias(?:ing)?\b|\breconstruct(?:ion|ed)?\b|"
    r"\brecover\b[^?.!]*\b(?:signal|waveform)\b|混叠|重建",
    re.IGNORECASE,
)
_SIGNAL_INFORMATION: re.Pattern[str] = re.compile(
    r"\b(?:bandwidth|band[\s-]?limited|bandlimit\w*|highest|maximum|spectrum|"
    r"spectral|baseband|tone|sin|cos|f[_\s]?(?:max|m|h))\b|"
    r"\d+(?:\.\d+)?\s*[kmg]?hz\b|带宽|带限|最高频率|频谱",
    re.IGNORECASE,
)
_VARIANCE_DENOMINATOR: re.Pattern[str] = re.compile(
    r"\bdenominator\b|\bdivid(?:e|ing)\s+by\s+n\b|"
    r"\b(?:using|with)\s+(?:the\s+)?n\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InputClarification:
    """Only the missing conditions and the question needed to supply them."""

    missing_fields: tuple[str, ...]
    question: str


def _matrix_clarification(question: str) -> InputClarification | None:
    """Do not mistake an unfamiliar supplied matrix for an absent matrix."""
    specific = _SPECIFIC_MATRIX.search(question)
    if not (_CALCULATION_REQUEST.search(question) or specific):
        return None
    if not _MATRIX_OPERATION.search(question):
        return None
    if not (_MATRIX_CONTEXT.search(question) or specific):
        return None
    # Requested entry counts describe the result, not supplied matrix values.
    supplied_data = _REQUESTED_MATRIX_ENTRIES.sub("", _MATRIX_SIZE.sub("", question))
    if _MATRIX_DATA.search(supplied_data):
        return None
    return InputClarification(("matrix",), "What are the entries of the matrix?")


def _gain_clarification(question: str) -> InputClarification | None:
    """Ask about unknown simple op-amp topology/roles, not unfamiliar wiring."""
    output_voltage: bool = bool(
        _OPAMP_CONTEXT.search(question) and _OUTPUT_VOLTAGE.search(question)
    )
    if not (_CLOSED_LOOP_GAIN.search(question) or output_voltage):
        return None
    values_present = len(_NUMBER.findall(question)) >= 2
    if not _CALCULATION_REQUEST.search(question) and not (values_present and "?" in question):
        return None
    if _CIRCUIT_DESCRIPTION.search(question):
        return None
    topology_missing = _TOPOLOGY.search(question) is None
    feedback_missing = _FEEDBACK_ROLE.search(question) is None
    # A supplied input voltage is not an input-resistor role.
    input_role: re.Pattern[str] = _INPUT_RESISTOR_ROLE if output_voltage else _INPUT_ROLE
    input_missing = input_role.search(question) is None
    missing: list[str] = []
    questions: list[str] = []
    if topology_missing:
        missing.append("amplifier_topology")
        questions.append("Is the op-amp inverting or non-inverting?")
    if feedback_missing or input_missing:
        missing.append("resistor_roles")
        if feedback_missing and input_missing:
            questions.append(
                "Which resistor is the feedback resistor, and which is the input/ground resistor?"
                if values_present
                else "What are the feedback and input/ground resistances?"
            )
        else:
            role = "feedback" if feedback_missing else "input/ground"
            questions.append(
                f"Which resistor is the {role} resistor?"
                if values_present
                else f"What is the {role} resistance?"
            )
    if not missing:
        return None
    return InputClarification(tuple(missing), " ".join(questions))


def _variance_clarification(question: str) -> InputClarification | None:
    """Clarify explicit conflicting denominators for a supplied numeric dataset."""
    if not (
        _CALCULATION_REQUEST.search(question)
        and re.search(r"\bvariance\b", question, re.IGNORECASE)
        and _VARIANCE_DENOMINATOR.search(question)
        and parse_number_list(question) is not None
        and _explicit_variance_convention(question) == "conflict"
    ):
        return None
    return InputClarification(
        ("variance_convention",),
        "Which variance convention should I use: population variance "
        "(divide by the number of values) or unbiased sample variance "
        "(divide by the number of values minus one)?",
    )


def check_input_requirements(
    question: str,
    *,
    subject: str,
    has_conversation_context: bool = False,
    image_provided: bool = False,
    retrieval_query: str | None = None,
) -> InputClarification | None:
    """Return a known missing-input question, or defer to the existing answer path.

    This is not a parser or a completeness proof. Context, images, retrieval hints,
    symbolic tasks and unfamiliar supplied data stay with the existing grounded
    path; prior assistant prose is never extracted into problem parameters here.
    ``retrieval_query`` must already have passed the pipeline's hint normalization.
    """
    if has_conversation_context or image_provided or retrieval_query is not None:
        return None
    if _GENERAL_REQUEST.search(question):
        return None
    selected = subject.strip().upper()
    if selected == "EM":
        return _matrix_clarification(question)
    if selected == "EPC":
        return _gain_clarification(question)
    if selected == "MMT":
        return _variance_clarification(question)
    if selected == "IS" and _SAMPLING_ADEQUACY.search(question):
        rate = _SAMPLING_RATE.search(question)
        if rate and not _SIGNAL_INFORMATION.search(
            question[: rate.start()] + question[rate.end() :]
        ):
            return InputClarification(
                ("input_signal_bandwidth",),
                "What is the input signal's highest frequency or spectrum, "
                "including any known band limit?",
            )
    return None
