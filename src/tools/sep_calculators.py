"""Bounded C int/double arithmetic; no C execution or general language parser."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

Number = int | float
Value = tuple[str, Number]
_INT_MIN: int = -(2**31)
_INT_MAX: int = 2**31 - 1
_TOKEN: re.Pattern[str] = re.compile(
    r"\s*(\+\+|--|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"|[0-9]+[eE][+-]?[0-9]+|[0-9]+|[A-Za-z_][A-Za-z_0-9]*|[()+*/=;,\-])"
)
_NAME: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_KEYWORDS: re.Pattern[str] = re.compile(
    r"(?:auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|"
    r"if|inline|int|long|register|restrict|return|short|signed|sizeof|static|struct|switch|typedef|"
    r"union|unsigned|void|volatile|while|_Bool|_Complex|_Imaginary)\Z"
)
_PROSE_END: re.Pattern[str] = re.compile(
    r"\.\s*(?:What|Explain|State)\b"
    r"|\s+give different values(?: in this case)?\?\s*State both values and explain\.\Z",
    re.IGNORECASE,
)
_EXPLANATION_CLAUSE: str = (
    r"(?:the quotient type|the order of the cast and division|"
    r"how the (?:negative )?integer quotient is (?:rounded|truncated))"
)


@dataclass(frozen=True)
class SEPCalculatorResult:
    """Calculator-owned values and typed worked steps for every declaration."""

    answer: str
    quantity: str
    unit: str
    steps: list[str]
    values: dict[str, Number]


def _checked(kind: str, number: Number) -> Value:
    if kind == "int":
        if not _INT_MIN <= number <= _INT_MAX:
            raise ValueError("integer result outside signed 32-bit range")
    elif not math.isfinite(number):
        raise ValueError("double result must be finite binary64")
    return kind, number


def _describe(value: Value) -> str:
    return f"{value[0]} {value[1]!r}"


class _Parser:
    """Evaluate only declarations and a fully consumed, side-effect-free grammar."""

    def __init__(self, program: str) -> None:
        if not program.strip() or len(program) > 1024:
            raise ValueError("C arithmetic input is empty or exceeds 1024 characters")
        self.tokens: list[str] = []
        cursor = 0
        while cursor < len(program.rstrip()):
            match = _TOKEN.match(program, cursor)
            if match is None or match.group(1) in {"++", "--"}:
                raise ValueError("unsupported C token or side effect")
            self.tokens.append(match.group(1))
            cursor = match.end()
        if len(self.tokens) > 128:
            raise ValueError("C arithmetic exceeds 128 tokens")
        self.position = 0
        self.variables: dict[str, Value] = {}
        self.steps: list[str] = []

    def peek(self) -> str:
        return self.tokens[self.position] if self.position < len(self.tokens) else ""

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if not token or (expected is not None and token != expected):
            raise ValueError(f"expected {expected or 'token'}, found {token!r}")
        self.position += 1
        return token

    def convert(self, kind: str, value: Value, context: str) -> Value:
        number = math.trunc(value[1]) if kind == "int" else float(value[1])
        result = _checked(kind, number)
        rule = (
            "; discard any fractional part toward zero"
            if kind == "int"
            else "; no integer truncation"
        )
        self.steps.append(f"{context}: {_describe(value)} -> {_describe(result)}{rule}.")
        return result

    def expression(self, minimum: int = 0) -> Value:
        left = self.unary()
        precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
        while self.peek() in precedence and precedence[self.peek()] >= minimum:
            operator = self.take()
            right = self.expression(precedence[operator] + 1)
            kind = "double" if "double" in (left[0], right[0]) else "int"
            if kind == "double":
                if left[0] != kind:
                    left = self.convert(kind, left, "Convert left operand before operation")
                if right[0] != kind:
                    right = self.convert(kind, right, "Convert right operand before operation")
            a, b = left[1], right[1]
            rule = ""
            if operator == "/":
                if b == 0:
                    raise ValueError("division by zero is unsupported")
                if kind == "int":
                    # Divide positive magnitudes, then restore sign; do not floor negatives.
                    number = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                    rule = "integer division truncates toward zero"
                else:
                    number = a / b
                    rule = "floating division after operand conversion"
            elif operator == "+":
                number = a + b
            elif operator == "-":
                number = a - b
            else:
                number = a * b
            result = _checked(kind, number)
            self.steps.append(
                f"{_describe(left)} {operator} {_describe(right)} = {_describe(result)}"
                f"{'; ' + rule if rule else ''}."
            )
            left = result
        return left

    def unary(self) -> Value:
        token = self.take()
        if token in {"+", "-"}:
            operand = self.unary()
            result = _checked(operand[0], operand[1] if token == "+" else -operand[1])
            self.steps.append(f"Unary {token}({_describe(operand)}) = {_describe(result)}.")
            return result
        if token == "(":
            if self.peek() in {"int", "double"}:
                kind = self.take()
                self.take(")")
                return self.convert(kind, self.unary(), "Explicit cast")
            result = self.expression()
            self.take(")")
            return result
        if _NAME.fullmatch(token):
            if token not in self.variables:
                raise ValueError(f"unknown variable: {token}")
            return self.variables[token]
        if not token[0].isdigit() and not token.startswith("."):
            raise ValueError("expected numeric literal, variable, or parenthesized expression")
        if any(marker in token for marker in (".", "e", "E")):
            return _checked("double", float(token))
        if len(token) > 1 and token.startswith("0"):
            raise ValueError("only decimal integer literals are supported")
        return _checked("int", int(token))

    def declarations(self) -> None:
        while self.peek():
            kind = self.take()
            if kind not in {"int", "double"}:
                raise ValueError("only initialized int/double declarations are supported")
            while True:
                name = self.take()
                if not _NAME.fullmatch(name) or _KEYWORDS.fullmatch(name) or name in self.variables:
                    raise ValueError("invalid or duplicate declaration name")
                self.take("=")
                self.variables[name] = self.convert(kind, self.expression(), f"Initialize {name}")
                if self.peek() != ",":
                    break
                self.take(",")
            self.take(";")


def calculate_c_arithmetic(*, program: str) -> SEPCalculatorResult:
    """Evaluate the bounded numerical model and preserve all declared outputs."""
    parser = _Parser(program)
    parser.declarations()
    assumptions = "C arithmetic subset: 32-bit signed int and finite binary64 double."
    values = {name: value[1] for name, value in parser.variables.items()}
    summary = "; ".join(
        f"{value[0]} {name} = {value[1]!r}" for name, value in parser.variables.items()
    )
    return SEPCalculatorResult(
        answer=f"{assumptions} {summary}.",
        quantity="C typed arithmetic",
        unit="dimensionless",
        steps=[assumptions, *parser.steps],
        values=values,
    )


def _request_is_supported(request: str, program: str) -> bool:
    """Match complete explanation clauses, not a bag of potentially executable task words."""
    if re.fullmatch(
        r"\.\s*What values result(?:,? and why)?\?|"
        r"\s+give different values(?: in this case)?\?\s*State both values and explain\.",
        request,
        re.IGNORECASE,
    ):
        return True
    if "/" in program and re.fullmatch(
        rf"\.\s*Explain {_EXPLANATION_CLAUSE}"
        rf"(?:(?:,\s*(?:and\s+)?|\s+and\s+){_EXPLANATION_CLAUSE})*\.",
        request,
        re.IGNORECASE,
    ):
        return True
    cast = re.fullmatch(
        r"\.\s*State both values and explain whether the (?:outer )?cast in "
        r"([A-Za-z_][A-Za-z_0-9]*) causes integer truncation(?: in this case)?\.",
        request,
        re.IGNORECASE,
    )
    if cast:
        return (
            re.search(
                rf"\b(?:int|double)\s+{re.escape(cast.group(1))}\s*=\s*\(\s*(?:int|double)\s*\)",
                program,
            )
            is not None
        )
    types = re.fullmatch(
        r"\.\s*Explain the type of ([A-Za-z_][A-Za-z_0-9]*(?: and [A-Za-z_][A-Za-z_0-9]*)+)"
        r" and why values result\.",
        request,
        re.IGNORECASE,
    )
    names = set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*=", program))
    return bool(types and set(re.split(r"\s+and\s+", types.group(1), flags=re.IGNORECASE)) <= names)


def _program_text(question: str) -> str | None:
    """Accept whole code or one restricted prose envelope, never extract loose snippets."""
    if len(question) > 1600:
        return None
    text = question.strip()
    fenced = re.fullmatch(r"```(?:c)?\s*\n(.*?)\n```|`([^`]+)`", text, re.DOTALL)
    if fenced:
        text = (fenced.group(1) if fenced.group(1) is not None else fenced.group(2)).strip()
    elif re.match(r"In C,\s*", text, re.IGNORECASE):
        text = re.sub(r"\AIn C,\s*(?:let\s+)?", "", text, flags=re.IGNORECASE)
        end = _PROSE_END.search(text)
        if end:
            if not _request_is_supported(text[end.start() :], text[: end.start()]):
                return None
            text = text[: end.start()]
            text = re.sub(r"\.\s*(?:Compare|Find|Do)\s+", ";", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+(?:and|with)\s+(?=(?:int|double)\s)", ";", text)
            text += ";"
    if not re.match(r"(?:int|double)\s", text) or not text.endswith(";"):
        return None
    return text


def classify_sep_formula_by_keywords(question: str) -> str | None:
    return "c_typed_arithmetic" if _program_text(question) is not None else None


def extract_sep_variables(formula_name: str, question: str) -> dict[str, str] | None:
    program = _program_text(question)
    return {"program": program} if formula_name == "c_typed_arithmetic" and program else None


def dispatch_formula(formula_name: str, *, program: str) -> SEPCalculatorResult:
    if formula_name != "c_typed_arithmetic":
        raise ValueError("unknown SEP calculator")
    return calculate_c_arithmetic(program=program)
