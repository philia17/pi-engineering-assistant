"""Deterministic calculators for MMT numerical-integration and probability questions."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class MMTCalculatorResult:
    """Structured MMT calculator result with a ready-to-grade answer string."""

    answer: str
    quantity: str
    unit: str
    steps: list[str]
    values: dict[str, float] = field(default_factory=dict)


def _require_positive(name: str, value: float) -> None:
    """Raise when a calculator input is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def calculate_composite_trapezoidal(
    *,
    values: Sequence[float],
    spacing: float,
) -> MMTCalculatorResult:
    """Estimate an integral from uniformly spaced samples with the trapezoidal rule.

    Uses ``int ≈ Δ (v_0/2 + Σ_{j=1}^{N-1} v_j + v_N/2)`` from the MMT numerical
    integration note (MMT_Core_04).
    """
    _require_positive("spacing", spacing)
    if len(values) < 2:
        raise ValueError("composite trapezoidal rule requires at least two samples")

    interior = sum(values[1:-1])
    weighted_sum = values[0] / 2 + interior + values[-1] / 2
    integral = spacing * weighted_sum
    steps = [
        "Composite trapezoidal rule: Δ (v_0/2 + Σ_{j=1}^{N-1} v_j + v_N/2).",
        (f"Weighted sum = {values[0]:g}/2 + {interior:g} + {values[-1]:g}/2 = {weighted_sum:g}."),
        f"∫ v dt ≈ {spacing:g} * {weighted_sum:g} = {integral:g}.",
    ]
    return MMTCalculatorResult(
        answer=f"∫ v(t) dt ≈ {integral:.6g} (integration-variable units).",
        quantity="integral",
        unit="integration-variable units",
        steps=steps,
        values={"integral": integral, "weighted_sum": weighted_sum, "spacing": spacing},
    )


def calculate_sample_mean(*, values: Sequence[float]) -> MMTCalculatorResult:
    """Compute the population mean ⟨x⟩ = (1/N) Σ x_j from supplied samples."""
    if not values:
        raise ValueError("values must be non-empty")
    count = len(values)
    mean = sum(values) / count
    steps = [
        f"Population mean ⟨x⟩ = (1/{count}) Σ x_j.",
        f"⟨x⟩ = ({'+'.join(f'{value:g}' for value in values)}) / {count} = {mean:g}.",
    ]
    return MMTCalculatorResult(
        answer=f"⟨x⟩ = {mean:.6g}.",
        quantity="mean",
        unit="same as x",
        steps=steps,
        values={"mean": mean, "count": float(count)},
    )


def calculate_composite_simpson(*, values: Sequence[float], spacing: float) -> MMTCalculatorResult:
    """Integrate equally spaced samples with composite Simpson's one-third rule."""
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Simpson spacing must be finite and positive")
    intervals: int = len(values) - 1
    if intervals < 2 or intervals % 2:
        raise ValueError("Simpson's one-third rule requires an even number of subintervals")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Simpson samples must be finite")
    try:
        odd_sum: float = math.fsum(values[1:-1:2])
        even_sum: float = math.fsum(values[2:-1:2])
    except OverflowError as error:
        raise ValueError("Simpson sample sums exceed the supported numeric range") from error
    weighted_sum: float = values[0] + 4 * odd_sum + 2 * even_sum + values[-1]
    integral: float = spacing * weighted_sum / 3
    if not math.isfinite(integral):
        raise ValueError("Simpson result must be finite")
    steps: list[str] = [
        f"Use {intervals} equal subintervals of width h = {spacing:g}; {intervals} is even.",
        f"Sample values: [{', '.join(f'{value:g}' for value in values)}].",
        "Simpson 1/3: integral ≈ h/3 * (y_0 + 4Σy_odd + 2Σy_even + y_n).",
        f"Weighted sum = {values[0]:g} + 4*({odd_sum:g}) + 2*({even_sum:g}) "
        f"+ {values[-1]:g} = {weighted_sum:g}.",
        f"Integral ≈ {spacing:g}/3 * {weighted_sum:g} = {integral:g} "
        "(integrand units × integration-variable units).",
    ]
    return MMTCalculatorResult(
        answer=f"Integral ≈ {integral:.6g} (integrand units × integration-variable units).",
        quantity="integral",
        unit="integrand units × integration-variable units",
        steps=steps,
        values={
            "integral": integral,
            "weighted_sum": weighted_sum,
            "spacing": spacing,
            "subintervals": float(intervals),
        },
    )


# These limits bound the explicitly supported polynomial grammar, not a general CAS.
_MAX_POLYNOMIAL_DEGREE: int = 12
_MAX_POLYNOMIAL_NODES: int = 64
_MAX_SIMPSON_SUBINTERVALS: int = 1000


def _multiply_polynomials(left: list[float], right: list[float]) -> list[float]:
    """Multiply coefficient lists without running user-supplied expressions."""
    degree: int = len(left) + len(right) - 2
    if degree > _MAX_POLYNOMIAL_DEGREE:
        raise ValueError("polynomial degree exceeds supported limit")
    coefficients: list[float] = [0.0] * (degree + 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            coefficients[i + j] += a * b
    return coefficients


def _polynomial_node(node: ast.AST, variable: str) -> list[float]:
    """Convert only numeric literals, one variable, +, -, *, and integer powers."""
    coefficients: list[float]
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        coefficients = [float(node.value)]
    elif isinstance(node, ast.Name) and node.id == variable:
        coefficients = [0.0, 1.0]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        sign: int = -1 if isinstance(node.op, ast.USub) else 1
        coefficients = [sign * value for value in _polynomial_node(node.operand, variable)]
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Mult):
        left: list[float] = _polynomial_node(node.left, variable)
        right: list[float] = _polynomial_node(node.right, variable)
        if isinstance(node.op, ast.Mult):
            coefficients = _multiply_polynomials(left, right)
        else:
            size: int = max(len(left), len(right))
            left += [0.0] * (size - len(left))
            right += [0.0] * (size - len(right))
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            coefficients = [a + sign * b for a, b in zip(left, right, strict=True)]
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        if (
            not isinstance(node.right, ast.Constant)
            or type(node.right.value) is not int
            or not 0 <= node.right.value <= _MAX_POLYNOMIAL_DEGREE
        ):
            raise ValueError("polynomial powers must be small nonnegative integer literals")
        base: list[float] = _polynomial_node(node.left, variable)
        coefficients = [1.0]
        for _ in range(node.right.value):
            coefficients = _multiply_polynomials(coefficients, base)
    else:
        raise ValueError(
            "unsupported polynomial expression; calls and other variables are forbidden"
        )
    if any(not math.isfinite(value) for value in coefficients):
        raise ValueError("polynomial coefficients must be finite")
    return coefficients


def _polynomial_coefficients(expression: str, variable: str) -> list[float]:
    """Parse a bounded polynomial; never use eval, exec, or executable callbacks."""
    if re.fullmatch(r"[a-z]", variable) is None or len(expression) > 256:
        raise ValueError("unsupported polynomial variable or expression length")
    normalized: str = expression.strip().replace("−", "-").replace("×", "*")
    normalized = normalized.replace("²", "**2").replace("³", "**3").replace("^", "**")
    normalized = re.sub(r"\b([a-z])\s+squared\b", r"\1**2", normalized)
    normalized = re.sub(r"\b([a-z])\s+cubed\b", r"\1**3", normalized)
    try:
        tree: ast.Expression = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ValueError("invalid polynomial syntax") from error
    if sum(1 for _ in ast.walk(tree)) > _MAX_POLYNOMIAL_NODES:
        raise ValueError("polynomial expression exceeds supported node limit")
    return _polynomial_node(tree.body, variable)


def calculate_polynomial_simpson(
    *, expression: str, variable: str, lower: float, upper: float, subintervals: int
) -> MMTCalculatorResult:
    """Apply Simpson to a small explicit polynomial on a finite increasing interval."""
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("Simpson requires finite bounds with upper > lower")
    if (
        type(subintervals) is not int
        or not 2 <= subintervals <= _MAX_SIMPSON_SUBINTERVALS
        or subintervals % 2
    ):
        raise ValueError("Simpson subintervals must be an even integer from 2 to 1000")
    coefficients: list[float] = _polynomial_coefficients(expression, variable)
    spacing: float = (upper - lower) / subintervals
    samples: list[float] = []
    for index in range(subintervals + 1):
        point: float = lower + index * spacing
        value: float = 0.0
        for coefficient in reversed(coefficients):
            value = value * point + coefficient
        samples.append(value)
    result: MMTCalculatorResult = calculate_composite_simpson(values=samples, spacing=spacing)
    result.steps[:0] = [
        f"Given f({variable}) = {expression}, integrate from {lower:g} to {upper:g}.",
        f"h = ({upper:g} - ({lower:g}))/{subintervals} = {spacing:g}; "
        f"evaluate f at {variable}_j = {lower:g} + j*{spacing:g}.",
    ]
    result.values.update({"lower": lower, "upper": upper})
    return result


def calculate_variance_stddev_from_moments(
    *,
    mean: float,
    mean_square: float,
) -> MMTCalculatorResult:
    """Compute population variance and standard deviation from ⟨x⟩ and ⟨x²⟩.

    Uses ``σ_x² = ⟨x²⟩ − ⟨x⟩²`` and ``σ_x = sqrt(σ_x²)`` from the MMT
    probability note (MMT_Core_07).
    """
    variance = mean_square - mean * mean
    if variance < 0 and abs(variance) < 1e-9:
        variance = 0.0
    if variance < 0:
        raise ValueError("moments imply a negative variance; inputs are inconsistent")
    stddev = math.sqrt(variance)
    steps = [
        "Population variance σ_x² = ⟨x²⟩ − ⟨x⟩².",
        f"σ_x² = {mean_square:g} − {mean:g}² = {variance:g}.",
        f"σ_x = sqrt({variance:g}) = {stddev:g}.",
    ]
    return MMTCalculatorResult(
        answer=f"σ_x² = {variance:.6g}; σ_x = {stddev:.6g}.",
        quantity="variance and standard deviation",
        unit="same as x and x²",
        steps=steps,
        values={"variance": variance, "stddev": stddev},
    )


def calculate_sample_variance_stddev(
    *,
    values: Sequence[float],
    ddof: int = 0,
) -> MMTCalculatorResult:
    """Compute population (ddof=0) or unbiased sample variance (ddof=1)."""
    if ddof not in {0, 1}:
        raise ValueError("variance ddof must be 0 (population) or 1 (sample)")
    if ddof == 1 and len(values) < 2:
        raise ValueError("sample variance requires at least two samples")
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("variance requires non-empty finite samples")
    count: int = len(values)
    denominator: int = count - ddof
    try:
        mean: float = math.fsum(values) / count
        squared_sum: float = math.fsum((value - mean) ** 2 for value in values)
    except OverflowError as error:
        raise ValueError("variance exceeds the supported numeric range") from error
    variance: float = squared_sum / denominator
    stddev: float = math.sqrt(variance)
    if not math.isfinite(variance):
        raise ValueError("variance result must be finite")
    convention: str = "Sample" if ddof else "Population"
    symbol: str = "s" if ddof else "σ_x"
    steps: list[str] = [
        f"Mean = ({'+'.join(f'({value:g})' for value in values)}) / {count} = {mean:g}.",
        "Sum all squared deviations: "
        + " + ".join(f"({value:g} - ({mean:g}))²" for value in values)
        + f" = {squared_sum:g}.",
        f"{convention} variance uses n{'−1' if ddof else ''} = {denominator}: "
        f"{symbol}² = {squared_sum:g}/{denominator} = {variance:g}.",
        f"{convention} standard deviation {symbol} = sqrt({variance:g}) = {stddev:g}.",
    ]
    return MMTCalculatorResult(
        answer=f"{symbol}² = {variance:.6g}; {symbol} = {stddev:.6g}.",
        quantity="variance and standard deviation",
        unit="same as x and x²",
        steps=steps,
        values={
            "mean": mean,
            "variance": variance,
            "stddev": stddev,
            "count": float(count),
            "denominator": float(denominator),
            "ddof": float(ddof),
            "sum_squared_deviations": squared_sum,
        },
    )


@dataclass(frozen=True)
class MMTFormulaSpec:
    """Metadata for a registered MMT calculator formula."""

    name: str
    topic: str
    description: str
    func: Callable[..., MMTCalculatorResult]
    param_names: tuple[str, ...]
    quantity: str
    unit: str


FORMULA_REGISTRY: dict[str, MMTFormulaSpec] = {
    "composite_simpson_rule": MMTFormulaSpec(
        name="composite_simpson_rule",
        topic="numerical_integration",
        description="Composite Simpson 1/3 for equally spaced samples and even subintervals",
        func=calculate_composite_simpson,
        param_names=("values", "spacing"),
        quantity="integral",
        unit="integrand units × integration-variable units",
    ),
    "polynomial_simpson_rule": MMTFormulaSpec(
        name="polynomial_simpson_rule",
        topic="numerical_integration",
        description="Simpson 1/3 for a bounded single-variable polynomial on a finite interval",
        func=calculate_polynomial_simpson,
        param_names=("expression", "variable", "lower", "upper", "subintervals"),
        quantity="integral",
        unit="integrand units × integration-variable units",
    ),
    "composite_trapezoidal_rule": MMTFormulaSpec(
        name="composite_trapezoidal_rule",
        topic="numerical_integration",
        description="Composite trapezoidal rule for uniformly spaced samples",
        func=calculate_composite_trapezoidal,
        param_names=("values", "spacing"),
        quantity="integral",
        unit="integration-variable units",
    ),
    "sample_mean": MMTFormulaSpec(
        name="sample_mean",
        topic="probability",
        description="Population mean of a list of sample values",
        func=calculate_sample_mean,
        param_names=("values",),
        quantity="mean",
        unit="same as x",
    ),
    "variance_stddev_from_moments": MMTFormulaSpec(
        name="variance_stddev_from_moments",
        topic="probability",
        description="Population variance and standard deviation from ⟨x⟩ and ⟨x²⟩",
        func=calculate_variance_stddev_from_moments,
        param_names=("mean", "mean_square"),
        quantity="variance and standard deviation",
        unit="same as x and x²",
    ),
    "sample_variance_stddev": MMTFormulaSpec(
        name="sample_variance_stddev",
        topic="probability",
        description="Population or unbiased sample variance and associated standard deviation",
        func=calculate_sample_variance_stddev,
        param_names=("values", "ddof"),
        quantity="variance and standard deviation",
        unit="same as x and x²",
    ),
}


_NUMBER: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_LIST_RE = re.compile(rf"\[(?P<values>\s*{_NUMBER}(?:\s*,\s*{_NUMBER})*\s*)\]")
_DELTA_EQUALS_RE = re.compile(
    r"(?:Δ|delta)\s*=\s*(?P<delta>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_DELTA_SPACING_RE = re.compile(
    r"spacing(?:\s+of)?\s+(?P<delta>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_MEAN_MOMENT_RE = re.compile(
    r"(?:⟨x⟩|<x>|〈x〉)\s*=\s*(?P<mean>[-+]?\d+(?:\.\d+)?)",
)
_MEAN_SQUARE_MOMENT_RE = re.compile(
    r"(?:⟨x²⟩|<x²>|⟨x\^2⟩|<x\^2>)\s*=\s*(?P<mean_square>[-+]?\d+(?:\.\d+)?)",
)
_ANY_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_EXPLANATION_MARKERS = (
    "what is",
    "what does",
    "why",
    "compare",
    "explain",
    "describe",
    "how are",
    "when is",
    "write the",
    "state the",
    "give the",
)
_SAMPLE_VARIANCE_MARKERS = (
    "sample variance",
    "sample standard deviation",
    "unbiased variance",
    "unbiased standard deviation",
    "variance of the sample",
    "standard deviation of the sample",
    "variance of samples",
    "standard deviation of samples",
    "unbiased estimate of variance",
    "unbiased estimate of the variance",
    "unbiased estimate of standard deviation",
    "unbiased estimate of the standard deviation",
)
_POPULATION_VARIANCE_MARKERS = (
    "population variance",
    "population standard deviation",
)
_SIMPSON_SPACING_RE: re.Pattern[str] = re.compile(
    rf"(?:\b(?:h|delta|spacing)\b|Δ)\s*(?:=|of|is)?\s*(?P<value>{_NUMBER})(?!\w|\.\d)",
    re.IGNORECASE,
)
_POLYNOMIAL_FUNCTION_RE: re.Pattern[str] = re.compile(
    r"\b[fg]\s*\(\s*(?P<variable>[a-z])\s*\)\s*=\s*(?P<expression>.+?)"
    r"(?=\s+(?:from|on|over|with|using)\b|[,;\n]|$)",
    re.IGNORECASE,
)
_INTEGRAL_EXPRESSION_RE: re.Pattern[str] = re.compile(
    r"\bintegral\s+of\s+(?P<expression>.+?)\s+from\b", re.IGNORECASE
)
_SIMPSON_BOUNDS_RE: re.Pattern[str] = re.compile(
    rf"\bfrom\s+(?P<lower>{_NUMBER})\s+to\s+(?P<upper>{_NUMBER})"
    r"(?=\s*(?:[,.;]|\b(?:with|using|and|on|over)\b|$))",
    re.IGNORECASE,
)
_SIMPSON_INTERVAL_RE: re.Pattern[str] = re.compile(
    rf"\b(?:on|over)\s*\[\s*(?P<lower>{_NUMBER})\s*,\s*(?P<upper>{_NUMBER})\s*\]",
    re.IGNORECASE,
)
_SMALL_EVEN_COUNTS: dict[str, int] = {
    "two": 2,
    "four": 4,
    "six": 6,
    "eight": 8,
    "ten": 10,
    "twelve": 12,
}
_SIMPSON_COUNT_RE: re.Pattern[str] = re.compile(
    r"\bn\s*=\s*(?P<n>\d+)(?!\w|\.\d)|(?<![\w.+−-])"
    r"(?P<count>\d+|two|four|six|eight|ten|twelve)"
    r"\s+(?:equal\s+)?(?:subintervals|intervals)\b",
    re.IGNORECASE,
)


_SIMPSON_OTHER_RULE_RE: re.Pattern[str] = re.compile(r"3\s*/\s*8|three[- ]eighths?", re.IGNORECASE)


def parse_number_list(text: str) -> list[float] | None:
    """Extract the first bracketed list of numbers from ``text``."""
    match = _NUMBER_LIST_RE.search(text)
    if match is None:
        return None
    return [float(part.strip()) for part in match.group("values").split(",")]


def _explicit_variance_convention(
    question: str,
) -> Literal["population", "sample", "conflict"] | None:
    """Resolve explicit conventions, retaining contradictory requests as conflicts."""
    normalized: str = question.casefold().replace("−", "-")
    population: bool = any(marker in normalized for marker in _POPULATION_VARIANCE_MARKERS)
    sample: bool = any(marker in normalized for marker in _SAMPLE_VARIANCE_MARKERS)
    sample = sample or re.search(r"\bn\s*-\s*1\b", normalized) is not None
    population = (
        population
        or re.search(
            r"\b(?:denominator\s*(?:of|=|is)?\s*|divid(?:e|ing)\s+by\s+)n\b(?!\s*-)",
            normalized,
        )
        is not None
    )
    if population and sample:
        return "conflict"
    if population:
        return "population"
    if sample:
        return "sample"
    return None


def _extract_polynomial_simpson(question: str) -> dict[str, Any] | None:
    """Extract an explicit polynomial, interval and subdivision count, without guessing."""
    function: re.Match[str] | None = _POLYNOMIAL_FUNCTION_RE.search(question)
    integral: re.Match[str] | None = _INTEGRAL_EXPRESSION_RE.search(question)
    bounds: set[tuple[float, float]] = {
        (float(match.group("lower")), float(match.group("upper")))
        for pattern in (_SIMPSON_BOUNDS_RE, _SIMPSON_INTERVAL_RE)
        for match in pattern.finditer(question)
    }
    counts: set[int] = set()
    for match in _SIMPSON_COUNT_RE.finditer(question):
        token: str = (match.group("n") or match.group("count")).casefold()
        counts.add(int(token) if token.isdigit() else _SMALL_EVEN_COUNTS[token])
    matched: re.Match[str] | None = function or integral
    if len(bounds) != 1 or len(counts) != 1 or matched is None:
        return None
    expression: str = matched.group("expression").strip().rstrip(".")
    variable: str = function.group("variable").casefold() if function else "x"
    if function is None:
        names: set[str] = set(re.findall(r"\b[a-z]\b", expression.casefold()))
        if len(names) == 1:
            variable = names.pop()
    try:
        _polynomial_coefficients(expression, variable)
    except (ValueError, OverflowError):
        return None
    lower: float
    upper: float
    lower, upper = bounds.pop()
    return {
        "expression": expression,
        "variable": variable,
        "lower": lower,
        "upper": upper,
        "subintervals": counts.pop(),
    }


def extract_mmt_variables(formula_name: str, question: str) -> dict[str, Any] | None:
    """Deterministically extract calculator variables from an MMT question.

    Returns ``None`` when the question lacks the structured values required by
    the named formula; formula-lookup and conceptual questions fall through to
    the grounded RAG path.
    """
    if formula_name in {"composite_simpson_rule", "polynomial_simpson_rule"} and (
        _SIMPSON_OTHER_RULE_RE.search(question)
    ):
        return None
    if formula_name == "composite_trapezoidal_rule":
        values = parse_number_list(question)
        if values is None or len(values) < 2:
            return None
        delta_match = _DELTA_EQUALS_RE.search(question) or _DELTA_SPACING_RE.search(question)
        if delta_match is None:
            return None
        return {"values": values, "spacing": float(delta_match.group("delta"))}

    if formula_name == "polynomial_simpson_rule":
        return _extract_polynomial_simpson(question)
    if formula_name == "composite_simpson_rule":
        if re.search(
            r"\b(?:non[- ]?uniform(?:ly)?|unequal(?:ly)?|uneven(?:ly)?)\b",
            question,
            re.IGNORECASE,
        ):
            return None
        lists: list[re.Match[str]] = list(_NUMBER_LIST_RE.finditer(question))
        spacings: set[float] = {
            float(match.group("value")) for match in _SIMPSON_SPACING_RE.finditer(question)
        }
        if len(lists) != 1 or len(spacings) != 1:
            return None
        samples: list[float] = [float(value) for value in lists[0].group("values").split(",")]
        for match in _SIMPSON_COUNT_RE.finditer(question):
            token: str = (match.group("n") or match.group("count")).casefold()
            intervals: int = int(token) if token.isdigit() else _SMALL_EVEN_COUNTS[token]
            if intervals != len(samples) - 1:
                return None
        return {"values": samples, "spacing": spacings.pop()}
    if formula_name in {"sample_mean", "sample_variance_stddev"}:
        values = parse_number_list(question)
        if values is None:
            return None
        if formula_name == "sample_variance_stddev":
            convention: str | None = _explicit_variance_convention(question)
            if convention == "conflict":
                return None
            if convention == "sample":
                return {"values": values, "ddof": 1}
        return {"values": values}

    if formula_name == "variance_stddev_from_moments":
        if _explicit_variance_convention(question) in {"sample", "conflict"}:
            return None
        mean_match = _MEAN_MOMENT_RE.search(question)
        mean_square_match = _MEAN_SQUARE_MOMENT_RE.search(question)
        if mean_match is None or mean_square_match is None:
            return None
        return {
            "mean": float(mean_match.group("mean")),
            "mean_square": float(mean_square_match.group("mean_square")),
        }

    return None


def classify_mmt_formula_by_keywords(question: str) -> str | None:
    """Classify unambiguous MMT calculator routes without an LLM call.

    Returns a registry key or ``None`` when the question is conceptual,
    formula-lookup, or otherwise not a supported deterministic calculation.
    """
    normalized = question.casefold()
    if any(marker in normalized for marker in _EXPLANATION_MARKERS) and (
        _ANY_NUMBER_RE.search(question) is None
    ):
        return None
    moments_present = _MEAN_MOMENT_RE.search(question) is not None and (
        _MEAN_SQUARE_MOMENT_RE.search(question) is not None
    )
    variance_cues = ("variance", "standard deviation", "σ")

    if "simpson" in normalized:
        if _SIMPSON_OTHER_RULE_RE.search(question):
            return None
        if _POLYNOMIAL_FUNCTION_RE.search(question) or _INTEGRAL_EXPRESSION_RE.search(question):
            return "polynomial_simpson_rule"
        if parse_number_list(question) is not None:
            return "composite_simpson_rule"
        return None
    if "trapezoidal" in normalized and any(
        cue in normalized for cue in ("composite", "integr", "∫", "estimate")
    ):
        return "composite_trapezoidal_rule"
    if moments_present and any(cue in normalized for cue in variance_cues):
        if _explicit_variance_convention(question) in {"sample", "conflict"}:
            return None
        return "variance_stddev_from_moments"
    if parse_number_list(question) is not None:
        if any(cue in normalized for cue in variance_cues):
            if _explicit_variance_convention(question) == "conflict":
                return None
            return "sample_variance_stddev"
        if "mean" in normalized or "average" in normalized:
            return "sample_mean"
    return None


def dispatch_formula(formula_name: str, **params: Any) -> MMTCalculatorResult:
    """Look up an MMT formula by name and call it with the given parameters."""
    spec = FORMULA_REGISTRY[formula_name]
    return spec.func(**params)


def schema_for_formula(formula_name: str) -> dict[str, str]:
    """Return a human-readable variable schema for the named formula."""
    spec = FORMULA_REGISTRY[formula_name]
    descriptions = {
        "values": "List of uniformly spaced sample values",
        "spacing": "Uniform sample spacing Δ",
        "mean": "Population mean ⟨x⟩",
        "mean_square": "Second moment ⟨x²⟩",
        "ddof": "0 for population variance; 1 for unbiased sample variance",
        "expression": "Small single-variable polynomial using +, -, *, and integer powers",
        "variable": "Single lowercase variable name",
        "lower": "Finite lower integration bound",
        "upper": "Finite upper integration bound greater than lower",
        "subintervals": "Even number of equal subintervals from 2 to 1000",
    }
    return {name: descriptions.get(name, name) for name in spec.param_names}
