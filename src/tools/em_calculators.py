"""Deterministic calculators for EM matrix, phasor, and impedance questions."""

from __future__ import annotations

import cmath
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EMCalculatorResult:
    """Structured EM calculator result with a ready-to-grade answer string."""

    answer: str
    quantity: str
    unit: str
    steps: list[str]
    values: dict[str, float] = field(default_factory=dict)
    matrix: tuple[tuple[float, ...], ...] | None = None


def _as_matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """Copy a matrix-like sequence into a mutable list of lists."""
    return [[float(value) for value in row] for row in rows]


def _format_matrix(matrix: Sequence[Sequence[float]]) -> str:
    """Format a matrix as ``[[a,b],[c,d]]`` with compact integers when exact."""

    def fmt(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.6g}"

    row_text = ",".join("[" + ",".join(fmt(value) for value in row) + "]" for row in matrix)
    return f"[{row_text}]"


def calculate_matrix_multiply(
    *,
    matrix_a: Sequence[Sequence[float]],
    matrix_b: Sequence[Sequence[float]],
) -> EMCalculatorResult:
    """Multiply matrix A by matrix B."""
    a = _as_matrix(matrix_a)
    b = _as_matrix(matrix_b)
    if not a or not b or not a[0] or not b[0]:
        raise ValueError("matrices must be non-empty")
    if any(len(row) != len(a[0]) for row in a):
        raise ValueError("matrix_a rows must have equal length")
    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("matrix_b rows must have equal length")
    if len(a[0]) != len(b):
        raise ValueError("inner dimensions must match for matrix multiplication")

    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    product = [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)
    ]
    steps = [
        f"A is {rows}x{inner}, B is {inner}x{cols}.",
        f"C = A B = {_format_matrix(product)}.",
    ]
    values = {f"c{i + 1}{j + 1}": product[i][j] for i in range(rows) for j in range(cols)}
    return EMCalculatorResult(
        answer=f"{_format_matrix(product)}.",
        quantity="C",
        unit="matrix",
        steps=steps,
        values=values,
        matrix=tuple(tuple(row) for row in product),
    )


def calculate_matrix_determinant(
    *,
    matrix: Sequence[Sequence[float]],
) -> EMCalculatorResult:
    """Compute the determinant of a 2x2 or 3x3 matrix."""
    m = _as_matrix(matrix)
    n = len(m)
    if n not in {2, 3} or any(len(row) != n for row in m):
        raise ValueError("determinant supports only square 2x2 or 3x3 matrices")

    if n == 2:
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        steps = [
            "For 2x2 matrix [[a,b],[c,d]], det = ad - bc.",
            (f"det = ({m[0][0]:g})({m[1][1]:g}) - ({m[0][1]:g})({m[1][0]:g}) = {det:g}."),
        ]
    else:
        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        steps = [
            "For 3x3 matrix use cofactor expansion along the first row.",
            f"det = {det:g}.",
        ]
    return EMCalculatorResult(
        answer=f"det = {det:g}.",
        quantity="det",
        unit="scalar",
        steps=steps,
        values={"det": det},
        matrix=tuple(tuple(row) for row in m),
    )


def calculate_matrix_inverse_2x2(
    *,
    matrix: Sequence[Sequence[float]],
) -> EMCalculatorResult:
    """Compute the inverse of a 2x2 matrix."""
    m = _as_matrix(matrix)
    if len(m) != 2 or any(len(row) != 2 for row in m):
        raise ValueError("inverse supports only 2x2 matrices")
    a, b = m[0]
    c, d = m[1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("matrix is singular and has no inverse")
    inv = [[d / det, -b / det], [-c / det, a / det]]
    steps = [
        "For 2x2 [[a,b],[c,d]], inverse = (1/det) * [[d,-b],[-c,a]].",
        f"det = {det:g}.",
        f"A^(-1) = {_format_matrix(inv)}.",
    ]
    return EMCalculatorResult(
        answer=f"{_format_matrix(inv)}.",
        quantity="A_inv",
        unit="matrix",
        steps=steps,
        values={
            "det": det,
            "c11": inv[0][0],
            "c12": inv[0][1],
            "c21": inv[1][0],
            "c22": inv[1][1],
        },
        matrix=tuple(tuple(row) for row in inv),
    )


def calculate_matrix_inverse_diagonal(*, matrix: Sequence[Sequence[float]]) -> EMCalculatorResult:
    """Invert a nonempty square diagonal matrix by reciprocating every diagonal entry."""
    m: list[list[float]] = _as_matrix(matrix)
    size: int = len(m)
    if not size or any(len(row) != size for row in m):
        raise ValueError("diagonal inverse requires a nonempty square matrix")
    if any(not math.isfinite(value) for row in m for value in row):
        raise ValueError("diagonal matrix entries must be finite")
    if any(value != 0 for i, row in enumerate(m) for j, value in enumerate(row) if i != j):
        raise ValueError("diagonal inverse requires every off-diagonal entry to be zero")
    if any(m[index][index] == 0 for index in range(size)):
        raise ValueError("matrix is singular: a diagonal entry is zero")
    inverse: list[list[float]] = [
        [1 / m[i][i] if i == j else 0.0 for j in range(size)] for i in range(size)
    ]
    if any(not math.isfinite(inverse[index][index]) for index in range(size)):
        raise ValueError("diagonal inverse entries must be finite")
    steps: list[str] = [
        f"A is {size}x{size}, every off-diagonal entry is zero, and every diagonal entry is nonzero.",
        "For diagonal A, (A^-1)_ii = 1/A_ii and every off-diagonal inverse entry is zero.",
        *[
            f"Entry ({index + 1},{index + 1}): 1/({m[index][index]:g}) "
            f"= {inverse[index][index]:.6g}."
            for index in range(size)
        ],
        f"A^(-1) = {_format_matrix(inverse)}.",
    ]
    return EMCalculatorResult(
        answer=f"A^(-1) = {_format_matrix(inverse)}.",
        quantity="A_inv",
        unit="matrix",
        steps=steps,
        values={f"c{i + 1}_{j + 1}": inverse[i][j] for i in range(size) for j in range(size)},
        matrix=tuple(tuple(row) for row in inverse),
    )


def _polar_from_rect(real: float, imag: float) -> tuple[float, float]:
    """Return magnitude and angle in degrees for a rectangular complex number."""
    magnitude = math.hypot(real, imag)
    angle_deg = math.degrees(math.atan2(imag, real))
    return magnitude, angle_deg


def calculate_phasor_rect_to_polar(
    *,
    real: float,
    imag: float,
) -> EMCalculatorResult:
    """Convert a rectangular phasor to polar form (degrees)."""
    magnitude, angle_deg = _polar_from_rect(real, imag)
    steps = [
        f"z = {real:g} + j{imag:g}.",
        f"|z| = sqrt({real:g}^2 + {imag:g}^2) = {magnitude:g}.",
        f"arg(z) = atan2({imag:g}, {real:g}) = {angle_deg:g} deg.",
    ]
    return EMCalculatorResult(
        answer=f"{magnitude:.4g} at {angle_deg:.4g} degrees.",
        quantity="phasor",
        unit="polar",
        steps=steps,
        values={
            "magnitude": magnitude,
            "phase_deg": angle_deg,
            "real": real,
            "imag": imag,
        },
    )


def calculate_phasor_polar_to_rect(
    *,
    magnitude: float,
    phase_deg: float,
) -> EMCalculatorResult:
    """Convert a polar phasor (degrees) to rectangular form."""
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")
    phase_rad = math.radians(phase_deg)
    real = magnitude * math.cos(phase_rad)
    imag = magnitude * math.sin(phase_rad)
    steps = [
        f"z = {magnitude:g} ∠ {phase_deg:g} deg.",
        f"Re = {magnitude:g} cos({phase_deg:g}) = {real:g}.",
        f"Im = {magnitude:g} sin({phase_deg:g}) = {imag:g}.",
    ]
    return EMCalculatorResult(
        answer=f"{real:.4g} + j{imag:.4g}.",
        quantity="phasor",
        unit="rectangular",
        steps=steps,
        values={
            "magnitude": magnitude,
            "phase_deg": phase_deg,
            "real": real,
            "imag": imag,
        },
    )


def calculate_phasor_add(
    *,
    magnitudes: Sequence[float],
    phases_deg: Sequence[float],
) -> EMCalculatorResult:
    """Add phasors given in polar form (sine-reference angles in degrees)."""
    if len(magnitudes) != len(phases_deg):
        raise ValueError("magnitudes and phases_deg must have the same length")
    if len(magnitudes) < 2:
        raise ValueError("phasor addition requires at least two phasors")

    total = 0j
    steps: list[str] = []
    for index, (magnitude, phase_deg) in enumerate(
        zip(magnitudes, phases_deg, strict=True), start=1
    ):
        if magnitude < 0:
            raise ValueError("magnitude must be non-negative")
        component = cmath.rect(magnitude, math.radians(phase_deg))
        total += component
        steps.append(
            f"V{index} = {magnitude:g} ∠ {phase_deg:g} deg "
            f"= {component.real:g} + j{component.imag:g}."
        )

    # Display rectangular components and amplitude to two decimal places.
    # Keep the existing presentation convention for angles near whole degrees.
    real_r = round(total.real, 2)
    imag_r = round(total.imag, 2)
    amplitude_rounded = round(math.hypot(real_r, imag_r), 2)
    if real_r == 0.0 and imag_r == 0.0:
        phase_rounded = 0.0
    else:
        phase_true = math.degrees(math.atan2(imag_r, real_r))
        if abs(phase_true - round(phase_true)) <= 0.1:
            phase_rounded = float(round(phase_true))
        else:
            phase_rounded = round(phase_true, 1)
    steps.extend(
        [
            f"Sum = {real_r:g} + j{imag_r:g}.",
            f"Amplitude = {amplitude_rounded:.2f} V.",
            f"Phase = {phase_rounded:.1f} degrees (sine reference).",
        ]
    )
    return EMCalculatorResult(
        answer=f"{amplitude_rounded:.2f} V at {phase_rounded:.1f} degrees.",
        quantity="resultant_phasor",
        unit="V",
        steps=steps,
        values={
            "amplitude_v": amplitude_rounded,
            "phase_deg": phase_rounded,
            "real": total.real,
            "imag": total.imag,
        },
    )


def calculate_phasor_magnitude_angle(
    *,
    real: float,
    imag: float,
) -> EMCalculatorResult:
    """Return magnitude and angle for a rectangular phasor."""
    return calculate_phasor_rect_to_polar(real=real, imag=imag)


def calculate_impedance_resistor(*, resistance_ohm: float) -> EMCalculatorResult:
    """Return resistor impedance Z_R = R."""
    if resistance_ohm < 0:
        raise ValueError("resistance_ohm must be non-negative")
    steps = [f"Z_R = R = {resistance_ohm:g} ohm."]
    return EMCalculatorResult(
        answer=f"Z_R = {resistance_ohm:g} ohm.",
        quantity="Z_R",
        unit="ohm",
        steps=steps,
        values={"resistance_ohm": resistance_ohm, "real": resistance_ohm, "imag": 0.0},
    )


def calculate_impedance_inductor(
    *,
    inductance_h: float,
    omega_rad_s: float,
) -> EMCalculatorResult:
    """Return inductor impedance Z_L = j ω L."""
    if inductance_h < 0:
        raise ValueError("inductance_h must be non-negative")
    if omega_rad_s < 0:
        raise ValueError("omega_rad_s must be non-negative")
    imag = omega_rad_s * inductance_h
    steps = [
        "Z_L = j ω L.",
        f"Z_L = j ({omega_rad_s:g}) ({inductance_h:g}) = j{imag:g} ohm.",
    ]
    return EMCalculatorResult(
        answer=f"Z_L = j{imag:g} ohm.",
        quantity="Z_L",
        unit="ohm",
        steps=steps,
        values={"inductance_h": inductance_h, "omega_rad_s": omega_rad_s, "imag": imag},
    )


def calculate_impedance_capacitor(
    *,
    capacitance_f: float | None = None,
    omega_rad_s: float | None = None,
) -> EMCalculatorResult:
    """Return capacitor impedance Z_C = 1/(j ω C) = -j/(ω C).

    When numeric values are omitted, return the symbolic identity used by
    formula-lookup items such as EM_NG_013.
    """
    if capacitance_f is None or omega_rad_s is None:
        latex = r"Z_C=\frac{1}{j\omega C}=-\frac{j}{\omega C}"
        steps = [
            "Ideal capacitor: I = j ω C V, so Z_C = V/I = 1/(j ω C).",
            "Rationalize: 1/(j ω C) = -j/(ω C).",
        ]
        return EMCalculatorResult(
            answer=f"{latex}.",
            quantity="Z_C",
            unit="ohm",
            steps=steps,
            values={},
        )

    if capacitance_f <= 0:
        raise ValueError("capacitance_f must be positive")
    if omega_rad_s <= 0:
        raise ValueError("omega_rad_s must be positive")
    imag = -1.0 / (omega_rad_s * capacitance_f)
    steps = [
        "Z_C = 1/(j ω C) = -j/(ω C).",
        (f"Z_C = -j / ({omega_rad_s:g} * {capacitance_f:g}) = j({imag:g}) ohm."),
    ]
    return EMCalculatorResult(
        answer=f"Z_C = j({imag:g}) ohm = -j/({omega_rad_s:g}*{capacitance_f:g}) ohm.",
        quantity="Z_C",
        unit="ohm",
        steps=steps,
        values={
            "capacitance_f": capacitance_f,
            "omega_rad_s": omega_rad_s,
            "imag": imag,
        },
    )


def calculate_impedance_series(
    *,
    impedances: Sequence[complex],
) -> EMCalculatorResult:
    """Return series combination Z = sum Z_k."""
    if not impedances:
        raise ValueError("impedances must be non-empty")
    total = sum(impedances, start=0j)
    steps = [
        "Series impedances add: Z = " + " + ".join(str(z) for z in impedances) + ".",
        f"Z = {total.real:g} + j{total.imag:g} ohm.",
    ]
    return EMCalculatorResult(
        answer=f"Z = {total.real:g} + j{total.imag:g} ohm.",
        quantity="Z_series",
        unit="ohm",
        steps=steps,
        values={"real": total.real, "imag": total.imag},
    )


def calculate_impedance_parallel(
    *,
    impedances: Sequence[complex] | None = None,
) -> EMCalculatorResult:
    """Return parallel combination, or the two-impedance formula when omitted."""
    if impedances is None:
        latex = r"Z_{total}=\frac{Z_1Z_2}{Z_1+Z_2}"
        steps = [
            "Parallel admittances add: 1/Z = 1/Z1 + 1/Z2.",
            "Therefore Z_total = Z1 Z2 / (Z1 + Z2).",
        ]
        return EMCalculatorResult(
            answer=f"{latex}.",
            quantity="Z_parallel",
            unit="ohm",
            steps=steps,
            values={},
        )
    if len(impedances) < 2:
        raise ValueError("parallel combination requires at least two impedances")
    if any(z == 0 for z in impedances):
        raise ValueError("parallel combination cannot include a zero impedance")
    admittance = sum(1 / z for z in impedances)
    if admittance == 0:
        raise ValueError("parallel admittance sum is zero")
    total = 1 / admittance
    steps = [
        "Parallel: 1/Z = " + " + ".join(f"1/({z})" for z in impedances) + ".",
        f"Z = {total.real:g} + j{total.imag:g} ohm.",
    ]
    return EMCalculatorResult(
        answer=f"Z = {total.real:g} + j{total.imag:g} ohm.",
        quantity="Z_parallel",
        unit="ohm",
        steps=steps,
        values={"real": total.real, "imag": total.imag},
    )


@dataclass(frozen=True)
class EMFormulaSpec:
    """Metadata for a registered EM calculator formula."""

    name: str
    topic: str
    description: str
    func: Callable[..., EMCalculatorResult]
    param_names: tuple[str, ...]
    quantity: str
    unit: str


FORMULA_REGISTRY: dict[str, EMFormulaSpec] = {
    "matrix_inverse_diagonal": EMFormulaSpec(
        name="matrix_inverse_diagonal",
        topic="matrices",
        description="Inverse of a square diagonal matrix with nonzero diagonal entries",
        func=calculate_matrix_inverse_diagonal,
        param_names=("matrix",),
        quantity="A_inv",
        unit="matrix",
    ),
    "matrix_multiply": EMFormulaSpec(
        name="matrix_multiply",
        topic="matrices",
        description="Multiply two matrices A and B",
        func=calculate_matrix_multiply,
        param_names=("matrix_a", "matrix_b"),
        quantity="C",
        unit="matrix",
    ),
    "matrix_determinant": EMFormulaSpec(
        name="matrix_determinant",
        topic="matrices",
        description="Determinant of a 2x2 or 3x3 matrix",
        func=calculate_matrix_determinant,
        param_names=("matrix",),
        quantity="det",
        unit="scalar",
    ),
    "matrix_inverse_2x2": EMFormulaSpec(
        name="matrix_inverse_2x2",
        topic="matrices",
        description="Inverse of a 2x2 matrix",
        func=calculate_matrix_inverse_2x2,
        param_names=("matrix",),
        quantity="A_inv",
        unit="matrix",
    ),
    "phasor_rect_to_polar": EMFormulaSpec(
        name="phasor_rect_to_polar",
        topic="phasors",
        description="Convert rectangular phasor to polar form",
        func=calculate_phasor_rect_to_polar,
        param_names=("real", "imag"),
        quantity="phasor",
        unit="polar",
    ),
    "phasor_polar_to_rect": EMFormulaSpec(
        name="phasor_polar_to_rect",
        topic="phasors",
        description="Convert polar phasor to rectangular form",
        func=calculate_phasor_polar_to_rect,
        param_names=("magnitude", "phase_deg"),
        quantity="phasor",
        unit="rectangular",
    ),
    "phasor_add": EMFormulaSpec(
        name="phasor_add",
        topic="phasors",
        description="Add sinusoidal phasors and return amplitude and phase",
        func=calculate_phasor_add,
        param_names=("magnitudes", "phases_deg"),
        quantity="resultant_phasor",
        unit="V",
    ),
    "phasor_magnitude_angle": EMFormulaSpec(
        name="phasor_magnitude_angle",
        topic="phasors",
        description="Magnitude and angle of a rectangular phasor",
        func=calculate_phasor_magnitude_angle,
        param_names=("real", "imag"),
        quantity="phasor",
        unit="polar",
    ),
    "impedance_resistor": EMFormulaSpec(
        name="impedance_resistor",
        topic="impedance",
        description="Resistor impedance Z_R = R",
        func=calculate_impedance_resistor,
        param_names=("resistance_ohm",),
        quantity="Z_R",
        unit="ohm",
    ),
    "impedance_inductor": EMFormulaSpec(
        name="impedance_inductor",
        topic="impedance",
        description="Inductor impedance Z_L = j ω L",
        func=calculate_impedance_inductor,
        param_names=("inductance_h", "omega_rad_s"),
        quantity="Z_L",
        unit="ohm",
    ),
    "impedance_capacitor": EMFormulaSpec(
        name="impedance_capacitor",
        topic="impedance",
        description="Capacitor impedance Z_C = 1/(j ω C) = -j/(ω C)",
        func=calculate_impedance_capacitor,
        param_names=("capacitance_f", "omega_rad_s"),
        quantity="Z_C",
        unit="ohm",
    ),
    "impedance_series": EMFormulaSpec(
        name="impedance_series",
        topic="impedance",
        description="Series combination of impedances",
        func=calculate_impedance_series,
        param_names=("impedances",),
        quantity="Z_series",
        unit="ohm",
    ),
    "impedance_parallel": EMFormulaSpec(
        name="impedance_parallel",
        topic="impedance",
        description="Parallel combination Z1 Z2/(Z1+Z2)",
        func=calculate_impedance_parallel,
        param_names=("impedances",),
        quantity="Z_parallel",
        unit="ohm",
    ),
}


_MATRIX_LITERAL_RE = re.compile(
    r"\[(?:\s*\[(?:\s*[-+]?\d*\.?\d+(?:\s*,\s*[-+]?\d*\.?\d+)*\s*)\]"
    r"(?:\s*,\s*\[(?:\s*[-+]?\d*\.?\d+(?:\s*,\s*[-+]?\d*\.?\d+)*\s*)\])+\s*)\]"
)
_SIN_PHASOR_RE = re.compile(
    r"(?P<mag>\d+(?:\.\d+)?)\s*sin\s*\(\s*omega\s*t\s*"
    r"(?P<sign>[+-])\s*(?P<phase>\d+(?:\.\d+)?)\s*degrees?\s*\)",
    re.IGNORECASE,
)
_COS_PHASOR_RE = re.compile(
    r"(?P<mag>\d+(?:\.\d+)?)\s*cos\s*\(\s*omega\s*t\s*"
    r"(?P<sign>[+-])\s*(?P<phase>\d+(?:\.\d+)?)\s*degrees?\s*\)",
    re.IGNORECASE,
)


def parse_matrix_literals(text: str) -> list[list[list[float]]]:
    """Extract matrix literals such as ``[[1,2,3],[0,1,2]]`` from text."""
    matrices: list[list[list[float]]] = []
    for match in _MATRIX_LITERAL_RE.finditer(text):
        literal = match.group(0)
        rows: list[list[float]] = []
        for row_match in re.finditer(r"\[([^\[\]]+)\]", literal[1:-1]):
            values = [float(part.strip()) for part in row_match.group(1).split(",")]
            rows.append(values)
        if rows:
            matrices.append(rows)
    return matrices


def parse_sinusoidal_phasors(text: str) -> tuple[list[float], list[float]]:
    """Extract sine/cosine phasor magnitudes and phases (degrees) from text."""
    magnitudes: list[float] = []
    phases: list[float] = []
    for match in _SIN_PHASOR_RE.finditer(text):
        magnitude = float(match.group("mag"))
        phase = float(match.group("phase"))
        if match.group("sign") == "-":
            phase = -phase
        magnitudes.append(magnitude)
        phases.append(phase)
    for match in _COS_PHASOR_RE.finditer(text):
        # cos(ωt + φ) = sin(ωt + φ + 90°)
        magnitude = float(match.group("mag"))
        phase = float(match.group("phase"))
        if match.group("sign") == "-":
            phase = -phase
        magnitudes.append(magnitude)
        phases.append(phase + 90.0)
    return magnitudes, phases


def extract_em_variables(formula_name: str, question: str) -> dict[str, Any] | None:
    """Deterministically extract calculator variables from an EM question.

    Returns ``None`` when the question does not contain enough structured
    values for the named formula. Formula-only impedance routes may return an
    empty dict.
    """
    if formula_name == "matrix_multiply":
        matrices = parse_matrix_literals(question)
        if len(matrices) >= 2:
            return {"matrix_a": matrices[0], "matrix_b": matrices[1]}
        return None

    if formula_name in {"matrix_determinant", "matrix_inverse_2x2"}:
        matrices = parse_matrix_literals(question)
        if matrices:
            return {"matrix": matrices[0]}
        return None

    if formula_name == "matrix_inverse_diagonal":
        matrices = parse_matrix_literals(question)
        return {"matrix": matrices[0]} if len(matrices) == 1 else None

    if formula_name == "phasor_add":
        magnitudes, phases = parse_sinusoidal_phasors(question)
        if len(magnitudes) >= 2:
            return {"magnitudes": magnitudes, "phases_deg": phases}
        return None

    if formula_name in {"phasor_rect_to_polar", "phasor_magnitude_angle"}:
        match = re.search(
            r"(?P<real>[-+]?\d+(?:\.\d+)?)\s*\+\s*j\s*(?P<imag>[-+]?\d+(?:\.\d+)?)",
            question,
            re.IGNORECASE,
        )
        if match:
            return {
                "real": float(match.group("real")),
                "imag": float(match.group("imag")),
            }
        return None

    if formula_name == "phasor_polar_to_rect":
        match = re.search(
            r"(?P<mag>\d+(?:\.\d+)?)\s*(?:∠|at)\s*(?P<phase>[-+]?\d+(?:\.\d+)?)\s*deg",
            question,
            re.IGNORECASE,
        )
        if match:
            return {
                "magnitude": float(match.group("mag")),
                "phase_deg": float(match.group("phase")),
            }
        return None

    if formula_name == "impedance_capacitor":
        # Symbolic formula questions need no numeric values.
        if re.search(r"\d", question) is None:
            return {}
        cap = re.search(
            r"(?P<c>\d+(?:\.\d+)?)\s*(?:f|farad)",
            question,
            re.IGNORECASE,
        )
        omega = re.search(
            r"(?:omega|ω)\s*=\s*(?P<w>\d+(?:\.\d+)?)",
            question,
            re.IGNORECASE,
        )
        if cap and omega:
            return {
                "capacitance_f": float(cap.group("c")),
                "omega_rad_s": float(omega.group("w")),
            }
        # Formula-lookup style still works with empty kwargs.
        return {}

    if formula_name == "impedance_inductor":
        ind = re.search(
            r"(?P<l>\d+(?:\.\d+)?)\s*(?:h|henry)",
            question,
            re.IGNORECASE,
        )
        omega = re.search(
            r"(?:omega|ω)\s*=\s*(?P<w>\d+(?:\.\d+)?)",
            question,
            re.IGNORECASE,
        )
        if ind and omega:
            return {
                "inductance_h": float(ind.group("l")),
                "omega_rad_s": float(omega.group("w")),
            }
        return None

    if formula_name == "impedance_resistor":
        res = re.search(
            r"(?P<r>\d+(?:\.\d+)?)\s*(?:ohm|Ω)",
            question,
            re.IGNORECASE,
        )
        if res:
            return {"resistance_ohm": float(res.group("r"))}
        return None

    if formula_name in {"impedance_series", "impedance_parallel"}:
        # Formula-only parallel/series statement.
        if "formula" in question.casefold() or "state the" in question.casefold():
            return {}
        return None

    return None


def dispatch_formula(formula_name: str, **params: Any) -> EMCalculatorResult:
    """Look up an EM formula by name and call it with the given parameters."""
    spec = FORMULA_REGISTRY[formula_name]
    return spec.func(**params)


def format_result_with_steps(result: EMCalculatorResult) -> str:
    """Format calculator answer including brief working steps."""
    if not result.steps:
        return result.answer
    working = " ".join(result.steps)
    return f"{result.answer} Working: {working}"


def schema_for_formula(formula_name: str) -> Mapping[str, str]:
    """Return a human-readable variable schema for the extractor prompt."""
    spec = FORMULA_REGISTRY[formula_name]
    descriptions = {
        "matrix_a": "Left-hand matrix A as a nested list of numbers",
        "matrix_b": "Right-hand matrix B as a nested list of numbers",
        "matrix": "Square matrix as a nested list of numbers",
        "real": "Real part of a rectangular phasor",
        "imag": "Imaginary part of a rectangular phasor",
        "magnitude": "Phasor magnitude",
        "phase_deg": "Phasor phase angle in degrees",
        "magnitudes": "List of phasor magnitudes",
        "phases_deg": "List of phasor phases in degrees",
        "resistance_ohm": "Resistance in ohms",
        "inductance_h": "Inductance in henries",
        "capacitance_f": "Capacitance in farads",
        "omega_rad_s": "Angular frequency in rad/s",
        "impedances": "List of complex impedances",
    }
    return {name: descriptions.get(name, name) for name in spec.param_names}
