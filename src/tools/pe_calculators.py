"""Deterministic calculators for PE transformer and AC power questions."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PECalculatorResult:
    """Structured PE calculator result with a ready-to-grade answer string."""

    answer: str
    quantity: str
    unit: str
    steps: list[str]
    values: dict[str, float] = field(default_factory=dict)


def _require_positive(name: str, value: float) -> None:
    """Raise when a calculator input is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(name: str, value: float) -> None:
    """Raise when a calculator input is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_power_factor(value: float) -> None:
    """Raise when a power factor is outside the closed interval [-1, 1]."""
    if not -1.0 <= value <= 1.0:
        raise ValueError("power_factor must be between -1 and 1")


def _format_complex_power(real_power_w: float, reactive_power_var: float) -> str:
    """Render ``S = P + jQ`` with an explicit minus sign for negative Q."""
    sign = "+" if reactive_power_var >= 0 else "-"
    return f"S = {real_power_w:.6g} {sign} j{abs(reactive_power_var):.6g} VA"


def calculate_ideal_transformer_turns_ratio(
    *,
    primary_turns: float,
    secondary_turns: float,
) -> PECalculatorResult:
    """Compute the ideal transformer turns ratio ``a = N1/N2 = V1/V2``.

    Grounded in the ideal-transformer relationships card in PE_Core_07:
    ``V1/V2 = a = N1/N2`` and ``I1/I2 = 1/a``.
    """
    _require_positive("primary_turns", primary_turns)
    _require_positive("secondary_turns", secondary_turns)
    turns_ratio = primary_turns / secondary_turns
    steps = [
        "Ideal transformer: V1/V2 = a = N1/N2.",
        f"a = {primary_turns:g} / {secondary_turns:g} = {turns_ratio:g}.",
        f"Current ratio I1/I2 = 1/a = {1 / turns_ratio:g}.",
    ]
    return PECalculatorResult(
        answer=f"a = {turns_ratio:.6g}.",
        quantity="turns ratio",
        unit="ratio",
        steps=steps,
        values={
            "turns_ratio": turns_ratio,
            "voltage_ratio": turns_ratio,
            "current_ratio": 1 / turns_ratio,
        },
    )


def calculate_transformer_reflected_impedance(
    *,
    turns_ratio: float,
    load_impedance_ohm: float,
) -> PECalculatorResult:
    """Compute the impedance reflected to the primary, ``Z_in = a² Z_L``."""
    _require_positive("turns_ratio", turns_ratio)
    _require_positive("load_impedance_ohm", load_impedance_ohm)
    reflected = turns_ratio * turns_ratio * load_impedance_ohm
    steps = [
        "Ideal transformer reflected impedance: Z_in = a² Z_L.",
        f"Z_in = {turns_ratio:g}² * {load_impedance_ohm:g} = {reflected:g} ohm.",
    ]
    return PECalculatorResult(
        answer=f"Z_in = {reflected:.6g} ohm.",
        quantity="reflected impedance",
        unit="ohm",
        steps=steps,
        values={"reflected_impedance_ohm": reflected},
    )


def calculate_single_phase_complex_power(
    *,
    voltage_v: float,
    current_a: float,
    phase_deg: float | None = None,
    power_factor: float | None = None,
    reactive_sign: float = 1.0,
) -> PECalculatorResult:
    """Compute complex, apparent, real and reactive power for one phase.

    Uses the PE_Core_03 relations ``S = V I* = P + jQ``, ``|S| = V I``,
    ``P = |S| cos φ``, ``Q = |S| sin φ`` and ``pf = P/|S|`` with RMS phasors.
    ``phase_deg`` is the voltage-current phase difference; alternatively a
    power factor may be supplied, in which case the reactive sign follows
    ``reactive_sign`` (+1 lagging, -1 leading) only when the question states it.
    """
    _require_nonnegative("voltage_v", voltage_v)
    _require_nonnegative("current_a", current_a)
    if phase_deg is None and power_factor is None:
        raise ValueError("provide phase_deg or power_factor")
    if (
        phase_deg is not None
        and power_factor is not None
        and abs(math.cos(math.radians(phase_deg)) - power_factor) > 1e-9
    ):
        raise ValueError("phase_deg and power_factor are inconsistent")

    apparent_va = voltage_v * current_a
    if phase_deg is not None:
        angle_rad = math.radians(phase_deg)
        effective_power_factor = math.cos(angle_rad)
        real_power_w = apparent_va * effective_power_factor
        reactive_power_var = apparent_va * math.sin(angle_rad)
        steps = [
            "Single-phase complex power: S = V I* = P + jQ.",
            f"|S| = V I = {voltage_v:g} * {current_a:g} = {apparent_va:g} VA.",
            f"P = |S| cos({phase_deg:g}°) = {real_power_w:g} W.",
            f"Q = |S| sin({phase_deg:g}°) = {reactive_power_var:g} var.",
            f"pf = P/|S| = cos({phase_deg:g}°) = {effective_power_factor:g}.",
        ]
    else:
        assert power_factor is not None
        _require_power_factor(power_factor)
        effective_power_factor = power_factor
        real_power_w = apparent_va * power_factor
        reactive_magnitude_var = math.sqrt(
            max(0.0, apparent_va * apparent_va - real_power_w * real_power_w)
        )
        reactive_power_var = reactive_sign * reactive_magnitude_var
        sign_note = "lagging" if reactive_sign >= 0 else "leading"
        steps = [
            "Single-phase power: |S| = V I and pf = P/|S|.",
            f"|S| = V I = {voltage_v:g} * {current_a:g} = {apparent_va:g} VA.",
            f"P = |S| pf = {apparent_va:g} * {power_factor:g} = {real_power_w:g} W.",
            (
                f"Q magnitude = sqrt(|S|² − P²) = {reactive_magnitude_var:g} var "
                f"(signed {sign_note} per the stated current direction)."
            ),
        ]

    return PECalculatorResult(
        answer=(
            f"{_format_complex_power(real_power_w, reactive_power_var)}; "
            f"|S| = {apparent_va:.6g} VA; P = {real_power_w:.6g} W; "
            f"Q = {reactive_power_var:.6g} var; pf = {effective_power_factor:.6g}."
        ),
        quantity="complex power",
        unit="VA",
        steps=steps,
        values={
            "apparent_power_va": apparent_va,
            "real_power_w": real_power_w,
            "reactive_power_var": reactive_power_var,
            "power_factor": effective_power_factor,
        },
    )


def calculate_power_from_real_reactive(
    *,
    real_power_w: float,
    reactive_power_var: float,
) -> PECalculatorResult:
    """Compute apparent power and power factor from real and reactive power.

    Uses ``|S| = sqrt(P² + Q²)`` and ``pf = P/|S|`` from PE_Core_03.
    """
    _require_nonnegative("real_power_w", real_power_w)
    apparent_va = math.hypot(real_power_w, reactive_power_var)
    if apparent_va == 0:
        raise ValueError("real and reactive power cannot both be zero")
    power_factor = real_power_w / apparent_va
    phase_deg = math.degrees(math.atan2(reactive_power_var, real_power_w))
    steps = [
        "Apparent power: |S| = sqrt(P² + Q²).",
        f"|S| = sqrt({real_power_w:g}² + {reactive_power_var:g}²) = {apparent_va:g} VA.",
        f"pf = P/|S| = {real_power_w:g} / {apparent_va:g} = {power_factor:g}.",
    ]
    return PECalculatorResult(
        answer=f"|S| = {apparent_va:.6g} VA; pf = {power_factor:.6g}.",
        quantity="apparent power and power factor",
        unit="VA",
        steps=steps,
        values={
            "apparent_power_va": apparent_va,
            "power_factor": power_factor,
            "phase_deg": phase_deg,
        },
    )


def calculate_wind_power_cube_scaling(*, speed_factor: float) -> PECalculatorResult:
    """Return the wind-power scale factor for a wind-speed multiplier.

    Grounded in PE_Core_02: ``P_wind = (1/2) ρ A v³``, so with ρ and A fixed,
    power scales with the cube of wind speed.
    """
    _require_nonnegative("speed_factor", speed_factor)
    power_factor = speed_factor**3
    steps = [
        "Available wind power P_wind = (1/2) ρ A v³.",
        "With ρ and A fixed, P_wind scales with the cube of wind speed.",
        f"Speed factor {speed_factor:g} gives power factor {speed_factor:g}³ = {power_factor:g}.",
    ]
    return PECalculatorResult(
        answer=f"Available wind power increases by a factor of {power_factor:.6g}.",
        quantity="wind power scale factor",
        unit="ratio",
        steps=steps,
        values={"power_factor": power_factor, "speed_factor": speed_factor},
    )


def calculate_rc_time_constant(
    *,
    resistance_ohm: float,
    capacitance_f: float,
) -> PECalculatorResult:
    """Compute the first-order RC time constant ``τ = R_eq C``.

    Also reports the fraction of the initial deviation remaining after one
    time constant (e^{-1} ≈ 0.368), matching PE_Core_04.
    """
    _require_positive("resistance_ohm", resistance_ohm)
    _require_positive("capacitance_f", capacitance_f)
    time_constant_s = resistance_ohm * capacitance_f
    fraction_remaining = math.exp(-1.0)
    steps = [
        "First-order RC time constant: τ = R_eq C.",
        f"τ = {resistance_ohm:g} * {capacitance_f:g} = {time_constant_s:g} s.",
        f"After one time constant, exp(-1) = {fraction_remaining:.4g} of the deviation remains.",
    ]
    return PECalculatorResult(
        answer=f"τ = {time_constant_s:.6g} s.",
        quantity="time constant",
        unit="s",
        steps=steps,
        values={
            "time_constant_s": time_constant_s,
            "fraction_remaining_one_tau": fraction_remaining,
        },
    )


def calculate_three_phase_power(
    *,
    line_voltage_v: float,
    line_current_a: float,
    power_factor: float,
) -> PECalculatorResult:
    """Compute total three-phase real and apparent power from line quantities.

    Uses ``P = √3 V_L I_L cos φ`` and ``|S| = √3 V_L I_L`` from PE_Core_05.
    The reactive-power sign is not determined by the power-factor value alone,
    so only its magnitude is reported.
    """
    _require_nonnegative("line_voltage_v", line_voltage_v)
    _require_nonnegative("line_current_a", line_current_a)
    _require_power_factor(power_factor)
    apparent_va = math.sqrt(3) * line_voltage_v * line_current_a
    real_power_w = apparent_va * power_factor
    reactive_magnitude_var = math.sqrt(max(0.0, apparent_va**2 - real_power_w**2))
    steps = [
        (
            rf"Use RMS line quantities: \(V_L = {line_voltage_v:g}\,\mathrm{{V}}\), "
            rf"\(I_L = {line_current_a:g}\,\mathrm{{A}}\), "
            rf"and power factor \(\cos\varphi = {power_factor:g}\)."
        ),
        (
            "For a balanced three-phase load, total real power is "
            r"\(P = \sqrt{3} V_L I_L \cos\varphi\)."
        ),
        (
            rf"First find total apparent power: \(|S| = \sqrt{{3}} V_L I_L "
            rf"= \sqrt{{3}}\times {line_voltage_v:g}\times {line_current_a:g} "
            rf"\approx {apparent_va:.6g}\,\mathrm{{VA}}\)."
        ),
        (
            rf"Apply the power factor: \(P = |S|\cos\varphi "
            rf"= \sqrt{{3}}\times {line_voltage_v:g}\times {line_current_a:g}"
            rf"\times {power_factor:g} \approx {real_power_w:.6g}\,\mathrm{{W}}\)."
        ),
        (
            rf"Therefore, total real power is \(P \approx {real_power_w:.6g}\,\mathrm{{W}} "
            rf"\approx {real_power_w / 1000:.4g}\,\mathrm{{kW}}\). "
            "Calculations use unrounded values; displayed powers are rounded."
        ),
    ]
    return PECalculatorResult(
        answer=f"P = {real_power_w:.6g} W.",
        quantity="three-phase real power",
        unit="W",
        steps=steps,
        values={
            "real_power_w": real_power_w,
            "apparent_power_va": apparent_va,
            "reactive_power_magnitude_var": reactive_magnitude_var,
        },
    )


def calculate_synchronous_speed(
    *,
    frequency_hz: float,
    poles: float,
) -> PECalculatorResult:
    """Compute induction-motor synchronous speed ``n_s = 120 f / P`` in r/min."""
    _require_positive("frequency_hz", frequency_hz)
    _require_positive("poles", poles)
    if not math.isfinite(frequency_hz) or not math.isfinite(poles):
        raise ValueError("frequency and poles must be finite")
    if not float(poles).is_integer() or int(poles) % 2:
        raise ValueError("total poles must be a positive even integer, not pole pairs")
    synchronous_speed_rpm = 120 * frequency_hz / poles
    steps = [
        "Synchronous speed: n_s = 120 f / P, where P is the total number of stator poles.",
        f"n_s = 120 * {frequency_hz:g} / {poles:g} = {synchronous_speed_rpm:g} r/min.",
    ]
    return PECalculatorResult(
        answer=f"n_s = {synchronous_speed_rpm:.6g} r/min.",
        quantity="synchronous speed",
        unit="r/min",
        steps=steps,
        values={"synchronous_speed_rpm": synchronous_speed_rpm},
    )


def calculate_phase_real_power(
    *,
    phase_voltages_v: Sequence[float],
    phase_currents_a: Sequence[float],
    phase_power_factors: Sequence[float],
) -> PECalculatorResult:
    """Sum real power using actual RMS load-phase quantities, balanced or unbalanced."""
    if any(
        len(values) != 3 for values in (phase_voltages_v, phase_currents_a, phase_power_factors)
    ):
        raise ValueError("three explicit phase voltages, currents and power factors are required")
    powers: list[float] = []
    steps: list[str] = [
        "Use actual RMS load-phase voltages and currents with each phase's stated power factor; "
        "no balanced-load or line-to-phase conversion is assumed.",
        "Total real power P = Σ V_phase,rms * I_phase,rms * pf_phase for the three phases.",
    ]
    for index, (voltage, current, power_factor) in enumerate(
        zip(phase_voltages_v, phase_currents_a, phase_power_factors, strict=True), start=1
    ):
        if not all(math.isfinite(value) for value in (voltage, current, power_factor)):
            raise ValueError("phase quantities must be finite")
        _require_nonnegative("phase_voltage_v", voltage)
        _require_nonnegative("phase_current_a", current)
        _require_power_factor(power_factor)
        power: float = voltage * current * power_factor
        if not math.isfinite(power):
            raise ValueError("phase power result must be finite")
        powers.append(power)
        steps.append(
            f"Phase {index}: P_{index} = {voltage:g} V * {current:g} A * {power_factor:g} "
            f"= {power:g} W."
        )
    total: float = sum(powers)
    if not math.isfinite(total):
        raise ValueError("total phase power result must be finite")
    steps.append(f"P = {' + '.join(f'{power:g}' for power in powers)} = {total:g} W.")
    return PECalculatorResult(
        answer=f"P_total = {total:.6g} W.",
        quantity="three-phase real power",
        unit="W",
        steps=steps,
        values={
            "real_power_w": total,
            **{f"phase_{index}_real_power_w": power for index, power in enumerate(powers, 1)},
        },
    )


@dataclass(frozen=True)
class PEFormulaSpec:
    """Metadata for a registered PE calculator formula."""

    name: str
    topic: str
    description: str
    func: Callable[..., PECalculatorResult]
    param_names: tuple[str, ...]
    quantity: str
    unit: str


FORMULA_REGISTRY: dict[str, PEFormulaSpec] = {
    "phase_real_power": PEFormulaSpec(
        name="phase_real_power",
        topic="three_phase",
        description="Total real power from three explicit actual RMS load-phase quantities",
        func=calculate_phase_real_power,
        param_names=("phase_voltages_v", "phase_currents_a", "phase_power_factors"),
        quantity="three-phase real power",
        unit="W",
    ),
    "ideal_transformer_turns_ratio": PEFormulaSpec(
        name="ideal_transformer_turns_ratio",
        topic="transformers",
        description="Ideal transformer turns ratio a = N1/N2 = V1/V2",
        func=calculate_ideal_transformer_turns_ratio,
        param_names=("primary_turns", "secondary_turns"),
        quantity="turns ratio",
        unit="ratio",
    ),
    "transformer_reflected_impedance": PEFormulaSpec(
        name="transformer_reflected_impedance",
        topic="transformers",
        description="Ideal transformer reflected input impedance Z_in = a² Z_L",
        func=calculate_transformer_reflected_impedance,
        param_names=("turns_ratio", "load_impedance_ohm"),
        quantity="reflected impedance",
        unit="ohm",
    ),
    "single_phase_complex_power": PEFormulaSpec(
        name="single_phase_complex_power",
        topic="ac_power",
        description="Single-phase complex, apparent, real and reactive power and power factor",
        func=calculate_single_phase_complex_power,
        param_names=("voltage_v", "current_a", "phase_deg", "power_factor", "reactive_sign"),
        quantity="complex power",
        unit="VA",
    ),
    "power_from_real_reactive": PEFormulaSpec(
        name="power_from_real_reactive",
        topic="ac_power",
        description="Apparent power and power factor from real and reactive power",
        func=calculate_power_from_real_reactive,
        param_names=("real_power_w", "reactive_power_var"),
        quantity="apparent power and power factor",
        unit="VA",
    ),
    "wind_power_cube_scaling": PEFormulaSpec(
        name="wind_power_cube_scaling",
        topic="renewables",
        description="Wind-power scale factor for a wind-speed multiplier",
        func=calculate_wind_power_cube_scaling,
        param_names=("speed_factor",),
        quantity="wind power scale factor",
        unit="ratio",
    ),
    "rc_time_constant": PEFormulaSpec(
        name="rc_time_constant",
        topic="transients",
        description="First-order RC time constant τ = R_eq C",
        func=calculate_rc_time_constant,
        param_names=("resistance_ohm", "capacitance_f"),
        quantity="time constant",
        unit="s",
    ),
    "three_phase_power": PEFormulaSpec(
        name="three_phase_power",
        topic="three_phase",
        description="Balanced three-phase real power P = √3 V_L I_L cos φ",
        func=calculate_three_phase_power,
        param_names=("line_voltage_v", "line_current_a", "power_factor"),
        quantity="three-phase real power",
        unit="W",
    ),
    "synchronous_speed": PEFormulaSpec(
        name="synchronous_speed",
        topic="machines",
        description="Induction-motor synchronous speed n_s = 120 f / P",
        func=calculate_synchronous_speed,
        param_names=("frequency_hz", "poles"),
        quantity="synchronous speed",
        unit="r/min",
    ),
}


_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_TURNS_PRIMARY_RE = re.compile(
    r"(?:N1|N_1|n1|primary(?: turns)?)\s*=\s*(?P<value>" + _NUMBER + r")",
    re.IGNORECASE,
)
_TURNS_SECONDARY_RE = re.compile(
    r"(?:N2|N_2|n2|secondary(?: turns)?)\s*=\s*(?P<value>" + _NUMBER + r")",
    re.IGNORECASE,
)
_VOLTAGE_PRIMARY_RE = re.compile(
    r"(?:V1|V_1|primary voltage)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*v",
    re.IGNORECASE,
)
_VOLTAGE_SECONDARY_RE = re.compile(
    r"(?:V2|V_2|secondary voltage)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*v",
    re.IGNORECASE,
)
_CURRENT_PRIMARY_RE = re.compile(
    r"(?:I1|I_1|primary current)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*a",
    re.IGNORECASE,
)
_CURRENT_SECONDARY_RE = re.compile(
    r"(?:I2|I_2|secondary current)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*a",
    re.IGNORECASE,
)
_TURNS_RATIO_RE = re.compile(
    r"(?:a|turns? ratio|turn ratio)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")",
    re.IGNORECASE,
)
_LOAD_IMPEDANCE_RE = re.compile(
    r"(?:Z[Ll]?_?|load|load impedance)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*(?:ohm|Ω)",
    re.IGNORECASE,
)
_PHASE_DEG_RE = re.compile(
    (
        r"(?:φ|phi|phase(?: difference| angle)?)\s*(?:=|\s+of\s+)?"
        r"\s*(?P<value>" + _NUMBER + r")\s*(?:°|degrees?|deg)"
    ),
    re.IGNORECASE,
)
_POWER_FACTOR_RE = re.compile(
    r"(?:power factor|pf)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")",
    re.IGNORECASE,
)
_REAL_POWER_RE = re.compile(
    r"(?:P|real power)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*w",
    re.IGNORECASE,
)
_REACTIVE_POWER_RE = re.compile(
    r"(?:Q|reactive power)\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*(?:var|vAr|VAR)",
    re.IGNORECASE,
)
_SPEED_FACTOR_RE = re.compile(
    r"(?P<value>" + _NUMBER + r")\s*(?:times|×|x)\s*(?:faster|as fast|the wind speed)",
    re.IGNORECASE,
)
_WIND_DOUBLE_RE = re.compile(r"\bdouble[sd]?\b", re.IGNORECASE)
_WIND_TRIPLE_RE = re.compile(r"\btriple[sd]?\b", re.IGNORECASE)
_WIND_QUADRUPLE_RE = re.compile(r"\bquadruple[sd]?\b", re.IGNORECASE)
_RESISTANCE_SI_RE = re.compile(
    r"(?P<value>" + _NUMBER + r")\s*(?P<prefix>[kKmMµuU]?)\s*(?:Ω|ohm|ohms)",
    re.IGNORECASE,
)
_CAPACITANCE_SI_RE = re.compile(
    r"(?P<value>" + _NUMBER + r")\s*(?P<prefix>[kKmMµuUnNpP]?)\s*(?:f|farad|farads)",
    re.IGNORECASE,
)
_LINE_VOLTAGE_RE = re.compile(
    (
        r"(?:line[- ]to[- ]line|line)\s*(?:voltage)?\s*(?:=|\s+of\s+)?"
        r"\s*(?P<value>" + _NUMBER + r")\s*v"
    ),
    re.IGNORECASE,
)
_LINE_CURRENT_RE = re.compile(
    r"line\s*current\s*(?:=|\s+of\s+)?\s*(?P<value>" + _NUMBER + r")\s*a",
    re.IGNORECASE,
)
_FREQUENCY_HZ_RE = re.compile(
    r"(?P<value>" + _NUMBER + r")\s*hz",
    re.IGNORECASE,
)
_POLE_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_POLES_RE = re.compile(
    rf"(?<![\w.-])(?P<value>{_NUMBER}|{'|'.join(_POLE_WORDS)})[\s-]+"
    r"(?:stator[\s-]+)?(?P<kind>poles?(?:[\s-]+pairs?)?)\b",
    re.IGNORECASE,
)
_UNBALANCED_RE: re.Pattern[str] = re.compile(
    r"\b(?:unbalanced|non[- ]balanced|not\s+balanced)\b", re.IGNORECASE
)
_PHASE_NUMBER: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ANY_NUMBER_RE = re.compile(_NUMBER)
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

_SI_PREFIX_FACTOR: dict[str, float] = {
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "m": 1e-3,
    "M": 1e6,
    "u": 1e-6,
    "U": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "N": 1e-9,
    "p": 1e-12,
    "P": 1e12,
}


def _match_value(match: re.Match[str] | None) -> float | None:
    """Return the numeric value captured by ``match``."""
    return float(match.group("value")) if match is not None else None


def _extract_rms_quantity(question: str, label: str, unit: str) -> float | None:
    """Extract a labeled RMS quantity, falling back to a bare number+unit."""
    labeled = re.search(
        rf"{label}\s*(?:=|\s+of\s+)?\s*(?P<value>{_NUMBER})\s*{unit}",
        question,
        re.IGNORECASE,
    )
    if labeled is not None:
        return float(labeled.group("value"))
    trailing = re.search(rf"(?P<value>{_NUMBER})\s*{unit}", question, re.IGNORECASE)
    if trailing is not None:
        return float(trailing.group("value"))
    return None


def _extract_speed_factor(question: str) -> float | None:
    """Return a wind-speed multiplier from explicit wording or number words."""
    match = _SPEED_FACTOR_RE.search(question)
    if match is not None:
        return float(match.group("value"))
    normalized = question.casefold()
    if _WIND_QUADRUPLE_RE.search(normalized):
        return 4.0
    if _WIND_TRIPLE_RE.search(normalized):
        return 3.0
    if _WIND_DOUBLE_RE.search(normalized):
        return 2.0
    return None


def _extract_si_quantity(
    question: str,
    pattern: re.Pattern[str],
) -> float | None:
    """Extract a metric-prefixed quantity (e.g. 2.0 kΩ → 2000.0)."""
    match = pattern.search(question)
    if match is None:
        return None
    value = float(match.group("value"))
    prefix = match.group("prefix")
    return value * _SI_PREFIX_FACTOR.get(prefix, 1.0)


def _extract_poles(question: str) -> float | None:
    """Read explicit total poles or pole pairs, rejecting inconsistent or odd totals."""
    normalized: str = question.casefold().replace("‑", "-").replace("–", "-").replace("−", "-")
    totals: set[float] = set()
    for match in _POLES_RE.finditer(normalized):
        token: str = match.group("value")
        count: float = float(_POLE_WORDS[token]) if token in _POLE_WORDS else float(token)
        total: float = count * (2 if "pair" in match.group("kind") else 1)
        if not math.isfinite(total) or total <= 0 or not total.is_integer() or int(total) % 2:
            return None
        totals.add(total)
    return totals.pop() if len(totals) == 1 else None


def _extract_phase_sequence(question: str, quantity: str) -> list[float] | None:
    """Read only explicitly labelled phase triplets or an explicit value for every phase."""
    unit: str = {"voltage": "V", "current": "A", "power_factor": ""}[quantity]
    label: str = (
        r"phase\s+(?:power\s+factors|pf)" if quantity == "power_factor" else rf"phase\s+{quantity}s"
    )
    token: str = rf"{_PHASE_NUMBER}(?:\s*{unit})?" if unit else _PHASE_NUMBER
    separator: str = r"\s*(?:,\s*(?:and\s+)?|and\s+)"
    triple: str = rf"{token}{separator}{token}{separator}{token}"
    pattern: re.Pattern[str] = re.compile(
        rf"(?P<leading_rms>rms\s+)?(?:actual\s+)?{label}\s*(?:are|=|of|:)?\s*"
        rf"(?P<values>\[\s*{triple}\s*\]|{triple})"
        rf"\s*(?P<unit>{unit or '(?!)'})?\s*(?P<trailing_rms>rms)?"
        rf"(?![\w\]]|\.\d|\s*,\s*{_PHASE_NUMBER})",
        re.IGNORECASE,
    )
    candidates: list[list[float]] = []
    for match in pattern.finditer(question):
        raw: str = match.group("values")
        if unit and (
            not (match.group("leading_rms") or match.group("trailing_rms"))
            or not (match.group("unit") or re.search(rf"\d\s*{unit}\b", raw, re.I))
        ):
            continue
        candidates.append([float(value) for value in re.findall(_PHASE_NUMBER, raw)])
    if quantity == "power_factor":
        shared: re.Pattern[str] = re.compile(
            rf"\beach\s+phase\s+(?:has\s+)?(?:a\s+)?"
            rf"(?:(?P<unity>unity)\s+power\s+factor|power\s+factor\s*(?:is|of|=)?\s*"
            rf"(?P<value>{_PHASE_NUMBER}))(?!\w|\.\d)",
            re.IGNORECASE,
        )
        for match in shared.finditer(question):
            candidates.append([1.0 if match.group("unity") else float(match.group("value"))] * 3)
    else:
        shared = re.compile(
            rf"\b(?:each\s+phase(?:'s)?\s+(?P<leading_rms>rms\s+)?{quantity}|"
            rf"all\s+phase\s+{quantity}s)\s*(?:is|are|of|=)?\s*"
            rf"(?P<value>{_PHASE_NUMBER})\s*{unit}\s*(?P<trailing_rms>rms)?(?!\w|\.\d)",
            re.IGNORECASE,
        )
        for match in shared.finditer(question):
            if match.group("leading_rms") or match.group("trailing_rms"):
                candidates.append([float(match.group("value"))] * 3)
    if not candidates or any(values != candidates[0] for values in candidates[1:]):
        return None
    return candidates[0]


def extract_pe_variables(formula_name: str, question: str) -> dict[str, Any] | None:
    """Deterministically extract calculator variables from a PE question.

    Returns ``None`` when the question lacks the structured values required by
    the named formula; formula-lookup and conceptual questions fall through to
    the grounded RAG path.
    """
    if formula_name == "ideal_transformer_turns_ratio":
        primary = _match_value(_TURNS_PRIMARY_RE.search(question))
        secondary = _match_value(_TURNS_SECONDARY_RE.search(question))
        if primary is None or secondary is None:
            primary_voltage = _match_value(_VOLTAGE_PRIMARY_RE.search(question))
            secondary_voltage = _match_value(_VOLTAGE_SECONDARY_RE.search(question))
            if primary_voltage is None or secondary_voltage is None:
                primary_current = _match_value(_CURRENT_PRIMARY_RE.search(question))
                secondary_current = _match_value(_CURRENT_SECONDARY_RE.search(question))
                if primary_current is None or secondary_current is None:
                    return None
                # I1/I2 = 1/a, so a = I2/I1.
                return {
                    "primary_turns": secondary_current,
                    "secondary_turns": primary_current,
                }
            # V1/V2 = a, so the voltage pair maps to the same ratio.
            return {
                "primary_turns": primary_voltage,
                "secondary_turns": secondary_voltage,
            }
        return {"primary_turns": primary, "secondary_turns": secondary}

    if formula_name == "transformer_reflected_impedance":
        turns_ratio = _match_value(_TURNS_RATIO_RE.search(question))
        if turns_ratio is None:
            primary = _match_value(_TURNS_PRIMARY_RE.search(question))
            secondary = _match_value(_TURNS_SECONDARY_RE.search(question))
            if primary is None or secondary is None or secondary == 0:
                return None
            turns_ratio = primary / secondary
        load_impedance = _match_value(_LOAD_IMPEDANCE_RE.search(question))
        if load_impedance is None:
            return None
        return {"turns_ratio": turns_ratio, "load_impedance_ohm": load_impedance}

    if formula_name == "single_phase_complex_power":
        voltage = _extract_rms_quantity(question, r"(?:V|voltage)", r"v")
        current = _extract_rms_quantity(question, r"(?:I|current)", r"a")
        if voltage is None or current is None:
            return None
        phase_deg = _match_value(_PHASE_DEG_RE.search(question))
        if phase_deg is not None:
            normalized = question.casefold()
            if "leading" in normalized:
                phase_deg = -abs(phase_deg)
            elif "lagging" in normalized:
                phase_deg = abs(phase_deg)
            return {"voltage_v": voltage, "current_a": current, "phase_deg": phase_deg}
        power_factor = _match_value(_POWER_FACTOR_RE.search(question))
        if power_factor is None:
            return None
        normalized = question.casefold()
        if "leading" in normalized:
            reactive_sign = -1.0
        elif "lagging" in normalized:
            reactive_sign = 1.0
        else:
            return None
        return {
            "voltage_v": voltage,
            "current_a": current,
            "power_factor": power_factor,
            "reactive_sign": reactive_sign,
        }

    if formula_name == "power_from_real_reactive":
        real = _match_value(_REAL_POWER_RE.search(question))
        reactive = _match_value(_REACTIVE_POWER_RE.search(question))
        if real is None or reactive is None:
            return None
        return {"real_power_w": real, "reactive_power_var": reactive}

    if formula_name == "wind_power_cube_scaling":
        speed_factor = _extract_speed_factor(question)
        if speed_factor is None:
            return None
        return {"speed_factor": speed_factor}

    if formula_name == "rc_time_constant":
        resistance = _extract_si_quantity(question, _RESISTANCE_SI_RE)
        capacitance = _extract_si_quantity(question, _CAPACITANCE_SI_RE)
        if resistance is None or capacitance is None:
            return None
        return {"resistance_ohm": resistance, "capacitance_f": capacitance}

    if formula_name == "three_phase_power":
        if _UNBALANCED_RE.search(question):
            return None
        line_voltage = _match_value(_LINE_VOLTAGE_RE.search(question))
        line_current = _match_value(_LINE_CURRENT_RE.search(question))
        power_factor = _match_value(_POWER_FACTOR_RE.search(question))
        if line_voltage is None or line_current is None or power_factor is None:
            return None
        return {
            "line_voltage_v": line_voltage,
            "line_current_a": line_current,
            "power_factor": power_factor,
        }

    if formula_name == "synchronous_speed":
        frequencies: set[float] = {
            float(match.group("value")) for match in _FREQUENCY_HZ_RE.finditer(question)
        }
        poles = _extract_poles(question)
        if len(frequencies) != 1 or poles is None:
            return None
        return {"frequency_hz": frequencies.pop(), "poles": poles}

    if formula_name == "phase_real_power":
        voltages: list[float] | None = _extract_phase_sequence(question, "voltage")
        currents: list[float] | None = _extract_phase_sequence(question, "current")
        factors: list[float] | None = _extract_phase_sequence(question, "power_factor")
        if voltages is None or currents is None or factors is None:
            return None
        return {
            "phase_voltages_v": voltages,
            "phase_currents_a": currents,
            "phase_power_factors": factors,
        }

    return None


def _requests_only_phase_real_power(question: str) -> bool:
    """Admit explicit P requests, not other outputs or ambiguous bare 'power'."""
    # Input cues must not hide another requested power quantity later in the text.
    if re.search(r"\b(?:(?:reactive|apparent|complex)\s+power|q|s)\b", question, re.I):
        return False
    for factor in re.finditer(r"\b(?:power\s+factors?|pf)\b", question, re.I):
        stated_value: bool = (
            re.match(
                rf"\s*(?:(?:are|is|of|=|:)\s*)?(?:\[\s*)?{_PHASE_NUMBER}",
                question[factor.end() :],
                re.I,
            )
            is not None
        )
        stated_unity: bool = re.search(r"\bunity\s+$", question[: factor.start()], re.I) is not None
        input_reference: bool = (
            re.search(
                r"\b(?:given|listed|stated|supplied|known|these|their)\s+(?:phase\s+)?$",
                question[: factor.start()],
                re.I,
            )
            is not None
        )
        if not (stated_value or stated_unity or input_reference):
            return False
    requests: list[re.Match[str]] = list(
        re.finditer(
            r"\b(?:calculate|compute|find|determine|give|what\s+is|what's)\b",
            question,
            re.IGNORECASE,
        )
    )
    for request in requests:
        # A stated power factor after an input cue is an input, not a requested output.
        target: str = re.split(
            r"\b(?:given|using|with)\b", question[request.end() :], maxsplit=1, flags=re.I
        )[0].strip(" \t\r\n.,?!:;")
        if (
            re.fullmatch(
                r"(?:the\s+)?(?:total\s+)?(?:(?:real|active|actual)\s+power|p)"
                r"(?:\s*\(p\))?(?:\s+in\s+(?:w|watts?|kw|kilowatts?))?",
                target,
                re.IGNORECASE,
            )
            is None
        ):
            return False
    return bool(requests)


def classify_pe_formula_by_keywords(question: str) -> str | None:
    """Classify unambiguous PE calculator routes without an LLM call.

    Returns a registry key or ``None`` when the question is conceptual,
    formula-lookup, or otherwise not a supported deterministic calculation.
    """
    normalized = question.casefold()
    if any(marker in normalized for marker in _EXPLANATION_MARKERS) and (
        _ANY_NUMBER_RE.search(question) is None and _extract_speed_factor(question) is None
    ):
        return None

    if "synchronous speed" in normalized:
        return "synchronous_speed"
    if "power" in normalized and (
        _UNBALANCED_RE.search(question)
        or re.search(r"\b(?:each\s+phase|phase\s+(?:voltages|currents))\b", normalized)
    ):
        # Do not let an unsupported phase request fall through to a single-phase tool.
        return "phase_real_power" if _requests_only_phase_real_power(question) else None
    if "time constant" in normalized and any(
        cue in normalized for cue in ("rc", "capacitor", "capacitance")
    ):
        return "rc_time_constant"
    if (
        "wind" in normalized
        and "power" in normalized
        and any(cue in normalized for cue in ("speed", "cube", "doubl"))
    ):
        return "wind_power_cube_scaling"
    if ("three-phase" in normalized or "3-phase" in normalized) and "power" in normalized:
        return "three_phase_power"
    if "transformer" in normalized or "winding" in normalized:
        if any(
            cue in normalized
            for cue in ("reflected", "input impedance", "seen from", "z_in", "zin")
        ):
            return "transformer_reflected_impedance"
        if any(
            cue in normalized
            for cue in (
                "turns ratio",
                "turn ratio",
                "n1",
                "n2",
                "primary",
                "secondary",
                "voltage ratio",
            )
        ):
            return "ideal_transformer_turns_ratio"
    if any(
        cue in normalized
        for cue in (
            "complex power",
            "apparent power",
            "power factor",
            "reactive power",
            "real power",
        )
    ):
        real = _match_value(_REAL_POWER_RE.search(question))
        reactive = _match_value(_REACTIVE_POWER_RE.search(question))
        if real is not None and reactive is not None:
            return "power_from_real_reactive"
        return "single_phase_complex_power"
    return None


def dispatch_formula(formula_name: str, **params: Any) -> PECalculatorResult:
    """Look up a PE formula by name and call it with the given parameters."""
    spec = FORMULA_REGISTRY[formula_name]
    return spec.func(**params)


def schema_for_formula(formula_name: str) -> dict[str, str]:
    """Return a human-readable variable schema for the named formula."""
    spec = FORMULA_REGISTRY[formula_name]
    descriptions = {
        "primary_turns": "Primary turns N1 (or primary voltage V1 / primary current I1)",
        "secondary_turns": "Secondary turns N2 (or secondary voltage V2 / secondary current I2)",
        "turns_ratio": "Ideal transformer turns ratio a",
        "load_impedance_ohm": "Secondary load impedance in ohms",
        "voltage_v": "RMS voltage in volts",
        "current_a": "RMS current in amps",
        "phase_deg": "Voltage-current phase difference in degrees",
        "power_factor": "Power factor (cos phi)",
        "reactive_sign": "+1 for lagging, -1 for leading",
        "real_power_w": "Real power in watts",
        "reactive_power_var": "Reactive power in vars",
        "speed_factor": "Wind-speed multiplier",
        "resistance_ohm": "Equivalent resistance in ohms",
        "capacitance_f": "Capacitance in farads",
        "line_voltage_v": "Line-to-line RMS voltage in volts",
        "line_current_a": "Line RMS current in amps",
        "frequency_hz": "Supply frequency in Hz",
        "poles": "Total number of stator poles",
        "phase_voltages_v": "Three explicit actual RMS load-phase voltages, in phase order",
        "phase_currents_a": "Three explicit RMS phase currents in the same phase order",
        "phase_power_factors": "Three explicit phase power factors; no unity default",
    }
    return {name: descriptions.get(name, name) for name in spec.param_names}
