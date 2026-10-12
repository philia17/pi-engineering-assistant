"""Deterministic calculators for high-frequency EECC converter questions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CalculatorResult:
    """Structured scalar calculator result with a traceable derivation."""

    value: float
    unit: str
    quantity: str
    steps: list[str]
    extra_values: dict[str, float] = field(default_factory=dict)


def _require_positive(name: str, value: float) -> None:
    """Raise when a calculator input is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_duty_cycle(duty_cycle: float) -> None:
    """Raise when duty cycle is outside the open interval (0, 1)."""
    if not 0 < duty_cycle < 1:
        raise ValueError("duty_cycle must be between 0 and 1")


def calculate_forward_turns_ratio(
    *,
    input_voltage_v: float,
    output_voltage_v: float,
    duty_cycle: float,
) -> CalculatorResult:
    """Calculate N1/N2 for an ideal isolated forward converter in CCM.

    Uses ``Vo/Vs = (N2/N1) * D`` and returns the requested primary-to-secondary
    turns ratio ``N1/N2``.
    """
    _require_positive("input_voltage_v", input_voltage_v)
    _require_positive("output_voltage_v", output_voltage_v)
    _require_duty_cycle(duty_cycle)

    secondary_to_primary = output_voltage_v / (input_voltage_v * duty_cycle)
    primary_to_secondary = 1 / secondary_to_primary
    return CalculatorResult(
        value=primary_to_secondary,
        unit="ratio",
        quantity="N1/N2",
        steps=[
            "Use isolated forward CCM gain: Vo/Vs = (N2/N1) * D.",
            f"N2/N1 = {output_voltage_v:g} / ({input_voltage_v:g} * {duty_cycle:g}) = {secondary_to_primary:g}.",
            f"N1/N2 = 1 / {secondary_to_primary:g} = {primary_to_secondary:g}.",
        ],
    )


def calculate_forward_inductance(
    *,
    input_voltage_v: float,
    output_voltage_v: float,
    duty_cycle: float,
    switching_frequency_hz: float,
    ripple_current_a: float,
    secondary_to_primary_turns_ratio: float,
) -> CalculatorResult:
    """Calculate forward-converter output inductance from ripple current.

    During the ON interval, the secondary-side applied voltage is
    ``Vs * (N2/N1)``. The output inductor voltage is therefore
    ``Vs * (N2/N1) - Vo``. Rule 1 gives ``Delta I = V_L,on * D * T / L``.
    """
    _require_positive("input_voltage_v", input_voltage_v)
    _require_positive("output_voltage_v", output_voltage_v)
    _require_duty_cycle(duty_cycle)
    _require_positive("switching_frequency_hz", switching_frequency_hz)
    _require_positive("ripple_current_a", ripple_current_a)
    _require_positive("secondary_to_primary_turns_ratio", secondary_to_primary_turns_ratio)

    reflected_secondary_voltage = input_voltage_v * secondary_to_primary_turns_ratio
    inductor_on_voltage = reflected_secondary_voltage - output_voltage_v
    if inductor_on_voltage <= 0:
        raise ValueError("reflected secondary voltage must exceed output voltage")

    period_s = 1 / switching_frequency_hz
    inductance_h = inductor_on_voltage * duty_cycle * period_s / ripple_current_a
    inductance_uh = inductance_h * 1_000_000
    return CalculatorResult(
        value=inductance_uh,
        unit="uH",
        quantity="L",
        steps=[
            f"Reflected secondary voltage = {input_voltage_v:g} * {secondary_to_primary_turns_ratio:g} = {reflected_secondary_voltage:g} V.",
            f"Inductor ON voltage = {reflected_secondary_voltage:g} - {output_voltage_v:g} = {inductor_on_voltage:g} V.",
            f"Switching period = 1 / {switching_frequency_hz:g} = {period_s:g} s.",
            f"L = {inductor_on_voltage:g} * {duty_cycle:g} * {period_s:g} / {ripple_current_a:g} = {inductance_h:g} H.",
            f"L = {inductance_uh:g} uH.",
        ],
    )


def calculate_forward_transistor_current(
    *,
    output_current_a: float,
    ripple_current_a: float,
    output_voltage_v: float,
    input_voltage_v: float,
    secondary_to_primary_turns_ratio: float,
) -> CalculatorResult:
    """Calculate peak and mean transistor current for an ideal forward converter.

    The output inductor current is a triangular ripple around the DC load
    current. With magnetising current neglected, the primary transistor current
    is the secondary-side current reflected by ``N2/N1``. The full-period mean
    transistor current is found from ideal input power balance.
    """
    _require_positive("output_current_a", output_current_a)
    _require_positive("ripple_current_a", ripple_current_a)
    _require_positive("output_voltage_v", output_voltage_v)
    _require_positive("input_voltage_v", input_voltage_v)
    _require_positive("secondary_to_primary_turns_ratio", secondary_to_primary_turns_ratio)

    secondary_inductor_peak_a = output_current_a + ripple_current_a / 2
    primary_peak_a = secondary_to_primary_turns_ratio * secondary_inductor_peak_a
    output_power_w = output_voltage_v * output_current_a
    mean_transistor_current_a = output_power_w / input_voltage_v
    return CalculatorResult(
        value=primary_peak_a,
        unit="A",
        quantity="peak transistor current",
        steps=[
            f"Secondary inductor peak = {output_current_a:g} + {ripple_current_a:g}/2 = {secondary_inductor_peak_a:g} A.",
            f"Reflected primary peak = {secondary_to_primary_turns_ratio:g} * {secondary_inductor_peak_a:g} = {primary_peak_a:g} A.",
            f"Output power = {output_voltage_v:g} * {output_current_a:g} = {output_power_w:g} W.",
            f"Full-period mean transistor current = {output_power_w:g} / {input_voltage_v:g} = {mean_transistor_current_a:g} A.",
        ],
        extra_values={
            "secondary_inductor_peak_a": secondary_inductor_peak_a,
            "mean_transistor_current_a": mean_transistor_current_a,
        },
    )


def calculate_flyback_turns_ratio_ccm(
    *,
    input_voltage_v: float,
    output_voltage_v: float,
    duty_cycle: float,
) -> CalculatorResult:
    """Calculate N1/N2 for an ideal isolated flyback converter in CCM.

    Uses ``Vo/Vs = (N2/N1) * D / (1-D)`` and returns the requested
    primary-to-secondary turns ratio ``N1/N2``.
    """
    _require_positive("input_voltage_v", input_voltage_v)
    _require_positive("output_voltage_v", output_voltage_v)
    _require_duty_cycle(duty_cycle)

    secondary_to_primary = output_voltage_v / input_voltage_v * (1 - duty_cycle) / duty_cycle
    primary_to_secondary = 1 / secondary_to_primary
    return CalculatorResult(
        value=primary_to_secondary,
        unit="ratio",
        quantity="N1/N2",
        steps=[
            "Use isolated flyback CCM gain: Vo/Vs = (N2/N1) * D / (1-D).",
            f"N2/N1 = ({output_voltage_v:g} / {input_voltage_v:g}) * ((1 - {duty_cycle:g}) / {duty_cycle:g}) = {secondary_to_primary:g}.",
            f"N1/N2 = 1 / {secondary_to_primary:g} = {primary_to_secondary:g}.",
        ],
    )


def calculate_non_isolated_flyback_duty_cycle(
    *,
    input_voltage_v: float,
    output_voltage_v: float,
) -> CalculatorResult:
    """Calculate duty cycle D for an ideal non-isolated flyback (buck-boost) converter in CCM.

    Uses Vout/Vin = D/(1-D) and returns D.
    """
    _require_positive("input_voltage_v", input_voltage_v)
    _require_positive("output_voltage_v", output_voltage_v)

    duty_cycle = output_voltage_v / (input_voltage_v + output_voltage_v)
    return CalculatorResult(
        value=duty_cycle,
        unit="ratio",
        quantity="D",
        steps=[
            "Use non-isolated flyback CCM gain: Vout/Vin = D/(1-D).",
            f"D = {output_voltage_v:g} / ({input_voltage_v:g} + {output_voltage_v:g}) = {duty_cycle:g}.",
        ],
    )


def calculate_flyback_turns_ratio_n2_n1(
    *,
    input_voltage_v: float,
    output_voltage_v: float,
    duty_cycle: float,
) -> CalculatorResult:
    """Calculate N2/N1 for an ideal isolated flyback converter in CCM.

    Uses Vo/Vs = (N2/N1) * D/(1-D) and returns secondary-to-primary ratio N2/N1.
    """
    _require_positive("input_voltage_v", input_voltage_v)
    _require_positive("output_voltage_v", output_voltage_v)
    _require_duty_cycle(duty_cycle)

    secondary_to_primary = output_voltage_v / input_voltage_v * (1 - duty_cycle) / duty_cycle
    return CalculatorResult(
        value=secondary_to_primary,
        unit="ratio",
        quantity="N2/N1",
        steps=[
            "Use isolated flyback CCM gain: Vo/Vs = (N2/N1) * D/(1-D).",
            f"N2/N1 = ({output_voltage_v:g} / {input_voltage_v:g}) * ((1 - {duty_cycle:g}) / {duty_cycle:g}) = {secondary_to_primary:g}.",
        ],
    )


def calculate_dc_machine_induced_emf(
    *,
    terminal_voltage_v: float,
    armature_current_a: float,
    armature_resistance_ohm: float,
) -> CalculatorResult:
    """Calculate induced armature EMF for a DC machine in generator and motor modes.

    Generator: E = Vt + Ia*Ra  (EMF > terminal voltage)
    Motor:     E = Vt - Ia*Ra  (EMF < terminal voltage)

    Returns generator EMF as primary value, motor EMF in extra_values.
    """
    _require_positive("terminal_voltage_v", terminal_voltage_v)
    _require_positive("armature_current_a", armature_current_a)
    _require_positive("armature_resistance_ohm", armature_resistance_ohm)

    voltage_drop = armature_current_a * armature_resistance_ohm
    generator_emf_v = terminal_voltage_v + voltage_drop
    motor_emf_v = terminal_voltage_v - voltage_drop

    return CalculatorResult(
        value=generator_emf_v,
        unit="V",
        quantity="induced EMF (generator)",
        steps=[
            f"Armature voltage drop = {armature_current_a:g} * {armature_resistance_ohm:g} = {voltage_drop:g} V.",
            f"Generator EMF = Vt + Ia*Ra = {terminal_voltage_v:g} + {voltage_drop:g} = {generator_emf_v:g} V.",
            f"Motor EMF = Vt - Ia*Ra = {terminal_voltage_v:g} - {voltage_drop:g} = {motor_emf_v:g} V.",
        ],
        extra_values={"motor_emf_v": motor_emf_v},
    )


@dataclass(frozen=True)
class FormulaSpec:
    """Metadata for a registered calculator formula."""

    name: str
    topic: str
    description: str
    func: Callable[..., CalculatorResult]
    param_names: tuple[str, ...]
    quantity: str
    unit: str


FORMULA_REGISTRY: dict[str, FormulaSpec] = {
    "forward_turns_ratio": FormulaSpec(
        name="forward_turns_ratio",
        topic="converter_circuits",
        description="Isolated forward converter turns ratio N1/N2 from Vo/Vs = (N2/N1)*D",
        func=calculate_forward_turns_ratio,
        param_names=("input_voltage_v", "output_voltage_v", "duty_cycle"),
        quantity="N1/N2",
        unit="ratio",
    ),
    "forward_inductance": FormulaSpec(
        name="forward_inductance",
        topic="converter_circuits",
        description="Forward converter output inductance from ripple current specification",
        func=calculate_forward_inductance,
        param_names=(
            "input_voltage_v",
            "output_voltage_v",
            "duty_cycle",
            "switching_frequency_hz",
            "ripple_current_a",
            "secondary_to_primary_turns_ratio",
        ),
        quantity="L",
        unit="uH",
    ),
    "forward_transistor_current": FormulaSpec(
        name="forward_transistor_current",
        topic="converter_circuits",
        description="Forward converter peak and mean transistor current from output specs",
        func=calculate_forward_transistor_current,
        param_names=(
            "output_current_a",
            "ripple_current_a",
            "output_voltage_v",
            "input_voltage_v",
            "secondary_to_primary_turns_ratio",
        ),
        quantity="peak transistor current",
        unit="A",
    ),
    "flyback_ccm_turns_ratio_n1_n2": FormulaSpec(
        name="flyback_ccm_turns_ratio_n1_n2",
        topic="converter_circuits",
        description="Isolated flyback CCM primary-to-secondary turns ratio N1/N2",
        func=calculate_flyback_turns_ratio_ccm,
        param_names=("input_voltage_v", "output_voltage_v", "duty_cycle"),
        quantity="N1/N2",
        unit="ratio",
    ),
    "flyback_ccm_turns_ratio_n2_n1": FormulaSpec(
        name="flyback_ccm_turns_ratio_n2_n1",
        topic="converter_circuits",
        description="Isolated flyback CCM secondary-to-primary turns ratio N2/N1",
        func=calculate_flyback_turns_ratio_n2_n1,
        param_names=("input_voltage_v", "output_voltage_v", "duty_cycle"),
        quantity="N2/N1",
        unit="ratio",
    ),
    "non_isolated_flyback_duty_cycle": FormulaSpec(
        name="non_isolated_flyback_duty_cycle",
        topic="converter_circuits",
        description="Non-isolated flyback (buck-boost) duty cycle D from Vin and Vout",
        func=calculate_non_isolated_flyback_duty_cycle,
        param_names=("input_voltage_v", "output_voltage_v"),
        quantity="D",
        unit="ratio",
    ),
    "dc_machine_induced_emf": FormulaSpec(
        name="dc_machine_induced_emf",
        topic="electrical_machines",
        description="DC machine induced armature EMF in generator and motor modes",
        func=calculate_dc_machine_induced_emf,
        param_names=(
            "terminal_voltage_v",
            "armature_current_a",
            "armature_resistance_ohm",
        ),
        quantity="induced EMF (generator)",
        unit="V",
    ),
}


def dispatch_formula(formula_name: str, **params: float) -> CalculatorResult:
    """Look up a formula by name and call it with the given parameters.

    Raises KeyError if formula_name is not in FORMULA_REGISTRY.
    """
    spec = FORMULA_REGISTRY[formula_name]
    return spec.func(**params)
