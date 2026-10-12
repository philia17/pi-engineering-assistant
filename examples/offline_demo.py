"""Fresh generic numerical examples; no model, course corpus or device needed."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.tools.calculator import calculate_shared_heatsink
from src.tools.pe_calculators import (
    calculate_ideal_transformer_turns_ratio,
    calculate_rc_time_constant,
)


def build_demo() -> dict[str, Any]:
    """Return independently specified numerical examples with explicit units."""
    thermal = calculate_shared_heatsink(
        devices=3,
        p_loss_w=6.0,
        theta_jc_c_per_w=1.0,
        theta_cs_c_per_w=0.5,
        ambient_c=35.0,
        junction_max_c=125.0,
    )
    transformer = calculate_ideal_transformer_turns_ratio(
        primary_turns=480.0, secondary_turns=120.0
    )
    rc = calculate_rc_time_constant(resistance_ohm=3300.0, capacitance_f=47e-6)
    return {
        "mode": "offline_calculators",
        "source": "new generic examples; no research benchmark data",
        "shared_heatsink": asdict(thermal),
        "ideal_transformer": transformer.values,
        "rc_circuit": rc.values,
    }


if __name__ == "__main__":
    print(json.dumps(build_demo(), indent=2))
