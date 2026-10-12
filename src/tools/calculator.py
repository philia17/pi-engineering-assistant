"""Deterministic engineering calculators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedHeatsinkResult:
    """Result for a shared-heatsink thermal resistance calculation."""

    total_power_w: float
    theta_sa_c_per_w: float


def calculate_shared_heatsink(
    *,
    devices: int,
    p_loss_w: float,
    theta_jc_c_per_w: float,
    theta_cs_c_per_w: float,
    ambient_c: float,
    junction_max_c: float,
) -> SharedHeatsinkResult:
    """Calculate total heatsink power and maximum heatsink-to-ambient resistance.

    Thermal model for identical devices on a shared heatsink:

    T_J = T_A + P_loss * (theta_JC + theta_CS) + P_total * theta_SA
    P_total = devices * P_loss
    """
    total_power = devices * p_loss_w
    package_rise = p_loss_w * (theta_jc_c_per_w + theta_cs_c_per_w)
    theta_sa = (junction_max_c - ambient_c - package_rise) / total_power
    return SharedHeatsinkResult(
        total_power_w=round(total_power, 4),
        theta_sa_c_per_w=round(theta_sa, 4),
    )
