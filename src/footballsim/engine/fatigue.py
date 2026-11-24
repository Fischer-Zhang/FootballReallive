"""體能與疲勞（§4.13）"""
from __future__ import annotations

import numpy as np

from ..model.attributes import ATTR


def intensity(sim, team: int, group: str) -> float:
    """每分鐘能量消耗的強度倍率：角色 × 戰術（壓迫、節奏）× 高溫"""
    f = sim.p.fatigue
    tac = sim.teams[team].tactics
    role = getattr(f.role, group)
    tactical = 1.0 + f.press_step * (tac.press - 3) + f.tempo_step * (tac.tempo - 3)
    heat = 1.0 + max(0.0, sim.ctx.temperature - f.heat_threshold) * f.heat_per_deg
    return role * tactical * heat


def drain_minute(sim) -> None:
    f = sim.p.fatigue
    stamina = sim.base[:, ATTR.STAMINA]
    work = sim.base[:, ATTR.WORK_RATE]
    drain = f.d0 * sim.intensity * (1.4 - 0.8 * stamina / 100) * (0.8 + 0.4 * work / 100)
    sim.energy = np.where(sim.active, np.maximum(0.05, sim.energy - drain), sim.energy)


def halftime(sim) -> None:
    f = sim.p.fatigue
    rec = f.ht_base + f.ht_nf * sim.base[:, ATTR.NATURAL_FITNESS] / 100
    sim.energy = np.minimum(sim.energy + rec, sim.energy_start)
