"""位置、位置群組、位置評分權重與陣型模板（§3.4、§4.4）"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np

from .attributes import INDEX, N_ATTR


class Pos(str, Enum):
    GK = "GK"
    CB = "CB"
    LB = "LB"
    RB = "RB"
    LWB = "LWB"
    RWB = "RWB"
    DM = "DM"
    CM = "CM"
    LM = "LM"
    RM = "RM"
    AM = "AM"
    LW = "LW"
    RW = "RW"
    ST = "ST"


GROUP: Dict[Pos, str] = {
    Pos.GK: "GK", Pos.CB: "CB", Pos.LB: "FB", Pos.RB: "FB", Pos.LWB: "FB", Pos.RWB: "FB",
    Pos.DM: "DM", Pos.CM: "CM", Pos.LM: "W", Pos.RM: "W", Pos.AM: "AM",
    Pos.LW: "W", Pos.RW: "W", Pos.ST: "ST",
}
SIDE: Dict[Pos, str] = {Pos.LB: "L", Pos.LWB: "L", Pos.LM: "L", Pos.LW: "L",
                        Pos.RB: "R", Pos.RWB: "R", Pos.RM: "R", Pos.RW: "R"}
DEFENSIVE_GROUPS = ("CB", "FB", "DM")

# 相近位置的熟練度（生成器用；未列出的位置為 15–35 的隨機值）
RELATED: Dict[Pos, Dict[Pos, int]] = {
    Pos.GK: {},
    Pos.CB: {Pos.DM: 55, Pos.LB: 45, Pos.RB: 45},
    Pos.LB: {Pos.LWB: 92, Pos.LM: 60, Pos.RB: 60, Pos.CB: 50, Pos.RWB: 55},
    Pos.RB: {Pos.RWB: 92, Pos.RM: 60, Pos.LB: 60, Pos.CB: 50, Pos.LWB: 55},
    Pos.LWB: {Pos.LB: 88, Pos.LM: 75, Pos.RWB: 60},
    Pos.RWB: {Pos.RB: 88, Pos.RM: 75, Pos.LWB: 60},
    Pos.DM: {Pos.CM: 85, Pos.CB: 55},
    Pos.CM: {Pos.DM: 75, Pos.AM: 75, Pos.LM: 55, Pos.RM: 55},
    Pos.AM: {Pos.CM: 70, Pos.LW: 60, Pos.RW: 60, Pos.ST: 55},
    Pos.LM: {Pos.LW: 90, Pos.LWB: 60, Pos.RM: 65, Pos.CM: 55},
    Pos.RM: {Pos.RW: 90, Pos.RWB: 60, Pos.LM: 65, Pos.CM: 55},
    Pos.LW: {Pos.LM: 88, Pos.RW: 70, Pos.AM: 60, Pos.ST: 55},
    Pos.RW: {Pos.RM: 88, Pos.LW: 70, Pos.AM: 60, Pos.ST: 55},
    Pos.ST: {Pos.AM: 55, Pos.LW: 50, Pos.RW: 50},
}

# 位置評分（選先發、選替補、算球隊強度）用的關鍵屬性權重
KEY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "GK": {"gk_reflexes": 3, "gk_positioning": 2.5, "gk_diving": 2, "gk_handling": 2,
           "gk_one_on_one": 1.5, "gk_aerial": 1.5, "gk_sweeping": 1, "gk_kicking": 1,
           "gk_communication": 1, "composure": 1, "concentration": 1},
    "CB": {"marking": 2.5, "tackling": 2.5, "def_positioning": 2.5, "heading": 2, "jumping": 1.5,
           "strength": 2, "anticipation": 1.5, "concentration": 1.5, "sprint_speed": 1,
           "acceleration": 1, "short_passing": 1, "composure": 1, "bravery": 1},
    "FB": {"tackling": 2, "marking": 1.5, "def_positioning": 1.5, "sprint_speed": 2,
           "acceleration": 2, "stamina": 2, "crossing": 1.5, "work_rate": 1.5,
           "short_passing": 1, "dribbling": 1, "anticipation": 1, "teamwork": 1},
    "DM": {"tackling": 2, "def_positioning": 2, "anticipation": 2, "short_passing": 2,
           "decisions": 1.5, "teamwork": 1.5, "work_rate": 1.5, "stamina": 1.5,
           "composure": 1, "marking": 1, "strength": 1, "long_passing": 1},
    "CM": {"short_passing": 2.5, "vision": 2, "decisions": 2, "first_touch": 1.5,
           "technique": 1.5, "stamina": 1.5, "work_rate": 1.5, "teamwork": 1.5,
           "long_passing": 1, "anticipation": 1, "composure": 1, "dribbling": 1, "tackling": 1},
    "AM": {"vision": 2.5, "technique": 2, "short_passing": 2, "dribbling": 2, "first_touch": 2,
           "flair": 1.5, "decisions": 1.5, "off_the_ball": 1.5, "long_shots": 1,
           "finishing": 1, "composure": 1, "agility": 1},
    "W": {"acceleration": 2.5, "sprint_speed": 2.5, "dribbling": 2.5, "crossing": 1.5,
          "agility": 1.5, "first_touch": 1.5, "technique": 1, "off_the_ball": 1.5,
          "finishing": 1, "flair": 1, "short_passing": 1, "stamina": 1},
    "ST": {"finishing": 3, "off_the_ball": 2.5, "composure": 2, "first_touch": 1.5,
           "heading": 1.5, "acceleration": 1.5, "sprint_speed": 1, "strength": 1,
           "anticipation": 1.5, "technique": 1, "shot_power": 1, "dribbling": 1},
}


def _weight_vector(weights: Dict[str, float]) -> np.ndarray:
    v = np.zeros(N_ATTR)
    for k, w in weights.items():
        v[INDEX[k]] = w
    return v / v.sum()


KEY_VECTORS: Dict[str, np.ndarray] = {g: _weight_vector(w) for g, w in KEY_WEIGHTS.items()}


def position_rating(attr_vec: np.ndarray, pos: Pos) -> float:
    return float(attr_vec @ KEY_VECTORS[GROUP[pos]])


@dataclass(frozen=True)
class Formation:
    """陣型模板。座標為「本隊進攻方向 = +x」的本隊座標系，y 越大越靠左路。

    oop / ip 的每列為 (x_rel, y)：x_rel ∈ [0, 1] 是在「本隊防線 → 最前線」之間的相對位置，
    y 是基準橫向位置（m）。GK 一律放在第 0 個位置，座標另由引擎計算。
    """
    name: str
    positions: Tuple[Pos, ...]
    oop: Tuple[Tuple[float, float], ...]  # 無球（防守）陣型
    ip: Tuple[Tuple[float, float], ...]   # 持球（進攻）陣型

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.oop, dtype=float), np.array(self.ip, dtype=float)


_GK = (0.0, 34.0)
FORMATIONS: Dict[str, Formation] = {
    "4-3-3": Formation(
        "4-3-3",
        (Pos.GK, Pos.LB, Pos.CB, Pos.CB, Pos.RB, Pos.DM, Pos.CM, Pos.CM, Pos.LW, Pos.RW, Pos.ST),
        (_GK, (0, 56), (0, 41), (0, 27), (0, 12), (0.28, 34), (0.5, 45), (0.5, 23),
         (0.64, 58), (0.64, 10), (1.0, 34)),
        (_GK, (0.32, 62), (0, 45), (0, 23), (0.32, 6), (0.24, 34), (0.56, 46), (0.56, 22),
         (0.92, 63), (0.92, 5), (1.0, 34)),
    ),
    "4-2-3-1": Formation(
        "4-2-3-1",
        (Pos.GK, Pos.LB, Pos.CB, Pos.CB, Pos.RB, Pos.DM, Pos.DM, Pos.LW, Pos.AM, Pos.RW, Pos.ST),
        (_GK, (0, 56), (0, 41), (0, 27), (0, 12), (0.3, 42), (0.3, 26), (0.6, 57),
         (0.66, 34), (0.6, 11), (1.0, 34)),
        (_GK, (0.34, 62), (0, 45), (0, 23), (0.34, 6), (0.26, 42), (0.26, 26), (0.9, 62),
         (0.76, 34), (0.9, 6), (1.0, 34)),
    ),
    "4-4-2": Formation(
        "4-4-2",
        (Pos.GK, Pos.LB, Pos.CB, Pos.CB, Pos.RB, Pos.LM, Pos.CM, Pos.CM, Pos.RM, Pos.ST, Pos.ST),
        (_GK, (0, 56), (0, 41), (0, 27), (0, 12), (0.42, 57), (0.42, 41), (0.42, 27),
         (0.42, 11), (0.95, 40), (1.0, 28)),
        (_GK, (0.3, 62), (0, 45), (0, 23), (0.3, 6), (0.8, 63), (0.46, 40), (0.46, 28),
         (0.8, 5), (1.0, 40), (1.0, 28)),
    ),
    "3-5-2": Formation(
        "3-5-2",
        (Pos.GK, Pos.CB, Pos.CB, Pos.CB, Pos.LWB, Pos.RWB, Pos.DM, Pos.CM, Pos.CM, Pos.ST, Pos.ST),
        (_GK, (0, 46), (0, 34), (0, 22), (0.05, 60), (0.05, 8), (0.3, 34), (0.46, 44),
         (0.46, 24), (0.95, 40), (1.0, 28)),
        (_GK, (0, 50), (0, 34), (0, 18), (0.78, 64), (0.78, 4), (0.24, 34), (0.56, 45),
         (0.56, 23), (1.0, 40), (1.0, 28)),
    ),
}
