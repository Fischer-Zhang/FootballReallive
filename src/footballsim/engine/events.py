"""事件與事件類型（SPADL 相容命名 + 少量擴充，§3.11）"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- 事件類型 ---
PASS = "pass"
CROSS = "cross"
THROUGH = "through_ball"
DRIBBLE = "dribble"            # 帶球推進（SPADL 的 dribble）
TAKE_ON = "take_on"
TACKLE = "tackle"
INTERCEPTION = "interception"
CLEARANCE = "clearance"
SHOT = "shot"
SHOT_PENALTY = "shot_penalty"
SHOT_FREEKICK = "shot_freekick"
KEEPER_SAVE = "keeper_save"
KEEPER_CLAIM = "keeper_claim"
FOUL = "foul"
CARD = "card"
BAD_TOUCH = "bad_touch"
AERIAL = "aerial_duel"
RECOVERY = "ball_recovery"
THROW_IN = "throw_in"
GOAL_KICK = "goalkick"
CORNER = "corner_crossed"
CORNER_SHORT = "corner_short"
FREEKICK_CROSSED = "freekick_crossed"
FREEKICK_SHORT = "freekick_short"
KICKOFF = "kickoff"
GK_DISTRIBUTION = "keeper_distribution"
OFFSIDE = "offside"
SUBSTITUTION = "substitution"
INJURY = "injury"
PERIOD_END = "period_end"

PASS_LIKE = {PASS, THROUGH, CROSS, THROW_IN, GOAL_KICK, CORNER_SHORT, FREEKICK_SHORT,
             KICKOFF, GK_DISTRIBUTION}
SHOT_TYPES = {SHOT, SHOT_PENALTY, SHOT_FREEKICK}

# --- 結果 ---
SUCCESS = "success"
FAIL = "fail"
GOAL = "goal"
SAVED = "saved"
BLOCKED = "blocked"
OFF_TARGET = "off_target"
YELLOW = "yellow"
RED = "red"
SECOND_YELLOW = "second_yellow"


@dataclass(slots=True)
class Event:
    period: int
    t: float                    # 半場內秒數（比賽時鐘）
    team: int
    player: Optional[str]
    type: str
    result: str
    x: float                    # 全域座標（主隊進攻方向 = +x）
    y: float
    end_x: float
    end_y: float
    body: str = "foot"
    opponent: Optional[str] = None
    xg: Optional[float] = None
    extra: Optional[dict] = None

    @property
    def minute(self) -> float:
        return (45.0 if self.period == 2 else 0.0) + self.t / 60.0
