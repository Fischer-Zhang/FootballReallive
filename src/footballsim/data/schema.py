"""資料模型（pydantic v2）：負責 I/O 驗證。比賽引擎的熱路徑只使用 numpy 向量（§3.12）。"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from ..model.attributes import GROUPS
from ..model.positions import Pos

Attr = Annotated[int, Field(ge=1, le=100)]
Pct = Annotated[float, Field(ge=0.0, le=1.0)]
Level = Annotated[int, Field(ge=1, le=5)]


class Foot(str, Enum):
    LEFT = "L"
    RIGHT = "R"
    BOTH = "B"


class Physical(BaseModel):
    acceleration: Attr
    sprint_speed: Attr
    agility: Attr
    balance: Attr
    strength: Attr
    jumping: Attr
    stamina: Attr
    natural_fitness: Attr


class BallSkills(BaseModel):
    first_touch: Attr
    dribbling: Attr
    shielding: Attr
    short_passing: Attr
    long_passing: Attr
    crossing: Attr
    technique: Attr


class Shooting(BaseModel):
    finishing: Attr
    long_shots: Attr
    shot_power: Attr
    heading: Attr


class Defending(BaseModel):
    tackling: Attr
    marking: Attr
    def_positioning: Attr


class Mental(BaseModel):
    anticipation: Attr
    decisions: Attr
    vision: Attr
    composure: Attr
    concentration: Attr
    off_the_ball: Attr
    teamwork: Attr
    work_rate: Attr
    aggression: Attr
    bravery: Attr
    flair: Attr
    leadership: Attr
    determination: Attr


class SetPieces(BaseModel):
    set_piece_delivery: Attr
    free_kick: Attr
    penalty: Attr
    long_throw: Attr


class Goalkeeping(BaseModel):
    gk_reflexes: Attr
    gk_diving: Attr
    gk_positioning: Attr
    gk_handling: Attr
    gk_one_on_one: Attr
    gk_aerial: Attr
    gk_sweeping: Attr
    gk_kicking: Attr
    gk_throwing: Attr
    gk_communication: Attr
    gk_penalty: Attr


class Hidden(BaseModel):
    consistency: Attr
    big_match: Attr
    injury_proneness: Attr
    temperament: Attr
    sportsmanship: Attr
    dirtiness: Attr
    professionalism: Attr
    ambition: Attr
    adaptability: Attr
    versatility: Attr


GROUP_MODELS = {"physical": Physical, "ball": BallSkills, "shooting": Shooting,
                "defending": Defending, "mental": Mental, "set_pieces": SetPieces,
                "goalkeeping": Goalkeeping, "hidden": Hidden}


class Development(BaseModel):
    potential: Attr
    growth_profile: Literal["early", "normal", "late"] = "normal"
    decline_rate: Annotated[float, Field(ge=0.5, le=1.5)] = 1.0


class Condition(BaseModel):
    """賽間動態狀態（§3.6）"""
    condition: Pct = 1.0
    match_sharpness: Pct = 0.85
    morale: Attr = 60
    form: float = 6.7
    acute_load: float = 0.0
    chronic_load: float = 0.0
    injury_days: int = 0
    suspended_matches: int = 0
    yellow_accum: int = 0


class Player(BaseModel):
    player_id: str
    name: str
    birth_date: date
    nationalities: List[str]
    position: Pos
    height_cm: Annotated[int, Field(ge=150, le=210)]
    weight_kg: Annotated[float, Field(ge=50, le=115)]
    preferred_foot: Foot
    weak_foot: Attr
    physical: Physical
    ball: BallSkills
    shooting: Shooting
    defending: Defending
    mental: Mental
    set_pieces: SetPieces
    goalkeeping: Goalkeeping
    hidden: Hidden
    positions: Dict[Pos, Annotated[int, Field(ge=0, le=100)]]
    traits: Dict[str, Pct] = {}
    development: Development
    state: Condition = Condition()

    def attr_vector(self) -> np.ndarray:
        """依 model.attributes.ATTR 的順序攤平成向量"""
        out = []
        for gname in GROUPS:
            group = getattr(self, gname)
            out.extend(getattr(group, n) for n in GROUPS[gname])
        return np.asarray(out, dtype=np.float64)

    def age_on(self, day: date) -> float:
        return (day - self.birth_date).days / 365.25

    def familiarity(self, pos: Pos) -> int:
        return self.positions.get(pos, 20)


class Tactics(BaseModel):
    """球隊戰術（§5）。1–5 級的參數以 3 為中性值。"""
    formation: str = "4-3-3"
    mentality: Level = 3
    line_height: Level = 3
    press: Level = 3
    width: Level = 3
    tempo: Level = 3
    directness: Level = 3
    shooting: Level = 3            # 1 = 打進禁區才射 … 5 = 見機就射
    buildup: Literal["short", "mixed", "long"] = "mixed"
    tackling: Literal["cautious", "normal", "aggressive"] = "normal"
    offside_trap: bool = False


class Stadium(BaseModel):
    name: str = ""
    capacity: int = 30000
    attendance_ratio: Pct = 0.9
    pitch_quality: Pct = 0.9


class Team(BaseModel):
    team_id: str
    name: str
    short_name: str
    level: float = 65.0
    players: List[Player]
    tactics: Tactics = Tactics()
    stadium: Stadium = Stadium()

    def player(self, pid: str) -> Player:
        for p in self.players:
            if p.player_id == pid:
                return p
        raise KeyError(pid)


class League(BaseModel):
    name: str
    season: str
    teams: List[Team]


class MatchContext(BaseModel):
    """比賽情境（§3.9）"""
    match_date: date = date(2026, 9, 1)
    importance: Pct = 0.3
    attendance: Optional[Pct] = None   # None → 依主場上座率
    temperature: float = 15.0
    rain: Literal[0, 1, 2] = 0
    referee_strictness: float = 0.0    # 約 −0.3 … +0.3（logit 位移）
    neutral: bool = False
