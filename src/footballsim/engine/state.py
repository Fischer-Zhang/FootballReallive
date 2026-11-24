"""比賽進行中的狀態結構"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..data.schema import Player, Tactics, Team
from ..model.positions import FORMATIONS, GROUP, Pos, position_rating


@dataclass(slots=True)
class Restart:
    kind: str                   # kickoff / throw_in / goal_kick / corner / free_kick / penalty / gk_hands
    team: int
    x: float                    # 全域座標
    y: float
    extra: Optional[dict] = None


@dataclass(slots=True)
class Outcome:
    duration: float = 0.0       # 球在場上的秒數
    events: list = field(default_factory=list)
    team: int = 0               # 之後控球的隊伍
    carrier: Optional[int] = None
    ball: Tuple[float, float] = (52.5, 34.0)   # 全域座標
    restart: Optional[Restart] = None
    goal: Optional[int] = None  # 進球的隊伍


def select_lineup(team: Team, formation_name: str, unavailable: set | None = None
                  ) -> tuple[list[Player], list[Player]]:
    """依陣型挑選先發 11 人與 9 名替補（位置熟練度 × 位置評分，稀缺位置先選）"""
    unavailable = unavailable or set()
    form = FORMATIONS[formation_name]
    pool = [p for p in team.players if p.player_id not in unavailable and p.state.injury_days == 0
            and p.state.suspended_matches == 0]
    vecs = {p.player_id: p.attr_vector() for p in pool}

    def score(p: Player, pos: Pos) -> float:
        return position_rating(vecs[p.player_id], pos) * (0.55 + 0.45 * p.familiarity(pos) / 100)

    order = sorted(range(11), key=lambda k: (form.positions[k] != Pos.GK,
                                             form.positions[k] not in (Pos.CB, Pos.ST)))
    chosen: dict[int, Player] = {}
    used: set[str] = set()
    for k in order:
        pos = form.positions[k]
        cands = [p for p in pool if p.player_id not in used]
        best = max(cands, key=lambda p: score(p, pos))
        chosen[k] = best
        used.add(best.player_id)
    starters = [chosen[k] for k in range(11)]
    rest = [p for p in pool if p.player_id not in used]
    gks = sorted([p for p in rest if p.position == Pos.GK],
                 key=lambda p: -position_rating(vecs[p.player_id], Pos.GK))
    outfield = sorted([p for p in rest if p.position != Pos.GK],
                      key=lambda p: -position_rating(vecs[p.player_id], p.position))
    bench = gks[:1] + outfield[:8]
    return starters, bench


class TeamRT:
    """比賽中的球隊狀態"""

    def __init__(self, idx: int, team: Team, is_home: bool):
        self.idx = idx
        self.team = team
        self.tactics: Tactics = team.tactics
        self.formation = FORMATIONS[team.tactics.formation]
        self.oop, self.ip = self.formation.arrays()
        self.slot_pos: List[Pos] = list(self.formation.positions)
        self.slot_group: List[str] = [GROUP[p] for p in self.slot_pos]
        self.starters, self.bench = select_lineup(team, team.tactics.formation)
        self.is_home = is_home
        self.score = 0
        self.subs_used = 0
        self.windows_used = 0
        self.mentality_offset = 0.0
        self.red_cards = 0
        self.strength = 65.0                # 先發陣容平均位置評分（MatchSim 設定）
        self.ip_adj = np.zeros(11)          # 特質造成的持球陣型前移（x_rel）
