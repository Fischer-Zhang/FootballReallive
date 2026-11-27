"""MatchSim：比賽主迴圈與狀態管理（§4.2）

每個事件：更新 22 人快照（shape）→ 持球者決策（decision）→ 屬性對抗解析（actions）
→ 套用結果 → 裁判、傷病、體能、教練 AI → 推進時鐘。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import load_params
from ..data.schema import MatchContext, Player, Team
from ..model.attributes import (ATTR, DAY_FORM, MENTAL_FATIGUE, MENTAL_GROUP, N_ATTR,
                                PHYSICAL_FATIGUE, TECHNICAL_FATIGUE, compile_composites)
from ..model.positions import position_rating
from . import actions, decision, fatigue, manager, referee, restarts, shape
from .events import PERIOD_END, Event
from .pitch import LENGTH, WIDTH, threat_for
from .state import Outcome, Restart, TeamRT

_DEFAULT_PARAMS = None
_POSITIONAL = np.array([ATTR.DEF_POSITIONING, ATTR.OFF_THE_BALL, ATTR.DECISIONS, ATTR.TEAMWORK])
_TECH_MENT = np.concatenate([TECHNICAL_FATIGUE, MENTAL_GROUP])


def default_params():
    global _DEFAULT_PARAMS
    if _DEFAULT_PARAMS is None:
        _DEFAULT_PARAMS = load_params()
    return _DEFAULT_PARAMS


def _compiled(params):
    comp = getattr(params, "_compiled", None)
    if comp is None:
        comp = compile_composites(params._raw["composites"])
        params._compiled = comp
    return comp


@dataclass
class MatchResult:
    home: str
    away: str
    home_id: str
    away_id: str
    score: tuple
    ht_score: tuple
    events: List[Event]
    players: Dict[str, dict]            # pid → {team, name, position}
    minutes: Dict[str, float]
    ball_in_play: float                 # 秒
    ball_in_play_half: tuple            # 秒（上, 下半場）
    possession_time: tuple              # 秒（主, 客）
    stoppage: tuple                     # 分鐘（上, 下半場）
    subs: tuple
    seed: int
    _stats: Optional[dict] = field(default=None, repr=False)

    def stats(self) -> dict:
        if self._stats is None:
            from ..analytics.stats import match_stats
            self._stats = match_stats(self)
        return self._stats


MAX_STEPS = 20000


class MatchSim:
    def __init__(self, home: Team, away: Team, params=None, ctx: Optional[MatchContext] = None,
                 seed: int = 0):
        self.p = params or default_params()
        p = self.p
        self.R = float(p.reference_level)
        self.ctx = ctx or MatchContext()
        self.seed = seed
        streams = np.random.SeedSequence(seed).spawn(5)
        (self.rng_shape, self.rng_dec, self.rng_act,
         self.rng_ref, self.rng_misc) = [np.random.default_rng(s) for s in streams]
        self.threat = threat_for(p)
        self.W, self.C = _compiled(p)

        dup = {p.player_id for p in home.players} & {p.player_id for p in away.players}
        if dup:
            raise ValueError(f"兩隊有重複的球員 ID（例如 {sorted(dup)[0]}）；每位球員的 ID 必須唯一")
        neutral = self.ctx.neutral
        self.teams = [TeamRT(0, home, not neutral), TeamRT(1, away, False)]
        att = self.ctx.attendance if self.ctx.attendance is not None else home.stadium.attendance_ratio
        self.attendance = 0.0 if neutral else float(att)
        hb = p.referee.home_bias * self.attendance
        self.ref_home_bias = [-hb, hb]     # 加在「犯規方」的犯規與出牌 logit 上
        self.ref_strict = float(self.ctx.referee_strictness)

        n = 22
        self.players: List[Optional[Player]] = [None] * n
        self.base = np.zeros((n, N_ATTR))
        self.eff = np.zeros((n, N_ATTR))
        self.comp = np.zeros((n, self.W.shape[1]))
        self.energy = np.ones(n)
        self.energy_start = np.ones(n)
        self.active = np.zeros(n, dtype=bool)
        self.yellow = np.zeros(n, dtype=int)
        self.knock = np.zeros(n)
        self.need_off = np.zeros(n, dtype=bool)
        self.height = np.full(n, 180.0)
        self.weak = np.full(n, 50.0)
        self.vmax = np.full(n, 8.5)
        self.tau = np.full(n, 0.8)
        self.jump = np.full(n, 50.0)
        self.intensity = np.ones(n)
        self.pos = np.zeros((n, 2))
        self.gpos: Optional[np.ndarray] = None   # 上一個快照的全域座標（位置連續性）
        self.last_shape_elapsed = 0.0
        self._box_cfg: Optional[int] = None      # 禁區跑位沿用的真實站位組合（shape._box_runs）
        self._hold: Optional[float] = None       # 這次決策的持球時間（open_play 抽一次）
        sp = p.sprint
        self.v_ref = sp.vmax_a + sp.vmax_b * self.R   # 平均水準球員的最高速度（逼近速度的基準）
        self.traits: List[dict] = [{} for _ in range(n)]

        self.events: List[Event] = []
        self.period = 1
        self.t = 0.0
        self.elapsed = 0.0
        self.bip = 0.0
        self.bip_half = [0.0, 0.0]
        self.poss_time = [0.0, 0.0]
        self.extra_dead = [0.0, 0.0]
        self.stoppage: List[Optional[int]] = [None, None]
        self.min_done = 0
        self.poss = 0
        self.carrier: Optional[int] = None
        self.ball = (52.5, 34.0)
        self.restart: Optional[Restart] = None
        self.ht_score = (0, 0)
        self.minutes_on: Dict[str, float] = {}
        self.minutes_off: Dict[str, float] = {}
        self.day_vec: Dict[str, np.ndarray] = {}
        self.first_kickoff = 0

        self._prepare_players()
        for t in (0, 1):
            team = self.teams[t]
            for k, pl in enumerate(team.starters):
                self.put(11 * t + k, pl, minute=0.0)
            # 教練對本隊實力的認知：先發陣容在各自位置上的平均評分
            team.strength = float(np.mean([position_rating(self.day_vec[pl.player_id], team.slot_pos[k])
                                           for k, pl in enumerate(team.starters)]))
        self.refresh()

    # ------------------------------------------------------------------ 球員與屬性
    def _prepare_players(self) -> None:
        """當日狀態（穩定性）、士氣、大場面、比賽感覺、主場因素 → 當日屬性向量"""
        p = self.p
        for t, team in enumerate(self.teams):
            for pl in team.starters + team.bench:
                v = pl.attr_vector()
                sigma = p.day_form.sigma_min + p.day_form.sigma_span * (1 - v[ATTR.CONSISTENCY] / 100)
                v[DAY_FORM] += self.rng_misc.normal(0.0, sigma)
                v[MENTAL_GROUP] += p.morale_points * (pl.state.morale - 60) / 40
                if self.ctx.importance >= p.big_match.threshold:
                    v[MENTAL_GROUP] += p.big_match.points * (v[ATTR.BIG_MATCH] - 50) / 50
                sharp = pl.state.match_sharpness
                if sharp < 0.7:
                    v[_TECH_MENT] -= 3.0 * (0.7 - sharp) / 0.7
                if not self.ctx.neutral:
                    if team.is_home:
                        v[ATTR.FIRST_TOUCH] += p.home.familiarity
                        v[ATTR.SHORT_PASSING] += p.home.familiarity
                    else:
                        v[ATTR.COMPOSURE] -= p.home.crowd_composure * self.attendance
                self.day_vec[pl.player_id] = np.clip(v, 1.0, 100.0)

    def put(self, slot: int, pl: Player, minute: float) -> None:
        t, k = divmod(slot, 11)
        team = self.teams[t]
        v = self.day_vec[pl.player_id].copy()
        fam = pl.familiarity(team.slot_pos[k])
        v[_POSITIONAL] *= 0.80 + 0.20 * fam / 100          # 非本位懲罰（§3.4）
        self.players[slot] = pl
        self.base[slot] = v
        start = pl.state.condition - (0.0 if team.is_home or self.ctx.neutral
                                      else self.p.home.away_energy)
        self.energy[slot] = start
        self.energy_start[slot] = start
        self.active[slot] = True
        self.yellow[slot] = 0
        self.knock[slot] = 0.0
        self.need_off[slot] = False
        self.height[slot] = pl.height_cm
        self.weak[slot] = pl.weak_foot
        self.traits[slot] = dict(pl.traits)
        self.intensity[slot] = fatigue.intensity(self, t, team.slot_group[k])
        late = pl.traits.get("arrives_late", 0.0)
        team.ip_adj[k] = self.p.shape.arrives_late * late if team.slot_group[k] in ("CM", "AM") else 0.0
        self.minutes_on[pl.player_id] = minute

    def refresh(self) -> None:
        """依能量重算有效屬性、複合能力與物理參數"""
        f = self.p.fatigue
        e = self.energy
        pf = np.where(e >= f.physical.threshold, 1.0,
                      np.maximum(f.physical.floor, 1.0 - f.physical.slope * (f.physical.threshold - e)))
        det = self.base[:, ATTR.DETERMINATION]
        mf = np.where(e >= f.mental.threshold, 1.0,
                      np.maximum(f.mental.floor,
                                 1.0 - f.mental.slope * (f.mental.threshold - e) * (1.2 - det / 100)))
        tf = 1.0 - f.technical_share * (1.0 - pf)
        eff = self.base.copy()
        eff[:, PHYSICAL_FATIGUE] *= pf[:, None]
        eff[:, TECHNICAL_FATIGUE] *= tf[:, None]
        eff[:, MENTAL_FATIGUE] *= mf[:, None]
        eff *= (1.0 - self.knock)[:, None]
        self.eff = eff
        self.comp = eff @ self.W
        sp, ae = self.p.sprint, self.p.aerial
        self.vmax = sp.vmax_a + sp.vmax_b * eff[:, ATTR.SPRINT_SPEED]
        self.tau = sp.tau_a + sp.tau_b * eff[:, ATTR.ACCELERATION]
        self.jump = ae.jump_a + ae.jump_b * eff[:, ATTR.JUMPING]

    # ------------------------------------------------------------------ 小工具
    def active_slots(self, t: int) -> List[int]:
        return [s for s in range(11 * t, 11 * t + 11) if self.active[s]]

    def gk_slot(self, t: int) -> Optional[int]:
        s = 11 * t
        return s if self.active[s] else None

    def is_gk(self, s: int) -> bool:
        return s % 11 == 0

    def match_minute(self) -> float:
        return (45.0 if self.period == 2 else 0.0) + self.t / 60.0

    def to_global(self, team: int, x: float, y: float) -> tuple:
        return (float(x), float(y)) if team == 0 else (LENGTH - float(x), WIDTH - float(y))

    to_frame = to_global   # 互為反函數

    def pid(self, s: Optional[int]) -> Optional[str]:
        if s is None:
            return None
        pl = self.players[s]
        return pl.player_id if pl is not None else None

    def ev(self, frame: int, s: Optional[int], etype: str, result: str, x: float, y: float,
           ex: Optional[float] = None, ey: Optional[float] = None, team: Optional[int] = None,
           **kw) -> Event:
        """建立事件。(x, y) 是 frame 隊的本隊座標，存檔時轉為全域座標；
        事件所屬隊伍由球員所在的 slot 決定（s // 11）。"""
        gx, gy = self.to_global(frame, x, y)
        if ex is None:
            gex, gey = gx, gy
        else:
            gex, gey = self.to_global(frame, ex, ey)
        if team is None:
            team = s // 11 if s is not None else frame
        return Event(self.period, self.t, team, self.pid(s), etype, result,
                     float(gx), float(gy), float(gex), float(gey), **kw)

    def place(self, frame: int, s: int, x: float, y: float) -> None:
        """直接設定某球員的位置（例如被過掉的防守者落在進攻者身後）"""
        if self.gpos is not None:
            self.gpos[s] = self.to_global(frame, x, y)

    def draw_hold(self) -> float:
        """持球到出手的時間（s）；戰術節奏越快越短"""
        lo, hi = self.p.decision.decide_time
        tempo = self.teams[self.poss].tactics.tempo
        return float(self.rng_act.uniform(lo, hi)) * (1.0 - 0.06 * (tempo - 3))

    def decide_time(self) -> float:
        """這次持球的持球時間。open_play 在每次決策開頭抽一次，逼搶者能逼近多遠、持球者能帶多遠、
        動作耗時都用同一個值（同一個物理量不抽兩次）"""
        return self._hold if self._hold is not None else self.draw_hold()

    def hold_time(self) -> float:
        """陣型計算用：決策中用抽到的值；定位球等沒有抽時用期望值"""
        if self._hold is not None:
            return self._hold
        lo, hi = self.p.decision.decide_time
        return 0.5 * (lo + hi) * (1.0 - 0.06 * (self.teams[self.poss].tactics.tempo - 3))

    def advance_dead(self, seconds: float) -> None:
        self.t += seconds
        self.elapsed += seconds
        self._process_minutes()

    # ------------------------------------------------------------------ 主迴圈
    def run(self) -> MatchResult:
        self.first_kickoff = int(self.rng_misc.integers(2))
        self.restart = Restart("kickoff", self.first_kickoff, 52.5, 34.0)
        self.poss = self.first_kickoff
        steps = 0
        while True:
            steps += 1
            if steps > MAX_STEPS:      # 安全上限：正常比賽約 1,500–2,500 步，失控時立即停止而不是吃光記憶體
                raise RuntimeError(f"比賽未正常結束：超過 {MAX_STEPS} 步（第 {self.period} 半場、時鐘 {self.t:.0f} 秒）")
            if self.t >= self.period_end():
                if self.period == 1:
                    self._half_time()
                    continue
                break
            if self.restart is not None:
                manager.at_stoppage(self)
                out = restarts.perform(self, self.restart)
            else:
                out = self.open_play()
            self.apply(out)
        return self._finish()

    def period_end(self) -> float:
        k = self.period - 1
        if self.stoppage[k] is None:
            if self.t < 2700.0:
                return 2700.0
            st = self.p.stoppage
            extra = self.extra_dead[k] + float(self.rng_misc.normal(0.0, 30.0))
            floor = st.min_first if k == 0 else st.min_second
            self.stoppage[k] = int(max(floor, math.ceil(extra / 60.0)))
        return 2700.0 + 60.0 * self.stoppage[k]

    def open_play(self) -> Outcome:
        team, c = self.poss, self.carrier
        bx, by = self.to_frame(team, *self.ball)
        self._hold = self.draw_hold()
        try:
            pos, off_line = shape.compute(self, team, bx, by, c)
            opt = decision.choose(self, team, c, pos, off_line, None)
            return actions.resolve(self, team, c, pos, off_line, opt, None)
        finally:
            self._hold = None

    def apply(self, out: Outcome) -> None:
        if not (math.isfinite(out.duration) and out.duration > 0):
            raise RuntimeError(f"動作耗時異常：{out.duration}（事件 {[e.type for e in out.events]}）")
        self.events.extend(out.events)
        self.poss_time[self.poss] += out.duration
        self.t += out.duration
        self.elapsed += out.duration
        self.bip += out.duration
        self.bip_half[self.period - 1] += out.duration
        if out.goal is not None:
            self.teams[out.goal].score += 1
            self.extra_dead[self.period - 1] += self.p.stoppage.goal
        if out.restart is not None:
            self.restart = out.restart
            self.carrier = None
            self.poss = out.restart.team
            self.ball = (out.restart.x, out.restart.y)
        else:
            self.restart = None
            self.poss = out.team
            self.carrier = out.carrier
            self.ball = out.ball
            if self.carrier is None or not self.active[self.carrier]:
                # 持球者已不在場上（極少見）→ 以界外球重新開始
                self.restart = Restart("throw_in", out.team, out.ball[0],
                                       0.5 if out.ball[1] < 34 else 67.5)
        self._process_minutes()

    def _process_minutes(self) -> None:
        total = int(self.elapsed // 60)
        changed = False
        while self.min_done < total:
            self.min_done += 1
            fatigue.drain_minute(self)
            manager.minute(self)
            referee.minute_hazards(self)
            changed = True
        if changed:
            self.refresh()

    def _half_time(self) -> None:
        self.ht_score = (self.teams[0].score, self.teams[1].score)
        self.events.append(Event(1, self.t, -1, None, PERIOD_END, "", 52.5, 34.0, 52.5, 34.0))
        self.period = 2
        self.t = 0.0
        fatigue.halftime(self)
        manager.halftime(self)
        self.refresh()
        kick = 1 - self.first_kickoff
        self.restart = Restart("kickoff", kick, 52.5, 34.0)
        self.poss = kick
        self.carrier = None

    def _finish(self) -> MatchResult:
        self.events.append(Event(2, self.t, -1, None, PERIOD_END, "", 52.5, 34.0, 52.5, 34.0))
        end_min = 90.0 + (self.stoppage[1] or 0)
        minutes: Dict[str, float] = {}
        for pid, on in self.minutes_on.items():
            off = self.minutes_off.get(pid, end_min)
            minutes[pid] = max(0.0, off - on)
        info = {}
        for t, team in enumerate(self.teams):
            for pl in team.starters + team.bench:
                info[pl.player_id] = {"team": t, "name": pl.name, "position": pl.position.value}
        return MatchResult(
            home=self.teams[0].team.name, away=self.teams[1].team.name,
            home_id=self.teams[0].team.team_id, away_id=self.teams[1].team.team_id,
            score=(self.teams[0].score, self.teams[1].score), ht_score=self.ht_score,
            events=self.events, players=info, minutes=minutes, ball_in_play=self.bip,
            ball_in_play_half=tuple(self.bip_half),
            possession_time=tuple(self.poss_time),
            stoppage=(self.stoppage[0] or 0, self.stoppage[1] or 0),
            subs=(self.teams[0].subs_used, self.teams[1].subs_used), seed=self.seed)


def simulate_match(home: Team, away: Team, params=None, ctx: Optional[MatchContext] = None,
                   seed: int = 0) -> MatchResult:
    return MatchSim(home, away, params, ctx, seed).run()
