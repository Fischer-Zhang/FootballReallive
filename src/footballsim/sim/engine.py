"""逐幀比賽主迴圈。

一個 tick（預設 0.1 s）的順序與依賴：
  1 球運動（掃掠偵測）→ 2 接觸判定（攔截、到達控制、無主球、搶斷）→ 3 持球者動作（決策、起腳）
  → 4 規則（出界、進球）→ 5 階段更新 → 6 目標點（每 1 s 或球權改變）→ 7 無球意圖（每 0.5 s）
  → 8 移動積分（轉向、體能、跑動距離）→ 9 時鐘與統計。
"""
from __future__ import annotations

import numpy as np

from ..engine.events import PERIOD_END
from ..engine.match import MatchResult
from . import phase, physics, play, rules, shape
from .world import Intent, Mode, World

MAX_TICKS = 80000


class TickMatch:
    def __init__(self, home, away, params=None, sim_params=None, seed: int = 0):
        self.w = World(home, away, params, sim_params, seed)
        self.first_kickoff = int(self.w.sim.rng_misc.integers(2))

    # ------------------------------------------------------------------ 一個 tick
    def step(self) -> None:
        w = self.w
        dt = w.sp.clock.dt
        if w.mode == Mode.DEAD:
            if w.restart is not None and w.clock >= w.restart["t"]:
                rules.perform_restart(w)
        else:
            prev, cur = physics.advance_ball(w, dt)
            if w.mode == Mode.FLIGHT:
                play.flight_step(w, prev, cur, dt)
            elif w.mode == Mode.FREE:
                play.free_step(w, dt)
            elif w.mode == Mode.CARRIED:
                play.tackle_step(w, dt)
            if w.mode == Mode.CARRIED:
                play.carrier_step(w)
            if w.mode != Mode.DEAD:
                rules.check_out(w)
        phase.update(w, dt)
        if w.clock >= w.next_shape:
            shape.update_targets(w)
            w.next_shape = w.clock + w.sp.shape.period
            w.next_intent = w.clock
        if w.clock >= w.next_intent:
            shape.assign_intents(w)
            w.next_intent = w.clock + w.sp.shape.intent_period
        for s, (it, tgt) in w.override.items():         # 持球推進、接球等每個 tick 都套用
            w.intent[s] = int(it)
            w.tgt[s] = tgt
        physics.integrate(w, dt)
        if w.mode == Mode.CARRIED and w.owner is not None:        # 球跟著持球者（移動後同步）
            w.ball[:2] = w.pos[w.owner]
        w.t += dt
        w.clock += dt
        w.tick_n += 1
        if w.mode != Mode.DEAD:
            w.bip += dt
            w.bip_half[w.period - 1] += dt
            w.poss_time[w.poss] += dt

    # ------------------------------------------------------------------ 整場
    def run(self) -> MatchResult:
        w = self.w
        sp = w.sp
        half = sp.clock.half_seconds
        w.stoppage = [int(round(s / 60.0)) for s in sp.clock.stoppage_seconds]
        ends = [half + sp.clock.stoppage_seconds[0], half + sp.clock.stoppage_seconds[1]]
        rules.start_restart(w, "kickoff", self.first_kickoff, 52.5, 34.0, 0.0)
        rules.kickoff_teleport(w)
        while True:
            if w.t >= ends[w.period - 1]:
                if w.period == 1:
                    w.ht_score = tuple(w.score)
                    w.events.append(self._period_end())
                    w.period, w.t = 2, 0.0
                    rules.start_restart(w, "kickoff", 1 - self.first_kickoff, 52.5, 34.0, 0.0)
                    rules.kickoff_teleport(w)
                    continue
                break
            if w.tick_n > MAX_TICKS:
                raise RuntimeError(f"比賽未正常結束：超過 {MAX_TICKS} 個 tick（第 {w.period} 半場、{w.t:.0f} 秒）")
            self.step()
        return self.result()

    def _period_end(self):
        from ..engine.events import Event
        return Event(self.w.period, self.w.t, -1, None, PERIOD_END, "", 52.5, 34.0, 52.5, 34.0)

    def result(self) -> MatchResult:
        w = self.w
        sim = w.sim
        w.events.append(self._period_end())
        end_min = 90.0 + w.stoppage[1]
        info, minutes = {}, {}
        for t, team in enumerate(sim.teams):
            for k, pl in enumerate(team.starters):
                minutes[pl.player_id] = end_min
            for pl in team.starters + team.bench:
                info[pl.player_id] = {"team": t, "name": pl.name, "position": pl.position.value}
        res = MatchResult(
            home=sim.teams[0].team.name, away=sim.teams[1].team.name,
            home_id=sim.teams[0].team.team_id, away_id=sim.teams[1].team.team_id,
            score=tuple(w.score), ht_score=w.ht_score, events=w.events, players=info, minutes=minutes,
            ball_in_play=w.bip, ball_in_play_half=tuple(w.bip_half), possession_time=tuple(w.poss_time),
            stoppage=tuple(w.stoppage), subs=(0, 0), seed=w.seed)
        res.distance = {w.pid(s): float(w.dist_run[s]) for s in range(22) if w.pid(s)}
        return res


def simulate_tick_match(home, away, params=None, sim_params=None, seed: int = 0) -> MatchResult:
    return TickMatch(home, away, params, sim_params, seed).run()
