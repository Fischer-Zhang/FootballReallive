"""攻防階段狀態機：IP（持球進攻）、DT（防守轉換）、OOP（無球防守）、AT（進攻轉換）。

觸發：球權改變的瞬間，失球方進入 DT，得球方進入 AT。窗口結束（或在窗口內奪回球）後：DT → OOP、AT → IP。
DT 窗口內離球最近的幾個人立即反搶（人數隨 press 戰術），其餘回防；AT 窗口內持球時間縮短（節奏提高）。
窗口長度與人數是模型假設（sim_params.yaml phase 區），可用 SkillCorner 的 phases_of_play 與 StatsBomb 的
counterpress／From Counter 校準。
"""
from __future__ import annotations

import numpy as np

from ..engine.pitch import LENGTH
from .world import Phase, World

TURNOVER = "turnover"
COUNTER_ATTACK = "counter_attack"
PRESS_RECOVERY = "press_recovery"


def dt_window(w: World, T: int) -> float:
    p = w.sp.phase
    press = w.sim.teams[T].tactics.press
    act = w.active(T)
    energy = float(np.mean(w.sim.energy[act])) if act else 1.0
    return max(1.0, (p.dt_window + p.dt_window_press * (press - 3)) * (0.7 + 0.3 * energy))


def possession_change(w: World, new_team: int, slot: int, how: str) -> None:
    """球權改變（斷球、搶斷、撲救後門將控球等）"""
    old = w.poss
    if new_team == old:
        return
    p = w.sp.phase
    w.phase[old] = Phase.DT
    w.phase_t[old] = dt_window(w, old)
    was_dt = w.phase[new_team] == Phase.DT          # 失球後立刻奪回 = 反搶成功
    w.phase[new_team] = Phase.AT
    w.phase_t[new_team] = p.at_window
    pos = w.ball[:2].copy()
    w.add_event(slot, TURNOVER, how, pos, team=old)
    if was_dt:
        w.add_event(slot, PRESS_RECOVERY, how, pos, team=new_team)
    w.counter_from[new_team] = (w.clock, float(w.tf(new_team, pos)[0]))
    w.counter_from[old] = None
    w.poss = new_team


def update(w: World, dt: float) -> None:
    p = w.sp.phase
    for T in (0, 1):
        ph = w.phase[T]
        if ph in (Phase.DT, Phase.AT):
            w.phase_t[T] -= dt
            if w.phase_t[T] <= 0:
                w.phase[T] = Phase.OOP if ph == Phase.DT else Phase.IP
                w.counter_from[T] = None if ph == Phase.DT else w.counter_from[T]
        cf = w.counter_from[T]
        if cf is not None and w.phase[T] in (Phase.AT, Phase.IP) and w.poss == T:
            t0, x0 = cf
            if w.clock - t0 <= p.at_window + 2.0 and w.tf(T, w.ball[:2])[0] - x0 >= p.counter_progress:
                w.add_event(w.owner, COUNTER_ATTACK, "success", w.ball[:2].copy(), team=T)
                w.counter_from[T] = None
            elif w.clock - t0 > p.at_window + 2.0:
                w.counter_from[T] = None


def counter_press_slots(w: World, T: int) -> list:
    """防守轉換窗口內負責反搶的球員：離球最近的 n 人（不含門將）"""
    n = int(w.sp.phase.counter_press_n[int(w.sim.teams[T].tactics.press) - 1])
    gk = w.sim.gk_slot(T)
    cand = [s for s in w.active(T) if s != gk]
    d = np.hypot(w.pos[cand, 0] - w.ball[0], w.pos[cand, 1] - w.ball[1])
    return [cand[i] for i in np.argsort(d)[:n]]
