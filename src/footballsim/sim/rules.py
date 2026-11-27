"""規則（MVP）：出界、進球、重啟、半場。越位、犯規、牌、換人、傷病留到第四階段。

出界後由最近的球員在 2–5 秒後於出界點重啟，並用 v1 的重啟模式（短傳為主）決策；進球後 30 秒開球。
"""
from __future__ import annotations

import numpy as np

from ..engine import events as E
from ..engine.pitch import CY, GOAL_HALF, LENGTH, WIDTH
from . import phase
from .world import Intent, Mode, Phase, World

GOAL_HEIGHT = 2.44


def _finalize_pending_fail(w: World) -> None:
    pend = getattr(w, "pending", None)
    if pend is not None:
        w.add_event(pend["slot"], pend["type"], E.FAIL, pend["start"], w.ball[:2].copy(), team=pend["team"],
                    **pend.get("kw", {}))
        w.pending = None


def goal_scored(w: World, T: int, scorer: int | None) -> None:
    w.score[T] += 1
    _finalize_pending_fail(w)
    start_restart(w, "kickoff", 1 - T, LENGTH / 2, CY, w.sp.rules.goal_delay)


def start_restart(w: World, kind: str, team: int, x: float, y: float, delay: float) -> None:
    _finalize_pending_fail(w)
    w.mode, w.owner, w.flight, w.carry, w.override = Mode.DEAD, None, None, None, {}
    w.restart = {"kind": kind, "team": team, "x": float(x), "y": float(y), "t": w.clock + delay}
    w.ball[:2], w.ball[2] = (x, y), 0.0
    w.bvel[:] = 0.0
    w.poss = team
    w.phase = [Phase.IP if t == team else Phase.OOP for t in (0, 1)]
    w.phase_t = [0.0, 0.0]
    w.counter_from = [None, None]
    w.next_shape = w.clock
    w.next_intent = w.clock
    w.fast_shape = True
    taker = _taker(w, kind, team, np.array([x, y]))        # 發球者走向發球點（不瞬移）
    w.restart["taker"] = taker
    w.override[taker] = (Intent.CHASE, np.array([x, y]))
    gap = float(np.hypot(*(w.pos[taker] - np.array([x, y]))))
    w.restart["t"] = max(w.restart["t"], w.clock + gap / 4.5 + 1.0)   # 至少給發球者以約 4.5 m/s 跑到的時間


def check_out(w: World) -> None:
    """球越過邊線或底線：進球、角球、球門球、界外球"""
    x, y = float(w.ball[0]), float(w.ball[1])
    if 0.0 < x < LENGTH and 0.0 < y < WIDTH:
        return
    lt = w.last_touch
    team = int(w.team_of[lt]) if lt is not None else w.poss
    delay = float(w.sim.rng_ref.uniform(*w.sp.rules.restart_delay))
    if x >= LENGTH or x <= 0.0:
        atk = 0 if x >= LENGTH else 1           # 這個球門被誰攻擊
        # 進球 = 射門（或擋出後的無主球、烏龍）越過球門線；傳球、解圍的飛行越過球門線算出界
        passing_flight = w.mode == Mode.FLIGHT and w.flight is not None and w.flight.kind != "shot"
        if abs(y - CY) <= GOAL_HALF and w.ball[2] < GOAL_HEIGHT and not passing_flight:
            goal_scored(w, atk, lt)
            return
        if team == atk:                          # 攻方出底線：球門球
            gx = LENGTH - 5.5 if atk == 0 else 5.5
            start_restart(w, "goal_kick", 1 - atk, gx, CY, delay)
        else:                                    # 守方碰出底線：攻方角球
            cx = LENGTH - 0.5 if atk == 0 else 0.5          # 擺在界內 0.5 m（剛好在線上會被判出界）
            cy = 0.5 if y < CY else WIDTH - 0.5
            start_restart(w, "corner_short", atk, cx, cy, delay)
    else:
        by = 0.5 if y <= 0.0 else WIDTH - 0.5
        start_restart(w, "throw_in", 1 - team, float(np.clip(x, 1.0, LENGTH - 1.0)), by, delay)


def _taker(w: World, kind: str, T: int, spot: np.ndarray) -> int:
    gk = w.sim.gk_slot(T)
    if kind == "goal_kick" and gk is not None:
        return gk
    cand = [s for s in w.active(T) if s != gk] or w.active(T)
    d = np.hypot(w.pos[cand, 0] - spot[0], w.pos[cand, 1] - spot[1])
    return cand[int(np.argmin(d))]


def perform_restart(w: World) -> None:
    r = w.restart
    T, kind = r["team"], r["kind"]
    spot = np.array([r["x"], r["y"]])
    if kind == "kickoff":
        for s in w.active(0) + w.active(1):          # 開球時雙方都在本方半場
            tt = int(w.team_of[s])
            p = w.tf(tt, w.pos[s])
            if p[0] > LENGTH / 2:
                p[0] = LENGTH / 2 - 1.0
                w.pos[s] = w.tf(tt, p)
                w.vel[s] = 0.0
    taker = r["taker"]
    gap = float(np.hypot(*(w.pos[taker] - spot)))
    if gap > 2.0 and w.clock < r["t"] + 20.0:              # 發球者還沒走到：再等（最多 20 s）
        return
    from .play import give_ball
    w.pos[taker] = spot                                    # 最後 2 m 以內直接就位
    w.vel[taker] = 0.0
    w.restart = None
    w.restart_mode = {"kickoff": "kickoff", "goal_kick": "goal_kick", "throw_in": "throw_in",
                      "corner_short": "corner_short"}[kind]
    give_ball(w, taker, "restart")


def kickoff_teleport(w: World) -> None:
    """全場開球前：球員直接站到陣型位置（本方半場內）"""
    from .shape import update_targets
    update_targets(w)
    for s in w.active(0) + w.active(1):
        tt = int(w.team_of[s])
        p = w.tf(tt, w.tgt[s])
        p[0] = min(p[0], LENGTH / 2 - 1.0)
        w.pos[s] = w.tf(tt, p)
        w.vel[s] = 0.0
