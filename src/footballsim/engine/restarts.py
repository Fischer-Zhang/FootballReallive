"""定位球與重新開始（§4.11）：開球、界外球、球門球、門將手中球、角球、自由球、十二碼"""
from __future__ import annotations

import math

import numpy as np

from . import actions, decision, shape
from . import events as E
from .core_math import logit, shot_geometry, sigmoid
from .decision import aerial_p, aerial_score, shot_model
from .pitch import CY, LENGTH, PEN_SPOT, WIDTH, shot_context
from .state import Outcome, Restart

_TIMING_KEY = {"kickoff": "kickoff", "throw_in": "throw_in", "goal_kick": "goal_kick",
               "gk_hands": "gk_hands", "corner": "corner", "free_kick": "free_kick",
               "penalty": "penalty"}


def _dead_time(sim, key: str, team: int) -> float:
    tm = sim.p.timing
    lo, hi = getattr(tm, key)
    sec = float(sim.rng_misc.uniform(lo, hi))
    me, op = sim.teams[team], sim.teams[1 - team]
    if sim.match_minute() >= 75 and me.score > op.score:       # 領先方拖延時間
        sec *= tm.time_wasting
        sim.extra_dead[sim.period - 1] += sec * (tm.time_wasting - 1) / tm.time_wasting
    return sec


def _is_direct_fk(sim, bx: float, by: float) -> bool:
    fk = sim.p.free_kick
    return (bx >= fk.direct_min_x and abs(by - CY) <= fk.direct_max_wide
            and math.hypot(LENGTH - bx, by - CY) <= fk.direct_max_dist)


def perform(sim, r: Restart) -> Outcome:
    T = r.team
    bx, by = sim.to_frame(T, r.x, r.y)
    key = _TIMING_KEY[r.kind]
    if r.kind == "free_kick" and _is_direct_fk(sim, bx, by):
        key = "free_kick_danger"
    sim.advance_dead(_dead_time(sim, key, T))
    sim.poss = T
    if r.kind == "corner":
        return _delivery(sim, T, LENGTH - 0.5, 0.5 if by < CY else WIDTH - 0.5, E.CORNER, corner=True)
    if r.kind == "penalty":
        return _penalty(sim, T)
    if r.kind == "free_kick" and _is_direct_fk(sim, bx, by):
        return _direct_free_kick(sim, T, bx, by)

    if r.kind == "kickoff":
        bx, by = 52.5, CY
    elif r.kind == "goal_kick":
        bx, by = 5.5, CY
    elif r.kind == "throw_in":
        by = 0.5 if by < CY else WIDTH - 0.5
    carrier = _taker(sim, T, r.kind, bx, by)
    mode = {"kickoff": "kickoff", "throw_in": "throw_in", "goal_kick": "goal_kick",
            "gk_hands": "gk_hands", "free_kick": "free_kick"}[r.kind]
    sim.carrier = carrier
    sim.ball = sim.to_global(T, bx, by)
    pos, off = shape.compute(sim, T, bx, by, carrier, set_piece=r.kind in ("free_kick", "kickoff"))
    opt = decision.choose(sim, T, carrier, pos, off, mode)
    return actions.resolve(sim, T, carrier, pos, off, opt, mode)


def _taker(sim, T: int, kind: str, bx: float, by: float) -> int:
    act = sim.active_slots(T)
    g = sim.gk_slot(T)
    if kind in ("goal_kick", "gk_hands") and g is not None:
        return g
    out = [s for s in act if s != g] or act
    if kind == "kickoff":
        st = [s for s in out if sim.teams[T].slot_pos[s % 11].value in ("ST", "AM")]
        return st[0] if st else out[-1]
    pos, _ = shape.compute(sim, T, bx, by, None, set_piece=True)
    d = np.hypot(pos[out, 0] - bx, pos[out, 1] - by)
    return out[int(np.argmin(d))]


def _penalty(sim, T: int) -> Outcome:
    pp, rng, C = sim.p.penalty, sim.rng_act, sim.C
    D = 1 - T
    out = [s for s in sim.active_slots(T) if not sim.is_gk(s)]
    taker = max(out, key=lambda s: sim.comp[s, C.PENALTY])
    g = sim.gk_slot(D)
    gkc = sim.comp[g, C.GK_PENALTY] if g is not None else 20.0
    z = (logit(pp.base) + pp.k_att * (sim.comp[taker, C.PENALTY] - sim.R) / 10
         - pp.k_gk * (gkc - sim.R) / 10 - pp.importance * (sim.ctx.importance - 0.3))
    p_goal = sigmoid(z)
    sim.extra_dead[sim.period - 1] += sim.p.stoppage.penalty
    px, py = PEN_SPOT
    sim.carrier = taker
    sim.ball = sim.to_global(T, px, py)
    pos, _ = shape.compute(sim, T, px, py, taker, set_piece=True)
    gy = CY + float(rng.uniform(-3.3, 3.3))
    if rng.random() < p_goal:
        evs = [sim.ev(T, taker, E.SHOT_PENALTY, E.GOAL, px, py, LENGTH, gy, xg=pp.base)]
        return Outcome(duration=2.0, events=evs, restart=Restart("kickoff", D, 52.5, CY), goal=T)
    if g is not None and rng.random() < pp.save_share:
        evs = [sim.ev(T, taker, E.SHOT_PENALTY, E.SAVED, px, py, LENGTH, gy, xg=pp.base),
               sim.ev(T, g, E.KEEPER_SAVE, E.SUCCESS, LENGTH - 1.0, gy, xg=pp.base)]
        u = rng.random()
        if u < 0.5:
            return Outcome(duration=2.0, events=evs, restart=actions.gk_hands(sim, D))
        if u < 0.75:
            return Outcome(duration=2.0, events=evs, restart=actions.corner(sim, T, gy))
        return actions.loose_ball(sim, T, pos, LENGTH - 8 + rng.normal(0, 2), CY + rng.normal(0, 4),
                                  2.5, evs)
    evs = [sim.ev(T, taker, E.SHOT_PENALTY, E.OFF_TARGET, px, py, LENGTH, CY + 5.0, xg=pp.base)]
    return Outcome(duration=2.0, events=evs, restart=actions.goal_kick(sim, D))


def _direct_free_kick(sim, T: int, bx: float, by: float) -> Outcome:
    fk, rng, C = sim.p.free_kick, sim.rng_act, sim.C
    central = abs(by - CY) <= 12
    if rng.random() >= (fk.direct_share if central else 0.25):
        return _delivery(sim, T, bx, by, E.FREEKICK_CROSSED, corner=False)
    out = [s for s in sim.active_slots(T) if not sim.is_gk(s)]
    taker = max(out, key=lambda s: sim.comp[s, C.FREE_KICK])
    D = 1 - T
    g = sim.gk_slot(D)
    dist = math.hypot(LENGTH - bx, by - CY)
    _, ang = shot_geometry(LENGTH - bx, abs(by - CY))
    xg = sigmoid(fk.b0 - fk.b_dist * dist) * min(1.0, ang / 0.3)
    sim.carrier = taker
    sim.ball = sim.to_global(T, bx, by)
    pos, _ = shape.compute(sim, T, bx, by, taker, set_piece=True)
    gkc = sim.comp[g, C.GK_SAVE] if g is not None else 20.0
    return actions.resolve_shot(sim, T, taker, pos, bx, by, xg, fk.wall_block,
                                sim.comp[taker, C.FREE_KICK], gkc, E.SHOT_FREEKICK, "foot", [], dur=1.5)


def _delivery(sim, T: int, bx: float, by: float, etype: str, corner: bool) -> Outcome:
    """角球／間接自由球傳中：落點品質 → 門將出擊 → 禁區爭頂 → 頭槌或解圍"""
    co, rng, C = sim.p.corner, sim.rng_act, sim.C
    D = 1 - T
    out = [s for s in sim.active_slots(T) if not sim.is_gk(s)]
    taker = max(out, key=lambda s: sim.comp[s, C.DELIVERY])
    if corner and rng.random() < co.short_share:
        cy = min(max(by, 1.0), WIDTH - 1.0)
        sim.carrier = taker
        sim.ball = sim.to_global(T, LENGTH - 1.0, cy)
        pos, off = shape.compute(sim, T, LENGTH - 1.0, cy, taker, set_piece=True)
        opt = decision.choose(sim, T, taker, pos, off, "corner_short")
        return actions.resolve(sim, T, taker, pos, off, opt, "corner_short")

    sim.carrier = taker
    sim.ball = sim.to_global(T, bx, by)
    pos, _ = shape.compute(sim, T, bx, by, taker, set_piece=True)
    att = sorted([s for s in out if s != taker], key=lambda s: -aerial_score(sim, s))[: int(co.n_attackers)]
    d_out = [s for s in sim.active_slots(D) if not sim.is_gk(s)]
    defs = sorted(d_out, key=lambda s: -aerial_score(sim, s))[:6]
    g = sim.gk_slot(D)
    side = 1.0 if by > CY else -1.0
    zone = int(rng.choice(3, p=[0.4, 0.3, 0.3]))
    tx, ty = ((100.5, CY + side * 4.0), (95.0, CY), (99.0, CY - side * 5.0))[zone]
    q = sigmoid(co.deliver_b0 + co.k * (sim.comp[taker, C.DELIVERY] - sim.R) / 10)
    dur = 2.5
    if not att or rng.random() >= q:
        evs = [sim.ev(T, taker, etype, E.FAIL, bx, by, tx, ty)]
        if rng.random() < co.out_share:
            return Outcome(duration=dur, events=evs, restart=actions.throw_in(sim, T, T, 90.0, ty))
        return actions.loose_ball(sim, T, pos, 84.0 + rng.normal(0, 3), CY + rng.normal(0, 10), dur, evs)
    w = np.array([0.5, 0.3, 0.2][: min(3, len(att))])
    a = att[int(rng.choice(len(w), p=w / w.sum()))]
    if g is not None:
        p_claim = sigmoid(co.claim_b0 + sim.p.cross.claim_k * (sim.comp[g, C.GK_CLAIM] - sim.R) / 10
                          + (0.5 if zone == 0 else 0.0))
        if rng.random() < p_claim:
            evs = [sim.ev(T, taker, etype, E.FAIL, bx, by, tx, ty),
                   sim.ev(T, g, E.KEEPER_CLAIM, E.SUCCESS, tx, ty)]
            return Outcome(duration=dur, events=evs, restart=actions.gk_hands(sim, D))
    k = defs[int(rng.integers(min(3, len(defs))))] if defs else None
    p_aer = aerial_p(sim, a, k, co.crowd) if k is not None else 0.8
    if rng.random() < p_aer:
        evs = [sim.ev(T, taker, etype, E.SUCCESS, bx, by, tx, ty, extra={"recv": sim.pid(a)}),
               sim.ev(T, a, E.AERIAL, E.SUCCESS, tx, ty, opponent=sim.pid(k))]
        if rng.random() >= co.header_shot_share:
            # 爭頂贏了但沒有直接攻門：頂給隊友或二點球
            return actions.loose_ball(sim, T, pos, tx - rng.uniform(3, 10), ty + rng.normal(0, 6), dur, evs)
        Dp = pos[d_out] if d_out else np.zeros((0, 2))
        n_cone, d_near, _ = shot_context(tx, ty, Dp)
        if k is not None:
            d_near = min(d_near, sim.p.cross.contest_dist)
        xg_h, pb_h, po_h = shot_model(sim, tx, ty, 1, n_cone, d_near)
        gkc = sim.comp[g, C.GK_SAVE] if g is not None else 20.0
        return actions.resolve_shot(sim, T, a, pos, tx, ty, xg_h, pb_h,
                                    sim.comp[a, C.HEADER_SHOT], gkc, E.SHOT, "head",
                                    [k] if k is not None else [], dur=dur, pre=evs, p_ot=po_h)
    evs = [sim.ev(T, taker, etype, E.FAIL, bx, by, tx, ty)]
    lx, ly = tx - rng.uniform(14, 24), ty + rng.normal(0, 8)
    if k is not None:
        evs.append(sim.ev(T, k, E.AERIAL, E.SUCCESS, tx, ty, opponent=sim.pid(a)))
        evs.append(sim.ev(T, k, E.CLEARANCE, E.SUCCESS, tx, ty, lx, ly, body="head"))   # 頭槌解圍
    if rng.random() < co.repeat_corner:
        return Outcome(duration=dur, events=evs, restart=actions.corner(sim, T, ty))
    return actions.loose_ball(sim, T, pos, lx, ly, dur, evs)
