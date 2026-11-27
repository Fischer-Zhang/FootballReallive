"""球員之間的接觸與動作：決策（v1 評分函式）、起腳、飛行中的攔截、接球控制、無主球、搶斷、過人、射門。

傳球、攔截、搶斷由「到達時間差與距離」決定結果（物理）；v1 擬合的成功率只用在決策時的預期與驗證。
射門沿用 v1 的三階段結果（封堵、射正、門將），球的彈道依結果動畫。
"""
from __future__ import annotations

import math

import numpy as np

from ..engine import decision as v1dec
from ..engine import events as E
from ..engine import shape as v1shape
from ..engine.core_math import shot_outcome_probs, sigmoid
from ..engine.pitch import CY, GOAL_HALF, LENGTH, WIDTH
from ..model.attributes import ATTR
from . import phase
from .physics import flight_state, ground_duration, seg_point_dist
from .world import Flight, Intent, Mode, World


# ------------------------------------------------------------------ 小工具
def _clip_lin(a: float, b: float, cap: float, x: float, lo: float | None = None) -> float:
    v = a + b * x
    return float(min(max(v, lo if lo is not None else a), cap))


def pressure_on(w: World, slot: int, radius: float = 2.0) -> float:
    """附近對手造成的壓力（高斯加權人數）"""
    T = int(w.team_of[slot])
    opp = w.active(1 - T)
    d = np.hypot(w.pos[opp, 0] - w.pos[slot, 0], w.pos[opp, 1] - w.pos[slot, 1])
    return float(np.exp(-d * d / (2 * radius * radius)).sum())


def give_ball(w: World, slot: int, how: str) -> None:
    """slot 取得控球權（接球、攔截、搶斷、撲救後）。處理待結算的傳球事件、階段轉換與狀態重置"""
    T = int(w.team_of[slot])
    pend = getattr(w, "pending", None)
    if pend is not None:
        ok = T == pend["team"]
        w.add_event(pend["slot"], pend["type"], E.SUCCESS if ok else E.FAIL, pend["start"], w.pos[slot].copy(),
                    team=pend["team"], **pend.get("kw", {}))
        w.pending = None
    if how == "interception":
        w.add_event(slot, E.INTERCEPTION, E.SUCCESS, w.pos[slot].copy())
    elif how == "recovery":
        w.add_event(slot, E.RECOVERY, E.SUCCESS, w.pos[slot].copy())
    vb = w.bvel[:2].copy()                                # 球到達時的運動方向：持球者順勢帶球的方向
    w.touch_dir = vb / np.hypot(*vb) if float(np.hypot(*vb)) > 1.0 else np.zeros(2)
    phase.possession_change(w, T, slot, how)
    w.mode, w.owner, w.flight = Mode.CARRIED, slot, None
    w.ball[:2], w.ball[2] = w.pos[slot], 0.0
    w.bvel[:] = 0.0
    w.last_touch, w.poss = slot, T
    w.carry = None
    w.override = {}
    w.next_shape = w.clock                       # 球權改變：陣型立即重算
    w.next_intent = w.clock
    if getattr(w, '_prev_poss', None) != T:
        w.fast_shape = True
    w._prev_poss = T


# ------------------------------------------------------------------ 決策（v1 評分函式）
def decide(w: World, c: int, mode=None):
    sim = w.sim
    T = int(w.team_of[c])
    pos_T = w.tf(T, w.pos)
    off = v1shape.offside_line(pos_T, w.active(1 - T))
    hold = sim.draw_hold()
    if w.phase[T].name == "AT":
        hold *= w.sp.phase.at_hold_factor
    sim.poss, sim.carrier, sim.ball = T, c, (float(w.ball[0]), float(w.ball[1]))
    sim._hold = hold
    # v1 的決策用 vmax 與 tau 估計「誰先到」（跑動路線傳球、直塞、帶球封堵）。換成 v2 實際的運動能力
    # （最高速 × vmax_scale、起步加速較慢），決策才不會選接球者實際到不了的傳球。
    vm, ta = sim.vmax, sim.tau
    sim.vmax, sim.tau = vm * w.sp.motion.vmax_scale, ta * w.sp.motion.decision_tau_scale
    try:
        opt = v1dec.choose(sim, T, c, pos_T, off, mode)
    finally:
        sim._hold = None
        sim.vmax, sim.tau = vm, ta
    return opt, hold


def carrier_step(w: World) -> None:
    """持球者：接球後先持球一段時間（持球時間，期間帶著球慢慢走），時間到才用「當下」的狀態決策。
    v1 的行為模型（射門、帶球、傳球傾向）擬合的是真實動作發生那一刻的狀態（最近對手距離、空間），不是接球那一刻；
    接球後 1–2 秒防守者已經逼近，所以在接球當下決策會高估空間與射門機會。
    決策為傳球／射門：動作時間（windup）後起腳；決策為帶球：跑到終點後立刻再決策（不再等待）。"""
    c = w.owner
    if c is None or w.mode != Mode.CARRIED:
        return
    sp, T = w.sp, int(w.team_of[c])
    st = w.carry
    if st is None:
        mode = getattr(w, "restart_mode", None)
        w.restart_mode = None
        hold = w.sim.draw_hold()
        if w.phase[T].name == "AT":
            hold *= sp.phase.at_hold_factor
        st = {"kind": "wait", "mode": mode, "until": w.clock + hold + sp.rules.hold_reaction}
        w.carry = st
    kind = st["kind"]
    if kind == "wait":
        w.override[c] = (Intent.WAIT, _drift_target(w, c, T))
        if w.clock >= st["until"]:
            mode = st.get("mode")
            w.kick_mode = mode                         # 重啟後的第一次出球要記成界外球、球門球、角球等事件
            opt, hold = decide(w, c, mode)
            if opt.kind in ("carry", "take_on"):
                end = w.tf(T, np.array([opt.x, opt.y]))
                w.carry = st = {"kind": opt.kind, "opt": opt, "end": end, "t_end": w.clock + max(hold, 1.0) * 1.6 + 1.0,
                                "k": opt.target if opt.kind == "take_on" else None}
            else:
                w.carry = st = {"kind": "windup", "opt": opt, "until": w.clock + sp.rules.windup,
                                "rpos0": w.pos[opt.target].copy() if opt.target is not None else None}
            kind = st["kind"]
    if kind in ("carry", "take_on"):
        w.override[c] = (Intent.CARRY, st["end"])
        if kind == "take_on":
            k = st["k"]
            if k is not None and np.hypot(*(w.pos[c] - w.pos[k])) <= sp.contest.tackle_range * 1.1:
                _duel(w, c, k, st["opt"])
                return
        if np.hypot(*(w.pos[c] - st["end"])) < 0.9 or w.clock > st["t_end"]:
            w.carry = {"kind": "wait", "mode": None, "until": w.clock}       # 帶完一段：馬上再決策
    elif kind == "windup":
        w.override[c] = (Intent.WAIT, _drift_target(w, c, T))
        if w.clock >= st["until"]:
            execute(w, c, st["opt"], st.get("rpos0"))


def _drift_target(w: World, c: int, T: int) -> np.ndarray:
    """持球等待出手時，持球者帶著球順勢走（追蹤資料：接球後持球者的速度中位數 1.9 → 3.3 m/s，不會站著不動；
    但「向前動量」只有 2%，所以不是往球門衝）：方向是球到達的方向與目前移動方向的混合，球不是傳來的（重啟）就原地。"""
    td = getattr(w, "touch_dir", np.zeros(2))
    v = w.vel[c]
    sp = float(np.hypot(*v))
    d = td if sp < 0.5 else 0.5 * td + 0.5 * v / sp
    n = float(np.hypot(*d))
    if n < 1e-6:
        return w.pos[c].copy()
    return w.pos[c] + d / n * w.sp.rules.wait_drift


def _duel(w: World, c: int, k: int, opt) -> None:
    """過人：與目標防守者接觸的瞬間，以 v1 的過人勝率（屬性對抗的擬合模型）決勝"""
    pc, ck = w.pos[c].copy(), w.pos[k].copy()
    if w.sim.rng_act.random() < opt.p:
        w.add_event(c, E.TAKE_ON, E.SUCCESS, pc, opponent=w.pid(k))
        w.stun[k] = w.sp.contest.duel_stun
        w.carry["kind"] = "carry"
        w.carry["k"] = None
    else:
        w.add_event(c, E.TAKE_ON, E.FAIL, pc, opponent=w.pid(k))
        w.add_event(k, E.TACKLE, E.SUCCESS, ck, opponent=w.pid(c))
        give_ball(w, k, "tackle")


def tackle_step(w: World, dt: float) -> None:
    """持球者被逼近時，範圍內的防守者以每秒 tackle_rate 的頻率嘗試搶斷"""
    c = w.owner
    if c is None or w.mode != Mode.CARRIED:
        return
    cp = w.sp.contest
    T = int(w.team_of[c])
    opp = np.array(w.active(1 - T), int)
    d = np.hypot(w.pos[opp, 0] - w.pos[c, 0], w.pos[opp, 1] - w.pos[c, 1])
    ok = (d <= cp.tackle_range) & (w.stun[opp] <= 0)
    if w.carry is not None and w.carry["kind"] == "take_on":
        ok &= opp != w.carry["k"]                       # 過人對象由決勝處理
    if not ok.any():
        return
    sim = w.sim
    for k in opp[ok]:
        if sim.rng_act.random() < cp.tackle_rate * dt:
            p = sigmoid(cp.tackle_b0 + cp.tackle_k * (sim.comp[k, sim.C.TACKLE] - sim.comp[c, sim.C.RETAIN]) / 10.0)
            if sim.rng_act.random() < p:
                w.add_event(k, E.TACKLE, E.SUCCESS, w.pos[k].copy(), opponent=w.pid(c))
                give_ball(w, int(k), "tackle")
                return


# ------------------------------------------------------------------ 動作
def execute(w: World, c: int, opt, rpos0=None) -> None:
    w.carry = None
    if opt.kind == "shot":
        shoot(w, c, opt)
    else:
        kick(w, c, opt, rpos0)


def kick(w: World, c: int, opt, rpos0=None) -> None:
    sim, sp = w.sim, w.sp
    T = int(w.team_of[c])
    start = w.pos[c].copy()
    aim = w.tf(T, np.array([opt.x, opt.y]))
    if rpos0 is not None and opt.target is not None and opt.kind != "clear":
        # 決策到起腳隔了持球時間，接球者已經移動：目標點跟著接球者的位移（保留決策時的相對位置，例如往前傳到他前方）
        aim = aim + (w.pos[opt.target] - rpos0)
    d = float(np.hypot(*(aim - start)))
    kind = opt.kind
    if kind == "cross" or kind == "clear":
        lofted = True
    elif kind == "through":
        lofted = d > 35.0
    else:
        q = float(opt.data[0]) if opt.data is not None and np.isscalar(opt.data[0]) else 0.0
        lofted = bool(sim.rng_act.random() < q)
    C = sim.C
    if kind == "through":
        skill = sim.comp[c, C.THROUGH_PASS]
    elif kind == "cross":
        skill = sim.comp[c, C.CROSS]
    elif sim.is_gk(c):
        skill = sim.comp[c, C.GK_KICK] if (lofted or d > 25) else sim.comp[c, C.GK_SHORT]
    else:
        skill = sim.comp[c, C.LONG_PASS] if (lofted or d > 25) else sim.comp[c, C.SHORT_PASS]
    pe = sp.ball.pass_err
    sigma = (pe[0] + pe[1] * d) * (1.5 - skill / 100.0) * (1.0 + 0.5 * pressure_on(w, c)) * (1.3 if lofted else 1.0)
    end = aim + sim.rng_act.normal(0.0, sigma, 2)
    L = float(np.hypot(*(end - start)))
    b = sp.ball
    if lofted:
        vh = _clip_lin(b.loft_vh[0], b.loft_vh[1], b.loft_vh[2], L)
        dur = max(L / vh, 0.3)
        hmax = _clip_lin(b.loft_height[0], b.loft_height[1], b.loft_height[2], L, lo=2.0)
        f = Flight(kind, c, T, start, end, w.t, dur, vh, True, hmax, opt.target, aim)
    else:
        v0 = _clip_lin(b.pass_v0[0], b.pass_v0[1], b.pass_v0[2], L)
        f = Flight(kind, c, T, start, end, w.t, ground_duration(max(L, 0.5), v0, b.ground_decel), v0, False, 0.0,
                   opt.target, aim, decel=b.ground_decel)
    if kind == "clear":
        w.add_event(c, E.CLEARANCE, E.SUCCESS, start, end, team=T)
        w.pending = None
    else:
        etype = {"pass": E.PASS, "cross": E.CROSS, "through": E.THROUGH}[kind]
        km = getattr(w, "kick_mode", None)
        w.kick_mode = None
        etype = {"throw_in": E.THROW_IN, "goal_kick": E.GOAL_KICK, "corner_short": E.CORNER_SHORT,
                 "kickoff": E.KICKOFF, "gk_hands": E.GK_DISTRIBUTION}.get(km, etype)
        w.pending = {"slot": c, "team": T, "type": etype, "start": start.copy()}
    w.mode, w.owner, w.flight, w.last_touch = Mode.FLIGHT, None, f, c
    w.override = {}
    if opt.target is not None and kind != "clear":
        w.override[int(opt.target)] = (Intent.RECEIVE, end)
    w.next_shape = w.clock


def shoot(w: World, c: int, opt) -> None:
    sim, sp = w.sim, w.sp
    T = int(w.team_of[c])
    D = 1 - T
    xg, p_block, one, shooter, gkc, blockers, p_ot, n_cone, d_near = opt.data[:9]
    shp = sim.p.shot
    shooter = shooter - shp.weak_foot_share * shp.weak_foot_penalty * (1 - sim.weak[c] / 100.0)
    pr = shot_outcome_probs(xg, shooter, gkc, p_block, p_ot, shp.k_ot, shp.k_place, shp.k_gk, sim.R)
    u = float(sim.rng_act.random())
    p_goal, p_on = pr["goal"], pr["on_target"]
    if u < pr["blocked"]:
        res = E.BLOCKED
    elif u < pr["blocked"] + p_goal:
        res = E.GOAL
    elif u < pr["blocked"] + p_on:
        res = E.SAVED
    else:
        res = E.OFF_TARGET
    start = w.pos[c].copy()
    goal_x = LENGTH if T == 0 else 0.0
    gk = sim.gk_slot(D)
    if res == E.GOAL:
        end = np.array([goal_x, CY + float(sim.rng_act.uniform(-GOAL_HALF + 0.4, GOAL_HALF - 0.4))])
    elif res == E.SAVED:
        gp = w.pos[gk] if gk is not None else np.array([goal_x, CY])
        end = gp + sim.rng_act.normal(0.0, 0.4, 2)
    elif res == E.BLOCKED:
        bl = [b for b in blockers] if blockers else []
        if bl:
            k = min(bl, key=lambda s: np.hypot(*(w.pos[s] - start)))
            frac = 0.92
            end = start + (w.pos[k] - start) * frac
        else:
            end = start + (np.array([goal_x, CY]) - start) * 0.2
    else:
        side = 1.0 if sim.rng_act.random() < 0.5 else -1.0
        end = np.array([goal_x, CY + side * float(sim.rng_act.uniform(GOAL_HALF + 0.5, GOAL_HALF + 8.0))])
    L = float(np.hypot(*(end - start)))
    b_ = sp.ball
    v = min(b_.shot_speed[0] + b_.shot_speed[1] * sim.comp[c, sim.C.LONG_SHOT if L > 18 else sim.C.FINISHING],
            b_.shot_speed[2])
    f = Flight("shot", c, T, start, end, w.t, max(L / v, 0.15), v, False, 0.0, None, end.copy(),
               shot={"result": res, "xg": xg})
    w.add_event(c, E.SHOT, res, start, end, team=T, xg=float(xg))
    w.pending = None
    w.mode, w.owner, w.flight, w.last_touch = Mode.FLIGHT, None, f, c
    w.override = {}
    w.next_shape = w.clock


# ------------------------------------------------------------------ 飛行中與到達
def flight_step(w: World, prev: np.ndarray, cur: np.ndarray, dt: float) -> None:
    f = w.flight
    if f is None:
        return
    sim, cp = w.sim, w.sp.contest
    if f.kind != "shot":
        opp = np.array([s for s in w.active(1 - f.team) if s not in f.tried], int)
        if len(opp):
            if f.lofted and w.ball[2] > cp.lofted_clear_z:
                pass
            else:
                reach = cp.intercept_reach + 0.012 * (sim.comp[opp, sim.C.INTERCEPTION] - sim.R)
                dseg = seg_point_dist(prev, cur, w.pos[opp])
                hit = opp[dseg <= reach]
                speed = float(np.hypot(*w.bvel[:2]))
                for s in sorted(hit, key=lambda s: float(seg_point_dist(prev, cur, w.pos[[s]])[0])):
                    f.tried.add(int(s))
                    p = sigmoid(cp.intercept_b0 + cp.intercept_k * (sim.comp[s, sim.C.INTERCEPTION] - sim.R) / 10.0
                                - 0.10 * (speed - 12.0))
                    if sim.rng_act.random() < p:
                        w.ball[:2] = w.pos[s]
                        give_ball(w, int(s), "interception")
                        return
    if w.t + dt >= f.t0 + f.dur - 1e-9:
        arrival(w)


def arrival(w: World) -> None:
    f = w.flight
    sim, cp, b = w.sim, w.sp.contest, w.sp.ball
    w.ball[:2], w.ball[2] = f.end, 0.0
    if f.kind == "shot":
        _shot_end(w, f)
        return
    speed = float(np.hypot(*w.bvel[:2]))
    allp = np.array(w.active(0) + w.active(1), int)
    d = np.hypot(w.pos[allp, 0] - f.end[0], w.pos[allp, 1] - f.end[1])
    R_c = np.where(allp == f.receiver, b.control_radius + 0.6, b.control_radius)
    cand = allp[d <= R_c]
    if len(cand):
        dc = np.hypot(w.pos[cand, 0] - f.end[0], w.pos[cand, 1] - f.end[1])
        react = cp.free_react - 0.0025 * (sim.eff[cand, ATTR.ANTICIPATION] - sim.R)
        tarr = dc / np.maximum(0.8 * sim.vmax[cand], 1e-9) + react - np.where(cand == f.receiver, 0.25, 0.0)
        for i in np.argsort(tarr):
            s = int(cand[i])
            same = int(w.team_of[s]) == f.team
            ft = sim.eff[s, ATTR.FIRST_TOUCH] if same else sim.comp[s, sim.C.INTERCEPTION]
            P = pressure_on(w, s, 1.5)
            z = (cp.control_b0 + cp.control_k * (ft - sim.R) / 10.0 - cp.control_speed * (speed - cp.control_speed_ref)
                 - cp.control_press * P)
            if sim.rng_act.random() < sigmoid(z):
                give_ball(w, s, "recovery" if not same else "pass")
                return
            break
    # 誰都沒控制住：球繼續滾（落點附近彈開）
    head = math.atan2(f.end[1] - f.start[1], f.end[0] - f.start[0])      # 球大致沿原方向繼續滾（不會朝任意方向亂彈）
    ang = head + float(sim.rng_act.uniform(-1.0, 1.0))
    sp_ = float(sim.rng_act.uniform(*b.loose_speed)) if f.lofted else max(speed * 0.5, 1.5)
    w.mode, w.owner, w.flight = Mode.FREE, None, None
    w.bvel[:2] = [math.cos(ang) * sp_, math.sin(ang) * sp_]
    w.override = {}


def _shot_end(w: World, f: Flight) -> None:
    sim = w.sim
    res = f.shot["result"]
    T, D = f.team, 1 - f.team
    if res == E.GOAL:
        from .rules import goal_scored
        goal_scored(w, T, f.kicker)
    elif res == E.SAVED:
        gk = sim.gk_slot(D)
        if gk is not None:
            w.add_event(gk, E.KEEPER_SAVE, E.SUCCESS, w.pos[gk].copy(), team=D)
        if gk is not None and sim.rng_act.random() < 0.7:
            w.restart_mode = "gk_hands"
            give_ball(w, gk, "recovery")
        else:                                       # 擋出：球彈向側面
            ang = float(sim.rng_act.uniform(-1.0, 1.0)) + (math.pi if T == 0 else 0.0)
            w.mode, w.owner, w.flight = Mode.FREE, None, None
            w.bvel[:2] = [math.cos(ang) * 5.0, math.sin(ang) * 5.0]
            w.override = {}
    elif res == E.BLOCKED:                          # 被擋下的球往射門者那一側彈回
        back = math.atan2(f.start[1] - f.end[1], f.start[0] - f.end[0])
        ang = back + float(sim.rng_act.uniform(-1.0, 1.0))
        w.mode, w.owner, w.flight = Mode.FREE, None, None
        w.bvel[:2] = [math.cos(ang) * 4.0, math.sin(ang) * 4.0]
        w.override = {}
    else:                                           # 射偏：球出底線（出界由 rules 判定）
        w.mode, w.owner, w.flight = Mode.FREE, None, None
        w.bvel[:2] = [3.0 if T == 0 else -3.0, 0.0]
        w.override = {}


def free_step(w: World, dt: float) -> None:
    """無主球：範圍內第一個到的人嘗試控制；失敗則球彈開"""
    sim, b = w.sim, w.sp.ball
    allp = np.array(w.active(0) + w.active(1), int)
    d = np.hypot(w.pos[allp, 0] - w.ball[0], w.pos[allp, 1] - w.ball[1])
    cand = allp[d <= b.loose_radius]
    if not len(cand):
        return
    dc = d[d <= b.loose_radius]
    s = int(cand[int(np.argmin(dc))])
    speed = float(np.hypot(*w.bvel[:2]))
    z = (w.sp.contest.control_b0 + w.sp.contest.control_k * (sim.comp[s, sim.C.LOOSE_BALL] - sim.R) / 10.0
         - 0.12 * (speed - 4.0))
    if sim.rng_act.random() < sigmoid(z):
        give_ball(w, s, "recovery" if int(w.team_of[s]) != w.poss or w.last_touch is None
                  or int(w.team_of[w.last_touch]) != int(w.team_of[s]) else "pass")
    else:
        ang = float(sim.rng_act.uniform(0, 2 * math.pi))
        sp_ = float(sim.rng_act.uniform(*b.loose_speed)) * 0.6
        w.bvel[:2] = [math.cos(ang) * sp_, math.sin(ang) * sp_]
