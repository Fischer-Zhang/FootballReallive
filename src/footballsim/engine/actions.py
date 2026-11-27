"""動作解析：決策 → 屬性對抗 → 結果（§4.7–§4.10）

所有座標參數都在持球隊 T 的本隊座標系；回傳 Outcome 時轉為全域座標。
"""
from __future__ import annotations

import math

import numpy as np

from ..model.attributes import ATTR
from . import events as E
from . import referee
from .core_math import shot_outcome_probs, sigmoid
from .decision import shot_model
from .pitch import BOX_HALF, BOX_X, CY, LENGTH, WIDTH, shot_context
from .state import Outcome, Restart

MODE_EVENT = {None: E.PASS, "kickoff": E.KICKOFF, "throw_in": E.THROW_IN, "goal_kick": E.GOAL_KICK,
              "free_kick": E.FREEKICK_SHORT, "corner_short": E.CORNER_SHORT,
              "gk_hands": E.GK_DISTRIBUTION}


# ------------------------------------------------------------------ 重新開始的位置
def goal_kick(sim, team: int) -> Restart:
    return Restart("goal_kick", team, *sim.to_global(team, 5.5, CY))


def corner(sim, team: int, y_in_team_frame: float) -> Restart:
    side = 0.5 if y_in_team_frame < CY else WIDTH - 0.5
    return Restart("corner", team, *sim.to_global(team, LENGTH - 0.5, side))


def gk_hands(sim, team: int, x: float = 6.0, y: float = CY) -> Restart:
    return Restart("gk_hands", team, *sim.to_global(team, x, y))


def throw_in(sim, team: int, frame: int, x: float, y: float) -> Restart:
    side = 0.5 if y < CY else WIDTH - 0.5
    return Restart("throw_in", team, *sim.to_global(frame, min(max(x, 1.0), LENGTH - 1.0), side))


# ------------------------------------------------------------------ 無主球
def loose_ball(sim, T: int, pos, lx: float, ly: float, dur: float, evs: list) -> Outcome:
    """無主球爭奪：距離 + 預判/加速/侵略性/投入度（§4.8 第二落點）"""
    lb = sim.p.loose_ball
    lx = min(max(lx, 0.5), LENGTH - 0.5)
    ly = min(max(ly, 0.5), WIDTH - 0.5)
    best = {}
    for team in (T, 1 - T):
        slots = [s for s in sim.active_slots(team)
                 if not sim.is_gk(s) or math.hypot(pos[s, 0] - lx, pos[s, 1] - ly) < 8.0]
        if not slots:
            continue
        P = pos[slots]
        score = -np.hypot(P[:, 0] - lx, P[:, 1] - ly) + lb.k * (sim.comp[slots, sim.C.LOOSE_BALL] - sim.R)
        i = int(np.argmax(score))
        best[team] = (float(score[i]), slots[i])
    if T in best and (1 - T) in best:
        p_att = sigmoid((best[T][0] - best[1 - T][0]) / lb.scale)
        team = T if sim.rng_act.random() < p_att else 1 - T
    else:
        team = T if T in best else 1 - T
    s = best[team][1]
    evs.append(sim.ev(T, s, E.RECOVERY, E.SUCCESS, lx, ly))
    return Outcome(duration=dur, events=evs, team=team, carrier=s, ball=sim.to_global(T, lx, ly))


def resolve(sim, T: int, c: int, pos, off_line: float, opt, mode=None) -> Outcome:
    kind = opt.kind
    if kind == "pass":
        return _pass(sim, T, c, pos, opt, mode)
    if kind == "through":
        return _through(sim, T, c, pos, opt)
    if kind == "carry":
        return _carry(sim, T, c, pos, opt)
    if kind == "take_on":
        return _take_on(sim, T, c, pos, opt)
    if kind == "shot":
        xg, p_block, one, shooter, gkc, blockers, p_ot = opt.data[:7]
        return resolve_shot(sim, T, c, pos, opt.x, opt.y, xg, p_block, shooter, gkc, E.SHOT, "foot",
                            blockers, weak_foot=True, one_on_one=one, p_ot=p_ot)
    if kind == "cross":
        return _cross(sim, T, c, pos, opt)
    return _clear(sim, T, c, pos, opt)


# ------------------------------------------------------------------ 傳球
def _pass(sim, T, c, pos, opt, mode) -> Outcome:
    pp, rng = sim.p.passing, sim.rng_act
    j = opt.target
    bx, by = float(pos[c, 0]), float(pos[c, 1])
    rx, ry = opt.x, opt.y
    # 傳球高度是技術選擇：以機率 q 踢高球（資料擬合），成功率依實際踢法；傳到跑動路線時再乘上追跑勝率
    q, lane, rp, thr, dfd, lead, p_g, p_l = opt.data[:8]
    lofted = bool(rng.random() < q)
    p_exec = p_l if lofted else p_g
    p_race = lead[0] if lead is not None else 1.0
    p_succ = p_exec * p_race
    dist = math.hypot(rx - bx, ry - by)
    dur = sim.decide_time() + pp.ctrl_time + dist / pp.ball_speed
    etype = MODE_EVENT.get(mode, E.PASS)
    extra = {"recv": sim.pid(j), "lofted": lofted}
    if lead is not None:
        extra["lead"] = True
    if rng.random() < p_succ:
        evs = [sim.ev(T, c, etype, E.SUCCESS, bx, by, rx, ry, extra=extra)]
        fo = sim.p.foul
        if rp > fo.recv_min and dfd:
            # 接球時的身體對抗：可能被犯規，也可能是進攻方犯規（爭頂推人）
            Dp = pos[dfd]
            k = dfd[int(np.argmin(np.hypot(Dp[:, 0] - rx, Dp[:, 1] - ry)))]
            if referee.receive_foul(sim, k, rp, lofted, rx, ry):
                if rng.random() < (fo.att_share_lofted if lofted else fo.att_share_ground):
                    return referee.attacker_foul(sim, T, j, k, rx, ry, dur, evs)
                return referee.award_foul(sim, T, j, k, rx, ry, pos, dur, evs, beaten=False)
        ft = sim.eff[j, ATTR.FIRST_TOUCH]
        p_bad = sigmoid(pp.ft_b0 + pp.ft_press * rp * (1.2 - ft / 100) - pp.ft_k * (ft - sim.R) / 10
                        + (0.4 if lofted else 0.0))
        if rng.random() < p_bad:
            evs.append(sim.ev(T, j, E.BAD_TOUCH, E.FAIL, rx, ry))
            return loose_ball(sim, T, pos, rx + rng.normal(0, 1.5), ry + rng.normal(0, 1.5), dur + 0.5, evs)
        return Outcome(duration=dur, events=evs, team=T, carrier=j, ball=sim.to_global(T, rx, ry))

    evs = [sim.ev(T, c, etype, E.FAIL, bx, by, rx, ry, extra=extra)]
    if lead is not None:
        # 傳到跑動路線失敗：若球傳得到位（執行成功），代表防守者先搶到落點
        k = lead[1]
        if rng.random() < p_exec * (1 - p_race) / max(1e-9, 1 - p_succ):
            evs.append(sim.ev(T, k, E.INTERCEPTION, E.SUCCESS, rx, ry))
            return Outcome(duration=dur, events=evs, team=1 - T, carrier=k, ball=sim.to_global(T, rx, ry))
    # 傳球在底線附近或禁區內被擋出底線 → 角球（比例皆為 StatsBomb 實測）
    p_corner = pp.byline_fail_corner if rx >= pp.byline_x else (
        pp.box_fail_corner if rx >= BOX_X and abs(ry - CY) <= BOX_HALF else 0.0)
    if p_corner and rng.random() < p_corner:
        return Outcome(duration=dur, events=evs, restart=corner(sim, T, ry))
    near_line = ry < 4 or ry > WIDTH - 4 or rx > LENGTH - 3
    w_int = (0.5 * rp) if lofted else lane
    w_con = 0.25 + 0.8 * rp
    w_out = pp.out_share * (2.5 if lofted else 1.0) * (2.0 if near_line else 0.6)
    u = rng.random() * (w_int + w_con + w_out)
    if u < w_int and dfd:
        if lofted or thr is None or float(np.sum(thr)) <= 0:
            Dp = pos[dfd]
            i = int(np.argmin(np.hypot(Dp[:, 0] - rx, Dp[:, 1] - ry)))
            k = dfd[i]
            ix, iy = float(Dp[i, 0]), float(Dp[i, 1])
        else:
            i = int(rng.choice(len(dfd), p=thr / thr.sum()))
            k = dfd[i]
            vx, vy = rx - bx, ry - by
            t = min(max(((pos[k, 0] - bx) * vx + (pos[k, 1] - by) * vy) / max(vx * vx + vy * vy, 1e-6),
                        0.0), 1.0)
            ix, iy = bx + t * vx, by + t * vy
        evs.append(sim.ev(T, k, E.INTERCEPTION, E.SUCCESS, ix, iy))
        return Outcome(duration=dur * 0.8, events=evs, team=1 - T, carrier=k, ball=sim.to_global(T, ix, iy))
    if u < w_int + w_con:
        return loose_ball(sim, T, pos, rx + rng.normal(0, 2.0), ry + rng.normal(0, 2.0), dur, evs)
    if rx > LENGTH - 8 and (lofted or rx > LENGTH - 3):
        return Outcome(duration=dur, events=evs, restart=goal_kick(sim, 1 - T))
    return Outcome(duration=dur, events=evs, restart=throw_in(sim, 1 - T, T, rx, ry))


def _through(sim, T, c, pos, opt) -> Outcome:
    rng = sim.rng_act
    p_exec, p_race, p_gk, p_off, k, thr, d_out = opt.data
    j = opt.target
    bx, by = float(pos[c, 0]), float(pos[c, 1])
    sx, sy = opt.x, opt.y
    dur = sim.decide_time() + math.hypot(sx - bx, sy - by) / 18.0 + 1.0
    D = 1 - T
    extra = {"recv": sim.pid(j)}
    fail = sim.ev(T, c, E.THROUGH, E.FAIL, bx, by, sx, sy, extra=extra)
    if rng.random() >= p_exec:
        if thr is not None and float(np.sum(thr)) > 0 and rng.random() < 0.6:
            i = int(rng.choice(len(d_out), p=thr / thr.sum()))
            kk = d_out[i]
            return Outcome(duration=dur * 0.7, events=[fail, sim.ev(T, kk, E.INTERCEPTION, E.SUCCESS,
                                                                     pos[kk, 0], pos[kk, 1])],
                           team=D, carrier=kk, ball=sim.to_global(T, pos[kk, 0], pos[kk, 1]))
        if sx > LENGTH - 6 or rng.random() < 0.3:
            return Outcome(duration=dur, events=[fail], restart=goal_kick(sim, D))
        return Outcome(duration=dur, events=[fail, sim.ev(T, k, E.RECOVERY, E.SUCCESS, sx, sy)],
                       team=D, carrier=k, ball=sim.to_global(T, sx, sy))
    if rng.random() < p_off:
        jx, jy = float(pos[j, 0]), float(pos[j, 1])
        evs = [sim.ev(T, c, E.THROUGH, E.OFFSIDE, bx, by, sx, sy, extra=extra),
               sim.ev(T, j, E.OFFSIDE, E.FAIL, jx, jy)]
        return Outcome(duration=dur * 0.6, events=evs,
                       restart=Restart("free_kick", D, *sim.to_global(T, jx, jy)))
    if rng.random() >= p_race:
        return Outcome(duration=dur, events=[fail, sim.ev(T, k, E.INTERCEPTION, E.SUCCESS, sx, sy)],
                       team=D, carrier=k, ball=sim.to_global(T, sx, sy))
    g = sim.gk_slot(D)
    if g is not None and rng.random() >= p_gk:
        evs = [fail, sim.ev(T, g, E.KEEPER_CLAIM, E.SUCCESS, sx, sy)]
        return Outcome(duration=dur, events=evs, restart=gk_hands(sim, D, LENGTH - sx, WIDTH - sy))
    evs = [sim.ev(T, c, E.THROUGH, E.SUCCESS, bx, by, sx, sy, extra=extra)]
    sim.place(T, k, sx - float(rng.uniform(1.5, 3.5)), sy + float(rng.normal(0, 1.5)))  # 輸掉衝刺者落後
    return Outcome(duration=dur, events=evs, team=T, carrier=j, ball=sim.to_global(T, sx, sy))


# ------------------------------------------------------------------ 帶球與過人
def _carry(sim, T, c, pos, opt) -> Outcome:
    cp, rng, C = sim.p.carry, sim.rng_act, sim.C
    p_att, p_tk, dist = opt.data[:3]
    k = opt.target
    bx, by = float(pos[c, 0]), float(pos[c, 1])
    ex, ey = opt.x, opt.y
    speed = max(sim.p.timing.carry_speed_min, cp.speed_frac * sim.vmax[c])
    dur = 0.6 * sim.decide_time() + dist / speed
    if k is not None and rng.random() < p_att:
        mx, my = (bx + ex) / 2, (by + ey) / 2
        if referee.foul_check(sim, k, c, sim.comp[c, C.RETAIN], sim.comp[k, C.TACKLE], False, mx, my):
            evs = [sim.ev(T, c, E.DRIBBLE, E.SUCCESS, bx, by, mx, my)]
            return referee.award_foul(sim, T, c, k, mx, my, pos, dur, evs, beaten=False)
        if rng.random() < p_tk:
            evs = [sim.ev(T, c, E.DRIBBLE, E.FAIL, bx, by, mx, my),
                   sim.ev(T, k, E.TACKLE, E.SUCCESS, mx, my, opponent=sim.pid(c))]
            if mx >= BOX_X and rng.random() < sim.p.take_on.byline_corner:
                return Outcome(duration=dur, events=evs, restart=corner(sim, T, my))
            if rng.random() < cp.loose_share:
                return loose_ball(sim, T, pos, mx + rng.normal(0, 1.5), my + rng.normal(0, 1.5), dur, evs)
            return Outcome(duration=dur, events=evs, team=1 - T, carrier=k, ball=sim.to_global(T, mx, my))
        evs = [sim.ev(T, c, E.DRIBBLE, E.SUCCESS, bx, by, ex, ey),
               sim.ev(T, k, E.TACKLE, E.FAIL, mx, my, opponent=sim.pid(c))]
        return Outcome(duration=dur, events=evs, team=T, carrier=c, ball=sim.to_global(T, ex, ey))
    return Outcome(duration=dur, events=[sim.ev(T, c, E.DRIBBLE, E.SUCCESS, bx, by, ex, ey)],
                   team=T, carrier=c, ball=sim.to_global(T, ex, ey))


def _take_on(sim, T, c, pos, opt) -> Outcome:
    tp, rng, C = sim.p.take_on, sim.rng_act, sim.C
    k = opt.target
    bx, by = float(pos[c, 0]), float(pos[c, 1])
    tx, ty = opt.x, opt.y
    dur = 0.5 * sim.decide_time() + tp.duration
    success = rng.random() < opt.p
    fx, fy = ((bx + tx) / 2, (by + ty) / 2) if success else (bx, by)
    if referee.foul_check(sim, k, c, sim.comp[c, C.TAKE_ON], sim.comp[k, C.TAKE_ON_DEF], success, fx, fy):
        evs = [sim.ev(T, c, E.TAKE_ON, E.SUCCESS if success else E.FAIL, bx, by, fx, fy,
                      opponent=sim.pid(k))]
        return referee.award_foul(sim, T, c, k, fx, fy, pos, dur, evs, beaten=success)
    if success:
        evs = [sim.ev(T, c, E.TAKE_ON, E.SUCCESS, bx, by, tx, ty, opponent=sim.pid(k))]
        sim.place(T, k, tx - float(rng.uniform(1.0, 2.5)), ty + float(rng.normal(0, 1.0)))  # 被過掉的人在身後
        return Outcome(duration=dur, events=evs, team=T, carrier=c, ball=sim.to_global(T, tx, ty))
    evs = [sim.ev(T, c, E.TAKE_ON, E.FAIL, bx, by, bx, by, opponent=sim.pid(k)),
           sim.ev(T, k, E.TACKLE, E.SUCCESS, bx, by, opponent=sim.pid(c))]
    if bx >= BOX_X and rng.random() < tp.byline_corner:     # 資料：禁區附近過人失敗 10% 出底線
        return Outcome(duration=dur, events=evs, restart=corner(sim, T, by))
    if rng.random() < tp.loose_share:
        return loose_ball(sim, T, pos, bx + rng.normal(0, 1.5), by + rng.normal(0, 1.5), dur, evs)
    return Outcome(duration=dur, events=evs, team=1 - T, carrier=k, ball=sim.to_global(T, bx, by))


# ------------------------------------------------------------------ 射門（三階段）
def resolve_shot(sim, T, c, pos, x, y, xg, p_block, shooter, gkc, etype, body, blockers=(),
                 weak_foot=False, one_on_one=False, dur=None, pre=None, p_ot=None) -> Outcome:
    sp, rng = sim.p.shot, sim.rng_act
    D = 1 - T
    g = sim.gk_slot(D)
    evs = list(pre or [])
    if dur is None:
        dur = 0.5 * sim.decide_time() + sim.p.timing.shot
    if weak_foot and rng.random() < sp.weak_foot_share:
        shooter = shooter - sp.weak_foot_penalty * (1 - sim.weak[c] / 100)
    p_ot = sp.p_ot_base if p_ot is None else p_ot
    pr = shot_outcome_probs(xg, shooter, gkc, p_block, p_ot, sp.k_ot, sp.k_place, sp.k_gk, sim.R)
    extra = {"one_on_one": True} if one_on_one else None
    gy = CY + float(rng.uniform(-3.4, 3.4))
    if rng.random() < p_block:
        blk = int(rng.choice(blockers)) if len(blockers) else None
        evs.append(sim.ev(T, c, etype, E.BLOCKED, x, y, x + 2.0, y, body=body, xg=xg,
                          opponent=sim.pid(blk), extra=extra))
        if rng.random() < sp.block_corner_share:
            return Outcome(duration=dur, events=evs, restart=corner(sim, T, y))
        return loose_ball(sim, T, pos, x - 4 + rng.normal(0, 2), y + rng.normal(0, 3), dur, evs)
    if rng.random() < pr["p_ot"]:
        if rng.random() < pr["p_g_ot"]:
            evs.append(sim.ev(T, c, etype, E.GOAL, x, y, LENGTH, gy, body=body, xg=xg, extra=extra))
            return Outcome(duration=dur, events=evs, restart=Restart("kickoff", D, 52.5, CY), goal=T)
        evs.append(sim.ev(T, c, etype, E.SAVED, x, y, LENGTH, gy, body=body, xg=xg, extra=extra))
        if g is None:
            return Outcome(duration=dur, events=evs, restart=goal_kick(sim, D))
        evs.append(sim.ev(T, g, E.KEEPER_SAVE, E.SUCCESS, LENGTH - 1.0, gy, xg=xg))
        hold = sigmoid(sp.hold_b0 + sp.hold_k * (sim.eff[g, ATTR.GK_HANDLING] - sim.R) / 10 - sp.hold_xg * xg)
        if rng.random() < hold:
            return Outcome(duration=dur, events=evs, restart=gk_hands(sim, D))
        if rng.random() < sp.parry_corner_share:
            return Outcome(duration=dur, events=evs, restart=corner(sim, T, gy))
        return loose_ball(sim, T, pos, LENGTH - 7 + rng.normal(0, 2), CY + rng.normal(0, 5), dur, evs)
    evs.append(sim.ev(T, c, etype, E.OFF_TARGET, x, y, LENGTH, CY + float(rng.choice([-1, 1])) * rng.uniform(4, 10),
                      body=body, xg=xg, extra=extra))
    return Outcome(duration=dur, events=evs, restart=goal_kick(sim, D))


# ------------------------------------------------------------------ 傳中
def tracked_defenders(sim, D_slots, pos, lx: float, ly: float, t: float) -> np.ndarray:
    """球飛行期間防守者朝落點追防：反應後以自身速度移動，最多貼到觸球範圍（不重疊在落點上）"""
    cr = sim.p.cross
    P = pos[D_slots].copy()
    if not len(P):
        return P
    v = np.array([lx, ly]) - P
    d = np.hypot(v[:, 0], v[:, 1])
    run = np.maximum(t - cr.def_react, 0.0)
    vmax, tau = sim.vmax[D_slots], sim.tau[D_slots]
    covered = vmax * (run - tau * (1 - np.exp(-run / tau)))
    step = np.minimum(covered, np.maximum(d - cr.reach, 0.0))
    P += v / np.maximum(d, 1e-9)[:, None] * step[:, None]
    return P


def _cross(sim, T, c, pos, opt) -> Outcome:
    """傳中結果（StatsBomb 360 擬合的比例）：被擋 → 出界 → 落點爭奪 → 頭槌攻門／控球／頂下，
    或門將沒收／拍出、防守者頭槌解圍。"""
    cr, rng, C = sim.p.cross, sim.rng_act, sim.C
    p_blk, p_out, p_win, k, gk_first, typ, throw = opt.data
    a = opt.target
    bx, by = float(pos[c, 0]), float(pos[c, 1])
    lx, ly = opt.x, opt.y
    D = 1 - T
    g = sim.gk_slot(D)
    etype = E.THROW_IN if throw else E.CROSS
    flight = math.hypot(lx - bx, ly - by) / cr.ball_speed
    dur = 0.6 * sim.decide_time() + flight + 0.3
    extra = {"recv": sim.pid(a), "cross_type": typ}
    fail = sim.ev(T, c, etype, E.FAIL, bx, by, lx, ly, extra=extra)
    if rng.random() < p_blk:
        if rng.random() < cr.block_corner_share:
            return Outcome(duration=dur, events=[fail], restart=corner(sim, T, by))
        return loose_ball(sim, T, pos, bx - 3 + rng.normal(0, 2), by + rng.normal(0, 2), dur, [fail])
    if rng.random() < p_out:
        if rng.random() < cr.out_goal_kick:
            return Outcome(duration=dur, events=[fail], restart=goal_kick(sim, D))
        return Outcome(duration=dur, events=[fail], restart=throw_in(sim, D, T, lx, WIDTH - by))
    high = typ == "high"
    if rng.random() < p_win:
        evs = [sim.ev(T, c, etype, E.SUCCESS, bx, by, lx, ly, extra=extra)]
        if not high:
            # 地面、低平傳中：接球者在落點拿球，之後照一般決策（可能直接射門）
            return Outcome(duration=dur, events=evs, team=T, carrier=a, ball=sim.to_global(T, lx, ly))
        evs.append(sim.ev(T, a, E.AERIAL, E.SUCCESS, lx, ly, opponent=sim.pid(k)))
        u = rng.random()
        if u < cr.head_shot_high:
            d_out = [s for s in sim.active_slots(D) if not sim.is_gk(s)]
            Dp = tracked_defenders(sim, d_out, pos, lx, ly, flight)
            n_cone, d_near, inc = shot_context(lx, ly, Dp)
            xg_h, pb_h, po_h = shot_model(sim, lx, ly, 1, n_cone, d_near)
            blockers = [d_out[i] for i in np.flatnonzero(inc)]
            gkc = sim.comp[g, C.GK_SAVE] if g is not None else 20.0
            return resolve_shot(sim, T, a, pos, lx, ly, xg_h, pb_h, sim.comp[a, C.HEADER_SHOT], gkc, E.SHOT,
                                "head", blockers, dur=dur, pre=evs, p_ot=po_h)
        if u < cr.head_shot_high + (1 - cr.head_shot_high) * cr.knockdown:
            # 頭槌頂給隊友或二點：無主球在落點附近
            return loose_ball(sim, T, pos, lx - rng.uniform(2, 7), ly + rng.normal(0, 4), dur, evs)
        return Outcome(duration=dur, events=evs, team=T, carrier=a, ball=sim.to_global(T, lx, ly))
    evs = [fail]
    if gk_first and g is not None:
        evs.append(sim.ev(T, g, E.KEEPER_CLAIM, E.SUCCESS, lx, ly))
        if rng.random() < cr.gk_punch:
            return loose_ball(sim, T, pos, lx - rng.uniform(10, 20), ly + rng.normal(0, 8), dur, evs)
        return Outcome(duration=dur, events=evs, restart=gk_hands(sim, D, LENGTH - lx, WIDTH - ly))
    lx2, ly2 = lx - rng.uniform(15, 25), ly + rng.normal(0, 8)
    if k is not None:
        if high:
            evs.append(sim.ev(T, k, E.AERIAL, E.SUCCESS, lx, ly, opponent=sim.pid(a)))
        evs.append(sim.ev(T, k, E.CLEARANCE, E.SUCCESS, lx, ly, lx2, ly2, body="head" if high else "foot"))
    if rng.random() < cr.def_corner_share:
        return Outcome(duration=dur, events=evs, restart=corner(sim, T, ly))
    return loose_ball(sim, T, pos, lx2, ly2, dur, evs)


def _clear(sim, T, c, pos, opt) -> Outcome:
    cl, rng = sim.p.clearance, sim.rng_act
    bx, by = float(pos[c, 0]), float(pos[c, 1])
    lx, ly = opt.x, opt.y
    dur = 0.4 * sim.decide_time() + 2.5
    evs = [sim.ev(T, c, E.CLEARANCE, E.SUCCESS, bx, by, lx, ly)]
    if bx <= 16.5 and rng.random() < cl.corner_share_box:
        # 禁區內倉促解圍出底線 → 對方角球（對方座標系的 y 需翻轉）
        return Outcome(duration=dur, events=evs, restart=corner(sim, 1 - T, WIDTH - by))
    if rng.random() < cl.out_share:
        return Outcome(duration=dur, events=evs, restart=throw_in(sim, 1 - T, T, lx, ly))
    return loose_ball(sim, T, pos, lx, ly, dur, evs)
