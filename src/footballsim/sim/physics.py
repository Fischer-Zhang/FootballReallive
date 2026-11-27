"""球的運動與球員的移動（逐幀）。

球員：加速度與減速度有上限的轉向；最高速取自屬性（v1 的 vmax），再依意圖乘上比例。
球：飛行的位置用解析式（地面球等減速、高球等速加拋物線高度），每個 tick 取線段做掃掠偵測，
球一個 tick 最多移動 2–3 m，不掃掠的話球會穿過球員。
"""
from __future__ import annotations

import math

import numpy as np

from ..engine.core_math import energy_drain_per_min
from ..engine.pitch import LENGTH, WIDTH
from ..model.attributes import ATTR
from .world import Flight, Intent, Mode, World


# ------------------------------------------------------------------ 幾何
def seg_point_dist(p0: np.ndarray, p1: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """點 pts (n, 2) 到線段 p0→p1 的最短距離（掃掠偵測）"""
    d = p1 - p0
    L2 = float(d @ d)
    if L2 < 1e-12:
        return np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
    t = np.clip(((pts - p0) @ d) / L2, 0.0, 1.0)
    proj = p0 + t[:, None] * d
    return np.hypot(pts[:, 0] - proj[:, 0], pts[:, 1] - proj[:, 1])


# ------------------------------------------------------------------ 球
def ground_duration(d: float, v0: float, mu: float) -> float:
    """地面球以初速 v0、減速度 mu 滾 d 公尺的時間；球會在 v0²/(2mu) 處停住（d 超過時回傳停住的時間）"""
    stop = v0 * v0 / (2 * mu)
    if d >= stop:
        return v0 / mu
    return (v0 - math.sqrt(max(v0 * v0 - 2 * mu * d, 0.0))) / mu


def flight_state(f: Flight, t: float) -> tuple[np.ndarray, float, float]:
    """飛行中的球在時間 t 的 (水平位置, 高度, 水平速率)"""
    u = min(max(t - f.t0, 0.0), f.dur)
    vec = f.end - f.start
    L = float(np.hypot(vec[0], vec[1]))
    if L < 1e-9:
        return f.start.copy(), 0.0, 0.0
    if f.lofted:
        s = L * (u / f.dur)
        h = 4.0 * f.hmax * u * (f.dur - u) / (f.dur * f.dur)
        speed = L / f.dur
    elif f.decel <= 0.0:                                  # 射門等等速直線飛行
        s = L * (u / f.dur)
        h = 0.0
        speed = L / f.dur
    else:
        mu = f.decel
        # 落點在減速停止之前：用「到達 L 的位移曲線」，並把時間縮放到 dur
        T_phys = ground_duration(L, f.v0, mu)
        uu = u / f.dur * T_phys
        s = f.v0 * uu - 0.5 * mu * uu * uu
        s = min(s, L)
        speed = max(f.v0 - mu * uu, 0.0) * (T_phys / f.dur)
        h = 0.0
    return f.start + vec / L * s, h, speed


def advance_ball(w: World, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """推進球一個 tick，回傳 (上一個位置, 目前位置)（全域水平座標，用於掃掠偵測）"""
    prev = w.ball[:2].copy()
    if w.mode == Mode.CARRIED and w.owner is not None:
        w.ball[:2] = w.pos[w.owner]
        w.ball[2] = 0.0
        w.bvel[:] = 0.0
    elif w.mode == Mode.FLIGHT and w.flight is not None:
        p, h, sp = flight_state(w.flight, w.t + dt)
        w.ball[:2], w.ball[2] = p, h
        v = (p - prev) / dt
        w.bvel[:2], w.bvel[2] = v, 0.0
    elif w.mode == Mode.FREE:
        v = w.bvel[:2]
        speed = float(np.hypot(v[0], v[1]))
        if speed > 1e-6:
            new_speed = max(speed - w.sp.ball.roll_decel * dt, 0.0)
            w.ball[:2] += v / speed * (speed + new_speed) / 2 * dt
            w.bvel[:2] = v / speed * new_speed
        w.ball[2] = 0.0
    return prev, w.ball[:2].copy()


# ------------------------------------------------------------------ 球員
def speed_cap(w: World) -> np.ndarray:
    """各球員這個 tick 的最高速（m/s）= 屬性決定的最高速 × 意圖比例"""
    fr = w.sp.motion.frac
    table = np.array([fr.hold, fr.run, fr.press, fr.recover, fr.receive, fr.chase, fr.carry])
    return w.sim.vmax * w.sp.motion.vmax_scale * table[w.intent]


def integrate(w: World, dt: float) -> None:
    """轉向、加速、體能與跑動距離"""
    m = w.sp.motion
    sim = w.sim
    act = sim.active
    d = w.tgt - w.pos
    dist = np.hypot(d[:, 0], d[:, 1])
    cap = speed_cap(w)
    # 保持陣型（HOLD）時小幅調整位置很從容（速度 = 距離 / arrive_tau）；其他意圖衝向目標，以煞車距離決定何時減速
    brake = np.sqrt(2.0 * 0.8 * m.a_dec * dist)
    v_des_mag = np.where(w.intent == int(Intent.HOLD), np.minimum(cap, dist / m.arrive_tau), np.minimum(cap, brake))
    v_des = d / np.maximum(dist, 1e-9)[:, None] * v_des_mag[:, None]
    dv = v_des - w.vel
    n = np.hypot(dv[:, 0], dv[:, 1])
    # 加速與減速的上限不同：速度變小（與目前速度反向）用 a_dec
    along = (dv * w.vel).sum(1)
    # 加速能力：v1 的 tau（加速屬性決定的時間常數）越小，起步越快；a_max 是平均球員（R）的值
    tau_ref = sim.p.sprint.tau_a + sim.p.sprint.tau_b * sim.R
    quick = tau_ref / np.maximum(sim.tau, 0.2)
    a = np.where(along < 0, m.a_dec, m.a_max) * quick
    scale = np.minimum(1.0, a * dt / np.maximum(n, 1e-9))
    w.vel += dv * scale[:, None]
    w.vel[~act] = 0.0
    stunned = w.stun > 0
    if stunned.any():                                   # 被過掉的人短暫不能動作
        w.vel[stunned] *= 0.5
        w.stun[stunned] = np.maximum(w.stun[stunned] - dt, 0.0)
    w.pos += w.vel * dt
    np.clip(w.pos[:, 0], 0.0, LENGTH, out=w.pos[:, 0])
    np.clip(w.pos[:, 1], 0.0, WIDTH, out=w.pos[:, 1])
    sp = np.hypot(w.vel[:, 0], w.vel[:, 1])
    w.dist_run += sp * dt
    # 體能：以 v1 的每分鐘耗能為基準，依速度比例調整
    st, wr = sim.eff[:, ATTR.STAMINA], sim.eff[:, ATTR.WORK_RATE]
    base = energy_drain_per_min(st, wr, sim.intensity) / 60.0
    load = np.clip(sp / np.maximum(sim.vmax * m.vmax_scale, 1e-9) / m.energy_ref_frac, 0.2, 4.0) ** 2
    sim.energy -= base * load * dt * act
    np.clip(sim.energy, 0.0, 1.0, out=sim.energy)
    if w.clock >= w.next_refresh:
        sim.refresh()
        w.next_refresh = w.clock + m.refresh_every
