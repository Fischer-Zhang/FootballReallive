"""比賽引擎核心公式（設計文件附錄 A；結構已定案，係數由 config 提供）"""
from __future__ import annotations

import math

import numpy as np

R_DEFAULT = 65.0


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def np_sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def np_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- 1. 屬性對抗
def duel_prob(base_p: float, att: float, dfn: float, k_att: float = 0.35, k_def: float = 0.35,
              mods: float = 0.0, R: float = R_DEFAULT) -> float:
    """base_p：R 水準雙方在此情境下的真實成功率；k_*：每 10 點能力差的 logit 位移"""
    z = logit(base_p) + k_att * (att - R) / 10 - k_def * (dfn - R) / 10 + mods
    return sigmoid(z)


def pressure_mod(pressure: float, composure: float, c: float = 0.9) -> float:
    """壓迫懲罰（加在 logit 上）。冷靜越高，壓迫影響越小。"""
    return -c * pressure * (1.25 - composure / 100)


# ---------------------------------------------------------------- 2. 衝刺（Furusawa 模型）
def vmax_ms(sprint_speed: float, a: float = 7.6, b: float = 0.022) -> float:
    return a + b * sprint_speed


def tau_s(acceleration: float, a: float = 1.0, b: float = -0.004) -> float:
    return a + b * acceleration


def sprint_time(d: float, vmax: float, tau: float) -> float:
    """由靜止跑 d 公尺所需秒數；解 d = vmax·(t − τ·(1 − e^(−t/τ)))"""
    if d <= 0:
        return 0.0
    t = d / vmax + tau
    for _ in range(8):
        e = math.exp(-t / tau)
        t -= (vmax * (t - tau * (1 - e)) - d) / (vmax * (1 - e))
    return t


def np_sprint_time(d, vmax, tau) -> np.ndarray:
    """sprint_time 的向量化版本（d、vmax、tau 可廣播）"""
    d = np.maximum(np.asarray(d, float), 0.0)
    vmax = np.asarray(vmax, float)
    tau = np.asarray(tau, float)
    t = d / vmax + tau
    for _ in range(8):
        e = np.exp(-t / tau)
        t = t - (vmax * (t - tau * (1 - e)) - d) / (vmax * (1 - e))
    return np.where(d > 0, t, 0.0)


def race_prob(t_att: float, t_def: float, noise: float = 0.12) -> float:
    return sigmoid((t_def - t_att) / noise)


# ---------------------------------------------------------------- 3. 爭頂
def jump_cm(jumping: float, a: float = 35.0, b: float = 0.30) -> float:
    return a + b * jumping


def aerial_prob(a: dict, d: dict) -> float:
    """a / d 需含 height, jumping, anticipation, heading, strength, bravery（附錄 A 驗證版）"""
    dh = (a["height"] + jump_cm(a["jumping"])) - (d["height"] + jump_cm(d["jumping"]))

    def timing(p: dict) -> float:
        return 0.6 * p["anticipation"] + 0.4 * p["heading"]

    z = (0.07 * dh + 0.025 * (timing(a) - timing(d))
         + 0.015 * (a["strength"] - d["strength"]) + 0.010 * (a["bravery"] - d["bravery"]))
    return sigmoid(z)


# ---------------------------------------------------------------- 4. 射門
GOAL_W = 7.32
XG_COEF = {"b0": -0.489, "b_dist": -0.041, "b_logang": 1.509, "b_head": -0.8}


def shot_geometry(x: float, y: float) -> tuple[float, float]:
    """x：離球門線距離(m)，y：與球門中心的橫向偏移(m) → (距離, 兩柱夾角 rad)"""
    ang = math.atan2(GOAL_W * x, x * x + y * y - (GOAL_W / 2) ** 2)
    if ang < 0:
        ang += math.pi
    return math.hypot(x, y), ang


def base_xg(x: float, y: float, header: bool = False, coef: dict | None = None) -> float:
    c = coef or XG_COEF
    dist, ang = shot_geometry(max(x, 0.3), y)
    z = c["b0"] + c["b_dist"] * dist + c["b_logang"] * math.log(max(ang, 1e-4))
    if header:
        z += c["b_head"]
    return sigmoid(z)


def shot_outcome_probs(xg: float, shooter: float, gk: float, p_block: float = 0.26,
                       p_ot_base: float = 0.47, k_ot: float = 0.12, k_place: float = 0.06,
                       k_gk: float = 0.20, R: float = R_DEFAULT) -> dict:
    """xG = (1 − p_block) · p_ot_base · p_goal_given_ot_base（R 水準射手對 R 水準門將時結果 = xG）"""
    p_block = min(max(p_block, 0.0), 0.95)
    p_g_ot_base = min(0.97, xg / ((1 - p_block) * p_ot_base))
    p_ot = sigmoid(logit(p_ot_base) + k_ot * (shooter - R) / 10)
    p_g_ot = sigmoid(logit(p_g_ot_base) + k_place * (shooter - R) / 10 - k_gk * (gk - R) / 10)
    return {"blocked": p_block, "on_target": (1 - p_block) * p_ot,
            "goal": (1 - p_block) * p_ot * p_g_ot, "save_rate": 1 - p_g_ot,
            "p_ot": p_ot, "p_g_ot": p_g_ot}


# ---------------------------------------------------------------- 5. 體能
def energy_drain_per_min(stamina: float, work_rate: float, intensity: float = 1.0,
                         d0: float = 0.0040) -> float:
    return d0 * intensity * (1.4 - 0.8 * stamina / 100) * (0.8 + 0.4 * work_rate / 100)


def physical_factor(energy: float, threshold: float = 0.75, slope: float = 0.375,
                    floor: float = 0.80) -> float:
    return 1.0 if energy >= threshold else max(floor, 1.0 - slope * (threshold - energy))


def mental_factor(energy: float, determination: float, threshold: float = 0.60,
                  slope: float = 0.27, floor: float = 0.90) -> float:
    if energy >= threshold:
        return 1.0
    return max(floor, 1.0 - slope * (threshold - energy) * (1.2 - determination / 100))


# ---------------------------------------------------------------- 6. 決策
def choice_probs(utilities, decisions: float, composure: float, pressure: float,
                 t0: float = 0.02) -> np.ndarray:
    temp = t0 * (1.5 - decisions / 100) * (1 + 0.5 * pressure * (1 - composure / 100))
    u = np.asarray(utilities, dtype=float) / temp
    p = np.exp(u - u.max())
    return p / p.sum()
