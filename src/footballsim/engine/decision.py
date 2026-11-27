"""持球者決策：感知（選項生成）→ 評估 → softmax 選擇（§4.6）

偏好（bias）與溫度皆為相對值：以當下最佳選項的效用為單位。

每個選項帶有「真實成功率 p」；球員看到的是加了估計誤差的 p̂（決策、壓迫、冷靜決定誤差大小），
效用 U = p̂·V_succ + (1 − p̂)·V_fail + 偏好；最後以溫度 T（決策越高越低）做 softmax。
解析階段直接沿用同一個 p，確保「想的」與「發生的」一致。
"""
from __future__ import annotations

import math

import numpy as np

from ..model.attributes import ATTR
from .core_math import (duel_prob, logit, np_logit, np_sigmoid, np_sprint_time, pressure_mod, race_prob,
                        shot_geometry, shot_outcome_probs, sigmoid, sprint_time)
from .pitch import BOX_HALF, BOX_X, CY, LENGTH, NEAR_CAP, WIDTH, shot_context
from .shape import interp

SET_PIECE_MODES = {"kickoff", "goal_kick", "free_kick", "corner_short", "gk_hands"}


class Opt:
    __slots__ = ("kind", "target", "x", "y", "p", "v_s", "v_f", "bias", "data", "feat")

    def __init__(self, kind, target, x, y, p, v_s, v_f, bias=0.0, data=None, feat=None):
        self.kind = kind
        self.target = target
        self.x = float(x)
        self.y = float(y)
        self.p = float(p)
        self.v_s = float(v_s)
        self.v_f = float(v_f)
        self.bias = float(bias)
        self.data = data
        self.feat = feat          # 傳球對象選擇模型的特徵（接球者在出球瞬間的處境）

    def __repr__(self) -> str:  # 方便除錯
        return (f"Opt({self.kind}, tgt={self.target}, ({self.x:.1f},{self.y:.1f}), p={self.p:.2f}, "
                f"vs={self.v_s:.3f}, vf={self.v_f:.3f}, b={self.bias:.3f})")


def carrier_pressure(sim, pos, bx: float, by: float, dfd) -> float:
    if not dfd:
        return 0.0
    dv = pos[dfd] - (bx, by)
    d2 = (dv * dv).sum(1)
    sig = sim.p.pressure.sigma
    return float((np.exp(-d2 / (2 * sig * sig)) * sim.comp[dfd, sim.C.PRESS] / sim.R).sum())


def aerial_p(sim, a: int, k: int, crowd: float = 1.0) -> float:
    """爭頂：有效高度（身高＋起跳）、時機、力量、勇敢（§4.9）"""
    ae, C, e = sim.p.aerial, sim.C, sim.eff
    dh = (sim.height[a] + sim.jump[a]) - (sim.height[k] + sim.jump[k])
    z = (ae.per_cm * dh + ae.timing * (sim.comp[a, C.AERIAL_TIMING] - sim.comp[k, C.AERIAL_TIMING])
         + ae.strength * (e[a, ATTR.STRENGTH] - e[k, ATTR.STRENGTH])
         + ae.bravery * (e[a, ATTR.BRAVERY] - e[k, ATTR.BRAVERY]))
    return sigmoid(z * crowd)


def aerial_score(sim, s: int) -> float:
    return float(sim.height[s] + sim.jump[s] + 0.25 * sim.comp[s, sim.C.AERIAL_TIMING])


def lane_threat(sim, bx, by, tx, ty, pos, dfd):
    """傳球線上的攔截威脅：回傳（總和, 各防守者威脅陣列）"""
    if not dfd:
        return 0.0, None
    pp = sim.p.passing
    vx, vy = tx - bx, ty - by
    L2 = max(vx * vx + vy * vy, 1e-6)
    Dp = pos[dfd]
    t = ((Dp[:, 0] - bx) * vx + (Dp[:, 1] - by) * vy) / L2
    px, py = bx + t * vx, by + t * vy
    l2 = (Dp[:, 0] - px) ** 2 + (Dp[:, 1] - py) ** 2
    sig = pp.lane_sigma0 + pp.lane_sigma_per_m * t * math.sqrt(L2)
    inl = (t > 0.08) & (t < 0.97)
    thr = np.where(inl, np.exp(-l2 / (2 * sig * sig)), 0.0) * sim.comp[dfd, sim.C.INTERCEPTION] / sim.R
    return float(thr.sum()), thr


SPACE_CAP = 15.0
SPACE_HALF_ANGLE = math.radians(40.0)


def space_ahead(bx: float, by: float, O: np.ndarray) -> float:
    """正前方（±40° 扇形）最近對手的距離，上限 15 m（帶球傾向的特徵；360 擬合與引擎共用）"""
    if len(O) == 0:
        return SPACE_CAP
    rel = O - (bx, by)
    dist = np.hypot(rel[:, 0], rel[:, 1])
    fwd = (rel[:, 0] > 0) & (np.abs(np.arctan2(rel[:, 1], rel[:, 0])) <= SPACE_HALF_ANGLE)
    return float(min(dist[fwd].min(), SPACE_CAP)) if fwd.any() else SPACE_CAP


class _Ctx:
    __slots__ = ("T", "D", "c", "bx", "by", "pos", "mates", "dfd", "d_out", "gk_d", "P",
                 "kappa", "m", "t_out_x", "mode", "off_line", "d_opp", "space", "carry_len")


def make_context(sim, T: int, c: int, pos, off_line: float, mode=None) -> _Ctx:
    """持球者的決策情境（位置、對手、受壓、行為模型的位置特徵）"""
    d = sim.p.decision
    x = _Ctx()
    x.T, x.D, x.c, x.pos, x.mode, x.off_line = T, 1 - T, c, pos, mode, off_line
    x.carry_len = None            # 這次想帶多遠；None＝帶球速度 × 持球時間（360 擬合時填入真實的帶球長度）
    x.bx, x.by = float(pos[c, 0]), float(pos[c, 1])
    x.mates = [s for s in sim.active_slots(T) if s != c]
    x.dfd = sim.active_slots(x.D)
    x.gk_d = sim.gk_slot(x.D)
    x.d_out = [s for s in x.dfd if s != x.gk_d]
    x.P = 0.0 if mode in SET_PIECE_MODES else carrier_pressure(sim, pos, x.bx, x.by, x.dfd)
    # 行為模型的位置特徵：最近對手（含門將）的距離，上限 8 m（與 360 資料擬合時的定義相同）
    x.d_opp, x.space = NEAR_CAP, SPACE_CAP
    if x.dfd and mode not in SET_PIECE_MODES:
        dv = pos[x.dfd] - (x.bx, x.by)
        x.d_opp = min(float(np.sqrt((dv * dv).sum(1)).min()), NEAR_CAP)
        x.space = space_ahead(x.bx, x.by, pos[x.dfd])
    team = sim.teams[T]
    x.m = team.tactics.mentality - 3 + team.mentality_offset
    x.kappa = d.kappa * (1 - d.kappa_mentality * x.m)
    x.t_out_x = pos[[s for s in sim.active_slots(T) if not sim.is_gk(s)], 0]
    return x


def choose(sim, T: int, c: int, pos, off_line: float, mode=None) -> Opt:
    x = make_context(sim, T, c, pos, off_line, mode)
    pos = x.pos
    opts: list[Opt] = []
    _passes(sim, x, opts)
    if mode is None:
        _take_on(sim, x, opts)
        _shot(sim, x, opts)
        _cross(sim, x, opts)
        _clearance(sim, x, opts)
    elif mode == "throw_in":
        if sim.eff[c, ATTR.LONG_THROW] >= 75:
            _cross(sim, x, opts, throw=True)
    elif mode == "free_kick":
        _through(sim, x, opts)
    shot = next((o for o in opts if o.kind == "shot"), None)
    cross = next((o for o in opts if o.kind == "cross"), None)
    clear = next((o for o in opts if o.kind == "clear"), None)
    if mode is not None:   # 定位球後的發球：沒有行為模型的門檻，直接比較效用
        cand = [o for o in opts if o.kind != "clear"]
        if not cand:
            _clearance(sim, x, cand, force=True)
        return _select(sim, x, cand)
    # 決策順序與資料定義一致：先以資料擬合的帶球傾向決定「這次是不是帶球 ≥ 3 m」（方向與距離再以效用比較）；
    # 射門／傳中／解圍的行為模型擬合時樣本不含帶球，所以只套用在不帶球的決策上。
    # 帶球選項要算 12 個方向的追截封堵，所以先抽閘門，通過了才生成（沒有可帶的方向時繼續走其他選項）
    if sim.rng_dec.random() < carry_probability(sim, x):
        carries: list[Opt] = []
        _carries(sim, x, carries)
        if carries:
            return _select_carry(sim, x, carries)
    # 身後球也是行為閘門：嘗試的機率由資料決定，通過了才在各接球者與深度之間比較效用
    runners = through_runners(sim, x)
    if runners and sim.rng_dec.random() < through_probability(sim, x, len(runners)):
        th: list[Opt] = []
        first = pick_through_runner(sim, x, runners)
        _through(sim, x, th, [first])
        if not th and len(runners) > 1:          # 這位接球者沒有可行的深度（防線附近沒有空間）：改看全部
            _through(sim, x, th, runners)
        if th:
            return _select(sim, x, th)
    rest = [o for o in opts if o.kind not in ("shot", "cross", "clear")]
    pick = _select(sim, x, rest) if rest else None
    if x.bx <= 35:
        # 本方三區：以資料擬合的機率選擇解圍（對手越近、越靠近本方球門越常直接踢遠）
        cp = sim.p.clear_propensity
        z = cp.k0 + cp.k_x * x.bx + cp.k_wide * abs(x.by - CY) + cp.k_near * x.d_opp + cp.align
        if sim.rng_dec.random() < sigmoid(z):
            if clear is None:
                tmp: list = []
                _clearance(sim, x, tmp, force=True)
                clear = tmp[0]
            return clear
    # 「要不要射門／傳中」由資料學到的行為模型決定，不與其他選項比效用（見 shoot_probability）
    if shot is not None and sim.rng_dec.random() < shoot_probability(sim, x, shot):
        return shot
    if cross is not None and sim.rng_dec.random() < cross_probability(sim, x, cross):
        return cross
    if pick is None:
        tmp = []
        _clearance(sim, x, tmp, force=True)
        return tmp[0]
    return pick


def carry_probability(sim, x: _Ctx) -> float:
    """帶球傾向：真實球員在此位置、最近對手在此距離、正前方有此空間時選擇帶球 ≥ 3 m 的機率（StatsBomb 360 擬合），
    再依持球能力調整。"""
    gp, R = sim.p.carry_propensity, sim.R
    z = (gp.g0 + gp.g_x * x.bx / 10 + gp.g_wide * abs(x.by - CY) / 10 + gp.g_near * x.d_opp
         + gp.g_space * x.space + gp.align + gp.k_skill * (sim.comp[x.c, sim.C.RETAIN] - R) / 10)
    return sigmoid(z)


def cross_probability(sim, x: _Ctx, opt: Opt) -> float:
    """傳中傾向：真實球員在此邊路位置、最近對手在此距離時選擇傳中的機率（StatsBomb 360 擬合），
    再依傳中能力、戰術寬度與特質調整。"""
    cp, R = sim.p.cross_propensity, sim.R
    z = (cp.c0 + cp.c_depth * (LENGTH - x.bx) + cp.c_wide * abs(x.by - CY) + cp.c_near * x.d_opp + cp.align
         + cp.k_skill * (sim.comp[x.c, sim.C.CROSS] - R) / 10
         + cp.k_policy * (sim.teams[x.T].tactics.width - 3)
         + cp.k_trait * sim.traits[x.c].get("crosses_early", 0.0))
    return sigmoid(z)


def shoot_probability(sim, x: _Ctx, opt: Opt) -> float:
    """射門傾向：真實球員在此位置、最近防守者在此距離時選擇射門的機率（StatsBomb 360 擬合），
    再依「射門路線上的封堵比該距離的典型射門多或少」調整（模型假設 k_quality，決策屬性越高越敏感），
    以及射手能力、戰術與特質。"""
    sp, R = sim.p.shot_propensity, sim.R
    dist, ang = shot_geometry(max(LENGTH - x.bx, 0.3), abs(x.by - CY))
    d_near = opt.data[8]
    z = sp.a0 + sp.a_dist * dist + sp.a_logang * math.log(max(ang, 1e-4)) + sp.a_near * d_near + sp.align
    ty = sim.p.shot_model.typical
    xg_typ = shot_model(sim, x.bx, x.by, 0, ty.cone_a + ty.cone_b * dist, d_near)[0]
    m = 1.0 + sp.k_dec * (sim.eff[x.c, ATTR.DECISIONS] - R) / 100
    z += sp.k_quality * m * (logit(opt.data[0]) - logit(xg_typ))
    z += sp.k_skill * (opt.data[3] - R) / 10
    z += sp.k_policy * (sim.teams[x.T].tactics.shooting - 3)
    if dist > 20:
        z += sp.k_trait * sim.traits[x.c].get("shoots_from_distance", 0.0)
    return sigmoid(z)


def _frac_ahead(x: _Ctx, fx) -> np.ndarray:
    if len(x.t_out_x) == 0:
        return np.zeros(np.shape(fx))
    return (x.t_out_x[None, :] > np.asarray(fx)[..., None]).mean(-1)


def _frac_ahead_at(x: _Ctx, fx: float) -> float:
    return float((x.t_out_x > fx).mean()) if len(x.t_out_x) else 0.0


def lane_threats(bx: float, by: float, A: np.ndarray, Dp: np.ndarray, w_int, pp) -> np.ndarray:
    """各防守者對「B → 各目標點」傳球路線的攔截威脅 (n_目標, n_防守者)；w_int 為攔截能力 / R"""
    V = A - (bx, by)
    dist2 = (V * V).sum(1)
    dist = np.sqrt(dist2)
    Wd = Dp - (bx, by)
    t = (V @ Wd.T) / np.maximum(dist2, 1e-6)[:, None]
    l2 = (Dp[None, :, 0] - (bx + t * V[:, 0:1])) ** 2 + (Dp[None, :, 1] - (by + t * V[:, 1:2])) ** 2
    sig = pp.lane_sigma0 + pp.lane_sigma_per_m * t * dist[:, None]
    inl = (t > 0.08) & (t < 0.97)
    return np.where(inl, np.exp(-l2 / (2 * sig * sig)), 0.0) * np.asarray(w_int)[None, :]


def arrival_pressure(dr: np.ndarray, dist: np.ndarray, w, pp, sigma: float) -> np.ndarray:
    """接球點的到達壓力（傳球成功率模型的特徵，360 擬合與引擎共用）：球在空中的這段時間，防守者反應
    arrive_react 秒後以 arrive_speed 朝接球點逼近，距離扣掉這段後再以高斯核加總。
    dr (n_目標, n_防守者) 為防守者到各接球點的距離，dist (n_目標,) 為傳球距離，w 為壓迫能力 / R。"""
    close = pp.arrive_speed * np.maximum(np.asarray(dist) / pp.flight_speed - pp.arrive_react, 0.0)
    de = np.maximum(dr - close[:, None], 0.0)
    return (np.exp(-de * de / (2 * sigma * sigma)) * np.asarray(w)[None, :]).sum(1)


def pass_success_logit(ps, dist, lane, rpa, P, back, gk, lofted):
    """傳球成功率的 logit（StatsBomb 360 擬合，平均能力；輸入皆可為陣列）。
    地面球主要受傳球路線上的攔截威脅影響；高球（含低平球）越過路線，但基礎成功率低、距離懲罰較大。
    共同的特徵：接球點的到達壓力、傳球者受到的壓迫、回傳、傳給門將。"""
    d = np.asarray(dist, float) / 10.0
    sl, sr, sp = np.sqrt(lane), np.sqrt(rpa), np.sqrt(P)
    z = (ps.s0 + ps.s_d * d + ps.s_d2 * d * d + ps.s_lane * sl + ps.s_recv * sr + ps.s_press * sp
         + ps.s_back * back + ps.s_gk * gk)
    return z + lofted * (ps.s_loft + ps.s_loft_lane * sl + ps.s_loft_d * d + ps.s_loft_recv * sr)


def lofted_logit(ph, dist, lane, P, back, gk, bx: float):
    """傳球高度的選擇（StatsBomb 360 擬合）：真實球員把球踢高（含低平球）的機率的 logit。
    距離越遠、地面路線越被封、自己越受壓越常踢高球；本方三區較常、進攻三區較少；回傳與傳給門將多走地面。"""
    d = np.asarray(dist, float) / 10.0
    return (ph.h0 + ph.h_d * d + ph.h_d2 * d * d + ph.h_lane * np.sqrt(lane) + ph.h_press * math.sqrt(P)
            + ph.h_back * back + ph.h_gk * gk + ph.h_own * float(bx < 35) + ph.h_final * float(bx >= 70))


PASS_CHOICE_FEATURES = ("dx", "dist", "dist2", "space", "lane", "back", "gk", "xt_gain", "dx_own", "dx_final")


def pass_choice_features(bx: float, by: float, A: np.ndarray, sep: np.ndarray, lane: np.ndarray,
                         gk: np.ndarray, xt_gain: np.ndarray) -> np.ndarray:
    """傳球對象選擇模型的特徵（360 擬合與引擎共用）：接球者在出球瞬間的位置與處境"""
    dx = A[:, 0] - bx
    d = np.hypot(dx, A[:, 1] - by) / 10
    own, fin = float(bx < 35), float(bx >= 70)
    return np.column_stack([dx / 10, d, d * d, np.minimum(sep, 10.0) / 5, lane, (dx < -3).astype(float),
                            gk.astype(float), 20 * xt_gain, dx / 10 * own, dx / 10 * fin])


def _pass_geometry(sim, x: _Ctx, A: np.ndarray):
    """傳球目標點 A (n, 2) 的幾何：(V, 距離, 各防守者的路線威脅, 路線威脅總和, 接球壓迫, 到達壓力)。
    接球壓迫（出球瞬間）用於接球後的處境；到達壓力（扣掉球飛行時防守者的逼近）用於傳球成功率。"""
    pp, C, R = sim.p.passing, sim.C, sim.R
    V = A - (x.bx, x.by)
    dist = np.sqrt((V * V).sum(1))
    if not x.dfd:
        z = np.zeros(len(A))
        return V, dist, np.zeros((len(A), 0)), z, z, z
    comp = sim.comp
    Dp = x.pos[x.dfd]
    thr = lane_threats(x.bx, x.by, A, Dp, comp[x.dfd, C.INTERCEPTION] / R, pp)
    dr = np.sqrt(((Dp[None, :, :] - A[:, None, :]) ** 2).sum(-1))
    rs = sim.p.pressure.recv_sigma
    wp = comp[x.dfd, C.PRESS] / R
    rp = (np.exp(-dr * dr / (2 * rs * rs)) * wp[None, :]).sum(1)
    rpa = arrival_pressure(dr, dist, wp, pp, rs)
    return V, dist, thr, thr.sum(1), rp, rpa


def _passes(sim, x: _Ctx, opts) -> None:
    pp, ps, ph, d, C, R = sim.p.passing, sim.p.pass_success, sim.p.pass_height, sim.p.decision, sim.C, sim.R
    if not x.mates:
        return
    comp, pos = sim.comp, x.pos
    A = pos[x.mates]
    n = len(x.mates)
    V, dist, thr, lane, rp, rpa = _pass_geometry(sim, x, A)
    valid = (dist >= d.min_pass) & (dist <= d.max_pass)
    if x.mode == "kickoff":
        valid &= (V[:, 0] <= 0.5) & (dist <= 25)
    elif x.mode == "throw_in":
        valid &= dist <= 30
    if not valid.any():
        return

    c = x.c
    if sim.is_gk(c):
        sk_g = comp[c, C.GK_SHORT]
        sk_l = comp[c, C.GK_THROW] if x.mode == "gk_hands" else comp[c, C.GK_KICK]
    else:
        sk_g, sk_l = comp[c, C.SHORT_PASS], comp[c, C.LONG_PASS]
    tac = sim.teams[x.T].tactics
    direct = tac.directness - 3                         # 教練的長傳指示（戰術）
    teamwork = 0.5 + sim.eff[c, ATTR.TEAMWORK] / 100
    # 傳球者受壓：擬合係數對應平均冷靜，冷靜每 +10，壓迫的影響減少 k_comp（模型假設）
    sq_p = math.sqrt(max(x.P, 0.0))
    z_cool = ps.k_comp * sq_p * (sim.eff[c, ATTR.COMPOSURE] - R) / 10
    gk_r = np.fromiter((sim.is_gk(s) for s in x.mates), bool, n)

    def evaluate(A, V, dist, lane, rpa, rp, gk_r, q):
        """每個目標點的 (成功率, 成功價值, 失敗價值, 偏好, 地面球成功率, 高球成功率)。
        高度是技術選擇：以機率 q 踢高球（資料擬合），選項的價值是兩種踢法的期望值，解析時再抽樣。"""
        back = (V[:, 0] < -3.0).astype(float)
        gkf = gk_r.astype(float)
        z0 = pass_success_logit(ps, dist, lane, rpa, x.P, back, gkf, 0.0) + z_cool
        z1 = pass_success_logit(ps, dist, lane, rpa, x.P, back, gkf, 1.0) + z_cool
        p_g = np_sigmoid(z0 + pp.k_att * (sk_g - R) / 10)
        p_l = np_sigmoid(z1 + pp.k_att * (sk_l - R) / 10)
        p_true = q * p_l + (1 - q) * p_g
        v_s = sim.threat.xt_many(A) * (1 - pp.recv_press_value * np.minimum(rp, 1.0))
        cb = d.counter_bonus
        fail_g = (x.bx, x.by) + 0.6 * V                 # 地面球在路線上被斷
        vf_g = -x.kappa * sim.threat.xt_opp_many(fail_g) * (1 + cb * _frac_ahead(x, fail_g[:, 0]))
        vf_l = -x.kappa * sim.threat.xt_opp_many(A) * (1 + cb * _frac_ahead(x, A[:, 0]))   # 高球在落點被爭走
        lose = np.maximum(1 - p_true, 1e-9)
        v_f = (q * (1 - p_l) * vf_l + (1 - q) * (1 - p_g) * vf_g) / lose
        bias = np.where(dist > 30, d.directness_bias * direct, -0.5 * d.directness_bias * direct)
        bias = bias + np.where(V[:, 0] > 5, d.tempo_bias * (tac.tempo - 3), 0.0)
        return p_true, v_s, v_f, bias * teamwork, p_g, p_l

    def height_q(dist, lane, V, gk_r):
        if x.mode in ("throw_in", "kickoff"):
            return np.zeros(len(dist))
        z = lofted_logit(ph, dist, lane, x.P, (V[:, 0] < -3.0).astype(float), gk_r.astype(float), x.bx)
        if x.bx < 35:     # 教練的出球方式（戰術）：長傳出球多踢高球、短傳出球少踢
            z = z + ph.k_buildup * {"short": -1.0, "mixed": 0.0, "long": 1.0}[tac.buildup]
        return np.where(dist > ps.ground_max, 1.0, np_sigmoid(z))

    q = height_q(dist, lane, V, gk_r)
    p_true, v_s, v_f, bias, p_g, p_l = evaluate(A, V, dist, lane, rpa, rp, gk_r, q)
    feat = None
    if x.mode is None:
        if x.dfd:
            Dp = pos[x.dfd]
            sep = np.sqrt(((A[:, None, :] - Dp[None, :, :]) ** 2).sum(-1)).min(1)
        else:
            sep = np.full(n, 10.0)
        feat = pass_choice_features(x.bx, x.by, A, sep, lane, gk_r,
                                    sim.threat.xt_many(A) - sim.threat.xt_at(x.bx, x.by))
    leads = (_lead_passes(sim, x, A, V, valid, p_true, v_s, v_f, evaluate, height_q)
             if x.mode is None else {})
    for i in np.flatnonzero(valid):
        fi = feat[i] if feat is not None else None
        if i in leads:
            leads[i].feat = fi
            opts.append(leads[i])
            continue
        opts.append(Opt("pass", x.mates[i], A[i, 0], A[i, 1], p_true[i], v_s[i], v_f[i], bias[i],
                        (float(q[i]), float(lane[i]), float(rp[i]), thr[i], x.dfd, None,
                         float(p_g[i]), float(p_l[i])), fi))


def _lead_passes(sim, x: _Ctx, A, V, valid, p_feet, vs_feet, vf_feet, evaluate, height_q) -> dict:
    """傳到跑動路線：在傳球者前方的隊友於出球瞬間往球門方向起跑，球傳到他前方 lead 公尺的空檔
    （360 資料：往前的傳球，接球者平均再向前跑 5–8 m）。落點不越過越位線（越線屬身後球）。
    成功率 = 球送到落點（傳球成功率模型，不含接球點的到達壓力）× 接球者比最快的防守者先到落點
    （追跑模型，防守者的反應時間與觸球範圍與帶球封堵相同，360 帶球資料擬合）。
    每位接球者只保留「傳腳下」與「傳跑動路線」中期望價值較高的一個。回傳 {接球者索引: Opt}。"""
    pp, cp, tp, C, R = sim.p.passing, sim.p.carry, sim.p.through, sim.C, sim.R
    if pp.lead <= 0 or not x.dfd:
        return {}
    gk = np.fromiter((sim.is_gk(s) for s in x.mates), bool, len(x.mates))
    cand = valid & ~gk & (V[:, 0] >= pp.lead_min_ahead)
    if not cand.any():
        return {}
    idx = np.flatnonzero(cand)
    Ai = A[idx]
    gxv, gyv = LENGTH - Ai[:, 0], pp.lead_goal_w * (CY - Ai[:, 1])
    nrm = np.maximum(np.hypot(gxv, gyv), 1e-6)
    L = Ai + pp.lead * np.column_stack([gxv / nrm, gyv / nrm])
    L[:, 0] = np.minimum(L[:, 0], np.maximum(x.off_line - 0.5, Ai[:, 0]))
    L[:, 0] = np.minimum(L[:, 0], LENGTH - 2.0)
    L[:, 1] = np.clip(L[:, 1], 1.0, WIDTH - 1.0)
    run = np.hypot(L[:, 0] - Ai[:, 0], L[:, 1] - Ai[:, 1])
    keep = run >= pp.lead_min_run
    if not keep.any():
        return {}
    idx, Ai, L, run = idx[keep], Ai[keep], L[keep], run[keep]
    VL, dL, thrL, laneL, rpL, _ = _pass_geometry(sim, x, L)
    no_gk = np.zeros(len(L), bool)
    qL = height_q(dL, laneL, VL, no_gk)
    p_exec, vsL, vfL, bL, peg, pel = evaluate(L, VL, dL, laneL, np.zeros(len(L)), rpL, no_gk, qL)
    js = [x.mates[i] for i in idx]
    t_a = np.maximum(tp.att_react - 0.002 * (sim.comp[js, C.ATT_TIMING] - R)
                     + np_sprint_time(run, sim.vmax[js], sim.tau[js]), dL / pp.ball_speed)
    Dp = x.pos[x.dfd]
    gap = np.hypot(Dp[None, :, 0] - L[:, 0:1], Dp[None, :, 1] - L[:, 1:2]) - cp.reach
    t_d = ((cp.def_react - 0.0025 * (sim.eff[x.dfd, ATTR.ANTICIPATION] - R))[None, :]
           + np_sprint_time(gap, sim.vmax[x.dfd][None, :], sim.tau[x.dfd][None, :]))
    kmin = t_d.argmin(1)
    p_race = np_sigmoid((t_d[np.arange(len(idx)), kmin] - t_a) / tp.race_noise)
    pL = p_exec * p_race
    # 失敗的價值：執行失誤（依踢法）或防守者先到落點（在落點丟球）
    vf_race = -x.kappa * sim.threat.xt_opp_many(L) * (1 + sim.p.decision.counter_bonus * _frac_ahead(x, L[:, 0]))
    lose = np.maximum(1 - pL, 1e-9)
    vfL = ((1 - p_exec) * vfL + p_exec * (1 - p_race) * vf_race) / lose
    u_lead = pL * vsL + (1 - pL) * vfL
    u_feet = p_feet[idx] * vs_feet[idx] + (1 - p_feet[idx]) * vf_feet[idx]
    out = {}
    for k in np.flatnonzero(u_lead > u_feet):
        i = int(idx[k])
        out[i] = Opt("pass", x.mates[i], L[k, 0], L[k, 1], pL[k], vsL[k], vfL[k], bL[k],
                     (float(qL[k]), float(laneL[k]), float(rpL[k]), thrL[k], x.dfd,
                      (float(p_race[k]), x.dfd[int(kmin[k])]), float(peg[k]), float(pel[k])))
    return out


def through_runners(sim, x: _Ctx) -> list:
    """身後球的候選接球者：防線附近（不比防線深 8 m 以上）、在持球者前方 2 m 以上的隊友（與 360 擬合的定義相同；
    上限 max_options 名只為了限制計算量，取最靠近防線的）。防線後到底線之間沒有空間時沒有身後球。
    （行為閘門與選項生成共用；傳給誰由 pick_through_runner 決定）"""
    tp = sim.p.through
    if LENGTH - tp.goal_line_margin - x.off_line < tp.min_space or not x.d_out:
        return []
    pos = x.pos
    runners = [s for s in x.mates if not sim.is_gk(s)
               and pos[s, 0] >= x.off_line - 8 and pos[s, 0] > x.bx + 2]
    return sorted(runners, key=lambda s: abs(pos[s, 0] - x.off_line))[: int(tp.max_options)]


THROUGH_CHOICE_FEATURES = ("dist", "dist2", "lat", "lat_gap", "depth", "space", "lane")


def through_choice_features(bx: float, by: float, line: float, R: np.ndarray, sep: np.ndarray,
                            lane: np.ndarray) -> np.ndarray:
    """身後球接球者選擇模型的特徵（360 擬合與引擎共用）：接球者 R (n, 2) 在出球瞬間的位置與處境。
    距離（含平方）、離中路的距離、與持球者的橫向差、已經比防線深多少、身邊空間、傳球路線的攔截威脅。"""
    dx, dy = R[:, 0] - bx, R[:, 1] - by
    d = np.hypot(dx, dy) / 10
    return np.column_stack([d, d * d, np.abs(R[:, 1] - CY) / 10, np.abs(dy) / 10, (R[:, 0] - line) / 10,
                            np.minimum(sep, 10.0) / 5, lane])


def pick_through_runner(sim, x: _Ctx, runners: list) -> int:
    """身後球傳給哪一位：資料擬合的條件 logit（真實球員在這些接球者之間的選擇），不再只取最靠前的幾人
    （那會讓直塞幾乎都傳給中路前鋒，邊路的跑動者被排除）。決策越高，越接近資料中的選擇方式（與傳球對象相同）。"""
    if len(runners) == 1:
        return runners[0]
    tc = sim.p.through_choice
    A = x.pos[runners]
    _, _, _, lane, _, _ = _pass_geometry(sim, x, A)
    Dp = x.pos[x.dfd]
    sep = np.sqrt(((A[:, None, :] - Dp[None, :, :]) ** 2).sum(-1)).min(1)
    F = through_choice_features(x.bx, x.by, x.off_line, A, sep, lane)
    beta = np.array([getattr(tc, f) for f in THROUGH_CHOICE_FEATURES])
    u = (1.0 + tc.k_dec * (sim.eff[x.c, ATTR.DECISIONS] - sim.R) / 100) * (F @ beta)
    w = np.exp(u - u.max())
    return runners[int(sim.rng_dec.choice(len(runners), p=w / w.sum()))]


def through_probability(sim, x: _Ctx, n_runners: int) -> float:
    """身後球傾向：真實球員在此位置、最近對手在此距離、防線附近有這麼多接球者時，把球送到防線之後的機率
    （StatsBomb 360 擬合），再依傳球者的直塞能力（視野、短傳、長傳、技術）、戰術 directness 與特質調整。
    選哪個接球者、球送多深，仍由效用比較決定（_through）。"""
    tq, R = sim.p.through_propensity, sim.R
    z = (tq.g0 + tq.g_x * x.bx / 10 + tq.g_wide * abs(x.by - CY) / 10 + tq.g_near * x.d_opp
         + tq.g_run * min(n_runners, 3) + tq.g_gap * (x.off_line - x.bx) / 10 + tq.align
         + tq.k_skill * (sim.comp[x.c, sim.C.THROUGH_PASS] - R) / 10
         + tq.k_policy * (sim.teams[x.T].tactics.directness - 3)
         + tq.k_trait * sim.traits[x.c].get("tries_killer_balls", 0.0))
    return sigmoid(z)


def _through(sim, x: _Ctx, opts, runners=None) -> None:
    tp, d, C, R = sim.p.through, sim.p.decision, sim.C, sim.R
    if runners is None:
        runners = through_runners(sim, x)
    if not runners:
        return
    pos = x.pos
    comp, eff = sim.comp, sim.eff
    c = x.c
    coord = float(comp[x.d_out, C.LINE_COORD].mean())
    if x.gk_d is not None:
        coord += 0.2 * (eff[x.gk_d, ATTR.GK_COMMUNICATION] - R)
    trap = tp.offside_trap if sim.teams[x.D].tactics.offside_trap else 0.0
    pm = pressure_mod(x.P, eff[c, ATTR.COMPOSURE], sim.p.pressure.c)
    Dp = pos[x.d_out]
    for j in runners:
        sy = float(pos[j, 1])
        last_sx = -1.0
        # 球送多深是傳球者的決策：在接球者前方多個深度各算一次成功率與價值（深的價值高，但接球者的跑動、
        # 防守者與守門員的競速更難、球越過底線的風險更大），由效用比較選擇
        for dep in tp.depths:
            sx = min(pos[j, 0] + dep, LENGTH - tp.goal_line_margin)
            if sx < x.off_line + 1.0 or sx <= last_sx:
                continue
            last_sx = sx
            _through_option(sim, x, opts, j, sx, sy, coord, trap, pm, Dp)


def _through_option(sim, x: _Ctx, opts, j: int, sx: float, sy: float, coord: float, trap: float, pm: float,
                    Dp: np.ndarray) -> None:
    tp, d, C, R = sim.p.through, sim.p.decision, sim.C, sim.R
    pos, comp, eff, c = x.pos, sim.comp, sim.eff, x.c
    lane, thr = lane_threat(sim, x.bx, x.by, sx, sy, pos, x.d_out)
    # 球送得到落點的機率：與一般傳球、跑動路線傳球相同的 360 擬合成功率模型（距離、路線攔截、傳球者受壓、
    # 傳球高度的選擇）；不含接球點的到達壓力（接球者與防守者誰先到由下面的追跑競速決定）。
    # 直塞的複合能力與冷靜只做偏移。
    ps, ph, pp = sim.p.pass_success, sim.p.pass_height, sim.p.passing
    _, dist_a, _, lane_a, _, _ = _pass_geometry(sim, x, np.array([[sx, sy]]))
    dS, lane_f = float(dist_a[0]), float(lane_a[0])
    shift = (ps.k_comp * math.sqrt(max(x.P, 0.0)) * (eff[c, ATTR.COMPOSURE] - R) / 10
             + pp.k_att * (comp[c, C.THROUGH_PASS] - R) / 10)
    p_g = sigmoid(float(pass_success_logit(ps, dS, lane_f, 0.0, x.P, 0.0, 0.0, 0.0)) + shift)
    p_l = sigmoid(float(pass_success_logit(ps, dS, lane_f, 0.0, x.P, 0.0, 0.0, 1.0)) + shift)
    q = 1.0 if dS > ps.ground_max else sigmoid(float(lofted_logit(ph, dS, lane_f, x.P, 0.0, 0.0, x.bx)))
    p_exec = q * p_l + (1.0 - q) * p_g
    t_a = (tp.att_react - 0.002 * (comp[j, C.ATT_TIMING] - R)
           + sprint_time(math.hypot(sx - pos[j, 0], sy - pos[j, 1]), sim.vmax[j], sim.tau[j]))
    dd = np.hypot(Dp[:, 0] - sx, Dp[:, 1] - sy)
    i = int(np.argmin(dd))
    k = x.d_out[i]
    t_d = (tp.def_react - 0.0025 * (eff[k, ATTR.ANTICIPATION] - R)
           - 0.001 * (eff[k, ATTR.CONCENTRATION] - R) + sprint_time(float(dd[i]), sim.vmax[k], sim.tau[k]))
    p_race = race_prob(t_a, t_d, tp.race_noise)
    p_gk = 1.0
    g = x.gk_d
    if g is not None and sx >= 85 and abs(sy - CY) <= 22:
        dg = math.hypot(pos[g, 0] - sx, pos[g, 1] - sy)
        t_g = (tp.gk_react - 0.003 * (eff[g, ATTR.GK_SWEEPING] - R)
               + sprint_time(dg, sim.vmax[g], sim.tau[g]) + tp.gk_handicap)
        p_gk = race_prob(t_a, t_g, tp.race_noise)
    p_off = sigmoid(tp.offside_b0 + tp.offside_k * (coord - comp[j, C.ATT_TIMING]) / 10 + trap)
    p_true = p_exec * p_race * p_gk * (1 - p_off)
    bias = (d.trait_bias * sim.traits[c].get("tries_killer_balls", 0.0)
            + 0.5 * d.trait_bias * sim.traits[j].get("plays_on_shoulder", 0.0)
            + d.flair_bias * (eff[c, ATTR.FLAIR] - 50) / 50)
    # 突破防線後的單刀價值：xT 只看位置，會低估中場附近的身後球
    brk = tp.breakaway_v0 * math.exp(-math.hypot(LENGTH - sx, CY - sy) / tp.breakaway_scale)
    opts.append(Opt("through", j, sx, sy, p_true, max(sim.threat.xt_at(sx, sy), brk),
                    -x.kappa * sim.threat.xt_opp(sx, sy), bias,
                    (p_exec, p_race, p_gk, p_off, k, thr, list(x.d_out))))


CARRY_CHOICE_FEATURES = ("fwd", "back", "len", "lat_in", "lat_abs", "space", "xt_gain", "end_dy",
                         "fwd_own", "fwd_final", "lat_final", "dy_final", "in_box",
                         "lat_wide", "fwd_wide", "back_wide", "space_final", "len_final")


def carry_choice_features(bx: float, by: float, ex: np.ndarray, ey: np.ndarray, dist: np.ndarray,
                          space_end: np.ndarray, xt_gain: np.ndarray) -> np.ndarray:
    """帶球方向選擇模型的特徵（360 擬合與引擎共用）：每個帶球選項（方向 + 可帶的距離）的終點處境。
    lat_in = 往中路的橫向位移（持球者在中路時為 0）；space = 終點最近防守者的距離；xt_gain = 終點與起點的 xT 差。"""
    dx = ex - bx
    lat = (ey - by) * float(np.sign(CY - by))
    own, fin = float(bx < 35), float(bx >= 70)
    dy_end = np.abs(ey - CY) / 10
    in_box = ((ex >= BOX_X) & (np.abs(ey - CY) <= BOX_HALF)).astype(float)
    wide = abs(by - CY) / 10                                  # 持球者離中路的程度：邊路的帶球方向與中路不同
    back = (dx < -2.0).astype(float)
    space = np.minimum(space_end, 10.0) / 5
    ln = np.asarray(dist) / 10
    return np.column_stack([dx / 10, back, ln, lat / 10, np.abs(ey - by) / 10, space, 20 * np.asarray(xt_gain), dy_end,
                            dx / 10 * own, dx / 10 * fin, lat / 10 * fin, dy_end * fin, in_box,
                            lat / 10 * wide, dx / 10 * wide, back * wide, space * fin, ln * fin])


def _select_carry(sim, x: _Ctx, carries) -> Opt:
    """帶球的方向：資料基準 + 屬性與戰術的偏移（設計原則 P2：基準來自資料，屬性只負責偏移）。
    - 基準：360 擬合的條件 logit（平均球員、中性戰術下，真實球員選擇各方向的傾向）。
    - 屬性：這名持球者的成功率相對平均持球者的差（帶球、觸球、平衡、力量，經由引擎自己的搶斷模型）乘上
      選項的價值差（成功價值 − 失敗價值），除以原本效用比較的溫度（決策、受壓與冷靜決定）。
      所以帶球好的球員敢往風險大、價值高的方向走，差的球員保守；失敗的代價隨心態（kappa）變。
    - 戰術：directness 偏向前進、width 偏向終點離中路遠（戰術指示；係數屬模型假設）；特質「內切」加在往中路的選項。
    決策越高，越接近資料中的選擇方式，也更能精準權衡風險。"""
    cc, d = sim.p.carry_choice, sim.p.decision
    if not cc.enabled or any(o.feat is None for o in carries):
        return _select(sim, x, carries)
    dec = sim.eff[x.c, ATTR.DECISIONS]
    beta = np.array([getattr(cc, f) for f in CARRY_CHOICE_FEATURES])
    F = np.vstack([o.feat for o in carries])
    u = (1.0 + cc.k_dec * (dec - sim.R) / 100) * (F @ beta)
    p = np.fromiter((o.p for o in carries), float, len(carries))
    p_ref = np.fromiter((1.0 - o.data[0] * o.data[3] for o in carries), float, len(carries))
    vs = np.fromiter((o.v_s for o in carries), float, len(carries))
    vf = np.fromiter((o.v_f for o in carries), float, len(carries))
    ref = max(abs(float((p * vs + (1 - p) * vf).max())), d.u_floor)
    stress = 1 + 0.5 * x.P * (1 - sim.eff[x.c, ATTR.COMPOSURE] / 100)
    u = u + (p - p_ref) * (vs - vf) / (d.t_rel * (1.5 - dec / 100) * stress * ref)
    tac = sim.teams[x.T].tactics
    u = u + cc.k_directness * (tac.directness - 3) * F[:, CARRY_CHOICE_FEATURES.index("fwd")]
    u = u + cc.k_width * (tac.width - 3) * F[:, CARRY_CHOICE_FEATURES.index("end_dy")]
    cut = sim.traits[x.c].get("cuts_inside", 0.0)
    if cut and abs(x.by - CY) > 10:
        u = u + cc.k_trait * cut * np.array([(o.y - x.by) * (CY - x.by) > 0 for o in carries], float)
    w = np.exp(u - u.max())
    return carries[int(sim.rng_dec.choice(len(carries), p=w / w.sum()))]


def desired_carry_length(sim, c: int) -> float:
    """這次持球想帶多遠（m）：持球者在持球時間內以帶球速度跑的距離（帶球速度 × 持球時間）。
    速度屬性、體能（有效最高速）與戰術節奏（縮短持球時間）都會改變它；實際能帶多遠再受防守者追截限制。"""
    return sim.p.carry.speed_frac * float(sim.vmax[c]) * sim.decide_time()


def _carries(sim, x: _Ctx, opts) -> None:
    cp, d, C, R = sim.p.carry, sim.p.decision, sim.C, sim.R
    pos, c = x.pos, x.c
    if sim.is_gk(c):
        return
    L = min(x.carry_len if x.carry_len is not None else desired_carry_length(sim, c), cp.max_dist)
    # 帶球路線的追跑封堵：對手（含門將）只要能比持球者先抵達路線上某點的觸球範圍，帶球就在那之前停下。
    # 持球者以帶球速度（最高速 × speed_frac）起跑；防守者要先反應（預判越好越快），再以自己的速度與加速度追。
    # 例外：站在持球者與球門之間、jockey_range 內的貼身者會邊退邊守，不衝去封堵路線（360：這種情況下帶球
    # 平均仍有 8 m）；他維持原本的距離跟著持球者，帶球結束時由他負責搶斷。
    S = np.arange(cp.step, L + 1e-9, cp.step)
    t_c = np_sprint_time(S, cp.speed_frac * sim.vmax[c], sim.tau[c])
    jockey = np.zeros(0, dtype=bool)
    if x.dfd:
        B_all = pos[x.dfd]
        rel = B_all - (x.bx, x.by)
        d0 = np.hypot(rel[:, 0], rel[:, 1])
        gx, gy = LENGTH - x.bx, CY - x.by
        gn = math.hypot(gx, gy) or 1.0
        jockey = ((rel[:, 0] * gx + rel[:, 1] * gy) / gn > 0) & (d0 <= cp.jockey_range)
        race = np.flatnonzero(~jockey)
        B = B_all[race]
        react = (cp.def_react - 0.0025 * (sim.eff[x.dfd, ATTR.ANTICIPATION] - R))[race][:, None]
        vb, tb = sim.vmax[x.dfd][race][:, None], sim.tau[x.dfd][race][:, None]
    retain = sim.comp[c, C.RETAIN]
    cut = sim.traits[c].get("cuts_inside", 0.0)
    tempo = sim.teams[x.T].tactics.tempo
    rs = sim.p.pressure.recv_sigma
    n0, spaces = len(opts), []
    for ang in cp.angles:
        ux, uy = math.cos(ang), math.sin(ang)
        px = np.clip(x.bx + ux * S, 1.0, LENGTH - 1.0)
        py = np.clip(x.by + uy * S, 1.0, WIDTH - 1.0)
        free = L
        if x.dfd and len(B):
            gap = np.hypot(B[:, 0:1] - px[None, :], B[:, 1:2] - py[None, :]) - cp.reach
            hit = ((react + np_sprint_time(gap, vb, tb)) <= t_c[None, :]).any(0)
            if hit.any():
                free = float(S[int(np.argmax(hit))]) - cp.stop_short
        dist = min(free, L)
        if dist < cp.min_dist:
            continue
        ex = min(max(x.bx + ux * dist, 1.0), LENGTH - 1.0)
        ey = min(max(x.by + uy * dist, 1.0), WIDTH - 1.0)
        k, p_att, p_tk, p_tk_ref = None, 0.0, 0.0, 0.0
        space_end = 10.0
        if x.dfd:
            # 帶球結束時最近的防守者：貼身者維持原本距離，其他人以站位計算
            de = np.where(jockey, d0, np.hypot(B_all[:, 0] - ex, B_all[:, 1] - ey))
            space_end = float(de.min())
            outfield = np.array([s != x.gk_d for s in x.dfd])
            if outfield.any():
                i = int(np.argmin(np.where(outfield, de, np.inf)))
                k = x.dfd[i]
                p_att = sigmoid(cp.attempt_b0 - cp.attempt_per_m * float(de[i])
                                + 0.15 * (sim.eff[k, ATTR.AGGRESSION] - R) / 10)
                p_tk = duel_prob(cp.base_tackle, sim.comp[k, C.TACKLE], retain, cp.k_att, cp.k_def,
                                 0.3 * x.P, R)
                p_tk_ref = duel_prob(cp.base_tackle, sim.comp[k, C.TACKLE], R, cp.k_att, cp.k_def, 0.3 * x.P, R)
        # 帶球結束時的處境：防守者在帶球期間（反應後）朝終點逼近，貼身者維持原本距離；
        # 終點的壓力與傳球接應者用同一個折價規則
        v_s = sim.threat.xt_at(ex, ey)
        if x.dfd:
            t_end = float(np_sprint_time(dist, cp.speed_frac * sim.vmax[c], sim.tau[c]))
            react_all = cp.def_react - 0.0025 * (sim.eff[x.dfd, ATTR.ANTICIPATION] - R)
            run = np.maximum(t_end - react_all, 0.0)
            v_all, t_all = sim.vmax[x.dfd], sim.tau[x.dfd]
            covered = v_all * (run - t_all * (1 - np.exp(-run / t_all)))
            dd = np.where(jockey, np.maximum(d0, cp.reach),
                          np.maximum(np.hypot(B_all[:, 0] - ex, B_all[:, 1] - ey) - covered, cp.reach))
            rp_end = float((np.exp(-dd * dd / (2 * rs * rs)) * sim.comp[x.dfd, C.PRESS] / R).sum())
            v_s *= 1 - sim.p.passing.recv_press_value * min(rp_end, 1.0)
        mx, my = (x.bx + ex) / 2, (x.by + ey) / 2
        v_f = -x.kappa * sim.threat.xt_opp(mx, my) * (1 + d.counter_bonus * _frac_ahead_at(x, mx))
        bias = d.tempo_bias * (tempo - 3)
        if cut and abs(x.by - CY) > 10 and (ey - x.by) * (CY - x.by) > 0:
            bias += d.trait_bias * cut
        opts.append(Opt("carry", k, ex, ey, 1 - p_att * p_tk, v_s, v_f, bias, (p_att, p_tk, dist, p_tk_ref)))
        spaces.append(space_end)
    new = opts[n0:]
    if new:
        E = np.array([[o.x, o.y] for o in new])
        F = carry_choice_features(x.bx, x.by, E[:, 0], E[:, 1], np.array([o.data[2] for o in new]), np.array(spaces),
                                  sim.threat.xt_many(E) - sim.threat.xt_at(x.bx, x.by))
        for o, f in zip(new, F):
            o.feat = f


def _take_on(sim, x: _Ctx, opts) -> None:
    tp, d, C = sim.p.take_on, sim.p.decision, sim.C
    if not x.d_out or sim.is_gk(x.c):
        return
    Dp = x.pos[x.d_out]
    rel = Dp - (x.bx, x.by)
    dist = np.hypot(rel[:, 0], rel[:, 1])
    cand = (rel[:, 0] > -0.5) & (dist <= tp.trigger_dist)
    if not cand.any():
        return
    i = int(np.argmin(np.where(cand, dist, np.inf)))
    k = x.d_out[i]
    gx, gy = LENGTH - x.bx, CY - x.by
    gn = math.hypot(gx, gy) or 1.0
    ux, uy = 0.7 + 0.3 * gx / gn, 0.3 * gy / gn
    un = math.hypot(ux, uy)
    tx = min(Dp[i, 0] + ux / un * tp.gain, LENGTH - 1.0)
    ty = min(max(Dp[i, 1] + uy / un * tp.gain, 1.0), WIDTH - 1.0)
    cover = np.hypot(Dp[:, 0] - tx, Dp[:, 1] - ty) < tp.cover_radius
    cover[i] = False
    p_true = duel_prob(tp.base, sim.comp[x.c, C.TAKE_ON], sim.comp[k, C.TAKE_ON_DEF], tp.k_att,
                       tp.k_def, -tp.cover_penalty * int(cover.sum()), sim.R)
    v_f = -x.kappa * sim.threat.xt_opp(x.bx, x.by) * (1 + d.counter_bonus * _frac_ahead_at(x, x.bx))
    bias = (d.trait_bias * sim.traits[x.c].get("takes_on", 0.0)
            + d.flair_bias * (sim.eff[x.c, ATTR.FLAIR] - 50) / 50)
    opts.append(Opt("take_on", k, tx, ty, p_true, sim.threat.xt_at(tx, ty), v_f, bias, None))


def shot_model(sim, bx: float, by: float, header: int, n_cone: int, d_near: float):
    """資料擬合的射門模型（StatsBomb freeze frame）：回傳 (xG, 被封堵機率, 未封堵時射正機率)。
    特徵定義與擬合時相同（pitch.shot_context）。"""
    m = sim.p.shot_model
    dist, ang = shot_geometry(max(LENGTH - bx, 0.3), abs(by - CY))
    g, b, o = m.goal, m.block, m.on_target
    xg = sigmoid(g.b0 + g.b_dist * dist + g.b_logang * math.log(max(ang, 1e-4)) + g.b_head * header
                 + g.b_cone * n_cone + g.b_near * d_near)
    p_block = sigmoid(b.c0 + b.c_cone * n_cone + b.c_near * d_near + b.c_dist * dist + b.c_head * header)
    p_ot = sigmoid(o.e0 + o.e_dist * dist + o.e_head * header + o.e_near * d_near)
    return min(xg, 0.95), p_block, p_ot


def after_value(sim, key: str, kappa: float) -> float:
    """動作失敗／射門未進之後的期望價值，機率與價值都由事件資料統計（transitions）"""
    tr = getattr(sim.p.transitions, key)
    return (tr.p_corner * sim.p.transitions.corner_value + tr.p_retain * tr.v_retain
            - kappa * tr.p_lose * tr.v_lose)


def shot_after_value(sim, pr: dict, bx: float, by: float, kappa: float) -> float:
    """射門沒進之後的期望價值，依「這次射門」的結果機率推算（與 actions.resolve_shot 的規則一致）：
    被封堵 → 角球或身前無主球；被撲出 → 門將抱住、撲出角球或門前補射；射偏 → 對方球門球。"""
    sp, th = sim.p.shot, sim.threat
    vc = sim.p.transitions.corner_value
    rr = sp.rebound_retain

    def loose(lx, ly):                       # 無主球：保住則己方在該處持球，丟了則對方在該處持球
        return rr * th.xt_at(lx, ly) - kappa * (1 - rr) * th.xt_opp(lx, ly)

    p_blk = pr["blocked"]
    p_ot, p_g_ot = pr["p_ot"], pr["p_g_ot"]
    v_block = sp.block_corner_share * vc + (1 - sp.block_corner_share) * loose(bx - 4, by)
    hold = sigmoid(sp.hold_b0)               # 平均門將的接球穩定度
    v_save = (1 - hold) * (sp.parry_corner_share * vc + (1 - sp.parry_corner_share) * loose(LENGTH - 7, CY))
    v_off = -kappa * th.xt_opp(LENGTH - 5.5, CY)   # 對方球門球（對方在自己的小禁區持球；本方座標系）
    rest = p_blk * v_block + (1 - p_blk) * (p_ot * (1 - p_g_ot) * v_save + (1 - p_ot) * v_off)
    return rest / max(1e-6, 1 - pr["goal"])


def _shot(sim, x: _Ctx, opts) -> None:
    sp, d, C, R = sim.p.shot, sim.p.decision, sim.C, sim.R
    bx, by, c = x.bx, x.by, x.c
    dist = math.hypot(LENGTH - bx, by - CY)
    if dist > sp.max_dist or bx < sp.min_x or sim.is_gk(c):
        return
    dpos = x.pos[x.d_out] if x.d_out else np.zeros((0, 2))
    n_cone, d_near, inc = shot_context(bx, by, dpos)
    blockers = [x.d_out[i] for i in np.flatnonzero(inc)]
    xg, p_block, p_ot = shot_model(sim, bx, by, 0, n_cone, d_near)
    if blockers:   # 封堵者的勇敢、站位、預判
        p_block = sigmoid(logit(p_block) + sp.k_block * (float(sim.comp[blockers, C.BLOCK].mean()) - R) / 10)
    one = n_cone == 0 and dist <= sp.one_on_one_dist
    g = x.gk_d
    if one:
        shooter = sim.comp[c, C.ONE_ON_ONE]
        gkc = sim.comp[g, C.GK_1V1] if g is not None else 20.0
    else:
        shooter = sim.comp[c, C.FINISHING] if dist <= 18.5 else sim.comp[c, C.LONG_SHOT]
        gkc = sim.comp[g, C.GK_SAVE] if g is not None else 20.0
    wf = sp.weak_foot_share * sp.weak_foot_penalty * (1 - sim.weak[c] / 100)
    pr = shot_outcome_probs(xg, shooter - wf, gkc, p_block, p_ot, sp.k_ot, sp.k_place, sp.k_gk, R)
    tac = sim.teams[x.T].tactics
    bias = d.shot_policy * (tac.shooting - 3)          # 教練的射門指示（戰術）
    if dist > 20:
        bias += d.trait_bias * sim.traits[c].get("shoots_from_distance", 0.0)   # 球員特質
    opts.append(Opt("shot", None, bx, by, pr["goal"], 1.0, shot_after_value(sim, pr, bx, by, x.kappa), bias,
                    (xg, p_block, one, float(shooter), float(gkc), blockers, p_ot, n_cone, d_near)))


CROSS_TYPES = ("ground", "high", "low")


def cross_contest(sim, T: int, a: int, k_slots, g, pos, bx: float, by: float, lx: float, ly: float,
                  high: bool, skill: float):
    """傳中落點的爭奪（StatsBomb 360 擬合）：攻方目標球員、最快的防守者、門將誰先到落點。
    真實資料中傳中前的站位幾乎無法預測成敗（成功率約 30%），位置的影響因此溫和；
    屬性在此發揮作用：傳中能力（落點精準）、爭頂（高球時與最近防守者比）、門將出擊。
    回傳 (攻方拿到第一點的機率, 最快防守者, 門將是否比防守者先到, 防守者到達落點的時間差)。"""
    cr, C, R = sim.p.cross, sim.C, sim.R
    t_b = math.hypot(lx - bx, ly - by) / cr.ball_speed
    t_a = max(cr.att_react - 0.002 * (sim.comp[a, C.ATT_TIMING] - R)
              + float(np_sprint_time(math.hypot(lx - pos[a, 0], ly - pos[a, 1]), sim.vmax[a], sim.tau[a])), t_b)
    k, t_d = None, t_a + 3.0
    if k_slots:
        D = pos[k_slots]
        gap = np.maximum(np.hypot(D[:, 0] - lx, D[:, 1] - ly) - cr.reach, 0.0)
        td = (cr.def_react - 0.0025 * (sim.eff[k_slots, ATTR.ANTICIPATION] - R)
              + np_sprint_time(gap, sim.vmax[k_slots], sim.tau[k_slots]))
        i = int(np.argmin(td))
        k, t_d = k_slots[i], float(td[i])
    t_g, gk_first = t_a + 3.0, False
    if g is not None and lx >= cr.gk_zone_x and abs(ly - CY) <= cr.gk_zone_half:
        gap = max(math.hypot(pos[g, 0] - lx, pos[g, 1] - ly) - cr.gk_reach, 0.0)
        t_g = (cr.gk_react - 0.003 * (sim.comp[g, C.GK_CLAIM] - R)
               + float(np_sprint_time(gap, sim.vmax[g], sim.tau[g])))
        gk_first = t_g < t_d
    z = (cr.c0 + cr.c_def * min(max(t_d - t_a, -cr.race_cap), cr.race_cap) + cr.c_gk * min(t_g - t_a, cr.race_cap)
         + (cr.c_high if high else 0.0) + cr.k * (skill - R) / 10)
    if high and k is not None:
        z += cr.k_aer * logit(min(max(aerial_p(sim, a, k), 0.02), 0.98))
    return sigmoid(z), k, gk_first, t_d - t_a


def _cross(sim, x: _Ctx, opts, throw: bool = False) -> None:
    """傳中：目標是禁區與前沿的跑位者（站位取自 360 資料），落點是他前插後的位置。
    傳中類型依傳中位置的深淺照資料比例抽樣（靠近底線多倒三角，較深處多為高球）。"""
    cr, C, R = sim.p.cross, sim.C, sim.R
    bx, by, c = x.bx, x.by, x.c
    if bx < cr.min_x or abs(by - CY) < cr.min_wide or sim.is_gk(c):
        return
    pos = x.pos
    runners = [s for s in x.mates if not sim.is_gk(s)
               and pos[s, 0] >= cr.zone_x and abs(pos[s, 1] - CY) <= cr.zone_half]
    if not runners:
        return
    if throw:
        typ = 1
    else:
        mix = cr.type_near if LENGTH - bx <= cr.type_depth else cr.type_far
        typ = int(sim.rng_dec.choice(3, p=np.asarray(mix) / sum(mix)))
    high = typ == 1
    a = runners[int(sim.rng_dec.integers(len(runners)))]
    side = 1.0 if by > CY else -1.0
    lx = min(pos[a, 0] + cr.run_dx, LENGTH - 1.0)
    ly = min(max(pos[a, 1] - side * cr.run_dy, CY - cr.zone_half), CY + cr.zone_half)
    skill = sim.eff[c, ATTR.LONG_THROW] if throw else sim.comp[c, C.CROSS]
    p_win, k, gk_first, _ = cross_contest(sim, x.T, a, x.d_out, x.gk_d, pos, bx, by, lx, ly, high, skill)
    # 傳出界（資料：高球 10.9%、地面 3%、低平 5.1%），傳中能力越好越少
    p_out = sigmoid(logit(cr.out_share[typ]) - cr.k * (skill - R) / 10
                    - pressure_mod(x.P, sim.eff[c, ATTR.COMPOSURE], sim.p.pressure.c))
    p_blk = 0.0
    if x.d_out and not throw:
        Dp = pos[x.d_out]
        dn = float(np.hypot(Dp[:, 0] - bx, Dp[:, 1] - by).min())
        p_blk = cr.block_base * math.exp(-dn * dn / (2 * cr.block_radius ** 2))
    p_true = (1 - p_blk) * (1 - p_out) * p_win
    opts.append(Opt("cross", a, lx, ly, p_true, sim.threat.xt_at(lx, ly), after_value(sim, "cross_fail", x.kappa),
                    0.0, (p_blk, p_out, p_win, k, gk_first, CROSS_TYPES[typ], throw)))


def _clearance(sim, x: _Ctx, opts, force: bool = False) -> None:
    cl = sim.p.clearance
    own_box = x.bx <= 16.5 and abs(x.by - CY) <= BOX_HALF
    if not force and not (own_box or (x.bx <= 30 and x.P >= cl.pressure_min)):
        return
    lx = min(x.bx + cl.dist, 85.0)
    ly = min(max(x.by + float(sim.rng_dec.normal(0.0, 8.0)), 2.0), WIDTH - 2.0)
    tr = sim.p.transitions.clearance          # 解圍後球權的實際回收率（事件資料）
    v = tr.p_retain * sim.threat.xt_at(lx, ly) - x.kappa * tr.p_lose * sim.threat.xt_opp(lx, ly)
    opts.append(Opt("clear", None, lx, ly, 1.0, v, 0.0, 0.0, None))


def vision_offset(sim, c: int, dist) -> np.ndarray:
    """視野對遠距離傳球的偏移（logit）：資料擬合的選擇模型描述的是平均視野的球員，視野比基準（R）高的球員
    更常找到遠處的隊友、低的更少。只有遠距離（vision_dist 以上，之後 vision_ramp 公尺內漸增到滿）的選項受影響
    （情境決定屬性價值）；近處的傳球隊友看得到，視野不改變選擇。係數與其他能力偏移同一量級（模型假設）。"""
    d = sim.p.decision
    far = np.clip((np.asarray(dist, float) - d.vision_dist) / d.vision_ramp, 0.0, 1.0)
    return d.k_vision * (sim.eff[c, ATTR.VISION] - sim.R) / 10 * far


def _select(sim, x: _Ctx, opts) -> Opt:
    d = sim.p.decision
    c = x.c
    dec = sim.eff[c, ATTR.DECISIONS]
    stress = 1 + 0.5 * x.P * (1 - sim.eff[c, ATTR.COMPOSURE] / 100)
    n = len(opts)
    p = np.fromiter((o.p for o in opts), float, n)
    vs = np.fromiter((o.v_s for o in opts), float, n)
    vf = np.fromiter((o.v_f for o in opts), float, n)
    bias = np.fromiter((o.bias for o in opts), float, n)
    noise = sim.rng_dec.normal(0.0, d.est_noise * (1.3 - dec / 100) * stress, n)
    certain = (p <= 0.0) | (p >= 1.0)
    ph = np.where(certain, p, np_sigmoid(np_logit(p) + noise))
    # 相對尺度：球員感知的是「與最佳選項的相對差距」，溫度與行為偏好都按當下最佳價值縮放，
    # 讓中場（xT 約 0.005）與禁區前（約 0.1）的決策雜訊在相對意義上一致。
    raw = ph * vs + (1 - ph) * vf
    ref = max(abs(float(raw.max())), d.u_floor)
    u = raw + bias * ref
    temp = d.t_rel * (1.5 - dec / 100) * stress * ref
    nested = sim.p.pass_choice.enabled and any(o.kind == "pass" and o.feat is not None for o in opts)
    if not nested:        # 定位球發球：沒有資料擬合的傳球對象模型，視野直接偏移各傳球選項的效用（logit × 溫度）
        isp = np.array([o.kind == "pass" for o in opts])
        if isp.any():
            dd = np.hypot(np.array([o.x for o in opts]) - x.bx, np.array([o.y for o in opts]) - x.by)
            u = u + np.where(isp, vision_offset(sim, c, dd), 0.0) * temp
    w = np.exp((u - u.max()) / temp)
    w /= w.sum()
    k = int(sim.rng_dec.choice(n, p=w))
    if opts[k].kind != "pass" or opts[k].feat is None or not sim.p.pass_choice.enabled:
        return opts[k]
    # 巢狀選擇：「傳球」這一類的機率照上面的效用計算；「傳給誰」由 360 資料擬合的傳球對象選擇模型決定
    # （xT 是真實踢法下的平均價值，逐步最大化的球員不會像真實球員那樣往前推進）。
    # 每位接球者保留效用最高的傳法（腳下／跑動路線／挑傳）；決策屬性越高，越接近資料中的選擇方式。
    best: dict = {}
    for i, o in enumerate(opts):
        if o.kind == "pass" and o.feat is not None and (o.target not in best or raw[i] > raw[best[o.target]]):
            best[o.target] = i
    cand = list(best.values())
    pc = sim.p.pass_choice
    beta = np.array([getattr(pc, f) for f in PASS_CHOICE_FEATURES])
    m = 1.0 + pc.k_dec * (dec - sim.R) / 100
    ub = m * (np.vstack([opts[i].feat for i in cand]) @ beta)
    ub = ub + vision_offset(sim, c, [math.hypot(opts[i].x - x.bx, opts[i].y - x.by) for i in cand])
    wb = np.exp(ub - ub.max())
    return opts[cand[int(sim.rng_dec.choice(len(cand), p=wb / wb.sum()))]]
