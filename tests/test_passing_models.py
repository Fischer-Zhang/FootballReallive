"""傳球成功率、傳球高度、帶球傾向（StatsBomb 360 擬合）與持球方縱向層次的方向性檢查"""
import numpy as np

from footballsim.config import load_params, load_raw
from footballsim.engine.core_math import sigmoid
from footballsim.engine.decision import arrival_pressure, lofted_logit, pass_success_logit, space_ahead
from footballsim.engine.shape import _att_ranks


def _p(ps, dist, lane=0.0, rpa=0.0, P=0.0, back=0.0, gk=0.0, lofted=0.0):
    return float(sigmoid(float(pass_success_logit(ps, dist, lane, rpa, P, back, gk, lofted))))


def test_fitted_passing_models_present_and_replace_hand_set_coefficients():
    raw = load_raw()
    for sec in ("pass_success", "pass_height", "carry_propensity"):
        assert sec in raw, sec
    # 舊的手設成功率係數已移除
    assert not ({"b0_ground", "b0_lofted", "c_recv", "recv_lofted_mult", "box_recv_mult", "backward_bonus"}
                & set(raw["passing"]))


def _z(ps, dist, lane=0.0, rpa=0.0, P=0.0, lofted=0.0):
    return float(pass_success_logit(ps, dist, lane, rpa, P, 0.0, 0.0, lofted))


def test_pass_success_geometry():
    ps = load_params().pass_success
    assert _p(ps, 15) > 0.95                                     # 無人干擾的地面短傳幾乎都成功
    assert _z(ps, 15, lane=0.8) < _z(ps, 15) - 1.0               # 路線被封，地面球成功率明顯下降（logit）
    assert _z(ps, 15, rpa=1.0) < _z(ps, 15) - 1.0                # 接球點有人逼近
    assert _z(ps, 15, P=1.0) < _z(ps, 15) - 0.5                  # 傳球者受壓
    assert _p(ps, 35, lofted=1.0) < _p(ps, 35)                   # 高球的基礎成功率較低
    # 高球越過路線：路線威脅對高球的影響遠小於地面球
    drop_ground = _z(ps, 25) - _z(ps, 25, lane=0.8)
    drop_lofted = _z(ps, 25, lofted=1.0) - _z(ps, 25, lane=0.8, lofted=1.0)
    assert drop_lofted < 0.3 * drop_ground
    # 距離的影響來自飛行時間：接球點 8 m 外有一名防守者時，45 m 的高球比 25 m 難（防守者趕得到）
    pp = load_params().passing
    rpa = arrival_pressure(np.array([[8.0], [8.0]]), np.array([25.0, 45.0]), np.ones(1), pp, 2.5)
    assert _p(ps, 45, rpa=rpa[1], lofted=1.0) < _p(ps, 25, rpa=rpa[0], lofted=1.0) - 0.05


def test_arrival_pressure_grows_with_flight_time():
    pp = load_params().passing
    dr = np.array([[6.0]])
    near = arrival_pressure(dr, np.array([10.0]), np.ones(1), pp, 2.5)[0]
    far = arrival_pressure(dr, np.array([40.0]), np.ones(1), pp, 2.5)[0]
    assert far > near                                            # 長傳飛行時間長，防守者能逼近接球點


def test_pass_height_choice_directions():
    ph = load_params().pass_height

    def q(dist, lane=0.0, P=0.0, bx=50.0):
        return float(sigmoid(float(lofted_logit(ph, dist, lane, P, 0.0, 0.0, bx))))

    assert q(12) < 0.2 and q(50) > 0.8                           # 短傳多走地面，長傳多挑高
    assert q(20, lane=0.8) > q(20)                               # 地面路線被封就挑高
    assert q(20, P=1.0) > q(20)                                  # 受壓時較常直接踢高
    assert q(25, bx=20) > q(25, bx=80)                           # 本方三區比進攻三區常踢高球


def test_carry_propensity_follows_space():
    gp = load_params().carry_propensity

    def pc(d_opp, space):
        return float(sigmoid(gp.g0 + gp.g_x * 5 + gp.g_wide * 1 + gp.g_near * d_opp + gp.g_space * space))

    assert pc(6.0, 15.0) > pc(6.0, 3.0)                          # 前方有空間就帶球
    assert pc(6.0, 10.0) > pc(1.5, 10.0)                         # 對手貼身時較少帶球
    assert space_ahead(50, 34, np.array([[55.0, 34.0], [45.0, 34.0]])) == 5.0   # 只看正前方
    assert space_ahead(50, 34, np.array([[50.0, 50.0]])) == 15.0  # 側面的對手不算


def test_att_ranks_spread_between_ball_and_line():
    p = load_params().shape
    pos = np.zeros((22, 2))
    mv = list(range(1, 11))
    pos[mv, 0] = np.linspace(20, 60, 10)                         # 模板順序：slot 10 最前
    _att_ranks(p, pos, mv, 40.0, 60.0, np.random.default_rng(0))
    top = sorted(pos[mv, 0])[::-1][:6]
    expect = [40.0 + f * 20.0 for f in p.att_ranks_mid]
    assert np.allclose(top, expect)
    assert pos[10, 0] == max(pos[mv, 0])                         # 陣型最前者仍最前
