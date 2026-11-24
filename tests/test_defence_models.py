"""防守站位（退守下限、區塊壓縮）與帶球封堵（貼身者邊退邊守）的方向性檢查"""
import numpy as np
import pytest

from footballsim.data.generator import generate_league
from footballsim.engine import decision, shape
from footballsim.engine.match import MatchSim


@pytest.fixture(scope="module")
def sim():
    lg = generate_league(seed=3, n_teams=4)
    return MatchSim(lg.teams[0], lg.teams[1], seed=1)


def _deepest_defenders(sim, bx):
    """球在 bx（持球隊 0 的座標系）時，防守方最深 4 名外野球員的平均 x（越大越靠近防守方球門）"""
    sim.gpos = None
    c = sim.active_slots(0)[6]
    pos, _ = shape.compute(sim, 0, bx, 34.0, c)
    d_out = [s for s in sim.active_slots(1) if s != sim.gk_slot(1)]
    return float(np.sort(pos[d_out, 0])[-4:].mean())


def test_line_retreats_then_holds_near_the_box(sim):
    mid, front, near = (_deepest_defenders(sim, bx) for bx in (50.0, 75.0, 86.0))
    assert front > mid + 8                    # 球進入前場時防線明顯後退
    assert near - front < 6                   # 之後守在禁區線附近，不隨球退進禁區
    assert 84 < near < 96                     # 球在禁區前時，防線大約在禁區線（88.5 m）附近


def _carry_options(sim, defender_xy):
    """持球者在 (60, 34)，只有一名防守者在 defender_xy，其餘防守者放到遠處"""
    T, D = 0, 1
    pos = np.zeros((22, 2))
    pos[:, 0] = 5.0
    pos[:, 1] = np.linspace(2, 66, 22)
    c = sim.active_slots(T)[6]
    pos[c] = (60.0, 34.0)
    d_slots = sim.active_slots(D)
    k = [s for s in d_slots if s != sim.gk_slot(D)][0]
    pos[k] = defender_xy
    pos[sim.gk_slot(D)] = (104.0, 34.0)
    x = decision.make_context(sim, T, c, pos, 80.0)
    opts = []
    decision._carries(sim, x, opts)
    return {round(np.degrees(np.arctan2(o.y - 34.0, o.x - 60.0))): o.data[2] for o in opts}


def test_close_goal_side_defender_jockeys_instead_of_blocking(sim):
    # 正前方 2.5 m 的貼身者邊退邊守：直線帶球仍可推進（360：這種情況下帶球平均約 8 m）
    goal_side = _carry_options(sim, (62.5, 34.0))
    assert goal_side.get(0, 0.0) >= 6.0
    # 同樣距離但在持球者身後的防守者會追，往前的路線不受影響、往回的才被擋
    behind = _carry_options(sim, (57.5, 34.0))
    assert behind.get(0, 0.0) >= 6.0


def test_box_zone_follows_real_configuration(sim):
    """禁區＋前沿的攻方人數照真實畫面：沒分到真實站位的人退到區域邊緣"""
    from footballsim.engine.shape import _box_runs

    p = sim.p.shape
    pos = np.zeros((22, 2))
    movers = list(range(1, 11))
    pos[movers, 0] = [60, 65, 70, 75, 82, 85, 88, 90, 92, 95]       # 6 人在區域內（x ≥ 80）
    pos[movers, 1] = np.linspace(20, 48, 10)
    sim._box_cfg = None
    assigned = _box_runs(sim, p, pos, movers, 80.0, 50.0, np.random.default_rng(0))
    in_zone = [s for s in movers if pos[s, 0] >= p.box_zone_x and abs(pos[s, 1] - 34) <= p.box_zone_half]
    assert set(in_zone) == assigned                                  # 區域內只剩分到真實站位的人
    assert len(assigned) <= 6 + 1


def test_booked_players_rarely_get_second_yellow(sim):
    """已吃黃牌者在同樣的犯規情境下較少再被判黃牌（2015/16：兩黃變紅 ÷ 黃牌 ≈ 2.3%）"""
    from footballsim.engine import referee

    k = sim.active_slots(1)[3]
    pos = np.tile([50.0, 34.0], (22, 1))
    rates = []
    for booked in (0, 1):
        sim.yellow[k] = booked
        cards = [referee.decide_card(sim, 0, k, 50.0, 34.0, pos, False, False) for _ in range(3000)]
        rates.append(sum(c == "yellow" for c in cards) / 3000)
    sim.yellow[k] = 0
    assert rates[1] < 0.5 * rates[0]
    assert abs(rates[1] / rates[0] - sim.p.card.booked_factor) < 0.15
