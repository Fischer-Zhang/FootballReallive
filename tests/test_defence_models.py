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
    x.carry_len = sim.p.carry.max_dist           # 想帶到最遠：看的是防守者讓他能帶多遠
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


def _close_rate(sim, bx, by, n=150):
    """球在 (bx, by)（持球隊 0 的座標系）時，最近的防守者在 3 m 內的比例（每次重新計算站位）"""
    c = sim.active_slots(0)[6]
    d_out = [s for s in sim.active_slots(1) if s != sim.gk_slot(1)]
    hit = 0
    for _ in range(n):
        sim.gpos = None
        pos, _ = shape.compute(sim, 0, bx, by, c)
        hit += float(np.hypot(*(pos[d_out] - (bx, by)).T).min() <= 3.0)
    return hit / n


def _open_field_carries(sim, carry_len=None):
    """持球者在 (60, 34)，防守者都在遠處：回傳帶球選項"""
    pos = np.zeros((22, 2))
    pos[:, 0] = 5.0
    pos[:, 1] = np.linspace(2, 66, 22)
    c = sim.active_slots(0)[6]
    pos[c] = (60.0, 34.0)
    pos[sim.gk_slot(1)] = (104.0, 34.0)
    x = decision.make_context(sim, 0, c, pos, 80.0)
    x.carry_len = carry_len
    opts = []
    decision._carries(sim, x, opts)
    return x, opts


def test_carry_options_cover_every_direction(sim):
    """帶球選項涵蓋一圈（含橫向與往回）；真實有 5–22% 的帶球往回，只給前方 5 個方向會讓引擎永遠不往回帶"""
    _, opts = _open_field_carries(sim, carry_len=8.0)
    angles = {round(np.degrees(np.arctan2(o.y - 34.0, o.x - 60.0))) for o in opts}
    assert {0, 90, -90, 180} <= angles or {0, 90, -90, -180} <= angles
    assert all(o.feat is not None and len(o.feat) == len(decision.CARRY_CHOICE_FEATURES) for o in opts)


def test_carry_choice_prefers_forward_and_open_space(sim):
    """擬合的方向選擇：往前與終點有空間者較常被選，往回仍有一定比例（真實 5–22%）"""
    x, opts = _open_field_carries(sim, carry_len=8.0)
    picks = [decision._select_carry(sim, x, opts) for _ in range(600)]
    fwd = np.mean([o.x - 60.0 > 2 for o in picks])
    back = np.mean([o.x - 60.0 < -2 for o in picks])
    assert fwd > back and 0.03 < back < 0.35


def _with_attrs(sim, slots, names, v):
    from footballsim.model.attributes import ATTR

    for s_ in slots:
        for n in names:
            sim.base[s_, getattr(ATTR, n)] = v
    sim.refresh()


@pytest.fixture()
def restore_attrs(sim):
    saved = sim.base.copy()
    yield
    sim.base[:] = saved
    sim.refresh()
    for t in sim.teams:
        t.tactics.directness = t.tactics.tempo = t.tactics.width = t.tactics.press = 3


def _close_rate(sim, bx, by, n=150):
    """球在 (bx, by)（持球隊 0 的座標系）時，最近的防守者在 3 m 內的比例（每次重新抽持球時間、計算站位）"""
    c = sim.active_slots(0)[6]
    d_out = [s for s in sim.active_slots(1) if s != sim.gk_slot(1)]
    hit = 0
    for _ in range(n):
        sim.gpos = None
        sim._hold = sim.draw_hold()
        pos, _ = shape.compute(sim, 0, bx, by, c)
        hit += float(np.hypot(*(pos[d_out] - (bx, by)).T).min() <= 3.0)
    sim._hold = None
    return hit / n


def test_closing_down_depends_on_defender_pressing_attributes(sim, restore_attrs):
    """球員層：防守者的逼搶屬性（跑動、侵略、加速、預判、團隊配合）越高，越常在持球者出手前貼上去"""
    d_out = [s for s in sim.active_slots(1) if s != sim.gk_slot(1)]
    names = ["WORK_RATE", "AGGRESSION", "ACCELERATION", "ANTICIPATION", "TEAMWORK"]
    _with_attrs(sim, d_out, names, 90)
    high = _close_rate(sim, 85.0, 5.0)
    _with_attrs(sim, d_out, names, 35)
    low = _close_rate(sim, 85.0, 5.0)
    assert high > low + 0.08


def test_longer_hold_time_gives_defenders_time_to_close(sim, restore_attrs):
    """物理：持球時間越長，防守者越能逼近（戰術節奏慢 → 持球者更常被貼身）"""
    c = sim.active_slots(0)[6]
    d_out = [s for s in sim.active_slots(1) if s != sim.gk_slot(1)]

    def near(hold):
        out = 0.0
        for _ in range(120):
            sim.gpos = None
            sim._hold = hold
            pos, _ = shape.compute(sim, 0, 85.0, 5.0, c)
            out += float(np.hypot(*(pos[d_out] - (85.0, 5.0)).T).min())
        sim._hold = None
        return out / 120

    assert near(2.0) < near(0.6)


def test_carry_choice_responds_to_skill_and_tactics(sim, restore_attrs):
    """決策層：帶球好的球員較不往回、戰術 directness 越高越往前（資料基準 + 屬性與戰術的偏移）"""
    c = sim.active_slots(0)[6]
    names = ["DRIBBLING", "FIRST_TOUCH", "BALANCE", "STRENGTH"]

    def shares(n=1200):
        pos = np.zeros((22, 2))
        pos[:, 0] = 5.0
        pos[:, 1] = np.linspace(2, 66, 22)
        pos[c] = (60.0, 34.0)
        pos[sim.gk_slot(1)] = (104.0, 34.0)
        d = [s for s in sim.active_slots(1) if s != sim.gk_slot(1)]
        pos[d[0]], pos[d[1]], pos[d[2]] = (65.5, 34.5), (63.0, 39.5), (57.0, 28.0)
        x = decision.make_context(sim, 0, c, pos, 80.0)
        x.carry_len = 8.0
        opts = []
        decision._carries(sim, x, opts)
        sim.rng_dec = np.random.default_rng(7)
        picks = [decision._select_carry(sim, x, opts) for _ in range(n)]
        return (np.mean([o.x - 60.0 > 2 for o in picks]), np.mean([o.x - 60.0 < -2 for o in picks]))

    _with_attrs(sim, [c], names, 90)
    _, back_good = shares()
    _with_attrs(sim, [c], names, 35)
    _, back_poor = shares()
    assert back_poor > back_good + 0.05
    sim.teams[0].tactics.directness = 1
    fwd_low, _ = shares(2000)
    sim.teams[0].tactics.directness = 5
    fwd_high, _ = shares(2000)
    assert fwd_high > fwd_low + 0.03


def test_carry_length_is_speed_times_hold_time(sim, restore_attrs):
    """物理：想帶多遠 = 帶球速度 × 持球時間；速度與戰術節奏都會改變它，不是從分布抽的"""
    c = sim.active_slots(0)[6]
    sim.poss = 0

    def mean_len(n=1500):
        sim.rng_act = np.random.default_rng(3)
        return float(np.mean([decision.desired_carry_length(sim, c) for _ in range(n)]))

    _with_attrs(sim, [c], ["SPRINT_SPEED", "ACCELERATION"], 90)
    fast = mean_len()
    _with_attrs(sim, [c], ["SPRINT_SPEED", "ACCELERATION"], 35)
    slow = mean_len()
    assert fast > slow * 1.05
    sim.teams[0].tactics.tempo = 1
    relaxed = mean_len()
    sim.teams[0].tactics.tempo = 5
    hurried = mean_len()
    assert relaxed > hurried * 1.15


def _through_state(sim, c):
    """持球者在 (50, 34)；一名跑動隊友在 (80, 34)，防線在 85 m；防守者都在遠處"""
    pos = np.zeros((22, 2))
    pos[:, 0] = 5.0
    pos[:, 1] = np.linspace(2, 66, 22)
    pos[c] = (50.0, 34.0)
    runner = [s_ for s_ in sim.active_slots(0) if s_ not in (c, sim.gk_slot(0))][0]
    pos[runner] = (80.0, 34.0)
    pos[sim.gk_slot(1)] = (104.0, 34.0)
    d = [s_ for s_ in sim.active_slots(1) if s_ != sim.gk_slot(1)]
    for k, s_ in enumerate(d[:4]):
        pos[s_] = (85.0, 20.0 + 10.0 * k)
    x = decision.make_context(sim, 0, c, pos, 85.0)
    return x, runner


def test_through_ball_attempts_follow_data_and_respond_to_skill_and_tactics(sim, restore_attrs):
    """決策層：身後球的嘗試機率（資料基準）隨傳球者的直塞能力與戰術 directness 改變"""
    c = sim.active_slots(0)[6]
    names = ["VISION", "SHORT_PASSING", "LONG_PASSING", "TECHNIQUE"]
    x, runner = _through_state(sim, c)
    assert decision.through_runners(sim, x) == [runner]
    _with_attrs(sim, [c], names, 90)
    high = decision.through_probability(sim, x, 1)
    _with_attrs(sim, [c], names, 35)
    low = decision.through_probability(sim, x, 1)
    assert high > low
    sim.teams[0].tactics.directness = 1
    p1 = decision.through_probability(sim, x, 1)
    sim.teams[0].tactics.directness = 5
    p5 = decision.through_probability(sim, x, 1)
    assert p5 > p1
    assert 0.0 < low < 0.5


def test_through_ball_depth_is_a_choice_and_not_a_fixed_spot(sim):
    """物理：球送多深由傳球者在多個深度中選擇，落點隨接球者位置與深度改變（不再固定在防線後 10 m、最遠 99 m）"""
    c = sim.active_slots(0)[6]
    x, runner = _through_state(sim, c)
    opts = []
    decision._through(sim, x, opts)
    ends = sorted({round(o.x, 1) for o in opts})
    assert len(ends) >= 2 and max(ends) > 90.0
    assert all(o.x >= x.off_line + 1.0 for o in opts)
    # 執行成功率用 360 擬合的傳球成功率模型：防守者都在遠處時，球送得到落點的機率很高
    assert max(o.data[0] for o in opts) > 0.6


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


def _far_pass_state(sim, c):
    """持球者在 (35, 34)；一名隊友在前方 45 m（x=80），其餘隊友在身邊 8–15 m；防守者都在遠處"""
    pos = np.zeros((22, 2))
    pos[:, 0] = 5.0
    pos[:, 1] = np.linspace(2, 66, 22)
    pos[c] = (35.0, 34.0)
    mates = [s_ for s_ in sim.active_slots(0) if s_ not in (c, sim.gk_slot(0))]
    far = mates[0]
    pos[far] = (80.0, 34.0)
    for k, s_ in enumerate(mates[1:]):
        a = 2 * np.pi * k / max(len(mates) - 1, 1)
        pos[s_] = (35.0 + (8.0 + 0.7 * k) * np.cos(a), 34.0 + (8.0 + 0.7 * k) * np.sin(a))
    d = [s_ for s_ in sim.active_slots(1) if s_ != sim.gk_slot(1)]
    for k, s_ in enumerate(d):
        pos[s_] = (95.0, 4.0 + 6.0 * k)
    pos[sim.gk_slot(1)] = (104.0, 34.0)
    return decision.make_context(sim, 0, c, pos, 85.0), far


def _far_share(sim, c, n=300):
    x, far = _far_pass_state(sim, c)
    opts = []
    decision._passes(sim, x, opts)
    far_in_menu = any(o.target == far for o in opts)
    hit = tot = 0
    for _ in range(n):
        o = decision._select(sim, x, opts)
        if o.kind == "pass":
            tot += 1
            hit += o.target == far
    return far_in_menu, hit / max(tot, 1)


def test_vision_is_a_relative_offset_on_far_passes_not_an_absolute_gate(sim, restore_attrs):
    """決策層（P2/P3）：視野不決定遠處的隊友「有沒有」，而是相對平均視野偏移選擇；只作用在遠距離，近處不受影響"""
    c = sim.active_slots(0)[6]
    d = sim.p.decision
    _with_attrs(sim, [c], ["VISION"], sim.R)
    assert float(decision.vision_offset(sim, c, [20.0, 45.0, 60.0]).max()) == 0.0   # 平均視野：資料基準不變
    _with_attrs(sim, [c], ["VISION"], sim.R + 20)
    off = decision.vision_offset(sim, c, [10.0, d.vision_dist, d.vision_dist + d.vision_ramp, 60.0])
    assert off[0] == 0.0 and off[1] == 0.0 and off[2] == pytest.approx(d.k_vision * 2) and off[3] == off[2]
    _with_attrs(sim, [c], ["VISION"], sim.R - 20)
    assert float(decision.vision_offset(sim, c, [45.0])[0]) < 0.0
    # 視野很低的球員，遠處的隊友仍在選項裡（不是被閘門濾掉）；視野高時被選的比例較高
    _with_attrs(sim, [c], ["VISION"], 30)
    in_menu_low, share_low = _far_share(sim, c)
    _with_attrs(sim, [c], ["VISION"], 95)
    in_menu_high, share_high = _far_share(sim, c)
    assert in_menu_low and in_menu_high
    assert share_high > share_low


def test_through_ball_receiver_is_chosen_by_the_data_model_not_by_depth_rank(sim, restore_attrs):
    """決策層（P2）：直塞傳給誰由資料擬合的條件 logit 決定，邊路跑動的接球者也有機會；不是只取最靠前的幾人"""
    c = sim.active_slots(0)[6]
    x, runner = _through_state(sim, c)
    mates = [s_ for s_ in sim.active_slots(0) if s_ not in (c, sim.gk_slot(0), runner)]
    wide = mates[0]
    x.pos[wide] = (80.0, 8.0)                                    # 同樣的深度，但在邊路
    x.mates = [s_ for s_ in x.mates]
    runners = decision.through_runners(sim, x)
    assert runner in runners and wide in runners
    picks = [decision.pick_through_runner(sim, x, [runner, wide]) for _ in range(400)]
    share_wide = picks.count(wide) / len(picks)
    assert 0.15 < share_wide < 0.5                               # 邊路的人有機會，但較近的中路接球者較常被選
    assert picks.count(runner) > picks.count(wide)


def _lateral_samples(sim, bx, by, n=250):
    """球在 (bx, by) 時，攻方球前方 8 m 以上的隊友與防守方外野球員的有號橫向位置（球所在側為正）"""
    c = sim.active_slots(0)[6]
    sg = 1.0 if by >= 34.0 else -1.0
    att = [s_ for s_ in sim.active_slots(0) if s_ not in (c, sim.gk_slot(0))]
    dfd = [s_ for s_ in sim.active_slots(1) if s_ != sim.gk_slot(1)]
    a, d = [], []
    for _ in range(n):
        sim.gpos = None
        pos, _ = shape.compute(sim, 0, bx, by, c)
        a += [(pos[s_, 1] - 34.0) * sg for s_ in att if pos[s_, 0] >= bx + 8.0]
        d += [(pos[s_, 1] - 34.0) * sg for s_ in dfd if pos[s_, 0] >= bx + 8.0]
    return np.array(a), np.array(d)


def test_attackers_spread_across_the_width_and_lean_to_the_ball_side(sim):
    """比賽層：攻方球前方的隊友不是固定的陣型位置，半空間有人、整體往球所在側偏；防守方比攻方緊湊（360 資料）"""
    p = sim.p.shape
    try:
        p.att_lateral = False
        tmpl, _ = _lateral_samples(sim, 52.0, 56.0)
        p.att_lateral = True
        att, dfd = _lateral_samples(sim, 52.0, 56.0)
    finally:
        p.att_lateral = True
    half = lambda v: float(((v >= 10.0) & (v < 20.0)).mean())
    assert half(tmpl) < 0.06 and half(att) > 0.10         # 陣型模板在 10–20 m 的半空間幾乎沒人（1–2%），資料分布有（15–19%）
    assert att.mean() > tmpl.mean()                        # 球在 +22 m：隊友往球所在側偏得更多（模板約 +2 m、資料約 +4～5 m）
    pool = shape.lateral_pool(sim)                         # 資料分布本身：球越偏，隊友的平均橫向位置越往球側
    mid = (pool["edges"][:-1] + pool["edges"][1:]) / 2
    means = [(pool["ahead"][1][b] * mid).sum() for b in range(5)]
    assert means[0] < means[2] < means[4] and means[4] > 6.0
    assert 14.0 < att.std() < 22.0                         # 攻方散布寬（360：標準差約 18 m）
    assert dfd.std() < att.std() - 2.0                     # 防守方較緊湊（360：約 12 m）
