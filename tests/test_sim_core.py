"""v2 逐幀引擎（footballsim.sim）的單元測試：動力學、球、接觸、階段、決策回歸"""
import math

import numpy as np
import pytest

from footballsim.data.generator import generate_league
from footballsim.engine import decision as v1dec
from footballsim.engine import shape as v1shape
from footballsim.model.attributes import ATTR
from footballsim.sim import physics, play, rules
from footballsim.sim import phase as phase_mod
from footballsim.sim import shape as sim_shape
from footballsim.sim.params import load_sim_params
from footballsim.sim.world import Flight, Intent, Mode, Phase, World


@pytest.fixture(scope="module")
def league():
    return generate_league(seed=3, n_teams=4)


@pytest.fixture()
def world(league):
    return World(league.teams[0], league.teams[1], seed=1)


def _spread(w):
    """把 22 人排在場上互不干擾的位置：主隊在左半場、客隊在右半場，橫向每 5 m 一人"""
    for s in range(22):
        k = s % 11
        w.pos[s] = (8.0 + 3.0 * k if s < 11 else 62.0 + 3.0 * k, 4.0 + 5.5 * k)
    w.vel[:] = 0.0
    w.tgt[:] = w.pos


# ------------------------------------------------------------------ motion
def test_speed_and_acceleration_never_exceed_limits(world):
    w = world
    _spread(w)
    s = 5
    w.tgt[s] = (100.0, w.pos[s, 1])
    w.intent[s] = int(Intent.RECOVER)
    cap = w.sim.vmax[s] * w.sp.motion.vmax_scale                  # 目標很遠（> 28 m）：衝刺到全速
    dt = w.sp.clock.dt
    prev = w.vel[s].copy()
    vmax_seen, amax_seen = 0.0, 0.0
    for _ in range(150):
        physics.integrate(w, dt)
        vmax_seen = max(vmax_seen, float(np.hypot(*w.vel[s])))
        amax_seen = max(amax_seen, float(np.hypot(*(w.vel[s] - prev))) / dt)
        prev = w.vel[s].copy()
    assert vmax_seen <= cap + 1e-6
    assert amax_seen <= max(w.sp.motion.a_max, w.sp.motion.a_dec) + 1e-6
    assert vmax_seen > 0.9 * cap                       # 跑得夠久會到最高速


def _time_to_cover(w, s, dist, dt=0.1):
    w.pos[s] = (5.0, 34.0)
    w.vel[s] = 0.0
    w.tgt[s] = (5.0 + dist, 34.0)
    w.intent[s] = int(Intent.CHASE)
    t = 0.0
    while w.pos[s, 0] < 5.0 + dist - 0.5 and t < 30:
        physics.integrate(w, dt)
        t += dt
    return t


def test_faster_player_covers_30m_sooner(world):
    w, s = world, 7
    slow = _time_to_cover(w, s, 30.0)
    w.sim.base[s, ATTR.SPRINT_SPEED] += 25
    w.sim.base[s, ATTR.ACCELERATION] += 25
    w.sim.refresh()
    fast = _time_to_cover(w, s, 30.0)
    assert fast < slow - 0.1


def test_running_costs_energy(world):
    w, s = world, 6
    e0 = float(w.sim.energy[s])
    w.pos[s], w.vel[s] = (5.0, 34.0), 0.0
    w.tgt[s] = (100.0, 34.0)
    w.intent[s] = int(Intent.RUN)
    for _ in range(300):
        physics.integrate(w, 0.1)
        if w.pos[s, 0] > 95:
            w.pos[s, 0] = 5.0
    assert w.sim.energy[s] < e0 and w.dist_run[s] > 100


# ------------------------------------------------------------------ ball
def _flight(**kw):
    base = dict(kind="pass", kicker=1, team=0, start=np.array([20.0, 34.0]), end=np.array([40.0, 34.0]),
                t0=0.0, dur=1.0, v0=15.0, lofted=False, hmax=0.0, decel=1.3)
    base.update(kw)
    return Flight(**base)


def test_ground_ball_arrives_at_target_at_closed_form_time():
    v0, mu, d = 19.0, 1.3, 20.0
    t = physics.ground_duration(d, v0, mu)
    closed = (v0 - math.sqrt(v0 * v0 - 2 * mu * d)) / mu
    assert t == pytest.approx(closed, rel=0.02)
    f = _flight(dur=t, v0=v0, decel=mu)
    p, h, speed = physics.flight_state(f, t)
    assert np.allclose(p, f.end, atol=1e-6) and h == 0.0
    mid, _, v_mid = physics.flight_state(f, t / 2)
    assert 20.0 < mid[0] < 40.0 and v_mid < v0            # 地面球沿途減速


def test_lofted_ball_lands_on_target_with_peak_in_the_middle():
    f = _flight(lofted=True, hmax=6.0, dur=2.0, v0=10.0)
    p_end, h_end, _ = physics.flight_state(f, 2.0)
    assert np.allclose(p_end, f.end) and h_end == pytest.approx(0.0, abs=1e-9)
    _, h_mid, _ = physics.flight_state(f, 1.0)
    assert h_mid == pytest.approx(6.0)


def test_sweep_catches_a_defender_a_fast_ball_would_otherwise_skip_past():
    p0, p1 = np.array([10.0, 10.0]), np.array([13.0, 10.0])        # 一個 tick 移動 3 m
    pts = np.array([[11.5, 10.3], [11.5, 12.0]])
    d = physics.seg_point_dist(p0, p1, pts)
    assert d[0] == pytest.approx(0.3) and d[1] == pytest.approx(2.0)
    # 只看 tick 結束時的位置（13, 10）距離球員都超過 1.5 m，會穿過去
    assert np.hypot(13 - 11.5, 10 - 10.3) > 1.5


def test_goal_only_counts_between_the_posts_and_under_the_bar(world):
    w = world
    w.mode = Mode.FREE
    w.last_touch = 3
    w.ball[:] = (105.2, 34.0, 0.5)
    rules.check_out(w)
    assert w.score == [1, 0] and w.restart["kind"] == "kickoff" and w.restart["team"] == 1
    w2 = World(w.sim.teams[0].team, w.sim.teams[1].team, seed=2)
    w2.mode, w2.last_touch = Mode.FREE, 3
    w2.ball[:] = (105.2, 40.0, 0.5)                                  # 球門柱外：球門球
    rules.check_out(w2)
    assert w2.score == [0, 0] and w2.restart["kind"] == "goal_kick" and w2.restart["team"] == 1
    w3 = World(w.sim.teams[0].team, w.sim.teams[1].team, seed=3)
    w3.mode, w3.last_touch = Mode.FREE, 14                           # 客隊碰出底線：主隊角球
    w3.ball[:] = (105.2, 40.0, 0.5)
    rules.check_out(w3)
    assert w3.restart["kind"] == "corner_short" and w3.restart["team"] == 0


# ------------------------------------------------------------------ contest
def _intercept_rate(w, defender_offset, skill, n=300):
    """主隊從 (20,34) 傳到 (40,34)；客隊一名防守者站在路線中點旁 defender_offset 公尺；回傳被攔截的比例"""
    hits = 0
    sim = w.sim
    d = 15  # 客隊後衛
    sim.base[d, ATTR.ANTICIPATION] = skill
    sim.base[d, ATTR.DEF_POSITIONING] = skill
    sim.base[d, ATTR.ACCELERATION] = skill
    sim.refresh()
    for i in range(n):
        _spread(w)
        w.pos[d] = (30.0, 34.0 + defender_offset)
        w.pos[1] = (20.0, 34.0)
        w.pos[2] = (40.0, 34.0)
        f = _flight(kicker=1, team=0, dur=physics.ground_duration(20.0, 19.0, 1.3), v0=19.0)
        w.flight, w.mode, w.pending = f, Mode.FLIGHT, None
        w.t = f.t0 + 0.1
        prev = np.array([29.05, 34.0])                      # 初速 19 m/s、一個 tick 移動 1.9 m
        cur = np.array([30.95, 34.0])
        w.ball[:2] = cur
        w.bvel[:2] = (cur - prev) / 0.1
        w.poss = 0
        play.flight_step(w, prev, cur, 0.1)
        hits += w.mode == Mode.CARRIED and w.owner == d
    return hits / n


def test_only_defenders_who_reach_the_lane_intercept_and_skill_matters(world):
    w = world
    far = _intercept_rate(w, 5.0, 65)
    near_hi = _intercept_rate(w, 0.0, 90)
    near_lo = _intercept_rate(w, 0.0, 30)
    assert far == 0.0
    assert near_hi > near_lo + 0.1 and near_hi > 0.2


def test_tackle_success_depends_on_tackler_versus_carrier(world):
    w = world
    sim = w.sim
    c, k = 3, 14
    results = {}
    for tag, (tk, ret) in {"strong": (90, 40), "weak": (30, 90)}.items():
        sim.base[k, ATTR.TACKLING] = tk
        sim.base[c, ATTR.DRIBBLING] = ret
        sim.base[c, ATTR.BALANCE] = ret
        sim.base[c, ATTR.STRENGTH] = ret
        sim.base[c, ATTR.FIRST_TOUCH] = ret
        sim.refresh()
        wins = 0
        for i in range(300):
            _spread(w)
            w.pos[c] = (50.0, 34.0)
            w.pos[k] = (50.8, 34.0)
            w.mode, w.owner, w.carry, w.poss = Mode.CARRIED, c, None, 0
            w.stun[:] = 0
            for _ in range(80):                          # 每次給 8 秒的接觸
                play.tackle_step(w, 0.1)
                if w.owner == k:
                    break
            wins += w.owner == k
        results[tag] = wins / 300
    assert results["strong"] > results["weak"] + 0.2


# ------------------------------------------------------------------ phase
def test_phase_changes_only_when_possession_changes_and_windows_expire(world):
    w = world
    w.ball[:2] = (50.0, 34.0)
    w.poss = 0
    w.phase, w.phase_t = [Phase.IP, Phase.OOP], [0.0, 0.0]
    phase_mod.possession_change(w, 0, 3, "pass")                       # 同隊：不變
    assert w.phase == [Phase.IP, Phase.OOP]
    phase_mod.possession_change(w, 1, 14, "tackle")
    assert w.phase == [Phase.DT, Phase.AT] and w.poss == 1
    assert w.phase_t[0] > 1.0 and w.phase_t[1] == w.sp.phase.at_window
    for _ in range(int((w.sp.phase.at_window + 1) / 0.1) + 60):
        phase_mod.update(w, 0.1)
    assert w.phase == [Phase.OOP, Phase.IP]
    # 失球後在 DT 窗口內奪回 = 反搶成功
    n0 = len([e for e in w.events if e.type == "press_recovery"])
    phase_mod.possession_change(w, 0, 3, "tackle")                     # 0 隊在 OOP 時搶回：不是反搶
    assert len([e for e in w.events if e.type == "press_recovery"]) == n0
    phase_mod.possession_change(w, 1, 14, "tackle")                    # 1 隊剛失球（DT）又立刻搶回：反搶成功
    pr = [e for e in w.events if e.type == "press_recovery"]
    assert len(pr) == n0 + 1 and pr[-1].team == 1


def test_counter_press_uses_more_players_with_higher_press_and_shorter_window_when_tired(world):
    w = world
    w.ball[:2] = (50.0, 34.0)
    _spread(w)
    w.sim.teams[0].tactics.press = 1
    low = len(phase_mod.counter_press_slots(w, 0))
    w.sim.teams[0].tactics.press = 5
    high = len(phase_mod.counter_press_slots(w, 0))
    assert high > low
    w.sim.energy[:] = 1.0
    fresh = phase_mod.dt_window(w, 0)
    w.sim.energy[:] = 0.3
    assert phase_mod.dt_window(w, 0) < fresh
    w.sim.teams[0].tactics.press = 3


# ------------------------------------------------------------------ shape
def test_targets_stay_on_the_pitch_and_high_line_pushes_the_defence_up(world):
    w = world
    w.mode, w.owner, w.poss = Mode.FREE, None, 0
    w.ball[:2] = (55.0, 34.0)
    xs = {}
    for lh in (1, 5):
        w.sim.teams[1].tactics.line_height = lh
        w.tgt_s = None
        sim_shape.update_targets(w)
        act = [s for s in w.active(1) if s != w.sim.gk_slot(1)]
        assert (w.tgt[:, 0] >= 0).all() and (w.tgt[:, 0] <= 105).all()
        assert (w.tgt[:, 1] >= 0).all() and (w.tgt[:, 1] <= 68).all()
        xs[lh] = float(np.sort(w.tgt[act, 0])[:4].mean())          # 客隊（往 −x 進攻）最深的 4 人 = x 最大者
    w.sim.teams[1].tactics.line_height = 3
    low_line = float(np.sort(w.tgt[[s for s in w.active(1) if s != w.sim.gk_slot(1)], 0])[-4:].mean())
    assert low_line > 0


# ------------------------------------------------------------------ decision
def test_on_ball_decision_is_the_v1_decision_given_the_same_state_and_random_streams(world):
    w = world
    _spread(w)
    c = 6
    w.pos[c] = (60.0, 34.0)
    w.ball[:2] = w.pos[c]
    for s in range(11, 22):                                          # 客隊放在主隊前方，產生選項
        w.pos[s] = (75.0 + 2.0 * (s - 11) % 10, 8.0 + 5.0 * (s - 11))
    sim = w.sim
    w.sp.motion.vmax_scale = 1.0                         # 決策函式的運動能力換算在這裡關掉，才能與 v1 直接比
    w.sp.motion.decision_tau_scale = 1.0
    st_act = sim.rng_act.bit_generator.state
    st_dec = sim.rng_dec.bit_generator.state
    opt2, hold2 = play.decide(w, c)
    sim.rng_act.bit_generator.state = st_act
    sim.rng_dec.bit_generator.state = st_dec
    pos_T = w.tf(0, w.pos)
    off = v1shape.offside_line(pos_T, w.active(1))
    hold = sim.draw_hold()
    sim.poss, sim.carrier, sim.ball = 0, c, (float(w.ball[0]), float(w.ball[1]))
    sim._hold = hold
    try:
        opt1 = v1dec.choose(sim, 0, c, pos_T, off, None)
    finally:
        sim._hold = None
    assert hold == pytest.approx(hold2)
    assert (opt1.kind, opt1.target) == (opt2.kind, opt2.target)
    assert (opt1.x, opt1.y) == pytest.approx((opt2.x, opt2.y))
