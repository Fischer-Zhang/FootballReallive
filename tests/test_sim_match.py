"""v2 逐幀引擎的整場性質測試：可重現、不變量、事件與統計一致、效能紀錄（用縮短的半場以加快測試）"""
import hashlib
import time

import numpy as np
import pytest

from footballsim.analytics.stats import match_stats
from footballsim.data.generator import generate_league
from footballsim.sim import TickMatch
from footballsim.sim.params import load_sim_params
from footballsim.sim.world import Mode


@pytest.fixture(scope="module")
def league():
    return generate_league(seed=3, n_teams=4)


def short_params(half=240.0):
    return load_sim_params(overrides={"clock.half_seconds": half, "clock.stoppage_seconds": [0.0, 0.0]})


def run(league, seed, half=240.0, a=0, b=1):
    return TickMatch(league.teams[a], league.teams[b], sim_params=short_params(half), seed=seed).run()


def signature(res) -> str:
    h = hashlib.sha256()
    for e in res.events:
        h.update(f"{e.period}|{e.t:.2f}|{e.team}|{e.player}|{e.type}|{e.result}|{e.x:.2f}|{e.y:.2f}".encode())
    return h.hexdigest()


def test_same_seed_reproduces_the_match_and_different_seeds_differ(league):
    a, b, c = run(league, 5), run(league, 5), run(league, 6)
    assert signature(a) == signature(b) and a.score == b.score
    assert signature(a) != signature(c)
    assert a.distance == b.distance


def test_invariants_hold_at_every_tick(league):
    m = TickMatch(league.teams[0], league.teams[1], sim_params=short_params(), seed=2)
    w = m.w
    from footballsim.sim import rules
    rules.start_restart(w, "kickoff", m.first_kickoff, 52.5, 34.0, 0.0)
    rules.kickoff_teleport(w)
    dt = w.sp.clock.dt
    cap = w.sim.vmax * w.sp.motion.vmax_scale * 1.05 + 0.5        # 最高速（含轉換時的小誤差）
    last_clock, last_pos = w.clock, w.pos.copy()
    modes = set()
    for _ in range(3000):
        m.step()
        act = w.sim.active
        assert (w.pos[act, 0] >= 0).all() and (w.pos[act, 0] <= 105).all()
        assert (w.pos[act, 1] >= 0).all() and (w.pos[act, 1] <= 68).all()
        assert w.clock > last_clock
        # 沒有瞬移：每個 tick 的位移不超過最高速 × dt（發球者最後 2 m 就位、開球時越過中線者拉回，位移都在 2 m 以內）
        disp = np.hypot(*(w.pos - last_pos).T)
        assert (disp[act] <= cap[act] * dt + 2.0 + 1e-6).all(), disp.max()
        last_clock, last_pos = w.clock, w.pos.copy()
        modes.add(w.mode)
        if w.mode == Mode.CARRIED:
            assert w.owner is not None and np.allclose(w.ball[:2], w.pos[w.owner], atol=1e-6)
        elif w.mode == Mode.FLIGHT:
            assert w.flight is not None and w.owner is None
        else:
            assert w.owner is None
    assert {Mode.CARRIED, Mode.FLIGHT} <= modes


def test_events_are_complete_and_statistics_are_rebuilt_from_them(league):
    r = run(league, 3, half=420.0)
    assert r.events and r.events[-1].type == "period_end"
    types = {e.type for e in r.events}
    assert {"pass", "turnover"} <= types
    for e in r.events:
        assert e.period in (1, 2) and 0 <= e.x <= 105 and 0 <= e.y <= 68
        assert e.team in (-1, 0, 1)
    st = match_stats(r)["teams"]
    for t in (0, 1):
        assert st[t]["passes_ok"] <= st[t]["passes"]
        assert st[t]["passes"] > 20
    shot_goals = sum(1 for e in r.events if e.type == "shot" and e.result == "goal")
    assert abs(shot_goals - sum(r.score)) <= 1
    assert sum(r.distance.values()) > 0 and all(v >= 0 for v in r.distance.values())
    assert r.ball_in_play > 0 and sum(r.possession_time) == pytest.approx(r.ball_in_play, rel=1e-6)


def test_phases_and_transitions_appear_in_the_event_stream(league):
    r = run(league, 4, half=420.0)
    n = {k: sum(1 for e in r.events if e.type == k) for k in ("turnover", "press_recovery", "counter_attack")}
    assert n["turnover"] > 10
    assert n["press_recovery"] <= n["turnover"]
    # 反搶成功緊接在失球事件之後，且由另一隊完成
    for i, e in enumerate(r.events):
        if e.type == "press_recovery":
            assert r.events[i - 1].type == "turnover" and r.events[i - 1].team != e.team


def test_speed_record(league):
    """效能紀錄：10 個模擬分鐘的耗時（全場外推）；寬鬆上限只用來抓出數量級的退步"""
    t0 = time.time()
    run(league, 1, half=300.0)
    dt = time.time() - t0
    full = dt * 5580.0 / 600.0
    print(f"10 個模擬分鐘 {dt:.1f} s，全場外推約 {full:.0f} s")
    assert full < 120.0
